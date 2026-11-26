// Port of the upstream Rust module emit/permlane_tests.rs, in Rust order.
//
// Synthetic row permutations on hosts supporting waves of at least 32 lanes.

#include "lower_card_util.h"

using namespace card;

namespace {

constexpr std::uint32_t THREADS = 128;
constexpr std::uint32_t OLD = 42;
constexpr std::uint32_t BASE = 0x1234'0000;

NativeProgram native(bool cross, bool fi, bool bc, bool alias, bool wave64) {
    auto mov = [](std::uint32_t dst, std::uint32_t src) { return 0x7E00'0200u | dst << 17 | src; };
    auto smov = [](std::uint32_t dst, std::uint32_t src) { return 0xBE80'0300u | dst << 16 | src; };
    std::vector<std::uint32_t> words = {
        smov(5, 128),
        0x3Du << 26 | 0x02u << 18 | 8u << 6 | 2u,
        125u << 25,   // s_load_dwordx4 s[8:11], s[4:5], 0
        0x340E'0082,  // v_lshlrev_b32 v7, 2, v0 (output byte offset)
        0x4A0A'00FF,
        BASE,  // v_add_nc_u32 v5, BASE, v0
        mov(6, alias ? 256 + 5 : 128 + OLD),
    };
    words.push_back(wave64 ? 0xBEFE'0402u : smov(126, 2));  // exec = s[2:3] / s2
    words.push_back(0xD400'0006u | (cross ? 0x378u : 0x377u) << 16 | std::uint32_t(fi) << 11 |
                    std::uint32_t(bc) << 12);
    words.push_back((256u + (alias ? 6u : 5u)) | (1u << 18));  // v_permlane[x]16_b32 v6, v5/v6, s0, s1
    words.push_back(wave64 ? 0xBEFE'04C1u : smov(126, 193));  // restore EXEC
    words.push_back(0x38u << 26 | 0x1Cu << 18 | 1u << 12);
    words.push_back(0x80u << 24 | 2u << 16 | 6u << 8 | 7u);  // buffer_store_dword v6, v7, s[8:11], 0 offen
    words.push_back(0xBF81'0000u);
    NativeProgram n;
    n.stage = HwStage::Cs;
    n.wave64 = wave64;
    n.vgprs = 8;
    n.layout = {root_param(RootType::Constants(), 0, 4, 0, 0, 0), root_param(RootType::Table(), 0, 1, 0, 4, 1)};
    n.used = {used_resource(ResourceType::Uav(), RootType::Table(), 0, 0, 0, 4, 4, 1)};
    n.io = compute_io(5 << 1, {THREADS, 1, 1});
    n.code = le_bytes(words);
    return n;
}

// Throws ShaderError from lowering (Rust `result?`).
HostShader lowered(const NativeProgram& native, std::uint32_t host_wave) {
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, native.vgprs);
    vs::structure::Structured st = vs::structure::structure(p);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered result;
    {
        HostWave was(host_wave);
        result = vs::lower::lower(parts_of(native, p, st, io, res));
    }
    HostShader shader = host_shader(std::move(result), res.root_indexed_views);
    std::string why;
    if (auto v = load_validator(&why)) {
        std::vector<std::uint8_t> copy = shader.dxil;
        try {
            v->validate(copy);
        } catch (const std::exception& e) {
            fail(std::string("DXIL validation: ") + e.what());
        }
        require(copy.size() >= 20 && shader.dxil.size() >= 20 &&
                    std::equal(copy.begin() + 4, copy.begin() + 20, shader.dxil.begin() + 4),
                "container hash");
    } else {
        std::fprintf(stderr, "DXIL validator unavailable: %s\n", why.c_str());
    }
    return shader;
}

// ---- rs:121 -------------------------------------------------------------------------------------
void permlane_fetch_inactive_outside_the_host_branch_is_rejected() {
    x2w::vs::lower::test::set_dump_name(
        "emit__permlane_tests__permlane_fetch_inactive_outside_the_host_branch_is_rejected");
    NativeProgram n = native(true, true, false, false, false);
    std::vector<std::uint32_t> words = le_words(n.code);
    // Skip the permutation when EXEC is empty. Structurisation turns this into a host lane branch,
    // so FI cannot read lanes that took the skip.
    words.insert(words.begin() + 8, 0xBF88'0002u);  // s_cbranch_execz +2 (to the EXEC restore)
    n.code = le_bytes(words);
    bool refused = false;
    try {
        lowered(n, 32);
    } catch (const ShaderError& e) {
        refused = is_unsupported(e, "a permlane fetch from lanes outside the host branch");
    }
    require(refused, "Err(Unsupported { what: \"a permlane fetch from lanes outside the host branch\" })");
}

// ---- rs:142 -------------------------------------------------------------------------------------
void permlane_rows_and_inactive_sources() {
    x2w::vs::lower::test::set_dump_name("emit__permlane_tests__permlane_rows_and_inactive_sources");
    std::lock_guard one(one_at_a_time());
    int ran = 0;
    const std::pair<const char*, bool> backends[] = {{"hardware", false}, {"WARP", true}};
    for (auto [backend, warp] : backends) {
        auto host = new_host(warp);
        if (!host) {
            std::fprintf(stderr, "no SM 6.6 %s; skipped\n", backend);
            continue;
        }
        auto [lanes_min, lanes_max] = wave_lane_counts(*host);
        std::fprintf(stderr, "%s waves %u..%u\n", backend, lanes_min, lanes_max);
        if (lanes_max < 32) {
            std::fprintf(stderr, "%s cannot hold two 16-lane rows; GPU checks skipped\n", backend);
            continue;
        }
        ran += 1;
        std::fprintf(stderr, "permlane: testing row and EXEC cases on %s\n", backend);
        for (bool wave64 : {false, true}) {
            if (lanes_min > (wave64 ? 64u : 32u)) {
                std::fprintf(stderr, "%s does not support the requested console wave size; skipped\n", backend);
                continue;
            }
            for (bool cross : {false, true}) {
                for (bool fi : {false, true}) {
                    for (bool bc : {false, true}) {
                        for (bool alias : {false, true}) {
                            HostShader shader;
                            try {
                                shader = lowered(native(cross, fi, bc, alias, wave64), lanes_max);
                            } catch (const ShaderError& e) {
                                fail("lower permlane: " + e.to_string());
                            }
                            std::string why;
                            auto pipe = compute_pipeline(*host, shader, &why);
                            if (!pipe) fail("permlane pipeline: " + why);
                            const std::array<std::uint32_t, 2> selector_cases[] = {
                                {0x7654'3210u, 0xFEDC'BA98u},  // identity
                                {0x89AB'CDEFu, 0x0123'4567u},  // reverse
                                {0u, 0u},                      // broadcast lane 0
                                {0xFFFF'FFFFu, 0xFFFF'FFFFu},  // broadcast lane 15
                                {0xF10E'2D3Cu, 0x4B5A'6978u},  // arbitrary; both selector words
                            };
                            for (const auto& selectors : selector_cases) {
                                for (std::uint32_t mask : {0xFFFF'FFFFu, 0x5555'5555u, 0xFFFF'0000u, 1u, 0u}) {
                                    const std::array<std::uint32_t, 4> constants = {selectors[0], selectors[1],
                                                                                    mask, mask};
                                    std::vector<std::uint32_t> ud(shader.user_data_dwords(), 0);
                                    for (const UserData& u : shader.user_data)
                                        ud.at(u.index) = u.root == RootType::Constants() ? constants.at(u.dword) : 0;
                                    const View views[] = {raw_uav(THREADS)};
                                    auto out = dispatch(*host, *pipe, std::move(ud), views, {}, {1, 1, 1},
                                                        "dispatch permlane");
                                    for (std::uint32_t t = 0; t < THREADS; ++t) {
                                        std::uint32_t pos = t % 16;
                                        std::uint64_t packed =
                                            std::uint64_t(selectors[0]) | std::uint64_t(selectors[1]) << 32;
                                        std::uint32_t pick = static_cast<std::uint32_t>(packed >> (4 * pos)) & 15;
                                        std::uint32_t src = ((t & ~15u) ^ (cross ? 16u : 0u)) | pick;
                                        std::uint32_t old = alias ? BASE + t : OLD;
                                        std::uint32_t want = ((mask >> (t % 32)) & 1) == 0 ? old
                                                             : (fi || ((mask >> (src % 32)) & 1) != 0) ? BASE + src
                                                             : bc ? 0
                                                                  : old;
                                        std::uint32_t got = out[0][t];
                                        if (got != want)
                                            fail(std::string(backend) + " wave64=" + (wave64 ? "true" : "false") +
                                                 " cross=" + (cross ? "true" : "false") +
                                                 " fi=" + (fi ? "true" : "false") + " bc=" + (bc ? "true" : "false") +
                                                 " alias=" + (alias ? "true" : "false") + " selectors=[" +
                                                 hex(selectors[0]) + ", " + hex(selectors[1]) + "] mask=" + hex(mask) +
                                                 " lane=" + std::to_string(t) + ": got " + hex(got) + ", want " +
                                                 hex(want));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        std::fprintf(stderr, "permlane: all row and EXEC cases passed on %s\n", backend);
    }
    if (ran == 0) std::fprintf(stderr, "no SM 6.6 device with 32-lane rows; GPU checks skipped\n");
}

}  // namespace

int main() {
    guarded("permlane_fetch_inactive_outside_the_host_branch_is_rejected",
            permlane_fetch_inactive_outside_the_host_branch_is_rejected);
    guarded("permlane_rows_and_inactive_sources", permlane_rows_and_inactive_sources);
    return test_util::finish("lower_permlane_test");
}
