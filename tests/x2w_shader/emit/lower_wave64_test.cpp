// Port of the upstream Rust module emit/wave64_tests.rs, in Rust order.
//
// A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a per-lane
// branch, with that branch taken wave-wide (structure::structure_wide) so that the two host waves
// can exchange their halves inside it. Lowered, validated by dxil.dll when present, and run on the
// host GPU when it takes waves of 32.

#include "lower_card_util.h"
#include "x2w_shader/emit/wave.h"  // WAVE_OR_V7_TO_S7

using namespace card;

namespace {

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

// Rust HashSet<usize> of branch blocks: structure_wide takes an ordered set here.
using Wide = std::set<std::size_t>;

NativeProgram native_of(std::span<const std::uint32_t> body, bool wave64, std::uint32_t threads);

// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4 root
// constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
NativeProgram native(std::span<const std::uint32_t> body) { return native_of(body, true, 64); }

// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads below
// s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store the result
// to dword 0 of the raw UAV. `skip`: the branch past the body when no thread is below s0:
// `s_cbranch_execz`, or `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
NativeProgram program(std::uint32_t skip) {
    std::vector<std::uint32_t> words = {
        0x7E0A'0281,  // v_mov_b32 v5, 1
        0x2C0C'0081,  // v_lshrrev_b32 v6, 1, v0
        0x340E'0B06,  // v_lshlrev_b32 v7, v6, v5
        0x7D88'0000,  // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C'246A,  // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,    // the branch to the restore of EXEC
    };
    words.insert(words.end(), vs::wave::WAVE_OR_V7_TO_S7.begin(), vs::wave::WAVE_OR_V7_TO_S7.end());
    for (std::uint32_t w : {
             0x7E00'0207u,  // v_mov_b32 v0, s7
             0x38u << 26 | 0x1Cu << 18,
             0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
             0xBEFE'040Cu,            // s_mov_b64 exec, s[12:13]
         })
        words.push_back(w);
    return native(words);
}

// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
NativeProgram native_of(std::span<const std::uint32_t> body, bool wave64, std::uint32_t threads) {
    std::vector<std::uint32_t> words = {
        0xBE85'0380,  // s_mov_b32 s5, 0
        0x3Du << 26 | 0x02u << 18 | 8u << 6 | 2u,
        125u << 25,  // s_load_dwordx4 s[8:11], s[4:5], 0
    };
    words.insert(words.end(), body.begin(), body.end());
    words.push_back(S_ENDPGM);
    NativeProgram n;
    n.stage = HwStage::Cs;
    n.wave64 = wave64;
    n.vgprs = 16;
    n.layout = {root_param(RootType::Constants(), 0, 4, 0, 0, 0), root_param(RootType::Table(), 0, 1, 0, 4, 1)};
    n.used = {used_resource(ResourceType::Uav(), RootType::Table(), 0, 0, 0, 4, 4, 1)};
    n.io = compute_io(5 << 1, {threads, 1, 1});  // 5 user SGPRs: s0..s4
    n.code = le_bytes(words);
    return n;
}

HostShader lowered_counted(const NativeProgram& native, const vs::ir::Program& p, const Wide& wide,
                           std::uint32_t host_wave, bool counted) {
    vs::structure::Structured st = vs::structure::structure_wide(p, wide);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered lowered;
    {
        HostWave was(host_wave);
        lowered = vs::lower::lower_counted(parts_of(native, p, st, io, res), counted);  // throws ShaderError
    }
    return host_shader(std::move(lowered), res.root_indexed_views);
}

// `lowered` for a host whose waves have at most `host_wave` lanes. Throws ShaderError.
HostShader lowered_on(const NativeProgram& native, const vs::ir::Program& p, const Wide& wide,
                      std::uint32_t host_wave) {
    return lowered_counted(native, p, wide, host_wave, false);
}

// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
HostShader lowered(const NativeProgram& native, const vs::ir::Program& p, const Wide& wide) {
    return lowered_on(native, p, wide, 32);
}

// The wave OR's words with v_add_f16 for v_or_b32 (opcode 50 for 28).
void extend_wave_sum(std::vector<std::uint32_t>& words) {
    for (std::uint32_t w : vs::wave::WAVE_OR_V7_TO_S7)
        words.push_back(w >> 25 == 28 ? (w & 0x01FF'FFFFu) | 50u << 25 : w);
}

// The half's float (normal numbers; with `zero`, 0 is +0).
float wide_half(std::uint16_t h) {
    return f32_from_bits(((std::uint32_t(h >> 10) + 112) << 23) | (std::uint32_t(h & 0x3FF) << 13));
}

// To the nearest half, ties to even (normal numbers).
std::uint16_t narrow_half(float f) {
    std::uint32_t b = f32_bits(f);
    std::uint32_t h = (((b >> 23) - 112) << 10) | ((b >> 13) & 0x3FF);
    std::uint32_t rest = b & 0x1FFF;
    return static_cast<std::uint16_t>(h + std::uint32_t(rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1)));
}

// The same, 0 for +0.
float wide_half_or_zero(std::uint16_t h) { return h == 0 ? 0.0f : wide_half(h); }
std::uint16_t narrow_half_or_zero(float f) { return f == 0.0f ? 0 : narrow_half(f); }

// Pairs, then pairs of pairs, ... (Rust `tree.chunks(2).map(|c| add(c[0], c[1]))` to one).
template <class Add>
std::uint16_t tree_of(std::vector<std::uint16_t> tree, Add add) {
    while (tree.size() > 1) {
        std::vector<std::uint16_t> next;
        for (std::size_t i = 0; i + 1 < tree.size(); i += 2) next.push_back(add(tree[i], tree[i + 1]));
        tree = std::move(next);
    }
    return tree[0];
}

// Dword 0 of the raw UAV after one group, root constants from `ud`.
std::vector<std::vector<std::uint32_t>> dispatch_one(const Host& host, const Pipeline& pipe,
                                                     std::vector<std::uint32_t> ud) {
    const View views[] = {raw_uav(1)};
    return dispatch(host, pipe, std::move(ud), views, {}, {1, 1, 1});
}

// ---- rs:148 -------------------------------------------------------------------------------------
void a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit");
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd mantissas, so
    // that additions round); all of them store the wave's sum
    std::vector<std::uint32_t> words = {
        0x340C'0085,  // v_lshlrev_b32 v6, 5, v0
        0x340E'0082,  // v_lshlrev_b32 v7, 2, v0
        0x4A0E'0F06,  // v_add_nc_u32 v7, v6, v7
        0x4A0E'0F00,  // v_add_nc_u32 v7, v0, v7
        0x4A0E'0EFF,
        0x0000'3C00,  // v_add_nc_u32 v7, 0x3c00, v7
    };
    extend_wave_sum(words);
    for (std::uint32_t w : {
             0x7E00'0207u,  // v_mov_b32 v0, s7
             0x38u << 26 | 0x1Cu << 18,
             0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
         })
        words.push_back(w);
    NativeProgram n = native(words);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    require(count_insts(p, "v_add_f16") == 6, "six v_add_f16");
    HostShader shader = lowered(n, p, {});
    validate(shader);
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits (to
    // nearest, ties to even); values here are normal numbers
    auto add = [](std::uint16_t a, std::uint16_t b) { return narrow_half(wide_half(a) + wide_half(b)); };
    std::vector<std::uint16_t> terms;
    for (std::uint16_t t = 0; t < 64; ++t) terms.push_back(static_cast<std::uint16_t>(0x3C00 + 37 * t));
    std::uint16_t tree = tree_of(terms, add);
    std::uint16_t in_a_row = terms[0];
    for (std::size_t i = 1; i < terms.size(); ++i) in_a_row = add(in_a_row, terms[i]);
    require(tree != in_a_row, "these terms tell the tree from another order");
    std::lock_guard one(one_at_a_time());
    auto host = new_host(false);
    if (!host) {
        std::fprintf(stderr, "no SM 6.6 hardware device; not run\n");
        return;
    }
    std::string why;
    auto pipe = compute_pipeline(*host, shader, &why);
    if (!pipe) {
        std::fprintf(stderr, "the hardware device does not take this program (waves of 32?): %s; not run\n",
                     why.c_str());
        return;
    }
    std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
    auto out = dispatch_one(*host, *pipe, std::move(ud));
    // (the high half is lane 0's term's: zero)
    require(out[0][0] == std::uint32_t(tree), "tree " + hex(tree) + ", in a row " + hex(in_a_row));
    std::fprintf(stderr, "ran on the hardware device: the sum %s of the console's tree\n", hex(tree).c_str());
}

// `native` for a host of 64-lane waves.
HostShader on_64_lanes(const NativeProgram& native, const vs::ir::Program& p) {
    vs::structure::Structured st = vs::structure::structure(p);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered lowered;
    {
        HostWave was(64);
        lowered = vs::lower::lower(parts_of(native, p, st, io, res));
    }
    require(lowered.wave_size == std::optional<std::uint32_t>(64), "wave_size == Some(64)");
    HostShader shader = host_shader(std::move(lowered), res.root_indexed_views);
    validate(shader);
    return shader;
}

// Dword 0 of the UAV after `shader` ran with root constants `constants`; none without a device
// that takes it.
std::optional<std::uint32_t> dispatched(const HostShader& shader, std::span<const std::uint32_t> constants) {
    std::lock_guard one(one_at_a_time());
    auto host = new_host(false);
    if (!host) {
        std::fprintf(stderr, "no SM 6.6 hardware device; not run\n");
        return std::nullopt;
    }
    std::string why;
    auto pipe = compute_pipeline(*host, shader, &why);
    if (!pipe) {
        std::fprintf(stderr, "the hardware device does not take this program (waves of 64?): %s; not run\n",
                     why.c_str());
        return std::nullopt;
    }
    std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
    for (const UserData& u : shader.user_data)
        if (u.root == RootType::Constants()) ud[u.index] = u.dword < constants.size() ? constants[u.dword] : 0;
    auto out = dispatch_one(*host, *pipe, std::move(ud));
    return out[0][0];
}

// ---- rs:222 -------------------------------------------------------------------------------------
// "Some lane is left" of a mask operation as a number: an upscaler's programs wait for other
// groups in a loop that ends when every lane saw its flag, or after enough rounds: both tests
// become -1 or 0 and SCC of `timeout | ~waiting` ends it.
void some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on() {
    x2w::vs::lower::test::set_dump_name("emit__wave64_tests__some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on");
    const std::uint32_t body[] = {
        0xD765'0001,
        0x0001'00C1,  // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766'0001,
        0x0002'02C1,  // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D84'0200,  // v_cmp_eq_u32 vcc, s0, v1
        0x8792'7E6A,  // s_and_b64 s[18:19], vcc, exec
        0x8592'80C1,  // s_cselect_b64 s[18:19], -1, 0
        0xBF09'8501,  // s_cmp_ge_u32 s1, 5
        0x8594'80C1,  // s_cselect_b64 s[20:21], -1, 0
        0x8B92'1214,  // s_orn2_b64 s[18:19], s[20:21], s[18:19]
        0x8506'8081,  // s_cselect_b32 s6, 1, 0
        0x7E04'0206,  // v_mov_b32 v2, s6
        0xE070'0000,
        0x8002'0200,  // buffer_store_dword v2, off, s[8:11], 0
    };
    NativeProgram n = native(body);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    HostShader shader = on_64_lanes(n, p);
    // (lane s0 waits, if there is one; s1 rounds)
    const std::array<std::uint32_t, 3> cases[] = {{0, 0, 0},  {37, 4, 0}, {63, 4, 0},
                                                  {64, 0, 1}, {37, 5, 1}, {64, 9, 1}};
    for (auto [wait, rounds, done] : cases) {
        const std::uint32_t constants[] = {wait, rounds};
        auto out = dispatched(shader, constants);
        if (!out) return;
        require(*out == done, "lane " + std::to_string(wait) + " after " + std::to_string(rounds) + " rounds");
    }
}

// ---- rs:328 -------------------------------------------------------------------------------------
// The same sum inside `if (t >= s0) { ... }` on a host of 64-lane waves: the lanes below s0 are
// elsewhere, and count as +0 in the console's tree.
void a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64");
    std::vector<std::uint32_t> words = {
        0x340C'0085,       // v_lshlrev_b32 v6, 5, v0
        0x340E'0082,       // v_lshlrev_b32 v7, 2, v0
        0x4A0E'0F06,       // v_add_nc_u32 v7, v6, v7
        0x4A0E'0F00,       // v_add_nc_u32 v7, v0, v7
        0x4A0E'0EFF,
        0x0000'3C00,       // v_add_nc_u32 v7, 0x3c00, v7
        0x7D86'0000,       // v_cmp_le_u32 vcc, s0, v0
        0xBE8C'246A,       // s_and_saveexec_b64 s[12:13], vcc
        0xBF88'0000 | 23,  // s_cbranch_execz to the restore of EXEC
    };
    extend_wave_sum(words);
    for (std::uint32_t w : {
             0x7E00'0207u,  // v_mov_b32 v0, s7
             0x38u << 26 | 0x1Cu << 18,
             0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
             0xBEFE'040Cu,            // s_mov_b64 exec, s[12:13]
         })
        words.push_back(w);
    NativeProgram n = native(words);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    HostShader shader = on_64_lanes(n, p);
    auto add = [](std::uint16_t a, std::uint16_t b) {
        return narrow_half_or_zero(wide_half_or_zero(a) + wide_half_or_zero(b));
    };
    for (std::uint32_t from : {0u, 1u, 5u, 31u, 32u, 37u, 63u}) {
        std::vector<std::uint16_t> terms;
        for (std::uint16_t t = 0; t < 64; ++t)
            terms.push_back(std::uint32_t(t) < from ? 0 : static_cast<std::uint16_t>(0x3C00 + 37 * t));
        std::uint16_t tree = tree_of(terms, add);
        const std::uint32_t constants[] = {from};
        auto out = dispatched(shader, constants);
        if (!out) return;
        require(*out == std::uint32_t(tree), "lanes from " + std::to_string(from));
    }
}

// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0 of the
// UAV; none (and a line) without a device that takes it.
std::optional<std::uint32_t> on_the_card(const HostShader& shader, std::array<std::uint32_t, 2> constants) {
    auto host = new_host(false);
    if (!host) {
        std::fprintf(stderr, "no SM 6.6 hardware device; not run\n");
        return std::nullopt;
    }
    auto pipe = compute_pipeline(*host, shader);
    if (!pipe) {
        std::fprintf(stderr, "the hardware device does not take this program; not run\n");
        return std::nullopt;
    }
    std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
    for (const UserData& u : shader.user_data)
        if (u.root == RootType::Constants() && u.slot == 0 && u.dword <= 1) ud[u.index] = constants[u.dword];
    auto out = dispatch_one(*host, *pipe, std::move(ud));
    return out[0][0];
}

// ---- rs:378 -------------------------------------------------------------------------------------
// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a thread past
// the end of the data): on the console they add +0, and after each step of the tree they hold
// their run's partial sum all the same. On a host of 64-lane waves (AMD) the lanes that are here
// read each partner run's sum from a lane of it that is here.
void a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit");
    std::vector<std::uint32_t> words = {
        0x7DA8'0000,  // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0,            // s_cbranch_execz to the end (filled in below)
        0x340C'0085,  // v_lshlrev_b32 v6, 5, v0
        0x340E'0082,  // v_lshlrev_b32 v7, 2, v0
        0x4A0E'0F06,  // v_add_nc_u32 v7, v6, v7
        0x4A0E'0F00,  // v_add_nc_u32 v7, v0, v7
        0x4A0E'0EFF,
        0x0000'3C00,  // v_add_nc_u32 v7, 0x3c00, v7
    };
    extend_wave_sum(words);
    for (std::uint32_t w : {
             0x7E00'0207u,  // v_mov_b32 v0, s7
             0x38u << 26 | 0x1Cu << 18,
             0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
         })
        words.push_back(w);
    // past every word after it, to the program's end
    words[1] = 0xBF88'0000u | (static_cast<std::uint32_t>(words.size()) - 2);
    NativeProgram n = native(words);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    HostShader shader = lowered_on(n, p, {}, 64);
    validate(shader);
    auto add = [](std::uint16_t a, std::uint16_t b) {
        return narrow_half_or_zero(wide_half_or_zero(a) + wide_half_or_zero(b));
    };
    std::lock_guard one(one_at_a_time());
    bool ran = false;
    for (std::uint32_t stay : {64u, 63u, 40u, 33u, 32u, 17u, 5u, 1u}) {
        std::vector<std::uint16_t> terms;
        for (std::uint32_t t = 0; t < 64; ++t) terms.push_back(t < stay ? static_cast<std::uint16_t>(0x3C00 + 37 * t) : 0);
        std::uint16_t tree = tree_of(terms, add);
        auto got = on_the_card(shader, {stay, 0});
        if (!got) return;
        ran = true;
        // (the high half is lane 0's term's: zero)
        require(*got == std::uint32_t(tree), std::to_string(stay) + " lanes stay: got " + hex(*got) +
                                                 ", the console's tree " + hex(tree));
    }
    if (ran) std::fprintf(stderr, "ran on the hardware device: the console's tree with lanes gone, in every case\n");
}

// ---- rs:448 -------------------------------------------------------------------------------------
// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2 (0x4000)
// is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The host's `f32tof16`
// may give 0x4201 (toward zero); with X2W_SHADERCFG_F16_NEAREST=1 the translation gives the console's
// whatever the host does.
void a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    x2w::vs::lower::test::set_dump_name("emit__wave64_tests__a_16_bit_sum_rounds_to_nearest_when_asked_to");
    const std::uint32_t words[] = {
        0x7E0E'02FF,
        0x0000'3C03,  // v_mov_b32 v7, 0x3c03
        0x7E0C'02FF,
        0x0000'4000,  // v_mov_b32 v6, 0x4000
        0x6400'0D07,  // v_add_f16 v0, v7, v6
        0x38u << 26 | 0x1Cu << 18,
        0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
    };
    NativeProgram n = native(words);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    require(count_insts(p, "v_add_f16") > 0, "a v_add_f16");
    std::lock_guard one(one_at_a_time());
    for (bool nearest : {false, true}) {
        HostShader shader;
        {
            F16Nearest was(nearest);
            shader = lowered(n, p, {});
        }
        validate(shader);
        auto host = new_host(false);
        if (!host) {
            std::fprintf(stderr, "no SM 6.6 hardware device; not run\n");
            return;
        }
        auto pipe = compute_pipeline(*host, shader);
        if (!pipe) {
            std::fprintf(stderr, "the hardware device does not take this program; not run\n");
            return;
        }
        std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
        auto out = dispatch_one(*host, *pipe, std::move(ud));
        std::uint32_t got = out[0][0] & 0xFFFF;
        std::fprintf(stderr, "v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest %s: %s\n",
                     nearest ? "true" : "false", hex(got).c_str());
        if (nearest)
            require(got == 0x4202, hex(got));
        else
            require(got == 0x4201 || got == 0x4202, hex(got));
    }
}

// ---- rs:507 -------------------------------------------------------------------------------------
void a_wave_or_inside(std::uint32_t skip, const char* name) {
    NativeProgram n = program(skip);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    std::optional<std::size_t> found;
    for (std::size_t b = 0; b < p.blocks.size() && !found; ++b)
        if (!p.blocks[b].insts.empty() && p.blocks[b].insts.back().name() == name) found = b;
    require(found.has_value(), "the branch");
    std::size_t branch = *found;
    // per lane, the two host waves cannot meet inside the branch
    vs::lower::take_needs();
    try {
        lowered(n, p, {});
        fail("lowered per lane");
    } catch (const ShaderError& e) {
        require(is_unsupported(e, "a wave reduction where not every lane runs"), e.to_string());
    }
    // ... and the lowering names the block, around which that branch is the one to widen
    std::vector<std::size_t> needs = vs::lower::take_needs();
    vs::structure::Structured st = vs::structure::structure(p);
    require(vs::lower::branches_around(st, needs) == std::optional<std::vector<std::size_t>>({branch}),
            "branches_around == Some([branch])");
    // (taken, past the body, when every lane's bit is clear)
    require(st.wave_any[branch] == std::optional<bool>(false), "wave_any[branch] == Some(false)");
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    const Wide wide = {branch};
    vs::structure::Structured st_wide = vs::structure::structure_wide(p, wide);
    require(p.blocks[branch].exit.is_branch(), "unreachable");
    std::size_t inside = p.blocks[branch].exit.not_taken;
    bool guarded_inside = false;
    for (const auto& g : st_wide.guards[inside]) guarded_inside |= g.has_value();
    require(guarded_inside, "a guard inside");
    HostShader shader = lowered(n, p, wide);
    validate(shader);
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of both
    // halves, whichever half's lanes take the branch.
    std::lock_guard one(one_at_a_time());
    auto host = new_host(false);
    if (!host) {
        std::fprintf(stderr, "no SM 6.6 hardware device; not run\n");
        return;
    }
    std::string why;
    auto pipe = compute_pipeline(*host, shader, &why);
    if (!pipe) {
        std::fprintf(stderr, "the hardware device does not take this program (waves of 32?): %s; not run\n",
                     why.c_str());
        return;
    }
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    const std::pair<std::uint32_t, std::uint32_t> cases[] = {
        {64, 0xFFFF'FFFF}, {40, 0x000F'FFFF}, {33, 0x0001'FFFF}, {10, 0x1F}, {1, 1}};
    for (auto [below, expected] : cases) {
        std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
        for (const UserData& u : shader.user_data)
            if (u.root == RootType::Constants() && u.slot == 0 && u.dword == 0) ud[u.index] = below;
        auto out = dispatch_one(*host, *pipe, std::move(ud));
        require(out[0][0] == expected, std::string(name) + ": " + std::to_string(below) + " threads in the branch");
    }
    std::fprintf(stderr, "ran on the hardware device (%s): the OR of both halves in every case\n", name);
}

void a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide");
    const std::pair<std::uint32_t, const char*> cases[] = {{0xBF88'0000, "s_cbranch_execz"},
                                                           {0xBF84'0000, "s_cbranch_scc0"}};
    for (auto [skip, name] : cases) a_wave_or_inside(skip, name);
}

// ---- rs:618 -------------------------------------------------------------------------------------
// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0) and a
// number on the other, and is read as a number after the join when a stack holds both masks and node numbers. The mask
// arrives as the ballot of its lanes.
void a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is");
    const std::uint32_t words[] = {
        0x7D88'0000,  // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94'036A,  // s_mov_b32 s20, vcc_lo
        0xBF06'8001,  // s_cmp_eq_u32 s1, 0
        0xBF85'0001,  // s_cbranch_scc1 over the next
        0xBE94'0385,  // s_mov_b32 s20, 5
        0x7E00'0214,  // v_mov_b32 v0, s20
        0x38u << 26 | 0x1Cu << 18,
        0x80u << 24 | 2u << 16,  // buffer_store_dword v0, off, s[8:11], 0
    };
    NativeProgram n = native_of(words, false, 32);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    HostShader shader = lowered(n, p, {});
    validate(shader);
    std::lock_guard one(one_at_a_time());
    const std::array<std::uint32_t, 3> cases[] = {{32, 0, 0xFFFF'FFFF}, {10, 0, 0x3FF}, {0, 0, 0}, {10, 1, 5}};
    for (auto [below, other, expected] : cases) {
        auto got = on_the_card(shader, {below, other});
        if (!got) return;
        require(*got == expected,
                std::to_string(below) + " lanes in the mask, s1 = " + std::to_string(other));
    }
    std::fprintf(stderr, "ran on the hardware device: the mask's bits, or the number\n");
}

// ---- rs:650 -------------------------------------------------------------------------------------
// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after the
// join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along. On the console the
// skip is taken by the wave, so with some lane inside every lane sees the inner test. A lane that
// skipped on its own would see "some lane is left" instead: refused per lane, right with the
// branch taken wave-wide. Lane 31 stores what it saw.
void scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    x2w::vs::lower::test::set_dump_name(
        "emit__wave64_tests__scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide");
    const std::uint32_t words[] = {
        0x7D88'0000,  // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D'3C6A,  // s_and_saveexec_b32 s13, vcc_lo
        0xBF88'0001,  // s_cbranch_execz over the next
        0xBF06'8701,  // s_cmp_eq_u32 s1, 7
        0xBEFE'030D,  // s_mov_b32 exec_lo, s13
        0xBF85'0002,  // s_cbranch_scc1 to "seven"
        0x7E0A'0281,  // v_mov_b32 v5, 1
        0xBF82'0001,  // s_branch over the next
        0x7E0A'0282,  // seven: v_mov_b32 v5, 2
        0x7DA4'009F,  // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38u << 26 | 0x1Cu << 18,
        0x80u << 24 | 2u << 16 | 5u << 8,  // buffer_store_dword v5, off, s[8:11], 0
    };
    NativeProgram n = native_of(words, false, 32);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    std::optional<std::size_t> found;
    for (std::size_t b = 0; b < p.blocks.size() && !found; ++b)
        if (!p.blocks[b].insts.empty() && p.blocks[b].insts.back().name() == "s_cbranch_execz") found = b;
    require(found.has_value(), "the skip");
    std::size_t branch = *found;
    vs::lower::take_needs();
    try {
        lowered(n, p, {});
        fail("lowered per lane");
    } catch (const ShaderError& e) {
        require(is_unsupported(e, "a lane mask read as data"), e.to_string());
    }
    vs::structure::Structured st = vs::structure::structure(p);
    require(vs::lower::branches_around(st, vs::lower::take_needs()) == std::optional<std::vector<std::size_t>>({branch}),
            "branches_around == Some([branch])");
    HostShader shader = lowered(n, p, Wide{branch});
    validate(shader);
    std::lock_guard one(one_at_a_time());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    const std::array<std::uint32_t, 3> cases[] = {{0, 7, 1}, {10, 7, 2}, {10, 3, 1}, {32, 3, 1}, {32, 7, 2}};
    for (auto [below, s1, expected] : cases) {
        auto got = on_the_card(shader, {below, s1});
        if (!got) return;
        require(*got == expected, std::to_string(below) + " lanes inside, s1 = " + std::to_string(s1));
    }
    std::fprintf(stderr,
                 "ran on the hardware device: the inner test's SCC for every lane once some lane is inside\n");
}

// ---- rs:700 -------------------------------------------------------------------------------------
// A dispatch counted in threads: of the last group only the lanes below the count store (after
// upstream commit bc3cd4b0).
void the_lanes_past_a_count_of_threads_do_not_run() {
    x2w::vs::lower::test::set_dump_name("emit__wave64_tests__the_lanes_past_a_count_of_threads_do_not_run");
    // Each lane that runs adds 1 to counter[local x].
    const std::uint32_t body[] = {
        0x3402'0082,  // v_lshlrev_b32 v1, 2, v0
        0x7e04'0281,  // v_mov_b32 v2, 1
        0xe0c8'1000, 0x8002'0201,  // buffer_atomic_add v2, v1, s[8:11], 0 offen
    };
    NativeProgram n = native(body);
    vs::ir::Program p = vs::ir::build(n.code, n.wave64, n.vgprs);
    HostShader shader = lowered_counted(n, p, {}, 64, true);
    const UserData* limit = nullptr;
    for (const UserData& u : shader.user_data)
        if (!limit && u.root == RootType::Other(vs::USER_DISPATCH_THREADS_X)) limit = &u;
    require(limit != nullptr, "the thread count's user data");
    require(limit->slot == 0 && limit->dword == 64, "(slot, dword) == (0, 64)");
    std::lock_guard one(one_at_a_time());
    auto host = new_host(true);
    if (!host) return;
    auto pipe = compute_pipeline(*host, shader);
    if (!pipe) return;
    for (std::uint32_t count : {0u, 1u, 7u, 63u, 64u, 65u, 74u, 77u, 127u, 128u}) {
        std::vector<std::uint32_t> constants(shader.user_data_dwords(), 0);
        constants[limit->index] = count;
        const View views[] = {raw_uav(68)};
        auto result = dispatch(*host, *pipe, std::move(constants), views, {}, {2, 1, 1});
        std::uint32_t active = count == 0 ? 128 : count;
        for (std::uint32_t i = 0; i < 64 && i < result[0].size(); ++i) {
            std::uint32_t want = std::uint32_t(i < active) + std::uint32_t(i + 64 < active);
            require(result[0][i] == want, "count " + std::to_string(count) + ", word " + std::to_string(i));
        }
    }
}

}  // namespace

int main() {
    guarded("a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit",
            a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit);
    guarded("some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on",
            some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on);
    guarded("a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64",
            a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64);
    guarded("a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit",
            a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit);
    guarded("a_16_bit_sum_rounds_to_nearest_when_asked_to", a_16_bit_sum_rounds_to_nearest_when_asked_to);
    guarded("a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide",
            a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide);
    guarded("a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is",
            a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is);
    guarded("scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide",
            scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide);
    guarded("the_lanes_past_a_count_of_threads_do_not_run", the_lanes_past_a_count_of_threads_do_not_run);
    return test_util::finish("lower_wave64_test");
}
