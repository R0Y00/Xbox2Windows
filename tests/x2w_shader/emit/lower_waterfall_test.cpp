// Port of the upstream Rust module emit/waterfall_tests.rs, in Rust order.
//
// End-to-end tests of waterfall loops over per-lane descriptors: the synthetic GFX10 programs of
// `resource::tests` lowered to DXIL, validated by dxil.dll when present, and run on the host GPU
// and WARP, each lane reading the resource its own descriptor names.

#include "lower_card_util.h"
#include "resource_fixtures.h"  // FRONT? vs-front: tests/x2w_shader/resource_fixtures.h (resource::tests)

using namespace card;
namespace fx = vs::resource::fixtures;  // FRONT?

namespace {

// Lowers a wave32 compute program of 32 threads with the table of views at s0 (textures, or with
// `buffers` structured buffers of dwords, at entries 0 and 1, a raw UAV at entry 2) and the table
// of samplers at s1.
HostShader build(std::span<const std::uint32_t> words, bool buffers) {
    auto table = [](std::uint8_t user_data) { return root_param(RootType::Table(), 0, 1, 0, user_data, user_data); };
    auto range = [](ResourceType kind, std::uint32_t table_offset_dwords, std::uint8_t dwords, std::uint32_t stride,
                    std::uint8_t user_data) {
        return used_resource(kind, RootType::Table(), table_offset_dwords, 0, stride, dwords, user_data, user_data);
    };
    UsedResource views = buffers ? range(ResourceType::Srv(), 0, 4, 4, 0) : range(ResourceType::Srv(), 0, 8, 0, 0);
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = 32;
    native.layout = {table(0), table(1)};
    native.used = {views, range(ResourceType::Uav(), 16, 4, 0, 0), range(ResourceType::Sampler(), 0, 4, 0, 1)};
    native.io = compute_io(2 << 1, {32, 1, 1});
    native.code = le_bytes(words);
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, native.vgprs);
    vs::structure::Structured st = vs::structure::structure(p);
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    Lowered lowered = vs::lower::lower(parts_of(native, p, st, io, res));
    // One access with each lane's own heap index, no wave operation.
    require(lowered.nonuniform_handles == 1, "nonuniform_handles == 1");
    require(!lowered.wave_ops, "!wave_ops");
    HostShader shader = host_shader(std::move(lowered), {});
    validate(shader, true);
    return shader;
}

// The red channel each lane sampled from two 1x1 textures, red 1.0 at heap index 0 and 2.0 at 1.
std::vector<std::pair<const char*, std::vector<float>>> sampled(const HostShader& shader) {
    const Texel first[] = {{1.0f, 0.0f, 0.0f, 1.0f}};
    const Texel second[] = {{2.0f, 0.0f, 0.0f, 1.0f}};
    const View views[] = {texture(1, 1, first), texture(1, 1, second), raw_uav(32)};
    std::vector<std::pair<const char*, std::vector<float>>> out;
    for (auto& [label, words] : run(shader, views)) {
        std::vector<float> red;
        for (std::uint32_t w : words) red.push_back(f32_from_bits(w));
        out.emplace_back(label, std::move(red));
    }
    return out;
}

// ---- rs:168 -------------------------------------------------------------------------------------
void lanes_sample_the_texture_their_if_else_arm_chose() {
    x2w::vs::lower::test::set_dump_name("emit__waterfall_tests__lanes_sample_the_texture_their_if_else_arm_chose");
    HostShader shader = build(fx::WATERFALL_JOINED_WAVE32, false);  // FRONT?
    for (auto& [label, red] : sampled(shader)) {
        // Lanes below 16 took the first texture, the others the second.
        std::vector<float> want;
        for (int l = 0; l < 32; ++l) want.push_back(l < 16 ? 1.0f : 2.0f);
        require(red == want, label);
    }
}

// ---- rs:179 -------------------------------------------------------------------------------------
void lanes_sample_the_texture_their_arm_loaded_or_picked() {
    x2w::vs::lower::test::set_dump_name("emit__waterfall_tests__lanes_sample_the_texture_their_arm_loaded_or_picked");
    HostShader shader = build(fx::WATERFALL_JOINED_MIXED_WAVE32, false);  // FRONT?
    for (auto& [label, red] : sampled(shader)) {
        // Lanes below 16 loaded entry (lane & 1); of the others, lanes below 24 picked the first,
        // the rest the second.
        std::vector<float> want;
        for (int l = 0; l < 32; ++l) {
            const float pair[] = {1.0f, 2.0f};
            want.push_back(l < 16 ? pair[l & 1] : l < 24 ? 1.0f : 2.0f);
        }
        require(red == want, label);
    }
}

// ---- rs:197 -------------------------------------------------------------------------------------
void lanes_load_from_the_buffer_their_nested_arms_chose() {
    x2w::vs::lower::test::set_dump_name("emit__waterfall_tests__lanes_load_from_the_buffer_their_nested_arms_chose");
    HostShader shader = build(fx::WATERFALL_JOINED_BUFFER_WAVE32, true);  // FRONT?
    std::vector<std::uint32_t> first, second;
    for (std::uint32_t i = 0; i < 32; ++i) {
        first.push_back(100 + i);
        second.push_back(200 + i);
    }
    const View views[] = {buffer(first, stride_structured(4)), buffer(second, stride_structured(4)), raw_uav(32)};
    for (auto& [label, got] : run(shader, views)) {
        // Dword `lane` of the first buffer below 16, of the second from 16 to 23; lanes from 24
        // on load and store nothing.
        std::vector<std::uint32_t> want;
        for (std::uint32_t l = 0; l < 32; ++l) want.push_back(l < 16 ? 100 + l : l < 24 ? 200 + l : 0);
        require(got == want, label);
    }
}

// ---- rs:228 -------------------------------------------------------------------------------------
void lanes_sample_the_entry_their_offset_names_in_the_console_waterfall() {
    x2w::vs::lower::test::set_dump_name(
        "emit__waterfall_tests__lanes_sample_the_entry_their_offset_names_in_the_console_waterfall");
    HostShader shader = build(fx::WATERFALL_INDEXED_WAVE32, false);  // FRONT?
    auto results = sampled(shader);
    require(!results.empty(), "!results.is_empty()");
    for (auto& [label, red] : results) {
        // Even lanes hold entry 0's offset, odd lanes entry 1's.
        std::vector<float> want;
        for (int l = 0; l < 32; ++l) want.push_back((l & 1) ? 2.0f : 1.0f);
        require(red == want, label);
    }
}

// ---- rs:241 -------------------------------------------------------------------------------------
void lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx() {
    x2w::vs::lower::test::set_dump_name(
        "emit__waterfall_tests__lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx");
    // Lanes below 16 hold the first descriptor, the others the second.
    std::vector<float> want;
    for (int l = 0; l < 32; ++l) want.push_back(l < 16 ? 1.0f : 2.0f);
    for (bool folded : {false, true}) {
        std::vector<std::uint32_t> words = fx::cmpx_waterfall(false, folded);  // FRONT? -> std::vector<std::uint32_t>
        HostShader shader = build(words, false);
        auto results = sampled(shader);
        require(!results.empty(), "!results.is_empty()");
        for (auto& [label, red] : results)
            require(red == want, std::string(label) + ", folded " + (folded ? "true" : "false"));
    }
}

// ---- rs:256 -------------------------------------------------------------------------------------
void immediate_sampler_chosen_by_control_flow() {
    x2w::vs::lower::test::set_dump_name("emit__waterfall_tests__immediate_sampler_chosen_by_control_flow");
    using W2 = std::array<std::uint32_t, 2>;
    auto s_mov_lit = [](std::uint8_t dst, std::uint32_t lit) { return W2{0xBE80'03FFu | std::uint32_t(dst) << 16, lit}; };
    auto s_mov_zero = [](std::uint8_t dst) { return 0xBE80'0380u | std::uint32_t(dst) << 16; };
    auto s_load_x8 = [](std::uint8_t dst, std::uint8_t base) {
        return W2{0x3Du << 26 | 0x03u << 18 | std::uint32_t(dst) << 6 | (std::uint32_t(base) >> 1), 125u << 25};
    };
    auto s_load_x4_off = [](std::uint8_t dst, std::uint8_t base, std::uint32_t off) {
        return W2{0x3Du << 26 | 0x02u << 18 | std::uint32_t(dst) << 6 | (std::uint32_t(base) >> 1), 125u << 25 | off};
    };
    // image_sample_lz v4, v[2:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
    const W2 image_sample_lz = {
        0x3Cu << 26 | 0x21u << 18 | 1u << 8 | 1u << 3,  // dmask=1, dim=2D(1)
        (16u >> 2) << 21 | (8u >> 2) << 16 | 4u << 8 | 2u,
    };
    // buffer_store_dword v4, off, s[20:23], 0
    const W2 buf_store = {0x38u << 26 | 0x1Cu << 18, 0x80u << 24 | (20u >> 2) << 16 | 4u << 8};

    // Two distinct valid immediate sampler descriptors (different wrap/filter bits).
    const std::array<std::uint32_t, 4> samp_a = {0x0000'0092, 0x00C0'0000, 0x1850'0000, 0};
    const std::array<std::uint32_t, 4> samp_b = {0x0000'0092, 0x00C0'0000, 0x1850'0001, 0};

    const std::pair<const char*, std::array<std::uint32_t, 4>> cases[] = {{"same_sampler", samp_a},
                                                                          {"different_samplers", samp_b}};
    for (const auto& [case_name, alt] : cases) {
        std::vector<std::uint32_t> words;
        auto extend = [&](const W2& w) { words.insert(words.end(), w.begin(), w.end()); };
        // s[2:3] = view table base (s0, 0).
        words.push_back(0xBE80'0300u | (2u << 16));  // s_mov_b32 s2, s0
        words.push_back(s_mov_zero(3));
        extend(s_load_x8(8, 2));           // s[8:15] = texture SRV (offset 0)
        extend(s_load_x4_off(20, 2, 64));  // s[20:23] = raw UAV (offset 16 dwords = 64B)
        // v2 = 0.5, v3 = 0.5 (sample coordinates).
        words.push_back(0x7E00'02F0u | (2u << 17));  // v_mov_b32 v2, 0.5
        words.push_back(0x7E00'02F0u | (3u << 17));  // v_mov_b32 v3, 0.5
        // s_cmp_eq_u32 s1, 0; s_cbranch_scc0 else_arm
        words.push_back(0xBF00'8001u);  // s_cmp_eq_u32 s1, 0 (const 128)
        std::size_t br_else = words.size();
        words.push_back(0);  // placeholder s_cbranch_scc0
        // then arm: load samp_a into s[16:19]
        for (std::size_t i = 0; i < samp_a.size(); ++i) extend(s_mov_lit(std::uint8_t(16 + i), samp_a[i]));
        std::size_t br_join = words.size();
        words.push_back(0);  // placeholder s_branch
        std::size_t else_pc = words.size();
        // else arm: load alt into s[16:19]
        for (std::size_t i = 0; i < alt.size(); ++i) extend(s_mov_lit(std::uint8_t(16 + i), alt[i]));
        std::size_t join_pc = words.size();
        words[br_else] = 0xBF84'0000u | (std::uint32_t(else_pc - br_else - 1) & 0xFFFF);
        words[br_join] = 0xBF82'0000u | (std::uint32_t(join_pc - br_join - 1) & 0xFFFF);
        extend(image_sample_lz);
        extend(buf_store);
        words.push_back(0xBF81'0000u);  // s_endpgm

        NativeProgram native;
        native.stage = HwStage::Cs;
        native.wave64 = false;
        native.vgprs = 8;
        native.layout = {root_param(RootType::Table(), 0, 1, 0, 0, 0), root_param(RootType::Constants(), 0, 1, 0, 1, 1)};
        native.used = {used_resource(ResourceType::Srv(), RootType::Table(), 0, 0, 0, 8, 0, 0),
                       used_resource(ResourceType::Uav(), RootType::Table(), 16, 0, 0, 4, 0, 0)};
        native.io = compute_io(2 << 1, {1, 1, 1});
        native.code = le_bytes(words);

        vs::ir::Program p = vs::ir::build(native.code, false, native.vgprs);
        vs::structure::Structured st = vs::structure::structure(p);
        vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
        vs::io::Io io = vs::io::raise(p, native);
        Lowered lowered;
        try {
            lowered = vs::lower::lower(parts_of(native, p, st, io, res));
        } catch (const ShaderError& e) {
            fail(std::string(case_name) + ": lower failed: " + e.to_string());
        }

        std::vector<vs::StaticSampler> static_samplers;
        for (const auto& w : lowered.samplers) static_samplers.push_back(vs::StaticSampler::from_words(w));
        HostShader shader = host_shader(std::move(lowered), {}, std::move(static_samplers));
        validate(shader);

        std::lock_guard one(one_at_a_time());
        const std::pair<const char*, bool> devices[] = {{"hardware", false}, {"warp", true}};
        for (auto [label, warp] : devices) {
            auto host = new_host(warp);
            if (!host) continue;
            std::string why;
            auto pipe = compute_pipeline(*host, shader, &why);
            if (!pipe) fail(std::string(case_name) + "/" + label + ": compute_pipeline failed: " + why);
            for (std::uint32_t branch : {0u, 1u}) {
                std::vector<std::uint32_t> ud(std::max<std::uint32_t>(shader.user_data_dwords(), 1), 0);
                for (const UserData& u : shader.user_data) {
                    if (u.root == RootType::Table() && u.slot == 0)
                        ud[u.index] = 0;
                    else if (u.root == RootType::Constants() && u.slot == 1)
                        ud[u.index] = branch;
                    else
                        fail("unexpected user data");
                }
                const Texel texels[] = {{1.0f, 0.0f, 0.0f, 1.0f}};
                const View views[] = {texture(1, 1, texels), texture(1, 1, texels), raw_uav(1)};
                auto out = dispatch(*host, *pipe, std::move(ud), views, {}, {1, 1, 1},
                                    std::string(case_name) + "/" + label);
                float got = f32_from_bits(out[0][0]);
                require(got == 1.0f, std::string(case_name) + "/" + label + " branch=" + std::to_string(branch));
            }
        }
    }
}

}  // namespace

int main() {
    guarded("lanes_sample_the_texture_their_if_else_arm_chose", lanes_sample_the_texture_their_if_else_arm_chose);
    guarded("lanes_sample_the_texture_their_arm_loaded_or_picked", lanes_sample_the_texture_their_arm_loaded_or_picked);
    guarded("lanes_load_from_the_buffer_their_nested_arms_chose", lanes_load_from_the_buffer_their_nested_arms_chose);
    guarded("lanes_sample_the_entry_their_offset_names_in_the_console_waterfall",
            lanes_sample_the_entry_their_offset_names_in_the_console_waterfall);
    guarded("lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx",
            lanes_sample_their_own_texture_in_a_waterfall_narrowed_by_v_cmpx);
    guarded("immediate_sampler_chosen_by_control_flow", immediate_sampler_chosen_by_control_flow);
    return test_util::finish("lower_waterfall_test");
}
