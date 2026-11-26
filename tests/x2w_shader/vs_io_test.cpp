// Port of the upstream Rust shader tests in src/io.rs.
#include "x2w_shader/io.h"

#include "vs_front_test.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

using namespace x2w::vs;
using namespace x2w::vs::io;
using native::HwStage;
using native::NativeProgram;
using native::PsInput;
using native::Usage;
using ir::Program;
using ir::Reg;

namespace {

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

std::vector<std::uint8_t> bytes(const std::vector<std::uint32_t>& words)
{
    std::vector<std::uint8_t> code;
    for (std::uint32_t w : words)
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<std::uint8_t>(w >> (8 * i)));
    return code;
}

Program program(const std::vector<std::uint32_t>& words) { return ir::build(bytes(words), false, 256); }

NativeProgram pixel(std::vector<PsInput> inputs, std::uint32_t input_addr)
{
    NativeProgram n;
    n.stage = HwStage::Ps;
    n.wave64 = false;
    n.vgprs = 256;
    native::PsIo ps;
    ps.inputs = std::move(inputs);
    ps.input_addr = input_addr;
    ps.db_shader_control = 0;
    n.io = ps;
    return n;
}

PsInput texcoord()
{
    PsInput i;
    i.usage = Usage::Generic();
    i.row = 1;
    i.mask = 3;
    i.attr = 0;
    i.flat = false;
    i.fp16 = false;
    i.pack2 = false;
    i.row2 = 1;
    return i;
}

const Attribute ATTRIBUTE{Usage::Generic(), 1, 0, false};

bool fails(const Program& p, const NativeProgram& n)
{
    return catch_shader_error([&] { raise(p, n); }).has_value();
}

}  // namespace

VS_TEST(interpolation_pairs_and_color_exports)
{
    // Blit's pixel shader: TEXCOORD.xy from the perspective-center i/j in v0/v1, then exported
    // unconverted to MRT0.
    const Program p = program({
        0xC808'0000,  // v_interp_p1_f32 v2, v0, attr0.x
        0xC809'0001,  // v_interp_p2_f32 v2, v1, attr0.x
        0xF800'180F,
        0x0302'0102,  // exp mrt0, v2, v1, v2, v3 done vm
        S_ENDPGM,
    });
    const Io io = raise(p, pixel({texcoord()}, 2));
    auto input = [&](std::uint8_t reg) {
        for (const auto& i : p.inputs)
            if (i.first == Reg::V(reg)) return i.second;
        throw std::runtime_error("no input");
    };
    auto center = [](std::uint8_t index) { return Input::Bary(Barycentric::PerspectiveCenter, index); };
    auto has = [&](const std::pair<ir::Value, Input>& e) {
        return std::find(io.inputs.begin(), io.inputs.end(), e) != io.inputs.end();
    };
    VS_CHECK(has({input(0), center(0)}));
    VS_CHECK(has({input(1), center(1)}));
    // v2 and v3 are read before written but hold nothing.
    const ir::Value v3 = input(3);
    VS_CHECK(std::none_of(io.inputs.begin(), io.inputs.end(), [&](const auto& i) { return i.first == v3; }));
    const std::vector<std::pair<Site, Interp>> want{
        {{0, 1}, Interp::Interpolated(ATTRIBUTE, Barycentric::PerspectiveCenter, {{0, 0}})}};
    VS_CHECK(io.interps == want);
    VS_REQUIRE(io.exports.size() == 1);
    const Export& e = io.exports[0];
    VS_CHECK(e.target == Target::Color(0) && e.done && !e.compressed);
    VS_CHECK(e.values[0] == p.blocks[0].insts[1].write(Reg::V(2)));
}

VS_TEST(unpaired_or_unknown_interpolation_fails)
{
    // p2 alone: its old dst is no p1's result.
    const Program lone = program({0xC809'0001, S_ENDPGM});  // v_interp_p2_f32 v2, v1, attr0.x
    VS_CHECK(fails(lone, pixel({texcoord()}, 2)));
    // attr1 is not in the stage data.
    const Program other = program({0xC808'0400, S_ENDPGM});  // v_interp_p1_f32 v2, v0, attr1.x
    VS_CHECK(fails(other, pixel({texcoord()}, 2)));
}

VS_TEST(computed_barycentrics_evaluate_each_half)
{
    // i scaled before the first half: both halves stand on their own.
    const Program p = program({
        0x1006'00F0,  // v_mul_f32 v3, 0.5, v0
        0xC808'0003,  // v_interp_p1_f32 v2, v3, attr0.x
        0xC809'0001,  // v_interp_p2_f32 v2, v1, attr0.x
        S_ENDPGM,
    });
    const Io io = raise(p, pixel({texcoord()}, 2));
    auto interps = io.interps;
    std::stable_sort(interps.begin(), interps.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const std::vector<std::pair<Site, Interp>> want{
        {{0, 1}, Interp::Weighted(ATTRIBUTE)},
        {{0, 2}, Interp::Weighted(ATTRIBUTE)},
    };
    VS_CHECK(interps == want);
}

VS_TEST(first_halves_read_through_a_phi_keep_their_partial_values)
{
    const Program p = program({
        0xBF06'8000,  // s_cmp_eq_u32 s0, 0
        0xBF85'0002,  // s_cbranch_scc1 -> second first half
        0xC808'0000,  // v_interp_p1_f32 v2, v0, attr0.x
        0xBF82'0001,  // s_branch -> export
        0xC808'0000,  // v_interp_p1_f32 v2, v0, attr0.x
        0xF800'180F,
        0x0202'0202,  // exp mrt0, v2, v2, v2, v2 done vm
        S_ENDPGM,
    });
    const Io io = raise(p, pixel({texcoord()}, 2));
    VS_CHECK(io.interps.size() == 2);
    VS_CHECK(std::all_of(io.interps.begin(), io.interps.end(),
                         [](const auto& i) { return i.second.kind == Interp::Kind::Weighted; }));
}

VS_TEST(a_standalone_f16_first_half_is_weighted)
{
    const Program p = program({
        0xD742'0002,
        0x0002'0000,  // v_interp_p1ll_f16 v2, v0, attr0.x
        0xF800'180F,
        0x0202'0202,
        S_ENDPGM,
    });
    PsInput in = texcoord();
    in.fp16 = true;
    const Io io = raise(p, pixel({in}, 2));
    VS_CHECK(io.interps.size() == 1 && io.interps[0].second.kind == Interp::Kind::Weighted);
}

VS_TEST(mesh_inputs_and_primitive_export)
{
    // Rust's `mesh(..)`: Ok(io) or Err.
    auto mesh = [](const std::vector<std::uint32_t>& words) -> std::optional<Io> {
        const auto code = bytes(words);
        const Program p = ir::build(code, true, 256);
        NativeProgram n;
        n.stage = HwStage::Ms;
        n.wave64 = true;
        n.vgprs = 256;
        native::MeshIo ms;
        ms.clip_cntl = 0;
        ms.threads = 128;
        ms.max_vertices = 128;
        ms.max_primitives = 128;
        ms.topology = 2;
        ms.lds_bytes = 0;
        n.io = ms;
        n.code = code;
        return try_shader([&] { return raise(p, n); });
    };
    const auto io = mesh({
        0x7E04'0505,  // v_readfirstlane_b32 s2, v5
        0xF800'0941,
        0x0000'0009,  // exp prim, v9, off, off, off done
        S_ENDPGM,
    });
    VS_REQUIRE(io.has_value());
    VS_CHECK(std::any_of(io->inputs.begin(), io->inputs.end(),
                         [](const auto& i) { return i.second == Input::Simple(Input::Kind::MeshGroup); }));
    VS_REQUIRE(!io->exports.empty());
    VS_CHECK(io->exports[0].target == Target::Simple(Target::Kind::Primitive));
    // The group input is uniform only as the first lane reads it.
    VS_CHECK(!mesh({0x7E02'0305, S_ENDPGM}).has_value());  // v_mov_b32 v1, v5
    // Also as an operand of a write to v5, or after a join.
    const std::vector<std::vector<std::uint32_t>> per_lane = {
        {0x4A0A'0305, S_ENDPGM},               // v_add_nc_u32 v5, v5, v1
        {0x7E0A'0D05, S_ENDPGM},               // v_cvt_f32_u32 v5, v5
        {0xDC30'8000, 0x0500'0005, S_ENDPGM},  // global_load_dword v5, v5, s[0:1]
        {0x7E0A'CB01, S_ENDPGM},               // v_swap_b32 v5, v1
        {
            0xBF06'8001,  // s_cmp_eq_u32 s1, 0
            0xBF85'0001,  // s_cbranch_scc1 -> 0xc
            0x7E0A'0280,  // v_mov_b32 v5, 0
            0x7E02'0305,  // 0xc: v_mov_b32 v1, v5
            S_ENDPGM,
        },
    };
    for (std::size_t k = 0; k < per_lane.size(); ++k) {
        const bool accepted = mesh(per_lane[k]).has_value();
        if (accepted) std::fprintf(stderr, "accepted per_lane[%zu]\n", k);
        VS_CHECK(!accepted);
    }
    // Overwriting v5 reads the group input only for lanes outside EXEC: v_mov_b32 v5, 0;
    // v_mov_b32 v1, v5.
    VS_CHECK(mesh({0x7E0A'0280, 0x7E02'0305, S_ENDPGM}).has_value());
    // Skipping that write with s_cbranch_execz is the same case: under the same mask, the read sees
    // only the 0.
    const std::vector<std::uint32_t> reuse = {
        0x7E04'0505,  // v_readfirstlane_b32 s2, v5
        0x7D88'0084,  // v_cmp_gt_u32 vcc, 4, v0
        0xBE86'246A,  // s_and_saveexec_b64 s[6:7], vcc
        0xBF88'0001,  // s_cbranch_execz -> 0x14
        0x7E0A'0280,  // v_mov_b32 v5, 0
        0x88FE'067E,  // 0x14: s_or_b64 exec, exec, s[6:7]
        0xBE86'246A,  // s_and_saveexec_b64 s[6:7], vcc
        0xBF88'0001,  // s_cbranch_execz -> 0x24
        0x7E02'0305,  // v_mov_b32 v1, v5
        0x88FE'067E,  // 0x24: s_or_b64 exec, exec, s[6:7]
        S_ENDPGM,
    };
    VS_CHECK(mesh(reuse).has_value());
}

VS_TEST(pixel_vgprs_follow_the_input_address)
{
    // Linear center (bit 5), then position x and y (bits 8, 9).
    const auto v = detail::ps_vgprs((1u << 5) | (1u << 8) | (1u << 9));
    const std::vector<std::pair<std::uint8_t, Input>> want{
        {std::uint8_t{0}, Input::Bary(Barycentric::LinearCenter, 0)},
        {std::uint8_t{1}, Input::Bary(Barycentric::LinearCenter, 1)},
        {std::uint8_t{2}, Input::Position(0)},
        {std::uint8_t{3}, Input::Position(1)},
    };
    VS_CHECK(v == want);
}

VS_TEST_MAIN()
