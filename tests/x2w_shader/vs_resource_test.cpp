// Port of the upstream Rust shader tests in src/resource.rs.
#include "x2w_shader/resource.h"

#include "resource_fixtures.h"
#include "vs_front_test.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace x2w::vs;
using namespace x2w::vs::native;
using namespace x2w::vs::resource;
using fixtures::cmpx_waterfall;
using fixtures::WATERFALL_FETCHED_WAVE32;
using fixtures::WATERFALL_FETCHED_WAVE64;
using fixtures::WATERFALL_INDEXED_WAVE32;
using fixtures::WATERFALL_JOINED_BUFFER_WAVE32;
using fixtures::WATERFALL_JOINED_MIXED_WAVE32;
using fixtures::WATERFALL_JOINED_WAVE32;
using fixtures::WATERFALL_LOOPED_WAVE32;
using fixtures::WATERFALL_PICKED_WAVE32;
using ir::Program;
using Words = std::vector<std::uint32_t>;
using Scalar = std::vector<std::pair<Site, Address>>;

namespace {

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

std::vector<std::uint8_t> bytes(std::span<const std::uint32_t> words)
{
    std::vector<std::uint8_t> code;
    for (std::uint32_t w : words)
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<std::uint8_t>(w >> (8 * i)));
    return code;
}

// Rust `[a, b, ..].concat()` of word slices.
Words concat(std::initializer_list<std::span<const std::uint32_t>> parts)
{
    Words w;
    for (const auto& part : parts) w.insert(w.end(), part.begin(), part.end());
    return w;
}

// `p` as one lane runs it, if it structures.
std::optional<structure::Structured> lanes(const Program& p)
{
    return try_shader([&] { return structure::structure(p); });
}

Resources track_words(const Words& words, const std::vector<RootParam>& layout, HwStage stage)
{
    const Program p = ir::build(bytes(words), false, 256);
    const auto l = lanes(p);
    return track(p, layout, stage, l ? &*l : nullptr);
}

RootParam param(RootType root, std::uint8_t user_data, std::uint32_t sgprs)
{
    RootParam r;
    r.root = root;
    r.table_offset_dwords = 0;
    r.sgprs = sgprs;
    r.base_register = 0;
    r.user_data = user_data;
    r.slot = 0;
    return r;
}

Address at(std::size_t param, std::uint32_t offset)
{
    Address a;
    a.param = param;
    a.offset = offset;
    a.dynamic = std::nullopt;
    return a;
}

// The address of the scalar access at `load` (Rust: `r.scalar.iter().find(|s| s.0 == load).unwrap().1`).
Address scalar_at(const Resources& r, Site load)
{
    const auto it = std::find_if(r.scalar.begin(), r.scalar.end(), [&](const auto& s) { return s.first == load; });
    VS_REQUIRE(it != r.scalar.end());
    return it->second;
}

// Rust `ShaderError::Unsupported { offset, what }` (`offset` nullopt: `..`), `what` by content.
bool is_unsupported(const std::optional<ShaderError>& e, std::optional<std::size_t> offset, std::string_view what)
{
    if (!e) {
        std::fprintf(stderr, "no error\n");
        return false;
    }
    const bool ok = e->kind == ShaderError::Kind::Unsupported && (!offset || e->offset == *offset) &&
                    std::string_view(e->reason) == what;
    if (!ok) std::fprintf(stderr, "%s\n", e->debug().c_str());
    return ok;
}

std::size_t moves_anywhere(const Words& words);

}  // namespace

VS_TEST(table_descriptors_are_loaded_from_their_range)
{
    const Resources r = track_words(
        {
            0xBE82'0309, // s_mov_b32 s2, s9
            0xBE89'0380, // s_mov_b32 s9, 0
            0xBE83'0380, // s_mov_b32 s3, 0
            0xF408'0204,
            0xFA00'0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C'0001,
            0xFA00'0000, // s_load_dwordx8 s[0:7], s[2:3], null
            0xF080'0F08,
            0x0040'0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
            S_ENDPGM,
        },
        {param(RootType::Table(), 8, 1), param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_CHECK((r.scalar == Scalar{{{0, 3}, at(0, 0)}, {{0, 4}, at(1, 0)}}));
    std::vector<std::pair<Role, Source>> roles;
    for (const auto& d : r.descriptors) roles.emplace_back(d.role, d.source);
    VS_CHECK((roles == std::vector<std::pair<Role, Source>>{
                           {Role::Resource, Source::Loaded({0, 4}, 0)},
                           {Role::Sampler, Source::Loaded({0, 3}, 0)},
                       }));
}

VS_TEST(root_descriptors_are_built_in_code)
{
    const Resources r = track_words(
        {
            0x8801'FF0B,
            0x0010'0000, // s_or_b32 s1, s11, 0x100000
            0xBE80'030A, // s_mov_b32 s0, s10
            0xBE82'03C1, // s_mov_b32 s2, -1
            0xBE83'03FF,
            0x1101'4FAC, // s_mov_b32 s3, 0x11014fac
            0xF430'0000,
            0xFA00'0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
            S_ENDPGM,
        },
        {param(RootType::Descriptor(), 10, 2)}, HwStage::Vs);
    VS_REQUIRE(!r.descriptors.empty());
    VS_CHECK((r.descriptors[0].source == Source::Built(at(0, 0), 0x10'0000, {UINT32_MAX, 0x1101'4FAC})));
    std::vector<std::pair<std::size_t, std::uint32_t>> params;
    for (const auto& p : r.params) params.emplace_back(std::get<1>(p), std::get<2>(p));
    VS_CHECK((params == std::vector<std::pair<std::size_t, std::uint32_t>>{{0, 0}, {0, 1}}));
}

VS_TEST(heap_indices_from_root_constants_in_a_merged_stage)
{
    // User data starts at s8: the constant (index 6) is s14, the heap
    // (index 10) s18.
    const Words code = {
        0x8F08'850E, // s_lshl_b32 s8, s14, 5
        0xBE93'0380, // s_mov_b32 s19, 0
        0xF408'0209,
        0x1000'0000, // s_load_dwordx4 s[8:11], s[18:19], s8
        0xE000'2000,
        0x8002'0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
        S_ENDPGM,
    };
    const Resources r =
        track_words(code, {param(RootType::Constants(), 6, 1), param(RootType::ResourceHeap(), 10, 1)}, HwStage::Ms);
    const auto found = std::find_if(r.params.begin(), r.params.end(), [](const auto& p) { return std::get<1>(p) == 0; });
    VS_REQUIRE(found != r.params.end());
    const ir::Value index = std::get<0>(*found);
    VS_REQUIRE(!r.scalar.empty());
    VS_REQUIRE(r.scalar[0].second.dynamic.has_value());
    const ir::Value shift = *r.scalar[0].second.dynamic;
    VS_CHECK(r.scalar[0].second.param == 1);
    // The computed term is the shifted constant, not the constant itself.
    VS_CHECK(shift != index);
    VS_REQUIRE(!r.descriptors.empty());
    VS_CHECK((r.descriptors[0].source == Source::Loaded({0, 2}, 0)));
    VS_CHECK(r.root_indexed_views.empty());
    RootParam table = param(RootType::Table(), 10, 1);
    table.slot = 7;
    const Resources indexed = track_words(code, {param(RootType::Constants(), 6, 1), table}, HwStage::Ms);
    VS_CHECK((indexed.root_indexed_views == std::vector<RootIndexedView>{RootIndexedView{
                                                .table_slot = 7,
                                                .constant_slot = 0,
                                                .constant_dword = 0,
                                                .byte_offset = 0,
                                                .shift = 5,
                                            }}));
    const Resources nonconstant = track_words(code, {param(RootType::Table(), 6, 1), table}, HwStage::Ms);
    VS_CHECK(nonconstant.root_indexed_views.empty());
}

VS_TEST(spilled_sgprs_are_followed_through_vgpr_lanes)
{
    const Resources r = track_words(
        {
            0xD761'0000,
            0x0001'0609, // v_writelane_b32 v0, s9, 3
            0xD760'0002,
            0x0001'0700, // v_readlane_b32 s2, v0, 3
            0xBE83'0380, // s_mov_b32 s3, 0
            0xF40C'0001,
            0xFA00'0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
            S_ENDPGM,
        },
        {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_CHECK((r.scalar == Scalar{{{0, 3}, at(0, 0x20)}}));
}

// A VGPR a wave reduction used is free again after it: from the first
// save on it holds parked scalars alone (an upscaler program parks a
// constant buffer's descriptor that way).
VS_TEST(a_vgpr_used_before_holds_spilled_sgprs_from_its_first_save)
{
    const Words before = {
        0x7E00'0281, // v_mov_b32 v0, 1
        0x7E02'0300, // v_mov_b32 v1, v0
    };
    const Words park = {
        0xD761'0000,
        0x0001'0609, // v_writelane_b32 v0, s9, 3
        0xD760'0002,
        0x0001'0700, // v_readlane_b32 s2, v0, 3
        0xBE83'0380, // s_mov_b32 s3, 0
    };
    const Words load = {
        0xF40C'0001,
        0xFA00'0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
        S_ENDPGM,
    };
    const std::span<const std::uint32_t> b(before), k(park), l(load);
    const Words words = concat({b, k, l});
    const Resources r = track_words(words, {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_CHECK((r.scalar == Scalar{{{0, 5}, at(0, 0x20)}}));
    // Still lane reads: a lane restored before it is saved, and a
    // register with an ordinary use after the first save.
    const Words restored_first = concat({b, k.subspan(2, 2), k.first(2), l});
    const Words used_after = concat({b, k, b.subspan(1), l});
    for (const Words& w : {restored_first, used_after}) {
        const Program p = ir::build(bytes(w), true, 8);
        std::size_t n = 0;
        for (const auto& blk : p.blocks)
            for (const auto& i : blk.insts) n += i.name() == "v_readlane_b32";
        VS_CHECK(n == 1);
    }
}

// Two lanes hold a pair (a lane mask, say) and later one of them a
// scalar, as in a water volume program that parks more than the lanes
// of one VGPR hold: the pair still moves as one.
VS_TEST(a_lane_spilled_to_again_later_does_not_split_an_earlier_pair)
{
    const Words pair = {
        0xD761'0000,
        0x0001'0004, // v_writelane_b32 v0, s4, 0
        0xD761'0000,
        0x0001'0205, // v_writelane_b32 v0, s5, 1
        0xBE84'0380, // s_mov_b32 s4, 0
        0xBE85'0380, // s_mov_b32 s5, 0
        0xD760'0004,
        0x0001'0100, // v_readlane_b32 s4, v0, 0
        0xD760'0005,
        0x0001'0300, // v_readlane_b32 s5, v0, 1
    };
    const Words scalar = {
        0xD761'0000,
        0x0001'0006, // v_writelane_b32 v0, s6, 0
        0xBE86'0380, // s_mov_b32 s6, 0
        0xD760'0006,
        0x0001'0100, // v_readlane_b32 s6, v0, 0
        S_ENDPGM,
    };
    const auto moves = [](const Words& words) {
        const Program p = ir::build(bytes(words), true, 8);
        std::vector<std::string_view> names;
        for (const auto& i : p.blocks[0].insts) names.push_back(i.name());
        VS_CHECK(std::none_of(names.begin(), names.end(),
                              [](std::string_view n) { return n.find("lane") != std::string_view::npos; }));
        return static_cast<std::size_t>(std::count(names.begin(), names.end(), std::string_view("s_mov_b64")));
    };
    const std::span<const std::uint32_t> pr(pair), sc(scalar);
    VS_CHECK(moves(concat({pr, sc})) == 2);
    // A restore both saves reach keeps the lane one scalar.
    const Words branch = {0xBF84'0002}; // s_cbranch_scc0 past the second save
    const Words joined = concat({
        pr.first(4),
        branch,
        sc.first(2),
        pr.subspan(4),
        sc.subspan(2),
    });
    VS_CHECK(moves_anywhere(joined) == 0);
}

namespace {

std::size_t moves_anywhere(const Words& words)
{
    const Program p = ir::build(bytes(words), true, 8);
    std::size_t n = 0;
    for (const auto& b : p.blocks)
        for (const auto& i : b.insts) n += i.name() == "s_mov_b64";
    return n;
}

}  // namespace

VS_TEST(descriptors_joined_by_control_flow_are_traced_per_edge)
{
    const Resources r = track_words(
        {
            0xBE82'0309, // s_mov_b32 s2, s9
            0xBE83'0380, // s_mov_b32 s3, 0
            0xBF06'8001, // s_cmp_eq_u32 s1, 0
            0xBF85'0003, // s_cbranch_scc1 -> 0x1c
            0xF408'0001,
            0xFA00'0000, // s_load_dwordx4 s[0:3], s[2:3], null
            0xBF82'0002, // s_branch -> 0x24
            0xF408'0001,
            0xFA00'0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
            0xF420'0100,
            0xFA00'0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
            S_ENDPGM,
        },
        {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_REQUIRE(!r.descriptors.empty());
    VS_REQUIRE(r.descriptors[0].source.kind == Source::Kind::Phi);
    std::vector<std::uint32_t> offsets;
    for (const Source& a : r.descriptors[0].source.args) {
        VS_REQUIRE(a.kind == Source::Kind::Loaded);
        offsets.push_back(scalar_at(r, a.load).offset);
    }
    VS_CHECK(offsets.size() == 2);
    VS_CHECK(std::find(offsets.begin(), offsets.end(), 0u) != offsets.end() &&
             std::find(offsets.begin(), offsets.end(), 0x10u) != offsets.end());
}

// Two paths add an index to one table in the code and meet before one load; the join is the computed term, counted
// from the `s_getpc_b64`.
VS_TEST(two_paths_into_one_code_table_load)
{
    const Resources r = track_words(
        {
            0xBE82'1F00, // 0x00: s_getpc_b64 s[2:3]
            0x8002'FF02,
            0x0000'0040, // s_add_u32 s2, s2, 0x40
            0x8203'8003, // s_addc_u32 s3, s3, 0
            0xBF06'8001, // 0x10: s_cmp_eq_u32 s1, 0
            0xBF85'0005, // 0x14: s_cbranch_scc1 -> 0x2c
            0xD70F'0457,
            0x0000'054A, // 0x18: v_add_co_u32 v87, s4, v74, s2
            0xD528'0458,
            0x0010'0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0xBF82'0004, // 0x28: s_branch -> 0x3c
            0xD70F'0457,
            0x0000'054B, // 0x2c: v_add_co_u32 v87, s4, v75, s2
            0xD528'0458,
            0x0010'0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0xDC34'8000,
            0x577D'0057, // 0x3c: global_load_dwordx2 v[87:88], v[87:88], null
            S_ENDPGM,
        },
        {}, HwStage::Cs);
    VS_REQUIRE(r.scalar.size() == 1);
    const Address a = r.scalar[0].second;
    VS_CHECK(a.param == CODE && a.offset == 0);
    // The join is the computed term, the byte in the code.
    VS_CHECK(a.dynamic.has_value());
}

// An address into a table in the code computed per lane and taken from
// the first lane (`avgLuminanceCompute` `ColorDownsample`): the per-lane
// sum carries the `s_getpc_b64` base, so the first-lane read is looked
// through and the lane's index is the computed term. (A first-lane read
// of an index alone stays the computed term itself; see
// `the_console_waterfall_over_a_first_lane_offset_is_one`.)
VS_TEST(a_code_table_address_taken_from_the_first_lane)
{
    const Resources r = track_words(
        {
            0xBE82'1F00, // 0x00: s_getpc_b64 s[2:3]
            0xD70F'0457,
            0x0000'054A, // v_add_co_u32 v87, s4, v74, s2
            0xD528'0458,
            0x0010'0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0x7E0C'0557, // v_readfirstlane_b32 s6, v87
            0x7E0E'0558, // v_readfirstlane_b32 s7, v88
            0xF400'0203,
            0xFA00'0000, // s_load_dword s8, s[6:7], 0x0
            S_ENDPGM,
        },
        {}, HwStage::Cs);
    VS_REQUIRE(r.scalar.size() == 1);
    const Address a = r.scalar[0].second;
    VS_CHECK(a.param == CODE && a.offset == 4);
    // The lane's index is the computed term.
    VS_CHECK(a.dynamic.has_value());
}

VS_TEST(a_descriptor_kept_per_lane_is_its_load_and_needs_no_waterfall)
{
    // The console compiler's way with a descriptor it joins per lane:
    // the dwords go to VGPRs and come back from the first lane. Every
    // lane holds the one load's dwords, so the descriptor is that load
    // (`lane_copied`), and no waterfall is asked for.
    Words words = {
        0xBE82'0309, // s_mov_b32 s2, s9
        0xBE83'0380, // s_mov_b32 s3, 0
        0xF40C'0001,
        0xFA00'0000, // s_load_dwordx8 s[0:7], s[2:3], null
    };
    // v_mov_b32 v10.., s0..; v_readfirstlane_b32 s16.., v10..
    for (std::uint32_t k = 0; k < 8; ++k) words.push_back(0x7E00'0200 | ((10 + k) << 17) | k);
    for (std::uint32_t k = 0; k < 8; ++k) words.push_back(0x7E00'0400 | ((16 + k) << 17) | (0x100 + 10 + k));
    words.insert(words.end(), {
                                  0xF000'1700,
                                  0x0004'0D06, // image_load v[13:15], v6, s[16:23]
                                  S_ENDPGM,
                              });
    const Resources r = track_words(words, {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    const Site load = {0, 2};
    VS_CHECK((r.scalar == Scalar{{load, at(0, 0)}}));
    std::vector<Source> sources;
    for (const auto& d : r.descriptors) sources.push_back(d.source);
    VS_CHECK((sources == std::vector<Source>{Source::Loaded(load, 0)}));
    VS_CHECK((r.lane_copied == std::vector<Site>{load}));
    VS_CHECK(r.waterfall == Waterfall{});
}

VS_TEST(loads_at_one_first_lane_offset_have_one_address)
{
    // Two loads of the table entry at the offset v4 holds, each with
    // its own first-lane read, as the console compiler loads a
    // descriptor again wherever it needs it: one address, by which
    // lowering knows them for the same descriptor.
    const Resources r = track_words(
        {
            0xBE82'0309, // s_mov_b32 s2, s9
            0xBE83'0380, // s_mov_b32 s3, 0
            0x7E08'0504, // v_readfirstlane_b32 s4, v4
            0xF40C'0201,
            0x0800'0000, // s_load_dwordx8 s[8:15], s[2:3], s4
            0x7E0A'0504, // v_readfirstlane_b32 s5, v4
            0xF40C'0401,
            0x0A00'0000, // s_load_dwordx8 s[16:23], s[2:3], s5
            0xF000'1700,
            0x0004'0D06, // image_load v[13:15], v6, s[16:23]
            S_ENDPGM,
        },
        {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_REQUIRE(r.scalar.size() == 2);
    const Address a = r.scalar[0].second;
    const Address b = r.scalar[1].second;
    VS_CHECK(a == b);
    VS_CHECK(a.param == 0 && a.offset == 0);
    // The lane's offset is the computed term.
    VS_CHECK(a.dynamic.has_value());
}

VS_TEST(descriptors_selected_by_scc_are_traced_per_outcome)
{
    // Two buffer descriptors loaded from the table at s9 and one
    // picked with s_cselect_b64 per half, as the compiler makes a
    // conditional choice of resources without a branch.
    const Resources r = track_words(
        {
            0xBE8C'0309, // s_mov_b32 s12, s9
            0xBE8D'0380, // s_mov_b32 s13, 0
            0xF408'0006,
            0xFA00'0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408'0106,
            0xFA00'0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06'800E, // s_cmp_eq_u32 s14, 0
            0x8588'0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A'0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
            0xF420'0404,
            0xFA00'0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        },
        {param(RootType::Table(), 9, 1)}, HwStage::Ps);
    VS_REQUIRE(!r.descriptors.empty());
    VS_REQUIRE(r.descriptors[0].source.kind == Source::Kind::Select);
    const std::vector<Source>& args = r.descriptors[0].source.args;
    // SCC set picks the first operand.
    VS_CHECK((args == std::vector<Source>{
                          Source::Loaded({0, 2}, 0),
                          Source::Loaded({0, 3}, 0),
                      }));
    VS_CHECK((r.scalar == Scalar{{{0, 2}, at(0, 0)}, {{0, 3}, at(0, 0x10)}}));
}

// An upscaler's byte accesses make a loaded buffer
// descriptor raw first (stride cleared, records times stride, bounds
// in bytes): the access reads the loaded descriptor. Any other change
// still fails.
VS_TEST(a_descriptor_made_raw_is_the_loaded_one)
{
    const auto words = [](std::uint32_t mask) -> std::vector<std::uint8_t> {
        const Words w = {
            0xBE88'0304, // s_mov_b32 s8, s4
            0xBE89'0380, // s_mov_b32 s9, 0
            0xF408'0204,
            0xFA00'0160, // s_load_dwordx4 s[8:11], s[8:9], 0x160
            0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
            0x938C'FF09,
            0x000E'0010, // s_bfe_u32 s12, s9, 0x000e0010
            0x848C'810C, // s_max_u32 s12, s12, 1
            0x8709'FF09,
            mask, // s_and_b32 s9, s9, mask
            0x870B'FF0B,
            0xCFFF'FFFF, // s_and_b32 s11, s11, 0xcfffffff
            0x930A'0C0A, // s_mul_i32 s10, s10, s12
            0x880B'FF0B,
            0x3000'0000, // s_or_b32 s11, s11, 0x30000000
            0xE078'1000,
            0x8002'0007, // buffer_store_dwordx4 v[0:3], v7, s[8:11], 0 offen
            S_ENDPGM,
        };
        return bytes(w);
    };
    const auto run = [&](std::uint32_t mask) {
        return catch_shader_error([&] {
            const Program p = ir::build(words(mask), false, 256);
            const std::vector<RootParam> layout = {param(RootType::Table(), 4, 1)};
            const auto l = lanes(p);
            return track(p, layout, HwStage::Cs, l ? &*l : nullptr);
        });
    };
    VS_CHECK(!run(0xC000'FFFF).has_value());
    VS_CHECK(is_unsupported(run(0xC000'FFFE), 0x3c, "a descriptor changed after loading"));
}

// Construct immediate words from a bit mask and a wide literal move.
VS_TEST(sampler_words_from_bit_masks_and_wide_literals_are_constants)
{
    const Words w = {
        0xB010'0007, // s_movk_i32 s16, 7
        0x9211'8583, // s_bfm_b32 s17, 3, 5
        0xBE92'04FF,
        0x1234'5678, // s_mov_b64 s[18:19], 0x12345678
        S_ENDPGM,
    };
    const std::vector<std::uint8_t> code = bytes(w);
    const Program p = ir::build(code, false, 256);
    const std::vector<RootParam> layout = {param(RootType::Table(), 4, 1)};
    const std::vector<resource::detail::Term> t = resource::detail::origins(p, layout, HwStage::Cs);
    const ir::Block& block = p.blocks[0];
    const auto value = [&](std::size_t n, std::size_t k) {
        const std::uint32_t v = block.insts.at(n).writes.at(k).second.n;
        VS_REQUIRE(v < t.size());
        return t[v];
    };
    VS_CHECK(value(1, 0) == resource::detail::Term::Const(7u << 5));
    VS_CHECK(value(2, 0) == resource::detail::Term::Const(0x1234'5678));
    VS_CHECK(value(2, 1) == resource::detail::Term::Const(0));
}

VS_TEST(a_selection_that_mixes_two_descriptors_fails)
{
    // The high half picked from the first load whatever SCC is: when
    // clear, the descriptor is half of one load and half of the other.
    const Words w = {
        0xBE8C'0309, // s_mov_b32 s12, s9
        0xBE8D'0380, // s_mov_b32 s13, 0
        0xF408'0006,
        0xFA00'0000, // s_load_dwordx4 s[0:3], s[12:13], null
        0xF408'0106,
        0xFA00'0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
        0xBF06'800E, // s_cmp_eq_u32 s14, 0
        0x8588'0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
        0x858A'0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
        0xF420'0404,
        0xFA00'0000, // s_buffer_load_dword s16, s[8:11], null
        S_ENDPGM,
    };
    const std::vector<std::uint8_t> code = bytes(w);
    const Program p = ir::build(code, false, 256);
    const std::vector<RootParam> layout = {param(RootType::Table(), 9, 1)};
    const auto l = lanes(p);
    const auto err = catch_shader_error([&] { return track(p, layout, HwStage::Ps, l ? &*l : nullptr); });
    VS_CHECK(is_unsupported(err, 36, "a descriptor changed after loading"));
}

VS_TEST(untraceable_addresses_fail_with_a_reason)
{
    const Words w = {0xF408'0001, 0xFA00'0000, S_ENDPGM}; // s_load_dwordx4 s[0:3], s[2:3]
    const std::vector<std::uint8_t> code = bytes(w);
    const Program p = ir::build(code, false, 256);
    const std::vector<RootParam> layout = {param(RootType::Table(), 9, 1)};
    const auto l = lanes(p);
    const auto err = catch_shader_error([&] { return track(p, layout, HwStage::Ps, l ? &*l : nullptr); });
    VS_CHECK(is_unsupported(err, 0, "an address in an SGPR that is no root parameter"));
}

// WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
// WATERFALL_JOINED_WAVE32 and WATERFALL_INDEXED_WAVE32 are in resource_fixtures.h.

namespace {

Resources waterfall(const Words& words, bool wave64)
{
    const Program p = ir::build(bytes(words), wave64, 256);
    const std::vector<RootParam> layout = {param(RootType::Table(), 0, 1), param(RootType::Table(), 1, 1)};
    const auto l = lanes(p);
    return track(p, layout, HwStage::Cs, l ? &*l : nullptr);
}

// The ShaderError waterfall(words, wave64) throws, if any.
std::optional<ShaderError> waterfall_error(const Words& words, bool wave64)
{
    return catch_shader_error([&] { return waterfall(words, wave64); });
}

}  // namespace

VS_TEST(the_console_waterfall_over_a_first_lane_offset_is_one)
{
    const Resources r = waterfall(WATERFALL_INDEXED_WAVE32, false);
    const auto image = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                    [](const Descriptor& d) { return d.role == Role::Resource; });
    VS_REQUIRE(image != r.descriptors.end());
    VS_REQUIRE(image->source.kind == Source::Kind::Loaded && image->source.first == 0);
    const Site load = image->source.load;
    const Address address = scalar_at(r, load);
    VS_CHECK(address.dynamic.has_value());
    VS_CHECK(r.waterfall.compares.size() == 1);
    VS_CHECK(r.waterfall.reads.size() == 1);
    VS_CHECK((r.waterfall.loads == std::vector<Site>{load}));
}

// WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_LOOPED_WAVE32 and WATERFALL_JOINED_BUFFER_WAVE32 are in
// resource_fixtures.h; the `waterfall` helper that follows them in Rust is above.

VS_TEST(descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them)
{
    const std::pair<const Words*, bool> cases[] = {
        {&WATERFALL_FETCHED_WAVE32, false},
        {&WATERFALL_FETCHED_WAVE64, true},
    };
    for (const auto& [words, wave64] : cases) {
        const Resources r = waterfall(*words, wave64);
        const auto image = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                        [](const Descriptor& d) { return d.role == Role::Resource; });
        VS_REQUIRE(image != r.descriptors.end());
        VS_REQUIRE(image->source.kind == Source::Kind::Fetched && image->source.first == 0);
        const Site load = image->source.load;
        // The first of the two loads: entry v0 * 32 of the table at s0.
        const Address a = scalar_at(r, load);
        VS_CHECK(a.param == 0 && a.offset == 0);
        VS_CHECK(a.dynamic.has_value());
        // The sampler is the same for every lane.
        const auto sampler = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                          [](const Descriptor& d) { return d.role == Role::Sampler; });
        VS_REQUIRE(sampler != r.descriptors.end());
        VS_CHECK(sampler->source.kind == Source::Kind::Loaded && sampler->source.first == 0);
        // Four 64-bit compares of the first lane's dwords with the lane's
        // own; the eight reads become the lane's own dwords.
        VS_CHECK(r.waterfall.compares.size() == 4);
        VS_CHECK(r.waterfall.reads.size() == 8);
    }
}

VS_TEST(descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane)
{
    const Resources r = waterfall(WATERFALL_PICKED_WAVE32, false);
    VS_REQUIRE(!r.descriptors.empty());
    const Descriptor& image = r.descriptors[0];
    VS_REQUIRE(image.source.kind == Source::Kind::Lanes);
    // Lanes with their bit set take the first load (offset 0), the
    // others the second (0x20).
    std::vector<std::uint32_t> offsets;
    for (const Source& a : image.source.args) {
        VS_REQUIRE(a.kind == Source::Kind::Loaded && a.first == 0);
        offsets.push_back(scalar_at(r, a.load).offset);
    }
    VS_CHECK((offsets == std::vector<std::uint32_t>{0, 0x20}));
    VS_CHECK(r.waterfall.compares.size() == 4);
}

VS_TEST(a_first_lane_read_outside_a_loop_is_no_waterfall)
{
    // The waterfall without its back edge: the s_cbranch_execnz is an
    // s_nop, so lanes the first one does not match never get a pass.
    Words words = WATERFALL_FETCHED_WAVE32;
    const auto back = std::find(words.begin(), words.end(), 0xBF89'FFE6u);
    VS_REQUIRE(back != words.end());
    *back = 0xBF80'0000;
    const auto err = waterfall_error(words, false);
    VS_CHECK(is_unsupported(err, std::nullopt, "a descriptor read from the first lane outside a waterfall loop"));
}

namespace {

// Why resource tracking refuses `words`.
std::string_view refusal(const Words& words, bool wave64)
{
    const auto err = waterfall_error(words, wave64);
    if (err && err->kind != ShaderError::Kind::Unsupported)
        std::fprintf(stderr, "not refused: %s\n", err->debug().c_str());
    // Not refused otherwise.
    VS_REQUIRE(err.has_value() && err->kind == ShaderError::Kind::Unsupported);
    return err->reason;
}

constexpr std::string_view NO_LOOP = "a descriptor read from the first lane outside a waterfall loop";
constexpr std::string_view NOT_SELECTED = "a waterfall loop whose compares do not select the access's lanes";
constexpr std::string_view BAD_RETRY = "a waterfall loop whose retry does not keep the lanes still waiting";
constexpr std::string_view SPLIT = "a waterfall loop nested in another or split over blocks";

// The index of `word` in `words`.
std::size_t word_at(const Words& words, std::uint32_t word)
{
    const auto it = std::find(words.begin(), words.end(), word);
    VS_REQUIRE(it != words.end());
    return static_cast<std::size_t>(it - words.begin());
}

}  // namespace

VS_TEST(a_first_lane_read_without_its_compare_is_no_waterfall)
{
    // The loop without the compare of dwords 0 and 1: nothing makes
    // the first lane's dwords the lane's own there.
    Words words = WATERFALL_FETCHED_WAVE32;
    const std::size_t cmp = word_at(words, 0x0002'080C);
    words[cmp - 1] = 0xBF80'0000;
    words[cmp] = 0xBF80'0000;
    VS_CHECK(refusal(words, false) == NOT_SELECTED);
}

VS_TEST(a_compare_that_does_not_mask_the_access_is_no_waterfall)
{
    // The compares are all there, but their result never narrows EXEC:
    // every lane samples with the first lane's descriptor (reported by
    // ifBars on upstream PR #566).
    Words words = WATERFALL_JOINED_WAVE32;
    const std::size_t save = word_at(words, 0xBE95'3C15);
    words[save] = 0xBE95'037E; // s_mov_b32 s21, exec_lo
    VS_CHECK(refusal(words, false) == NOT_SELECTED);
}

VS_TEST(an_access_under_a_further_mask_is_no_waterfall)
{
    // The access runs for the matching lanes that VCC also holds, yet
    // the retry removes every matching lane.
    Words words = WATERFALL_JOINED_WAVE32;
    const std::size_t nop = word_at(words, 0xBE95'3C15) + 1;
    VS_REQUIRE(nop < words.size());
    VS_CHECK(words[nop] == 0xBF80'0000);
    words[nop] = 0x877E'6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
    VS_CHECK(refusal(words, false) == NOT_SELECTED);
}

VS_TEST(a_loop_that_drops_unmatched_lanes_is_no_waterfall)
{
    Words words = WATERFALL_JOINED_WAVE32;
    const std::size_t retry = word_at(words, 0x897E'157E);
    words[retry] = 0xBEFE'0380; // s_mov_b32 exec_lo, 0
    VS_CHECK(refusal(words, false) == BAD_RETRY);
}

VS_TEST(a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall)
{
    // The retry keeps the lanes of VCC that did not just sample, not
    // the waiting ones: lanes 16 and up never sample on the console.
    Words words = WATERFALL_JOINED_WAVE32;
    const std::size_t retry = word_at(words, 0x897E'157E);
    words[retry] = 0x8A7E'7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
    VS_CHECK(refusal(words, false) == BAD_RETRY);
}

VS_TEST(a_wave64_retry_of_one_half_is_no_waterfall)
{
    // Only the low half of EXEC loses the lanes just served.
    Words words = WATERFALL_FETCHED_WAVE64;
    const std::size_t retry = word_at(words, 0x89FE'167E);
    words[retry] = 0x897E'167E; // s_xor_b32 exec_lo, exec_lo, s22
    VS_CHECK(refusal(words, true) == BAD_RETRY);
}

VS_TEST(a_descriptor_used_after_exec_is_restored_is_no_waterfall)
{
    Words words = WATERFALL_JOINED_WAVE32;
    const std::size_t sample = word_at(words, 0xF09C'0F08);
    const std::size_t restore = word_at(words, 0xBEFE'0317);
    const Words image = {words[sample], words[sample + 1]};
    std::fill(words.begin() + static_cast<std::ptrdiff_t>(sample),
              words.begin() + static_cast<std::ptrdiff_t>(sample + 2), 0xBF80'0000u); // Keep the branch offsets.
    words.insert(words.begin() + static_cast<std::ptrdiff_t>(restore + 1), image.begin(), image.end());
    VS_CHECK(refusal(words, false) == NO_LOOP);
}

VS_TEST(a_wave64_access_with_exec_hi_overwritten_is_no_waterfall)
{
    Words words = WATERFALL_FETCHED_WAVE64;
    const std::size_t save = word_at(words, 0xBE96'2416);
    words[save + 1] = 0xBEFF'03C1; // s_mov_b32 exec_hi, -1
    VS_CHECK(refusal(words, true) == NOT_SELECTED);
}

// cmpx_waterfall is in resource_fixtures.h.

VS_TEST(a_waterfall_narrowed_by_v_cmpx_is_one)
{
    for (const bool wave64 : {false, true}) {
        for (const bool folded : {false, true}) {
            const Words words = cmpx_waterfall(wave64, folded);
            std::optional<Resources> r;
            const auto err = catch_shader_error([&] { r = waterfall(words, wave64); });
            if (err) std::fprintf(stderr, "wave64 %d folded %d: %s\n", wave64, folded, err->debug().c_str());
            VS_REQUIRE(r.has_value());
            if (r->waterfall.compares.size() != 4 || r->waterfall.reads.size() != 8)
                std::fprintf(stderr, "%d %d\n", wave64, folded);
            VS_CHECK(r->waterfall.compares.size() == 4);
            VS_CHECK(r->waterfall.reads.size() == 8);
        }
    }
}

VS_TEST(a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none)
{
    Words words = cmpx_waterfall(false, false);
    const std::size_t retry = word_at(words, 0x8A15'7E15);
    words[retry] = 0x8A15'7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
    VS_CHECK(refusal(words, false) == BAD_RETRY);
}

VS_TEST(a_waterfall_nested_in_another_is_refused_by_name)
{
    // An image whose resource and sampler are both per lane, with one
    // loop per operand, the one for the sampler inside the one for the
    // resource (what the SelectionDAG legalisation in LLVM would give).
    // The console gives each lane its own pair. Refused for now: the
    // access of the inner loop reads the first-lane dwords of the outer
    // loop, which another block holds.
    const Words words = {
        0xBE84'0300, // s_mov_b32 s4, s0
        0xBE85'0380, // s_mov_b32 s5, 0
        0xBE86'0301, // s_mov_b32 s6, s1
        0xBE87'0380, // s_mov_b32 s7, 0
        0xF408'0A02,
        0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04'0D00, // v_cvt_f32_u32 v2, v0
        0x7E06'0302, // v_mov_b32 v3, v2
        0x3420'0082, // v_lshlrev_b32 v16, 2, v0
        0x3602'0081, // v_and_b32 v1, 1, v0
        0x3402'0285, // v_lshlrev_b32 v1, 5, v1
        0x3622'0080, // v_and_b32 v17, 0, v0
        0x3422'2284, // v_lshlrev_b32 v17, 4, v17
        0xDC38'8000,
        0x0404'0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38'8010,
        0x0804'0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xDC38'8000,
        0x1406'0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
        0xBF8C'0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94'037E, // s_mov_b32 s20, exec_lo
        // L1, the loop over the resource:
        0x7E18'0504, // v_readfirstlane_b32 s12, v4
        0x7E1A'0505,
        0x7E1C'0506,
        0x7E1E'0507,
        0x7E20'0508,
        0x7E22'0509,
        0x7E24'050A,
        0x7E26'050B, // v_readfirstlane_b32 s19, v11
        0xD4E2'0015,
        0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2'0016,
        0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715'1615, // s_and_b32 s21, s21, s22
        0xD4E2'0016,
        0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715'1615, // s_and_b32 s21, s21, s22
        0xD4E2'0016,
        0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715'1615, // s_and_b32 s21, s21, s22
        0xBE95'3C15, // s_and_saveexec_b32 s21, s21
        0xBE9A'037E, // s_mov_b32 s26, exec_lo
        // L2, the loop over the sampler:
        0x7E10'0514, // v_readfirstlane_b32 s8, v20
        0x7E12'0515, // v_readfirstlane_b32 s9, v21
        0x7E14'0516, // v_readfirstlane_b32 s10, v22
        0x7E16'0517, // v_readfirstlane_b32 s11, v23
        0xD4E2'0018,
        0x0002'2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
        0xD4E2'0019,
        0x0002'2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
        0x8718'1918, // s_and_b32 s24, s24, s25
        0xBE98'3C18, // s_and_saveexec_b32 s24, s24
        0xF09C'0F08,
        0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
        0x897E'187E, // s_xor_b32 exec_lo, exec_lo, s24
        0xBF89'FFF2, // s_cbranch_execnz L2
        0xBEFE'031A, // s_mov_b32 exec_lo, s26
        0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89'FFDA, // s_cbranch_execnz L1
        0xBEFE'0314, // s_mov_b32 exec_lo, s20
        0xE070'1000,
        0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81'0000, // s_endpgm
    };
    VS_CHECK(refusal(words, false) == SPLIT);
}

VS_TEST(descriptors_read_from_the_first_lane_of_lds_fail_naming_it)
{
    // The waterfall over descriptors each lane read from LDS:
    // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
    // in place of the global loads.
    Words words = WATERFALL_FETCHED_WAVE32;
    const Words lds = {0xDBFC'0000, 0x0400'0001, 0xDBFC'0010, 0x0800'0001};
    std::copy(lds.begin(), lds.end(), words.begin() + 5);
    const auto err = waterfall_error(words, false);
    VS_CHECK(is_unsupported(err, std::nullopt, "a descriptor read from the first lane of an LDS read"));
}

namespace {

// The block of `p` at byte `offset`.
std::size_t block_at(const Words& words, std::size_t offset)
{
    const Program p = ir::build(bytes(words), false, 256);
    const auto it = std::find_if(p.blocks.begin(), p.blocks.end(), [&](const ir::Block& b) { return b.offset == offset; });
    VS_REQUIRE(it != p.blocks.end());
    return static_cast<std::size_t>(it - p.blocks.begin());
}

// The table offset of what the scalar load of `s` reads.
std::uint32_t table_offset(const Resources& r, const Source& s)
{
    VS_REQUIRE(s.kind == Source::Kind::Loaded && s.first == 0);
    return scalar_at(r, s.load).offset;
}

}  // namespace

VS_TEST(descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path)
{
    const Words& words = WATERFALL_JOINED_WAVE32;
    const Resources r = waterfall(words, false);
    const auto image = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                    [](const Descriptor& d) { return d.source.kind == Source::Kind::LanePhi; });
    VS_REQUIRE(image != r.descriptors.end());
    // The join after the else arm: lanes that ran it (lanes from 16 on)
    // take the second entry (0x20); the others come through the flow
    // block, whose join gives lanes that ran the then arm the first one
    // (0). No lane that skipped the then arm comes that way, so the
    // dwords they held before (the inputs) are not traced.
    VS_REQUIRE(image->source.kind == Source::Kind::LanePhi);
    {
        const std::size_t block = image->source.block;
        const auto& args = image->source.lane_args;
        VS_CHECK(block == block_at(words, 0x9c));
        VS_REQUIRE(args.size() == 2 && args[0].has_value() && args[1].has_value());
        const Source& flow = *args[0];
        const Source& other = *args[1];
        VS_CHECK(table_offset(r, other) == 0x20);
        VS_REQUIRE(flow.kind == Source::Kind::LanePhi);
        VS_CHECK(flow.block == block_at(words, 0x70));
        const auto& flow_args = flow.lane_args;
        VS_REQUIRE(flow_args.size() == 2 && !flow_args[0].has_value() && flow_args[1].has_value());
        const Source& then = *flow_args[1];
        VS_CHECK(table_offset(r, then) == 0);
    }
    // The loop's compares are of the joined dwords; still recognised.
    VS_CHECK(r.waterfall.compares.size() == 4);
    VS_CHECK(r.waterfall.reads.size() == 8);
    // Without the lanes' paths the join is refused.
    const std::vector<std::uint8_t> code = bytes(words);
    const Program p = ir::build(code, false, 256);
    const std::vector<RootParam> layout = {param(RootType::Table(), 0, 1), param(RootType::Table(), 1, 1)};
    const auto err = catch_shader_error([&] { return track(p, layout, HwStage::Cs, nullptr); });
    VS_CHECK(is_unsupported(err, std::nullopt,
                            "a descriptor read from the first lane of a phi, without the lanes' paths"));
}

VS_TEST(descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path)
{
    const Words& words = WATERFALL_JOINED_MIXED_WAVE32;
    const Resources r = waterfall(words, false);
    const auto image = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                    [](const Descriptor& d) { return d.source.kind == Source::Kind::LanePhi; });
    VS_REQUIRE(image != r.descriptors.end());
    VS_REQUIRE(image->source.kind == Source::Kind::LanePhi);
    const std::size_t block = image->source.block;
    const auto& args = image->source.lane_args;
    VS_CHECK(block == block_at(words, 0xb8));
    VS_REQUIRE(args.size() == 2 && args[0].has_value() && args[1].has_value());
    VS_REQUIRE(args[0]->kind == Source::Kind::LanePhi && args[1]->kind == Source::Kind::Lanes);
    const auto& flow = args[0]->lane_args;
    const std::vector<Source>& picked = args[1]->args;
    // The then arm loads each lane's entry from the table at s0.
    VS_REQUIRE(flow.size() == 2 && !flow[0].has_value() && flow[1].has_value());
    VS_REQUIRE(flow[1]->kind == Source::Kind::Fetched && flow[1]->first == 0);
    const Site load = flow[1]->load;
    const Address a = scalar_at(r, load);
    VS_CHECK(a.param == 0 && a.offset == 0);
    VS_CHECK(a.dynamic.has_value());
    // The else arm picks the first entry where vcc is set.
    std::vector<std::uint32_t> offsets;
    for (const Source& s : picked) offsets.push_back(table_offset(r, s));
    VS_CHECK((offsets == std::vector<std::uint32_t>{0, 0x20}));
    VS_CHECK(r.waterfall.compares.size() == 4);
}

VS_TEST(a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it)
{
    const auto err = waterfall_error(WATERFALL_LOOPED_WAVE32, false);
    VS_CHECK(is_unsupported(err, std::nullopt, "a descriptor read from the first lane of a phi at a loop header"));
}

VS_TEST(a_per_lane_descriptor_some_lanes_do_not_write_fails)
{
    // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
    // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
    // of the then arm, which keep the first entry; the second is not
    // every lane's.
    Words words = WATERFALL_JOINED_WAVE32;
    const auto skip = std::find(words.rbegin(), words.rend(), 0xBF88'0008u);
    VS_REQUIRE(skip != words.rend());
    *skip = 0xBF80'0000;
    const auto err = waterfall_error(words, false);
    VS_CHECK(is_unsupported(err, std::nullopt,
                            "a per-lane descriptor written where some lanes may keep the old one"));
}

VS_TEST(a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load)
{
    const Words& words = WATERFALL_JOINED_BUFFER_WAVE32;
    const Resources r = waterfall(words, false);
    const auto buffer = std::find_if(r.descriptors.begin(), r.descriptors.end(),
                                     [](const Descriptor& d) { return d.source.kind == Source::Kind::LanePhi; });
    VS_REQUIRE(buffer != r.descriptors.end());
    // The outer join: lanes from the else arm's inner join, or (below
    // 16) through the flow block.
    VS_REQUIRE(buffer->source.kind == Source::Kind::LanePhi);
    const std::size_t block = buffer->source.block;
    const auto& args = buffer->source.lane_args;
    VS_CHECK(block == block_at(words, 0x78));
    VS_REQUIRE(args.size() == 2 && args[0].has_value() && args[1].has_value());
    VS_REQUIRE(args[0]->kind == Source::Kind::LanePhi && args[1]->kind == Source::Kind::LanePhi);
    const auto& flow = args[0]->lane_args;
    const std::size_t inner = args[1]->block;
    const auto& nested = args[1]->lane_args;
    VS_REQUIRE(flow.size() == 2 && !flow[0].has_value() && flow[1].has_value());
    const Source& then = *flow[1];
    VS_CHECK(table_offset(r, then) == 0);
    // Lanes from 24 on skip the inner arm and keep the inputs, but do
    // not load: that edge is not traced.
    VS_CHECK(inner == block_at(words, 0x74));
    VS_REQUIRE(nested.size() == 2 && !nested[0].has_value() && nested[1].has_value());
    const Source& other = *nested[1];
    VS_CHECK(table_offset(r, other) == 0x20);
    VS_CHECK(r.waterfall.compares.size() == 2);
    VS_CHECK(r.waterfall.reads.size() == 4);
}

VS_TEST_MAIN()
