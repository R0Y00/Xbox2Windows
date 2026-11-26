// Port of the tests in the upstream Rust module emit/lower.rs (#[cfg(test)] mod tests), in Rust order.
//
// None of these runs on a GPU: they lower synthetic GFX10 compute programs and check that the
// lowering succeeds or what it reports (features, handle counts), and test lower.rs's private
// helpers (packing, buffer formats, small floats) directly.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/lower_impl.h"  // buffer_format, half_shift, packed_half, half_input_partners
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "resource_fixtures.h"  // vs-front: tests/x2w_shader/resource_fixtures.h
#include "x2w_shader/structure.h"
#include "test_util.h"

namespace vs = x2w::vs;
using vs::container::SigElement;
using vs::lower::Lowered;
using vs::lower::Numeric;
using vs::lower::Parts;
using vs::native::ComputeIo;
using vs::native::HwStage;
using vs::native::NativeProgram;
using vs::native::ResourceType;
using vs::native::RootParam;
using vs::native::RootType;
using vs::native::StageIo;
using vs::native::UsedResource;

namespace {

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

std::vector<std::uint8_t> le_bytes(std::span<const std::uint32_t> words) {
    std::vector<std::uint8_t> code;
    code.reserve(words.size() * 4);
    for (std::uint32_t w : words)
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    return code;
}

// Rust `lower(&Parts { native, p, st, io, res, inputs: vec![], outputs: vec![], prims: vec![],
// mesh_stride: 0 })` over the front end's results for `native`. `track_may_fail`: Rust's `?` on
// resource::track (lower_wave) vs. `.unwrap()` (lowers): both throw ShaderError here.
Lowered lower_native(const NativeProgram& native) {
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, 256);
    vs::structure::Structured st = vs::structure::structure(p);
    // FRONT? resource::track(p, layout, stage, Option<&Structured>) -> Resources (throws)
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    // FRONT? io::raise(p, native) -> Io (throws)
    vs::io::Io io = vs::io::raise(p, native);
    Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    return vs::lower::lower(parts);
}

// ---- rs:12722 -----------------------------------------------------------------------------------
void packed_pixel_inputs_find_their_second_row() {
    auto entry = [](std::uint8_t row, std::uint8_t row2, bool pack2) {
        vs::native::PsInput e;
        e.usage = vs::native::Usage::Generic();
        e.row = row;
        e.mask = 0xF;
        e.attr = row;
        e.flat = false;
        e.fp16 = true;
        e.pack2 = pack2;
        e.row2 = row2;
        return e;
    };
    auto element = [](std::uint32_t row) {
        SigElement e;
        e.name = "TEXCOORD";
        e.index = row;
        e.kind = 0;
        e.comp = 9;
        e.interp = 2;
        e.row = row;
        e.col = 0;
        e.cols = 4;
        e.usage = 0;
        e.half = true;
        return e;
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its own (6). Unnamed 16-bit
    // rows: 1, 4, 6, 7.
    vs::native::PsIo ps;
    ps.inputs = {entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)};
    ps.input_addr = 0;
    ps.db_shader_control = 0;
    StageIo io = ps;
    std::vector<SigElement> sig;
    for (std::uint32_t r : {0u, 1u, 3u, 4u, 5u, 6u, 7u}) sig.push_back(element(r));
    auto p = vs::lower::half_input_partners(io, sig);
    auto get = [&](std::uint8_t k) -> std::optional<std::uint8_t> {
        auto it = p.find(k);
        if (it == p.end()) return std::nullopt;
        return it->second;
    };
    CHECK(get(0) == std::optional<std::uint8_t>(1));
    CHECK(get(3) == std::optional<std::uint8_t>(4));
    CHECK(get(5) == std::nullopt);
}

// ---- rs:12767 -----------------------------------------------------------------------------------
// Each element starts on a whole dword, even when it shares a source register.
void packed_elements_behind_another_take_the_first_free_dword() {
    auto element = [](std::uint32_t index, std::uint32_t row, std::uint8_t col, std::uint8_t cols) {
        SigElement e;
        e.name = "TEXCOORD";
        e.index = index;
        e.kind = 0;
        e.comp = 9;
        e.interp = 2;
        e.row = row;
        e.col = col;
        e.cols = cols;
        e.usage = 0;
        e.half = true;
        return e;
    };
    using Pack = std::pair<std::uint8_t, std::optional<std::uint8_t>>;
    using Got = std::optional<std::pair<std::uint8_t, std::uint8_t>>;
    auto some = [](std::uint8_t a, std::uint8_t b) { return Got(std::pair<std::uint8_t, std::uint8_t>(a, b)); };
    // Rows 8 and 13 occupy four dwords. The trailing scalar from row 8
    // must move to the spare dword in a separate, unpaired parameter.
    {
        std::vector<SigElement> sig = {
            element(0, 8, 0, 3), element(1, 8, 3, 1),
            element(2, 13, 0, 3), element(3, 17, 0, 2),
        };
        std::vector<Pack> packs = {{8, 13}, {17, std::nullopt}};
        auto at = [&](std::uint8_t a, std::uint8_t h) { return vs::lower::packed_half(sig, packs, a, h); };
        CHECK(at(8, 2) == some(8, 2));
        CHECK(at(8, 3) == std::nullopt);
        CHECK(at(8, 4) == some(13, 0));
        CHECK(at(8, 7) == std::nullopt);
        CHECK(at(17, 1) == some(17, 1));
        CHECK(at(17, 2) == some(8, 3));
        CHECK(at(17, 3) == std::nullopt);
    }
    // A scalar lead leaves half a dword unused. Its three-component
    // neighbor starts only after the partner row, even with shuffled inputs.
    {
        std::vector<SigElement> sig = {
            element(2, 23, 0, 2), element(1, 21, 1, 3), element(0, 21, 0, 1),
        };
        std::vector<Pack> packs = {{34, std::nullopt}, {21, 23}};
        auto at = [&](std::uint8_t a, std::uint8_t h) { return vs::lower::packed_half(sig, packs, a, h); };
        CHECK(at(21, 0) == some(21, 0));
        CHECK(at(21, 1) == std::nullopt);
        CHECK(at(21, 2) == some(23, 0));
        CHECK(at(21, 3) == some(23, 1));
        CHECK(at(21, 4) == some(21, 1));
        CHECK(at(21, 6) == some(21, 3));
        CHECK(at(21, 7) == std::nullopt);
        CHECK(at(34, 0) == std::nullopt);
    }
}

// ---- rs:12844 -----------------------------------------------------------------------------------
// Whether a synthetic compute program translates with these input registers. Throws ShaderError.
void lowers(std::span<const std::uint32_t> words, std::uint32_t rsrc2) {
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = 256;
    ComputeIo cs;
    cs.rsrc2 = rsrc2;
    cs.threads = {1, 1, 1};
    native.io = cs;
    native.code = le_bytes(words);
    (void)lower_native(native);
}

// rs:12887: lower_with in wave64 if `wave64`. Throws ShaderError.
Lowered lower_wave(std::span<const std::uint32_t> words, std::vector<RootParam> layout,
                   std::vector<UsedResource> used, bool wave64) {
    std::uint32_t sgprs = 0;
    for (const RootParam& p : layout) sgprs += p.sgprs;
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = wave64;
    native.vgprs = 256;
    native.layout = std::move(layout);
    native.used = std::move(used);
    ComputeIo cs;
    // The layout's user SGPRs (at least two), 512 bytes of LDS.
    cs.rsrc2 = 1u << 15 | (sgprs > 2 ? sgprs : 2) << 1;
    cs.threads = {1, 1, 1};
    native.io = cs;
    native.code = le_bytes(words);
    return lower_native(native);
}

// rs:12878: lowers a compute program with the root parameters `layout` (in user SGPRs from s0 on)
// and the used resources `used`.
Lowered lower_with(std::span<const std::uint32_t> words, std::vector<RootParam> layout,
                   std::vector<UsedResource> used) {
    return lower_wave(words, std::move(layout), std::move(used), false);
}

// rs:12926: root parameter `slot` in user SGPRs from `user_data` on.
RootParam root(RootType root, std::uint8_t user_data, std::uint8_t slot) {
    RootParam p;
    p.root = root;
    p.table_offset_dwords = 0;
    p.sgprs = root == RootType::Descriptor() ? 2 : 1;
    p.base_register = 0;
    p.user_data = user_data;
    p.slot = slot;
    return p;
}

// rs:12940: lowers a one-thread compute program that copies four root constants (s0..s3) into
// v1..v4, runs `body`, and stores v0 to a raw buffer from a descriptor table, so the vector
// instructions in `body` have an effect. Throws ShaderError.
void lowers_valu(std::span<const std::uint32_t> body) {
    std::vector<std::uint32_t> words = {
        0xBE85'0380,  // s_mov_b32 s5, 0
        0xF408'0202,  // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00'0000,
    };
    // v_mov_b32 v1..v4, s0..s3
    for (std::uint32_t i = 0; i < 4; ++i) words.push_back(0x7E00'0200u | (i + 1) << 17 | i);
    words.insert(words.end(), body.begin(), body.end());
    for (std::uint32_t w : {0xE070'0000u,  // buffer_store_dword v0, off, s[8:11], 0
                            0x8002'0000u, S_ENDPGM})
        words.push_back(w);
    RootParam constants = root(RootType::Constants(), 0, 0);
    constants.sgprs = 4;
    UsedResource buffer;
    buffer.kind = ResourceType::Uav();
    buffer.root = RootType::Table();
    buffer.table_offset_dwords = 0;
    buffer.lower_bound = 0;
    buffer.stride = 0;
    buffer.dwords = 4;
    buffer.user_data = 4;
    buffer.slot = 1;
    (void)lower_with(words, {constants, root(RootType::Table(), 4, 1)}, {buffer});
}

// Rust `.unwrap_or_else(|e| panic!(...))` / `.unwrap()` / `.is_ok()` for a throwing call.
template <class F>
bool succeeds(F&& f, const char* what) {
    try {
        f();
        return true;
    } catch (const vs::ShaderError& e) {
        std::fprintf(stderr, "%s: %s\n", what, e.to_string().c_str());
        return false;
    }
}

// ---- rs:12977 -----------------------------------------------------------------------------------
void the_24_bit_multiplies_lower_both_halves() {
    // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low dword) and v_mul_hi_i32_i24
    // and v_mul_hi_u32_u24 (the high one).
    for (std::uint32_t op : {0x09u, 0x0Au, 0x0Bu, 0x0Cu}) {
        std::uint32_t word = op << 25 | 2u << 9 | 0x101u;
        char what[32];
        std::snprintf(what, sizeof what, "VOP2 %#x", op);
        CHECK(succeeds([&] { lowers_valu(std::span<const std::uint32_t>(&word, 1)); }, what));
    }
}

// ---- rs:12987 -----------------------------------------------------------------------------------
void buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the host to invalidate its
    // cached view of a buffer.
    const std::uint32_t gl0[] = {0xE1C4'0000, 0, S_ENDPGM};
    const std::uint32_t gl1[] = {0xE1C8'0000, 0, S_ENDPGM};
    CHECK(succeeds([&] { lowers(gl0, 0); }, "buffer_gl0_inv"));
    CHECK(succeeds([&] { lowers(gl1, 0); }, "buffer_gl1_inv"));
}

// ---- rs:12995 -----------------------------------------------------------------------------------
void sums_of_absolute_differences_lower() {
    // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32 and v_msad_u8.
    for (std::uint32_t op : {0x15Au, 0x15Bu, 0x15Cu, 0x15Du, 0x171u}) {
        const std::uint32_t words[] = {0xD400'0000u | op << 16, 0x101u | 0x102u << 9 | 0x103u << 18};
        char what[32];
        std::snprintf(what, sizeof what, "VOP3 %#x", op);
        CHECK(succeeds([&] { lowers_valu(words); }, what));
    }
}

// ---- rs:13005 -----------------------------------------------------------------------------------
void a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the carry (borrow) the high
    // half leaves in SCC, and a branch on it (only code with an effect needs its operands).
    const std::uint32_t add[] = {
        0xBE80'03C1,  // s_mov_b32 s0, -1
        0xBE81'0380,  // s_mov_b32 s1, 0
        0x8000'8100,  // s_add_u32 s0, s0, 1
        0x8201'8001,  // s_addc_u32 s1, s1, 0
        0x8502'0001,  // s_cselect_b32 s2, s1, s0
        0xBF07'8002,  // s_cmp_lg_u32 s2, 0
        0xBF85'0001,  // s_cbranch_scc1 -> s_endpgm
        0xBE83'0380,  // s_mov_b32 s3, 0
        S_ENDPGM,
    };
    const std::uint32_t sub[] = {
        0xBE80'0380,  // s_mov_b32 s0, 0
        0xBE81'0380,  // s_mov_b32 s1, 0
        0x8080'8100,  // s_sub_u32 s0, s0, 1
        0x8281'8001,  // s_subb_u32 s1, s1, 0
        0x8502'0001,  // s_cselect_b32 s2, s1, s0
        0xBF07'8002,  // s_cmp_lg_u32 s2, 0
        0xBF85'0001,  // s_cbranch_scc1 -> s_endpgm
        0xBE83'0380,  // s_mov_b32 s3, 0
        S_ENDPGM,
    };
    CHECK(succeeds([&] { lowers(add, 0); }, "add"));
    CHECK(succeeds([&] { lowers(sub, 0); }, "sub"));
}

// ---- rs:13037 -----------------------------------------------------------------------------------
std::uint64_t perm_features(std::uint32_t selector) {
    // Synthetic compute shader: v0 is the thread ID and can supply a dynamic selector. Store the
    // result in LDS to keep the permutation live.
    const std::uint32_t words[] = {
        0x7E02'02FFu, 0x817F'34AB, 0x7E04'02FF, 0xFE01'56CD, 0xD744'0003,
        257u | (258u << 9) | (selector << 18), 0x3408'0082, 0xD834'0000, 0x0000'0304, S_ENDPGM,
    };
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = 256;
    ComputeIo cs;
    cs.rsrc2 = 1u << 15;
    cs.threads = {64, 1, 1};
    native.io = cs;
    native.code = le_bytes(words);

    vs::ir::Program p = vs::ir::build(native.code, false, 256);
    bool perm = false;
    for (const auto& b : p.blocks)
        for (const auto& i : b.insts) perm = perm || i.name() == "v_perm_b32";
    CHECK(perm);

    Lowered emitted = lower_native(native);
    vs::dxbc::Container container = vs::dxbc::Container::parse(emitted.dxil);
    const vs::dxbc::Part* features = container.part(vs::dxbc::FourCC::SFI0);
    CHECK(features != nullptr);
    if (!features) return ~std::uint64_t{0};
    CHECK(features->data.size() == 8);  // Rust `try_into().unwrap()` of 8 bytes
    if (features->data.size() != 8) return ~std::uint64_t{0};
    std::uint64_t v = 0;
    for (int k = 0; k < 8; ++k) v |= std::uint64_t(features->data[k]) << (8 * k);
    return v;
}

// ---- rs:13104 -----------------------------------------------------------------------------------
void perm_constant_selector_needs_no_int64() { CHECK((perm_features(128) & 0x8000) == 0); }

// ---- rs:13109 -----------------------------------------------------------------------------------
void perm_dynamic_selector_needs_no_int64() {
    // dynamic v_perm_b32 must not require INT64
    CHECK((perm_features(256) & 0x8000) == 0);
}

// ---- rs:13118 -----------------------------------------------------------------------------------
void a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by a root constant (s1) with
    // s_cselect_b64 per half, and loaded through by index and stored to LDS so the load is needed.
    const std::uint32_t words[] = {
        0xBE84'0300,  // s_mov_b32 s4, s0
        0xBE85'0380,  // s_mov_b32 s5, 0
        0xF408'0202,
        0xFA00'0000,  // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408'0302,
        0xFA00'0010,  // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06'8001,  // s_cmp_eq_u32 s1, 0
        0x8590'0C08,  // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592'0E0A,  // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030'2000,
        0x8004'0100,  // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834'0000,
        0x0000'0100,  // ds_write_b32 v0, v1
        S_ENDPGM,
    };
    std::vector<RootParam> layout = {root(RootType::Table(), 0, 0), root(RootType::Constants(), 1, 1)};
    // Both are structured buffers of dwords in the table's range.
    UsedResource u;
    u.kind = ResourceType::Srv();
    u.root = RootType::Table();
    u.table_offset_dwords = 0;
    u.lower_bound = 0;
    u.stride = 4;
    u.dwords = 4;
    u.user_data = 0;
    u.slot = 0;
    CHECK(succeeds([&] { (void)lower_with(words, layout, {u}); }, "scc-selected descriptor"));
}

// ---- rs:13156 -----------------------------------------------------------------------------------
// Tables of 2D textures at s0 and samplers at s1.
std::pair<std::vector<RootParam>, std::vector<UsedResource>> texture_tables() {
    auto range = [](ResourceType kind, std::uint8_t user_data, std::uint8_t slot) {
        UsedResource u;
        u.kind = kind;
        u.root = RootType::Table();
        u.table_offset_dwords = 0;
        u.lower_bound = 0;
        u.stride = 0;
        u.dwords = kind == ResourceType::Sampler() ? 4 : 8;
        u.user_data = user_data;
        u.slot = slot;
        return u;
    };
    return {
        {root(RootType::Table(), 0, 0), root(RootType::Table(), 1, 1)},
        {range(ResourceType::Srv(), 0, 0), range(ResourceType::Sampler(), 1, 1)},
    };
}

// ---- rs:13177 -----------------------------------------------------------------------------------
void waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    namespace fx = vs::resource::fixtures;
    const std::pair<std::span<const std::uint32_t>, bool> cases[] = {
        {fx::WATERFALL_FETCHED_WAVE32, false},
        {fx::WATERFALL_FETCHED_WAVE64, true},
        {fx::WATERFALL_PICKED_WAVE32, false},
    };
    for (const auto& [words, wave64] : cases) {
        auto [layout, used] = texture_tables();
        std::optional<Lowered> l;
        CHECK(succeeds([&] { l = lower_wave(words, layout, used, wave64); }, "waterfall"));
        if (!l) continue;
        // The loop's reads of the first lane became each lane's own dwords: no wave operation is
        // left.
        CHECK(l->nonuniform_handles == 1);
        CHECK(!l->wave_ops);
    }
}

// ---- rs:13196 -----------------------------------------------------------------------------------
// texture_tables with a raw buffer UAV at dword 16 of the first (pub(crate) in Rust: the
// waterfall/size tests use it too).
std::pair<std::vector<RootParam>, std::vector<UsedResource>> texture_and_uav_tables() {
    auto [layout, used] = texture_tables();
    UsedResource u;
    u.kind = ResourceType::Uav();
    u.root = RootType::Table();
    u.table_offset_dwords = 16;
    u.lower_bound = 0;
    u.stride = 0;
    u.dwords = 4;
    u.user_data = 0;
    u.slot = 0;
    used.push_back(u);
    return {layout, used};
}

// ---- rs:13212 -----------------------------------------------------------------------------------
void waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    namespace fx = vs::resource::fixtures;
    const std::span<const std::uint32_t> cases[] = {fx::WATERFALL_JOINED_WAVE32,
                                                    fx::WATERFALL_JOINED_MIXED_WAVE32};
    for (const auto& words : cases) {
        auto [layout, used] = texture_and_uav_tables();
        std::optional<Lowered> l;
        CHECK(succeeds([&] { l = lower_wave(words, layout, used, false); }, "joined waterfall"));
        if (!l) continue;
        // One image handle, from the join's per-lane heap index.
        CHECK(l->nonuniform_handles == 1);
        CHECK(!l->wave_ops);
    }
}

// ---- rs:13224 -----------------------------------------------------------------------------------
void scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only code with an effect
    // needs its operands).
    const std::uint32_t tail[] = {
        0xBF85'0001,  // s_cbranch_scc1 -> s_endpgm
        0xBE84'0380,  // s_mov_b32 s4, 0
        S_ENDPGM,
    };
    const std::vector<std::vector<std::uint32_t>> bit_tests = {
        {0xBE81'0387, 0xBF0C'8301},  // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        {0xBE81'0387, 0xBF0D'8301},  // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        {
            0xBE80'0387,  // s_mov_b32 s0, 7
            0xBE81'0380,  // s_mov_b32 s1, 0
            0xBF0E'8300,  // s_bitcmp0_b64 s[0:1], 3
        },
        {
            0xBE80'0387,  // s_mov_b32 s0, 7
            0xBE81'0380,  // s_mov_b32 s1, 0
            0xBF0F'8300,  // s_bitcmp1_b64 s[0:1], 3
        },
    };
    for (const auto& head : bit_tests) {
        std::vector<std::uint32_t> words = head;
        words.insert(words.end(), std::begin(tail), std::end(tail));
        CHECK(succeeds([&] { lowers(words, 0); }, "bit test"));
    }
}

// ---- rs:13254 -----------------------------------------------------------------------------------
// #[ignore = "ours marks joins read through joins only with X2W_SHADERCFG_STRICT_JOINS (40 programs stop
// otherwise)"]: compiled, run only with the argument `--ignored`.
void a_missing_value_stays_missing_through_two_live_joins() {
    std::uint32_t words[] = {
        0xF490'0180,
        0,            // s_memtime s[6:7]
        0xBF06'8000,  // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85'0001,  // skip the first overwrite
        0xBE86'0380,  // s_mov_b32 s6, 0
        0xBF06'8101,  // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85'0001,  // skip the second overwrite
        0xBE86'0381,  // s_mov_b32 s6, 1
        0xBF07'8006,  // s_cmp_lg_u32 s6, 0
        0xBF85'0001,  // use the result to branch
        0xBE83'0380,
        S_ENDPGM,
    };
    const std::uint32_t group_ids_xy = 3u << 7;
    // Compare to a direct read so future timestamp support works in both cases. Rust panics on
    // any error other than Unsupported; that is a failed CHECK here.
    bool other_error = false;
    auto refusal = [&](std::span<const std::uint32_t> code) -> std::optional<std::string> {
        try {
            lowers(code, group_ids_xy);
            return std::nullopt;
        } catch (const vs::ShaderError& e) {
            if (e.kind == vs::ShaderError::Kind::Unsupported) return std::string(e.reason);
            std::fprintf(stderr, "%s\n", e.to_string().c_str());
            other_error = true;
            return std::nullopt;
        }
    };
    std::vector<std::uint32_t> direct(words, words + 2);
    direct.insert(direct.end(), words + 8, std::end(words));
    CHECK(refusal(words) == refusal(direct));
    CHECK(!other_error);
    words[8] = 0xBF07'8000;  // read the group id instead; the timestamp is dead
    CHECK(succeeds([&] { lowers(words, group_ids_xy); }, "dead timestamp"));
}

// ---- rs:13283 -----------------------------------------------------------------------------------
bool format_is(std::uint8_t format, std::initializer_list<std::uint32_t> bits, Numeric numeric) {
    auto f = vs::lower::buffer_format(format);
    if (!f) return false;
    if (f->second != numeric || f->first.size() != bits.size()) return false;
    return std::equal(f->first.begin(), f->first.end(), bits.begin());
}

void buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM, 2_10_10_10_UNORM, 16_FLOAT,
    // 32_32_32_32_FLOAT.
    CHECK(format_is(56, {8, 8, 8, 8}, Numeric::Unorm));
    CHECK(format_is(24, {16, 16}, Numeric::Snorm));
    CHECK(format_is(50, {10, 10, 10, 2}, Numeric::Unorm));
    CHECK(format_is(13, {16}, Numeric::Float));
    CHECK(format_is(77, {32, 32, 32, 32}, Numeric::Float));
    // Invalid, past the table.
    CHECK(!vs::lower::buffer_format(0));
    CHECK(!vs::lower::buffer_format(78));
}

// ---- rs:13303 -----------------------------------------------------------------------------------
void packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43, 10_10_10_2 44-49. Channels from the
    // lowest bits: R11G11B10_FLOAT is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    CHECK(format_is(30, {11, 11, 10}, Numeric::Unorm));
    CHECK(format_is(36, {11, 11, 10}, Numeric::Float));
    CHECK(format_is(37, {10, 11, 11}, Numeric::Unorm));
    CHECK(format_is(43, {10, 11, 11}, Numeric::Float));
    CHECK(format_is(44, {2, 10, 10, 10}, Numeric::Unorm));
    CHECK(format_is(45, {2, 10, 10, 10}, Numeric::Snorm));
    CHECK(format_is(46, {2, 10, 10, 10}, Numeric::Uscaled));
    CHECK(format_is(47, {2, 10, 10, 10}, Numeric::Sscaled));
    CHECK(format_is(48, {2, 10, 10, 10}, Numeric::Uint));
    CHECK(format_is(49, {2, 10, 10, 10}, Numeric::Sint));
}

// rs:13339: a half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as legacyF16ToF32
// widens it. 2f32.powi(n) is exact for these n: std::ldexp(1.0f, n).
float half(std::uint32_t h) {
    std::uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) {
        v = static_cast<float>(m) * std::ldexp(1.0f, -24);
    } else if (e == 31 && m == 0) {
        v = std::numeric_limits<float>::infinity();
    } else if (e == 31) {
        v = std::numeric_limits<float>::quiet_NaN();
    } else {
        v = static_cast<float>(1024 + m) * std::ldexp(1.0f, static_cast<int>(e) - 25);
    }
    return s == 1 ? -v : v;
}

// ---- rs:13351 -----------------------------------------------------------------------------------
void small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a 6- or 5-bit mantissa, no
    // sign; values from the D3D/GL spec.
    auto f11 = [](std::uint32_t x) { return half(x << vs::lower::half_shift(11)); };
    auto f10 = [](std::uint32_t x) { return half(x << vs::lower::half_shift(10)); };
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(f11(0) == 0.0f);
    CHECK(f11(0x001) == std::ldexp(1.0f, -20));          // smallest denormal
    CHECK(f11(0x03F) == 63.0f * std::ldexp(1.0f, -20));  // largest denormal
    CHECK(f11(0x040) == std::ldexp(1.0f, -14));          // smallest normal
    CHECK(f11(0x3C0) == 1.0f);
    CHECK(f11(0x3E0) == 1.5f);
    CHECK(f11(0x7BF) == 65024.0f);  // largest
    CHECK(f11(0x7C0) == inf);
    CHECK(std::isnan(f11(0x7C1)));
    CHECK(f10(0x001) == std::ldexp(1.0f, -19));
    CHECK(f10(0x020) == std::ldexp(1.0f, -14));
    CHECK(f10(0x1E0) == 1.0f);
    CHECK(f10(0x3DF) == 64512.0f);
    CHECK(f10(0x3E0) == inf);
    CHECK(std::isnan(f10(0x3FF)));
    // A 16-bit float is a half already, sign included.
    CHECK(half(0xBC00u << vs::lower::half_shift(16)) == -1.0f);
}

}  // namespace

int main(int argc, char** argv) {
    bool ignored = argc > 1 && std::string_view(argv[1]) == "--ignored";
    if (ignored) {
        x2w::vs::lower::test::set_dump_name("emit__lower__tests__a_missing_value_stays_missing_through_two_live_joins");
        a_missing_value_stays_missing_through_two_live_joins();
        return test_util::finish("lower_test --ignored");
    }
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__packed_pixel_inputs_find_their_second_row");
    packed_pixel_inputs_find_their_second_row();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__packed_elements_behind_another_take_the_first_free_dword");
    packed_elements_behind_another_take_the_first_free_dword();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__the_24_bit_multiplies_lower_both_halves");
    the_24_bit_multiplies_lower_both_halves();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__buffer_cache_invalidations_lower_as_no_ops");
    buffer_cache_invalidations_lower_as_no_ops();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__sums_of_absolute_differences_lower");
    sums_of_absolute_differences_lower();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__a_scalar_carry_chain_reaches_the_select");
    a_scalar_carry_chain_reaches_the_select();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__perm_constant_selector_needs_no_int64");
    perm_constant_selector_needs_no_int64();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__perm_dynamic_selector_needs_no_int64");
    perm_dynamic_selector_needs_no_int64();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__a_descriptor_selected_by_scc_picks_its_heap_index");
    a_descriptor_selected_by_scc_picks_its_heap_index();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own");
    waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own");
    waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__scalar_bit_tests_lower");
    scalar_bit_tests_lower();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__buffer_formats_by_number");
    buffer_formats_by_number();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__packed_buffer_formats_by_number");
    packed_buffer_formats_by_number();
    x2w::vs::lower::test::set_dump_name("emit__lower__tests__small_floats_widen_as_halves");
    small_floats_widen_as_halves();
    return test_util::finish("lower_test");
}
