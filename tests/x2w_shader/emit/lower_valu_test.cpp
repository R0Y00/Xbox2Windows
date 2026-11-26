// Port of the upstream Rust module emit/valu_tests.rs, the tests in Rust order.
//
// End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated by dxil.dll when
// present, and executed on the host GPU and WARP.
//
// Covers RDNA2 hardware semantics that differ from naive DXIL lowering: v_med3_f32/v_med3_f16
// with NaN operands, v_fract_f32/v_fract_f16 near 1.0, turn-range reduction for v_sin_*/v_cos_*,
// and the LDS float atomics ds_min_f32/ds_max_f32.
//
// A Rust assert! inside a check is a non-fatal CHECK here (the test goes on); a Rust panic!/expect
// outside one throws, and main counts the test as failed. `#[ignore]` tests run only with
// --ignored. Each test names its lowered modules for VS_DUMP after the Rust test thread
// (emit::valu_tests::<name>, `::` as `__`).

#ifndef NOMINMAX
#define NOMINMAX  // windows.h (through d3d12.h) would make std::max a macro
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <initializer_list>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "x2w_shader/emit/emit.h"       // HostShader
#include "x2w_shader/emit/host.h"     
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/settings.h"   // rdna::ALL
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"
#include "test_util.h"

namespace vs = x2w::vs;
using vs::HostShader;
using vs::native::ComputeIo;
using vs::native::HwStage;
using vs::native::NativeProgram;
using vs::native::ResourceType;
using vs::native::RootParam;
using vs::native::RootType;
using vs::native::StageIo;
using vs::native::UsedResource;

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using Words = std::vector<u32>;
using In = std::array<u32, 4>;

// A failed Rust assert!/assert_eq! in a check: reported, counted, the test goes on.
void ensure(bool ok, const std::string& what, const char* file, int line) {
    if (!ok) {
        std::fprintf(stderr, "%s:%d: %s\n", file, line, what.c_str());
        ++test_util::failures();
    }
}
#define ENSURE(ok, ...) ensure(static_cast<bool>(ok), std::format(__VA_ARGS__), __FILE__, __LINE__)

u32 f2b(float x) { return std::bit_cast<u32>(x); }
float b2f(u32 b) { return std::bit_cast<float>(b); }
u64 d2b(double x) { return std::bit_cast<u64>(x); }
double b2d(u64 b) { return std::bit_cast<double>(b); }

// Rust {:08x?} of a slice of words.
std::string hex_words(std::span<const u32> w) {
    std::string s = "[";
    for (std::size_t i = 0; i < w.size(); ++i) s += std::format("{}{:08x}", i ? ", " : "", w[i]);
    return s + "]";
}
// Rust {:x?} of an input set.
std::string hex_in(const In& w) {
    return std::format("[{:x}, {:x}, {:x}, {:x}]", w[0], w[1], w[2], w[3]);
}

void append(Words& to, std::span<const u32> w) { to.insert(to.end(), w.begin(), w.end()); }

constexpr u32 S_ENDPGM = 0xBF81'0000;

// SOP1 s_mov_b32 sdst, src0.
constexpr u32 s_mov_b32(u8 sdst, u8 src0) { return 0xBE80'0000u | u32(sdst) << 16 | 0x03u << 8 | u32(src0); }

// SOP2 op sdst, src0, src1.
constexpr u32 sop2(u8 op, u8 sdst, u8 src0, u8 src1) {
    return 0x8000'0000u | (u32(op) & 0x7F) << 23 | (u32(sdst) & 0x7F) << 16 | u32(src1) << 8 | u32(src0);
}

// SMEM s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0 (soffset = null).
constexpr std::array<u32, 2> s_load_dwordx4(u8 dst, u8 base) {
    return {0x3Du << 26 | 0x02u << 18 | u32(dst) << 6 | (u32(base) >> 1), 125u << 25};
}

// VOP1 op vdst, src0 (src0: 0..105 = SGPR, 256..511 = VGPR).
constexpr u32 vop1(u16 op, u8 vdst, u16 src0) {
    return 0x7E00'0000u | u32(vdst) << 17 | u32(op) << 9 | (u32(src0) & 0x1FF);
}

// VOP3 op vdst, src0, src1, src2 with explicit abs, neg, omod, clamp.
constexpr std::array<u32, 2> vop3_mod(u16 op, u8 vdst, u16 src0, u16 src1, u16 src2, u8 abs, u8 neg, u8 omod,
                                      bool clamp) {
    return {0x35u << 26 | (u32(op) & 0x3FF) << 16 | u32(clamp) << 15 | (u32(abs) & 7) << 8 | u32(vdst),
            (u32(neg) & 7) << 29 | (u32(omod) & 3) << 27 | (u32(src0) & 0x1FF) | (u32(src1) & 0x1FF) << 9 |
                (u32(src2) & 0x1FF) << 18};
}

// VOP3 op vdst, src0, src1, src2 (no modifiers).
constexpr std::array<u32, 2> vop3(u16 op, u8 vdst, u16 src0, u16 src1, u16 src2) {
    return vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false);
}

// MUBUF buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off.
constexpr std::array<u32, 2> buffer_store_dword(u8 vdata, u8 srsrc, u16 byte_off) {
    return {0x38u << 26 | 0x1Cu << 18 | (u32(byte_off) & 0xFFF),
            0x80u << 24 | ((u32(srsrc) >> 2) & 0x1F) << 16 | u32(vdata) << 8};
}

constexpr u16 sgpr(u8 n) { return u16(n); }
constexpr u16 vgpr(u8 n) { return u16(256 + n); }

std::vector<u8> le_bytes(std::span<const u32> words) {
    std::vector<u8> code;
    code.reserve(words.size() * 4);
    for (u32 w : words)
        for (int k = 0; k < 4; ++k) code.push_back(u8(w >> (8 * k)));
    return code;
}

// The layout and resources of the synthetic compute programs: 4 root constants in s0..s3, a table
// in s4 holding one raw UAV.
NativeProgram synthetic_cs(std::vector<u8> code, std::uint16_t vgprs, u32 rsrc2, std::array<u32, 3> threads) {
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = vgprs;
    RootParam constants;
    constants.root = RootType::Constants();
    constants.table_offset_dwords = 0;
    constants.sgprs = 4;
    constants.base_register = 0;
    constants.user_data = 0;
    constants.slot = 0;
    RootParam table;
    table.root = RootType::Table();
    table.table_offset_dwords = 0;
    table.sgprs = 1;
    table.base_register = 0;
    table.user_data = 4;
    table.slot = 1;
    native.layout = {constants, table};
    UsedResource uav;
    uav.kind = ResourceType::Uav();
    uav.root = RootType::Table();
    uav.table_offset_dwords = 0;
    uav.lower_bound = 0;
    uav.stride = 0;
    uav.dwords = 4;
    uav.user_data = 4;
    uav.slot = 1;
    native.used = {uav};
    ComputeIo cs;
    cs.rsrc2 = rsrc2;
    cs.threads = threads;
    native.io = StageIo{cs};
    native.code = std::move(code);
    return native;
}

// The front end and lower over `native`, as a HostShader; `indexed_views`: Rust takes
// res.root_indexed_views (build_cs) or none (whole_wave_f64_sum_and_cvt_f32_f64).
HostShader lower_cs(const NativeProgram& native, bool indexed_views) {
    vs::ir::Program p = vs::ir::build(native.code, native.wave64, native.vgprs);  // expect("ir::build")
    vs::structure::Structured st = vs::structure::structure(p);                 // expect("structure")
    vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    vs::io::Io io = vs::io::raise(p, native);
    vs::lower::Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    vs::lower::Lowered lowered = vs::lower::lower(parts);  // expect("lower")

    HostShader shader;
    shader.stage = HwStage::Cs;
    shader.dxil = std::move(lowered.dxil);
    shader.user_data = std::move(lowered.user_data);
    shader.samplers = std::move(lowered.samplers);
    shader.static_samplers = {};
    shader.root_cbvs = std::move(lowered.root_cbvs);
    shader.root_srvs = std::move(lowered.root_srvs);
    shader.root_uavs = std::move(lowered.root_uavs);
    if (indexed_views) shader.root_indexed_views = res.root_indexed_views;
    return shader;
}

// Validator::load returns std::optional (Rust `if let Ok(mut v) = ...`), validate returns
// the error text or nothing.
void validate(const HostShader& shader, bool check_hash) {
    auto v = vs::validator::Validator::load(vs::validator::Validator::default_path());
    if (!v) return;
    std::vector<u8> copy = shader.dxil;
    std::vector<u8> ours(copy.begin() + 4, copy.begin() + 20);
    if (auto e = v->validate(copy)) throw std::runtime_error("DXIL validation failed: " + *e);
    if (check_hash) ENSURE(std::equal(ours.begin(), ours.end(), copy.begin() + 4), "container hash mismatch");
}

// build_valu_cs with `threads` threads in x (thread index in v0 on entry, before `body`) and
// `lds_bytes` of LDS (a multiple of 512).
HostShader build_cs(std::span<const u32> body, u32 threads, u32 lds_bytes) {
    Words words;
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push_back(s_mov_b32(5, 0x80));  // inline constant 0
    append(words, s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on VGPRs just as they
    // do in real shaders.
    for (u8 i = 0; i < 4; ++i) words.push_back(vop1(0x001, u8(i + 1), sgpr(i)));  // v_mov_b32
    append(words, body);
    append(words, buffer_store_dword(0, 8, 0));
    words.push_back(S_ENDPGM);

    NativeProgram native =
        synthetic_cs(le_bytes(words), 8, (5u << 1) | ((lds_bytes / 512) << 15), {threads, 1, 1});  // 5 user SGPRs: s0..s4
    HostShader shader = lower_cs(native, true);
    validate(shader, true);
    return shader;
}

// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives 4 dwords of root
// constants in s0..s3, runs `body` (leaving its result in v0), and stores v0 to dword 0 of the raw
// UAV at heap index 0.
HostShader build_valu_cs(std::span<const u32> body) { return build_cs(body, 1, 0); }
HostShader build_valu_cs(std::initializer_list<u32> body) { return build_cs(std::span(body.begin(), body.size()), 1, 0); }

// One GPU test at a time (Rust's ONE_AT_A_TIME: dispatches from several test threads at once on
// the hardware adapter and WARP now and then never complete, and two at once on WARP can end the
// process). Each C++ test binary runs its tests one after another; this only guards this binary.
std::mutex ONE_AT_A_TIME;

using Check = std::function<void(const char*, std::size_t, In, u32)>;
using Devices = std::vector<std::pair<const char*, bool>>;

// The user-data dwords for `inputs`, as Rust's run_cases_on and whole_wave_f64_sum_and_cvt_f32_f64
// fill them.
std::vector<u32> user_data_for(const HostShader& shader, const In& inputs) {
    std::vector<u32> ud(std::max<u32>(shader.user_data_dwords(), 1), 0);
    for (const auto& u : shader.user_data) {
        if (u.root == RootType::Constants() && u.slot == 0)
            ud[u.index] = inputs[u.dword];
        else if (u.root == RootType::Table() && u.slot == 1)
            ud[u.index] = 0;
        else
            throw std::runtime_error(std::format("unexpected user data (root kind {}, slot {})", int(u.root.kind), u.slot));
    }
    return ud;
}

// one dispatch of `shader` with root constants `ud` and one raw UAV dword; the UAV's dword 0.
u32 dispatch_one(const vs::host::Host& host, ID3D12RootSignature* root, ID3D12PipelineState* pso, std::vector<u32> ud,
                 const char* label) {
    std::array<vs::host::View, 1> views{vs::host::View::raw_uav(1)};
    vs::host::Dispatch d;
    d.roots.emplace_back(0u, vs::host::Root::constants(std::move(ud)));
    d.views = views;
    d.samplers = {};
    d.groups = {1, 1, 1};
    std::vector<std::vector<u32>> out;
    try {
        out = host.dispatch(root, pso, d);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::format("{}: dispatch failed: {}", label, e.what()));
    }
    return out[0][0];
}

void run_cases_on(const HostShader& shader, std::span<const In> cases, const Devices& devices, const Check& check) {
    std::lock_guard one(ONE_AT_A_TIME);
    int ran = 0;
    for (const auto& [label, warp] : devices) {
        std::optional<vs::host::Host> host;
        try {
            host.emplace(warp);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "no SM 6.6 %s device (%s); skipped\n", label, e.what());
            continue;
        }
        Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
        try {
            auto [r, p] = host->compute_pipeline(shader);
            root = r;
            pso = p;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("{}: compute_pipeline failed: {}", label, e.what()));
        }
        for (std::size_t i = 0; i < cases.size(); ++i) {
            In inputs = cases[i];
            u32 got = dispatch_one(*host, root.Get(), pso.Get(), user_data_for(shader, inputs), label);
            check(label, i, inputs, got);
        }
        ++ran;
    }
    bool all_hardware = std::all_of(devices.begin(), devices.end(), [](const auto& d) { return !d.second; });
    if (ran == 0 && all_hardware) {
        // A case for the hardware adapter alone, on a machine without one.
        std::fprintf(stderr, "no SM 6.6 hardware device; a hardware-only case not run\n");
        return;
    }
    if (ran == 0) throw std::runtime_error("no SM 6.6 device available");
}

const Devices HARDWARE_AND_WARP{{"hardware", false}, {"warp", true}};
const Devices HARDWARE_ONLY{{"hardware", false}};

// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for each 4-dword input set
// and invokes check(backend, case_index, inputs, result_dword).
void run_cases(const HostShader& shader, std::span<const In> cases, const Check& check) {
    run_cases_on(shader, cases, HARDWARE_AND_WARP, check);
}
void run_cases(const HostShader& shader, std::initializer_list<In> cases, const Check& check) {
    run_cases_on(shader, std::span(cases.begin(), cases.size()), HARDWARE_AND_WARP, check);
}

bool is_f16_nan(u16 bits) { return (bits & 0x7FFF) > 0x7C00; }

float f16_to_f32(u16 bits) {
    u32 sign = u32(bits >> 15) << 31;
    u32 exp = (bits >> 10) & 0x1F;
    u32 mant = bits & 0x3FF;
    u32 f32_bits;
    if (exp == 0) {
        if (mant == 0) {
            f32_bits = sign;
        } else {
            u32 m = mant;
            u32 e = 127 - 14;
            while ((m & 0x400) == 0) {
                m <<= 1;
                e -= 1;
            }
            f32_bits = sign | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        f32_bits = sign | (0xFFu << 23) | (mant << 13);
    } else {
        f32_bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    return b2f(f32_bits);
}

// MTBUF op vdata, off, s[srsrc:srsrc+3], 0 format:format (at byte 0).
constexpr std::array<u32, 2> mtbuf(u8 op, u8 vdata, u8 srsrc, u8 format) {
    return {0x3Au << 26 | (u32(format) & 0x7F) << 19 | (u32(op) & 7) << 16,
            0x80u << 24 | (u32(op) >> 3 & 1) << 21 | (u32(srsrc) >> 2 & 0x1F) << 16 | u32(vdata) << 8};
}

// Stores v1 to the buffer, then reads it back into v0 with the tbuffer load `op` in buffer format
// `format`.
HostShader tbuffer_load_of_v1(u8 op, u8 format) {
    Words body;
    append(body, buffer_store_dword(1, 8, 0));
    append(body, mtbuf(op, 0, 8, format));
    return build_valu_cs(body);
}

// ---- rs:265 -------------------------------------------------------------------------------------
void scalar_quad_masks_pack_nibbles_and_set_scc() {
    auto pack_words = [](u32 opcode) {
        return std::array<u32, 5>{
            0xBE90'0000u | opcode << 8,  // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),    // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),     // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),      // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),        // v_mov_b32 v0, s16
        };
    };
    const std::array<std::pair<In, std::array<u32, 2>>, 4> cases{{
        {{0, 0, 0, 0}, {0, 0}},
        {{0x8000'0420, 0x0100'0001, 0, 0}, {0x1'0086, 0x1'4186}},
        {{0, 0x8000'0000, 0, 0}, {0, 0x1'8000}},
        {{UINT32_MAX, UINT32_MAX, 0, 0}, {0x1'00ff, 0x1'ffff}},
    }};
    std::array<In, 4> inputs;
    for (std::size_t k = 0; k < 4; ++k) inputs[k] = cases[k].first;
    const std::array<u32, 2> opcodes{0x2c, 0x2d};
    for (std::size_t wide = 0; wide < opcodes.size(); ++wide) {
        u32 opcode = opcodes[wide];
        HostShader shader = build_valu_cs(pack_words(opcode));
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
            ENSURE(got == cases[i].second[wide], "{}, opcode {:#x}: {} != {}", backend, opcode, got, cases[i].second[wide]);
        });
    }
    HostShader high = build_valu_cs({
        0xBE90'2D00u,  // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    });
    run_cases(high, inputs, [&](const char* backend, std::size_t, In, u32 got) {
        ENSURE(got == 0, "{}: high dword: {}", backend, got);
    });
}

// ---- rs:302 -------------------------------------------------------------------------------------
void a_branch_to_fallthrough_keeps_both_values_at_a_join() {
    HostShader shader = build_valu_cs({
        0xBF06'0100u,             // s_cmp_eq_u32 s0, s1
        0xBF85'0002u,             // s_cbranch_scc1 -> the s2 = 1 path
        s_mov_b32(2, 128),        // s2 = 0
        0xBF82'0002u,             // s_branch -> the join
        s_mov_b32(2, 129),        // s2 = 1
        0xBF85'0000u,             // s_cbranch_scc1 -> the following instruction
        vop1(0x001, 0, sgpr(2)),  // join: v0 = s2
    });
    run_cases(shader, {In{7, 7, 0, 0}, In{7, 5, 0, 0}}, [&](const char* backend, std::size_t, In inputs, u32 actual) {
        u32 want = u32(inputs[0] == inputs[1]);
        ENSURE(actual == want, "{}: {} != {}", backend, actual, want);
    });
}

// ---- rs:365 -------------------------------------------------------------------------------------
void tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    const u8 x = 8, xy = 9;
    // (opcode, format, dword stored, dword loaded)
    const std::array<std::tuple<u8, u8, u32, u32>, 4> cases{{
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        {xy, 29, 0x4000'3C00, 0x4000'3C00},
        // 16_16_UINT: the 16-bit integers as they were stored.
        {xy, 27, 0xBEEF'1234, 0xBEEF'1234},
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        {xy, 23, 0xFFFF'0000, 0x3C00'0000},
        // 16_FLOAT, one channel: the high half is 0.
        {x, 13, 0x4000'3C00, 0x0000'3C00},
    }};
    for (const auto& [op, format, stored, want] : cases) {
        HostShader shader = tbuffer_load_of_v1(op, format);
        run_cases(shader, {In{stored, 0, 0, 0}}, [&](const char* backend, std::size_t, In, u32 got) {
            ENSURE(got == want, "{}: d16 opcode {} format {} of {:#010x} = {:#010x}, want {:#010x}", backend, op, format,
                   stored, got, want);
        });
    }
}

// ---- rs:391 -------------------------------------------------------------------------------------
void normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    const u8 x = 0;
    // (format, dword stored, value loaded)
    const std::array<std::tuple<u8, u32, float>, 8> cases{{
        {1, 0xFF, 1.0f},    // 8_UNORM
        {7, 0xFFFF, 1.0f},  // 16_UNORM
        {2, 0x7F, 1.0f},    // 8_SNORM: the top, the one above the lowest, the lowest
        {2, 0x81, -1.0f},
        {2, 0x80, -1.0f},
        {8, 0x7FFF, 1.0f},  // 16_SNORM
        {8, 0x8001, -1.0f},
        {8, 0x8000, -1.0f},
    }};
    for (const auto& [format, stored, want] : cases) {
        HostShader shader = tbuffer_load_of_v1(x, format);
        run_cases(shader, {In{stored, 0, 0, 0}}, [&](const char* backend, std::size_t, In, u32 got) {
            ENSURE(got == f2b(want), "{}: format {} of {:#x} = {:#010x}, want {}", backend, format, stored, got, want);
        });
    }
}

// Codes of the normalized formats sampled below: every 8-bit code; for 16 bits a stride, the
// RX 6700 XT's division misses (257, 261, ...) and the ends.
std::vector<u32> normalized_codes(u32 bits) {
    std::vector<u32> out;
    if (bits <= 10) {
        for (u32 a = 0; a < (1u << bits); ++a) out.push_back(a);
        return out;
    }
    const u32 edges[] = {1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE};
    for (u32 a = 0; a <= 0xFFFF; a += 257) out.push_back(a);
    for (u32 e : edges) out.push_back(e);
    return out;
}

std::vector<In> first_inputs(const std::vector<u32>& codes) {
    std::vector<In> cases;
    for (u32 a : codes) cases.push_back(In{a, 0, 0, 0});
    return cases;
}

// ---- rs:434 -------------------------------------------------------------------------------------
// Every normalized code loads as the float nearest its exact quotient, as the console's buffer unit
// converts it, not as the host divides (the RX 6700 XT gave 125 of the 8-bit codes one ulp high and
// 16-bit codes such as 257 one ulp low): unsigned and signed, 8, 10 (the x channel of 2_10_10_10)
// and 16 bits, the signed formats' two lowest codes -1.0.
void normalized_buffer_formats_load_correctly_rounded_quotients() {
    const u8 x = 0;
    // (format, bits, signed): 8_UNORM, 16_UNORM, 8_SNORM, 16_SNORM, 2_10_10_10_UNORM and _SNORM.
    const std::array<std::tuple<u8, u32, bool>, 6> formats{{
        {1, 8, false}, {7, 16, false}, {2, 8, true}, {8, 16, true}, {50, 10, false}, {51, 10, true}}};
    for (const auto& [format, bits, is_signed] : formats) {
        HostShader shader = tbuffer_load_of_v1(x, format);
        std::vector<In> cases = first_inputs(normalized_codes(bits));
        run_cases(shader, cases, [&](const char* backend, std::size_t, In in, u32 got) {
            u32 a = in[0];
            u32 want;
            if (is_signed) {
                i32 top = (i32(1) << (bits - 1)) - 1;
                i32 s = i32(a << (32 - bits)) >> (32 - bits);
                want = f2b(float(std::max(s, -top)) / float(top));
            } else {
                want = f2b(float(a) / float((1u << bits) - 1));
            }
            const char* sign = is_signed ? "signed" : "unsigned";
            ENSURE(got == want, "{}: {}-bit {} normalized {:#x} loads as {:#010x}, want {:#010x}", backend, bits, sign, a,
                   got, want);
        });
    }
}

// ---- rs:460 -------------------------------------------------------------------------------------
// A title truncates x * (2^n - 1) back to the code it stored (the material index a mesh program
// reads from a vertex colour's alpha, as v_mul_legacy_f32 by 255.0 then v_cvt_u32_f32): every code
// comes back as itself, on hardware and WARP, which a quotient one ulp low breaks.
void normalized_codes_truncate_back_to_themselves() {
    const u8 x = 0;
    const std::array<std::pair<u8, u32>, 2> formats{{{1, 8}, {7, 16}}};
    for (const auto& [format, bits] : formats) {
        u32 scale = f2b(float((1u << bits) - 1));
        Words body;
        append(body, buffer_store_dword(1, 8, 0));
        append(body, mtbuf(x, 0, 8, format));
        // v_mul_legacy_f32 v0, literal, v0; v_cvt_u32_f32 v0, v0.
        body.push_back(0x07u << 25 | 0xFF);
        body.push_back(scale);
        body.push_back(vop1(0x07, 0, vgpr(0)));
        HostShader shader = build_valu_cs(body);
        std::vector<In> cases = first_inputs(normalized_codes(bits));
        run_cases(shader, cases, [&](const char* backend, std::size_t, In in, u32 got) {
            u32 a = in[0];
            ENSURE(got == a, "{}: {}-bit code {} comes back as {}", backend, bits, a, got);
        });
    }
}

// ---- rs:478 -------------------------------------------------------------------------------------
void packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves. The same GFX10
    // source field (248) has a different bit pattern for a packed floating-point and a packed
    // integer instruction.
    const std::array<std::pair<std::array<u32, 2>, u32>, 3> cases{{
        // v_pk_add_f16 v0, 0x3118, v1
        {{0xcc0f4000, 0x1a0202f8}, 0x0000'3118},
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        {{0xcc0f4000, 0x120202f8}, 0x3118'3118},
        // v_pk_add_u16 v0, 0x3e22f983, v1
        {{0xcc0a4000, 0x1a0202f8}, 0x3e22'f983},
    }};
    for (const auto& [words, expected] : cases) {
        HostShader shader = build_valu_cs(words);
        run_cases(shader, {In{0, 0, 0, 0}}, [&](const char* backend, std::size_t, In, u32 got) {
            ENSURE(got == expected, "{}: {}: {:#x} != {:#x}", backend, hex_words(words), got, expected);
        });
    }
}

// The float v_cvt_f32_f64 gives for x: Rust's rounding (to nearest even), with results below the
// smallest normal float flushed to signed zero as the console's default float mode does.
u32 cvt_f32_f64_reference(double x) {
    float r = static_cast<float>(x);
    bool subnormal = std::fpclassify(r) == FP_SUBNORMAL;
    if (subnormal || (r == 0.0f && x != 0.0)) return f2b(r) & 0x8000'0000;
    return f2b(r);
}

std::vector<In> f64_inputs(std::span<const double> values) {
    std::vector<In> inputs;
    for (double x : values) {
        u64 b = d2b(x);
        inputs.push_back(In{u32(b), u32(b >> 32), 0, 0});
    }
    return inputs;
}

// ---- rs:511 -------------------------------------------------------------------------------------
// #[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering
// (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
void v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    HostShader shader = build_valu_cs({vop1(0x00f, 0, vgpr(1))});
    double tie_even = 1.0 + double(FLT_EPSILON) / 2.0;  // ties to 1.0
    double tie_odd = 1.0 + 1.5 * double(FLT_EPSILON);   // ties up to 1 + 2 ulp
    const std::vector<double> values{
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::numbers::pi,
        1.0e-3,
        123'456'789.123,
        tie_even,
        tie_odd,
        1.0 + double(FLT_EPSILON) * 0.75,
        double(FLT_MAX),
        double(FLT_MAX) * 1.000'000'1,  // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40,  // a float denormal: flushed
        -1.0e-300,
        double(FLT_MIN),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
    };
    std::vector<In> inputs = f64_inputs(values);
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
        double x = values[i];
        if (std::isnan(x)) {
            ENSURE(std::isnan(b2f(got)), "{}: cvt({}) = {:#010x}, want NaN", backend, x, got);
            return;
        }
        u32 want = cvt_f32_f64_reference(x);
        ENSURE(got == want, "{}: cvt({:e}) = {:#010x}, want {:#010x}", backend, x, got, want);
    });
}

// The lane v_permlane16_b32 (cross false) or v_permlanex16_b32 reads for `lane`, with selector
// dwords s1 (lanes 0-7 of a row) and s2.
u32 permlane_source(u32 lane, u32 s1, u32 s2, bool cross) {
    u64 sel = u64(s1) | u64(s2) << 32;
    u32 pick = u32(sel >> (4 * (lane & 15)) & 15);
    u32 row = cross ? ((lane & ~31u) | ((lane & 16) ^ 16)) : (lane & ~15u);
    return row | pick;
}

// ---- rs:571 -------------------------------------------------------------------------------------
void v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1); lane `read` of v1 into
    // s12, then into v0 for the store.
    const std::array<std::pair<u32, u32>, 3> selectors{{
        {0x7654'3210, 0xfedc'ba98},  // identity within the row
        {0x89ab'cdef, 0x0123'4567},  // reversed
        {0x1111'1111, 0xf0f0'f0f0},  // repeats
    }};
    const std::array<std::pair<u16, bool>, 2> ops{{{0x377, false}, {0x378, true}}};
    for (const auto& [op, cross] : ops) {
        for (u8 read : {u8(0), u8(5), u8(15), u8(16), u8(22), u8(31)}) {
            Words body;
            append(body, vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            append(body, vop3(0x360, 12, vgpr(1), u16(128 + u16(read)), 0));
            body.push_back(vop1(0x001, 0, sgpr(12)));  // v_mov_b32 v0, s12
            HostShader shader = build_cs(body, 32, 0);
            std::vector<In> inputs;
            for (const auto& [a, b] : selectors) inputs.push_back(In{a, b, 0, 0});
            // Cross-lane code declares WaveSize(32), which WARP's pipeline creation refuses:
            // hardware only.
            run_cases_on(shader, inputs, HARDWARE_ONLY, [&](const char* backend, std::size_t i, In inp, u32 got) {
                u32 want = permlane_source(u32(read), inp[0], inp[1], cross);
                ENSURE(got == want, "{}: op {:#x} selectors [{:08x}, {:08x}], lane {}: {:#x} != {:#x}", backend, op,
                       selectors[i].first, selectors[i].second, read, got, want);
            });
        }
    }
}

// ---- rs:603 -------------------------------------------------------------------------------------
void v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    HostShader shader = build_valu_cs(vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    const u32 nan = f2b(std::numeric_limits<float>::quiet_NaN());
    auto f = f2b;

    const std::vector<std::pair<In, std::optional<float>>> cases{
        // Normal medians (all orderings).
        {{f(1.0f), f(2.0f), f(3.0f), 0}, 2.0f},
        {{f(3.0f), f(1.0f), f(2.0f), 0}, 2.0f},
        {{f(-5.0f), f(0.0f), f(5.0f), 0}, 0.0f},
        {{f(1.0f), f(1.0f), f(2.0f), 0}, 1.0f},
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        {{f(1.0f), f(2.0f), nan, 0}, 1.0f},
        {{nan, f(1.0f), f(0.0f), 0}, 0.0f},
        {{nan, f(0.0f), f(1.0f), 0}, 0.0f},
        {{f(2.0f), nan, f(1.0f), 0}, 1.0f},
        {{f(-3.0f), nan, f(4.0f), 0}, -3.0f},
        // Two NaNs: returns the sole non-NaN operand.
        {{nan, f(5.0f), nan, 0}, 5.0f},
        {{nan, nan, f(-2.0f), 0}, -2.0f},
        {{f(7.0f), nan, nan, 0}, 7.0f},
        // Three NaNs: returns NaN.
        {{nan, nan, nan, 0}, std::nullopt},
    };

    std::vector<In> inputs;
    for (const auto& c : cases) inputs.push_back(c.first);
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        std::optional<float> expect = cases[i].second;
        float got_f = b2f(got);
        if (expect) {
            float want = *expect;
            ENSURE(got_f == want, "{}: v_med3_f32({}, {}, {}) = {} ({:#010x}), want {}", backend, b2f(inp[0]),
                   b2f(inp[1]), b2f(inp[2]), got_f, got, want);
        } else {
            ENSURE(std::isnan(got_f), "{}: v_med3_f32(NaN, NaN, NaN) = {} ({:#010x}), want NaN", backend, got_f, got);
        }
    });
}

// ---- rs:651 -------------------------------------------------------------------------------------
// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and FMin compare and select,
// which can give either zero.
void max_and_min_order_negative_zero_below_positive_zero() {
    const u32 pz = 0, nz = 0x8000'0000;
    // Each case has a -0 and a +0 among its first two operands.
    const std::array<In, 4> cases{{
        {nz, pz, nz, 0},
        {pz, nz, nz, 0},
        {nz, pz, pz, 0},
        {pz, nz, pz, 0},
    }};
    const std::array<std::tuple<const char*, u16, u32>, 4> ops{{
        {"v_max_f32", 0x110, pz},
        {"v_min_f32", 0x10F, nz},
        {"v_max3_f32", 0x154, pz},
        {"v_min3_f32", 0x151, nz},
    }};
    for (const auto& [name, op, want] : ops) {
        HostShader shader = build_valu_cs(vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(shader, cases, [&](const char* backend, std::size_t, In inp, u32 got) {
            ENSURE(got == want, "{}: {} {}: {:#x} != {:#x}", backend, name, hex_in(inp), got, want);
        });
    }
}

// ---- rs:677 -------------------------------------------------------------------------------------
// The f16 forms go through the same max_min helper, after widening to f32.
void f16_max_and_min_order_negative_zero_below_positive_zero() {
    const u32 pz = 0, nz = 0x8000;
    const std::array<In, 4> cases{{
        {nz, pz, nz, 0},
        {pz, nz, nz, 0},
        {nz, pz, pz, 0},
        {pz, nz, pz, 0},
    }};
    const std::array<std::tuple<const char*, u16, u32>, 4> ops{{
        {"v_max_f16", 0x139, pz},
        {"v_min_f16", 0x13A, nz},
        {"v_max3_f16", 0x354, pz},
        {"v_min3_f16", 0x351, nz},
    }};
    for (const auto& [name, op, want] : ops) {
        HostShader shader = build_valu_cs(vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(shader, cases, [&](const char* backend, std::size_t, In inp, u32 got) {
            // The f16 result is the low half of v0.
            ENSURE((got & 0xFFFF) == want, "{}: {} {}: {:#x} != {:#x}", backend, name, hex_in(inp), got & 0xFFFF, want);
        });
    }
}

// ---- rs:702 -------------------------------------------------------------------------------------
void v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    using Pick = u32 (*)(std::array<u32, 3>);
    const std::array<std::tuple<const char*, u16, Pick>, 6> ops{{
        {"v_min3_i32", 0x152,
         [](std::array<u32, 3> t) {
             return *std::min_element(t.begin(), t.end(), [](u32 a, u32 b) { return i32(a) < i32(b); });
         }},
        {"v_min3_u32", 0x153, [](std::array<u32, 3> t) { return *std::min_element(t.begin(), t.end()); }},
        // Rust max_by_key takes the last of equal maxima; equal keys are equal values here.
        {"v_max3_i32", 0x155,
         [](std::array<u32, 3> t) {
             return *std::max_element(t.begin(), t.end(), [](u32 a, u32 b) { return i32(a) < i32(b); });
         }},
        {"v_max3_u32", 0x156, [](std::array<u32, 3> t) { return *std::max_element(t.begin(), t.end()); }},
        {"v_med3_i32", 0x158,
         [](std::array<u32, 3> t) {
             std::stable_sort(t.begin(), t.end(), [](u32 a, u32 b) { return i32(a) < i32(b); });
             return t[1];
         }},
        {"v_med3_u32", 0x159,
         [](std::array<u32, 3> t) {
             std::sort(t.begin(), t.end());
             return t[1];
         }},
    }};
    // Every ordering of three values, with the sign bit set on some so that the signed and unsigned
    // answers differ.
    const std::array<u32, 5> values{0u, 1, 0x7FFF'FFFF, 0x8000'0000, 0xFFFF'FFFF};
    std::vector<In> cases;
    for (u32 a : values)
        for (u32 b : values)
            for (u32 c : values) cases.push_back(In{a, b, c, 0});
    for (const auto& [name, op, pick] : ops) {
        HostShader shader = build_valu_cs(vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(shader, cases, [&](const char* backend, std::size_t, In inp, u32 got) {
            u32 want = pick({inp[0], inp[1], inp[2]});
            ENSURE(got == want, "{}: {}({:#010x}, {:#010x}, {:#010x}) = {:#010x}, want {:#010x}", backend, name, inp[0],
                   inp[1], inp[2], got, want);
        });
    }
}

// ---- rs:748 -------------------------------------------------------------------------------------
void v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    HostShader shader = build_valu_cs(vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    const u32 h0 = 0x0000;   // 0.0
    const u32 h1 = 0x3C00;   // 1.0
    const u32 h2 = 0x4000;   // 2.0
    const u32 h3 = 0x4200;   // 3.0
    const u32 h5 = 0x4500;   // 5.0
    const u32 hn2 = 0xC000;  // -2.0
    const u32 nan = 0x7E00;

    const std::vector<std::pair<In, std::optional<u16>>> cases{
        {{h1, h2, h3, 0}, u16(h2)},
        {{hn2, h0, h3, 0}, u16(h0)},
        {{h1, h2, nan, 0}, u16(h1)},
        {{nan, h1, h0, 0}, u16(h0)},
        {{nan, h0, h1, 0}, u16(h0)},
        {{h2, nan, h1, 0}, u16(h1)},
        {{nan, h5, nan, 0}, u16(h5)},
        {{nan, nan, nan, 0}, std::nullopt},
    };

    std::vector<In> inputs;
    for (const auto& c : cases) inputs.push_back(c.first);
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        std::optional<u16> expect = cases[i].second;
        u16 got_h = u16(got);
        if (expect) {
            u16 want = *expect;
            ENSURE(got_h == want, "{}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {:#06x}, want {:#06x}", backend, inp[0],
                   inp[1], inp[2], got_h, want);
        } else {
            ENSURE(is_f16_nan(got_h), "{}: v_med3_f16(NaN, NaN, NaN) = {:#06x}, want NaN", backend, got_h);
        }
    });
}

// ---- rs:789 -------------------------------------------------------------------------------------
// v_cmp_lg_* is "less or greater": ordered and not equal, so false when either operand is a NaN
// (RDNA2 ISA; the same on gfx1036 hardware).
void v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    const u32 nan = f2b(std::numeric_limits<float>::quiet_NaN());
    const u32 inf = f2b(std::numeric_limits<float>::infinity());
    const u32 ninf = f2b(-std::numeric_limits<float>::infinity());
    auto f = f2b;
    using Cases = std::vector<std::pair<In, u32>>;
    const Cases f32_cases{
        {{f(1.0f), f(2.0f), 0, 0}, 1},
        {{f(2.0f), f(1.0f), 0, 0}, 1},
        {{f(1.0f), f(1.0f), 0, 0}, 0},
        {{f(0.0f), f(-0.0f), 0, 0}, 0},
        {{inf, ninf, 0, 0}, 1},
        {{inf, inf, 0, 0}, 0},
        {{nan, f(1.0f), 0, 0}, 0},
        {{f(1.0f), nan, 0, 0}, 0},
        {{nan, nan, 0, 0}, 0},
        {{0x7F80'0001, f(0.0f), 0, 0}, 0},  // a signalling NaN
    };
    const Cases f16_cases{
        {{0x3C00, 0x4000, 0, 0}, 1},
        {{0x4000, 0x3C00, 0, 0}, 1},
        {{0x3C00, 0x3C00, 0, 0}, 0},
        {{0x0000, 0x8000, 0, 0}, 0},
        {{0x7C00, 0xFC00, 0, 0}, 1},
        {{0x7E00, 0x3C00, 0, 0}, 0},
        {{0x3C00, 0x7E00, 0, 0}, 0},
        {{0x7E00, 0x7E00, 0, 0}, 0},
        {{0x7C01, 0x0000, 0, 0}, 0},            // a signalling NaN
        {{0xABCD'3C00, 0x1234'7E00, 0, 0}, 0},  // only the low halves are read
    };
    const std::array<std::tuple<u16, const char*, const Cases*>, 2> forms{{
        {0x005, "v_cmp_lg_f32", &f32_cases},
        {0x0cd, "v_cmp_lg_f16", &f16_cases},
    }};
    for (const auto& [op, what, cases_ptr] : forms) {
        const Cases& cases = *cases_ptr;
        Words body;
        append(body, vop3(op, 20, vgpr(1), vgpr(2), 0x80));
        append(body, vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        HostShader shader = build_valu_cs(body);
        std::vector<In> inputs;
        for (const auto& c : cases) inputs.push_back(c.first);
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
            u32 want = cases[i].second;
            ENSURE(got == want, "{}: {}({:#010x}, {:#010x}) = {}, want {}", backend, what, inp[0], inp[1], got, want);
        });
    }
}

// ---- rs:841 -------------------------------------------------------------------------------------
void v_cvt_off_f32_i4_uses_the_signed_low_nibble() {
    // v_cvt_off_f32_i4 v0, v1 (VOP1 opcode 0x00e)
    HostShader shader = build_valu_cs({vop1(0x00e, 0, vgpr(1))});
    // C++20: << of a signed value wraps as Rust's does; >> of a negative one is arithmetic.
    auto nibble = [](u32 n) { return float((i32(n) << 28) >> 28) / 16.0f; };

    std::vector<u32> cases;
    for (u32 k = 0; k < 16; ++k) cases.push_back(k);
    // Bits above the low nibble are ignored.
    for (u32 k : {0xABCD'EF05u, 0xFFFF'FFF8u, 0x1234'5670u, 0x8000'000Fu, 0x0000'00F7u}) cases.push_back(k);
    std::vector<In> inputs = first_inputs(cases);
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = f2b(nibble(cases[i]));
        ENSURE(got == want, "{}: v_cvt_off_f32_i4({:#010x}) = {:#010x} ({}), want {:#010x}", backend, inp[0], got, b2f(got),
               want);
    });
    // Spot-check the reference itself: 0x5 -> 0.3125, 0x8 -> -0.5, 0xF -> -0.0625.
    CHECK(nibble(0xABCD'EF05) == 0.3125f);
    CHECK(nibble(8) == -0.5f);
    CHECK(nibble(0xF) == -0.0625f);
}

// ---- rs:866 -------------------------------------------------------------------------------------
void v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    HostShader shader = build_valu_cs({vop1(0x020, 0, vgpr(1))});
    auto f = f2b;
    const u32 below_one = 0x3F7F'FFFF;

    const std::vector<std::pair<u32, std::optional<u32>>> cases{
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        {f(-1e-10f), below_one},
        {f(-1e-8f), below_one},
        {f(-std::ldexp(1.0f, -25)), below_one},
        {f(-std::ldexp(1.0f, -24)), below_one},
        // Normal values.
        {f(0.0f), f(0.0f)},
        {f(-0.0f), f(0.0f)},
        {f(0.25f), f(0.25f)},
        {f(3.75f), f(0.75f)},
        {f(-0.25f), f(0.75f)},
        {f(-3.25f), f(0.75f)},
        {f(1.0f), f(0.0f)},
        {f(-1.0f), f(0.0f)},
        // Non-finite inputs yield NaN.
        {f(std::numeric_limits<float>::quiet_NaN()), std::nullopt},
        {f(std::numeric_limits<float>::infinity()), std::nullopt},
        {f(-std::numeric_limits<float>::infinity()), std::nullopt},
    };

    std::vector<In> inputs;
    for (const auto& [x, _] : cases) inputs.push_back(In{x, 0, 0, 0});
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        std::optional<u32> expect = cases[i].second;
        if (expect) {
            u32 want = *expect;
            ENSURE(got == want, "{}: v_fract_f32({} / {:#010x}) = {:#010x} ({}), want {:#010x}", backend, b2f(inp[0]),
                   inp[0], got, b2f(got), want);
        } else {
            ENSURE(std::isnan(b2f(got)), "{}: v_fract_f32({}) = {:#010x}, want NaN", backend, b2f(inp[0]), got);
        }
    });
}

// ---- rs:915 -------------------------------------------------------------------------------------
void v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    HostShader shader = build_valu_cs({vop1(0x05f, 0, vgpr(1))});
    const u16 below_one = 0x3BFF;  // 1.0 - 2^-11

    CHECK(f16_to_f32(0x9400) == -std::ldexp(1.0f, -10));
    CHECK(f16_to_f32(0x9000) == -std::ldexp(1.0f, -11));
    CHECK(f16_to_f32(0x8C00) == -std::ldexp(1.0f, -12));

    const std::vector<std::pair<u16, std::optional<u16>>> cases{
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        {0x8C00, below_one},  // -2^-12
        {0x8400, below_one},  // -2^-14
        {0x8001, below_one},  // -2^-24 (smallest negative f16 subnormal)
        {0x9000, below_one},  // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        {0x9400, u16(0x3BFE)},  // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        {0x0000, u16(0x0000)},  // 0.0 -> 0.0
        {0x3400, u16(0x3400)},  // 0.25 -> 0.25
        {0xB400, u16(0x3A00)},  // -0.25 -> 0.75
        {0x4300, u16(0x3800)},  // 3.5 -> 0.5
        {0x3C00, u16(0x0000)},  // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        {0x7E00, std::nullopt},  // NaN
        {0x7C00, std::nullopt},  // +inf
        {0xFC00, std::nullopt},  // -inf
    };

    std::vector<In> inputs;
    for (const auto& [x, _] : cases) inputs.push_back(In{u32(x), 0, 0, 0});
    run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        std::optional<u16> expect = cases[i].second;
        u16 got_h = u16(got);
        if (expect) {
            u16 want = *expect;
            ENSURE(got_h == want, "{}: v_fract_f16({:#06x}) = {:#06x}, want {:#06x}", backend, inp[0], got_h, want);
        } else {
            ENSURE(is_f16_nan(got_h), "{}: v_fract_f16({:#06x}) = {:#06x}, want NaN", backend, inp[0], got_h);
        }
    });
}

// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or 65535, round the exact
// product to nearest even; NaN gives 0.
u16 cvt_norm16_reference(u16 bits, bool is_signed) {
    float x = f16_to_f32(bits);
    if (std::isnan(x)) return 0;
    float lo = is_signed ? -1.0f : 0.0f;
    double scale = is_signed ? 32767.0 : 65535.0;
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    double p = double(std::clamp(x, lo, 1.0f)) * scale;
    // round_ties_even: nearbyint in the default rounding mode (to nearest even); `as i32` of a value
    // within [-32767, 65535], then `as u16` wraps.
    return u16(i32(std::nearbyint(p)));
}

// ---- rs:983 -------------------------------------------------------------------------------------
void v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would round before the rounding
    // to an integer, ties, and the edges.
    std::vector<u16> bits;
    for (u32 b = 0; b <= 0xFFFF; b += 251) bits.push_back(u16(b));
    for (u16 b : {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
                  0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
                  0x3A9B, 0xBA9B, 0x2E66, 0x1400})
        bits.push_back(b);
    const std::array<std::tuple<u16, bool, const char*>, 2> ops{{
        {0x063, true, "v_cvt_norm_i16_f16"},
        {0x064, false, "v_cvt_norm_u16_f16"},
    }};
    for (const auto& [op, is_signed, what] : ops) {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        HostShader shader = build_valu_cs({vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))});
        std::vector<In> inputs;
        for (u16 b : bits) inputs.push_back(In{u32(b), 0, 0, 0});
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
            u16 want = cvt_norm16_reference(bits[i], is_signed);
            ENSURE(got == u32(want), "{}: {}({:#06x}) = {:#010x}, want {:#06x}", backend, what, bits[i], got, want);
        });
    }
}

// The approximate checks of the sin/cos tests: within `tol` of the value or NaN.
void check_near(const char* backend, const char* what, std::optional<float> expect, In inp, float got_f, float tol,
                const char* file, int line) {
    if (expect) {
        float want = *expect;
        ensure(std::fabs(got_f - want) < tol,
               std::format("{}: {}({}) = {}, want {} (err = {})", backend, what, b2f(inp[0]), got_f, want,
                           std::fabs(got_f - want)),
               file, line);
    } else {
        ensure(std::isnan(got_f), std::format("{}: {}({}) = {}, want NaN", backend, what, b2f(inp[0]), got_f), file, line);
    }
}

// ---- rs:1012 ------------------------------------------------------------------------------------
void v_sin_and_v_cos_f32_reduce_large_turns() {
    HostShader sin_shader = build_valu_cs({vop1(0x035, 0, vgpr(1))});
    HostShader cos_shader = build_valu_cs({vop1(0x036, 0, vgpr(1))});
    auto f = f2b;
    const float half_sqrt2 = 0.707106781186547524400844362104849039f;  // Rust std::f32::consts::FRAC_1_SQRT_2
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float INF = std::numeric_limits<float>::infinity();
    using Cases = std::vector<std::pair<float, std::optional<float>>>;

    const Cases sin_cases{
        {0.0f, 0.0f},
        {0.25f, 1.0f},
        {255.0f, 0.0f},
        {255.25f, 1.0f},
        {255.5f, 0.0f},
        {-255.25f, -1.0f},
        {10'000.125f, half_sqrt2},
        {NaN, std::nullopt},
        {INF, std::nullopt},
        {-INF, std::nullopt},
    };
    std::vector<In> sin_inputs;
    for (const auto& [x, _] : sin_cases) sin_inputs.push_back(In{f(x), 0, 0, 0});
    run_cases(sin_shader, sin_inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        check_near(backend, "v_sin_f32", sin_cases[i].second, inp, b2f(got), 1e-5f, __FILE__, __LINE__);
    });

    const Cases cos_cases{
        {0.0f, 1.0f},
        {0.25f, 0.0f},
        {255.0f, 1.0f},
        {255.25f, 0.0f},
        {255.5f, -1.0f},
        {-255.5f, -1.0f},
        {10'000.125f, half_sqrt2},
        {NaN, std::nullopt},
        {INF, std::nullopt},
        {-INF, std::nullopt},
    };
    std::vector<In> cos_inputs;
    for (const auto& [x, _] : cos_cases) cos_inputs.push_back(In{f(x), 0, 0, 0});
    run_cases(cos_shader, cos_inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        check_near(backend, "v_cos_f32", cos_cases[i].second, inp, b2f(got), 1e-5f, __FILE__, __LINE__);
    });

    // Main's trig_turns preserves sin(-0.0) = -0.0 and produces exact +0.0 at half-turns (sin) and
    // quarter-turns (cos).
    const std::vector<std::pair<u32, u32>> exact_sin{
        {0x8000'0000, 0x8000'0000},  // sin(-0.0) = -0.0
        {f(255.0f), 0},              // sin(255.0) = +0.0
        {f(255.5f), 0},              // sin(255.5) = +0.0
    };
    std::vector<In> exact_sin_in;
    for (const auto& [x, _] : exact_sin) exact_sin_in.push_back(In{x, 0, 0, 0});
    run_cases(sin_shader, exact_sin_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = exact_sin[i].second;
        ENSURE(got == want, "{}: v_sin_f32({:#010x}) = {:#010x}, want {:#010x}", backend, inp[0], got, want);
    });
    const std::vector<std::pair<u32, u32>> exact_cos{
        {f(0.25f), 0},    // cos(0.25) = +0.0
        {f(255.25f), 0},  // cos(255.25) = +0.0
    };
    std::vector<In> exact_cos_in;
    for (const auto& [x, _] : exact_cos) exact_cos_in.push_back(In{x, 0, 0, 0});
    run_cases(cos_shader, exact_cos_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = exact_cos[i].second;
        ENSURE(got == want, "{}: v_cos_f32({:#010x}) = {:#010x}, want {:#010x}", backend, inp[0], got, want);
    });
}

// ---- rs:1112 ------------------------------------------------------------------------------------
void v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    const std::vector<std::tuple<u16, u16, u16>> cases{
        // (f16 input, u16 result, i16 result)
        {0x0000, 0, 0},           // 0
        {0x8000, 0, 0},           // -0
        {0x3E00, 1, 1},           // 1.5 -> 1 (toward zero)
        {0xBE00, 0, 0xFFFF},      // -1.5 -> 0 / -1
        {0x57D0, 125, 125},       // 125.0
        {0x7800, 32768, 32767},   // 32768 saturates i16
        {0x77FF, 32752, 32752},   // 32752, the largest f16 below 32768
        {0xF800, 0, 0x8000},      // -32768 fits i16
        {0xF801, 0, 0x8000},      // -32800 saturates i16
        {0x7BFF, 65504, 32767},   // max finite f16
        {0xFBFF, 0, 0x8000},      // -65504
        {0x7C00, 0xFFFF, 32767},  // +inf
        {0xFC00, 0, 0x8000},      // -inf
        {0x7E00, 0, 0},           // NaN
        {0xFE00, 0, 0},           // -NaN
        {0x0001, 0, 0},           // smallest subnormal
    };
    const std::array<std::tuple<u16, const char*, std::size_t>, 2> ops{{
        {0x052, "v_cvt_u16_f16", 0},
        {0x053, "v_cvt_i16_f16", 1},
    }};
    for (const auto& [op, what, pick] : ops) {
        HostShader shader = build_valu_cs({vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))});
        std::vector<In> inputs;
        for (const auto& c : cases) inputs.push_back(In{u32(std::get<0>(c)), 0, 0, 0});
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
            u16 want = pick == 0 ? std::get<1>(cases[i]) : std::get<2>(cases[i]);
            ENSURE(got == u32(want), "{}: {}({:#06x}) = {:#010x}, want {:#06x}", backend, what, std::get<0>(cases[i]), got,
                   want);
        });
    }
}

// ---- rs:1152 ------------------------------------------------------------------------------------
void v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    // Only integers f16 holds exactly: how the others round is the narrowing's business, not this
    // conversion's. nullopt: the result depends on rounding.
    using O = std::optional<u16>;
    const std::vector<std::tuple<u16, O, O>> cases{
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        {0x0000, O(0x0000), O(0x0000)},     // 0
        {0x0001, O(0x3C00), O(0x3C00)},     // 1.0
        {0x07FF, O(0x67FF), O(0x67FF)},     // 2047, the last integer below 2048 that f16 holds
        {0x8000, O(0x7800), O(0xF800)},     // 32768 / -32768
        {0xFC00, O(0x7BE0), O(0xE400)},     // 64512 / -1024
        {0xFFFE, std::nullopt, O(0xC000)},  // 65534 does not fit f16 / -2
        {0xFFFF, std::nullopt, O(0xBC00)},  // 65535 does not fit f16 / -1
    };
    const std::array<std::tuple<u16, const char*, std::size_t>, 2> ops{{
        {0x050, "v_cvt_f16_u16", 0},
        {0x051, "v_cvt_f16_i16", 1},
    }};
    for (const auto& [op, what, pick] : ops) {
        HostShader shader = build_valu_cs({vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))});
        std::vector<In> inputs;
        for (const auto& c : cases) inputs.push_back(In{u32(std::get<0>(c)), 0, 0, 0});
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
            O want = pick == 0 ? std::get<1>(cases[i]) : std::get<2>(cases[i]);
            if (!want) return;
            ENSURE(got == u32(*want), "{}: {}({:#06x}) = {:#010x}, want {:#06x}", backend, what, std::get<0>(cases[i]),
                   got, *want);
        });
    }
}

// The f16 forms of check_near: the result is the low half, compared as f32.
void check_near_f16(const char* backend, const char* what, std::optional<float> expect, In inp, u32 got,
                    const char* file, int line) {
    u16 got_h = u16(got);
    if (expect) {
        float want = *expect;
        float got_f = f16_to_f32(got_h);
        ensure(std::fabs(got_f - want) < 1e-3f,
               std::format("{}: {}({:#06x} = {}) = {} ({:#06x}), want {}", backend, what, inp[0], f16_to_f32(u16(inp[0])),
                           got_f, got_h, want),
               file, line);
    } else {
        ensure(is_f16_nan(got_h), std::format("{}: {}({:#06x}) = {:#06x}, want NaN", backend, what, inp[0], got_h), file,
               line);
    }
}

// ---- rs:1186 ------------------------------------------------------------------------------------
void v_sin_and_v_cos_f16_reduce_large_turns() {
    HostShader sin_shader = build_valu_cs({vop1(0x060, 0, vgpr(1))});
    HostShader cos_shader = build_valu_cs({vop1(0x061, 0, vgpr(1))});

    // Sanity-check the f16 bit encodings of large turns.
    CHECK(f16_to_f32(0x5BF8) == 255.0f);
    CHECK(f16_to_f32(0x5BFA) == 255.25f);
    CHECK(f16_to_f32(0x5BFC) == 255.5f);
    CHECK(f16_to_f32(0x7BFF) == 65504.0f);

    using Cases = std::vector<std::pair<u16, std::optional<float>>>;
    const Cases sin_cases{
        {0x5BF8, 0.0f},          // 255.0
        {0x5BFA, 1.0f},          // 255.25
        {0x5BFC, 0.0f},          // 255.5
        {0xDBFA, -1.0f},         // -255.25
        {0x7BFF, 0.0f},          // 65504.0 (largest finite f16, exact integer)
        {0x7E00, std::nullopt},  // NaN
        {0x7C00, std::nullopt},  // +inf
        {0xFC00, std::nullopt},  // -inf
    };
    std::vector<In> sin_inputs;
    for (const auto& [x, _] : sin_cases) sin_inputs.push_back(In{u32(x), 0, 0, 0});
    run_cases(sin_shader, sin_inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        check_near_f16(backend, "v_sin_f16", sin_cases[i].second, inp, got, __FILE__, __LINE__);
    });

    const Cases cos_cases{
        {0x5BF8, 1.0f},          // 255.0
        {0x5BFA, 0.0f},          // 255.25
        {0x5BFC, -1.0f},         // 255.5
        {0x7BFF, 1.0f},          // 65504.0 (largest finite f16, exact integer)
        {0x7E00, std::nullopt},  // NaN
        {0x7C00, std::nullopt},  // +inf
        {0xFC00, std::nullopt},  // -inf
    };
    std::vector<In> cos_inputs;
    for (const auto& [x, _] : cos_cases) cos_inputs.push_back(In{u32(x), 0, 0, 0});
    run_cases(cos_shader, cos_inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        check_near_f16(backend, "v_cos_f16", cos_cases[i].second, inp, got, __FILE__, __LINE__);
    });
}

// ---- rs:1267 ------------------------------------------------------------------------------------
// A 32-bit literal in a signed 64-bit operand is sign-extended (an unsigned one's is
// zero-extended): -100 compared with v[1:2], and shifted by s_ashr_i64.
void signed_64_bit_operands_sign_extend_their_literal() {
    constexpr u32 MINUS_100 = u32(-100);
    using Cases = std::vector<std::pair<In, u32>>;
    auto check = [](const char* name, std::span<const u32> body, const Cases& cases) {
        HostShader shader = build_valu_cs(body);
        std::vector<In> inputs;
        for (const auto& c : cases) inputs.push_back(c.first);
        run_cases(shader, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
            u32 want = cases[i].second;
            ENSURE(got == want, "{}: {} {}: {:#x} != {:#x}", backend, name, hex_in(inp), got, want);
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then v_cndmask_b32 v0, 0, 1, s6.
    auto compare = [](u16 op) {
        auto [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        auto [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        return std::array<u32, 5>{a, b, MINUS_100, c, d};
    };
    // An i64 as the two input dwords of v[1:2].
    auto pair = [](i64 n) { return In{u32(n), u32(n >> 32), 0, 0}; };
    const Cases eq{{pair(-100), 1}, {In{MINUS_100, 0, 0, 0}, 0}, {pair(0), 0}};
    check("v_cmp_eq_i64", compare(0x0A2), eq);
    const Cases lt{{pair(-200), 1}, {pair(-50), 0}, {pair(0), 0}};
    check("v_cmp_lt_i64", compare(0x0A1), lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    const std::array<u32, 3> ashr{sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))};
    check("s_ashr_i64", ashr, Cases{{In{0, 0, 0, 0}, UINT32_MAX}});
}

// ---- rs:1299 ------------------------------------------------------------------------------------
void bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    HostShader v_signed = build_valu_cs(vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    HostShader v_unsigned = build_valu_cs(vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    HostShader s_signed = build_valu_cs({sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))});

    using Cases = std::vector<std::pair<In, u32>>;
    const Cases signed_cases{
        // Field crosses bit 31: extends from bit 31.
        {{0xd2f9'0159, 28, 16, 0}, 0xffff'fffd},
        {{0x8000'0000, 31, 4, 0}, 0xffff'ffff},
        {{0x7000'0000, 28, 8, 0}, 0x0000'0007},
        // Normal fields inside 32 bits.
        {{0x0000'0080, 4, 4, 0}, 0xffff'fff8},
        {{0x0000'0070, 4, 4, 0}, 0x0000'0007},
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0), while 31 extracts all
        // remaining bits from offset 4.
        {{0xffff'ffff, 4, 0, 0}, 0},
        {{0x8000'0000, 4, 31, 0}, 0xf800'0000},
    };
    std::vector<In> v_in;
    for (const auto& c : signed_cases) v_in.push_back(c.first);
    run_cases(v_signed, v_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = signed_cases[i].second;
        ENSURE(got == want, "{}: v_bfe_i32({:#010x}, {}, {}) = {:#010x}, want {:#010x}", backend, inp[0], inp[1], inp[2],
               got, want);
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(v_signed, {In{0x8000'0000, 4, 32, 0}}, [&](const char* backend, std::size_t, In, u32 got) {
        ENSURE(got == 0, "{}: v_bfe_i32 width 32 should mask to 0", backend);
    });

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    Cases s_cases = signed_cases;
    s_cases.push_back({In{0x8000'0000, 4, 32, 0}, 0xf800'0000});
    std::vector<In> s_in;
    for (const auto& [inp, _] : s_cases) s_in.push_back(In{inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0});
    run_cases(s_signed, s_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = s_cases[i].second;
        ENSURE(got == want, "{}: s_bfe_i32({:#010x}, {:#010x}) = {:#010x}, want {:#010x}", backend, inp[0], inp[1], got,
               want);
    });

    const Cases unsigned_cases{
        {{0xd2f9'0159, 28, 16, 0}, 0x0000'000d},
        {{0x8000'0000, 31, 4, 0}, 0x0000'0001},
        {{0xffff'ffff, 4, 0, 0}, 0},
        {{0x8000'0000, 4, 31, 0}, 0x0800'0000},
    };
    std::vector<In> u_in;
    for (const auto& c : unsigned_cases) u_in.push_back(c.first);
    run_cases(v_unsigned, u_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = unsigned_cases[i].second;
        ENSURE(got == want, "{}: v_bfe_u32({:#010x}, {}, {}) = {:#010x}, want {:#010x}", backend, inp[0], inp[1], inp[2],
               got, want);
    });
}

// ---- rs:1372 ------------------------------------------------------------------------------------
void f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    HostShader mul2 = build_valu_cs(vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    HostShader mul4 = build_valu_cs(vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    HostShader div2 = build_valu_cs(vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    HostShader div2_clamp = build_valu_cs(vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000,
    // -0.5=0xB800
    run_cases(mul2, {In{0x3400, 0x3800, 0, 0}},  // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
              [&](const char* backend, std::size_t, In, u32 got) {
                  ENSURE(u16(got) == 0x3E00, "{}: v_add_f16 mul:2", backend);
              });
    run_cases(mul4, {In{0x3400, 0x3400, 0, 0}},  // (0.25 + 0.25) * 4 = 2.0 (0x4000)
              [&](const char* backend, std::size_t, In, u32 got) {
                  ENSURE(u16(got) == 0x4000, "{}: v_add_f16 mul:4", backend);
              });
    run_cases(div2, {In{0x3E00, 0x3E00, 0, 0}},  // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
              [&](const char* backend, std::size_t, In, u32 got) {
                  ENSURE(u16(got) == 0x3E00, "{}: v_add_f16 div:2", backend);
              });
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00), whereas clamp
    // before div:2 would give 0.5 (0x3800).
    const std::vector<std::tuple<u16, u16, u16>> clamp_cases{
        {0x3A00, 0x3A00, 0x3A00},  // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        {0x3E00, 0x3E00, 0x3C00},  // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        {0xB800, 0xB800, 0x0000},  // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    };
    std::vector<In> inputs;
    for (const auto& [a, b, _] : clamp_cases) inputs.push_back(In{u32(a), u32(b), 0, 0});
    run_cases(div2_clamp, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
        u16 want = std::get<2>(clamp_cases[i]);
        ENSURE(u16(got) == want, "{}: v_add_f16 div:2 clamp case {}: got {:#06x}, want {:#06x}", backend, i, u16(got),
               want);
    });
}

// ---- rs:1423 ------------------------------------------------------------------------------------
void cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    HostShader cvt_u16 = build_valu_cs(vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    HostShader cvt_u16_abs = build_valu_cs(vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    HostShader cvt_i16 = build_valu_cs(vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    HostShader cvt_i16_neg = build_valu_cs(vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    using Cases = std::vector<std::pair<u16, u16>>;
    const Cases u16_cases{
        {0xBC00, 0},       // -1.0 -> 0
        {0xFC00, 0},       // -inf -> 0
        {0x7E00, 0},       // NaN -> 0
        {0x57BC, 123},     // 123.75 -> 123
        {0x7BFF, 65504},   // 65504.0 (max finite f16) -> 65504
        {0x7C00, 0xFFFF},  // +inf -> 65535 (saturated)
    };
    std::vector<In> u16_in;
    for (const auto& [x, _] : u16_cases) u16_in.push_back(In{u32(x), 0, 0, 0});
    run_cases(cvt_u16, u16_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u16 want = u16_cases[i].second;
        ENSURE(u16(got) == want, "{}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {:#06x}", backend, inp[0], u16(got), want);
    });
    run_cases(cvt_u16_abs, {In{0xD7BC, 0, 0, 0}},  // |-123.75| -> 123
              [&](const char* backend, std::size_t, In, u32 got) {
                  ENSURE(u16(got) == 123, "{}: v_cvt_u16_f16 |src0|", backend);
              });

    const Cases i16_cases{
        {0xFC00, 0x8000},          // -inf -> -32768
        {0xF8E2, 0x8000},          // -40000.0 -> -32768 (saturated, not wrapped)
        {0xD7BC, u16(-123)},       // -123.75 -> -123
        {0x7E00, 0},               // NaN -> 0
        {0x57BC, 123},             // 123.75 -> 123
        {0x78E2, 0x7FFF},          // +40000.0 -> 32767 (saturated, not wrapped)
        {0x7C00, 0x7FFF},          // +inf -> 32767
    };
    std::vector<In> i16_in;
    for (const auto& [x, _] : i16_cases) i16_in.push_back(In{u32(x), 0, 0, 0});
    run_cases(cvt_i16, i16_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u16 want = i16_cases[i].second;
        ENSURE(u16(got) == want, "{}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {:#06x}", backend, inp[0], u16(got), want);
    });
    run_cases(cvt_i16_neg, {In{0x57BC, 0, 0, 0}, In{0x78E2, 0, 0, 0}},  // -(123.75) -> -123; -(40000.0) -> -32768
              [&](const char* backend, std::size_t i, In, u32 got) {
                  const std::array<u16, 2> wants{u16(-123), 0x8000};
                  u16 want = wants[i];
                  ENSURE(u16(got) == want, "{}: v_cvt_i16_f16 -src0 case {}", backend, i);
              });
}

constexpr u32 S_BARRIER = 0xBF8A'0000;

// VOP2 op vdst, src0, vsrc1.
constexpr u32 vop2(u8 op, u8 vdst, u16 src0, u8 vsrc1) {
    return (u32(op) & 0x3F) << 25 | u32(vdst) << 17 | u32(vsrc1) << 9 | (u32(src0) & 0x1FF);
}

// DS op vdst, addr, data0 at offset 0.
constexpr std::array<u32, 2> ds(u8 op, u8 vdst, u8 addr, u8 data0) {
    return {0x36u << 26 | u32(op) << 18, u32(vdst) << 24 | u32(data0) << 8 | u32(addr)};
}

constexpr u8 DS_WRITE_B32 = 0x0D;
constexpr u8 DS_MIN_F32 = 0x12;
constexpr u8 DS_MAX_F32 = 0x13;
constexpr u8 DS_MIN_RTN_F32 = 0x32;
constexpr u8 DS_MAX_RTN_F32 = 0x33;
constexpr u8 DS_READ_B32 = 0x36;

// What RDNA2 writes for ds_max_f32 / ds_min_f32 over a dword holding `old` with `data` (bits), as
// measured on gfx1036 with f32 denormals kept: an sNaN, data first, then memory, is written
// quieted; otherwise a NaN loses to a number (two quiet NaNs give the data); numbers compare as
// floats with -0 below +0.
u32 lds_min_max_ref(u32 old, u32 data, bool max) {
    auto nan = [](u32 x) { return (x & 0x7FFF'FFFF) > 0x7F80'0000; };
    auto snan = [&](u32 x) { return nan(x) && (x & 0x40'0000) == 0; };
    auto key = [](u32 x) {
        i32 b = i32(x);
        return b < 0 ? (b ^ 0x7FFF'FFFF) : b;
    };
    if (snan(data)) return data | 0x40'0000;
    if (snan(old)) return old | 0x40'0000;
    bool nd = nan(data), no = nan(old);
    if (no) return data;   // (true, true) | (false, true)
    if (nd) return old;    // (true, false)
    bool take = max ? key(data) > key(old) : key(data) < key(old);
    return take ? data : old;
}

// ---- rs:1538 ------------------------------------------------------------------------------------
void ds_min_max_f32_order_floats_and_return_the_old_value() {
    const std::array<float, 14> values{
        0.0f,
        -0.0f,
        1.0f,
        -1.0f,
        2.5f,
        -2.5f,
        1e-30f,
        -1e-30f,
        b2f(1),  // denormal
        b2f(0x8000'0001),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        FLT_MAX,
        -FLT_MAX,
    };
    // Quiet and signalling NaNs of both signs, in memory and as data.
    const std::array<u32, 5> nans{0x7FC0'0000u, 0xFFC0'0000, 0x7F80'0001, 0xFFA0'0000, 0x7FBF'FFFF};
    std::vector<u32> bits;
    for (float v : values) bits.push_back(f2b(v));
    for (u32 n : nans) bits.push_back(n);
    std::vector<In> cases;
    for (u32 old : bits)
        for (u32 data : bits) cases.push_back(In{old, data, 0, 0});
    const std::array<std::tuple<u8, u8, bool, const char*>, 2> forms{{
        {DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"},
        {DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"},
    }};
    for (const auto& [rtn, plain, max, what] : forms) {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value (rtn) or the dword afterwards
        // (read back).
        Words returns_old{vop1(0x001, 5, 0x80)};
        append(returns_old, ds(DS_WRITE_B32, 0, 5, 1));
        append(returns_old, ds(rtn, 0, 5, 2));
        Words returns_new{vop1(0x001, 5, 0x80)};
        append(returns_new, ds(DS_WRITE_B32, 0, 5, 1));
        append(returns_new, ds(plain, 0, 5, 2));
        append(returns_new, ds(DS_READ_B32, 0, 5, 0));
        const std::array<std::pair<const Words*, bool>, 2> bodies{{{&returns_old, true}, {&returns_new, false}}};
        for (const auto& [body, old_result] : bodies) {
            // One thread with LDS declares WaveSize(32), which WARP's pipeline creation refuses:
            // hardware only (the 64-thread test below runs on WARP too).
            HostShader shader = build_cs(*body, 1, 512);
            run_cases_on(shader, cases, HARDWARE_ONLY, [&](const char* backend, std::size_t, In inp, u32 got) {
                float old = b2f(inp[0]), data = b2f(inp[1]);
                u32 want = old_result ? inp[0] : lds_min_max_ref(inp[0], inp[1], max);
                ENSURE(got == want, "{}: {}_f32 on {:e} [{:#010x}] with {:e} [{:#010x}] gave {:#010x}, want {:#010x} ({})",
                       backend, what, old, inp[0], data, inp[1], got, want, old_result ? "old value" : "dword after");
            });
        }
    }
}

// ---- rs:1618 ------------------------------------------------------------------------------------
void ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then every thread applies
    // (thread - 32) as a float; v0 = LDS[0].
    const float INF = std::numeric_limits<float>::infinity();
    const std::array<std::tuple<u8, bool, float, float>, 4> forms{{
        {DS_MAX_F32, true, -INF, 31.0f},
        {DS_MAX_F32, true, 100.0f, 100.0f},
        {DS_MIN_F32, false, INF, -32.0f},
        {DS_MIN_F32, false, -0.0f, -32.0f},
    }};
    for (const auto& [op, max, init, want] : forms) {
        Words body{vop1(0x001, 5, 0x80)};
        append(body, ds(DS_WRITE_B32, 0, 5, 1));
        body.push_back(S_BARRIER);
        body.push_back(vop2(0x26, 6, vgpr(0), 3));  // v_sub_nc_u32 v6, v0, v3
        body.push_back(vop1(0x005, 6, vgpr(6)));    // v_cvt_f32_i32
        append(body, ds(op, 0, 5, 6));
        body.push_back(S_BARRIER);
        append(body, ds(DS_READ_B32, 0, 5, 0));
        HostShader shader = build_cs(body, 64, 512);
        run_cases(shader, {In{f2b(init), 0, 32, 0}}, [&](const char* backend, std::size_t, In, u32 got) {
            ENSURE(b2f(got) == want, "{}: {} from {} over 64 threads: {} != {}", backend, max ? "max" : "min", init,
                   b2f(got), want);
        });
    }
}

// ---- rs:1652 ------------------------------------------------------------------------------------
void v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    HostShader cmp_lo = build_valu_cs({
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    });
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    auto [w0, w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1u << 11;  // opsel[0] = 1 (high half)
    w1 |= 1u << 29;  // neg[0] = 1
    HostShader cmp_hi_neg = build_valu_cs({
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    });

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    const std::vector<std::pair<u32, u32>> classes{
        {0x7C01, 0},  // sNaN (+sNaN, quiet bit 0x0200 clear)
        {0xFDFF, 0},  // -sNaN (sign ignored for NaNs)
        {0x7E00, 1},  // +qNaN
        {0xFE01, 1},  // -qNaN
        {0xFC00, 2},  // -inf
        {0xBC00, 3},  // -normal (-1.0)
        {0x8001, 4},  // -denormal
        {0x8000, 5},  // -0.0
        {0x0000, 6},  // +0.0
        {0x0001, 7},  // +denormal
        {0x3C00, 8},  // +normal (+1.0)
        {0x7C00, 9},  // +inf
    };
    std::vector<std::pair<In, u32>> cases;
    for (const auto& [bits, cls] : classes) {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push_back({In{bits, 1u << cls, 0, 0}, 1u});
        cases.push_back({In{bits, 0x3FFu & ~(1u << cls), 0, 0}, 0u});
    }
    std::vector<In> inputs;
    for (const auto& c : cases) inputs.push_back(c.first);
    run_cases(cmp_lo, inputs, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = cases[i].second;
        ENSURE(got == want, "{}: v_cmp_class_f16({:#06x}, {:#05x}) = {}, want {}", backend, inp[0], inp[1], got, want);
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3); high half 0x7C01
    // (+sNaN) negated -> still sNaN (class 0, not quieted).
    const std::vector<std::pair<In, u32>> hi_cases{
        {{0x3C00'0000, 1u << 3, 0, 0}, 1},
        {{0x3C00'0000, 1u << 8, 0, 0}, 0},
        {{0x7C01'FFFF, 1u << 0, 0, 0}, 1},
        {{0x7C01'FFFF, 1u << 1, 0, 0}, 0},
    };
    std::vector<In> hi_in;
    for (const auto& c : hi_cases) hi_in.push_back(c.first);
    run_cases(cmp_hi_neg, hi_in, [&](const char* backend, std::size_t i, In inp, u32 got) {
        u32 want = hi_cases[i].second;
        ENSURE(got == want, "{}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {}, want {}", backend, inp[0], inp[1], got,
               want);
    });
}

// ---- rs:1722 ------------------------------------------------------------------------------------
void v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    HostShader plain = build_valu_cs({vop1(0x00f, 0, vgpr(1))});
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    auto [w0, w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1u << 8;
    w1 |= 1u << 29;
    HostShader neg_abs = build_valu_cs({w0, w1});

    const double INF = std::numeric_limits<double>::infinity();
    const std::vector<double> values{
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        b2d(0x3FF0'0000'1000'0000),  // 1.0 + 0.5 ulp(f32) -> 1.0
        b2d(0x3FF0'0000'3000'0000),  // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        b2d(0x3FF0'0000'1000'0001),  // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        double(FLT_MAX),
        double(FLT_MAX) * 1.000'001,
        -double(FLT_MAX) * 1.000'001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        double(FLT_MIN),
        double(FLT_MIN) * 0.5,
        double(FLT_MIN) * 0.75,
        double(b2f(0x007F'FFFF)),
        b2d(0x380F'FFFF'F000'0000),  // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        b2d(0x3690'0000'0000'0000),
        b2d(0x3690'0000'0000'0001),
        b2d(0xB690'0000'0000'0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        INF,
        -INF,
        b2d(0x7FF8'0000'0000'0000),  // +qNaN
        b2d(0x7FF0'0000'0000'0001),  // +sNaN -> quieted
    };
    std::vector<In> inputs = f64_inputs(values);
    run_cases(plain, inputs, [&](const char* backend, std::size_t i, In, u32 got) {
        double x = values[i];
        // Rust `x as f32`: round to nearest even, NaN stays a NaN (its quiet bit is not checked
        // here, only on the result).
        u32 want = f2b(static_cast<float>(x));
        if (std::isnan(x)) {
            ENSURE(std::isnan(b2f(got)) && (got & 0x0040'0000) != 0, "{}: v_cvt_f32_f64({:#018x}) = {:#010x}, expected qNaN",
                   backend, d2b(x), got);
        } else {
            ENSURE(got == want, "{}: v_cvt_f32_f64({} / {:#018x}) = {:#010x}, want {:#010x}", backend, x, d2b(x), got,
                   want);
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    const std::vector<double> mod_values{2.5, -2.5, 0.0, INF};
    std::vector<In> mod_in = f64_inputs(mod_values);
    run_cases(neg_abs, mod_in, [&](const char* backend, std::size_t i, In, u32 got) {
        double x = mod_values[i];
        u32 want = f2b(static_cast<float>(-std::fabs(x)));
        ENSURE(got == want, "{}: v_cvt_f32_f64(-|{}|) = {:#010x}, want {:#010x}", backend, x, got, want);
    });
}

// ---- rs:1811 ------------------------------------------------------------------------------------
void whole_wave_f64_sum_and_cvt_f32_f64() {
    auto s_or_saveexec_b32 = [](u8 sdst, u8 src0) { return 0xBE80'0000u | u32(sdst) << 16 | 0x3Du << 8 | u32(src0); };
    auto v_mov_dpp8 = [](u8 vdst, u8 vsrc, u32 sel) {
        return std::array<u32, 2>{vop1(0x001, vdst, 0xE9), u32(vsrc) | (sel << 8)};
    };
    auto v_mov_row_mirror = [](u8 vdst, u8 vsrc) {
        return std::array<u32, 2>{vop1(0x001, vdst, 0xFA),
                                  u32(vsrc) | (0x140u << 8) | (1u << 19) | (0xFu << 24) | (0xFu << 28)};
    };

    Words words;
    words.push_back(s_mov_b32(5, 0x80));
    append(words, s_load_dwordx4(8, 4));
    // Per-lane f64 input in v[6:7] = (s0, s1).
    words.push_back(vop1(0x001, 6, sgpr(0)));
    words.push_back(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push_back(s_or_saveexec_b32(12, 0xC1));
    append(words, vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    append(words, vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
    for (u32 sel : {14570689u, 9658386u, 2340u}) {
        append(words, v_mov_dpp8(4, 2, sel));
        append(words, v_mov_dpp8(5, 3, sel));
        append(words, vop3(0x164, 2, vgpr(4), vgpr(2), 0));  // v_add_f64
    }
    append(words, v_mov_row_mirror(4, 2));
    append(words, v_mov_row_mirror(5, 3));
    append(words, vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    append(words, vop3(0x378, 4, vgpr(2), 0x80, 0x80));  // v_permlanex16_b32
    append(words, vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    append(words, vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    append(words, vop3(0x360, 14, vgpr(2), 0x80, 0));  // v_readlane_b32 s14, v2, 0
    append(words, vop3(0x360, 15, vgpr(3), 0x80, 0));  // v_readlane_b32 s15, v3, 0
    words.push_back(s_mov_b32(16, 14));
    words.push_back(s_mov_b32(17, 15));
    words.push_back(s_mov_b32(126, 12));       // restore exec_lo
    words.push_back(vop1(0x00f, 0, sgpr(16)));  // v_cvt_f32_f64 v0, s[16:17]
    append(words, buffer_store_dword(0, 8, 0));
    words.push_back(S_ENDPGM);

    NativeProgram native = synthetic_cs(le_bytes(words), 12, 5u << 1, {4, 1, 1});
    HostShader shader = lower_cs(native, false);
    validate(shader, false);

    const std::array<std::pair<double, float>, 2> cases{{{1.25, 5.0f}, {-0.75, -3.0f}}};
    std::lock_guard one(ONE_AT_A_TIME);
    for (const auto& [label, warp] : HARDWARE_AND_WARP) {
        std::optional<vs::host::Host> host;
        try {
            host.emplace(warp);
        } catch (const std::exception&) {
            continue;
        }
        Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
        Microsoft::WRL::ComPtr<ID3D12PipelineState> pso;
        try {
            auto [r, p] = host->compute_pipeline(shader);
            root = r;
            pso = p;
        } catch (const std::exception& e) {
            // WARP does not support fixed WaveSize(32) on SM 6.6.
            if (warp) {
                std::fprintf(stderr, "%s: compute_pipeline skipped (%s)\n", label, e.what());
                continue;
            }
            throw std::runtime_error(std::format("{}: compute_pipeline failed: {}", label, e.what()));
        }
        for (const auto& [x, want] : cases) {
            u64 b = d2b(x);
            In inputs{u32(b), u32(b >> 32), 0, 0};
            u32 got = dispatch_one(*host, root.Get(), pso.Get(), user_data_for(shader, inputs), label);
            ENSURE(got == f2b(want), "{}: wave f64 sum({}) = {:#010x}, want {:#010x}", label, x, got, f2b(want));
        }
    }
}

// Whether the hardware adapter is an AMD RDNA card's: what the translator writes for one
// (emit::rdna) holds there.
bool rdna_card() {
    std::optional<vs::host::Host> host;
    try {
        host.emplace(false);
    } catch (const std::exception&) {
        return false;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
    bool amd = false;
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (SUCCEEDED(factory->EnumAdapterByLuid(host->device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)))) {
            DXGI_ADAPTER_DESC1 desc{};
            amd = SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002;
        }
    }
    return amd && SUCCEEDED(host->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof(waves))) &&
           waves.WaveLaneCountMin == 32 && waves.WaveLaneCountMax == 64;
}

// ---- rs:2002 ------------------------------------------------------------------------------------
// The forms written for the console's own kind of hardware give, on such a card, what the ones
// spelled out for every host give; a legacy product's zero may have the other sign.
void rdna_forms_give_what_the_spelled_out_ones_give() {
    if (!rdna_card()) {
        std::fprintf(stderr, "no RDNA card; skipped\n");
        return;
    }
    constexpr u16 ONE = 242;
    constexpr u16 MINUS_ONE = 243;
    const u16 a = vgpr(1), b = vgpr(2), c = vgpr(3);
    auto saturated = vop3_mod(0x110, 5, vgpr(1), vgpr(1), 0, 0, 0, 0, true);
    auto two = [](std::array<u32, 2> x, std::array<u32, 2> y) { return Words{x[0], x[1], y[0], y[1]}; };
    auto vec = [](std::array<u32, 2> x) { return Words{x[0], x[1]}; };
    const std::vector<std::tuple<const char*, Words, bool>> bodies{
        {"max", vec(vop3(0x110, 0, a, b, 0)), true},
        {"min", vec(vop3(0x10F, 0, a, b, 0)), true},
        {"max3", vec(vop3(0x154, 0, a, b, c)), true},
        {"min3", vec(vop3(0x151, 0, a, b, c)), true},
        {"med3 0 1", vec(vop3(0x157, 0, a, 128, ONE)), true},
        {"med3 1 0", vec(vop3(0x157, 0, ONE, a, 128)), true},
        {"med3 1 -1", vec(vop3(0x157, 0, ONE, MINUS_ONE, a)), true},
        {"fract", Words{vop1(0x020, 0, a)}, true},
        {"mul_legacy", vec(vop3(0x107, 0, a, b, 0)), false},
        {"mul_legacy by itself", vec(vop3(0x107, 0, a, a, 0)), false},
        {"mul_legacy by a constant", vec(vop3(0x107, 0, a, MINUS_ONE, 0)), false},
        {"mul_legacy of a saturated", two(saturated, vop3(0x107, 0, vgpr(5), b, 0)), false},
        {"mad_legacy", vec(vop3(0x140, 0, a, b, c)), false},
        {"max of a mul_legacy", two(vop3(0x107, 5, a, b, 0), vop3(0x110, 0, vgpr(5), c, 0)), false},
    };
    const std::array<float, 10> numbers{0.0f,
                                        -0.0f,
                                        1.0f,
                                        -1.0f,
                                        0.3f,
                                        -2.5e-7f,
                                        3e38f,
                                        std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::quiet_NaN()};
    std::vector<In> cases;
    for (std::size_t i = 0; i < numbers.size(); ++i) {
        for (std::size_t j = 0; j < numbers.size(); ++j) {
            float z = numbers[(3 * i + j) % numbers.size()];
            cases.push_back(In{f2b(numbers[i]), f2b(numbers[j]), f2b(z), 0});
        }
    }
    for (const auto& [name, body, exact] : bodies) {
        std::vector<std::vector<u32>> got;
        std::vector<std::vector<u8>> written;
        for (u32 bits : {0u, vs::rdna::ALL}) {
            vs::lower::test::set_rdna(bits);
            HostShader shader = build_valu_cs(body);
            vs::lower::test::set_rdna(std::nullopt);
            written.push_back(shader.dxil);
            std::vector<u32> all;
            run_cases_on(shader, cases, HARDWARE_ONLY, [&](const char*, std::size_t, In, u32 r) { all.push_back(r); });
            got.push_back(std::move(all));
        }
        ENSURE(written[0] != written[1], "{}: written alike", name);
        for (std::size_t k = 0; k < cases.size() && k < got[0].size() && k < got[1].size(); ++k) {
            u32 was = got[0][k], is = got[1][k];
            float x = b2f(was), y = b2f(is);
            bool same = was == is || (std::isnan(x) && std::isnan(y)) || (!exact && x == 0.0f && y == 0.0f);
            const In& cs = cases[k];
            ENSURE(same, "{} of [{}, {}, {}, {}]: {} ({:#x}), now {} ({:#x})", name, b2f(cs[0]), b2f(cs[1]), b2f(cs[2]),
                   b2f(cs[3]), x, was, y, is);
        }
    }
}

// Runs one test as the Rust test thread `emit::valu_tests::<name>` would: VS_DUMP names its modules
// after it, and a panic (here: an exception) fails it.
void run(const char* name, void (*test)()) {
    x2w::vs::lower::test::set_dump_name(std::string("emit__valu_tests__") + name);
    try {
        test();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: panicked: %s\n", name, e.what());
        ++test_util::failures();
    }
}

}  // namespace

#define RUN(name) run(#name, name)

int main(int argc, char** argv) {
    bool ignored = argc > 1 && std::string_view(argv[1]) == "--ignored";
    if (ignored) {
        RUN(v_cvt_f32_f64_rounds_to_nearest_even);
        return test_util::finish("lower_valu_test --ignored");
    }
    RUN(scalar_quad_masks_pack_nibbles_and_set_scc);
    RUN(a_branch_to_fallthrough_keeps_both_values_at_a_join);
    RUN(tbuffer_d16_loads_pack_two_channels_per_dword);
    RUN(normalized_buffer_formats_load_their_ends_exactly);
    RUN(normalized_buffer_formats_load_correctly_rounded_quotients);
    RUN(normalized_codes_truncate_back_to_themselves);
    RUN(packed_add_selects_halves_of_inline_constants);
    RUN(v_permlane16_and_x16_read_the_lanes_their_selectors_pick);
    RUN(v_med3_f32_matches_rdna2_with_nan);
    RUN(max_and_min_order_negative_zero_below_positive_zero);
    RUN(f16_max_and_min_order_negative_zero_below_positive_zero);
    RUN(v_min3_max3_med3_integer_forms_match_rdna2);
    RUN(v_med3_f16_matches_rdna2_with_nan);
    RUN(v_cmp_lg_is_false_for_nan);
    RUN(v_cvt_off_f32_i4_uses_the_signed_low_nibble);
    RUN(v_fract_f32_never_returns_one);
    RUN(v_fract_f16_never_returns_one);
    RUN(v_cvt_norm_i16_and_u16_f16_round_the_exact_product);
    RUN(v_sin_and_v_cos_f32_reduce_large_turns);
    RUN(v_cvt_u16_and_i16_f16_saturate);
    RUN(v_cvt_f16_u16_and_i16_widen_the_integer_first);
    RUN(v_sin_and_v_cos_f16_reduce_large_turns);
    RUN(signed_64_bit_operands_sign_extend_their_literal);
    RUN(bfe_signed_and_unsigned_match_rdna2);
    RUN(f16_omod_applies_before_clamp);
    RUN(cvt_16bit_int_from_f16_saturates_and_honors_modifiers);
    RUN(ds_min_max_f32_order_floats_and_return_the_old_value);
    RUN(ds_min_max_f32_are_atomic_across_waves);
    RUN(v_cmp_class_f16_matches_rdna2);
    RUN(v_cvt_f32_f64_matches_ieee754);
    RUN(whole_wave_f64_sum_and_cvt_f32_f64);
    RUN(rdna_forms_give_what_the_spelled_out_ones_give);
    return test_util::finish("lower_valu_test");
}
