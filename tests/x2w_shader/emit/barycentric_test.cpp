// Port of the tests in the upstream Rust module emit/mod.rs `mod barycentric_tests` (rs:2491), in Rust order.
//
// Each Rust #[test] is a function; main runs them all, each under its own dump name
// ("emit__barycentric_tests__<name>"). A test that throws (a Rust panic or `.unwrap()` of an error)
// fails and the others still run. first_half_values_render_on_hardware_and_warp (Rust
// #[cfg(windows)]) skips, and says so, without an SM 6.6 device with barycentrics. The upstream
// valu_tests::ONE_AT_A_TIME lock is not needed: the tests run one after another on one thread.

#include <windows.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "test_util.h"
#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/host.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/signature.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"

namespace vs = x2w::vs;
using vs::HostShader;
using vs::ShaderError;
using vs::container::SigElement;
using vs::dxbc::Bytes;
using vs::dxbc::Container;
using vs::dxbc::FourCC;
using vs::native::HwStage;
using vs::native::NativeProgram;
using vs::native::PsInput;
using vs::native::PsIo;
using vs::native::Usage;
using vs::validator::Validator;

#define CHECK_MSG(expr, msg)                                                               \
    do {                                                                                   \
        const bool check_ok_ = static_cast<bool>(expr);                                    \
        if (!check_ok_) std::fprintf(stderr, "  %s\n", std::string(msg).c_str());          \
        ::test_util::check(check_ok_, #expr, __FILE__, __LINE__);                          \
    } while (0)

namespace {

std::vector<std::uint8_t> le_bytes(std::span<const std::uint32_t> words) {
    std::vector<std::uint8_t> code;
    code.reserve(words.size() * 4);
    for (std::uint32_t w : words)
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    return code;
}

// Rust `.unwrap()` of a Result<_, ShaderError>: the error is a panic.
template <class F>
auto unwrap(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const ShaderError& e) {
        throw std::logic_error("called `Result::unwrap()` on an `Err` value: " + e.to_string());
    }
}

Bytes part_data(const Container& c, FourCC fourcc) {
    const vs::dxbc::Part* p = c.part(fourcc);
    if (!p) throw std::logic_error("no " + fourcc.to_string() + " part");
    return p->data;
}

// `haystack.windows(needle.len()).any(|w| w == needle)` / `.position(..)`.
std::optional<std::size_t> find_window(Bytes haystack, std::span<const std::uint8_t> needle) {
    if (needle.size() > haystack.size()) return std::nullopt;
    for (std::size_t at = 0; at + needle.size() <= haystack.size(); ++at)
        if (std::equal(needle.begin(), needle.end(), haystack.begin() + at)) return at;
    return std::nullopt;
}

std::string describe(const std::vector<SigElement>& elements) {
    std::string s = "[";
    for (const SigElement& e : elements) {
        s += "{" + e.name + " index " + std::to_string(e.index) + " kind " + std::to_string(e.kind) + " row " +
             (e.row ? std::to_string(*e.row) : std::string("None")) + " col " + std::to_string(e.col) +
             " cols " + std::to_string(e.cols) + "} ";
    }
    return s + "]";
}

// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
constexpr std::uint32_t PERSP_CENTER = 1u << 1;
constexpr std::uint32_t LINEAR_CENTER = 1u << 5;

// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
std::array<std::uint32_t, 2> exp_mrt0(std::uint32_t r) { return {0xF800'180Fu, r * 0x0101'0101u}; }

// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
std::uint32_t v_add_f32(std::uint32_t dst, std::uint32_t a, std::uint32_t b) {
    return 3u << 25 | dst << 17 | b << 9 | (256 + a);
}

SigElement element(const char* name, std::uint8_t kind, std::uint32_t row) {
    SigElement e;
    e.name = name;
    e.index = 0;
    e.kind = kind;
    e.comp = 9;
    e.interp = 0;
    e.row = row;
    e.col = 0;
    e.cols = 4;
    e.usage = 0;
    e.half = false;
    return e;
}

// Translates a pixel program with these input signature elements and one float4 TEXCOORD0
// attribute (attr 0, row 0) in the stage data. Throws ShaderError (Rust `?`).
HostShader pixel_with_inputs(std::span<const std::uint32_t> words, std::uint32_t input_addr,
                             std::span<const SigElement> inputs) {
    NativeProgram native;
    native.stage = HwStage::Ps;
    native.wave64 = false;
    native.vgprs = 8;
    PsInput in;
    in.usage = Usage::Generic();
    in.row = 0;
    in.mask = 0b1111;
    in.attr = 0;
    in.flat = false;
    in.fp16 = false;
    in.pack2 = false;
    in.row2 = 0;
    PsIo io_data;
    io_data.inputs = {in};
    io_data.input_addr = input_addr;
    io_data.db_shader_control = 0;
    native.io = io_data;
    native.code = le_bytes(words);
    SigElement target = element("SV_Target", 16, 0);
    target.usage = 15;
    const std::vector<std::uint8_t> isg1 = vs::container::signature(inputs, true);
    const std::vector<SigElement> outs = {target};
    const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::ISG1, isg1}, {FourCC::OSG1, osg1}});
    const Container c = unwrap([&] { return Container::parse(bytes); });
    const std::vector<std::uint8_t> code = native.patched_code();
    const vs::ir::Program p = vs::ir::build(code, native.wave64, native.vgprs);
    const vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, nullptr);
    const vs::io::Io io = vs::io::raise(p, native);
    const vs::structure::Structured st = vs::structure::structure(p);
    return vs::emit(c, native, p, st, io, res);
}

// Translates a pixel program with the input VGPRs `input_addr` enables and one float4 TEXCOORD0
// attribute (attr 0, row 0). Throws ShaderError.
HostShader pixel(std::span<const std::uint32_t> words, std::uint32_t input_addr) {
    const std::array<SigElement, 1> inputs = {element("TEXCOORD", 0, 0)};
    return pixel_with_inputs(words, input_addr, inputs);
}

// The translation's `SFI0` features and `ISG1` rows.
std::pair<std::uint64_t, std::vector<SigElement>> parts(const HostShader& shader) {
    const Container c = unwrap([&] { return Container::parse(shader.dxil); });
    const Bytes sfi0 = part_data(c, FourCC::SFI0);
    if (sfi0.size() != 8) throw std::logic_error("SFI0 is not 8 bytes");
    std::uint64_t features = 0;
    for (std::size_t b = 0; b < 8; ++b) features |= std::uint64_t(sfi0[b]) << (8 * b);
    std::vector<SigElement> inputs = unwrap([&] { return vs::signature(part_data(c, FourCC::ISG1)); });
    return {features, std::move(inputs)};
}

void validate(const HostShader& shader) {
    std::optional<Validator> v = Validator::load(Validator::default_path());
    if (!v) return;
    std::vector<std::uint8_t> copy = shader.dxil;
    if (std::optional<std::string> e = v->validate(copy)) CHECK_MSG(!e, "validator: " + *e);
}

// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
std::uint8_t texcoord_mode(const HostShader& shader) {
    const Container c = unwrap([&] { return Container::parse(shader.dxil); });
    const Bytes psv = part_data(c, FourCC::PSV0);
    const std::array<std::uint8_t, 5> row = {1, 0, 0x44, 0, 3};
    const std::optional<std::size_t> at = find_window(psv, row);
    if (!at) throw std::logic_error("called `Option::unwrap()` on a `None` value: no TEXCOORD0 row in PSV0");
    if (*at + 5 >= psv.size()) throw std::logic_error("PSV0 ends in the TEXCOORD0 row");
    return psv[*at + 5];
}

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

std::size_t count_barycentrics(const std::vector<SigElement>& inputs) {
    return static_cast<std::size_t>(
        std::count_if(inputs.begin(), inputs.end(), [](const SigElement& e) { return e.kind == 28; }));
}

// ---- rs:2569 ------------------------------------------------------------------------------------
void an_unused_packed_input_matches_its_provoking_vertex_sibling() {
    const auto [e0, e1] = exp_mrt0(3);
    // v_interp_mov_f32 v3, p0, attr0.z followed by exp mrt0.
    const std::array<std::uint32_t, 4> words = {0xC80E'0202u, e0, e1, 0xBF81'0000};
    SigElement xy = element("TEXCOORD", 0, 0);
    xy.cols = 2;
    SigElement zw = element("NORMAL", 0, 0);
    zw.col = 2;
    zw.cols = 2;
    const std::array<SigElement, 2> inputs = {xy, zw};
    const HostShader shader = unwrap([&] { return pixel_with_inputs(words, PERSP_CENTER, inputs); });
    const Container c = unwrap([&] { return Container::parse(shader.dxil); });
    const Bytes psv = part_data(c, FourCC::PSV0);
    for (std::uint8_t columns : {std::uint8_t{0x42}, std::uint8_t{0x62}}) {
        const std::array<std::uint8_t, 6> record = {1, 0, columns, 0, 3, 1};
        CHECK(find_window(psv, record).has_value());
    }
    const Bytes in = part_data(c, FourCC::ISG1);
    if (in.size() <= 8 + 32 + 25) throw std::logic_error("ISG1 shorter than two rows");
    CHECK(in[8 + 25] == 0);
    CHECK(in[8 + 32 + 25] == 4);
    validate(shader);
}

// ---- rs:2588 ------------------------------------------------------------------------------------
// A pixel program's first-lane loop (no table entry in it, so not a waterfall the tracker knows)
// asks for helper lanes, which is valid DXIL (upstream #946's program).
void a_pixel_first_lane_loop_serves_helper_lanes() {
    const std::array<std::uint32_t, 16> words = {
        0xD765'0000u, 0x0001'00C1,  // v_mbcnt_lo_u32_b32 v0, -1, 0
        0x7E02'0280,                // v_mov_b32 v1, 0
        0xBE97'037E,                // s_mov_b32 s23, exec_lo
        0x7E18'0500,                // v_readfirstlane_b32 s12, v0
        0x7D84'000C,                // v_cmp_eq_u32 vcc, s12, v0
        0xBE95'3C6A,                // s_and_saveexec_b32 s21, vcc_lo
        0xBF80'0000,                // s_nop 0
        0x7E02'02FF, 0x3F80'0000,   // v_mov_b32 v1, 1.0
        0x897E'157E,                // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89'FFF8,                // s_cbranch_execnz loop
        0xBEFE'0317,                // s_mov_b32 exec_lo, s23
        0xF800'180F, 0x0101'0101,   // exp mrt0 v1 v1 v1 v1 done vm
        0xBF81'0000,                // s_endpgm
    };
    const HostShader shader = unwrap([&] { return pixel(words, 0); });
    validate(shader);
}

// ---- rs:2634 ------------------------------------------------------------------------------------
// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as plain floats: they are channels
// y and z of `SV_Barycentrics`, the weights of vertices 1 and 2.
void barycentrics_read_as_values_are_sv_barycentrics() {
    const auto [e0, e1] = exp_mrt0(2);
    const std::array<std::uint32_t, 4> words = {v_add_f32(2, 0, 1), e0, e1, 0xBF81'0000};
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    CHECK_MSG((features & 0x2'0000) != 0, "no Barycentrics feature");
    std::vector<const SigElement*> bary;
    for (const SigElement& e : inputs)
        if (e.kind == 28) bary.push_back(&e);
    CHECK_MSG(bary.size() == 1, describe(inputs));
    if (bary.empty()) throw std::logic_error("index out of bounds: no SV_Barycentrics row");
    CHECK(bary[0]->index == 0);
    CHECK(bary[0]->col == 0);
    CHECK(bary[0]->cols == 3);
    // Not packed: the validator wants packing location -1.
    CHECK(bary[0]->row == std::nullopt);
    // Only y and z (I and J) are read.
    const Container isg1_c = unwrap([&] { return Container::parse(shader.dxil); });
    const Bytes isg1 = part_data(isg1_c, FourCC::ISG1);
    if (isg1.size() <= 8 + 32 + 25) throw std::logic_error("ISG1 shorter than two rows");
    std::vector<std::uint8_t> rw;
    for (std::size_t k = 0; k < 2; ++k) rw.push_back(isg1[8 + 32 * k + 25]);
    CHECK_MSG(std::find(rw.begin(), rw.end(), std::uint8_t{0b110}) != rw.end(),
              "[" + std::to_string(rw[0]) + ", " + std::to_string(rw[1]) + "]");
    validate(shader);
}

// ---- rs:2657 ------------------------------------------------------------------------------------
// Two kinds of barycentrics read directly: one element per mode.
void two_barycentric_modes_get_two_elements() {
    // v0-v1 perspective i, j; v2-v3 linear i, j.
    const auto [e0, e1] = exp_mrt0(4);
    const std::array<std::uint32_t, 4> words = {v_add_f32(4, 0, 3), e0, e1, 0xBF81'0000};
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER | LINEAR_CENTER); });
    const auto [features, inputs] = parts(shader);
    (void)features;
    std::vector<std::uint32_t> indices;
    for (const SigElement& e : inputs)
        if (e.kind == 28) indices.push_back(e.index);
    std::sort(indices.begin(), indices.end());
    CHECK_MSG((indices == std::vector<std::uint32_t>{0, 1}), describe(inputs));
    validate(shader);
}

// ---- rs:2675 ------------------------------------------------------------------------------------
// I and J only fed to `v_interp_*` stay host interpolation: no barycentrics are declared.
void interpolation_alone_declares_no_barycentrics() {
    const auto [e0, e1] = exp_mrt0(2);
    const std::array<std::uint32_t, 5> words = {
        0xC808'0000u,  // v_interp_p1_f32 v2, v0, attr0.x
        0xC809'0001,   // v_interp_p2_f32 v2, v1, attr0.x
        e0,
        e1,
        0xBF81'0000,
    };
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    CHECK((features & 0x2'0000) == 0);
    CHECK_MSG(count_barycentrics(inputs) == 0, describe(inputs));
    validate(shader);
}

HostShader first_half_pixel(bool paired) {
    const auto [e0, e1] = exp_mrt0(2);
    std::vector<std::uint32_t> words = {0xC808'0000u};  // v_interp_p1_f32 v2, v0, attr0.x
    if (paired) {
        words.push_back(0x7E06'0302);  // v_mov_b32 v3, v2 (keep the partial value)
        words.push_back(0xC809'0001);  // v_interp_p2_f32 v2, v1, attr0.x
        words.push_back(v_add_f32(2, 2, 3));
    }
    words.push_back(e0);
    words.push_back(e1);
    words.push_back(0xBF81'0000);
    return unwrap([&] { return pixel(words, PERSP_CENTER); });
}

// ---- rs:2694 ------------------------------------------------------------------------------------
void a_standalone_first_half_reads_the_vertices() {
    const HostShader shader = first_half_pixel(false);
    const auto [features, inputs] = parts(shader);
    (void)features;
    CHECK(count_barycentrics(inputs) == 1);
    CHECK(texcoord_mode(shader) == 1);
    validate(shader);
}

// ---- rs:2703 ------------------------------------------------------------------------------------
void a_first_half_read_beside_its_pair_keeps_its_partial_value() {
    const HostShader shader = first_half_pixel(true);
    CHECK(texcoord_mode(shader) == 1);
    validate(shader);
}

// ---- rs:2724 ------------------------------------------------------------------------------------
void first_half_values_render_on_hardware_and_warp() {
    using vs::bitcode::Attr;
    using vs::bitcode::Term;
    using vs::host::Draw;
    using vs::host::Host;
    // Pass POSITION and TEXCOORD through, with TEXCOORD in the same output row as the translated
    // pixel shader's input signature.
    vs::shader::Shader s(false);
    s.inputs = {element("POSITION", 0, 0), element("TEXCOORD", 0, 1)};
    for (SigElement& e : s.inputs) e.usage = 15;
    s.outputs = {element("TEXCOORD", 0, 0), element("SV_Position", 3, 1)};
    for (std::size_t i = 0; i < s.outputs.size(); ++i) {
        s.outputs[i].usage = 15;
        s.outputs[i].interp = static_cast<std::uint8_t>(i == 1 ? 4 : 2);
    }
    for (std::uint32_t row = 0; row < 2; ++row) {
        for (std::uint8_t col = 0; col < 4; ++col) {
            // Braced: evaluated left to right, as Rust's array.
            const std::array<vs::bitcode::V, 4> args = {s.m.i32(row), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)};
            const vs::bitcode::V v =
                s.op(s.entry, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, std::span<const vs::bitcode::V>(args));
            const std::array<vs::bitcode::V, 4> store = {s.m.i32(1 - row), s.m.i32(0), s.m.i8(col), v};
            s.op(s.entry, "storeOutput.f32", Attr::NoUnwind, s.t.void_, 5, std::span<const vs::bitcode::V>(store));
        }
    }
    s.m.terminate(s.entry, Term::ret());
    HostShader vertex;
    vertex.stage = HwStage::Vs;
    vertex.dxil = s.finish();
    validate(vertex);
    // One triangle covers an 8x8 target, with attribute values 2, 6, 10.
    const std::array<float, 24> vertices = {
        -1.0f, 1.0f,  0.5f, 1.0f, 2.0f,  0.0f, 0.0f, 0.0f, -1.0f, -3.0f, 0.5f, 1.0f,
        6.0f,  0.0f,  0.0f, 0.0f, 3.0f,  1.0f, 0.5f, 1.0f, 10.0f, 0.0f,  0.0f, 0.0f,
    };
    int ran = 0;
    for (const auto& [backend, warp] : {std::pair<const char*, bool>{"hardware", false}, {"WARP", true}}) {
        std::optional<Host> host;
        try {
            host.emplace(warp);
        } catch (const vs::host::Error&) {
            continue;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS3 options{};
        const HRESULT hr = host->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &options,
                                                             static_cast<UINT>(sizeof options));
        if (FAILED(hr)) throw std::logic_error("CheckFeatureSupport(OPTIONS3): " + vs::host::message(hr));
        if (!options.BarycentricsSupported) {
            std::fprintf(stderr, "no barycentrics on %s; skipped\n", backend);
            continue;
        }
        for (bool paired : {false, true}) {
            const HostShader ps = first_half_pixel(paired);
            validate(ps);
            vs::host::Pipeline pipeline;
            try {
                pipeline = host->pipeline(vertex, ps);
            } catch (const vs::host::Error& e) {
                throw std::logic_error(std::string("interpolation pipeline: ") + e.what());
            }
            Draw d;
            d.vertices = vertices;
            d.stride = 32;
            d.size = {8, 8};
            d.mesh_groups = 0;
            std::vector<std::array<float, 4>> out;
            try {
                out = host->draw(pipeline.first.Get(), pipeline.second.Get(), d);
            } catch (const vs::host::Error& e) {
                throw std::logic_error(std::string("draw partial interpolation: ") + e.what());
            }
            for (std::size_t n = 0; n < out.size(); ++n) {
                const float i = static_cast<float>(n / 8) / 16.0f + 0.5f / 16.0f;
                const float j = static_cast<float>(n % 8) / 16.0f + 0.5f / 16.0f;
                const float partial = 2.0f + i * 4.0f;
                const float want = paired ? partial + 2.0f + i * 4.0f + j * 8.0f : partial;
                for (float got : out[n]) {
                    char msg[160];
                    std::snprintf(msg, sizeof msg, "%s paired=%s pixel=%zu: got %g, want %g", backend,
                                  paired ? "true" : "false", n, static_cast<double>(got), static_cast<double>(want));
                    CHECK_MSG(std::fabs(got - want) < 1e-5f, msg);
                }
            }
        }
        ran += 1;
        std::fprintf(stderr, "partial interpolation renders correctly on %s\n", backend);
    }
    if (ran == 0) std::fprintf(stderr, "no SM 6.6 host with barycentrics; GPU checks skipped\n");
}

// ---- rs:2841 ------------------------------------------------------------------------------------
// Both: the attribute is interpolated and J is used as a value.
void interpolation_and_a_barycentric_value_together() {
    const auto [e0, e1] = exp_mrt0(3);
    const std::array<std::uint32_t, 6> words = {
        0xC808'0000u,        // v_interp_p1_f32 v2, v0, attr0.x
        0xC809'0001,         // v_interp_p2_f32 v2, v1, attr0.x
        v_add_f32(3, 2, 1),  // v3 = attribute + j
        e0,
        e1,
        0xBF81'0000,
    };
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    CHECK((features & 0x2'0000) != 0);
    CHECK(count_barycentrics(inputs) == 1);
    validate(shader);
}

// ---- rs:2870 ------------------------------------------------------------------------------------
// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads the attribute's vertices:
// TEXCOORD0 becomes constant, read with `attributeAtVertex`; i and j are `SV_Barycentrics` values.
void computed_barycentrics_read_the_vertices() {
    const auto [e0, e1] = exp_mrt0(2);
    const std::array<std::uint32_t, 6> words = {
        0x1006'00F0u,  // v_mul_f32 v3, 0.5, v0
        0xC808'0003,   // v_interp_p1_f32 v2, v3, attr0.x
        0xC809'0001,   // v_interp_p2_f32 v2, v1, attr0.x
        e0,
        e1,
        S_ENDPGM,
    };
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    CHECK_MSG((features & 0x2'0000) != 0, "no Barycentrics feature");
    CHECK(count_barycentrics(inputs) == 1);
    CHECK(texcoord_mode(shader) == 1);
    validate(shader);
}

// ---- rs:2892 ------------------------------------------------------------------------------------
// Weights from constants alone: the vertices, no `SV_Barycentrics`.
void constant_weights_need_no_barycentrics() {
    const auto [e0, e1] = exp_mrt0(2);
    const std::array<std::uint32_t, 6> words = {
        0x7E06'02F0u,  // v_mov_b32 v3, 0.5
        0xC808'0003,   // v_interp_p1_f32 v2, v3, attr0.x
        0xC809'0003,   // v_interp_p2_f32 v2, v3, attr0.x
        e0,
        e1,
        S_ENDPGM,
    };
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    CHECK_MSG((features & 0x2'0000) != 0, "no Barycentrics feature");
    CHECK_MSG(count_barycentrics(inputs) == 0, describe(inputs));
    CHECK(texcoord_mode(shader) == 1);
    validate(shader);
}

// ---- rs:2912 ------------------------------------------------------------------------------------
// The same attribute interpolated with the loaded i and j and with a computed j: both from its
// vertices, the first weighed with `SV_Barycentrics`.
void computed_and_loaded_barycentrics_on_one_attribute() {
    const auto [e0, e1] = exp_mrt0(2);
    const std::array<std::uint32_t, 9> words = {
        0xC810'0100u,  // v_interp_p1_f32 v4, v0, attr0.y
        0xC811'0101,   // v_interp_p2_f32 v4, v1, attr0.y
        0x1006'02F0,   // v_mul_f32 v3, 0.5, v1
        0xC808'0000,   // v_interp_p1_f32 v2, v0, attr0.x
        0xC809'0003,   // v_interp_p2_f32 v2, v3, attr0.x
        v_add_f32(2, 2, 4),
        e0,
        e1,
        S_ENDPGM,
    };
    const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
    const auto [features, inputs] = parts(shader);
    (void)features;
    CHECK(count_barycentrics(inputs) == 1);
    CHECK(texcoord_mode(shader) == 1);
    validate(shader);
}

// ---- rs:2936 ------------------------------------------------------------------------------------
// P10 of an attribute the program interpolates is P1 - P0.
void p10_of_an_interpolated_attribute() {
    {
        const auto [e0, e1] = exp_mrt0(2);
        const std::array<std::uint32_t, 7> words = {
            0xC808'0000u,  // v_interp_p1_f32 v2, v0, attr0.x
            0xC809'0001,   // v_interp_p2_f32 v2, v1, attr0.x
            0xC812'0000,   // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        };
        const HostShader shader = unwrap([&] { return pixel(words, PERSP_CENTER); });
        CHECK(texcoord_mode(shader) == 1);
        validate(shader);
    }
    // Alone, what the hardware holds is not known.
    const auto [e0, e1] = exp_mrt0(4);
    const std::array<std::uint32_t, 4> words = {0xC812'0000u, e0, e1, S_ENDPGM};
    CHECK(vs::catch_shader_error([&] { (void)pixel(words, PERSP_CENTER); }).has_value());
}

int run_failed = 0;

void run(const char* name, void (*test)()) {
    x2w::vs::lower::test::set_dump_name(std::string("emit__barycentric_tests__") + name);
    const int before = test_util::failures();
    try {
        test();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: panicked: %s\n", name, e.what());
        ++test_util::failures();
    }
    if (test_util::failures() != before) {
        ++run_failed;
        std::fprintf(stderr, "%s: FAILED\n", name);
    }
}

}  // namespace

#define RUN(test) run(#test, test)

int main() {
    RUN(an_unused_packed_input_matches_its_provoking_vertex_sibling);
    RUN(a_pixel_first_lane_loop_serves_helper_lanes);
    RUN(barycentrics_read_as_values_are_sv_barycentrics);
    RUN(two_barycentric_modes_get_two_elements);
    RUN(interpolation_alone_declares_no_barycentrics);
    RUN(a_standalone_first_half_reads_the_vertices);
    RUN(a_first_half_read_beside_its_pair_keeps_its_partial_value);
    RUN(first_half_values_render_on_hardware_and_warp);
    RUN(interpolation_and_a_barycentric_value_together);
    RUN(computed_barycentrics_read_the_vertices);
    RUN(constant_weights_need_no_barycentrics);
    RUN(computed_and_loaded_barycentrics_on_one_attribute);
    RUN(p10_of_an_interpolated_attribute);
    if (run_failed) std::fprintf(stderr, "barycentric_test: %d test(s) failed\n", run_failed);
    return test_util::finish("barycentric_test");
}
