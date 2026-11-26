// Port of the upstream Rust module emit/rect.rs's device tests that container_test.cpp does not cover: the
// validator part of the_rectangle_stage_is_a_geometry_shader, and rectangles_are_drawn_whole
// (with pass_through). Skips the validator checks without dxil.dll and the draw without a
// hardware SM 6.6 device (and says so).
//
// pass_through returns two FakeShaders (what the host reads of a HostShader, as in host_test):
// HostShader needs the front end's native.h, which this test does not link.

#include <windows.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/host.h"
#include "x2w_shader/emit/rect.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"

using namespace x2w::vs;
using container::SigElement;
using shader::Shader;
using validator::Validator;

namespace {

SigElement element(const char* name, std::uint32_t index, std::uint8_t kind, std::uint8_t comp, std::uint32_t row,
                   std::uint8_t cols) {
    SigElement e;
    e.name = name;
    e.index = index;
    e.kind = kind;
    e.comp = comp;
    e.interp = 0;
    e.row = row;
    e.col = 0;
    e.cols = cols;
    e.usage = 0;
    e.half = false;
    return e;
}

// A position, a float interpolant and an integer one.
std::vector<SigElement> outputs() {
    return {element("SV_Position", 0, 3, 9, 0, 4), element("TEXCOORD", 0, 0, 9, 1, 4),
            element("TEXCOORD", 1, 0, 5, 2, 2)};
}

std::optional<Validator> load_validator() {
    std::string error;
    std::optional<Validator> v = Validator::load(Validator::default_path(), &error);
    if (!v) std::printf("rect_test: no validator (%s); validator checks skipped\n", error.c_str());
    return v;
}

// Validates a copy of container; a rejection fails the test (Rust's unwrap).
void validate(Validator& v, const std::vector<std::uint8_t>& container, const char* what) {
    std::vector<std::uint8_t> copy = container;
    std::optional<std::string> r = v.validate(copy);
    CHECK(!r);
    if (r) std::fprintf(stderr, "%s rejected: %s\n", what, r->c_str());
}

// The stage is a geometry shader of triangles in and six strip vertices out, with the vertex
// shader's outputs on both sides and the primitive id the pixel shader declares; the validator,
// where there is one, accepts it. (The container checks are in container_test.)
void the_rectangle_stage_is_a_geometry_shader() {
    const std::vector<SigElement> wanted = {element("SV_PrimitiveID", 0, 10, 5, 3, 1)};
    const std::vector<std::uint8_t> gs = rect_build(outputs(), wanted);
    if (std::optional<Validator> v = load_validator()) {
        validate(*v, gs, "rectangle stage");
        const std::vector<std::uint8_t> plain = rect_build(outputs(), {});
        validate(*v, plain, "rectangle stage without a primitive id");
        std::printf("rect_test: rectangle stages validated (%zu and %zu bytes)\n", gs.size(), plain.size());
    }
}

// What the host reads of a HostShader (Stage converts from it).
struct FakeShader {
    std::vector<std::uint8_t> dxil;
    std::vector<sampler::StaticSampler> static_samplers;
    std::vector<std::uint8_t> root_cbvs, root_srvs, root_uavs;
    // No user data (Rust: user_data empty).
    std::uint32_t user_data_dwords() const { return 0; }
};

// A vertex shader that passes a position and a float4 through, and a pixel shader that writes
// the float4.
std::pair<FakeShader, FakeShader> pass_through() {
    SigElement position = element("SV_Position", 0, 3, 9, 0, 4);
    position.interp = 4;
    position.usage = 15;
    SigElement coord = element("TEXCOORD", 0, 0, 9, 1, 4);
    coord.interp = 2;
    coord.usage = 15;
    Shader vs(false);
    SigElement vs_pos = element("POSITION", 0, 0, 9, 0, 4);
    vs_pos.usage = 15;
    SigElement vs_coord = coord;
    vs_coord.interp = 0;
    vs.inputs = {vs_pos, vs_coord};
    vs.outputs = {position, coord};
    Shader ps(true);
    SigElement ps_pos = position;
    ps_pos.usage = 0;
    ps.inputs = {ps_pos, coord};
    SigElement target = element("SV_Target", 0, 16, 9, 0, 4);
    target.usage = 15;
    ps.outputs = {target};
    // Vertex: output id from input id. Pixel: target 0 from input 1.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> vs_copies = {{0, 0}, {1, 1}};
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> ps_copies = {{1, 0}};
    const std::pair<Shader*, const std::vector<std::pair<std::uint32_t, std::uint32_t>>*> stages[] = {
        {&vs, &vs_copies}, {&ps, &ps_copies}};
    for (auto [s, copies] : stages) {
        const std::size_t b = s->entry;
        for (auto [from, to] : *copies) {
            for (std::uint8_t col = 0; col < 4; ++col) {
                // In Rust's order (each call numbers a value).
                const bitcode::V a0 = s->m.i32(from);
                const bitcode::V a1 = s->m.i32(0);
                const bitcode::V a2 = s->m.i8(col);
                const bitcode::V a3 = s->m.undef(s->t.i32);
                const bitcode::V v = s->op(b, "loadInput.f32", bitcode::Attr::ReadNone, s->t.f32, 4, {a0, a1, a2, a3});
                const bitcode::V b0 = s->m.i32(to);
                const bitcode::V b1 = s->m.i32(0);
                const bitcode::V b2 = s->m.i8(col);
                s->op(b, "storeOutput.f32", bitcode::Attr::NoUnwind, s->t.void_, 5, {b0, b1, b2, v});
            }
        }
        s->m.terminate(b, bitcode::Term::ret());
    }
    FakeShader vs_h;
    vs_h.dxil = vs.finish();
    FakeShader ps_h;
    ps_h.dxil = ps.finish();
    return {std::move(vs_h), std::move(ps_h)};
}

// Three corners of the whole target, the corner between the other two first (as the tiled
// lighting sends its tiles), second (as full-screen rectangles come) or third: with the stage
// every texel is drawn and the interpolant continues into the fourth corner's half; without it
// part of the target is not drawn.
void rectangles_are_drawn_whole() {
    std::optional<host::Host> h;
    try {
        h.emplace(false);
    } catch (const host::Error& e) {
        std::printf("rect_test: no SM 6.6 device (%s); rectangles_are_drawn_whole skipped\n", e.what());
        return;
    }
    host::Host& hst = *h;
    {
        DXGI_ADAPTER_DESC1 desc{};
        Microsoft::WRL::ComPtr<IDXGIFactory4> f;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
        if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f))) &&
            SUCCEEDED(f->EnumAdapterByLuid(hst.device->GetAdapterLuid(), IID_PPV_ARGS(&a))))
            a->GetDesc1(&desc);
        std::printf("rect_test: device %ls\n", desc.Description);
    }
    auto [vs, ps] = pass_through();
    if (std::optional<Validator> v = load_validator()) {
        validate(*v, vs.dxil, "pass-through vertex shader");
        validate(*v, ps.dxil, "pass-through pixel shader");
    }
    const std::vector<std::uint8_t> gs = rect_list_gs(vs.dxil, std::span<const std::uint8_t>(ps.dxil));
    auto root = hst.root_signature(vs, ps);
    // Position (z 0.5, w 1) and an interpolant that is the position.
    using Corner = std::pair<float, float>;
    const std::pair<const char*, std::array<Corner, 3>> cases[] = {
        {"corner first", {{{-1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, 1.0f}}}},
        {"corner second", {{{-1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}}}},
        {"corner third", {{{-1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}}}},
    };
    for (const auto& [name, corners] : cases) {
        std::vector<float> vertices;
        for (auto [x, y] : corners) vertices.insert(vertices.end(), {x, y, 0.5f, 1.0f, x, y, 0.0f, 1.0f});
        auto draw = [&](std::span<const std::uint8_t> with_gs) {
            auto pso = hst.pipeline_with_gs(root.Get(), vs.dxil, with_gs, ps.dxil);
            host::Draw d;
            d.vertices = vertices;
            d.stride = 32;
            d.size = {8, 8};
            d.mesh_groups = 0;
            return hst.draw(root.Get(), pso.Get(), d);
        };
        const std::vector<std::array<float, 4>> whole = draw(gs);
        CHECK(whole.size() == 64);
        bool all = true;
        for (std::size_t i = 0; i < whole.size(); ++i) {
            const float x = float(i % 8), y = float(i / 8);
            const std::array<float, 4> want = {(x + 0.5f) / 4.0f - 1.0f, 1.0f - (y + 0.5f) / 4.0f, 0.0f, 1.0f};
            bool ok = true;
            for (int c = 0; c < 4; ++c) ok = ok && std::fabs(whole[i][c] - want[c]) < 1e-4f;
            if (!ok) {
                std::fprintf(stderr, "%s: texel %zu is [%g, %g, %g, %g], not [%g, %g, %g, %g]\n", name, i, whole[i][0],
                             whole[i][1], whole[i][2], whole[i][3], want[0], want[1], want[2], want[3]);
                all = false;
                break;  // Rust's assert! stops at the first
            }
        }
        CHECK(all);
        const std::vector<std::array<float, 4>> half = draw({});
        std::size_t missing = 0;
        for (std::size_t i = 0; i < half.size() && i < whole.size(); ++i) missing += half[i] != whole[i];
        if (missing < 24) std::fprintf(stderr, "%s: only %zu texels need the stage\n", name, missing);
        CHECK(missing >= 24);
        std::printf("rect_test: %s: every texel drawn with the stage, %zu missing without\n", name, missing);
    }
}

template <class F>
void run(const char* name, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        CHECK(false);
    }
}

}  // namespace

int main() {
    run("the_rectangle_stage_is_a_geometry_shader", the_rectangle_stage_is_a_geometry_shader);
    run("rectangles_are_drawn_whole", rectangles_are_drawn_whole);
    return test_util::finish("rect_test");
}
