// the geometry shader that completes the console's rectangle lists (list topology 17, pipeline topology
// type 19).
//
// A rectangle comes as three vertices, three corners of a screen-aligned rectangle; the hardware
// draws the whole rectangle and continues every vertex shader output linearly into the fourth
// corner. The host has no such topology: the draw goes out as a triangle list with this shader
// behind the translated vertex shader, so vertex counts, vertex numbers, indices and indirect
// arguments stay the title's, and each rectangle arrives here as one triangle.
//
// Which vertex is the corner between the other two varies: the corner is the vertex that shares x
// with one of the others and y with the other; the fourth vertex lies opposite it, a + b - corner
// for the position and every float output. That holds for three vertices with one positive w;
// three that differ in w, or are not such corners, stay the triangle they make.
//
// Flat values are the first vertex's: both triangles start with it, which is the vertex the host
// takes a flat input from, and the fourth vertex's integer outputs are the first vertex's.
//
// shortcut: that the console takes a rectangle's flat values from its first vertex follows from
// the hardware's provoking-vertex setting for triangles; it is not measured on a rectangle.

#include "x2w_shader/emit/rect.h"

#include <algorithm>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/signature.h"
#include "x2w_shader/error.h"

namespace x2w::vs {

namespace {

using bitcode::Attr;
using bitcode::Bin;
using bitcode::Term;
using bitcode::V;
namespace pred = bitcode::pred;
using container::SigElement;
using shader::Geometry;
using shader::Shader;

// Vertices emitted per rectangle: two triangles.
constexpr std::uint32_t VERTICES = 6;

// The scalars a geometry shader may emit per invocation (vertices times output components).
constexpr std::uint32_t MAX_SCALARS = 1024;

// Column col of input id at vertex `vertex` of the triangle.
V load(Shader& s, std::size_t id, std::uint8_t col, std::uint32_t vertex) {
    const bool f = s.inputs[id].comp == 9;
    const char* name = f ? "loadInput.f32" : "loadInput.i32";
    bitcode::Ty ty = f ? s.t.f32 : s.t.i32;
    const V args[4] = {s.m.i32(std::uint32_t(id)), s.m.i32(0), s.m.i8(col), s.m.i32(vertex)};
    std::size_t entry = s.entry;
    return s.op(entry, name, Attr::ReadNone, ty, 4, args);
}

// Two floats are equal.
V same(Shader& s, V a, V b) { return s.m.cmp(s.entry, pred::OEQ, a, b); }
V both(Shader& s, V a, V b) { return s.m.bin(s.entry, Bin::And, a, b); }
V either(Shader& s, V a, V b) { return s.m.bin(s.entry, Bin::Or, a, b); }
V not_(Shader& s, V a) {
    std::size_t entry = s.entry;
    V on = s.m.bool_(true);
    return s.m.bin(entry, Bin::Xor, a, on);
}

}  // namespace

std::vector<std::uint8_t> rect_build(std::vector<SigElement> outputs, std::span<const SigElement> ps_inputs) {
    auto unsupported = [](const char* what) { return ShaderError::unsupported(0, what); };
    if (std::any_of(outputs.begin(), outputs.end(), [](const SigElement& e) { return !e.row; }))
        throw unsupported("a rectangle list output without a register");
    std::size_t position = outputs.size();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const SigElement& e = outputs[i];
        if (e.kind == 3 && e.comp == 9 && e.cols == 4) {
            position = i;
            break;
        }
    }
    if (position == outputs.size()) throw unsupported("a rectangle list without a four-component position");
    for (SigElement& e : outputs) {
        // Interpolation as emit_program declares a vertex shader's outputs; a geometry shader's
        // inputs carry the same modes.
        if (e.kind == 3)
            e.interp = 4;
        else if ((e.kind == 0 || e.kind == 6 || e.kind == 7) && e.comp == 9)
            e.interp = 2;
        else if (e.kind == 0)
            e.interp = 1;
        else
            e.interp = 0;
        e.usage = std::uint8_t((1u << e.cols) - 1);
    }
    Shader s(false);
    s.geometry = Geometry{VERTICES};
    s.inputs = outputs;
    // The primitive id the pixel shader declares, as one more output.
    const SigElement* wanted = nullptr;
    for (const SigElement& e : ps_inputs)
        if (e.kind == 10 && e.row) {
            wanted = &e;
            break;
        }
    if (wanted && std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind == 10; }))
        wanted = nullptr;
    std::optional<std::size_t> primitive;
    if (wanted) {
        const SigElement& e = *wanted;
        // Alone in its columns, and in a row of constant elements.
        const bool clash = std::any_of(outputs.begin(), outputs.end(), [&](const SigElement& o) {
            const bool overlap = o.col < e.col + e.cols && e.col < o.col + o.cols;
            return o.row == e.row && (o.interp != 1 || overlap);
        });
        if (clash) throw unsupported("a rectangle list whose primitive id shares a register with a vertex output");
        SigElement p = e;
        p.interp = 1;
        p.usage = std::uint8_t((1u << e.cols) - 1);
        outputs.push_back(std::move(p));
        primitive = outputs.size() - 1;
    }
    std::uint32_t scalars = 0;
    for (const SigElement& e : outputs) scalars += e.cols;
    if (scalars * VERTICES > MAX_SCALARS)
        throw unsupported("a rectangle list with more output components than a geometry shader emits");
    s.outputs = outputs;
    const std::size_t b = s.entry;

    // Every input of the three vertices: [element][column][vertex].
    using V3 = std::array<V, 3>;
    std::vector<std::vector<V3>> corners;
    for (std::size_t id = 0; id < s.inputs.size(); ++id) {
        std::vector<V3> columns;
        for (std::uint8_t col = 0; col < s.inputs[id].cols; ++col) {
            V3 c;
            for (std::uint32_t vertex = 0; vertex < 3; ++vertex) c[vertex] = load(s, id, col, vertex);
            columns.push_back(c);
        }
        corners.push_back(std::move(columns));
    }

    // One positive w, and no two vertices in one place.
    const V3 x = corners[position][0], y = corners[position][1], w = corners[position][3];
    V zero = s.m.f32(0);
    V plain = s.m.cmp(b, pred::OGT, w[0], zero);
    for (std::size_t k = 1; k < 3; ++k) {
        V one_w = same(s, w[k], w[0]);
        plain = both(s, plain, one_w);
    }
    for (auto [i, j] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}}) {
        V same_x = same(s, x[i], x[j]);
        V same_y = same(s, y[i], y[j]);
        V together = both(s, same_x, same_y);
        V apart = not_(s, together);
        plain = both(s, plain, apart);
    }
    // corner[i]: vertex i is the corner between the other two (the first that shares x with one
    // of them and y with the other).
    std::vector<V> corner;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t a = (i + 1) % 3, c = (i + 2) % 3;
        V xa = same(s, x[i], x[a]);
        V yc = same(s, y[i], y[c]);
        V xc = same(s, x[i], x[c]);
        V ya = same(s, y[i], y[a]);
        V one = both(s, xa, yc);
        V other = both(s, xc, ya);
        V shares = either(s, one, other);
        V here = both(s, shares, plain);
        for (V found : corner) {
            V earlier = not_(s, found);
            here = both(s, here, earlier);
        }
        corner.push_back(here);
    }

    // What the six emitted vertices carry: [element][column][vertex].
    using V6 = std::array<V, 6>;
    std::vector<std::vector<V6>> emitted;
    for (std::size_t id = 0; id < corners.size(); ++id) {
        const bool fl = s.inputs[id].comp == 9;
        std::vector<V6> out;
        for (const V3& pc : corners[id]) {
            const V p0 = pc[0], p1 = pc[1], p2 = pc[2];
            // The fourth vertex: a + b - corner. Integers (flat), and three vertices without a
            // corner, keep the first vertex's value; the second triangle of the latter is then
            // empty.
            V fourth = p0;
            if (fl) {
                const std::array<std::array<V, 3>, 3> sides = {{{p1, p2, p0}, {p2, p0, p1}, {p0, p1, p2}}};
                for (std::size_t k = 0; k < 3; ++k) {
                    V sum = s.m.bin(b, Bin::Add, sides[k][0], sides[k][1]);
                    V opposite = s.m.bin(b, Bin::Sub, sum, sides[k][2]);
                    fourth = s.m.select(b, corner[k], opposite, fourth);
                }
            }
            // Corner first: (v0, v1, v3), (v0, v3, v2). Second: (v0, v1, v2), (v0, v2, v3).
            // Third: (v0, v1, v2), (v0, v3, v1).
            V third = s.m.select(b, corner[0], fourth, p2);
            V fifth = s.m.select(b, corner[1], p2, fourth);
            V sixth = s.m.select(b, corner[0], p2, p1);
            sixth = s.m.select(b, corner[1], fourth, sixth);
            out.push_back(V6{p0, p1, third, p0, fifth, sixth});
        }
        emitted.push_back(std::move(out));
    }

    // The rectangle's number: the triangle's, three vertices a rectangle.
    std::optional<std::pair<std::size_t, V>> number;
    if (primitive) {
        V n = s.op(b, "primitiveID.i32", Attr::ReadNone, s.t.i32, 108, std::span<const V>());
        number = std::make_pair(*primitive, n);
    }
    V stream = s.m.i8(0);
    for (std::size_t vertex = 0; vertex < VERTICES; ++vertex) {
        for (std::size_t id = 0; id < emitted.size(); ++id) {
            const char* name = s.outputs[id].comp == 9 ? "storeOutput.f32" : "storeOutput.i32";
            for (std::size_t col = 0; col < emitted[id].size(); ++col) {
                const V args[4] = {s.m.i32(std::uint32_t(id)), s.m.i32(0), s.m.i8(std::uint8_t(col)),
                                   emitted[id][col][vertex]};
                s.op(b, name, Attr::NoUnwind, s.t.void_, 5, args);
            }
        }
        if (number) {
            const V args[4] = {s.m.i32(std::uint32_t(number->first)), s.m.i32(0), s.m.i8(0), number->second};
            s.op(b, "storeOutput.i32", Attr::NoUnwind, s.t.void_, 5, args);
        }
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void_, 97, {stream});
        // The strip ends after the first triangle.
        if (vertex == 2) s.op(b, "cutStream", Attr::NoUnwind, s.t.void_, 98, {stream});
    }
    s.m.terminate(b, Term::ret());
    return s.finish();
}

std::vector<std::uint8_t> rect_list_gs(std::span<const std::uint8_t> vs, std::optional<std::span<const std::uint8_t>> ps) {
    auto sig = [](std::span<const std::uint8_t> dxil, dxbc::FourCC fourcc) -> std::vector<SigElement> {
        const dxbc::Container c = dxbc::Container::parse(dxil);
        if (const dxbc::Part* part = c.part(fourcc)) return signature(part->data);
        return {};
    };
    std::vector<SigElement> outputs = sig(vs, dxbc::FourCC::OSG1);
    std::vector<SigElement> wanted;
    if (ps) wanted = sig(*ps, dxbc::FourCC::ISG1);
    return rect_build(std::move(outputs), wanted);
}

}  // namespace x2w::vs
