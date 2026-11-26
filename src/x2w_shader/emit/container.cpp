
#include "x2w_shader/emit/container.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>

#include "x2w_shader/emit/shader.h"

namespace x2w::vs::container {

namespace {

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(std::uint8_t(v));
    out.push_back(std::uint8_t(v >> 8));
    out.push_back(std::uint8_t(v >> 16));
    out.push_back(std::uint8_t(v >> 24));
}

void put32_at(std::uint8_t* at, std::uint32_t v) {
    at[0] = std::uint8_t(v);
    at[1] = std::uint8_t(v >> 8);
    at[2] = std::uint8_t(v >> 16);
    at[3] = std::uint8_t(v >> 24);
}

void put16_at(std::uint8_t* at, std::uint16_t v) {
    at[0] = std::uint8_t(v);
    at[1] = std::uint8_t(v >> 8);
}

std::optional<std::size_t> find(std::span<const std::uint8_t> hay, std::span<const std::uint8_t> needle) {
    if (needle.size() > hay.size()) return std::nullopt;
    for (std::size_t i = 0; i + needle.size() <= hay.size(); ++i)
        if (std::equal(needle.begin(), needle.end(), hay.begin() + std::ptrdiff_t(i))) return i;
    return std::nullopt;
}

void md5_block(std::array<std::uint32_t, 4>& state, const std::uint8_t* block) {
    static constexpr std::uint32_t S[16] = {7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};
    static constexpr std::uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    };
    std::uint32_t x[16];
    for (int i = 0; i < 16; ++i)
        x[i] = std::uint32_t(block[4 * i]) | std::uint32_t(block[4 * i + 1]) << 8 |
               std::uint32_t(block[4 * i + 2]) << 16 | std::uint32_t(block[4 * i + 3]) << 24;
    std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    for (std::uint32_t i = 0; i < 64; ++i) {
        std::uint32_t f = 0, g = 0;
        switch (i / 16) {
        case 0:
            f = (b & c) | (~b & d);
            g = i;
            break;
        case 1:
            f = (d & b) | (~d & c);
            g = (5 * i + 1) % 16;
            break;
        case 2:
            f = b ^ c ^ d;
            g = (3 * i + 5) % 16;
            break;
        default:
            f = c ^ (b | ~d);
            g = (7 * i) % 16;
            break;
        }
        std::uint32_t rotated = std::rotl(a + f + K[i] + x[g], int(S[(i / 16) * 4 + i % 4]));
        std::uint32_t nb = b + rotated;
        a = d;
        d = c;
        c = b;
        b = nb;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

}  // namespace

// (Exposed for the test of the MD5 core.)
void md5_block_for_test(std::array<std::uint32_t, 4>& state, const std::uint8_t* block) { md5_block(state, block); }

std::uint32_t SigElement::system_value() const {
    switch (kind) {
    case 1: return 6;    // vertex id
    case 2: return 8;    // instance id
    case 3: return 1;    // position
    case 4: return 4;    // render target array index
    case 5: return 5;    // viewport array index
    case 6: return 2;    // clip distance
    case 7: return 3;    // cull distance
    case 10: return 7;   // primitive id
    case 12: return 10;
    case 13: return 9;   // front face
    case 14: return 66;
    case 15: return 70;
    case 16: return 64;  // target
    case 17: return 65;  // depth
    case 18: return 68;
    case 19: return 67;
    case 20: return 69;
    case 28: return 23;  // barycentrics
    case 30: return 25;  // cull primitive
    default: return 0;
    }
}

std::uint32_t SigElement::sig_comp() const {
    switch (comp) {
    case 1:
    case 5:
        return 1;  // bool and u32
    case 4:
        return 2;
    case 9:
        return 3;
    default:
        return 0;
    }
}

std::uint8_t SigElement::mask() const { return std::uint8_t(((1u << cols) - 1) << col); }

std::vector<std::uint8_t> signature(std::span<const SigElement> elements, bool input) {
    std::vector<std::uint8_t> out;
    put32(out, std::uint32_t(elements.size()));
    put32(out, 8);
    // Validator 1.6 writes one name per element in element order, repeats included, except that
    // system values share theirs (it keys names by the semantic's string, which is one static
    // string per system value); the rows sorted by register (unallocated last), then name offset.
    std::vector<std::uint32_t> offsets;
    std::uint32_t next = 8 + 32 * std::uint32_t(elements.size());
    std::vector<const std::string*> names;
    for (std::size_t i = 0; i < elements.size(); ++i) {
        const SigElement& e = elements[i];
        std::optional<std::size_t> shared;
        if (e.kind != 0) {
            for (std::size_t j = 0; j < i; ++j)
                if (elements[j].kind == e.kind) {
                    shared = j;
                    break;
                }
        }
        if (shared) {
            offsets.push_back(offsets[*shared]);
        } else {
            offsets.push_back(next);
            next += std::uint32_t(e.name.size()) + 1;
            names.push_back(&e.name);
        }
    }
    std::vector<std::size_t> order(elements.size());
    std::iota(order.begin(), order.end(), std::size_t(0));
    std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
        auto kx = std::make_pair(elements[x].row.value_or(UINT32_MAX), offsets[x]);
        auto ky = std::make_pair(elements[y].row.value_or(UINT32_MAX), offsets[y]);
        return kx < ky;
    });
    for (std::size_t i : order) {
        const SigElement& e = elements[i];
        std::uint32_t offset = offsets[i];
        auto used = std::uint8_t(e.usage << e.col);
        std::uint8_t rw = input ? used : std::uint8_t(~used & 0xf);
        for (std::uint32_t v : {0u, offset, e.index, e.system_value(), e.sig_comp(), e.row.value_or(UINT32_MAX)})
            put32(out, v);
        out.push_back(e.mask());
        out.push_back(rw);
        out.push_back(0);
        out.push_back(0);
        put32(out, 0);
    }
    for (const std::string* name : names) {
        out.insert(out.end(), name->begin(), name->end());
        out.push_back(0);
    }
    return out;
}

std::uint64_t mesh_output_bytes(std::span<const SigElement> vertex, std::span<const SigElement> primitive,
                                std::uint32_t max_vertices, std::uint32_t max_primitives) {
    auto scalars = [](std::span<const SigElement> els) {
        std::uint64_t n = 0;
        for (const SigElement& e : els) n += e.cols;
        return n;
    };
    auto align = [](std::uint32_t n) { return (std::uint64_t(n) + 31) / 32 * 32; };
    return 4 * (scalars(vertex) * align(max_vertices) + scalars(primitive) * align(max_primitives));
}

std::vector<std::uint8_t> psv(const Psv& p) {
    auto vectors = [](std::span<const SigElement> els) {
        std::uint32_t m = 0;
        for (const SigElement& e : els)
            if (e.row) m = std::max(m, *e.row + 1);
        return m;
    };
    const std::uint32_t in_vectors = vectors(p.inputs), out_vectors = vectors(p.outputs);
    std::vector<std::uint8_t> out;
    put32(out, 48);
    std::uint8_t info[48] = {};
    info[0] = std::uint8_t(p.flag);
    if (p.stage == 0) {
        // PSInfo.SampleFrequency: an input interpolated per sample, or the sample index.
        bool any = false;
        for (const SigElement& e : p.inputs)
            if (e.interp == 6 || e.interp == 7 || e.kind == 12) any = true;
        info[1] = std::uint8_t(any);
    }
    std::uint32_t wmin = 0, wmax = UINT32_MAX;
    if (p.wave_size) wmin = wmax = *p.wave_size;
    put32_at(info + 16, wmin);
    put32_at(info + 20, wmax);
    info[24] = p.stage;
    for (std::size_t k = 0; k < 3; ++k) put32_at(info + 36 + 4 * k, p.threads[k]);
    info[28] = std::uint8_t(p.inputs.size());
    info[29] = std::uint8_t(p.outputs.size());
    info[31] = std::uint8_t(in_vectors);
    info[32] = std::uint8_t(out_vectors);
    if (p.mesh) {
        const shader::Mesh& mesh = *p.mesh;
        put32_at(info + 0, p.mesh_lds_bytes);
        put16_at(info + 12, std::uint16_t(mesh.max_vertices));
        put16_at(info + 14, std::uint16_t(mesh.max_primitives));
        info[26] = std::uint8_t(vectors(p.prims));
        info[27] = mesh.topology;
        info[30] = std::uint8_t(p.prims.size());
    }
    if (p.geometry) {
        // Input primitive (3: triangles), output topology (5: triangle strips), stream mask,
        // position written; the vertex maximum.
        put32_at(info + 0, 3);
        put32_at(info + 4, 5);
        put32_at(info + 8, 1);
        info[12] = std::uint8_t(p.flag);
        put16_at(info + 26, std::uint16_t(*p.geometry));
    }
    out.insert(out.end(), info, info + 48);
    put32(out, std::uint32_t(p.bindings.size()));
    if (!p.bindings.empty()) {
        put32(out, 24);
        for (const Binding& b : p.bindings)
            for (std::uint32_t v : {b.kind, b.space, b.register_, b.register_, b.dxil_kind, 0u}) put32(out, v);
    }
    // Names of arbitrary semantics (system values use ""), then indices.
    std::vector<std::uint8_t> strings = {0};
    std::vector<std::uint32_t> indices;
    struct Row {
        std::uint32_t name, index;
        std::uint8_t start, cols;
        const SigElement* e;
    };
    std::vector<Row> elements;
    auto add = [&](const SigElement& e) {
        std::size_t name = 0;
        if (e.kind == 0) {
            std::vector<std::uint8_t> needle(e.name.begin(), e.name.end());
            needle.push_back(0);
            if (auto at = find(strings, needle)) {
                name = *at;
            } else {
                strings.insert(strings.end(), needle.begin(), needle.end());
                name = strings.size() - needle.size();
            }
        }
        std::size_t index = 0;
        auto it = std::find(indices.begin(), indices.end(), e.index);
        if (it != indices.end()) {
            index = std::size_t(it - indices.begin());
        } else {
            indices.push_back(e.index);
            index = indices.size() - 1;
        }
        std::uint8_t start = 0, alloc = 0;
        if (e.row) {
            start = std::uint8_t(*e.row);
            alloc = std::uint8_t(0x40 | (e.col << 4));
        }
        elements.push_back(Row{std::uint32_t(name), std::uint32_t(index), start, std::uint8_t(e.cols | alloc), &e});
    };
    for (const SigElement& e : p.inputs) add(e);
    for (const SigElement& e : p.outputs) add(e);
    for (const SigElement& e : p.prims) add(e);
    strings.resize((strings.size() + 3) / 4 * 4, 0);
    put32(out, std::uint32_t(strings.size()));
    out.insert(out.end(), strings.begin(), strings.end());
    put32(out, std::uint32_t(indices.size()));
    for (std::uint32_t i : indices) put32(out, i);
    if (!elements.empty()) {
        put32(out, 16);
        for (const Row& r : elements) {
            put32(out, r.name);
            put32(out, r.index);
            const std::uint8_t rec[8] = {1, r.start, r.cols, r.e->kind, std::uint8_t(r.e->sig_comp()), r.e->interp, 0, 0};
            out.insert(out.end(), rec, rec + 8);
        }
    }
    // Input-to-output dependencies, as the view-id state declares them: every output scalar on
    // every input scalar.
    if (in_vectors > 0 && out_vectors > 0) {
        auto scalars = [](std::span<const SigElement> els) {
            std::uint32_t m = 0;
            for (const SigElement& e : els)
                if (e.row) m = std::max(m, *e.row * 4 + std::uint32_t(std::uint8_t(e.col + e.cols)));
            return m;
        };
        const std::uint32_t ni = scalars(p.inputs), no = scalars(p.outputs);
        const std::uint32_t per = (out_vectors * 4 + 31) / 32;
        for (std::uint32_t i = 0; i < in_vectors * 4; ++i) {
            for (std::uint32_t k = 0; k < per; ++k) {
                std::uint32_t bits = 0;
                if (i < ni) bits = std::min(no > 32 * k ? no - 32 * k : 0u, 32u);
                std::uint32_t mask = bits == 32 ? UINT32_MAX : (1u << bits) - 1;
                put32(out, mask);
            }
        }
    }
    return out;
}

std::vector<std::uint8_t> build(std::span<const Part> parts) {
    const std::size_t table = 0x20 + 4 * parts.size();
    std::size_t total = table;
    for (const Part& p : parts) total += 8 + p.data.size();
    std::vector<std::uint8_t> out;
    out.reserve(total);
    const char magic[4] = {'D', 'X', 'B', 'C'};
    out.insert(out.end(), magic, magic + 4);
    out.insert(out.end(), 16, 0);
    out.push_back(1);
    out.push_back(0);
    out.push_back(0);
    out.push_back(0);
    put32(out, std::uint32_t(total));
    put32(out, std::uint32_t(parts.size()));
    std::size_t offset = table;
    for (const Part& p : parts) {
        put32(out, std::uint32_t(offset));
        offset += 8 + p.data.size();
    }
    for (const Part& p : parts) {
        out.insert(out.end(), p.fourcc.begin(), p.fourcc.end());
        put32(out, std::uint32_t(p.data.size()));
        out.insert(out.end(), p.data.begin(), p.data.end());
    }
    auto digest = hash(std::span<const std::uint8_t>(out).subspan(20));
    std::copy(digest.begin(), digest.end(), out.begin() + 4);
    return out;
}

std::array<std::uint8_t, 16> hash(std::span<const std::uint8_t> data) {
    const std::size_t len = data.size();
    const std::size_t left = len % 64;
    std::vector<std::uint8_t> tail;
    tail.reserve(128);
    auto put = [&](std::uint32_t v) { put32(tail, v); };
    if (left < 56) {
        put(std::uint32_t(len) << 3);
        tail.insert(tail.end(), data.begin() + std::ptrdiff_t(len - left), data.end());
        tail.push_back(0x80);
        tail.resize(60, 0);
    } else {
        tail.insert(tail.end(), data.begin() + std::ptrdiff_t(len - left), data.end());
        tail.push_back(0x80);
        tail.resize(64, 0);
        put(std::uint32_t(len) << 3);
        tail.resize(124, 0);
    }
    put((std::uint32_t(len) << 1) | 1);
    std::array<std::uint32_t, 4> state = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    for (std::size_t at = 0; at + 64 <= len - left; at += 64) md5_block(state, data.data() + at);
    for (std::size_t at = 0; at + 64 <= tail.size(); at += 64) md5_block(state, tail.data() + at);
    std::array<std::uint8_t, 16> out{};
    for (std::size_t k = 0; k < 4; ++k) put32_at(out.data() + 4 * k, state[k]);
    return out;
}

}  // namespace x2w::vs::container
