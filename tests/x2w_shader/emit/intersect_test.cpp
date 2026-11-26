// Port of the upstream Rust module emit/intersect_tests.rs. The ray-tracing node test end to end: a synthetic
// compute program that runs image_bvh64_intersect_ray (or the 32-bit form) once per thread,
// lowered to DXIL, validated by dxil.dll when present, run on the host GPU and WARP, and compared
// dword for dword with the CPU reference (bvh.h).
//
// The nodes are in a raw buffer. Two ways to name it are run: the buffer's descriptor from the
// program's table (a resource of the host, BvhNodes::Resource), and the title's way, a descriptor
// of numbers over title memory with the structure's address in the node pointer or in the
// descriptor. These tests have no host that keeps copies of title memory and their directory,
// so for the second they put a stand-in under the node reader (lower::test::set_memory, Rust's
// lower::intersect::TEST_MEMORY); without it the same program reads through the directory, which
// is only checked to lower and validate.
//
// Each Rust #[test] is a function here, run one after the other (Rust's ONE_AT_A_TIME lock has
// nothing to serialise); a Rust panic is an exception that fails that function.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "test_util.h"
#include "x2w_shader/bvh.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/host.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"

using namespace x2w::vs;
using bvh::Ray;
using host::Dispatch;
using host::Host;
using host::Root;
using host::Stride;
using host::View;
using native::ComputeIo;
using native::HwStage;
using native::NativeProgram;
using native::ResourceType;
using native::RootParam;
using native::RootType;
using native::UsedResource;
using resource::BvhNodes;
namespace nodes = bvh::testing;

namespace {

[[noreturn]] void panic(const std::string& what) { throw std::logic_error(what); }

// Threads of one group, each with its own record.
constexpr std::uint32_t THREADS = 32;
// Bytes of a thread's record: the node pointer (two dwords), the ray's extent, origin, direction
// and inverse direction, and whether the thread runs the instruction.
constexpr std::uint32_t RECORD = 64;
// Where the nodes start in the buffer.
constexpr std::uint32_t NODES = THREADS * RECORD;
// What the result registers hold before the instruction.
constexpr bvh::Result4 BEFORE = {1, 2, 3, 4};

// How the program names the nodes' memory (Rust enum Descriptor).
struct Descriptor {
    // Table: the buffer's descriptor from the table. Else Numbers, as the title builds it:
    // dwords 0 and 1 (the base address in 256-byte units and the box sorting bit); the size and
    // the rest of the flags as the title sets them.
    bool table = true;
    std::uint32_t word0 = 0;
    std::uint32_t word1 = 0;

    static Descriptor Table() { return {true, 0, 0}; }
    static Descriptor Numbers(std::uint32_t w0, std::uint32_t w1) { return {false, w0, w1}; }
};

// The descriptor the title builds for the 64-bit form: the whole address space, sorted boxes,
// barycentrics returned.
const Descriptor WHOLE = Descriptor::Numbers(0, bvh::BOX_SORT);

// The program: per thread, the record's twelve operand dwords in v1-v12, the instruction's
// result in v13-v16 (under a lane mask of the threads whose record says so), stored to the UAV at
// 16 bytes a thread.
std::vector<std::uint32_t> program(bool wide, Descriptor descriptor) {
    std::vector<std::uint32_t> w = {
        0xBE85'0380,  // s_mov_b32 s5, 0
        0xF408'0202,
        0xFA00'0000,  // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408'0302,
        0xFA00'0020,  // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428'0086,  // v_lshlrev_b32 v20, 6, v0
        0xE038'1000,
        0x8002'0114,  // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038'1010,
        0x8002'0514,  // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038'1020,
        0x8002'0914,  // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030'1030,
        0x8002'1114,  // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    };
    for (std::uint32_t k = 0; k < 4; ++k) {
        // v_mov_b32 v13+k, 1+k
        w.push_back(0x7E00'0200u | ((13 + k) << 17) | (0x81 + k));
    }
    std::uint32_t rsrc;
    if (descriptor.table) {
        rsrc = 8;
    } else {
        // s_mov_b32 s24, word0
        w.insert(w.end(), {0xBE98'03FFu, descriptor.word0});
        if (descriptor.word1 == bvh::BOX_SORT) {
            // As the title makes it.
            w.push_back(0xBE99'0B81);  // s_brev_b32 s25, 1
        } else {
            w.insert(w.end(), {0xBE99'03FFu, descriptor.word1});  // s_mov_b32 s25, word1
        }
        w.insert(w.end(), {
                              0xBE9A'03C1u,  // s_mov_b32 s26, -1
                              0xBE9B'03FFu,
                              0x8900'03FFu,  // s_mov_b32 s27, 0x890003ff
                          });
        rsrc = 24;
    }
    w.insert(w.end(), {
                          0xBE94'037Eu,  // s_mov_b32 s20, exec_lo
                          0x7DAA'2280u,  // v_cmpx_ne_u32 exec_lo, 0, v17
                      });
    const std::uint32_t first = 0x0000'0D01u | ((rsrc >> 2) << 16);
    if (wide) {
        // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[rsrc:rsrc+3]
        w.insert(w.end(), {0xF19C'9F07u, first, 0x0504'0302u, 0x0908'0706u, 0x000C'0B0Au});
    } else {
        // image_bvh_intersect_ray v[13:16], [v1, v3 .. v12], s[rsrc:rsrc+3]
        w.insert(w.end(), {0xF198'9F07u, first, 0x0605'0403u, 0x0A09'0807u, 0x0000'0C0Bu});
    }
    w.insert(w.end(), {
                          0xBEFE'0314u,  // s_mov_b32 exec_lo, s20
                          0x342A'0084u,  // v_lshlrev_b32 v21, 4, v0
                          0xE078'1000u,
                          0x8003'0D15u,  // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
                          0xBF81'0000u,  // s_endpgm
                      });
    return w;
}

NativeProgram make_native(const std::vector<std::uint32_t>& words) {
    auto table = [](ResourceType kind, std::uint32_t at) {
        UsedResource u;
        u.kind = kind;
        u.root = RootType::Table();
        u.table_offset_dwords = at;
        u.lower_bound = 0;
        u.stride = 4;
        u.dwords = 4;
        u.user_data = 4;
        u.slot = 1;
        return u;
    };
    NativeProgram n;
    n.stage = HwStage::Cs;
    n.wave64 = false;
    n.vgprs = 22;
    RootParam constants;
    constants.root = RootType::Constants();
    constants.table_offset_dwords = 0;
    constants.sgprs = 4;
    constants.base_register = 0;
    constants.user_data = 0;
    constants.slot = 0;
    RootParam tbl;
    tbl.root = RootType::Table();
    tbl.table_offset_dwords = 0;
    tbl.sgprs = 1;
    tbl.base_register = 0;
    tbl.user_data = 4;
    tbl.slot = 1;
    n.layout = {constants, tbl};
    n.used = {table(ResourceType::Srv(), 0), table(ResourceType::Uav(), 8)};
    ComputeIo io;
    io.rsrc2 = 5 << 1;
    io.threads = {THREADS, 1, 1};
    n.io = io;
    for (std::uint32_t x : words)
        for (int b = 0; b < 4; ++b) n.code.push_back(std::uint8_t(x >> (8 * b)));
    return n;
}

// `native`, a compute program, translated (the tests' own way: no container). Rust:
// build_compute_shader.
std::pair<HostShader, resource::Resources> build_compute_shader(const NativeProgram& native) {
    const ir::Program p = ir::build(native.code, native.wave64, native.vgprs);
    const structure::Structured st = structure::structure(p);
    resource::Resources res = resource::track(p, native.layout, native.stage, &st);
    const io::Io io = io::raise(p, native);
    lower::Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    lower::Lowered lowered = lower::lower(parts);
    HostShader shader;
    shader.stage = HwStage::Cs;
    shader.dxil = std::move(lowered.dxil);
    shader.user_data = std::move(lowered.user_data);
    shader.samplers = std::move(lowered.samplers);
    shader.root_cbvs = std::move(lowered.root_cbvs);
    shader.root_srvs = std::move(lowered.root_srvs);
    shader.root_uavs = std::move(lowered.root_uavs);
    shader.root_indexed_views = res.root_indexed_views;
    if (std::optional<validator::Validator> v = validator::Validator::load(validator::Validator::default_path())) {
        std::vector<std::uint8_t> copy = shader.dxil;
        if (std::optional<std::string> e = v->validate(copy)) panic("DXIL validation failed: " + *e);
    }
    return {std::move(shader), std::move(res)};
}

// Lowers the program; the DXIL is validated when the validator is there.
std::pair<HostShader, resource::Resources> build(bool wide, Descriptor descriptor) {
    return build_compute_shader(make_native(program(wide, descriptor)));
}

// One thread's test: a node (by its index in the node list), the kind its pointer carries, and
// the ray; nullopt leaves the thread masked out.
struct Test {
    std::size_t n;
    std::uint32_t kind;
    Ray ray;
};
using Case = std::optional<Test>;

// The scene: node dwords (a 32-bit box node takes 32, the others 16) and where each node starts,
// in dwords from the first node.
struct Scene {
    std::vector<std::uint32_t> words;
    std::vector<std::size_t> starts;

    // Adds a node of `dwords` (16 or 32) and returns its index.
    std::size_t add(const bvh::Words& node, std::size_t dwords) {
        starts.push_back(words.size());
        std::array<std::uint32_t, 32> padded{};
        std::copy(node.begin(), node.end(), padded.begin());
        // Bytes the instruction must not read as part of the node: what follows a 64-byte node,
        // and a box node's last 16 bytes.
        for (std::size_t k = dwords == 16 ? 16 : 28; k < 32; ++k) padded[k] = 0x7fc0'dead;
        words.insert(words.end(), padded.begin(), padded.begin() + std::ptrdiff_t(dwords));
        return starts.size() - 1;
    }

    // The pointer of node `n` with `kind`, counted from the first node.
    std::uint32_t pointer(std::size_t n, std::uint32_t kind) const {
        return ((4 * std::uint32_t(starts.at(n))) >> 3) | kind;
    }

    // What the instruction reads of node `n`.
    bvh::Words node(std::size_t n) const {
        bvh::Words w{};
        for (std::size_t k = 0; k < w.size(); ++k) {
            const std::size_t at = starts.at(n) + k;
            w[k] = at < words.size() ? words[at] : 0;
        }
        return w;
    }
};

// A small generator of repeatable numbers.
struct Numbers {
    std::uint64_t state;

    std::uint32_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;  // wrapping
        return std::uint32_t(state >> 33);
    }

    // A float in -1..1 in steps of 1/4096 (exactly representable).
    float unit() { return float(next() % 8193) / 4096.0f - 1.0f; }

    bvh::Vec3 vector(float scale) {
        // In order: x, y, z.
        const float x = unit() * scale;
        const float y = unit() * scale;
        const float z = unit() * scale;
        return {x, y, z};
    }
};

bvh::Vec3 vabs(bvh::Vec3 v) { return {std::fabs(v[0]), std::fabs(v[1]), std::fabs(v[2])}; }

// The scene and batches of cases over it.
std::pair<Scene, std::vector<std::vector<Case>>> scene() {
    Scene s;
    Numbers g{0x5eed};
    constexpr bvh::Box UNIT = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    auto moved = [](bvh::Box b, bvh::Vec3 by) {
        bvh::Box r;
        for (std::size_t k = 0; k < 6; ++k) r[k] = b[k] + by[k % 3];
        return r;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const bvh::Box off = {nan, nan, nan, nan, nan, nan};
    std::vector<std::pair<std::size_t, std::uint32_t>> box_nodes;
    // The boxes of the reference's own tests, in both widths.
    const bvh::Boxes lined = {
        moved(UNIT, {30.0f, 0.0f, 0.0f}),
        moved(UNIT, {10.0f, 0.0f, 0.0f}),
        moved(UNIT, {10.0f, 5.0f, 0.0f}),
        moved(UNIT, {20.0f, 0.0f, 0.0f}),
    };
    const bvh::Result4 ids = {0x105, 0x20c, 0x304, 0x400};
    box_nodes.emplace_back(s.add(nodes::box32(ids, lined), 32), 5);
    box_nodes.emplace_back(s.add(nodes::box16(ids, lined), 16), 4);
    // Boxes around the origin, one switched off, one child missing.
    const bvh::Boxes around = {
        UNIT,
        moved(UNIT, {0.5f, 0.5f, 0.5f}),
        off,
        bvh::Box{-4.0f, -0.25f, -0.25f, 4.0f, 0.25f, 0.25f},
    };
    const bvh::Result4 ids2 = {8, bvh::NONE, 24, 0x1fff'fffc};
    box_nodes.emplace_back(s.add(nodes::box32(ids2, around), 32), 5);
    box_nodes.emplace_back(s.add(nodes::box16(ids2, around), 16), 4);
    // A child that is not there with inverted infinite bounds (the title's builder leaves such
    // slots): every ray "enters" it at minus infinity.
    bvh::Boxes empty = lined;
    empty[1] = {inf, inf, inf, -inf, -inf, -inf};
    const bvh::Result4 ids3 = {0x105, bvh::NONE, 0x304, 0x400};
    box_nodes.emplace_back(s.add(nodes::box32(ids3, empty), 32), 5);
    // Random boxes.
    for (std::uint32_t n = 0; n < 6; ++n) {
        bvh::Boxes boxes;
        for (std::size_t b = 0; b < 4; ++b) {
            const bvh::Vec3 c = g.vector(8.0f);
            const bvh::Vec3 h = vabs(g.vector(2.0f));
            for (std::size_t k = 0; k < 6; ++k) boxes[b][k] = c[k % 3] + (k < 3 ? -h[k] : h[k - 3]);
        }
        bvh::Result4 rids;
        for (std::uint32_t k = 0; k < 4; ++k) rids[k] = 0x1000 * (n + 1) + 8 * k + 5;
        if (n % 2 == 0)
            box_nodes.emplace_back(s.add(nodes::box32(rids, boxes), 32), 5);
        else
            box_nodes.emplace_back(s.add(nodes::box16(rids, boxes), 16), 4);
    }
    // Triangles: one of the reference's tests, a pair as the title's builder stores it, all four
    // triangles of a full node with turned ids, random ones.
    std::vector<std::pair<std::size_t, std::uint32_t>> triangle_nodes;
    const bvh::Vec3 front[3] = {{0.0f, 2.0f, 5.0f}, {2.0f, -2.0f, 5.0f}, {-2.0f, -2.0f, 5.0f}};
    triangle_nodes.emplace_back(s.add(nodes::triangles(front, 3, 9), 16), 0);
    const bvh::Vec3 pair_v[4] = {front[0], front[1], front[2], {2.0f, 2.0f, 5.0f}};
    const std::size_t pair = s.add(nodes::triangles(pair_v, 4, 9 | 0x80 | (6 << 8)), 16);
    triangle_nodes.emplace_back(pair, 0);
    triangle_nodes.emplace_back(pair, 1);
    const bvh::Vec3 five[5] = {
        {-3.0f, -1.0f, 6.0f}, {-1.0f, 2.0f, 6.5f}, {1.0f, -1.0f, 6.0f}, {3.0f, 2.0f, 6.5f}, {5.0f, -1.0f, 7.0f},
    };
    // I and J per triangle: (1, 2), (2, 0), (0, 1), (1, 2).
    const std::size_t full = s.add(nodes::triangles(five, 5, 9 | (2 << 8) | (4 << 16) | (9u << 24)), 16);
    for (std::uint32_t k = 0; k < 4; ++k) triangle_nodes.emplace_back(full, k);
    for (int r = 0; r < 5; ++r) {
        const bvh::Vec3 c = g.vector(6.0f);
        bvh::Vec3 v[3];
        for (auto& vert : v) {
            const bvh::Vec3 d = g.vector(3.0f);
            vert = {c[0] + d[0], c[1] + d[1], c[2] + d[2]};
        }
        triangle_nodes.emplace_back(s.add(nodes::triangles(v, 3, 9), 16), 0);
    }
    std::vector<Case> cases;
    // Rays of the reference's tests against the lined-up boxes.
    struct R {
        bvh::Vec3 origin, direction;
        float extent;
    };
    const R reference_rays[] = {
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f},
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 19.5f},
        {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 19.0f},
        {{0.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 1000.0f},
        {{40.0f, 0.0f, 0.0f}, {-2.0f, 0.0f, 0.0f}, 1000.0f},
        {{0.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f},
        {{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f},
        {{0.0f, 0.5f, 0.25f}, {1.0f, -0.0f, 0.0f}, 1000.0f},
        {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 1000.0f},
    };
    for (std::size_t i = 0; i < 2; ++i) {
        const auto [n, kind] = box_nodes[i];
        for (const R& r : reference_rays) cases.push_back(Test{n, kind, Ray::make(r.origin, r.direction, r.extent)});
    }
    // Random rays at every box node, half of them aimed at a box.
    for (const auto& [n, kind] : box_nodes) {
        const bvh::Words words = s.node(n);
        const bvh::Boxes boxes = kind == 5 ? bvh::boxes32(words) : bvh::boxes16(words);
        for (std::size_t k = 0; k < 12; ++k) {
            const bvh::Vec3 origin = g.vector(12.0f);
            bvh::Vec3 direction;
            if (k % 2 == 0) {
                const bvh::Box b = boxes[(k / 2) % 4];
                if (std::isnan(b[0])) {
                    direction = g.vector(1.0f);
                } else {
                    for (std::size_t a = 0; a < 3; ++a)
                        direction[a] = (b[a] + b[3 + a]) / 2.0f - origin[a] + g.unit() / 2.0f;
                }
            } else {
                direction = g.vector(1.0f);
            }
            const float extent = k % 3 == 0 ? 6.0f : 100.0f;
            cases.push_back(Test{n, kind, Ray::make(origin, direction, extent)});
        }
    }
    // Rays at every triangle: aimed inside, at an edge, at a vertex, past it, from both sides,
    // along each axis.
    for (const auto& [n, kind] : triangle_nodes) {
        const bvh::Triangle p = bvh::triangle_vertices(s.node(n), kind);
        auto at = [&p](bvh::Vec3 w) {
            bvh::Vec3 r;
            for (std::size_t a = 0; a < 3; ++a) r[a] = w[0] * p[0][a] + w[1] * p[1][a] + w[2] * p[2][a];
            return r;
        };
        const bvh::Vec3 weights[] = {
            {0.5f, 0.25f, 0.25f}, {0.125f, 0.75f, 0.125f}, {0.5f, 0.5f, 0.0f},  {0.0f, 0.5f, 0.5f},
            {1.0f, 0.0f, 0.0f},   {0.75f, 0.75f, -0.5f},   {-0.25f, 0.5f, 0.75f}, {0.25f, 0.25f, 0.5f},
        };
        for (std::size_t k = 0; k < 8; ++k) {
            const bvh::Vec3 target = at(weights[k]);
            const bvh::Vec3 origin = k % 2 == 0 ? bvh::Vec3{0.0f, 0.0f, 0.0f} : g.vector(16.0f);
            bvh::Vec3 direction;
            for (std::size_t a = 0; a < 3; ++a) direction[a] = target[a] - origin[a];
            if (k == 7) {
                // The triangle lies behind.
                for (float& d : direction) d = -d;
            }
            cases.push_back(Test{n, kind, Ray::make(origin, direction, 100.0f)});
        }
        for (std::size_t axis = 0; axis < 3; ++axis) {
            for (float sign : {1.0f, -1.0f}) {
                const bvh::Vec3 target = at({0.25f, 0.5f, 0.25f});
                bvh::Vec3 origin = target;
                origin[axis] -= 8.0f * sign;
                bvh::Vec3 direction = {0.0f, 0.0f, 0.0f};
                direction[axis] = sign;
                cases.push_back(Test{n, kind, Ray::make(origin, direction, 100.0f)});
            }
        }
    }
    // Kinds the instruction does not read, and threads masked out.
    const Ray ray = Ray::make({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 100.0f);
    cases.push_back(Test{triangle_nodes[0].first, 6, ray});
    cases.push_back(Test{triangle_nodes[0].first, 7, ray});
    cases.push_back(std::nullopt);
    cases.push_back(Test{box_nodes[0].first, 5, Ray::make({0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, 1000.0f)});
    cases.push_back(std::nullopt);
    std::vector<std::vector<Case>> batches;
    for (std::size_t i = 0; i < cases.size(); i += THREADS)
        batches.emplace_back(cases.begin() + std::ptrdiff_t(i),
                             cases.begin() + std::ptrdiff_t(std::min<std::size_t>(i + THREADS, cases.size())));
    return {std::move(s), std::move(batches)};
}

// The buffer for one batch: the threads' records, then the nodes. Node pointers count from
// `base` before the first node, in bytes.
std::vector<std::uint32_t> buffer(const Scene& scene, const std::vector<Case>& batch, std::uint64_t base) {
    std::vector<std::uint32_t> data(NODES / 4, 0);
    for (std::size_t t = 0; t < batch.size(); ++t) {
        if (!batch[t]) continue;
        const Test& c = *batch[t];
        std::uint32_t* r = &data.at(t * (RECORD / 4));
        const std::uint64_t pointer = (base >> 3) + std::uint64_t(scene.pointer(c.n, c.kind));
        r[0] = std::uint32_t(pointer);
        r[1] = std::uint32_t(pointer >> 32);
        r[2] = std::bit_cast<std::uint32_t>(c.ray.extent);
        for (std::size_t a = 0; a < 3; ++a) {
            r[3 + a] = std::bit_cast<std::uint32_t>(c.ray.origin[a]);
            r[6 + a] = std::bit_cast<std::uint32_t>(c.ray.direction[a]);
            r[9 + a] = std::bit_cast<std::uint32_t>(c.ray.inverse[a]);
        }
        r[12] = 1;
    }
    data.insert(data.end(), scene.words.begin(), scene.words.end());
    return data;
}

bvh::Result4 expected(const Scene& scene, const Case& c, bool sort) {
    if (c) return bvh::intersect(scene.node(c->n), c->kind, c->ray, sort);
    return BEFORE;
}

// Sets the tests' title memory for this thread until destroyed.
struct Memory {
    explicit Memory(std::uint64_t base) {
        // Heap slot 0: the buffer of records and nodes.
        lower::test::set_memory(std::pair<std::uint32_t, std::uint64_t>{0, base});
    }
    ~Memory() { lower::test::set_memory(std::nullopt); }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
};

std::string words4(const std::uint32_t* w) {
    char b[160];
    std::snprintf(b, sizeof b, "[%x, %x, %x, %x] = [%g, %g, %g, %g]", w[0], w[1], w[2], w[3],
                  double(std::bit_cast<float>(w[0])), double(std::bit_cast<float>(w[1])),
                  double(std::bit_cast<float>(w[2])), double(std::bit_cast<float>(w[3])));
    return b;
}

// Runs every batch on the hardware adapter and WARP and compares with the reference. `pointers`
// is what node pointers count from, in bytes before the first node.
void run(bool wide, Descriptor descriptor, std::uint64_t pointers, bool sort) {
    auto [shader, res] = build(wide, descriptor);
    if (res.bvh.size() != 1) panic("res.bvh.len() is " + std::to_string(res.bvh.size()) + ", not 1");
    auto [scene_v, batches] = scene();
    const Scene& sc = scene_v;
    // The reference must see a bit of everything, or the comparison proves little.
    std::vector<const Case*> all;
    for (const auto& b : batches)
        for (const Case& c : b) all.push_back(&c);
    std::vector<bvh::Result4> results;
    for (const Case* c : all) results.push_back(expected(sc, *c, sort));
    auto boxes = [&](std::size_t hits) {
        std::size_t n = 0;
        for (std::size_t i = 0; i < all.size(); ++i) {
            const Case& c = *all[i];
            if (c && (c->kind == 4 || c->kind == 5) &&
                std::size_t(std::count_if(results[i].begin(), results[i].end(),
                                          [](std::uint32_t x) { return x != bvh::NONE; })) == hits)
                ++n;
        }
        return n;
    };
    auto triangles = [&](bool hit) {
        std::size_t n = 0;
        const std::uint32_t inf = std::bit_cast<std::uint32_t>(std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < all.size(); ++i) {
            const Case& c = *all[i];
            if (c && c->kind <= 3 && (results[i][0] != inf) == hit) ++n;
        }
        return n;
    };
    if (!(boxes(0) > 5 && boxes(1) > 5 && boxes(2) > 5 && boxes(3) > 0)) panic("box cases");
    if (!(triangles(true) > 30 && triangles(false) > 30)) panic("triangle cases");
    // Sorting must matter to some case, or the unsorted run proves nothing.
    if (!std::any_of(all.begin(), all.end(),
                     [&](const Case* c) { return expected(sc, *c, true) != expected(sc, *c, false); }))
        panic("no case tells sorted from unsorted");
    int ran = 0;
    const std::pair<const char*, bool> devices[] = {{"hardware", false}, {"warp", true}};
    for (const auto& [label, warp] : devices) {
        std::optional<Host> h;
        try {
            h.emplace(warp);
        } catch (const host::Error&) {
            std::fprintf(stderr, "no SM 6.6 %s device; skipped\n", label);
            continue;
        }
        const Host& host = *h;
        host::Pipeline pl;
        try {
            pl = host.compute_pipeline(shader);
        } catch (const host::Error& e) {
            panic(std::string(label) + ": compute_pipeline failed: " + e.what());
        }
        for (std::size_t b = 0; b < batches.size(); ++b) {
            const std::vector<Case>& batch = batches[b];
            const std::vector<std::uint32_t> data = buffer(sc, batch, pointers);
            const View views[] = {View::buffer(data, Stride::raw()), View::raw_uav(4 * THREADS)};
            Dispatch d;
            d.roots.emplace_back(0, Root::constants(std::vector<std::uint32_t>(5, 0)));
            d.views = views;
            d.groups = {1, 1, 1};
            std::vector<std::vector<std::uint32_t>> out;
            try {
                out = host.dispatch(pl.first.Get(), pl.second.Get(), d);
            } catch (const host::Error& e) {
                panic(std::string(label) + ": dispatch failed: " + e.what());
            }
            if (out.empty() || out[0].size() < 4 * THREADS) panic(std::string(label) + ": short output");
            for (std::size_t t = 0; t < THREADS; ++t) {
                // A thread past the batch: its record is zeros, so it is masked out.
                const bvh::Result4 want = t < batch.size() ? expected(sc, batch[t], sort) : BEFORE;
                const std::uint32_t* got = &out[0][4 * t];
                if (!std::equal(want.begin(), want.end(), got)) {
                    std::string c = "None";
                    if (t < batch.size() && batch[t]) {
                        const Test& x = *batch[t];
                        c = "Some(node " + std::to_string(x.n) + ", kind " + std::to_string(x.kind) + ")";
                    }
                    panic(std::string(label) + ", batch " + std::to_string(b) + " thread " + std::to_string(t) + ": " +
                          c + "\n got " + words4(got) + "\nwant " + words4(want.data()));
                }
            }
        }
        ++ran;
    }
    if (ran == 0) panic("no SM 6.6 device available");
}

void the_64_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    // Pointers count from the buffer's start.
    run(true, Descriptor::Table(), NODES, true);
    CHECK(build(true, Descriptor::Table()).second.bvh.at(0).second == BvhNodes::Resource());
}

void the_32_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    run(false, Descriptor::Table(), NODES, true);
}

// Where the tests' title memory puts the buffer: past 4 GiB, on 256 bytes.
constexpr std::uint64_t BASE = 0x12'3456'7800ull;

void the_64_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's way: one descriptor for all of memory, the structure's address folded into a
    // 64-bit pointer.
    const Memory memory(BASE);
    run(true, WHOLE, BASE + NODES, true);
}

void the_32_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's other way: the structure's address in the descriptor (in 256-byte units, 40
    // bits), a 32-bit pointer from there. Here with an address past 2^40 bytes, and without box
    // sorting.
    const std::uint64_t base = 0x312'3456'7800ull;
    const std::uint64_t structure = base + NODES;
    const Memory memory(base);
    // Rust's `as u32`: the low 32 bits.
    const std::uint32_t word0 = std::uint32_t((structure >> 8) & 0xFFFF'FFFFu);
    const std::uint32_t word1 = std::uint32_t(structure >> 40);
    run(false, Descriptor::Numbers(word0, word1), 0, false);
    // And sorted.
    run(false, Descriptor::Numbers(word0, word1 | bvh::BOX_SORT), 0, true);
}

void a_descriptor_of_numbers_is_title_memory_read_through_the_directory() {
    for (bool wide : {true, false}) {
        const NativeProgram native = make_native(program(wide, WHOLE));
        const ir::Program p = ir::build(native.code, native.wave64, native.vgprs);
        const structure::Structured st = structure::structure(p);
        const resource::Resources res = resource::track(p, native.layout, native.stage, &st);
        // The flags the title sets: sorted boxes, barycentrics, a ray-tracing structure. No
        // resource is taken for it: the buffer loads' and the store's descriptors are all there is.
        CHECK(res.bvh.size() == 1);
        if (!res.bvh.empty()) CHECK(res.bvh[0].second == BvhNodes::Memory(0x8000'0000u, 0x8900'03ffu));
        CHECK(res.descriptors.size() == 5);
        // Without the tests' stand-in the nodes are read as the title's programs will read them:
        // through the host's directory of title memory. The program lowers and validates, and
        // asks the host for the directory and for the list of pages it found no copy of.
        const auto [shader, unused] = build(wide, WHOLE);
        CHECK(std::find(shader.root_srvs.begin(), shader.root_srvs.end(), SLOT_MEMORY) != shader.root_srvs.end());
        CHECK(std::find(shader.root_uavs.begin(), shader.root_uavs.end(), SLOT_FAULTS) != shader.root_uavs.end());
    }
}

void a_structure_that_returns_triangle_ids_is_refused_by_name() {
    // Triangle return mode 0 (bit 24 of dword 3 clear): the last two result dwords would be a
    // triangle id and a hit status.
    std::vector<std::uint32_t> words = program(true, WHOLE);
    auto flags = std::find(words.begin(), words.end(), 0x8900'03FFu);
    if (flags == words.end()) panic("no flags word");
    *flags = 0x8800'03FF;
    const NativeProgram native = make_native(words);
    const ir::Program p = ir::build(native.code, native.wave64, native.vgprs);
    const structure::Structured st = structure::structure(p);
    const resource::Resources res = resource::track(p, native.layout, native.stage, &st);
    const io::Io io = io::raise(p, native);
    lower::Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    try {
        lower::lower(parts);
        std::fprintf(stderr, "lowered: Ok(())\n");
        CHECK(false);
    } catch (const ShaderError& e) {
        if (e.kind != ShaderError::Kind::Unsupported) std::fprintf(stderr, "%s\n", e.debug().c_str());
        CHECK(e.kind == ShaderError::Kind::Unsupported);
        CHECK(std::strcmp(e.reason, "a ray-tracing structure that returns triangle ids") == 0);
    }
}

template <class F>
void run_test(const char* name, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", name, e.what());
        CHECK(false);
    }
}

}  // namespace

int main() {
    run_test("the_64_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card",
             the_64_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card);
    run_test("the_32_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card",
             the_32_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card);
    run_test("the_64_bit_node_test_over_title_memory_matches_the_reference_on_the_card",
             the_64_bit_node_test_over_title_memory_matches_the_reference_on_the_card);
    run_test("the_32_bit_node_test_over_title_memory_matches_the_reference_on_the_card",
             the_32_bit_node_test_over_title_memory_matches_the_reference_on_the_card);
    run_test("a_descriptor_of_numbers_is_title_memory_read_through_the_directory",
             a_descriptor_of_numbers_is_title_memory_read_through_the_directory);
    run_test("a_structure_that_returns_triangle_ids_is_refused_by_name",
             a_structure_that_returns_triangle_ids_is_refused_by_name);
    return test_util::finish("intersect_test");
}
