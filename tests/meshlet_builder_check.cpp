#include "meshlet_builder.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using x2w::meshlet::MeshletBounds;
using x2w::meshlet::MeshletBufferData;
using x2w::meshlet::MeshletVertex;

void require(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "meshlet_builder_check failed: %s\n", what);
        throw std::runtime_error(what);
    }
}

MeshletVertex make_vertex(float x, float y, float z, float nz = -1.0f) {
    return MeshletVertex{{x, y, z}, 0.0f, {0.0f, 0.0f, nz}, 0.0f, {0.2f, 0.4f, 0.6f, 1.0f}};
}

bool is_backface_culled(const MeshletBounds& b, float cam_x, float cam_y, float cam_z) {
    if (b.cone_cutoff >= 1.5f) return false;
    const float dx = b.cone_apex[0] - cam_x;
    const float dy = b.cone_apex[1] - cam_y;
    const float dz = b.cone_apex[2] - cam_z;
    const float len_sq = dx * dx + dy * dy + dz * dz;
    if (len_sq <= 1e-12f) return false;
    const float inv_len = 1.0f / std::sqrt(len_sq);
    const float d = (dx * b.cone_axis[0] + dy * b.cone_axis[1] + dz * b.cone_axis[2]) * inv_len;
    return d >= b.cone_cutoff;
}

void test_single_triangle() {
    const std::vector<MeshletVertex> verts{
        make_vertex(-0.5f, -0.5f, 0.5f),
        make_vertex(0.0f, 0.5f, 0.5f),
        make_vertex(0.5f, -0.5f, 0.5f),
    };
    const std::vector<std::uint32_t> indices{0, 1, 2};
    const MeshletBufferData data = x2w::meshlet::build_meshlets(verts, indices);

    require(data.meshlets.size() == 1, "single triangle produces 1 meshlet");
    require(data.bounds.size() == 1, "single triangle produces 1 bounds entry");
    require(data.meshlets[0].vertex_count == 3, "single triangle vertex_count == 3");
    require(data.meshlets[0].primitive_count == 1, "single triangle primitive_count == 1");
    require(data.unique_vertex_indices.size() == 3, "unique_vertex_indices size == 3");
    require(data.packed_primitives.size() == 1, "packed_primitives size == 1");
    require(data.packed_primitives[0] == 0x00020100u, "packed primitive indices match (0,1,2)");
    require(!is_backface_culled(data.bounds[0], 0.0f, 0.0f, -2.0f), "CW triangle is front-facing from z=-2");
    require(is_backface_culled(data.bounds[0], 0.0f, 0.0f, 2.0f), "CW triangle is back-facing from z=+2");
}

void test_grid_partitioning() {
    // 10x10 grid = 121 vertices, 200 triangles: exceeds both kMaxMeshletVertices (64) and kMaxMeshletPrimitives (126).
    constexpr int grid = 10;
    std::vector<MeshletVertex> verts;
    std::vector<std::uint32_t> indices;
    for (int y = 0; y <= grid; ++y) {
        for (int x = 0; x <= grid; ++x) {
            verts.push_back(make_vertex(static_cast<float>(x), static_cast<float>(y), 0.0f));
        }
    }
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            const std::uint32_t v0 = static_cast<std::uint32_t>(y * (grid + 1) + x);
            const std::uint32_t v1 = v0 + (grid + 1);
            const std::uint32_t v2 = v1 + 1;
            const std::uint32_t v3 = v0 + 1;
            indices.insert(indices.end(), {v0, v1, v2, v0, v2, v3});
        }
    }

    const MeshletBufferData data = x2w::meshlet::build_meshlets(verts, indices);
    require(data.meshlets.size() >= 2, "grid splits into multiple meshlets");
    require(data.meshlets.size() == data.bounds.size(), "meshlets and bounds counts match");

    std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> reconstructed;
    for (std::size_t m = 0; m < data.meshlets.size(); ++m) {
        const auto& d = data.meshlets[m];
        const auto& b = data.bounds[m];
        require(d.vertex_count > 0 && d.vertex_count <= x2w::meshlet::kMaxMeshletVertices,
                "meshlet vertex count within limit");
        require(d.primitive_count > 0 && d.primitive_count <= x2w::meshlet::kMaxMeshletPrimitives,
                "meshlet primitive count within limit");
        require(d.vertex_offset + d.vertex_count <= data.unique_vertex_indices.size(),
                "meshlet vertex range in bounds");
        require(d.primitive_offset + d.primitive_count <= data.packed_primitives.size(),
                "meshlet primitive range in bounds");

        std::set<std::uint32_t> seen_vertices;
        for (std::uint32_t v = 0; v < d.vertex_count; ++v) {
            const std::uint32_t global_idx = data.unique_vertex_indices[d.vertex_offset + v];
            require(global_idx < verts.size(), "unique vertex index in bounds");
            require(seen_vertices.insert(global_idx).second, "meshlet vertices are deduplicated");
            const float dx = verts[global_idx].position[0] - b.center[0];
            const float dy = verts[global_idx].position[1] - b.center[1];
            const float dz = verts[global_idx].position[2] - b.center[2];
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            require(dist <= b.radius + 1e-4f, "bounding sphere encloses all meshlet vertices");
        }

        for (std::uint32_t p = 0; p < d.primitive_count; ++p) {
            const std::uint32_t packed = data.packed_primitives[d.primitive_offset + p];
            const std::uint32_t l0 = packed & 0xFFu;
            const std::uint32_t l1 = (packed >> 8) & 0xFFu;
            const std::uint32_t l2 = (packed >> 16) & 0xFFu;
            require(l0 < d.vertex_count && l1 < d.vertex_count && l2 < d.vertex_count,
                    "packed primitive local indices in bounds");
            reconstructed.emplace_back(
                data.unique_vertex_indices[d.vertex_offset + l0],
                data.unique_vertex_indices[d.vertex_offset + l1],
                data.unique_vertex_indices[d.vertex_offset + l2]);
        }
    }

    require(reconstructed.size() == indices.size() / 3, "all triangles preserved");
    for (std::size_t t = 0; t < reconstructed.size(); ++t) {
        const auto expected = std::make_tuple(indices[t * 3 + 0], indices[t * 3 + 1], indices[t * 3 + 2]);
        require(reconstructed[t] == expected, "triangle order and indices match input");
    }
}

void test_double_sided_disables_culling() {
    const std::vector<MeshletVertex> verts{
        make_vertex(-0.5f, -0.5f, 0.5f),
        make_vertex(-0.5f, 0.5f, 0.5f),
        make_vertex(0.5f, 0.5f, 0.5f),
    };
    const std::vector<std::uint32_t> indices{0, 1, 2, 0, 2, 1};
    const MeshletBufferData data = x2w::meshlet::build_meshlets(verts, indices);
    require(data.bounds.size() == 1, "double-sided triangles in 1 meshlet");
    require(data.bounds[0].cone_cutoff > 1.0f, "opposing normals disable normal cone culling");
    require(!is_backface_culled(data.bounds[0], 0.0f, 0.0f, -2.0f), "double-sided meshlet not culled from front");
    require(!is_backface_culled(data.bounds[0], 0.0f, 0.0f, 2.0f), "double-sided meshlet not culled from back");
}

void test_culling_scene() {
    const MeshletBufferData scene = x2w::meshlet::build_culling_test_scene();
    require(scene.meshlets.size() == 4, "culling test scene has 4 meshlets");
    require(!is_backface_culled(scene.bounds[0], 0.0f, 0.0f, -2.0f), "meshlet 0 is front-facing");
    require(!is_backface_culled(scene.bounds[1], 0.0f, 0.0f, -2.0f), "meshlet 1 is front-facing");
    require(is_backface_culled(scene.bounds[2], 0.0f, 0.0f, -2.0f), "meshlet 2 is back-facing");
    require(!is_backface_culled(scene.bounds[3], 0.0f, 0.0f, -2.0f), "meshlet 3 is front-facing");
}

// Two triangles whose normals are tilted by +/- half_angle around Y, so the cone axis is +Z and
// min_dot == cos(half_angle). Appends 6 vertices at x offset `x0`.
std::vector<MeshletVertex> tilted_pair(float half_angle) {
    const float c = std::cos(half_angle);
    const float s = std::sin(half_angle);
    std::vector<MeshletVertex> v;
    // normal (s,0,c): tangent (c,0,-s) x (0,1,0)
    v.push_back(make_vertex(0.0f, 0.0f, 0.0f));
    v.push_back(make_vertex(c, 0.0f, -s));
    v.push_back(make_vertex(0.0f, 1.0f, 0.0f));
    // normal (-s,0,c): tangent (c,0,s) x (0,1,0)
    v.push_back(make_vertex(5.0f, 0.0f, 0.0f));
    v.push_back(make_vertex(5.0f + c, 0.0f, s));
    v.push_back(make_vertex(5.0f, 1.0f, 0.0f));
    return v;
}

void test_cone_width_threshold() {
    constexpr float kDeg = 3.14159265f / 180.0f;
    const std::vector<std::uint32_t> indices{0, 1, 2, 3, 4, 5};

    // Moderate cone (min_dot = cos 60 = 0.5): kept, cutoff = sin(60 deg).
    const MeshletBufferData moderate = x2w::meshlet::build_meshlets(tilted_pair(60.0f * kDeg), indices);
    require(moderate.bounds.size() == 1, "moderate cone is one meshlet");
    require(std::fabs(moderate.bounds[0].cone_cutoff - std::sin(60.0f * kDeg)) < 1e-3f,
            "moderate cone keeps cutoff = sin(half angle)");
    require(std::fabs(moderate.bounds[0].cone_axis[2] - 1.0f) < 1e-3f, "moderate cone axis is +Z");

    // Wide cone (min_dot = cos 87 = 0.052, inside (-0.1, 0.1]): too wide to cull, culling disabled.
    const MeshletBufferData wide = x2w::meshlet::build_meshlets(tilted_pair(87.0f * kDeg), indices);
    require(wide.bounds.size() == 1, "wide cone is one meshlet");
    require(wide.bounds[0].cone_cutoff > 1.0f, "cone wider than the threshold disables culling");

    // Just inside the accepted side (min_dot = cos 80 = 0.17 > 0.1): still culls.
    const MeshletBufferData edge = x2w::meshlet::build_meshlets(tilted_pair(80.0f * kDeg), indices);
    require(edge.bounds[0].cone_cutoff < 1.0f, "cone just under the threshold still culls");
}

void test_degenerate_triangle_ignored_for_cone() {
    // One real +Z triangle plus a sliver (cross product ~1e-8, squared ~1e-16) wound the other way.
    // The sliver must not enter the cone; otherwise the opposing normals cancel and culling is lost.
    const float e = 1e-4f;
    const std::vector<MeshletVertex> verts{
        make_vertex(0.0f, 0.0f, 0.0f), make_vertex(1.0f, 0.0f, 0.0f), make_vertex(0.0f, 1.0f, 0.0f),
        make_vertex(5.0f, 0.0f, 0.0f), make_vertex(5.0f, e, 0.0f),     make_vertex(5.0f + e, 0.0f, 0.0f),
    };
    const MeshletBufferData data = x2w::meshlet::build_meshlets(verts, {0, 1, 2, 3, 4, 5});
    require(data.bounds.size() == 1, "sliver case is one meshlet");
    require(data.bounds[0].cone_cutoff < 0.01f, "sliver triangle is ignored by the cone");
    require(data.bounds[0].cone_axis[2] > 0.99f, "cone axis follows the real triangle");
}

void require_within_limits(const MeshletBufferData& data, std::uint32_t max_vertices, std::uint32_t max_primitives,
                           std::size_t triangles) {
    std::size_t total = 0;
    for (const auto& d : data.meshlets) {
        require(d.vertex_count > 0 && d.vertex_count <= max_vertices, "meshlet vertex count within requested limit");
        require(d.primitive_count > 0 && d.primitive_count <= max_primitives,
                "meshlet primitive count within requested limit");
        total += d.primitive_count;
    }
    require(total == triangles, "all triangles preserved under small limits");
}

void test_small_limits() {
    const int grid = 6;
    std::vector<MeshletVertex> verts;
    for (int y = 0; y <= grid; ++y)
        for (int x = 0; x <= grid; ++x) verts.push_back(make_vertex(float(x), float(y), 0.0f));
    std::vector<std::uint32_t> indices;
    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            const std::uint32_t v0 = static_cast<std::uint32_t>(y * (grid + 1) + x);
            const std::uint32_t v1 = v0 + (grid + 1);
            indices.insert(indices.end(), {v0, v1, v1 + 1, v0, v1 + 1, v0 + 1});
        }
    }
    const std::size_t tris = indices.size() / 3;
    require_within_limits(x2w::meshlet::build_meshlets(verts, indices, 3, 126), 3, 126, tris);
    require_within_limits(x2w::meshlet::build_meshlets(verts, indices, 64, 1), 64, 1, tris);
    require_within_limits(x2w::meshlet::build_meshlets(verts, indices, 4, 2), 4, 2, tris);
    require_within_limits(x2w::meshlet::build_meshlets(verts, indices, 256, 256), 256, 256, tris);
    require_within_limits(x2w::meshlet::build_meshlets(verts, indices), x2w::meshlet::kMaxMeshletVertices,
                          x2w::meshlet::kMaxMeshletPrimitives, tris);

    // Triangles that repeat a vertex need fewer slots but must still respect the limit.
    const std::vector<std::uint32_t> degenerate{0, 0, 1, 2, 2, 2, 3, 4, 5, 1, 1, 1};
    require_within_limits(x2w::meshlet::build_meshlets(verts, degenerate, 3, 126), 3, 126, degenerate.size() / 3);
}

void test_negative_controls() {
    const std::vector<MeshletVertex> verts{make_vertex(0, 0, 0), make_vertex(0, 1, 0), make_vertex(1, 0, 0)};
    auto expect_throw = [](auto&& fn, const char* what) {
        bool threw = false;
        try {
            fn();
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, what);
    };
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1}); }, "non-multiple-of-3 indices rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 3}); }, "out-of-range vertex index rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 2}, 0, 10); }, "max_vertices=0 rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 2}, 1, 10); }, "max_vertices=1 rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 2}, 2, 10); }, "max_vertices=2 rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 2}, 257, 10); }, "max_vertices>256 rejected");
    expect_throw([&] { x2w::meshlet::build_meshlets(verts, {0, 1, 2}, 64, 257); }, "max_primitives>256 rejected");
}

} // namespace

int main() {
    try {
        test_single_triangle();
        test_grid_partitioning();
        test_double_sided_disables_culling();
        test_culling_scene();
        test_cone_width_threshold();
        test_degenerate_triangle_ignored_for_cone();
        test_small_limits();
        test_negative_controls();
        std::puts("meshlet_builder_check: PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "meshlet_builder_check error: %s\n", e.what());
        return 1;
    }
}
