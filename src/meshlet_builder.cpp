#include "meshlet_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace x2w::meshlet {

namespace {

struct Vec3 {
    float x;
    float y;
    float z;
};

Vec3 to_vec3(const float v[3]) {
    return {v[0], v[1], v[2]};
}

Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 operator*(const Vec3& a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

float length_sq(const Vec3& v) {
    return dot(v, v);
}

bool normalize(const Vec3& v, Vec3& out) {
    const float len_sq = length_sq(v);
    if (len_sq <= 1e-12f) return false;
    const float inv_len = 1.0f / std::sqrt(len_sq);
    out = v * inv_len;
    return true;
}

std::uint32_t pack_primitive(std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
    return (i0 & 0xFFu) | ((i1 & 0xFFu) << 8) | ((i2 & 0xFFu) << 16);
}

MeshletBounds compute_bounds(
    const std::vector<MeshletVertex>& vertices,
    const std::uint32_t* unique_vertices,
    std::uint32_t vertex_count,
    const std::uint32_t* packed_primitives,
    std::uint32_t primitive_count) {
    MeshletBounds result{};
    result.cone_cutoff = 2.0f; // > 1.0 disables backface cone culling by default.

    if (vertex_count == 0) return result;

    Vec3 min_p = to_vec3(vertices[unique_vertices[0]].position);
    Vec3 max_p = min_p;
    for (std::uint32_t i = 1; i < vertex_count; ++i) {
        const Vec3 p = to_vec3(vertices[unique_vertices[i]].position);
        min_p.x = std::min(min_p.x, p.x);
        min_p.y = std::min(min_p.y, p.y);
        min_p.z = std::min(min_p.z, p.z);
        max_p.x = std::max(max_p.x, p.x);
        max_p.y = std::max(max_p.y, p.y);
        max_p.z = std::max(max_p.z, p.z);
    }

    const Vec3 center = (min_p + max_p) * 0.5f;
    float max_dist_sq = 0.0f;
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const Vec3 p = to_vec3(vertices[unique_vertices[i]].position);
        max_dist_sq = std::max(max_dist_sq, length_sq(p - center));
    }

    result.center[0] = center.x;
    result.center[1] = center.y;
    result.center[2] = center.z;
    result.radius = std::sqrt(max_dist_sq);

    struct TrianglePlane {
        Vec3 p0;
        Vec3 normal;
    };
    std::vector<TrianglePlane> planes;
    planes.reserve(primitive_count);

    Vec3 normal_sum{0.0f, 0.0f, 0.0f};
    for (std::uint32_t p = 0; p < primitive_count; ++p) {
        const std::uint32_t packed = packed_primitives[p];
        const std::uint32_t i0 = unique_vertices[packed & 0xFFu];
        const std::uint32_t i1 = unique_vertices[(packed >> 8) & 0xFFu];
        const std::uint32_t i2 = unique_vertices[(packed >> 16) & 0xFFu];
        const Vec3 v0 = to_vec3(vertices[i0].position);
        const Vec3 v1 = to_vec3(vertices[i1].position);
        const Vec3 v2 = to_vec3(vertices[i2].position);
        Vec3 n{};
        if (!normalize(cross(v1 - v0, v2 - v0), n)) continue;
        normal_sum = normal_sum + n;
        planes.push_back({v0, n});
    }

    Vec3 cone_axis{};
    if (planes.empty() || !normalize(normal_sum, cone_axis)) {
        result.cone_apex[0] = center.x;
        result.cone_apex[1] = center.y;
        result.cone_apex[2] = center.z;
        return result;
    }

    float min_dot = 1.0f;
    for (const auto& plane : planes) {
        min_dot = std::min(min_dot, dot(plane.normal, cone_axis));
    }

    result.cone_axis[0] = cone_axis.x;
    result.cone_axis[1] = cone_axis.y;
    result.cone_axis[2] = cone_axis.z;

    if (min_dot <= 0.1f) {
        result.cone_apex[0] = center.x;
        result.cone_apex[1] = center.y;
        result.cone_apex[2] = center.z;
        result.cone_cutoff = 2.0f;
        return result;
    }

    float max_t = 0.0f;
    for (const auto& plane : planes) {
        const float denom = dot(cone_axis, plane.normal);
        max_t = std::max(max_t, dot(center - plane.p0, plane.normal) / denom);
    }

    const Vec3 cone_apex = center - cone_axis * max_t;
    result.cone_apex[0] = cone_apex.x;
    result.cone_apex[1] = cone_apex.y;
    result.cone_apex[2] = cone_apex.z;
    result.cone_cutoff = std::sqrt(std::max(0.0f, 1.0f - min_dot * min_dot));
    return result;
}

void append_quad(
    std::vector<MeshletVertex>& vertices,
    std::vector<std::uint32_t>& indices,
    float x0, float y0,
    float x1, float y1,
    float z,
    bool front_facing_cw,
    const std::array<float, 4>& color) {
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    const float nz = front_facing_cw ? -1.0f : 1.0f;
    const auto make_vertex = [&](float x, float y) -> MeshletVertex {
        return MeshletVertex{
            {x, y, z}, 0.0f,
            {0.0f, 0.0f, nz}, 0.0f,
            {color[0], color[1], color[2], color[3]},
        };
    };
    vertices.push_back(make_vertex(x0, y0)); // 0: bottom-left
    vertices.push_back(make_vertex(x0, y1)); // 1: top-left
    vertices.push_back(make_vertex(x1, y1)); // 2: top-right
    vertices.push_back(make_vertex(x1, y0)); // 3: bottom-right

    if (front_facing_cw) {
        indices.insert(indices.end(), {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});
    } else {
        indices.insert(indices.end(), {base + 0, base + 2, base + 1, base + 0, base + 3, base + 2});
    }
}

} // namespace

MeshletBufferData build_meshlets(
    const std::vector<MeshletVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    std::uint32_t max_vertices,
    std::uint32_t max_primitives) {
    if (indices.size() % 3 != 0) {
        throw std::invalid_argument("Meshlet index buffer length must be a multiple of 3");
    }
    if (max_vertices < 3 || max_vertices > kHardwareMaxMeshletVertices) {
        throw std::invalid_argument("Meshlet max_vertices must be in [3, 256]");
    }
    if (max_primitives == 0 || max_primitives > kHardwareMaxMeshletPrimitives) {
        throw std::invalid_argument("Meshlet max_primitives must be in [1, 256]");
    }
    for (const std::uint32_t idx : indices) {
        if (idx >= vertices.size()) {
            throw std::invalid_argument("Meshlet index exceeds vertex buffer bounds");
        }
    }

    MeshletBufferData out;
    out.vertices = vertices;
    out.source_indices = indices;

    if (indices.empty()) return out;

    std::unordered_map<std::uint32_t, std::uint32_t> local_map;
    local_map.reserve(max_vertices);
    std::uint32_t current_vertex_offset = 0;
    std::uint32_t current_primitive_offset = 0;
    std::uint32_t current_vertex_count = 0;
    std::uint32_t current_primitive_count = 0;

    auto flush_meshlet = [&]() {
        if (current_primitive_count == 0) return;
        MeshletDescriptor desc{
            current_vertex_count,
            current_vertex_offset,
            current_primitive_count,
            current_primitive_offset,
        };
        out.meshlets.push_back(desc);
        out.bounds.push_back(compute_bounds(
            out.vertices,
            out.unique_vertex_indices.data() + current_vertex_offset,
            current_vertex_count,
            out.packed_primitives.data() + current_primitive_offset,
            current_primitive_count));
        current_vertex_offset = static_cast<std::uint32_t>(out.unique_vertex_indices.size());
        current_primitive_offset = static_cast<std::uint32_t>(out.packed_primitives.size());
        current_vertex_count = 0;
        current_primitive_count = 0;
        local_map.clear();
    };

    const std::size_t tri_count = indices.size() / 3;
    for (std::size_t t = 0; t < tri_count; ++t) {
        const std::uint32_t tri[3]{
            indices[t * 3 + 0],
            indices[t * 3 + 1],
            indices[t * 3 + 2],
        };

        std::uint32_t new_vertices = 0;
        for (int k = 0; k < 3; ++k) {
            bool seen_in_tri = false;
            for (int prev = 0; prev < k; ++prev) {
                if (tri[prev] == tri[k]) {
                    seen_in_tri = true;
                    break;
                }
            }
            if (!seen_in_tri && local_map.find(tri[k]) == local_map.end()) {
                ++new_vertices;
            }
        }

        if (current_primitive_count > 0 &&
            (current_primitive_count + 1 > max_primitives ||
             current_vertex_count + new_vertices > max_vertices)) {
            flush_meshlet();
        }

        std::uint32_t local_tri[3]{};
        for (int k = 0; k < 3; ++k) {
            auto [it, inserted] = local_map.emplace(tri[k], current_vertex_count);
            if (inserted) {
                out.unique_vertex_indices.push_back(tri[k]);
                ++current_vertex_count;
            }
            local_tri[k] = it->second;
        }
        out.packed_primitives.push_back(pack_primitive(local_tri[0], local_tri[1], local_tri[2]));
        ++current_primitive_count;
    }

    flush_meshlet();
    return out;
}

MeshletBufferData build_culling_test_scene() {
    std::vector<MeshletVertex> vertices;
    std::vector<std::uint32_t> indices;
    vertices.reserve(16);
    indices.reserve(24);

    // Meshlet 0: visible green quad on the left.
    append_quad(vertices, indices, -0.8f, -0.7f, -0.1f, 0.7f, 0.5f, true, {0.0f, 1.0f, 0.0f, 1.0f});
    // Meshlet 1: visible blue quad on the right.
    append_quad(vertices, indices, 0.1f, -0.7f, 0.8f, 0.7f, 0.5f, true, {0.0f, 0.0f, 1.0f, 1.0f});
    // Meshlet 2: back-facing red quad in the center (culled by AS normal cone and VS CULL_MODE_BACK).
    append_quad(vertices, indices, -0.4f, -0.4f, 0.4f, 0.4f, 0.25f, false, {1.0f, 0.0f, 0.0f, 1.0f});
    // Meshlet 3: front-facing yellow quad outside the right frustum plane (culled by AS frustum test).
    append_quad(vertices, indices, 2.5f, -0.5f, 3.5f, 0.5f, 0.5f, true, {1.0f, 1.0f, 0.0f, 1.0f});

    return build_meshlets(vertices, indices, kMaxMeshletVertices, 2);
}

} // namespace x2w::meshlet
