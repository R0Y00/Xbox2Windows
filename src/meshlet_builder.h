#pragma once

#include <cstdint>
#include <vector>

namespace x2w::meshlet {

inline constexpr std::uint32_t kMaxMeshletVertices = 64;
inline constexpr std::uint32_t kMaxMeshletPrimitives = 126;
inline constexpr std::uint32_t kHardwareMaxMeshletVertices = 256;
inline constexpr std::uint32_t kHardwareMaxMeshletPrimitives = 256;

struct alignas(16) MeshletVertex {
    float position[3];
    float pad0;
    float normal[3];
    float pad1;
    float color[4];
};
static_assert(sizeof(MeshletVertex) == 48, "MeshletVertex must match HLSL StructuredBuffer stride (48 bytes)");

struct MeshletDescriptor {
    std::uint32_t vertex_count;
    std::uint32_t vertex_offset;
    std::uint32_t primitive_count;
    std::uint32_t primitive_offset;
};
static_assert(sizeof(MeshletDescriptor) == 16, "MeshletDescriptor must match HLSL StructuredBuffer stride (16 bytes)");

struct alignas(16) MeshletBounds {
    float center[3];
    float radius;
    float cone_apex[3];
    float cone_cutoff;
    float cone_axis[3];
    float pad0;
};
static_assert(sizeof(MeshletBounds) == 48, "MeshletBounds must match HLSL StructuredBuffer stride (48 bytes)");

struct MeshletBufferData {
    std::vector<MeshletVertex> vertices;
    std::vector<std::uint32_t> source_indices;
    std::vector<MeshletDescriptor> meshlets;
    std::vector<MeshletBounds> bounds;
    std::vector<std::uint32_t> unique_vertex_indices;
    std::vector<std::uint32_t> packed_primitives;
};

// Partitions an indexed triangle list into meshlets bounded by max_vertices and max_primitives,
// emitting per-meshlet bounding spheres and normal cones for amplification shader culling.
// max_vertices must be in [3, 256] (a triangle can need three vertex slots) and max_primitives in [1, 256];
// no meshlet exceeds either limit.
MeshletBufferData build_meshlets(
    const std::vector<MeshletVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    std::uint32_t max_vertices = kMaxMeshletVertices,
    std::uint32_t max_primitives = kMaxMeshletPrimitives);

// Synthetic 4-meshlet scene exercising visible, backface-culled, and frustum-culled meshlets.
MeshletBufferData build_culling_test_scene();

} // namespace x2w::meshlet
