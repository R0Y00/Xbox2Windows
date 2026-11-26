// The console GPU's ray-tracing node test, `image_bvh_intersect_ray` and
// `image_bvh64_intersect_ray`, on the CPU: the node formats, and what the
// instruction returns for one ray and one node. The emitter lowers the
// instruction to the same arithmetic in the same order (emit/intersect), so
// its code can be compared against this dword for dword.
//
// BVH node pointers, box nodes, and triangle intersections.
//
// The instruction tests one ray against one 64-byte node of a bounding
// volume hierarchy and returns four dwords. Traversal (the stack, instance
// transforms, culling, the hit distance `numerator / denominator`) is the
// program's own code.
//
// Node pointers: the low three bits are the node's kind, the rest its byte
// offset in 8-byte units: the node is at `(pointer >> 3) << 6` past the base
// address of the descriptor (40 bits in dword 0 and the low byte of dword 1,
// in 256-byte units).
//
//   kind 0-3  triangle 0-3 of a triangle node
//   kind 4    box node, 16-bit floats (64 bytes)
//   kind 5    box node, 32-bit floats (128 bytes)
//   kind 6    instance: not read by the instruction
//   kind 7    procedural primitive: not read by the instruction
//
// Not known: how the console rounds, which of two boxes entered at the same
// distance it returns first, and what it returns for a ray exactly through an
// edge (here: a hit in both triangles of the edge) or for kinds 6 and 7
// (here: a triangle miss).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace x2w::vs::bvh {

// The result dword for a box the ray misses, or a child that is not there.
inline constexpr std::uint32_t NONE = 0xffff'ffffu;

// The node's vertices that make triangle 0-3, by the triangle's own vertex
// numbers 0-2.
inline constexpr std::array<std::array<std::uint32_t, 3>, 4> TRIANGLE_VERTICES{{
    {0, 1, 2}, {1, 3, 2}, {2, 3, 4}, {2, 4, 0}}};

// Descriptor dword 1: boxes are returned nearest first.
inline constexpr std::uint32_t BOX_SORT = 1u << 31;
// Descriptor dword 3: triangles return the numerators of I and J.
inline constexpr std::uint32_t RETURN_BARYCENTRICS = 1u << 24;
// Descriptor dword 3, bits 31:28: the descriptor's type, a ray-tracing
// structure.
inline constexpr std::uint32_t DESCRIPTOR_TYPE = 8;

using Vec3 = std::array<float, 3>;
using Words = std::array<std::uint32_t, 28>;
using Box = std::array<float, 6>;
using Boxes = std::array<Box, 4>;
using Result4 = std::array<std::uint32_t, 4>;
using Triangle = std::array<Vec3, 3>;

// The ray as the instruction takes it.
struct Ray {
    float extent = 0;
    Vec3 origin{};
    Vec3 direction{};
    // What the program computed as `1 / direction`.
    Vec3 inverse{};

    // A ray with the inverse direction a program computes by division.
    static Ray make(Vec3 origin, Vec3 direction, float extent);
    bool operator==(const Ray&) const = default;
};

// The kind of the node a pointer names.
std::uint32_t kind(std::uint64_t pointer);

// The byte address of the node `pointer` names, for a descriptor whose first
// two dwords are `word0` and `word1`.
std::uint64_t node_address(std::uint32_t word0, std::uint32_t word1, std::uint64_t pointer);

// What the instruction returns for the node at `words` (its first 28 dwords;
// a 64-byte node's dwords 16-27 are not read) named by a pointer of this
// `kind`, with box sorting as `sort` says and the triangle return mode set.
Result4 intersect(const Words& words, std::uint32_t kind, const Ray& ray, bool sort);

// A box node's four child pointers.
Result4 children(const Words& words);
// The four boxes of a 16-bit box node: minimum x y z, maximum x y z.
Boxes boxes16(const Words& words);
// The four boxes of a 32-bit box node.
Boxes boxes32(const Words& words);
// TRIANGLE_VERTICES of triangle `k` in four bits each, vertex 0 lowest.
std::uint32_t vertex_map(std::uint32_t k);
// The vertices of triangle `k` (0-3) of a triangle node, the first being the
// one neither barycentric weighs, then I's, then J's.
Triangle triangle_vertices(const Words& words, std::uint32_t k);
// The box test: child pointers of the boxes the ray enters, NONE for the
// others; sorted by entry distance when `sort`, in child order otherwise.
Result4 boxes(const Result4& children, const Boxes& boxes, const Ray& ray, bool sort);

// What a triangle the ray misses returns.
inline constexpr std::array<float, 4> MISS{std::numeric_limits<float>::infinity(), 1.0f, 0.0f, 0.0f};
// The triangle test (watertight): the hit distance's numerator, the
// denominator, and the numerators of the weights of vertices 1 and 2; MISS
// otherwise.
std::array<float, 4> triangle(const Triangle& p, const Ray& ray);
// A 16-bit float as a 32-bit one (exact).
float half_to_f32(std::uint16_t h);

// `values` as their bit patterns.
Result4 to_bits(const std::array<float, 4>& values);

// Test helpers shared with the emitter tests.
namespace testing {
// The smallest 16-bit float not below `x` (`up`) or not above it.
std::uint16_t f32_to_half(float x, bool up);
// A 32-bit box node.
Words box32(const Result4& children, const Boxes& boxes);
// A 16-bit box node holding `boxes` rounded outward.
Words box16(const Result4& children, const Boxes& boxes);
// A triangle node: up to five vertices, and the dword at byte 60.
Words triangles(const Vec3* vertices, std::size_t count, std::uint32_t id);
}  // namespace testing

}  // namespace x2w::vs::bvh
