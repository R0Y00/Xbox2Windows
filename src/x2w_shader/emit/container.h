#pragma once

// The PC container around emitted DXIL: feature flags (SFI0), the signatures (ISG1, OSG1), pipeline state
// validation (PSV0) and the program (DXIL), laid out as validator version 1.6 lays them out, plus the
// container hash that D3D12 checks.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace x2w::vs::shader {
struct Mesh;
}

namespace x2w::vs::container {

// One signature element: one row (D3D12 links stages by semantic).
struct SigElement {
    std::string name;
    std::uint32_t index = 0;
    // DXIL semantic kind (0 arbitrary, 1 vertex id, 3 position, 16 target...).
    std::uint8_t kind = 0;
    // DXIL component type (4 i32, 5 u32, 9 f32, 1 i1).
    std::uint8_t comp = 0;
    // DXIL interpolation mode (0 for none).
    std::uint8_t interp = 0;
    // Start row, if the element has a register.
    std::optional<std::uint32_t> row;
    std::uint8_t col = 0;
    std::uint8_t cols = 0;
    // Components read (inputs) or written (outputs), relative to col.
    std::uint8_t usage = 0;
    // 16-bit on the console (declared 32-bit here).
    bool half = false;

    std::uint32_t system_value() const;
    std::uint32_t sig_comp() const;
    std::uint8_t mask() const;
    bool operator==(const SigElement&) const = default;
};

// ISG1/OSG1: 32-byte rows, then the names.
std::vector<std::uint8_t> signature(std::span<const SigElement> elements, bool input);

// The most a mesh shader's outputs may take, in bytes (D3D12).
inline constexpr std::uint64_t MESH_OUTPUT_LIMIT = 32 * 1024;

// A mesh shader's output size as the DXIL validator measures it against MESH_OUTPUT_LIMIT: 4
// bytes per component of every vertex element and primitive element, times the maxima rounded up
// to multiples of 32.
std::uint64_t mesh_output_bytes(std::span<const SigElement> vertex, std::span<const SigElement> primitive,
                                std::uint32_t max_vertices, std::uint32_t max_primitives);

// A resource bound by register, as PSV0 lists it.
struct Binding {
    // PSV resource type: 1 sampler, 2 CBV, 4 raw SRV, 7 raw UAV.
    std::uint32_t kind = 0;
    std::uint32_t space = 0;
    std::uint32_t register_ = 0;
    // DXIL resource kind (13 cbuffer, 14 sampler).
    std::uint32_t dxil_kind = 0;
};

// The spans describing PSV0 borrow their input storage.
struct Psv {
    // PSV shader kind: 0 pixel, 1 vertex, 2 geometry, 5 compute.
    std::uint8_t stage = 0;
    // Vertex: writes SV_Position. Pixel: writes depth.
    bool flag = false;
    std::span<const SigElement> inputs;
    std::span<const SigElement> outputs;
    // Mesh: per-primitive outputs.
    std::span<const SigElement> prims;
    // Mesh: its state and the group-shared bytes it declares. (A pointer: Mesh is declared in
    // shader.h, which includes this header.)
    const shader::Mesh* mesh = nullptr;
    std::uint32_t mesh_lds_bytes = 0;
    // Geometry (triangles in, triangle strips out on stream 0): its maximum vertex count. flag:
    // writes SV_Position.
    std::optional<std::uint32_t> geometry;
    std::span<const Binding> bindings;
    // Compute: thread-group size and required wave size.
    std::array<std::uint32_t, 3> threads{};
    std::optional<std::uint32_t> wave_size;
};

// PSV0 with runtime info version 2 (validator 1.6).
std::vector<std::uint8_t> psv(const Psv& p);

// One part: its fourcc and data.
struct Part {
    std::array<std::uint8_t, 4> fourcc;
    std::span<const std::uint8_t> data;
};

// The container: parts in order, hashed.
std::vector<std::uint8_t> build(std::span<const Part> parts);

// The container hash D3D12 checks: MD5 whose last block carries the bit count first and
// 2 x bytes + 1 last instead of standard padding.
std::array<std::uint8_t, 16> hash(std::span<const std::uint8_t> data);

// The MD5 compression of one 64-byte block (for the test of the MD5 core).
void md5_block_for_test(std::array<std::uint32_t, 4>& state, const std::uint8_t* block);

}  // namespace x2w::vs::container
