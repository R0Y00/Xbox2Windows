// I/O raising: what a program's input registers, its interpolation and its exports mean.
//
// Pixel shaders get the VGPRs `SPI_PS_INPUT_ADDR` enables, in a fixed order (barycentrics, then
// position, front face...). `v_interp_p1` and `v_interp_p2` (and their f16 forms) evaluate an
// attribute from a pair of barycentrics; `v_interp_mov` reads one vertex's value. After the 32-bit
// export patch (NativeProgram::patched_code) exports carry plain 32-bit values: `mrt<n>` is render
// target n, `mrtz` depth, stencil and sample mask, `null` nothing. A lane whose EXEC bit is clear at
// the export that ends the program (`done`) is discarded: that export's guard after
// structurisation.
//
// Vertex shaders that run as the hardware VS get the vertex id in v0, the primitive id in v2 and the
// instance id in v3; the fetch shader loads channel k of each attribute into its VGPR + k.
// `exp pos0` is the position, `exp param<n>` the output with parameter n.
//
// Attributes and outputs are named by their row: a register of the container's signature part.
//
// Compute programs get the thread id in the group in v0-v2 and the group id in the SGPRs after the
// user data, as `COMPUTE_PGM_RSRC2` enables them.
//
// Mesh programs run as one NGG stage. The program computes its thread index from the wave index
// (s3 bits 24-27) and the lane (`v_mbcnt`), and its group id from v5 of the first lane
// (`v_readfirstlane`: the group times the group size). Lane n exports vertex n (`exp pos`,
// `exp param`) and primitive n (`exp prim`: 9-bit vertex indices at bits 0, 10, 20, each with an
// edge flag above it, bit 31 culls); `param` entries with bit 7 are per-primitive outputs. One lane
// sends the output counts (`s_sendmsg` GS_ALLOC_REQ, m0 = vertices | primitives << 12).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"

namespace x2w::vs {
namespace resource {
// (block, instruction). Also declared in resource.h.
using Site = std::pair<std::size_t, std::size_t>;
}  // namespace resource

namespace io {

using resource::Site;

enum class Barycentric : std::uint8_t {
    PerspectiveSample,
    PerspectiveCenter,
    PerspectiveCentroid,
    PerspectivePullModel,
    LinearSample,
    LinearCenter,
    LinearCentroid,
};

struct Input {
    enum class Kind : std::uint8_t {
        VertexId,
        PrimitiveId,
        InstanceId,
        // A vertex attribute the fetch shader loads (row, channel).
        Attribute,
        // `index` 0 is i, 1 is j (2 the pull model's w).
        Barycentric,
        // Pixel position channel (n: 0-3, x, y, z, w).
        Position,
        FrontFace,
        // Sample id and other bits.
        Ancillary,
        SampleCoverage,
        FixedPosition,
        LineStipple,
        // Compute: thread-group id component n (an SGPR).
        GroupId,
        // Compute: thread id within the group, component n (a VGPR).
        ThreadIdInGroup,
        // Compute: the thread-group info SGPR (wave index and count).
        GroupInfo,
        // Mesh: v5, the group id times the group size (read as uniform).
        MeshGroup,
        // Mesh: s3, the wave's index in the group in bits 24-27.
        MergedWaveInfo,
    };
    Kind kind = Kind::VertexId;
    std::uint8_t row = 0;      // Attribute
    std::uint8_t channel = 0;  // Attribute
    Barycentric barycentric = Barycentric::PerspectiveSample;  // Barycentric kind
    std::uint8_t index = 0;    // Barycentric
    std::uint8_t n = 0;        // Position, GroupId, ThreadIdInGroup

    static constexpr Input Simple(Kind k) { return {k, 0, 0, Barycentric::PerspectiveSample, 0, 0}; }
    static constexpr Input Attribute(std::uint8_t row, std::uint8_t channel)
    {
        return {Kind::Attribute, row, channel, Barycentric::PerspectiveSample, 0, 0};
    }
    static constexpr Input Bary(Barycentric kind, std::uint8_t index) { return {Kind::Barycentric, 0, 0, kind, index, 0}; }
    static constexpr Input Position(std::uint8_t c) { return {Kind::Position, 0, 0, Barycentric::PerspectiveSample, 0, c}; }
    static constexpr Input GroupId(std::uint8_t c) { return {Kind::GroupId, 0, 0, Barycentric::PerspectiveSample, 0, c}; }
    static constexpr Input ThreadIdInGroup(std::uint8_t c)
    {
        return {Kind::ThreadIdInGroup, 0, 0, Barycentric::PerspectiveSample, 0, c};
    }
    constexpr bool operator==(const Input&) const = default;
};

// An attribute channel of the input signature.
struct Attribute {
    native::Usage usage;
    std::uint8_t row = 0;
    std::uint8_t channel = 0;
    // The upper f16 half of the channel (packed fp16 inputs).
    bool high = false;
    bool operator==(const Attribute&) const = default;
};

struct Interp {
    enum class Kind : std::uint8_t {
        // This site's result (`v_interp_p2*`) is the attribute interpolated with `barycentric`;
        // `first` are the `v_interp_p1*` halves it completes (one per path where the compiler shares a
        // first half).
        Interpolated,
        // One vertex's value (`v_interp_mov_f32`), as encoded: 0 for P10, 1 for P20, 2 for P0 (the
        // provoking vertex).
        Vertex,
        // A half of an interpolation whose i or j is computed rather than loaded: each half
        // evaluates as the hardware does, from the attribute's vertex values (`v_interp_p1*`:
        // P0 + i P10, with P10 = P1 - P0; `v_interp_p2*`: its old dst + j P20, with P20 = P2 - P0).
        Weighted,
    };
    Kind kind = Kind::Weighted;
    Attribute attribute;
    Barycentric barycentric = Barycentric::PerspectiveSample;  // Interpolated
    std::vector<Site> first;                                    // Interpolated
    std::uint8_t vertex = 0;                                    // Vertex

    static Interp Interpolated(Attribute a, Barycentric b, std::vector<Site> first)
    {
        return {Kind::Interpolated, a, b, std::move(first), 0};
    }
    static Interp Vertex(Attribute a, std::uint8_t vertex)
    {
        return {Kind::Vertex, a, Barycentric::PerspectiveSample, {}, vertex};
    }
    static Interp Weighted(Attribute a) { return {Kind::Weighted, a, Barycentric::PerspectiveSample, {}, 0}; }
    bool operator==(const Interp&) const = default;
};

struct Target {
    enum class Kind : std::uint8_t {
        Color,
        // Depth, stencil, sample mask (channels as `DB_SHADER_CONTROL` enables).
        Depth,
        Null,
        Position,
        Param,
        // `param0` of a vertex shader without outputs: the hardware wants one parameter export
        // regardless.
        Unread,
        // The primitive export of a merged or mesh stage.
        Primitive,
        // A mesh program's per-primitive output (a row of `PSG1`).
        PrimitiveParam,
    };
    Kind kind = Kind::Null;
    std::uint8_t n = 0;    // Color, Position
    native::Usage usage;   // Param, PrimitiveParam
    std::uint8_t row = 0;  // Param, PrimitiveParam

    static constexpr Target Simple(Kind k) { return {k, 0, {}, 0}; }
    static constexpr Target Color(std::uint8_t n) { return {Kind::Color, n, {}, 0}; }
    static constexpr Target Position(std::uint8_t n) { return {Kind::Position, n, {}, 0}; }
    static constexpr Target Param(native::Usage u, std::uint8_t row) { return {Kind::Param, 0, u, row}; }
    static constexpr Target PrimitiveParam(native::Usage u, std::uint8_t row)
    {
        return {Kind::PrimitiveParam, 0, u, row};
    }
    bool operator==(const Target&) const = default;
};

struct Export {
    Site site;
    Target target;
    // Per channel, as the instruction enables them.
    std::array<std::optional<ir::Value>, 4> values{};
    // Two f16 values per dword.
    bool compressed = false;
    // The program's last export (for pixels: kills the lanes not in EXEC).
    bool done = false;
    bool operator==(const Export&) const = default;
};

struct Io {
    // Input values the hardware loads, with their meaning; other VGPRs read before written hold
    // undefined values.
    std::vector<std::pair<ir::Value, Input>> inputs;
    std::vector<std::pair<Site, Interp>> interps;
    std::vector<Export> exports;
    // Compute: how many SGPRs the hardware initializes (ComputeIo::initialized_sgprs).
    std::optional<std::uint8_t> initialized_sgprs;
    bool operator==(const Io&) const = default;
};

// Raises the I/O of `p`, built from `native`'s patched code. Throws ShaderError for stages whose
// stage data is not read and on constructs not covered yet.
Io raise(const ir::Program& p, const native::NativeProgram& native);

namespace detail {
// The pixel input VGPRs that `addr` enables, with their meaning.
std::vector<std::pair<std::uint8_t, Input>> ps_vgprs(std::uint32_t addr);
}  // namespace detail

}  // namespace io
}  // namespace x2w::vs
