// The DXIL program inside a container's `DXIL` (or `STAT`) part.
//
//   0x00  program version u32   (kind << 16 | major << 4 | minor)
//   0x04  size in dwords u32
//   0x08  "DXIL"                <- bitcode offset is relative to here
//   0x0C  dxil version u32
//   0x10  bitcode offset u32    (16 in standard PC DXIL)
//   0x14  bitcode size u32
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/error.h"

namespace x2w::vs::dxil {

using dxbc::Bytes;

// Values beyond Node retain their numeric stage identifier.
enum class ShaderKind : std::uint16_t {
    Pixel = 0,
    Vertex = 1,
    Geometry = 2,
    Hull = 3,
    Domain = 4,
    Compute = 5,
    Library = 6,
    RayGeneration = 7,
    Intersection = 8,
    AnyHit = 9,
    ClosestHit = 10,
    Miss = 11,
    Callable = 12,
    Mesh = 13,
    Amplification = 14,
    Node = 15,
};

inline bool is_unknown(ShaderKind k) { return static_cast<std::uint16_t>(k) > 15; }
ShaderKind shader_kind_from_raw(std::uint16_t v);
// Diagnostic spelling: "Pixel",..., "Unknown(17)".
std::string debug(ShaderKind k);

// Borrows the part it was parsed from.
struct DxilProgram {
    ShaderKind kind = ShaderKind::Pixel;
    std::pair<std::uint8_t, std::uint8_t> shader_model{};
    std::uint32_t dxil_version = 0;
    // LLVM bitcode (starts with `BC C0 DE`).
    Bytes bitcode;
    // Bytes between the bitcode header and the bitcode. Empty on PC.
    Bytes prefix;
    // Bytes after the bitcode, up to the program size. Empty on PC.
    Bytes trailer;

    // Throws ShaderError::malformed.
    static DxilProgram parse(Bytes part);
    // True for a program laid out exactly like PC DXIL.
    bool is_standard_layout() const;
    struct XboxExtras xbox_extras() const;
};

// Xbox additions recognised around the bitcode.
struct XboxExtras {
    // The prefix starts with an `XDAT` table.
    bool xdat = false;
    // The trailer contains an `xbop` blob (compiler options).
    bool xbop = false;
    bool operator==(const XboxExtras&) const = default;
};

// Builds a standard PC-layout program around `bitcode` (test helper).
std::vector<std::uint8_t> build_standard(std::uint16_t kind, std::pair<std::uint8_t, std::uint8_t> sm,
                                         std::uint32_t dxil_version, Bytes bitcode);

}  // namespace x2w::vs::dxil
