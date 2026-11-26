// The DXBC container: a header, then a table of FourCC-tagged parts.
//
//   0x00  "DXBC"
//   0x04  digest[16]
//   0x14  version major u16, minor u16
//   0x18  total size u32
//   0x1C  part count u32
//   0x20  part offsets u32[count]
//   part: fourcc[4], size u32, data[size]
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "x2w_shader/error.h"

namespace x2w::vs::dxbc {

using Bytes = std::span<const std::uint8_t>;

struct FourCC {
    std::array<std::uint8_t, 4> b{};

    static const FourCC DXIL;
    // Native code another compiler made, in place of `DXIL`.
    static const FourCC ACCC;
    static const FourCC STAT;
    static const FourCC RTS0;
    static const FourCC PSV0;
    static const FourCC ISG1;
    static const FourCC OSG1;
    // Per-primitive outputs of a mesh shader.
    static const FourCC PSG1;
    static const FourCC SFI0;
    static const FourCC HASH;
    static const FourCC ILDB;
    static const FourCC ILDN;

    // Display and Debug: the four bytes, '.' for any that is not ASCII graphic.
    std::string to_string() const;
    bool operator==(const FourCC&) const = default;
    auto operator<=>(const FourCC&) const = default;
};

constexpr FourCC fourcc(const char (&s)[5]) {
    return FourCC{{std::uint8_t(s[0]), std::uint8_t(s[1]), std::uint8_t(s[2]), std::uint8_t(s[3])}};
}
inline constexpr FourCC FourCC::DXIL = fourcc("DXIL");
inline constexpr FourCC FourCC::ACCC = fourcc("ACCC");
inline constexpr FourCC FourCC::STAT = fourcc("STAT");
inline constexpr FourCC FourCC::RTS0 = fourcc("RTS0");
inline constexpr FourCC FourCC::PSV0 = fourcc("PSV0");
inline constexpr FourCC FourCC::ISG1 = fourcc("ISG1");
inline constexpr FourCC FourCC::OSG1 = fourcc("OSG1");
inline constexpr FourCC FourCC::PSG1 = fourcc("PSG1");
inline constexpr FourCC FourCC::SFI0 = fourcc("SFI0");
inline constexpr FourCC FourCC::HASH = fourcc("HASH");
inline constexpr FourCC FourCC::ILDB = fourcc("ILDB");
inline constexpr FourCC FourCC::ILDN = fourcc("ILDN");

struct Part {
    FourCC fourcc;
    // Offset of the part header within the container.
    std::size_t offset = 0;
    Bytes data;
};

// Borrows the bytes it was parsed from: keep them alive.
struct Container {
    std::array<std::uint8_t, 16> digest{};
    std::pair<std::uint16_t, std::uint16_t> version{};
    std::vector<Part> parts;

    // Throws ShaderError::not_dxbc / malformed.
    static Container parse(Bytes data);
    // The first part of that tag, or nullptr.
    const Part* part(FourCC fourcc) const;
};

// Little-endian reads; throw ShaderError::malformed("truncated") past the end (pub(crate)).
std::uint16_t u16_at(Bytes data, std::size_t at);
std::uint32_t u32_at(Bytes data, std::size_t at);
std::uint64_t u64_at(Bytes data, std::size_t at);

// Builds a container from parts for tests.
std::vector<std::uint8_t> build(const std::vector<std::pair<FourCC, Bytes>>& parts);

}  // namespace x2w::vs::dxbc
