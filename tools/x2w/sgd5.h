// `SGD5`: the engine's shader file. Each record holds one shader container, deflate-packed. The header
// tables are not decoded yet, so records are found by their marker and cross-checked by their sizes:
//
//   u8 name length (with NUL), name, NUL, u8 kind,
//   u32 n, u32 0x000DDCE7, u32 n, u32 unpacked size, raw deflate (n - 12 bytes)
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace x2w::tool::sgd5 {

inline constexpr std::uint8_t MAGIC[4] = {'S', 'G', 'D', '5'};

// Borrows the file it was found in.
struct Record {
    // Offset of the marker in the file.
    std::size_t offset = 0;
    // Like `PS_<name>_<hash>_Wrapped`; nullopt if the bytes before the record do not read as one.
    std::optional<std::string_view> name;
    // Record type, meaning unknown (0x32 most shaders, 0x41 mesh shaders and some material
    // families, 0x3c compute, 0x40, 0x42).
    std::uint8_t kind = 0;
    std::span<const std::uint8_t> packed;
    std::size_t unpacked_size = 0;

    // The container, or nullopt if the stream is corrupt (does not end, or does not inflate to
    // exactly the stated size).
    std::optional<std::vector<std::uint8_t>> unpack() const;
};

// An SGD5 file, or a file holding one (an archive: records are found by their marker, so no index
// is needed).
bool is_sgd5(std::span<const std::uint8_t> file);

// Every record in an SGD5 file, in file order.
std::vector<Record> records(std::span<const std::uint8_t> file);

// Offsets of every occurrence of `needle` (non-overlapping, as memchr::memmem::find_iter).
std::vector<std::size_t> find_all(std::span<const std::uint8_t> hay, std::span<const std::uint8_t> needle);

}  // namespace x2w::tool::sgd5
