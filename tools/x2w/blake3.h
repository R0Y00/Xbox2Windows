// BLAKE3 (https://github.com/BLAKE3-team/BLAKE3-specs), the plain hash mode with 32-byte output: a compact,
// portable, single-threaded implementation of the spec for the `x2w shaders` keys (A shader key uses the
// first 16 hex digits of the BLAKE3 of its container).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace x2w::tool {

using Blake3Hash = std::array<std::uint8_t, 32>;

// Incremental hashing: update() any number of times, then finalize().
class Blake3 {
public:
    Blake3();
    void update(std::span<const std::uint8_t> data);
    Blake3Hash finalize() const;

private:
    struct Chunk {
        std::array<std::uint32_t, 8> cv{};
        std::uint64_t counter = 0;
        std::array<std::uint8_t, 64> block{};
        std::size_t block_len = 0;
        std::uint8_t blocks_compressed = 0;
        std::size_t len() const { return 64 * std::size_t{blocks_compressed} + block_len; }
    };
    void chunk_update(std::span<const std::uint8_t> data);
    void push_cv(std::array<std::uint32_t, 8> cv, std::uint64_t total_chunks);

    Chunk chunk_;
    // Chaining values of completed subtrees (at most 54 for 2^64 bytes).
    std::vector<std::array<std::uint32_t, 8>> stack_;
};

Blake3Hash blake3(std::span<const std::uint8_t> data);
// Lowercase hexadecimal encoding of all 32 bytes.
std::string to_hex(const Blake3Hash& h);

}  // namespace x2w::tool
