// BLAKE3 hash mode, after the spec's reference implementation (reference_impl.rs): chunks of
// 1024 bytes in 64-byte blocks, a binary tree of parent nodes merged as chunks complete, the root
// node compressed with ROOT for the output.
#include "blake3.h"

#include <algorithm>

namespace x2w::tool {
namespace {

constexpr std::array<std::uint32_t, 8> IV = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
constexpr std::array<std::size_t, 16> MSG_PERMUTATION = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
constexpr std::uint32_t CHUNK_START = 1;
constexpr std::uint32_t CHUNK_END = 2;
constexpr std::uint32_t PARENT = 4;
constexpr std::uint32_t ROOT = 8;
constexpr std::size_t CHUNK_LEN = 1024;
constexpr std::size_t BLOCK_LEN = 64;

using Words16 = std::array<std::uint32_t, 16>;
using Cv = std::array<std::uint32_t, 8>;

constexpr std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void g(Words16& s, std::size_t a, std::size_t b, std::size_t c, std::size_t d, std::uint32_t mx, std::uint32_t my) {
    s[a] = s[a] + s[b] + mx;
    s[d] = rotr(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my;
    s[d] = rotr(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 7);
}

void rounds_one(Words16& s, const Words16& m) {
    g(s, 0, 4, 8, 12, m[0], m[1]);
    g(s, 1, 5, 9, 13, m[2], m[3]);
    g(s, 2, 6, 10, 14, m[4], m[5]);
    g(s, 3, 7, 11, 15, m[6], m[7]);
    g(s, 0, 5, 10, 15, m[8], m[9]);
    g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7, 8, 13, m[12], m[13]);
    g(s, 3, 4, 9, 14, m[14], m[15]);
}

Words16 compress(const Cv& cv, const Words16& block, std::uint64_t counter, std::uint32_t block_len,
                 std::uint32_t flags) {
    Words16 s = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7], IV[0], IV[1], IV[2], IV[3],
                 static_cast<std::uint32_t>(counter), static_cast<std::uint32_t>(counter >> 32), block_len, flags};
    Words16 m = block;
    for (int r = 0; r < 7; ++r) {
        rounds_one(s, m);
        if (r == 6) break;
        Words16 p{};
        for (std::size_t i = 0; i < 16; ++i) p[i] = m[MSG_PERMUTATION[i]];
        m = p;
    }
    for (std::size_t i = 0; i < 8; ++i) {
        s[i] ^= s[i + 8];
        s[i + 8] ^= cv[i];
    }
    return s;
}

Words16 words_of(const std::array<std::uint8_t, 64>& block) {
    Words16 w{};
    for (std::size_t i = 0; i < 16; ++i)
        w[i] = std::uint32_t{block[4 * i]} | std::uint32_t{block[4 * i + 1]} << 8 |
               std::uint32_t{block[4 * i + 2]} << 16 | std::uint32_t{block[4 * i + 3]} << 24;
    return w;
}

Cv first8(const Words16& w) {
    Cv cv{};
    std::copy_n(w.begin(), 8, cv.begin());
    return cv;
}

// A node not yet compressed: the root's output, or a chaining value.
struct Output {
    Cv input_cv;
    Words16 block;
    std::uint64_t counter;
    std::uint32_t block_len;
    std::uint32_t flags;
    Cv chaining_value() const { return first8(compress(input_cv, block, counter, block_len, flags)); }
};

Output parent_output(const Cv& left, const Cv& right) {
    Words16 block{};
    std::copy(left.begin(), left.end(), block.begin());
    std::copy(right.begin(), right.end(), block.begin() + 8);
    return Output{IV, block, 0, static_cast<std::uint32_t>(BLOCK_LEN), PARENT};
}

}  // namespace

Blake3::Blake3() { chunk_.cv = IV; }

void Blake3::chunk_update(std::span<const std::uint8_t> data) {
    while (!data.empty()) {
        // A full block is compressed only once more input follows: the last one is the output's.
        if (chunk_.block_len == BLOCK_LEN) {
            const std::uint32_t start = chunk_.blocks_compressed == 0 ? CHUNK_START : 0;
            chunk_.cv = first8(compress(chunk_.cv, words_of(chunk_.block), chunk_.counter,
                                        static_cast<std::uint32_t>(BLOCK_LEN), start));
            ++chunk_.blocks_compressed;
            chunk_.block.fill(0);
            chunk_.block_len = 0;
        }
        const std::size_t take = std::min(BLOCK_LEN - chunk_.block_len, data.size());
        std::copy_n(data.begin(), take, chunk_.block.begin() + static_cast<std::ptrdiff_t>(chunk_.block_len));
        chunk_.block_len += take;
        data = data.subspan(take);
    }
}

void Blake3::push_cv(Cv cv, std::uint64_t total_chunks) {
    // Each trailing zero bit of the chunk count completes a subtree.
    while ((total_chunks & 1) == 0) {
        cv = parent_output(stack_.back(), cv).chaining_value();
        stack_.pop_back();
        total_chunks >>= 1;
    }
    stack_.push_back(cv);
}

void Blake3::update(std::span<const std::uint8_t> data) {
    while (!data.empty()) {
        if (chunk_.len() == CHUNK_LEN) {
            const Output out{chunk_.cv, words_of(chunk_.block), chunk_.counter,
                             static_cast<std::uint32_t>(chunk_.block_len),
                             (chunk_.blocks_compressed == 0 ? CHUNK_START : 0) | CHUNK_END};
            const std::uint64_t total = chunk_.counter + 1;
            push_cv(out.chaining_value(), total);
            chunk_ = Chunk{};
            chunk_.cv = IV;
            chunk_.counter = total;
        }
        const std::size_t take = std::min(CHUNK_LEN - chunk_.len(), data.size());
        chunk_update(data.first(take));
        data = data.subspan(take);
    }
}

Blake3Hash Blake3::finalize() const {
    Output out{chunk_.cv, words_of(chunk_.block), chunk_.counter, static_cast<std::uint32_t>(chunk_.block_len),
               (chunk_.blocks_compressed == 0 ? CHUNK_START : 0) | CHUNK_END};
    for (std::size_t i = stack_.size(); i-- > 0;) out = parent_output(stack_[i], out.chaining_value());
    const Words16 w = compress(out.input_cv, out.block, 0, out.block_len, out.flags | ROOT);
    Blake3Hash h{};
    for (std::size_t i = 0; i < 8; ++i)
        for (std::size_t k = 0; k < 4; ++k) h[4 * i + k] = static_cast<std::uint8_t>(w[i] >> (8 * k));
    return h;
}

Blake3Hash blake3(std::span<const std::uint8_t> data) {
    Blake3 h;
    h.update(data);
    return h.finalize();
}

std::string to_hex(const Blake3Hash& h) {
    static const char digits[] = "0123456789abcdef";
    std::string s;
    s.reserve(64);
    for (std::uint8_t b : h) {
        s.push_back(digits[b >> 4]);
        s.push_back(digits[b & 15]);
    }
    return s;
}

}  // namespace x2w::tool
