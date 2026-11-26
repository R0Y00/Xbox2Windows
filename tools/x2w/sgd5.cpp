// The raw deflate decoder is X2W's own (startup_stubs/inflate.h, written from RFC 1951) in place of
// zlib-rs.
#include "sgd5.h"

#include <cstring>

#include "startup_stubs/inflate.h"

namespace x2w::tool::sgd5 {
namespace {

constexpr std::uint8_t MARKER[4] = {0xE7, 0xDC, 0x0D, 0x00};  // 0x000D_DCE7 little endian
// Containers are at most a few MiB; refuse to allocate for absurd sizes.
constexpr std::size_t MAX_UNPACKED = std::size_t{64} << 20;

std::optional<std::size_t> u32_at(std::span<const std::uint8_t> file, std::size_t o) {
    if (o > file.size() || file.size() - o < 4) return std::nullopt;
    return std::size_t{file[o]} | std::size_t{file[o + 1]} << 8 | std::size_t{file[o + 2]} << 16 |
           std::size_t{file[o + 3]} << 24;
}

bool is_ascii_graphic(std::uint8_t b) { return b >= 0x21 && b <= 0x7E; }

// The printable name that ends at `nul` and follows its length byte.
std::optional<std::string_view> name_before(std::span<const std::uint8_t> file, std::size_t nul) {
    std::size_t run = 0;
    for (std::size_t p = nul; p-- > 0;) {
        if (!is_ascii_graphic(file[p])) {
            run = p + 1;
            break;
        }
    }
    // From 32 characters on, the length byte is printable too.
    for (std::size_t l = run == 0 ? 0 : run - 1; l < nul; ++l) {
        if (std::size_t{file[l]} == nul - l) {
            // (Every byte after the length byte is printable ASCII: valid UTF-8.)
            return std::string_view(reinterpret_cast<const char*>(file.data() + l + 1), nul - l - 1);
        }
    }
    return std::nullopt;
}

std::optional<Record> record(std::span<const std::uint8_t> file, std::size_t at) {
    if (at < 4) return std::nullopt;
    const auto n = u32_at(file, at - 4);
    if (!n) return std::nullopt;
    const auto unpacked_size = u32_at(file, at + 8);
    if (!unpacked_size) return std::nullopt;
    const auto n2 = u32_at(file, at + 4);
    if (!n2 || *n2 != *n || *n < 12 || *unpacked_size > MAX_UNPACKED) return std::nullopt;
    if (at < 6) return std::nullopt;
    const std::size_t nul = at - 6;
    Record r;
    r.offset = at;
    r.name = file[nul] == 0 ? name_before(file, nul) : std::nullopt;
    r.kind = file[at - 5];
    if (at + *n > file.size()) return std::nullopt;
    r.packed = file.subspan(at + 12, *n - 12);
    r.unpacked_size = *unpacked_size;
    return r;
}

}  // namespace

std::optional<std::vector<std::uint8_t>> Record::unpack() const {
    // Decode raw deflate with a 15-bit window. Require an end marker and exactly unpacked_size output bytes.
    // Ignore trailing input.
    std::vector<std::uint8_t> out(unpacked_size);
    std::uint8_t none = 0;
    x2w_inflate::Stream s{packed.data(), packed.size(), 0, out.empty() ? &none : out.data(), out.size()};
    if (!x2w_inflate::inflate_raw(s) || s.out_pos != unpacked_size) return std::nullopt;
    return out;
}

std::vector<std::size_t> find_all(std::span<const std::uint8_t> hay, std::span<const std::uint8_t> needle) {
    std::vector<std::size_t> out;
    if (needle.empty() || hay.size() < needle.size()) return out;
    const std::uint8_t* base = hay.data();
    const std::size_t last = hay.size() - needle.size();
    std::size_t at = 0;
    while (at <= last) {
        const void* hit = std::memchr(base + at, needle[0], last - at + 1);
        if (!hit) break;
        at = static_cast<std::size_t>(static_cast<const std::uint8_t*>(hit) - base);
        if (std::memcmp(base + at, needle.data(), needle.size()) == 0) {
            out.push_back(at);
            at += needle.size();
        } else {
            ++at;
        }
    }
    return out;
}

bool is_sgd5(std::span<const std::uint8_t> file) {
    const std::uint8_t* end = file.data() + file.size();
    if (file.size() < 4) return false;
    for (const std::uint8_t* p = file.data(); p + 4 <= end;) {
        const void* hit = std::memchr(p, MAGIC[0], static_cast<std::size_t>(end - p) - 3);
        if (!hit) return false;
        p = static_cast<const std::uint8_t*>(hit);
        if (std::memcmp(p, MAGIC, 4) == 0) return true;
        ++p;
    }
    return false;
}

std::vector<Record> records(std::span<const std::uint8_t> file) {
    std::vector<Record> out;
    for (std::size_t at : find_all(file, MARKER))
        if (auto r = record(file, at)) out.push_back(*r);
    return out;
}

}  // namespace x2w::tool::sgd5
