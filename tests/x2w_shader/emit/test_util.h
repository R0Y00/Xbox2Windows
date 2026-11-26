#pragma once

// Minimal test helpers for the x2w_shader/emit tests (no framework: each test is an executable
// that returns nonzero on failure).

#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace test_util {

inline int& failures() {
    static int n = 0;
    return n;
}

inline void check(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", file, line, expr);
        ++failures();
    }
}

inline int finish(const char* name) {
    if (failures()) {
        std::fprintf(stderr, "%s: %d failure(s)\n", name, failures());
        return 1;
    }
    std::printf("%s: ok\n", name);
    return 0;
}

inline std::uint32_t u32le(std::span<const std::uint8_t> b, std::size_t at) {
    return std::uint32_t(b[at]) | std::uint32_t(b[at + 1]) << 8 | std::uint32_t(b[at + 2]) << 16 |
           std::uint32_t(b[at + 3]) << 24;
}

// FNV-1a, to pin a written module's exact bytes.
inline std::uint64_t fnv(std::span<const std::uint8_t> bytes) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::uint8_t b : bytes) h = (h ^ b) * 0x00000100000001b3ull;
    return h;
}

}  // namespace test_util

#define CHECK(expr) ::test_util::check(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
