// Emitter settings and loop-limit parsing.

#include "x2w_shader/emit/settings.h"

#include <cstdlib>
#include <string>

#include "x2w_shader/emit/opt.h"

namespace x2w::vs {

namespace {

std::optional<std::string_view> env(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string_view(v);
}

std::string_view trim(std::string_view s) {
    auto space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    while (!s.empty() && space(s.front())) s.remove_prefix(1);
    while (!s.empty() && space(s.back())) s.remove_suffix(1);
    return s;
}

// Accept an optional plus sign followed by one or more digits. Reject overflow.
std::optional<std::uint32_t> parse_u32(std::string_view s, unsigned radix) {
    if (!s.empty() && s.front() == '+') s.remove_prefix(1);
    if (s.empty()) return std::nullopt;
    std::uint64_t n = 0;
    for (char c : s) {
        unsigned d;
        if (c >= '0' && c <= '9')
            d = unsigned(c - '0');
        else if (c >= 'a' && c <= 'z')
            d = unsigned(c - 'a') + 10;
        else if (c >= 'A' && c <= 'Z')
            d = unsigned(c - 'A') + 10;
        else
            return std::nullopt;
        if (d >= radix) return std::nullopt;
        n = n * radix + d;
        if (n > UINT32_MAX) return std::nullopt;
    }
    return std::uint32_t(n);
}

// value.map(str::trim).and_then(0x hex or decimal).
std::optional<std::uint32_t> parse_number(std::optional<std::string_view> value) {
    if (!value) return std::nullopt;
    std::string_view v = trim(*value);
    if (v.substr(0, 2) == "0x") return parse_u32(v.substr(2), 16);
    return parse_u32(v, 10);
}

}  // namespace

std::uint32_t parse_set(std::optional<std::string_view> value, std::uint32_t unset, std::uint32_t all) {
    auto parsed = parse_number(value);
    return parsed ? (*parsed & all) : unset;
}

std::uint32_t parse_shader_opt(std::optional<std::string_view> value) {
    auto parsed = parse_number(value);
    return parsed ? (*parsed & bitcode::opt::ALL) : bitcode::opt::DEFAULT;
}

std::uint32_t parse_loop_limit(std::optional<std::string_view> value) {
    auto parsed = parse_number(value);
    if (parsed && *parsed > 0) return *parsed;
    return DEFAULT_LOOP_LIMIT;
}

std::uint32_t loop_limit() {
#ifdef X2W_VS_DIAGNOSTICS
    static const std::uint32_t limit = parse_loop_limit(env("X2W_SHADERCFG_SHADER_LOOP_LIMIT"));
    return limit;
#else
    return DEFAULT_LOOP_LIMIT;
#endif
}

bool helper_own() {
    static const bool on = [] {
        auto v = env("X2W_SHADERCFG_HELPER_OWN");
        return !v || *v != "0";
    }();
    return on;
}

std::uint8_t group_sync() {
    static const std::uint8_t mode = [] {
        auto v = env("X2W_SHADERCFG_GROUP_SYNC");
        if (v) {
            std::string_view t = trim(*v);
            if (t == "0") return std::uint8_t(0);
            if (t == "1") return std::uint8_t(1);
        }
        return std::uint8_t(2);
    }();
    return mode;
}

bool builder_fences() {
    static const bool on = [] {
        auto v = env("X2W_SHADERCFG_BVH_FENCES");
        return !v || *v != "0";
    }();
    return on;
}

bool builder_bounds() {
    static const bool on = [] {
        auto v = env("X2W_SHADERCFG_BVH_BOUNDS");
        return !v || *v != "0";
    }();
    return on;
}

std::uint32_t shader_opt() {
    static const std::uint32_t passes = parse_shader_opt(env("X2W_SHADERCFG_SHADER_OPT"));
    return passes;
}

std::uint32_t rdna_float() {
    static const std::uint32_t bits = parse_set(env("X2W_SHADERCFG_RDNA_FLOAT"), rdna::ALL, rdna::ALL);
    return bits;
}

}  // namespace x2w::vs
