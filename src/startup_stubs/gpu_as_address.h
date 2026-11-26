#pragma once

#include <cstdint>

namespace x2w_as_address {
// X2WV native view token leaves words 3 and 6..7 for metadata. Preserve its
// capture id and resource when a placed AS view is made at a desktop GPU VA.
constexpr std::uint32_t token = 0x56573258;
constexpr std::uint32_t annotation = 0x41573258;
inline bool annotate(std::uint32_t* words, std::uint64_t title)
{
    if (words[0] != token || words[2] != 18) return false;
    words[3] = annotation;
    words[6] = std::uint32_t(title);
    words[7] = std::uint32_t(title >> 32);
    return true;
}
inline std::uint64_t address(const std::uint32_t* words, std::uint64_t captured)
{
    return words[3] == annotation ? std::uint64_t{words[6]} | (std::uint64_t{words[7]} << 32) : captured;
}
}
