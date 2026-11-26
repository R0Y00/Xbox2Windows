// Hashers for SSA values, sites and pairs use the FxHash multiply-rotate step. Tables are used for lookup
// or collected in a stable order before their contents affect output.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "x2w_shader/ir.h"

namespace x2w::vs::fhash {

constexpr std::uint64_t mix(std::uint64_t h, std::uint64_t n)
{
    return (((h << 5) | (h >> 59)) ^ n) * 0x517c'c1b7'2722'0a95ull;
}

struct ValueHash {
    std::size_t operator()(ir::Value v) const { return static_cast<std::size_t>(mix(0, v.n)); }
};

// (Value, Value): a lane mask by its dwords.
struct ValuePairHash {
    std::size_t operator()(const std::pair<ir::Value, ir::Value>& k) const
    {
        return static_cast<std::size_t>(mix(mix(0, k.first.n), k.second.n));
    }
};

// (block, instruction).
struct SiteHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& s) const
    {
        return static_cast<std::size_t>(mix(mix(0, s.first), s.second));
    }
};

}  // namespace x2w::vs::fhash
