
#include "x2w_shader/emit/sampler.h"

#include <algorithm>

namespace x2w::vs::sampler {

std::uint32_t StaticSampler::filter_for_host(bool aniso_point_mip_supported) const {
    if (!aniso_point_mip_supported && (filter & ~0x180u) == 0x54) return filter | 1;
    return filter;
}

StaticSampler StaticSampler::from_words(std::array<std::uint32_t, 4> w) {
    auto bits = [](std::uint32_t word, std::uint32_t at, std::uint32_t n) { return (word >> at) & ((1u << n) - 1); };
    // wrap, mirror, clamp, mirror once; the half-border modes have no D3D form and become their
    // nearest.
    auto address = [](std::uint32_t m) {
        static const std::uint32_t table[8] = {1, 2, 3, 5, 3, 5, 4, 5};
        return table[m];
    };
    const std::uint32_t mag = bits(w[2], 20, 2), min = bits(w[2], 22, 2), mip = bits(w[2], 26, 2);
    std::uint32_t reduction = 0;
    switch (bits(w[0], 29, 2)) {
    case 1:
        reduction = 2;
        break;
    case 2:
        reduction = 3;
        break;
    default:
        reduction = 0;
    }
    std::uint32_t f = 0;
    if (mag >= 2 || min >= 2)
        f = 0x54 | std::uint32_t(mip == 2) | reduction << 7;
    else
        f = (min & 1) << 4 | (mag & 1) << 2 | std::uint32_t(mip == 2) | reduction << 7;
    auto lod = [](std::uint32_t v) { return float(v) / 256.0f; };
    // Without a mip filter the base level is sampled.
    float min_lod = 0, max_lod = 0;
    if (mip != 0) {
        min_lod = lod(bits(w[1], 0, 12));
        max_lod = lod(bits(w[1], 12, 12));
    }
    StaticSampler s;
    s.filter = f;
    s.address = {address(bits(w[0], 0, 3)), address(bits(w[0], 3, 3)), address(bits(w[0], 6, 3))};
    s.mip_lod_bias = float(std::int32_t(bits(w[2], 0, 14) << 18) >> 18) / 256.0f;
    s.max_anisotropy = 1u << std::min(bits(w[0], 9, 3), 4u);
    s.comparison = bits(w[0], 12, 3) + 1;
    // shortcut: a custom border colour (type 3, a table entry) becomes transparent black; upgrade
    // when a title uses one.
    s.border = bits(w[3], 30, 2) % 3;
    s.min_lod = min_lod;
    s.max_lod = max_lod;
    return s;
}

}  // namespace x2w::vs::sampler
