#pragma once

// Immediate samplers: the console's four sampler words (the GPU's public SQ_IMG_SAMP_WORD0-3 layout) as a
// D3D12 static sampler.
//
// Word 0: address modes x, y, z (bits 0-8, 3 each), log2 of the maximum anisotropy (9-11),
// comparison function (12-14), reduction (29-30: 0 filtering, 1 minimum, 2 maximum). Word 1:
// minimum and maximum LOD (0-11, 12-23, unsigned 4.8). Word 2: LOD bias (0-13, signed 5.8),
// magnification and minification filters (20-21, 22-23: point, linear, anisotropic point,
// anisotropic linear), mip filter (26-27: none, point, linear). Word 3: border colour (30-31:
// transparent black, opaque black, opaque white, custom).

#include <array>
#include <cstdint>

namespace x2w::vs::sampler {

// A static sampler in D3D12 terms; the integers are the D3D12_FILTER,
// D3D12_TEXTURE_ADDRESS_MODE, D3D12_COMPARISON_FUNC and D3D12_STATIC_BORDER_COLOR values.
struct StaticSampler {
    std::uint32_t filter = 0;
    // U, V, W.
    std::array<std::uint32_t, 3> address{};
    float mip_lod_bias = 0;
    std::uint32_t max_anisotropy = 0;
    std::uint32_t comparison = 0;
    std::uint32_t border = 0;
    float min_lod = 0;
    float max_lod = 0;

    // The filter supported by the host device. Anisotropic point-mip filtering is optional
    // (D3D12_OPTIONS19); older devices retain the previous linear-mip approximation.
    std::uint32_t filter_for_host(bool aniso_point_mip_supported) const;
    // Decodes the four sampler words.
    static StaticSampler from_words(std::array<std::uint32_t, 4> w);
    // (The floats come from fixed-point fields: never NaN.)
    bool operator==(const StaticSampler&) const = default;
};

}  // namespace x2w::vs::sampler
