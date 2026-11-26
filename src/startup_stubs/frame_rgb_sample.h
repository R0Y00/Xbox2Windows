#pragma once

#include <DirectXPackedVector.h>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace x2w_frame_sample {
enum class Format { rgba16f, rgba8, bgra8 };
struct Sample { float r, g, b, a; };

inline Sample decode(const unsigned char* p, Format format)
{
    if (format == Format::rgba16f) {
        std::uint16_t words[4];
        std::memcpy(words, p, sizeof(words));
        using DirectX::PackedVector::XMConvertHalfToFloat;
        return {XMConvertHalfToFloat(words[0]), XMConvertHalfToFloat(words[1]),
                XMConvertHalfToFloat(words[2]), XMConvertHalfToFloat(words[3])};
    }
    const bool bgra = format == Format::bgra8;
    return {p[bgra ? 2 : 0] / 255.f, p[1] / 255.f, p[bgra ? 0 : 2] / 255.f, p[3] / 255.f};
}

struct Counts {
    unsigned rgb = 0, alpha_only = 0, nonfinite = 0;
    float peak = 0;
    void add(Sample s)
    {
        if (!std::isfinite(s.r) || !std::isfinite(s.g) || !std::isfinite(s.b) || !std::isfinite(s.a)) {
            ++nonfinite;
        } else if (s.r != 0 || s.g != 0 || s.b != 0) {
            ++rgb;
            peak = std::fmax(peak, std::fmax(std::fabs(s.r), std::fmax(std::fabs(s.g), std::fabs(s.b))));
        } else if (s.a != 0) {
            ++alpha_only;
        }
    }
};

inline unsigned coordinate(unsigned size, unsigned cell, unsigned cells)
{
    return static_cast<unsigned>((2ull * cell + 1) * size / (2 * cells));
}
}
