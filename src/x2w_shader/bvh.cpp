// Preserve operation order and disable FMA contraction. The lowering is compared against this arithmetic
// dword for dword.
#include "x2w_shader/bvh.h"

#include <bit>
#include <cmath>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

namespace x2w::vs::bvh {

namespace {

float from_bits(std::uint32_t b) { return std::bit_cast<float>(b); }
std::uint32_t bits_of(float f) { return std::bit_cast<std::uint32_t>(f); }

// A NaN operand gives the other operand.
float fmin_rs(float a, float b) { return std::fmin(a, b); }
float fmax_rs(float a, float b) { return std::fmax(a, b); }

}  // namespace

Ray Ray::make(Vec3 origin, Vec3 direction, float extent)
{
    Ray r;
    r.extent = extent;
    r.origin = origin;
    r.direction = direction;
    for (int i = 0; i < 3; ++i) r.inverse[i] = 1.0f / direction[i];
    return r;
}

std::uint32_t kind(std::uint64_t pointer) { return static_cast<std::uint32_t>(pointer & 7); }

std::uint64_t node_address(std::uint32_t word0, std::uint32_t word1, std::uint64_t pointer)
{
    const std::uint64_t base = ((std::uint64_t{word1 & 0xff} << 32) | std::uint64_t{word0}) << 8;
    return base + ((pointer >> 3) << 6);  // wrapping
}

Result4 to_bits(const std::array<float, 4>& values)
{
    return {bits_of(values[0]), bits_of(values[1]), bits_of(values[2]), bits_of(values[3])};
}

Result4 intersect(const Words& words, std::uint32_t k, const Ray& ray, bool sort)
{
    switch (k & 7) {
    case 4: return boxes(children(words), boxes16(words), ray, sort);
    case 5: return boxes(children(words), boxes32(words), ray, sort);
    case 0:
    case 1:
    case 2:
    case 3: return to_bits(triangle(triangle_vertices(words, k & 7), ray));
    default: return to_bits(MISS);
    }
}

Result4 children(const Words& words) { return {words[0], words[1], words[2], words[3]}; }

Boxes boxes16(const Words& words)
{
    Boxes out{};
    for (std::size_t b = 0; b < 4; ++b) {
        for (std::size_t k = 0; k < 6; ++k) {
            const std::size_t half = 6 * b + k;
            out[b][k] = half_to_f32(static_cast<std::uint16_t>(words[4 + half / 2] >> (16 * (half % 2))));
        }
    }
    return out;
}

Boxes boxes32(const Words& words)
{
    Boxes out{};
    for (std::size_t b = 0; b < 4; ++b)
        for (std::size_t k = 0; k < 6; ++k) out[b][k] = from_bits(words[4 + 6 * b + k]);
    return out;
}

std::uint32_t vertex_map(std::uint32_t k)
{
    const auto& v = TRIANGLE_VERTICES[k & 3];
    return v[0] | (v[1] << 4) | (v[2] << 8);
}

Triangle triangle_vertices(const Words& words, std::uint32_t k)
{
    const std::uint32_t id = words[15] >> (8 * k);
    const std::uint32_t i = id & 3, j = (id >> 2) & 3;
    // An id naming one vertex twice, or a vertex 3, is malformed; it reads
    // what this arithmetic lands on (a vertex of the node), as the emitted
    // code does.
    const std::uint32_t first = (3u - i - j) & 3;
    const std::uint32_t names[3] = {first, i, j};
    Triangle out{};
    for (int t = 0; t < 3; ++t) {
        const std::size_t vertex = (vertex_map(k) >> (4 * names[t])) & 0xf;
        for (std::size_t c = 0; c < 3; ++c) out[t][c] = from_bits(words[3 * vertex + c]);
    }
    return out;
}

Result4 boxes(const Result4& children, const Boxes& bounds, const Ray& ray, bool sort)
{
    // An infinite inverse (a direction component of 0) would turn an origin
    // exactly on a box's face into a NaN. (Maximum then minimum, as the
    // emitted code: a NaN becomes the lower bound.)
    constexpr float big = std::numeric_limits<float>::max();
    Vec3 inverse{};
    for (int a = 0; a < 3; ++a) inverse[a] = fmin_rs(fmax_rs(ray.inverse[a], -big), big);
    Result4 ids{NONE, NONE, NONE, NONE};
    const float inf = std::numeric_limits<float>::infinity();
    std::array<float, 4> distances{inf, inf, inf, inf};
    for (std::size_t n = 0; n < 4; ++n) {
        const Box& b = bounds[n];
        float near_[3], far_[3];
        for (int a = 0; a < 3; ++a) {
            const float low = (b[a] - ray.origin[a]) * inverse[a];
            const float high = (b[3 + a] - ray.origin[a]) * inverse[a];
            near_[a] = fmin_rs(low, high);
            far_[a] = fmax_rs(low, high);
        }
        const float enter = fmax_rs(fmax_rs(near_[0], near_[1]), near_[2]);
        const float leave = fmin_rs(fmin_rs(far_[0], far_[1]), far_[2]);
        // A box whose minimum x is NaN is switched off. A child that is not
        // there may have inverted infinite bounds, which every ray enters at
        // minus infinity: it would be first of the sorted children and end the
        // walk before them.
        if (children[n] != NONE && !std::isnan(b[0]) && leave >= fmax_rs(enter, 0.0f) && enter < ray.extent) {
            ids[n] = children[n];
            distances[n] = enter;
        }
    }
    if (sort) {
        constexpr std::pair<int, int> network[5] = {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}};
        for (auto [a, b] : network) {
            if (distances[b] < distances[a]) {
                std::swap(distances[a], distances[b]);
                std::swap(ids[a], ids[b]);
            }
        }
    }
    return ids;
}

std::array<float, 4> triangle(const Triangle& p, const Ray& ray)
{
    const Vec3& d = ray.direction;
    const Vec3 size{std::fabs(d[0]), std::fabs(d[1]), std::fabs(d[2])};
    // The largest axis, and the other two in an order that keeps the
    // triangle's winding.
    int kz;
    if (size[0] >= size[1])
        kz = size[0] >= size[2] ? 0 : 2;
    else if (size[1] >= size[2])
        kz = 1;
    else
        kz = 2;
    int kx = (kz + 1) % 3, ky = (kz + 2) % 3;
    if (d[kz] < 0.0f) std::swap(kx, ky);
    const float sz = ray.inverse[kz];
    const float sx = d[kx] * sz, sy = d[ky] * sz;
    struct Sheared {
        float x, y, z;
    };
    auto sheared = [&](const Vec3& v) {
        const float r[3] = {v[0] - ray.origin[0], v[1] - ray.origin[1], v[2] - ray.origin[2]};
        const float x = r[kx] - sx * r[kz];
        const float y = r[ky] - sy * r[kz];
        const float z = sz * r[kz];
        return Sheared{x, y, z};
    };
    const Sheared a = sheared(p[0]);
    const Sheared b = sheared(p[1]);
    const Sheared c = sheared(p[2]);
    const float u = c.x * b.y - c.y * b.x;
    const float v = a.x * c.y - a.y * c.x;
    const float w = b.x * a.y - b.y * a.x;
    const bool behind = u < 0.0f || v < 0.0f || w < 0.0f;
    const bool before = u > 0.0f || v > 0.0f || w > 0.0f;
    const float det = u + (v + w);
    const float t = (u * a.z + v * b.z) + w * c.z;
    const bool ahead = (det > 0.0f && t >= 0.0f) || (det < 0.0f && t <= 0.0f);
    if (!(behind && before) && ahead) return {t, det, v, w};
    return MISS;
}

float half_to_f32(std::uint16_t h)
{
    const std::uint32_t sign = std::uint32_t{static_cast<std::uint32_t>(h >> 15)} << 31;
    const std::uint32_t exponent = (h >> 10) & 0x1f, mantissa = h & 0x3ffu;
    std::uint32_t bits;
    if (exponent == 0 && mantissa == 0) {
        bits = sign;
    } else if (exponent == 0) {
        // Subnormal: normalise.
        const std::uint32_t shift = static_cast<std::uint32_t>(std::countl_zero(mantissa)) - 21;
        bits = sign | ((113 - shift) << 23) | (((mantissa << shift) & 0x3ff) << 13);
    } else if (exponent == 0x1f) {
        bits = sign | (0xffu << 23) | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return from_bits(bits);
}

namespace testing {

std::uint16_t f32_to_half(float x, bool up)
{
    bool have = false;
    std::uint16_t best = 0;
    for (std::uint32_t hh = 0; hh <= 0xffff; ++hh) {
        const auto h = static_cast<std::uint16_t>(hh);
        const float f = half_to_f32(h);
        if (std::isnan(f) || (up ? f < x : f > x)) continue;
        if (have && std::fabs(half_to_f32(best) - x) <= std::fabs(f - x)) continue;
        best = h;
        have = true;
    }
    return best;  // a half always exists for the values the tests use
}

Words box32(const Result4& children, const Boxes& bounds)
{
    Words w{};
    for (int i = 0; i < 4; ++i) w[i] = children[i];
    for (std::size_t b = 0; b < 4; ++b)
        for (std::size_t k = 0; k < 6; ++k) w[4 + 6 * b + k] = bits_of(bounds[b][k]);
    return w;
}

Words box16(const Result4& children, const Boxes& bounds)
{
    Words w{};
    for (int i = 0; i < 4; ++i) w[i] = children[i];
    for (std::size_t b = 0; b < 4; ++b) {
        for (std::size_t k = 0; k < 6; ++k) {
            const std::size_t half = 6 * b + k;
            const float x = bounds[b][k];
            const std::uint16_t h = std::isnan(x) ? std::uint16_t{0x7e00} : f32_to_half(x, k >= 3);
            w[4 + half / 2] |= std::uint32_t{h} << (16 * (half % 2));
        }
    }
    return w;
}

Words triangles(const Vec3* vertices, std::size_t count, std::uint32_t id)
{
    Words w{};
    for (std::size_t n = 0; n < count; ++n)
        for (std::size_t c = 0; c < 3; ++c) w[3 * n + c] = bits_of(vertices[n][c]);
    w[15] = id;
    return w;
}

}  // namespace testing

}  // namespace x2w::vs::bvh
