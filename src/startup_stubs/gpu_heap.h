// Device-independent mirror heap layout, stride tracking, descriptor diffs and dynamic slot allocation.
//
// For a host supporting 2^22 views, slots [0,count) mirror the title heap. Dynamic table slots follow. Shaders
// clamp indices to the null slot at 2^21 - 1. The raw twin of slot i is at 2^21 + i. Unused twin slots hold null
// raw buffer views.
#pragma once

#include "startup_stubs/gpu_abi.h"
#include "startup_stubs/gpu_as_address.h"
#include "startup_stubs/gpu_depth_twins.h"
#include "startup_stubs/gpu_late_texture.h"
#include <map>
#include <set>

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace gpu_heap {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// What a host that caps view heaps (NVIDIA) takes.
constexpr u32 portable_views = 1'000'000;
// Every view heap on a host that takes more; the last is the null slot.
constexpr u32 uncapped_views = 1u << 21;
// Where the raw twins start in a heap that has them.
constexpr u32 raw_twins_first = 1u << 21;
// Base dynamic slot count.
constexpr u32 dynamic_base = 1u << 18;
// X2W_SHADERCFG_DYNAMIC_SLOTS defaults to DYNAMIC << 2.
constexpr u32 dynamic_most_default = dynamic_base << 2;

// Table of buffer element strides at t255 in the root CBV space. Each view slot has one dword.
constexpr u32 slot_strides_count = 1u << 21;
constexpr u32 view_stride_bits = (1u << 30) - 1;
constexpr u32 view_integer = 1u << 31;
constexpr u32 view_signed = 1u << 30;

static_assert(raw_twins_first == x2w_gpu::abi::raw_views, "twins at the export's raw_views");
static_assert(uncapped_views - 1 == x2w_gpu::abi::view_heap_limit, "the null slot the export clamps to");
static_assert(2 * raw_twins_first == x2w_gpu::abi::heap_views_with_twins, "2^22 views with twins");
static_assert(dynamic_base == x2w_gpu::abi::dynamic_min && dynamic_most_default == x2w_gpu::abi::dynamic_max,
              "dynamic_slots bounds");

// Bytes per descriptor by heap type (views, samplers, RTV, DSV).
inline std::optional<u32> increment(u32 kind)
{
    constexpr u32 steps[4] = {32, 16, 128, 128};
    if (kind < 4) return steps[kind];
    return std::nullopt;
}

// What probe_view_limit settles once: the null slot every view index is clamped to, and where
// the raw twins start (0: none).
struct Limits {
    u32 views = 0;      // heap ends at the null slot views - 1 (without twins)
    u32 raw = 0;        // raw twins' first slot, 0 if none
    u32 null() const { return views - 1; }
};

// `fits(count)` says whether the host takes a shader-visible view heap of `count` descriptors. With twins wanted, a
// host that takes 2 * RAW_TWINS gets them.
template <class Fits>
Limits probe_view_limit(bool twins_wanted, Fits&& fits)
{
    Limits l;
    const bool twins = twins_wanted && fits(2 * raw_twins_first);
    if (twins) l.raw = raw_twins_first;
    if (twins) l.views = raw_twins_first;
    else if (fits(uncapped_views)) l.views = uncapped_views;
    else l.views = portable_views;
    return l;
}

// DYNAMIC, or up to four times as many where the host heap has room.
inline u32 dynamic_slots(u32 count, u32 null, u32 most = dynamic_most_default)
{
    if (null < portable_views) return dynamic_base;
    const u32 room = null > count ? null - count : 0;
    return (std::max)((std::min)(room, most), dynamic_base);
}

// The layout of one host heap: what is made, and where.
struct Layout {
    bool sampler = false;
    u32 title = 0;      // the title's own count
    u32 count = 0;      // title slots that have host slots (fewer if capped)
    u32 dynamic = 0;    // dynamic slots, from `count` on
    u32 null = 0;       // the null slot (0: none, sampler heaps)
    u32 raw = 0;        // first twin slot (0: none)
    u32 size = 0;       // descriptors in the host heap
    u32 twins = 0;      // slots with a twin: [0, twins)
    bool capped = false;

    u32 twin(u32 index) const { return raw + index; }
    bool has_twin(u32 index) const { return raw != 0 && index < twins; }
    bool is_dynamic(u32 index) const { return index >= count && index - count < dynamic; }
    // The host slot a shader's view index reads (indices at or past the null slot are clamped
    // there by the translated shader, twins excepted).
    u32 clamp(u32 index) const { return null && index > null ? null : index; }
    // How many slots fill_null makes plain null views before the twins' null buffer views.
    u32 null_views_end() const { return raw ? (std::min)(raw, size) : size; }
};

inline Layout layout_for(bool sampler, u32 count, const Limits& limits, u32 most = dynamic_most_default)
{
    Layout l;
    l.sampler = sampler;
    l.title = count;
    const u32 null = sampler ? 0 : limits.null();
    const u32 raw = sampler ? 0 : limits.raw;
    const u32 extra = sampler ? 0 : dynamic_slots(count, null, most);
    l.dynamic = extra;
    l.null = null;
    l.raw = raw;
    if (null == 0) {
        l.count = count;
        l.size = count + extra;
    } else {
        l.size = raw == 0 ? null + 1 : 2 * raw;
        if (u64{count} + extra <= null) {
            l.count = count;
        } else {
            // Descriptors beyond the host limit read the null slot.
            l.count = null > extra ? null - extra : 0;
            l.capped = true;
        }
    }
    l.twins = raw == 0 ? 0 : l.count + extra;
    return l;
}

// Bytes per element and the element's width in texels, by DXGI format.
struct Element { u64 bytes, block; };
inline std::optional<Element> element(u32 f)
{
    if (f >= 1 && f <= 4) return Element{16, 1};
    if (f >= 5 && f <= 8) return Element{12, 1};
    if (f >= 9 && f <= 22) return Element{8, 1};
    if ((f >= 23 && f <= 47) || f == 67 || (f >= 87 && f <= 93) || f == 116 || f == 117 || f == 189 || f == 191)
        return Element{4, 1};
    if (f >= 118 && f <= 120) return Element{3, 1};
    if ((f >= 48 && f <= 59) || f == 68 || f == 69 || f == 85 || f == 86 || f == 115) return Element{2, 1};
    if ((f >= 60 && f <= 65) || f == 190) return Element{1, 1};
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) return Element{8, 4};
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) return Element{16, 4};
    return std::nullopt;
}
inline u64 element_bytes(u32 f) { auto e = element(f); return e ? e->bytes : 0; }

// The console's 16-bit depth with stencil has no PC format.
inline u32 pc_format(u32 f)
{
    switch (f) {
    case 118: return 45;
    case 119: return 46;
    case 120: return 47;
    default: return f;
    }
}

// The typeless format of a format's family (itself if none).
inline u32 family(u32 f)
{
    constexpr u32 families[22][2] = {{1, 4}, {5, 8}, {9, 14}, {15, 18}, {19, 22}, {23, 25}, {27, 32}, {33, 38},
                                     {39, 43}, {44, 47}, {48, 52}, {53, 59}, {60, 64}, {70, 72}, {73, 75}, {76, 78},
                                     {79, 81}, {82, 84}, {90, 91}, {92, 93}, {94, 96}, {97, 99}};
    if (f == 87) return 90;
    if (f == 88) return 92;
    for (const auto& r : families)
        if (f >= r[0] && f <= r[1]) return r[0];
    return f;
}

constexpr u32 format_unknown = 0;
constexpr u32 format_r32_typeless = 39;

// Element layout of a buffer view.
struct Elements {
    u32 stride = 0;
    bool raw = false;
    u32 format = 0;
};

// Bytes per element as the console's descriptor holds them for indexed reads (structured: the stride; raw: 1, the
// index is the byte; typed: the format's).
inline u32 view_stride(const Elements& e)
{
    if (e.stride > 0) return e.stride;
    if (e.raw) return 1;
    return static_cast<u32>(element_bytes(e.format));
}

// VIEW_INTEGER (and VIEW_SIGNED) for a typed buffer view of an integer format.
inline u32 view_kind(const Elements& e)
{
    if (e.stride != 0 || e.raw) return 0;
    switch (e.format) {
    case 3: case 7: case 12: case 17: case 25: case 30: case 36: case 42: case 50: case 57: case 62:
        return view_integer;
    case 4: case 8: case 14: case 18: case 32: case 38: case 43: case 52: case 59: case 64:
        return view_integer | view_signed;
    default:
        return 0;
    }
}

// The word the table of strides holds for a slot: 0 if it is no buffer view.
inline u32 stride_word(const std::optional<Elements>& e)
{
    if (!e) return 0;
    return view_stride(*e) | view_kind(*e);
}

// A buffer view that begins `offset` bytes into its host copy where no host view can begin.
inline bool off_elements(u64 offset, u32 stride, bool raw, u64 element)
{
    if (raw) return offset % 16 != 0;
    if (stride > 0) return offset % 16 != 0 && offset % stride != 0;
    return element > 0 && offset % element != 0;
}

// A structured view (`first`, `count` elements of `stride` bytes, `offset` bytes into its host copy) as a raw view
// of the same bytes: first dword and dwords. Only when the bytes start on 16 and are whole dwords.
struct RawWindow { u64 first; u32 dwords; };
inline std::optional<RawWindow> raw_window(u64 offset, u64 first, u32 count, u32 stride)
{
    if (stride == 0) return std::nullopt;
    if (first > (~u64{0} - offset) / stride) return std::nullopt;
    const u64 start = first * stride + offset;
    const u64 bytes = u64{count} * stride;
    if (start % 16 != 0 || bytes % 4 != 0) return std::nullopt;
    if (bytes / 4 > 0xffffffffull) return std::nullopt;
    return RawWindow{start / 4, static_cast<u32>(bytes / 4)};
}

// Whether the host accepts a buffer view with these fields of a buffer of `width` bytes (it removes the device for
// one it does not).
inline bool buffer_view_ok(u32 format, u64 first, u32 count, u32 stride, bool raw, u64 width)
{
    u64 el = 0;
    if (raw) {
        if (format != format_r32_typeless || stride != 0 || first % 4 != 0) return false;
        el = 4;
    } else if (stride > 0) {
        if (format != format_unknown || stride > 2048 || stride % 4 != 0) return false;
        el = stride;
    } else {
        const bool typeless = family(format) == format && family(format + 1) == format;
        const auto e = element(format);
        if (!e || e->block != 1 || format == 0 || typeless || e->bytes == 0) return false;
        el = e->bytes;
    }
    if (count == 0) return false;
    const u64 end = first + count;
    if (end < first) return false;
    if (el && end > ~u64{0} / el) return false;
    return end * el <= width;
}

// The bytes a buffer view the host takes shows as a raw view: first dword, dwords, and the bytes past that dword it
// begins (a raw view begins on 16 bytes).
struct TwinWindow { u64 first; u32 dwords; u32 by; };
inline std::optional<TwinWindow> twin_window(u32 format, u64 first, u32 count, u32 stride, bool raw, u64 width)
{
    u64 el = raw ? 4 : stride > 0 ? stride : element_bytes(format);
    const u64 start = first * el;
    u64 end = (first + count) * el;
    end = (end + 3) & ~u64{3};
    end = (std::min)(end, width & ~u64{3});
    const u64 from = start & ~u64{15};
    if (end <= start) return std::nullopt;
    const u64 dwords = (end - from) / 4;
    if (dwords > 0xffffffffull) return std::nullopt;
    return TwinWindow{from / 4, static_cast<u32>(dwords), static_cast<u32>(start - from)};
}

// Whole 256s, at most the 64 KiB a shader can index.
inline u32 constant_bytes(u32 size)
{
    constexpr u32 largest = 64u << 10;
    const u32 s = (std::min)(size, largest);
    return (s + 255) & ~255u;
}

// The records (`step` bytes each, the first is slot `first`) that differ between `last` and `records` (of one
// length) go into `out` and `last`. Sixteen at a time.
inline void changed_records(unsigned char* last, const unsigned char* records, std::size_t len, std::size_t step,
                            u32 first, std::vector<u32>& out)
{
    constexpr std::size_t at_a_time = 16;
    const std::size_t block = at_a_time * step;
    for (std::size_t b = 0, at = 0; at < len; ++b, at += block) {
        const std::size_t n = (std::min)(block, len - at);
        if (std::memcmp(last + at, records + at, n) == 0) continue;
        for (std::size_t r = 0; r + step <= n; r += step) {
            if (std::memcmp(last + at + r, records + at + r, step) != 0) {
                std::memcpy(last + at + r, records + at + r, step);
                out.push_back(first + static_cast<u32>(b * at_a_time + r / step));
            }
        }
    }
}

// The console's sampler description (CreateSamplerX, 27 words: flags, magnification, minification, mip and depth
// filters, filter mode, three address modes, minimum and maximum LOD, LOD bias, secondary bias, anisotropy, its
// bias and threshold, comparison, border colour, then tuning words) as the host's.
inline float lod_finite(float value, float fallback)
{
    if (value != value) return fallback;
    return (std::min)((std::max)(value, -D3D12_FLOAT32_MAX), D3D12_FLOAT32_MAX);
}
inline D3D12_SAMPLER_DESC console_sampler(const u32 w[27])
{
    const auto f = [&](int i) {
        float v;
        std::memcpy(&v, &w[i], 4);
        return v;
    };
    const u32 mag = w[1], min = w[2], mip = w[3], mode = w[5], anisotropy = w[13], compare = w[16];
    const auto linear = [](u32 x) { return u32{x == 1 || x == 3}; };
    const bool aniso = mag >= 2 || min >= 2;
    // A sampler given a real test (less ... greater-or-equal) is a comparison sampler; never, always and none not.
    const bool compares = compare >= 2 && compare <= 7;
    const u32 reduction = mode == 1 ? 2 : mode == 2 ? 3 : u32{compares};
    const u32 filter = aniso ? 0x55 : (linear(min) << 4 | linear(mag) << 2 | u32{mip == 2});
    const float min_lod = lod_finite(f(9), 0.0f);
    // Without a mip filter the console clamps sampling to min_lod.
    const float max_lod = mip == 0 ? min_lod : (std::max)(lod_finite(f(10), min_lod), min_lod);
    const auto address = [](u32 m) {
        return m >= 1 && m <= 5 ? static_cast<D3D12_TEXTURE_ADDRESS_MODE>(m) : D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    };
    D3D12_SAMPLER_DESC d{};
    d.Filter = static_cast<D3D12_FILTER>(reduction << 7 | filter);
    d.AddressU = address(w[6]);
    d.AddressV = address(w[7]);
    d.AddressW = address(w[8]);
    const float bias = f(11);
    d.MipLODBias = bias != bias ? 0.0f : (std::min)((std::max)(bias, D3D12_MIP_LOD_BIAS_MIN), D3D12_MIP_LOD_BIAS_MAX);
    d.MaxAnisotropy = aniso ? (std::min)((std::max)(anisotropy, 1u), 16u) : 1;
    d.ComparisonFunc = reduction == 1 ? static_cast<D3D12_COMPARISON_FUNC>(compare) : D3D12_COMPARISON_FUNC_NEVER;
    for (int i = 0; i < 4; ++i) d.BorderColor[i] = f(17 + i);
    d.MinLOD = min_lod;
    d.MaxLOD = max_lod;
    return d;
}

// The record WP5 writes into a title slot for a placed buffer view (device slots 70 and 71) that X2W's handler left
// without a record it can follow: {"X2WP", address lo, hi, first element, elements, stride, format | raw << 16 |
// unordered << 17, the buffer's width}. Copies of it travel with the title's memory copies like X2WV tokens.
constexpr u32 placed_magic = 0x50573258;  // "X2WP"
struct PlacedRecord {
    u64 address = 0;
    u32 first = 0, elements = 0, stride = 0, format = 0;
    bool raw = false, unordered = false;
    u32 width = 0;
};
inline void placed_words(const PlacedRecord& p, u32 out[8])
{
    out[0] = placed_magic;
    out[1] = static_cast<u32>(p.address);
    out[2] = static_cast<u32>(p.address >> 32);
    out[3] = p.first;
    out[4] = p.elements;
    out[5] = p.stride;
    out[6] = (p.format & 0xffff) | (p.raw ? 1u << 16 : 0) | (p.unordered ? 1u << 17 : 0);
    out[7] = p.width;
}
inline bool placed_of(const u32 w[8], PlacedRecord& p)
{
    if (w[0] != placed_magic) return false;
    p.address = w[1] | (u64{w[2]} << 32);
    p.first = w[3];
    p.elements = w[4];
    p.stride = w[5];
    p.format = w[6] & 0xffff;
    p.raw = (w[6] >> 16) & 1;
    p.unordered = (w[6] >> 17) & 1;
    p.width = w[7];
    return p.address != 0 && p.elements != 0;
}

// Reserve dynamic slots until release. The caller holds the heap lock.
class DynamicRuns {
public:
    DynamicRuns() = default;
    DynamicRuns(u32 count, u32 dynamic) { reset(count, dynamic); }
    void reset(u32 count, u32 dynamic)
    {
        count_ = count;
        dynamic_ = dynamic;
        taken_.assign(dynamic, 0);
        next_ = 0;
        free_ = dynamic;
        too_long_ = ~0u;
    }
    // `n` consecutive free dynamic slots: the first's heap index.
    std::optional<u32> take(u32 n)
    {
        if (n == 0 || n > dynamic_) return std::nullopt;
        const bool enough = free_ >= n && n < too_long_;
        if (!enough) return std::nullopt;
        const std::size_t cursor = next_, end = dynamic_;
        const std::size_t second_end = (std::min)(cursor + (n - 1), end);
        std::size_t run = 0, last = ~std::size_t{0};
        std::optional<std::size_t> found;
        auto look = [&](std::size_t at) {
            if (at != last + 1) run = 0;
            last = at;
            run = taken_[at] ? 0 : run + 1;
            if (run == n) found = at + 1 - n;
            return found.has_value();
        };
        for (std::size_t at = cursor; at < end && !found; ++at) look(at);
        for (std::size_t at = 0; at < second_end && !found; ++at) look(at);
        if (!found) {
            too_long_ = (std::min)(too_long_, n);
            return std::nullopt;
        }
        std::fill(taken_.begin() + *found, taken_.begin() + *found + n, std::uint8_t{1});
        free_ -= n;
        next_ = static_cast<u32>((*found + n) % end);
        return count_ + static_cast<u32>(*found);
    }
    // Gives back `n` dynamic slots from heap index `first`.
    void release(u32 first, u32 n)
    {
        if (first < count_) return;
        const u32 at = first - count_;
        const u64 end = (std::min<u64>)(u64{at} + n, dynamic_);
        if (at >= end) return;
        u32 given = 0;
        for (u64 i = at; i < end; ++i) {
            given += taken_[i] ? 1 : 0;
            taken_[i] = 0;
        }
        free_ += given;
        too_long_ = ~0u;
    }
    u32 free() const { return free_; }
    u32 size() const { return dynamic_; }

private:
    u32 count_ = 0, dynamic_ = 0, next_ = 0, free_ = 0, too_long_ = ~0u;
    std::vector<std::uint8_t> taken_;
};

// One slot's bit in a set of slots: put, remove, take the changed ones.
class SlotSet {
public:
    void resize(std::size_t n) { words_.assign((n + 63) / 64, 0); }
    bool put(u32 i) { return flip(i, true); }
    bool remove(u32 i) { return flip(i, false); }
    bool has(u32 i) const { return i / 64 < words_.size() && (words_[i / 64] >> (i % 64) & 1); }
    template <class F> void each(F&& f) const
    {
        for (std::size_t w = 0; w < words_.size(); ++w)
            for (u64 bits = words_[w]; bits; bits &= bits - 1) {
                unsigned long b = 0;
                for (u64 t = bits; !(t & 1); t >>= 1) ++b;
                f(static_cast<u32>(w * 64 + b));
            }
    }
    std::vector<u32> take()
    {
        std::vector<u32> out;
        each([&](u32 i) { out.push_back(i); });
        std::fill(words_.begin(), words_.end(), 0);
        return out;
    }

private:
    // Whether the bit was set before.
    bool flip(u32 i, bool on)
    {
        if (i / 64 >= words_.size()) return false;
        u64& w = words_[i / 64];
        const u64 bit = u64{1} << (i % 64);
        const bool was = (w & bit) != 0;
        w = on ? (w | bit) : (w & ~bit);
        return was;
    }
    std::vector<u64> words_;
};

}  // namespace gpu_heap
