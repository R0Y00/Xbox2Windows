#pragma once

#include "startup_stubs/gfx10_layout_local.h"
#include <d3d12.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace x2w_resident_mips {
struct Suffix {
    unsigned first;
    std::uint64_t prefix;
};

inline std::uint64_t prefix_bytes(const x2w_startup::gfx10_local::Layout& layout, unsigned first)
{
    if (!first) return layout.slice_size;
    const auto end = layout.level[first - 1].offset;
    return !end && !layout.linear ? std::uint64_t{1} << layout.block_log2 : end;
}

// Only omit levels already hidden by the title SRV. A shader using GetDimensions,
// Load, or explicit SampleLevel must see the same logical mip zero after rebasing.
inline UINT host_base(const D3D12_RESOURCE_DESC& desc, const D3D12_SHADER_RESOURCE_VIEW_DESC& view,
                      const x2w_startup::gfx10_local::Layout& layout, bool compressed)
{
    if (!layout.valid || layout.thick || layout.slices != 1 || !layout.mips ||
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1 ||
        view.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D || view.Texture2D.MostDetailedMip >= layout.mips)
        return 0;
    UINT base = view.Texture2D.MostDetailedMip;
    while (base && !layout.linear && !layout.level[base - 1].offset) --base;
    while (base && compressed && ((desc.Width >> base) < 4 || (desc.Height >> base) < 4 ||
           ((desc.Width >> base) % 4) || ((desc.Height >> base) % 4))) --base;
    return prefix_bytes(layout, base) < layout.slice_size ? base : 0;
}

inline D3D12_RESOURCE_DESC host_desc(D3D12_RESOURCE_DESC desc, UINT base)
{
    if (!base) return desc;
    desc.Width = (std::max)(UINT64{1}, desc.Width >> base);
    desc.Height = (std::max)(1u, desc.Height >> base);
    desc.MipLevels = static_cast<UINT16>(desc.MipLevels - base);
    return desc;
}

inline void rebase(D3D12_SHADER_RESOURCE_VIEW_DESC& view, UINT base)
{
    if (!base) return;
    auto& v = view.Texture2D;
    v.MostDetailedMip -= base;
    v.ResourceMinLODClamp = (std::max)(0.0f, v.ResourceMinLODClamp - static_cast<float>(base));
}

// Each thin slice stores its smallest mip(s) first. Tiled mip tails share an entire block.
// Require the same suffix in every array slice; array indexing must never reach an unfilled slice.
template<class Readable>
Suffix find(const x2w_startup::gfx10_local::Layout& layout, std::uintptr_t address, Readable readable)
{
    const Suffix none{layout.mips, 0};
    if (!layout.valid || layout.thick || !layout.mips || !layout.slice_size || !address ||
        layout.size > (1ull << 30) || layout.size > UINTPTR_MAX - address) return none;
    const auto available = [&](unsigned first) {
        const auto bytes = prefix_bytes(layout, first);
        if (!bytes || bytes > layout.slice_size) return false;
        for (unsigned slice = 0; slice < layout.slices; ++slice)
            if (!readable(address + slice * layout.slice_size, bytes)) return false;
        return true;
    };
    unsigned lo = 0, hi = layout.mips;
    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (available(mid)) hi = mid;
        else lo = mid + 1;
    }
    return lo == layout.mips ? none : Suffix{lo, prefix_bytes(layout, lo)};
}

inline bool stage_rows(const x2w_startup::gfx10_local::Layout& layout, Suffix suffix, UINT mip,
                       const unsigned char* source, unsigned char* destination, UINT columns, UINT rows, UINT pitch)
{
    if (!source || !destination || mip < suffix.first || mip >= layout.mips) return false;
    const auto element = 1u << layout.elem_log2;
    if (std::uint64_t(columns) * element > pitch) return false;
    for (UINT y = 0; y < rows; ++y) {
        auto* out = destination + std::uint64_t(y) * pitch;
        for (UINT x = 0; x < columns; ++x) {
            const auto offset = layout.address(mip, x, y, 0);
            if (offset + element > suffix.prefix) return false;
            std::memcpy(out + std::uint64_t(x) * element, source + offset, element);
        }
    }
    return true;
}
inline bool limit_view(D3D12_SHADER_RESOURCE_VIEW_DESC& view, unsigned mips, unsigned first)
{
    const auto apply = [&](auto& v) {
        if (v.MostDetailedMip >= mips || first >= mips || !v.MipLevels) return false;
        const UINT count = (std::min)(v.MipLevels, mips - v.MostDetailedMip);
        if (first >= v.MostDetailedMip + count) return false;
        if (first <= v.MostDetailedMip) return true;
        // Preserve the shader's mip numbering and dimensions. For a rebased title view,
        // defer until its own base is resident instead of combining two LOD origins.
        if (v.MostDetailedMip) return false;
        v.ResourceMinLODClamp = (std::max)(v.ResourceMinLODClamp, static_cast<float>(first));
        return true;
    };
    switch (view.ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE2D: return apply(view.Texture2D);
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY: return apply(view.Texture2DArray);
    default: return false;
    }
}
}
