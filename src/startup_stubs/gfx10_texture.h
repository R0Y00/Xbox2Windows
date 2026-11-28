#pragma once

// GFX10 texture tiling, detiling, and mip-tail layout runtime.
//
// The micro-tile patterns, macro-tile dimensions, and mip-tail calculations
// follow AMD's addrlib in Mesa (src/amd/addrlib, SPDX: MIT):
//
//   Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.
//   SPDX-License-Identifier: MIT

#include <d3d12.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace x2w_startup {

enum Gfx10SwizzleMode : UINT {
    SW_LINEAR = 0,
    SW_256B_S = 1,
    SW_256B_D = 2,
    SW_4KB_S  = 5,
    SW_4KB_D  = 6,
    SW_64KB_S = 9,
    SW_64KB_D = 10,
};

struct Gfx10FormatInfo {
    UINT bpe = 0;       // Bytes per element (texel or block)
    UINT block_dim = 0; // 4 for block-compressed (BC), 1 for uncompressed
};

inline Gfx10FormatInfo gfx10_format_info(DXGI_FORMAT fmt)
{
    switch (fmt) {
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return {16, 4};

    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        return {8, 4};

    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return {16, 1};

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
        return {8, 1};

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return {4, 1};

    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B5G5R5A1_UNORM:
        return {2, 1};

    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
        return {1, 1};

    default:
        return {0, 0};
    }
}

inline UINT gfx10_swizzle_from_desc(const D3D12_RESOURCE_DESC& d)
{
    const UINT layout = static_cast<UINT>(d.Layout);
    return (layout & 0xFF00) == 0x100 ? (layout & 0xFF) : 0xFFu;
}

inline UINT gfx10_swizzle_block_log2(UINT mode)
{
    switch (mode) {
    case SW_LINEAR: return 0;
    case SW_256B_S:
    case SW_256B_D: return 8;
    case SW_4KB_S:
    case SW_4KB_D:  return 12;
    case SW_64KB_S:
    case SW_64KB_D: return 16;
    default: return 0;
    }
}

struct Gfx10Texture {
    UINT width = 0, height = 0, levels = 0, slices = 0;
    UINT bpe = 0, block_dim = 1;
    UINT swizzle_mode = SW_64KB_S;
    UINT block_log2 = 16;
    UINT tile_width = 0, tile_height = 0, tile_bytes = 0;
    UINT micro_w = 0, micro_h = 0;
    UINT first_tail = 0;
    UINT64 total_slice_bytes = 0;

    std::array<UINT, 256> x_table{};
    std::array<UINT, 256> y_table{};
    std::array<UINT64, 16> mip_start{};
    std::array<UINT, 16> tiles_w{};
    std::array<UINT, 16> tiles_h{};
    std::array<UINT, 16> tail_origin_x{};
    std::array<UINT, 16> tail_origin_y{};

    static bool supports(const D3D12_RESOURCE_DESC& d)
    {
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
        if (d.Width < 1 || d.Height < 1 || d.MipLevels < 1 || d.MipLevels > 16 || d.DepthOrArraySize < 1) return false;
        if (d.SampleDesc.Count != 1) return false;
        if (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) return false;
        const auto fi = gfx10_format_info(d.Format);
        if (fi.bpe == 0) return false;
        const UINT mode = gfx10_swizzle_from_desc(d);
        return mode == SW_LINEAR || mode == SW_256B_S || mode == SW_256B_D ||
               mode == SW_4KB_S || mode == SW_4KB_D || mode == SW_64KB_S || mode == SW_64KB_D;
    }

    static bool is_tiled(const D3D12_RESOURCE_DESC& d)
    {
        const UINT layout = static_cast<UINT>(d.Layout);
        if ((layout & 0xFF00) == 0x100) return (layout & 0xFF) != SW_LINEAR;
        return false;
    }

    Gfx10Texture() = default;
    explicit Gfx10Texture(const D3D12_RESOURCE_DESC& d)
        : width(static_cast<UINT>(d.Width)), height(d.Height),
          levels(d.MipLevels ? d.MipLevels : 1), slices(d.DepthOrArraySize ? d.DepthOrArraySize : 1)
    {
        const auto fi = gfx10_format_info(d.Format);
        bpe = fi.bpe;
        block_dim = fi.block_dim;
        swizzle_mode = gfx10_swizzle_from_desc(d);
        block_log2 = gfx10_swizzle_block_log2(swizzle_mode);

        if (block_log2 == 0) {
            init_linear();
            return;
        }

        UINT element_log2 = 0;
        while ((1u << element_log2) < bpe) ++element_log2;
        const UINT elements_log2 = block_log2 - element_log2;
        const UINT w = (elements_log2 + 1) / 2;
        tile_width = 1u << w;
        tile_height = 1u << (elements_log2 - w);
        tile_bytes = 1u << block_log2;

        static const UINT MICRO_2D[5][2] = {
            {16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}
        };
        micro_w = MICRO_2D[element_log2][0];
        micro_h = MICRO_2D[element_log2][1];

        static const char* const PATTERN_S[5] = {
            "x0x1x2x3y0y1y2y3",
            "--x0x1x2y0y1y2x3",
            "----x0x1y0y1y2x2",
            "------x0y0y1x1x2",
            "--------y0y1x0x1"
        };
        static const char* const PATTERN_D[5] = {
            "x0x1x2y1y0y2x3y3",
            "--x0x1x2y0y1y2x3",
            "----x0x1y0y1x2y2",
            "------x0y0x1x2y1",
            "--------x0y0x1y1"
        };
        const bool display = (swizzle_mode == SW_256B_D || swizzle_mode == SW_4KB_D || swizzle_mode == SW_64KB_D);
        const char* pat = display ? PATTERN_D[element_log2] : PATTERN_S[element_log2];

        std::array<UINT, 16> x_target_bit{};
        std::array<UINT, 16> y_target_bit{};
        UINT next_x = 0, next_y = 0;
        for (UINT bit = 0; bit < 8; ++bit) {
            const char axis = pat[2 * bit];
            const char idx = pat[2 * bit + 1];
            if (axis == 'x') {
                const UINT i = static_cast<UINT>(idx - '0');
                x_target_bit[i] = bit;
                if (i + 1 > next_x) next_x = i + 1;
            } else if (axis == 'y') {
                const UINT i = static_cast<UINT>(idx - '0');
                y_target_bit[i] = bit;
                if (i + 1 > next_y) next_y = i + 1;
            }
        }
        for (UINT bit = 8; bit < block_log2; ++bit) {
            if ((bit & 1) == 0) {
                y_target_bit[next_y++] = bit;
            } else {
                x_target_bit[next_x++] = bit;
            }
        }

        for (UINT x = 0; x < tile_width; ++x) {
            UINT val = 0;
            for (UINT bit = 0; bit < next_x; ++bit) {
                if ((x >> bit) & 1) val |= (1u << x_target_bit[bit]);
            }
            x_table[x] = val;
        }
        for (UINT y = 0; y < tile_height; ++y) {
            UINT val = 0;
            for (UINT bit = 0; bit < next_y; ++bit) {
                if ((y >> bit) & 1) val |= (1u << y_target_bit[bit]);
            }
            y_table[y] = val;
        }

        const UINT tail_max_w = tile_width >> 1;
        const UINT tail_max_h = tile_height;
        const UINT max_in_tail = (block_log2 >= 12) ? (block_log2 - 4) : 0;
        first_tail = levels;

        if (block_log2 >= 12 && levels > 1) {
            for (UINT i = 0; i < levels; ++i) {
                const UINT pw = extent_columns(i);
                const UINT ph = extent_rows(i);
                if (pw <= tail_max_w && ph <= tail_max_h && (levels - i) <= max_in_tail) {
                    first_tail = i;
                    break;
                }
            }
        }

        UINT64 offset = (first_tail < levels) ? tile_bytes : 0;
        for (int i = static_cast<int>(first_tail) - 1; i >= 0; --i) {
            const UINT cols = extent_columns(static_cast<UINT>(i));
            const UINT r = extent_rows(static_cast<UINT>(i));
            tiles_w[i] = (cols + tile_width - 1) / tile_width;
            tiles_h[i] = (r + tile_height - 1) / tile_height;
            mip_start[i] = offset;
            offset += UINT64{tiles_w[i]} * tiles_h[i] * tile_bytes;
        }
        total_slice_bytes = offset;

        if (first_tail < levels) {
            for (UINT i = first_tail; i < levels; ++i) {
                const UINT m = max_in_tail - 1 - (i - first_tail);
                const UINT tail_offset = (m > 6) ? (16u << m) : (m << 8);
                UINT x_idx = 0, y_idx = 0;
                for (UINT b = 0; b < 6; ++b) {
                    y_idx |= ((tail_offset >> (8 + 2 * b)) & 1) << b;
                    x_idx |= ((tail_offset >> (9 + 2 * b)) & 1) << b;
                }
                tail_origin_x[i] = x_idx * micro_w;
                tail_origin_y[i] = y_idx * micro_h;
                mip_start[i] = tail_offset;
            }
        }
    }

    void init_linear()
    {
        tile_width = 256 / bpe;
        tile_height = 1;
        tile_bytes = 256;
        first_tail = levels;
        UINT64 offset = 0;
        for (int i = static_cast<int>(levels) - 1; i >= 0; --i) {
            const UINT cols = extent_columns(static_cast<UINT>(i));
            const UINT r = extent_rows(static_cast<UINT>(i));
            tiles_w[i] = (cols + tile_width - 1) / tile_width;
            tiles_h[i] = r;
            const UINT64 pitch = UINT64{tiles_w[i]} * 256;
            mip_start[i] = offset;
            offset += pitch * r;
        }
        total_slice_bytes = offset;
    }

    // Extents for layout allocation and tile strides (AMD addrlib GetMipSize / ShiftCeil)
    UINT extent_columns(UINT mip) const
    {
        const UINT cols0 = (width + block_dim - 1) / block_dim;
        return (std::max)(1u, (cols0 + (1u << mip) - 1) >> mip);
    }
    UINT extent_rows(UINT mip) const
    {
        const UINT rows0 = (height + block_dim - 1) / block_dim;
        return (std::max)(1u, (rows0 + (1u << mip) - 1) >> mip);
    }

    // Subresource dimensions in elements for texel copying and bounds checking
    UINT columns(UINT mip) const
    {
        return ((std::max)(1u, width >> mip) + block_dim - 1) / block_dim;
    }
    UINT rows(UINT mip) const
    {
        return ((std::max)(1u, height >> mip) + block_dim - 1) / block_dim;
    }
    UINT tiles_wide(UINT mip) const
    {
        return (mip < first_tail) ? tiles_w[mip] : (extent_columns(mip) + tile_width - 1) / tile_width;
    }
    UINT tiles_high(UINT mip) const
    {
        return (mip < first_tail) ? tiles_h[mip] : (extent_rows(mip) + tile_height - 1) / tile_height;
    }
    UINT padded_columns(UINT mip) const
    {
        if (mip >= first_tail) {
            const UINT tail_max_w = tile_width >> 1;
            return (std::max)(micro_w, tail_max_w >> (mip - first_tail));
        }
        return tiles_w[mip] * tile_width;
    }
    UINT padded_rows(UINT mip) const
    {
        if (mip >= first_tail) {
            const UINT tail_max_h = tile_height;
            return (std::max)(micro_h, tail_max_h >> (mip - first_tail));
        }
        return tiles_h[mip] * tile_height;
    }

    UINT64 bytes() const { return total_slice_bytes * slices; }
    UINT64 slice_bytes() const { return total_slice_bytes; }
    UINT64 alignment() const { return (block_log2 > 0) ? (1ull << block_log2) : 256; }
    UINT64 start(UINT mip) const { return mip_start[mip]; }

    UINT64 address(UINT mip, UINT x, UINT y, UINT slice) const
    {
        const UINT64 slice_base = UINT64{slice} * total_slice_bytes;
        if (swizzle_mode == SW_LINEAR) {
            const UINT64 pitch = UINT64{tiles_w[mip]} * 256;
            return slice_base + mip_start[mip] + UINT64{y} * pitch + UINT64{x} * bpe;
        }
        if (mip >= first_tail) {
            const UINT tx = tail_origin_x[mip] + x;
            const UINT ty = tail_origin_y[mip] + y;
            return slice_base + (x_table[tx] | y_table[ty]);
        }
        const UINT64 tile_idx = UINT64{y / tile_height} * tiles_w[mip] + (x / tile_width);
        const UINT within = x_table[x % tile_width] | y_table[y % tile_height];
        return slice_base + mip_start[mip] + tile_idx * tile_bytes + within;
    }

    template<bool into>
    void copy_subresource(UINT mip, UINT slice, unsigned char* dst, const unsigned char* src, UINT pitch) const
    {
        const UINT cols = columns(mip);
        const UINT r = rows(mip);
        for (UINT y = 0; y < r; ++y) {
            const std::size_t linear_row = static_cast<std::size_t>(y) * pitch;
            for (UINT x = 0; x < cols; ++x) {
                const UINT64 tiled = address(mip, x, y, slice);
                const std::size_t linear = linear_row + static_cast<std::size_t>(x) * bpe;
                if constexpr (into) {
                    std::memcpy(dst + tiled, src + linear, bpe);
                } else {
                    std::memcpy(dst + linear, src + tiled, bpe);
                }
            }
        }
    }
};

}  // namespace x2w_startup
