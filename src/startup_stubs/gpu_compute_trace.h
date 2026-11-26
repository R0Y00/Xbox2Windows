#pragma once

#include "startup_stubs/draw_bmp.h"
#include "startup_stubs/list_copy.h"
#include "startup_stubs/gpu_material_trace.h"
#include "startup_stubs/gpu_indirect_capture.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace x2w_compute_trace {
using Microsoft::WRL::ComPtr;

inline bool supported(DXGI_FORMAT f)
{
    return x2w_draw_bmp::trace_supported(f) || f == DXGI_FORMAT_R32G32B32A32_FLOAT || f == DXGI_FORMAT_R32_FLOAT || f == DXGI_FORMAT_R32_UINT ||
           f == DXGI_FORMAT_R16_FLOAT || f == DXGI_FORMAT_R16_UINT || f == DXGI_FORMAT_R8_UNORM ||
           f == DXGI_FORMAT_R8_UINT || f == DXGI_FORMAT_R8G8B8A8_TYPELESS || f == DXGI_FORMAT_R10G10B10A2_UNORM;
}

inline bool depth_format(DXGI_FORMAT f)
{
    return f == DXGI_FORMAT_R32_TYPELESS || f == DXGI_FORMAT_D32_FLOAT || f == DXGI_FORMAT_R32G8X24_TYPELESS ||
           f == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
}

struct Binding {
    ID3D12Resource* texture{};
    UINT slot{}, offset{};
    bool write{}, priority{};
    UINT mip{}, slice{};
    DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
    UINT plane{};
};
struct Bindings {
    std::array<Binding, 32> values{};
    UINT count = 0, reads = 0, writes = 0;
    void add(Binding b, bool priority = false, bool wide = false)
    {
        const UINT limit = wide ? 16u : 3u;
        if (!b.texture || (!priority && (b.write ? writes : reads) >= limit)) return;
        b.priority = priority;
        const auto d = b.texture->GetDesc();
        const bool flat = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.DepthOrArraySize == 1;
        const bool depth = depth_format(d.Format) || (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        const auto format = b.view_format == DXGI_FORMAT_UNKNOWN ? d.Format : b.view_format;
        if (d.SampleDesc.Count != 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
            (!wide && ((!flat && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D) || d.MipLevels != 1)) ||
            (depth ? ((!priority && !wide) || b.write || !depth_format(d.Format)) : !supported(format))) return;
        for (UINT i = 0; i < count; ++i)
            if (values[i].texture == b.texture && values[i].write == b.write && values[i].mip == b.mip &&
                values[i].slice == b.slice && values[i].plane == b.plane && values[i].view_format == b.view_format) {
                if (priority && !values[i].priority) values[i] = b;
                return;
            }
        if ((b.write ? writes : reads) >= limit) {
            for (UINT i = count; i > 0; --i) {
                auto& old = values[i - 1];
                if (old.write == b.write && !old.priority) { old = b; return; }
            }
            return;
        }
        values[count++] = b;
        ++(b.write ? writes : reads);
    }
};

struct Image {
    struct Focus { UINT x, y; }; // thousandths of the full image
    static D3D12_BOX color_box(UINT width, UINT height, Focus focus)
    {
        // One comparison covers the same screen area on both sides of a 1440p to 4K draw.
        const UINT w = (std::min)(256u, (width + 14) / 15);
        const UINT h = (std::min)(128u, (height * 8 + 134) / 135);
        const auto origin = [](UINT extent, UINT length, UINT center) {
            const UINT pixel = (std::min)(1000u, center) * extent / 1000;
            return (std::min)(extent - length, pixel > length / 2 ? pixel - length / 2 : 0u);
        };
        const UINT x = origin(width, w, focus.x), y = origin(height, h, focus.y);
        return {x,y,0,x+w,y+h,1};
    }
    Binding binding;
    ComPtr<ID3D12Resource> source, pre, post;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    D3D12_BOX box{};
    UINT64 row_bytes = 0, size = 0;
    UINT source_depth = 1;
    UINT source_subresource{};
    bool sparse = false, full_depth_plane = false;
    UINT width() const { return box.right - box.left; }
    UINT height() const { return box.bottom - box.top; }
    UINT64 output_row_bytes() const { return row_bytes / footprint.Footprint.Width * width(); }
    const unsigned char* row(const unsigned char* pixels, UINT y, UINT z) const
    {
        const auto& layout = footprint.Footprint;
        const UINT64 row_index = UINT64(z) * layout.Height + y + (full_depth_plane ? box.top : 0);
        return pixels + row_index * layout.RowPitch + (full_depth_plane ? box.left * (row_bytes / layout.Width) : 0);
    }
    UINT source_z(UINT z) const { return sparse && z > 0 ? (z == 1 ? source_depth / 2 : source_depth - 1) : z; }
    bool init(ID3D12Device* device, Binding b, const Focus* focus = nullptr)
    {
        binding = b;
        source = b.texture;
        auto d = source->GetDesc();
        const auto original = d;
        const bool volume = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if (b.plane) {
            D3D12_FEATURE_DATA_FORMAT_INFO info{d.Format};
            if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info))) || b.plane >= info.PlaneCount) return false;
        }
        const UINT slices = volume ? 1u : d.DepthOrArraySize;
        source_subresource = b.mip + b.slice * d.MipLevels + b.plane * slices * d.MipLevels;
        if (b.view_format != DXGI_FORMAT_UNKNOWN) {
            if (!x2w_startup::compatible_formats(d.Format, b.view_format)) return false;
            d.Format = b.view_format;
        }
        if (b.mip >= d.MipLevels || (!volume && b.slice >= d.DepthOrArraySize)) return false;
        d.Width = (std::max)(UINT64{1}, d.Width >> b.mip);
        d.Height = (std::max)(1u, d.Height >> b.mip);
        if (volume) d.DepthOrArraySize = static_cast<UINT16>((std::max)(1, d.DepthOrArraySize >> b.mip));
        UINT width = static_cast<UINT>((std::min)(d.Width, volume ? UINT64{64} : UINT64{256}));
        UINT height = (std::min)(d.Height, volume ? 64u : 128u);
        source_depth = volume ? d.DepthOrArraySize : 1;
        sparse = volume && source_depth > 3 && (d.Width > width || d.Height > height || source_depth > 64);
        const UINT x = (static_cast<UINT>(d.Width) - width) / 2, y = (d.Height - height) / 2;
        box = {x, y, 0, x + width, y + height, source_depth};
        const bool screen_aspect = d.Width * 9 == UINT64{d.Height} * 16;
        if (focus && !volume && ((d.Width >= 1024 && d.Height >= 512) || (screen_aspect && d.Height >= 90))) {
            box = color_box(UINT(d.Width), d.Height, *focus);
            width = box.right - box.left;
            height = box.bottom - box.top;
        }
        full_depth_plane = depth_format(d.Format) || (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        if (!full_depth_plane) {
            d.Width = width;
            d.Height = height;
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            d.Alignment = 0;
            d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            d.MipLevels = 1;
        }
        // Depth-stencil planes must be copied whole. Extract the center from the completed readback.
        const UINT64 limit = full_depth_plane ? 32ull << 20 : (volume ? 1024ull : 512ull) << 10;
        const auto layout = [&] {
            if (full_depth_plane) {
                device->GetCopyableFootprints(&original, source_subresource, 1, 0, &footprint, nullptr, &row_bytes, &size);
                return;
            }
            d.DepthOrArraySize = static_cast<UINT16>(sparse ? 3 : source_depth);
            device->GetCopyableFootprints(&d, 0, 1, 0, &footprint, nullptr, &row_bytes, &size);
        };
        layout();
        if (volume && size > limit && !sparse && source_depth > 3) {
            sparse = true;
            layout();
        }
        if (!size || size > limit) return false;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC buf{};
        buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf.Width = size;
        buf.Height = buf.DepthOrArraySize = buf.MipLevels = buf.SampleDesc.Count = 1;
        buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        auto make = [&](ComPtr<ID3D12Resource>& out) {
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buf,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&out)));
        };
        return make(pre) && (!b.write || make(post));
    }
    void record(ID3D12GraphicsCommandList* list, bool after) const
    {
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = after ? post.Get() : pre.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        src.pResource = source.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = source_subresource;
        if (!sparse) {
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, full_depth_plane ? nullptr : &box);
            return;
        }
        for (UINT z = 0; z < footprint.Footprint.Depth; ++z) {
            auto plane = box;
            plane.front = source_z(z);
            plane.back = plane.front + 1;
            list->CopyTextureRegion(&dst, 0, 0, z, &src, &plane);
        }
    }
};

struct Capture {
    bool pixel = false;
    bool paired = false;
    IndirectArguments indirect;
    x2w_material_trace::Probe material;
    ComPtr<ID3D12DescriptorHeap> feedback_heap;
    ComPtr<ID3D12Resource> feedback_source;
    long request{}, frame{};
    std::string key;
    std::vector<Image> images;
    std::vector<std::pair<ID3D12Fence*, UINT64>> fences;
    bool published = false, unfenced = false;
};
}
