#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>

namespace x2w_depth_twins {
using Microsoft::WRL::ComPtr;

struct Twins {
    ComPtr<ID3D12Resource> depth, colour, depth_plane, through;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> stencil_rows;

    bool create(ID3D12Device* device, ID3D12Resource* original, const D3D12_RESOURCE_DESC& title)
    {
        constexpr auto both = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (!original || (title.Flags & both) != both) return false;
        auto d = original->GetDesc();
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1) return false;
        const auto of = d;
        bool stencil = false;
        switch (d.Format) {
        case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: d.Format = DXGI_FORMAT_R32_TYPELESS; break;
        case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: d.Format = DXGI_FORMAT_R16_TYPELESS; break;
        case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT:
            stencil = true;
            d.Format = DXGI_FORMAT_R8_TYPELESS;
            break;
        default: return false;
        }
        d.Alignment = 0;
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto allocate = [&](const D3D12_RESOURCE_DESC& desc, ComPtr<ID3D12Resource>& result) {
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                              nullptr, IID_PPV_ARGS(&result)));
        };
        if (!allocate(d, colour)) return false;
        if (stencil) {
            const UINT planes = UINT{of.MipLevels} * of.DepthOrArraySize;
            stencil_rows.resize(planes);
            UINT64 bytes = 0;
            device->GetCopyableFootprints(&of, planes, planes, 0, stencil_rows.data(), nullptr, nullptr, &bytes);
            if (!bytes || bytes == UINT64_MAX) return false;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = bytes;
            buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (!allocate(buffer, through)) return false;
            const auto format = static_cast<UINT>(title.Format);
            d.Format = format >= 118 && format <= 120 ? DXGI_FORMAT_R16_TYPELESS : DXGI_FORMAT_R32_TYPELESS;
            if (!allocate(d, depth_plane)) return false;
        }
        depth = original;
        return true;
    }

    // Extract the selected depth/stencil plane into a single-plane host texture.
    ID3D12Resource* view(const D3D12_UNORDERED_ACCESS_VIEW_DESC* input, D3D12_UNORDERED_ACCESS_VIEW_DESC& out) const
    {
        if (!depth || !colour) return nullptr;
        out = input ? *input : D3D12_UNORDERED_ACCESS_VIEW_DESC{};
        if (!input) {
            const UINT slices = depth->GetDesc().DepthOrArraySize;
            out.ViewDimension = slices > 1 ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY : D3D12_UAV_DIMENSION_TEXTURE2D;
            if (slices > 1) out.Texture2DArray.ArraySize = slices;
        }
        UINT* plane = nullptr;
        if (out.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D) plane = &out.Texture2D.PlaneSlice;
        if (out.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY) plane = &out.Texture2DArray.PlaneSlice;
        if (!plane || *plane > 1) return nullptr;
        if (through && (*plane == 1 || out.Format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT || out.Format == DXGI_FORMAT_X24_TYPELESS_G8_UINT)) {
            *plane = 0;
            out.Format = DXGI_FORMAT_R8_UINT;
            return colour.Get();
        }
        if (*plane != 0) return nullptr;
        ID3D12Resource* backing = through ? depth_plane.Get() : colour.Get();
        if (!backing) return nullptr;
        const DXGI_FORMAT f = out.Format;
        const bool depth_stencil = (f >= 19 && f <= 21) || (f >= 44 && f <= 46);
        const bool steps = backing->GetDesc().Format == DXGI_FORMAT_R16_TYPELESS;
        if (f == DXGI_FORMAT_UNKNOWN || depth_stencil || (steps && (f == DXGI_FORMAT_R16_TYPELESS || f == DXGI_FORMAT_D16_UNORM)) ||
            (!steps && (f == DXGI_FORMAT_R32_TYPELESS || f == DXGI_FORMAT_D32_FLOAT)))
            out.Format = steps ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R32_FLOAT;
        return backing;
    }

    // One-plane depth is a bit-compatible copy. Stencil uses the depth resource's plane footprints,
    // including row alignment, for both legs. The caller owns the states of depth and colour.
    template<class SetState>
    void copy_colour(ID3D12GraphicsCommandList* list, bool to_colour, SetState&& state) const
    {
        ID3D12Resource* from = to_colour ? depth.Get() : colour.Get();
        ID3D12Resource* to = to_colour ? colour.Get() : depth.Get();
        state(from, D3D12_RESOURCE_STATE_COPY_SOURCE);
        state(to, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!through) {
            list->CopyResource(to, from);
            return;
        }
        const UINT plane_count = static_cast<UINT>(stencil_rows.size());
        const auto leg = [&](bool into_buffer) {
            state(through.Get(), into_buffer ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE);
            for (UINT i = 0; i < plane_count; ++i) {
                D3D12_TEXTURE_COPY_LOCATION buffer{through.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                buffer.PlacedFootprint = stencil_rows[i];
                D3D12_TEXTURE_COPY_LOCATION texture{into_buffer ? from : to, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                texture.SubresourceIndex = i + (texture.pResource == depth.Get() ? plane_count : 0);
                list->CopyTextureRegion(into_buffer ? &buffer : &texture, 0, 0, 0, into_buffer ? &texture : &buffer, nullptr);
            }
        };
        leg(true);
        leg(false);
    }
};
}
