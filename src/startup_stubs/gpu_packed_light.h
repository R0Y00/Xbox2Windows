#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include "memory_srv_buffer.h"

namespace x2w_packed_light {
// One table-binding pass, ending before the title's draw or dispatch can write.
struct BoundCopy {
    ID3D12Resource* source = nullptr;
    ID3D12Resource* destination = nullptr;

    bool contains(ID3D12Resource* from, ID3D12Resource* to) const
    {
        return source == from && destination == to;
    }
    void invalidate(ID3D12Resource* texture)
    {
        if (texture == source || texture == destination) *this = {};
    }
};

inline bool shape(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b)
{
    return a.Dimension == b.Dimension && (a.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        a.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) && a.Width == b.Width && a.Height == b.Height &&
        a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels && a.MipLevels == b.MipLevels &&
        a.SampleDesc.Count == 1 && b.SampleDesc.Count == 1 && !a.SampleDesc.Quality && !b.SampleDesc.Quality;
}

// Typed, uncompressed, single-plane color formats. Packed and planar YUV are excluded.
inline UINT color_bytes(DXGI_FORMAT format)
{
    switch (format) {
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return 4;
    case DXGI_FORMAT_A8_UNORM: return 1;
    default: return x2w_startup::typed_buffer_format_bytes(format);
    }
}

// A native resource already passed device size limits. Without the title allocation
// size, its unpadded texels still give a lower bound on the occupied byte range.
inline UINT64 minimum_span(const D3D12_RESOURCE_DESC& desc)
{
    UINT64 width = desc.Width, height = desc.Height, depth = desc.DepthOrArraySize, bytes = 0;
    for (UINT mip = 0; mip < desc.MipLevels; ++mip) {
        bytes += width * height * depth * color_bytes(desc.Format);
        if (width > 1) width >>= 1;
        if (height > 1) height >>= 1;
        if (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && depth > 1) depth >>= 1;
    }
    return bytes;
}

inline bool compatible(const D3D12_RESOURCE_DESC& writer, const D3D12_RESOURCE_DESC& reader)
{
    const UINT count = writer.MipLevels * (writer.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : writer.DepthOrArraySize);
    const UINT bytes = color_bytes(writer.Format);
    return count && count <= 128 && bytes && bytes == color_bytes(reader.Format) && writer.Format != reader.Format &&
        (writer.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) &&
        !(writer.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) &&
        !(reader.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) && shape(writer, reader);
}

struct Copy {
    LONG last_trace_frame{};
    Microsoft::WRL::ComPtr<ID3D12Resource> source, destination, scratch;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> source_rows, dest_rows;

    bool init(ID3D12Device* device, ID3D12Resource* from, ID3D12Resource* to, ID3D12Resource* reusable = nullptr)
    {
        const auto a = from->GetDesc(), b = to->GetDesc();
        if (!compatible(a, b)) return false;
        const UINT count = a.MipLevels * (a.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : a.DepthOrArraySize);

        source_rows.resize(count);
        dest_rows.resize(count);
        UINT64 bytes = 0, dest_bytes = 0;
        device->GetCopyableFootprints(&a, 0, count, 0, source_rows.data(), nullptr, nullptr, &bytes);
        device->GetCopyableFootprints(&b, 0, count, 0, dest_rows.data(), nullptr, nullptr, &dest_bytes);
        if (!bytes || bytes != dest_bytes || bytes == UINT64_MAX) return false;
        for (UINT i = 0; i < count; ++i) {
            const auto& x = source_rows[i];
            const auto& y = dest_rows[i];
            if (x.Offset != y.Offset || x.Footprint.RowPitch != y.Footprint.RowPitch ||
                x.Footprint.Depth != y.Footprint.Depth || x.Footprint.Height != y.Footprint.Height ||
                x.Footprint.Width != y.Footprint.Width) return false;
        }
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (reusable && reusable->GetDesc().Width >= bytes) scratch = reusable;
        else if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON,
                                                         nullptr, IID_PPV_ARGS(&scratch)))) return false;
        source = from;
        destination = to;
        return true;
    }

    template<class Require>
    void record(ID3D12GraphicsCommandList* list, Require require, D3D12_RESOURCE_STATES previous_source_state)
    {
        auto transfer = [&](bool to_buffer) {
            auto& rows = to_buffer ? source_rows : dest_rows;
            for (UINT i = 0; i < rows.size(); ++i) {
                D3D12_TEXTURE_COPY_LOCATION linear{}, texture{};
                linear.pResource = scratch.Get();
                linear.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                linear.PlacedFootprint = rows[i];
                texture.pResource = to_buffer ? source.Get() : destination.Get();
                texture.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                texture.SubresourceIndex = i;
                if (to_buffer) list->CopyTextureRegion(&linear, 0, 0, 0, &texture, nullptr);
                else list->CopyTextureRegion(&texture, 0, 0, 0, &linear, nullptr);
            }
        };
        require(source.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        require(scratch.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        transfer(true);
        require(source.Get(), previous_source_state);
        require(scratch.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        require(destination.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        transfer(false);
        require(scratch.Get(), D3D12_RESOURCE_STATE_COMMON);
    }
};
}
