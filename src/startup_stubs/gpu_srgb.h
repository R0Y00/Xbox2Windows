#pragma once
#include <d3d12.h>

namespace x2w_srgb {
inline DXGI_FORMAT family(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8X8_TYPELESS;
    default: return f;
    }
}

inline DXGI_FORMAT linear(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8X8_UNORM;
    default: return f;
    }
}

inline DXGI_FORMAT backing_format(const D3D12_RESOURCE_DESC& d)
{
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER && (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
        return family(d.Format);
    return d.Format;
}

// These resources allow unordered access, which rules out multisampling. A default view addresses
// all slices and the first mip for a target, or all mips for a shader resource.
inline D3D12_SHADER_RESOURCE_VIEW_DESC srv(const D3D12_RESOURCE_DESC& d, DXGI_FORMAT format)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    const UINT slices = d.DepthOrArraySize;
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && slices == 1) {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D; v.Texture1D.MipLevels = d.MipLevels;
    } else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY; v.Texture1DArray.MipLevels = d.MipLevels; v.Texture1DArray.ArraySize = slices;
    } else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D; v.Texture3D.MipLevels = d.MipLevels;
    } else if (slices == 1) {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; v.Texture2D.MipLevels = d.MipLevels;
    } else {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY; v.Texture2DArray.MipLevels = d.MipLevels; v.Texture2DArray.ArraySize = slices;
    }
    return v;
}

inline D3D12_UNORDERED_ACCESS_VIEW_DESC uav(const D3D12_RESOURCE_DESC& d, DXGI_FORMAT format)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
    v.Format = linear(format);
    const UINT slices = d.DepthOrArraySize;
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && slices == 1) v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
    else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY; v.Texture1DArray.ArraySize = slices;
    } else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D; v.Texture3D.WSize = slices;
    } else if (slices == 1) v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    else { v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY; v.Texture2DArray.ArraySize = slices; }
    return v;
}

inline D3D12_RENDER_TARGET_VIEW_DESC rtv(const D3D12_RESOURCE_DESC& d, DXGI_FORMAT format)
{
    D3D12_RENDER_TARGET_VIEW_DESC v{};
    v.Format = format;
    const UINT slices = d.DepthOrArraySize;
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && slices == 1) v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;
    else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) {
        v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY; v.Texture1DArray.ArraySize = slices;
    } else if (d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
        v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE3D; v.Texture3D.WSize = slices;
    } else if (slices == 1) v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    else { v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY; v.Texture2DArray.ArraySize = slices; }
    return v;
}
} // namespace x2w_srgb
