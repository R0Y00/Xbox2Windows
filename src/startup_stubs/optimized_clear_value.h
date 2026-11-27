// The optimized clear value passed to CreateCommittedResource/CreatePlacedResource, checked against the
// resource that is really created. D3D12 (ID3D12Device::CreateCommittedResource, D3D12_CLEAR_VALUE): a clear
// value is only for a texture that allows render-target or depth-stencil use, its format must be fully typed,
// and a depth-stencil clear needs a depth format with Depth in [0, 1]. Otherwise the call fails (or the debug
// layer reports it), so the value is dropped, or its depth clamped, before the call.
#pragma once

#include <d3d12.h>

namespace x2w_startup {
enum class ClearDrop { none, buffer, flags, format, kind };
inline bool clear_depth_format(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_D32_FLOAT || f == DXGI_FORMAT_D16_UNORM ||
           f == DXGI_FORMAT_D24_UNORM_S8_UINT || f == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
}
inline bool clear_typeless_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_UNKNOWN: case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: case DXGI_FORMAT_R32G32_TYPELESS: case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: case DXGI_FORMAT_X24_TYPELESS_G8_UINT: case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC7_TYPELESS: return true;
    default: return false;
    }
}
// Returns `mapped`, `&copy` (depth clamped) or null (dropped; `reason` says why).
inline const D3D12_CLEAR_VALUE* optimized_clear_value(const D3D12_CLEAR_VALUE* mapped,
        const D3D12_RESOURCE_DESC& created, D3D12_CLEAR_VALUE& copy, ClearDrop* reason = nullptr) {
    ClearDrop why = ClearDrop::none;
    if (!mapped) return nullptr;
    const bool ds = (created.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
    const bool rt = (created.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0;
    if (created.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) why = ClearDrop::buffer;
    else if (!rt && !ds) why = ClearDrop::flags;
    else if (clear_typeless_format(mapped->Format)) why = ClearDrop::format;
    else if (clear_depth_format(mapped->Format) != ds) why = ClearDrop::kind;
    if (reason) *reason = why;
    if (why != ClearDrop::none) return nullptr;
    if (ds && (!(mapped->DepthStencil.Depth >= 0.0f) || mapped->DepthStencil.Depth > 1.0f)) {
        copy = *mapped;
        copy.DepthStencil.Depth = !(mapped->DepthStencil.Depth >= 0.0f) ? 0.0f : 1.0f;
        return &copy;
    }
    return mapped;
}
}  // namespace x2w_startup
