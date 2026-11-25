// Experimental (docs/startup-runtime.md): the format of the view the window presenter samples.
//
// The presenter draws the presented surface with a shader that reads a float4. A texture created
// with a typeless format cannot be viewed with that format: a shader resource view needs a typed
// one. The 8-bit and 10:10:10:2 colour families are viewed as UNORM, the only member that a float4
// read takes unchanged; no sRGB conversion is added, because that is a colour choice. The 16-bit
// four-channel family (R16G16B16A16_TYPELESS) holds either half-float (HDR scene) or UNORM texels and
// the format does not say which; a wrong view reads garbage. It is therefore viewed only when the caller
// names the reading: DXGI_FORMAT_R16G16B16A16_FLOAT or DXGI_FORMAT_R16G16B16A16_UNORM (a known typed
// producer view, or the explicit diagnostic choice X2W_PRESENT_RGBA16_VIEW=float|unorm). With no choice it
// is refused. A float4 read takes half-float texels unchanged, including values above 1 and below 0 (the
// window's 8-bit back buffer clamps them; no tone mapping is added, because that is a colour choice too).
// A typeless format that still names several readings (32 bits a channel, 16 bits in fewer channels: float,
// unorm, snorm, uint, sint; depth/stencil planes;
// block-compressed data) is refused instead of guessed. A typed format passes unchanged.
// DXGI_FORMAT_UNKNOWN means refused.
#pragma once

#include <d3d12.h>

#include <cstdlib>
#include <cstring>

namespace x2w_startup {
// The explicit diagnostic choice for typeless 16-bit four-channel surfaces: X2W_PRESENT_RGBA16_VIEW=float|unorm.
// Unset or any other value: UNKNOWN (not chosen).
inline DXGI_FORMAT present_rgba16_choice()
{
    const char* value = std::getenv("X2W_PRESENT_RGBA16_VIEW");
    if (value && !std::strcmp(value, "float")) return DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (value && !std::strcmp(value, "unorm")) return DXGI_FORMAT_R16G16B16A16_UNORM;
    return DXGI_FORMAT_UNKNOWN;
}

// `rgba16_choice` is only read for R16G16B16A16_TYPELESS; anything but FLOAT or UNORM means "not chosen".
inline DXGI_FORMAT present_view_format(DXGI_FORMAT format, DXGI_FORMAT rgba16_choice = DXGI_FORMAT_UNKNOWN)
{
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: return DXGI_FORMAT_B8G8R8X8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return rgba16_choice == DXGI_FORMAT_R16G16B16A16_FLOAT || rgba16_choice == DXGI_FORMAT_R16G16B16A16_UNORM
                   ? rgba16_choice : DXGI_FORMAT_UNKNOWN;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_UNKNOWN: return DXGI_FORMAT_UNKNOWN;
    default: return format;
    }
}

inline bool present_surface_supported(const D3D12_RESOURCE_DESC& desc, DXGI_FORMAT rgba16_choice = DXGI_FORMAT_UNKNOWN)
{
    return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
           desc.SampleDesc.Count == 1 && desc.Width != 0 && desc.Height != 0 &&
           (desc.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) == 0 &&
           present_view_format(desc.Format, rgba16_choice) != DXGI_FORMAT_UNKNOWN;
}
}  // namespace x2w_startup
