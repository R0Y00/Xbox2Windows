// The format of a D3D12_CLEAR_VALUE the title passes with a resource, mapped to the desktop format.
//
// The resource description already goes through desktop_format() before it reaches desktop D3D12, but the
// clear value used to be forwarded as given. A console-only format value in it (116 and 117, two 10-bit
// float formats; 118, D16_UNORM_S8_UINT) is refused by desktop (debug layer messages 816 and 817), and the
// creation then only succeeded in the retry without a clear value. The same mapping as for the resource
// is applied here so the clear value matches the resource it belongs to.
//
// 119 and 120 (the typed views of the depth plane) are not clear value formats on desktop either and are
// left unchanged, so the existing fallback (create without a clear value) still applies to them.
#pragma once

#include <d3d12.h>

namespace x2w_startup {

inline DXGI_FORMAT desktop_clear_format(DXGI_FORMAT format)
{
    switch (static_cast<UINT>(format)) {
    case 116:
    case 117:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case 118:
        return DXGI_FORMAT_D24_UNORM_S8_UINT;
    default:
        return format;
    }
}

// A copy of `clear` with its format mapped; the colour or depth and stencil values are kept as they are.
inline D3D12_CLEAR_VALUE desktop_clear_value(const D3D12_CLEAR_VALUE& clear)
{
    D3D12_CLEAR_VALUE copy = clear;
    copy.Format = desktop_clear_format(clear.Format);
    return copy;
}

}  // namespace x2w_startup
