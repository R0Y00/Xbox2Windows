// Experimental (docs/startup-runtime.md): whether a view description's ViewDimension suits the
// resource it is made on. Desktop D3D12 removes the device (DXGI_ERROR_INVALID_CALL, debug-layer
// message "The ViewDimension in the View Desc is incompatible with the type of the Resource")
// instead of failing the call, so the stand-in checks first and skips such a view with a log.
// Values are the public d3d12.h enumerations. Pure, no Windows calls, so a control can check it.
#pragma once

namespace x2w_view_dimension {

enum ResourceDimension : unsigned { unknown = 0, buffer = 1, texture1d = 2, texture2d = 3, texture3d = 4 };

// Which resource dimension one view dimension needs, by the device slot that makes the view
// (18 shader resource, 19 unordered access, 20 render target, 21 depth stencil). unknown when the
// view dimension is not one of that view type's values (or is 0); such a view is not judged here.
inline unsigned needed(int slot, unsigned view_dimension) noexcept
{
    switch (slot) {
    case 18:  // D3D12_SRV_DIMENSION
        switch (view_dimension) {
        case 1: return buffer;
        case 2: case 3: return texture1d;
        case 4: case 5: case 6: case 7: case 9: case 10: return texture2d;  // 2D, arrays, MS, cube, cube array
        case 8: return texture3d;
        default: return unknown;  // 11 is a raytracing structure, made with a null resource
        }
    case 19:  // D3D12_UAV_DIMENSION
    case 20:  // D3D12_RTV_DIMENSION (same numbering)
        switch (view_dimension) {
        case 1: return buffer;
        case 2: case 3: return texture1d;
        case 4: case 5: case 6: case 7: return texture2d;
        case 8: return texture3d;
        default: return unknown;
        }
    case 21:  // D3D12_DSV_DIMENSION
        switch (view_dimension) {
        case 1: case 2: return texture1d;
        case 3: case 4: case 5: case 6: return texture2d;
        default: return unknown;
        }
    default:
        return unknown;
    }
}

// False only when the view dimension names a resource dimension other than the resource's.
inline bool fits(int slot, unsigned view_dimension, unsigned resource_dimension) noexcept
{
    const unsigned need = needed(slot, view_dimension);
    return need == unknown || need == resource_dimension;
}

// A description with ViewDimension 0 (UNKNOWN) is always refused by desktop D3D12. Measured: a
// title module makes one on a two-dimensional RGBA8 texture (format R32G32B32A32_FLOAT, default
// component mapping). The stand-in makes the resource's default view instead (a null description),
// which desktop allows only for a texture with a typed format, and for a shader resource view not
// a depth format. Whether the console gives UNKNOWN that meaning is a hypothesis.
inline bool typeless(unsigned format) noexcept
{
    switch (format) {
    case 1: case 5: case 9: case 15: case 19: case 23: case 27: case 33: case 39: case 44: case 48: case 53: case 60:
    case 70: case 73: case 76: case 79: case 82: case 90: case 92: case 94: case 97:
        return true;
    default:
        return false;
    }
}

inline bool depth_format(unsigned format) noexcept { return format == 20 || format == 40 || format == 45 || format == 55; }

// Which slots may get the default view. It is measured only for shader resource views (slot 18) of
// typed textures, so that is the default. The default view also drops the requested format and
// component mapping, so for the other view types it stays an explicit experiment
// (X2W_D3D12_VIEW_DEFAULTS=all). off: every unknown-dimension view is skipped.
enum class DefaultViews : unsigned { off, srv, all };

// X2W_D3D12_VIEW_DEFAULTS: "all" or "off"; anything else, or unset, is srv.
inline DefaultViews default_views_mode(const char* value) noexcept
{
    if (!value) return DefaultViews::srv;
    if (value[0] == 'a' && value[1] == 'l' && value[2] == 'l' && value[3] == 0) return DefaultViews::all;
    if (value[0] == 'o' && value[1] == 'f' && value[2] == 'f' && value[3] == 0) return DefaultViews::off;
    return DefaultViews::srv;
}

inline bool default_view_allowed(int slot, unsigned resource_dimension, unsigned resource_format, DefaultViews mode) noexcept
{
    if (slot < 18 || slot > 21) return false;
    if (mode == DefaultViews::off || (mode == DefaultViews::srv && slot != 18)) return false;
    if (resource_dimension == unknown || resource_dimension == buffer) return false;
    if (resource_format == 0 || typeless(resource_format)) return false;
    if (slot == 18 && depth_format(resource_format)) return false;
    return true;
}

// The dimension of the null SRV that replaces a refused one in its destination slot, so that the slot
// cannot keep the view it held before (a skipped call would otherwise leave that descriptor in place,
// bindable and copyable, after the slot's record has dropped its resource). The requested dimension
// when it is a texture dimension (2 to 10), else texture 2D (4).
inline unsigned refused_srv_null_dimension(unsigned requested) noexcept
{
    return requested >= 2 && requested <= 10 ? requested : 4;
}

}  // namespace x2w_view_dimension
