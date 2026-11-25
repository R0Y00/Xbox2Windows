// Experimental (#1999, item B): the cull, front-face and fill state in a PA_SU_SC_MODE_CNTL register
// value, as a D3D12 rasterizer description.
//
// This header decodes the register value only. Where a guest pipeline keeps that word is not part of it:
// the caller supplies the value once its location is independently established.
//
// Field positions follow the amdgpu register header gc_10_3_0_sh_mask.h (Linux
// drivers/gpu/drm/amd/include/asic_reg/gc, MIT) and Mesa's src/amd/registers/gfx10.json, PA_SU_SC_MODE_CNTL:
//   CULL_FRONT [0], CULL_BACK [1], FACE [2], POLY_MODE [4:3] (0 disabled, 1 dual),
//   POLYMODE_FRONT_PTYPE [7:5] and POLYMODE_BACK_PTYPE [10:8] (0 points, 1 lines, 2 triangles),
//   POLY_OFFSET_FRONT_ENABLE [11], POLY_OFFSET_BACK_ENABLE [12], POLY_OFFSET_PARA_ENABLE [13],
//   PROVOKING_VTX_LAST [19], PERSP_CORR_DIS [20].
// FACE 0 selects counter-clockwise front faces (radeonsi writes S_028814_FACE(!front_ccw)). Whether the
// guest's window coordinates and the D3D12 viewport agree on that winding is not established here, so the
// caller passes flip_winding to invert it.
//
// Refused with a reason, so the caller can keep the state it had: both faces culled (D3D12 has no such
// cull mode), a polygon mode that draws points or differs between front and back faces, a reserved
// POLY_MODE value, any polygon offset enable (the offset words are not decoded here), the last vertex
// as provoking vertex (D3D12 uses the first), and disabled perspective correction (a D3D12 pipeline has
// no such switch). The other fields (VTX_WINDOW_OFFSET_ENABLE, MULTI_PRIM_IB_ENA, the gradient and quad
// decomposition bits, KEEP_TOGETHER_ENABLE) are not read.
// A pure function, not used by the draw path yet.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace x2w_rasterizer_state {

constexpr std::uint32_t field(std::uint32_t v, unsigned lo, unsigned hi) { return (v >> lo) & ((1u << (hi - lo + 1)) - 1); }

struct Raster {
    bool ok = false;
    const char* reason = nullptr;
    D3D12_RASTERIZER_DESC desc{};
};

inline Raster decode_mode_cntl(std::uint32_t mode, bool flip_winding)
{
    Raster out;
    D3D12_RASTERIZER_DESC& r = out.desc;
    const std::uint32_t poly_mode = field(mode, 3, 4);
    if (poly_mode > 1) {
        out.reason = "reserved POLY_MODE value";
        return out;
    }
    r.FillMode = D3D12_FILL_MODE_SOLID;
    if (poly_mode == 1) {
        const std::uint32_t front = field(mode, 5, 7), back = field(mode, 8, 10);
        if (front != back || (front != 1 && front != 2)) {
            out.reason = "polygon mode with points, an unknown type, or different front and back types";
            return out;
        }
        r.FillMode = front == 1 ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
    }
    const bool cull_front = field(mode, 0, 0) != 0, cull_back = field(mode, 1, 1) != 0;
    if (cull_front && cull_back) {
        out.reason = "both faces culled";
        return out;
    }
    if (field(mode, 11, 13)) {
        out.reason = "polygon offset enabled (offset words not decoded)";
        return out;
    }
    if (field(mode, 19, 19)) {
        out.reason = "last vertex as provoking vertex";
        return out;
    }
    if (field(mode, 20, 20)) {
        out.reason = "perspective correction disabled";
        return out;
    }
    r.CullMode = cull_front ? D3D12_CULL_MODE_FRONT : cull_back ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
    r.FrontCounterClockwise = (field(mode, 2, 2) == 0) != flip_winding;
    r.DepthBias = 0;
    r.DepthBiasClamp = 0.0f;
    r.SlopeScaledDepthBias = 0.0f;
    r.DepthClipEnable = TRUE;
    r.MultisampleEnable = FALSE;
    r.AntialiasedLineEnable = FALSE;
    r.ForcedSampleCount = 0;
    r.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    out.ok = true;
    return out;
}

}  // namespace x2w_rasterizer_state
