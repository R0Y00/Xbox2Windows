// Device-free check of src/startup_stubs/rasterizer_state_decode.h (#1999, item B): synthetic
// PA_SU_SC_MODE_CNTL values built from the Mesa gfx10 field layout, decoded to D3D12 cull, winding and fill
// state, plus every refused form.
//
// startup_rasterizer_decode_check [--mutate face|cull|fill]
// A mutation changes the expectations the way a wrong field reading would (FACE read inverted, CULL_FRONT
// and CULL_BACK swapped, the polygon types swapped); the check must then fail (CTest WILL_FAIL controls).
#include "startup_stubs/rasterizer_state_decode.h"

#include <cstdio>
#include <cstring>

namespace {

using namespace x2w_rasterizer_state;

int failures = 0;
bool mutate_face = false, mutate_cull = false, mutate_fill = false;

constexpr std::uint32_t CULL_FRONT = 1u << 0, CULL_BACK = 1u << 1, FACE_CW = 1u << 2, POLY_DUAL = 1u << 3;
constexpr std::uint32_t front_type(std::uint32_t t) { return t << 5; }
constexpr std::uint32_t back_type(std::uint32_t t) { return t << 8; }
constexpr std::uint32_t PTYPE_POINTS = 0, PTYPE_LINES = 1, PTYPE_TRIANGLES = 2;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void accepted(const char* label, std::uint32_t mode, bool flip, D3D12_CULL_MODE cull, bool front_ccw, D3D12_FILL_MODE fill)
{
    if (mutate_cull && cull != D3D12_CULL_MODE_NONE)
        cull = cull == D3D12_CULL_MODE_FRONT ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_FRONT;
    if (mutate_face)
        front_ccw = !front_ccw;
    if (mutate_fill && (mode & POLY_DUAL))
        fill = fill == D3D12_FILL_MODE_SOLID ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
    const Raster r = decode_mode_cntl(mode, flip);
    char what[160];
    std::snprintf(what, sizeof what, "%s (mode %#x flip %d): ok=%d reason=%s cull=%d ccw=%d fill=%d", label, mode, flip,
                  r.ok, r.reason ? r.reason : "-", r.desc.CullMode, r.desc.FrontCounterClockwise, r.desc.FillMode);
    expect(r.ok && r.desc.CullMode == cull && (r.desc.FrontCounterClockwise != FALSE) == front_ccw &&
               r.desc.FillMode == fill && r.desc.DepthClipEnable && r.desc.DepthBias == 0 &&
               r.desc.SlopeScaledDepthBias == 0.0f && !r.desc.MultisampleEnable &&
               r.desc.ConservativeRaster == D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
           what);
}

void refused(const char* label, std::uint32_t mode, const char* reason)
{
    const Raster r = decode_mode_cntl(mode, false);
    char what[160];
    std::snprintf(what, sizeof what, "%s (mode %#x) refused with \"%s\", got ok=%d reason=%s", label, mode, reason, r.ok,
                  r.reason ? r.reason : "-");
    expect(!r.ok && r.reason && std::strstr(r.reason, reason), what);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc > 2 && std::strcmp(argv[1], "--mutate") == 0) {
        mutate_face = std::strcmp(argv[2], "face") == 0;
        mutate_cull = std::strcmp(argv[2], "cull") == 0;
        mutate_fill = std::strcmp(argv[2], "fill") == 0;
    }
    if (argc > 1 && !mutate_face && !mutate_cull && !mutate_fill) {
        std::printf("usage: startup_rasterizer_decode_check [--mutate face|cull|fill]\n");
        return 2;
    }
    // Cull mode x winding, with the caller's flip both ways.
    for (int flip = 0; flip < 2; ++flip) {
        const std::uint32_t faces[] = {0u, FACE_CW};
        for (std::uint32_t face : faces) {
            const bool ccw = (face == 0) != (flip != 0);
            accepted("no cull", face, flip != 0, D3D12_CULL_MODE_NONE, ccw, D3D12_FILL_MODE_SOLID);
            accepted("cull front", face | CULL_FRONT, flip != 0, D3D12_CULL_MODE_FRONT, ccw, D3D12_FILL_MODE_SOLID);
            accepted("cull back", face | CULL_BACK, flip != 0, D3D12_CULL_MODE_BACK, ccw, D3D12_FILL_MODE_SOLID);
        }
    }
    // Dual polygon mode with the same type on both faces; the types are ignored while POLY_MODE is 0.
    accepted("lines", POLY_DUAL | front_type(PTYPE_LINES) | back_type(PTYPE_LINES), false, D3D12_CULL_MODE_NONE, true,
             D3D12_FILL_MODE_WIREFRAME);
    accepted("triangles", POLY_DUAL | front_type(PTYPE_TRIANGLES) | back_type(PTYPE_TRIANGLES) | CULL_BACK, false,
             D3D12_CULL_MODE_BACK, true, D3D12_FILL_MODE_SOLID);
    accepted("types without POLY_MODE", front_type(PTYPE_POINTS) | back_type(PTYPE_LINES), false, D3D12_CULL_MODE_NONE, true,
             D3D12_FILL_MODE_SOLID);
    // Fields this decoder does not read leave the result alone (MULTI_PRIM_IB_ENA [21], KEEP_TOGETHER_ENABLE [24]).
    accepted("unread fields", CULL_BACK | 1u << 21 | 1u << 24, false, D3D12_CULL_MODE_BACK, true, D3D12_FILL_MODE_SOLID);
    // Refused forms.
    refused("both faces culled", CULL_FRONT | CULL_BACK, "both faces culled");
    refused("points", POLY_DUAL | front_type(PTYPE_POINTS) | back_type(PTYPE_POINTS), "polygon mode");
    refused("front and back differ", POLY_DUAL | front_type(PTYPE_LINES) | back_type(PTYPE_TRIANGLES), "polygon mode");
    refused("unknown polygon type", POLY_DUAL | front_type(3) | back_type(3), "polygon mode");
    refused("reserved POLY_MODE", 2u << 3, "reserved POLY_MODE");
    refused("front offset", 1u << 11, "polygon offset");
    refused("back offset", 1u << 12, "polygon offset");
    refused("para offset", 1u << 13, "polygon offset");
    refused("last provoking vertex", 1u << 19, "provoking vertex");
    refused("no perspective correction", 1u << 20, "perspective correction");
    if (failures) {
        std::printf("startup_rasterizer_decode: %d failures\n", failures);
        return 1;
    }
    std::printf("startup_rasterizer_decode: PASS (cull none/front/back x FACE x flip, wireframe and solid dual polygon "
                "mode, unread fields ignored, 10 refused forms)\n");
    return 0;
}
