// Source-built control for view_dimension.h: whether a view's ViewDimension suits the dimension of
// the resource it is made on (desktop D3D12 removes the device for a mismatch). Values are the
// public d3d12.h enumerations. --control swaps the resource dimension of the mismatched cases for
// the matching one and enables default views for every slot in the gated cases, so the checks
// that expect a refusal must print FAIL.
#include <cstdio>
#include <cstring>

#include "startup_stubs/view_dimension.h"

namespace {
int failures;
bool control;

void check(const char* what, bool ok)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// A refusal case: in the control the resource is given the dimension the view needs.
bool refused(int slot, unsigned view_dimension, unsigned resource_dimension)
{
    using namespace x2w_view_dimension;
    if (control) resource_dimension = needed(slot, view_dimension);
    return !fits(slot, view_dimension, resource_dimension);
}
}  // namespace

int main(int argc, char** argv)
{
    using namespace x2w_view_dimension;
    control = argc > 1 && std::strcmp(argv[1], "--control") == 0;

    // Matching shapes are accepted.
    check("SRV texture 2D on a 2D texture", fits(18, 4, texture2d));
    check("SRV cube on a 2D texture", fits(18, 9, texture2d));
    check("SRV cube array on a 2D texture", fits(18, 10, texture2d));
    check("SRV texture 3D on a 3D texture", fits(18, 8, texture3d));
    check("SRV buffer on a buffer", fits(18, 1, buffer));
    check("SRV texture 1D array on a 1D texture", fits(18, 3, texture1d));
    check("UAV texture 2D array on a 2D texture", fits(19, 5, texture2d));
    check("UAV texture 3D on a 3D texture", fits(19, 8, texture3d));
    check("RTV buffer on a buffer", fits(20, 1, buffer));
    check("RTV texture 2D on a 2D texture", fits(20, 4, texture2d));
    check("DSV texture 1D on a 1D texture", fits(21, 1, texture1d));
    check("DSV texture 2D on a 2D texture", fits(21, 3, texture2d));

    // Values this check does not judge pass on: unknown dimension, raytracing structure, other slots.
    check("SRV unknown dimension is not judged", fits(18, 0, texture3d));
    check("SRV raytracing structure is not judged", fits(18, 11, unknown));
    check("out-of-range SRV dimension is not judged", fits(18, 99, buffer));
    check("DSV dimension 7 is not judged", fits(21, 7, buffer));
    check("sampler slot is not judged", fits(22, 4, buffer));

    // Mismatches are refused (the control turns each into a match, so these print FAIL).
    check("SRV texture 2D on a 3D texture is refused", refused(18, 4, texture3d));
    check("SRV texture 3D on a 2D texture is refused", refused(18, 8, texture2d));
    check("SRV buffer on a 2D texture is refused", refused(18, 1, texture2d));
    check("SRV cube on a 3D texture is refused", refused(18, 9, texture3d));
    check("UAV texture 2D on a buffer is refused", refused(19, 4, buffer));
    check("RTV texture 3D on a 2D texture is refused", refused(20, 8, texture2d));
    check("DSV texture 2D on a 1D texture is refused", refused(21, 3, texture1d));

    // ViewDimension 0 (unknown): a default view only where desktop allows a null description. Under
    // X2W_D3D12_VIEW_DEFAULTS=all (experimental) every view type may get one.
    const auto all = DefaultViews::all, srv = DefaultViews::srv, off = DefaultViews::off;
    check("default SRV of an RGBA8 2D texture", default_view_allowed(18, texture2d, 28, all));
    check("default RTV of an RGBA8 2D texture", default_view_allowed(20, texture2d, 28, all));
    check("default DSV of a D32 2D texture", default_view_allowed(21, texture2d, 40, all));
    check("default SRV of a 3D texture", default_view_allowed(18, texture3d, 10, all));
    check("default SRV of an R16_SINT 2D texture", default_view_allowed(18, texture2d, 59, all));
    const auto no_default = [all](int slot, unsigned dimension, unsigned format) {
        if (control) dimension = texture2d, format = 28;  // a texture that has a default view
        return !default_view_allowed(slot, dimension, format, all);
    };
    check("no default view of a buffer", no_default(18, buffer, 28));
    check("no default view of a typeless texture", no_default(18, texture2d, 27));
    check("no default view of a typeless BC7 texture", no_default(19, texture2d, 97));
    check("no default view of a typeless R8 texture", no_default(18, texture2d, 60));
    check("no default SRV of a depth texture", no_default(18, texture2d, 40));
    check("no default view of an unknown format", no_default(18, texture2d, 0));
    check("no default view on the constant-buffer slot", !default_view_allowed(17, texture2d, 28, all));

    // The default (srv): only the measured typed-SRV case; UAV, RTV and DSV need the experiment.
    // The control enables all slots, so the three refusals print FAIL.
    const auto gated = [all, srv](int slot, unsigned dimension, unsigned format) {
        return !default_view_allowed(slot, dimension, format, control ? all : srv);
    };
    check("default SRV of an RGBA8 2D texture without the experiment", default_view_allowed(18, texture2d, 28, srv));
    check("default SRV of a 3D texture without the experiment", default_view_allowed(18, texture3d, 10, srv));
    check("no default SRV of a depth texture without the experiment", !default_view_allowed(18, texture2d, 40, srv));
    check("no default UAV without the experiment", gated(19, texture2d, 28));
    check("no default RTV without the experiment", gated(20, texture2d, 28));
    check("no default DSV without the experiment", gated(21, texture2d, 40));
    check("off: no default SRV", !default_view_allowed(18, texture2d, 28, off));
    check("off: no default RTV", !default_view_allowed(20, texture2d, 28, off));

    // X2W_D3D12_VIEW_DEFAULTS parsing: exact "all" or "off", anything else is the default.
    check("unset is srv", default_views_mode(nullptr) == srv);
    check("empty is srv", default_views_mode("") == srv);
    check("all", default_views_mode("all") == all);
    check("off", default_views_mode("off") == off);
    check("ALL is srv", default_views_mode("ALL") == srv);
    check("allx is srv", default_views_mode("allx") == srv);
    check("1 is srv", default_views_mode("1") == srv);

    // A refused SRV's null descriptor keeps a texture dimension and turns the rest into texture 2D.
    check("null for a refused 3D view is 3D", refused_srv_null_dimension(8) == 8);
    check("null for a refused cube array is a cube array", refused_srv_null_dimension(10) == 10);
    check("null for a refused 1D view is 1D", refused_srv_null_dimension(2) == 2);
    check("null for a refused unknown view is 2D", refused_srv_null_dimension(0) == 4);
    check("null for a refused buffer view is 2D", refused_srv_null_dimension(1) == 4);
    check("null for a refused raytracing view is 2D", refused_srv_null_dimension(11) == 4);

    if (failures) {
        std::printf("startup_view_dimension: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_view_dimension: ok");
    return 0;
}
