// Source-built control for view_dimension.h: whether a view's ViewDimension suits the dimension of
// the resource it is made on (desktop D3D12 removes the device for a mismatch). Values are the
// public d3d12.h enumerations. --control swaps the resource dimension of the mismatched cases for
// the matching one, so the checks that expect a refusal must print FAIL.
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

    // ViewDimension 0 (unknown): a default view only where desktop allows a null description.
    check("default SRV of an RGBA8 2D texture", default_view_allowed(18, texture2d, 28));
    check("default RTV of an RGBA8 2D texture", default_view_allowed(20, texture2d, 28));
    check("default DSV of a D32 2D texture", default_view_allowed(21, texture2d, 40));
    check("default SRV of a 3D texture", default_view_allowed(18, texture3d, 10));
    const auto no_default = [](int slot, unsigned dimension, unsigned format) {
        if (control) dimension = texture2d, format = 28;  // a texture that has a default view
        return !default_view_allowed(slot, dimension, format);
    };
    check("no default view of a buffer", no_default(18, buffer, 28));
    check("no default view of a typeless texture", no_default(18, texture2d, 27));
    check("no default view of a typeless BC7 texture", no_default(19, texture2d, 97));
    check("no default SRV of a depth texture", no_default(18, texture2d, 40));
    check("no default view of an unknown format", no_default(18, texture2d, 0));
    check("no default view on the constant-buffer slot", !default_view_allowed(17, texture2d, 28));

    if (failures) {
        std::printf("startup_view_dimension: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_view_dimension: ok");
    return 0;
}
