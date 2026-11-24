// Checks the mapping of the format in a clear value (src/startup_stubs/clear_value_format.h) without a device.
// --mutate d16: expect console format 118 to stay unchanged.
// --mutate values: expect the mapped copy to lose the depth and stencil values.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/clear_value_format.h"

#include <cstdio>
#include <cstring>

namespace {

using namespace x2w_startup;

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

DXGI_FORMAT console(UINT value) { return static_cast<DXGI_FORMAT>(value); }

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];

    check("118 is the 24-bit depth with stencil", mutated("d16") ? desktop_clear_format(console(118)) == console(118)
                                                                 : desktop_clear_format(console(118)) == DXGI_FORMAT_D24_UNORM_S8_UINT);
    check("116 maps", desktop_clear_format(console(116)) == DXGI_FORMAT_R10G10B10A2_UNORM);
    check("117 maps", desktop_clear_format(console(117)) == DXGI_FORMAT_R10G10B10A2_UNORM);
    check("119 stays", desktop_clear_format(console(119)) == console(119));
    check("120 stays", desktop_clear_format(console(120)) == console(120));
    check("a desktop format stays", desktop_clear_format(DXGI_FORMAT_D32_FLOAT) == DXGI_FORMAT_D32_FLOAT &&
                                        desktop_clear_format(DXGI_FORMAT_R8G8B8A8_UNORM) == DXGI_FORMAT_R8G8B8A8_UNORM);
    check("unknown stays", desktop_clear_format(DXGI_FORMAT_UNKNOWN) == DXGI_FORMAT_UNKNOWN);

    D3D12_CLEAR_VALUE depth{};
    depth.Format = console(118);
    depth.DepthStencil.Depth = 0.25f;
    depth.DepthStencil.Stencil = 0x5a;
    const D3D12_CLEAR_VALUE mapped = desktop_clear_value(depth);
    check("copy format", mapped.Format == (mutated("d16") ? console(118) : DXGI_FORMAT_D24_UNORM_S8_UINT));
    check("copy depth", mapped.DepthStencil.Depth == (mutated("values") ? 0.0f : 0.25f));
    check("copy stencil", mapped.DepthStencil.Stencil == (mutated("values") ? 0 : 0x5a));
    check("original untouched", depth.Format == console(118));

    D3D12_CLEAR_VALUE colour{};
    colour.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    colour.Color[0] = 1.0f;
    colour.Color[3] = 0.5f;
    const D3D12_CLEAR_VALUE kept = desktop_clear_value(colour);
    check("colour kept", kept.Format == DXGI_FORMAT_R8G8B8A8_UNORM && kept.Color[0] == 1.0f && kept.Color[3] == 0.5f);

    if (failures) return 1;
    std::printf("startup_clear_value_format_check: ok\n");
    return 0;
}
