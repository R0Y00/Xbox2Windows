// Checks the rules that gate a title's ResolveSubresource (src/startup_stubs/resolve_subresource.h, #870).
#include "../src/startup_stubs/resolve_subresource.h"

#include <cstdio>

using namespace x2w_resolve;

namespace {
int failures;
void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}
constexpr std::uint32_t rgba8 = 28;
}  // namespace

int main()
{
    const Side single{1280, 720, 1, 1, 1, true};
    const Side msaa{1280, 720, 1, 1, 4, true};

    expect(check(single, 0, msaa, 0, rgba8) == Verdict::ok, "a 4x target resolves into a single-sample one");
    expect(check(msaa, 0, msaa, 0, rgba8) == Verdict::destination_multisampled, "destination must be single-sample");
    expect(check(single, 0, single, 0, rgba8) == Verdict::source_not_multisampled, "source must be multisampled");
    expect(check(single, 0, msaa, 0, 0) == Verdict::format_unknown, "format must be named");

    Side buffer = single;
    buffer.is_texture2d = false;
    expect(check(buffer, 0, msaa, 0, rgba8) == Verdict::not_texture, "buffers are refused");

    const Side smaller{640, 360, 1, 1, 1, true};
    expect(check(smaller, 0, msaa, 0, rgba8) == Verdict::size_mismatch, "sizes must match");

    const Side mips{1280, 720, 3, 2, 1, true};
    expect(check(mips, 5, msaa, 0, rgba8) == Verdict::size_mismatch, "mip 2 of the destination is 320x180");
    expect(check(mips, 6, msaa, 0, rgba8) == Verdict::subresource_out_of_range, "subresource 6 of 6 is out of range");
    expect(check(mips, 3, msaa, 0, rgba8) == Verdict::ok, "array slice 1, mip 0 matches the source");
    expect(check(single, 0, msaa, 1, rgba8) == Verdict::subresource_out_of_range, "source subresource 1 of 1");

    const Side tiny{1, 1, 1, 1, 1, true};
    const Side tiny_msaa{1, 1, 1, 1, 2, true};
    expect(check(tiny, 0, tiny_msaa, 0, rgba8) == Verdict::ok, "1x1 resolves");
    const Side deep{1, 1, 70, 1, 1, true};
    expect(check(deep, 69, tiny_msaa, 0, rgba8) == Verdict::ok, "a mip past the size floor stays 1x1");

    if (!failures) std::printf("startup_resolve_subresource: ok\n");
    return failures ? 1 : 0;
}
