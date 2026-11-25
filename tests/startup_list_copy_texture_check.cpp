// Experimental (docs/startup-runtime.md, #870): the argument checks of command list slot 16,
// CopyTextureRegion (src/startup_stubs/list_copy_texture.h). Plain standard C++, synthetic sizes only.
// --mutate overflow drops the destination bounds check; the run must then fail (negative control).
#include "../src/startup_stubs/list_copy_texture.h"

#include <cstdio>
#include <cstring>

using namespace x2w_list_copy_texture;

namespace {
int failures;
void expect(Verdict got, Verdict want, const char* what)
{
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s: got %s, want %s\n", what, name(got), name(want));
        ++failures;
    }
}
Side texture(std::uint32_t w, std::uint32_t h, std::uint32_t block = 1)
{
    return {Kind::subresource, w, h, 1, 0, 0, 0, 0, false, block, block};
}
Side footprint(std::uint32_t w, std::uint32_t h, std::uint64_t offset, std::uint32_t pitch, std::uint32_t bpb,
               std::uint64_t bytes, std::uint32_t block = 1)
{
    return {Kind::placed_footprint, w, h, 1, offset, pitch, bpb, bytes, false, block, block};
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate = argc > 2 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "overflow");

    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), nullptr), Verdict::ok, "whole texture");
    expect(check(texture(64, 64), 32, 32, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 32, 32, 1}), Verdict::ok, "corner box");
    // The destination origin pushes the box past the destination.
    const Verdict past = check(texture(64, 64), 40, 0, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 32, 32, 1});
    expect(mutate ? Verdict::ok : past, Verdict::dest_overflow, "dest overflow");
    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), &(const Box&)Box{0, 0, 0, 65, 8, 1}), Verdict::source_overflow, "source overflow");
    expect(check(texture(64, 64), 0, 0, 0, texture(64, 64), &(const Box&)Box{8, 0, 0, 8, 8, 1}), Verdict::bad_box, "empty box");
    expect(check(footprint(16, 16, 0, 64, 4, 4096), 0, 0, 0, footprint(16, 16, 0, 64, 4, 4096), nullptr), Verdict::buffer_to_buffer, "two buffers");

    // Texture to a 16x16 RGBA8 footprint: 64-byte rows would be below 256, so the pitch is refused.
    expect(check(footprint(16, 16, 0, 64, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::bad_pitch, "pitch not 256");
    expect(check(footprint(16, 16, 0, 256, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::ok, "footprint ok");
    expect(check(footprint(16, 16, 256, 256, 4, 1 << 20), 0, 0, 0, texture(16, 16), nullptr), Verdict::bad_offset, "offset not 512");
    // 16 rows of 256 bytes, the last row 64 bytes: 15 * 256 + 64 = 3904 bytes from offset 512.
    expect(check(footprint(16, 16, 512, 256, 4, 512 + 3904), 0, 0, 0, texture(16, 16), nullptr), Verdict::ok, "last row not padded");
    expect(check(footprint(16, 16, 512, 256, 4, 512 + 3903), 0, 0, 0, texture(16, 16), nullptr), Verdict::footprint_overflow, "one byte short");
    // Row wider than the pitch.
    expect(check(footprint(128, 4, 0, 256, 4, 1 << 20), 0, 0, 0, texture(128, 4), nullptr), Verdict::bad_pitch, "row over pitch");

    // Block-compressed: 4x4 blocks of 8 bytes; a 6-texel start is not on a block.
    expect(check(texture(64, 64, 4), 6, 0, 0, texture(64, 64, 4), nullptr), Verdict::bad_alignment, "block alignment");
    expect(check(footprint(64, 64, 0, 256, 8, 256 * 16, 4), 0, 0, 0, texture(64, 64, 4), nullptr), Verdict::ok, "bc footprint");

    // The destination origin counts: a 1x1 RGBA8 copy at Y=63 of a 64x64, 256-pitch footprint ends at byte 16132.
    expect(check(footprint(64, 64, 0, 256, 4, 256), 0, 63, 0, texture(1, 1), nullptr), Verdict::footprint_overflow, "dest origin counts");
    expect(check(footprint(64, 64, 0, 256, 4, 16132), 0, 63, 0, texture(1, 1), nullptr), Verdict::ok, "dest origin fits");
    // The source box origin counts too.
    expect(check(texture(1, 1), 0, 0, 0, footprint(64, 64, 0, 256, 4, 256), &(const Box&)Box{0, 63, 0, 1, 64, 1}), Verdict::footprint_overflow, "box origin counts");
    // 3D: a slice is as tall as the footprint (64 rows), not as the copy (1 row).
    Side volume = footprint(64, 64, 0, 256, 4, 256 * 64);
    volume.depth = 4;
    expect(check(volume, 0, 0, 1, texture(1, 1), nullptr), Verdict::footprint_overflow, "slice stride from footprint height");

    Side ms = texture(64, 64);
    ms.multisampled = true;
    expect(check(texture(64, 64), 0, 0, 0, ms, nullptr), Verdict::sample_mismatch, "multisampled source");

    if (failures) return 1;
    std::puts("ok");
    return 0;
}
