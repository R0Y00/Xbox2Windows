// Experimental (docs/startup-runtime.md): the planes of a console present (CommandQueue slot 32).
//
// The present takes a plane count and an array of plane records. Measured from the first presents
// of a console executable: one record is 0x48 bytes, with +0x00 a frame token, +0x08 the resource
// count and +0x10 a pointer to the array of resources; the second plane's token sits at +0x48 when
// two planes are presented. The fields after +0x10 (source and destination rectangles, flags) are
// not decoded here. Pure: the caller passes a readability check, so a control can run it on
// synthetic records.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_startup {
inline constexpr std::size_t present_plane_stride = 0x48;

// The first resource of plane `index`, or 0 when the plane, its record or its resource array is
// not readable or has no resource. readable(address, bytes) answers whether memory can be read.
template <class Readable>
std::uint64_t present_plane_resource(const void* planes, unsigned count, unsigned index, Readable&& readable,
                                     std::size_t stride = present_plane_stride)
{
    if (!planes || index >= count) return 0;
    const auto* record = static_cast<const unsigned char*>(planes) + std::size_t{index} * stride;
    if (!readable(record, 0x18)) return 0;
    std::uint64_t resources = 0, resource_count = 0, first = 0;
    std::memcpy(&resource_count, record + 0x08, 8);
    std::memcpy(&resources, record + 0x10, 8);
    if (resource_count == 0 || !resources || !readable(reinterpret_cast<const void*>(resources), 8)) return 0;
    std::memcpy(&first, reinterpret_cast<const void*>(resources), 8);
    return first;
}

// One draw of the window composite: plane `plane`, either replacing what is below (the bottom
// layer) or blended over it by the plane's own alpha.
struct PlaneStep {
    unsigned plane = 0;
    bool blended = false;
};

// The window composite of a present, bottom to top, the way planes stack on the console: plane 0
// first, each later plane over it. `resources[i]` is plane i's surface, 0 when it cannot be shown;
// those planes and repeats of a surface already drawn are left out. The lowest drawn plane is
// opaque (nothing is below it), every later one blended. Returns the number of steps written.
// Measured on a two-plane present: plane 0 was black and plane 1, with alpha 255 wherever the
// picture was, held it; showing plane 0 alone gave a black window.
inline unsigned composite_steps(const std::uint64_t* resources, unsigned count, PlaneStep* steps, unsigned max_steps)
{
    unsigned written = 0;
    for (unsigned i = 0; resources && i < count && written < max_steps; ++i) {
        if (!resources[i]) continue;
        bool repeat = false;
        for (unsigned j = 0; j < written; ++j) repeat = repeat || resources[steps[j].plane] == resources[i];
        if (repeat) continue;
        steps[written] = PlaneStep{i, written != 0};
        ++written;
    }
    return written;
}
}  // namespace x2w_startup
