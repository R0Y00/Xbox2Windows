// Checks src/startup_stubs/present_planes.h on synthetic plane records: two 0x48-byte records, each
// with a frame token, a resource count and a pointer to its resource array.
// Also checks the window composite order (composite_steps) on synthetic plane surfaces.
// --control reads the records with a 0x40-byte stride (the record size before the second plane was
// measured) and composites plane 0 alone (the window present before planes were composited); the
// second-plane rows must print FAIL.
#include "startup_stubs/present_planes.h"

#include <cstdio>
#include <cstring>

namespace {
int failures;

void check(const char* what, bool ok)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void put(unsigned char* at, std::uint64_t value) { std::memcpy(at, &value, 8); }
}  // namespace

int main(int argc, char** argv)
{
    const bool control = argc > 1 && std::strcmp(argv[1], "--control") == 0;
    const std::size_t stride = control ? 0x40 : x2w_startup::present_plane_stride;
    const std::uint64_t first_plane[] = {0x1111}, second_plane[] = {0x2222, 0x3333};
    alignas(8) unsigned char records[2 * 0x48] = {};
    alignas(8) unsigned char empty[0x48] = {};
    // Readable: only this fixture's own records and resource arrays.
    const auto inside = [](const void* address, std::size_t bytes, const void* base, std::size_t size) {
        const auto* a = static_cast<const unsigned char*>(address);
        const auto* b = static_cast<const unsigned char*>(base);
        return a >= b && a + bytes <= b + size;
    };
    const auto readable = [&](const void* address, std::size_t bytes) {
        return inside(address, bytes, records, sizeof(records)) || inside(address, bytes, empty, sizeof(empty)) ||
               inside(address, bytes, first_plane, sizeof(first_plane)) || inside(address, bytes, second_plane, sizeof(second_plane));
    };
    put(records + 0x00, 7);  // frame token
    put(records + 0x08, 1);
    put(records + 0x10, reinterpret_cast<std::uint64_t>(first_plane));
    put(records + 0x48, 7);
    put(records + 0x50, 2);
    put(records + 0x58, reinterpret_cast<std::uint64_t>(second_plane));

    using x2w_startup::present_plane_resource;
    check("plane 0 resource", present_plane_resource(records, 2, 0, readable, stride) == 0x1111);
    check("plane 1 resource (the second record starts at +0x48)", present_plane_resource(records, 2, 1, readable, stride) == 0x2222);
    check("a plane past the count has no resource", present_plane_resource(records, 1, 1, readable, stride) == 0);
    check("no records", present_plane_resource(nullptr, 2, 0, readable, stride) == 0);

    put(empty + 0x10, reinterpret_cast<std::uint64_t>(first_plane));
    check("a plane with no resources has none", present_plane_resource(empty, 1, 0, readable, stride) == 0);
    const auto unreadable = [](const void*, std::size_t) { return false; };
    check("an unreadable record has no resource", present_plane_resource(records, 2, 0, unreadable, stride) == 0);

    // Composite order: the lowest shown plane is opaque, later ones blend over it.
    using x2w_startup::PlaneStep;
    const auto composite = [&](const std::uint64_t* resources, unsigned count, PlaneStep* steps) {
        return x2w_startup::composite_steps(resources, control && count > 1 ? 1 : count, steps, 4);
    };
    PlaneStep steps[4];
    const std::uint64_t two[] = {0x1111, 0x2222};
    unsigned n = composite(two, 2, steps);
    check("two planes give two steps", n == 2);
    check("plane 0 is drawn first, opaque", n >= 1 && steps[0].plane == 0 && !steps[0].blended);
    check("plane 1 is drawn over plane 0, blended", n >= 2 && steps[1].plane == 1 && steps[1].blended);
    const std::uint64_t missing_base[] = {0, 0x2222};
    n = composite(missing_base, 2, steps);
    check("with plane 0 not shown, plane 1 is the opaque base", n == 1 && steps[0].plane == 1 && !steps[0].blended);
    const std::uint64_t repeated[] = {0x1111, 0x1111};
    n = x2w_startup::composite_steps(repeated, 2, steps, 4);
    check("a surface presented twice is drawn once", n == 1 && steps[0].plane == 0);
    const std::uint64_t one[] = {0x1111};
    n = composite(one, 1, steps);
    check("one plane is one opaque step", n == 1 && steps[0].plane == 0 && !steps[0].blended);
    check("no planes, no steps", x2w_startup::composite_steps(nullptr, 2, steps, 4) == 0);

    if (failures) {
        std::printf("startup_present_planes: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_present_planes: ok");
    return 0;
}
