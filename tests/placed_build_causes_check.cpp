// The device-free rules of the placed acceleration-structure build (src/startup_stubs/placed_build.h). Standard
// C++17 only.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "startup_stubs/placed_build.h"

namespace {
using x2w_startup::PlacedBuildCause;
using Cause = PlacedBuildCause;
using x2w_startup::check_update_shape;
using x2w_startup::ranges_overlap;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

constexpr std::uint64_t size = 4096;
constexpr std::uint64_t base = 0x10000;

void causes_have_distinct_names()
{
    for (unsigned a = 0; a < static_cast<unsigned>(Cause::count); ++a) {
        const std::string_view name = x2w_startup::cause_name(static_cast<Cause>(a));
        CHECK(name != "unknown");
        for (unsigned b = a + 1; b < static_cast<unsigned>(Cause::count); ++b)
            CHECK(name != x2w_startup::cause_name(static_cast<Cause>(b)));
    }
    CHECK(std::string_view(x2w_startup::cause_name(Cause::count)) == "unknown");
}

void a_plain_build_has_no_source()
{
    CHECK(check_update_shape(false, 0, base, size) == Cause::ok);
    CHECK(check_update_shape(false, base, base + size, size) == Cause::source_without_update);
    CHECK(check_update_shape(true, 0, base, size) == Cause::update_without_source);
}

void an_update_runs_in_place_or_elsewhere()
{
    CHECK(check_update_shape(true, base, base, size) == Cause::ok);                 // in place
    CHECK(check_update_shape(true, base, base + size, size) == Cause::ok);          // next to it, touching
    CHECK(check_update_shape(true, base + 4 * size, base, size) == Cause::ok);      // far above
    CHECK(check_update_shape(true, base, base + 4 * size, size) == Cause::ok);      // far below
}

void partial_overlap_is_refused()
{
    CHECK(check_update_shape(true, base + 256, base, size) == Cause::source_dest_partial_overlap);
    CHECK(check_update_shape(true, base, base + 256, size) == Cause::source_dest_partial_overlap);
    CHECK(check_update_shape(true, base + size - 256, base, size) == Cause::source_dest_partial_overlap);  // last 256 bytes
    CHECK(check_update_shape(true, base + size, base, size) == Cause::ok);                                // first byte past
}

void the_source_must_be_aligned()
{
    CHECK(check_update_shape(true, base + 1, base + 8 * size, size) == Cause::source_unaligned);
    CHECK(check_update_shape(true, base + 255, base + 8 * size, size) == Cause::source_unaligned);
    CHECK(check_update_shape(true, base + 256, base + 8 * size, size) == Cause::ok);
    // Alignment is named before overlap.
    CHECK(check_update_shape(true, base + 1, base, size) == Cause::source_unaligned);
}

void overlap_edges()
{
    CHECK(!ranges_overlap(base, 0, base, size));
    CHECK(!ranges_overlap(base, size, base, 0));
    CHECK(ranges_overlap(base, 1, base, 1));
    CHECK(!ranges_overlap(base, 1, base + 1, 1));
    CHECK(ranges_overlap(base, size, base + size - 1, 1));
    CHECK(!ranges_overlap(base, size, base + size, 1));
    // A range that would pass the top of the address space stops at it.
    CHECK(ranges_overlap(UINT64_MAX - 15, 64, UINT64_MAX, 1));
    CHECK(!ranges_overlap(UINT64_MAX - 15, 64, 0, 16));
    CHECK(ranges_overlap(0, UINT64_MAX, UINT64_MAX - 1, 1));
}

void source_causes_read_as_the_source()
{
    CHECK(x2w_startup::as_source_cause(Cause::unaligned) == Cause::source_unaligned);
    CHECK(x2w_startup::as_source_cause(Cause::not_placed) == Cause::source_not_placed);
    CHECK(x2w_startup::as_source_cause(Cause::wrong_state) == Cause::source_wrong_state);
    CHECK(x2w_startup::as_source_cause(Cause::range) == Cause::source_range);
    CHECK(x2w_startup::as_source_cause(Cause::ok) == Cause::ok);
    CHECK(x2w_startup::as_source_cause(Cause::source_unaligned) == Cause::source_unaligned);
}
}  // namespace

int main()
{
    causes_have_distinct_names();
    a_plain_build_has_no_source();
    an_update_runs_in_place_or_elsewhere();
    partial_overlap_is_refused();
    the_source_must_be_aligned();
    overlap_edges();
    source_causes_read_as_the_source();
    std::puts("placed build causes: all checks passed");
    return 0;
}
