// Experimental (docs/startup-runtime.md, #870): the checks before a title's ResolveSubresource (command
// list slot 19, desktop order) is forwarded to the desktop list. Plain standard C++ with no Windows types,
// so the rules can be compiled and tested on their own.
//
// A resolve copies one subresource of a multisampled texture into one subresource of a single-sample
// texture of the same size. The desktop runtime treats any other call as a device-removing error, so a
// call that does not fit is refused here and logged instead of reaching the driver.
#pragma once

#include <cstdint>

namespace x2w_resolve {

struct Side {
    std::uint64_t width;
    std::uint32_t height;
    std::uint32_t mip_levels;
    std::uint32_t array_size;
    std::uint32_t sample_count;
    bool is_texture2d;
};

enum class Verdict {
    ok,
    not_texture,
    destination_multisampled,
    source_not_multisampled,
    size_mismatch,
    subresource_out_of_range,
    format_unknown,
};

inline std::uint64_t subresource_count(const Side& s) { return std::uint64_t{s.mip_levels} * s.array_size; }

// The size of a subresource at `mip`: each mip halves the base size and stops at 1.
inline std::uint64_t mip_extent(std::uint64_t base, std::uint32_t mip)
{
    const std::uint64_t value = mip >= 63 ? 0 : base >> mip;
    return value ? value : 1;
}

inline Verdict check(const Side& destination, std::uint32_t destination_subresource, const Side& source,
                     std::uint32_t source_subresource, std::uint32_t format)
{
    if (!destination.is_texture2d || !source.is_texture2d) return Verdict::not_texture;
    if (destination.sample_count != 1) return Verdict::destination_multisampled;
    if (source.sample_count <= 1) return Verdict::source_not_multisampled;
    if (!format) return Verdict::format_unknown;
    if (!destination.mip_levels || !destination.array_size || !source.mip_levels || !source.array_size ||
        destination_subresource >= subresource_count(destination) || source_subresource >= subresource_count(source))
        return Verdict::subresource_out_of_range;
    const std::uint32_t destination_mip = destination_subresource % destination.mip_levels;
    const std::uint32_t source_mip = source_subresource % source.mip_levels;
    if (mip_extent(destination.width, destination_mip) != mip_extent(source.width, source_mip) ||
        mip_extent(destination.height, destination_mip) != mip_extent(source.height, source_mip))
        return Verdict::size_mismatch;
    return Verdict::ok;
}

}  // namespace x2w_resolve
