#pragma once
#include "mesh_pixel_heap.h"
#include <d3d12.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace x2w_startup
{
inline std::vector<unsigned char> normalize_mesh57_descriptor(std::vector<unsigned char> desc)
{
    if (desc.size() < 15 * sizeof(std::uint64_t))
        return {};
    for (const auto index : {0, 1, 3, 5, 13})
        std::memset(desc.data() + index * sizeof(std::uint64_t), 0, sizeof(std::uint64_t));

    // Extended fixed descriptors carry two shader-record blocks. Keep the distances
    // between their internal pointers, while removing the process relocation. The
    // repeated allocation address at +0x160/+0x2c0 varies independently.
    if (desc.size() != 0x300)
        return desc;
    const auto word = [&](std::size_t offset) {
        std::uint32_t value;
        std::memcpy(&value, desc.data() + offset, sizeof(value));
        return value;
    };
    const auto address = [&](std::size_t offset) {
        std::uint64_t value;
        std::memcpy(&value, desc.data() + offset, sizeof(value));
        return value;
    };
    const auto is_pointer = [](std::uint64_t value) {
        return value >= 0x10000 && value < 0x800000000000ull;
    };
    const auto section = address(0x38);
    const auto allocation = address(0x160);
    if (address(0x48) || address(0x50) || !address(0x30) || !address(0x70) ||
        word(0x11c) != 0x04010000 || word(0x27c) != 0x04010000 ||
        !is_pointer(section) || !is_pointer(allocation) || allocation != address(0x2c0) ||
        word(0x110) + 0x10u != word(0x288) || word(0x270) + 0x10u != word(0x120))
        return desc;

    constexpr std::size_t section_pointers[]{
        0x38, 0x78, 0x120, 0x128, 0x168, 0x178, 0x198, 0x1d8,
        0x280, 0x288, 0x2c8, 0x2d8, 0x2f8
    };
    for (const auto offset : section_pointers) {
        const auto value = address(offset);
        if (!is_pointer(value) || (value >= section ? value - section : section - value) > INT32_MAX)
            return desc;
    }
    for (const auto offset : section_pointers) {
        const auto relative = address(offset) - section;
        std::memcpy(desc.data() + offset, &relative, sizeof(relative));
    }
    for (const auto offset : {0x110, 0x270}) {
        const auto relative = word(offset) - static_cast<std::uint32_t>(section);
        std::memcpy(desc.data() + offset, &relative, sizeof(relative));
    }
    for (const auto offset : {0x160, 0x2c0})
        std::memset(desc.data() + offset, 0, sizeof(allocation));
    return desc;
}
// A reviewed fixed-state/entry record. The header is followed by the exact normalized fixed
// descriptor, the two register blocks, creation root blob, guest MS and guest PS, in this order.
// Descriptor normalization removes the relocations recognized in each fixed layout.
struct Mesh57Profile {
    char magic[8];
    UINT lengths[6];
    UINT threads, vertices, primitives, topology, group_stride, wave_size;
    D3D12_BLEND_DESC blend;
    D3D12_RASTERIZER_DESC rasterizer;
    D3D12_DEPTH_STENCIL_DESC depth;
    D3D12_RT_FORMAT_ARRAY targets;
    DXGI_FORMAT depth_format;
    DXGI_SAMPLE_DESC samples;
    UINT sample_mask;
};
static_assert(sizeof(Mesh57Profile) == 532, "update the offline mesh57 profile reader when the ABI changes");
// Version 2 stores this immediately after Mesh57Profile, before the six byte spans.
struct Mesh57PixelEntry {
    UINT enable, address, wave_size;
};
// Version 3 adds one reviewed MS entry mapping after the version 2 pixel entry.
struct Mesh57ConstantEntry {
    UINT sgpr, root_parameter, count;
};
// Version 4 adds a count followed by this many MS table/root-address SGPR mappings.
// Version 5 then stores the physical resource-heap SGPR; selector and view class come from the MS.
struct Mesh57ViewEntry {
    UINT sgpr, root_parameter, root_address;
};
struct Mesh57BufferRead {
    UINT root_parameter, offset, minimum_bytes;
    bool root_address;
};
} // namespace x2w_startup
