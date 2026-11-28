// Descriptor heap requests repaired before the first desktop call, so the debug layer sees a valid one.
// D3D12 "Hardware Tiers" (Resource Binding): every tier guarantees 1,000,000 descriptors in a shader-visible
// CBV/SRV/UAV heap and 2048 in a shader-visible sampler heap; larger requests are driver-dependent and the
// debug layer reports them. RTV and DSV heaps cannot be shader visible ("Descriptor Heaps Overview").
// Tier 3 lists "1000000+" (Resource Binding, Hardware Tiers): a tier 3 device keeps requests up to 2^20 = 1,048,576
// descriptors (the largest count the API documents), and the shim's capacity probe shortens it when the driver refuses.
// Descriptors past the clamped count stay in the CPU-only tail heap made by the device shim.
#pragma once
#include <d3d12.h>
#include <cstdint>
#include <algorithm>
namespace x2w_startup {
constexpr UINT max_shader_visible_view_descriptors = 1000000;
constexpr UINT max_tier3_view_descriptors = 1048576;
constexpr UINT max_shader_visible_sampler_descriptors = 2048;
inline D3D12_DESCRIPTOR_HEAP_DESC safe_heap_desc(const D3D12_DESCRIPTOR_HEAP_DESC& input, UINT view_limit = max_shader_visible_view_descriptors) {
    auto result = input;
    if (input.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
        if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && input.NumDescriptors > view_limit)
            result.NumDescriptors = view_limit;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER &&
                 input.NumDescriptors > max_shader_visible_sampler_descriptors)
            result.NumDescriptors = max_shader_visible_sampler_descriptors;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV || input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
            result.Flags = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS>(input.Flags & ~D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    }
    return result;
}
// Halve from the ceiling and test the floor itself as the last size, so a driver whose maximum lies between the
// floor and the last power-of-two step (65,536 to 124,999 from a 1,000,000 ceiling) is still found. 0 when even the
// floor is refused.
template<class Probe>
std::uint64_t largest_halving_size(std::uint64_t ceiling, std::uint64_t floor, Probe&& probe) {
    for (std::uint64_t size = ceiling;; size /= 2) {
        const std::uint64_t tried = (std::max)(size, floor);
        if (probe(tried)) return tried;
        if (tried == floor) return 0;
    }
}
}  // namespace x2w_startup
