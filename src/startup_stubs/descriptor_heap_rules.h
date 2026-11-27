// Descriptor heap requests repaired before the first desktop call, so the debug layer sees a valid one.
// D3D12 "Hardware Tiers" (Resource Binding): every tier guarantees 1,000,000 descriptors in a shader-visible
// CBV/SRV/UAV heap and 2048 in a shader-visible sampler heap; larger requests are driver-dependent and the
// debug layer reports them. RTV and DSV heaps cannot be shader visible ("Descriptor Heaps Overview").
// Descriptors past the clamped count stay in the CPU-only tail heap made by the device shim.
#pragma once
#include <d3d12.h>
namespace x2w_startup {
constexpr UINT max_shader_visible_view_descriptors = 1000000;
constexpr UINT max_shader_visible_sampler_descriptors = 2048;
inline D3D12_DESCRIPTOR_HEAP_DESC safe_heap_desc(const D3D12_DESCRIPTOR_HEAP_DESC& input) {
    auto result = input;
    if (input.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
        if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && input.NumDescriptors > max_shader_visible_view_descriptors)
            result.NumDescriptors = max_shader_visible_view_descriptors;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER &&
                 input.NumDescriptors > max_shader_visible_sampler_descriptors)
            result.NumDescriptors = max_shader_visible_sampler_descriptors;
        else if (input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_RTV || input.Type == D3D12_DESCRIPTOR_HEAP_TYPE_DSV)
            result.Flags = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS>(input.Flags & ~D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    }
    return result;
}
}  // namespace x2w_startup
