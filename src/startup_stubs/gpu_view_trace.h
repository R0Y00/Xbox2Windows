#pragma once

#include "startup_stubs/local_log.h"
#include <d3d12.h>
#include <type_traits>

namespace gpu_view_trace {
inline bool watching(ID3D12Device* device) noexcept
{
    static const bool enabled = x2w_local::env_on("X2W_GPU_VIEW_TRACE");
    return enabled && device && device->GetDeviceRemovedReason() == S_OK;
}

// A view creation returns void. Only claim a transition when the device was healthy immediately before the call.
// Another queue or thread can still remove the device during the same interval.
template <typename Desc>
inline void report(const char* site, const char* kind, HRESULT reason, ID3D12Resource* resource, ID3D12Resource* counter,
                   const Desc* desc, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    const auto facts = resource ? resource->GetDesc() : D3D12_RESOURCE_DESC{};
    const auto counter_facts = counter ? counter->GetDesc() : D3D12_RESOURCE_DESC{};
    unsigned words[16]{};
    static_assert(sizeof(Desc) <= sizeof(words));
    const bool readable = desc && x2w_local::readable(desc, sizeof(*desc));
    if (readable) std::memcpy(words, desc, sizeof(*desc));
    x2w_local::logf("x2w-view-transition: %s %s tid=%lu before=S_OK after=0x%08lx dest=0x%llx resource=%p "
                    "dim=%u width=%llu height=%u depthOrArray=%u mips=%u format=%u flags=0x%x samples=%u "
                    "counter=%p counterWidth=%llu desc=%p readable=%d words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                    site, kind, GetCurrentThreadId(), static_cast<unsigned long>(reason), static_cast<unsigned long long>(at.ptr),
                    static_cast<void*>(resource), facts.Dimension, facts.Width, facts.Height, facts.DepthOrArraySize, facts.MipLevels,
                    facts.Format, facts.Flags, facts.SampleDesc.Count, static_cast<void*>(counter), counter_facts.Width,
                    static_cast<const void*>(desc), int(readable), words[0], words[1], words[2], words[3], words[4], words[5],
                    words[6], words[7], words[8], words[9], words[10], words[11], words[12]);
    if (!readable) return;
    if constexpr (std::is_same_v<Desc, D3D12_UNORDERED_ACCESS_VIEW_DESC>) {
        if (desc->ViewDimension == D3D12_UAV_DIMENSION_BUFFER)
            x2w_local::logf("x2w-view-detail UAV format=%u first=%llu elements=%u stride=%u flags=%u counterOffset=%llu",
                            desc->Format, desc->Buffer.FirstElement, desc->Buffer.NumElements, desc->Buffer.StructureByteStride,
                            desc->Buffer.Flags, desc->Buffer.CounterOffsetInBytes);
    } else if constexpr (std::is_same_v<Desc, D3D12_SHADER_RESOURCE_VIEW_DESC>) {
        if (desc->ViewDimension == D3D12_SRV_DIMENSION_BUFFER)
            x2w_local::logf("x2w-view-detail SRV format=%u first=%llu elements=%u stride=%u flags=%u", desc->Format,
                            desc->Buffer.FirstElement, desc->Buffer.NumElements, desc->Buffer.StructureByteStride, desc->Buffer.Flags);
    } else if constexpr (std::is_same_v<Desc, D3D12_CONSTANT_BUFFER_VIEW_DESC>) {
        x2w_local::logf("x2w-view-detail CBV address=0x%llx bytes=%u", desc->BufferLocation, desc->SizeInBytes);
    }
}

inline LONG reported = 0;
template <typename Desc>
inline void after(const char* site, const char* kind, HRESULT reason, ID3D12Resource* resource, ID3D12Resource* counter,
                  const Desc* desc, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    if (FAILED(reason) && InterlockedCompareExchange(&reported, 1, 0) == 0)
        report(site, kind, reason, resource, counter, desc, at);
}

struct MirrorCall {
    ID3D12Device* device;
    const char* site;
    void srv(ID3D12Resource* r, const D3D12_SHADER_RESOURCE_VIEW_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE at) const
    {
        const bool watch = watching(device);
        device->CreateShaderResourceView(r, d, at);
        if (watch) after(site, "mirror SRV", device->GetDeviceRemovedReason(), r, nullptr, d, at);
    }
    void uav(ID3D12Resource* r, ID3D12Resource* c, const D3D12_UNORDERED_ACCESS_VIEW_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE at) const
    {
        const bool watch = watching(device);
        device->CreateUnorderedAccessView(r, c, d, at);
        if (watch) after(site, "mirror UAV", device->GetDeviceRemovedReason(), r, c, d, at);
    }
    void cbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE at) const
    {
        const bool watch = watching(device);
        device->CreateConstantBufferView(d, at);
        if (watch) after(site, "mirror CBV", device->GetDeviceRemovedReason(), nullptr, nullptr, d, at);
    }
    void sampler(const D3D12_SAMPLER_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE at) const
    {
        const bool watch = watching(device);
        device->CreateSampler(d, at);
        if (watch) after(site, "mirror sampler", device->GetDeviceRemovedReason(), nullptr, nullptr, d, at);
    }
    void copy(UINT count, D3D12_CPU_DESCRIPTOR_HANDLE dest, D3D12_CPU_DESCRIPTOR_HANDLE src, D3D12_DESCRIPTOR_HEAP_TYPE heap) const
    {
        const bool watch = watching(device);
        device->CopyDescriptorsSimple(count, dest, src, heap);
        if (!watch) return;
        const HRESULT reason = device->GetDeviceRemovedReason();
        if (FAILED(reason) && InterlockedCompareExchange(&reported, 1, 0) == 0)
            x2w_local::logf("x2w-view-transition: %s mirror copy tid=%lu before=S_OK after=0x%08lx "
                            "dest=0x%llx src=0x%llx count=%u heap=%u", site, GetCurrentThreadId(),
                            static_cast<unsigned long>(reason), static_cast<unsigned long long>(dest.ptr),
                            static_cast<unsigned long long>(src.ptr), count, heap);
    }
};
}
