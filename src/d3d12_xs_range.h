#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>

// The GPU range a D3D12ConfigureVirtualMemoryX configuration describes (#874). The title keeps a CPU
// range A and a range B of the same size and maps B's 64 KiB pages to physical pages through a queue.
// On desktop D3D12 the range is a set of reserved buffers ("windows"); a caller maps their tiles with
// ID3D12CommandQueue::UpdateTileMappings.
//
// Ownership: d3d12_xs_range::create returns an owning reference. The windows are device children and
// hold the device until the range releases them, either on Close() or when the last reference is
// released. Nothing else holds the range, so a device whose range was closed or released is destroyed
// by its own last release. D3D12XboxCreateDevice does not create a range yet: that waits for an owner
// of the title's device that can close it before the device's final release.
//
// Before Close() or the final Release, the owner waits until GPU work that uses the windows has
// completed (for example with a fence on each queue that used them). Close() is serialized with the
// other methods: a call that overlaps Close() sees either the open range or an empty one.
struct __declspec(uuid("c0b6a313-7c1e-4273-b375-cd666d84f420")) IX2WGpuRange : IUnknown {
    // Bytes covered by all windows together; 0 after Close().
    virtual UINT64 STDMETHODCALLTYPE GetSize() = 0;
    // Bytes of each window; every window but the last has this size.
    virtual UINT64 STDMETHODCALLTYPE GetWindowSize() = 0;
    virtual UINT STDMETHODCALLTYPE GetWindowCount() = 0;
    // The reserved buffer of window `index`, with a reference added. It starts at range offset
    // index * GetWindowSize(). Returns E_INVALIDARG for an index out of range or a null output.
    virtual HRESULT STDMETHODCALLTYPE GetWindow(UINT index, ID3D12Resource** window) = 0;
    // GPU virtual address of [offset, offset + bytes) when the span lies inside one window; 0 otherwise.
    // A byte count of 0 is treated as 1.
    virtual D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress(UINT64 offset, UINT64 bytes) = 0;
    // Releases the windows. Afterwards the range is empty. Calling it again does nothing. Window
    // references obtained from GetWindow are the caller's and stay valid until it releases them.
    virtual void STDMETHODCALLTYPE Close() = 0;
};

namespace d3d12_xs_range {
// Creates one window; the default calls ID3D12Device::CreateReservedResource.
using CreateWindowFn = HRESULT (*)(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** window);

// Reserves a positive multiple of 64 KiB in windows of at most 4 GiB, or less where the adapter's
// per-resource limit is lower, and returns the owning reference in *range. The count must fit UINT.
// On failure returns the HRESULT with *range
// null, and every window created so far is released.
HRESULT create(ID3D12Device* device, UINT64 size, IX2WGpuRange** range, CreateWindowFn create_window = nullptr) noexcept;
}
