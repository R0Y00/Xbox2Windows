#include "d3d12_xs_range.h"

#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {
constexpr UINT64 tile_bytes = UINT64{64} << 10;
// Adapters tested in #874 lose tile mappings at or above 4 GiB inside one reserved buffer.
constexpr UINT64 max_window_bytes = UINT64{4} << 30;

HRESULT create_reserved(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** window)
{
    return device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(window));
}

class SharedLock {
public:
    explicit SharedLock(SRWLOCK& lock) : lock_(lock) { AcquireSRWLockShared(&lock_); }
    ~SharedLock() { ReleaseSRWLockShared(&lock_); }
    SharedLock(const SharedLock&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;

private:
    SRWLOCK& lock_;
};

class GpuRange final : public IX2WGpuRange {
public:
    HRESULT reserve(ID3D12Device* device, UINT64 size, d3d12_xs_range::CreateWindowFn create_window) noexcept
    {
        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va{};
        HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va));
        if (FAILED(hr)) return hr;
        if (va.MaxGPUVirtualAddressBitsPerResource < 17) return E_FAIL;
        // Below 4 GiB, the largest whole number of tiles a resource of this adapter can span.
        const UINT64 per_resource = va.MaxGPUVirtualAddressBitsPerResource >= 64
                                        ? max_window_bytes
                                        : (UINT64{1} << va.MaxGPUVirtualAddressBitsPerResource) - tile_bytes;
        const UINT64 window = std::min(max_window_bytes, per_resource) & ~(tile_bytes - 1);
        const UINT64 count = 1 + (size - 1) / window;
        if (count > std::numeric_limits<UINT>::max() || count > windows_.max_size()) return E_INVALIDARG;
        try {
            windows_.reserve(static_cast<std::size_t>(count));
        } catch (const std::bad_alloc&) {
            return E_OUTOFMEMORY;
        }
        for (UINT64 i = 0; i < count; ++i) {
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = std::min(window, size - i * window);
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Microsoft::WRL::ComPtr<ID3D12Resource> created;
            hr = create_window(device, desc, &created);
            if (FAILED(hr)) return hr;
            windows_.push_back(std::move(created));  // capacity reserved above
        }
        size_ = size;
        window_ = window;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown) && iid != __uuidof(IX2WGpuRange)) return E_NOINTERFACE;
        *out = static_cast<IX2WGpuRange*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG left = --refs_;
        if (!left) delete this;
        return left;
    }

    UINT64 STDMETHODCALLTYPE GetSize() override
    {
        SharedLock shared(lock_);
        return size_;
    }
    UINT64 STDMETHODCALLTYPE GetWindowSize() override { return window_; }  // fixed after create
    UINT STDMETHODCALLTYPE GetWindowCount() override
    {
        SharedLock shared(lock_);
        return static_cast<UINT>(windows_.size());
    }
    HRESULT STDMETHODCALLTYPE GetWindow(UINT index, ID3D12Resource** window) override
    {
        if (!window) return E_INVALIDARG;
        *window = nullptr;
        SharedLock shared(lock_);
        if (index >= windows_.size()) return E_INVALIDARG;
        *window = windows_[index].Get();
        (*window)->AddRef();
        return S_OK;
    }
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress(UINT64 offset, UINT64 bytes) override
    {
        if (!bytes) bytes = 1;
        SharedLock shared(lock_);
        if (offset >= size_ || bytes > size_ - offset) return 0;
        const UINT64 index = offset / window_;
        const UINT64 inside = offset - index * window_;
        if (bytes > window_ - inside) return 0;
        return windows_[static_cast<std::size_t>(index)]->GetGPUVirtualAddress() + inside;
    }
    void STDMETHODCALLTYPE Close() override
    {
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> released;
        AcquireSRWLockExclusive(&lock_);
        size_ = 0;
        released.swap(windows_);
        ReleaseSRWLockExclusive(&lock_);
        // The windows are released here, outside the lock; the last one may destroy the device.
    }

private:
    ~GpuRange() = default;

    std::atomic<ULONG> refs_{1};
    SRWLOCK lock_ = SRWLOCK_INIT;
    UINT64 size_ = 0;
    UINT64 window_ = 0;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> windows_;
};
}

namespace d3d12_xs_range {
HRESULT create(ID3D12Device* device, UINT64 size, IX2WGpuRange** range, CreateWindowFn create_window) noexcept
{
    if (!range) return E_POINTER;
    *range = nullptr;
    if (!device || size == 0 || size % tile_bytes != 0) return E_INVALIDARG;
    auto* created = new (std::nothrow) GpuRange;
    if (!created) return E_OUTOFMEMORY;
    const HRESULT hr = created->reserve(device, size, create_window ? create_window : create_reserved);
    if (FAILED(hr)) {
        created->Release();  // releases every window created so far
        return hr;
    }
    *range = created;
    return S_OK;
}
}
