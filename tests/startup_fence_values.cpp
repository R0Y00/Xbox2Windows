// Compare the console fence's method and completed-value address with the desktop fence.
// Signals deliberately decrease the value: a fence is not an atomic maximum.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    // The temporary private-data bridge is also used by startup_resource_budget.
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

bool wait_value(ID3D12Fence* fence, UINT64 value)
{
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool done = event && SUCCEEDED(fence->SetEventOnCompletion(value, event)) &&
                      WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    return done && fence->GetCompletedValue() == value;
}

bool shadow_is(volatile LONG64* shadow, UINT64 value)
{
    const ULONGLONG end = GetTickCount64() + 10000;
    do {
        if (static_cast<UINT64>(*shadow) == value) return true;
        Sleep(1);
    } while (GetTickCount64() < end);
    return false;
}
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "--queue-rewind"))) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (!std::strcmp(argv[1], "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 77;
    } else {
        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                   IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    }
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(adapter->GetDesc(&desc))) return 3;
    LARGE_INTEGER driver{};
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    std::printf("adapter=%ls driver=%llu\n", desc.Description, static_cast<unsigned long long>(driver.QuadPart));
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(guest.GetAddressOf()))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 3;

    ComPtr<IUnknown> fence;
    if (FAILED(invoke(guest.Get(), 36, UINT64{0}, D3D12_FENCE_FLAG_NONE,
                      &__uuidof(ID3D12Fence), fence.GetAddressOf()))) return 4;
    const auto real = desktop_object<ID3D12Fence>(guest.Get(), device.Get(), fence.Get());
    if (!real) return 4;
    auto* shadow = invoke<volatile LONG64*>(fence.Get(), 12, UINT64{0}, UINT64{0}, UINT64{0});
    if (!shadow) return 4;

    unsigned failures = 0;

    // Two callers cross a barrier, then each Signal its own value. The fixture repeats this
    // so a stale post-call observation cannot disagree with the native fence or method.
    std::atomic<unsigned> concurrent_failures{};
    for (unsigned pair = 0; pair < 128; ++pair) {
        std::atomic<unsigned> ready{};
        std::atomic<bool> go{};
        auto caller = [&](UINT64 value) {
            ++ready;
            while (!go.load()) std::this_thread::yield();
            if (FAILED(invoke(fence.Get(), 10, value))) ++concurrent_failures;
        };
        std::thread one(caller, UINT64{1});
        std::thread two(caller, UINT64{2});
        while (ready.load() != 2) std::this_thread::yield();
        go.store(true);
        one.join();
        two.join();

        const UINT64 native = real->GetCompletedValue();
        const UINT64 direct = static_cast<UINT64>(*shadow);
        const UINT64 observed = invoke<UINT64>(fence.Get(), 8);
        if (native != direct || native != observed) ++concurrent_failures;
    }
    const UINT64 concurrent_native = real->GetCompletedValue();
    std::printf("concurrent CPU signals: native=%llu method=%llu address=%llu %s\n",
                concurrent_native, invoke<UINT64>(fence.Get(), 8),
                static_cast<UINT64>(*shadow), concurrent_failures.load() ? "FAIL" : "PASS");
    failures += concurrent_failures.load() ? 1 : 0;

    for (const UINT64 value : {UINT64{9}, UINT64{2}, UINT64{0}, UINT64{7}}) {
        if (FAILED(invoke(fence.Get(), 10, value))) return 5;
        const UINT64 native = real->GetCompletedValue();
        // CPU Signal must publish synchronously, before a method read could refresh the word.
        const UINT64 direct = static_cast<UINT64>(*shadow);
        const UINT64 observed = invoke<UINT64>(fence.Get(), 8);
        const bool ok = native == value && observed == native && direct == native;
        std::printf("cpu Signal(%llu): desktop=%llu method=%llu address=%llu %s\n", value,
                    native, observed, direct, ok ? "PASS" : "FAIL");
        failures += !ok;
    }
    if (argc == 2) {
        std::printf("CPU fence values: %u failures\n", failures);
        return failures ? 1 : 0;
    }
    // Optional diagnostic for the separate, unresolved queued-rewind/address contract.
    // Not registered as a passing CTest: a target-value event is already satisfied before
    // a lower queue signal executes, so a private monotonic queue marker is needed.
    // The queue cannot reach its lower signal until the CPU releases the blocker.
    ComPtr<IUnknown> blocker, marker, queue;
    for (auto* output : {blocker.GetAddressOf(), marker.GetAddressOf()})
        if (FAILED(invoke(guest.Get(), 36, UINT64{0}, D3D12_FENCE_FLAG_NONE,
                          &__uuidof(ID3D12Fence), output))) return 4;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    if (FAILED(invoke(guest.Get(), 8, &queue_desc, &__uuidof(ID3D12CommandQueue), queue.GetAddressOf()))) return 4;
    const auto real_marker = desktop_object<ID3D12Fence>(guest.Get(), device.Get(), marker.Get());
    if (!real_marker) return 4;
    if (FAILED(invoke(queue.Get(), 17, blocker.Get(), UINT64{1})) ||
        FAILED(invoke(queue.Get(), 16, fence.Get(), UINT64{3})) ||
        FAILED(invoke(queue.Get(), 16, marker.Get(), UINT64{1}))) return 5;
    Sleep(25);
    if (real->GetCompletedValue() != 7 || static_cast<UINT64>(*shadow) != 7) {
        std::puts("FAIL: pending queue signal changed the completed value");
        ++failures;
    }
    if (FAILED(invoke(blocker.Get(), 10, UINT64{1})) || !wait_value(real_marker.Get(), 1)) return 5;
    // Read only the address until the callback publishes: GetCompletedValue must not hide a stale address.
    const bool published = shadow_is(shadow, 3);
    const UINT64 native = real->GetCompletedValue(), observed = invoke<UINT64>(fence.Get(), 8);
    const bool ok = published && native == 3 && observed == native;
    std::printf("queue Signal(3): desktop=%llu method=%llu address=%llu %s\n", native,
                observed, static_cast<UINT64>(*shadow), ok ? "PASS" : "FAIL");
    failures += !ok;
    std::printf("fence values: %u failures\n", failures);
    return failures ? 1 : 0;
}
