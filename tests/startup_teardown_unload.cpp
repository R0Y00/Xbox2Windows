// Loads the built d3d12_xs.dll by explicit path, warms the two caches that keep desktop D3D12 objects
// (#2572) through that module, and walks both detach paths it has to survive:
//
//   startup_teardown_unload <path-to-d3d12_xs.dll> <warp|hardware>
//
// Unload: FreeLibrary detaches the module with lpReserved == nullptr. Each cache is loaded and warmed on
// its own, so its release can be seen alone: the device's reference count must return to the count the
// test itself holds (otherwise an owned object was leaked), the module must be gone, and the process must
// still run.
//
// Termination: a second load warms both caches and is deliberately left loaded. On process exit the
// loader detaches the module with lpReserved != nullptr, after Windows has stopped the display driver;
// releasing a pipeline state then faults inside the driver (amdxc64 offset 0x35f2a19, through
// D3D12Core!CPipelineState::FinalRelease) and the process dies with 0xC0000409 before it can report a
// result. The caches must detach their references instead. This is the path the five
// startup_indirect_*_hardware runs hit.
//
// The fixture is source-built and loads no title.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <string>
#include <string_view>

using Microsoft::WRL::ComPtr;

namespace {

using XboxCreateDevice = HRESULT(WINAPI*)(IUnknown*, const void*, REFIID, void**);
// Probe bits: 1 the indexed-validation pipeline, 2 the rectangle-list argument converter.
using CachedObjectsProbe = UINT(WINAPI*)(ID3D12Device*, UINT);

struct CacheCase {
    const char* name;
    UINT probe_bits;
    UINT owned;  // objects the warmed cache holds for one device
};

// One case per cache, so a leak names the cache that kept it.
constexpr CacheCase cache_cases[] = {
    {"indexed-validation", 1u, 3u},  // device, root signature, pipeline
    {"rect-converter", 2u, 2u},      // root signature, pipeline
};

// Same selection convention as the other D3D12 fixtures: WARP on request, otherwise the first
// non-software adapter in high-performance order. 77 when the machine has no physical adapter.
int select_adapter(bool warp, ComPtr<IDXGIFactory6>& factory, ComPtr<IDXGIAdapter>& adapter)
{
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::fprintf(stderr, "CreateDXGIFactory2 failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return 3;
    }
    if (warp) {
        hr = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        if (FAILED(hr) || !adapter) {
            std::fprintf(stderr, "WARP adapter unavailable: 0x%08lx\n", static_cast<unsigned long>(hr));
            return 3;
        }
        return 0;
    }
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> candidate;
        hr = factory->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
        if (hr == DXGI_ERROR_NOT_FOUND) return 77;
        if (FAILED(hr)) {
            std::fprintf(stderr, "EnumAdapterByGpuPreference failed: 0x%08lx\n", static_cast<unsigned long>(hr));
            return 3;
        }
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc))) return 3;
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            adapter = candidate;
            return 0;
        }
    }
}

std::wstring widen(const char* path)
{
    const int length = MultiByteToWideChar(CP_ACP, 0, path, -1, nullptr, 0);
    if (length <= 0) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    if (!MultiByteToWideChar(CP_ACP, 0, path, -1, wide.data(), length)) return std::wstring();
    return wide;
}

HMODULE load_shim(const std::wstring& path, XboxCreateDevice& create, CachedObjectsProbe& probe)
{
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) {
        std::fprintf(stderr, "LoadLibraryExW(%ls) failed: %lu\n", path.c_str(),
                     static_cast<unsigned long>(GetLastError()));
        return nullptr;
    }
    create = reinterpret_cast<XboxCreateDevice>(GetProcAddress(module, "D3D12XboxCreateDevice"));
    probe = reinterpret_cast<CachedObjectsProbe>(GetProcAddress(module, "x2w_d3d12_cached_objects_probe"));
    if (!create || !probe) {
        std::fputs("d3d12_xs.dll does not export D3D12XboxCreateDevice and x2w_d3d12_cached_objects_probe\n", stderr);
        FreeLibrary(module);
        return nullptr;
    }
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(probe), &owner) ||
        owner != module) {
        std::fputs("x2w_d3d12_cached_objects_probe does not resolve inside d3d12_xs.dll\n", stderr);
        FreeLibrary(module);
        return nullptr;
    }
    return module;
}

// The count returned includes the caller's own reference: AddRef then Release leaves the object with the
// references it had and answers how many there are.
ULONG reference_count(IUnknown* object)
{
    object->AddRef();
    return object->Release();
}

// Phase 1: load, warm one cache, unload explicitly, and check that this cache released what it owned.
int unload_path(const std::wstring& path, IDXGIAdapter* adapter, const CacheCase& cache)
{
    XboxCreateDevice create = nullptr;
    CachedObjectsProbe probe = nullptr;
    HMODULE shim = load_shim(path, create, probe);
    if (!shim) return 1;

    ComPtr<ID3D12Device> device;
    HRESULT hr = create(adapter, nullptr, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(device.GetAddressOf()));
    if (FAILED(hr) || !device) {
        std::fprintf(stderr, "D3D12XboxCreateDevice failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        FreeLibrary(shim);
        return 4;
    }

    const ULONG before = reference_count(device.Get());
    const UINT owned = probe(device.Get(), cache.probe_bits);
    if (owned != cache.owned) {
        std::fprintf(stderr, "%s: probe reports %u cached objects, expected %u\n", cache.name, owned, cache.owned);
        FreeLibrary(shim);
        return 5;
    }
    const ULONG warm = reference_count(device.Get());
    if (warm <= before) {
        std::fprintf(stderr, "%s: the cache holds no reference on the device: %lu before, %lu warm\n", cache.name,
                     static_cast<unsigned long>(before), static_cast<unsigned long>(warm));
        FreeLibrary(shim);
        return 6;
    }

    if (!FreeLibrary(shim)) {
        std::fprintf(stderr, "%s: FreeLibrary failed: %lu\n", cache.name,
                     static_cast<unsigned long>(GetLastError()));
        return 7;
    }
    if (GetModuleHandleW(L"d3d12_xs.dll")) {
        std::fprintf(stderr, "%s: d3d12_xs.dll is still loaded after FreeLibrary\n", cache.name);
        return 8;
    }
    const ULONG after = reference_count(device.Get());
    if (after != before) {
        std::fprintf(stderr, "%s: the unloaded module left %lu device references behind (expected %lu)\n",
                     cache.name, static_cast<unsigned long>(after), static_cast<unsigned long>(before));
        return 9;
    }
    std::printf("unload %s: %u objects released, references %lu -> %lu, module unloaded\n", cache.name, owned,
                static_cast<unsigned long>(warm), static_cast<unsigned long>(after));
    return 0;
}

// Phase 2: load and warm both caches, then let the process exit with the module still loaded. Nothing may
// be released from the caches then, so nothing reaches the driver Windows has already stopped.
int termination_path(const std::wstring& path, IDXGIAdapter* adapter)
{
    XboxCreateDevice create = nullptr;
    CachedObjectsProbe probe = nullptr;
    HMODULE shim = load_shim(path, create, probe);
    if (!shim) return 1;

    ComPtr<ID3D12Device> device;
    HRESULT hr = create(adapter, nullptr, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(device.GetAddressOf()));
    if (FAILED(hr) || !device) {
        std::fprintf(stderr, "D3D12XboxCreateDevice failed on the second load: 0x%08lx\n",
                     static_cast<unsigned long>(hr));
        return 4;
    }
    UINT expected = 0;
    for (const CacheCase& cache : cache_cases) expected += cache.owned;
    const UINT owned = probe(device.Get(), 3u);
    if (owned != expected) {
        std::fprintf(stderr, "termination: probe reports %u cached objects, expected %u\n", owned, expected);
        return 5;
    }
    // Deliberately no FreeLibrary: the shim stays loaded, so the loader detaches it during process exit.
    // Releasing the device here, while the process and the driver still run, is the test's own reference.
    std::printf("termination: %u objects left to the process exit path\n", owned);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <path-to-d3d12_xs.dll> <warp|hardware>\n", argv[0]);
        return 2;
    }
    const std::string_view backend = argv[2];
    if (backend != "warp" && backend != "hardware") {
        std::fputs("unknown backend\n", stderr);
        return 2;
    }
    const std::wstring path = widen(argv[1]);
    if (path.empty()) {
        std::fputs("cannot convert the d3d12_xs.dll path\n", stderr);
        return 2;
    }

    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    const int selected = select_adapter(backend == "warp", factory, adapter);
    if (selected) return selected;
    DXGI_ADAPTER_DESC desc{};
    if (SUCCEEDED(adapter->GetDesc(&desc)))
        std::printf("startup_teardown_unload backend=%s adapter=%ls vendor=%04x device=%04x\n", argv[2],
                    desc.Description, desc.VendorId, desc.DeviceId);

    int result = 0;
    for (const CacheCase& cache : cache_cases) {
        result = unload_path(path, adapter.Get(), cache);
        if (result) break;
    }
    if (result == 0) result = termination_path(path, adapter.Get());
    std::printf("startup_teardown_unload %s: %s\n", argv[2], result == 0 ? "PASS" : "FAIL");
    return result;
}
