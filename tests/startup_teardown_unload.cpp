// Loads the built d3d12_xs.dll by explicit path, warms the four caches that keep desktop D3D12 objects
// (#2572) through that module, and walks both detach paths it has to survive:
//
//   startup_teardown_unload <path-to-d3d12_xs.dll> <warp|hardware>
//
// Unload: FreeLibrary detaches the module with lpReserved == nullptr. Each cache is loaded and warmed on
// its own, so its release can be seen alone: the module must be gone, what it owned must be released, and
// the process must still run. The two original caches hold a reference on the device handle this fixture
// holds, so the count of that handle is the measure there. The caches added with the placed page aliases
// and the draw dumps own desktop objects instead, and a desktop object holds no reference on the wrapper:
// for them the probe hands over one reference of every object an entry owns (x2w_d3d12_cached_object_probe)
// and that reference has to stay alone once the module is gone -- the count of the entry's own object is
// the release measure.
//
// The placed-alias case also owns the heap its buffer is placed on. The probe creates that heap and hands
// its one reference over (the entry keeps the heap as a key only, as the real route does), so the fixture
// holds it across the unload and releases it only after the cache released the placed buffer: a heap must
// outlive the resources placed in it. That release has to be the last reference on the heap, and the case
// fails when it is not, because a heap left with no owner is the leak this check exists to catch.
//
// Termination: a second load warms every cache and is deliberately left loaded. On process exit the
// loader detaches the module with lpReserved != nullptr, after Windows has stopped the display driver;
// releasing a pipeline state then faults inside the driver (amdxc64 offset 0x35f2a19, through
// D3D12Core!CPipelineState::FinalRelease) and the process dies with 0xC0000409 before it can report a
// result. The caches must detach their references instead. This is the path the five
// startup_indirect_*_hardware runs hit.
//
// That second load runs in a child process (this executable again, --termination-only), because the
// evidence only exists while that process tears the module down: the shim prints one line per destroyed
// cache on stderr,
//
//   x2w-teardown <cache> detached=<n> released=<n>
//
// when X2W_TEARDOWN_REPORT is set. The child is run with it set and its report is required to say, for
// both caches that own something new here, that every owned reference was detached and none released.
//
// Negative control, the test hook the review of the ownership change asked for: X2W_TEARDOWN_NO_GUARD=
// <pages|readbacks|all> makes that cache's destructor release instead of detach during process exit, so
// the termination check has to fail with it set. Every passing run re-runs the child once per cache with
// the hook set and requires the check to fail, so a check that passes with and without a guard cannot
// report success. Set in this process's own environment the hook is passed through to the child, and this
// run fails with the reason.
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
// Probe bits: 1 the indexed-validation pipeline, 2 the rectangle-list argument converter, 4 the placed
// page aliases, 8 the draw-dump readbacks. Bit 4 also creates a heap and hands its one reference over
// through the third argument, which only it fills; the alias entry keeps that heap as a key only.
using CachedObjectsProbe = UINT(WINAPI*)(ID3D12Device*, UINT, ID3D12Heap**);
// One reference of an object a warmed entry owns: cache 4 slot 0 the placed buffer, cache 8 slot 0 the
// readback buffer, cache 8 slot 1 its fence.
using CachedObjectProbe = IUnknown*(WINAPI*)(UINT, UINT);

struct CacheCase {
    const char* name;
    UINT probe_bits;
    UINT owned;     // objects the warmed cache holds for one device
    unsigned slots; // owned objects the probe hands out; 0: measured through the device count
    bool heap;      // the probe also hands over the placed buffer's backing heap
};

// One case per cache, so a leak names the cache that kept it.
constexpr CacheCase cache_cases[] = {
    {"indexed-validation", 1u, 3u, 0u, false},  // device, root signature, pipeline
    {"rect-converter", 2u, 2u, 0u, false},      // root signature, pipeline
    {"placed-pages", 4u, 1u, 1u, true},         // the placed buffer behind the alias entry, and its heap
    {"draw-readbacks", 8u, 2u, 2u, false},      // the readback buffer and the fence that owns its completion
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
    if (length <= 1) return std::wstring();
    std::wstring wide(static_cast<std::size_t>(length), L'\0');
    if (!MultiByteToWideChar(CP_ACP, 0, path, -1, wide.data(), length)) return std::wstring();
    // -1 makes the conversion include the terminating null. Keeping it would put a null inside the path,
    // and a command line built from this string would end there: CreateProcessW truncates at the first one.
    wide.resize(static_cast<std::size_t>(length) - 1);
    return wide;
}

std::wstring narrow_to_wide(const std::string& text)
{
    return std::wstring(text.begin(), text.end());
}

std::wstring quoted(const std::wstring& text)
{
    return L'"' + text + L'"';
}

std::wstring executable_path()
{
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
        if (!length) return std::wstring();
        if (length < path.size()) {
            path.resize(length);
            return path;
        }
        path.resize(path.size() * 2);
    }
}

HMODULE load_shim(const std::wstring& path, XboxCreateDevice& create, CachedObjectsProbe& probe,
                  CachedObjectProbe& object_probe)
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
    object_probe = reinterpret_cast<CachedObjectProbe>(GetProcAddress(module, "x2w_d3d12_cached_object_probe"));
    if (!create || !probe || !object_probe) {
        std::fputs("d3d12_xs.dll does not export D3D12XboxCreateDevice, x2w_d3d12_cached_objects_probe "
                   "and x2w_d3d12_cached_object_probe\n", stderr);
        FreeLibrary(module);
        return nullptr;
    }
    const LPCWSTR entries[] = {reinterpret_cast<LPCWSTR>(probe), reinterpret_cast<LPCWSTR>(object_probe)};
    for (LPCWSTR entry : entries) {
        HMODULE owner = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                entry, &owner) ||
            owner != module) {
            std::fputs("a cache probe does not resolve inside d3d12_xs.dll\n", stderr);
            FreeLibrary(module);
            return nullptr;
        }
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

void release_objects(IUnknown* (&objects)[2])
{
    for (IUnknown*& object : objects) {
        if (object) object->Release();
        object = nullptr;
    }
}

// Gives back what the fixture holds of a placed alias entry on an error path: the placed buffer's
// reference first, then the heap behind it. A heap must outlive the resources placed in it, so the two
// releases are not interchangeable; the success path below releases them one at a time instead, because
// it reports each count.
void release_placed(ComPtr<ID3D12Heap>& heap, IUnknown* (&objects)[2], bool unloaded = true)
{
    release_objects(objects);
    if (unloaded) heap.Reset();
    else (void)heap.Detach();  // The still-loaded cache owns a placed buffer until process exit.
}

// Phase 1: load, warm one cache, unload explicitly, and check that this cache released what it owned.
int unload_path(const std::wstring& path, IDXGIAdapter* adapter, const CacheCase& cache)
{
    XboxCreateDevice create = nullptr;
    CachedObjectsProbe probe = nullptr;
    CachedObjectProbe object_probe = nullptr;
    HMODULE shim = load_shim(path, create, probe, object_probe);
    if (!shim) return 1;

    ComPtr<ID3D12Device> device;
    HRESULT hr = create(adapter, nullptr, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(device.GetAddressOf()));
    if (FAILED(hr) || !device) {
        std::fprintf(stderr, "D3D12XboxCreateDevice failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        FreeLibrary(shim);
        return 4;
    }

    ULONG before = 0, warm = 0;
    IUnknown* objects[2] = {nullptr, nullptr};
    ULONG held[2] = {0, 0};
    // The backing heap of the placed buffer, when this case has one: the probe hands it over with the one
    // reference it holds, so the fixture owns it from here to the end of this function -- across the
    // unload -- and it is released last, after the cache and the fixture both let the placed buffer go.
    ComPtr<ID3D12Heap> heap;
    const auto unload_after_error = [&] {
        const bool unloaded = FreeLibrary(shim) && !GetModuleHandleW(L"d3d12_xs.dll");
        release_placed(heap, objects, unloaded);
    };
    ULONG heap_held = 0;
    if (!cache.slots) before = reference_count(device.Get());

    ID3D12Heap* warmed_heap = nullptr;
    const UINT owned = probe(device.Get(), cache.probe_bits, &warmed_heap);
    if (warmed_heap) heap.Attach(warmed_heap);
    if (owned != cache.owned) {
        std::fprintf(stderr, "%s: probe reports %u cached objects, expected %u\n", cache.name, owned, cache.owned);
        unload_after_error();
        return 5;
    }
    if (cache.heap != (heap.Get() != nullptr)) {
        std::fprintf(stderr, "%s: the probe %s a backing heap for the placed buffer, expected %s\n", cache.name,
                     heap ? "also handed over" : "handed over no", cache.heap ? "one" : "none");
        unload_after_error();
        return 15;
    }
    if (heap) {
        heap_held = reference_count(heap.Get());
        std::printf("warm %s: backing heap handed over with %lu reference%s\n", cache.name,
                    static_cast<unsigned long>(heap_held), heap_held == 1 ? "" : "s");
    }

    if (cache.slots) {
        // One reference of each object the entry owns: before the unload it is the cache's plus this one.
        for (unsigned slot = 0; slot < cache.slots; ++slot) {
            objects[slot] = object_probe(cache.probe_bits, slot);
            if (!objects[slot]) {
                std::fprintf(stderr, "%s: the warmed cache owns no object in slot %u\n", cache.name, slot);
                unload_after_error();
                return 6;
            }
            held[slot] = reference_count(objects[slot]);
            if (held[slot] < 2) {
                std::fprintf(stderr, "%s: slot %u reports %lu references, expected the cache's and this fixture's\n",
                             cache.name, slot, static_cast<unsigned long>(held[slot]));
                unload_after_error();
                return 6;
            }
        }
    } else {
        warm = reference_count(device.Get());
        if (warm <= before) {
            std::fprintf(stderr, "%s: the cache holds no reference on the device: %lu before, %lu warm\n", cache.name,
                         static_cast<unsigned long>(before), static_cast<unsigned long>(warm));
            FreeLibrary(shim);
            return 6;
        }
    }

    if (!FreeLibrary(shim)) {
        std::fprintf(stderr, "%s: FreeLibrary failed: %lu\n", cache.name,
                     static_cast<unsigned long>(GetLastError()));
        release_placed(heap, objects, false);
        return 7;
    }
    if (GetModuleHandleW(L"d3d12_xs.dll")) {
        std::fprintf(stderr, "%s: d3d12_xs.dll is still loaded after FreeLibrary\n", cache.name);
        release_placed(heap, objects, false);
        return 8;
    }

    if (cache.slots) {
        for (unsigned slot = 0; slot < cache.slots; ++slot) {
            const ULONG after = reference_count(objects[slot]);
            if (after + 1 != held[slot]) {
                std::fprintf(stderr, "%s: the unloaded module kept a reference on slot %u: %lu -> %lu (expected %lu)\n",
                             cache.name, slot, static_cast<unsigned long>(held[slot]),
                             static_cast<unsigned long>(after), static_cast<unsigned long>(held[slot] - 1));
                release_placed(heap, objects);
                return 9;
            }
            std::printf("unload %s: slot %u reference %lu -> %lu\n", cache.name, slot,
                        static_cast<unsigned long>(held[slot]), static_cast<unsigned long>(after));
        }
        // The module dropped its reference on the placed buffer above; this drops the fixture's. The heap
        // has to outlive the buffer placed in it, so it is released only now -- and that release is the
        // last reference, which is what proves the fixture, and not the cache, owned the heap.
        release_objects(objects);
        if (heap) {
            // Still alive with the module gone, and what keeps it alive is the fixture's reference: it was
            // taken from the probe before the unload and this scope is where it ends.
            const ULONG alive = reference_count(heap.Get());
            std::printf("unload %s: heap alive after the unload, %lu reference%s, placed buffer released first\n",
                        cache.name, static_cast<unsigned long>(alive), alive == 1 ? "" : "s");
            ID3D12Heap* released = heap.Detach();
            const ULONG last = released->Release();
            if (last != 0) {
                std::fprintf(stderr, "%s: the fixture's heap release was not the last: %lu references left, so "
                                     "something else owns the backing heap\n",
                             cache.name, static_cast<unsigned long>(last));
                return 16;
            }
            std::printf("unload %s: heap released last, %lu -> 0\n", cache.name, static_cast<unsigned long>(alive));
        }
        std::printf("unload %s: %u objects released, module unloaded\n", cache.name, owned);
        return 0;
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

// The child body: load, warm every cache, return without unloading, so the loader detaches the module
// during process exit. Nothing else runs here -- the report lines come out of the CRT teardown.
int termination_warm(const std::wstring& path, const std::string& backend)
{
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    const int selected = select_adapter(backend == "warp", factory, adapter);
    if (selected) return selected;
    DXGI_ADAPTER_DESC desc{};
    if (SUCCEEDED(adapter->GetDesc(&desc)))
        std::printf("termination child backend=%s adapter=%ls vendor=%04x device=%04x\n", backend.c_str(),
                    desc.Description, desc.VendorId, desc.DeviceId);

    XboxCreateDevice create = nullptr;
    CachedObjectsProbe probe = nullptr;
    CachedObjectProbe object_probe = nullptr;
    HMODULE shim = load_shim(path, create, probe, object_probe);
    if (!shim) return 1;

    ComPtr<ID3D12Device> device;
    HRESULT hr = create(adapter.Get(), nullptr, __uuidof(ID3D12Device),
                        reinterpret_cast<void**>(device.GetAddressOf()));
    if (FAILED(hr) || !device) {
        std::fprintf(stderr, "D3D12XboxCreateDevice failed on the termination load: 0x%08lx\n",
                     static_cast<unsigned long>(hr));
        return 4;
    }
    UINT caches = 0, expected = 0;
    for (const CacheCase& cache : cache_cases) {
        caches |= cache.probe_bits;
        expected += cache.owned;
    }
    ID3D12Heap* placed_heap = nullptr;
    const UINT owned = probe(device.Get(), caches, &placed_heap);
    if (owned != expected) {
        std::fprintf(stderr, "termination: probe reports %u cached objects, expected %u\n", owned, expected);
        return 5;
    }
    if (!placed_heap) {
        std::fputs("termination: the placed alias probe handed over no backing heap\n", stderr);
        return 6;
    }
    // Deliberately no FreeLibrary: the shim stays loaded, so the loader detaches it during process exit.
    // Releasing the device here, while the process and the driver still run, is the test's own reference.
    // The backing heap is kept for the same reason and is not released either: the exit path detaches the
    // placed buffer instead of releasing it, so the heap has to outlive the process exit. Unlike before
    // this reference has an owner -- this child -- instead of being abandoned by the probe.
    std::printf("termination: %u objects left to the process exit path\n", owned);
    return 0;
}

// One x2w-teardown line of the child's report, if it printed one for this cache.
bool teardown_line(const std::string& text, const char* cache, unsigned& detached, unsigned& released)
{
    const std::string prefix = std::string("x2w-teardown ") + cache + " detached=";
    for (std::size_t at = text.find(prefix); at != std::string::npos; at = text.find(prefix, at + 1)) {
        unsigned detached_value = 0, released_value = 0;
        if (std::sscanf(text.c_str() + at + prefix.size(), "%u released=%u", &detached_value, &released_value) == 2) {
            detached = detached_value;
            released = released_value;
            return true;
        }
    }
    return false;
}

// The termination contract: the child exited cleanly and both caches that own something here report every
// reference detached and none released. Answers the failures found, empty when the contract holds.
struct ChildRun {
    DWORD exit_code = 0;
    std::string text;  // everything the child wrote, stdout and stderr on one pipe
};

std::string termination_failure(const ChildRun& run)
{
    std::string failure;
    char line[256];
    const auto add = [&failure](const char* text) {
        if (!failure.empty()) failure += "; ";
        failure += text;
    };
    if (run.exit_code != 0) {
        std::snprintf(line, sizeof(line), "the child exited 0x%08lx instead of returning",
                      static_cast<unsigned long>(run.exit_code));
        add(line);
    }
    struct Expectation {
        const char* cache;
        unsigned detached;
    };
    constexpr Expectation expectations[] = {{"pages", 1u}, {"readbacks", 2u}};
    for (const Expectation& want : expectations) {
        unsigned detached = 0, released = 0;
        if (!teardown_line(run.text, want.cache, detached, released)) {
            std::snprintf(line, sizeof(line), "x2w-teardown %s: no report line", want.cache);
            add(line);
        } else if (detached != want.detached || released != 0) {
            std::snprintf(line, sizeof(line), "x2w-teardown %s detached=%u released=%u (want detached=%u released=0)",
                          want.cache, detached, released, want.detached);
            add(line);
        }
    }
    return failure;
}

// An environment variable of this process; empty when it is unset or empty (the shim reads both the same
// way).
std::wstring variable(const wchar_t* name)
{
    wchar_t value[64]{};
    const DWORD length = GetEnvironmentVariableW(name, value, 64);
    return length && length < 64 ? std::wstring(value, length) : std::wstring();
}

// Sets (non-empty) or clears (empty) an environment variable of this process.
void set_variable(const wchar_t* name, const std::wstring& value)
{
    SetEnvironmentVariableW(name, value.empty() ? nullptr : value.c_str());
}

// The caller's own X2W_TEARDOWN_NO_GUARD, passed through to the guarded child so that setting it makes
// this run fail -- that is the documented negative control at fixture level.
std::wstring caller_no_guard()
{
    return variable(L"X2W_TEARDOWN_NO_GUARD");
}

// Runs this executable once as a termination child. Its stdout and stderr share one pipe, read while it
// runs, so a child that dies mid-teardown still leaves whatever it printed; stdin comes from NUL. The
// child inherits this process's environment, so the two teardown variables are set here around
// CreateProcessW and the caller's values are put back afterwards.
bool run_child(const std::wstring& command, const std::wstring& no_guard, ChildRun& run)
{
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_end = nullptr, write_end = nullptr;
    if (!CreatePipe(&read_end, &write_end, &security, 0)) {
        std::fprintf(stderr, "CreatePipe failed: %lu\n", static_cast<unsigned long>(GetLastError()));
        return false;
    }
    HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0,
                               nullptr);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = input;
    startup.hStdOutput = write_end;
    startup.hStdError = write_end;
    PROCESS_INFORMATION process{};
    const std::wstring saved_report = variable(L"X2W_TEARDOWN_REPORT");
    const std::wstring saved_guard = variable(L"X2W_TEARDOWN_NO_GUARD");
    set_variable(L"X2W_TEARDOWN_REPORT", L"1");
    set_variable(L"X2W_TEARDOWN_NO_GUARD", no_guard);
    std::wstring line = command;  // CreateProcessW may write to its command line
    const BOOL created = CreateProcessW(nullptr, line.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                                        nullptr, &startup, &process);
    const DWORD error = GetLastError();
    set_variable(L"X2W_TEARDOWN_REPORT", saved_report);
    set_variable(L"X2W_TEARDOWN_NO_GUARD", saved_guard);
    CloseHandle(write_end);
    if (input != INVALID_HANDLE_VALUE) CloseHandle(input);
    if (!created) {
        std::fprintf(stderr, "CreateProcessW failed: %lu for %ls\n", static_cast<unsigned long>(error),
                     command.c_str());
        CloseHandle(read_end);
        return false;
    }
    CloseHandle(process.hThread);
    for (;;) {
        char buffer[4096];
        DWORD got = 0;
        if (!ReadFile(read_end, buffer, sizeof(buffer), &got, nullptr) || !got) break;
        run.text.append(buffer, got);
    }
    CloseHandle(read_end);
    WaitForSingleObject(process.hProcess, INFINITE);
    GetExitCodeProcess(process.hProcess, &run.exit_code);
    CloseHandle(process.hProcess);
    return true;
}

// Phase 2: the termination load runs in a child of this process. The guarded child has to pass; with
// X2W_TEARDOWN_NO_GUARD naming one cache, the same checks have to fail -- the negative control the review
// asked for, run per cache on every pass so that a check insensitive to either guard cannot go green.
int termination_path(const std::wstring& path, const std::string& backend)
{
    const std::wstring executable = executable_path();
    if (executable.empty()) {
        std::fputs("GetModuleFileNameW failed: cannot run the termination child\n", stderr);
        return 10;
    }
    const std::wstring command = quoted(executable) + L" --termination-only " + quoted(path) + L" " +
                                 narrow_to_wide(backend);
    const std::wstring no_guard = caller_no_guard();
    if (!no_guard.empty())
        std::printf("termination: X2W_TEARDOWN_NO_GUARD=%ls is set in this environment: the guarded check must fail\n",
                    no_guard.c_str());

    ChildRun guarded;
    if (!run_child(command, no_guard, guarded)) return 10;
    std::printf("%s", guarded.text.c_str());
    const std::string failure = termination_failure(guarded);
    if (!failure.empty()) {
        std::fprintf(stderr, "termination: %s\n", failure.c_str());
        return 11;
    }
    std::printf("termination: both caches detached every owned reference, released none\n");
    if (!no_guard.empty()) {
        // The caller set the hook and the check still passed: the hook did nothing, so nothing here
        // proves the guard is needed.
        std::fprintf(stderr, "negative control: the termination check passed with X2W_TEARDOWN_NO_GUARD=%ls: "
                             "the hook has no effect on this build\n", no_guard.c_str());
        return 14;
    }

    for (const char* cache : {"pages", "readbacks"}) {
        ChildRun unguarded;
        if (!run_child(command, narrow_to_wide(cache), unguarded)) return 12;
        const std::string unguarded_failure = termination_failure(unguarded);
        if (unguarded_failure.empty()) {
            std::fprintf(stderr, "negative control: the termination check passed with X2W_TEARDOWN_NO_GUARD=%s, "
                                 "so it does not prove the %s guard is needed\n", cache, cache);
            std::fprintf(stderr, "%s", unguarded.text.c_str());
            return 13;
        }
        std::printf("negative control X2W_TEARDOWN_NO_GUARD=%s: the check failed as it must (%s)\n", cache,
                    unguarded_failure.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 4 && std::string_view(argv[1]) == "--termination-only") {
        const std::string_view backend = argv[3];
        if (backend != "warp" && backend != "hardware") {
            std::fputs("unknown backend\n", stderr);
            return 2;
        }
        const std::wstring path = widen(argv[2]);
        if (path.empty()) {
            std::fputs("cannot convert the d3d12_xs.dll path\n", stderr);
            return 2;
        }
        return termination_warm(path, std::string(backend));
    }
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
        std::printf("startup_teardown_unload backend=%.*s adapter=%ls vendor=%04x device=%04x\n",
                    static_cast<int>(backend.size()), backend.data(), desc.Description, desc.VendorId, desc.DeviceId);

    int result = 0;
    for (const CacheCase& cache : cache_cases) {
        result = unload_path(path, adapter.Get(), cache);
        if (result) break;
    }
    if (result == 0) result = termination_path(path, std::string(backend));
    std::printf("startup_teardown_unload %.*s: %s\n", static_cast<int>(backend.size()), backend.data(),
                result == 0 ? "PASS" : "FAIL");
    return result;
}
