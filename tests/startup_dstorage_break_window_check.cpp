// A DirectStorage request whose destination lies in an XMem run that is being taken apart must wait for the break
// and then land, not be refused.
//
// With shared views (X2W_XMEM_SHARED_VIEWS=1) adjacent pages of one batch share a view, and unmapping one page of
// such a run unmaps the whole run and maps its other pages again, under xmem's mapping mutex. For that moment a
// surviving page reads as reserved. The xmem build with its test controls holds an unmap in that window
// (X2W_XMEM_TEST_HOLD_WINDOW=<name>: it sets "<name>.unmapped" and waits up to five seconds for "<name>.release").
//
// usage: startup_dstorage_break_window_check <xmem.dll built with its test controls> <XCRDAPI.dll>
// Cases, each against the real stand-in's queue with a memory source:
// - no break: a request into a mapped page of the run lands;
// - read-only destination: refused, nothing written (the writability check also runs on the xmem path);
// - break held: while an unmap holds page 2's run apart (page 2 reads MEM_RESERVE), a request into page 2 must not
//   complete for the whole hold window (1000 ms, polled every 10 ms); after the release it completes with S_OK and
//   every byte landed. An XCRDAPI that judges the destination with its own VirtualQuery instead refuses inside the
//   window: "FAIL break window: refused while the run was broken" (the negative control in CMakeLists.txt).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using u64 = std::uint64_t;
struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using VirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using AllocatePagesFn = BOOL(WINAPI*)(ULONG_PTR, PULONG_PTR, PULONG_PTR);
using MapPagesFn = void*(WINAPI*)(void*, ULONG_PTR, const ULONG_PTR*);
using ScatterFn = BOOL(WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);

constexpr SIZE_T page = 65536;
constexpr unsigned run_pages = 8;
constexpr unsigned bytes = 4096;
constexpr DWORD hold_ms = 1000;  // the window in which a request into the broken run must not complete
constexpr DWORD poll_ms = 10;
unsigned char source[bytes];
int failures;

Object* status;
Object* queue;
unsigned next_index;

void fail(const char* name, const char* what)
{
    std::printf("FAIL %s: %s\n", name, what);
    ++failures;
}

// Enqueues one uncompressed memory-source request and submits it; returns its status index.
unsigned submit(unsigned char* destination, unsigned capacity, unsigned size)
{
    unsigned char request[0x48] = {};
    *reinterpret_cast<u64*>(request + 0x00) = 0x400000;
    *reinterpret_cast<unsigned char**>(request + 0x08) = destination;
    *reinterpret_cast<unsigned*>(request + 0x18) = capacity;
    *reinterpret_cast<const void**>(request + 0x20) = source;
    *reinterpret_cast<unsigned*>(request + 0x30) = size;
    const unsigned index = next_index++;
    reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
    reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, index);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    return index;
}

// Polls the status entry for up to milliseconds; true with its HRESULT when it completed.
bool completed_within(unsigned index, DWORD milliseconds, long& hr)
{
    for (DWORD waited = 0;; waited += poll_ms) {
        if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index)) {
            hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, index);
            return true;
        }
        if (waited >= milliseconds) return false;
        Sleep(poll_ms);
    }
}

bool filled(const unsigned char* at, unsigned char value, unsigned count)
{
    for (unsigned i = 0; i < count; ++i)
        if (at[i] != value) return false;
    return true;
}

DWORD query_state(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) ? info.State : 0;
}

struct Unmap {
    ScatterFn scatter;
    void* address;
    BOOL result;
};

DWORD WINAPI unmap_one(void* context)
{
    auto* unmap = static_cast<Unmap*>(context);
    unmap->result = unmap->scatter(&unmap->address, 1, nullptr);  // no pages: unmap
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::printf("usage: startup_dstorage_break_window_check <xmem.dll with test controls> <XCRDAPI.dll>\n");
        return 2;
    }
    char hold_name[64];
    std::snprintf(hold_name, sizeof(hold_name), "x2w-dstorage-break-window-%lu", GetCurrentProcessId());
    SetEnvironmentVariableA("X2W_XMEM_SHARED_VIEWS", "1");
    SetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", hold_name);

    // xmem first: XCRDAPI finds it by module name (GetModuleHandleW(L"xmem.dll")).
    HMODULE xmem = LoadLibraryA(argv[1]);
    if (!xmem || GetModuleHandleW(L"xmem.dll") != xmem) {
        std::printf("FAIL setup: %s is not loaded as xmem.dll\n", argv[1]);
        return 1;
    }
    const auto virtual_alloc = reinterpret_cast<VirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto allocate = reinterpret_cast<AllocatePagesFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<MapPagesFn>(GetProcAddress(xmem, "XMemMapPhysicalPages"));
    const auto scatter = reinterpret_cast<ScatterFn>(GetProcAddress(xmem, "XMemMapPhysicalPagesScatter"));
    HMODULE library = LoadLibraryA(argv[2]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    if (!virtual_alloc || !allocate || !map || !scatter || !get_factory) {
        std::printf("FAIL setup: missing export\n");
        return 1;
    }
    void* factory_pointer = nullptr;
    if (get_factory(nullptr, &factory_pointer) != 0 || !factory_pointer) return 1;
    auto* factory = static_cast<Object*>(factory_pointer);
    void* status_pointer = nullptr;
    void* queue_pointer = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 8, "Status", nullptr, &status_pointer) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_pointer) != 0)
        return 1;
    status = static_cast<Object*>(status_pointer);
    queue = static_cast<Object*>(queue_pointer);
    for (unsigned i = 0; i < bytes; ++i) source[i] = static_cast<unsigned char>(i % 251 + 1);

    // Eight physical pages mapped as one run at the start of a startup reservation (as xmem_page_runs_check.py).
    auto* whole = static_cast<unsigned char*>(virtual_alloc(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    ULONG_PTR count = run_pages, tokens[run_pages] = {};
    unsigned char* base = whole ? whole + page : nullptr;
    if (!base || !allocate(0x20400000, &count, tokens) || count != run_pages || map(base, run_pages, tokens) != base) {
        std::printf("FAIL setup: could not map %u pages (err %lu)\n", run_pages, GetLastError());
        return 1;
    }
    MEMORY_BASIC_INFORMATION run{};
    if (!VirtualQuery(base, &run, sizeof(run)) || run.Type != MEM_MAPPED || run.RegionSize < run_pages * page) {
        std::printf("FAIL setup: the %u pages are not one view (region 0x%zx), so no unmap takes a run apart\n", run_pages,
                    static_cast<std::size_t>(run.RegionSize));
        return 1;
    }
    unsigned char* target = base + 2 * page;

    // No break: the request lands.
    std::memset(target, 0xAA, bytes);
    long hr = -1;
    if (!completed_within(submit(target, bytes, bytes), 5000, hr)) fail("no break", "no completion");
    else if (hr != 0) fail("no break", "request failed");
    else if (std::memcmp(target, source, bytes) != 0) fail("no break", "bytes differ");
    else std::printf("ok: no break: the request into page 2 of the run landed\n");

    // Read-only destination: refused before any write, on the xmem path too.
    auto* plain = static_cast<unsigned char*>(VirtualAlloc(nullptr, 2 * bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    DWORD old = 0;
    if (!plain) {
        fail("read-only destination", "VirtualAlloc");
    } else {
        std::memset(plain, 0xAA, 2 * bytes);
        if (!VirtualProtect(plain + bytes, bytes, PAGE_READONLY, &old)) fail("read-only destination", "VirtualProtect");
        else if (!completed_within(submit(plain, 2 * bytes, bytes), 5000, hr)) fail("read-only destination", "no completion");
        else if (hr == 0) fail("read-only destination", "request succeeded");
        else if (!filled(plain, 0xAA, 2 * bytes)) fail("read-only destination", "bytes were written");
        else std::printf("ok: read-only destination: refused, nothing written\n");
        VirtualFree(plain, 0, MEM_RELEASE);
    }

    // Break held: an unmap of page 5 takes the run apart and holds it there until released.
    char event_name[96];
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", hold_name);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", hold_name);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::memset(target, 0xAA, bytes);
    Unmap unmap{scatter, base + 5 * page, FALSE};
    HANDLE thread = unmapped && release ? CreateThread(nullptr, 0, unmap_one, &unmap, 0, nullptr) : nullptr;
    if (!thread || WaitForSingleObject(unmapped, 10000) != WAIT_OBJECT_0) {
        std::printf("FAIL break window: the unmap never held the run apart (hook missing or timed out)\n");
        if (release) SetEvent(release);
        return 1;
    }
    if (query_state(target) != MEM_RESERVE) {
        std::printf("FAIL break window: page 2 is not reserved while the run is apart (state 0x%lx)\n", query_state(target));
        SetEvent(release);
        return 1;
    }
    std::printf("ok: break held: page 2 reads MEM_RESERVE while the run is apart\n");
    const unsigned index = submit(target, bytes, bytes);
    const bool early = completed_within(index, hold_ms, hr);
    const long early_hr = hr;
    SetEvent(release);
    const bool joined = WaitForSingleObject(thread, 10000) == WAIT_OBJECT_0;
    if (early) {
        if (early_hr != 0) fail("break window", "refused while the run was broken");
        else fail("break window", "completed while the run was broken");
    } else {
        std::printf("ok: break held: no completion within %lu ms\n", hold_ms);
        if (!completed_within(index, 5000, hr)) fail("break window", "no completion after the release");
        else if (hr != 0) fail("break window", "request failed after the release");
        else if (std::memcmp(target, source, bytes) != 0) fail("break window", "bytes differ after the release");
        else std::printf("ok: break released: the request completed with S_OK and every byte landed\n");
    }
    if (!joined || !unmap.result) fail("break window", "the unmap of page 5 did not finish");
    CloseHandle(thread);

    if (failures) return 1;
    std::printf("dstorage break window: the request waited for the break and landed\n");
    return 0;
}
