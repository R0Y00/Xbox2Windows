// A DirectStorage request whose destination is not writable up to its declared capacity is refused
// before any byte is written (#870: a write fault inside the stand-in's copy, not in title code).
//
// Memory-source requests go through the queue of the real stand-in DLL (path in argv[1]). The
// destination is three pages that start out 0xAA. Each refusal case makes one part of them unusable
// and expects: a failure status, the process alive, every readable byte still 0xAA. After the
// refusals, a request into a fully writable buffer on the same queue must still succeed.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

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
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);

constexpr unsigned page = 4096;
constexpr unsigned pages = 3;
constexpr unsigned bytes = pages * page;
constexpr unsigned cases = 15;
unsigned char source[bytes];
int failures;
volatile LONG guard_notifications;  // first-chance guard-page exceptions any thread raised

LONG CALLBACK count_guard(EXCEPTION_POINTERS* info)
{
    if (info->ExceptionRecord->ExceptionCode == STATUS_GUARD_PAGE_VIOLATION) InterlockedIncrement(&guard_notifications);
    return EXCEPTION_CONTINUE_SEARCH;
}

Object* status;
Object* queue;
unsigned next_index;

void fail(const char* name, const char* what)
{
    std::printf("FAIL %s: %s\n", name, what);
    ++failures;
}

// Enqueues one uncompressed memory-source request and returns its status HRESULT, or -1 on timeout.
long run_request(unsigned char* destination, unsigned capacity, unsigned size, u64 options = 0x400000,
                 const void* from = source, u64 offset = 0)
{
    unsigned char request[0x48] = {};
    *reinterpret_cast<u64*>(request + 0x00) = options;
    *reinterpret_cast<unsigned char**>(request + 0x08) = destination;
    *reinterpret_cast<unsigned*>(request + 0x18) = capacity;
    *reinterpret_cast<const void**>(request + 0x20) = from;
    *reinterpret_cast<u64*>(request + 0x28) = offset;
    *reinterpret_cast<unsigned*>(request + 0x30) = size;
    const unsigned index = next_index++;
    reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
    reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, index);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    for (int i = 0; i < 500; ++i) {
        if (reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, index))
            return reinterpret_cast<GetHResultFn>(status->vtable[4])(status, index);
        Sleep(10);
    }
    return -1;
}

bool untouched(const unsigned char* region, unsigned from, unsigned to)
{
    for (unsigned i = from; i < to; ++i)
        if (region[i] != 0xAA) return false;
    return true;
}

unsigned char* fresh_region()
{
    auto* region = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (region) std::memset(region, 0xAA, bytes);
    return region;
}

// A guard page must still be armed after the check; other protections are not checked here.
bool guard_kept(const unsigned char* at, DWORD protect)
{
    if (!(protect & PAGE_GUARD)) return true;
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(at, &info, sizeof(info)) && (info.Protect & PAGE_GUARD);
}

// Refusal: the middle page, or the last, is made unusable in the way `how` names.
void refused(const char* name, DWORD protect, bool decommit_last)
{
    unsigned char* region = fresh_region();
    if (!region) { fail(name, "VirtualAlloc"); return; }
    DWORD old = 0;
    bool ok = true;
    if (decommit_last) ok = VirtualFree(region + 2 * page, page, MEM_DECOMMIT) != 0;
    else ok = VirtualProtect(region + page, page, protect, &old) != 0;
    if (!ok) { fail(name, "could not change the page"); VirtualFree(region, 0, MEM_RELEASE); return; }
    const LONG guards_before = guard_notifications;
    const long hr = run_request(region, bytes, decommit_last ? page : bytes);
    if (hr == -1) fail(name, "no completion");
    else if (guard_notifications != guards_before) fail(name, "the check raised a guard-page exception");
    else if (hr == 0) fail(name, "request succeeded");
    // The page the case changed is not read (a guard or no-access page would trip the check itself).
    else if (!untouched(region, 0, page)) fail(name, "first page was written");
    else if (!decommit_last && !untouched(region, 2 * page, bytes)) fail(name, "last page was written");
    else if (!guard_kept(region + page, protect)) fail(name, "guard page lost its guard bit");
    else std::printf("ok: %s refused, bytes before the bad page intact\n", name);
    VirtualFree(region, 0, MEM_RELEASE);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::printf("usage: startup_dstorage_destination_check <XCRDAPI.dll>\n");
        return 2;
    }
    HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    void* factory_pointer = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_pointer) != 0 || !factory_pointer) return 1;
    auto* factory = static_cast<Object*>(factory_pointer);
    void* status_pointer = nullptr;
    void* queue_pointer = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, cases, "Status", nullptr, &status_pointer) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_pointer) != 0)
        return 1;
    status = static_cast<Object*>(status_pointer);
    queue = static_cast<Object*>(queue_pointer);
    for (unsigned i = 0; i < bytes; ++i) source[i] = static_cast<unsigned char>(i % 251);
    // Observes every first-chance exception before any handler of the stand-in: the check must raise none.
    if (!AddVectoredExceptionHandler(1, count_guard)) return 1;

    refused("read-only middle page", PAGE_READONLY, false);
    refused("guard middle page", PAGE_READWRITE | PAGE_GUARD, false);
    refused("no-access middle page", PAGE_NOACCESS, false);
    // The capacity runs one page past what is committed; the data itself (one page) would fit.
    refused("capacity past the committed pages", 0, true);

    // Control and recovery on the same queue: a fully writable destination still succeeds exactly.
    unsigned char* region = fresh_region();
    if (!region) return 1;
    const long hr = run_request(region, bytes, bytes);
    if (hr != 0) fail("writable control", "request failed");
    else if (std::memcmp(region, source, bytes) != 0) fail("writable control", "bytes differ");
    else std::printf("ok: writable control after the refusals wrote every byte\n");
    VirtualFree(region, 0, MEM_RELEASE);

    // Committed but never touched: not in the working set, so judged by the VirtualQuery walk, and writable.
    auto* untouched_region = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (!untouched_region) return 1;
    const long cold_hr = run_request(untouched_region, bytes, bytes);
    if (cold_hr != 0) fail("non-resident destination", "request failed");
    else if (std::memcmp(untouched_region, source, bytes) != 0) fail("non-resident destination", "bytes differ");
    else std::printf("ok: non-resident writable destination wrote every byte\n");
    VirtualFree(untouched_region, 0, MEM_RELEASE);

    // The check covers every page of the declared capacity; past the data it must leave each byte as it was.
    unsigned char* roomy = fresh_region();
    if (!roomy) return 1;
    const long roomy_hr = run_request(roomy, bytes, page);
    if (roomy_hr != 0) fail("capacity past the data", "request failed");
    else if (std::memcmp(roomy, source, page) != 0) fail("capacity past the data", "data bytes differ");
    else if (!untouched(roomy, page, bytes)) fail("capacity past the data", "a byte past the data changed");
    else std::printf("ok: capacity past the data wrote the data, later bytes unchanged\n");
    VirtualFree(roomy, 0, MEM_RELEASE);

    // Copy-on-write destination (PAGE_WRITECOPY) is also allowed and must succeed.
    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, bytes, nullptr);
    auto* cow = mapping ? static_cast<unsigned char*>(MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, bytes)) : nullptr;
    if (!cow) {
        fail("copy-on-write destination", "MapViewOfFile(FILE_MAP_COPY)");
    } else {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cow, &info, sizeof(info)) || (info.Protect & 0xFF) != PAGE_WRITECOPY) {
            fail("copy-on-write destination", "mapping is not PAGE_WRITECOPY");
        } else {
            const long cow_hr = run_request(cow, bytes, bytes);
            if (cow_hr != 0) fail("copy-on-write destination", "request failed");
            else if (std::memcmp(cow, source, bytes) != 0) fail("copy-on-write destination", "bytes differ");
            else std::printf("ok: copy-on-write destination wrote every byte\n");
        }
        UnmapViewOfFile(cow);
    }
    if (mapping) CloseHandle(mapping);

    // A copy-on-write page followed by a no-access page: the request is refused, and the first page stays
    // shared (the check took no private copy of it) with its bytes unchanged.
    HANDLE shared_mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 2 * page, nullptr);
    auto* filler = shared_mapping ? static_cast<unsigned char*>(MapViewOfFile(shared_mapping, FILE_MAP_WRITE, 0, 0, 2 * page)) : nullptr;
    auto* cow_pair = shared_mapping ? static_cast<unsigned char*>(MapViewOfFile(shared_mapping, FILE_MAP_COPY, 0, 0, 2 * page)) : nullptr;
    DWORD old_protect = 0;
    if (!filler || !cow_pair) {
        fail("copy-on-write before no-access", "MapViewOfFile");
    } else {
        std::memset(filler, 0xAA, 2 * page);
        volatile unsigned char sink = cow_pair[0];  // a read maps the first page in, still shared
        (void)sink;
        PSAPI_WORKING_SET_EX_INFORMATION first{};
        first.VirtualAddress = cow_pair;
        if (!VirtualProtect(cow_pair + page, page, PAGE_NOACCESS, &old_protect)) {
            fail("copy-on-write before no-access", "VirtualProtect");
        } else if (!QueryWorkingSetEx(GetCurrentProcess(), &first, sizeof(first)) || !first.VirtualAttributes.Valid ||
                   !first.VirtualAttributes.Shared) {
            fail("copy-on-write before no-access", "first page not resident and shared before the request");
        } else {
            const long pair_hr = run_request(cow_pair, 2 * page, 2 * page);
            first.VirtualAttributes.Flags = 0;
            const bool queried = QueryWorkingSetEx(GetCurrentProcess(), &first, sizeof(first)) != 0;
            const bool still_shared = queried && (first.VirtualAttributes.Valid ? first.VirtualAttributes.Shared
                                                                                : first.VirtualAttributes.Invalid.Shared);
            if (pair_hr == -1) fail("copy-on-write before no-access", "no completion");
            else if (pair_hr == 0) fail("copy-on-write before no-access", "request succeeded");
            else if (!untouched(cow_pair, 0, page)) fail("copy-on-write before no-access", "first page was written");
            else if (!still_shared) fail("copy-on-write before no-access", "first page became private");
            else std::printf("ok: copy-on-write before no-access refused, first page still shared\n");
        }
    }
    if (cow_pair) UnmapViewOfFile(cow_pair);
    if (filler) UnmapViewOfFile(filler);
    if (shared_mapping) CloseHandle(shared_mapping);

    // Distinct failure codes: a destination too small for the data, and a read past the end of a file.
    const long insufficient = static_cast<long>(HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER));
    const long end_of_file = static_cast<long>(HRESULT_FROM_WIN32(ERROR_HANDLE_EOF));
    unsigned char* small = fresh_region();
    if (!small) return 1;
    const long small_hr = run_request(small, page, bytes);
    if (small_hr != insufficient) fail("destination too small", "status is not ERROR_INSUFFICIENT_BUFFER");
    else if (!untouched(small, 0, bytes)) fail("destination too small", "destination was written");
    else std::printf("ok: destination too small reports ERROR_INSUFFICIENT_BUFFER, nothing written\n");

    wchar_t self[MAX_PATH];
    void* file_pointer = nullptr;
    if (!GetModuleFileNameW(nullptr, self, MAX_PATH) ||
        reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, self, nullptr, &file_pointer) != 0 || !file_pointer)
        return 1;
    const long eof_hr = run_request(small, bytes, 4, 0, file_pointer, 1ull << 40);
    if (eof_hr != end_of_file) fail("read past end of file", "status is not ERROR_HANDLE_EOF");
    else if (!untouched(small, 0, bytes)) fail("read past end of file", "destination was written");
    else std::printf("ok: read past the end of a file reports ERROR_HANDLE_EOF, nothing written\n");
    VirtualFree(small, 0, MEM_RELEASE);

    // Null arguments are refused with their own codes and leave the output null.
    void* out = &out;
    void* queue_out = &queue_out;
    const long no_path = reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, nullptr, nullptr, &out);
    if (no_path != E_INVALIDARG || out != nullptr) fail("OpenFile null path", "not E_INVALIDARG with a null output");
    else std::printf("ok: OpenFile with a null path returns E_INVALIDARG\n");
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, self, nullptr, nullptr) != E_POINTER)
        fail("OpenFile null output", "not E_POINTER");
    const long no_desc = reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, nullptr, nullptr, &queue_out);
    if (no_desc != E_INVALIDARG || queue_out != nullptr) fail("CreateQueue null descriptor", "not E_INVALIDARG with a null output");
    else std::printf("ok: CreateQueue with a null descriptor returns E_INVALIDARG\n");

    if (failures) return 1;
    std::printf("dstorage destination: refused without writes, queue kept working\n");
    return 0;
}
