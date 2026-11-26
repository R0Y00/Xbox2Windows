// Queue slot 31 (X2WMapPageBatch) into the middle of an XMem reservation needs a placeholder split. Where the region
// query does not report placeholders (Wine), xmem probes whether a placeholder release refuses a plain reservation
// and then offers reserved regions to the split (xmem placeholder_fallback).
//   xmem_placeholder_split_check <xmem.dll>          split succeeds; a committed target still fails and stays intact
//   xmem_placeholder_split_check <xmem.dll> control  the same checks, run with X2W_XMEM_TEST_NO_PLACEHOLDER_FALLBACK=1
//                                                    by CTest, which expects the split step to fail; exits 77 where
//                                                    the OS reports the bit, because there the fallback is never used
#include <windows.h>
#include "xmem_page_batch.h"
#include <cstdio>
#include <cstring>
#include <cwchar>

namespace {
constexpr SIZE_T page_bytes = 65536;
constexpr UINT pool_pages = 4;
using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using AllocPages = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
using Scatter = BOOL (WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);

bool require(bool ok, const char* step)
{
    if (!ok) std::printf("FAIL: %s (last error %lu)\n", step, GetLastError());
    return ok;
}

bool os_reports_placeholder_bit()
{
    void* probe = VirtualAlloc2(GetCurrentProcess(), nullptr, 2 * page_bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                PAGE_NOACCESS, nullptr, 0);
    if (!probe) return false;
    WIN32_MEMORY_REGION_INFORMATION info{};
    const bool bit = QueryVirtualMemoryInformation(GetCurrentProcess(), probe, MemoryRegionInfo, &info, sizeof(info),
                                                   nullptr) && (info.Flags & 0x100);
    VirtualFree(probe, 0, MEM_RELEASE);
    return bit;
}

MEMORY_BASIC_INFORMATION query(const BYTE* pointer)
{
    MEMORY_BASIC_INFORMATION info{};
    VirtualQuery(pointer, &info, sizeof(info));
    return info;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    const bool control = argc == 3 && std::wcscmp(argv[2], L"control") == 0;
    if (argc != 2 && !control) return 2;
    const bool bit = os_reports_placeholder_bit();
    std::printf("placeholder bit reported=%d\n", bit);
    if (control && bit) {
        std::puts("SKIP: the OS reports the placeholder bit, so the fallback is never used");
        return 77;
    }
    const HMODULE xmem = LoadLibraryW(argv[1]);
    if (!require(xmem != nullptr, "load xmem.dll")) return 1;
    const auto reserve = reinterpret_cast<Reserve>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto alloc = reinterpret_cast<AllocPages>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto scatter = reinterpret_cast<Scatter>(GetProcAddress(xmem, "XMemMapPhysicalPagesScatter"));
    const auto batch_map = reinterpret_cast<decltype(&X2WMapPageBatch)>(GetProcAddress(xmem, "X2WMapPageBatch"));
    if (!require(reserve && alloc && scatter && batch_map, "find the xmem exports")) return 1;

    auto* const pool = static_cast<BYTE*>(reserve(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!require(pool != nullptr, "reserve XMem addresses")) return 1;
    ULONG_PTR grants[pool_pages]{}, count = pool_pages;
    void* addresses[pool_pages]{};
    for (UINT i = 0; i < pool_pages; ++i) addresses[i] = pool + i * page_bytes;
    if (!require(alloc(0x280, &count, grants) && count == pool_pages, "allocate the pool") ||
        !require(scatter(addresses, pool_pages, grants), "map the pool"))
        return 1;
    for (UINT i = 0; i < pool_pages; ++i) std::memset(addresses[i], 0x51 + i, page_bytes);

    // Negative control: a committed page inside the reservation is not a placeholder. The batch must fail and leave
    // the page committed with its bytes.
    BYTE* const committed = pool + 64 * page_bytes;
    if (!require(VirtualFree(committed, page_bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER), "split out a page") ||
        !require(VirtualAlloc2(GetCurrentProcess(), committed, page_bytes, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                               PAGE_READWRITE, nullptr, 0) == committed, "commit that page"))
        return 1;
    std::memset(committed, 0x77, page_bytes);
    const X2WPageRange one{0, 0, 1};
    const X2WPageBatch onto_committed{reinterpret_cast<UINT64>(committed), 1, &one};
    const HRESULT refused = batch_map(1, &onto_committed, reinterpret_cast<UINT64>(pool), pool_pages, 0, nullptr);
    std::printf("batch onto a committed page -> 0x%08lx\n", static_cast<unsigned long>(refused));
    // The code is the OS's map failure: Windows 487, Wine 87 (measured 2026-11-26).
    if (!require(FAILED(refused), "a committed target is refused") ||
        !require(query(committed).State == MEM_COMMIT && committed[0] == 0x77 && committed[page_bytes - 1] == 0x77,
                 "the committed target keeps its bytes"))
        return 1;

    // The split: pool pages 1 and 2 into the middle of an untouched part of the reservation.
    BYTE* const window = pool + 128 * page_bytes;
    const X2WPageRange two{0, 1, 2};
    const X2WPageBatch split{reinterpret_cast<UINT64>(window + page_bytes), 1, &two};
    const HRESULT hr = batch_map(1, &split, reinterpret_cast<UINT64>(pool), pool_pages, 0, nullptr);
    std::printf("batch into the middle of the reservation -> 0x%08lx\n", static_cast<unsigned long>(hr));
    if (!require(SUCCEEDED(hr), "map two pool pages into the middle of the reservation") ||
        !require(window[page_bytes] == 0x52 && window[2 * page_bytes] == 0x53 && window[3 * page_bytes - 1] == 0x53,
                 "the destination reads the pool pages") ||
        !require(query(window).State == MEM_RESERVE && query(window + 3 * page_bytes).State == MEM_RESERVE,
                 "the neighbours stay reserved"))
        return 1;
    window[page_bytes] = 0x60;
    if (!require(pool[page_bytes] == 0x60, "a write through the destination reaches the pool page")) return 1;
    std::puts("slot31 split inside an XMem reservation matched; a committed target was refused and kept");
    return 0;
}
