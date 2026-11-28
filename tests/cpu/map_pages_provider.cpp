// Controlled recording provider for XMemMapPhysicalPages and XMemMapPhysicalPagesScatter.
// Validates proposed call contracts and records trace of calls:
// - 64 KiB address alignment, non-zero page count.
// - Mapping vs unmapping (NULL page array).
// - Verification of fixture input arguments (expected count, page IDs, ordered scatter addresses).
// - Rejection and tracking of changed-argument controls.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>

namespace {
constexpr std::uintptr_t page_alignment_mask = 0xFFFF; // 64 KiB alignment requirement

// Specific fixture inputs chosen for the controlled test
constexpr std::uintptr_t FIXTURE_ALIGNED_ADDR = 0x10000000;
constexpr std::uintptr_t FIXTURE_SCATTER_ADDR0 = 0x10000000;
constexpr std::uintptr_t FIXTURE_SCATTER_ADDR1 = 0x10010000;
constexpr ULONG_PTR FIXTURE_PAGE_0 = 1;
constexpr ULONG_PTR FIXTURE_PAGE_1 = 2;
constexpr ULONG_PTR FIXTURE_PAGE_COUNT = 2;

volatile LONG map_calls = 0;
volatile LONG unmap_calls = 0;
volatile LONG scatter_map_calls = 0;
volatile LONG scatter_unmap_calls = 0;
volatile LONG argument_mismatches = 0;

// Recorded trace from valid calls
volatile std::uintptr_t recorded_map_address = 0;
volatile ULONG_PTR recorded_map_count = 0;
volatile ULONG_PTR recorded_map_pages[2] = {0, 0};

volatile std::uintptr_t recorded_scatter_addresses[2] = {0, 0};
volatile ULONG_PTR recorded_scatter_count = 0;
volatile ULONG_PTR recorded_scatter_pages[2] = {0, 0};
}

extern "C" __declspec(dllexport) LPVOID WINAPI XMemMapPhysicalPages(
    LPVOID lpAddress,
    ULONG_PTR cPages,
    const ULONG_PTR* rgPages)
{
    const auto address = reinterpret_cast<std::uintptr_t>(lpAddress);
    if (!lpAddress || (address & page_alignment_mask) != 0 || cPages == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    if (!rgPages) {
        // Unmap request: unbinds physical pages under the proposed policy.
        InterlockedIncrement(&unmap_calls);
        return lpAddress;
    }

    // Changed-argument control: distinguish expected fixture page IDs and count
    if (cPages != FIXTURE_PAGE_COUNT || rgPages[0] != FIXTURE_PAGE_0 || rgPages[1] != FIXTURE_PAGE_1) {
        InterlockedIncrement(&argument_mismatches);
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }

    // Valid map request: record trace and bind
    recorded_map_address = address;
    recorded_map_count = cPages;
    recorded_map_pages[0] = rgPages[0];
    recorded_map_pages[1] = rgPages[1];

    InterlockedIncrement(&map_calls);
    return lpAddress;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemMapPhysicalPagesScatter(
    LPVOID* rgVirtualAddresses,
    ULONG_PTR cPages,
    const ULONG_PTR* rgPages)
{
    if (!rgVirtualAddresses || cPages == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }

    for (ULONG_PTR i = 0; i < cPages; ++i) {
        const auto addr = reinterpret_cast<std::uintptr_t>(rgVirtualAddresses[i]);
        if (!rgVirtualAddresses[i] || (addr & page_alignment_mask) != 0) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return FALSE;
        }
    }

    if (!rgPages) {
        // Unmap scatter request: unbinds physical pages for all listed virtual addresses.
        InterlockedIncrement(&scatter_unmap_calls);
        return TRUE;
    }

    // Changed-argument control: distinguish expected ordered scatter destinations and page IDs
    if (cPages != FIXTURE_PAGE_COUNT ||
        reinterpret_cast<std::uintptr_t>(rgVirtualAddresses[0]) != FIXTURE_SCATTER_ADDR0 ||
        reinterpret_cast<std::uintptr_t>(rgVirtualAddresses[1]) != FIXTURE_SCATTER_ADDR1 ||
        rgPages[0] != FIXTURE_PAGE_0 ||
        rgPages[1] != FIXTURE_PAGE_1)
    {
        InterlockedIncrement(&argument_mismatches);
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }

    // Valid scatter map request: record trace and bind
    recorded_scatter_addresses[0] = reinterpret_cast<std::uintptr_t>(rgVirtualAddresses[0]);
    recorded_scatter_addresses[1] = reinterpret_cast<std::uintptr_t>(rgVirtualAddresses[1]);
    recorded_scatter_count = cPages;
    recorded_scatter_pages[0] = rgPages[0];
    recorded_scatter_pages[1] = rgPages[1];

    InterlockedIncrement(&scatter_map_calls);
    return TRUE;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_map_pages_call_count(void)
{
    return static_cast<uint32_t>(map_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_map_pages_unmap_count(void)
{
    return static_cast<uint32_t>(unmap_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_scatter_map_call_count(void)
{
    return static_cast<uint32_t>(scatter_map_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_scatter_unmap_count(void)
{
    return static_cast<uint32_t>(scatter_unmap_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_argument_mismatch_count(void)
{
    return static_cast<uint32_t>(argument_mismatches);
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_map_address(void)
{
    return static_cast<uint64_t>(recorded_map_address);
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_map_count(void)
{
    return static_cast<uint64_t>(recorded_map_count);
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_map_page(uint32_t index)
{
    return index < 2 ? static_cast<uint64_t>(recorded_map_pages[index]) : 0;
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_scatter_address(uint32_t index)
{
    return index < 2 ? static_cast<uint64_t>(recorded_scatter_addresses[index]) : 0;
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_scatter_count(void)
{
    return static_cast<uint64_t>(recorded_scatter_count);
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_scatter_page(uint32_t index)
{
    return index < 2 ? static_cast<uint64_t>(recorded_scatter_pages[index]) : 0;
}

extern "C" __declspec(dllexport) void WINAPI reset_map_pages_records(void)
{
    InterlockedExchange(&map_calls, 0);
    InterlockedExchange(&unmap_calls, 0);
    InterlockedExchange(&scatter_map_calls, 0);
    InterlockedExchange(&scatter_unmap_calls, 0);
    InterlockedExchange(&argument_mismatches, 0);
    recorded_map_address = 0;
    recorded_map_count = 0;
    recorded_map_pages[0] = 0;
    recorded_map_pages[1] = 0;
    recorded_scatter_addresses[0] = 0;
    recorded_scatter_addresses[1] = 0;
    recorded_scatter_count = 0;
    recorded_scatter_pages[0] = 0;
    recorded_scatter_pages[1] = 0;
}
