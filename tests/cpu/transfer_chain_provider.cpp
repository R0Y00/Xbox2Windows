// Controlled provider for testing the memory transfer prerequisite and failure paths.
// Simulates tools-memory query availability, working-set statistics, and transfer memory calls:
// - Verifies prerequisite query blocks transfer when tools memory is unavailable (no-tools desktop policy).
// - Verifies failed working-set statistics queries leave transfer uncalled.
// - Verifies zero available budget skips transfer.
// - Records transfer arguments and returns the configured result without transferring pages.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
constexpr std::uint32_t SELECTOR_WORKING_SET = 0x2000000A;
constexpr std::uint32_t POLICY_PARTITION = 1;
constexpr std::uint64_t POLICY_ALIGNMENT_MASK = 0x1FFFFF;
constexpr std::uint64_t POLICY_MAX_BYTES = 0x1000000;

// Configurable test provider behaviors
volatile LONG cfg_tools_mem_available = 1;
volatile LONG cfg_working_set_success = 1;
volatile LONG cfg_transfer_success = 1;

volatile uint64_t cfg_budget_total = 0x2000000; // 32 MiB
volatile uint64_t cfg_budget_used = 0x1000000;  // 16 MiB (diff = 16 MiB)

// Call counters and recorded trace
volatile LONG tools_mem_calls = 0;
volatile LONG working_set_calls = 0;
volatile LONG transfer_calls = 0;
volatile uint64_t transferred_bytes = 0;
volatile uint32_t transferred_partition = 0;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        char buf[2];
        if (GetEnvironmentVariableA("XMEM_TEST_TOOLS_MEM_AVAILABLE", buf, sizeof(buf)) == 1 && buf[0] == '0') {
            cfg_tools_mem_available = 0;
        }
        if (GetEnvironmentVariableA("XMEM_TEST_WORKING_SET_SUCCESS", buf, sizeof(buf)) == 1 && buf[0] == '0') {
            cfg_working_set_success = 0;
        }
        if (GetEnvironmentVariableA("XMEM_TEST_ZERO_BUDGET", buf, sizeof(buf)) == 1 && buf[0] == '1') {
            cfg_budget_total = 0x1000000;
            cfg_budget_used = 0x1000000;
        }
        if (GetEnvironmentVariableA("XMEM_TEST_TRANSFER_SUCCESS", buf, sizeof(buf)) == 1 && buf[0] == '0') {
            cfg_transfer_success = 0;
        }
    }
    return TRUE;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemIsToolsMemAvailable(void)
{
    InterlockedIncrement(&tools_mem_calls);
    return cfg_tools_mem_available ? TRUE : FALSE;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemGetWorkingSetStatistics(
    UINT32 selector,
    void* statistics)
{
    InterlockedIncrement(&working_set_calls);
    if (!statistics || selector != SELECTOR_WORKING_SET || !cfg_working_set_success) {
        SetLastError(ERROR_NOT_SUPPORTED);
        return FALSE;
    }

    std::memset(statistics, 0, 128);
    auto* qwords = static_cast<std::uint64_t*>(statistics);
    qwords[2] = cfg_budget_total; // offset +0x10
    qwords[3] = cfg_budget_used;  // offset +0x18
    return TRUE;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemTransferMemory(
    UINT32 partition,
    UINT64 byte_count)
{
    InterlockedIncrement(&transfer_calls);
    transferred_partition = partition;
    transferred_bytes = byte_count;

    if (!cfg_transfer_success ||
        partition != POLICY_PARTITION ||
        byte_count == 0 ||
        (byte_count & POLICY_ALIGNMENT_MASK) != 0 ||
        byte_count > POLICY_MAX_BYTES)
    {
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
    return TRUE;
}

// Verification and configuration exports
extern "C" __declspec(dllexport) void WINAPI set_chain_tools_mem_available(BOOL val)
{
    InterlockedExchange(&cfg_tools_mem_available, val ? 1 : 0);
}

extern "C" __declspec(dllexport) void WINAPI set_chain_working_set_success(BOOL val)
{
    InterlockedExchange(&cfg_working_set_success, val ? 1 : 0);
}

extern "C" __declspec(dllexport) void WINAPI set_chain_working_set_budget(uint64_t total, uint64_t used)
{
    cfg_budget_total = total;
    cfg_budget_used = used;
}

extern "C" __declspec(dllexport) void WINAPI set_chain_transfer_success(BOOL val)
{
    InterlockedExchange(&cfg_transfer_success, val ? 1 : 0);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_chain_tools_mem_calls(void)
{
    return static_cast<uint32_t>(tools_mem_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_chain_working_set_calls(void)
{
    return static_cast<uint32_t>(working_set_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_chain_transfer_calls(void)
{
    return static_cast<uint32_t>(transfer_calls);
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_chain_transferred_bytes(void)
{
    return transferred_bytes;
}

extern "C" __declspec(dllexport) void WINAPI reset_chain_records(void)
{
    InterlockedExchange(&tools_mem_calls, 0);
    InterlockedExchange(&working_set_calls, 0);
    InterlockedExchange(&transfer_calls, 0);
    transferred_bytes = 0;
    transferred_partition = 0;
    cfg_tools_mem_available = 1;
    cfg_working_set_success = 1;
    cfg_transfer_success = 1;
    cfg_budget_total = 0x2000000;
    cfg_budget_used = 0x1000000;
}
