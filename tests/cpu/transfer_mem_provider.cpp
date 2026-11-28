// Controlled recording provider for XMemTransferMemory.
// Validates proposed test-provider call policies and records the ordered trace of calls:
// - Partition selector 1 (test-provider policy for tools memory to title partition).
// - 2 MiB boundary alignment and non-zero byte count <= 16 MiB (test-provider policies).
// - BOOL return value (1 for accepted under test-provider policy, 0 for rejected).
// - Rejection and tracking of invalid partition, zero bytes, unaligned bytes, and oversized requests.
// Native console transfer semantics and partition effects remain open.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstddef>
#include <cstdint>

namespace {
// Test-provider policies (synthetic fixture boundaries)
constexpr std::uint32_t PROVIDER_POLICY_PARTITION = 1;
constexpr std::uint64_t PROVIDER_POLICY_ALIGNMENT_MASK = 0x1FFFFF; // 2 MiB alignment policy
constexpr std::uint64_t PROVIDER_POLICY_MAX_BYTES = 0x1000000;    // 16 MiB maximum policy
constexpr std::size_t MAX_RECORDED_CALLS = 16;

struct CallRecord {
    std::uint32_t partition;
    std::uint64_t bytes;
    std::uint32_t result;
};

volatile LONG total_calls = 0;
volatile LONG accepted_calls = 0;
volatile LONG rejected_calls = 0;

CallRecord recorded_calls[MAX_RECORDED_CALLS] = {};
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemTransferMemory(
    UINT32 partition,
    UINT64 byte_count)
{
    const BOOL accepted = (partition == PROVIDER_POLICY_PARTITION &&
                           byte_count != 0 &&
                           (byte_count & PROVIDER_POLICY_ALIGNMENT_MASK) == 0 &&
                           byte_count <= PROVIDER_POLICY_MAX_BYTES) ? TRUE : FALSE;

    const LONG idx = InterlockedIncrement(&total_calls) - 1;
    if (idx >= 0 && static_cast<std::size_t>(idx) < MAX_RECORDED_CALLS) {
        recorded_calls[idx].partition = partition;
        recorded_calls[idx].bytes = byte_count;
        recorded_calls[idx].result = static_cast<std::uint32_t>(accepted);
    }

    if (accepted) {
        InterlockedIncrement(&accepted_calls);
        SetLastError(ERROR_SUCCESS);
        return TRUE;
    } else {
        InterlockedIncrement(&rejected_calls);
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_total_call_count(void)
{
    return static_cast<uint32_t>(total_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_transfer_call_count(void)
{
    return static_cast<uint32_t>(accepted_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_argument_rejection_count(void)
{
    return static_cast<uint32_t>(rejected_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_recorded_call_partition(uint32_t index)
{
    return index < MAX_RECORDED_CALLS ? recorded_calls[index].partition : 0;
}

extern "C" __declspec(dllexport) uint64_t WINAPI get_recorded_call_bytes(uint32_t index)
{
    return index < MAX_RECORDED_CALLS ? recorded_calls[index].bytes : 0;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_recorded_call_result(uint32_t index)
{
    return index < MAX_RECORDED_CALLS ? recorded_calls[index].result : 0;
}

extern "C" __declspec(dllexport) void WINAPI reset_transfer_mem_records(void)
{
    InterlockedExchange(&total_calls, 0);
    InterlockedExchange(&accepted_calls, 0);
    InterlockedExchange(&rejected_calls, 0);
    for (std::size_t i = 0; i < MAX_RECORDED_CALLS; ++i) {
        recorded_calls[i] = {};
    }
}
