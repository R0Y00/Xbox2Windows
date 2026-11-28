// Test-only recording provider for acphal.dll!ApuAlloc and acphal.dll!ApuFree.
//
// This is not a production acphal.dll shim. The call shape comes from public MIT call
// sites in microsoft/DirectXTK12 at f3555e3014a53c81a6202b786dc4d6509b95a936, which
// assign the ApuAlloc result to an HRESULT, test it with FAILED, pass the address of a
// void* as the first argument, nullptr as the second, a UINT32 size as the third and an
// alignment constant as the fourth, and pass the returned void* to ApuFree. The public
// sources do not declare the type of the second argument, the width of the alignment
// argument or the ApuFree return type. This provider accepts only a null second
// argument, reads the alignment as 32 bits and returns nothing from ApuFree.
//
// Everything below the call shape is proposed desktop policy for the fixture: a fixed
// 64 KiB arena stands in for the heap that ApuCreateHeap would create (its signature is
// not public, so it is not exported), a request that does not fit fails with
// E_OUTOFMEMORY and leaves the output untouched, and ApuFree returns the block's space.
// APU_TEST_LENIENT builds a control provider that never refuses for size.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>

namespace {
constexpr std::uint32_t arena_size = 0x10000;
constexpr int max_blocks = 8;
constexpr int max_records = 8;

alignas(4096) unsigned char arena[arena_size];

struct Block {
    std::uintptr_t address;
    std::uint32_t size;
};

SRWLOCK lock = SRWLOCK_INIT;
Block live[max_blocks];
int live_count = 0;
std::uint32_t bump = 0;

std::uint32_t alloc_calls = 0;
std::uint32_t free_calls = 0;
std::uint32_t rejected_frees = 0;
std::uint32_t recorded_sizes[max_records];
std::uint32_t recorded_alignments[max_records];
HRESULT recorded_results[max_records];

void record(std::uint32_t size, std::uint32_t alignment, HRESULT result)
{
    if (alloc_calls < max_records) {
        recorded_sizes[alloc_calls] = size;
        recorded_alignments[alloc_calls] = alignment;
        recorded_results[alloc_calls] = result;
    }
    ++alloc_calls;
}

HRESULT allocate(void** out, std::uint32_t size, std::uint32_t alignment)
{
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(arena);
    const std::uintptr_t start = (base + bump + alignment - 1) & ~static_cast<std::uintptr_t>(alignment - 1);
    const std::uint64_t end = static_cast<std::uint64_t>(start - base) + size;
    if (end > arena_size || live_count == max_blocks) {
#ifdef APU_TEST_LENIENT
        // Control: accept any size from separate memory so a missing refusal is visible.
        void* extra = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (!extra)
            return E_OUTOFMEMORY;
        *out = extra;
        return S_OK;
#else
        return E_OUTOFMEMORY;
#endif
    }
    live[live_count++] = {start, size};
    bump = static_cast<std::uint32_t>(end);
    *out = reinterpret_cast<void*>(start);
    return S_OK;
}
}

extern "C" HRESULT WINAPI ApuAlloc(void** virtual_address, void* second_argument, UINT32 size, UINT32 alignment)
{
    AcquireSRWLockExclusive(&lock);
    HRESULT result;
    if (!virtual_address || second_argument || size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0)
        result = E_INVALIDARG;
    else
        result = allocate(virtual_address, size, alignment);
    record(size, alignment, result);
    ReleaseSRWLockExclusive(&lock);
    return result;
}

extern "C" void WINAPI ApuFree(void* virtual_address)
{
    AcquireSRWLockExclusive(&lock);
    ++free_calls;
    const auto address = reinterpret_cast<std::uintptr_t>(virtual_address);
    int found = -1;
    for (int i = 0; i < live_count; ++i) {
        if (live[i].address == address) {
            found = i;
            break;
        }
    }
    if (found < 0) {
        ++rejected_frees;
    } else {
        live[found] = live[--live_count];
        if (live_count == 0)
            bump = 0;
    }
    ReleaseSRWLockExclusive(&lock);
}

extern "C" UINT32 WINAPI get_apu_alloc_call_count(void)
{
    return alloc_calls;
}

extern "C" UINT32 WINAPI get_apu_free_call_count(void)
{
    return free_calls;
}

extern "C" UINT32 WINAPI get_apu_live_block_count(void)
{
    return static_cast<UINT32>(live_count);
}

extern "C" UINT32 WINAPI get_apu_rejected_free_count(void)
{
    return rejected_frees;
}

extern "C" UINT32 WINAPI get_apu_recorded_size(UINT32 index)
{
    return index < max_records ? recorded_sizes[index] : 0;
}

extern "C" UINT32 WINAPI get_apu_recorded_alignment(UINT32 index)
{
    return index < max_records ? recorded_alignments[index] : 0;
}

extern "C" UINT32 WINAPI get_apu_recorded_result(UINT32 index)
{
    return index < max_records ? static_cast<UINT32>(recorded_results[index]) : 0;
}

extern "C" void WINAPI reset_apu_records(void)
{
    AcquireSRWLockExclusive(&lock);
    live_count = 0;
    bump = 0;
    alloc_calls = 0;
    free_calls = 0;
    rejected_frees = 0;
    ReleaseSRWLockExclusive(&lock);
}
