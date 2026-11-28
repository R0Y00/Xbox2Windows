// Test-only recording provider for acphal.dll address and memory mapping exports:
//   acphal.dll!ApuAlloc
//   acphal.dll!ApuFree
//   acphal.dll!ApuMapVirtualAddress
//   acphal.dll!ApuMapApuAddress
//   acphal.dll!ApuIsVirtualAddressValid
//
// Call shapes and public citations:
// - ApuAlloc and ApuFree follow public MIT call sites in microsoft/DirectXTK12
//   (SoundEffect.cpp, WaveBankReader.cpp, SoundStreamInstance.cpp).
// - ApuMapVirtualAddress translates a CPU virtual address pointer (const void*)
//   to an APU device address (UINT64 / APU_ADDRESS).
// - ApuMapApuAddress translates an APU device address (UINT64) back to a CPU
//   virtual address pointer (void*).
// - ApuIsVirtualAddressValid returns BOOL indicating whether a virtual address
//   belongs to an active, allocated APU memory region.
//
// This is not a production acphal.dll shim. The arena, APU device address space
// mapping, and refusal sentinels are proposed desktop policies for this fixture.
// APU_TEST_FAULTY builds a negative control provider that returns corrupted mappings.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>

namespace {
constexpr std::uint32_t arena_size = 0x10000; // 64 KiB test arena policy
constexpr std::uint64_t apu_device_base = 0x200000000ull; // Synthetic APU device address base
constexpr int max_blocks = 8;

alignas(4096) unsigned char arena[arena_size];

struct Block {
    std::uintptr_t address;
    std::uint64_t apu_base;
    std::uint32_t size;
    bool active;
};

SRWLOCK lock = SRWLOCK_INIT;
Block blocks[max_blocks] = {};
int block_count = 0;
std::uint32_t bump = 0;

std::uint32_t alloc_calls = 0;
std::uint32_t free_calls = 0;
std::uint32_t map_virtual_calls = 0;
std::uint32_t map_apu_calls = 0;
std::uint32_t is_valid_calls = 0;
std::uint32_t unmapped_refusals = 0;

HRESULT allocate_block(void** out, std::uint32_t size, std::uint32_t alignment)
{
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(arena);
    const std::uintptr_t start = (base + bump + alignment - 1) & ~static_cast<std::uintptr_t>(alignment - 1);
    const std::uint64_t end = static_cast<std::uint64_t>(start - base) + size;
    if (end > arena_size || block_count == max_blocks) {
        return E_OUTOFMEMORY;
    }
    const std::uint64_t apu_base = apu_device_base + (start - base);
    blocks[block_count++] = {start, apu_base, size, true};
    bump = static_cast<std::uint32_t>(end);
    *out = reinterpret_cast<void*>(start);
    return S_OK;
}

const Block* find_block_by_cpu(const void* ptr)
{
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    for (int i = 0; i < block_count; ++i) {
        if (blocks[i].active && addr >= blocks[i].address && addr < blocks[i].address + blocks[i].size) {
            return &blocks[i];
        }
    }
    return nullptr;
}

const Block* find_block_by_apu(std::uint64_t apu_addr)
{
    for (int i = 0; i < block_count; ++i) {
        if (blocks[i].active && apu_addr >= blocks[i].apu_base && apu_addr < blocks[i].apu_base + blocks[i].size) {
            return &blocks[i];
        }
    }
    return nullptr;
}
} // namespace

extern "C" __declspec(dllexport) HRESULT WINAPI ApuAlloc(
    void** virtual_address,
    void* second_argument,
    UINT32 size,
    UINT32 alignment)
{
    AcquireSRWLockExclusive(&lock);
    HRESULT result;
    if (!virtual_address || second_argument || size == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        result = E_INVALIDARG;
    } else {
        result = allocate_block(virtual_address, size, alignment);
    }
    ++alloc_calls;
    ReleaseSRWLockExclusive(&lock);
    return result;
}

extern "C" __declspec(dllexport) void WINAPI ApuFree(void* virtual_address)
{
    AcquireSRWLockExclusive(&lock);
    ++free_calls;
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(virtual_address);
    for (int i = 0; i < block_count; ++i) {
        if (blocks[i].active && blocks[i].address == addr) {
            blocks[i].active = false;
            break;
        }
    }
    ReleaseSRWLockExclusive(&lock);
}

extern "C" __declspec(dllexport) UINT64 WINAPI ApuMapVirtualAddress(const void* virtual_address)
{
    AcquireSRWLockExclusive(&lock);
    ++map_virtual_calls;
    UINT64 result = 0;
    if (const Block* b = find_block_by_cpu(virtual_address)) {
        const std::uintptr_t offset = reinterpret_cast<std::uintptr_t>(virtual_address) - b->address;
        result = b->apu_base + offset;
    } else {
        ++unmapped_refusals;
    }
    ReleaseSRWLockExclusive(&lock);
    return result;
}

extern "C" __declspec(dllexport) void* WINAPI ApuMapApuAddress(UINT64 apu_address)
{
    AcquireSRWLockExclusive(&lock);
    ++map_apu_calls;
    void* result = nullptr;
    if (const Block* b = find_block_by_apu(apu_address)) {
#ifdef APU_TEST_FAULTY
        // Faulty provider negative control: returns nullptr
        result = nullptr;
#else
        const std::uint64_t offset = apu_address - b->apu_base;
        result = reinterpret_cast<void*>(b->address + offset);
#endif
    } else {
        ++unmapped_refusals;
    }
    ReleaseSRWLockExclusive(&lock);
    return result;
}

extern "C" __declspec(dllexport) BOOL WINAPI ApuIsVirtualAddressValid(const void* virtual_address)
{
    AcquireSRWLockExclusive(&lock);
    ++is_valid_calls;
    BOOL result = FALSE;
    if (find_block_by_cpu(virtual_address) != nullptr) {
        result = TRUE;
    } else {
        ++unmapped_refusals;
    }
    ReleaseSRWLockExclusive(&lock);
    return result;
}

// Verification and query exports
extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_map_virtual_calls(void)
{
    return map_virtual_calls;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_map_apu_calls(void)
{
    return map_apu_calls;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_is_valid_calls(void)
{
    return is_valid_calls;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_unmapped_refusals(void)
{
    return unmapped_refusals;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_alloc_count(void)
{
    return alloc_calls;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_apu_free_count(void)
{
    return free_calls;
}

extern "C" __declspec(dllexport) void WINAPI reset_apu_address_records(void)
{
    AcquireSRWLockExclusive(&lock);
    for (int i = 0; i < max_blocks; ++i) {
        blocks[i] = {};
    }
    block_count = 0;
    bump = 0;
    alloc_calls = 0;
    free_calls = 0;
    map_virtual_calls = 0;
    map_apu_calls = 0;
    is_valid_calls = 0;
    unmapped_refusals = 0;
    ReleaseSRWLockExclusive(&lock);
}
