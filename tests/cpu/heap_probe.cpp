// Heap-block ownership probe for tests/cpu/xmem_default_pair.asm. A freed heap block usually stays
// inside committed heap memory, so VirtualQuery cannot show its release. heap_block_busy walks every
// process heap with HeapWalk and reports whether an allocated (busy) entry contains the address:
//   1  a busy entry contains it
//   0  no busy entry contains it
//   2  a heap could not be listed or walked
// The walk allocates nothing, and each heap is locked while it is walked.
#include <windows.h>

#include <cstdint>

extern "C" __declspec(dllexport) std::uint32_t heap_block_busy(const void* address)
{
    HANDLE heaps[128];
    const DWORD count = GetProcessHeaps(ARRAYSIZE(heaps), heaps);
    if (count == 0 || count > ARRAYSIZE(heaps)) return 2;
    const auto target = reinterpret_cast<std::uintptr_t>(address);
    std::uint32_t result = 0;
    for (DWORD i = 0; i < count && result == 0; ++i) {
        if (!HeapLock(heaps[i])) return 2;
        PROCESS_HEAP_ENTRY entry{};
        while (HeapWalk(heaps[i], &entry)) {
            if (!(entry.wFlags & PROCESS_HEAP_ENTRY_BUSY)) continue;
            const auto start = reinterpret_cast<std::uintptr_t>(entry.lpData);
            if (target >= start && target - start < entry.cbData) {
                result = 1;
                break;
            }
        }
        const DWORD error = GetLastError();
        HeapUnlock(heaps[i]);
        if (result == 0 && error != ERROR_NO_MORE_ITEMS) return 2;
    }
    return result;
}
