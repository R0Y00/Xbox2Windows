// src/startup_stubs/tracy_new.cpp replaces the DLL's operator new/delete in X2W_TRACY builds. This fixture compiles it
// against a recording stand-in for Tracy's memory macros and checks what a caller of the C++ heap relies on.
#include "../src/startup_stubs/tracy_new.cpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
int failures = 0;

void expect(bool condition, const char* what)
{
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool pool_is_heap(const char* pool) { return pool && std::strcmp(pool, X2W_TRACY_HEAP) == 0; }
}

int main()
{
    // An empty allocation is legal C++: it must return a usable pointer, and be reported with the requested size.
    void* empty = operator new(0);
    expect(empty != nullptr, "operator new(0) returns a block");
    expect(g_tracy_event_count == 1 && g_tracy_events[0].alloc && g_tracy_events[0].pointer == empty &&
               g_tracy_events[0].bytes == 0 && pool_is_heap(g_tracy_events[0].pool),
           "operator new(0) is reported once to the C++ heap pool with 0 bytes");

    operator delete(empty);
    expect(g_tracy_event_count == 2 && !g_tracy_events[1].alloc && g_tracy_events[1].pointer == empty &&
               pool_is_heap(g_tracy_events[1].pool),
           "operator delete reports the same pointer and pool");

    // Over-aligned blocks come from _aligned_malloc, so they must go back through _aligned_free (a plain free()
    // here corrupts the heap).
    g_tracy_event_count = 0;
    const std::align_val_t alignment{4096};
    void* aligned = operator new(100, alignment);
    expect(aligned != nullptr, "aligned operator new returns a block");
    expect(reinterpret_cast<std::uintptr_t>(aligned) % 4096 == 0, "aligned operator new honours the alignment");
    std::memset(aligned, 0xA5, 100);
    expect(g_tracy_event_count == 1 && g_tracy_events[0].bytes == 100, "aligned allocation reports its size");
    operator delete(aligned, alignment);
    expect(g_tracy_event_count == 2 && !g_tracy_events[1].alloc && g_tracy_events[1].pointer == aligned,
           "aligned delete reports the same pointer");

    void* aligned_empty = operator new(0, alignment);
    expect(aligned_empty != nullptr, "aligned operator new(0) returns a block");
    operator delete(aligned_empty, alignment);

    // Deleting a null pointer is a no-op and must not reach the profiler.
    g_tracy_event_count = 0;
    operator delete(nullptr);
    operator delete(nullptr, alignment);
    expect(g_tracy_event_count == 0, "operator delete(nullptr) reports nothing");

    if (failures) return 1;
    std::puts("tracy_new_check: ok");
    return 0;
}
