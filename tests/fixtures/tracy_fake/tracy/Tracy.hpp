// Test double for Tracy's memory macros: records every pool event so tracy_new_check can assert on them.
// Used only by tracy_new_check; the real Tracy is never fetched for it.
#pragma once

#include <cstddef>

struct TracyFakeEvent {
    bool alloc;
    const void* pointer;
    std::size_t bytes;
    const char* pool;
};

constexpr std::size_t kTracyFakeMax = 16;
inline TracyFakeEvent g_tracy_events[kTracyFakeMax];
inline std::size_t g_tracy_event_count = 0;

inline void tracy_fake_record(bool alloc, const void* pointer, std::size_t bytes, const char* pool)
{
    if (g_tracy_event_count < kTracyFakeMax)
        g_tracy_events[g_tracy_event_count++] = {alloc, pointer, bytes, pool};
}

#define TracyAllocN(pointer, bytes, pool) tracy_fake_record(true, (pointer), (bytes), (pool))
#define TracyFreeN(pointer, pool) tracy_fake_record(false, (pointer), 0, (pool))
#define TracyPlot(name, value) ((void)0)
#define ZoneScopedN(name) ((void)0)
#define FrameMark ((void)0)
