// Test-only recording provider for d3d12_xs.dll frame-event exports (#874):
//   d3d12_xs.dll!SetFrameIntervalX
//   d3d12_xs.dll!ScheduleFrameEventX
//   d3d12_xs.dll!WaitFrameEventX
//
// Established by public Microsoft sample (InGameStore/DeviceResources.cpp):
// - Registers ORIGIN scheduling once, then waits each frame.
// - Uses INFINITE wait and a NULL event handle.
// - Follow-up: the returned token is subsequently passed into PresentX (not called here).
//
// Labeled as fixture behavior / policy (requiring separate evidence):
// - PE imports: exposing these APIs as d3d12_xs.dll exports is a fixture choice (sample calls device methods).
// - Null output: passing NULL to SetFrameIntervalX is a fixture choice (sample passes an actual IDXGIOutput*).
// - Numeric values: intervals, timeouts, and seeded token representation (monotonic integer increment).
// - Controlled tick driver: after one registration, a worker delivers each tick only after the
//   caller enters the wait and atomically releases the shared lock in SleepConditionVariableSRW.
// - FAULT_CLOBBER / FAULT_EARLY_RELEASE: two distinct fault controls for refusal sentinel overwrite
//   and premature wait release.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdint>

namespace {
std::atomic<std::uint32_t> g_mode{0};
std::atomic<std::uint64_t> g_token_counter{0};
std::atomic<std::uint64_t> g_origin_waits{0};
std::atomic<std::uint64_t> g_blocked_waits{0};
std::atomic<std::uint64_t> g_timeouts{0};
std::atomic<std::uint64_t> g_rejections{0};
std::atomic<std::uint64_t> g_schedule_calls{0};

HANDLE g_worker_thread = nullptr;
SRWLOCK g_lock = SRWLOCK_INIT;
CONDITION_VARIABLE g_frame_changed = CONDITION_VARIABLE_INIT;
bool g_wait_pending = false;
bool g_tick_ready = false;
bool g_stop = false;

DWORD WINAPI tick_injector(LPVOID)
{
    AcquireSRWLockExclusive(&g_lock);
    for (unsigned frame = 0; frame < 2; ++frame) {
        while (!g_wait_pending && !g_stop) {
            if (!SleepConditionVariableSRW(&g_frame_changed, &g_lock, 10000, 0)) {
                ReleaseSRWLockExclusive(&g_lock);
                return 1;
            }
        }
        if (g_stop) break;
        g_wait_pending = false;
        g_tick_ready = true;
        g_blocked_waits.fetch_add(1);
        WakeAllConditionVariable(&g_frame_changed);
    }
    ReleaseSRWLockExclusive(&g_lock);
    return 0;
}
} // namespace

extern "C" HRESULT WINAPI SetFrameIntervalX(
    void* output,
    UINT interval,
    UINT period,
    UINT flags)
{
    (void)interval;
    (void)period;
    (void)flags;
    g_mode.store(output != nullptr ? 1 : 0);
    return S_OK;
}

extern "C" HRESULT WINAPI ScheduleFrameEventX(
    UINT event_type,
    UINT offset_us,
    HANDLE event_handle,
    UINT flags)
{
    (void)offset_us;
    (void)event_handle;
    (void)flags;
    // Only D3D12XBOX_FRAME_EVENT_ORIGIN (0) has an established contract.
    if (event_type != 0) {
        g_rejections.fetch_add(1);
        return static_cast<HRESULT>(0x80070057); // E_INVALIDARG (fixture policy)
    }

    g_schedule_calls.fetch_add(1);

    // Register ORIGIN once: spawn the controlled injector thread delivering two ticks
    AcquireSRWLockExclusive(&g_lock);
    if (!g_worker_thread) {
        g_worker_thread = CreateThread(nullptr, 0, tick_injector, nullptr, 0, nullptr);
    }
    const bool started = g_worker_thread != nullptr;
    ReleaseSRWLockExclusive(&g_lock);

    return started ? S_OK : E_FAIL;
}

extern "C" HRESULT WINAPI WaitFrameEventX(
    UINT event_type,
    UINT timeout_ms,
    HANDLE event_handle,
    UINT flags,
    UINT64* out_token)
{
    (void)event_handle;
    (void)flags;
    if (!out_token) {
        return static_cast<HRESULT>(0x80004003); // E_POINTER
    }

    // Only D3D12XBOX_FRAME_EVENT_ORIGIN (0) has an established contract.
    if (event_type != 0) {
        g_rejections.fetch_add(1);
#ifdef FAULT_CLOBBER
        // Mutation 1: overwrite caller's sentinel on refusal
        *out_token = 0xBADF00DDEADBEEFull;
#endif
        return static_cast<HRESULT>(0x80070057); // E_INVALIDARG (fixture policy)
    }

#ifdef FAULT_EARLY_RELEASE
    // Mutation 2: release wait early without blocking on tick
    *out_token = g_token_counter.fetch_add(1) + 1;
    return S_OK;
#else
    AcquireSRWLockExclusive(&g_lock);
    if (g_worker_thread) {
        g_wait_pending = true;
        WakeAllConditionVariable(&g_frame_changed);
    }

    const ULONGLONG deadline = GetTickCount64() + timeout_ms;
    HRESULT result = S_OK;
    while (!g_tick_ready) {
        const ULONGLONG now = GetTickCount64();
        const DWORD remaining = now < deadline ? static_cast<DWORD>(deadline - now) : 0;
        if (!SleepConditionVariableSRW(&g_frame_changed, &g_lock, remaining, 0) && !g_tick_ready) {
            if (GetLastError() == ERROR_TIMEOUT) {
                g_timeouts.fetch_add(1);
                result = static_cast<HRESULT>(WAIT_TIMEOUT); // Fixture policy; preserve the output.
            } else {
                result = E_FAIL;
            }
            break;
        }
    }

    if (result == S_OK) {
        g_tick_ready = false;
        g_origin_waits.fetch_add(1);
        *out_token = g_token_counter.fetch_add(1) + 1;
    }
    HANDLE worker = nullptr;
    if (g_worker_thread && (result != S_OK || g_origin_waits.load() == 2)) {
        g_stop = true;
        g_wait_pending = false;
        WakeAllConditionVariable(&g_frame_changed);
        worker = g_worker_thread;
        g_worker_thread = nullptr;
    }
    ReleaseSRWLockExclusive(&g_lock);

    if (worker) {
        const DWORD joined = WaitForSingleObject(worker, 10000);
        CloseHandle(worker);
        if (joined != WAIT_OBJECT_0) return E_FAIL;
    }
    return result;
#endif
}

// ---- Instrumentation exports for test verification ----
extern "C" std::uint64_t WINAPI get_frame_events_origin_waits()
{
    return g_origin_waits.load();
}

extern "C" std::uint64_t WINAPI get_frame_events_last_token()
{
    return g_token_counter.load();
}

extern "C" std::uint64_t WINAPI get_frame_events_blocked_waits()
{
    return g_blocked_waits.load();
}

extern "C" std::uint64_t WINAPI get_frame_events_timeouts()
{
    return g_timeouts.load();
}

extern "C" std::uint64_t WINAPI get_frame_events_rejections()
{
    return g_rejections.load();
}

extern "C" std::uint64_t WINAPI get_frame_events_schedule_calls()
{
    return g_schedule_calls.load();
}

extern "C" std::uint32_t WINAPI get_frame_events_mode()
{
    return g_mode.load();
}
