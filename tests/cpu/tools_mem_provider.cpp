// Controlled recording provider for XMemIsToolsMemAvailable.
// Exercises both BOOL return values: desktop without a tools partition (FALSE)
// and console devkit with a tools partition (TRUE).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdint>

namespace {
volatile LONG call_count = 0;
}

extern "C" __declspec(dllexport) BOOL WINAPI XMemIsToolsMemAvailable(void)
{
    InterlockedIncrement(&call_count);
#ifdef TOOLS_PARTITION_PRESENT
    // Console devkit mode: tools partition is present and available.
    return TRUE;
#else
    // Desktop policy / retail console mode: no tools partition is present.
    return FALSE;
#endif
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_tools_mem_call_count(void)
{
    return static_cast<uint32_t>(call_count);
}

extern "C" __declspec(dllexport) void WINAPI reset_tools_mem_call_count(void)
{
    InterlockedExchange(&call_count, 0);
}
