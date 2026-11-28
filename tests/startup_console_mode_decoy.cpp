// Source-built package-style DLL for startup_console_mode: exports RegOpenKeyExW and
// RegQueryValueExW under the registry names (startup_console_mode_decoy.def) with a result no
// registry returns. The console-mode hooks must leave it alone: neither route may hook it or keep
// its address as the real registry function.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
LONG calls;
constexpr LONG decoy_status = 0x5EC0;
}

extern "C" LONG WINAPI decoy_calls() { return InterlockedCompareExchange(&calls, 0, 0); }

extern "C" LONG WINAPI decoy_open_key_ex_w(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY out)
{
    InterlockedIncrement(&calls);
    if (out) *out = reinterpret_cast<HKEY>(static_cast<ULONG_PTR>(0xDEC0));
    return decoy_status;
}

extern "C" LONG WINAPI decoy_query_value_ex_w(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD)
{
    InterlockedIncrement(&calls);
    return decoy_status;
}
