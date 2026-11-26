// Unrelated DLL that also exports QueryApiImpl and RegQueryValueExW; must not become the cached
// gaming-runtime provider or the remembered registry function.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
LONG calls;
LONG registry_calls;
}

extern "C" __declspec(dllexport) LONG WINAPI query_calls() { return InterlockedCompareExchange(&calls, 0, 0); }

extern "C" __declspec(dllexport) HRESULT WINAPI QueryApiImpl(const GUID*, const GUID*, void** out)
{
    InterlockedIncrement(&calls);
    if (out) *out = nullptr;
    return 0xA11C0001;  // distinct from the fake xgameruntime path
}

extern "C" LONG WINAPI registry_query_calls() { return InterlockedCompareExchange(&registry_calls, 0, 0); }

// Exported as RegQueryValueExW (decoy.def): a result no registry returns.
extern "C" LONG WINAPI decoy_reg_query_value_ex_w(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD)
{
    InterlockedIncrement(&registry_calls);
    return 0x5EC1;
}
