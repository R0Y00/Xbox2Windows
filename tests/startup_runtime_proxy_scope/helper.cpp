// Package helper: imports GetProcAddress by name so the runtime proxy can rewrite its IAT after load.
// Also delay-imports xgameruntime!QueryApiImpl. The host calls call_delay_query before the proxy
// attaches, so the slot holds the real export for the later rewrite. A second copy is never called.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

extern "C" HRESULT WINAPI QueryApiImpl(const GUID*, const GUID*, void**);

extern "C" __declspec(dllexport) FARPROC WINAPI resolve_query(HMODULE module)
{
    return GetProcAddress(module, "QueryApiImpl");
}

extern "C" __declspec(dllexport) HRESULT WINAPI call_delay_query(const GUID* provider, const GUID* iid, void** out)
{
    return QueryApiImpl(provider, iid, out);
}
