// Windows-only: EXE + package helpers through the real loader.
//
// Loads a fake xgameruntime.dll and an unrelated decoy that also exports QueryApiImpl, then one helper,
// then a carrier DLL that hosts runtime_proxy (X2W_RUNTIME_PROXY=1). That helper calls its
// delay-imported QueryApiImpl before the carrier attaches, so the bind goes through the unhooked
// GetProcAddress and the slot holds the real export. After attach that slot must become the proxy:
// the hooked lookup is not what writes it. A second helper loads after the carrier and is never
// called. Its GetProcAddress hook needs the load-driven rescan, and its delay thunk must stay inside
// its own image. Rewrites run after linking (outside LdrDllNotification). Looking up QueryApiImpl on
// xgameruntime must return the proxy;
// the same name on the decoy must still be the decoy's export. A QueryApiImpl pointer cached before
// the proxy attaches stays the real export (pre-attach gap). RegQueryValueExW is hooked on
// advapi32.dll only: the decoy's same-named export stays its own, with X2W_DEVICE_TYPE unset and set,
// and never becomes the registry the hook calls. No game files.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include "startup_stubs/runtime_gpa_patch.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures;

void check(const char* label, bool ok)
{
    std::printf("%s: %s\n", label, ok ? "ok" : "MISMATCH");
    if (!ok) ++failures;
}

using ResolveFn = FARPROC(WINAPI*)(HMODULE);
using QueryFn = HRESULT(WINAPI*)(const GUID*, const GUID*, void**);
using RegQueryFn = LONG(WINAPI*)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
using SeenFn = LONG(WINAPI*)();

std::wstring dir_of_self()
{
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring full(path);
    const auto slash = full.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : full.substr(0, slash);
}

HMODULE load_beside(const std::wstring& dir, const wchar_t* name)
{
    const std::wstring path = dir + L"\\" + name;
    return LoadLibraryW(path.c_str());
}

bool wait_for_helper_hook(ResolveFn resolve, HMODULE xgr, FARPROC expect_proxy, int ms)
{
    const DWORD start = GetTickCount();
    for (;;) {
        const FARPROC got = resolve(xgr);
        if (got == expect_proxy) return true;
        if (GetTickCount() - start > static_cast<DWORD>(ms)) return false;
        Sleep(10);
    }
}

ULONGLONG* delay_query_slot(HMODULE module)
{
    if (!module) return nullptr;
    auto* base = reinterpret_cast<unsigned char*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return nullptr;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return nullptr;
    const auto image_size = static_cast<std::size_t>(nt->OptionalHeader.SizeOfImage);
    ULONGLONG* found = nullptr;
    x2w_runtime_patch::visit_imports(base, image_size, [&](const x2w_runtime_patch::Hit& hit) {
        if (!hit.delay_import || hit.kind != x2w_runtime_patch::Kind::query_api_impl) return;
        found = reinterpret_cast<ULONGLONG*>(base + hit.first_thunk_rva);
    });
    return found;
}

bool wait_for_slot(ULONGLONG* slot, ULONGLONG expect, int ms)
{
    if (!slot) return false;
    const DWORD start = GetTickCount();
    for (;;) {
        if (*slot == expect) return true;
        if (GetTickCount() - start > static_cast<DWORD>(ms)) return false;
        Sleep(10);
    }
}

bool slot_in_module(HMODULE module, ULONGLONG value)
{
    if (!module) return false;
    auto* base = reinterpret_cast<const unsigned char*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return false;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return false;
    return x2w_runtime_patch::address_in_image(value, reinterpret_cast<ULONGLONG>(module), nt->OptionalHeader.SizeOfImage);
}
}

int main()
{
    SetEnvironmentVariableW(L"X2W_RUNTIME_PROXY", L"1");
    const std::wstring dir = dir_of_self();
    SetDllDirectoryW(dir.c_str());

    const HMODULE decoy = load_beside(dir, L"startup_runtime_proxy_decoy.dll");
    check("load decoy", decoy != nullptr);
    const FARPROC decoy_query = decoy ? GetProcAddress(decoy, "QueryApiImpl") : nullptr;
    check("decoy exports QueryApiImpl", decoy_query != nullptr);

    const HMODULE xgr = load_beside(dir, L"xgameruntime.dll");
    check("load fake xgameruntime", xgr != nullptr);
    const FARPROC xgr_query = xgr ? GetProcAddress(xgr, "QueryApiImpl") : nullptr;
    check("xgameruntime exports QueryApiImpl", xgr_query != nullptr);
    check("decoy and xgameruntime exports differ", decoy_query && xgr_query && decoy_query != xgr_query);
    const FARPROC decoy_reg = decoy ? GetProcAddress(decoy, "RegQueryValueExW") : nullptr;
    check("decoy exports RegQueryValueExW", decoy_reg != nullptr);
    const HMODULE advapi = LoadLibraryW(L"advapi32.dll");
    const FARPROC advapi_reg = advapi ? GetProcAddress(advapi, "RegQueryValueExW") : nullptr;
    check("advapi32.dll exports RegQueryValueExW", advapi_reg != nullptr && advapi_reg != decoy_reg);

    // Pre-attach cache: resolve before the proxy carrier loads. This pointer must stay the real
    // export after attach (IAT/GPA hooks do not rewrite already-held function pointers).
    const FARPROC cached_before_proxy = xgr_query;

    // Bind the delay slot before the proxy exists, through the helper's own GetProcAddress.
    const HMODULE helper = load_beside(dir, L"startup_runtime_proxy_helper.dll");
    check("load package helper before the proxy", helper != nullptr);
    const auto call_delay = helper ? reinterpret_cast<QueryFn>(GetProcAddress(helper, "call_delay_query")) : nullptr;
    check("helper exports call_delay_query", call_delay != nullptr);
    if (call_delay) call_delay(nullptr, nullptr, nullptr);
    ULONGLONG* delay_slot = delay_query_slot(helper);
    check("helper has delay QueryApiImpl slot", delay_slot != nullptr);
    check("delay slot holds the real export before attach",
          delay_slot && xgr_query && *delay_slot == reinterpret_cast<ULONGLONG>(xgr_query));
    if (failures) return 1;

    const HMODULE carrier = load_beside(dir, L"startup_runtime_proxy_carrier.dll");
    check("load proxy carrier", carrier != nullptr);
    if (!carrier) return 1;
    check("release the caller's carrier reference", FreeLibrary(carrier) != FALSE);
    check("enabled proxy stays live through process exit",
          GetModuleHandleW(L"startup_runtime_proxy_carrier.dll") == carrier);
    if (failures) return 1;

    // Host GetProcAddress is rewritten on attach; xgameruntime lookups become the proxy.
    const FARPROC host_xgr = xgr ? GetProcAddress(xgr, "QueryApiImpl") : nullptr;
    check("host GetProcAddress(xgameruntime) is proxied", host_xgr && host_xgr != xgr_query);
    check("pre-attach cached QueryApiImpl stays real", cached_before_proxy == xgr_query);
    check("release the caller's runtime reference", xgr && FreeLibrary(xgr) != FALSE);
    check("the cached provider stays live", GetModuleHandleW(L"xgameruntime.dll") == xgr);
    if (failures) return 1;

    const FARPROC host_decoy = decoy ? GetProcAddress(decoy, "QueryApiImpl") : nullptr;
    check("host GetProcAddress(decoy) stays original", host_decoy == decoy_query);

    const auto resolve = helper ? reinterpret_cast<ResolveFn>(GetProcAddress(helper, "resolve_query")) : nullptr;
    check("helper exports resolve_query", resolve != nullptr);

    check("helper sees proxied xgameruntime QueryApiImpl",
          resolve && wait_for_helper_hook(resolve, xgr, host_xgr, 2000));
    const FARPROC helper_decoy = resolve ? resolve(decoy) : nullptr;
    check("helper GetProcAddress(decoy) stays original", helper_decoy == decoy_query);

    // The slot already held the real export. Only the delay-slot rewrite turns it into the proxy.
    check("bound delay slot rewritten to proxy",
          delay_slot && host_xgr && wait_for_slot(delay_slot, reinterpret_cast<ULONGLONG>(host_xgr), 2000));

    // After the carrier's first scan. The load notification has to request another one.
    const HMODULE unbound = load_beside(dir, L"startup_runtime_proxy_helper_unbound.dll");
    check("late-load package helper", unbound != nullptr);
    const auto unbound_resolve = unbound ? reinterpret_cast<ResolveFn>(GetProcAddress(unbound, "resolve_query")) : nullptr;
    check("late-load helper exports resolve_query", unbound_resolve != nullptr);
    ULONGLONG* unbound_slot = delay_query_slot(unbound);
    check("unbound helper has delay QueryApiImpl slot", unbound_slot != nullptr);
    check("unbound delay thunk starts inside its image", unbound_slot && slot_in_module(unbound, *unbound_slot));
    check("late-load helper sees proxied xgameruntime QueryApiImpl",
          unbound_resolve && host_xgr && wait_for_helper_hook(unbound_resolve, xgr, host_xgr, 2000));
    check("unbound delay thunk stays inside its image", unbound_slot && slot_in_module(unbound, *unbound_slot));

    // Intended call: invoke the proxied entry; the fake xgameruntime must run.
    const auto xgr_seen = xgr ? reinterpret_cast<SeenFn>(GetProcAddress(xgr, "query_calls")) : nullptr;
    const auto decoy_seen = decoy ? reinterpret_cast<SeenFn>(GetProcAddress(decoy, "query_calls")) : nullptr;
    const LONG xgr_before = xgr_seen ? xgr_seen() : -1;
    const LONG decoy_before = decoy_seen ? decoy_seen() : -1;
    if (host_xgr) reinterpret_cast<QueryFn>(host_xgr)(nullptr, nullptr, nullptr);
    if (helper_decoy) reinterpret_cast<QueryFn>(helper_decoy)(nullptr, nullptr, nullptr);
    check("proxied call reached xgameruntime", xgr_seen && xgr_seen() == xgr_before + 1);
    check("decoy call reached decoy only", decoy_seen && decoy_seen() == decoy_before + 1);

    // Pre-attach cache still reaches the real export (not the proxy entry).
    if (cached_before_proxy) reinterpret_cast<QueryFn>(cached_before_proxy)(nullptr, nullptr, nullptr);
    check("pre-attach cache still calls real export", xgr_seen && xgr_seen() == xgr_before + 2);

    // Registry names, with the switch unset and set. The decoy is looked up first each time, so a
    // match by name alone would keep the decoy as the registry the hook calls.
    const auto decoy_reg_seen = decoy ? reinterpret_cast<SeenFn>(GetProcAddress(decoy, "registry_query_calls")) : nullptr;
    DWORD type = 0, bytes = 0;
    const LONG expect_real = advapi_reg ? reinterpret_cast<RegQueryFn>(advapi_reg)(nullptr, L"ConsoleMode", nullptr, &type, nullptr, &bytes) : -1;
    check("advapi32.dll's own answer differs from the decoy's", expect_real != 0x5EC1);
    for (const wchar_t* device : {static_cast<const wchar_t*>(nullptr), L"7"}) {
        SetEnvironmentVariableW(L"X2W_DEVICE_TYPE", device);
        const std::string pass = device ? "X2W_DEVICE_TYPE=7: " : "switch unset: ";
        const FARPROC host_decoy_reg = decoy ? GetProcAddress(decoy, "RegQueryValueExW") : nullptr;
        check((pass + "host GetProcAddress(decoy, RegQueryValueExW) stays original").c_str(), host_decoy_reg == decoy_reg);
        const LONG decoy_reg_before = decoy_reg_seen ? decoy_reg_seen() : -1;
        const LONG decoy_answer =
            host_decoy_reg ? reinterpret_cast<RegQueryFn>(host_decoy_reg)(nullptr, L"ConsoleMode", nullptr, &type, nullptr, &bytes) : -1;
        check((pass + "decoy RegQueryValueExW answers its own result").c_str(),
              decoy_answer == 0x5EC1 && decoy_reg_seen && decoy_reg_seen() == decoy_reg_before + 1);
        const FARPROC host_advapi_reg = advapi ? GetProcAddress(advapi, "RegQueryValueExW") : nullptr;
        check((pass + "host GetProcAddress(advapi32, RegQueryValueExW) is hooked").c_str(),
              host_advapi_reg && host_advapi_reg != advapi_reg && host_advapi_reg != decoy_reg);
        const LONG hooked_answer =
            host_advapi_reg ? reinterpret_cast<RegQueryFn>(host_advapi_reg)(nullptr, L"ConsoleMode", nullptr, &type, nullptr, &bytes) : -1;
        check((pass + "hooked RegQueryValueExW reaches advapi32.dll, not the decoy").c_str(),
              hooked_answer == expect_real && decoy_reg_seen && decoy_reg_seen() == decoy_reg_before + 1);
    }
    SetEnvironmentVariableW(L"X2W_DEVICE_TYPE", nullptr);

    if (failures) {
        std::printf("startup_runtime_proxy_scope: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_runtime_proxy_scope: ok");
    return 0;
}
