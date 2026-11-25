// Windows-only: EXE + late-loaded package helper through the real loader.
//
// Loads a carrier DLL that hosts runtime_proxy (X2W_RUNTIME_PROXY=1), a fake xgameruntime.dll, an
// unrelated decoy that also exports QueryApiImpl, then a helper beside the EXE. The helper's
// GetProcAddress IAT is rewritten after linking (outside LdrDllNotification). Looking up
// QueryApiImpl on xgameruntime must return the proxy; the same name on the decoy must still be the
// decoy's export. A QueryApiImpl pointer cached before the proxy attaches stays the real export
// (pre-attach gap). A delay-imported QueryApiImpl that has already bound is rewritten to the proxy;
// an unbound delay thunk must not be. No game files.
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

    // Pre-attach cache: resolve before the proxy carrier loads. This pointer must stay the real
    // export after attach (IAT/GPA hooks do not rewrite already-held function pointers).
    const FARPROC cached_before_proxy = xgr_query;

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

    const HMODULE helper = load_beside(dir, L"startup_runtime_proxy_helper.dll");
    check("late-load package helper", helper != nullptr);
    const auto resolve = helper ? reinterpret_cast<ResolveFn>(GetProcAddress(helper, "resolve_query")) : nullptr;
    check("helper exports resolve_query", resolve != nullptr);
    const auto call_delay = helper ? reinterpret_cast<QueryFn>(GetProcAddress(helper, "call_delay_query")) : nullptr;
    check("helper exports call_delay_query", call_delay != nullptr);

    check("helper sees proxied xgameruntime QueryApiImpl",
          resolve && wait_for_helper_hook(resolve, xgr, host_xgr, 2000));
    const FARPROC helper_decoy = resolve ? resolve(decoy) : nullptr;
    check("helper GetProcAddress(decoy) stays original", helper_decoy == decoy_query);

    ULONGLONG* delay_slot = delay_query_slot(helper);
    check("helper has delay QueryApiImpl slot", delay_slot != nullptr);
    if (delay_slot) {
        const ULONGLONG before_bind = *delay_slot;
        check("delay slot unbound before first use",
              !x2w_runtime_patch::delay_query_slot_is_bound(before_bind, reinterpret_cast<ULONGLONG>(xgr_query),
                                                            reinterpret_cast<ULONGLONG>(host_xgr)));
        // Negative: give the quiet rebind pass time; an unbound thunk must stay untouched.
        Sleep(300);
        check("unbound delay slot not rewritten", *delay_slot == before_bind);
        if (call_delay) call_delay(nullptr, nullptr, nullptr);
        // The quiet rebind pass may already have swapped the slot for the proxy.
        const ULONGLONG after_bind = *delay_slot;
        check("delay bind wrote real export or proxy",
              after_bind == reinterpret_cast<ULONGLONG>(xgr_query) ||
                  after_bind == reinterpret_cast<ULONGLONG>(host_xgr));
        check("bound delay slot rewritten to proxy",
              wait_for_slot(delay_slot, reinterpret_cast<ULONGLONG>(host_xgr), 2000));
    }

    // Intended call: invoke the proxied entry; the fake xgameruntime must run.
    using SeenFn = LONG(WINAPI*)();
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

    if (failures) {
        std::printf("startup_runtime_proxy_scope: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_runtime_proxy_scope: ok");
    return 0;
}
