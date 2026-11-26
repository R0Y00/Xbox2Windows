// Source-built control for console_mode.h and registry_console_mode.inl: the console mode a title
// reads from HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM (value ConsoleMode).
//
// The real registry functions are replaced by local fakes that report the value as missing (a PC),
// so no machine state is read or changed. The hooks are reached the way a title reaches them
// through GetProcAddress. Cases: unset passes the read to the fake; 6 and 7 answer 0x60 and 0x64
// without calling it; 1 answers missing; other keys, values and closed handles pass through.
// --control leaves X2W_DEVICE_TYPE unset for the answered cases, which must print FAIL.
//
// The fake registry lives in this EXE, which X2W_CONSOLE_MODE_FIXTURE counts as a registry
// provider. --decoy <dll> loads a source-built DLL that exports RegOpenKeyExW and RegQueryValueExW
// with a distinct result. With the switch unset and set, neither GetProcAddress on the decoy nor a
// synthetic IAT entry naming it (or naming advapi32.dll but bound to the decoy) may be hooked or
// remembered, and the hooks must still reach the fake registry. --control provider matches
// registry imports by name alone, which must print FAIL.
#define X2W_CONSOLE_MODE_FIXTURE 1
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "startup_stubs/console_mode.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/runtime_gpa_patch.h"

namespace x2w_local {
std::string fixture_log;
void fixture_logf(const char* format, ...)
{
    char line[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    fixture_log += line;
    fixture_log += '\n';
}
}  // namespace x2w_local

#define logf fixture_logf
namespace {
#include "../src/startup_stubs/registry_console_mode.inl"
}
#undef logf

namespace {
int failures;
bool control;

void check(const char* what, bool ok)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

HKEY const fake_oem = reinterpret_cast<HKEY>(0x5100);
HKEY const fake_other = reinterpret_cast<HKEY>(0x5200);
int real_queries, real_gets, real_closes;

LSTATUS WINAPI fake_open_ex(HKEY root, LPCWSTR subkey, DWORD, REGSAM, PHKEY out)
{
    if (root != HKEY_LOCAL_MACHINE || !out) return ERROR_FILE_NOT_FOUND;
    *out = x2w_console_mode::is_oem_subkey(subkey) ? fake_oem : fake_other;
    return ERROR_SUCCESS;
}
LSTATUS WINAPI fake_open(HKEY root, LPCWSTR subkey, PHKEY out) { return fake_open_ex(root, subkey, 0, 0, out); }
LSTATUS WINAPI fake_query(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD)
{
    ++real_queries;
    return ERROR_FILE_NOT_FOUND;  // as on a PC: the value is absent
}
LSTATUS WINAPI fake_get(HKEY, LPCWSTR, LPCWSTR, DWORD, LPDWORD, PVOID, LPDWORD)
{
    ++real_gets;
    return ERROR_FILE_NOT_FOUND;
}
LSTATUS WINAPI fake_close(HKEY)
{
    ++real_closes;
    return ERROR_SUCCESS;
}

// The fakes live in this EXE, the fixture's registry provider.
template <class Fn>
Fn hook(const char* name, Fn fake)
{
    return reinterpret_cast<Fn>(registry_hook_for(GetModuleHandleW(nullptr), name, reinterpret_cast<FARPROC>(fake)));
}

void device(const char* value, bool answered_case)
{
    // In the control the answered cases run without the switch, so their checks must fail.
    if (!value || (control && answered_case))
        SetEnvironmentVariableA("X2W_DEVICE_TYPE", nullptr);
    else
        SetEnvironmentVariableA("X2W_DEVICE_TYPE", value);
}

void pure_checks()
{
    using namespace x2w_console_mode;
    check("OEM subkey", is_oem_subkey(L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM"));
    check("OEM subkey in capitals", is_oem_subkey(L"SOFTWARE\\MICROSOFT\\WINDOWS NT\\CURRENTVERSION\\OEM"));
    check("OEM subkey with a leading and a trailing backslash",
          is_oem_subkey(L"\\Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM\\"));
    check("a longer subkey is not the OEM key", !is_oem_subkey(L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEMx"));
    check("a child of the OEM key is not the OEM key", !is_oem_subkey(L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM\\a"));
    check("a parent is not the OEM key", !is_oem_subkey(L"Software\\Microsoft\\Windows NT\\CurrentVersion"));
    check("null subkey", !is_oem_subkey(nullptr));
    check("value name ignores case", is_console_mode_value(L"consolemode"));
    check("other value name", !is_console_mode_value(L"ConsoleModeX"));

    std::uint32_t mode = 0;
    check("unset passes", answer_for(false, 7, &mode) == Answer::pass);
    check("6 is Series S", answer_for(true, 6, &mode) == Answer::value && mode == series_s);
    check("7 is Series X", answer_for(true, 7, &mode) == Answer::value && mode == series_x);
    check("1 is missing", answer_for(true, 1, &mode) == Answer::missing);
    check("8 is passed on", answer_for(true, 8, &mode) == Answer::pass);

    unsigned long type = 0, bytes = 0;
    unsigned char data[8] = {};
    check("size query", fill_query(series_x, &type, nullptr, &bytes) == status_ok && bytes == 4 && type == type_dword);
    bytes = 3;
    check("short buffer", fill_query(series_x, &type, data, &bytes) == status_more_data && bytes == 4 && data[0] == 0);
    check("data without size", fill_query(series_x, &type, data, nullptr) == status_invalid);
    bytes = sizeof(data);
    check("value written", fill_query(series_x, &type, data, &bytes) == status_ok && bytes == 4 && data[0] == 0x64 && data[1] == 0);
    bytes = sizeof(data);
    check("RegGetValue refuses a string-only request", fill_get(0x2 /* RRF_RT_REG_SZ */, series_s, &type, data, &bytes) == status_wrong_type);
    bytes = sizeof(data);
    check("RegGetValue with RRF_RT_DWORD", fill_get(0x18, series_s, &type, data, &bytes) == status_ok && data[0] == 0x60);
}

DWORD query(RegQueryValueExWFn query_fn, HKEY key, const wchar_t* name, LSTATUS* status)
{
    DWORD type = 0, value = 0xffffffff, bytes = sizeof(value);
    *status = query_fn(key, name, nullptr, &type, reinterpret_cast<LPBYTE>(&value), &bytes);
    return value;
}

void hook_checks()
{
    const auto open_ex = hook("RegOpenKeyExW", &fake_open_ex);
    const auto open = hook("RegOpenKeyW", &fake_open);
    const auto query_fn = hook("RegQueryValueExW", &fake_query);
    const auto get = hook("RegGetValueW", &fake_get);
    const auto close = hook("RegCloseKey", &fake_close);
    check("every registry name has a hook", open_ex && open && query_fn && get && close);
    if (!(open_ex && open && query_fn && get && close)) return;
    const HMODULE exe = GetModuleHandleW(nullptr);
    check("other names have no hook", !registry_hook_for(exe, "RegSetValueExW", reinterpret_cast<FARPROC>(&fake_close)));
    check("a hook is never kept as the real function", !registry_hook_for(exe, "RegCloseKey", reinterpret_cast<FARPROC>(close)));

    const wchar_t* oem_path = L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM";
    HKEY key = nullptr, other = nullptr;
    check("OEM key opens through the hook", open_ex(HKEY_LOCAL_MACHINE, oem_path, 0, KEY_READ, &key) == ERROR_SUCCESS && key == fake_oem);
    check("other key opens through the hook", open_ex(HKEY_LOCAL_MACHINE, L"Software\\Other", 0, KEY_READ, &other) == ERROR_SUCCESS);

    LSTATUS status = 0;
    device(nullptr, false);
    int before = real_queries;
    query(query_fn, key, L"ConsoleMode", &status);
    check("unset: the read reaches the registry", real_queries == before + 1 && status == ERROR_FILE_NOT_FOUND);

    device("6", true);
    before = real_queries;
    DWORD value = query(query_fn, key, L"ConsoleMode", &status);
    check("6: Series S answered locally", status == ERROR_SUCCESS && value == 0x60 && real_queries == before);

    device("7", true);
    value = query(query_fn, key, L"ConsoleMode", &status);
    check("7: Series X answered locally", status == ERROR_SUCCESS && value == 0x64 && real_queries == before);

    DWORD type = 0, bytes = 0;
    status = query_fn(key, L"ConsoleMode", nullptr, &type, nullptr, &bytes);
    check("7: size query", status == ERROR_SUCCESS && bytes == 4 && type == REG_DWORD);

    DWORD got = 0, got_bytes = sizeof(got);
    before = real_gets;
    status = get(HKEY_LOCAL_MACHINE, oem_path, L"ConsoleMode", RRF_RT_REG_DWORD, &type, &got, &got_bytes);
    check("7: RegGetValueW by path answered locally", status == ERROR_SUCCESS && got == 0x64 && real_gets == before);

    device("1", false);
    before = real_queries;
    query(query_fn, key, L"ConsoleMode", &status);
    check("1: missing, as on a PC", status == ERROR_FILE_NOT_FOUND && real_queries == before);

    device("7", false);
    before = real_queries;
    query(query_fn, key, L"OtherValue", &status);
    check("7: another value passes through", real_queries == before + 1);
    query(query_fn, other, L"ConsoleMode", &status);
    check("7: ConsoleMode under another key passes through", real_queries == before + 2);
    device("not-a-number", false);
    query(query_fn, key, L"ConsoleMode", &status);
    check("a value that is not a number passes through", real_queries == before + 3);

    HKEY second = nullptr;
    device("7", true);
    check("RegOpenKeyW opens the OEM key", open(HKEY_LOCAL_MACHINE, oem_path, &second) == ERROR_SUCCESS && second == fake_oem);
    check("RegCloseKey reaches the registry", close(key) == ERROR_SUCCESS && real_closes == 1);
    device("7", false);
    before = real_queries;
    query(query_fn, key, L"ConsoleMode", &status);
    check("a closed OEM handle passes through", real_queries == before + 1);

    // A full table of OEM handles: one more open is logged once and not tracked.
    const char* full_line = "OEM key handles already open, another is not tracked";
    check("no full-table line before the table fills", x2w_local::fixture_log.find(full_line) == std::string::npos);
    HKEY extra = nullptr;
    for (int i = 0; i < 17; ++i) open_ex(HKEY_LOCAL_MACHINE, oem_path, 0, KEY_READ, &extra);
    const std::size_t full_at = x2w_local::fixture_log.find(full_line);
    check("a full OEM handle table is logged", full_at != std::string::npos);
    check("the full-table line is logged once", full_at != std::string::npos &&
                                                   x2w_local::fixture_log.find(full_line, full_at + 1) == std::string::npos);
    close(second);
    check("closing the OEM handle empties the table", !oem_key(fake_oem));

    device(nullptr, false);
    const std::string& log = x2w_local::fixture_log;
    check("open is logged", log.find("RegOpenKeyExW opened the OEM key, real status 0") != std::string::npos);
    check("pass-through is logged", log.find("RegQueryValueExW ConsoleMode passed through -> status 2") != std::string::npos);
    check("answer is logged", log.find("ConsoleMode -> 0x64 for X2W_DEVICE_TYPE=7") != std::string::npos);
}

// ---- a package DLL with the same export names -----------------------------------------------------
constexpr LSTATUS decoy_status = 0x5EC0;  // startup_console_mode_decoy.cpp

struct Import {
    const char* name;
    void* bound;
};
struct ImportDll {
    const char* dll;
    std::vector<Import> imports;
};

// Synthetic PE32+ image with one regular import descriptor per entry, each slot already bound.
std::vector<unsigned char> build_image(const std::vector<ImportDll>& dlls, std::vector<std::vector<std::uint32_t>>* slots)
{
    namespace patch = x2w_runtime_patch;
    std::vector<unsigned char> image(0x1000, 0);
    auto* dos = reinterpret_cast<patch::DosStub*>(image.data());
    dos->e_magic = 0x5A4D;
    dos->e_lfanew = 0x80;
    auto* nt = image.data() + 0x80;
    *reinterpret_cast<std::uint32_t*>(nt) = 0x00004550;
    *reinterpret_cast<std::uint16_t*>(nt + 24) = 0x20B;
    auto* dirs = reinterpret_cast<patch::DataDir*>(nt + 24 + 112);
    dirs[1].virtual_address = 0x200;
    dirs[1].size = static_cast<std::uint32_t>(sizeof(patch::ImportDesc) * (dlls.size() + 1));
    std::uint32_t next = 0x400;
    const auto take = [&](std::size_t bytes) {
        const std::uint32_t at = next;
        next = static_cast<std::uint32_t>((next + bytes + 7) & ~std::size_t{7});
        return at;
    };
    slots->clear();
    for (std::size_t d = 0; d < dlls.size(); ++d) {
        auto* imp = reinterpret_cast<patch::ImportDesc*>(image.data() + 0x200) + d;
        const std::size_t count = dlls[d].imports.size();
        imp->original_first_thunk = take(8 * (count + 1));
        imp->first_thunk = take(8 * (count + 1));
        imp->name = take(std::strlen(dlls[d].dll) + 1);
        std::memcpy(image.data() + imp->name, dlls[d].dll, std::strlen(dlls[d].dll) + 1);
        slots->emplace_back();
        for (std::size_t i = 0; i < count; ++i) {
            const char* name = dlls[d].imports[i].name;
            const std::uint32_t hint = take(2 + std::strlen(name) + 1);
            std::memcpy(image.data() + hint + 2, name, std::strlen(name) + 1);
            reinterpret_cast<patch::Thunk64*>(image.data() + imp->original_first_thunk)[i].value = hint;
            reinterpret_cast<patch::Thunk64*>(image.data() + imp->first_thunk)[i].value =
                reinterpret_cast<std::uint64_t>(dlls[d].imports[i].bound);
            slots->back().push_back(imp->first_thunk + static_cast<std::uint32_t>(8 * i));
        }
    }
    return image;
}

// What hooked_get_proc_address returns: the hook for a registry provider, else the real export.
FARPROC resolve(HMODULE module, const char* name)
{
    const FARPROC real = GetProcAddress(module, name);
    const FARPROC hooked = real ? registry_hook_for(module, name, real) : nullptr;
    return hooked ? hooked : real;
}

void* remembered(RegistryImport which) { return real_registry[which].load(); }

void decoy_checks(HMODULE decoy, const char* pass)
{
    using SeenFn = LONG(WINAPI*)();
    const auto seen = reinterpret_cast<SeenFn>(GetProcAddress(decoy, "decoy_calls"));
    void* const decoy_open = reinterpret_cast<void*>(GetProcAddress(decoy, "RegOpenKeyExW"));
    void* const decoy_query = reinterpret_cast<void*>(GetProcAddress(decoy, "RegQueryValueExW"));
    const std::string label = std::string(pass) + ": ";
    const auto say = [&](const char* what, bool ok) { check((label + what).c_str(), ok); };
    say("the decoy exports both registry names", seen && decoy_open && decoy_query);
    if (!(seen && decoy_open && decoy_query)) return;

    // GetProcAddress on the decoy: its own exports, which keep their own result.
    const auto open_fn = reinterpret_cast<RegOpenKeyExWFn>(resolve(decoy, "RegOpenKeyExW"));
    const auto query_fn = reinterpret_cast<RegQueryValueExWFn>(resolve(decoy, "RegQueryValueExW"));
    say("GetProcAddress(decoy, RegOpenKeyExW) is the decoy's export", reinterpret_cast<void*>(open_fn) == decoy_open);
    say("GetProcAddress(decoy, RegQueryValueExW) is the decoy's export", reinterpret_cast<void*>(query_fn) == decoy_query);
    LONG before = seen();
    HKEY out = nullptr;
    LSTATUS status = 0;
    query(query_fn, fake_other, L"ConsoleMode", &status);
    say("the decoy's RegQueryValueExW answers its own result", status == decoy_status && seen() == before + 1);
    say("the decoy's RegOpenKeyExW answers its own result",
        open_fn(HKEY_LOCAL_MACHINE, L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM", 0, KEY_READ, &out) == decoy_status &&
            seen() == before + 2);
    say("a decoy address is refused under the provider module",
        !registry_hook_for(GetModuleHandleW(nullptr), "RegQueryValueExW", reinterpret_cast<FARPROC>(decoy_query)));

    // A synthetic IAT: the decoy by name, advapi32.dll by name but bound to the decoy, and
    // advapi32.dll bound to the fake registry (the one entry that is rewritten).
    std::vector<std::vector<std::uint32_t>> slots;
    auto image = build_image({{"startup_console_mode_decoy.dll", {{"RegOpenKeyExW", decoy_open}, {"RegQueryValueExW", decoy_query}}},
                              {"ADVAPI32.dll", {{"RegQueryValueExW", decoy_query}}},
                              {"advapi32.dll", {{"RegQueryValueExW", reinterpret_cast<void*>(&fake_query)},
                                                {"RegCloseKey", reinterpret_cast<void*>(&fake_close)}}}},
                             &slots);
    const auto slot = [&](std::size_t d, std::size_t i) { return *reinterpret_cast<void**>(image.data() + slots[d][i]); };
    const int patched = patch_registry_imports(image.data(), image.size());
    say("only the provider-bound imports are rewritten", patched == 2);
    say("the decoy's RegOpenKeyExW slot is untouched", slot(0, 0) == decoy_open);
    say("the decoy's RegQueryValueExW slot is untouched", slot(0, 1) == decoy_query);
    say("advapi32.dll bound to the decoy is untouched", slot(1, 0) == decoy_query);
    say("advapi32.dll bound to the registry is rewritten",
        slot(2, 0) == registry_hooks[reg_query_value_ex_w] && slot(2, 1) == registry_hooks[reg_close_key]);
    before = seen();
    query(reinterpret_cast<RegQueryValueExWFn>(slot(0, 1)), fake_other, L"ConsoleMode", &status);
    say("a call through the decoy's slot reaches the decoy", status == decoy_status && seen() == before + 1);

    // Neither route made the decoy the real registry: the hooks still reach the fake registry.
    say("the decoy's RegOpenKeyExW is never remembered", remembered(reg_open_key_ex_w) != decoy_open);
    say("the decoy's RegQueryValueExW is never remembered",
        remembered(reg_query_value_ex_w) != decoy_query && remembered(reg_query_value_ex_w) == reinterpret_cast<void*>(&fake_query));
    before = seen();
    const int real_before = real_queries;
    query(reinterpret_cast<RegQueryValueExWFn>(registry_hooks[reg_query_value_ex_w]), fake_other, L"ConsoleMode", &status);
    say("the hook still reaches the registry, not the decoy",
        status == ERROR_FILE_NOT_FOUND && real_queries == real_before + 1 && seen() == before);
}

void provider_checks(HMODULE decoy)
{
    const HMODULE advapi = LoadLibraryW(L"advapi32.dll");  // this EXE does not import it
    check("advapi32.dll is a registry provider", registry_provider_module(advapi));
    check("kernelbase.dll is a registry provider", registry_provider_module(GetModuleHandleW(L"kernelbase.dll")));
    check("advapi32.dll's RegQueryValueExW lies in a provider",
          advapi && registry_provider_address(reinterpret_cast<void*>(GetProcAddress(advapi, "RegQueryValueExW"))));
    check("the decoy is not a registry provider", decoy && !registry_provider_module(decoy));
    check("import names: advapi32.dll in any case", registry_provider_dll("ADVAPI32.dll"));
    check("import names: the registry API set", registry_provider_dll("api-ms-win-core-registry-l1-1-0.dll"));
    check("import names: kernelbase.dll", registry_provider_dll("KernelBase.dll"));
    check("import names: the decoy is refused", !registry_provider_dll("startup_console_mode_decoy.dll"));
    check("import names: a longer name is refused", !registry_provider_dll("advapi32.dll.x"));
    check("import names: another API set is refused", !registry_provider_dll("api-ms-win-core-file-l1-1-0.dll"));
}
}  // namespace

int main(int argc, char** argv)
{
    const char* decoy_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--control") == 0) {
            if (i + 1 < argc && std::strcmp(argv[i + 1], "provider") == 0) {
                registry_any_provider = true;  // name-only matching; the decoy checks must fail
                ++i;
            } else {
                control = true;
            }
        } else if (std::strcmp(argv[i], "--decoy") == 0 && i + 1 < argc) {
            decoy_path = argv[++i];
        }
    }
    pure_checks();
    std::string decoy_file = decoy_path ? decoy_path : "";
    for (char& c : decoy_file)
        if (c == '/') c = '\\';
    const HMODULE decoy = decoy_path ? LoadLibraryA(decoy_file.c_str()) : nullptr;
    check("load the decoy DLL (--decoy)", decoy != nullptr);
    if (!registry_any_provider) provider_checks(decoy);
    // Before hook_checks registers the fake registry, so a name-only match would keep the decoy first.
    if (decoy) {
        SetEnvironmentVariableA("X2W_DEVICE_TYPE", nullptr);
        decoy_checks(decoy, "decoy, switch unset");
        SetEnvironmentVariableA("X2W_DEVICE_TYPE", "7");
        decoy_checks(decoy, "decoy, X2W_DEVICE_TYPE=7");
        SetEnvironmentVariableA("X2W_DEVICE_TYPE", nullptr);
    }
    hook_checks();
    if (failures) {
        std::printf("startup_console_mode: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_console_mode: ok");
    return 0;
}
