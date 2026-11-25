// Source-built control for console_mode.h and registry_console_mode.inl: the console mode a title
// reads from HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM (value ConsoleMode).
//
// The real registry functions are replaced by local fakes that report the value as missing (a PC),
// so no machine state is read or changed. The hooks are reached the way a title reaches them
// through GetProcAddress. Cases: unset passes the read to the fake; 6 and 7 answer 0x60 and 0x64
// without calling it; 1 answers missing; other keys, values and closed handles pass through.
// --control leaves X2W_DEVICE_TYPE unset for the answered cases, which must print FAIL.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

template <class Fn>
Fn hook(const char* name, Fn fake)
{
    return reinterpret_cast<Fn>(registry_hook_for(name, reinterpret_cast<FARPROC>(fake)));
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
    check("other names have no hook", !registry_hook_for("RegSetValueExW", reinterpret_cast<FARPROC>(&fake_close)));
    check("a hook is never kept as the real function", !registry_hook_for("RegCloseKey", reinterpret_cast<FARPROC>(close)));

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

    device(nullptr, false);
    const std::string& log = x2w_local::fixture_log;
    check("open is logged", log.find("RegOpenKeyExW opened the OEM key, real status 0") != std::string::npos);
    check("pass-through is logged", log.find("RegQueryValueExW ConsoleMode passed through -> status 2") != std::string::npos);
    check("answer is logged", log.find("ConsoleMode -> 0x64 for X2W_DEVICE_TYPE=7") != std::string::npos);
}
}  // namespace

int main(int argc, char** argv)
{
    control = argc > 1 && std::strcmp(argv[1], "--control") == 0;
    pure_checks();
    hook_checks();
    if (failures) {
        std::printf("startup_console_mode: FAIL (%d)\n", failures);
        return 1;
    }
    std::puts("startup_console_mode: ok");
    return 0;
}
