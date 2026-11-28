// Experimental (docs/startup-runtime.md): registry reads of the console mode (see console_mode.h).
// Included inside runtime_proxy.cpp's anonymous namespace, and on its own by
// tests/startup_console_mode_check.cpp. Needs <windows.h>, <atomic>, local_log.h,
// runtime_gpa_patch.h and console_mode.h.
//
// With X2W_RUNTIME_PROXY the stand-in rewrites five registry imports of the EXE and package helpers
// (regular IAT and GetProcAddress by name), only where the Windows registry provides them: see
// registry_provider_module below. Every call goes to the real function except a read of
// ConsoleMode under HKLM\...\OEM while X2W_DEVICE_TYPE names a known console. Opening the OEM
// key and reading the value are logged a few times in either case, so a run shows whether the
// title asks at all.

enum RegistryImport { reg_open_key_ex_w, reg_open_key_w, reg_query_value_ex_w, reg_get_value_w, reg_close_key, registry_imports };
constexpr const char* registry_import_names[registry_imports] = {"RegOpenKeyExW", "RegOpenKeyW", "RegQueryValueExW", "RegGetValueW",
                                                                 "RegCloseKey"};
using RegOpenKeyExWFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
using RegOpenKeyWFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, PHKEY);
using RegQueryValueExWFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
using RegGetValueWFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, LPCWSTR, DWORD, LPDWORD, PVOID, LPDWORD);
using RegCloseKeyFn = LSTATUS(WINAPI*)(HKEY);

std::atomic<void*> real_registry[registry_imports];

int registry_import_index(const char* name) noexcept
{
    if (!name) return -1;
    for (int i = 0; i < registry_imports; ++i)
        if (std::strcmp(name, registry_import_names[i]) == 0) return i;
    return -1;
}

// ---- which modules are the Windows registry -----------------------------------------------------
// A package DLL can export the same names. Only these system modules count as the registry, for the
// GetProcAddress module, for the module that holds a bound IAT address and for what is remembered
// as the real function. The api-ms-win-core-registry API sets resolve to kernelbase.dll.
constexpr const wchar_t* registry_provider_names[] = {L"advapi32.dll", L"kernelbase.dll", L"kernel32.dll"};

#if defined(X2W_CONSOLE_MODE_FIXTURE)
// tests/startup_console_mode_check.cpp keeps its fake registry in the test EXE, which counts as one
// more provider there. Its provider control sets registry_any_provider (name-only matching), which
// the decoy checks must catch.
bool registry_any_provider;
bool fixture_registry_provider(HMODULE module) noexcept { return module == GetModuleHandleW(nullptr); }
#else
constexpr bool registry_any_provider = false;
constexpr bool fixture_registry_provider(HMODULE) noexcept { return false; }
#endif

// The loaded system-directory copy of a provider, by full path, so a same-named DLL elsewhere does not match.
bool registry_provider_module(HMODULE module) noexcept
{
    if (registry_any_provider) return true;
    if (!module) return false;
    if (fixture_registry_provider(module)) return true;
    const DWORD saved = GetLastError();
    wchar_t path[MAX_PATH]{};
    const UINT length = GetSystemDirectoryW(path, MAX_PATH);
    bool found = false;
    for (const wchar_t* name : registry_provider_names) {
        if (found || !length || length >= MAX_PATH) break;
        std::size_t at = length, i = 0;
        path[at++] = L'\\';
        for (; name[i] && at + 1 < MAX_PATH; ++i) path[at++] = name[i];
        if (name[i]) break;
        path[at] = 0;
        found = GetModuleHandleW(path) == module;
    }
    SetLastError(saved);
    return found;
}

bool registry_provider_address(const void* address) noexcept
{
    if (registry_any_provider) return true;
    const DWORD saved = GetLastError();
    HMODULE module = nullptr;
    const bool known = address && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                                     static_cast<LPCWSTR>(address), &module);
    SetLastError(saved);
    return known && registry_provider_module(module);
}

// Import DLL names (any case) that supply the registry: the providers above and the
// api-ms-win-core-registry-* API sets.
bool registry_provider_dll(const char* dll) noexcept
{
    if (registry_any_provider) return true;
    if (!dll) return false;
    const auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
    const auto equal = [&](const char* expect) {
        std::size_t i = 0;
        for (; expect[i]; ++i)
            if (lower(dll[i]) != expect[i]) return false;
        return dll[i] == 0;
    };
    if (equal("advapi32.dll") || equal("kernelbase.dll") || equal("kernel32.dll")) return true;
    static constexpr char api_set[] = "api-ms-win-core-registry-";
    for (std::size_t i = 0; api_set[i]; ++i)
        if (lower(dll[i]) != api_set[i]) return false;
    return true;
}

// ---- OEM key handles the title opened -----------------------------------------------------------
SRWLOCK oem_lock = SRWLOCK_INIT;
HKEY oem_keys[16];

void note_oem_table_full() noexcept
{
    static volatile LONG logged;
    if (x2w_local::first_calls(logged, 1))
        x2w_local::logf("x2w-runtime console mode: %u OEM key handles already open, another is not tracked",
                        static_cast<unsigned>(sizeof(oem_keys) / sizeof(oem_keys[0])));
}

void track_oem(HKEY key) noexcept
{
    AcquireSRWLockExclusive(&oem_lock);
    bool tracked = false;
    for (HKEY& slot : oem_keys) {
        if (!slot) {
            slot = key;
            tracked = true;
            break;
        }
    }
    ReleaseSRWLockExclusive(&oem_lock);
    if (!tracked) note_oem_table_full();
}

bool oem_key(HKEY key) noexcept
{
    if (!key) return false;
    AcquireSRWLockShared(&oem_lock);
    bool found = false;
    for (HKEY slot : oem_keys) found = found || slot == key;
    ReleaseSRWLockShared(&oem_lock);
    return found;
}

void untrack_oem(HKEY key) noexcept
{
    AcquireSRWLockExclusive(&oem_lock);
    for (HKEY& slot : oem_keys)
        if (slot == key) slot = nullptr;
    ReleaseSRWLockExclusive(&oem_lock);
}

// ---- the answer ---------------------------------------------------------------------------------
x2w_console_mode::Answer console_answer(std::uint32_t* mode, unsigned long* device) noexcept
{
    const DWORD saved = GetLastError();
    char text[16]{};
    const DWORD length = GetEnvironmentVariableA("X2W_DEVICE_TYPE", text, sizeof(text));
    SetLastError(saved);
    char* end = nullptr;
    const unsigned long value = length && length < sizeof(text) ? std::strtoul(text, &end, 0) : 0;
    const bool set = length && length < sizeof(text) && end && end != text && !*end;
    *device = value;
    return x2w_console_mode::answer_for(set, value, mode);
}

void note_oem_open(const char* via, LSTATUS status) noexcept
{
    static volatile LONG logged;
    if (x2w_local::first_calls(logged, 4))
        x2w_local::logf("x2w-runtime console mode: %s opened the OEM key, real status %ld", via, static_cast<long>(status));
}

// One line per read (the first eight): what was answered, and the real status or value when passed on.
void note_console_read(const char* via, x2w_console_mode::Answer answer, unsigned long device, std::uint32_t value, LSTATUS status) noexcept
{
    static volatile LONG logged;
    if (!x2w_local::first_calls(logged, 8)) return;
    if (answer == x2w_console_mode::Answer::value)
        x2w_local::logf("x2w-runtime console mode: %s ConsoleMode -> 0x%x for X2W_DEVICE_TYPE=%lu, status %ld", via, value, device,
                        static_cast<long>(status));
    else if (answer == x2w_console_mode::Answer::missing)
        x2w_local::logf("x2w-runtime console mode: %s ConsoleMode -> missing (PC) for X2W_DEVICE_TYPE=%lu", via, device);
    else if (status == ERROR_SUCCESS)
        x2w_local::logf("x2w-runtime console mode: %s ConsoleMode passed through -> 0x%x", via, value);
    else
        x2w_local::logf("x2w-runtime console mode: %s ConsoleMode passed through -> status %ld", via, static_cast<long>(status));
}

template <class Fn>
Fn real_reg(RegistryImport which) noexcept
{
    return reinterpret_cast<Fn>(real_registry[which].load(std::memory_order_acquire));
}

LSTATUS WINAPI hooked_reg_open_key_ex_w(HKEY root, LPCWSTR subkey, DWORD options, REGSAM access, PHKEY out)
{
    const auto real = real_reg<RegOpenKeyExWFn>(reg_open_key_ex_w);
    if (!real) return ERROR_PROC_NOT_FOUND;
    const LSTATUS status = real(root, subkey, options, access, out);
    if (root == HKEY_LOCAL_MACHINE && x2w_console_mode::is_oem_subkey(subkey)) {
        note_oem_open("RegOpenKeyExW", status);
        if (status == ERROR_SUCCESS && out) track_oem(*out);
    }
    return status;
}

LSTATUS WINAPI hooked_reg_open_key_w(HKEY root, LPCWSTR subkey, PHKEY out)
{
    const auto real = real_reg<RegOpenKeyWFn>(reg_open_key_w);
    if (!real) return ERROR_PROC_NOT_FOUND;
    const LSTATUS status = real(root, subkey, out);
    if (root == HKEY_LOCAL_MACHINE && x2w_console_mode::is_oem_subkey(subkey)) {
        note_oem_open("RegOpenKeyW", status);
        if (status == ERROR_SUCCESS && out) track_oem(*out);
    }
    return status;
}

std::uint32_t dword_read(LSTATUS status, const void* data, const DWORD* bytes) noexcept
{
    std::uint32_t value = 0;
    if (status == ERROR_SUCCESS && data && bytes && *bytes >= sizeof(value)) std::memcpy(&value, data, sizeof(value));
    return value;
}

LSTATUS WINAPI hooked_reg_query_value_ex_w(HKEY key, LPCWSTR name, LPDWORD reserved, LPDWORD type, LPBYTE data, LPDWORD bytes)
{
    const auto real = real_reg<RegQueryValueExWFn>(reg_query_value_ex_w);
    if (!x2w_console_mode::is_console_mode_value(name) || !oem_key(key))
        return real ? real(key, name, reserved, type, data, bytes) : ERROR_PROC_NOT_FOUND;
    std::uint32_t mode = 0;
    unsigned long device = 0;
    const auto answer = console_answer(&mode, &device);
    LSTATUS status = ERROR_FILE_NOT_FOUND;
    if (answer == x2w_console_mode::Answer::value)
        status = x2w_console_mode::fill_query(mode, reinterpret_cast<unsigned long*>(type), data, reinterpret_cast<unsigned long*>(bytes));
    else if (answer == x2w_console_mode::Answer::pass)
        status = real ? real(key, name, reserved, type, data, bytes) : ERROR_PROC_NOT_FOUND;
    note_console_read("RegQueryValueExW", answer, device, answer == x2w_console_mode::Answer::value ? mode : dword_read(status, data, bytes), status);
    return status;
}

LSTATUS WINAPI hooked_reg_get_value_w(HKEY key, LPCWSTR subkey, LPCWSTR name, DWORD flags, LPDWORD type, PVOID data, LPDWORD bytes)
{
    const auto real = real_reg<RegGetValueWFn>(reg_get_value_w);
    const bool oem = (key == HKEY_LOCAL_MACHINE && x2w_console_mode::is_oem_subkey(subkey)) || (oem_key(key) && (!subkey || !*subkey));
    if (!oem || !x2w_console_mode::is_console_mode_value(name))
        return real ? real(key, subkey, name, flags, type, data, bytes) : ERROR_PROC_NOT_FOUND;
    std::uint32_t mode = 0;
    unsigned long device = 0;
    const auto answer = console_answer(&mode, &device);
    LSTATUS status = ERROR_FILE_NOT_FOUND;
    if (answer == x2w_console_mode::Answer::value)
        status = x2w_console_mode::fill_get(flags, mode, reinterpret_cast<unsigned long*>(type), data, reinterpret_cast<unsigned long*>(bytes));
    else if (answer == x2w_console_mode::Answer::pass)
        status = real ? real(key, subkey, name, flags, type, data, bytes) : ERROR_PROC_NOT_FOUND;
    note_console_read("RegGetValueW", answer, device, answer == x2w_console_mode::Answer::value ? mode : dword_read(status, data, bytes), status);
    return status;
}

LSTATUS WINAPI hooked_reg_close_key(HKEY key)
{
    const auto real = real_reg<RegCloseKeyFn>(reg_close_key);
    untrack_oem(key);
    return real ? real(key) : ERROR_PROC_NOT_FOUND;
}

void* const registry_hooks[registry_imports] = {
    reinterpret_cast<void*>(&hooked_reg_open_key_ex_w), reinterpret_cast<void*>(&hooked_reg_open_key_w),
    reinterpret_cast<void*>(&hooked_reg_query_value_ex_w), reinterpret_cast<void*>(&hooked_reg_get_value_w),
    reinterpret_cast<void*>(&hooked_reg_close_key)};

// Keep the first real address for each import: one inside a registry provider, never one of the
// hooks themselves.
bool remember_registry(int which, void* real) noexcept
{
    if (which < 0 || which >= registry_imports || !real) return false;
    for (void* hook : registry_hooks)
        if (real == hook) return false;
    if (!registry_provider_address(real)) return false;
    void* expected = nullptr;
    real_registry[which].compare_exchange_strong(expected, real, std::memory_order_acq_rel);
    return true;
}

// GetProcAddress by name: the hook for a registry import looked up on a registry provider, or null
// for any other name or module.
FARPROC registry_hook_for(HMODULE module, LPCSTR name, FARPROC real) noexcept
{
    const int which = registry_import_index(name);
    if (which < 0 || !registry_provider_module(module) || !remember_registry(which, reinterpret_cast<void*>(real))) return nullptr;
    return reinterpret_cast<FARPROC>(registry_hooks[which]);
}

// Rewrite one image's regular registry IAT slots that import from a registry provider and are bound
// inside one. Delay-import slots hold a thunk and are left alone.
int patch_registry_imports(unsigned char* base, std::size_t image_size) noexcept
{
    int patched = 0;
    const auto classify = [](const char* dll, const char* name, x2w_runtime_patch::Kind* kind) {
        if (registry_import_index(name) < 0 || !registry_provider_dll(dll)) return false;
        *kind = x2w_runtime_patch::Kind::registry;
        return true;
    };
    x2w_runtime_patch::visit_imports_with(base, image_size, classify, [&](const x2w_runtime_patch::Hit& hit) {
        if (hit.delay_import) return;
        const int which = registry_import_index(hit.import_name);
        auto* slot = reinterpret_cast<ULONGLONG*>(base + hit.first_thunk_rva);
        if (*slot == reinterpret_cast<ULONGLONG>(registry_hooks[which])) return;
        if (!remember_registry(which, reinterpret_cast<void*>(*slot))) return;
        DWORD old = 0;
        if (!VirtualProtect(slot, 8, PAGE_READWRITE, &old)) return;
        *slot = reinterpret_cast<ULONGLONG>(registry_hooks[which]);
        VirtualProtect(slot, 8, old, &old);
        ++patched;
    });
    return patched;
}
