// Experimental (docs/startup-runtime.md): which console the title is told it runs on.
//
// Public GDK documentation describes XSystemGetDeviceType and the XSystemDeviceType values
// (Pc = 1, XboxScarlettLockhart = 6, XboxScarlettAnaconda = 7). The proxied runtime calls seen in
// startup runs include no device-type method. A public runtime reimplementation's notes
// (github.com/Marvisak/VIctory, facts only) say the value is not a runtime call: title code reads
// HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM, DWORD value ConsoleMode, where 0x60 is
// Series S, 0x64 is Series X and a missing value is a PC. Those three readings are that source's,
// not measured here.
//
// X2W_DEVICE_TYPE=<n> (with X2W_RUNTIME_PROXY) answers that one value: 6 -> 0x60, 7 -> 0x64,
// 1 -> missing. Unset or any other number passes every read to the real registry.
// Pure helpers, no Windows calls, so a source-built control can check them.
#pragma once

#include <cstddef>
#include <cstdint>

namespace x2w_console_mode {

constexpr long status_ok = 0, status_not_found = 2, status_invalid = 87, status_more_data = 234, status_wrong_type = 1630;
constexpr unsigned long type_dword = 4;
constexpr unsigned long rrf_rt_reg_dword = 0x10;
constexpr std::uint32_t series_s = 0x60, series_x = 0x64;

inline wchar_t fold(wchar_t c) noexcept
{
    return (c >= L'A' && c <= L'Z') ? static_cast<wchar_t>(c - L'A' + L'a') : c;
}

inline bool equal_ci(const wchar_t* a, const wchar_t* b) noexcept
{
    if (!a || !b) return false;
    for (;; ++a, ++b) {
        if (fold(*a) != fold(*b)) return false;
        if (!*a) return true;
    }
}

// Subkey of HKEY_LOCAL_MACHINE. One leading or trailing backslash is accepted.
inline bool is_oem_subkey(const wchar_t* subkey) noexcept
{
    static constexpr wchar_t expect[] = L"software\\microsoft\\windows nt\\currentversion\\oem";
    if (!subkey) return false;
    if (*subkey == L'\\') ++subkey;
    std::size_t i = 0;
    for (; expect[i]; ++i)
        if (fold(subkey[i]) != expect[i]) return false;
    return !subkey[i] || (subkey[i] == L'\\' && !subkey[i + 1]);
}

inline bool is_console_mode_value(const wchar_t* name) noexcept
{
    return equal_ci(name, L"ConsoleMode");
}

enum class Answer { pass, value, missing };

// device_set is false when X2W_DEVICE_TYPE is absent or not a number.
inline Answer answer_for(bool device_set, unsigned long device_type, std::uint32_t* mode) noexcept
{
    if (!device_set) return Answer::pass;
    switch (device_type) {
    case 1: return Answer::missing;
    case 6: if (mode) *mode = series_s; return Answer::value;
    case 7: if (mode) *mode = series_x; return Answer::value;
    default: return Answer::pass;
    }
}

// RegQueryValueExW's output rules for one REG_DWORD.
inline long fill_query(std::uint32_t value, unsigned long* type, unsigned char* data, unsigned long* bytes) noexcept
{
    if (data && !bytes) return status_invalid;
    if (type) *type = type_dword;
    if (!data) {
        if (bytes) *bytes = sizeof(value);
        return status_ok;
    }
    if (*bytes < sizeof(value)) {
        *bytes = sizeof(value);
        return status_more_data;
    }
    for (std::size_t i = 0; i < sizeof(value); ++i) data[i] = static_cast<unsigned char>(value >> (8 * i));
    *bytes = sizeof(value);
    return status_ok;
}

// RegGetValueW's: the flags must allow a DWORD (RRF_RT_REG_DWORD, part of RRF_RT_DWORD and RRF_RT_ANY).
inline long fill_get(unsigned long flags, std::uint32_t value, unsigned long* type, void* data, unsigned long* bytes) noexcept
{
    if (!(flags & rrf_rt_reg_dword)) return status_wrong_type;
    return fill_query(value, type, static_cast<unsigned char*>(data), bytes);
}

}  // namespace x2w_console_mode
