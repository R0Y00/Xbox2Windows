#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdint>

extern "C" __declspec(dllimport) void* XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);

namespace {
constexpr DWORD reserve = 0x20402000;
constexpr SIZE_T gib = SIZE_T{1} << 30;

bool reserved(void* address, SIZE_T bytes)
{
    MEMORY_BASIC_INFORMATION info;
    return VirtualQuery(address, &info, sizeof(info)) == sizeof(info) &&
           info.BaseAddress == address && info.AllocationBase == address &&
           info.RegionSize == bytes && info.State == MEM_RESERVE;
}

bool refused(void* base, SIZE_T bytes, DWORD type, ULONGLONG attributes, DWORD protection)
{
    SetLastError(ERROR_SUCCESS);
    return !XMemVirtualAlloc(base, bytes, type, attributes, protection) &&
           GetLastError() == ERROR_INVALID_PARAMETER;
}

DWORD check_pair(SIZE_T bytes)
{
    void* first = XMemVirtualAlloc(nullptr, bytes, reserve, 0x21, PAGE_READWRITE);
    void* second = XMemVirtualAlloc(nullptr, bytes, reserve, 3, 0x801);
    if (!first || !second) return 1;
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    if (a % 65536 || b % 65536 || (a < b ? b - a : a - b) < bytes) return 2;
    if (!reserved(first, bytes) || !reserved(second, bytes)) return 3;

    if (!refused(first, bytes, reserve, 0x21, PAGE_READWRITE) ||
        !refused(nullptr, 0, reserve, 0x21, PAGE_READWRITE) ||
        !refused(nullptr, bytes, reserve | MEM_COMMIT, 0x21, PAGE_READWRITE) ||
        !refused(nullptr, bytes, reserve, 3, PAGE_READWRITE)) return 4;
    if (!reserved(first, bytes) || !reserved(second, bytes)) return 5;

    if (!VirtualFree(first, 0, MEM_RELEASE) || !VirtualFree(second, 0, MEM_RELEASE)) return 6;
    void* reused = VirtualAlloc(first, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (reused != first || !reserved(reused, bytes) || !VirtualFree(reused, 0, MEM_RELEASE)) return 7;
    return 0;
}
}

extern "C" DWORD WINAPI reservation_entry()
{
    if (DWORD code = check_pair(16 * gib)) return code;
    if (DWORD code = check_pair(24 * gib)) return 10 + code;
    return 0;
}
