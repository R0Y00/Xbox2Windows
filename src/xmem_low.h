#pragma once
#include <windows.h>

namespace x2w_low {
inline bool on() noexcept
{
    static const bool value = [] {
        char text[8];
        const DWORD n = GetEnvironmentVariableA("X2W_XMEM_LOW40", text, sizeof text);
        return n && n < sizeof text && text[0] == '1';
    }();
    return value;
}

using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
inline VirtualAlloc2Fn virtual_alloc2() noexcept
{
    static const VirtualAlloc2Fn fn = [] {
        const HMODULE base = GetModuleHandleW(L"kernelbase.dll");
        return base ? reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(base, "VirtualAlloc2")) : nullptr;
    }();
    return fn;
}

// address == nullptr only; type/protect as for VirtualAlloc2 (placeholders allowed).
inline void* alloc(SIZE_T bytes, ULONG type, ULONG protect) noexcept
{
    const auto fn = virtual_alloc2();
    if (!fn) return VirtualAlloc(nullptr, bytes, type, protect);
    if (on()) {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        // Floor at 4 GiB: low 32-bit addresses can collide with values the title treats as non-pointers.
        requirements.LowestStartingAddress = reinterpret_cast<void*>(ULONG_PTR{1} << 32);
        requirements.HighestEndingAddress = reinterpret_cast<void*>((ULONG_PTR{1} << 40) - 1);
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        if (void* low = fn(GetCurrentProcess(), nullptr, bytes, type, protect, &parameter, 1)) return low;
    }
    return fn(GetCurrentProcess(), nullptr, bytes, type, protect, nullptr, 0);
}
}  // namespace x2w_low
