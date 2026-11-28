// Recording stand-ins for d3d12_xs.dll and xmem.dll, used by the configure_vm fixture (#874).
// They implement no configuration or reservation; each call appends one line to the file named
// by CONFIGURE_VM_LOG. CONFIGURE_VM_HRESULT selects the configuration result, and
// CONFIGURE_VM_WRITE=1/2 makes the stand-in write inside/after the caller's 32-byte block.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cwchar>

namespace {
void append(const char* line, int length)
{
    wchar_t path[MAX_PATH];
    const DWORD size = GetEnvironmentVariableW(L"CONFIGURE_VM_LOG", path, MAX_PATH);
    if (!size || size >= MAX_PATH || length <= 0) return;
    HANDLE file = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                              OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    DWORD written = 0;
    WriteFile(file, line, static_cast<DWORD>(length), &written, nullptr);
    CloseHandle(file);
}

unsigned long env_number(const wchar_t* name, unsigned long fallback)
{
    wchar_t text[32];
    const DWORD size = GetEnvironmentVariableW(name, text, 32);
    if (!size || size >= 32) return fallback;
    return std::wcstoul(text, nullptr, 0);
}

// Distinct non-null results; the fixture never dereferences them.
std::uintptr_t next_reservation = 0x7f0000000000;
}

extern "C" HRESULT WINAPI configure_vm_record(unsigned char* config)
{
    char line[96];
    int length = std::snprintf(line, sizeof(line), "D3D12ConfigureVirtualMemoryX ");
    if (!config) {
        length += std::snprintf(line + length, sizeof(line) - length, "null\n");
    } else {
        for (int i = 0; i < 32; ++i) {
            length += std::snprintf(line + length, sizeof(line) - length, "%02x", config[i]);
        }
        line[length++] = '\n';
    }
    append(line, length);
    if (config) {
        const unsigned long write = env_number(L"CONFIGURE_VM_WRITE", 0);
        if (write == 1) config[31] ^= 1;
        if (write == 2) config[32] ^= 1;
    }
    return static_cast<HRESULT>(env_number(L"CONFIGURE_VM_HRESULT", S_OK));
}

extern "C" void* WINAPI xmem_virtual_alloc_record(
    void* address, SIZE_T size, DWORD type, ULONG64 fourth, DWORD fifth)
{
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "XMemVirtualAlloc %llx %llx %lx %llx %lx\n",
        static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(address)),
        static_cast<unsigned long long>(size), static_cast<unsigned long>(type),
        static_cast<unsigned long long>(fourth), static_cast<unsigned long>(fifth));
    append(line, length);
    next_reservation += 0x1000000000;
    return reinterpret_cast<void*>(next_reservation);
}
