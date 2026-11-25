// Lazy commit through the main module's imports (#1199). Source-built check of the two import wrappers that
// X2W_XMEM_LAZY_COMMIT=1 and X2W_XMEM_LAZY_STARTUP=1 install in the main module when xmem.dll loads
// (src/xmem_mapped_pages.cpp). The Python checks cannot reach them: there the main module is python.exe.
//
// - ReadFile: this program's own ReadFile import reads a file into a mapped physical page that was never
//   touched. A kernel-mode write into an uncommitted page fails with ERROR_NOACCESS, so the read succeeds only
//   if the wrapper commits the buffer first.
// - VirtualFree: this program releases a lazy startup block through its own VirtualFree import, reserves the
//   same addresses again, and touches them. The touch must fault: a block still on the lazy list would be
//   committed by the fault handler as if it were the old block.
//
// Usage: xmem_lazy_imports_check <path to xmem.dll>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdio>
#include <cstring>

namespace {
using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*, SIZE_T*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, SIZE_T, SIZE_T*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(SIZE_T, SIZE_T*);

constexpr SIZE_T page = SIZE_T{64} << 10;
constexpr SIZE_T pages = 4;
constexpr SIZE_T block_bytes = SIZE_T{64} << 20;

int failures = 0;

void check(bool ok, const char* name)
{
    std::printf("%s: %s\n", ok ? "ok" : "check_failed", name);
    if (!ok) ++failures;
}

// True when reading the byte faults; kept free of C++ objects for __try.
bool read_faults(volatile const unsigned char* address)
{
    __try {
        (void)*address;
        return false;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return true;
    }
}

DWORD state(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) ? info.State : 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("usage: xmem_lazy_imports_check <xmem.dll>\n");
        return 2;
    }
    // xmem.dll reads the switches and wraps this program's imports when it loads.
    SetEnvironmentVariableA("X2W_XMEM_LAZY_COMMIT", "1");
    SetEnvironmentVariableA("X2W_XMEM_LAZY_STARTUP", "1");
    const HMODULE xmem = LoadLibraryA(argv[1]);
    if (!xmem) {
        std::printf("check_failed: load (error %lu)\n", GetLastError());
        return 1;
    }
    const auto virtual_alloc = reinterpret_cast<XMemVirtualAllocFn>(GetProcAddress(xmem, "XMemVirtualAlloc"));
    const auto allocate = reinterpret_cast<XMemAllocatePhysicalPagesFn>(GetProcAddress(xmem, "XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(GetProcAddress(xmem, "XMemMapPhysicalPages"));
    const auto release = reinterpret_cast<XMemFreePhysicalPagesFn>(GetProcAddress(xmem, "XMemFreePhysicalPages"));
    if (!virtual_alloc || !allocate || !map || !release) {
        std::printf("check_failed: exports\n");
        return 1;
    }

    // ReadFile into an untouched mapped page, through this program's import.
    void* const reservation = virtual_alloc(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE);
    SIZE_T count = pages, tokens[pages] = {};
    const bool mapped = reservation && allocate(0x20400000, &count, tokens) && count == pages &&
                        map(reservation, pages, tokens) == reservation;
    check(mapped, "map");
    if (mapped) {
        auto* const target = static_cast<unsigned char*>(reservation) + page;
        check(state(target) == MEM_RESERVE, "page_left_lazy");
        char data[4096];
        for (size_t i = 0; i < sizeof(data); ++i) data[i] = static_cast<char>(i * 7);
        char path[MAX_PATH], name[MAX_PATH];
        GetTempPathA(MAX_PATH, path);
        GetTempFileNameA(path, "x2w", 0, name);
        HANDLE file = CreateFileA(name, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
        DWORD written = 0;
        WriteFile(file, data, sizeof(data), &written, nullptr);
        CloseHandle(file);
        file = CreateFileA(name, GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        DWORD read = 0;
        const BOOL ok = file != INVALID_HANDLE_VALUE && ReadFile(file, target, sizeof(data), &read, nullptr);
        const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);  // FILE_FLAG_DELETE_ON_CLOSE removes the file
        else
            DeleteFileA(name);
        if (!ok) std::printf("  ReadFile failed with error %lu\n", error);
        check(ok && read == sizeof(data) && std::memcmp(target, data, sizeof(data)) == 0, "readfile_into_lazy_page");
        map(reservation, pages, nullptr);
        release(pages, tokens);
    }
    if (reservation) VirtualFree(reservation, 0, MEM_RELEASE);

    // A released lazy startup block leaves the list: the same addresses, reserved again, still fault.
    void* const block = virtual_alloc(nullptr, block_bytes, 0x20403000, 0x181, PAGE_READWRITE);
    check(block && state(block) == MEM_RESERVE, "block_left_lazy");
    if (block) {
        check(VirtualFree(block, 0, MEM_RELEASE) != FALSE, "virtualfree_release");
        void* const again = VirtualAlloc(block, block_bytes, MEM_RESERVE, PAGE_READWRITE);
        check(again == block, "reserve_same_addresses");
        if (again == block) {
            check(read_faults(static_cast<volatile const unsigned char*>(again) + page), "released_block_not_committed");
            check(state(again) == MEM_RESERVE, "released_block_still_reserved");
            VirtualFree(again, 0, MEM_RELEASE);
        }
    }
    std::printf(failures ? "FAIL (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
