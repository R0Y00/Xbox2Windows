#include <windows.h>

#include <cstdint>
#include <cstdio>

namespace {

bool reserved_at(const void* base, SIZE_T bytes)
{
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(base, &info, sizeof(info)) == sizeof(info) &&
        info.State == MEM_RESERVE && info.BaseAddress == base &&
        info.AllocationBase == base && info.RegionSize == bytes) return true;
    std::fprintf(stderr, "VirtualQuery: expected base %p and %zu reserved bytes, observed base %p, "
                         "allocation base %p, %zu bytes and state 0x%lx\n",
                 base, bytes, info.BaseAddress, info.AllocationBase, info.RegionSize, info.State);
    return false;
}

bool free_at(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && info.State == MEM_FREE;
}

bool check(bool ok, const char* operation)
{
    if (!ok) std::fprintf(stderr, "%s failed (last Win32 error %lu)\n", operation, GetLastError());
    return ok;
}

} // namespace

int main()
{
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    const SIZE_T unit = system.dwAllocationGranularity;
    const HANDLE process = GetCurrentProcess();
    auto* left = static_cast<std::uint8_t*>(VirtualAlloc2(
        process, nullptr, 5 * unit, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
    if (!check(left != nullptr, "Reserve five allocation units")) return 1;
    auto* middle = left + 2 * unit;
    auto* right = middle + unit;
    if (!check(reserved_at(left, 5 * unit),
               "Observe a single original reservation") ||
        !check(VirtualFree(middle, unit, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
               "Split one interior placeholder")) return 1;

    const auto partition_intact = [&] {
        return reserved_at(left, 2 * unit) && reserved_at(middle, unit) &&
               reserved_at(right, 2 * unit);
    };
    if (!check(partition_intact(), "Observe left, middle and right placeholders")) return 1;

    // The requested span starts in the left placeholder and ends in the middle.
    // Its endpoints are aligned, but it is not contained in one placeholder.
    if (!check(!VirtualFree(left + unit, 2 * unit, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
               "Reject a split crossing a placeholder boundary") ||
        !check(partition_intact(), "Keep all three partitions after a refused split")) return 1;

    const HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                              0, static_cast<DWORD>(unit), nullptr);
    if (!check(section != nullptr, "Create a pagefile-backed section")) return 1;
    void* view = MapViewOfFile3(section, process, middle, 0, unit, MEM_REPLACE_PLACEHOLDER,
                                PAGE_READWRITE, nullptr, 0);
    CloseHandle(section);
    if (!check(view == middle, "Map the exact middle placeholder")) return 1;
    static_cast<volatile std::uint8_t*>(view)[0] = 0x5a;
    if (!check(static_cast<volatile const std::uint8_t*>(view)[0] == 0x5a &&
               reserved_at(left, 2 * unit) && reserved_at(right, 2 * unit),
               "Read the mapped byte and keep both neighbors reserved") ||
        !check(UnmapViewOfFile2(process, view, MEM_PRESERVE_PLACEHOLDER), "Restore the middle placeholder") ||
        !check(partition_intact(), "Keep all three partitions after unmapping")) return 1;

    if (!check(VirtualFree(middle, 0, MEM_RELEASE), "Release only the middle") ||
        !check(free_at(middle) && reserved_at(left, 2 * unit) && reserved_at(right, 2 * unit),
               "Keep both neighbors after releasing the middle") ||
        !check(VirtualFree(left, 0, MEM_RELEASE), "Release only the left neighbor") ||
        !check(free_at(left) && free_at(middle) && reserved_at(right, 2 * unit),
               "Keep the right neighbor after releasing the left") ||
        !check(VirtualFree(right, 0, MEM_RELEASE) && free_at(right), "Release the right neighbor")) return 1;

    std::printf("Windows placeholder partition passed (allocation granularity %zu bytes)\n", unit);
    return 0;
}
