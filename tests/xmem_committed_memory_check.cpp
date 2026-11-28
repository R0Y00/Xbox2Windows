#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>

namespace {
struct ReleaseRegion {
    void operator()(BYTE* base) const { VirtualFree(base, 0, MEM_RELEASE); }
};

struct CommitResult {
    BYTE* address;
    SIZE_T bytes;
    DWORD error;
};

// Test-only CPU backing policy. Keep the guest base alignment independent of the host commit size.
CommitResult commit_pages(BYTE* base, SIZE_T bytes, SIZE_T page, DWORD protection)
{
    if (!bytes || protection != PAGE_READWRITE) return {nullptr, 0, ERROR_INVALID_PARAMETER};
    const SIZE_T tail = page - 1;
    if (bytes > std::numeric_limits<SIZE_T>::max() - tail)
        return {nullptr, 0, ERROR_ARITHMETIC_OVERFLOW};
    const SIZE_T rounded = (bytes + tail) & ~tail;
    auto* address = static_cast<BYTE*>(VirtualAlloc(base, rounded, MEM_COMMIT, protection));
    if (!address) return {nullptr, 0, GetLastError()};
    return {address, rounded, ERROR_SUCCESS};
}

bool check_region(const BYTE* address, const BYTE* allocation, DWORD state, SIZE_T bytes)
{
    MEMORY_BASIC_INFORMATION region{};
    if (VirtualQuery(address, &region, sizeof(region)) == sizeof(region) &&
        region.BaseAddress == address && region.AllocationBase == allocation &&
        region.State == state && region.RegionSize == bytes &&
        (state != MEM_COMMIT || (region.Protect == PAGE_READWRITE && region.Type == MEM_PRIVATE))) {
        return true;
    }
    std::fprintf(stderr, "region mismatch: state=%#lx expected=%#lx bytes=%zu expected=%zu protect=%#lx\n",
                 region.State, state, region.RegionSize, bytes, region.Protect);
    return false;
}

BYTE written_byte(SIZE_T offset, SIZE_T page)
{
    if (offset >= page - 8 && offset < page + 8) return static_cast<BYTE>(0x40 + offset - (page - 8));
    return offset == 2 * page - 1 ? 0x9b : 0;
}

bool check_bytes(const volatile BYTE* base, SIZE_T page, bool written)
{
    for (SIZE_T i = 0; i < 2 * page; ++i) {
        const BYTE expected = written ? written_byte(i, page) : 0;
        const BYTE actual = base[i];
        if (actual != expected) {
            std::fprintf(stderr, "byte %zu: read %#x, expected %#x\n", i, unsigned(actual), unsigned(expected));
            return false;
        }
    }
    return true;
}

bool check_refusals(BYTE* base, SIZE_T page)
{
    const struct {
        const char* name;
        BYTE* target;
        SIZE_T bytes;
        DWORD protection;
        DWORD error;
    } cases[] = {
        {"rounding overflow", base, std::numeric_limits<SIZE_T>::max(), PAGE_READWRITE, ERROR_ARITHMETIC_OVERFLOW},
        {"unsupported protection", base + 2 * page, page, PAGE_READONLY, ERROR_INVALID_PARAMETER},
        // The fourth page lies beyond this reservation. Commit must fail without changing the first three.
        {"host overcommit", base, 4 * page, PAGE_READWRITE, ERROR_INVALID_ADDRESS},
    };
    for (const auto& test : cases) {
        const auto result = commit_pages(test.target, test.bytes, page, test.protection);
        if (result.address || result.bytes || result.error != test.error) {
            std::fprintf(stderr, "%s: returned %p, %zu bytes, error %lu, expected %lu\n",
                         test.name, static_cast<void*>(result.address), result.bytes, result.error, test.error);
            return false;
        }
        if (!check_region(base, base, MEM_COMMIT, 2 * page) ||
            !check_region(base + 2 * page, base, MEM_RESERVE, page) ||
            !check_bytes(base, page, true)) {
            std::fprintf(stderr, "%s changed the existing reservation or data\n", test.name);
            return false;
        }
    }
    std::printf("refused overflow=%lu protection=%lu host_error=%lu with existing pages unchanged\n",
                cases[0].error, cases[1].error, cases[2].error);
    return true;
}

bool check_alignment(SIZE_T alignment, SIZE_T page)
{
    MEM_ADDRESS_REQUIREMENTS requirements{};
    requirements.Alignment = alignment;
    MEM_EXTENDED_PARAMETER parameter{};
    parameter.Type = MemExtendedParameterAddressRequirements;
    parameter.Pointer = &requirements;
    std::unique_ptr<BYTE, ReleaseRegion> reservation(static_cast<BYTE*>(
        VirtualAlloc2(GetCurrentProcess(), nullptr, 3 * page, MEM_RESERVE, PAGE_NOACCESS, &parameter, 1)));
    auto* base = reservation.get();
    if (!base) {
        std::fprintf(stderr, "aligned reserve failed (%lu)\n", GetLastError());
        return false;
    }
    if (reinterpret_cast<std::uintptr_t>(base) % alignment) {
        std::fprintf(stderr, "reserve did not return %zu-byte alignment\n", alignment);
        return false;
    }
    if (!check_region(base, base, MEM_RESERVE, 3 * page)) return false;

    const SIZE_T requested = page + 5;
    const auto commit = commit_pages(base, requested, page, PAGE_READWRITE);
    if (commit.address != base || commit.bytes != 2 * page || commit.error != ERROR_SUCCESS) {
        std::fprintf(stderr, "commit returned %p, %zu bytes, error %lu\n",
                     static_cast<void*>(commit.address), commit.bytes, commit.error);
        return false;
    }
    if (!check_region(base, base, MEM_COMMIT, 2 * page) ||
        !check_region(base + 2 * page, base, MEM_RESERVE, page) ||
        !check_bytes(base, page, false)) return false;

    volatile BYTE* observed = base;
    for (SIZE_T i = page - 8; i < page + 8; ++i) observed[i] = written_byte(i, page);
    observed[commit.bytes - 1] = written_byte(commit.bytes - 1, page);
    if (!check_bytes(base, page, true) || !check_refusals(base, page)) return false;

    if (!VirtualFree(base, 0, MEM_RELEASE)) {
        std::fprintf(stderr, "release failed (%lu)\n", GetLastError());
        return false;
    }
    reservation.release();
    MEMORY_BASIC_INFORMATION released{};
    if (VirtualQuery(base, &released, sizeof(released)) != sizeof(released) || released.State != MEM_FREE) {
        std::fputs("released range is not free\n", stderr);
        return false;
    }
    std::printf("alignment=%zu requested=%zu committed=%zu zero_fill=%zu cross_page=16 released=1\n",
                alignment, requested, commit.bytes, commit.bytes);
    return true;
}
}

int main()
{
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    constexpr SIZE_T kib = 1024;
    return check_alignment(64 * kib, info.dwPageSize) && check_alignment(2048 * kib, info.dwPageSize) ? 0 : 1;
}
