#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>

#include "era_physical.h"
#include "xmem_low.h"
#include "xmem_census.h"
#include "xmem_gpu_pool.h"

#include <cstring>
#include "startup_stubs/local_log.h"

#include <vector>
#include "xmem_slab_slots.h"
#include "xmem_section_lease.h"
#include "xmem_page_batch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "startup_stubs/local_log.h"
#include "startup_stubs/profile_zone.h"

#include <vector>

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <unordered_map>
#include <utility>

namespace {
struct Backing {
    HANDLE section;
    ULONG_PTR views = 0;
    ULONG64 offset = 0;  // the page is one 64 KiB slot of a shared section (see section_for)
    std::size_t slab = 0;  // index of that section among the slabs
    bool committed = true;  // false while a lazily mapped slot is untouched (X2W_XMEM_LAZY_COMMIT)
};

using Slot = std::pair<HANDLE, ULONG64>;
struct SlotPin { unsigned references{}; bool retired{}; };
struct SectionLease {
    UINT version;
    SIZE_T first_byte;
    SIZE_T length;
    std::vector<Slot> slots;
};

// Keep grant transitions and section views serialized. A grant can outlive all its
// views so its bytes survive unmapping. Its backing is released before reuse.
struct PhysicalMemory {
    std::mutex mutex;
    std::map<ULONG_PTR, Backing> backings;
    std::map<ULONG_PTR, ULONG_PTR> views;
    std::map<ULONG_PTR, SIZE_T> reservations;
    // A no-access graphics reservation (protection 0x801) made right after a
    // read-write one of the same size is that memory's GPU view on the console. Pages mapped into
    // the CPU view are mapped at the same offset into the GPU view too (the same section).
    ULONG_PTR last_cpu_base = 0;
    SIZE_T last_cpu_bytes = 0;
    ULONG_PTR mirror_cpu = 0, mirror_gpu = 0;
    SIZE_T mirror_bytes = 0;
    std::set<ULONG_PTR> explicit_mirror_pages;
    std::map<ULONG_PTR, SIZE_T> runs;  // OS views: base -> bytes
    std::vector<HANDLE> slabs;
    ULONG64 slab_used = 0;
    std::set<std::pair<HANDLE, ULONG64>> free_slots;  // ordered, so that pages reused together are adjacent slots again
    std::set<HANDLE> file_slabs;
    ULONG64 migrations = 0, migration_failures = 0;
    std::map<HANDLE, unsigned char*> pool_views;
    unsigned char* newest_pool_view = nullptr;  // pool_views of slabs.back(), or nullptr when it is not a pool slab
    ULONG64 pool_slabs = 0;  // not counted against the slab budget: already charged by the holder
    // A grant may be freed after unmapping while a submitted GPU operation
    // still uses the section. Keep those slots out of the reusable pool (free_slots).
    std::map<Slot, SlotPin> pins;
    std::map<ULONG64, SectionLease> leases;
    ULONG64 next_lease = 1;

    ~PhysicalMemory()
    {
        for (const auto& run : runs)
            UnmapViewOfFile(reinterpret_cast<void*>(run.first));
        for (const HANDLE slab : slabs)
            CloseHandle(slab);
    }
};

PhysicalMemory memory;
constexpr SIZE_T page_bytes = era_physical::page_bytes;

volatile LONG64 startup_block_bytes = 0, startup_block_calls = 0;   // commit_startup_block
volatile LONG64 permissive_commit_bytes = 0, permissive_commit_calls = 0;  // local_virtual_alloc commits
volatile LONG slab_refusals = 0;

// X2W_XMEM_SLAB_BUDGET_MIB=<n>: no new slab once the slabs would pass n MiB; the page mapping or
// page write that needed it fails (ERROR_COMMITMENT_LIMIT), as on a host without memory, but
// inside the title instead of exhausting the host. 0 or unset: no budget.
ULONG64 slab_budget_bytes() noexcept
{
    static const ULONG64 value = [] {
        char text[24]{};
        const DWORD n = GetEnvironmentVariableA("X2W_XMEM_SLAB_BUDGET_MIB", text, sizeof(text));
        return n && n < sizeof(text) ? std::strtoull(text, nullptr, 10) << 20 : 0ull;
    }();
    return value;
}

// CPU and GPU views still name the same grant.
bool mirrored_range(ULONG_PTR address, SIZE_T length)
{
    if (!length || !memory.mirror_gpu || address < memory.mirror_gpu || address > UINT64_MAX - length)
        return false;
    const auto offset = address - memory.mirror_gpu;
    return offset < memory.mirror_bytes && length <= memory.mirror_bytes - offset;
}

const Backing* mirrored_backing(ULONG_PTR page)
{
    const auto mirror = memory.views.find(page);
    const auto cpu = memory.views.find(memory.mirror_cpu + (page - memory.mirror_gpu));
    if (mirror == memory.views.end() || (!memory.explicit_mirror_pages.count(page) &&
        (cpu == memory.views.end() || cpu->second != mirror->second)))
        return nullptr;
    const auto backing = memory.backings.find(mirror->second);
    return backing != memory.backings.end() && backing->second.committed ? &backing->second : nullptr;
}

// Caller holds memory.mutex.
void unpin_slot(const Slot& slot)
{
    const auto found = memory.pins.find(slot);
    if (--found->second.references) return;
    if (found->second.retired) memory.free_slots.insert(slot);
    memory.pins.erase(found);
}

BOOL physical_result(era_physical::Status status) noexcept
{
    switch (status) {
    case era_physical::Status::ok:
        SetLastError(ERROR_SUCCESS);
        return TRUE;
    case era_physical::Status::exhausted:
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return FALSE;
    default:
        SetLastError(ERROR_INVALID_PARAMETER);
        return FALSE;
    }
}

constexpr ULONG64 slab_bytes = 4096 * ULONG64{page_bytes};

const wchar_t* slab_dir() noexcept
{
    static wchar_t dir[MAX_PATH]{};
    static const bool usable = [] {
        const DWORD n = GetEnvironmentVariableW(L"X2W_XMEM_SLAB_DIR", dir, MAX_PATH);
        if (!n || n >= MAX_PATH - 40) return false;
        const bool absolute = n >= 3 && dir[1] == L':' && (dir[2] == L'\\' || dir[2] == L'/');
        wchar_t cwd[MAX_PATH]{};
        GetCurrentDirectoryW(MAX_PATH, cwd);
        const bool same_drive = absolute && (cwd[0] | 0x20) == (dir[0] | 0x20) && cwd[1] == L':';
        const DWORD attributes = absolute ? GetFileAttributesW(dir) : INVALID_FILE_ATTRIBUTES;
        const bool ok = absolute && !same_drive && attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
        x2w_local::logf("x2w-xmem slab directory %ls: %s", dir,
                        ok ? "file-backed slabs (no commit charge; GPU slabs move to the pagefile)"
                           : "refused (must be an existing absolute directory off the current drive); pagefile slabs");
        return ok;
    }();
    return usable ? dir : nullptr;
}

ULONG64 pagefile_slabs() noexcept { return memory.slabs.size() - memory.file_slabs.size() - memory.pool_slabs; }

HANDLE pagefile_slab() noexcept
{
    const ULONG64 budget = slab_budget_bytes();
    if (budget && (pagefile_slabs() + 1) * slab_bytes > budget) {
        if (x2w_local::first_of(InterlockedIncrement(&slab_refusals), 4))
            x2w_local::logf("x2w-xmem slab budget reached (X2W_XMEM_SLAB_BUDGET_MIB=%llu, %llu pagefile slabs of 256 MiB): refused",
                            static_cast<unsigned long long>(budget >> 20), static_cast<unsigned long long>(pagefile_slabs()));
        SetLastError(ERROR_COMMITMENT_LIMIT);
        return nullptr;
    }
    return CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(slab_bytes >> 32),
                              static_cast<DWORD>(slab_bytes), nullptr);
}

HANDLE gpu_slab(bool* pooled) noexcept
{
    *pooled = false;
    const ULONG64 budget = slab_budget_bytes();
    if (!budget || (pagefile_slabs() + 1) * slab_bytes <= budget)
        if (const HANDLE section = x2w_gpu_pool::take(slab_bytes)) {
            *pooled = true;
            return section;
        }
    return pagefile_slab();
}

HANDLE pool_slab(unsigned char** view) noexcept
{
    static bool exhausted = false;
    if (exhausted) return nullptr;
    const HANDLE section = x2w_gpu_pool::take(slab_bytes);
    void* mapped = section ? MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 0) : nullptr;
    if (!mapped) {
        if (section) CloseHandle(section);
        exhausted = true;
        x2w_local::logf("x2w-xmem GPU slab pool (all): no further pool section after %llu (error %lu): fresh %s slabs from "
                        "now on%s", static_cast<unsigned long long>(memory.pool_slabs), GetLastError(),
                        slab_dir() ? "file-backed" : "pagefile",
                        slab_dir() ? " (moved for the GPU as without the pool)" : " (under the slab budget)");
        return nullptr;
    }
    *view = static_cast<unsigned char*>(mapped);
    return section;
}

// A new slab: from the pool with X2W_XMEM_GPU_POOL=all, else file-backed when X2W_XMEM_SLAB_DIR allows it, else from
// the pagefile. *file says whether it is file-backed, *pool_view is the pool section's view (or nullptr).
HANDLE new_slab(bool* file, unsigned char** pool_view) noexcept
{
    *file = false;
    *pool_view = nullptr;
    if (x2w_gpu_pool::mode() == x2w_gpu_pool::Mode::all)
        if (const HANDLE slab = pool_slab(pool_view)) return slab;
    if (const wchar_t* dir = slab_dir()) {
        static LONG serial;
        wchar_t path[MAX_PATH];
        swprintf_s(path, L"%ls\\x2w-slab-%lu-%ld.tmp", dir, GetCurrentProcessId(), InterlockedIncrement(&serial));
        const HANDLE backing = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                           FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (backing != INVALID_HANDLE_VALUE) {
            DWORD returned = 0;
            DeviceIoControl(backing, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr);
            const HANDLE slab = CreateFileMappingW(backing, nullptr, PAGE_READWRITE, static_cast<DWORD>(slab_bytes >> 32),
                                                   static_cast<DWORD>(slab_bytes), nullptr);
            const DWORD error = GetLastError();
            CloseHandle(backing);  // the section keeps the file; it is deleted when the section goes
            if (slab) {
                *file = true;
                return slab;
            }
            static LONG failed;
            if (x2w_local::first_calls(failed, 4))
                x2w_local::logf("x2w-xmem file-backed slab %ls failed (error %lu): pagefile slab instead", path, error);
        } else {
            static LONG failed;
            if (x2w_local::first_calls(failed, 4))
                x2w_local::logf("x2w-xmem slab file %ls could not be created (error %lu): pagefile slab instead", path, GetLastError());
        }
    }
    return pagefile_slab();
}

// Pages are slots of 256 MiB pagefile-backed sections rather than one section each: one handle and
// one view per page is what makes large batches slow. A slab is created with SEC_RESERVE, so it costs
// no commit until a slot is used; a slot is committed here, when its page is first mapped or written,
// and stays committed after it is freed so that reuse cannot fail for lack of commit. A freed slot is
// zeroed when it is reused, as a fresh section was. Fails with the system error (for instance
// ERROR_COMMITMENT_LIMIT) when the slot cannot be committed, leaving no state changed.
bool commit_slot(HANDLE slab, ULONG64 offset, bool zero)
{
#ifdef X2W_XMEM_TEST_CONTROLS
    // Refuse the first commit in a dedicated test process without consuming system commit.
    static bool refuse_commit = x2w_local::env_on("X2W_XMEM_TEST_COMMIT_FAILURE");
    if (refuse_commit) {
        refuse_commit = false;
        SetLastError(ERROR_COMMITMENT_LIMIT);
        return false;
    }
#endif
    void* view = MapViewOfFile(slab, FILE_MAP_WRITE, static_cast<DWORD>(offset >> 32), static_cast<DWORD>(offset), page_bytes);
    if (!view) return false;
    bool ok = VirtualAlloc(view, page_bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    const DWORD error = ok ? ERROR_SUCCESS : GetLastError();
    if (ok && zero) std::memset(view, 0, page_bytes);  // a reused slot starts zero-filled
    UnmapViewOfFile(view);
    if (!ok) SetLastError(error);
    return ok;
}

// Experimental (X2W_XMEM_LAZY_COMMIT=1, #1199): a fresh slot is not committed when its page is mapped. The first
// CPU access faults and lazy_commit_fault commits that slot. Writers that do not fault -- kernel-mode I/O into
// mapped pages, GPU placement, stand-in writes behind a commit-state check -- call x2w_xmem_commit_range first.
// Reused slots and x2w_xmem_write_physical still commit before use.
bool lazy_commit_requested()
{
    static const bool on = x2w_local::env_on("X2W_XMEM_LAZY_COMMIT");
    return on;
}

// Set when xmem.dll loads, once the fault handler is installed. Without the handler a lazy page could never be
// committed, so the switch then acts as off: pages and blocks are committed up front.
bool lazy_handler_installed = false;

bool lazy_commit()
{
    return lazy_handler_installed && lazy_commit_requested();
}

// Experimental (X2W_XMEM_LAZY_STARTUP=1 with X2W_XMEM_LAZY_COMMIT=1, #1199): the large startup blocks
// XMemVirtualAlloc commits for the title (commit_startup_block, 32 MiB and up) are only reserved. The fault
// handler commits them 256 KiB at a time on first touch. x2w_xmem_commit_range commits the chunks behind a range
// for writers that cannot fault, and x2w_xmem_commit_block commits a whole block before the GPU uses it.
bool lazy_startup()
{
    static const bool on = x2w_local::env_on("X2W_XMEM_LAZY_STARTUP");
    return on && lazy_commit();
}

constexpr SIZE_T lazy_chunk = SIZE_T{256} << 10;
constexpr SIZE_T lazy_block_min = SIZE_T{32} << 20;

// The reserved startup blocks that are still committed lazily: base -> bytes.
struct LazyBlocks {
    SRWLOCK lock = SRWLOCK_INIT;
    std::map<ULONG_PTR, SIZE_T> blocks;
};
LazyBlocks lazy_blocks;
volatile LONG lazy_block_releases = 0;  // lazy blocks the title has released (keys the fault handler's retry)

// The lazy block holding address, as base and size; false when there is none. The caller holds lazy_blocks.lock
// until it has committed what it found, so that a release (virtual_free_lazy) cannot run in between and leave it
// committing addresses that no longer belong to the block.
bool lazy_block_at(ULONG_PTR address, ULONG_PTR& base, SIZE_T& bytes)
{
    auto it = lazy_blocks.blocks.upper_bound(address);
    const bool inside = it != lazy_blocks.blocks.begin() && (--it, address - it->first < it->second);
    if (inside) {
        base = it->first;
        bytes = it->second;
    }
    return inside;
}

// Commits the whole chunks of a lazy block that [from, to) touches; false with the system error if that fails.
bool commit_block_chunks(ULONG_PTR base, SIZE_T bytes, ULONG_PTR from, ULONG_PTR to)
{
    const SIZE_T first = (from - base) & ~(lazy_chunk - 1);
    SIZE_T last = to - base < bytes ? to - base : bytes;  // offsets in the block; to is exclusive
    last = (last + lazy_chunk - 1) & ~(lazy_chunk - 1);
    if (last > bytes) last = bytes;
    return VirtualAlloc(reinterpret_cast<void*>(base + first), last - first, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

// Commits a lazily mapped slot under memory.mutex; false with the system error if the commit fails.
bool commit_backing(Backing& backing)
{
    if (backing.committed) return true;
    if (!commit_slot(backing.section, backing.offset, false)) return false;
    backing.committed = true;
    return true;
}

// The backing behind a mapped address (CPU view, or its GPU-view mirror), under memory.mutex.
Backing* backing_at(ULONG_PTR address)
{
    const auto view = memory.views.find(address & ~(ULONG_PTR{page_bytes} - 1));
    if (view == memory.views.end()) return nullptr;
    const auto backing = memory.backings.find(view->second);
    return backing == memory.backings.end() ? nullptr : &backing->second;
}

bool valid_lease_range(UINT version, ULONG_PTR address, SIZE_T length)
{
    if (version == x2w_xmem_section_lease_version) return mirrored_range(address, length);
    return version == x2w_xmem_single_page_lease_version && length &&
           length <= page_bytes - (address % page_bytes) && address <= UINTPTR_MAX - (length - 1);
}

const Backing* lease_backing(UINT version, ULONG_PTR page)
{
    if (version == x2w_xmem_section_lease_version || mirrored_range(page, page_bytes))
        return mirrored_backing(page);
    const auto* backing = backing_at(page);
    return backing && backing->committed ? backing : nullptr;
}

LONG CALLBACK lazy_commit_fault(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR address = record->ExceptionInformation[1];
    if (lazy_startup()) {
        ULONG_PTR base = 0;
        SIZE_T bytes = 0;
        MEMORY_BASIC_INFORMATION info{};
        AcquireSRWLockShared(&lazy_blocks.lock);
        const bool lazy = lazy_block_at(address, base, bytes) &&
                          VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) && info.State == MEM_RESERVE;
        const bool committed = lazy && commit_block_chunks(base, bytes, address, address + 1);
        ReleaseSRWLockShared(&lazy_blocks.lock);
        if (lazy) return committed ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
        // Another thread committed this chunk (or the whole block, which then leaves the list) between the fault
        // and the query: committed read-write memory cannot fault, so retry once. The retry is keyed on the chunk
        // and on the release count, so a fault that stays is passed on, while a new block later placed at the same
        // addresses can still be raced once (separate from the slot path's retry below).
        thread_local ULONG_PTR retried_chunk = 0;
        thread_local LONG retried_releases = 0;
        const ULONG_PTR chunk = address & ~(ULONG_PTR{lazy_chunk} - 1);
        const LONG releases = lazy_block_releases;
        if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) && info.Type == MEM_PRIVATE &&
            info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE &&
            !(retried_chunk == chunk && retried_releases == releases)) {
            retried_chunk = chunk;
            retried_releases = releases;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    // A thread that raced another one to the same slot finds it committed: retry once, then pass the fault on.
    thread_local ULONG_PTR retried = 0;
    const ULONG_PTR page = address & ~(ULONG_PTR{page_bytes} - 1);
    const std::lock_guard guard(memory.mutex);
    Backing* backing = backing_at(address);
    if (!backing) return EXCEPTION_CONTINUE_SEARCH;
    if (backing->committed) {
        if (retried == page) return EXCEPTION_CONTINUE_SEARCH;
        retried = page;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (!commit_backing(*backing)) return EXCEPTION_CONTINUE_SEARCH;
    retried = 0;
    return EXCEPTION_CONTINUE_EXECUTION;
}

HANDLE section_for(ULONG_PTR page, bool /*need_commit*/ = true)
{
    const auto existing = memory.backings.find(page);
    if (existing != memory.backings.end())
        return existing->second.section;
    if (!memory.free_slots.empty()) {
        const auto [section, offset] = *memory.free_slots.begin();
        if (const auto pool = memory.pool_views.find(section); pool != memory.pool_views.end()) {
            std::memset(pool->second + offset, 0, page_bytes);  // through the pool slab's own view
        } else {
            void* view = MapViewOfFile(section, FILE_MAP_WRITE, static_cast<DWORD>(offset >> 32), static_cast<DWORD>(offset), page_bytes);
            if (!view) return nullptr;
            std::memset(view, 0, page_bytes);  // a reused slot starts zero-filled, as a fresh section did
            UnmapViewOfFile(view);
        }
        memory.backings.emplace(page, Backing{section, 0, offset});
        memory.free_slots.erase(memory.free_slots.begin());
        return section;
    }
    if (memory.slabs.empty() || memory.slab_used == slab_bytes) {
        bool file = false;
        unsigned char* pool_view = nullptr;
        const HANDLE slab = new_slab(&file, &pool_view);
        if (!slab) return nullptr;
        try {
            memory.slabs.push_back(slab);
            if (file) memory.file_slabs.insert(slab);
            if (pool_view) memory.pool_views.emplace(slab, pool_view), ++memory.pool_slabs;
        } catch (const std::bad_alloc&) {
            if (pool_view) UnmapViewOfFile(pool_view);
            CloseHandle(slab);
            throw;
        }
        memory.newest_pool_view = pool_view;
        memory.slab_used = 0;
    }
    const HANDLE section = memory.slabs.back();
    if (memory.newest_pool_view) std::memset(memory.newest_pool_view + memory.slab_used, 0, page_bytes);
    memory.backings.emplace(page, Backing{section, 0, memory.slab_used});
    memory.slab_used += page_bytes;
    return section;
}

auto owned_reservation(ULONG_PTR address)
{
    auto it = memory.reservations.upper_bound(address);
    if (it == memory.reservations.begin()) return memory.reservations.end();
    --it;
    return address - it->first < it->second ? it : memory.reservations.end();
}

struct MapErrors {
    DWORD map = 0, split = 0, retry = 0;
    bool split_tried = false;
};
thread_local MapErrors map_errors;

// One view of bytes at address: replace the placeholder, or split it out of a bigger recorded placeholder first.
bool map_placeholder(ULONG_PTR address, HANDLE section, ULONG64 section_offset, SIZE_T bytes)
{
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(section, GetCurrentProcess(), target, section_offset, bytes, MEM_REPLACE_PLACEHOLDER,
                              PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    map_errors = MapErrors{};
    if (map()) return true;
    map_errors.map = GetLastError();
    if (owned_reservation(address) == memory.reservations.end()) {
        SetLastError(map_errors.map);
        return false;
    }
    map_errors.split_tried = true;
    if (!VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
        map_errors.split = GetLastError();
        return false;
    }
    if (map()) return true;
    map_errors.retry = GetLastError();
    return false;
}

bool map_view(ULONG_PTR address, const Backing& backing)
{
    return map_placeholder(address, backing.section, backing.offset, page_bytes);
}

// Maps bytes of one section at address as a single view; falls back to one view per page when the
// range is not inside a single placeholder.
bool map_run(ULONG_PTR address, const Backing& backing, SIZE_T bytes,
             std::vector<std::pair<ULONG_PTR, SIZE_T>>& made, ULONG_PTR& failed_address)
{
    if (map_placeholder(address, backing.section, backing.offset, bytes)) {
        memory.runs[address] = bytes;
        made.emplace_back(address, bytes);
        return true;
    }
    failed_address = address;
    if (bytes == page_bytes) return false;
    for (SIZE_T done = 0; done < bytes; done += page_bytes) {
        Backing one = backing;
        one.offset += done;
        if (!map_view(address + done, one)) {
            failed_address = address + done;
            return false;
        }
        memory.runs[address + done] = page_bytes;
        made.emplace_back(address + done, page_bytes);
    }
    return true;
}

bool unmap_view(ULONG_PTR address)
{
    return UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(address),
                           MEM_PRESERVE_PLACEHOLDER) != FALSE;
}

template <class Pages>
void coalesce_unused(const Pages& batch)
{
    auto previous = memory.reservations.end();
    for (const auto& entry : batch) {
        const auto reservation = owned_reservation(entry.first);
        if (reservation == memory.reservations.end() || reservation == previous) continue;
        previous = reservation;
        const auto [base, size] = *reservation;
        const auto view = memory.views.lower_bound(base);
        if (view == memory.views.end() || view->first - base >= size)
            VirtualFree(reinterpret_cast<void*>(base), size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
    }
}

// Unmapping a page inside a run unmaps the whole run and maps its other pages again (map_pages).
// A thread that touches one of those pages in between faults. The break records the address
// ranges of the runs it takes apart and a generation that is odd while it is in progress.
struct RemapWindow {
    SRWLOCK lock = SRWLOCK_INIT;
    std::vector<std::pair<ULONG_PTR, SIZE_T>> ranges;  // the runs of the break in progress, or of the last one
    ULONG64 generation = 0;
};
RemapWindow remap_window;

void begin_break(const std::map<ULONG_PTR, SIZE_T>& broken)
{
    std::vector<std::pair<ULONG_PTR, SIZE_T>> ranges(broken.begin(), broken.end());
    AcquireSRWLockExclusive(&remap_window.lock);
    remap_window.ranges.swap(ranges);
    ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
}
void end_break()
{
    AcquireSRWLockExclusive(&remap_window.lock);
    ++remap_window.generation;
    ReleaseSRWLockExclusive(&remap_window.lock);
}

struct EndBreak { ~EndBreak() { end_break(); } };

// Retries an access violation only when all of this holds: the address is inside a run that the
// break in progress, or the last finished one, took apart; after that break has finished (the
// mutex) the page is recorded and covered by a view; and this thread has not already retried this
// page for this break. Everything else is passed on, so a fault that is not the transient one is
// reported after at most one retry.
thread_local bool mapping_thread = false;
LONG CALLBACK retry_remapped_page(EXCEPTION_POINTERS* e)
{
    const EXCEPTION_RECORD* record = e->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || mapping_thread)
        return EXCEPTION_CONTINUE_SEARCH;
    const ULONG_PTR address = record->ExceptionInformation[1];
    const ULONG_PTR page = address & ~(ULONG_PTR{page_bytes} - 1);
    bool inside = false;
    AcquireSRWLockShared(&remap_window.lock);
    const ULONG64 seen = remap_window.generation;
    for (const auto& [base, size] : remap_window.ranges)
        inside |= address - base < size;
    ReleaseSRWLockShared(&remap_window.lock);
    if (!inside || seen == 0) return EXCEPTION_CONTINUE_SEARCH;
    const ULONG64 break_id = seen | 1;  // the break in progress, or the one that just finished
    thread_local ULONG_PTR retried_page = 0;
    thread_local ULONG64 retried_break = 0;
    if (retried_page == page && retried_break == break_id) return EXCEPTION_CONTINUE_SEARCH;
    bool mapped = false;
    {
        const std::lock_guard guard(memory.mutex);  // held by the mapping thread until its break is over
        auto run = memory.runs.upper_bound(page);
        mapped = memory.views.count(page) && run != memory.runs.begin() && page - std::prev(run)->first < std::prev(run)->second;
    }
    if (!mapped) return EXCEPTION_CONTINUE_SEARCH;
    retried_page = page;
    retried_break = break_id;
    return EXCEPTION_CONTINUE_EXECUTION;
}
struct MappingThread {
    MappingThread() { mapping_thread = true; }
    ~MappingThread() { mapping_thread = false; }
};

struct MapFailure;
ULONG_PTR map_recorded(ULONG_PTR base, SIZE_T size, bool fail, MapFailure* failure = nullptr);  // defined below (upstream MapFailure)

bool retry_handler_ready() noexcept
{
    static const bool ready = [] {
        HMODULE pinned = nullptr;
        return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                  reinterpret_cast<LPCWSTR>(&retry_remapped_page), &pinned) &&
               AddVectoredExceptionHandler(1, retry_remapped_page) != nullptr;
    }();
    return ready;
}

HANDLE migrate_slab(HANDLE old_slab)
{
    const ULONG64 began = GetTickCount64();
    if (!retry_handler_ready()) {
        ++memory.migrations, ++memory.migration_failures;
        x2w_local::logf("x2w-xmem slab move refused: the fault handler could not be set up");
        return nullptr;
    }
    for (const auto& [slot, pin] : memory.pins)
        if (slot.first == old_slab) {
            ++memory.migrations, ++memory.migration_failures;
            x2w_local::logf("x2w-xmem slab move refused: a section lease pins one of its slots");
            return nullptr;
        }
    std::map<ULONG_PTR, SIZE_T> broken;
    for (const auto& [address, page] : memory.views) {
        if (memory.backings.at(page).section != old_slab) continue;
        auto run = memory.runs.upper_bound(address);
        if (run == memory.runs.begin()) continue;
        --run;
        if (address - run->first < run->second) broken.insert(*run);
    }
    bool pooled = false;
    const HANDLE moved = gpu_slab(&pooled);
    if (!moved) {
        ++memory.migration_failures;
        x2w_local::logf("x2w-xmem slab move refused: no pagefile slab (error %lu)", GetLastError());
        return nullptr;
    }
    const MappingThread marker;
    begin_break(broken);
    DWORD error = ERROR_SUCCESS;
    std::vector<std::pair<ULONG_PTR, SIZE_T>> unmapped;
    for (const auto& [base, size] : broken) {
        if (!UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER)) {
            error = GetLastError();
            break;
        }
        memory.runs.erase(base);
        unmapped.emplace_back(base, size);
    }
    bool copied = false;
    if (error == ERROR_SUCCESS) {
        void* from = MapViewOfFile(old_slab, FILE_MAP_READ, 0, 0, 0);
        void* to = MapViewOfFile(moved, FILE_MAP_WRITE, 0, 0, 0);
        if (from && to) {
            // Only the used part of the newest slab holds pages.
            const bool newest = !memory.slabs.empty() && memory.slabs.back() == old_slab;
            std::memcpy(to, from, newest ? memory.slab_used : slab_bytes);
            // A pool section holds an earlier run's bytes: the unused tail must read as zero, as a fresh one.
            if (pooled && newest) std::memset(static_cast<char*>(to) + memory.slab_used, 0, slab_bytes - memory.slab_used);
            copied = true;
        } else {
            error = GetLastError();
        }
        if (from) UnmapViewOfFile(from);
        if (to) UnmapViewOfFile(to);
    }
    if (copied) {
        for (auto& [page, backing] : memory.backings)
            if (backing.section == old_slab) backing.section = moved;
        std::vector<std::pair<HANDLE, ULONG64>> slots;
        for (auto it = memory.free_slots.begin(); it != memory.free_slots.end();) {
            if (it->first == old_slab) {
                slots.emplace_back(moved, it->second);
                it = memory.free_slots.erase(it);
            } else {
                ++it;
            }
        }
        memory.free_slots.insert(slots.begin(), slots.end());
        std::replace(memory.slabs.begin(), memory.slabs.end(), old_slab, moved);
        memory.file_slabs.erase(old_slab);
    }
    ULONG_PTR lost = 0;
    for (const auto& [base, size] : unmapped) lost += map_recorded(base, size, false);
    end_break();
    ++memory.migrations;
    if (!copied) {
        ++memory.migration_failures;
        CloseHandle(moved);
        x2w_local::logf("x2w-xmem slab move failed (error %lu); %llu page(s) could not be mapped again", error,
                        static_cast<unsigned long long>(lost));
        return nullptr;
    }
    CloseHandle(old_slab);
    x2w_local::logf("x2w-xmem slab moved to the pagefile for the GPU (#%llu): %llu view(s) in %llu ms, %llu page(s) lost, "
                    "%llu file-backed slab(s) left", static_cast<unsigned long long>(memory.migrations),
                    static_cast<unsigned long long>(unmapped.size()), static_cast<unsigned long long>(GetTickCount64() - began),
                    static_cast<unsigned long long>(lost), static_cast<unsigned long long>(memory.file_slabs.size()));
    return moved;
}

// Shared views are opt-in: X2W_XMEM_SHARED_VIEWS=1, read once by the first mapping call. Without
// it every page has a view of its own, as before: unmapping a page never takes a neighbour away,
// no handler is registered, and the module unloads like any DLL.
//
// With it, the first mapping call registers the handler and pins the module. The handler is
// process-wide and Windows offers no way to remove one that also waits for every dispatch already
// under way, so it is never removed; the pin keeps xmem.dll, the handler's code and this state in
// place until the process ends, also across a host's FreeLibrary and LoadLibrary. Nothing is
// waited for or torn down at DLL detach. If the handler cannot be registered or the module cannot
// be pinned, the opt-in is refused and pages get a view each.
struct FaultRetry {
    SRWLOCK lock = SRWLOCK_INIT;
    bool tried = false;
    bool guaranteed = false;  // handler registered and module pinned: shared views may be created

    bool ready()
    {
        AcquireSRWLockExclusive(&lock);
        if (!tried) {
            tried = true;
#ifdef X2W_XMEM_TEST_CONTROLS
            // Test build only: shared views without the handler, to show that a reader then faults.
            if (x2w_local::env_on("X2W_XMEM_NO_FAULT_RETRY")) {
                guaranteed = true;
                ReleaseSRWLockExclusive(&lock);
                return true;
            }
#endif
            HMODULE pinned = nullptr;
            if (!x2w_local::env_on("X2W_XMEM_SHARED_VIEWS")) {
                // The default: one view per page, nothing process-wide, unloadable.
            } else if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                   reinterpret_cast<LPCWSTR>(&retry_remapped_page), &pinned) &&
                AddVectoredExceptionHandler(1, retry_remapped_page)) {
                guaranteed = true;
                x2w_local::logf("x2w-xmem shared views are on (X2W_XMEM_SHARED_VIEWS): xmem.dll stays loaded until the process ends");
            } else
                x2w_local::logf("x2w-xmem shared views are off: the fault handler could not be set up (error %lu)", GetLastError());
        }
        const bool result = guaranteed;
        ReleaseSRWLockExclusive(&lock);
        return result;
    }
};
FaultRetry fault_retry;

// Test build only (X2W_XMEM_TEST_CONTROLS; tests/xmem_page_runs_check.py): X2W_XMEM_TEST_FAULT=<stage>:<skip> makes one step of
// the next unmap fail as if Windows had refused it, after skipping <skip> such steps. Stage 1 is
// unmapping a run, stage 2 mapping surviving pages again, stage 3 both the unmap and the restore.
#ifdef X2W_XMEM_TEST_CONTROLS
struct TestFault {
    unsigned stage = 0, skip = 0;
    TestFault()
    {
        char text[32];
        if (GetEnvironmentVariableA("X2W_XMEM_TEST_FAULT", text, sizeof(text)) && text[0] >= '1' && text[0] <= '3' && text[1] == ':') {
            stage = static_cast<unsigned>(text[0] - '0');
            skip = static_cast<unsigned>(std::strtoul(text + 2, nullptr, 10));
        }
    }
    bool hit(unsigned at)
    {
        if (stage != at && !(stage == 3 && at == 1)) return false;
        if (skip) {
            --skip;
            return false;
        }
        if (stage != 3) stage = 0;      // one failure
        else if (at == 1) return true;  // stage 3: the unmap now, the restore next
        return true;
    }
};
#else
struct TestFault {
    unsigned stage = 0;
    bool hit(unsigned) { return false; }
};
#endif

// Test build only: X2W_XMEM_TEST_HOLD_WINDOW=<name> keeps an unmap in the window where the run is
// unmapped and not yet mapped again. It sets the event "<name>.unmapped", then waits up to
// five seconds for "<name>.release", so that a reader thread can reach the window every time.
#ifdef X2W_XMEM_TEST_CONTROLS
void hold_window()
{
    char name[96];
    const DWORD length = GetEnvironmentVariableA("X2W_XMEM_TEST_HOLD_WINDOW", name, sizeof(name) - 16);
    if (!length || length >= sizeof(name) - 16) return;
    char event_name[128];
    std::snprintf(event_name, sizeof(event_name), "%s.unmapped", name);
    HANDLE unmapped = CreateEventA(nullptr, TRUE, FALSE, event_name);
    std::snprintf(event_name, sizeof(event_name), "%s.release", name);
    HANDLE release = CreateEventA(nullptr, TRUE, FALSE, event_name);
    if (unmapped && release) {
        SetEvent(unmapped);
        if (WaitForSingleObject(release, 5000) != WAIT_OBJECT_0)
            x2w_local::logf("x2w-xmem test hold: nobody released the unmapped window within 5 s");
    }
    if (unmapped) CloseHandle(unmapped);
    if (release) CloseHandle(release);
}
#else
void hold_window() {}
#endif

// Drops a page that has no view any more from the registry, so that the registry never names a
// page Windows does not show.
void forget_page(ULONG_PTR address)
{
    const auto live = memory.views.find(address);
    if (live == memory.views.end()) return;
    --memory.backings.at(live->second).views;
    memory.views.erase(live);
}

// Maps every recorded page of [base, base + size) that has no view: neighbours on adjacent slots
// together, one page at a time when a group is refused. A page that cannot be mapped is forgotten
// and counted. fail is true when the test control refuses this range.
struct MapFailure {
    const char* stage = nullptr;
    ULONG_PTR address = 0;
    ULONG_PTR page = 0;
    DWORD error = ERROR_SUCCESS;
    MEMORY_BASIC_INFORMATION region{};
    SIZE_T view_bytes = page_bytes;

    void record(const char* step, ULONG_PTR at, ULONG_PTR id, DWORD code, SIZE_T bytes = page_bytes) noexcept
    {
        if (stage) return;
        stage = step;
        address = at;
        page = id;
        error = code;
        view_bytes = bytes;
        VirtualQuery(reinterpret_cast<void*>(at), &region, sizeof(region));
        SetLastError(code);
    }
};

ULONG_PTR map_recorded(ULONG_PTR base, SIZE_T size, bool fail, MapFailure* failure)
{
    ULONG_PTR lost = 0;
    ULONG_PTR at = base;
    while (at < base + size) {
        const auto live = memory.views.find(at);
        if (live == memory.views.end()) {
            at += page_bytes;
            continue;
        }
        const Backing head = memory.backings.at(live->second);
        SIZE_T bytes = page_bytes;
        for (; at + bytes < base + size; bytes += page_bytes) {
            const auto next = memory.views.find(at + bytes);
            if (next == memory.views.end()) break;
            const Backing& b = memory.backings.at(next->second);
            if (b.section != head.section || b.offset != head.offset + bytes) break;
        }
        std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
        ULONG_PTR failed_address;
        if (fail || !map_run(at, head, bytes, made, failed_address)) {
            // map_run may have mapped some pages one at a time before it stopped; keep those.
            for (SIZE_T done = 0; done < bytes; done += page_bytes) {
                if (memory.runs.count(at + done)) continue;
                Backing one = head;
                one.offset += done;
                if (!fail && map_view(at + done, one)) {
                    memory.runs[at + done] = page_bytes;
                } else {
                    if (failure)
                        failure->record("restore-view", at + done, memory.views.at(at + done),
                                        fail ? ERROR_INVALID_ADDRESS : GetLastError());
                    forget_page(at + done);
                    ++lost;
                }
            }
        }
        at += bytes;
    }
    return lost;
}

struct PageEvent {
    ULONG64 tick = 0, serial = 0;
    ULONG_PTR page = 0;
    DWORD thread = 0;
    unsigned char kind = 0;  // 1 map, 2 unmap
};
std::unordered_map<ULONG_PTR, PageEvent> page_events;
ULONG64 page_event_serial = 0;  // one per successful map or unmap step
ULONG64 remap_gaps = 0;        // addresses unmapped and mapped again in one call (P -> Q): a VirtualQuery from another
                               // thread inside that call sees them uncommitted

void note_events(const std::map<ULONG_PTR, ULONG_PTR>& batch, unsigned char kind) noexcept
{
    const ULONG64 tick = GetTickCount64(), serial = ++page_event_serial;
    const DWORD thread = GetCurrentThreadId();
    try {
        for (const auto& [address, page] : batch) page_events[address] = PageEvent{tick, serial, page, thread, kind};
    } catch (const std::bad_alloc&) {
    }
}

void log_refusal(const char* what, bool mapping, ULONG_PTR count, ULONG_PTR index, ULONG_PTR address, ULONG_PTR page,
                 const char* check, DWORD error, ULONG_PTR mapped_before, bool windows) noexcept
{
    MEMORY_BASIC_INFORMATION info{};
    const bool queried = address && VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) == sizeof(info);
    const auto live = memory.views.find(address);
    const auto reservation = owned_reservation(address);
    char windows_text[160] = "";
    if (windows)
        std::snprintf(windows_text, sizeof(windows_text), " MapViewOfFile3 err %lu, split %s%lu, map again err %lu;",
                      map_errors.map, map_errors.split_tried ? "err " : "not tried ", map_errors.split, map_errors.retry);
    x2w_local::logf("x2w-xmem REFUSED %s (%s) thread %lu: entry %llu of %llu, address %p page 0x%llx (token 0x%llx, %s): %s, "
                    "error %lu;%s %llu page(s) of the call mapped before it were taken back | Windows at the address: base %p "
                    "allocation %p size 0x%llx state 0x%lx type 0x%lx protect 0x%lx%s | registry: %s 0x%llx, reservation %p+0x%llx | "
                    "slabs %llu (%llu pool), %llu pages live, %llu free slots, budget %llu MiB",
                    what, mapping ? "map" : "unmap", GetCurrentThreadId(), static_cast<unsigned long long>(index),
                    static_cast<unsigned long long>(count), reinterpret_cast<void*>(address), static_cast<unsigned long long>(page),
                    static_cast<unsigned long long>(page << 4), page && era_physical::owns(page) ? "granted" : "NOT granted",
                    check, static_cast<unsigned long>(error), windows_text, static_cast<unsigned long long>(mapped_before),
                    queried ? info.BaseAddress : nullptr, queried ? info.AllocationBase : nullptr,
                    static_cast<unsigned long long>(queried ? info.RegionSize : 0), queried ? info.State : 0ul,
                    queried ? info.Type : 0ul, queried ? info.Protect : 0ul, queried ? "" : " (VirtualQuery failed)",
                    live != memory.views.end() ? "view of page" : "no view", static_cast<unsigned long long>(
                        live != memory.views.end() ? live->second : 0),
                    reservation != memory.reservations.end() ? reinterpret_cast<void*>(reservation->first) : nullptr,
                    static_cast<unsigned long long>(reservation != memory.reservations.end() ? reservation->second : 0),
                    static_cast<unsigned long long>(memory.slabs.size()), static_cast<unsigned long long>(memory.pool_slabs),
                    static_cast<unsigned long long>(memory.backings.size()),
                    static_cast<unsigned long long>(memory.free_slots.size()),
                    static_cast<unsigned long long>(slab_budget_bytes() >> 20));
    SetLastError(error);
}

// Destinations must be caller-owned, page-sized Windows placeholders. Live views
// are owned by this shim until it unmaps them. The caller must not replace them.
// Called with memory.mutex held and a MappingThread marker on this thread (map_pages, local_apply).
template <typename AddressAt>
BOOL map_pages_locked(ULONG_PTR count, const ULONG_PTR* tokens, const AddressAt& address_at, bool share_views,
                      const char* what, MapFailure* failure = nullptr) noexcept
{
    if (!count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    try {
        std::map<ULONG_PTR, ULONG_PTR> batch;
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto address = address_at(i);
            const auto live = memory.views.find(address);
            const char* check = !address || address % page_bytes ? "address is zero or not 64 KiB aligned"
                                : tokens && live != memory.views.end() ? "a view is already recorded at the address"
                                : tokens && !era_physical::owns(tokens[i]) ? "the page is not granted (era_physical::owns)"
                                : !tokens && live == memory.views.end() ? "no view is recorded at the address"
                                                                         : nullptr;
            if (!check && !batch.emplace(address, tokens ? tokens[i] : live->second).second)
                check = "the address is named twice in the call";
            if (check) {
                log_refusal(what, tokens != nullptr, count, i, address,
                            tokens ? tokens[i] : live != memory.views.end() ? live->second : 0, check,
                            ERROR_INVALID_PARAMETER, 0, false);
                if (failure) {
                    const ULONG_PTR page = tokens ? tokens[i] : live != memory.views.end() ? live->second : 0;
                    const char* stage = !address || address % page_bytes ? "invalid-address"
                                        : tokens && live != memory.views.end() ? "already-mapped"
                                        : tokens && !era_physical::owns(page) ? "unowned-page"
                                        : !tokens && live == memory.views.end() ? "unmapped-address"
                                                                                 : "duplicate-address";
                    failure->record(stage, address, page, ERROR_INVALID_PARAMETER);
                }
                return physical_result(era_physical::Status::invalid);
            }
        }
        // Adjacent pages that are adjacent slots of one section share a single view (a "run"): a
        // multi-gigabyte batch then takes a few views instead of one per 64 KiB page.
        if (tokens) {
            ULONG_PTR index = 0;
            for (const auto& [address, page] : batch) {
                if (!section_for(page, false)) {
                    const DWORD section_error = GetLastError();
                    log_refusal(what, true, count, index, address, page, "no backing section for the page (section_for)",
                                section_error, 0, false);
                    if (failure) failure->record("section-backing", address, page, section_error);
                    return FALSE;
                }
                ++index;
            }
            std::vector<std::pair<ULONG_PTR, SIZE_T>> made;
            auto first = batch.begin();
            while (first != batch.end()) {
                const Backing& head = memory.backings.at(first->second);
                SIZE_T bytes = page_bytes;
                auto last = std::next(first);
                for (; share_views && last != batch.end(); ++last, bytes += page_bytes) {
                    const Backing& b = memory.backings.at(last->second);
                    if (last->first != first->first + bytes || b.section != head.section || b.offset != head.offset + bytes) break;
                }
                ULONG_PTR failed_address;
                if (!map_run(first->first, head, bytes, made, failed_address)) {
                    const DWORD error = GetLastError();
                    // The page map_run stopped at: the first page of the run that has no view.
                    ULONG_PTR at = first->first;
                    while (memory.runs.count(at) && at < first->first + bytes) at += page_bytes;
                    const auto failed = batch.find(at);
                    log_refusal(what, true, count, static_cast<ULONG_PTR>(std::distance(batch.begin(), failed)), at,
                                failed != batch.end() ? failed->second : 0,
                                "Windows refused the view (sorted entry; MapViewOfFile3 / placeholder split)", error,
                                made.size(), true);
                    if (failure) failure->record("map-view", failed_address, batch.at(failed_address), error);
                    for (const auto& [base, size] : made) {
                        if (!UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(base), MEM_PRESERVE_PLACEHOLDER))
                            x2w_local::logf("x2w-xmem REFUSED %s: taking back the view %p+0x%llx failed (error %lu): a view the "
                                            "registry does not know stays there", what, reinterpret_cast<void*>(base),
                                            static_cast<unsigned long long>(size), GetLastError());
                        memory.runs.erase(base);
                    }
                    coalesce_unused(batch);
                    SetLastError(error);
                    return FALSE;
                }
                first = last;
            }
            for (const auto& [address, page] : batch)
                ++memory.backings.at(page).views;
            note_events(batch, 1);
            memory.views.merge(batch);
        } else {
            // Unmap every run that holds a page of the batch, then map the surviving pages of those
            // runs again. The registry (views, runs) matches what Windows shows on every way out:
            // - an unmap is refused: the runs already unmapped are mapped again whole, nothing is
            //   removed, and the call fails with that error;
            // - a surviving page cannot be mapped again: it is forgotten, the batch is unmapped, and
            //   the call fails.
            std::map<ULONG_PTR, SIZE_T> broken;
            for (const auto& [address, page] : batch) {
                auto run = memory.runs.upper_bound(address);
                if (run == memory.runs.begin()) continue;
                --run;
                if (address - run->first < run->second) broken.insert(*run);
            }
            TestFault test;
            begin_break(broken);
            const EndBreak end;
            DWORD error = ERROR_SUCCESS;
            std::vector<std::pair<ULONG_PTR, SIZE_T>> unmapped;
            for (const auto& [base, size] : broken) {
                const bool inject_failure = test.hit(1);
                if (inject_failure || !unmap_view(base)) {
                    error = inject_failure ? ERROR_INVALID_ADDRESS : GetLastError();
                    if (failure) failure->record("unmap-view", base, memory.views.at(base), error, size);
                    break;
                }
                memory.runs.erase(base);
                unmapped.emplace_back(base, size);
            }
            const bool refused = error != ERROR_SUCCESS;
            if (!refused) {
                note_events(batch, 2);
                for (const auto& [address, page] : batch) {
                    --memory.backings.at(page).views;
                    memory.views.erase(address);
                }
            }
            if (!refused && !unmapped.empty()) hold_window();
            ULONG_PTR lost = 0;
            for (const auto& [base, size] : unmapped)
                lost += map_recorded(base, size, refused ? test.stage == 3 && test.hit(3) : test.hit(2), failure);
            if (!refused) coalesce_unused(batch);
            if (refused || lost) {
                x2w_local::logf("x2w-xmem unmap of %llu page(s) failed (error %lu): %s; %llu surviving page(s) could not be mapped "
                                "again and were forgotten", static_cast<unsigned long long>(batch.size()),
                                static_cast<unsigned long>(error), refused ? "an unmap was refused and no page of the batch was removed" : "the pages were unmapped",
                                static_cast<unsigned long long>(lost));
                SetLastError(refused ? error : ERROR_NOT_ENOUGH_MEMORY);
                return FALSE;
            }
        }
    } catch (const std::bad_alloc&) {
        return physical_result(era_physical::Status::exhausted);
    }
    return physical_result(era_physical::Status::ok);
}

template <typename AddressAt>
BOOL map_pages(ULONG_PTR count, const ULONG_PTR* tokens, const AddressAt& address_at, const char* what,
               MapFailure* failure = nullptr) noexcept
{
    X2W_ZONE("xmem map_pages");
    if (!count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    const bool share_views = fault_retry.ready();
    const MappingThread marker;
    const std::lock_guard guard(memory.mutex);
    return map_pages_locked(count, tokens, address_at, share_views, what, failure);
}
}

// ---------------------------------------------------------------------------------------------
// Experimental (docs/startup-runtime.md). With X2W_XMEM_PERMISSIVE=1 the exports
// below accept the call shapes of a startup run on top of the section-backed pages above.
// Observations behind it:
// - Page numbers: the caller rebuilds them as (21-bit entry << 4), so it expects 4 KiB frame numbers
//   with the low four bits clear. Callers get page << 4 and pass it back.
// - XMemAllocatePhysicalPages flags other than MEM_64K_PAGES occur (0x280) and are served the same.
// - XMemVirtualAlloc: commits (type 0x20003000/0x20403000) become plain commits; reservations become
//   placeholders that pages can be mapped into; a "replace placeholder" call (0x20406000) on a piece
//   the caller split from its own placeholder keeps the placeholder and records it for mapping.
// - Scatter mapping passes page 0 to unmap and may name pages that are already mapped elsewhere
//   (a second view; the sections give real sharing).
namespace {
bool local_permissive() noexcept { return x2w_local::env_on("X2W_XMEM_PERMISSIVE"); }

void record_reservation(void* base, SIZE_T bytes)
{
    const std::lock_guard guard(memory.mutex);
    memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(base), bytes);
}

void* local_virtual_alloc(void* address, SIZE_T bytes, DWORD type, ULONGLONG attributes, DWORD protection)
{
    const DWORD low = protection & 0xFF;
    const bool accessible = low == PAGE_READONLY || low == PAGE_READWRITE || low == PAGE_EXECUTE_READ ||
                            low == PAGE_EXECUTE_READWRITE;
    const DWORD protect = accessible ? low : PAGE_READWRITE;
    void* result = nullptr;
    const char* how = "";
    if ((type & MEM_REPLACE_PLACEHOLDER) && address) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(address, &info, sizeof(info)) && info.State == MEM_RESERVE) {
            record_reservation(address, bytes);
            result = address;
        }
        how = "placeholder kept for page mapping";
    } else if (type & MEM_COMMIT) {
        if (!address) {
            result = x2w_low::alloc(bytes, MEM_COMMIT | MEM_RESERVE, protect);
            how = "commit";
            if (result) InterlockedAdd64(&permissive_commit_bytes, static_cast<LONG64>(bytes)), InterlockedIncrement64(&permissive_commit_calls);
        } else {
            result = VirtualAlloc(address, bytes, MEM_COMMIT, protect);
            if (!result) {
                VirtualFree(address, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                result = VirtualAlloc2(GetCurrentProcess(), address, bytes,
                                       MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, protect, nullptr, 0);
            }
            how = "commit at address";
            if (result) InterlockedAdd64(&permissive_commit_bytes, static_cast<LONG64>(bytes)), InterlockedIncrement64(&permissive_commit_calls);
        }
    } else {
        result = address ? VirtualAlloc2(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
                                         nullptr, 0)
                         : x2w_low::alloc(bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS);
        // A caller that asks for a placeholder splits it itself; other reservations are split here on mapping.
        if (result && !(type & MEM_RESERVE_PLACEHOLDER)) record_reservation(result, bytes);
        how = "placeholder reserve";
    }
    const DWORD error = result ? 0 : GetLastError();
    x2w_local::logf("x2w-xmem XMemVirtualAlloc addr=%p size=0x%llx type=0x%lx attr=0x%llx prot=0x%lx -> %p (%s, err %lu)",
                    address, static_cast<unsigned long long>(bytes), static_cast<unsigned long>(type),
                    static_cast<unsigned long long>(attributes), static_cast<unsigned long>(protection), result, how,
                    static_cast<unsigned long>(error));
    SetLastError(error);
    return result;
}

void log_apply_failure(const char* what, const MapFailure& failure, ULONG_PTR count, const ULONG_PTR* addresses)
{
    // The apply batch is sorted and the last occurrence of each address wins. Report the caller's
    // index, including when the failing OS unmap covers a whole run containing a requested page.
    ULONG_PTR index = count;
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (addresses[i] == failure.address) index = i;
    }
    if (index == count && failure.view_bytes > page_bytes) {
        for (ULONG_PTR i = 0; i < count; ++i) {
            if (addresses[i] - failure.address < failure.view_bytes) {
                index = i;
                break;
            }
        }
    }
    char request_index[24] = "none";
    if (index != count)
        std::snprintf(request_index, sizeof(request_index), "%llu", static_cast<unsigned long long>(index));
    x2w_local::logf("x2w-xmem %s failed stage=%s index=%s address=%p page=0x%llx error=%lu "
                    "region_base=%p allocation_base=%p region_size=0x%llx state=0x%lx protect=0x%lx type=0x%lx",
                    what, failure.stage, request_index, reinterpret_cast<void*>(failure.address),
                    static_cast<unsigned long long>(failure.page), static_cast<unsigned long>(failure.error),
                    failure.region.BaseAddress, failure.region.AllocationBase,
                    static_cast<unsigned long long>(failure.region.RegionSize),
                    static_cast<unsigned long>(failure.region.State), static_cast<unsigned long>(failure.region.Protect),
                    static_cast<unsigned long>(failure.region.Type));
}

void log_failed_ranges(const char* what, const char* operation, const std::vector<ULONG_PTR>& addresses)
{
    // Addresses are sorted by local_apply. A failed map rolls back its completed views, so all
    // of its destinations matter when correlating a later fault, even those preceding the error.
    constexpr std::size_t max_ranges = 2048;
    std::size_t first = 0;
    for (std::size_t range = 0; first < addresses.size() && range < max_ranges; ++range) {
        std::size_t end = first + 1;
        while (end < addresses.size() && addresses[end] - addresses[end - 1] == page_bytes) ++end;
        x2w_local::logf("x2w-xmem %s failed batch %s first=%p last=%p pages=%llu", what, operation,
                        reinterpret_cast<void*>(addresses[first]), reinterpret_cast<void*>(addresses[end - 1]),
                        static_cast<unsigned long long>(end - first));
        first = end;
    }
    if (first < addresses.size())
        x2w_local::logf("x2w-xmem %s failed batch %s truncated omitted_pages=%llu", what, operation,
                        static_cast<unsigned long long>(addresses.size() - first));
}

// Applies (address, page) pairs; page 0 unmaps. An address that already shows the requested page is
// left alone; one that shows another page is unmapped first.
BOOL local_apply_locked(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages,
                        const char* what, bool share_views)
{
    // The pairs in ascending address order, the last pair for a repeated address winning, as an
    // ordered map assigned in call order would give; but one allocation instead of a tree node per
    // page (a single call can carry ~100,000 pages).
    std::vector<std::pair<ULONG_PTR, ULONG_PTR>> want;
    want.reserve(count);
    bool ascending = true;
    for (ULONG_PTR i = 0; i < count; ++i) {
        ascending = ascending && (!i || addresses[i] > addresses[i - 1]);
        want.emplace_back(addresses[i], pages[i]);
    }
    if (!ascending) {
        std::stable_sort(want.begin(), want.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::size_t kept = 0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            if (i + 1 < want.size() && want[i + 1].first == want[i].first) continue;  // a later pair names it again
            want[kept++] = want[i];
        }
        want.resize(kept);
    }
    std::vector<ULONG_PTR> unmap, map_at, map_page;
    // The plan (which addresses to unmap, which to map) and both steps run under ONE hold of the mutex (the
    // caller's, local_apply): a call from another thread naming the same addresses in between would make the plan
    // wrong and the call fail, where the console applies both calls one after the other.
    for (const auto& [address, page] : want) {
        const auto live = memory.views.find(address);
        if (!page) {
            if (live != memory.views.end()) unmap.push_back(address);
            continue;
        }
        if (live != memory.views.end()) {
            if (live->second == page) continue;
            unmap.push_back(address);
            ++remap_gaps;
        }
        map_at.push_back(address);
        map_page.push_back(page);
    }
    BOOL ok = TRUE;
    DWORD error = 0;
    MapFailure failure;
    if (!unmap.empty() && !map_pages_locked(unmap.size(), nullptr, [&](ULONG_PTR i) { return unmap[i]; },
                                            share_views, what, &failure)) {
        ok = FALSE;
        error = GetLastError();
    }
    if (ok && !map_at.empty() &&
        !map_pages_locked(map_at.size(), map_page.data(), [&](ULONG_PTR i) { return map_at[i]; },
                          share_views, what, &failure)) {
        ok = FALSE;
        error = GetLastError();
    }
    if (failure.stage) log_apply_failure(what, failure, count, addresses);
    if (!ok && x2w_local::env_on("X2W_XMEM_TRACE_FAILED_BATCH")) {
        log_failed_ranges(what, "map", map_at);
        log_failed_ranges(what, "unmap", unmap);
    }
    static LONG calls;
    if (!ok || x2w_local::first_calls(calls, 16))
        x2w_local::logf("x2w-xmem %s count=0x%llx first=%p map=%llu unmap=%llu -> %d (err %lu) thread %lu, %s, lowest %p "
                        "highest %p%s", what, static_cast<unsigned long long>(count),
                        count ? reinterpret_cast<void*>(addresses[0]) : nullptr, static_cast<unsigned long long>(map_at.size()),
                        static_cast<unsigned long long>(unmap.size()), ok, static_cast<unsigned long>(error),
                        GetCurrentThreadId(), ascending ? "ascending" : "unsorted",
                        want.empty() ? nullptr : reinterpret_cast<void*>(want.front().first),
                        want.empty() ? nullptr : reinterpret_cast<void*>(want.back().first),
                        ok ? "" : " (the x2w-xmem REFUSED line above names the entry and the cause; no page of the refused step is mapped)");
    SetLastError(error);
    return ok;
}

// Applies the same (address, page) pairs again at the GPU view for those addresses in the CPU view.
void local_mirror_locked(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages,
                         bool share_views)
{
    const auto cpu = memory.mirror_cpu, gpu = memory.mirror_gpu;
    const auto bytes = memory.mirror_bytes;
    if (!cpu) return;
    std::vector<ULONG_PTR> at, page;
    for (ULONG_PTR i = 0; i < count; ++i)
        if (addresses[i] >= cpu && addresses[i] - cpu < bytes) {
            const auto address = addresses[i] - cpu + gpu;
            if (memory.explicit_mirror_pages.count(address)) continue;
            at.push_back(address);
            page.push_back(pages[i]);
        }
    if (!at.empty()) local_apply_locked(at.size(), at.data(), page.data(), "GPU view mirror", share_views);
}

BOOL local_apply(ULONG_PTR count, const ULONG_PTR* addresses, const ULONG_PTR* pages, const char* what)
{
    const bool share_views = fault_retry.ready();
    const MappingThread marker;
    const std::lock_guard guard(memory.mutex);
    if (!local_apply_locked(count, addresses, pages, what, share_views)) return FALSE;
    local_mirror_locked(count, addresses, pages, share_views);
    return TRUE;
}

bool to_pages(ULONG_PTR count, const ULONG_PTR* tokens, std::vector<ULONG_PTR>& pages)
{
    if (count > era_physical::capacity_pages) return false;  // as the strict path: no grant is larger
    pages.resize(count);
    for (ULONG_PTR i = 0; i < count; ++i) {
        if (tokens[i] & 0xF) return false;
        pages[i] = tokens[i] >> 4;
    }
    return true;
}
}  // namespace

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages);
}

extern "C" BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR flags, PULONG_PTR count, PULONG_PTR pages)
{
    x2w_census::start();
    static_assert(sizeof(ULONG_PTR) == sizeof(std::uint64_t));
    if (local_permissive()) {
        const ULONG_PTR requested = count ? *count : 0;
        era_physical::Status status;
        {
            const std::lock_guard guard(memory.mutex);
            status = era_physical::allocate(era_physical::mem_64k_pages, reinterpret_cast<std::uint64_t*>(count),
                                            reinterpret_cast<std::uint64_t*>(pages));
        }
        if (status == era_physical::Status::ok)
            for (ULONG_PTR i = 0; i < *count; ++i) pages[i] <<= 4;
        x2w_local::logf("x2w-xmem XMemAllocatePhysicalPages flags=0x%llx pages=0x%llx -> %d",
                        static_cast<unsigned long long>(flags), static_cast<unsigned long long>(requested),
                        status == era_physical::Status::ok);
        return physical_result(status);
    }
    const std::lock_guard guard(memory.mutex);
    return physical_result(era_physical::allocate(flags, reinterpret_cast<std::uint64_t*>(count),
                                                  reinterpret_cast<std::uint64_t*>(pages)));
}

extern "C" BOOL WINAPI XMemFreePhysicalPages(ULONG_PTR count, PULONG_PTR pages)
{
    if (local_permissive() && pages && count && count <= era_physical::capacity_pages) {
        std::vector<ULONG_PTR> converted;
        if (!to_pages(count, pages, converted)) return physical_result(era_physical::Status::invalid);
        return free_pages(count, converted.data());
    }
    return free_pages(count, pages);
}

namespace {
BOOL free_pages(ULONG_PTR count, PULONG_PTR pages)
{
    if (!pages || !count || count > era_physical::capacity_pages)
        return physical_result(era_physical::Status::invalid);
    const std::lock_guard guard(memory.mutex);
    for (ULONG_PTR i = 0; i < count; ++i) {
        const auto backing = memory.backings.find(pages[i]);
        if (backing != memory.backings.end() && backing->second.views)
            return physical_result(era_physical::Status::invalid);
    }
    const auto status = era_physical::release(count, reinterpret_cast<const std::uint64_t*>(pages));
    if (status == era_physical::Status::ok) {
        for (ULONG_PTR i = 0; i < count; ++i) {
            const auto backing = memory.backings.find(pages[i]);
            if (backing != memory.backings.end()) {
                const Slot slot{backing->second.section, backing->second.offset};
                if (const auto pin = memory.pins.find(slot); pin != memory.pins.end())
                    pin->second.retired = true;
                else
                    memory.free_slots.insert(slot);
                memory.backings.erase(backing);
            }
        }
    }
    return physical_result(status);
}
}  // namespace

namespace {
struct AliasRange {
    ULONG_PTR address;
    X2WPageRange source;
};

// Keep the guarded reads outside the transaction and out of C++ functions that need unwinding.
template <typename T>
bool read_alias_record(const T* array, UINT index, T& record)
{
    __try {
        record = array[index];
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ||
                GetExceptionCode() == EXCEPTION_IN_PAGE_ERROR ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

struct AliasEdit {
    ULONG_PTR old;
    ULONG_PTR desired;
    ULONG_PTR actual;
};

using AliasPlan = std::map<ULONG_PTR, AliasEdit>;
using ViewRuns = std::map<ULONG_PTR, SIZE_T>;

bool alias_placeholders(ULONG_PTR address, SIZE_T bytes)
{
    // The region query exposes the placeholder reservation flag in Flags even though the
    // SDK does not name that bit. VirtualQuery alone cannot distinguish a reserved private
    // allocation from a placeholder. PRESERVE_PLACEHOLDER can silently release that allocation.
    constexpr ULONG placeholder_reservation = 0x100;
    const auto end = address + bytes;
    while (address < end) {
        WIN32_MEMORY_REGION_INFORMATION info{};
        if (!QueryVirtualMemoryInformation(GetCurrentProcess(), reinterpret_cast<void*>(address),
                                           MemoryRegionInfo, &info, sizeof(info), nullptr) ||
            !(info.Flags & placeholder_reservation))
            return false;
        const auto base = reinterpret_cast<ULONG_PTR>(info.AllocationBase);
        if (address - base >= info.RegionSize) return false;
        address += info.RegionSize - (address - base);
    }
    return true;
}

// Both maps have the same node type. The node for each possible run base was reserved before
// unmapping anything, so recording a successful OS call cannot fail for lack of C++ heap memory.
bool install_alias_view(ULONG_PTR address, const Backing& backing, SIZE_T bytes,
                        ViewRuns& nodes, X2WPageBatchStats& stats)
{
    auto* target = reinterpret_cast<void*>(address);
    auto map = [&] {
        return MapViewOfFile3(backing.section, GetCurrentProcess(), target, backing.offset, bytes,
                              MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0) != nullptr;
    };
    if (!map()) {
        const DWORD original_error = GetLastError();
        const auto reservation = owned_reservation(address);
        if (reservation == memory.reservations.end() ||
            bytes > reservation->second - (address - reservation->first) || !alias_placeholders(address, bytes)) {
            SetLastError(original_error);
            return false;
        }
        // A subrange of one placeholder must be split. Several exact placeholders left by old
        // views may be coalesced instead. Both operations stay within the recorded reservation.
        if (!VirtualFree(target, bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) &&
            (bytes == page_bytes || !VirtualFree(target, bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS))) {
            SetLastError(original_error);
            return false;
        }
        if (!map()) return false;
    }
    auto node = nodes.extract(address);
    node.mapped() = bytes;
    memory.runs.insert(std::move(node));
    ++stats.mapCalls;
    return true;
}

DWORD install_alias_plan(AliasPlan& plan, ViewRuns& nodes, bool share_views, bool restoring,
                         std::vector<ULONG_PTR>& made, X2WPageBatchStats& stats)
{
    DWORD error = ERROR_SUCCESS;
    auto token = [&](const AliasEdit& edit) { return restoring ? edit.old : edit.desired; };
    auto it = plan.begin();
    while (it != plan.end()) {
        if (it->second.actual || !token(it->second)) {
            ++it;
            continue;
        }
        const auto address = it->first;
        const auto& head = memory.backings.at(token(it->second));
        auto end = std::next(it);
        SIZE_T bytes = page_bytes;
        for (; share_views && end != plan.end(); ++end, bytes += page_bytes) {
            if (end->first != address + bytes || end->second.actual || !token(end->second)) break;
            const auto& next = memory.backings.at(token(end->second));
            if (next.section != head.section || next.offset != head.offset + bytes) break;
        }
        if (install_alias_view(address, head, bytes, nodes, stats)) {
            made.push_back(address);
            for (auto page = it; page != end; ++page)
                page->second.actual = token(page->second);
        } else if (bytes == page_bytes) {
            error = GetLastError();
            if (!restoring) break;
        } else {
            // A grouped view may cross placeholder boundaries. Retry each page once.
            for (auto page = it; page != end; ++page) {
                if (install_alias_view(page->first, memory.backings.at(token(page->second)),
                                       page_bytes, nodes, stats)) {
                    made.push_back(page->first);
                    page->second.actual = token(page->second);
                } else {
                    error = GetLastError();
                    if (!restoring) return error;
                }
            }
        }
        it = end;
    }
    return error;
}

HRESULT remap_alias_pages(AliasPlan& plan, bool share_views, X2WPageBatchStats& stats)
{
    ViewRuns broken;
    for (const auto& [address, edit] : plan) {
        if (!edit.old) continue;
        const auto run = memory.runs.upper_bound(address);
        if (run == memory.runs.begin() || address - std::prev(run)->first >= std::prev(run)->second)
            return E_UNEXPECTED;
        broken.insert(*std::prev(run));
    }
    for (const auto& [base, size] : broken) {
        // Surviving neighbours of a shared view are part of the transaction as well.
        for (auto view = memory.views.lower_bound(base);
             view != memory.views.end() && view->first - base < size; ++view) {
            const auto page = view->second;
            plan.try_emplace(view->first, AliasEdit{page, page, page});
        }
    }

    ViewRuns run_nodes;
    std::map<ULONG_PTR, ULONG_PTR> view_nodes;
    std::vector<ULONG_PTR> made;
    made.reserve(plan.size());
    for (const auto& [address, edit] : plan) {
        run_nodes.emplace(address, page_bytes);
        view_nodes.emplace(address, 0);
    }
    auto clear_run = [&](ULONG_PTR address, SIZE_T size) {
        ++stats.unmapCalls;
        for (auto it = plan.lower_bound(address); it != plan.end() && it->first - address < size; ++it)
            it->second.actual = 0;
    };

    begin_break(broken);
    const EndBreak end;
    DWORD error = ERROR_SUCCESS;
    for (const auto& [base, size] : broken) {
        if (!unmap_view(base)) {
            error = GetLastError();
            break;
        }
        memory.runs.erase(base);
        clear_run(base, size);
    }
    if (!error)
        error = install_alias_plan(plan, run_nodes, share_views, false, made, stats);
    bool restore_failed = false;
    if (error) {
        for (const auto address : made) {
            if (!unmap_view(address)) {
                restore_failed = true;
                continue;
            }
            auto run = memory.runs.extract(address);
            clear_run(address, run.mapped());
            run_nodes.insert(std::move(run));
        }
        made.clear();
        if (install_alias_plan(plan, run_nodes, share_views, true, made, stats))
            restore_failed = true;
    }
    // Publish the OS state, including any views left over if Windows refused rollback.
    for (const auto& [address, edit] : plan) {
        if (edit.old == edit.actual) continue;
        forget_page(address);
        if (edit.actual) {
            ++memory.backings.at(edit.actual).views;
            auto node = view_nodes.extract(address);
            node.mapped() = edit.actual;
            memory.views.insert(std::move(node));
        }
    }
    if (restore_failed) {
        x2w_local::logf("x2w-xmem page batch rollback incomplete (original error %lu)", error);
        return E_UNEXPECTED;
    }
    if (!error) coalesce_unused(plan);
    return error ? HRESULT_FROM_WIN32(error) : S_OK;
}
}  // namespace

extern "C" HRESULT WINAPI X2WMapPageBatch(UINT count, const X2WPageBatch* batches, UINT64 pool,
                                           UINT poolPages, UINT flags, X2WPageBatchStats* stats)
{
    // Buffers are valid for this call. The descriptor arrays may be supplied by the caller's
    // address space, so take a bounded snapshot before taking the memory lock.
    if (stats) *stats = {};
    constexpr UINT limit = static_cast<UINT>(era_physical::capacity_pages);
    if (flags || count > limit || (count && !batches) ||
        (poolPages && !pool) || pool % page_bytes || pool > UINT64_MAX - UINT64{poolPages} * page_bytes)
        return E_INVALIDARG;
    if (!count) return S_OK;
    try {
        std::vector<AliasRange> ranges;
        UINT total = 0;
        bool ordered_disjoint = true;
        UINT64 previous_end = 0;
        for (UINT i = 0; i < count; ++i) {
            X2WPageBatch batch{};
            if (!read_alias_record(batches, i, batch)) return E_POINTER;
            if (!batch.destination || batch.destination % page_bytes || !batch.ranges ||
                !batch.count || batch.count > limit - total)
                return E_INVALIDARG;
            auto address = batch.destination;
            for (UINT j = 0; j < batch.count; ++j) {
                X2WPageRange range{};
                if (!read_alias_record(batch.ranges, j, range)) return E_POINTER;
                const auto bytes = UINT64{range.count} * page_bytes;
                if (!range.count || range.count > limit - total || address > UINT64_MAX - bytes ||
                    range.kind > 3 || (range.kind <= 1 && range.start >= poolPages) ||
                    (range.kind == 0 && range.count > poolPages - range.start))
                    return E_INVALIDARG;
                ordered_disjoint &= address >= previous_end;
                ranges.push_back({address, range});
                total += range.count;
                address += bytes;
                previous_end = address;
            }
        }

        const bool share_views = fault_retry.ready();
        const MappingThread marker;
        const std::lock_guard guard(memory.mutex);
        constexpr ULONG_PTR keep = UINT64_MAX;
        AliasPlan plan;
        std::set<ULONG_PTR> explicit_pages;
        X2WPageBatchStats result{};
        const auto add_edit = [&](ULONG_PTR address, ULONG_PTR desired) {
            if (desired == keep) {
                ++result.pagesSkipped;
                return;
            }
            const auto live = memory.views.find(address);
            const auto old = live != memory.views.end() ? live->second : 0;
            if (memory.mirror_bytes && address - memory.mirror_gpu < memory.mirror_bytes &&
                !memory.explicit_mirror_pages.count(address))
                explicit_pages.insert(address);
            if (old == desired) {
                ++result.pagesSkipped;
            } else {
                plan.emplace(address, AliasEdit{old, desired, old});
                ++result.pagesChanged;
            }
        };
        std::map<ULONG_PTR, ULONG_PTR> requested;
        for (const auto& [base, range] : ranges) {
            for (UINT i = 0; i < range.count; ++i) {
                const auto address = base + SIZE_T{i} * page_bytes;
                if (owned_reservation(address) == memory.reservations.end()) return E_INVALIDARG;
                ULONG_PTR desired = range.kind == 3 ? keep : 0;
                if (range.kind <= 1) {
                    const auto source = pool + SIZE_T{range.start + (range.kind == 0 ? i : 0)} * page_bytes;
                    const auto view = memory.views.find(source);
                    if (view == memory.views.end()) return E_INVALIDARG;
                    desired = view->second;
                }
                // Ordered, disjoint ranges need no per-page nodes for already satisfied edits.
                if (ordered_disjoint)
                    add_edit(address, desired);
                else if (desired == keep)
                    requested.try_emplace(address, keep);
                else
                    requested.insert_or_assign(address, desired);
            }
        }
        for (const auto& [address, desired] : requested) add_edit(address, desired);
        const HRESULT hr = plan.empty() ? S_OK : remap_alias_pages(plan, share_views, result);
        if (SUCCEEDED(hr)) {
            memory.explicit_mirror_pages.merge(explicit_pages);
            if (stats) *stats = result;
        }
        return hr;
    } catch (const std::bad_alloc&) {
        return E_OUTOFMEMORY;
    }
}

// The view is a dense range of committed slots in one section. A caller can
// map it as a separate view at section_offset and open that address as a heap.
extern "C" BOOL WINAPI x2w_xmem_lease_section(UINT version, ULONG_PTR address, SIZE_T length,
                                              X2WXMemSectionLease* out, SIZE_T bytes)
{
    if (!out || bytes != sizeof(*out)) return FALSE;
    *out = {};
    constexpr SIZE_T max_pages = 64;
    if (length > max_pages * page_bytes) return FALSE;
    const std::lock_guard guard(memory.mutex);
    if (!memory.next_lease || !valid_lease_range(version, address, length)) return FALSE;
    const auto first = address & ~(page_bytes - 1);
    const auto first_byte = address - first;
    const auto count = (first_byte + length + page_bytes - 1) / page_bytes;
    if (count > max_pages) return FALSE;
    std::vector<Slot> selected;
    std::size_t pinned = 0;
    HANDLE section = nullptr, duplicate = nullptr;
    try {
        selected.reserve(count);
        for (SIZE_T i = 0; i < count; ++i) {
            const auto page = first + i * page_bytes;
            const auto* slot = lease_backing(version, page);
            if (!slot) return FALSE;
            if (selected.empty())
                section = slot->section;
            else if (slot->section != section ||
                     slot->offset != selected.front().second + i * page_bytes)
                return FALSE;
            selected.emplace_back(slot->section, slot->offset);
        }
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE,
                             DUPLICATE_SAME_ACCESS))
            return FALSE;
        for (const auto& slot : selected) {
            ++memory.pins[slot].references;
            ++pinned;
        }
        const auto lease = memory.next_lease;
        memory.leases.emplace(lease, SectionLease{version, first_byte, length, selected});
        ++memory.next_lease;
        *out = {duplicate, lease, selected.front().second, count * page_bytes, first_byte};
        return TRUE;
    } catch (const std::bad_alloc&) {
        if (duplicate) CloseHandle(duplicate);
        for (std::size_t i = 0; i < pinned; ++i) unpin_slot(selected[i]);
        return FALSE;
    }
}

extern "C" BOOL WINAPI x2w_xmem_match_section(UINT version, ULONG64 lease, SIZE_T byte_offset,
                                              ULONG_PTR address, SIZE_T length)
{
    if (byte_offset % page_bytes != address % page_bytes) return FALSE;
    const std::lock_guard guard(memory.mutex);
    const auto found = memory.leases.find(lease);
    if (found == memory.leases.end() || found->second.version != version ||
        !valid_lease_range(version, address, length)) return FALSE;
    const auto& pin = found->second;
    if (byte_offset < pin.first_byte || byte_offset - pin.first_byte >= pin.length ||
        length > pin.length - (byte_offset - pin.first_byte)) return FALSE;
    const SIZE_T first_slot = byte_offset / page_bytes;
    const SIZE_T last_slot = (byte_offset + length - 1) / page_bytes;
    const auto first_page = address & ~(page_bytes - 1);
    for (SIZE_T slot = first_slot; slot <= last_slot; ++slot) {
        const auto* backing = lease_backing(version, first_page + (slot - first_slot) * page_bytes);
        if (!backing || pin.slots[slot] != Slot{backing->section, backing->offset}) return FALSE;
    }
    return TRUE;
}

extern "C" BOOL WINAPI x2w_xmem_release_section(UINT version, ULONG64 lease)
{
    const std::lock_guard guard(memory.mutex);
    const auto found = memory.leases.find(lease);
    if (found == memory.leases.end() || found->second.version != version) return FALSE;
    for (const auto& slot : found->second.slots) unpin_slot(slot);
    memory.leases.erase(found);
    return TRUE;
}

// Experimental export: one snapshot of the physical-page accounting for the startup heartbeat
// (#1199). A page gets a 64 KiB slot of a shared slab on first use, by a mapping or by a page-token write, and
// loses it when the grant is released; the number of pages holding a slot is reported so runs can be compared.
extern "C" BOOL WINAPI x2w_xmem_page_stats(ULONGLONG* live_sections)
{
    if (!live_sections) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *live_sections = static_cast<ULONGLONG>(memory.backings.size());
    return TRUE;
}

extern "C" BOOL WINAPI x2w_xmem_memory_stats(X2wXmemStats* stats)
{
    if (!stats) return FALSE;
    const std::lock_guard guard(memory.mutex);
    stats->slabs = memory.slabs.size();
    stats->slab_bytes = memory.slabs.size() * slab_bytes;
    stats->live_pages = memory.backings.size();
    stats->free_slots = memory.free_slots.size();
    stats->slab_budget_bytes = slab_budget_bytes();
    stats->slab_refusals = static_cast<ULONGLONG>(slab_refusals);
    stats->startup_block_bytes = static_cast<ULONGLONG>(startup_block_bytes);
    stats->startup_block_calls = static_cast<ULONGLONG>(startup_block_calls);
    stats->permissive_commit_bytes = static_cast<ULONGLONG>(permissive_commit_bytes);
    stats->permissive_commit_calls = static_cast<ULONGLONG>(permissive_commit_calls);
    stats->reservations = memory.reservations.size();
    stats->file_slabs = memory.file_slabs.size();
    stats->migrations = memory.migrations;
    stats->migration_failures = memory.migration_failures;
    return TRUE;
}

extern "C" SIZE_T WINAPI x2w_xmem_reserved_ranges(ULONG_PTR* pairs, SIZE_T capacity)
{
    const std::lock_guard guard(memory.mutex);
    SIZE_T i = 0;
    for (const auto& [base, bytes] : memory.reservations) {
        if (pairs && i < capacity) {
            pairs[2 * i] = base;
            pairs[2 * i + 1] = bytes;
        }
        ++i;
    }
    return i;
}
// Experimental export: the slab accounting behind those pages (#1199): slabs open, slots in use and slots free
// in the open slabs. A slab is reserved when it is created, its slots are committed on first use, and it is closed only at process end, so
// a slot freed by a grant stays committed and these numbers do not show the commit held.
extern "C" BOOL WINAPI x2w_xmem_slab_stats(ULONGLONG* slabs, ULONGLONG* slots_in_use, ULONGLONG* slots_free)
{
    if (!slabs || !slots_in_use || !slots_free) return FALSE;
    const std::lock_guard guard(memory.mutex);
    *slabs = memory.slabs.size();
    *slots_in_use = memory.backings.size();
    *slots_free = memory.free_slots.size() + (memory.slabs.empty() ? 0 : (slab_bytes - memory.slab_used) / page_bytes);
    return TRUE;
}

// Experimental export (X2W_XMEM_LAZY_COMMIT): commits the lazily mapped slots behind [address, address + bytes)
// before a writer that cannot fault one in uses the range: kernel-mode I/O into mapped pages fails with
// STATUS_ACCESS_VIOLATION instead of faulting, and the GPU reads and writes placed resources without the CPU.
// GPU-view addresses resolve to the same slots. TRUE without the switch or when nothing in the range is lazy;
// FALSE with the system error when a commit fails.
extern "C" BOOL WINAPI x2w_xmem_commit_range(const void* address, SIZE_T bytes)
{
    if (!lazy_commit() || !address || !bytes) return TRUE;
    const auto start = reinterpret_cast<ULONG_PTR>(address);
    const ULONG_PTR end = bytes > ~ULONG_PTR{0} - start ? ~ULONG_PTR{0} : start + bytes;
    ULONG_PTR from = start;  // the part of the range past any lazy startup blocks it starts in
    if (lazy_startup()) {
        ULONG_PTR base = 0;
        SIZE_T block_bytes = 0;
        bool committed = true;
        AcquireSRWLockShared(&lazy_blocks.lock);
        for (; from < end && lazy_block_at(from, base, block_bytes); from = base + block_bytes) {
            committed = commit_block_chunks(base, block_bytes, from, end);
            if (!committed) break;
        }
        ReleaseSRWLockShared(&lazy_blocks.lock);  // keeps the commit's error
        if (!committed) return FALSE;
        if (from >= end) return TRUE;  // the range lies in lazy startup blocks
    }
    const std::lock_guard guard(memory.mutex);
    for (ULONG_PTR page = from & ~(ULONG_PTR{page_bytes} - 1); page < end; page += page_bytes) {
        Backing* backing = backing_at(page);
        if (backing && !commit_backing(*backing)) return FALSE;
        if (page > ~ULONG_PTR{0} - page_bytes) break;
    }
    return TRUE;
}

// Experimental export (X2W_XMEM_LAZY_STARTUP): commits the whole lazy startup block holding address and stops
// committing it lazily. A GPU heap opened on the block covers the committed run from its base and is reused for
// later resources in the block, so the block is committed whole before the first one is placed. TRUE without the
// switch or when the address is in no lazy block; FALSE with the system error when the commit fails.
extern "C" BOOL WINAPI x2w_xmem_commit_block(const void* address)
{
    if (!lazy_startup()) return TRUE;
    ULONG_PTR base = 0;
    SIZE_T bytes = 0;
    AcquireSRWLockExclusive(&lazy_blocks.lock);
    const bool lazy = lazy_block_at(reinterpret_cast<ULONG_PTR>(address), base, bytes);
    const bool committed = !lazy || VirtualAlloc(reinterpret_cast<void*>(base), bytes, MEM_COMMIT, PAGE_READWRITE);
    if (lazy && committed) lazy_blocks.blocks.erase(base);
    ReleaseSRWLockExclusive(&lazy_blocks.lock);  // keeps the commit's error
    return committed;
}

namespace {
// The title reads files straight into mapped physical pages; a kernel-mode write into an uncommitted slot fails
// with STATUS_ACCESS_VIOLATION (no fault), so with X2W_XMEM_LAZY_COMMIT the main module's ReadFile import commits
// the buffer first.
using ReadFileFn = BOOL(WINAPI*)(HANDLE, LPVOID, DWORD, LPDWORD, LPOVERLAPPED);
ReadFileFn real_read_file;

BOOL WINAPI read_file_committed(HANDLE file, LPVOID buffer, DWORD bytes, LPDWORD read, LPOVERLAPPED overlapped)
{
    if (buffer && bytes && !x2w_xmem_commit_range(buffer, bytes)) return FALSE;
    return real_read_file(file, buffer, bytes, read, overlapped);
}

// The title releases its startup blocks through its VirtualFree import: with X2W_XMEM_LAZY_STARTUP a released
// block leaves the lazy list, so a later allocation at the same addresses is never committed as part of it. The
// release runs under the list lock, so no commit of the block is in flight, and the block leaves the list only if
// the release succeeded: a refused release (a nonzero size, for instance) leaves it reserved and still lazy.
using VirtualFreeFn = BOOL(WINAPI*)(LPVOID, SIZE_T, DWORD);
VirtualFreeFn real_virtual_free;

BOOL WINAPI virtual_free_lazy(LPVOID address, SIZE_T bytes, DWORD type)
{
    if (!(type & MEM_RELEASE)) return real_virtual_free(address, bytes, type);
    AcquireSRWLockExclusive(&lazy_blocks.lock);
    const BOOL released = real_virtual_free(address, bytes, type);
    const DWORD error = GetLastError();
    if (released && lazy_blocks.blocks.erase(reinterpret_cast<ULONG_PTR>(address)))
        InterlockedIncrement(&lazy_block_releases);
    ReleaseSRWLockExclusive(&lazy_blocks.lock);
    SetLastError(error);
    return released;
}

// Points the main module's imports of name at replacement; returns how many import slots were changed.
int patch_main_import(const char* name, const void* replacement) noexcept
{
    auto* base = reinterpret_cast<unsigned char*>(GetModuleHandleW(nullptr));
    if (!base) return 0;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    int patched = 0;
    for (auto* imp = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); dir.Size && imp->Name; ++imp) {
        const auto* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + (imp->OriginalFirstThunk ? imp->OriginalFirstThunk : imp->FirstThunk));
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            const auto* by_name = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), name) != 0) continue;
            DWORD old = 0;
            if (!VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), PAGE_READWRITE, &old)) continue;
            slots->u1.Function = reinterpret_cast<ULONGLONG>(replacement);
            VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), old, &old);
            ++patched;
        }
    }
    return patched;
}

// Test build only (tests/xmem_lazy_imports_check.cpp): X2W_XMEM_TEST_LAZY_HANDLER_FAILURE=1 refuses the handler.
bool add_lazy_handler()
{
#ifdef X2W_XMEM_TEST_CONTROLS
    if (x2w_local::env_on("X2W_XMEM_TEST_LAZY_HANDLER_FAILURE")) return false;
#endif
    return AddVectoredExceptionHandler(1, lazy_commit_fault) != nullptr;
}

// With the switch, install the fault handler and the ReadFile wrapper when xmem.dll loads, before the title runs.
// The module is pinned: the handler must outlive any unload. If the handler cannot be installed, lazy commit stays
// off and nothing is wrapped.
const bool lazy_commit_installed = [] {
    if (!lazy_commit_requested()) return false;
    HMODULE self = nullptr;
    lazy_handler_installed = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                                 reinterpret_cast<LPCWSTR>(&lazy_commit_fault), &self) &&
                             add_lazy_handler();
    if (!lazy_handler_installed) {
        x2w_local::logf("x2w-xmem lazy commit (X2W_XMEM_LAZY_COMMIT): fault handler failed, pages committed when mapped");
        return false;
    }
    real_read_file = reinterpret_cast<ReadFileFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "ReadFile"));
    const int reads = real_read_file ? patch_main_import("ReadFile", reinterpret_cast<const void*>(&read_file_committed)) : 0;
    x2w_local::logf("x2w-xmem lazy commit (X2W_XMEM_LAZY_COMMIT): fault handler on, %d ReadFile import(s) wrapped", reads);
    if (lazy_startup()) {
        real_virtual_free = reinterpret_cast<VirtualFreeFn>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "VirtualFree"));
        const int frees = real_virtual_free ? patch_main_import("VirtualFree", reinterpret_cast<const void*>(&virtual_free_lazy)) : 0;
        x2w_local::logf("x2w-xmem lazy startup blocks (X2W_XMEM_LAZY_STARTUP): %d VirtualFree import(s) wrapped", frees);
    }
    return true;
}();
}  // namespace

// Experimental export: writes bytes into physical pages named by the title's
// page tokens (page number << 4, as XMemMapPhysicalPages takes them), starting offset bytes into the
// first page. DirectStorage requests can name such pages as their destination.
extern "C" BOOL WINAPI x2w_xmem_write_physical(const ULONG_PTR* tokens, ULONG_PTR count, SIZE_T offset, const void* data, SIZE_T bytes)
{
    X2W_ZONE("x2w_xmem_write_physical");
    if (!tokens || !data || offset >= page_bytes) return FALSE;
    const auto* source = static_cast<const unsigned char*>(data);
    ULONG_PTR index = 0;
    while (bytes) {
        if (index >= count || (tokens[index] & 0xF)) return FALSE;
        // The lock is held across the write: a slab move (migrate_slab) must not happen between
        // finding the section and writing into it.
        const std::lock_guard guard(memory.mutex);
        const HANDLE section = section_for(tokens[index] >> 4);
        if (!section) return FALSE;
        const ULONG64 section_offset = memory.backings.at(tokens[index] >> 4).offset;
        void* view = MapViewOfFile(section, FILE_MAP_WRITE, static_cast<DWORD>(section_offset >> 32),
                                   static_cast<DWORD>(section_offset), page_bytes);
        if (!view) return FALSE;
        const SIZE_T chunk = bytes < page_bytes - offset ? bytes : page_bytes - offset;
        std::memcpy(static_cast<unsigned char*>(view) + offset, source, chunk);
        UnmapViewOfFile(view);
        source += chunk;
        bytes -= chunk;
        offset = 0;
        ++index;
    }
    return TRUE;
}

extern "C" void* WINAPI XMemMapPhysicalPages(void* base, ULONG_PTR count, const ULONG_PTR* tokens)
{
    const auto address = reinterpret_cast<ULONG_PTR>(base);
    if (!address || count > (std::numeric_limits<ULONG_PTR>::max() - address) / page_bytes) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    if (local_permissive()) {
        if (count > era_physical::capacity_pages) {  // as the strict path; sizes the vectors below
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        // A null array unmaps (seen on a cleanup path, also with a zero count).
        std::vector<ULONG_PTR> addresses(count), pages(count, 0);
        for (ULONG_PTR i = 0; i < count; ++i) addresses[i] = address + i * page_bytes;
        if (tokens && !to_pages(count, tokens, pages)) {
            SetLastError(ERROR_INVALID_PARAMETER);
            return nullptr;
        }
        if (!count) return base;
        if (!local_apply(count, addresses.data(), pages.data(), "XMemMapPhysicalPages")) return nullptr;
        return base;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return address + i * page_bytes; }, "XMemMapPhysicalPages") ? base : nullptr;
}

extern "C" BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, ULONG_PTR count,
                                                   const ULONG_PTR* tokens)
{
    if (!addresses) return physical_result(era_physical::Status::invalid);
    if (local_permissive() && tokens) {
        std::vector<ULONG_PTR> pages;
        if (!to_pages(count, tokens, pages)) return physical_result(era_physical::Status::invalid);
        if (!local_apply(count, reinterpret_cast<const ULONG_PTR*>(addresses), pages.data(), "XMemMapPhysicalPagesScatter"))
            return FALSE;
        return TRUE;
    }
    return map_pages(count, tokens, [=](ULONG_PTR i) { return reinterpret_cast<ULONG_PTR>(addresses[i]); }, "XMemMapPhysicalPagesScatter");
}

// Startup compatibility configuration (#1758, under #874): X2W_XMEM_STARTUP_BELOW_40=1 places the
// large-page pool blocks with attribute 0x283 (protection 0x2004 or 0x2404) so that the whole block,
// last byte included, lies below 2^40. The title passes the CPU address of such a block straight
// through as a GPU address, and a package acceleration-structure builder refuses destinations with
// more than 40 address bits. Off by default; every other shape keeps its placement. The low range is
// a requirement, never a preference: an allocation that cannot be placed there fails with the host
// error, and nothing is truncated, masked or retried higher.
constexpr ULONG_PTR startup_low_limit = ULONG_PTR{1} << 40;  // exclusive
constexpr SIZE_T startup_large_page = SIZE_T{2} << 20;

#ifdef X2W_XMEM_TEST_CONTROLS
// Test build only (tests/xmem_startup_below_40_check.cpp): X2W_XMEM_TEST_BELOW_40 names mutants
// that drop the address limit ("no_limit"), the check after the allocation ("no_check") or the
// release of a refused block ("no_release").
static bool startup_below_40_mutant(const char* name) noexcept
{
    char text[64]{};
    return GetEnvironmentVariableA("X2W_XMEM_TEST_BELOW_40", text, sizeof(text)) && std::strstr(text, name);
}
#else
static bool startup_below_40_mutant(const char*) noexcept { return false; }
#endif

// Checks a block allocated with the 2^40 requirement: 2 MiB aligned and base + bytes - 1 < 2^40,
// computed without overflow. A block outside that range is released and refused.
static void* keep_startup_block_below_40(void* block, SIZE_T bytes, ULONGLONG attributes, DWORD protection) noexcept
{
    if (!block) {
        const DWORD error = GetLastError();
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, no 2 MiB-aligned "
                        "range ending below 2^40 (host error %lu)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection),
                        static_cast<unsigned long>(error));
        SetLastError(error);
        return nullptr;
    }
    const auto base = reinterpret_cast<ULONG_PTR>(block);
    const bool inside = base % startup_large_page == 0 && bytes <= startup_low_limit && base <= startup_low_limit - bytes;
    if (!inside && !startup_below_40_mutant("no_check")) {
        const BOOL released = startup_below_40_mutant("no_release") ? FALSE : VirtualFree(block, 0, MEM_RELEASE);
        x2w_local::logf("x2w-xmem startup block below 2^40 refused: size=0x%llx attr=0x%llx prot=0x%lx, placed at %p "
                        "outside the 2 MiB-aligned range below 2^40 (%s)",
                        static_cast<unsigned long long>(bytes), attributes, static_cast<unsigned long>(protection), block,
                        released ? "released" : "not released");
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    x2w_local::logf("x2w-xmem startup block below 2^40: %p size=0x%llx last=%p attr=0x%llx prot=0x%lx", block,
                    static_cast<unsigned long long>(bytes), reinterpret_cast<void*>(base + (bytes - 1)), attributes,
                    static_cast<unsigned long>(protection));
    return block;
}

// Reserve+commit desktop profile for the startup blocks observed on #870: the pool loop asks
// for type 0x20003000 with 0x183/0x22004, 0x283/0x2404 or 0x283/0x2004, and later startup code for type
// 0x20403000 (64 KiB pages) with 0x181/PAGE_READWRITE, each with a null base. Proposed
// desktop policy: commit a fresh block of ordinary zero-filled read/write pages, whose size
// must be a nonzero multiple of 64 KiB and is committed exactly, without rounding. The
// MEM_LARGE_PAGES type gets a 2 MiB-aligned base (the x64 large-page size); the
// MEM_64K_PAGES type (MEM_LARGE_PAGES | MEM_PHYSICAL) keeps the 64 KiB allocation
// granularity. Large pages themselves and the console protection bits above PAGE_READWRITE
// are not reproduced. A host failure returns NULL with the host error. X2W_XMEM_STARTUP_BELOW_40
// adds an address limit for the 0x283 blocks (keep_startup_block_below_40).
static void* commit_startup_block_raw(SIZE_T bytes, DWORD allocation_type, ULONGLONG attributes,
                                      DWORD protection) noexcept
{
    constexpr SIZE_T granularity = SIZE_T{64} << 10;
    const bool large_pages = allocation_type == 0x20003000;
    const bool supported = large_pages
                               ? (attributes == 0x183 && protection == 0x22004)
                                     || (attributes == 0x283 && (protection == 0x2404 || protection == 0x2004))
                               : attributes == 0x181 && protection == PAGE_READWRITE;
    if (!bytes || bytes % granularity || !supported) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    const bool below_40 = large_pages && attributes == 0x283 && x2w_local::env_on("X2W_XMEM_STARTUP_BELOW_40");
    // X2W_XMEM_LAZY_STARTUP: a large block is reserved and committed on first touch (lazy_startup).
    const bool lazy = lazy_startup() && bytes >= lazy_block_min && !below_40;
    const DWORD type = lazy ? MEM_RESERVE : MEM_RESERVE | MEM_COMMIT;
    void* block = nullptr;
    if (!large_pages) {
        block = x2w_low::alloc(bytes, type, PAGE_READWRITE);
    } else {
        MEM_ADDRESS_REQUIREMENTS requirements{};
        requirements.Alignment = SIZE_T{2} << 20;
        if (x2w_low::on()) {
            requirements.LowestStartingAddress = reinterpret_cast<void*>(ULONG_PTR{1} << 32);
            requirements.HighestEndingAddress = reinterpret_cast<void*>((ULONG_PTR{1} << 40) - 1);
        }
        // supported already limits attribute 0x283 to protection 0x2004 or 0x2404.
        if (below_40 && !startup_below_40_mutant("no_limit"))
            requirements.HighestEndingAddress = reinterpret_cast<void*>(startup_low_limit - 1);
        MEM_EXTENDED_PARAMETER parameter{};
        parameter.Type = MemExtendedParameterAddressRequirements;
        parameter.Pointer = &requirements;
        block = VirtualAlloc2(GetCurrentProcess(), nullptr, bytes, type, PAGE_READWRITE, &parameter, 1);
        if (below_40) return keep_startup_block_below_40(block, bytes, attributes, protection);
    }
    if (block && lazy) {
        bool listed = false;
        AcquireSRWLockExclusive(&lazy_blocks.lock);
        try {
            lazy_blocks.blocks[reinterpret_cast<ULONG_PTR>(block)] = bytes;
            listed = true;
        } catch (const std::bad_alloc&) {
        }
        ReleaseSRWLockExclusive(&lazy_blocks.lock);
        if (!listed && !VirtualAlloc(block, bytes, MEM_COMMIT, PAGE_READWRITE)) {  // not listed: commit it now
            const DWORD error = GetLastError();
            VirtualFree(block, 0, MEM_RELEASE);
            SetLastError(error);
            return nullptr;
        }
    }
    return block;
}

static void* commit_startup_block(SIZE_T bytes, DWORD allocation_type, ULONGLONG attributes,
                                  DWORD protection) noexcept
{
    void* block = commit_startup_block_raw(bytes, allocation_type, attributes, protection);
    const DWORD error = GetLastError();
    if (block) {
        InterlockedAdd64(&startup_block_bytes, static_cast<LONG64>(bytes));
        x2w_census::note_block(block, bytes);
    }
    const LONG64 n = InterlockedIncrement64(&startup_block_calls);
    if (n <= 64 || !block)
        x2w_local::logf("x2w-xmem startup block #%lld size=0x%llx type=0x%lx attr=0x%llx prot=0x%lx -> %p (err %lu), %lld MiB so far",
                        static_cast<long long>(n), static_cast<unsigned long long>(bytes), static_cast<unsigned long>(allocation_type),
                        attributes, static_cast<unsigned long>(protection), block, block ? 0ul : static_cast<unsigned long>(error),
                        static_cast<long long>(startup_block_bytes >> 20));
    SetLastError(error);
    return block;
}

// The startup commit profile above, and the reserve-only desktop profile from the source-built
// #888 comparison.
extern "C" void* WINAPI XMemVirtualAlloc(void* address, SIZE_T bytes, DWORD allocation_type,
                                        ULONGLONG attributes, DWORD protection)
{
    x2w_census::start();
    constexpr DWORD startup_reserve = 0x20402000;
    constexpr SIZE_T gib = SIZE_T{1} << 30;
    if (!address && (allocation_type == 0x20003000 || allocation_type == (startup_reserve | MEM_COMMIT)))
        return commit_startup_block(bytes, allocation_type, attributes, protection);
    const bool supported = (attributes == 0x21 && protection == PAGE_READWRITE)
                           || (attributes == 3 && protection == (0x800 | PAGE_NOACCESS));
    if (address || (bytes != 16 * gib && bytes != 24 * gib) || allocation_type != startup_reserve || !supported) {
        if (local_permissive()) return local_virtual_alloc(address, bytes, allocation_type, attributes, protection);
        SetLastError(ERROR_INVALID_PARAMETER);
        return nullptr;
    }
    const std::lock_guard guard(memory.mutex);
    // Callers release unused reservations with Win32. Drop entries for released
    // ranges before another allocation can reuse those virtual addresses.
    for (auto it = memory.reservations.begin(); it != memory.reservations.end();) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<void*>(it->first), &info, sizeof(info)) && info.State == MEM_FREE)
            it = memory.reservations.erase(it);
        else
            ++it;
    }
    auto* reservation = x2w_low::alloc(bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS);
    if (!reservation) return nullptr;
    try {
        memory.reservations.insert_or_assign(reinterpret_cast<ULONG_PTR>(reservation), bytes);
        const auto base = reinterpret_cast<ULONG_PTR>(reservation);
        if (protection == PAGE_READWRITE) {
            memory.last_cpu_base = base;
            memory.last_cpu_bytes = bytes;
        } else if (memory.last_cpu_base && memory.last_cpu_bytes == bytes && x2w_local::env_on("X2W_XMEM_GPU_MIRROR")) {
            memory.mirror_cpu = memory.last_cpu_base;
            memory.mirror_gpu = base;
            memory.mirror_bytes = bytes;
            memory.explicit_mirror_pages.clear();
            x2w_local::logf("x2w-xmem GPU view %p (+0x%llx) mirrors CPU view %p, 0x%llx bytes", reservation,
                            static_cast<unsigned long long>(base - memory.last_cpu_base), reinterpret_cast<void*>(memory.last_cpu_base),
                            static_cast<unsigned long long>(bytes));
        }
    } catch (const std::bad_alloc&) {
        VirtualFree(reservation, 0, MEM_RELEASE);
        SetLastError(ERROR_NOT_ENOUGH_MEMORY);
        return nullptr;
    }
    return reservation;
}

extern "C" BOOL WINAPI x2w_xmem_page_event(const void* address, ULONG* event_kind, ULONG64* page, ULONG* thread,
                                           ULONG64* age_ms, ULONG64* serial, ULONG64* remaps)
{
    const auto at = reinterpret_cast<ULONG_PTR>(address) & ~(ULONG_PTR{page_bytes} - 1);
    const std::lock_guard guard(memory.mutex);
    const auto event = page_events.find(at);
    const PageEvent e = event != page_events.end() ? event->second : PageEvent{};
    if (event_kind) *event_kind = e.kind;
    if (page) *page = e.page;
    if (thread) *thread = e.thread;
    if (age_ms) *age_ms = e.tick ? GetTickCount64() - e.tick : 0;
    if (serial) *serial = e.serial;
    if (remaps) *remaps = remap_gaps;
    return memory.views.count(at) != 0;
}

bool range_writable(ULONG_PTR start, SIZE_T bytes) noexcept
{
    for (ULONG_PTR cursor = start; cursor < start + bytes;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) || info.State != MEM_COMMIT ||
            (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)))
            return false;
        const DWORD protect = info.Protect & 0xFF;
        if (protect != PAGE_READWRITE && protect != PAGE_WRITECOPY && protect != PAGE_EXECUTE_READWRITE &&
            protect != PAGE_EXECUTE_WRITECOPY)
            return false;
        const ULONG_PTR end = reinterpret_cast<ULONG_PTR>(info.BaseAddress) + info.RegionSize;
        if (end <= cursor) return false;
        cursor = end;
    }
    return true;
}

extern "C" BOOL WINAPI x2w_xmem_write_settled(void* destination, const void* data, SIZE_T bytes, SIZE_T capacity)
{
    if (!destination || (bytes && !data) || bytes > capacity) return FALSE;
    const auto start = reinterpret_cast<ULONG_PTR>(destination);
    if (capacity > std::numeric_limits<ULONG_PTR>::max() - start) return FALSE;
    const MappingThread marker;  // a fault in the copy is never retried by retry_remapped_page (it takes the mutex)
    std::unique_lock guard(memory.mutex);
    // Inside a recorded reservation, or one starts inside the destination.
    const auto next = memory.reservations.upper_bound(start);
    const bool ours = owned_reservation(start) != memory.reservations.end() ||
                      (next != memory.reservations.end() && next->first < start + capacity);
    if (!ours) guard.unlock();
    if (!range_writable(start, capacity)) return FALSE;
    std::memcpy(destination, data, bytes);
    return TRUE;
}

extern "C" BOOL WINAPI x2w_xmem_write_settled_wait(void* destination, const void* data, SIZE_T bytes, SIZE_T capacity,
                                                   ULONG wait_ms, ULONG* waited_ms)
{
    const ULONG64 began = GetTickCount64();
    for (;;) {
        const BOOL ok = x2w_xmem_write_settled(destination, data, bytes, capacity);
        const ULONG64 spent = GetTickCount64() - began;
        if (ok || spent >= wait_ms) {
            if (waited_ms) *waited_ms = static_cast<ULONG>(spent);
            return ok;
        }
        Sleep(5);
    }
}

extern "C" BOOL WINAPI x2w_xmem_gpu_location(const void* address, HANDLE* section, ULONG64* offset)
{
    if (!section || !offset) return FALSE;
    const auto at = reinterpret_cast<ULONG_PTR>(address);
    const std::lock_guard guard(memory.mutex);
    const auto view = memory.views.find(at - at % page_bytes);
    if (view == memory.views.end()) return FALSE;
    const auto backing = memory.backings.find(view->second);
    if (backing == memory.backings.end()) return FALSE;
    if (memory.file_slabs.count(backing->second.section) && !migrate_slab(backing->second.section)) return FALSE;
    *section = backing->second.section;  // the pagefile slab after a move
    *offset = backing->second.offset + at % page_bytes;
    return TRUE;
}

extern "C" BOOL WINAPI x2w_xmem_pool_run(const void* address, SIZE_T bytes, HANDLE* section, ULONG64* offset)
{
    if (!section || !offset || !bytes) return FALSE;
    const auto at = reinterpret_cast<ULONG_PTR>(address);
    if (at % page_bytes) return FALSE;
#ifdef X2W_XMEM_TEST_CONTROLS
    // Test build only: any slab counts as a pool section (a fixture must not claim the real pool).
    static const bool any_slab = x2w_local::env_on("X2W_XMEM_TEST_POOL_ANY");
#else
    constexpr bool any_slab = false;
#endif
    const std::lock_guard guard(memory.mutex);
    HANDLE first_section = nullptr;
    ULONG64 first_offset = 0;
    for (SIZE_T done = 0; done < bytes; done += page_bytes) {
        const auto view = memory.views.find(at + done);
        if (view == memory.views.end()) return FALSE;
        const auto backing = memory.backings.find(view->second);
        if (backing == memory.backings.end()) return FALSE;
        const HANDLE s = backing->second.section;
        if (!done) {
            if (!memory.pool_views.count(s) && !any_slab) return FALSE;  // not a pool section
            first_section = s;
            first_offset = backing->second.offset;
        } else if (s != first_section || backing->second.offset != first_offset + done) {
            return FALSE;  // the pages are not consecutive in one section
        }
    }
    *section = first_section;
    *offset = first_offset;
    return TRUE;
}
