// Source-built check of the AcpHal stand-in APU heap (src/startup_stubs/acphal.cpp): ApuCreateHeap reserves,
// ApuAlloc commits on demand, ApuFree returns blocks, coalesces neighbours and decommits interior pages without
// disturbing shared boundary pages, bad requests leave the heap alone and concurrent callers never overlap.
// Usage: startup_acphal_apu_free <AcpHal.dll>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using u64 = std::uint64_t;
using AllocFn = long(__cdecl*)(void**, std::uint32_t*, u64, u64, u64);
using FreeFn = long(__cdecl*)(void*);
using CreateFn = long(__cdecl*)(u64, u64);
using MapFn = std::uint32_t(__cdecl*)(void*);
using MapApuFn = void*(__cdecl*)(u64);
using ValidFn = int(__cdecl*)(void*);
using HalCreateFn = long(__cdecl*)(void**);
using ShapeAllocFn = long(__cdecl*)(void*);
using ShapeReleaseFn = long(__cdecl*)(void*);
using HalConnectFn = long(__stdcall*)(void*, unsigned, unsigned);
using HalSubmitFn = long(__stdcall*)(void*, unsigned, u64, unsigned, const unsigned*, u64);

constexpr long ok = 0, invalid = static_cast<long>(0x80070057), no_memory = static_cast<long>(0x8007000E),
               no_pointer = static_cast<long>(0x80004003), not_impl = static_cast<long>(0x80004001);
constexpr u64 heap_size = 0x200000, first_offset = 0x10000, usable = heap_size - first_offset;

struct ShapeContexts {
    std::uint32_t counts[6]{};
    unsigned char* arrays[6]{};
    std::uint32_t apu[6]{};
};
static_assert(sizeof(ShapeContexts) == 0x60, "ShapeContexts layout");
static_assert(offsetof(ShapeContexts, arrays) == 0x18, "arrays offset");
static_assert(offsetof(ShapeContexts, apu) == 0x48, "apu offset");

AllocFn alloc_fn;
FreeFn free_fn;
MapFn map_fn;
MapApuFn map_apu_fn;
ValidFn valid_fn;
HalCreateFn hal_create_fn;
ShapeAllocFn shape_alloc_fn;
ShapeReleaseFn shape_release_fn;
int failures;

void check(bool condition, const char* what)
{
    if (condition) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

long take(void** out, u64 size, u64 alignment = 16, std::uint32_t* apu = nullptr) { return alloc_fn(out, apu, size, alignment, 0); }

void* take_ok(u64 size, u64 alignment = 16)
{
    void* p = nullptr;
    const long hr = take(&p, size, alignment);
    check(hr == ok && p, "allocation succeeds");
    return p;
}

// The whole heap is free again exactly when one block of the full usable size fits at the first offset.
void check_empty(const char* where)
{
    std::uint32_t apu = 0;
    void* p = nullptr;
    const long hr = take(&p, usable, 16, &apu);
    if (hr != ok || !p || apu != first_offset) {
        std::printf("FAIL: heap not whole after %s (hr=0x%08lx apu=0x%x)\n", where, static_cast<unsigned long>(hr), apu);
        ++failures;
        return;
    }
    check(free_fn(p) == ok, "whole-heap block frees");
}

void churn_beyond_capacity()
{
    void* a = take_ok(0x100000, 4096);
    void* b = nullptr;
    check(take(&b, 0x100000, 4096) == no_memory && !b, "two 1 MiB blocks do not fit while the first is live");
    check(free_fn(a) == ok, "free first");
    for (int round = 0; round < 64; ++round) {  // 64 MiB handed out in total, over 30 times the heap
        void* p = take_ok(0x100000, 4096);
        check(p == a, "same block comes back");
        if (!p) return;
        check(free_fn(p) == ok, "churn free");
    }
    check_empty("churn");
}

void reuse_and_alignment()
{
    for (u64 alignment : {16u, 256u, 4096u, 65536u}) {
        void* keep = take_ok(0x30, 16);
        void* a = take_ok(0x2000, alignment);
        void* after = take_ok(0x30, 16);
        check(reinterpret_cast<std::uintptr_t>(a) % alignment == 0, "alignment honoured");
        std::memset(a, 0xAA, 0x2000);
        check(free_fn(a) == ok, "free");
        void* again = take_ok(0x2000, alignment);
        check(again == a, "same size returns the freed block");
        bool zero = true;
        for (int i = 0; i < 0x2000; ++i) zero = zero && static_cast<unsigned char*>(again)[i] == 0;
        check(zero, "reused block is zeroed like fresh heap memory");
        check(free_fn(again) == ok, "free again");
        void* smaller = take_ok(0x1000, alignment);
        check(smaller >= a && static_cast<unsigned char*>(smaller) + 0x1000 <= static_cast<unsigned char*>(a) + 0x2000,
              "smaller request lands inside the freed block");
        std::uint32_t apu = 0;
        void* mapped = nullptr;
        check(take(&mapped, 0x40, 64, &apu) == ok && map_fn(mapped) == apu && apu, "APU address maps back to the block");
        for (void* p : {keep, after, smaller, mapped}) check(free_fn(p) == ok, "cleanup free");
        check_empty("reuse");
    }
}

void coalescing()
{
    constexpr u64 part = 0x10000;
    void* a = take_ok(part);
    void* b = take_ok(part);
    void* c = take_ok(part);
    void* rest = take_ok(usable - 3 * part);  // fills the heap: only A, B, C can ever be free
    auto* base = static_cast<unsigned char*>(a);
    check(b == base + part && c == base + 2 * part, "A, B and C are contiguous");
    check(free_fn(a) == ok && free_fn(c) == ok, "free A and C");
    void* big = nullptr;
    check(take(&big, 3 * part) == no_memory && !big, "A+B+C does not fit while B is live");
    check(free_fn(b) == ok, "free B");
    big = take_ok(3 * part);
    check(big == a, "A+B+C is one block at A");
    check(free_fn(big) == ok && free_fn(rest) == ok, "cleanup");
    check_empty("coalescing");
    void* x = take_ok(part);  // reverse order, middle first
    void* y = take_ok(part);
    void* z = take_ok(part);
    check(free_fn(y) == ok && free_fn(z) == ok && free_fn(x) == ok, "free middle, last, first");
    check_empty("coalescing reverse");
}

void bad_requests()
{
    const void* const sentinel = reinterpret_cast<void*>(0x1234);
    void* out = const_cast<void*>(sentinel);
    std::uint32_t apu = 0xDEADBEEF;
    check(free_fn(nullptr) == ok, "free(null) is a no-op");
    check(take(nullptr, 64) == no_pointer, "null output");
    if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
        auto* ro_ptr = static_cast<void**>(ro);
        auto* ro_apu = static_cast<std::uint32_t*>(ro);
        check(take(ro_ptr, 64, 16, &apu) == no_pointer && apu == 0xDEADBEEF,
              "read-only virtual_address returns E_POINTER and leaves apu unchanged");
        out = const_cast<void*>(sentinel);
        check(take(&out, 64, 16, ro_apu) == no_pointer && !out,
              "read-only apu_address returns E_POINTER and zeroes virtual_address");
        out = const_cast<void*>(sentinel);
        check(take(&out, 0, 16, ro_apu) == no_pointer && !out,
              "read-only apu_address with size 0 returns E_POINTER");
        VirtualFree(ro, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc PAGE_READONLY for ApuAlloc negative control");
    }
    check_empty("unwritable ApuAlloc outputs");
    out = const_cast<void*>(sentinel);
    check(take(&out, usable + 16, 16, &apu) == no_memory && !out && apu == 0xDEADBEEF, "larger than the heap");
    out = const_cast<void*>(sentinel);
    check(take(&out, 0xFFFFFFFFull, 16, &apu) == no_memory && !out, "size near 2^32");
    check(take(&out, ~u64{0}, 16, &apu) == no_memory && !out, "size 2^64-1 is cut to 32 bits");
    check(take(&out, 0, 16, &apu) == invalid && !out && apu == 0xDEADBEEF, "size 0");
    check(take(&out, 0x100000000ull, 16, &apu) == invalid && !out, "size 2^32 is size 0");
    // Not a power of two means 16, so these succeed and honour 16.
    for (u64 odd : {u64{24}, u64{0xFFFFFFFF}, u64{3}}) {
        void* r = take_ok(64, odd);
        check(reinterpret_cast<std::uintptr_t>(r) % 16 == 0, "non-power-of-two alignment falls back to 16");
        check(free_fn(r) == ok, "free fallback block");
    }
    // No placement can satisfy a heap-sized block at 2 GiB alignment plus 16 spare bytes.
    out = const_cast<void*>(sentinel);
    check(take(&out, usable + 16, 0x80000000ull, &apu) == no_memory && !out && apu == 0xDEADBEEF, "no-fit request with 2 GiB alignment");
    check_empty("refused requests");

    void* p = take_ok(0x40, 0);  // 0 means the default of 16
    check(reinterpret_cast<std::uintptr_t>(p) % 16 == 0, "alignment 0");
    void* q = take_ok(0x40, 4);
    check(reinterpret_cast<std::uintptr_t>(q) % 16 == 0, "alignment 4 is raised to 16");
    check(free_fn(static_cast<unsigned char*>(p) + 8) == invalid, "free inside a live block");
    check(free_fn(&failures) == invalid, "free of a foreign address");
    check(free_fn(reinterpret_cast<void*>(0x10)) == invalid, "free of a tiny address");
    check(free_fn(p) == ok, "free");
    check(free_fn(p) == invalid, "double free");
    void* big = nullptr;
    check(take(&big, usable) == no_memory, "q is still live after the refused frees");
    check(free_fn(q) == ok && free_fn(q) == invalid, "free q, then a double free");
    check_empty("refused frees");
}

void table_full()
{
    std::vector<void*> blocks;
    void* p = nullptr;
    while (blocks.size() < 5000 && take(&p, 256) == ok) blocks.push_back(p);
    check(blocks.size() < 5000 && blocks.size() * 256 < usable, "extent table limit refuses before the heap is full");
    check(!blocks.empty() && take(&p, 256) == no_memory && !p, "refusal is E_OUTOFMEMORY and repeatable");
    for (size_t i = 0; i < blocks.size(); i += 2) check(free_fn(blocks[i]) == ok, "free every other block");
    for (size_t i = 1; i < blocks.size(); i += 2) check(free_fn(blocks[i]) == ok, "free the rest");
    check_empty("table full");
}

// Deterministic generator, so a failure is reproducible.
struct Rng {
    std::uint32_t state;
    std::uint32_t next() { return state = state * 1664525u + 1013904223u; }
};

struct Live {
    std::map<std::uintptr_t, std::uintptr_t> blocks;  // start -> end, shared by all threads
    std::mutex lock;
    bool add(std::uintptr_t start, std::uintptr_t end)
    {
        std::lock_guard<std::mutex> guard(lock);
        auto next = blocks.lower_bound(start);
        if (next != blocks.end() && next->first < end) return false;
        if (next != blocks.begin() && std::prev(next)->second > start) return false;
        blocks[start] = end;
        return true;
    }
    void remove(std::uintptr_t start)
    {
        std::lock_guard<std::mutex> guard(lock);
        blocks.erase(start);
    }
};

void worker(unsigned id, unsigned rounds, std::atomic<int>* ready, std::atomic<int>* errors, Live* live)
{
    struct Mine {
        unsigned char* p;
        u64 size;
        unsigned char tag;
    };
    std::vector<Mine> mine;
    Rng rng{0xC0FFEEu + id * 7919u};
    ready->fetch_add(1);
    while (ready->load() < 4) {}
    auto release = [&](std::size_t index) {
        const Mine m = mine[index];
        for (u64 i = 0; i < m.size; ++i)
            if (m.p[i] != m.tag) { errors->fetch_add(1); break; }
        live->remove(reinterpret_cast<std::uintptr_t>(m.p));
        if (free_fn(m.p) != ok) errors->fetch_add(1);
        mine.erase(mine.begin() + static_cast<std::ptrdiff_t>(index));
    };
    for (unsigned round = 0; round < rounds; ++round) {
        if (mine.size() >= 12 || (!mine.empty() && rng.next() % 3 == 0)) release(rng.next() % mine.size());
        const u64 size = 16 + (rng.next() % 256) * 16;
        const u64 alignment = u64{16} << (rng.next() % 5);
        void* p = nullptr;
        if (take(&p, size, alignment) != ok || !p || reinterpret_cast<std::uintptr_t>(p) % alignment) {
            errors->fetch_add(1);
            continue;
        }
        const auto start = reinterpret_cast<std::uintptr_t>(p);
        if (!live->add(start, start + size)) errors->fetch_add(1);
        const unsigned char tag = static_cast<unsigned char>(id * 64 + round % 61 + 1);
        std::memset(p, tag, size);
        mine.push_back({static_cast<unsigned char*>(p), size, tag});
    }
    while (!mine.empty()) release(mine.size() - 1);
}

void concurrent()
{
    Live live;
    std::atomic<int> ready{0}, errors{0};
    std::vector<std::thread> threads;
    for (unsigned id = 0; id < 4; ++id) threads.emplace_back(worker, id, 1000u, &ready, &errors, &live);
    for (auto& t : threads) t.join();
    check(errors.load() == 0, "no overlap, torn pattern or unexpected error across 4 threads x 1000 rounds");
    check(live.blocks.empty(), "every block was released");
    check_empty("concurrent run");
}
// Competing first-time creators: every caller gets a block right after its create returns and keeps it
// while the others are still creating. A second creation that reset the table would hand out a
// block that overlaps a live one, or leave a live block unfreeable.
void first_creation(CreateFn create)
{
    // A heap that would cross 4 GiB by one byte (rounded up to 64 KiB) is refused before any memory is
    // committed and leaves no heap behind; the creations below still succeed.
    check(create(0xFFFFFFFFull, 2) == invalid, "heap crossing 4 GiB by one byte is refused");
    check(create(0xFFFFFFFFull, 0xFFFFFFFFull) == invalid, "two maximal 32-bit sizes are refused");
    void* refused = nullptr;
    check(take(&refused, 64) == no_memory && !refused, "refused ApuCreateHeap leaves no heap");
    constexpr unsigned callers = 4;
    std::atomic<bool> go{false};
    std::atomic<int> bad{0};
    void* blocks[callers] = {};
    std::vector<std::thread> threads;
    for (unsigned id = 0; id < callers; ++id)
        threads.emplace_back([&, id] {
            while (!go.load()) std::this_thread::yield();
            if (create(heap_size, 0) != ok) ++bad;
            void* p = nullptr;
            if (take(&p, 0x1000, 16) != ok || !p) ++bad;
            blocks[id] = p;
        });
    go = true;
    for (auto& t : threads) t.join();
    check(bad.load() == 0, "concurrent first ApuCreateHeap calls and allocations succeed");
    for (unsigned i = 0; i < callers; ++i)
        for (unsigned j = i + 1; j < callers; ++j)
            check(blocks[i] != blocks[j], "competing creators got distinct blocks");
    check(create(heap_size, 0) == ok, "repeat ApuCreateHeap");
    void* extra = take_ok(0x1000);
    for (void* b : blocks) check(extra != b, "block taken after a repeat creation does not reuse a live block");
    for (void* b : blocks) check(free_fn(b) == ok, "live block survives competing creation and frees");
    check(free_fn(extra) == ok, "free extra");
    check_empty("first creation");
}

void shape_contexts()
{
    check(shape_alloc_fn(nullptr) == no_pointer, "AcpHalAllocateShapeContexts(null) returns E_POINTER");
    if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
        check(shape_alloc_fn(ro) == no_pointer, "AcpHalAllocateShapeContexts(read-only) returns E_POINTER");
        VirtualFree(ro, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc PAGE_READONLY for shape_contexts negative control");
    }

    ShapeContexts zero_ctx{};
    check(shape_alloc_fn(&zero_ctx) == ok, "AcpHalAllocateShapeContexts with all-zero counts succeeds");
    check(!zero_ctx.arrays[4] && zero_ctx.apu[4] == 0, "all-zero counts leaves outputs zero");
    check_empty("shape contexts zero counts");

    ShapeContexts happy{};
    happy.counts[4] = 350;
    check(shape_alloc_fn(&happy) == ok, "AcpHalAllocateShapeContexts XMA-only happy path succeeds");
    check(happy.arrays[4] != nullptr, "XMA array pointer is populated");
    check(reinterpret_cast<std::uintptr_t>(happy.arrays[4]) % 64 == 0, "XMA array pointer is 64-byte aligned");
    check(happy.apu[4] != 0, "XMA APU address is populated");
    check(map_fn(happy.arrays[4]) == happy.apu[4], "XMA APU address matches ApuMapVirtualAddress");
    check(map_apu_fn(happy.apu[4]) == happy.arrays[4], "ApuMapApuAddress round-trips XMA array pointer");
    check(valid_fn(happy.arrays[4]) != 0, "XMA array pointer is valid APU virtual address");
    std::memset(happy.arrays[4], 0x5A, 350 * 64);
    check(free_fn(happy.arrays[4]) == ok, "XMA array frees via ApuFree");
    check_empty("shape contexts happy path");

    // Partial failure: i == 4 allocates, then counts[5] != 0 triggers E_NOTIMPL and must roll back.
    ShapeContexts partial{};
    partial.counts[4] = 350;
    partial.counts[5] = 1;
    check(shape_alloc_fn(&partial) == not_impl, "AcpHalAllocateShapeContexts with counts[5] != 0 returns E_NOTIMPL");
    check(partial.arrays[4] == nullptr && partial.apu[4] == 0,
          "partial-failure rollback zeroes arrays[4] and apu[4]");
    check_empty("shape contexts partial failure rollback");

    // Early unsupported type: i == 0 fails before i == 4.
    ShapeContexts early{};
    early.counts[0] = 1;
    early.counts[4] = 350;
    check(shape_alloc_fn(&early) == not_impl, "AcpHalAllocateShapeContexts with counts[0] != 0 returns E_NOTIMPL");
    check(early.arrays[4] == nullptr && early.apu[4] == 0, "early failure leaves arrays[4] and apu[4] zero");
    check_empty("shape contexts early failure");

    // Overflow / out-of-memory counts.
    for (std::uint32_t huge_count : {0x04000000u, 0xFFFFFFFFu}) {
        ShapeContexts oom{};
        oom.counts[4] = huge_count;
        check(shape_alloc_fn(&oom) == no_memory, "AcpHalAllocateShapeContexts huge count returns E_OUTOFMEMORY");
        check(oom.arrays[4] == nullptr && oom.apu[4] == 0, "OOM leaves arrays[4] and apu[4] zero");
        check_empty("shape contexts oom");
    }
}

void hal_and_address_mapping()
{
    check(hal_create_fn(nullptr) == no_pointer, "AcpHalCreate(null) returns E_POINTER");
    if (void* ro = VirtualAlloc(nullptr, 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_READONLY)) {
        check(hal_create_fn(static_cast<void**>(ro)) == no_pointer, "AcpHalCreate(read-only) returns E_POINTER");
        VirtualFree(ro, 0, MEM_RELEASE);
    } else {
        check(false, "VirtualAlloc PAGE_READONLY for AcpHalCreate negative control");
    }
    void* hal = nullptr;
    check(hal_create_fn(&hal) == ok && hal != nullptr, "AcpHalCreate returns stand-in object");
    {
        constexpr unsigned callers = 4;
        std::atomic<bool> go{false};
        std::atomic<int> bad{0};
        void* objects[callers] = {};
        std::vector<std::thread> threads;
        for (unsigned id = 0; id < callers; ++id)
            threads.emplace_back([&, id] {
                while (!go.load()) std::this_thread::yield();
                if (hal_create_fn(&objects[id]) != ok || !objects[id]) ++bad;
            });
        go = true;
        for (auto& t : threads) t.join();
        check(bad.load() == 0, "concurrent AcpHalCreate calls succeed");
        for (unsigned i = 0; i < callers; ++i)
            check(objects[i] == hal, "concurrent AcpHalCreate returns the same stand-in object");
    }
    if (hal) {
        auto* const* vtable = *reinterpret_cast<void* const* const*>(hal);
        auto connect = reinterpret_cast<HalConnectFn>(vtable[0]);
        auto submit = reinterpret_cast<HalSubmitFn>(vtable[2]);
        const unsigned command_val = 0x12345678u;
        check(connect(hal, 0x1000, 0x1000) == ok, "AcpHal::slot0(connect) returns S_OK");
        check(submit(hal, 1, 0x1234, 2, &command_val, 0) == ok, "AcpHal::slot2(submit) returns S_OK");
    }

    std::uint32_t apu = 0;
    void* block = nullptr;
    check(take(&block, 0x100, 64, &apu) == ok && block && apu, "allocate block for mapping checks");
    auto* inner = static_cast<unsigned char*>(block) + 0x80;
    check(valid_fn(block) != 0, "ApuIsVirtualAddressValid accepts block start");
    check(valid_fn(inner) != 0, "ApuIsVirtualAddressValid accepts interior pointer");
    check(map_fn(block) == apu, "ApuMapVirtualAddress maps block start");
    check(map_fn(inner) == apu + 0x80, "ApuMapVirtualAddress maps interior pointer");
    check(map_apu_fn(apu) == block, "ApuMapApuAddress maps block APU address");
    check(map_apu_fn(apu + 0x80) == inner, "ApuMapApuAddress maps interior APU address");

    // Negative controls for address validity and mapping.
    auto* heap_base = static_cast<unsigned char*>(block) - apu;
    check(valid_fn(nullptr) == 0, "ApuIsVirtualAddressValid(null) is 0");
    check(valid_fn(&failures) == 0, "ApuIsVirtualAddressValid(foreign) is 0");
    check(valid_fn(heap_base - 1) == 0, "ApuIsVirtualAddressValid(before heap) is 0");
    check(valid_fn(heap_base) == 0, "ApuIsVirtualAddressValid(reserved offset 0) is 0");
    check(valid_fn(heap_base + 0x100) == 0, "ApuIsVirtualAddressValid(inside reserved guard) is 0");
    check(valid_fn(heap_base + first_offset - 1) == 0, "ApuIsVirtualAddressValid(last reserved byte) is 0");
    check(valid_fn(heap_base + first_offset) != 0, "ApuIsVirtualAddressValid(first allocatable byte) is 1");
    check(valid_fn(heap_base + heap_size - 1) != 0, "ApuIsVirtualAddressValid(last heap byte) is 1");
    check(valid_fn(heap_base + heap_size) == 0, "ApuIsVirtualAddressValid(at heap end) is 0");
    check(map_fn(nullptr) == 0, "ApuMapVirtualAddress(null) is 0");
    check(map_fn(&failures) == 0, "ApuMapVirtualAddress(foreign) is 0");
    check(map_fn(heap_base - 1) == 0, "ApuMapVirtualAddress(before heap) is 0");
    check(map_fn(heap_base) == 0, "ApuMapVirtualAddress(reserved offset 0) is 0");
    check(map_fn(heap_base + 0x100) == 0, "ApuMapVirtualAddress(inside reserved guard) is 0");
    check(map_fn(heap_base + first_offset - 1) == 0, "ApuMapVirtualAddress(last reserved byte) is 0");
    check(map_fn(heap_base + first_offset) == first_offset, "ApuMapVirtualAddress(first allocatable byte)");
    check(map_fn(heap_base + heap_size - 1) == heap_size - 1, "ApuMapVirtualAddress(last heap byte)");
    check(map_fn(heap_base + heap_size) == 0, "ApuMapVirtualAddress(at heap end) is 0");
    check(map_apu_fn(0) == nullptr, "ApuMapApuAddress(0) is null");
    check(map_apu_fn(1) == nullptr, "ApuMapApuAddress(1 in reserved guard) is null");
    check(map_apu_fn(0x100) == nullptr, "ApuMapApuAddress(0x100 in reserved guard) is null");
    check(map_apu_fn(first_offset - 1) == nullptr, "ApuMapApuAddress(last reserved byte) is null");
    check(map_apu_fn(first_offset) == heap_base + first_offset, "ApuMapApuAddress(first allocatable byte)");
    check(map_apu_fn(heap_size - 1) == heap_base + heap_size - 1, "ApuMapApuAddress(last heap byte)");
    check(map_apu_fn(heap_size) == nullptr, "ApuMapApuAddress(heap_size) is null");
    check(map_apu_fn(heap_size + 0x1000) == nullptr, "ApuMapApuAddress(past end) is null");

    check(free_fn(block) == ok, "free mapping test block");
    check_empty("address mapping");
}

int run_trap_control(const char* dll_path, const char* export_name)
{
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", nullptr);
    SetEnvironmentVariableA("X2W_TRAP_CONTINUE", nullptr);
    SetEnvironmentVariableA("X2W_TRAP_FAIL", nullptr);
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return 2;
    FARPROC proc = GetProcAddress(module, export_name);
    if (!proc) return 3;
    if (std::strcmp(export_name, "AcpHalAllocateShapeContexts") == 0) {
        ShapeContexts ctx{};
        reinterpret_cast<ShapeAllocFn>(proc)(&ctx);
    } else if (std::strcmp(export_name, "AcpHalReleaseShapeContexts") == 0) {
        reinterpret_cast<ShapeReleaseFn>(proc)(nullptr);
    } else if (std::strcmp(export_name, "AcpHalCreate") == 0) {
        void* out = nullptr;
        reinterpret_cast<HalCreateFn>(proc)(&out);
    } else if (std::strcmp(export_name, "ApuIsVirtualAddressValid") == 0) {
        reinterpret_cast<ValidFn>(proc)(nullptr);
    } else if (std::strcmp(export_name, "ApuMapApuAddress") == 0) {
        reinterpret_cast<MapApuFn>(proc)(0);
    } else if (std::strcmp(export_name, "ApuMapVirtualAddress") == 0) {
        reinterpret_cast<MapFn>(proc)(nullptr);
    } else {
        return 5;
    }
    return 4;
}

void check_trap_case(const char* self_exe, const char* dll_path, const char* export_name, DWORD expected_exit)
{
    std::string cmd = std::string("\"") + self_exe + "\" --trap-control \"" + dll_path + "\" " + export_name;
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        check(false, "CreateProcessA for non-permissive trap control");
        return;
    }
    const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 10000);
    DWORD exit_code = 0;
    const BOOL got_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    check(wait_rc == WAIT_OBJECT_0, "non-permissive trap child exited within timeout");
    if (!got_code || exit_code != expected_exit) {
        std::printf("FAIL: %s trap expected 0x%08lx got 0x%08lx\n", export_name, static_cast<unsigned long>(expected_exit),
                    static_cast<unsigned long>(exit_code));
        ++failures;
    }
}

void check_non_permissive_traps(const char* self_exe, const char* dll_path)
{
    check_trap_case(self_exe, dll_path, "AcpHalAllocateShapeContexts", 0xE2570100u);
    check_trap_case(self_exe, dll_path, "AcpHalCreate", 0xE2570101u);
    check_trap_case(self_exe, dll_path, "ApuIsVirtualAddressValid", 0xE2570106u);
    check_trap_case(self_exe, dll_path, "ApuMapApuAddress", 0xE2570107u);
    check_trap_case(self_exe, dll_path, "ApuMapVirtualAddress", 0xE2570108u);
    check_trap_case(self_exe, dll_path, "AcpHalReleaseShapeContexts", 0xE2570102u);
}

SIZE_T pagefile_usage(SIZE_T* peak_out = nullptr)
{
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        check(false, "GetProcessMemoryInfo succeeds");
        if (peak_out) *peak_out = 0;
        return 0;
    }
    if (peak_out) *peak_out = pmc.PeakPagefileUsage;
    return pmc.PagefileUsage;
}

bool query_mem(const void* address, MEMORY_BASIC_INFORMATION& mbi)
{
    if (VirtualQuery(address, &mbi, sizeof(mbi)) == sizeof(mbi)) return true;
    check(false, "VirtualQuery succeeds");
    return false;
}

int run_commit_check(const char* dll_path)
{
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    HMODULE module = LoadLibraryA(dll_path);
    if (!module) return std::printf("FAIL: cannot load %s\n", dll_path), 2;
    alloc_fn = reinterpret_cast<AllocFn>(GetProcAddress(module, "ApuAlloc"));
    free_fn = reinterpret_cast<FreeFn>(GetProcAddress(module, "ApuFree"));
    map_apu_fn = reinterpret_cast<MapApuFn>(GetProcAddress(module, "ApuMapApuAddress"));
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(module, "ApuCreateHeap"));
    if (!alloc_fn || !free_fn || !map_apu_fn || !create) return std::printf("FAIL: missing export\n"), 2;

    // Warm up GetProcessMemoryInfo before measuring baseline commit.
    (void)pagefile_usage();
    const SIZE_T base_commit = pagefile_usage();

    constexpr u64 large_heap = 64ull * 1024 * 1024;
    constexpr u64 page_size = 0x1000;
    constexpr SIZE_T slack = 256 * 1024;
    check(create(large_heap, 0) == ok, "64 MiB ApuCreateHeap succeeds");
    const SIZE_T after_create_commit = pagefile_usage();
    check(after_create_commit <= base_commit + slack,
          "PagefileUsage stays near 0 growth after 64 MiB ApuCreateHeap");

    auto* first_addr = static_cast<unsigned char*>(map_apu_fn(first_offset));
    check(first_addr != nullptr, "ApuMapApuAddress(first_offset) returns non-null after 64 MiB ApuCreateHeap");
    if (!first_addr) return 1;
    unsigned char* const base = first_addr - first_offset;

    MEMORY_BASIC_INFORMATION mbi{};
    if (query_mem(base, mbi)) {
        check(mbi.State == MEM_RESERVE && mbi.RegionSize == large_heap,
              "entire 64 MiB heap starts in MEM_RESERVE state");
    }
    if (query_mem(first_addr, mbi)) {
        check(mbi.State == MEM_RESERVE && mbi.RegionSize == large_heap - first_offset,
              "allocatable 64 MiB range starts in MEM_RESERVE state");
    }

    // Allocate A (4 MiB + 2 KiB -> 1025 pages) and adjacent B (1 MiB + 2 KiB -> shares boundary page).
    constexpr u64 size_a = 0x400800;
    constexpr u64 size_b = 0x100800;
    void* a = take_ok(size_a, 16);
    check(a == first_addr, "first sub-block starts at first_offset");
    const SIZE_T after_a_commit = pagefile_usage();
    check(after_a_commit >= after_create_commit + 0x400000 && after_a_commit <= after_create_commit + 0x401000 + slack,
          "PagefileUsage grows only by first sub-block pages");
    if (query_mem(a, mbi)) {
        check(mbi.State == MEM_COMMIT && mbi.RegionSize >= 0x401000, "first sub-block pages are MEM_COMMIT");
    }
    if (query_mem(first_addr + 0x401000, mbi)) {
        check(mbi.State == MEM_RESERVE, "pages past first sub-block remain MEM_RESERVE");
    }
    std::memset(a, 0xA5, size_a);

    void* b = take_ok(size_b, 16);
    check(b == static_cast<unsigned char*>(a) + size_a, "second sub-block is adjacent on shared boundary page");
    const SIZE_T after_b_commit = pagefile_usage();
    check(after_b_commit <= after_a_commit + 0x100000 + slack,
          "PagefileUsage grows only by second sub-block newly committed pages");
    bool a_intact = true;
    for (u64 i = 0; i < size_a; ++i) a_intact = a_intact && static_cast<const unsigned char*>(a)[i] == 0xA5;
    check(a_intact, "first sub-block bytes remain intact after committing adjacent block on shared page");
    std::memset(b, 0x5A, size_b);

    // Free A: interior pages [0x10000, 0x410000) must return to MEM_RESERVE while shared page [0x410000, 0x411000) stays MEM_COMMIT.
    check(free_fn(a) == ok, "ApuFree first sub-block succeeds");
    if (query_mem(first_addr, mbi)) {
        check(mbi.State == MEM_RESERVE && mbi.RegionSize == 0x400000,
              "interior pages of freed first sub-block return to MEM_RESERVE");
    }
    if (query_mem(first_addr + 0x400000, mbi)) {
        check(mbi.State == MEM_COMMIT,
              "shared boundary page stays MEM_COMMIT while adjacent second block is live");
    }
    const SIZE_T after_free_a_commit = pagefile_usage();
    check(after_free_a_commit + 0x380000 <= after_b_commit,
          "PagefileUsage drops after interior freed pages are decommitted");
    if (query_mem(b, mbi)) {
        check(mbi.State == MEM_COMMIT && mbi.RegionSize >= size_b,
              "adjacent live block stays MEM_COMMIT after freeing first block");
        if (mbi.State == MEM_COMMIT) {
            bool b_intact = true;
            for (u64 i = 0; i < size_b; ++i) b_intact = b_intact && static_cast<const unsigned char*>(b)[i] == 0x5A;
            check(b_intact, "adjacent live block bytes on shared boundary page remain intact and readable");
        }
    }

    // Reallocate A while B is still live: re-commits interior pages and zeroes still-committed shared boundary page.
    void* a2 = take_ok(size_a, 16);
    check(a2 == a, "reallocated sub-block reuses freed span");
    bool a2_zero = true;
    for (u64 i = 0; i < size_a; ++i) a2_zero = a2_zero && static_cast<const unsigned char*>(a2)[i] == 0;
    check(a2_zero, "reallocated sub-block is zeroed including shared boundary page");
    bool b_still_intact = true;
    for (u64 i = 0; i < size_b; ++i) b_still_intact = b_still_intact && static_cast<const unsigned char*>(b)[i] == 0x5A;
    check(b_still_intact, "adjacent live block remains untouched when neighbor reallocates");

    check(free_fn(a2) == ok, "ApuFree reallocated first sub-block");
    check(free_fn(b) == ok, "ApuFree second sub-block");
    if (query_mem(first_addr, mbi)) {
        check(mbi.State == MEM_RESERVE && mbi.RegionSize == large_heap - first_offset,
              "entire allocatable heap returns to MEM_RESERVE after freeing all blocks");
    }
    const SIZE_T after_free_all_commit = pagefile_usage();
    check(after_free_all_commit <= after_create_commit + slack,
          "PagefileUsage returns near baseline after freeing all blocks");

    // Verify commit-failure rollback leaves the heap intact if a process commit limit can be applied.
    SIZE_T peak_commit = 0;
    (void)pagefile_usage(&peak_commit);
    if (HANDLE job = CreateJobObjectW(nullptr, nullptr)) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY;
        limits.ProcessMemoryLimit = peak_commit + 2 * 1024 * 1024;
        if (SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits) &&
            AssignProcessToJobObject(job, GetCurrentProcess())) {
            void* oom = reinterpret_cast<void*>(0x1234);
            std::uint32_t apu = 0xDEADBEEF;
            check(take(&oom, 16ull * 1024 * 1024, 16, &apu) == no_memory && !oom && apu == 0xDEADBEEF,
                  "commit failure in VirtualAlloc(MEM_COMMIT) returns E_OUTOFMEMORY cleanly");
            void* small_ok = nullptr;
            std::uint32_t small_apu = 0;
            check(take(&small_ok, page_size, 16, &small_apu) == ok && small_ok == first_addr && small_apu == first_offset,
                  "heap free list remains uncorrupted after commit failure");
            if (small_ok) check(free_fn(small_ok) == ok, "free small block after commit-failure rollback check");
        }
        CloseHandle(job);
    }

    return failures ? 1 : 0;
}

void check_on_demand_commit(const char* self_exe, const char* dll_path)
{
    std::string cmd = std::string("\"") + self_exe + "\" --commit-check \"" + dll_path + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        check(false, "CreateProcessA for on-demand commit check");
        return;
    }
    const DWORD wait_rc = WaitForSingleObject(pi.hProcess, 15000);
    DWORD exit_code = 1;
    const BOOL got_code = GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    check(wait_rc == WAIT_OBJECT_0 && got_code && exit_code == 0, "on-demand commit child check passes");
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 4 && std::strcmp(argv[1], "--trap-control") == 0) return run_trap_control(argv[2], argv[3]);
    if (argc == 3 && std::strcmp(argv[1], "--commit-check") == 0) return run_commit_check(argv[2]);
    if (argc != 2) return std::printf("usage: startup_acphal_apu_free <AcpHal.dll>\n"), 2;
    SetEnvironmentVariableA("X2W_AUDIO_PERMISSIVE", "1");
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return std::printf("FAIL: cannot load %s\n", argv[1]), 2;
    alloc_fn = reinterpret_cast<AllocFn>(GetProcAddress(module, "ApuAlloc"));
    free_fn = reinterpret_cast<FreeFn>(GetProcAddress(module, "ApuFree"));
    map_fn = reinterpret_cast<MapFn>(GetProcAddress(module, "ApuMapVirtualAddress"));
    map_apu_fn = reinterpret_cast<MapApuFn>(GetProcAddress(module, "ApuMapApuAddress"));
    valid_fn = reinterpret_cast<ValidFn>(GetProcAddress(module, "ApuIsVirtualAddressValid"));
    hal_create_fn = reinterpret_cast<HalCreateFn>(GetProcAddress(module, "AcpHalCreate"));
    shape_alloc_fn = reinterpret_cast<ShapeAllocFn>(GetProcAddress(module, "AcpHalAllocateShapeContexts"));
    shape_release_fn = reinterpret_cast<ShapeReleaseFn>(GetProcAddress(module, "AcpHalReleaseShapeContexts"));
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(module, "ApuCreateHeap"));
    if (!alloc_fn || !free_fn || !map_fn || !map_apu_fn || !valid_fn || !hal_create_fn || !shape_alloc_fn ||
        !shape_release_fn || !create)
        return std::printf("FAIL: missing export\n"), 2;
    void* none = nullptr;
    check(take(&none, 64) == no_memory && !none, "no heap yet");
    check(valid_fn(nullptr) == 0 && map_apu_fn(first_offset) == nullptr && map_fn(nullptr) == 0,
          "address queries fail cleanly before ApuCreateHeap");
    first_creation(create);
    churn_beyond_capacity();
    reuse_and_alignment();
    coalescing();
    bad_requests();
    shape_contexts();
    hal_and_address_mapping();
    table_full();
    concurrent();
    check_non_permissive_traps(argv[0], argv[1]);
    check_on_demand_commit(argv[0], argv[1]);
    std::printf(failures ? "startup_acphal_apu_free: %d failure(s)\n" : "startup_acphal_apu_free: ok\n", failures);
    return failures ? 1 : 0;
}
