// LOCAL stall: probe_readable with and without its unreadable-range cache (local_log.h, X2W_PROBE_CACHE).
//
//   startup_probe_cache <cache|nocache>
//   (cache: X2W_PROBE_CACHE=1 X2W_PROBE_CACHE_MS=100)
//
// A 1 MiB reservation with its first 64 KiB committed. First-chance access violations are counted by a vectored
// handler (each one is what costs a debugger round trip in a run).
//   1. 100 probes of [base, base + 128 KiB): 64 KiB readable every time. Violations: cache 1, nocache 100.
//   2. 100 probes at base + 512 KiB (reserved): 0 readable. Violations added: cache 0 (same region), nocache 100.
//   3. +64 KiB committed, no invalidation, within the expiry: cache still says 64 KiB (the stale direction is
//      "unreadable", never "readable"); nocache 128 KiB.
//   4. invalidate_probe_cache(): cache 128 KiB.
//   5. Expiry: a probe at base + 512 KiB faults once (cached), again 0 faults, then after 150 ms faults once more.
//   6. The reservation released (free memory): 0 readable, no crash.
// Exit 0 pass, other values name the step.
#include "startup_stubs/local_log.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
volatile LONG violations;
LONG CALLBACK count_violation(EXCEPTION_POINTERS* e)
{
    if (e->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) InterlockedIncrement(&violations);
    return EXCEPTION_CONTINUE_SEARCH;
}
int failures;
void expect(const char* what, long long got, long long want)
{
    std::printf("%s: %lld (want %lld)%s\n", what, got, want, got == want ? "" : "  <-- FAIL");
    if (got != want) ++failures;
}
}  // namespace

// More live textures than cache slots. Their bases are mapped, but full uploads cross a missing page.
bool streamed_texture_prefixes()
{
    constexpr SIZE_T page = 64u << 10, stride = 4 * page, count = 96;
    auto* memory = static_cast<BYTE*>(VirtualAlloc(nullptr, stride * count, MEM_RESERVE, PAGE_READWRITE));
    if (!memory) return false;
    bool ok = true;
    for (SIZE_T i = 0; i < count; ++i)
        ok &= VirtualAlloc(memory + i * stride, (1 + i % 3) * page, MEM_COMMIT, PAGE_READWRITE) != nullptr;
    const auto sweep = [&](bool texture) {
        const LONG before = violations;
        for (SIZE_T i = 0; i < count; ++i) {
            const auto at = reinterpret_cast<std::uintptr_t>(memory + i * stride);
            const auto bytes = texture ? x2w_local::probe_descriptor_address(at, stride)
                                       : x2w_local::probe_readable(at, stride);
            ok &= bytes == (1 + i % 3) * page;
        }
        return violations - before;
    };
    x2w_local::invalidate_probe_cache();
    if (ok) {
        expect("ordinary first pass over sparse textures", sweep(false), count);
        x2w_local::invalidate_probe_cache();
        expect("texture upload across missing interior pages", sweep(true), 0);
        x2w_local::invalidate_probe_cache();
        expect("repeated texture uploads after cache eviction", sweep(true), 0);
        ok &= VirtualAlloc(memory + page, page, MEM_COMMIT, PAGE_READWRITE) != nullptr;
        x2w_local::invalidate_probe_cache();
        expect("new texture page included", x2w_local::probe_descriptor_address(reinterpret_cast<std::uintptr_t>(memory), stride), 2 * page);
    }
    VirtualFree(memory, 0, MEM_RELEASE);
    x2w_local::invalidate_probe_cache();
    return ok;
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode != "cache" && mode != "nocache" && mode != "query" && mode != "lazy") return 2;
    const bool cache = mode != "nocache";
    const bool query = mode == "query";
    expect("query-first eligibility", x2w_local::probe_query_first(), query);
    if (x2w_local::probe_cache_on() != cache) {
        std::printf("X2W_PROBE_CACHE is %s, mode %s\n", x2w_local::probe_cache_on() ? "on" : "off", mode.c_str());
        return 3;
    }
    AddVectoredExceptionHandler(1, count_violation);
    auto* base = static_cast<unsigned char*>(VirtualAlloc(nullptr, 1u << 20, MEM_RESERVE, PAGE_READWRITE));
    if (!base || !VirtualAlloc(base, 64u << 10, MEM_COMMIT, PAGE_READWRITE)) return 4;
    std::memset(base, 1, 64u << 10);
    const auto b = reinterpret_cast<std::uintptr_t>(base);

    if (query) {
        if (!streamed_texture_prefixes()) return 6;
        violations = 0;
        // A normal pointer still uses first-touch probing even with the switch on.
        expect("ordinary reserved prefix", x2w_local::probe_readable(b + (512u << 10), 256), 0);
        expect("ordinary probe retained", violations, 1);
        x2w_local::invalidate_probe_cache();
        auto* candidate = base + (512u << 10);
        bool missing = true;
        for (int i = 0; i < 100; ++i) {
            x2w_local::invalidate_probe_cache();
            missing &= x2w_local::probe_readable(b + (512u << 10), 256) == 0;
        }
        expect("ordinary repeated missing page", missing, true);
        expect("ordinary retries after invalidation add no exceptions", violations, 1);
        x2w_local::invalidate_probe_cache();
        expect("different page in the same reserved region", x2w_local::probe_readable(b + (640u << 10), 256), 0);
        expect("ordinary first touch retained for a different page", violations, 2);
        if (!VirtualAlloc(candidate, 4096, MEM_COMMIT, PAGE_READWRITE)) return 5;
        x2w_local::invalidate_probe_cache();
        expect("ordinary newly committed page", x2w_local::probe_readable(b + (512u << 10), 256), 256);
        DWORD old_protect = 0;
        if (!VirtualProtect(candidate, 4096, PAGE_READWRITE | PAGE_GUARD, &old_protect)) return 5;
        expect("guarded page is unreadable", x2w_local::probe_readable(b + (512u << 10), 256), 0);
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(candidate, &info, sizeof(info))) return 5;
        expect("guard is restored", (info.Protect & PAGE_GUARD) != 0, true);
        if (!VirtualFree(candidate, 4096, MEM_DECOMMIT)) return 5;
        x2w_local::invalidate_probe_cache();
        violations = 0;
    }
    bool all = true;
    for (int i = 0; i < 100; ++i) all = all && x2w_local::probe_descriptor_address(b, 128u << 10) == (64u << 10);
    expect("1. 100 probes of 128 KiB, all 64 KiB readable", all, 1);
    expect("1. access violations", violations, query ? 0 : cache ? 1 : 100);

    LONG before = violations;
    all = true;
    for (int i = 0; i < 100; ++i) all = all && x2w_local::probe_descriptor_address(b + (512u << 10), 4096) == 0;
    expect("2. 100 probes in the reserved part, all 0", all, 1);
    expect("2. access violations added", violations - before, cache ? 0 : 100);

    if (!VirtualAlloc(base + (64u << 10), 64u << 10, MEM_COMMIT, PAGE_READWRITE)) return 5;
    expect("3. committed +64 KiB, no invalidation: readable", static_cast<long long>(x2w_local::probe_descriptor_address(b, 128u << 10)),
           cache ? (64u << 10) : (128u << 10));
    x2w_local::invalidate_probe_cache();
    expect("4. after invalidate_probe_cache: readable", static_cast<long long>(x2w_local::probe_descriptor_address(b, 128u << 10)), 128u << 10);

    x2w_local::invalidate_probe_cache();
    before = violations;
    x2w_local::probe_descriptor_address(b + (512u << 10), 4096);
    expect("5. first probe after invalidation: violations", violations - before, query ? 0 : 1);
    before = violations;
    x2w_local::probe_descriptor_address(b + (512u << 10), 4096);
    expect("5. second probe: violations", violations - before, cache ? 0 : 1);
    Sleep(150);
    before = violations;
    x2w_local::probe_descriptor_address(b + (512u << 10), 4096);
    expect("5. after 150 ms (expiry 100 ms): violations", violations - before, query ? 0 : 1);

    if (mode == "query" || mode == "cache") {
        LARGE_INTEGER t0, t1, freq;
        QueryPerformanceFrequency(&freq);
        unsigned mismatches = 0;
        const LONG initial_faults = violations;
        QueryPerformanceCounter(&t0);
        for (int i = 0; i < 20000; ++i) {
            x2w_local::invalidate_probe_cache();
            if (x2w_local::probe_descriptor_address(b + (512u << 10), 256)) ++mismatches;
        }
        QueryPerformanceCounter(&t1);
        expect("reserved candidate mismatches", mismatches, 0);
        expect("reserved candidate exceptions", violations - initial_faults, query ? 0 : 20000);
        std::printf("reserved candidate benchmark: 20000 queries %.2f ms\n",
                    1000.0 * (t1.QuadPart - t0.QuadPart) / freq.QuadPart);
        QueryPerformanceCounter(&t0);
        for (int i = 0; i < 20000; ++i) {
            if (x2w_local::probe_descriptor_address(b, 256) != 256) ++mismatches;
        }
        QueryPerformanceCounter(&t1);
        expect("committed candidate mismatches", mismatches, 0);
        std::printf("committed candidate benchmark: 20000 probes %.2f ms\n",
                    1000.0 * (t1.QuadPart - t0.QuadPart) / freq.QuadPart);
    }
    VirtualFree(base, 0, MEM_RELEASE);
    x2w_local::invalidate_probe_cache();
    expect("6. released: readable", static_cast<long long>(x2w_local::probe_descriptor_address(b, 4096)), 0);

    const x2w_local::ProbeCache& c = x2w_local::probe_cache();
    std::printf("cache counters: hits %lld faults %lld queries %lld (%.3f ms total) ranges kept %lld\n", c.hits, c.faults, c.queries,
                [&] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return static_cast<double>(c.query_ticks) * 1000.0 / static_cast<double>(f.QuadPart); }(),
                c.cached);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 16 : 0;
}
