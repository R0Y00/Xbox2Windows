// Checks the negative cache of refused buffer memory SRVs (device slot 70, src/startup_stubs/srv_refusal_cache.h)
// without a device. A refusal decided by the request and the stand-in state alone (memory independent) is reused for
// the same request and placement generation. A refusal that depends on what is committed at the address is never
// reused by the default lookup: pages can move with the same committed total (reserve 128 KiB, commit the upper
// 64 KiB, refuse; decommit it, commit the lower 64 KiB, retry at once), which no region stamp can see.
// --mutate generation: expect a changed placement generation to still hit.
// --mutate memory: expect the default lookup to reuse a memory dependent refusal (the page-move case).
// --mutate key: expect a different view to hit an entry noted for another view.
// --mutate device: expect the same request on another device to hit an entry noted for the first device.
// --mutate age: expect an entry older than the age limit to still hit.
// --mutate hits: expect an entry used more than the hit limit to still hit.
// --mutate invalid: expect a memory dependent entry with an unreadable stamp to hit under the opt-in.
// Each mutated run is a negative control and must fail.
#include "startup_stubs/srv_refusal_cache.h"

#include <cstdio>
#include <cstring>

namespace {

int failures;
const char* mutation = "";

bool mutated(const char* name)
{
    return std::strcmp(mutation, name) == 0;
}

void check(const char* name, bool condition)
{
    if (condition) return;
    ++failures;
    std::printf("FAIL %s\n", name);
}

x2w_startup::SrvRefusalKey request(std::uint64_t address, std::uint64_t first_element)
{
    x2w_startup::SrvRefusalKey key;
    key.device = 0x5000;
    key.address = address;
    key.span_bytes = 0x4000;
    key.view_a = 0x1234;
    key.view_b = first_element;
    key.view_c = 64;
    key.resource_a = 1;
    key.resource_b = 0x10000;
    return key;
}

x2w_startup::SrvRefusalStamp stamp(std::uint64_t generation, std::uint64_t commit)
{
    x2w_startup::SrvRefusalStamp s;
    s.valid = true;
    s.generation = generation;
    s.base = 0x7000000000ull;
    s.size = 0x20000;  // 128 KiB reservation
    s.commit = commit;
    s.flags = 1;
    return s;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];
    using namespace x2w_startup;
    const std::uint64_t kHalf = 0x10000;  // 64 KiB

    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        check("an empty cache misses", !cache.hit(key, stamp(5, 0), 100));
        cache.note(key, stamp(5, 0), 100, true);
        check("a memory independent refusal hits for the same request and generation", cache.hit(key, stamp(5, 0), 150));
        check("a hit repeats", cache.hit(key, stamp(5, 0), 160));
        // The commit state is irrelevant to such a refusal: it was decided without reading memory.
        check("a memory independent refusal ignores the committed total", cache.hit(key, stamp(5, kHalf), 170));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0), 100, true);
        const bool hit = cache.hit(key, stamp(6, 0), 110);
        check("a new placement generation misses (a resource may now cover the address)", mutated("generation") ? hit : !hit);
    }
    {
        // Reported sequence: reserve 128 KiB, commit the upper 64 KiB, the slot-70 request refuses; decommit the
        // upper half, commit and fill the lower half (same base, same reservation, same committed total), retry at once.
        SrvRefusalCache cache;
        const auto key = request(0x7000000000ull, 0);
        const auto upper_committed = stamp(5, kHalf);
        const auto lower_committed = stamp(5, kHalf);  // identical stamp: the move is invisible
        cache.note(key, upper_committed, 100, false);
        bool hit = cache.hit(key, lower_committed, 101);
        if (mutated("memory")) hit = cache.hit(key, lower_committed, 101, true) || hit;
        check("the page move with an equal stamp runs the full path", !hit);
        check("a memory dependent refusal never hits by default", !cache.hit(key, upper_committed, 102));
        // Opt-in accepts exactly this staleness, so it must be off unless asked for.
        check("the opt-in reuses a memory dependent refusal for an equal stamp", cache.hit(key, upper_committed, 103, true));
        check("the opt-in still misses on a different committed total", !cache.hit(key, stamp(5, 2 * kHalf), 104, true));
        auto moved = stamp(5, kHalf);
        moved.base += 0x10000;
        check("the opt-in misses on another base", !cache.hit(key, moved, 104, true));
        auto flagged = stamp(5, kHalf);
        flagged.flags = 2;
        check("the opt-in misses on other flags", !cache.hit(key, flagged, 104, true));
        check("the opt-in misses on a new generation", !cache.hit(key, stamp(6, kHalf), 104, true));
    }
    {
        SrvRefusalCache cache;
        cache.note(request(0x7000001000ull, 0), stamp(5, 0), 100, true);
        const bool other_view = cache.hit(request(0x7000001000ull, 8), stamp(5, 0), 110);
        check("another view of the same address misses", mutated("key") ? other_view : !other_view);
        check("another address misses", !cache.hit(request(0x7000002000ull, 0), stamp(5, 0), 110));
        auto other_device = request(0x7000001000ull, 0);
        other_device.device += 0x100;
        const bool device_hit = cache.hit(other_device, stamp(5, 0), 110);
        check("the same request on another device misses (placement lookups are per device)",
              mutated("device") ? device_hit : !device_hit);
        auto wider = request(0x7000001000ull, 0);
        wider.span_bytes += 4;
        check("another span misses", !cache.hit(wider, stamp(5, 0), 110));
        auto resource = request(0x7000001000ull, 0);
        resource.resource_b += 1;
        check("another resource width misses", !cache.hit(resource, stamp(5, 0), 110));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0), 100, true);
        const bool old_hit = cache.hit(key, stamp(5, 0), 100 + SrvRefusalCache::kMaxAgeMs + 1);
        check("an entry older than the age limit misses", mutated("age") ? old_hit : !old_hit);
        check("an entry exactly at the age limit still hits", [&] {
            SrvRefusalCache fresh;
            fresh.note(key, stamp(5, 0), 100, true);
            return fresh.hit(key, stamp(5, 0), 100 + SrvRefusalCache::kMaxAgeMs);
        }());
        check("a clock that went backwards misses", [&] {
            SrvRefusalCache fresh;
            fresh.note(key, stamp(5, 0), 100, true);
            return !fresh.hit(key, stamp(5, 0), 50);
        }());
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0), 100, true);
        bool all = true;
        for (std::uint32_t i = 0; i < SrvRefusalCache::kMaxHits; ++i) all = cache.hit(key, stamp(5, 0), 110) && all;
        check("the entry hits up to the hit limit", all);
        const bool over = cache.hit(key, stamp(5, 0), 110);
        check("past the hit limit the full path runs again", mutated("hits") ? over : !over);
        cache.note(key, stamp(5, 0), 120, true);  // the full path refused again and re-noted
        check("a re-noted entry hits again", cache.hit(key, stamp(5, 0), 121));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        SrvRefusalStamp unreadable = stamp(5, kHalf);
        unreadable.valid = false;
        cache.note(key, unreadable, 100, false);
        const bool hit = cache.hit(key, unreadable, 110, true);
        check("a memory dependent refusal with an unreadable stamp is never cached", mutated("invalid") ? hit : !hit);
        // An unreadable stamp does not matter to a memory independent refusal.
        cache.note(key, unreadable, 100, true);
        check("a memory independent refusal does not need the stamp", cache.hit(key, unreadable, 110));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0), 100, true);
        cache.clear();
        check("clear forgets everything", !cache.hit(key, stamp(5, 0), 110));
        const auto before = srv_placement_generation().load();
        note_placement_change();
        check("note_placement_change advances the generation", srv_placement_generation().load() == before + 1);
    }

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("srv refusal cache checks passed\n");
    return 0;
}
