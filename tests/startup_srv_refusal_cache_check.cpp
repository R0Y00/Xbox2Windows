// Checks the negative cache of refused buffer memory SRVs (device slot 70, src/startup_stubs/srv_refusal_cache.h)
// without a device: a refusal is reused only for the same request, placement generation and memory stamp, and
// only while it is young and lightly used.
// --mutate generation: expect a changed placement generation to still hit.
// --mutate commit: expect a changed committed-byte count of the allocation to still hit.
// --mutate key: expect a different view to hit an entry noted for another view.
// --mutate device: expect the same request on another device to hit an entry noted for the first device.
// --mutate age: expect an entry older than the age limit to still hit.
// --mutate hits: expect an entry used more than the hit limit to still hit.
// --mutate invalid: expect a stamp that could not be read to be cached.
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
    s.size = 0x200000;
    s.commit = commit;
    s.flags = 1;
    return s;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) mutation = argv[2];
    using namespace x2w_startup;

    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        check("an empty cache misses", !cache.hit(key, stamp(5, 0x1000), 100));
        cache.note(key, stamp(5, 0x1000), 100);
        check("the same request, generation and stamp hit", cache.hit(key, stamp(5, 0x1000), 150));
        check("a hit repeats", cache.hit(key, stamp(5, 0x1000), 160));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0x1000), 100);
        const bool hit = cache.hit(key, stamp(6, 0x1000), 110);
        check("a new placement generation misses (a resource may now cover the address)", mutated("generation") ? hit : !hit);
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0x1000), 100);
        const bool hit = cache.hit(key, stamp(5, 0x2000), 110);
        check("more committed bytes in the allocation miss (the buffer may now be readable)", mutated("commit") ? hit : !hit);
        auto moved = stamp(5, 0x1000);
        moved.base += 0x10000;
        check("another allocation base misses", !cache.hit(key, moved, 110));
        auto resized = stamp(5, 0x1000);
        resized.size += 0x10000;
        check("another allocation size misses", !cache.hit(key, resized, 110));
        auto flagged = stamp(5, 0x1000);
        flagged.flags = 2;
        check("other allocation flags miss", !cache.hit(key, flagged, 110));
    }
    {
        SrvRefusalCache cache;
        cache.note(request(0x7000001000ull, 0), stamp(5, 0x1000), 100);
        const bool other_view = cache.hit(request(0x7000001000ull, 8), stamp(5, 0x1000), 110);
        check("another view of the same address misses", mutated("key") ? other_view : !other_view);
        check("another address misses", !cache.hit(request(0x7000002000ull, 0), stamp(5, 0x1000), 110));
        auto other_device = request(0x7000001000ull, 0);
        other_device.device += 0x100;
        const bool device_hit = cache.hit(other_device, stamp(5, 0x1000), 110);
        check("the same request on another device misses (placement lookups are per device)",
              mutated("device") ? device_hit : !device_hit);
        auto wider = request(0x7000001000ull, 0);
        wider.span_bytes += 4;
        check("another span misses", !cache.hit(wider, stamp(5, 0x1000), 110));
        auto resource = request(0x7000001000ull, 0);
        resource.resource_b += 1;
        check("another resource width misses", !cache.hit(resource, stamp(5, 0x1000), 110));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0x1000), 100);
        const bool old_hit = cache.hit(key, stamp(5, 0x1000), 100 + SrvRefusalCache::kMaxAgeMs + 1);
        check("an entry older than the age limit misses", mutated("age") ? old_hit : !old_hit);
        check("an entry exactly at the age limit still hits", [&] {
            SrvRefusalCache fresh;
            fresh.note(key, stamp(5, 0x1000), 100);
            return fresh.hit(key, stamp(5, 0x1000), 100 + SrvRefusalCache::kMaxAgeMs);
        }());
        check("a clock that went backwards misses", [&] {
            SrvRefusalCache fresh;
            fresh.note(key, stamp(5, 0x1000), 100);
            return !fresh.hit(key, stamp(5, 0x1000), 50);
        }());
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0x1000), 100);
        bool all = true;
        for (std::uint32_t i = 0; i < SrvRefusalCache::kMaxHits; ++i) all = cache.hit(key, stamp(5, 0x1000), 110) && all;
        check("the entry hits up to the hit limit", all);
        const bool over = cache.hit(key, stamp(5, 0x1000), 110);
        check("past the hit limit the full path runs again", mutated("hits") ? over : !over);
        cache.note(key, stamp(5, 0x1000), 120);  // the full path refused again and re-noted
        check("a re-noted entry hits again", cache.hit(key, stamp(5, 0x1000), 121));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        SrvRefusalStamp unreadable = stamp(5, 0x1000);
        unreadable.valid = false;
        cache.note(key, unreadable, 100);
        const bool hit = cache.hit(key, unreadable, 110);
        check("a stamp that could not be read is never cached", mutated("invalid") ? hit : !hit);
        check("and never hits", !cache.hit(key, unreadable, 110));
    }
    {
        SrvRefusalCache cache;
        const auto key = request(0x7000001000ull, 0);
        cache.note(key, stamp(5, 0x1000), 100);
        cache.clear();
        check("clear forgets everything", !cache.hit(key, stamp(5, 0x1000), 110));
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
