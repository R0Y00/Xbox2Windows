#pragma once

// A bounded negative cache for refused buffer memory SRVs (device slot 70, reason buffer_unavailable).
//
// A refused SRV used to repeat the whole resolution on every call: the placement scans, the resource index
// scan, and the memory questions of placed_buffer_range.h (a region-information call and, for partly committed
// or mapped allocations, a VirtualQuery walk that costs milliseconds per GiB). A title that re-publishes the same
// view every frame paid that for every refusal.
//
// The answer is not remembered blindly: placed_buffer_range.h explains why title memory may change behind the
// stand-ins. A refusal is reused only while all of this still holds:
//   - the same request: address, span and every view / resource field the resolution reads (SrvRefusalKey);
//   - the same stand-in state: the placement generation, which advances whenever a resource is indexed, retired,
//     published or released, or an unplaced buffer is created or dropped (note_placement_change);
//   - the same memory state: one cheap region-information stamp of the allocation holding the address (base, size,
//     committed bytes, flags; no page scan). A stamp that could not be read is never cached;
//   - an entry younger than kMaxAgeMs and used fewer than kMaxHits times, so a change the stamp cannot see (the
//     same number of pages committed elsewhere) is picked up after at most that long.
// A miss runs the full resolution unchanged. The only wrong answer a stale hit can give is a refusal where the
// full path would now publish, delayed by at most kMaxAgeMs; it can never publish a view.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace x2w_startup {

inline std::atomic<std::uint64_t>& srv_placement_generation()
{
    static std::atomic<std::uint64_t> generation{1};
    return generation;
}

// Call after any change of what existing_memory_buffer / unplaced_memory_buffer can find.
inline void note_placement_change()
{
    srv_placement_generation().fetch_add(1, std::memory_order_acq_rel);
}

struct SrvRefusalKey {
    std::uint64_t address = 0;
    std::uint64_t span_bytes = 0;
    std::uint64_t view_a = 0;      // format, dimension, component mapping
    std::uint64_t view_b = 0;      // first element
    std::uint64_t view_c = 0;      // element count, stride, flags
    std::uint64_t resource_a = 0;  // dimension, flags
    std::uint64_t resource_b = 0;  // width
    bool operator==(const SrvRefusalKey& o) const
    {
        return address == o.address && span_bytes == o.span_bytes && view_a == o.view_a && view_b == o.view_b &&
               view_c == o.view_c && resource_a == o.resource_a && resource_b == o.resource_b;
    }
};

struct SrvRefusalStamp {
    bool valid = false;  // the region information could be read
    std::uint64_t generation = 0;
    std::uint64_t base = 0;
    std::uint64_t size = 0;
    std::uint64_t commit = 0;
    std::uint64_t flags = 0;
    bool operator==(const SrvRefusalStamp& o) const
    {
        return valid && o.valid && generation == o.generation && base == o.base && size == o.size &&
               commit == o.commit && flags == o.flags;
    }
};

class SrvRefusalCache {
public:
    static constexpr std::size_t kSlots = 256;
    static constexpr std::uint64_t kMaxAgeMs = 250;
    static constexpr std::uint32_t kMaxHits = 1024;

    // True when this request was refused under the same stand-in and memory state a moment ago.
    bool hit(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms)
    {
        if (!stamp.valid) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        Entry& e = entries_[slot(key)];
        if (!e.used || !(e.key == key) || !(e.stamp == stamp) || now_ms < e.noted_ms ||
            now_ms - e.noted_ms > kMaxAgeMs || e.hits >= kMaxHits) {
            return false;
        }
        ++e.hits;
        return true;
    }

    // Remember a refusal the full resolution just produced.
    void note(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms)
    {
        if (!stamp.valid) return;
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[slot(key)] = Entry{true, key, stamp, now_ms, 0};
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (Entry& e : entries_) e = Entry{};
    }

private:
    struct Entry {
        bool used = false;
        SrvRefusalKey key;
        SrvRefusalStamp stamp;
        std::uint64_t noted_ms = 0;
        std::uint32_t hits = 0;
    };

    static std::size_t slot(const SrvRefusalKey& k)
    {
        std::uint64_t h = 1469598103934665603ull;
        for (std::uint64_t v : {k.address, k.span_bytes, k.view_a, k.view_b, k.view_c, k.resource_a, k.resource_b})
            h = (h ^ v) * 1099511628211ull;
        return static_cast<std::size_t>((h ^ (h >> 32)) % kSlots);
    }

    std::mutex mutex_;
    std::array<Entry, kSlots> entries_{};
};

} // namespace x2w_startup
