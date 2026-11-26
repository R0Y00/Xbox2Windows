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
//   - an entry younger than kMaxAgeMs and used fewer than kMaxHits times.
// Two kinds of refusal exist, and only one is exact:
//   - memory independent: decided by the request and by stand-in state alone (a live resource overlaps, the span has no
//     page extent, a placed buffer owns the page, the found buffer is too small for the view). The key and the
//     placement generation prove it, so it is cached by default;
//   - memory dependent: decided by what is committed at the address. Commit and decommit happen behind the stand-ins,
//     and a region-information stamp (base, size, committed bytes, flags) cannot see pages that move with the same
//     total (reserve 128 KiB, commit the upper half, refuse; decommit it, commit the lower half, retry at once: the
//     stamp is identical but the buffer is now readable). Such a refusal is cached only when the caller opts in
//     (allow_memory_dependent), accepting that a retry within kMaxAgeMs can be refused again.
// A miss runs the full resolution unchanged. The default lookup never refuses what the full path would publish.

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
    std::uint64_t device = 0;      // the device wrapper: every placement lookup is per device
    std::uint64_t address = 0;
    std::uint64_t span_bytes = 0;
    std::uint64_t view_a = 0;      // format, dimension, component mapping
    std::uint64_t view_b = 0;      // first element
    std::uint64_t view_c = 0;      // element count, stride, flags
    std::uint64_t resource_a = 0;  // dimension, flags
    std::uint64_t resource_b = 0;  // width
    bool operator==(const SrvRefusalKey& o) const
    {
        return device == o.device && address == o.address && span_bytes == o.span_bytes && view_a == o.view_a && view_b == o.view_b &&
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
    bool hit(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms,
             bool allow_memory_dependent = false)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry& e = entries_[slot(key)];
        if (!e.used || !(e.key == key) || now_ms < e.noted_ms ||
            now_ms - e.noted_ms > kMaxAgeMs || e.hits >= kMaxHits) {
            return false;
        }
        if (e.memory_independent) {
            if (e.stamp.generation != stamp.generation) return false;
        } else if (!allow_memory_dependent || !(e.stamp == stamp)) {
            return false;
        }
        ++e.hits;
        return true;
    }

    // Remember a refusal the full resolution just produced. memory_independent: it does not depend on what is
    // committed at the address (see above); the stamp's generation is then the only state it needs.
    void note(const SrvRefusalKey& key, const SrvRefusalStamp& stamp, std::uint64_t now_ms, bool memory_independent)
    {
        if (!memory_independent && !stamp.valid) return;
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[slot(key)] = Entry{true, key, stamp, now_ms, 0, memory_independent};
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
        bool memory_independent = false;
    };

    static std::size_t slot(const SrvRefusalKey& k)
    {
        std::uint64_t h = 1469598103934665603ull;
        for (std::uint64_t v : {k.device, k.address, k.span_bytes, k.view_a, k.view_b, k.view_c, k.resource_a, k.resource_b})
            h = (h ^ v) * 1099511628211ull;
        return static_cast<std::size_t>((h ^ (h >> 32)) % kSlots);
    }

    std::mutex mutex_;
    std::array<Entry, kSlots> entries_{};
};

} // namespace x2w_startup
