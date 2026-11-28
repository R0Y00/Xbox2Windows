// X2W_D3D12_DEBUG=1 log budget: the debug-layer drain keeps up to 4000 error/corruption lines and 400 others,
// at most 32 lines per message id, then one "repeated N times" line at each power of two.
// consume_map_count: a desktop Unmap is forwarded only for a counted, successful desktop Map.
#pragma once
#include <cstdint>
#include <new>
#include <unordered_map>

namespace x2w_startup {
inline bool consume_map_count(std::uint64_t& count) { if (!count) return false; --count; return true; }

class ResourceMapCounts {
    using Subresources = std::unordered_map<unsigned, std::uint64_t>;
    std::unordered_map<const void*, Subresources> resources_;

public:
    void record(const void* resource, unsigned subresource) {
        auto entry = resources_.find(resource);
        if (entry == resources_.end()) {
            resources_.emplace(resource, Subresources{{subresource, 1}});
        } else {
            ++entry->second[subresource];
        }
    }

    bool consume(const void* resource, unsigned subresource) {
        auto entry = resources_.find(resource);
        if (entry == resources_.end()) return false;
        auto& subresources = entry->second;
        auto count = subresources.find(subresource);
        if (count == subresources.end() || !consume_map_count(count->second)) return false;
        if (!count->second) subresources.erase(count);
        if (subresources.empty()) resources_.erase(entry);
        return true;
    }

    void retire(const void* resource) { resources_.erase(resource); }
};

class DebugDrainBudget {
    std::unordered_map<unsigned, std::uint64_t> counts_;
    unsigned errors_ = 0, other_ = 0;
public:
    enum class Decision { suppress, message, repeat };
    // Best effort: when the per-id counter cannot be allocated the line is dropped, never thrown to the caller.
    Decision record(unsigned severity, unsigned id) noexcept {
        std::uint64_t count;
        try {
            count = ++counts_[id];
        } catch (const std::bad_alloc&) {
            return Decision::suppress;
        }
        if (count > 32) return (count & (count - 1)) == 0 ? Decision::repeat : Decision::suppress;
        unsigned& used = severity <= 1 ? errors_ : other_;
        const unsigned limit = severity <= 1 ? 4000 : 400;
        if (used >= limit) return Decision::suppress;
        ++used;
        return Decision::message;
    }
    std::uint64_t count(unsigned id) const { auto it = counts_.find(id); return it == counts_.end() ? 0 : it->second; }
};
}  // namespace x2w_startup
