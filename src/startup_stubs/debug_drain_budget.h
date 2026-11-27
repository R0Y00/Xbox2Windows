// X2W_D3D12_DEBUG=1 log budget: the debug-layer drain keeps up to 4000 error/corruption lines and 400 others,
// at most 32 lines per message id, then one "repeated N times" line at each power of two.
// consume_map_count: a desktop Unmap is forwarded only for a counted, successful desktop Map.
#pragma once
#include <cstdint>
#include <unordered_map>

namespace x2w_startup {
inline bool consume_map_count(std::uint64_t& count) { if (!count) return false; --count; return true; }
class DebugDrainBudget {
    std::unordered_map<unsigned, std::uint64_t> counts_;
    unsigned errors_ = 0, other_ = 0;
public:
    enum class Decision { suppress, message, repeat };
    Decision record(unsigned severity, unsigned id) {
        const auto count = ++counts_[id];
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
