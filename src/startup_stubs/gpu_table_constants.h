#pragma once

#include <windows.h>
#include <d3d12.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace x2w_gpu {
// Ring offsets survive until first execution; generations distinguish a recycled
// backing chunk. The original CPU bytes let us compare without reading GPU upload memory.
struct TableConstantSlice {
    UINT chunk;
    UINT64 generation, offset;
};
struct TableConstantRefresh {
    size_t checked = 0, changed = 0, unreadable = 0, stale = 0;
};
class TableConstants {
    struct Fill {
        const unsigned char* source;
        TableConstantSlice slice;
        size_t start, bytes;
    };
    std::vector<Fill> fills_;
    std::vector<unsigned char> original_;
public:
    // Take one snapshot for both the original descriptor and the comparison.
    const unsigned char* remember(const void* source, size_t bytes, TableConstantSlice slice)
    {
        const auto start = original_.size();
        const auto* ptr = static_cast<const unsigned char*>(source);
        original_.insert(original_.end(), ptr, ptr + bytes);
        fills_.push_back({ptr, slice, start, bytes});
        return original_.data() + start;
    }
    template<class Write> TableConstantRefresh first_submit(Write write)
    {
        TableConstantRefresh result;
        if (fills_.empty()) return result;
        static thread_local std::array<unsigned char, D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16> scratch;
        for (const auto& fill : fills_) {
            ++result.checked;
            SIZE_T read = 0;
            if (fill.bytes > scratch.size() || !ReadProcessMemory(GetCurrentProcess(), fill.source, scratch.data(), fill.bytes, &read) ||
                read != fill.bytes) {
                ++result.unreadable;
                continue;
            }
            if (std::memcmp(original_.data() + fill.start, scratch.data(), fill.bytes) == 0) continue;
            if (write(fill.slice, scratch.data(), fill.bytes)) ++result.changed;
            else ++result.stale;
        }
        clear();
        return result;
    }
    void clear() { fills_.clear(); original_.clear(); }
};
}
