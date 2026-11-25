#pragma once

// Descriptor tables of the compute root of one command list: command-list slot 35 (desktop 31 + 4,
// SetComputeRootDescriptorTable). Slot 33 (SetComputeRootSignature) keeps its own desktop bind; it only tells this
// state which root signature is current. Tables are recorded for a later Dispatch handler and are not forwarded.
//
// The rules follow the graphics side (slots 34 and 36): setting the root signature that is already set keeps the
// tables, setting a different one (or none) leaves no table bound, and so does a change of the title's view heap.
// A parameter past the kept range is refused and counted.

#include <cstdint>

#include "startup_stubs/logical_view_table.h"

namespace x2w_startup {

class ComputeTables {
public:
    static constexpr unsigned max_parameters = 16;  // same range as the graphics tables

    // The root signature is identified by any stable pointer (the title wrapper's parameter list); null clears it.
    void set_signature(const void* signature)
    {
        if (signature != signature_) clear();
        signature_ = signature;
    }

    // Slot 32 with another view heap: the recorded addresses belong to the old one.
    void clear()
    {
        for (auto& table : tables_) table = {};
    }

    // False when the parameter is out of range (counted).
    bool set_table(unsigned parameter, const ViewTable& table)
    {
        if (parameter >= max_parameters) {
            ++out_of_range_;
            return false;
        }
        tables_[parameter] = table;
        return true;
    }

    const void* signature() const { return signature_; }
    ViewTable table(unsigned parameter) const { return parameter < max_parameters ? tables_[parameter] : ViewTable{}; }
    std::uint32_t out_of_range_count() const { return out_of_range_; }

private:
    const void* signature_ = nullptr;
    ViewTable tables_[max_parameters] = {};
    std::uint32_t out_of_range_ = 0;
};

}  // namespace x2w_startup
