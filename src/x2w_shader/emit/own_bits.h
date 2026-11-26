#pragma once

// A wave OR whose bits a loop then visits one at a time, each lane testing its own bit: a host wave may
// take over its own lanes wherever they stand. X2W_SHADERCFG_WAVE_OWN_BITS=0 turns this off.

#include <set>

#include "x2w_shader/ir.h"

namespace x2w::vs::structure {
struct Structured;
}

namespace x2w::vs::own_bits {

// Sets of values established by find.
struct OwnBits {
    // Wave ORs (the values their sections leave) a host wave may take over its own lanes.
    std::set<ir::Value> totals;
    // VGPR values of those loops that every lane of a host wave holds alike (the bits still to
    // visit): a first-lane read of one is the lane's own value.
    std::set<ir::Value> alike;
};

OwnBits find(const ir::Program& p, const structure::Structured& st, bool why);

}  // namespace x2w::vs::own_bits
