#pragma once

// Jump threading for refused programs: edges along which a register's value is a known constant, so that
// the program can be retried with those edges threaded (ir::thread_edges).

#include <cstddef>
#include <tuple>
#include <vector>

#include "x2w_shader/ir.h"

namespace x2w::vs::structure {
struct Structured;
}

namespace x2w::vs::threading {

// (predecessor, join, destination) triples, as ir::thread_edges takes them: along the edge from
// `predecessor` into `join`, the join's closing branch goes to `destination` (its taken or
// not-taken successor). At most one edge per join.
std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> constant_edges(const ir::Program& p,
                                                                               const structure::Structured& st);

}  // namespace x2w::vs::threading
