#pragma once

// Counts of a module's instructions by kind (a diagnostic): what a translated program costs before the
// driver's compiler sees it.
//
// X2W_SHADERCFG_SHADER_CENSUS (any value) turns the counting on; the counts of the module a thread
// wrote last are read with take().

#include <array>
#include <cstdint>
#include <optional>
#include <span>

#include "x2w_shader/emit/bitcode.h"

namespace x2w::vs::bitcode::census {

// The kinds, in the order of Census::counts.
inline constexpr std::array<const char*, 26> KINDS = {
    "total",     "arith",        "math_call",    "bitcast",  "cast",         "select",  "guard_select",
    "phi",       "extract",      "buffer_load",  "cbuffer_load", "texture_read", "lds_load", "buffer_store",
    "lds_store", "atomic",       "handle",       "annotate", "dimensions",   "barrier", "wave_op",
    "io_other",  "gep",          "clamp",        "loop_counter", "blocks",
};

// Instruction counts of one written module, by KINDS.
struct Census {
    std::array<std::uint32_t, KINDS.size()> counts{};
    bool operator==(const Census&) const = default;
};

// Whether modules are counted (X2W_SHADERCFG_SHADER_CENSUS set).
bool enabled();

// The counts of the module whose bitcode the container `dxil` holds, if this thread wrote it
// since the last call (which forgets the others).
std::optional<Census> take(std::span<const std::uint8_t> dxil);

// Counts m, whose written form is `bitcode`, and keeps the result for take().
void note(const Module& m, std::span<const std::uint8_t> bitcode);

// The kind (index into KINDS) of a call of the function `name`.
std::size_t kind(std::string_view name);

}  // namespace x2w::vs::bitcode::census
