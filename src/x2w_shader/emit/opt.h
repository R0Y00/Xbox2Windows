#pragma once

// Clean-up of a finished module before it is written. Every pass is exact: the module computes the same
// values and has the same effects for every input. The passes run as Module::optimise(passes); which ones
// is shader_opt() (X2W_SHADERCFG_SHADER_OPT), part of the translation and so of the shader cache's key.

#include <cstdint>

#include "x2w_shader/emit/bitcode.h"

namespace x2w::vs::bitcode::opt {

// Dead-code elimination.
inline constexpr std::uint32_t DEAD = 1;
// Constants folded, identities removed.
inline constexpr std::uint32_t FOLD = 2;
// One instruction for one value.
inline constexpr std::uint32_t REUSE = 4;
// What a branch taken says about its condition.
inline constexpr std::uint32_t FACTS = 8;
// One read for one place within a block, while nothing is stored.
inline constexpr std::uint32_t LOADS = 16;
// Not a pass over the module: the lowering of a first-lane read in a wave64 program run as two
// host waves skips the exchange that asks which half has a lane where every lane is in EXEC.
inline constexpr std::uint32_t FIRST = 32;
// Not a pass over the module either: the lowering reads the dwords of one console load by one
// call for every four of them (lower `raw_dwords`).
inline constexpr std::uint32_t WIDE = 64;
// The passes over the module.
inline constexpr std::uint32_t MODULE = DEAD | FOLD | REUSE | FACTS | LOADS;
// Every pass.
inline constexpr std::uint32_t ALL = MODULE | FIRST | WIDE;
// The passes that run when nothing says which: every one but WIDE.
inline constexpr std::uint32_t DEFAULT = ALL & ~WIDE;

// What the instructions a module writes use, of the features a shader declares by flag (the
// validator's "flags must match usage").
struct Usage {
    // A wave or quad operation.
    bool wave_ops = false;
    // A rasterizer ordered view.
    bool rovs = false;
    // A 64-bit integer, as a result or an operand.
    bool int64 = false;
    // A typed load of several components from a UAV.
    bool typed_uav_loads = false;
    // A handle from the view heap.
    bool heap_views = false;
    // A handle from the sampler heap.
    bool heap_samplers = false;
    // A 64-bit atomic.
    bool atomic64 = false;
    bool operator==(const Usage&) const = default;
};

// What a call does besides returning its result.
enum class Effect : std::uint8_t {
    // Nothing, and the result depends on the arguments alone for the whole run of the shader.
    Pure,
    // Nothing, but the result depends on memory a store may change (buffer and texture reads).
    Reads,
    // Nothing, but the result depends on which lanes run the instruction (wave and quad ops).
    Lanes,
    // Stores, atomics, barriers, outputs, discards: always kept.
    Writes,
};

Effect effect(const Func& f);

// The comparison that holds exactly when p does not (floats 0-15, integers 32-41).
std::uint8_t opposite(std::uint8_t p);

}  // namespace x2w::vs::bitcode::opt
