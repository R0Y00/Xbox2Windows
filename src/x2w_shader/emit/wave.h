#pragma once

// Whole-wave sections: code between `s_or_saveexec -1` (or `s_nand_saveexec 0`) and the restore of the
// saved mask runs in every lane of the console's wave, the inactive ones included, which is how the
// compiler expands HLSL wave reductions and prefix sums. Each section is simulated on the console's lanes
// with symbolic values, and a value it leaves is lowered by what it turned out to be: the reduction or
// prefix of one operation over the lanes' (masked) inputs.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <unordered_map>

#include "x2w_shader/ir.h"

namespace x2w::vs::wave {

// An associative operation and its identity.
enum class Op : std::uint8_t {
    AddU,
    AddF,
    // A sum of 16-bit floats in the low halves (v_add_f16), each addition rounded to 16 bits: not
    // associative, so only the balanced tree over the lanes in their order is followed (buddies).
    AddH,
    MinU,
    MaxU,
    MinI,
    MaxI,
    MinF,
    MaxF,
    Or,
    And,
    Xor,
};

// The operation a VALU instruction name stands for.
std::optional<Op> op_of(std::string_view name);
// Whether k leaves the other operand unchanged.
bool identity(Op op, std::uint32_t k);
// Counting a lane twice changes the result.
bool counts(Op op);

// Whether the lane sets a and b are the two halves of one aligned run of lanes.
bool buddies(std::uint64_t a, std::uint64_t b);

// A lane's value inside a section. Fields by kind: Unknown;  K: k;  Own: src (the lane's own value of a
// VGPR value from before the section); Uni: src (a uniform value from before the section); Red: op
// (nullopt: one lane), src, mask, ident, lanes  (op over lanes of mask ? src : ident); Red64F: lo = src,
// hi, mask, lanes, high  (low/high 32 bits of the f64 sum over lanes of mask ? f64(lo, hi) : 0.0).
struct Sym {
    enum Kind : std::uint8_t { Unknown, K, Own, Uni, Red, Red64F };
    Kind kind = Unknown;
    std::uint32_t k = 0;
    std::optional<Op> op;
    ir::Value src{};
    ir::Value hi{};
    ir::Value mask{};
    std::uint32_t ident = 0;
    std::uint64_t lanes = 0;
    bool high = false;
    bool operator==(const Sym&) const = default;

    // Leave unused fields at their defaults so equality compares only relevant state.
    static Sym unknown() { return Sym{}; }
    static Sym konst(std::uint32_t k) {
        Sym s;
        s.kind = K;
        s.k = k;
        return s;
    }
    static Sym own(ir::Value v) {
        Sym s;
        s.kind = Own;
        s.src = v;
        return s;
    }
    static Sym uni(ir::Value v) {
        Sym s;
        s.kind = Uni;
        s.src = v;
        return s;
    }
    static Sym red(std::optional<Op> op, ir::Value src, ir::Value mask, std::uint32_t ident, std::uint64_t lanes) {
        Sym s;
        s.kind = Red;
        s.op = op;
        s.src = src;
        s.mask = mask;
        s.ident = ident;
        s.lanes = lanes;
        return s;
    }
    static Sym red64f(ir::Value lo, ir::Value hi, ir::Value mask, std::uint64_t lanes, bool high) {
        Sym s;
        s.kind = Red64F;
        s.src = lo;
        s.hi = hi;
        s.mask = mask;
        s.lanes = lanes;
        s.high = high;
        return s;
    }
};

// What a value a section leaves holds in every lane. Fields by kind: K(k);  Value(a); Masked(a = src, b =
// mask, ident): the lane's own mask ? src : ident; Active(op, a = src, b = mask, ident): op over every
// lane; ActiveF64(a = lo, b = hi, c = mask, flag = high): low/high 32 bits of the f64 sum over every lane;
// Prefix(op, a = src, b = mask, ident, flag = inclusive): op over the lanes below (inclusive: and
// including) each lane.
struct Shape {
    enum Kind : std::uint8_t { K, Value, Masked, Active, ActiveF64, Prefix };
    Kind kind = K;
    std::uint32_t k = 0;  // K's constant; ident for Masked, Active, Prefix
    Op op = Op::AddU;
    ir::Value a{};
    ir::Value b{};
    ir::Value c{};
    bool flag = false;
    bool operator==(const Shape&) const = default;

    static Shape konst(std::uint32_t k) { return Shape{K, k}; }
    static Shape value(ir::Value v) { return Shape{Value, 0, Op::AddU, v}; }
    static Shape masked(ir::Value src, ir::Value mask, std::uint32_t ident) {
        return Shape{Masked, ident, Op::AddU, src, mask};
    }
    static Shape active(Op op, ir::Value src, ir::Value mask, std::uint32_t ident) {
        return Shape{Active, ident, op, src, mask};
    }
    static Shape active_f64(ir::Value lo, ir::Value hi, ir::Value mask, bool high) {
        return Shape{ActiveF64, 0, Op::AddU, lo, hi, mask, high};
    }
    static Shape prefix(Op op, ir::Value src, ir::Value mask, std::uint32_t ident, bool inclusive) {
        return Shape{Prefix, ident, op, src, mask, ir::Value{}, inclusive};
    }
};

// The section starting at instruction `start` of a block: its restore index and the shapes of the values it
// writes.
struct Section {
    std::size_t end = 0;
    std::map<ir::Value, Shape> shapes;
    // What a v_readlane_b32 of one lane takes, later in the block, from a VGPR the section leaves
    // (by the scalar it writes): the last lanes of a reduction hold it before the first do.
    std::map<ir::Value, Shape> reads;
};

// Every whole-wave section of block, by start index.
std::unordered_map<std::size_t, Section> sections(const ir::Program& p, std::size_t block);

// Test program words.
extern const std::array<std::uint32_t, 20> WAVE_OR_V7_TO_S7;

// Internal declarations used by the tests, threading.cpp and own_bits.cpp.
namespace detail {

// resource::inline_constant shared by wave analysis, threading and own_bits.
std::optional<std::uint32_t> inline_constant(std::uint8_t code);

// The lane `l` reads with DPP, whether it reads 0 instead, and whether it writes; nullopt for
// patterns not modelled.
std::optional<std::tuple<std::uint32_t, bool, bool>> dpp_lane(const gfx10::Dpp& d, std::uint32_t l);
// The value an `s_mov_b32`/`s_mov_b64` to EXEC sets (its low dwords), if constant.
std::optional<std::uint64_t> constant(const ir::Inst& inst);
// The shape of a VGPR value from its lanes, where every lane agrees on one.
std::optional<Shape> per_lane(std::span<const Sym> lanes, std::uint64_t all);

}  // namespace detail

}  // namespace x2w::vs::wave
