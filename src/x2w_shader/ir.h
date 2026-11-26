// One-to-one SSA form of a native program.
//
// Every machine instruction becomes one Inst, except scheduling-only ones (`s_nop`, `s_waitcnt*`,
// `s_inst_prefetch`...) and redundant conditional branches. Each register dword is a variable: a
// read names the Value it sees, a write defines a new one, and Phis merge values where control flow
// joins. Nothing is raised: an instruction keeps its opcode and decoded fields, so its exact
// semantics stay with it.
//
// Implicit operands are explicit: SCC, M0, and EXEC, which every vector instruction reads. A write
// to a VGPR also reads the register's old value, because lanes disabled in EXEC keep it.
//
// M0-relative VGPR access (`v_movrels/d/sd_b32`, `v_swaprel_b32`) names its operand's whole possible
// range: from the encoded register up to the program's VGPR count, all read (and all written, for a
// destination).
#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "x2w_shader/error.h"
#include "x2w_shader/gfx10/mod.h"

namespace x2w::vs::ir {

// One register dword. Sort scalar registers before SCC, then vector registers. Scalar codes as in
// Operand::Scalar: 0-105 SGPRs, 106/107 VCC, 108-123 trap temporaries, 124 M0, 126/127 EXEC. Codes from
// PARKED up are scalars the compiler parked in VGPR lanes (`unspill`): no machine instruction names them.
struct Reg {
    enum class Kind : std::uint8_t { S = 0, Scc = 1, V = 2 };
    Kind kind = Kind::S;
    // S: the scalar code; V: the VGPR; Scc: 0.
    std::uint8_t n = 0;

    static constexpr Reg S(std::uint8_t code) { return {Kind::S, code}; }
    static constexpr Reg V(std::uint8_t reg) { return {Kind::V, reg}; }
    static const Reg Scc;

    constexpr bool is_s() const { return kind == Kind::S; }
    constexpr bool is_scc() const { return kind == Kind::Scc; }
    constexpr bool is_v() const { return kind == Kind::V; }

    constexpr bool operator==(const Reg&) const = default;
    constexpr auto operator<=>(const Reg&) const = default;
    // Diagnostic spelling: "S(5)", "Scc", "V(3)".
    std::string debug() const;
};
inline constexpr Reg Reg::Scc = {Reg::Kind::Scc, 0};

inline constexpr std::uint8_t VCC = 106;
inline constexpr std::uint8_t M0 = 124;
inline constexpr std::uint8_t EXEC = 126;
// The first scalar code past the machine's (0-127): see Reg.
inline constexpr std::uint8_t PARKED = 128;

// An SSA value holds one register dword, or one bit for SCC.
struct Value {
    std::uint32_t n = 0;
    constexpr bool operator==(const Value&) const = default;
    constexpr auto operator<=>(const Value&) const = default;
    // Diagnostic spelling: "Value(3)".
    std::string debug() const;
};

struct Exit {
    enum class Kind : std::uint8_t {
        Jump,
        // `s_cbranch_*`, the block's last instruction.
        Branch,
        // `s_endpgm` and its variants.
        End,
        // `s_setpc_b64` to an address held in registers, e.g. the handover to the next half of a
        // merged hardware stage.
        Indirect,
    };
    Kind kind = Kind::End;
    std::size_t target = 0;     // Jump
    std::size_t taken = 0;      // Branch
    std::size_t not_taken = 0;  // Branch

    static constexpr Exit Jump(std::size_t b) { return {Kind::Jump, b, 0, 0}; }
    static constexpr Exit Branch(std::size_t taken, std::size_t not_taken) {
        return {Kind::Branch, 0, taken, not_taken};
    }
    static const Exit End;
    static const Exit Indirect;

    constexpr bool is_jump() const { return kind == Kind::Jump; }
    constexpr bool is_branch() const { return kind == Kind::Branch; }
    constexpr bool is_end() const { return kind == Kind::End; }
    constexpr bool is_indirect() const { return kind == Kind::Indirect; }

    std::vector<std::size_t> successors() const;
    constexpr bool operator==(const Exit&) const = default;
    // Diagnostic spelling: "Jump(2)", "Branch { taken: 2, not_taken: 1 }", "End", "Indirect".
    std::string debug() const;
};
inline constexpr Exit Exit::End = {Exit::Kind::End, 0, 0, 0};
inline constexpr Exit Exit::Indirect = {Exit::Kind::Indirect, 0, 0, 0};

struct Phi {
    Reg reg;
    Value value;
    // One per predecessor, in Block::preds order.
    std::vector<Value> args;
    bool operator==(const Phi&) const = default;
};

struct Inst {
    gfx10::Instruction machine;
    gfx10::Fields fields;
    // Every register dword read, with the value it holds.
    std::vector<std::pair<Reg, Value>> reads;
    // Every register dword written, with the value it receives.
    std::vector<std::pair<Reg, Value>> writes;

    // The opcode name ("" if unknown). Static storage: compare by content.
    std::string_view name() const;
    // The value `reg` holds when this instruction reads it.
    std::optional<Value> read(Reg reg) const;
    // The value this instruction writes to `reg`.
    std::optional<Value> write(Reg reg) const;
    bool operator==(const Inst&) const = default;
};

struct Block {
    // Byte offset of the block's first machine instruction (shared by a block and its copies).
    std::size_t offset = 0;
    std::vector<std::size_t> preds;
    std::vector<Phi> phis;
    std::vector<Inst> insts;
    Exit exit = Exit::End;
    bool operator==(const Block&) const = default;
};

struct Program {
    bool wave64 = false;
    // VGPRs v0 up to this exist; bounds M0-relative access.
    std::uint16_t vgprs = 0;
    // In code order, then copies made to give every loop one entry; block 0 is the entry.
    std::vector<Block> blocks;
    // Registers read before any write, with their value on entry: what the hardware or a called
    // fetch routine set up (user SGPRs, system values, vertex inputs).
    std::vector<std::pair<Reg, Value>> inputs;
    bool operator==(const Program&) const = default;
};

// Decodes `code` and builds its SSA form; `vgprs` is the program's VGPR count. Throws
// ShaderError::unsupported on instructions the IR cannot express yet (and ::isa on bad code).
Program build(std::span<const std::uint8_t> code, bool wave64, std::uint16_t vgprs);

// Duplicate a join for each proven (predecessor, join, destination) edge. The instructions still
// execute on that path and define fresh SSA values (pub(crate)). Throws ShaderError.
Program thread_edges(const Program& p, const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>& edges);

// The program without its early-outs (an `s_cbranch_execz` to a block that only ends the program);
// nullopt if the program has no such branch (pub(crate)).
std::optional<Program> without_early_outs(const Program& p);

// As without_early_outs, and also without the branches that only skip a region when no lane is left
// for it; `deep` looks closer at what the code after the branch can see. nullopt if nothing changes
// (pub(crate)).
std::optional<Program> without_skips(const Program& p, bool deep);

// The registers `inst` of `p` reads as operands: its reads without the old value of a VGPR it only
// writes (pub(crate)).
std::vector<Reg> operand_reads(const Program& p, const Inst& inst);

std::vector<std::size_t> reverse_postorder(std::span<const Block> blocks);

// An edge to an earlier block in reverse postorder whose target does not dominate its source, if
// any: there the control flow is irreducible. `index` is each block's position in `order`.
std::optional<std::pair<std::size_t, std::size_t>> irreducible_edge(std::span<const Block> blocks,
                                                                    std::span<const std::size_t> order,
                                                                    std::span<const std::size_t> index);

// Each block's immediate dominator (the entry's is itself, unreachable blocks' SIZE_MAX); `order` is
// reverse postorder, `index` each block's position in it.
std::vector<std::size_t> idoms(std::span<const Block> blocks, std::span<const std::size_t> order,
                               std::span<const std::size_t> index);

// Internal SSA helpers exposed to unit tests.
namespace detail {

// Placeholder before renaming.
inline constexpr Value NONE = {0xFFFF'FFFFu};

// Registers read, registers written.
using Effects = std::pair<std::vector<Reg>, std::vector<Reg>>;

// Registers an instruction reads and writes, one per dword, or nullopt if the IR drops it.
std::optional<Effects> effects(const gfx10::Instruction& inst, const gfx10::Fields& fields,
                               std::string_view name, std::uint8_t lanes, std::uint16_t vgprs);

// Recomputes every block's predecessors from the exits.
void link(std::vector<Block>& blocks);
void split_irreducible(std::vector<Block>& blocks);
Program rename(std::vector<Block> blocks, bool wave64, std::uint16_t vgprs);
// Returns the union-find steps taken (a test bounds them).
std::size_t remove_trivial_phis_steps(std::vector<Block>& blocks, std::uint32_t values);

}  // namespace detail

}  // namespace x2w::vs::ir

// Hashes for Reg and Value. Use ordered maps and sets where iteration order affects output.
template <>
struct std::hash<x2w::vs::ir::Reg> {
    std::size_t operator()(const x2w::vs::ir::Reg& r) const noexcept {
        return std::hash<unsigned>{}(unsigned(r.kind) << 8 | r.n);
    }
};
template <>
struct std::hash<x2w::vs::ir::Value> {
    std::size_t operator()(const x2w::vs::ir::Value& v) const noexcept { return std::hash<std::uint32_t>{}(v.n); }
};
