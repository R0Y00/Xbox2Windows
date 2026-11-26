// Control-flow structurisation: the program as one lane runs it, without EXEC.
//
// Lane masks (EXEC, compare and carry results, the SGPRs they are saved in) become per-lane booleans
// (Bools). A branch on a wave-wide test of a mask (`s_cbranch_execz`, `vccz`, SCC of a mask
// operation) becomes a branch on the lane's own bit: compiled code only skips work that lanes
// outside the mask would not change, or tests a value that is the same in every lane, so either way
// each lane can decide alone. A path-sensitive pass then proves, for each vector instruction, that
// EXEC holds in every lane that reaches it; what it cannot prove stays as the instruction's guard, a
// per-lane `if` around it.
//
// The control-flow graph keeps its shape and must be reducible (DXIL's only requirement). Helper
// lanes of a pixel shader count as live: on PC their side effects are dropped too, and what they
// compute is only read through derivatives.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "x2w_shader/error.h"
#include "x2w_shader/front_hash.h"
#include "x2w_shader/ir.h"

namespace x2w::vs::structure {

// A per-lane boolean node. Equal functions share a node. Bool::False and Bool::True avoid the macros in
// windows.h.
struct Bool {
    std::uint32_t n = 0;
    static const Bool False;
    static const Bool True;
    constexpr bool operator==(const Bool&) const = default;
    constexpr auto operator<=>(const Bool&) const = default;
};
inline constexpr Bool Bool::False = {0};
inline constexpr Bool Bool::True = {1};

struct BoolHash {
    std::size_t operator()(Bool b) const { return static_cast<std::size_t>(fhash::mix(0, b.n)); }
};

// Meaning of a variable in Bools. Unused members stay zero.
struct Atom {
    enum class Kind : std::uint8_t {
        // The lane's bit of the compare or carry result in this VALU-written value, before EXEC
        // masking.
        Lane,
        // An SCC value, the same in every lane.
        Uniform,
        // A lane mask joined at a block entry, by its low dword's phi.
        Phi,
        // `s_wqm` result: some lane of the quad is in the source mask.
        Quad,
        // A mask the pass cannot see into (loaded, built from scalars, an EXEC write it does not
        // follow): the lane's bit of the register pair this value is a dword of.
        Opaque,
        // A constant mask other than all lanes or none.
        Constant,
    };
    Kind kind = Kind::Lane;
    ir::Value value;
    std::uint64_t constant = 0;

    static constexpr Atom Lane(ir::Value v) { return {Kind::Lane, v, 0}; }
    static constexpr Atom Uniform(ir::Value v) { return {Kind::Uniform, v, 0}; }
    static constexpr Atom Phi(ir::Value v) { return {Kind::Phi, v, 0}; }
    static constexpr Atom Quad(ir::Value v) { return {Kind::Quad, v, 0}; }
    static constexpr Atom Opaque(ir::Value v) { return {Kind::Opaque, v, 0}; }
    static constexpr Atom Constant(std::uint64_t c) { return {Kind::Constant, {}, c}; }
    constexpr bool operator==(const Atom&) const = default;
};

struct AtomHash {
    std::size_t operator()(const Atom& a) const
    {
        return static_cast<std::size_t>(
            fhash::mix(fhash::mix(fhash::mix(0, static_cast<std::uint64_t>(a.kind)), a.value.n), a.constant));
    }
};

// What Bools::meet found out about one Bools, for its later calls: the conditions no lane satisfies,
// and the case splits it took to find that out of each.
struct Meets {
    struct KeyHash {
        std::size_t operator()(const std::vector<std::uint32_t>& k) const
        {
            std::uint64_t h = 0;
            for (std::uint32_t x : k) h = fhash::mix(h, x);
            return static_cast<std::size_t>(fhash::mix(h, k.size()));
        }
    };
    std::unordered_map<std::vector<std::uint32_t>, std::size_t, KeyHash> none;
};

// Case splits Bools::meet makes before it gives up.
inline constexpr std::size_t MAX_MEET = std::size_t{1} << 16;
// Conditions Meets keeps; past that, they are searched anew.
inline constexpr std::size_t MAX_MEETS = std::size_t{1} << 18;

// Reduced ordered binary decision diagrams over Atoms.
class Bools {
public:
    Bools();

    // The atom tested at the top of `b` and the results when it is false and true; nullopt for the
    // constants.
    std::optional<std::tuple<Atom, Bool, Bool>> node(Bool b) const;
    // The atoms `b` depends on, in variable order.
    std::vector<Atom> support(Bool b) const;

    // Whether some lane can satisfy every `(b, holds)` of `all` at once: `b` where `holds`, its
    // negation where not. Yes when the search grows too large to finish.
    bool meet(const std::vector<std::pair<Bool, bool>>& all) const;
    // meet(), which looks up in `seen` what earlier calls on these Bools found out and adds what it
    // finds. The answer is the one meet gives.
    bool meet_seen(const std::vector<std::pair<Bool, bool>>& all, Meets& seen) const;

    // Internal helpers shared by this module and its tests.
    Bool atom(const Atom& atom, std::size_t born);
    Bool mk(std::uint32_t v, Bool lo, Bool hi);
    std::pair<Bool, Bool> split(Bool b, std::uint32_t v) const;
    Bool ite(Bool f, Bool g, Bool h);
    Bool and_(Bool a, Bool b);
    Bool or_(Bool a, Bool b);
    Bool not_(Bool a);
    Bool xor_(Bool a, Bool b);
    bool meet_within(const std::vector<std::pair<Bool, bool>>& all, std::size_t budget, Meets& seen) const;
    bool meet_in(std::size_t at, std::vector<std::uint32_t>& work, std::size_t& budget, Meets& seen) const;
    bool implies(Bool a, Bool b);
    Bool forget(Bool b, std::size_t from, std::unordered_map<Bool, Bool, BoolHash>& memo);
    Bool restrict(Bool b, std::uint32_t v, bool value, std::unordered_map<Bool, Bool, BoolHash>& memo);
    Bool given(Bool b, Bool known);
    // The variable at the top of `b` (TERMINAL for the constants).
    std::uint32_t top(Bool b) const { return std::get<0>(nodes_[b.n]); }
    bool overflow() const { return overflow_; }

    static constexpr std::uint32_t TERMINAL = 0xFFFF'FFFFu;

private:
    struct TripleHash {
        std::size_t operator()(const std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>& t) const
        {
            return static_cast<std::size_t>(
                fhash::mix(fhash::mix(fhash::mix(0, std::get<0>(t)), std::get<1>(t)), std::get<2>(t)));
        }
    };
    std::vector<std::uint32_t> vars(Bool b) const;

    // (variable, low, high); nodes 0 and 1 are the constants.
    std::vector<std::tuple<std::uint32_t, Bool, Bool>> nodes_;
    std::unordered_map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, Bool, TripleHash> unique_;
    std::unordered_map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, Bool, TripleHash> memo_;
    std::vector<Atom> atoms_;
    // Reverse-postorder index of the block that defines each atom.
    std::vector<std::size_t> born_;
    std::unordered_map<Atom, std::uint32_t, AtomHash> ids_;
    bool overflow_ = false;
};

// How a lane leaves a block.
struct LaneExit {
    enum class Kind : std::uint8_t {
        Jump,
        // To `taken` in lanes where `when` holds. A condition over Atom::Uniform only is a uniform
        // branch.
        Branch,
        End,
        Indirect,
    };
    Kind kind = Kind::End;
    std::size_t target = 0;  // Jump
    Bool when;               // Branch
    std::size_t taken = 0;
    std::size_t not_taken = 0;

    static constexpr LaneExit Jump(std::size_t t) { return {Kind::Jump, t, {}, 0, 0}; }
    static constexpr LaneExit Branch(Bool when, std::size_t taken, std::size_t not_taken)
    {
        return {Kind::Branch, 0, when, taken, not_taken};
    }
    static constexpr LaneExit End() { return {Kind::End, 0, {}, 0, 0}; }
    static constexpr LaneExit Indirect() { return {Kind::Indirect, 0, {}, 0, 0}; }
    constexpr bool is_branch() const { return kind == Kind::Branch; }
    constexpr bool operator==(const LaneExit&) const = default;
};

using ValueHash = fhash::ValueHash;
// A lane mask by the values of its low and high dword (both the low one in wave32).
using Key = std::pair<ir::Value, ir::Value>;

// A Program as one lane runs it: same blocks, per-lane branches.
struct Structured {
    Bools bools;
    // Per block.
    std::vector<LaneExit> exits;
    // Per block and instruction: the condition, beyond reaching it, under which a lane runs a vector
    // instruction; nullopt when reaching it is enough (lanes outside EXEC then never get there, so a
    // VGPR write needs no old value). Always nullopt for scalar instructions.
    std::vector<std::vector<std::optional<Bool>>> guards;
    // Per block: what holds in every lane that enters it (over forward edges; a loop header forgets
    // what its loop redefines).
    std::vector<Bool> reach;
    // The lane's bit of each lane-mask value, by its low dword's value.
    std::unordered_map<ir::Value, Bool, ValueHash> masks;
    // Exact Wave64 pairs for checking masks at their uses. See mask_bits.
    std::unordered_map<Key, Bool, fhash::ValuePairHash> paired_masks;
    // Wave64 masks: low dword's value by the high dword's.
    std::unordered_map<ir::Value, ir::Value, ValueHash> highs;
    // Masks (low dwords) set only in lanes whose EXEC bit is set where they are written: every such
    // lane reaches the writing block, so a ballot there gives the mask as data. No split low dword,
    // as in `masks`.
    std::unordered_set<ir::Value, ValueHash> active;
    // Incoming lane boolean per predecessor edge (in block predecessor order) for each Atom::Phi id.
    std::unordered_map<ir::Value, std::vector<Bool>, ValueHash> phi_masks;
    // Blocks whose closing branch the whole wave takes together (structure_wide): its `when` is
    // still the lane's own bit; the wave goes by all of them.
    std::set<std::size_t> wide;
    // Per block closed by a branch on a lane mask: whether the console takes it when some lane's
    // `when` holds (true) or when every lane's does (false).
    std::vector<std::optional<bool>> wave_any;
    // SCC values a mask operation sets when some lane's bit of its result is set, with that lane
    // bit: as data such an SCC is "some lane of the wave has it".
    std::unordered_map<ir::Value, Bool, ValueHash> scc_some;

    // The lane bit of the mask whose dwords are `low` and `high` (`high` nullopt: by the low dword
    // alone, as wave32 masks are) (pub(crate)).
    std::optional<Bool> mask_bits(ir::Value low, std::optional<ir::Value> high) const;
};

// Builds the per-lane form of `program`. Throws ShaderError.
Structured structure(const ir::Program& program);

// structure(), with the branches that close the blocks of `wide` kept as the console has them: one
// decision for the whole wave (`s_cbranch_execz` and `vccz`: taken when no lane's bit is set;
// `execnz` and `vccnz`: when some lane's is; SCC of a mask operation likewise,
// Structured::wave_any), which every lane follows. A lane then learns nothing about its own bit from
// the way it went, so past such a branch EXEC stays a guard on every vector instruction, as it is on
// the console. For a console wave run as two host waves, which exchange that decision and can then
// meet (a group barrier) inside the branch.
Structured structure_wide(const ir::Program& program, const std::set<std::size_t>& wide);

}  // namespace x2w::vs::structure
