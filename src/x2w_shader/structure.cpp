// Control-flow structurisation. See structure.h.
//
// Node IDs follow creation order. Evaluate calls that may create a node (atom, mk, ite, mask, sop, operand)
// in separate statements so the IDs are reproducible.
#include "x2w_shader/structure.h"

#include <algorithm>
#include <string_view>
#include <variant>

namespace x2w::vs::structure {

namespace {

using gfx10::Operand;
using ir::Exit;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;

constexpr std::size_t MAX_NODES = std::size_t{1} << 21;
constexpr std::size_t MAX_ROUNDS = 256;

ShaderError too_large() { return ShaderError::isa(0, "lane conditions too large"); }

using ValueSet = std::unordered_set<Value, ValueHash>;
template <class V>
using KeyMap = std::unordered_map<Key, V, fhash::ValuePairHash>;
template <class V>
using ValueMap = std::unordered_map<Value, V, ValueHash>;
using BoolMemo = std::unordered_map<Bool, Bool, BoolHash>;

std::optional<std::string_view> strip_prefix(std::string_view s, std::string_view p)
{
    if (!s.starts_with(p)) return std::nullopt;
    return s.substr(p.size());
}
std::optional<std::string_view> strip_suffix(std::string_view s, std::string_view p)
{
    if (!s.ends_with(p)) return std::nullopt;
    return s.substr(0, s.size() - p.size());
}

}  // namespace

// ---- Bools ----

Bools::Bools()
{
    nodes_.push_back({TERMINAL, Bool::False, Bool::False});
    nodes_.push_back({TERMINAL, Bool::True, Bool::True});
}

std::optional<std::tuple<Atom, Bool, Bool>> Bools::node(Bool b) const
{
    const auto& [v, lo, hi] = nodes_.at(b.n);
    if (v == TERMINAL) return std::nullopt;
    return std::tuple<Atom, Bool, Bool>{atoms_[v], lo, hi};
}

std::vector<Atom> Bools::support(Bool b) const
{
    std::vector<std::uint32_t> vs = vars(b);
    std::sort(vs.begin(), vs.end());
    std::vector<Atom> out;
    out.reserve(vs.size());
    for (std::uint32_t v : vs) out.push_back(atoms_[v]);
    return out;
}

std::vector<std::uint32_t> Bools::vars(Bool b) const
{
    std::vector<Bool> seen;
    std::vector<std::uint32_t> out;
    std::vector<Bool> stack{b};
    while (!stack.empty()) {
        const Bool x = stack.back();
        stack.pop_back();
        const auto& [v, lo, hi] = nodes_[x.n];
        if (v == TERMINAL || std::find(seen.begin(), seen.end(), x) != seen.end()) continue;
        seen.push_back(x);
        if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
        stack.push_back(lo);
        stack.push_back(hi);
    }
    return out;
}

Bool Bools::atom(const Atom& atom, std::size_t born)
{
    std::uint32_t v;
    const auto it = ids_.find(atom);
    if (it != ids_.end()) {
        v = it->second;
    } else {
        atoms_.push_back(atom);
        born_.push_back(born);
        v = static_cast<std::uint32_t>(atoms_.size() - 1);
        ids_.emplace(atom, v);
    }
    return mk(v, Bool::False, Bool::True);
}

Bool Bools::mk(std::uint32_t v, Bool lo, Bool hi)
{
    if (lo == hi) return lo;
    const auto key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{v, lo.n, hi.n};
    if (const auto it = unique_.find(key); it != unique_.end()) return it->second;
    if (nodes_.size() >= MAX_NODES) {
        // The caller fails once it sees `overflow`.
        overflow_ = true;
        return Bool::False;
    }
    const Bool b{static_cast<std::uint32_t>(nodes_.size())};
    nodes_.push_back({v, lo, hi});
    unique_.emplace(key, b);
    return b;
}

std::pair<Bool, Bool> Bools::split(Bool b, std::uint32_t v) const
{
    const auto& [bv, lo, hi] = nodes_[b.n];
    if (bv == v) return {lo, hi};
    return {b, b};
}

Bool Bools::ite(Bool f, Bool g, Bool h)
{
    if (f == Bool::True || g == h) return g;
    if (f == Bool::False) return h;
    if (g == Bool::True && h == Bool::False) return f;
    const auto key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>{f.n, g.n, h.n};
    if (const auto it = memo_.find(key); it != memo_.end()) return it->second;
    const std::uint32_t v = std::min(std::min(top(f), top(g)), top(h));
    const auto [f0, f1] = split(f, v);
    const auto [g0, g1] = split(g, v);
    const auto [h0, h1] = split(h, v);
    const Bool lo = ite(f0, g0, h0);
    const Bool hi = ite(f1, g1, h1);
    const Bool r = mk(v, lo, hi);
    memo_.insert_or_assign(key, r);
    return r;
}

Bool Bools::and_(Bool a, Bool b) { return ite(a, b, Bool::False); }
Bool Bools::or_(Bool a, Bool b) { return ite(a, Bool::True, b); }
Bool Bools::not_(Bool a) { return ite(a, Bool::False, Bool::True); }
Bool Bools::xor_(Bool a, Bool b)
{
    const Bool nb = not_(b);
    return ite(a, nb, b);
}

bool Bools::meet(const std::vector<std::pair<Bool, bool>>& all) const
{
    Meets seen;
    return meet_seen(all, seen);
}

bool Bools::meet_seen(const std::vector<std::pair<Bool, bool>>& all, Meets& seen) const
{
    return meet_within(all, MAX_MEET, seen);
}

bool Bools::meet_within(const std::vector<std::pair<Bool, bool>>& all, std::size_t budget, Meets& seen) const
{
    std::vector<std::uint32_t> work;
    work.reserve(all.size() * 8);
    for (const auto& [b, holds] : all) work.push_back((b.n << 1) | (holds ? 1u : 0u));
    return meet_in(0, work, budget, seen);
}

// The search of meet for the conditions `work[at..]` (a node doubled, plus one where it must hold), which
// it takes off `work`. A case split on the top variable; `seen` keeps the conditions that no lane satisfies
// with the case splits it took to find that out.
bool Bools::meet_in(std::size_t at, std::vector<std::uint32_t>& work, std::size_t& budget, Meets& seen) const
{
    // Constants either fail or say nothing.
    std::size_t kept = at;
    for (std::size_t i = at; i < work.size(); ++i) {
        const std::uint32_t b = work[i] >> 1;
        const bool holds = (work[i] & 1) != 0;
        if (b > Bool::True.n) {
            work[kept] = work[i];
            ++kept;
        } else if ((b == Bool::True.n) != holds) {
            work.resize(at);
            return false;
        }
    }
    work.resize(kept);
    if (kept == at || budget == 0) {
        work.resize(at);
        return true;
    }
    // As a set: neither order nor repeats change the search.
    std::sort(work.begin() + static_cast<std::ptrdiff_t>(at), work.end());
    std::size_t end = at + 1;
    for (std::size_t i = at + 1; i < kept; ++i) {
        if (work[i] != work[end - 1]) {
            work[end] = work[i];
            ++end;
        }
    }
    work.resize(end);
    {
        const std::vector<std::uint32_t> key(work.begin() + static_cast<std::ptrdiff_t>(at), work.end());
        if (const auto it = seen.none.find(key); it != seen.none.end()) {
            const std::size_t splits = it->second;
            const bool within = splits <= budget;
            budget = budget >= splits ? budget - splits : 0;
            work.resize(at);
            return !within;
        }
    }
    const std::size_t before = budget;
    budget -= 1;
    std::uint32_t v = TERMINAL;
    for (std::size_t i = at; i < end; ++i) v = std::min(v, top(Bool{work[i] >> 1}));
    bool found = false;
    for (const bool value : {false, true}) {
        for (std::size_t i = at; i < end; ++i) {
            const auto [lo, hi] = split(Bool{work[i] >> 1}, v);
            const std::uint32_t code = ((value ? hi : lo).n << 1) | (work[i] & 1);
            work.push_back(code);
        }
        if (meet_in(end, work, budget, seen)) {
            found = true;
            break;
        }
    }
    if (!found && seen.none.size() < MAX_MEETS) {
        std::vector<std::uint32_t> key(work.begin() + static_cast<std::ptrdiff_t>(at),
                                       work.begin() + static_cast<std::ptrdiff_t>(end));
        seen.none.insert_or_assign(std::move(key), before - budget);
    }
    work.resize(at);
    return found;
}

bool Bools::implies(Bool a, Bool b)
{
    const Bool nb = not_(b);
    return and_(a, nb) == Bool::False;
}

Bool Bools::forget(Bool b, std::size_t from, BoolMemo& memo)
{
    const auto [v, lo0, hi0] = nodes_[b.n];
    if (v == TERMINAL) return b;
    if (const auto it = memo.find(b); it != memo.end()) return it->second;
    const Bool lo = forget(lo0, from, memo);
    const Bool hi = forget(hi0, from, memo);
    const Bool r = born_[v] >= from ? or_(lo, hi) : mk(v, lo, hi);
    memo.insert_or_assign(b, r);
    return r;
}

Bool Bools::restrict(Bool b, std::uint32_t v, bool value, BoolMemo& memo)
{
    const auto [bv, lo0, hi0] = nodes_[b.n];
    if (bv == TERMINAL || bv > v) return b;
    if (bv == v) return value ? hi0 : lo0;
    if (const auto it = memo.find(b); it != memo.end()) return it->second;
    const Bool l = restrict(lo0, v, value, memo);
    const Bool h = restrict(hi0, v, value, memo);
    const Bool r = mk(bv, l, h);
    memo.insert_or_assign(b, r);
    return r;
}

Bool Bools::given(Bool b, Bool known)
{
    for (const std::uint32_t v : vars(b)) {
        const Bool x = mk(v, Bool::False, Bool::True);
        const Bool nx = not_(x);
        if (implies(known, x)) {
            BoolMemo memo;
            b = restrict(b, v, true, memo);
        } else if (implies(known, nx)) {
            BoolMemo memo;
            b = restrict(b, v, false, memo);
        }
    }
    return b;
}

std::optional<Bool> Structured::mask_bits(Value low, std::optional<Value> high) const
{
    if (high) {
        const auto it = paired_masks.find(Key{low, *high});
        if (it == paired_masks.end()) return std::nullopt;
        return it->second;
    }
    const auto it = masks.find(low);
    if (it == masks.end()) return std::nullopt;
    return it->second;
}

// ---- Lanes ----

namespace {

// A mask join taken as `value` in every lane that reaches `block`: each incoming mask held (or
// failed) on its edge.
struct Join {
    std::size_t block;
    // The phi value naming the join.
    Value id;
    bool value;
    bool derived;
    std::vector<Key> args;
};

// A mask join represented as Atom::Phi, whose incoming edge masks are resolved once every block
// (including loop back edges) has been visited.
struct PhiJoin {
    std::size_t block;
    Value id;
    bool derived;
    std::vector<Key> args;
};

struct SopMasks {
    std::optional<std::pair<Key, Bool>> dst;
    std::optional<std::pair<Key, Bool>> exec;
    // Some operand derives from EXEC or a VALU result.
    bool derived = false;
    // For branches: SCC ("result is not zero") per lane.
    std::optional<Bool> scc;
};

struct Lanes {
    const Program* p;
    std::vector<std::size_t> order;
    // Reverse-postorder index per block.
    std::vector<std::size_t> index;
    // Per block, in predecessor order: (block, over its taken edge).
    std::vector<std::vector<std::pair<std::size_t, bool>>> incoming;
    // Block and instruction index writing each value.
    ValueMap<std::pair<std::size_t, std::size_t>> defs;
    // Block and index of each phi value.
    ValueMap<std::pair<std::size_t, std::size_t>> phis;
    // Mask joins found wrong when resolved to a constant: atoms instead.
    ValueSet unresolved;
    // Phi ids whose joins depend on EXEC or VALU results (including along back edges), carried
    // across resets so their joins stay in `masks`.
    ValueSet derived_phis;
    Bools b;
    // Masks derived from EXEC and VALU results.
    KeyMap<Bool> masks;
    // Other values read as masks: constants, uniform selects, opaque ones.
    KeyMap<Bool> other;
    // Pairs read as masks that `mask` does not see: carry-ins, and pairs joins take over back edges.
    std::vector<Key> seen;
    // For branches: SCC of a mask operation ("some lane is set") as the lane's own bit.
    ValueMap<Bool> scc_lane;
    // Joins resolved to a constant, to check once all edges are known.
    std::vector<Join> joins;
    // Atom::Phi joins created in the current pass, resolved in check_joins and exported in finish.
    std::vector<PhiJoin> phi_joins;
    ValueSet phi_join_ids;
    // Per block: what a lane knows on entry, over forward edges only.
    std::vector<Bool> first;
    // Per block ending in a branch: when a lane takes it.
    std::vector<std::optional<Bool>> when;
    ValueSet active;
    std::set<std::size_t> wide;
    // SCC values of scc_lane that are set when EVERY lane's bit is (`s_cmp_eq` of two masks: "no lane
    // differs"); the others are set when some lane's is.
    ValueSet scc_every;
    std::vector<std::optional<bool>> wave_any;

    explicit Lanes(const Program& program);
    void reset();
    std::vector<std::vector<std::optional<Bool>>> masks_in_order();
    Bool edge(Bool c, std::size_t q, bool taken);
    std::pair<std::vector<Value>, std::vector<Value>> check_joins(const std::vector<Bool>& cond);
    Structured finish(const std::vector<std::vector<std::optional<Bool>>>& execs, const std::vector<Bool>& cond);
    template <class F>
    std::optional<Key> key(F value, std::uint8_t code) const;
    std::size_t born(Value v) const;
    Bool exec(const Inst& inst);
    std::pair<Bool, bool> mask(Key k);
    std::optional<std::pair<Bool, bool>> join(Key k);
    std::optional<std::pair<Bool, bool>> operand(const Inst& inst, const Operand& op);
    std::optional<SopMasks> sop(const Inst& inst);
    bool clear_after_loop(Value scc) const;
    std::optional<Bool> combine(std::string_view f, Bool a, Bool c);
    void step(const Inst& inst, std::optional<Bool> exec);
    std::optional<bool> wave_kind(const Inst& inst) const;
    Bool branch(const Inst& inst);
    std::vector<Bool> paths();
};

const Inst* def_of(const Program& p, std::optional<std::pair<std::size_t, std::size_t>> d)
{
    if (!d) return nullptr;
    return &p.blocks[d->first].insts[d->second];
}

template <class M>
std::optional<std::pair<std::size_t, std::size_t>> lookup(const M& m, Value v)
{
    const auto it = m.find(v);
    if (it == m.end()) return std::nullopt;
    return it->second;
}

// `ir::build` splits irreducible loops; checked again because everything here assumes a reducible
// graph.
void check_reducible(const Program& p, const std::vector<std::size_t>& order, const std::vector<std::size_t>& index)
{
    if (const auto e = ir::irreducible_edge(p.blocks, order, index))
        throw ShaderError::unsupported(p.blocks[e->first].offset, "irreducible control flow");
}

Lanes::Lanes(const Program& program) : p(&program)
{
    const auto& blocks = program.blocks;
    order = ir::reverse_postorder(blocks);
    index.assign(blocks.size(), SIZE_MAX);
    for (std::size_t i = 0; i < order.size(); ++i) index[order[i]] = i;
    check_reducible(program, order, index);
    incoming.assign(blocks.size(), {});
    for (std::size_t bl = 0; bl < blocks.size(); ++bl) {
        const auto succ = blocks[bl].exit.successors();
        for (std::size_t k = 0; k < succ.size(); ++k) incoming[succ[k]].push_back({bl, k == 0});
        for (std::size_t i = 0; i < blocks[bl].phis.size(); ++i)
            phis.insert_or_assign(blocks[bl].phis[i].value, std::pair{bl, i});
        for (std::size_t i = 0; i < blocks[bl].insts.size(); ++i)
            for (const auto& w : blocks[bl].insts[i].writes) defs.insert_or_assign(w.second, std::pair{bl, i});
    }
    first.assign(blocks.size(), Bool::False);
    when.assign(blocks.size(), std::nullopt);
    wave_any.assign(blocks.size(), std::nullopt);
}

void Lanes::reset()
{
    b = Bools();
    masks.clear();
    other.clear();
    seen.clear();
    scc_lane.clear();
    scc_every.clear();
    std::fill(wave_any.begin(), wave_any.end(), std::nullopt);
    joins.clear();
    phi_joins.clear();
    phi_join_ids.clear();
    std::fill(first.begin(), first.end(), Bool::False);
    std::fill(when.begin(), when.end(), std::nullopt);
    active.clear();
}

template <class F>
std::optional<Key> Lanes::key(F value, std::uint8_t code) const
{
    const std::optional<Value> lo = value(Reg::S(code));
    if (!lo) return std::nullopt;
    Value hi = *lo;
    if (p->wave64) {
        const std::optional<Value> h = value(Reg::S(static_cast<std::uint8_t>(code + 1)));
        if (!h) return std::nullopt;
        hi = *h;
    }
    return Key{*lo, hi};
}

#define READS [&inst](Reg r) { return inst.read(r); }
#define WRITES [&inst](Reg r) { return inst.write(r); }

std::vector<std::vector<std::optional<Bool>>> Lanes::masks_in_order()
{
    const Program& prog = *p;
    auto input = [&](Reg reg) -> std::optional<Value> {
        for (const auto& i : prog.inputs)
            if (i.first == reg) return i.second;
        return std::nullopt;
    };
    if (const auto lo = input(Reg::S(ir::EXEC))) {
        const Value hi = prog.wave64 ? input(Reg::S(ir::EXEC + 1)).value_or(*lo) : *lo;
        masks.insert_or_assign(Key{*lo, hi}, Bool::True);
        active.insert(*lo);
    }
    std::vector<std::vector<std::optional<Bool>>> execs;
    for (const auto& blk : prog.blocks) execs.emplace_back(blk.insts.size());
    for (std::size_t n = 0; n < order.size(); ++n) {
        const std::size_t bl = order[n];
        Bool c = n == 0 ? Bool::True : Bool::False;
        for (std::size_t k = 0; k < incoming[bl].size(); ++k) {
            const auto [q, taken] = incoming[bl][k];
            if (index[q] < n) {
                const Bool e = edge(first[q], q, taken);
                c = b.or_(c, e);
            }
        }
        first[bl] = c;
        // EXEC joined at the block's entry holds the lanes that arrive.
        for (const auto& phi : prog.blocks[bl].phis)
            if (phi.reg == Reg::S(ir::EXEC)) active.insert(phi.value);
        // EXEC where known in this block (read or written so far).
        std::optional<Bool> now;
        const auto& insts = prog.blocks[bl].insts;
        for (std::size_t i = 0; i < insts.size(); ++i) {
            const Inst& inst = insts[i];
            if (inst.read(Reg::S(ir::EXEC))) {
                const Bool e = exec(inst);
                execs[bl][i] = e;
                now = e;
            }
            step(inst, now);
            if (const auto k = key(WRITES, ir::EXEC)) {
                if (const auto it = masks.find(*k); it != masks.end())
                    now = it->second;
                else if (const auto jt = other.find(*k); jt != other.end())
                    now = jt->second;
                else
                    now = std::nullopt;
            }
        }
        if (prog.blocks[bl].exit.is_branch() && !insts.empty()) {
            const Inst& last = insts.back();
            when[bl] = branch(last);
            wave_any[bl] = wave_kind(last);
        }
    }
    return execs;
}

// `c` narrowed by the branch at the end of `q` towards one successor.
Bool Lanes::edge(Bool c, std::size_t q, bool taken)
{
    if (wide.count(q)) {
        // the wave's decision: it says nothing about this lane's own bit
        return c;
    }
    const std::optional<Bool> w = when[q];
    if (!w) return c;
    if (taken) return b.and_(c, *w);
    const Bool nw = b.not_(*w);
    return b.and_(c, nw);
}

// Joins whose constant fails on some edge, and joins that need another pass because a later edge
// turned out to derive from EXEC or VALU.
std::pair<std::vector<Value>, std::vector<Value>> Lanes::check_joins(const std::vector<Bool>& cond)
{
    std::vector<Value> wrong;
    std::vector<Value> new_derived;
    ValueSet seen_derived;
    std::size_t j = 0;
    std::size_t pj = 0;
    // Checking an argument can resolve another join; it is checked too.
    while (j < joins.size() || pj < phi_joins.size()) {
        while (j < joins.size()) {
            const std::size_t block = joins[j].block;
            const Value id = joins[j].id;
            const bool value = joins[j].value;
            const bool was_derived = joins[j].derived;
            bool failed = false;
            const std::vector<Key> args = joins[j].args;
            for (std::size_t n = 0; n < args.size(); ++n) {
                const auto [q, taken] = incoming[block].at(n);
                const Bool e = edge(cond[q], q, taken);
                auto [m, d] = mask(args[n]);
                if (!was_derived && d && !derived_phis.count(id) && seen_derived.insert(id).second)
                    new_derived.push_back(id);
                if (!value) m = b.not_(m);
                if (!failed && !b.implies(e, m)) {
                    wrong.push_back(id);
                    failed = true;
                }
            }
            ++j;
        }
        while (pj < phi_joins.size()) {
            const Value id = phi_joins[pj].id;
            const bool was_derived = phi_joins[pj].derived;
            const std::vector<Key> args = phi_joins[pj].args;
            for (const Key& a : args) {
                const bool d = mask(a).second;
                if (!was_derived && d && !derived_phis.count(id) && seen_derived.insert(id).second)
                    new_derived.push_back(id);
            }
            ++pj;
        }
    }
    return {wrong, new_derived};
}

Structured Lanes::finish(const std::vector<std::vector<std::optional<Bool>>>& execs, const std::vector<Bool>& cond)
{
    std::vector<PhiJoin> pjs = std::move(phi_joins);
    phi_joins.clear();
    ValueMap<std::vector<Bool>> phi_masks;
    for (PhiJoin& pj : pjs) {
        std::vector<Bool> in;
        for (std::size_t n = 0; n < pj.args.size(); ++n) {
            const auto [q, taken] = incoming[pj.block].at(n);
            const Bool e = edge(cond[q], q, taken);
            const Bool m = mask(pj.args[n]).first;
            in.push_back(b.given(m, e));
        }
        phi_masks.insert_or_assign(pj.id, std::move(in));
    }
    const auto& blocks = p->blocks;
    std::vector<std::vector<std::optional<Bool>>> guards;
    for (std::size_t n = 0; n < blocks.size(); ++n) {
        // Reachability is fixed within the block. Vector instructions with the same EXEC therefore
        // need the same guard.
        std::optional<std::pair<Bool, std::optional<Bool>>> last_guard;
        std::vector<std::optional<Bool>> row;
        const auto& insts = blocks[n].insts;
        for (std::size_t i = 0; i < insts.size(); ++i) {
            const std::optional<Bool>& ex = execs[n][i];
            const auto& f = insts[i].fields;
            const bool scalar = std::holds_alternative<gfx10::Sop>(f) || std::holds_alternative<gfx10::Sopp>(f) ||
                                std::holds_alternative<gfx10::Smem>(f);
            if (!ex || scalar) {
                row.push_back(std::nullopt);
                continue;
            }
            const Bool e = *ex;
            if (last_guard && last_guard->first == e) {
                row.push_back(last_guard->second);
                continue;
            }
            std::optional<Bool> guard;
            if (!b.implies(cond[n], e)) guard = b.given(e, cond[n]);
            last_guard = std::pair{e, guard};
            row.push_back(guard);
        }
        guards.push_back(std::move(row));
    }
    std::vector<LaneExit> exits;
    for (std::size_t n = 0; n < blocks.size(); ++n) {
        const Exit& x = blocks[n].exit;
        switch (x.kind) {
        case Exit::Kind::Jump: exits.push_back(LaneExit::Jump(x.target)); break;
        case Exit::Kind::End: exits.push_back(LaneExit::End()); break;
        case Exit::Kind::Indirect: exits.push_back(LaneExit::Indirect()); break;
        case Exit::Kind::Branch: {
            const Bool w = b.given(when[n].value_or(Bool::False), cond[n]);
            exits.push_back(LaneExit::Branch(w, x.taken, x.not_taken));
            break;
        }
        }
    }
    if (b.overflow()) throw too_large();
    // Exports go by low dword. A low dword read with two high dwords (one rewritten alone) names no
    // one mask: it is left out, so lowering refuses what reads it instead of taking the other pair's
    // lanes 32-63. Not where the low dword holds a number: see the shortcut in `mask`.
    ValueMap<Value> high;
    ValueSet split;
    auto note = [&](const Key& k) {
        const auto [it, inserted] = high.try_emplace(k.first, k.second);
        if (it->second != k.second) split.insert(k.first);
    };
    for (const auto& kv : masks) note(kv.first);
    for (const auto& kv : other) note(kv.first);
    for (const Key& k : seen) note(k);
    for (auto it = masks.begin(); it != masks.end();) {
        if (split.count(it->first.first))
            it = masks.erase(it);
        else
            ++it;
    }
    for (auto it = active.begin(); it != active.end();) {
        if (split.count(*it))
            it = active.erase(it);
        else
            ++it;
    }
    Structured s;
    for (const auto& [k, m] : masks) s.masks.insert_or_assign(k.first, m);
    // Iterate keys in sorted order so that if two low dwords share a high dword, the larger low dword wins.
    std::vector<Key> keys;
    for (const auto& kv : masks) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (const Key& k : keys)
        if (k.first != k.second) s.highs.insert_or_assign(k.second, k.first);
    if (p->wave64) s.paired_masks = masks;
    s.active = active;
    s.wide = wide;
    s.wave_any = wave_any;
    for (const auto& [scc, m] : scc_lane)
        if (!scc_every.count(scc)) s.scc_some.insert_or_assign(scc, m);
    s.phi_masks = std::move(phi_masks);
    s.bools = std::move(b);
    s.exits = std::move(exits);
    s.guards = std::move(guards);
    s.reach = cond;
    return s;
}

std::size_t Lanes::born(Value v) const
{
    if (const auto d = lookup(defs, v)) return index[d->first];
    if (const auto f = lookup(phis, v)) return index[f->first];
    return 0;
}

Bool Lanes::exec(const Inst& inst)
{
    if (const auto k = key(READS, ir::EXEC)) return mask(*k).first;
    return Bool::True;
}

// The mask a value pair holds, and whether it derives from EXEC or a VALU result.
std::pair<Bool, bool> Lanes::mask(Key k)
{
    if (const auto it = masks.find(k); it != masks.end()) return {it->second, true};
    if (const auto it = other.find(k); it != other.end()) return {it->second, false};
    const Program& prog = *p;
    std::pair<Bool, bool> result;
    if (const auto j = join(k)) {
        result = *j;
    } else {
        const auto lo = lookup(defs, k.first);
        const auto hi = lookup(defs, k.second);
        std::optional<Bool> m;
        if (lo && hi && *lo == *hi) {
            if (const Inst* i = def_of(prog, lo)) {
                if (const auto r = sop(*i)) {
                    if (r->dst) m = r->dst->second;
                }
            }
        }
        // What an EXEC write the pass does not follow leaves there.
        auto exec_write = [&](std::optional<std::pair<std::size_t, std::size_t>> d, Value v) {
            const Inst* i = def_of(prog, d);
            return i && (i->write(Reg::S(ir::EXEC)) == v || i->write(Reg::S(ir::EXEC + 1)) == v);
        };
        const bool derived = exec_write(lo, k.first) || exec_write(hi, k.second);
        // shortcut: the atom names the pair by its low dword, and lowering takes the high dword
        // written beside it. For halves written apart (two `s_mov_b32`) lanes 32-63 are wrong unless
        // lowering has no value for the low dword (EXEC, compares) and refuses. Give the atom both
        // dwords when such code turns up.
        if (!m) {
            const std::size_t bn = born(k.first);
            m = b.atom(Atom::Opaque(k.first), bn);
        }
        result = {*m, derived};
    }
    auto& cache = result.second ? masks : other;
    cache.insert_or_assign(k, result.first);
    return result;
}

// A mask joined by phis at a block entry, and whether a mask coming in on a forward edge derives
// from EXEC or a VALU result. Where every lane arriving so far has the incoming mask set (or clear),
// the join is that constant, for a lane can only read it after arriving.
std::optional<std::pair<Bool, bool>> Lanes::join(Key k)
{
    const Program& prog = *p;
    const auto lo = lookup(phis, k.first);
    const auto hi = lookup(phis, k.second);
    std::size_t block;
    if (lo && hi && lo->first != hi->first) return std::nullopt;
    if (lo)
        block = lo->first;
    else if (hi)
        block = hi->first;
    else
        return std::nullopt;
    const ir::Block& target = prog.blocks[block];
    auto arg = [&](std::optional<std::pair<std::size_t, std::size_t>> phi, Value own, std::size_t n) {
        return phi ? target.phis[phi->second].args.at(n) : own;
    };
    std::vector<Key> args;
    for (std::size_t n = 0; n < target.preds.size(); ++n) args.push_back(Key{arg(lo, k.first, n), arg(hi, k.second, n)});
    const Value id = lo ? k.first : k.second;
    bool derived = derived_phis.count(id) != 0;
    // (edge condition, mask) per forward edge; back edges come later.
    std::vector<std::pair<Bool, Bool>> forward;
    for (std::size_t n = 0; n < args.size(); ++n) {
        const auto [q, taken] = incoming[block].at(n);
        if (index[q] < index[block]) {
            const auto [m, d] = mask(args[n]);
            derived |= d;
            const Bool e = edge(first[q], q, taken);
            forward.push_back({e, m});
        } else {
            seen.push_back(args[n]);
        }
    }
    if (forward.size() == args.size() && !forward.empty()) {
        const Bool m = forward.front().second;
        if (std::all_of(forward.begin(), forward.end(), [&](const auto& f) { return f.second == m; }))
            return std::pair{m, derived};
    }
    if (!unresolved.count(id)) {
        for (const bool value : {true, false}) {
            bool holds = true;
            for (const auto& [e, m0] : forward) {
                const Bool m = value ? m0 : b.not_(m0);
                if (!b.implies(e, m)) {
                    holds = false;
                    break;
                }
            }
            if (holds) {
                joins.push_back(Join{block, id, value, derived, args});
                return std::pair{value ? Bool::True : Bool::False, derived};
            }
        }
    }
    if (phi_join_ids.insert(id).second) phi_joins.push_back(PhiJoin{block, id, derived, args});
    const std::size_t bn = index[block];
    const Bool a = b.atom(Atom::Phi(id), bn);
    return std::pair{a, derived};
}

std::optional<std::pair<Bool, bool>> Lanes::operand(const Inst& inst, const Operand& op)
{
    const bool wave64 = p->wave64;
    auto constant = [&](std::uint64_t v) {
        v = wave64 ? v : (v & 0xFFFF'FFFFull);
        if (v == 0) return Bool::False;
        if (v == (~std::uint64_t{0} >> (32 * (wave64 ? 0 : 1)))) return Bool::True;
        return b.atom(Atom::Constant(v), 0);
    };
    switch (op.kind) {
    case Operand::Kind::Scalar: {
        if (op.code == gfx10::NULL_CODE) return std::nullopt;
        const auto k = key(READS, op.code);
        if (!k) return std::nullopt;
        return mask(*k);
    }
    case Operand::Kind::Const: {
        if (op.code < 128 || op.code > 208) return std::nullopt;
        const std::int64_t v = op.code <= 192 ? std::int64_t{op.code} - 128 : 192 - std::int64_t{op.code};
        return std::pair{constant(static_cast<std::uint64_t>(v)), false};
    }
    case Operand::Kind::Literal: return std::pair{constant(std::uint64_t{op.literal}), false};
    default: return std::nullopt;
    }
}

// Lane-mask meaning of a lane-width scalar operation.
std::optional<SopMasks> Lanes::sop(const Inst& inst)
{
    const auto* f = std::get_if<gfx10::Sop>(&inst.fields);
    if (!f) return std::nullopt;
    const std::string_view width = p->wave64 ? "_b64" : "_b32";
    const auto op0 = strip_prefix(inst.name(), "s_");
    if (!op0) return std::nullopt;
    const auto op1 = strip_suffix(*op0, width);
    if (!op1) return std::nullopt;
    const std::string_view op = *op1;
    std::optional<Key> dst;
    if (f->dst && f->dst->is_scalar() && f->dst->code != gfx10::NULL_CODE) dst = key(WRITES, f->dst->code);
    const std::optional<Key> exec_out = key(WRITES, ir::EXEC);
    std::optional<bool> save;
    std::string_view fn = op;
    if (const auto s = strip_suffix(op, "_saveexec")) {
        save = true;
        fn = *s;
    } else if (const auto w = strip_suffix(op, "_wrexec")) {
        save = false;
        fn = *w;
    }
    bool derived = exec_out.has_value() || save.has_value();
    auto arg = [&](std::size_t n) -> std::optional<Bool> {
        if (n >= f->src.size()) return std::nullopt;
        const auto r = operand(inst, f->src[n]);
        if (!r) return std::nullopt;
        derived |= r->second;
        return r->first;
    };
    auto with = [&](const std::optional<Key>& k, Bool m) -> std::optional<std::pair<Key, Bool>> {
        if (!k) return std::nullopt;
        return std::pair{*k, m};
    };
    if (save) {
        const auto s = arg(0);
        if (!s) return std::nullopt;
        const Bool ex = exec(inst);
        const auto nw = combine(fn, *s, ex);
        if (!nw) return std::nullopt;
        const Bool saved = *save ? ex : *nw;
        return SopMasks{with(dst, saved), with(exec_out, *nw), derived, *nw};
    }
    Bool result;
    std::optional<Bool> scc;
    if (op == "mov") {
        const auto a = arg(0);
        if (!a) return std::nullopt;
        result = *a;
    } else if (op == "not") {
        const auto a = arg(0);
        if (!a) return std::nullopt;
        result = b.not_(*a);
        scc = result;
    } else if (op == "wqm") {
        const auto a = arg(0);
        if (!a) return std::nullopt;
        const std::optional<Key> d = dst ? dst : exec_out;
        if (!d) return std::nullopt;
        const Value lo = d->first;
        const std::size_t bn = born(lo);
        const Bool q = b.atom(Atom::Quad(lo), bn);
        result = b.or_(*a, q);
        scc = result;
    } else if (op == "cselect") {
        const auto a = arg(0);
        if (!a) return std::nullopt;
        const auto c = arg(1);
        if (!c) return std::nullopt;
        const auto s = inst.read(Reg::Scc);
        if (!s) return std::nullopt;
        if (clear_after_loop(*s)) {
            result = *c;
        } else {
            const std::size_t bn = born(*s);
            const Bool u = b.atom(Atom::Uniform(*s), bn);
            result = b.ite(u, *a, *c);
        }
    } else {
        const auto a = arg(0);
        if (!a) return std::nullopt;
        const auto c = arg(1);
        if (!c) return std::nullopt;
        const auto r = combine(op, *a, *c);
        if (!r) return std::nullopt;
        result = *r;
        scc = result;
    }
    return SopMasks{with(dst, result), with(exec_out, result), derived, scc};
}

// Whether SCC value `scc` is clear wherever it is read: the instruction before an
// `s_cbranch_execnz` that closes a loop wrote it as "the EXEC it leaves is not empty"
// (`s_andn2_wrexec`, say), and the wave leaves `do { .. } while (EXEC)` with EXEC empty. A read
// inside the loop or behind another way out of it sees a join of this value, not the value itself.
bool Lanes::clear_after_loop(Value scc) const
{
    const auto d = lookup(defs, scc);
    if (!d) return false;
    const auto [bl, i] = *d;
    const ir::Block& block = p->blocks[bl];
    if (i >= block.insts.size() || block.insts.empty()) return false;
    const Inst& writer = block.insts[i];
    const Inst& last = block.insts.back();
    const std::optional<Value> ex = writer.write(Reg::S(ir::EXEC));
    const bool back = block.exit.is_branch() && index[block.exit.taken] <= index[bl];
    return back && i + 2 == block.insts.size() && ex.has_value() && last.name() == "s_cbranch_execnz" &&
           last.read(Reg::S(ir::EXEC)) == ex &&
           (!p->wave64 || last.read(Reg::S(ir::EXEC + 1)) == writer.write(Reg::S(ir::EXEC + 1)));
}

std::optional<Bool> Lanes::combine(std::string_view f, Bool a, Bool c)
{
    if (f == "and") return b.and_(a, c);
    if (f == "or") return b.or_(a, c);
    if (f == "xor") return b.xor_(a, c);
    if (f == "andn2") {
        const Bool nc = b.not_(c);
        return b.and_(a, nc);
    }
    if (f == "orn2") {
        const Bool nc = b.not_(c);
        return b.or_(a, nc);
    }
    if (f == "andn1") {
        const Bool na = b.not_(a);
        return b.and_(na, c);
    }
    if (f == "orn1") {
        const Bool na = b.not_(a);
        return b.or_(na, c);
    }
    if (f == "nand") {
        const Bool r = b.and_(a, c);
        return b.not_(r);
    }
    if (f == "nor") {
        const Bool r = b.or_(a, c);
        return b.not_(r);
    }
    if (f == "xnor") {
        const Bool r = b.xor_(a, c);
        return b.not_(r);
    }
    return std::nullopt;
}

// Records the masks `inst` defines; `exec` is EXEC there, if known.
void Lanes::step(const Inst& inst, std::optional<Bool> exec_now)
{
    const std::string_view name = inst.name();
    if (const auto* f = std::get_if<gfx10::Sop>(&inst.fields)) {
        bool takes_mask = false;
        for (const Operand& s : f->src) {
            if (!s.is_scalar() || s.code == gfx10::NULL_CODE) continue;
            const auto k = key(READS, s.code);
            if (!k) continue;
            const bool phi = phis.count(k->first) || phis.count(k->second);
            bool t = masks.count(*k) != 0;
            if (!t && phi) t = mask(*k).second;
            takes_mask |= t;
        }
        const bool touches_exec = inst.write(Reg::S(ir::EXEC)).has_value() || inst.read(Reg::S(ir::EXEC)).has_value();
        if (takes_mask || touches_exec) {
            const auto r = sop(inst);
            if (r && r->derived) {
                std::vector<std::pair<Key, Bool>> outs;
                if (r->dst) outs.push_back(*r->dst);
                if (r->exec) outs.push_back(*r->exec);
                for (const auto& [k, m] : outs) {
                    masks.insert_or_assign(k, m);
                    // EXEC itself holds the lanes that run here.
                    bool act = r->exec && r->exec->first == k;
                    if (!act && exec_now) act = b.implies(m, *exec_now);
                    if (act) active.insert(k.first);
                }
                const auto scc = inst.write(Reg::Scc);
                if (scc && r->scc) scc_lane.insert_or_assign(*scc, *r->scc);
            }
        }
        // `s_cmp_eq/lg` of a mask: "no lane differs" per lane.
        const std::string_view width = p->wave64 ? "64" : "32";
        if (takes_mask) {
            const auto rest = strip_prefix(name, "s_cmp_");
            const auto scc = inst.write(Reg::Scc);
            if (rest && scc) {
                const auto r = strip_suffix(*rest, width);
                if (r && (*r == "eq_u" || *r == "lg_u") && f->src.size() == 2) {
                    const auto a = operand(inst, f->src[0]);
                    const auto c = operand(inst, f->src[1]);
                    if (a && c) {
                        const Bool ne = b.xor_(a->first, c->first);
                        const bool eq = rest->starts_with("eq");
                        const Bool m = eq ? b.not_(ne) : ne;
                        scc_lane.insert_or_assign(*scc, m);
                        if (eq) scc_every.insert(*scc);
                    }
                }
            }
        }
        // `s_bcnt1` of a mask every running lane sees alike (EXEC under a scalar condition, or EXEC
        // itself): its SCC, "some lane is set", is each lane's own bit. A branch on it then reads no
        // ballot.
        if (takes_mask && name.starts_with("s_bcnt1_i32_b")) {
            const auto scc = inst.write(Reg::Scc);
            if (scc && f->src.size() == 1) {
                const auto a = operand(inst, f->src[0]);
                if (a && a->second) {
                    const Bool m = a->first;
                    const Bool e = exec_now.value_or(Bool::True);
                    const Bool own = b.given(m, e);
                    if (b.implies(m, e)) {
                        const auto sup = b.support(own);
                        if (std::all_of(sup.begin(), sup.end(), [](const Atom& x) { return x.kind == Atom::Kind::Uniform; }))
                            scc_lane.insert_or_assign(*scc, own);
                    }
                }
            }
        }
        return;
    }
    if (const auto* v = std::get_if<gfx10::Valu>(&inst.fields)) {
        const Bool ex = exec(inst);
        auto lane = [&](const Key& k) {
            const std::size_t bn = born(k.first);
            const Bool a = b.atom(Atom::Lane(k.first), bn);
            const Bool m = b.and_(ex, a);
            masks.insert_or_assign(k, m);
            active.insert(k.first);
        };
        // A select reads a mask nothing else may: a join of compare results is resolved here, as for
        // EXEC. A carry-in is only recorded for `finish`: resolving its join would turn a refusal into
        // a per-lane phi, wrong after a divergent branch.
        if ((name == "v_cndmask_b32" || name.find("_co_ci_") != std::string_view::npos) && v->src.size() > 2 &&
            v->src[2].is_scalar() && v->src[2].code != gfx10::NULL_CODE) {
            if (const auto k = key(READS, v->src[2].code)) {
                if (name == "v_cndmask_b32")
                    mask(*k);
                else
                    seen.push_back(*k);
            }
        }
        if (name == "v_div_fmas_f32" || name == "v_div_fmas_f64") {
            if (const auto k = key(READS, ir::VCC)) mask(*k);
        }
        if (name.starts_with("v_cmpx")) {
            if (const auto k = key(WRITES, ir::EXEC)) lane(*k);
        } else {
            std::vector<Operand> ops;
            if (v->dst && name.starts_with("v_cmp")) ops.push_back(*v->dst);
            if (v->carry) ops.push_back(*v->carry);
            for (const Operand& o : ops) {
                if (!o.is_scalar()) continue;
                if (const auto k = key(WRITES, o.code)) lane(*k);
            }
        }
    }
}

// How the console decides the branch `inst` on a lane mask, in terms of the lanes' own `when`
// (branch): taken when some lane's holds (true) or when every lane's does (false). nullopt for a
// branch on a scalar condition.
std::optional<bool> Lanes::wave_kind(const Inst& inst) const
{
    const std::string_view n = inst.name();
    if (n == "s_cbranch_execnz" || n == "s_cbranch_vccnz") return true;
    if (n == "s_cbranch_execz" || n == "s_cbranch_vccz") return false;
    if (n == "s_cbranch_scc0" || n == "s_cbranch_scc1") {
        const auto scc = inst.read(Reg::Scc);
        if (!scc) return std::nullopt;
        if (!scc_lane.count(*scc)) return std::nullopt;
        // set: SCC is 1 when some lane's bit is set (every lane's, for a compare)
        const bool every = scc_every.count(*scc) != 0;
        return (n == "s_cbranch_scc1") != every;
    }
    return std::nullopt;
}

// The per-lane condition under which a block's closing branch is taken.
Bool Lanes::branch(const Inst& inst)
{
    const Bool ex = exec(inst);
    const std::string_view n = inst.name();
    if (n == "s_cbranch_execz") return b.not_(ex);
    if (n == "s_cbranch_execnz") return ex;
    if (n == "s_cbranch_vccz" || n == "s_cbranch_vccnz") {
        Bool vcc = Bool::False;
        if (const auto k = key(READS, ir::VCC)) vcc = mask(*k).first;
        const Bool any = b.and_(vcc, ex);
        return n.ends_with("nz") ? any : b.not_(any);
    }
    const auto scc = inst.read(Reg::Scc);
    if (!scc) return Bool::False;
    Bool set;
    if (const auto it = scc_lane.find(*scc); it != scc_lane.end()) {
        set = it->second;
    } else {
        const std::size_t bn = born(*scc);
        set = b.atom(Atom::Uniform(*scc), bn);
    }
    return n == "s_cbranch_scc1" ? set : b.not_(set);
}

// What a lane knows on entering each block, to a fixpoint. A loop header forgets what the back edge
// says about atoms the loop redefines.
std::vector<Bool> Lanes::paths()
{
    const std::size_t n = p->blocks.size();
    std::vector<Bool> cond(n, Bool::False);
    if (!order.empty()) cond[order.front()] = Bool::True;
    for (std::size_t round = 0; round < MAX_ROUNDS; ++round) {
        bool changed = false;
        for (std::size_t i = 1; i < order.size(); ++i) {
            const std::size_t s = order[i];
            Bool c = Bool::False;
            for (std::size_t k = 0; k < incoming[s].size(); ++k) {
                const auto [q, taken] = incoming[s][k];
                Bool e = edge(cond[q], q, taken);
                if (i <= index[q]) {
                    BoolMemo memo;
                    e = b.forget(e, i, memo);
                }
                c = b.or_(c, e);
            }
            if (c != cond[s]) {
                cond[s] = c;
                changed = true;
            }
        }
        if (b.overflow()) throw too_large();
        if (!changed) return cond;
    }
    throw ShaderError::isa(0, "lane conditions do not settle");
}

#undef READS
#undef WRITES

}  // namespace

Structured structure(const Program& program) { return structure_wide(program, {}); }

Structured structure_wide(const Program& program, const std::set<std::size_t>& wide)
{
    Lanes l(program);
    l.wide = wide;
    for (;;) {
        const auto execs = l.masks_in_order();
        const auto cond = l.paths();
        auto [wrong, new_derived] = l.check_joins(cond);
        if (l.b.overflow()) throw too_large();
        if (wrong.empty() && new_derived.empty()) return l.finish(execs, cond);
        for (Value v : wrong) l.unresolved.insert(v);
        for (Value v : new_derived) l.derived_phis.insert(v);
        l.reset();
    }
}

}  // namespace x2w::vs::structure
