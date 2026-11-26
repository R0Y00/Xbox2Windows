// Wave ORs that only steer a loop over each lane's own bits.
//
// The title's light loops are "scalarised": the lanes' light masks are ORed over the wave and the
// wave walks the set bits of the result together, each lane working only where its own mask has
// the bit:
//
//     merged = WaveActiveBitOr(mask);            // over the lanes that run
//     while (merged != 0) {
//         bit = firstbitlow(merged);
//         if (mask & (1 << bit)) { ... }         // the body
//         merged ^= 1 << bit;
//     }
//
// A console wave64 program in two host waves of 32 cannot exchange the halves' ORs where not every
// lane runs (a group barrier must be reached by every thread). It does not have to: as one thread
// runs it, the loop visits the bits of `merged` in rising order and does something only at the bits
// of the thread's own mask, so any `merged` that holds the thread's own bits gives that thread the
// same run. The host wave's own OR is such a value.
//
// That holds only if the OR is used for nothing else, and if a round of the loop at a bit the
// thread does not have leaves nothing behind. find proves both for each wave OR, with no
// instruction sequence taken on trust:
//
// - the OR and what derives from it as "the bits still to visit" (copies, the loop's join, the
//   first-lane read of it, the set less the visited bit) are read only by the loop's own steps: the
//   test for an empty set, the lowest set bit, the removal of that bit, the test that ends the loop;
// - the lane's test is its own bit of the very value that was ORed, and "the body" is what a lane
//   only reaches with that test true in the same round;
// - outside the body the loop has no side effect and no cross-lane step, writes nothing that is
//   read after the loop or carried into a later round, and is left only by the test for "no bits
//   left";
// - no lane mask that depends on the test is used outside the round that made it.
//
// Anything else is refused there as before ("a wave reduction where not every lane runs") and left
// to the forms of the program in which every lane runs that code (`emit`'s retries: the skips left
// out, the branches around it taken wave-wide), where the halves do exchange.
// `X2W_SHADERCFG_WAVE_OWN_BITS=0` turns this off (checked by the lowering).

#include "x2w_shader/emit/own_bits.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "x2w_shader/emit/wave.h"
#include "x2w_shader/structure.h"

namespace x2w::vs::lower {
// Declared in emit/lower.h. A forward declaration avoids pulling in the lowering headers.
bool reads_mask_as_data(const structure::Structured& st, const ir::Inst& i);
}  // namespace x2w::vs::lower

namespace x2w::vs::own_bits {

using gfx10::Fields;
using gfx10::NULL_CODE;
using gfx10::Operand;
using gfx10::Valu;
using ir::EXEC;
using ir::Exit;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;
using structure::Atom;
using structure::Bool;
using structure::BoolHash;
using structure::LaneExit;
using structure::Structured;
using wave::Op;
using wave::Shape;

namespace {

using Why = const char*;

constexpr Why USED = "a wave OR used as data";
constexpr Why NO_LOOP = "a wave OR that steers no loop over its bits";
constexpr Why SHAPE = "a loop over a wave OR's bits of another shape";
constexpr Why TEST = "a loop over a wave OR's bits whose lanes do not test their own bit";
constexpr Why LEAVES = "a value of a round over a wave OR's bits that outlives the round";
constexpr Why EFFECT = "an effect in a loop over a wave OR's bits outside the lanes' own bits";
constexpr Why CROSS = "a cross-lane step in a loop over a wave OR's bits outside the lanes' own bits";
constexpr Why EXIT = "a loop over a wave OR's bits left another way";
constexpr Why STALE = "a lane's own-bit test used outside its round";
constexpr Why MASK = "a wave OR over a mask a running lane may be outside";

using Site = std::pair<std::size_t, std::size_t>;

struct User {
    enum Kind : std::uint8_t { Inst, Phi };
    Kind kind;
    std::size_t b;
    std::size_t i;  // Inst: the instruction; Phi: the phi
    bool operator==(const User&) const = default;
};

struct Arg {
    enum Kind : std::uint8_t { Val, K, Other };
    Kind kind = Other;
    Value v{};
    std::uint32_t k = 0;
    bool operator==(const Arg&) const = default;
    static Arg val(Value v) { return Arg{Val, v, 0}; }
    static Arg konst(std::uint32_t k) { return Arg{K, Value{}, k}; }
    static Arg other() { return Arg{}; }
};

// Where and why a wave OR is not such a loop.
struct Err {
    std::size_t offset;
    Why what;
};

template <class C, class T>
bool contains(const C& c, const T& x) {
    return std::find(c.begin(), c.end(), x) != c.end();
}

bool is_cmp_scalar(std::string_view name) {
    return name == "s_cmp_eq_u32" || name == "s_cmp_lg_u32" || name == "s_cmp_eq_i32" || name == "s_cmp_lg_i32";
}

// The values instruction `i` of block `b` uses: what its operands read, without EXEC (lane masks
// are followed as booleans) and without the old value of a VGPR it writes, which only lanes
// outside a guard keep.
std::vector<Value> used(const Program& p, const Structured& st, std::size_t b, std::size_t i) {
    const Inst& inst = p.blocks.at(b).insts.at(i);
    const std::vector<Reg> ops = ir::operand_reads(p, inst);
    const bool guarded = st.guards.at(b).at(i).has_value();
    std::vector<Value> out;
    for (const auto& [r, v] : inst.reads) {
        bool keep = true;
        if (r.is_s() && (r.n == EXEC || r.n == EXEC + 1)) {
            keep = false;
        } else if (r.is_v()) {
            keep = contains(ops, r) || guarded;
        }
        if (keep) out.push_back(v);
    }
    return out;
}

std::optional<Value> value(const Inst& inst, const Operand& op) {
    if (op.kind == Operand::Kind::Scalar && op.code != NULL_CODE) return inst.read(Reg::S(op.code));
    if (op.kind == Operand::Kind::Vector) return inst.read(Reg::V(op.reg));
    return std::nullopt;
}

// The sources of a scalar or unmodified vector operation.
std::vector<Arg> args(const Inst& inst) {
    const std::vector<Operand>* src = nullptr;
    if (const auto* s = std::get_if<gfx10::Sop>(&inst.fields)) {
        src = &s->src;
    } else if (const auto* v = std::get_if<Valu>(&inst.fields);
               v && !v->dpp && !v->sdwa && v->abs == 0 && v->neg == 0 && !v->clamp && v->omod == 0) {
        src = &v->src;
    }
    std::vector<Arg> out;
    if (!src) return out;
    for (const Operand& o : *src) {
        if (const auto x = value(inst, o)) {
            out.push_back(Arg::val(*x));
        } else if (o.kind == Operand::Kind::Const) {
            const auto c = wave::detail::inline_constant(o.code);  // resource::inline
            out.push_back(c ? Arg::konst(*c) : Arg::other());
        } else if (o.kind == Operand::Kind::Literal) {
            out.push_back(Arg::konst(o.literal));
        } else {
            out.push_back(Arg::other());
        }
    }
    return out;
}

// The other source of a two-source operation that reads `a`.
std::optional<Arg> other(const Inst& inst, Value a) {
    const auto s = args(inst);
    if (s.size() != 2) return std::nullopt;
    if (s[0].kind == Arg::Val && s[0].v == a) return s[1];
    if (s[1].kind == Arg::Val && s[1].v == a) return s[0];
    return std::nullopt;
}

// The scalar (not SCC) or vector dword an instruction writes, if it writes one.
std::optional<Value> result(const Inst& inst) {
    std::optional<Value> one;
    std::size_t count = 0;
    for (const auto& w : inst.writes) {
        if (w.first == Reg::Scc) continue;
        if (++count == 1) one = w.second;
        if (count == 2) return std::nullopt;
    }
    return one;
}

// The same test as `wave_local`'s and `Lower::inst`'s `effect`, and the messages a program sends.
bool effect(const Inst& inst) {
    const std::string_view name = inst.name();
    const bool ds = std::holds_alternative<gfx10::Ds>(inst.fields);
    return std::holds_alternative<gfx10::Exp>(inst.fields) ||
           (std::holds_alternative<gfx10::Flat>(inst.fields) && !name.starts_with("global_load_")) ||
           name.find("store") != std::string_view::npos || name.find("atomic") != std::string_view::npos ||
           (ds && !(name.starts_with("ds_read") || name.starts_with("ds_swizzle") ||
                    name.find("permute") != std::string_view::npos)) ||
           name == "s_barrier" || name == "s_sendmsg" || name == "s_sendmsghalt";
}

bool contains_text(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

class Ctx {
public:
    Ctx(const Program& program, const Structured& structured) : p(program), st(structured) {
        for (std::size_t b = 0; b < p.blocks.size(); ++b) {
            const auto& block = p.blocks[b];
            for (std::size_t k = 0; k < block.phis.size(); ++k) {
                for (Value a : block.phis[k].args) users[a].push_back(User{User::Phi, b, k});
            }
            for (std::size_t i = 0; i < block.insts.size(); ++i) {
                for (Value v : used(p, st, b, i)) {
                    users[v].push_back(User{User::Inst, b, i});
                    live.insert(v);
                }
                for (const auto& w : block.insts[i].writes) def.insert_or_assign(w.second, Site{b, i});
            }
        }
        // through joins
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& block : p.blocks) {
                for (const auto& phi : block.phis) {
                    if (live.contains(phi.value)) {
                        for (Value a : phi.args) changed |= live.insert(a).second;
                    }
                }
            }
        }
        const std::vector<std::size_t> order = ir::reverse_postorder(p.blocks);
        std::vector<std::size_t> index(p.blocks.size(), SIZE_MAX);
        for (std::size_t n = 0; n < order.size(); ++n) index.at(order[n]) = n;
        idom = ir::idoms(p.blocks, order, index);
        sections.reserve(p.blocks.size());
        for (std::size_t b = 0; b < p.blocks.size(); ++b) sections.push_back(wave::sections(p, b));
    }

    const Program& p;
    const Structured& st;
    // User vectors preserve insertion order. The lookup tables are used only by key.
    std::map<Value, std::vector<User>> users;
    // The instruction writing each value.
    std::map<Value, Site> def;
    // Values an instruction reads, directly or through joins.
    std::set<Value> live;
    std::vector<std::size_t> idom;
    std::vector<std::unordered_map<std::size_t, wave::Section>> sections;

    const Inst& inst(Site u) const { return p.blocks.at(u.first).insts.at(u.second); }

    std::size_t at(Site u) const { return inst(u).machine.offset; }

    const std::vector<User>* users_of(Value v) const {
        const auto it = users.find(v);
        return it == users.end() ? nullptr : &it->second;
    }

    // Whether block `a` dominates block `b`.
    bool dominates(std::size_t a, std::size_t b) const {
        while (true) {
            if (a == b) return true;
            const std::size_t d = b < idom.size() ? idom[b] : SIZE_MAX;
            if (d == b || d == SIZE_MAX) return false;
            b = d;
        }
    }

    std::vector<Site> readers(Value v) const {
        std::vector<Site> out;
        if (const auto* us = users_of(v)) {
            for (const User& u : *us) {
                if (u.kind == User::Inst) out.emplace_back(u.b, u.i);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // The joins reading `v` whose value something reads.
    std::vector<Site> joins(Value v) const {
        std::vector<Site> out;
        if (const auto* us = users_of(v)) {
            for (const User& u : *us) {
                if (u.kind == User::Phi && live.contains(p.blocks.at(u.b).phis.at(u.i).value)) out.emplace_back(u.b, u.i);
            }
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    // The loop of header `h`: the blocks that reach one of its back edges without passing it.
    std::vector<bool> natural_loop(std::size_t h) const {
        std::vector<bool> inside(p.blocks.size(), false);
        std::vector<std::size_t> work;
        for (std::size_t q : p.blocks.at(h).preds) {
            if (dominates(h, q)) work.push_back(q);
        }
        if (work.empty()) return inside;
        inside.at(h) = true;
        while (!work.empty()) {
            const std::size_t b = work.back();
            work.pop_back();
            if (inside.at(b)) continue;
            inside[b] = true;
            const auto& preds = p.blocks.at(b).preds;
            work.insert(work.end(), preds.begin(), preds.end());
        }
        return inside;
    }

    // Whether `b` holds only in lanes where `a` does.
    bool implied(Bool b, const Atom& a, std::unordered_map<Bool, bool, BoolHash>& memo) const {
        if (b == Bool::False) return true;
        if (const auto it = memo.find(b); it != memo.end()) return it->second;
        bool r = false;
        const auto node = st.bools.node(b);
        if (!node) {
            r = false;
        } else if (std::get<0>(*node) == a) {
            r = std::get<1>(*node) == Bool::False;
        } else {
            r = implied(std::get<1>(*node), a, memo) && implied(std::get<2>(*node), a, memo);
        }
        memo.insert_or_assign(b, r);
        return r;
    }

    struct BranchOn {
        Site u;
        std::size_t set;
        std::size_t clear;
    };

    // The conditional branch reading `scc`, as (its place, the block it goes to when SCC is set,
    // the other one).
    std::optional<BranchOn> branch(Value scc) const {
        const auto rs = readers(scc);
        if (rs.size() != 1) return std::nullopt;
        const Site u = rs[0];
        const Exit& ex = p.blocks.at(u.first).exit;
        if (ex.kind != Exit::Kind::Branch) return std::nullopt;
        if (u.second + 1 != p.blocks.at(u.first).insts.size()) return std::nullopt;
        const std::string_view name = inst(u).name();
        if (name == "s_cbranch_scc1") return BranchOn{u, ex.taken, ex.not_taken};
        if (name == "s_cbranch_scc0") return BranchOn{u, ex.not_taken, ex.taken};
        return std::nullopt;
    }

    // One wave OR `t` of `x` over the lanes of `mask`, left by a section of block `sb`: the VGPR
    // values its loop keeps alike, or where and why it is not such a loop.
    std::variant<std::vector<Value>, Err> site(std::size_t sb, Value t, Value x, Value mask) const;
};

std::variant<std::vector<Value>, Err> Ctx::site(std::size_t sb, Value t, Value x, Value mask) const {
    const std::size_t n = p.blocks.size();
    const std::size_t here = p.blocks.at(sb).offset;
    auto is_mask = [&](Value v) { return st.masks.contains(v) || st.highs.contains(v); };
    // every lane that runs the section is one of the lanes ORed
    const auto mit = st.masks.find(mask);
    if (mit == st.masks.end()) return Err{here, MASK};
    const Bool mb = mit->second;
    if (st.bools.meet({{st.reach.at(sb), true}, {mb, false}})) return Err{here, MASK};
    // Bits still to visit and steps executed by the loop.
    std::set<Value> rem{t};
    std::set<Site> pattern;
    std::vector<Value> alike;
    // an unguarded copy
    auto copy = [&](Site u) -> std::optional<Value> {
        const Inst& i = inst(u);
        const std::string_view name = i.name();
        if ((name == "v_mov_b32" || name == "s_mov_b32") && args(i).size() == 1 &&
            !st.guards.at(u.first).at(u.second).has_value()) {
            return result(i);
        }
        return std::nullopt;
    };
    std::vector<Value> init{t};
    // (SCC, whether it is set when the OR is empty)
    std::optional<std::pair<Value, bool>> zero;
    for (const Site u : readers(t)) {
        const Inst& i = inst(u);
        const std::string_view name = i.name();
        std::optional<Value> scc;
        if (is_cmp_scalar(name) && other(i, t) == Arg::konst(0) && !zero && (scc = i.write(Reg::Scc))) {
            zero = std::pair{*scc, contains_text(name, "_eq_")};
            rem.insert(*scc);
        } else if (const auto w = copy(u)) {
            init.push_back(*w);
            rem.insert(*w);
            if (i.writes.at(0).first.is_v()) alike.push_back(*w);
        } else {
            return Err{at(u), USED};
        }
        pattern.insert(u);
    }
    // the loop's join of them
    std::optional<Site> join;
    for (const Value v : init) {
        if (v != t) {
            const auto rs = readers(v);
            if (!rs.empty()) return Err{at(rs[0]), USED};
        }
        for (const Site j : joins(v)) {
            if (join && *join != j) return Err{p.blocks.at(j.first).offset, USED};
            join = j;
        }
    }
    if (!join) return Err{here, NO_LOOP};
    const std::size_t h = join->first;
    const std::size_t k = join->second;
    const ir::Phi& phi = p.blocks.at(h).phis.at(k);
    const Value r = phi.value;
    rem.insert(r);
    const std::size_t head = p.blocks.at(h).offset;
    const std::vector<bool> lp = natural_loop(h);
    if (!lp[h] || lp.at(sb)) return Err{head, NO_LOOP};
    // the set as a scalar: the first lane's copy of it
    Value cur = r;
    if (phi.reg.is_v()) {
        alike.push_back(r);
        const auto rs = readers(r);
        if (rs.size() != 1) return Err{head, USED};
        const Site u = rs[0];
        const Inst& i = inst(u);
        if (i.name() != "v_readfirstlane_b32" || args(i).size() != 1 || !lp.at(u.first)) return Err{at(u), USED};
        pattern.insert(u);
        const auto c = result(i);
        if (!c) return Err{at(u), SHAPE};
        if (!joins(r).empty()) return Err{head, USED};
        cur = *c;
    }
    rem.insert(cur);
    // its lowest bit, the set without it, the tests for "that was the last"
    std::optional<Value> idx;
    std::optional<std::pair<Value, Value>> next;
    // (SCC, whether it is set while bits remain)
    std::vector<std::pair<Value, bool>> more;
    std::vector<Value> bits;
    for (const Site u : readers(cur)) {
        const Inst& i = inst(u);
        const std::string_view name = i.name();
        if (!lp.at(u.first)) return Err{at(u), USED};
        const std::vector<Arg> a = args(i);
        if (name == "s_ff1_i32_b32" && a.size() == 1 && a[0].kind == Arg::Val && !idx) {
            idx = result(i);
        } else if ((name == "s_xor_b32" || (name == "s_andn2_b32" && a.size() == 2 && a[0].kind == Arg::Val &&
                                            a[1].kind == Arg::Val)) &&
                   !next) {
            Value bit{};
            if (name == "s_andn2_b32") {
                // ("s_andn2_b32", [Arg::Val(c), Arg::Val(b)]) if *c == cur && *b != cur
                if (!(a[0].v == cur && a[1].v != cur)) return Err{at(u), USED};
                bit = a[1].v;
            } else {
                const auto o = other(i, cur);
                if (!(o && o->kind == Arg::Val && o->v != cur)) return Err{at(u), USED};
                bit = o->v;
            }
            const auto w = result(i);
            if (!w) return Err{at(u), SHAPE};
            next = std::pair{*w, bit};
            bits.push_back(bit);
            if (const auto scc = i.write(Reg::Scc)) {
                more.emplace_back(*scc, true);
                rem.insert(*scc);
            }
        } else if (is_cmp_scalar(name)) {
            const auto o = other(i, cur);
            const auto scc = i.write(Reg::Scc);
            if (!(o && o->kind == Arg::Val && scc)) return Err{at(u), USED};
            bits.push_back(o->v);
            more.emplace_back(*scc, contains_text(name, "_lg_"));
            rem.insert(*scc);
        } else {
            return Err{at(u), USED};
        }
        pattern.insert(u);
    }
    if (cur != r && !joins(cur).empty()) return Err{head, USED};
    if (!idx || !next) return Err{head, SHAPE};
    const Value nx = next->first;
    const Value bit = next->second;
    if (std::any_of(bits.begin(), bits.end(), [&](Value b) { return b != bit; })) return Err{head, SHAPE};
    // the bit: 1 shifted by the lowest set bit's number (a shift takes five bits of it)
    auto made = [&](Value v) -> std::optional<Site> {
        const auto it = def.find(v);
        if (it == def.end()) return std::nullopt;
        return it->second;
    };
    std::optional<Value> shifted;
    if (const auto u = made(bit); u && inst(*u).name() == "s_lshl_b32" && lp.at(u->first)) {
        const auto a = args(inst(*u));
        if (a.size() == 2 && a[0] == Arg::konst(1) && a[1].kind == Arg::Val) shifted = a[1].v;
    }
    bool by_idx = false;
    if (shifted) {
        const Value sh = *shifted;
        if (sh == *idx) {
            by_idx = true;
        } else if (const auto u = made(sh)) {
            const Inst& i = inst(*u);
            by_idx = i.name() == "s_and_b32" && other(i, *idx) == Arg::konst(31);
        }
    }
    if (!by_idx) return Err{head, SHAPE};
    // the set less the bit: back into the join, or tested for empty
    rem.insert(nx);
    std::vector<Value> nexts{nx};
    for (const Site u : readers(nx)) {
        const Inst& i = inst(u);
        const std::string_view name = i.name();
        std::optional<Value> scc;
        if (is_cmp_scalar(name) && other(i, nx) == Arg::konst(0) && (scc = i.write(Reg::Scc))) {
            more.emplace_back(*scc, contains_text(name, "_lg_"));
            rem.insert(*scc);
        } else if (const auto w = copy(u)) {
            nexts.push_back(*w);
            rem.insert(*w);
            if (i.writes.at(0).first.is_v()) alike.push_back(*w);
        } else {
            return Err{at(u), USED};
        }
        pattern.insert(u);
    }
    for (const Value v : nexts) {
        if (v != nx) {
            const auto rs = readers(v);
            if (!rs.empty()) return Err{at(rs[0]), USED};
        }
        const auto js = joins(v);
        if (std::any_of(js.begin(), js.end(), [&](Site j) { return j != Site{h, k}; })) return Err{head, USED};
    }
    {
        const auto& preds = p.blocks.at(h).preds;
        const std::size_t m = std::min(preds.size(), phi.args.size());
        for (std::size_t e = 0; e < m; ++e) {
            const bool ok = lp.at(preds[e]) ? contains(nexts, phi.args[e]) : contains(init, phi.args[e]);
            if (!ok) return Err{head, SHAPE};
        }
    }
    // the loop goes on while bits remain and is left when none does
    std::set<std::size_t> leaves;
    for (const auto& [scc, set_while_more] : more) {
        if (!joins(scc).empty()) return Err{head, USED};
        if (readers(scc).empty()) continue;
        const auto br = branch(scc);
        if (!br) return Err{head, USED};
        const std::size_t on = set_while_more ? br->set : br->clear;
        const std::size_t out = set_while_more ? br->clear : br->set;
        if (!lp.at(br->u.first) || !lp.at(on) || lp.at(out)) return Err{at(br->u), SHAPE};
        leaves.insert(br->u.first);
        pattern.insert(br->u);
    }
    if (leaves.empty()) return Err{head, SHAPE};
    // "no bit at all" skips the loop; what lies between that test and the loop runs only when
    // there is one
    std::vector<bool> pre(n, false);
    if (zero) {
        const auto [scc, set_when_empty] = *zero;
        if (!joins(scc).empty()) return Err{here, USED};
        if (!readers(scc).empty()) {
            const auto br = branch(scc);
            if (!br) return Err{here, USED};
            const Site u = br->u;
            const std::size_t none = set_when_empty ? br->set : br->clear;
            const std::size_t some = set_when_empty ? br->clear : br->set;
            if (none == some || lp.at(u.first)) return Err{at(u), SHAPE};
            if (some != h) {
                const auto& sp = p.blocks.at(some).preds;
                if (!(sp.size() == 1 && sp[0] == u.first) || !dominates(some, h)) return Err{at(u), SHAPE};
                for (std::size_t b = 0; b < n; ++b) pre[b] = dominates(some, b) && !dominates(h, b);
                for (std::size_t b = 0; b < n; ++b) {
                    if (!pre[b]) continue;
                    const Exit& e = p.blocks[b].exit;
                    bool inside = false;
                    if (e.kind != Exit::Kind::End && e.kind != Exit::Kind::Indirect) {
                        const auto succ = e.successors();
                        inside = std::all_of(succ.begin(), succ.end(), [&](std::size_t s) { return pre.at(s) || s == h; });
                    }
                    if (!inside) return Err{p.blocks[b].offset, SHAPE};
                }
            }
            pattern.insert(u);
        }
    }
    // ---- the lane's test of its own bit ----
    // (the atom of the compare, its block, its index)
    struct Test {
        Atom atom;
        std::size_t b;
        std::size_t i;
    };
    std::vector<Test> tests;
    for (const Site u : readers(bit)) {
        const Inst& i = inst(u);
        if (i.name() != "v_and_b32" || other(i, bit) != Arg::val(x) || st.guards.at(u.first).at(u.second).has_value()) {
            continue;
        }
        const auto y = result(i);
        if (!y) continue;
        for (const Site c : readers(*y)) {
            const Inst& ci = inst(c);
            const std::string_view cn = ci.name();
            const bool ne = cn == "v_cmp_ne_u32" || cn == "v_cmpx_ne_u32" || cn == "v_cmp_lg_u32" ||
                            cn == "v_cmpx_lg_u32" || cn == "v_cmp_ne_i32" || cn == "v_cmpx_ne_i32" ||
                            cn == "v_cmp_lg_i32" || cn == "v_cmpx_lg_i32";
            if (ne && other(ci, *y) == Arg::konst(0) && lp.at(c.first)) {
                for (const auto& w : ci.writes) tests.push_back(Test{Atom::Lane(w.second), c.first, c.second});
            }
        }
    }
    if (tests.empty()) return Err{head, TEST};
    std::vector<std::unordered_map<Bool, bool, BoolHash>> memo(tests.size());
    // what a lane only reaches with its own bit set, in the round of that test
    std::vector<bool> body_block(n, false);
    for (std::size_t b = 0; b < n; ++b) {
        if (!lp[b]) continue;
        bool any = false;
        for (std::size_t j = 0; j < tests.size() && !any; ++j) {
            const Test& tj = tests[j];
            any = b != tj.b && dominates(tj.b, b) && implied(st.reach.at(b), tj.atom, memo[j]);
        }
        body_block[b] = any;
    }
    auto body_inst = [&](std::size_t b, std::size_t i) {
        if (body_block[b]) return true;
        if (!lp[b]) return false;
        const auto& g = st.guards.at(b).at(i);
        if (!g) return false;
        for (std::size_t j = 0; j < tests.size(); ++j) {
            const Test& tj = tests[j];
            if (((b == tj.b && i > tj.i) || (b != tj.b && dominates(tj.b, b))) && implied(*g, tj.atom, memo[j])) {
                return true;
            }
        }
        return false;
    };
    // ---- a round outside the body leaves nothing behind ----
    for (std::size_t b = 0; b < n; ++b) {
        if (!(lp[b] || pre[b])) continue;
        const auto& insts = p.blocks[b].insts;
        for (std::size_t i = 0; i < insts.size(); ++i) {
            const Inst& in = insts[i];
            if (lp[b] && body_inst(b, i)) continue;
            const Site u{b, i};
            for (const auto& [reg, w] : in.writes) {
                if (rem.contains(w) || is_mask(w) || (reg.is_s() && (reg.n == EXEC || reg.n == EXEC + 1))) continue;
                if (const auto* us = users_of(w)) {
                    for (const User& user : *us) {
                        bool stays = false;
                        if (user.kind == User::Inst) {
                            stays = lp.at(user.b) || pre.at(user.b);
                        } else {
                            stays = body_block.at(user.b) || !live.contains(p.blocks.at(user.b).phis.at(user.i).value);
                        }
                        if (!stays) return Err{at(u), LEAVES};
                    }
                }
            }
            if (pattern.contains(u)) continue;
            if (effect(in)) return Err{at(u), EFFECT};
            const std::string_view name = in.name();
            bool cross = name.starts_with("ds_swizzle") || contains_text(name, "permute") ||
                         name.starts_with("v_mbcnt") || contains_text(name, "lane");
            if (!cross) {
                const auto* v = std::get_if<Valu>(&in.fields);
                cross = v && v->dpp.has_value();
            }
            if (!cross) cross = lower::reads_mask_as_data(st, in);
            if (!cross) {
                for (const auto& [s, sec] : sections.at(b)) {
                    if (i >= s && i <= sec.end) {
                        cross = true;
                        break;
                    }
                }
            }
            if (cross) return Err{at(u), CROSS};
            const auto uv = used(p, st, b, i);
            if (std::any_of(uv.begin(), uv.end(), [&](Value v) { return rem.contains(v); })) return Err{at(u), USED};
        }
        // left only by "no bits left"; the body may leave it as it likes
        if (lp[b] && !body_block[b]) {
            const Exit& e = p.blocks[b].exit;
            bool ok = false;
            if (e.kind != Exit::Kind::End && e.kind != Exit::Kind::Indirect) {
                const auto succ = e.successors();
                ok = leaves.contains(b) || std::all_of(succ.begin(), succ.end(), [&](std::size_t s) { return lp.at(s); });
            }
            if (!ok) return Err{p.blocks[b].offset, EXIT};
        }
    }
    // ---- no mask that depends on the test outlives its round ----
    std::unordered_set<Atom, structure::AtomHash> dep;
    for (const Test& tt : tests) dep.insert(tt.atom);
    auto depends = [&](Bool e) {
        const auto sup = st.bools.support(e);
        return std::any_of(sup.begin(), sup.end(), [&](const Atom& a) { return dep.contains(a); });
    };
    bool changed = true;
    while (changed) {
        changed = false;
        // (a fixed point: the order of the map does not change it)
        for (const auto& [id, edges] : st.phi_masks) {
            if (!dep.contains(Atom::Phi(id)) && std::any_of(edges.begin(), edges.end(), depends)) {
                dep.insert(Atom::Phi(id));
                changed = true;
            }
        }
    }
    std::unordered_map<Bool, bool, BoolHash> seen;
    auto tainted = [&](Bool b) -> bool {
        if (const auto it = seen.find(b); it != seen.end()) return it->second;
        const bool r = depends(b);
        seen.emplace(b, r);
        return r;
    };
    for (std::size_t b = 0; b < n; ++b) {
        if (lp[b] && std::any_of(tests.begin(), tests.end(), [&](const Test& tt) { return dominates(tt.b, b); })) {
            continue;
        }
        bool stale = false;
        for (const auto& g : st.guards.at(b)) {
            if (g && tainted(*g)) {
                stale = true;
                break;
            }
        }
        if (!stale) {
            const LaneExit& e = st.exits.at(b);
            stale = e.kind == LaneExit::Kind::Branch && tainted(e.when);
        }
        if (stale) return Err{p.blocks[b].offset, STALE};
    }
    // (any mask found gives the same error: the order of the map does not matter)
    for (const auto& [v, m] : st.masks) {
        if (!tainted(m)) continue;
        if (const auto* us = users_of(v)) {
            for (const User& user : *us) {
                bool outside = false;
                if (user.kind == User::Inst) {
                    outside = !lp.at(user.b);
                } else {
                    outside = !lp.at(user.b) && live.contains(p.blocks.at(user.b).phis.at(user.i).value);
                }
                if (outside) return Err{head, STALE};
            }
        }
    }
    return alike;
}

}  // namespace

OwnBits find(const Program& p, const Structured& st, bool why) {
    OwnBits out;
    std::optional<Ctx> ctx;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto sections = wave::sections(p, b);
        std::vector<std::size_t> starts;
        starts.reserve(sections.size());
        for (const auto& [s, _] : sections) starts.push_back(s);
        std::sort(starts.begin(), starts.end());
        for (const std::size_t s : starts) {
            const wave::Section& sec = sections.at(s);
            std::vector<std::pair<Value, Shape>> left(sec.shapes.begin(), sec.shapes.end());
            std::sort(left.begin(), left.end(), [](const auto& l, const auto& r) { return l.first.n < r.first.n; });
            for (const auto& [t, shape] : left) {
                if (!(shape.kind == Shape::Active && shape.op == Op::Or && shape.k == 0)) continue;
                const Value x = shape.a;
                const Value mask = shape.b;
                if (!ctx) ctx.emplace(p, st);
                // (values only the section itself reads are not results)
                auto inside = [&](const User& u) { return u.kind == User::Inst && u.b == b && u.i > s && u.i < sec.end; };
                const auto* us = ctx->users_of(t);
                if (!us || std::all_of(us->begin(), us->end(), inside)) continue;
                auto r = ctx->site(b, t, x, mask);
                if (auto* alike = std::get_if<std::vector<Value>>(&r)) {
                    out.totals.insert(t);
                    out.alike.insert(alike->begin(), alike->end());
                } else if (why) {
                    const Err& e = std::get<Err>(r);
                    std::fprintf(stderr,
                                 "WAVE64-OWN-BITS the wave OR at 0x%zx: not a loop over the lanes' own bits at 0x%zx: %s\n",
                                 p.blocks[b].insts.at(s).machine.offset, e.offset, e.what);
                }
            }
        }
    }
    return out;
}

}  // namespace x2w::vs::own_bits
