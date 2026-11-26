// clean-up of a finished module before it is written. The lowering emits one instruction after another as
// the console code names them, and nothing stands between it and the driver's compiler. Every pass here is
// exact: the module computes the same values and has the same effects for every input.
//
// - DEAD: instructions without an effect whose result nothing reads are left out (the lowering
//   computes every dword of every register a console instruction writes; a guarded write reads
//   the register's old value whether anything reads the new one or not).
// - FOLD: integer arithmetic on constants is done here, and identities (`x + 0`, `x & -1`, a
//   select between a value and itself, a cast there and back, a join of one value) give their
//   operand. Float arithmetic is left alone: `x + 0.0` and `x * 1.0` are not `x` for every `x`.
// - REUSE: an instruction without an effect whose operands are those of an earlier one that
//   every path to it has run gives that one's result (a descriptor handle made again in each
//   block, an index clamped again, a register read as a float again).
// - FACTS: in the blocks only one side of a branch leads to, the branch's condition is that
//   side's constant, and so is what the condition was made of (both terms of an `and` that held,
//   the opposite comparison). The lowering branches on a lane's condition and then guards each
//   instruction behind the branch with the same condition.
// - LOADS: within a block, a read of the same place as an earlier read gives that one's result
//   while no store, atomic, barrier or other call with an effect stands between them (another
//   thread's unsynchronised store may fall between any two reads or before both: the title's
//   code cannot tell which).
//
// The known, read, joins, facts and func_ids tables are only queried or updated. Their iteration order does
// not affect output.

#include "x2w_shader/emit/opt.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace x2w::vs::bitcode {

namespace {

using opt::Effect;

constexpr std::size_t NONE = std::numeric_limits<std::size_t>::max();

// a NaN operand gives the other.
float fmax_rs(float a, float b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return a > b ? a : b;
}

void mix(std::size_t& h, std::uint64_t v) { h ^= std::size_t(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); }

std::uint64_t vbits(V v) { return (std::uint64_t(v.kind) << 32) | v.i; }

// What makes two instructions without an effect the same value.
struct Key {
    enum Kind : std::uint8_t { Bin, Cast, Cmp, Select, Extract, Call, Gep, Load };
    Kind kind = Bin;
    std::uint8_t code = 0;
    // Cast: type. Extract: index. Call: function. Gep: pointee type.
    std::uint32_t n = 0;
    std::vector<V> vs;
    bool operator==(const Key&) const = default;
};

struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        std::size_t h = k.kind;
        mix(h, k.code);
        mix(h, k.n);
        for (V v : k.vs) mix(h, vbits(v));
        return h;
    }
};

using Cmp3 = std::tuple<std::uint8_t, V, V>;

struct Cmp3Hash {
    std::size_t operator()(const Cmp3& k) const noexcept {
        std::size_t h = std::get<0>(k);
        mix(h, vbits(std::get<1>(k)));
        mix(h, vbits(std::get<2>(k)));
        return h;
    }
};

using JoinKey = std::vector<std::pair<std::size_t, V>>;

struct JoinKeyHash {
    std::size_t operator()(const JoinKey& k) const noexcept {
        std::size_t h = k.size();
        for (const auto& [b, v] : k) {
            mix(h, b);
            mix(h, vbits(v));
        }
        return h;
    }
};

// The blocks reached from the entry, each after the blocks that dominate it, and each block's
// children in the dominator tree.
struct Tree {
    std::vector<std::size_t> order;
    std::vector<std::vector<std::size_t>> children;
    // Per block: the reached blocks that branch to it, once per edge.
    std::vector<std::vector<std::size_t>> preds;
};

// What holds in the block being walked: values that are a known bit there, and comparisons (by
// their operands) with a known outcome.
struct Facts {
    std::unordered_map<V, bool, VHash> bits;
    std::unordered_map<Cmp3, bool, Cmp3Hash> compares;
};

// The facts a block added, to forget when its part of the tree is done.
struct Learned {
    std::vector<V> bits;
    std::vector<Cmp3> compares;
};

std::uint64_t mask(std::uint32_t bits) { return bits >= 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1; }

// `x` of `bits` bits as a signed number.
std::int64_t signed_(std::uint32_t bits, std::uint64_t x) {
    if (bits < 64 && ((x >> (bits - 1)) & 1) != 0) return std::int64_t(x | ~mask(bits));
    return std::int64_t(x);
}

// What `v` stands for after the replacements so far.
V resolve(const std::vector<std::optional<V>>& repl, V v) {
    while (v.kind == V::Inst) {
        if (v.i >= repl.size()) bug("index out of bounds: resolve");
        const auto& r = repl[v.i];
        if (!r) break;
        v = *r;
    }
    return v;
}

// Replaces every value `op` reads by `f` of it.
template <class F>
void map_operands(Op& op, F&& f) {
    switch (op.kind) {
    case Op::Bin:
    case Op::Cmp:
    case Op::Store:
    case Op::AtomicRmw:
        op.a = f(op.a);
        op.b = f(op.b);
        break;
    case Op::Cast:
    case Op::Extract:
    case Op::Load:
        op.a = f(op.a);
        break;
    case Op::Select:
    case Op::CmpXchg:
        op.a = f(op.a);
        op.b = f(op.b);
        op.c = f(op.c);
        break;
    case Op::Call:
        for (V& a : op.list) a = f(a);
        break;
    case Op::Phi:
        for (auto& x : op.incoming) x.first = f(x.first);
        break;
    case Op::Gep:
        op.a = f(op.a);
        for (V& i : op.list) i = f(i);
        break;
    }
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}
std::string_view strip_dx_op(std::string_view name) {
    return starts_with(name, "dx.op.") ? name.substr(6) : name;
}

std::optional<std::uint32_t> int_bits(const Module& m, Ty ty) {
    const TypeDef& d = m.types.at(ty.id);
    if (d.kind == TypeDef::Int) return d.n;
    return std::nullopt;
}

bool is_op(const Module& m, V v, Op::Kind kind) { return v.kind == V::Inst && m.insts.at(v.i).op.kind == kind; }

Tree dominator_tree(const Module& m) {
    const std::size_t n = m.blocks.size();
    auto succs = [&](std::size_t b) { return successors(m.blocks.at(b).term); };
    // Reverse postorder, as `write` lays the blocks out.
    std::vector<bool> seen(n, false);
    std::vector<std::size_t> post;
    std::vector<std::pair<std::size_t, std::size_t>> stack = {{0, 0}};
    seen.at(0) = true;
    while (!stack.empty()) {
        auto& top = stack.back();
        auto [b, k] = top;
        top.second += 1;
        Succs s = succs(b);
        if (k < s.size()) {
            std::size_t t = s.to[k];
            if (!seen.at(t)) {
                seen[t] = true;
                stack.push_back({t, 0});
            }
        } else {
            post.push_back(b);
            stack.pop_back();
        }
    }
    std::reverse(post.begin(), post.end());
    std::vector<std::size_t> order = std::move(post);
    std::vector<std::size_t> number(n, NONE);
    for (std::size_t k = 0; k < order.size(); ++k) number[order[k]] = k;
    std::vector<std::vector<std::size_t>> preds(n);
    for (std::size_t b : order)
        for (std::size_t s : succs(b)) preds.at(s).push_back(b);
    // Immediate dominators (Cooper, Harvey and Kennedy's iteration).
    std::vector<std::size_t> idom(n, NONE);
    idom[0] = 0;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t o = 1; o < order.size(); ++o) {
            std::size_t b = order[o];
            std::size_t nw = NONE;
            for (std::size_t p : preds[b]) {
                if (idom[p] == NONE) continue;
                if (nw == NONE) {
                    nw = p;
                    continue;
                }
                std::size_t x = p, y = nw;
                while (x != y) {
                    while (number[x] > number[y]) x = idom[x];
                    while (number[y] > number[x]) y = idom[y];
                }
                nw = x;
            }
            if (idom[b] != nw) {
                idom[b] = nw;
                changed = true;
            }
        }
    }
    std::vector<std::vector<std::size_t>> children(n);
    for (std::size_t o = 1; o < order.size(); ++o) {
        std::size_t b = order[o];
        if (idom[b] >= n) bug("index out of bounds: idom");
        children[idom[b]].push_back(b);
    }
    return Tree{std::move(order), std::move(children), std::move(preds)};
}

// Notes that `v` (an i1) is `holds` in the block being walked, and what follows from it for the
// values `v` was made of.
void learn(const Module& m, V v, bool holds, Facts& facts, Learned& learned, std::uint32_t depth) {
    if (v.kind != V::Inst) return;
    if (facts.bits.contains(v)) return;
    facts.bits.emplace(v, holds);
    learned.bits.push_back(v);
    if (depth == 0) return;
    const Op& op = m.insts.at(v.i).op;
    if (op.kind == Op::Bin && op.code == std::uint8_t(Bin::And) && holds) {
        // Both terms of an `and` that holds hold; neither of an `or` that does not.
        learn(m, op.a, true, facts, learned, depth - 1);
        learn(m, op.b, true, facts, learned, depth - 1);
    } else if (op.kind == Op::Bin && op.code == std::uint8_t(Bin::Or) && !holds) {
        learn(m, op.a, false, facts, learned, depth - 1);
        learn(m, op.b, false, facts, learned, depth - 1);
    } else if (op.kind == Op::Bin && op.code == std::uint8_t(Bin::Xor)) {
        auto ka = m.int_k(op.a);
        auto kb = m.int_k(op.b);
        V x;
        std::uint64_t k;
        if (!ka && kb && kb->first == 1) {
            x = op.a;
            k = kb->second;
        } else if (ka && ka->first == 1 && !kb) {
            x = op.b;
            k = ka->second;
        } else {
            return;
        }
        learn(m, x, holds != (k != 0), facts, learned, depth - 1);
    } else if (op.kind == Op::Cmp) {
        // The opposite comparison of the same operands.
        Cmp3 key{opt::opposite(op.code), op.a, op.b};
        if (!facts.compares.contains(key)) {
            facts.compares.emplace(key, !holds);
            learned.compares.push_back(key);
        }
    }
}

// The constant `x` of the integer type `ty`, kept sign-extended as Module::i32 and Module::bool_
// keep theirs.
V k_(Module& m, Ty ty, std::uint32_t bits, std::uint64_t x) { return m.int_const(ty, signed_(bits, x & mask(bits))); }

bool int_pred_holds(std::uint8_t p, std::uint64_t x, std::uint64_t y, std::uint32_t bits) {
    std::int64_t sx = signed_(bits, x), sy = signed_(bits, y);
    switch (p) {
    case pred::EQ:
        return x == y;
    case pred::NE:
        return x != y;
    case pred::IUGT:
        return x > y;
    case pred::IUGE:
        return x >= y;
    case pred::IULT:
        return x < y;
    case pred::IULE:
        return x <= y;
    case pred::SGT:
        return sx > sy;
    case pred::SGE:
        return sx >= sy;
    case pred::SLT:
        return sx < sy;
    default:
        return sx <= sy;
    }
}

// The value instruction `i` is known to have without it, if any: a constant or an earlier value.
// May instead give `i` simpler operands (`changed` says so).
std::optional<V> simplify(Module& m, std::uint32_t i, bool& changed) {
    const Ty ty = m.insts.at(i).ty;
    // Constants made below grow consts and types, never insts: the reference stays valid.
    const Op& op = m.insts[i].op;
    switch (op.kind) {
    case Op::Bin: {
        // Floats: `x + 0.0`, `x * 1.0` and the like are not `x` for every `x` (a negative zero,
        // a signalling NaN).
        auto bits_ = int_bits(m, ty);
        if (!bits_) return std::nullopt;
        const std::uint32_t bits = *bits_;
        const V a = op.a, b = op.b;
        std::optional<std::uint64_t> ka, kb;
        if (auto k = m.int_k(a)) ka = k->second;
        if (auto k = m.int_k(b)) kb = k->second;
        const std::uint64_t all = mask(bits);
        const Bin bo = Bin(op.code);
        if (ka && kb) {
            const std::uint64_t x = *ka, y = *kb;
            // A shift by the width or more and a division by zero have no defined result: left
            // as they are.
            std::optional<std::uint32_t> amount;
            if (y <= 0xFFFFFFFFull && std::uint32_t(y) < bits) amount = std::uint32_t(y);
            std::uint64_t r;
            switch (bo) {
            case Bin::Add:
                r = x + y;
                break;
            case Bin::Sub:
                r = x - y;
                break;
            case Bin::Mul:
                r = x * y;
                break;
            case Bin::And:
                r = x & y;
                break;
            case Bin::Or:
                r = x | y;
                break;
            case Bin::Xor:
                r = x ^ y;
                break;
            case Bin::Shl:
                if (!amount) return std::nullopt;
                r = x << *amount;
                break;
            case Bin::LShr:
                if (!amount) return std::nullopt;
                r = x >> *amount;
                break;
            case Bin::AShr:
                if (!amount) return std::nullopt;
                r = std::uint64_t(signed_(bits, x) >> *amount);
                break;
            case Bin::UDiv:
                if (y == 0) return std::nullopt;
                r = x / y;
                break;
            case Bin::URem:
                if (y == 0) return std::nullopt;
                r = x % y;
                break;
            default:
                return std::nullopt;
            }
            return k_(m, ty, bits, r);
        }
        auto is = [](std::optional<std::uint64_t> k, std::uint64_t v) { return k && *k == v; };
        auto any = [&](std::initializer_list<Bin> l) { return std::find(l.begin(), l.end(), bo) != l.end(); };
        if (any({Bin::Add, Bin::Sub, Bin::Or, Bin::Xor, Bin::Shl, Bin::LShr, Bin::AShr}) && is(kb, 0)) return a;
        if (any({Bin::Add, Bin::Or, Bin::Xor}) && is(ka, 0)) return b;
        if (any({Bin::Mul, Bin::UDiv}) && is(kb, 1)) return a;
        if (bo == Bin::Mul && is(ka, 1)) return b;
        if (any({Bin::Mul, Bin::And}) && is(kb, 0)) return b;
        if (any({Bin::Mul, Bin::And}) && is(ka, 0)) return a;
        if (bo == Bin::And && is(kb, all)) return a;
        if (bo == Bin::And && is(ka, all)) return b;
        if (bo == Bin::Or && is(kb, all)) return b;
        if (bo == Bin::Or && is(ka, all)) return a;
        if (any({Bin::And, Bin::Or}) && a == b) return a;
        return std::nullopt;
    }
    case Op::Cmp: {
        const std::uint8_t p = op.code;
        const V a = op.a, b = op.b;
        if (p <= pred::TRUE_) {
            // Two float constants compare one way (`v_mul_legacy` asks whether its constant
            // factor is zero).
            if (a.kind != V::Const || b.kind != V::Const) return std::nullopt;
            const Const& cx = m.consts.at(a.i).second;
            const Const& cy = m.consts.at(b.i).second;
            if (cx.kind != Const::Float || cy.kind != Const::Float) return std::nullopt;
            const float x = std::bit_cast<float>(cx.bits), y = std::bit_cast<float>(cy.bits);
            const bool unordered = std::isnan(x) || std::isnan(y);
            // Bits 0-2 of a predicate: equal, greater, less; bit 3: or unordered.
            bool holds;
            if (unordered)
                holds = (p & 8) != 0;
            else
                holds = ((p & 1) != 0 && x == y) || ((p & 2) != 0 && x > y) || ((p & 4) != 0 && x < y);
            return m.bool_(holds);
        }
        if (!(p >= pred::EQ && p <= pred::SLE)) return std::nullopt;
        auto ka = m.int_k(a);
        auto kb = m.int_k(b);
        if (ka && kb) return m.bool_(int_pred_holds(p, ka->second, kb->second, ka->first));
        if (a == b) return m.bool_(int_pred_holds(p, 0, 0, 32));
        // A bit widened and tested again is the bit.
        if (p == pred::NE && kb && kb->second == 0 && a.kind == V::Inst) {
            const Op& z = m.insts.at(a.i).op;
            if (z.kind == Op::Cast && z.code == std::uint8_t(Cast::ZExt) && m.type_of(z.a) == ty) return z.a;
        }
        return std::nullopt;
    }
    case Op::Select: {
        const V c = op.a;
        V t = op.b, f = op.c;
        if (auto kc = m.int_k(c)) return kc->second != 0 ? t : f;
        // Under the same condition an inner select is one of its operands (two guarded writes of
        // one register in a row).
        bool inner = false;
        while (t.kind == V::Inst) {
            const Op& o = m.insts.at(t.i).op;
            if (!(o.kind == Op::Select && o.a == c)) break;
            t = o.b;
            inner = true;
        }
        while (f.kind == V::Inst) {
            const Op& o = m.insts.at(f.i).op;
            if (!(o.kind == Op::Select && o.a == c)) break;
            f = o.c;
            inner = true;
        }
        if (t == f) return t;
        auto kt = m.int_k(t);
        auto kf = m.int_k(f);
        if (int_bits(m, ty) == std::optional<std::uint32_t>(1) && kt && kt->second == 1 && kf && kf->second == 0)
            return c;
        if (inner) {
            Op& o = m.insts[i].op;
            o.a = c;
            o.b = t;
            o.c = f;
            changed = true;
        }
        return std::nullopt;
    }
    case Op::Cast: {
        const Cast co = Cast(op.code);
        const V x = op.a;
        const Ty from = m.type_of(x);
        std::optional<std::pair<Cast, V>> through;
        if (x.kind == V::Inst) {
            const Op& o = m.insts.at(x.i).op;
            if (o.kind == Op::Cast) through = std::make_pair(Cast(o.code), o.a);
        }
        switch (co) {
        case Cast::Bitcast: {
            if (from == ty) return x;
            // There and back: the bits are the same.
            if (through && through->first == Cast::Bitcast && m.type_of(through->second) == ty) return through->second;
            if (x.kind != V::Const) return std::nullopt;
            const bool to_float = m.types.at(ty.id).kind == TypeDef::Float;
            const Const c = m.consts.at(x.i).second;
            if (c.kind == Const::Int && to_float && int_bits(m, from) == std::optional<std::uint32_t>(32))
                return m.f32(std::uint32_t(std::uint64_t(c.value)));
            if (c.kind == Const::Float && int_bits(m, ty) == std::optional<std::uint32_t>(32))
                return m.int_const(ty, std::int64_t(std::int32_t(c.bits)));
            return std::nullopt;
        }
        case Cast::ZExt:
        case Cast::SExt:
        case Cast::Trunc: {
            auto to = int_bits(m, ty);
            if (!to) return std::nullopt;
            if (auto kx = m.int_k(x)) {
                std::uint64_t k = co == Cast::SExt ? std::uint64_t(signed_(kx->first, kx->second)) : kx->second;
                return k_(m, ty, *to, k);
            }
            // Widened and narrowed again.
            if (co == Cast::Trunc && through && (through->first == Cast::ZExt || through->first == Cast::SExt) &&
                m.type_of(through->second) == ty)
                return through->second;
            return std::nullopt;
        }
        default:
            return std::nullopt;
        }
    }
    default:
        return std::nullopt;
    }
}

// What kind of instruction `i` is for merging.
Effect class_of(const Module& m, const std::vector<Effect>& effects, std::uint32_t i) {
    const Op& op = m.insts.at(i).op;
    switch (op.kind) {
    case Op::Bin:
    case Op::Cast:
    case Op::Cmp:
    case Op::Select:
    case Op::Extract:
    case Op::Gep:
        return Effect::Pure;
    case Op::Load: {
        // A constant table reads the same whatever is stored.
        V p = op.a;
        V base = p;
        if (p.kind == V::Inst) {
            const Op& g = m.insts.at(p.i).op;
            if (g.kind == Op::Gep) base = g.a;
        }
        if (base.kind == V::Global && m.globals.at(base.i).init.has_value()) return Effect::Pure;
        return Effect::Reads;
    }
    case Op::Store:
    case Op::AtomicRmw:
    case Op::CmpXchg:
        return Effect::Writes;
    case Op::Phi:
        return Effect::Lanes;
    case Op::Call:
        return effects.at(op.index);
    }
    bug("unreachable: class");
}

std::optional<Key> key_of(const Module& m, std::uint32_t i) {
    const Inst& inst = m.insts.at(i);
    const Op& op = inst.op;
    Key k;
    switch (op.kind) {
    case Op::Bin: {
        // Integer sums, products and bit operations whichever way round; float ones as written
        // (which NaN an operation of two NaNs gives is the hardware's choice by operand).
        const Bin bo = Bin(op.code);
        const bool commutes = (bo == Bin::Add || bo == Bin::Mul || bo == Bin::And || bo == Bin::Or || bo == Bin::Xor) &&
                              int_bits(m, inst.ty).has_value();
        k.kind = Key::Bin;
        k.code = op.code;
        if (commutes && op.b < op.a)
            k.vs = {op.b, op.a};
        else
            k.vs = {op.a, op.b};
        return k;
    }
    case Op::Cast:
        k.kind = Key::Cast;
        k.code = op.code;
        k.n = inst.ty.id;
        k.vs = {op.a};
        return k;
    case Op::Cmp:
        k.kind = Key::Cmp;
        k.code = op.code;
        k.vs = {op.a, op.b};
        return k;
    case Op::Select:
        k.kind = Key::Select;
        k.vs = {op.a, op.b, op.c};
        return k;
    case Op::Extract:
        k.kind = Key::Extract;
        k.n = op.index;
        k.vs = {op.a};
        return k;
    case Op::Call:
        k.kind = Key::Call;
        k.n = op.index;
        k.vs = op.list;
        return k;
    case Op::Gep:
        k.kind = Key::Gep;
        k.n = op.ty.id;
        k.vs.reserve(op.list.size() + 1);
        k.vs.push_back(op.a);
        k.vs.insert(k.vs.end(), op.list.begin(), op.list.end());
        return k;
    case Op::Load:
        k.kind = Key::Load;
        k.vs = {op.a};
        return k;
    default:
        return std::nullopt;
    }
}

// FOLD, REUSE, FACTS and LOADS: one walk over the dominator tree. Returns whether anything
// changed.
bool merge(Module& m, std::uint32_t passes) {
    if (m.blocks.empty()) return false;
    const std::vector<Effect> effects = m.effects();
    const Tree tree = dominator_tree(m);
    std::vector<std::optional<V>> repl(m.insts.size());
    bool changed = false;
    // Joins of one value: until none is left (one gone can leave another with one value).
    if ((passes & opt::FOLD) != 0) {
        bool again = true;
        while (again) {
            again = false;
            for (std::size_t b : tree.order) {
                for (std::uint32_t p : m.blocks[b].phis) {
                    if (repl.at(p)) continue;
                    const Op& op = m.insts.at(p).op;
                    if (op.kind != Op::Phi) continue;
                    std::optional<V> one;
                    bool same = true;
                    for (const auto& inc : op.incoming) {
                        V v = resolve(repl, inc.first);
                        if (v == V::inst(p)) continue;
                        if (!one)
                            one = v;
                        else if (*one != v)
                            same = false;
                    }
                    if (same && one) {
                        repl[p] = *one;
                        again = true;
                        changed = true;
                    }
                }
            }
        }
    }
    // The values every path to the block being walked has computed (those of the blocks above it
    // in the tree, and its own so far).
    std::unordered_map<Key, V, KeyHash> known;
    // The reads of the block being walked since its last store.
    std::unordered_map<Key, V, KeyHash> read;
    Facts facts;
    const V yes = m.bool_(true);
    const V no = m.bool_(false);
    struct Step {
        bool leave = false;
        std::size_t b = 0;
        std::vector<Key> keys;
        Learned learned;
    };
    std::vector<Step> steps;
    steps.push_back(Step{false, 0, {}, {}});
    auto known_bit = [&](V v) {
        auto it = facts.bits.find(v);
        if (it == facts.bits.end()) return v;
        return it->second ? yes : no;
    };
    while (!steps.empty()) {
        Step step = std::move(steps.back());
        steps.pop_back();
        if (step.leave) {
            for (const Key& k : step.keys) known.erase(k);
            for (V v : step.learned.bits) facts.bits.erase(v);
            for (const Cmp3& k : step.learned.compares) facts.compares.erase(k);
            continue;
        }
        const std::size_t b = step.b;
        std::vector<Key> added;
        // Reached from one block alone, by one side of its branch: the condition is that side's,
        // here and in every block below.
        Learned learned;
        if ((passes & opt::FACTS) != 0 && tree.preds.at(b).size() == 1) {
            std::size_t p = tree.preds[b][0];
            const auto& term = m.blocks.at(p).term;
            if (term && term->kind == Term::CondBr && term->t != term->f)
                learn(m, resolve(repl, term->cond), b == term->t, facts, learned, 8);
        }
        read.clear();
        // Two joins of the same values from the same blocks are one.
        if ((passes & opt::REUSE) != 0 && m.blocks[b].phis.size() > 1) {
            std::unordered_map<JoinKey, V, JoinKeyHash> joins;
            for (std::uint32_t p : m.blocks[b].phis) {
                if (repl.at(p)) continue;
                const Ty ty = m.insts.at(p).ty;
                const Op& op = m.insts[p].op;
                if (op.kind != Op::Phi) continue;
                JoinKey key;
                key.reserve(op.incoming.size() + 1);
                for (const auto& [v, from] : op.incoming) key.emplace_back(from, resolve(repl, v));
                std::sort(key.begin(), key.end());
                // The type too: an undefined value of one type is not that of another.
                key.emplace_back(std::size_t(ty.id), V::global(0xFFFFFFFFu));
                auto it = joins.find(key);
                if (it != joins.end()) {
                    repl[p] = it->second;
                    changed = true;
                } else {
                    joins.emplace(std::move(key), V::inst(p));
                }
            }
        }
        std::vector<std::uint32_t> body = std::move(m.blocks[b].body);
        m.blocks[b].body.clear();
        std::vector<std::uint32_t> kept;
        kept.reserve(body.size());
        for (std::uint32_t i : body) {
            map_operands(m.insts.at(i).op, [&](V v) {
                V r = resolve(repl, v);
                V bit = known_bit(r);
                changed |= bit != r;
                return bit;
            });
            {
                const Op& op = m.insts[i].op;
                if (op.kind == Op::Cmp) {
                    auto it = facts.compares.find(Cmp3{op.code, op.a, op.b});
                    if (it != facts.compares.end()) {
                        repl[i] = it->second ? yes : no;
                        changed = true;
                        continue;
                    }
                }
            }
            if ((passes & opt::FOLD) != 0) {
                if (auto v = simplify(m, i, changed)) {
                    repl[i] = *v;
                    changed = true;
                    continue;
                }
            }
            const Effect cls = class_of(m, effects, i);
            if ((passes & opt::REUSE) != 0 && cls == Effect::Pure) {
                if (auto key = key_of(m, i)) {
                    auto it = known.find(*key);
                    if (it != known.end()) {
                        repl[i] = it->second;
                        changed = true;
                        continue;
                    }
                    known.emplace(*key, V::inst(i));
                    added.push_back(std::move(*key));
                }
            }
            // Reads of memory, and wave operations (the same lanes run the whole block): the
            // block's own earlier one.
            if ((passes & opt::LOADS) != 0 && (cls == Effect::Reads || cls == Effect::Lanes)) {
                if (auto key = key_of(m, i)) {
                    auto it = read.find(*key);
                    if (it != read.end()) {
                        repl[i] = it->second;
                        changed = true;
                        continue;
                    }
                    read.emplace(std::move(*key), V::inst(i));
                }
            }
            if (cls == Effect::Writes) read.clear();
            kept.push_back(i);
        }
        m.blocks[b].body = std::move(kept);
        if (auto& term = m.blocks[b].term; term && term->kind == Term::CondBr) term->cond = known_bit(resolve(repl, term->cond));
        steps.push_back(Step{true, 0, std::move(added), std::move(learned)});
        const auto& ch = tree.children.at(b);
        for (auto it = ch.rbegin(); it != ch.rend(); ++it) steps.push_back(Step{false, *it, {}, {}});
    }
    if (!changed) return false;
    // Everything that read a replaced value reads its replacement (joins read values from blocks
    // walked after them).
    for (Block& blk : m.blocks) {
        std::erase_if(blk.phis, [&](std::uint32_t p) { return repl.at(p).has_value(); });
        if (blk.term && blk.term->kind == Term::CondBr) blk.term->cond = resolve(repl, blk.term->cond);
    }
    auto res = [&](V v) { return resolve(repl, v); };
    for (const Block& blk : m.blocks) {
        for (std::uint32_t i : blk.phis) map_operands(m.insts.at(i).op, res);
        for (std::uint32_t i : blk.body) map_operands(m.insts.at(i).op, res);
    }
    return true;
}

// FOLD: a branch on a constant goes one way. The side not taken loses the branching block's
// values at its joins, and so do the joins of blocks that nothing reaches any more. Returns
// whether a branch changed.
bool prune(Module& m) {
    bool changed = false;
    for (std::size_t b = 0; b < m.blocks.size(); ++b) {
        if (!m.blocks[b].term || m.blocks[b].term->kind != Term::CondBr) continue;
        const Term t = *m.blocks[b].term;
        // Both sides one block: two edges into it, left as they are.
        if (t.t == t.f) continue;
        auto k = m.int_k(t.cond);
        if (!k) continue;
        std::size_t taken = k->second != 0 ? t.t : t.f;
        std::size_t not_ = k->second != 0 ? t.f : t.t;
        m.blocks[b].term = Term::br(taken);
        changed = true;
        for (std::uint32_t p : m.blocks.at(not_).phis) {
            Op& op = m.insts.at(p).op;
            if (op.kind == Op::Phi) std::erase_if(op.incoming, [&](const auto& x) { return x.second == b; });
        }
    }
    if (!changed) return false;
    const std::vector<bool> reached = m.reached();
    for (std::size_t b = 0; b < m.blocks.size(); ++b) {
        if (!reached[b]) continue;
        for (std::uint32_t p : m.blocks[b].phis) {
            Op& op = m.insts.at(p).op;
            if (op.kind == Op::Phi)
                std::erase_if(op.incoming, [&](const auto& x) { return !reached.at(x.second); });
        }
    }
    return true;
}

// Whether instruction `i` must stay whatever reads it.
bool has_effect(const Module& m, const std::vector<Effect>& effects, std::uint32_t i) {
    const Op& op = m.insts.at(i).op;
    switch (op.kind) {
    case Op::Store:
    case Op::AtomicRmw:
    case Op::CmpXchg:
        return true;
    case Op::Call:
        return effects.at(op.index) == Effect::Writes;
    default:
        return false;
    }
}

// A constant table no remaining load reads leaves the module (the validator refuses an internal
// declaration nothing uses). Shared memory stays declared whether it is read or not.
void drop_unused_tables(Module& m) {
    std::vector<bool> used;
    used.reserve(m.globals.size());
    for (const Global& g : m.globals) used.push_back(!g.init.has_value());
    if (std::all_of(used.begin(), used.end(), [](bool u) { return u; })) return;
    auto note = [&](V v) {
        if (v.kind == V::Global) used.at(v.i) = true;
    };
    for (const Block& blk : m.blocks) {
        for (std::uint32_t i : blk.phis) for_each_operand(m.insts.at(i).op, note);
        for (std::uint32_t i : blk.body) for_each_operand(m.insts.at(i).op, note);
    }
    for (const MdDef& md : m.mds)
        if (md.kind == MdDef::Value && md.v.kind == V::Global) used.at(md.v.i) = true;
    if (std::all_of(used.begin(), used.end(), [](bool u) { return u; })) return;
    std::vector<std::uint32_t> nw(used.size(), 0xFFFFFFFFu);
    {
        std::uint32_t next = 0;
        for (std::size_t old = 0; old < used.size(); ++old)
            if (used[old]) nw[old] = next++;
    }
    {
        std::vector<Global> kept;
        for (std::size_t g = 0; g < m.globals.size(); ++g)
            if (used[g]) kept.push_back(std::move(m.globals[g]));
        m.globals = std::move(kept);
    }
    auto renumber = [&](V v) {
        if (v.kind == V::Global && used.at(v.i)) return V::global(nw[v.i]);
        return v;
    };
    for (Inst& inst : m.insts) map_operands(inst.op, renumber);
    for (MdDef& md : m.mds)
        if (md.kind == MdDef::Value) md.v = renumber(md.v);
}

// Declarations no remaining call names leave the module (the validator refuses an external
// function nothing uses); the others keep their order, so `main` stays function 0.
void drop_unused_functions(Module& m) {
    std::vector<bool> used;
    used.reserve(m.funcs.size());
    for (const Func& f : m.funcs) used.push_back(f.body);
    auto note = [&](std::uint32_t i) {
        const Op& op = m.insts.at(i).op;
        if (op.kind == Op::Call) used.at(op.index) = true;
    };
    for (const Block& blk : m.blocks) {
        for (std::uint32_t i : blk.phis) note(i);
        for (std::uint32_t i : blk.body) note(i);
    }
    for (const MdDef& md : m.mds)
        if (md.kind == MdDef::Value && md.v.kind == V::Func) used.at(md.v.i) = true;
    if (std::all_of(used.begin(), used.end(), [](bool u) { return u; })) return;
    std::vector<std::uint32_t> nw(used.size(), 0xFFFFFFFFu);
    {
        std::uint32_t next = 0;
        for (std::size_t old = 0; old < used.size(); ++old)
            if (used[old]) nw[old] = next++;
    }
    {
        std::vector<Func> kept;
        for (std::size_t f = 0; f < m.funcs.size(); ++f)
            if (used[f]) kept.push_back(std::move(m.funcs[f]));
        m.funcs = std::move(kept);
    }
    m.func_ids.clear();
    for (std::size_t i = 0; i < m.funcs.size(); ++i) m.func_ids[m.funcs[i].name] = std::uint32_t(i);
    // Instructions that left their block keep a stale number: nothing reads them again.
    for (Inst& inst : m.insts)
        if (inst.op.kind == Op::Call && used.at(inst.op.index)) inst.op.index = nw[inst.op.index];
    for (MdDef& md : m.mds)
        if (md.kind == MdDef::Value && md.v.kind == V::Func) md.v.i = nw.at(md.v.i);
}

// DEAD: keeps the instructions with an effect, the branch conditions, and what they read,
// transitively; the rest leave their blocks. Blocks no branch reaches are emptied (they are not
// written).
void drop_dead(Module& m) {
    const std::vector<Effect> effects = m.effects();
    const std::vector<bool> reached = m.reached();
    std::vector<bool> live(m.insts.size(), false);
    std::vector<std::uint32_t> work;
    auto mark = [&](V v) {
        if (v.kind == V::Inst && !live.at(v.i)) {
            live[v.i] = true;
            work.push_back(v.i);
        }
    };
    for (std::size_t b = 0; b < m.blocks.size(); ++b) {
        if (!reached[b]) continue;
        const Block& blk = m.blocks[b];
        for (std::uint32_t i : blk.body)
            if (has_effect(m, effects, i)) mark(V::inst(i));
        if (blk.term && blk.term->kind == Term::CondBr) mark(blk.term->cond);
    }
    while (!work.empty()) {
        std::uint32_t i = work.back();
        work.pop_back();
        for_each_operand(m.insts.at(i).op, mark);
    }
    for (std::size_t b = 0; b < m.blocks.size(); ++b) {
        Block& blk = m.blocks[b];
        if (reached[b]) {
            std::erase_if(blk.body, [&](std::uint32_t i) { return !live.at(i); });
            std::erase_if(blk.phis, [&](std::uint32_t i) { return !live.at(i); });
        } else {
            blk.body.clear();
            blk.phis.clear();
        }
    }
    drop_unused_functions(m);
    drop_unused_tables(m);
}

}  // namespace

namespace opt {

Effect effect(const Func& f) {
    const std::string_view name = strip_dx_op(f.name);
    if (f.attr == Attr::ReadNone) return Effect::Pure;
    if (f.attr == Attr::ReadOnly) {
        // A shader cannot write a constant buffer or resize a resource.
        if (starts_with(name, "cbufferLoadLegacy") || starts_with(name, "getDimensions")) return Effect::Pure;
        if (starts_with(name, "wave")) return Effect::Lanes;
        return Effect::Reads;
    }
    // Declared with the attribute of calls that have effects, and have none: they read the other
    // lanes' values.
    if (starts_with(name, "wave") || starts_with(name, "quad")) return Effect::Lanes;
    return Effect::Writes;
}

std::uint8_t opposite(std::uint8_t p) {
    if (p <= 15) return std::uint8_t(15 - p);
    switch (p) {
    case pred::EQ:
        return pred::NE;
    case pred::NE:
        return pred::EQ;
    case pred::IUGT:
        return pred::IULE;
    case pred::IULE:
        return pred::IUGT;
    case pred::IUGE:
        return pred::IULT;
    case pred::IULT:
        return pred::IUGE;
    case pred::SGT:
        return pred::SLE;
    case pred::SLE:
        return pred::SGT;
    case pred::SGE:
        return pred::SLT;
    default:
        return pred::SGE;
    }
}

}  // namespace opt

void Module::optimise(std::uint32_t passes) {
    if ((passes & (opt::FOLD | opt::REUSE | opt::FACTS | opt::LOADS)) != 0) {
        // Values merged in one round can make two joins the same in the next (a loop's registers
        // that only go round), and a branch that goes one way leaves blocks with one way in.
        for (int round = 0; round < 4; ++round) {
            bool changed = merge(*this, passes);
            if ((passes & opt::FOLD) != 0) changed |= prune(*this);
            if (!changed) break;
        }
    }
    if ((passes & opt::DEAD) != 0) drop_dead(*this);
}

std::vector<opt::Effect> Module::effects() const {
    std::vector<opt::Effect> r;
    r.reserve(funcs.size());
    for (const Func& f : funcs) r.push_back(opt::effect(f));
    return r;
}

std::vector<bool> Module::reached() const {
    std::vector<bool> seen(blocks.size(), false);
    if (blocks.empty()) return seen;
    std::vector<std::size_t> stack = {0};
    seen[0] = true;
    while (!stack.empty()) {
        std::size_t b = stack.back();
        stack.pop_back();
        for (std::size_t s : successors(blocks[b].term)) {
            if (!seen.at(s)) {
                seen[s] = true;
                stack.push_back(s);
            }
        }
    }
    return seen;
}

// An integer constant's width and value (the bits of that width).
std::optional<std::pair<std::uint32_t, std::uint64_t>> Module::int_k(V v) const {
    if (v.kind != V::Const) return std::nullopt;
    const auto& [ty, c] = consts.at(v.i);
    if (c.kind != Const::Int) return std::nullopt;
    auto bits = int_bits(*this, ty);
    if (!bits) return std::nullopt;
    return std::make_pair(*bits, std::uint64_t(c.value) & mask(*bits));
}

// The bits of a 32-bit float that is a constant, given as one or as a cast of an integer constant.
std::optional<float> Module::float_k(V v) const {
    bool cast = false;
    if (v.kind == V::Inst) {
        const Op& op = insts.at(v.i).op;
        if (!(op.kind == Op::Cast && op.code == std::uint8_t(Cast::Bitcast))) return std::nullopt;
        v = op.a;
        cast = true;
    }
    if (v.kind != V::Const) return std::nullopt;
    const Const& c = consts.at(v.i).second;
    if (c.kind == Const::Float && !cast) return std::bit_cast<float>(c.bits);
    if (c.kind == Const::Int && cast) {
        auto k = int_k(v);
        if (!k) return std::nullopt;
        if (k->first != 32) return std::nullopt;
        return std::bit_cast<float>(std::uint32_t(k->second));
    }
    return std::nullopt;
}

// For a float that is a number and not an infinity whatever the program reads: a power of two
// its magnitude does not pass.
std::optional<float> Module::bounded(V v, std::uint32_t depth) const {
    if (auto k = float_k(v)) {
        if (!std::isfinite(*k)) return std::nullopt;
        return std::ceil(std::log2(fmax_rs(std::fabs(*k), 1e-30f)));
    }
    if (v.kind != V::Inst) return std::nullopt;
    if (depth == 0) return std::nullopt;
    const std::uint32_t d = depth - 1;
    auto of = [&](V x) { return bounded(x, d); };
    // (A power that overflows, or might with rounding, is no bound.)
    auto held = [](float b) -> std::optional<float> {
        if (b < 126.0f) return b;
        return std::nullopt;
    };
    const Op& op = insts.at(v.i).op;
    switch (op.kind) {
    case Op::Cast: {
        const Cast co = Cast(op.code);
        if (co == Cast::UiToFp || co == Cast::SiToFp) return 32.0f;
        if (co == Cast::Bitcast) {
            V x = op.a;
            if (x.kind != V::Inst) return std::nullopt;
            const Op& o = insts.at(x.i).op;
            if (o.kind == Op::Cast && o.code == std::uint8_t(Cast::Bitcast) && type_of(o.a) == type_of(v)) return of(o.a);
            return std::nullopt;
        }
        return std::nullopt;
    }
    case Op::Bin: {
        const Bin bo = Bin(op.code);
        if (bo == Bin::Mul) {
            auto a = of(op.a);
            if (!a) return std::nullopt;
            auto b = of(op.b);
            if (!b) return std::nullopt;
            return held(*a + *b);
        }
        if (bo == Bin::Add || bo == Bin::Sub) {
            auto a = of(op.a);
            if (!a) return std::nullopt;
            auto b = of(op.b);
            if (!b) return std::nullopt;
            return held(fmax_rs(*a, *b) + 1.0f);
        }
        return std::nullopt;
    }
    case Op::Select: {
        auto a = of(op.b);
        if (!a) return std::nullopt;
        auto b = of(op.c);
        if (!b) return std::nullopt;
        return fmax_rs(*a, *b);
    }
    case Op::Call: {
        const auto& args = op.list;
        if (args.empty()) return std::nullopt;
        auto kc = int_k(args[0]);
        if (!kc) return std::nullopt;
        const std::uint64_t code = kc->second;
        const std::size_t n = args.size() - 1;
        // Saturated, a fraction, a cosine or sine (of a number).
        if (code == 7 && n == 1) return 0.0f;
        if (code == 6 && n == 1) return of(args[1]);
        if ((code == 22 || code == 12 || code == 13) && n == 1) {
            if (of(args[1])) return 0.0f;
            return std::nullopt;
        }
        if (code >= 26 && code <= 29 && n == 1) {
            auto x = of(args[1]);
            if (!x) return std::nullopt;
            return held(*x + 1.0f);
        }
        // Of two numbers; or one of them the other way round: the minimum (36) of a constant and
        // a maximum of another.
        if ((code == 35 || code == 36) && n == 2) {
            const V a = args[1], b = args[2];
            auto x = of(a);
            auto y = of(b);
            if (x && y) return fmax_rs(*x, *y);
            const std::uint64_t other = code == 35 ? 36 : 35;
            V inner, kv;
            if (float_k(b).has_value()) {
                inner = a;
                kv = b;
            } else {
                inner = b;
                kv = a;
            }
            if (!float_k(kv)) return std::nullopt;
            auto k = of(kv);
            if (!k) return std::nullopt;
            if (inner.kind != V::Inst) return std::nullopt;
            const Op& io = insts.at(inner.i).op;
            if (io.kind != Op::Call) return std::nullopt;
            if (io.list.empty()) return std::nullopt;
            auto ic = int_k(io.list[0]);
            if (!ic) return std::nullopt;
            std::optional<V> low;
            for (std::size_t j = 1; j < io.list.size(); ++j) {
                if (float_k(io.list[j]).has_value()) {
                    low = io.list[j];
                    break;
                }
            }
            if (!low) return std::nullopt;
            if (ic->second != other) return std::nullopt;
            auto l = of(*low);
            if (!l) return std::nullopt;
            return fmax_rs(*k, *l);
        }
        if (code == 46 && n == 3) {
            auto a = of(args[1]);
            if (!a) return std::nullopt;
            auto b = of(args[2]);
            if (!b) return std::nullopt;
            auto c = of(args[3]);
            if (!c) return std::nullopt;
            return held(fmax_rs(*a + *b, *c) + 1.0f);
        }
        return std::nullopt;
    }
    default:
        return std::nullopt;
    }
}

// See Usage: over the blocks reached from the entry.
opt::Usage Module::usage() const {
    opt::Usage u;
    const std::vector<bool> rch = reached();
    auto wide = [&](Ty ty) {
        const TypeDef& d = types.at(ty.id);
        return d.kind == TypeDef::Int && d.n == 64 && d.of.id == 0 && !d.name && d.list.empty();
    };
    // The first field of a handle's annotation: its kind, with the UAV bit; the second: the
    // component type and count.
    auto annotation = [&](V h) -> std::optional<std::pair<std::int64_t, std::int64_t>> {
        if (h.kind != V::Inst) return std::nullopt;
        const Op& op = insts.at(h.i).op;
        if (op.kind != Op::Call) return std::nullopt;
        if (funcs.at(op.index).name != "dx.op.annotateHandle") return std::nullopt;
        if (op.list.size() <= 2) return std::nullopt;
        V props = op.list[2];
        if (props.kind != V::Const) return std::nullopt;
        const Const& pc = consts.at(props.i).second;
        if (pc.kind != Const::Aggregate) return std::nullopt;
        const Const& f0 = consts.at(pc.ids.at(0)).second;
        const Const& f1 = consts.at(pc.ids.at(1)).second;
        if (f0.kind == Const::Int && f1.kind == Const::Int) return std::make_pair(f0.value, f1.value);
        return std::nullopt;
    };
    auto one = [&](std::uint32_t i) {
        const Inst& inst = insts.at(i);
        u.int64 |= wide(inst.ty);
        for_each_operand(inst.op, [&](V v) { u.int64 |= wide(type_of(v)); });
        if (inst.op.kind != Op::Call) return;
        const auto& args = inst.op.list;
        const std::string_view name = strip_dx_op(funcs.at(inst.op.index).name);
        if (starts_with(name, "wave") || starts_with(name, "quad")) {
            u.wave_ops = true;
        } else if (name == "createHandleFromHeap") {
            std::optional<std::pair<std::uint32_t, std::uint64_t>> k;
            if (args.size() > 2) k = int_k(args[2]);
            if (k && k->second == 0)
                u.heap_views = true;
            else
                u.heap_samplers = true;
        } else if (name == "annotateHandle") {
            auto a = annotation(V::inst(i));
            u.rovs |= a && (a->first & 0x2000) != 0;
        } else if (starts_with(name, "atomic") && ends_with(name, ".i64")) {
            u.atomic64 = true;
        } else if (starts_with(name, "textureLoad") || starts_with(name, "bufferLoad")) {
            std::optional<std::pair<std::int64_t, std::int64_t>> a;
            if (args.size() > 1) a = annotation(args[1]);
            u.typed_uav_loads |= !a || ((a->first & 0x1000) != 0 && ((a->second >> 8) & 0xff) > 1);
        }
    };
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!rch[b]) continue;
        for (std::uint32_t i : blocks[b].phis) one(i);
        for (std::uint32_t i : blocks[b].body) one(i);
    }
    return u;
}

}  // namespace x2w::vs::bitcode
