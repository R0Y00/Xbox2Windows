// Resource tracking implementation. See resource.h.
//
// R<T> holds a value or a static error message. TRY propagates a failure and reasons are compared by
// content.
#include "x2w_shader/resource.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <map>
#include <string_view>
#include <tuple>
#include <variant>

#include "x2w_shader/bvh.h"

namespace x2w::vs::resource {

namespace {

using detail::Term;
using gfx10::Operand;
using ir::Exit;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;
using native::HwStage;
using native::RootParam;
using native::RootType;
using structure::Bool;
using structure::LaneExit;
using structure::Meets;
using structure::Structured;

// The reasons an address traces to no root parameter for which the address is then taken as it
// stands (MEMORY).
constexpr const char* RAW[5] = {
    "an address not from a root parameter",
    "an address from a root parameter that is no address",
    "an address in an SGPR that is no root parameter",
    "an address loaded from memory",
    "an address from a phi",
};

constexpr std::size_t MAX_ROUNDS = 64;
constexpr std::size_t MAX_JOINS = 8;
// VGPR values followed for one dword of a descriptor kept per lane.
constexpr std::size_t MAX_COPIES = 64;

struct Err {
    const char* why;
};

template <class T>
struct R {
    T value{};
    const char* err = nullptr;
    R(T v) : value(std::move(v)) {}
    R(Err e) : err(e.why) {}
    bool ok() const { return err == nullptr; }
};

#define TRY(var, expr)                         \
    auto var##_r = (expr);                     \
    if (!var##_r.ok()) return Err{var##_r.err}; \
    auto var = std::move(var##_r.value)

bool same_text(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

// A 32-bit sum: optional root parameter (as its first dword), constant, optional computed term.
struct Sum {
    std::optional<Value> base;
    std::uint32_t offset = 0;
    std::optional<Value> dynamic;
};

// What lanes are known to satisfy: each Bool where its flag is set, its negation where not.
using Known = std::vector<std::pair<Bool, bool>>;

struct Moved {
    bool is_const = false;
    std::uint32_t n = 0;
    static Moved Const(std::uint32_t c) { return {true, c}; }
    static Moved Of(Value v) { return {false, v.n}; }
    bool operator==(const Moved&) const = default;
};

bool starts(std::string_view s, std::string_view p) { return s.starts_with(p); }

bool plain_valu(const gfx10::Valu& f) { return f.abs == 0 && f.neg == 0 && !f.sdwa && !f.dpp; }

// A lane select given as a constant.
std::optional<std::uint32_t> lane_of(const Operand& op)
{
    if (op.is_const()) return inline_constant(op.code);
    if (op.is_literal()) return op.literal;
    return std::nullopt;
}

// What dword `k` of a move's result copies.
std::optional<Moved> copy_source(const Inst& inst, std::size_t k)
{
    const auto* f = std::get_if<gfx10::Sop>(&inst.fields);
    if (!f) return std::nullopt;
    const std::string_view name = inst.name();
    if (name == "s_movk_i32") {
        if (!f->imm) return std::nullopt;
        return Moved::Const(static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int16_t>(*f->imm))));
    }
    // A mask of S0 bits from bit S1, as the upscaler's compiler makes an immediate sampler's word.
    if (name == "s_bfm_b32" && k == 0 && f->src.size() == 2 && f->src[0].is_const() && f->src[1].is_const()) {
        const auto w0 = inline_constant(f->src[0].code);
        if (!w0) return std::nullopt;
        const auto at0 = inline_constant(f->src[1].code);
        if (!at0) return std::nullopt;
        const std::uint32_t w = *w0 & 31, at = *at0 & 31;
        return Moved::Const(((1u << w) - 1) << at);
    }
    if (f->src.empty()) return std::nullopt;
    const Operand& src = f->src[0];
    const bool mov = name == "s_mov_b32" || name == "s_mov_b64";
    if (mov && src.is_scalar() && src.code != gfx10::NULL_CODE) {
        const auto v = inst.read(Reg::S(static_cast<std::uint8_t>(src.code + k)));
        if (!v) return std::nullopt;
        return Moved::Of(*v);
    }
    if (mov && src.is_scalar()) return Moved::Const(0);
    if (name == "s_mov_b32" && src.is_literal()) return Moved::Const(src.literal);
    if (name == "s_mov_b32" && src.is_const()) {
        const auto c = inline_constant(src.code);
        if (!c) return std::nullopt;
        return Moved::Const(*c);
    }
    // shortcut: only the one-operand ways the compiler makes a constant that has no inline form;
    // fold more if descriptors need it.
    if (name == "s_brev_b32" && src.is_const()) {
        const auto c = inline_constant(src.code);
        if (!c) return std::nullopt;
        std::uint32_t x = *c, r = 0;
        for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i);
        return Moved::Const(r);
    }
    if (name == "s_not_b32" && src.is_const()) {
        const auto c = inline_constant(src.code);
        if (!c) return std::nullopt;
        return Moved::Const(~*c);
    }
    // shortcut: a literal below 2^31, whose high dword is 0 whether the hardware extends it with
    // zeros or with its sign.
    if (name == "s_mov_b64" && src.is_literal() && (src.literal >> 31) == 0)
        return Moved::Const(k == 0 ? src.literal : 0);
    // 64-bit inline integers are sign-extended.
    if (name == "s_mov_b64" && src.is_const() && src.code >= 128 && src.code <= 208) {
        const auto low = inline_constant(src.code);
        if (!low) return std::nullopt;
        return Moved::Const(k == 0 ? *low : static_cast<std::uint32_t>(static_cast<std::int32_t>(*low) >> 31));
    }
    return std::nullopt;
}

template <class T>
bool contains(const std::vector<T>& v, const T& x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

struct Tracer {
    const Program* p;
    std::span<const RootParam> layout;
    HwStage stage;
    // Defining instruction per value (phis and inputs have none).
    std::vector<std::optional<Site>> defs;
    // Block and index of each phi value.
    std::vector<std::optional<std::pair<std::size_t, std::size_t>>> phis;
    // Register per input value.
    std::vector<std::optional<Reg>> inputs;
    std::vector<Term> origin_;
    // See Resources::lane_copied.
    mutable std::vector<Site> lane_copied;
    const Structured* lanes;
    // While set, sum() keeps the first-lane read of an index as the computed term instead of what
    // the lanes hold.
    mutable bool read_first_lane = false;
    // What the conditions asked of `lanes` so far came down to: the proofs of one program ask about
    // the same blocks and masks again and again.
    mutable Meets meets;
    // Reverse-postorder index per block.
    std::vector<std::size_t> rpo;
    // The block of the access being traced (live()).
    mutable std::optional<std::size_t> at;

    Tracer(const Program& prog, std::span<const RootParam> lay, HwStage st, const Structured* l)
        : p(&prog), layout(lay), stage(st), lanes(l)
    {
        std::optional<std::size_t> max;
        auto see = [&](Value v) {
            const std::size_t n = std::size_t{v.n} + 1;
            if (!max || n > *max) max = n;
        };
        for (const auto& i : prog.inputs) see(i.second);
        for (const auto& b : prog.blocks) {
            for (const auto& phi : b.phis) see(phi.value);
            for (const auto& i : b.insts)
                for (const auto& w : i.writes) see(w.second);
        }
        const std::size_t n = max.value_or(0);
        defs.assign(n, std::nullopt);
        phis.assign(n, std::nullopt);
        inputs.assign(n, std::nullopt);
        for (const auto& [reg, v] : prog.inputs) inputs[v.n] = reg;
        for (std::size_t b = 0; b < prog.blocks.size(); ++b) {
            const auto& block = prog.blocks[b];
            for (std::size_t i = 0; i < block.phis.size(); ++i) phis[block.phis[i].value.n] = std::pair{b, i};
            for (std::size_t i = 0; i < block.insts.size(); ++i)
                for (const auto& w : block.insts[i].writes) defs[w.second.n] = Site{b, i};
        }
        rpo.assign(prog.blocks.size(), SIZE_MAX);
        const auto order = ir::reverse_postorder(prog.blocks);
        for (std::size_t k = 0; k < order.size(); ++k) rpo[order[k]] = k;
        origin_ = propagate_copies(n);
    }

    // Optimistic copy propagation: a move's result is its source's origin, a phi's the one origin
    // its arguments agree on. Throws ShaderError.
    std::vector<Term> propagate_copies(std::size_t n) const
    {
        std::vector<std::optional<Term>> o(n);
        for (const auto& [reg, v] : p->inputs) o[v.n] = Term::Of(v);
        const auto order = ir::reverse_postorder(p->blocks);
        for (std::size_t round = 0; round < MAX_ROUNDS; ++round) {
            bool changed = false;
            for (const std::size_t b : order) {
                const auto& block = p->blocks[b];
                for (const auto& phi : block.phis) {
                    const Term own = Term::Of(phi.value);
                    std::optional<Term> first;
                    bool all = true;
                    for (const Value a : phi.args) {
                        const auto t = o[a.n];
                        if (!t || *t == own) continue;
                        if (!first)
                            first = *t;
                        else if (*t != *first) {
                            all = false;
                            break;
                        }
                    }
                    std::optional<Term> nw;
                    if (first) nw = all ? *first : own;
                    if (nw && nw != o[phi.value.n]) {
                        o[phi.value.n] = nw;
                        changed = true;
                    }
                }
                for (const auto& inst : block.insts) {
                    for (std::size_t k = 0; k < inst.writes.size(); ++k) {
                        const Value v = inst.writes[k].second;
                        auto moved = copy_source(inst, k);
                        if (!moved) moved = restored(inst);
                        std::optional<Term> nw;
                        if (!moved)
                            nw = Term::Of(v);
                        else if (moved->is_const)
                            nw = Term::Const(moved->n);
                        else
                            nw = o[moved->n];
                        if (nw && nw != o[v.n]) {
                            o[v.n] = nw;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) {
                std::vector<Term> out;
                out.reserve(n);
                for (std::size_t v = 0; v < n; ++v) out.push_back(o[v].value_or(Term::Of(Value{static_cast<std::uint32_t>(v)})));
                return out;
            }
        }
        throw ShaderError::unsupported(0, "copies that do not settle");
    }

    // Whether some lane can satisfy all of `conditions` (Bools::meet).
    bool meet(const Structured& l, const Known& conditions) const { return l.bools.meet_seen(conditions, meets); }

    const Inst& inst(Site s) const { return p->blocks[s.first].insts[s.second]; }
    const Inst* def(Value v) const
    {
        const auto& d = defs[v.n];
        return d ? &inst(*d) : nullptr;
    }
    Term origin(Value v) const { return origin_[v.n]; }

    // The origin of a one-dword operand as `inst` reads it.
    std::optional<Term> term(const Inst& i, const Operand& op) const
    {
        if (op.is_scalar() && op.code == gfx10::NULL_CODE) return Term::Const(0);
        if (op.is_scalar()) {
            const auto v = i.read(Reg::S(op.code));
            if (!v) return std::nullopt;
            return origin(*v);
        }
        if (op.is_const()) {
            const auto c = inline_constant(op.code);
            if (!c) return std::nullopt;
            return Term::Const(*c);
        }
        if (op.is_literal()) return Term::Const(op.literal);
        return std::nullopt;
    }

    // Layout index and dword of the root parameter `reg` holds on entry.
    std::optional<std::pair<std::size_t, std::uint32_t>> param_of(Reg reg) const
    {
        if (!reg.is_s()) return std::nullopt;
        const std::uint32_t code = reg.n;
        for (std::size_t i = 0; i < layout.size(); ++i) {
            const std::uint32_t first = native::user_sgpr(stage, layout[i].user_data);
            if (code < first) continue;
            const std::uint32_t dword = code - first;
            if (dword < layout[i].sgprs) return std::pair{i, dword};
        }
        return std::nullopt;
    }

    // The SOP2 operands of `v` if an instruction with one of `names` wrote it.
    std::optional<std::pair<const Inst*, std::array<Term, 2>>> sop2(Value v,
                                                                     std::initializer_list<std::string_view> names) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        if (std::find(names.begin(), names.end(), i->name()) == names.end()) return std::nullopt;
        if (i->writes.empty() || i->writes[0].second != v) return std::nullopt;
        const auto* f = std::get_if<gfx10::Sop>(&i->fields);
        if (!f || f->src.empty()) return std::nullopt;
        const auto a = term(*i, f->src[0]);
        if (!a) return std::nullopt;
        if (f->src.size() < 2) return std::nullopt;
        const auto b = term(*i, f->src[1]);
        if (!b) return std::nullopt;
        return std::pair{i, std::array<Term, 2>{*a, *b}};
    }
    std::optional<std::array<Term, 2>> sop2_terms(Value v, std::initializer_list<std::string_view> names) const
    {
        const auto s = sop2(v, names);
        if (!s) return std::nullopt;
        return s->second;
    }

    std::optional<RootIndexedView> root_indexed_view(const Descriptor& d,
                                                     const std::vector<std::pair<Site, Address>>& scalar) const
    {
        if (d.role != Role::Resource) return std::nullopt;
        if (d.source.kind != Source::Kind::Loaded) return std::nullopt;
        const Site load = d.source.load;
        const auto it = std::lower_bound(scalar.begin(), scalar.end(), load,
                                         [](const auto& e, const Site& s) { return e.first < s; });
        if (it == scalar.end() || it->first != load) return std::nullopt;
        const Address address = it->second;
        if (address.param >= layout.size()) return std::nullopt;
        const RootParam& base = layout[address.param];
        if (!(base.root == RootType::Table())) return std::nullopt;
        if (!address.dynamic) return std::nullopt;
        const Value dynamic = *address.dynamic;
        Value input = dynamic;
        std::uint8_t shift = 0;
        if (const auto t = sop2_terms(dynamic, {"s_lshl_b32"}); t && (*t)[0].is_value() && (*t)[1].is_const()) {
            input = (*t)[0].value();
            shift = static_cast<std::uint8_t>((*t)[1].n & 31);
        }
        const auto reg = inputs[input.n];
        if (!reg) return std::nullopt;
        const auto pc = param_of(*reg);
        if (!pc) return std::nullopt;
        const auto [param, constant_dword] = *pc;
        if (param >= layout.size()) return std::nullopt;
        const RootParam& constant = layout[param];
        if (!(constant.root == RootType::Constants())) return std::nullopt;
        return RootIndexedView{base.slot, constant.slot, constant_dword,
                               address.offset + std::uint32_t{d.source.first} * 4, shift};
    }

    Sum sum(Term t) const
    {
        if (t.is_const()) return Sum{std::nullopt, t.n, std::nullopt};
        const Value v = t.value();
        const Sum dynamic{std::nullopt, 0, v};
        if (const auto reg = inputs[v.n]) {
            const auto pr = param_of(*reg);
            if (pr && pr->second == 0) return Sum{v, 0, std::nullopt};
            return dynamic;
        }
        if (getpc(v)) return Sum{v, 0, std::nullopt};
        // Paths that each add an index to a table in the code and meet before one load: the join
        // itself is the computed term. Lowering gives every `s_getpc_b64` its own offset in the code
        // as its value, so the join is the byte in the code whichever table a path took (address()
        // adds no base to it).
        if (const auto pc = code_join(v)) return Sum{*pc, 0, v};
        // An address computed in VGPRs and taken from the first lane (a table in the code indexed by
        // a value held per lane, or a table offset held per lane, the console's waterfall): the sum
        // of what the lanes hold. Loads at one offset in different loops (each with its own
        // `v_readfirstlane_b32`) then have one address, which is what tells lowering that they read
        // one descriptor. With `read_first_lane` a first-lane read of an index alone stays the
        // computed term (first_lane_offset).
        if (const auto lane = first_lane(v)) {
            const Sum per_lane = sum(vorigin(*lane));
            return per_lane.base || !read_first_lane ? per_lane : dynamic;
        }
        auto terms = sop2_terms(v, {"s_add_u32", "s_add_i32"});
        if (!terms) terms = vadd(v);
        if (!terms) return dynamic;
        const Sum x = sum((*terms)[0]);
        const Sum y = sum((*terms)[1]);
        if ((x.base && y.base) || (x.dynamic && y.dynamic)) return dynamic;
        return Sum{x.base ? x.base : y.base, x.offset + y.offset, x.dynamic ? x.dynamic : y.dynamic};
    }

    // For a join `v` whose every incoming value adds to an `s_getpc_b64` low dword: the first of
    // those (one level of joins only). The paths may add to different ones: two tables in the code,
    // one picked by control flow.
    std::optional<Value> code_join(Value v) const
    {
        const auto ph = phis[v.n];
        if (!ph) return std::nullopt;
        std::optional<Value> base;
        for (const Value a : p->blocks[ph->first].phis[ph->second].args) {
            const Term t = vorigin(a);
            if (!t.is_value()) return std::nullopt;
            const Value x = t.value();
            if (x == v) continue;
            const auto pc = code_base(x, 4);
            if (!pc) return std::nullopt;
            if (!base) base = pc;
        }
        return base;
    }

    // `t` is the high dword of an `s_getpc_b64`, or `s_addc_u32`s of one.
    bool code_high(Term t) const { return code_high_in(t, 4); }

    bool code_high_in(Term t, std::uint32_t depth) const
    {
        if (!t.is_value()) return false;
        const Value x = t.value();
        if (const Inst* i = def(x); i && i->name() == "s_getpc_b64" && i->writes.size() > 1 && i->writes[1].second == x)
            return true;
        if (depth == 0) return false;
        const auto s = sop2_terms(x, {"s_addc_u32"});
        return s && (code_high_in((*s)[0], depth - 1) || code_high_in((*s)[1], depth - 1));
    }

    // The `s_getpc_b64` low dword among the terms `v` adds up (`depth` additions deep; joins are
    // not followed).
    std::optional<Value> code_base(Value v, std::uint32_t depth) const
    {
        if (getpc(v)) return v;
        // Computed in VGPRs and taken from the first lane.
        if (const auto lane = first_lane(v)) {
            const Term t = vorigin(*lane);
            if (t.is_value()) return code_base(t.value(), depth);
            return std::nullopt;
        }
        if (depth == 0 || phis[v.n]) return std::nullopt;
        auto terms = sop2_terms(v, {"s_add_u32", "s_add_i32"});
        if (!terms) terms = vadd(v);
        if (!terms) return std::nullopt;
        for (const Term t : *terms) {
            if (!t.is_value()) continue;
            if (const auto r = code_base(t.value(), depth - 1)) return r;
        }
        return std::nullopt;
    }

    // `t` is `high` or an `s_addc_u32` of it.
    bool is_high(Term t, Value high) const
    {
        if (!t.is_value()) return false;
        const Value x = t.value();
        if (x == high) return true;
        const auto s = sop2_terms(x, {"s_addc_u32"});
        return s && ((*s)[0] == Term::Of(high) || (*s)[1] == Term::Of(high));
    }

    // The VGPR value `v_readfirstlane_b32` reads to write the scalar `v`.
    std::optional<Value> first_lane(Value v) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        if (i->name() != "v_readfirstlane_b32" || i->writes.empty() || i->writes[0].second != v) return std::nullopt;
        const auto* f = std::get_if<gfx10::Valu>(&i->fields);
        if (!f) return std::nullopt;
        if (!plain_valu(*f)) return std::nullopt;
        if (f->src.empty() || !f->src[0].is_vector()) return std::nullopt;
        return i->read(Reg::V(f->src[0].reg));
    }

    // The `s_getpc_b64` writing `lo` as its low dword.
    const Inst* getpc(Value lo) const
    {
        const Inst* i = def(lo);
        if (!i) return nullptr;
        if (i->name() == "s_getpc_b64" && !i->writes.empty() && i->writes[0].second == lo) return i;
        return nullptr;
    }

    // The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`) writing `v`, through VGPR
    // copies of scalars.
    std::optional<std::array<Term, 2>> vadd(Value v) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* f = std::get_if<gfx10::Valu>(&i->fields);
        if (!f) return std::nullopt;
        const std::string_view name = i->name();
        if (!(name == "v_add_co_u32" || name == "v_add_nc_u32")) return std::nullopt;
        if (!f->dst || !f->dst->is_vector()) return std::nullopt;
        if (i->write(Reg::V(f->dst->reg)) != v) return std::nullopt;
        if (!plain_valu(*f)) return std::nullopt;
        if (f->src.empty()) return std::nullopt;
        const auto a = vterm(*i, f->src[0]);
        if (!a) return std::nullopt;
        if (f->src.size() < 2) return std::nullopt;
        const auto b = vterm(*i, f->src[1]);
        if (!b) return std::nullopt;
        return std::array<Term, 2>{*a, *b};
    }

    // The origin of a VALU operand.
    std::optional<Term> vterm(const Inst& i, const Operand& op) const
    {
        if (op.is_vector()) {
            const auto v = i.read(Reg::V(op.reg));
            if (!v) return std::nullopt;
            return vorigin(*v);
        }
        return term(i, op);
    }

    // A VGPR value's origin through `v_mov_b32` copies.
    Term vorigin(Value v) const
    {
        if (const Inst* i = def(v); i && i->name() == "v_mov_b32") {
            if (const auto* f = std::get_if<gfx10::Valu>(&i->fields); f && plain_valu(*f) && !f->src.empty()) {
                if (const auto t = vterm(*i, f->src[0])) return *t;
            }
        }
        return origin(v);
    }

    // The address of an access that traces to no root parameter (`why` says how it failed): title
    // memory at the address as it stands, in a compute program.
    R<Address> by_address(const char* why) const
    {
        const bool raw = std::any_of(std::begin(RAW), std::end(RAW), [&](const char* r) { return same_text(r, why); });
        if (stage == HwStage::Cs && raw) return Address{MEMORY, 0, std::nullopt};
        return Err{why};
    }

    // A global memory address: an SGPR pair plus a VGPR offset, or a VGPR pair.
    R<Address> flat_address(const Inst& i, const std::optional<Operand>& addr, const std::optional<Operand>& saddr,
                            std::int16_t offset) const
    {
        if (!addr || !addr->is_vector()) return Err{"a global address not in VGPRs"};
        const std::uint8_t reg = addr->reg;
        const auto off = static_cast<std::uint32_t>(static_cast<std::int32_t>(offset));
        if (saddr && saddr->is_scalar() && saddr->code != gfx10::NULL_CODE) {
            const std::uint8_t code = saddr->code;
            auto word = [&](std::uint8_t k) -> std::optional<Term> {
                const auto v = i.read(Reg::S(static_cast<std::uint8_t>(code + k)));
                if (!v) return std::nullopt;
                return origin(*v);
            };
            const auto lo = word(0);
            const auto hi = word(1);
            if (!lo || !hi) return Err{"an address not in SGPRs"};
            const auto v = i.read(Reg::V(reg));
            if (!v) return Err{"a global address"};
            Sum extra = sum(vorigin(*v));
            if (extra.base) return Err{"an offset from a root parameter"};
            extra.offset += off;
            return address(*lo, *hi, extra);
        }
        auto word = [&](std::uint8_t k) -> std::optional<Term> {
            const auto v = i.read(Reg::V(static_cast<std::uint8_t>(reg + k)));
            if (!v) return std::nullopt;
            return vorigin(*v);
        };
        const auto lo = word(0);
        const auto hi = word(1);
        if (!lo || !hi) return Err{"a global address"};
        return address(*lo, *hi, Sum{std::nullopt, off, std::nullopt});
    }

    // The address `lo` and `hi` hold, plus `extra` (an instruction's immediate and offset register).
    R<Address> address(Term lo, Term hi0, Sum extra) const
    {
        // The high half as the lanes that read through the address hold it, where that is a constant
        // or what a register held on entry (the code clears a table address's high half inside a
        // branch and uses the address after the join).
        Term hi = hi0;
        {
            const Term l = live(hi0);
            if (l.is_const())
                hi = l;
            else if (inputs[l.n])
                hi = l;
        }
        const Sum s = sum(lo);
        if (s.dynamic && extra.dynamic) return Err{"an address with two computed terms"};
        if (!s.base) return Err{why_no_base(s.dynamic)};
        const Value base = *s.base;
        if (const Inst* pc = getpc(base)) {
            // A table in the code: the base is the next instruction's offset.
            std::optional<Value> high;
            if (pc->writes.size() > 1) high = pc->writes[1].second;
            if (!high_half(hi, high) && !(high && carried(hi, *high))) return Err{"an address with an untracked high half"};
            // A join of paths into tables in the code is the byte in the code already: each path's
            // `s_getpc_b64` counts as its own offset.
            const bool joined = s.dynamic && code_join(*s.dynamic).has_value();
            const std::uint32_t first = joined ? 0 : static_cast<std::uint32_t>(pc->machine.offset) + 4;
            return Address{CODE, first + s.offset + extra.offset, s.dynamic ? s.dynamic : extra.dynamic};
        }
        const auto reg = inputs[base.n];
        if (!reg) return Err{"an address not from a root parameter"};
        const auto pr = param_of(*reg);
        if (!pr) return Err{"an address not from a root parameter"};
        const std::size_t param = pr->first;
        std::optional<Value> high;
        const RootType root = layout[param].root;
        if (root == RootType::Table() || root == RootType::ResourceHeap() || root == RootType::SamplerHeap()) {
            high = std::nullopt;
        } else if (root == RootType::Descriptor()) {
            const Reg h = Reg::S(static_cast<std::uint8_t>(native::user_sgpr(stage, layout[param].user_data) + 1));
            const auto it = std::find_if(p->inputs.begin(), p->inputs.end(), [&](const auto& e) { return e.first == h; });
            if (it == p->inputs.end()) return Err{"a root descriptor without its high half"};
            high = it->second;
        } else {
            return Err{"an address from a root parameter that is no address"};
        }
        // shortcut: with a computed term, a high half that adds anything to the base's is taken as the
        // carry of that term (offsets below 4 GB).
        const std::optional<Value> dynamic = s.dynamic ? s.dynamic : extra.dynamic;
        if (!high_half(hi, high) && !(dynamic && high && carried(hi, *high)))
            return Err{"an address with an untracked high half"};
        return Address{param, s.offset + extra.offset, dynamic};
    }

    // `hi` is `high` (0 for a table), possibly plus a carry.
    bool high_half(Term hi, std::optional<Value> high) const
    {
        const Term want = high ? Term::Of(*high) : Term::Const(0);
        if (hi == want) return true;
        if (!hi.is_value()) return false;
        const auto s = sop2_terms(hi.value(), {"s_addc_u32"});
        if (!s) return false;
        const Term a = (*s)[0], b = (*s)[1];
        return (a == want && b == Term::Const(0)) || (b == want && a == Term::Const(0));
    }

    // Whether one of the first two VALU operands of `i` satisfies `pred` (through vterm).
    template <class F>
    bool carry_operand(const Inst& i, const gfx10::Valu& f, F pred) const
    {
        const std::size_t n = std::min<std::size_t>(2, f.src.size());
        for (std::size_t k = 0; k < n; ++k) {
            const auto t = vterm(i, f.src[k]);
            if (t && pred(*t)) return true;
        }
        return false;
    }

    // `hi` adds something (with carry) to `high`: `s_addc_u32` or `v_add_co_ci_u32` with `high` as an
    // operand.
    bool carried(Term hi, Value high) const
    {
        if (!hi.is_value()) return false;
        const Value v = hi.value();
        if (v == high) return true;
        // A join (code_join): every path carries into the table's high half.
        if (const auto ph = phis[v.n]) {
            // A path may carry into another table's high half (code_join).
            auto high_of = [&](Term t) { return is_high(t, high) || code_high(t); };
            for (const Value a : p->blocks[ph->first].phis[ph->second].args) {
                const Term t = vorigin(a);
                if (t == Term::Of(v) || high_of(t)) continue;
                if (!t.is_value()) return false;
                Value x = t.value();
                // Computed in VGPRs and taken from the first lane.
                if (const auto lane = first_lane(x)) {
                    const Term y = vorigin(*lane);
                    if (y.is_const()) return false;
                    x = y.value();
                }
                const Inst* i = def(x);
                if (!i) return false;
                const auto* f = std::get_if<gfx10::Valu>(&i->fields);
                if (!f) return false;
                if (!(i->name() == "v_add_co_ci_u32" && carry_operand(*i, *f, high_of))) return false;
            }
            return true;
        }
        if (const auto lane = first_lane(v)) return carried(vorigin(*lane), high);
        if (const auto s = sop2_terms(v, {"s_addc_u32"})) return carried((*s)[0], high) || carried((*s)[1], high);
        const Inst* i = def(v);
        if (!i) return false;
        const auto* f = std::get_if<gfx10::Valu>(&i->fields);
        if (!f) return false;
        return i->name() == "v_add_co_ci_u32" && carry_operand(*i, *f, [&](Term t) { return carried(t, high); });
    }

    const char* why_no_base(std::optional<Value> dynamic) const
    {
        if (!dynamic) return "a constant address";
        const Value v = *dynamic;
        if (inputs[v.n]) return "an address in an SGPR that is no root parameter";
        std::vector<Value> ls;
        leaves(v, ls);
        if (std::any_of(ls.begin(), ls.end(), [&](Value l) {
                const Inst* i = def(l);
                return i && i->name() == "s_getpc_b64";
            }))
            return "an address relative to the program counter";
        const Inst* i = def(v);
        if (!i) return "an address from a phi";
        if (i->name().starts_with("s_load_")) return "an address loaded from memory";
        return "an address not from a root parameter";
    }

    // The terms of the additions that compute `v`.
    void leaves(Value v, std::vector<Value>& out) const
    {
        const auto s = sop2_terms(v, {"s_add_u32", "s_add_i32"});
        if (!s) {
            out.push_back(v);
            return;
        }
        for (const Term t : *s)
            if (t.is_value()) leaves(t.value(), out);
    }

    R<Address> smem_address(const Inst& i, const Operand& base, std::int32_t offset, const Operand& soffset) const
    {
        if (!base.is_scalar()) return Err{"an address not in SGPRs"};
        const std::uint8_t code = base.code;
        auto word = [&](std::uint8_t k) -> std::optional<Term> {
            const auto v = i.read(Reg::S(static_cast<std::uint8_t>(code + k)));
            if (!v) return std::nullopt;
            return origin(*v);
        };
        const auto lo = word(0);
        const auto hi = word(1);
        if (!lo || !hi) return Err{"an address not in SGPRs"};
        const auto so = term(i, soffset);
        if (!so) return Err{"an unknown offset operand"};
        Sum extra = sum(*so);
        if (extra.base) return Err{"an offset from a root parameter"};
        extra.offset += static_cast<std::uint32_t>(offset);
        return address(*lo, *hi, extra);
    }

    // The dwords of descriptor operand `op` as `i` reads them, as origins.
    std::optional<std::vector<Term>> operand_words(const Inst& i, const Operand& op) const
    {
        std::vector<Term> words;
        for (std::uint8_t k = 0; k < op.dwords; ++k) {
            const auto v = i.read(Reg::S(static_cast<std::uint8_t>(op.code + k)));
            if (!v) return std::nullopt;
            words.push_back(origin(*v));
        }
        return words;
    }

    // The source of the descriptor operand `op` of `i`, in block `at`.
    R<Source> descriptor(const Inst& i, const Operand& op, std::size_t blk) const
    {
        if (!op.is_scalar()) return Err{"a descriptor not in SGPRs"};
        auto w = operand_words(i, op);
        if (!w) return Err{"a descriptor not in SGPRs"};
        std::vector<Term> words = std::move(*w);
        // A byte access through a buffer descriptor made raw reads what the descriptor's view does with a raw host
        // view.
        if (i.name().find("format") == std::string_view::npos) {
            if (const auto raw = made_raw(words)) words.assign(raw->begin(), raw->end());
        }
        // Only the lanes that reach the access read the descriptor.
        Known known;
        if (lanes) known.push_back({lanes->reach[blk], true});
        std::vector<std::size_t> joins;
        return classify(words, joins, known);
    }

    // The source of a descriptor's words; `joins` are the phi blocks being resolved, `known` holds in
    // every lane that reads them.
    R<Source> classify(const std::vector<Term>& words, std::vector<std::size_t>& joins, Known& known) const
    {
        {
            TRY(s, phi(words, joins, known));
            if (s) return std::move(*s);
        }
        {
            TRY(s, select(words, joins, known));
            if (s) return std::move(*s);
        }
        auto read = [&](const Term& w) -> std::optional<Value> {
            if (w.is_value()) return first_lane_source(w.value());
            return std::nullopt;
        };
        std::vector<Value> sources;
        bool all = true;
        for (const Term& w : words) {
            const auto r = read(w);
            if (!r) {
                all = false;
                break;
            }
            sources.push_back(*r);
        }
        if (all) {
            // One descriptor the code keeps per lane: the lanes agree.
            if (auto s = lane_copies(words)) return std::move(*s);
            // Read from the first active lane, as a waterfall loop does: the dwords each lane holds.
            std::vector<Term> per;
            for (const Value v : sources) per.push_back(Term::Of(v));
            return per_lane(per, joins, known);
        }
        if (std::any_of(words.begin(), words.end(), [&](const Term& w) { return read(w).has_value(); }))
            return Err{"a descriptor partly read from the first lane"};
        if (std::all_of(words.begin(), words.end(), [](const Term& w) { return w.is_const(); })) {
            std::vector<std::uint32_t> consts;
            for (const Term& w : words) consts.push_back(w.n);
            return Source::Constant(std::move(consts));
        }
        if (auto s = loaded(words)) return std::move(*s);
        if (words.size() == 4 && words[2].is_const() && words[3].is_const()) {
            Term hi = words[1];
            std::uint32_t bits = 0;
            while (hi.is_value()) {
                const auto sb = set_bits(hi.value());
                if (!sb) break;
                hi = sb->first;
                bits |= sb->second;
            }
            TRY(addr, address(words[0], hi, Sum{}));
            return Source::Built(addr, bits, {words[2].n, words[3].n});
        }
        const bool any_loaded = std::any_of(words.begin(), words.end(), [&](const Term& w) {
            if (!w.is_value()) return false;
            const Inst* i = def(w.value());
            return i && i->name().starts_with("s_load_");
        });
        return Err{any_loaded ? "a descriptor changed after loading" : "a descriptor neither loaded nor built"};
    }

    // Pushes what lanes leaving block `q` for `block` know (its reach, and its branch's way).
    void push_edge(Known& known, const Structured& l, std::size_t q, std::size_t block) const
    {
        known.push_back({l.reach[q], true});
        const LaneExit& e = l.exits[q];
        if (e.is_branch() && e.taken != e.not_taken) known.push_back({e.when, e.taken == block});
    }

    // The one edge into the join `block` that lanes satisfying `known` can have taken, if the lanes'
    // paths (Structured) rule out every other. None at loop headers and without the lanes' paths.
    std::optional<std::size_t> only_edge(std::size_t block, Known& known) const
    {
        if (!lanes) return std::nullopt;
        const auto& preds = p->blocks[block].preds;
        if (preds.size() < 2 ||
            std::any_of(preds.begin(), preds.end(), [&](std::size_t q) { return rpo[q] >= rpo[block]; }))
            return std::nullopt;
        std::optional<std::size_t> only;
        for (std::size_t j = 0; j < preds.size(); ++j) {
            const std::size_t n = known.size();
            push_edge(known, *lanes, preds[j], block);
            const bool possible = meet(*lanes, known);
            known.resize(n);
            if (possible) {
                if (only) return std::nullopt;
                only = j;
            }
        }
        return only;
    }

    // Whether lowering can make a handle for `source` in any block: its heap index comes from root
    // parameters and constants alone (a table offset the code computes exists only where the load
    // is).
    bool anywhere(const Source& source) const
    {
        switch (source.kind) {
        case Source::Kind::Constant: return true;
        case Source::Kind::Built: return !source.address.dynamic;
        case Source::Kind::Loaded: {
            const Inst& i = inst(source.load);
            const auto* f = std::get_if<gfx10::Smem>(&i.fields);
            if (!f) return false;
            const auto a = smem_address(i, f->base, f->offset, f->soffset);
            return a.ok() && !a.value.dynamic;
        }
        default: return false;
        }
    }

    // `t` as the lanes that reach the access being traced hold it: through joins they can have
    // entered by one edge only (only_edge).
    Term live(Term t) const
    {
        if (!at || !lanes) return t;
        Known known{{lanes->reach[*at], true}};
        for (std::size_t k = 0; k < MAX_JOINS; ++k) {
            if (!t.is_value()) break;
            const auto ph = phis[t.n];
            if (!ph) break;
            const auto j = only_edge(ph->first, known);
            if (!j) break;
            t = origin(p->blocks[ph->first].phis[ph->second].args[*j]);
        }
        return t;
    }

    std::optional<std::pair<std::size_t, std::size_t>> phi_of(const Term& w) const
    {
        if (w.is_value()) return phis[w.n];
        return std::nullopt;
    }

    // Words that phis of one block join, traced per predecessor (words defined before the join hold
    // on every edge).
    R<std::optional<Source>> phi(const std::vector<Term>& words, std::vector<std::size_t>& joins, Known& known) const
    {
        std::optional<std::size_t> found;
        for (const Term& w : words)
            if (const auto ph = phi_of(w)) {
                found = ph->first;
                break;
            }
        if (!found) return std::optional<Source>{};
        const std::size_t block = *found;
        for (const Term& w : words)
            if (const auto ph = phi_of(w); ph && ph->first != block) return Err{"a descriptor joined in several blocks"};
        if (contains(joins, block)) return Err{"a descriptor that changes in a loop"};
        // Each level multiplies the sources by the predecessor count.
        if (joins.size() >= MAX_JOINS) return Err{"a descriptor joined too often"};
        joins.push_back(block);
        const auto& block_phis = p->blocks[block].phis;
        auto edge = [&](std::size_t j) {
            std::vector<Term> e;
            for (const Term& w : words) {
                const auto ph = phi_of(w);
                e.push_back(ph ? origin(block_phis[ph->second].args[j]) : w);
            }
            return e;
        };
        std::vector<Source> args;
        const char* error = nullptr;
        for (std::size_t j = 0; j < p->blocks[block].preds.size(); ++j) {
            auto a = classify(edge(j), joins, known);
            if (!a.ok()) {
                error = a.err;
                break;
            }
            args.push_back(std::move(a.value));
        }
        R<std::optional<Source>> source = std::optional<Source>{};
        if (!error) {
            source = std::optional<Source>{Source::Phi(block, std::move(args))};
        } else {
            // An edge holds no descriptor. With only one edge the lanes that read the descriptor can
            // have come by, it is what the registers hold on that one (a wave that took another edge
            // has no lane left that reads it), if that can be named where it is read.
            source = Err{error};
            if (const auto j = only_edge(block, known)) {
                auto s = classify(edge(*j), joins, known);
                if (s.ok() && anywhere(s.value)) source = std::optional<Source>{std::move(s.value)};
            }
        }
        joins.pop_back();
        return source;
    }

    // Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced per outcome (other words
    // are the same for both).
    R<std::optional<Source>> select(const std::vector<Term>& words, std::vector<std::size_t>& joins,
                                    Known& known) const
    {
        std::vector<std::optional<std::tuple<Value, Term, Term>>> picks;
        for (const Term& w : words) picks.push_back(w.is_value() ? picked(w.value()) : std::nullopt);
        std::optional<Value> scc;
        for (const auto& pk : picks)
            if (pk) {
                scc = std::get<0>(*pk);
                break;
            }
        if (!scc) return std::optional<Source>{};
        for (const auto& pk : picks)
            if (pk && std::get<0>(*pk) != *scc) return Err{"a descriptor chosen by several conditions"};
        // Each level doubles the sources; selects count like joins (a marker no block index reaches).
        if (joins.size() >= MAX_JOINS) return Err{"a descriptor joined too often"};
        joins.push_back(SIZE_MAX);
        auto side = [&](bool set) {
            std::vector<Term> s;
            for (std::size_t k = 0; k < words.size(); ++k)
                s.push_back(picks[k] ? (set ? std::get<1>(*picks[k]) : std::get<2>(*picks[k])) : words[k]);
            return s;
        };
        std::vector<Source> args;
        const char* error = nullptr;
        for (const bool set : {true, false}) {
            auto a = classify(side(set), joins, known);
            if (!a.ok()) {
                error = a.err;
                break;
            }
            args.push_back(std::move(a.value));
        }
        joins.pop_back();
        if (error) return Err{error};
        return std::optional<Source>{Source::Select(*scc, std::move(args))};
    }

    // For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`: the SCC it reads and the dword
    // it is when SCC is set, then clear.
    std::optional<std::tuple<Value, Term, Term>> picked(Value v) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* f = std::get_if<gfx10::Sop>(&i->fields);
        if (!f) return std::nullopt;
        const auto pos = std::find_if(i->writes.begin(), i->writes.end(), [&](const auto& w) { return w.second == v; });
        if (pos == i->writes.end()) return std::nullopt;
        const auto k = static_cast<std::uint8_t>(pos - i->writes.begin());
        const std::string_view name = i->name();
        bool wide;
        if (name == "s_cselect_b32" || name == "s_cmov_b32")
            wide = false;
        else if (name == "s_cselect_b64" || name == "s_cmov_b64")
            wide = true;
        else
            return std::nullopt;
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are sign-extended; 64-bit
        // literals and float constants are not followed.
        auto dword = [&](const Operand& op) -> std::optional<Term> {
            if (op.is_scalar() && op.code == gfx10::NULL_CODE) return Term::Const(0);
            if (op.is_scalar()) {
                const auto r = i->read(Reg::S(static_cast<std::uint8_t>(op.code + k)));
                if (!r) return std::nullopt;
                return origin(*r);
            }
            if (op.is_const() && !wide) {
                const auto c = inline_constant(op.code);
                if (!c) return std::nullopt;
                return Term::Const(*c);
            }
            if (op.is_const() && op.code >= 128 && op.code <= 208) {
                const auto low = inline_constant(op.code);
                if (!low) return std::nullopt;
                return Term::Const(k == 0 ? *low : static_cast<std::uint32_t>(static_cast<std::int32_t>(*low) >> 31));
            }
            if (op.is_literal() && !wide) return Term::Const(op.literal);
            return std::nullopt;
        };
        if (f->src.empty()) return std::nullopt;
        const auto set = dword(f->src[0]);
        if (!set) return std::nullopt;
        std::optional<Term> clear;
        if (name.starts_with("s_cmov")) {
            if (!f->dst) return std::nullopt;
            clear = dword(*f->dst);
        } else {
            if (f->src.size() < 2) return std::nullopt;
            clear = dword(f->src[1]);
        }
        if (!clear) return std::nullopt;
        const auto scc = i->read(Reg::Scc);
        if (!scc) return std::nullopt;
        return std::tuple{*scc, *set, *clear};
    }

    // `v` as some other dword with constant bits set.
    std::optional<std::pair<Term, std::uint32_t>> set_bits(Value v) const
    {
        if (const auto s = sop2_terms(v, {"s_or_b32"})) {
            if ((*s)[1].is_const()) return std::pair{(*s)[0], (*s)[1].n};
            if ((*s)[0].is_const()) return std::pair{(*s)[1], (*s)[0].n};
        }
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* f = std::get_if<gfx10::Sop>(&i->fields);
        if (!f || !f->dst || !f->dst->is_scalar()) return std::nullopt;
        if (f->src.empty()) return std::nullopt;
        const auto t = term(*i, f->src[0]);
        if (!t) return std::nullopt;
        if (i->name() == "s_bitset1_b32" && t->is_const()) {
            const auto r = i->read(Reg::S(f->dst->code));
            if (!r) return std::nullopt;
            return std::pair{origin(*r), 1u << (t->n & 31)};
        }
        return std::nullopt;
    }

    // `x op c` with `c` either side.
    std::optional<Term> with(Value v, std::string_view name, std::uint32_t c) const
    {
        const auto s = sop2_terms(v, {name});
        if (!s) return std::nullopt;
        if ((*s)[1].is_const() && (*s)[1].n == c) return (*s)[0];
        if ((*s)[0].is_const() && (*s)[0].n == c) return (*s)[1];
        return std::nullopt;
    }

    // The words of a buffer descriptor that `words` makes raw, as the upscaler's compiler does before
    // byte accesses: the stride (dword 1 bits 16-29) cleared, the record count times max(stride, 1),
    // out-of-bounds checking by bytes (dword 3 bits 28-29 set). The bounds are then the view's size
    // in bytes either way.
    std::optional<std::array<Term, 4>> made_raw(const std::vector<Term>& words) const
    {
        if (words.size() != 4 || !words[1].is_value() || !words[2].is_value() || !words[3].is_value())
            return std::nullopt;
        const Term w0 = words[0];
        const auto d1 = with(words[1].value(), "s_and_b32", 0xC000'FFFF);
        if (!d1) return std::nullopt;
        const auto m3 = with(words[3].value(), "s_or_b32", 0x3000'0000);
        if (!m3 || !m3->is_value()) return std::nullopt;
        const auto d3 = with(m3->value(), "s_and_b32", 0xCFFF'FFFF);
        if (!d3) return std::nullopt;
        const auto ab = sop2_terms(words[2].value(), {"s_mul_i32"});
        if (!ab) return std::nullopt;
        const Term a = (*ab)[0], b = (*ab)[1];
        auto max = [&](Term t) -> std::optional<Term> {
            if (!t.is_value()) return std::nullopt;
            return with(t.value(), "s_max_u32", 1);
        };
        const auto ma = max(a);
        const auto mb = max(b);
        Term d2, field;
        if (mb) {
            d2 = a;
            field = *mb;
        } else if (ma) {
            d2 = b;
            field = *ma;
        } else {
            return std::nullopt;
        }
        if (!field.is_value()) return std::nullopt;
        // s_bfe_u32: offset 16, width 14.
        const auto bfe = sop2_terms(field.value(), {"s_bfe_u32"});
        if (!bfe || !((*bfe)[1] == Term::Const(0x000E'0010))) return std::nullopt;
        if ((*bfe)[0] != *d1) return std::nullopt;
        return std::array<Term, 4>{w0, *d1, d2, *d3};
    }

    // For a `v_readfirstlane_b32` result: the VGPR value it reads.
    std::optional<Value> first_lane_source(Value v) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* f = std::get_if<gfx10::Valu>(&i->fields);
        if (!f) return std::nullopt;
        if (i->name() != "v_readfirstlane_b32" || f->sdwa || f->dpp) return std::nullopt;
        if (f->src.empty() || !f->src[0].is_vector()) return std::nullopt;
        return i->read(Reg::V(f->src[0].reg));
    }

    // The `v_readfirstlane_b32` result the scalar load at `load` adds to its address as a table offset
    // (the console's waterfall, `s_load_dwordx8 .., s[4:5], s20`): `track` proves the waterfall from
    // it, and lowering reads it per lane there.
    std::optional<Value> first_lane_offset(Site load) const
    {
        const Inst& i = inst(load);
        const auto* f = std::get_if<gfx10::Smem>(&i.fields);
        if (!f) return std::nullopt;
        read_first_lane = true;
        const auto address = smem_address(i, f->base, f->offset, f->soffset);
        read_first_lane = false;
        if (!address.ok() || !address.value.dynamic) return std::nullopt;
        const Term t = origin(*address.value.dynamic);
        if (t.is_value() && first_lane_source(t.value())) return t.value();
        return std::nullopt;
    }

    // Whether the descriptor operand `op` of `i` is one the code keeps per lane (lane_copies).
    bool kept_per_lane(const Inst& i, const Operand& op) const
    {
        if (!op.is_scalar()) return false;
        const auto words = operand_words(i, op);
        return words && lane_copies(*words).has_value();
    }

    // The `v_readfirstlane_b32` results the descriptor operand `op` of `i` is, when every dword is
    // one.
    std::optional<std::vector<Value>> first_lane_words(const Inst& i, const Operand& op) const
    {
        if (!op.is_scalar()) return std::nullopt;
        std::vector<Value> out;
        for (std::uint8_t k = 0; k < op.dwords; ++k) {
            const auto r = i.read(Reg::S(static_cast<std::uint8_t>(op.code + k)));
            if (!r) return std::nullopt;
            const Term t = origin(*r);
            if (!t.is_value() || !first_lane_source(t.value())) return std::nullopt;
            out.push_back(t.value());
        }
        return out;
    }

    // `t` through VGPR copies (of VGPRs or of scalars) and copies.
    Term lane_term(Term t) const
    {
        for (std::size_t k = 0; k < MAX_JOINS * 2; ++k) {
            if (!t.is_value()) break;
            const Term next = vorigin(t.value());
            if (next == t) break;
            t = next;
        }
        return t;
    }

    // Whether `t` is a value each lane holds its own of (one in VGPRs).
    bool varies(Term t) const
    {
        if (!t.is_value()) return false;
        const Value v = t.value();
        std::optional<Reg> reg;
        if (inputs[v.n]) {
            reg = inputs[v.n];
        } else if (const auto ph = phis[v.n]) {
            reg = p->blocks[ph->first].phis[ph->second].reg;
        } else if (const Inst* i = def(v)) {
            for (const auto& w : i->writes)
                if (w.second == v) {
                    reg = w.first;
                    break;
                }
        }
        return reg && reg->is_v();
    }

    // lane_term(), where every lane in `known` runs each VGPR copy it follows.
    R<Term> lane_value(Term t, const Known& known) const
    {
        for (std::size_t k = 0; k < MAX_JOINS * 2; ++k) {
            if (!t.is_value()) break;
            const Value v = t.value();
            const Term next = vstep(v);
            if (next == t) break;
            if (varies(t)) {
                if (const auto site = defs[v.n]) {
                    TRY(ok, runs(*site, known));
                    (void)ok;
                }
            }
            t = next;
        }
        return t;
    }

    // One copy of vorigin(): what the `v_mov_b32` writing `v` reads, or `v`'s origin.
    Term vstep(Value v) const
    {
        if (const Inst* i = def(v); i && i->name() == "v_mov_b32") {
            if (const auto* f = std::get_if<gfx10::Valu>(&i->fields); f && plain_valu(*f) && !f->src.empty()) {
                if (const auto t = raw(*i, f->src[0])) return *t;
            }
        }
        return origin(v);
    }

    // A VALU operand as `i` reads it: a VGPR's value itself, other operands' origins.
    std::optional<Term> raw(const Inst& i, const Operand& op) const
    {
        if (op.is_vector()) {
            const auto v = i.read(Reg::V(op.reg));
            if (!v) return std::nullopt;
            return Term::Of(*v);
        }
        return term(i, op);
    }

    // Fails unless every lane in `known` that reaches the vector instruction at `site` runs it: one
    // with its EXEC bit clear there keeps what the registers held before.
    R<bool> runs(Site site, const Known& known) const
    {
        if (!lanes) return true;
        if (site.first >= lanes->guards.size() || site.second >= lanes->guards[site.first].size()) return true;
        const auto& guard = lanes->guards[site.first][site.second];
        if (!guard) return true;
        Known skips = known;
        skips.push_back({lanes->reach[site.first], true});
        skips.push_back({*guard, false});
        if (meet(*lanes, skips)) return Err{"a per-lane descriptor written where some lanes may keep the old one"};
        return true;
    }

    // The source of descriptor dwords each lane holds its own of (a waterfall loop reads them from
    // the first active lane): the same in every lane, loaded per lane from memory a heap address
    // points to, or picked per lane between such sources, by a lane mask or by the path the lane
    // took. `known` holds in every lane that accesses them.
    R<Source> per_lane(const std::vector<Term>& words0, std::vector<std::size_t>& joins, Known& known) const
    {
        std::vector<Term> words;
        for (const Term& w : words0) {
            TRY(t, lane_value(w, known));
            words.push_back(t);
        }
        const auto lane = std::find_if(words.begin(), words.end(), [&](const Term& w) { return varies(w); });
        if (lane == words.end()) return classify(words, joins, known);
        {
            TRY(s, fetched(words, known));
            if (s) return std::move(*s);
        }
        {
            TRY(s, lane_select(words, joins, known));
            if (s) return std::move(*s);
        }
        {
            TRY(s, lane_phi(words, joins, known));
            if (s) return std::move(*s);
        }
        const Value v = lane->value();
        const Inst* i = def(v);
        if (inputs[v.n]) return Err{"a descriptor read from the first lane of an input VGPR"};
        if (!i) return Err{"a descriptor read from the first lane of a phi"};
        const std::string_view n = i->name();
        if (n.starts_with("ds_read")) return Err{"a descriptor read from the first lane of an LDS read"};
        if (n.starts_with("buffer_load")) return Err{"a descriptor read from the first lane of a buffer load"};
        if (n.starts_with("global_load") || n.starts_with("flat_load"))
            return Err{"a descriptor read from the first lane of a partial or mixed load"};
        return Err{"a descriptor read from the first lane of computed VGPRs"};
    }

    // Dwords of `global_load_dword*` that line up: `first..` of one load, then the bytes that follow
    // (an image descriptor takes two `global_load_dwordx4`).
    R<std::optional<Source>> fetched(const std::vector<Term>& words, const Known& known) const
    {
        std::vector<std::pair<Site, std::size_t>> sites;
        for (const Term& w : words) {
            if (!w.is_value()) return std::optional<Source>{};
            const Value v = w.value();
            const auto site = defs[v.n];
            if (!site) return std::optional<Source>{};
            const Inst& i = inst(*site);
            if (!i.name().starts_with("global_load_dword")) return std::optional<Source>{};
            std::optional<std::size_t> k;
            std::size_t n = 0;
            for (const auto& wr : i.writes) {
                if (!wr.first.is_v()) continue;
                if (wr.second == v) {
                    k = n;
                    break;
                }
                ++n;
            }
            if (!k) return std::optional<Source>{};
            sites.push_back({*site, *k});
        }
        auto address = [&](Site site) -> R<Address> {
            const Inst& i = inst(site);
            const auto* f = std::get_if<gfx10::Flat>(&i.fields);
            if (!f) return Err{"a global load"};
            return flat_address(i, f->addr, f->saddr, f->offset);
        };
        const auto [load, first] = sites[0];
        TRY(a0, address(load));
        for (std::size_t k = 0; k < sites.size(); ++k) {
            const auto [site, dword] = sites[k];
            TRY(ok, runs(site, known));
            (void)ok;
            TRY(a, address(site));
            const std::uint32_t want = a0.offset + static_cast<std::uint32_t>(4 * (first + k));
            if (a.param != a0.param || a.dynamic != a0.dynamic || a.offset + static_cast<std::uint32_t>(4 * dword) != want)
                return Err{"a descriptor read from the first lane of loads that do not line up"};
        }
        return std::optional<Source>{Source::Fetched(load, static_cast<std::uint8_t>(first))};
    }

    // Words that `v_cndmask_b32` picks by one lane mask, traced per outcome (other words are the same
    // for both).
    R<std::optional<Source>> lane_select(const std::vector<Term>& words, std::vector<std::size_t>& joins,
                                         Known& known) const
    {
        std::vector<std::optional<std::tuple<Value, Term, Term>>> picks;
        for (const Term& w : words) picks.push_back(w.is_value() ? lane_picked(w.value()) : std::nullopt);
        std::optional<Value> mask;
        for (const auto& pk : picks)
            if (pk) {
                mask = std::get<0>(*pk);
                break;
            }
        if (!mask) return std::optional<Source>{};
        for (const auto& pk : picks)
            if (pk && std::get<0>(*pk) != *mask) return Err{"a descriptor chosen per lane by several masks"};
        for (std::size_t k = 0; k < words.size(); ++k) {
            if (words[k].is_value() && picks[k]) {
                if (const auto site = defs[words[k].n]) {
                    TRY(ok, runs(*site, known));
                    (void)ok;
                }
            }
        }
        if (joins.size() >= MAX_JOINS) return Err{"a descriptor joined too often"};
        joins.push_back(SIZE_MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        std::optional<Bool> bit;
        if (lanes) {
            if (const auto it = lanes->masks.find(*mask); it != lanes->masks.end()) bit = it->second;
        }
        auto side = [&](bool set) {
            std::vector<Term> s;
            for (std::size_t k = 0; k < words.size(); ++k)
                s.push_back(picks[k] ? (set ? std::get<1>(*picks[k]) : std::get<2>(*picks[k])) : words[k]);
            return s;
        };
        std::vector<Source> args;
        const char* error = nullptr;
        for (const bool set : {true, false}) {
            const std::size_t n = known.size();
            if (bit) known.push_back({*bit, set});
            auto arg = per_lane(side(set), joins, known);
            known.resize(n);
            if (!arg.ok()) {
                error = arg.err;
                break;
            }
            args.push_back(std::move(arg.value));
        }
        joins.pop_back();
        if (error) return Err{error};
        return std::optional<Source>{Source::Lanes(*mask, std::move(args))};
    }

    // For `v` written by a plain `v_cndmask_b32`: the low dword of the lane mask it reads, and the
    // dword it is where the lane's bit is set (src1), then clear (src0).
    std::optional<std::tuple<Value, Term, Term>> lane_picked(Value v) const
    {
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* f = std::get_if<gfx10::Valu>(&i->fields);
        if (!f) return std::nullopt;
        if (!f->dst || !f->dst->is_vector()) return std::nullopt;
        if (i->name() != "v_cndmask_b32" || i->write(Reg::V(f->dst->reg)) != v || f->abs != 0 || f->neg != 0 ||
            f->clamp || f->omod != 0 || f->sdwa || f->dpp)
            return std::nullopt;
        if (f->src.size() != 3 || !f->src[2].is_scalar()) return std::nullopt;
        const Operand& clear = f->src[0];
        const Operand& set = f->src[1];
        const auto mask = i->read(Reg::S(f->src[2].code));
        if (!mask) return std::nullopt;
        const auto s = raw(*i, set);
        if (!s) return std::nullopt;
        const auto c = raw(*i, clear);
        if (!c) return std::nullopt;
        return std::tuple{*mask, *s, *c};
    }

    // Per-lane words that phis of one block join, as an if/else on a per-lane condition leaves them:
    // traced on each edge into the block that lanes in `known` can take (other words hold on every
    // edge). Lanes take their own paths (Structured), so each gets the value of the edge it came by;
    // an edge they cannot take is nullopt.
    R<std::optional<Source>> lane_phi(const std::vector<Term>& words, std::vector<std::size_t>& joins,
                                      Known& known) const
    {
        std::optional<std::size_t> found;
        for (const Term& w : words)
            if (const auto ph = phi_of(w)) {
                found = ph->first;
                break;
            }
        if (!found) return std::optional<Source>{};
        const std::size_t block = *found;
        for (const Term& w : words)
            if (const auto ph = phi_of(w); ph && ph->first != block) return Err{"a descriptor joined in several blocks"};
        if (contains(joins, block)) return Err{"a descriptor that changes in a loop"};
        const auto& preds = p->blocks[block].preds;
        if (std::any_of(preds.begin(), preds.end(), [&](std::size_t q) { return rpo[q] >= rpo[block]; })) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err{"a descriptor read from the first lane of a phi at a loop header"};
        }
        if (!lanes) return Err{"a descriptor read from the first lane of a phi, without the lanes' paths"};
        // The other words must be there before the join: defined in a block that dominates it, which
        // (both dominating the access) is one the join does not reach.
        for (const Term& w : words) {
            if (!w.is_value() || phi_of(w)) continue;
            const auto d = defs[w.n];
            if (d && (d->first == block || reaches(block, d->first))) return Err{"a descriptor partly joined by control flow"};
        }
        if (joins.size() >= MAX_JOINS) return Err{"a descriptor joined too often"};
        joins.push_back(block);
        const auto& block_phis = p->blocks[block].phis;
        std::vector<std::optional<Source>> args;
        for (std::size_t j = 0; j < preds.size(); ++j) {
            const std::size_t n = known.size();
            push_edge(known, *lanes, preds[j], block);
            R<std::optional<Source>> arg = std::optional<Source>{};
            if (meet(*lanes, known)) {
                std::vector<Term> edge;
                for (const Term& w : words) {
                    const auto ph = phi_of(w);
                    edge.push_back(ph ? Term::Of(block_phis[ph->second].args[j]) : w);
                }
                auto s = per_lane(edge, joins, known);
                if (s.ok())
                    arg = std::optional<Source>{std::move(s.value)};
                else
                    arg = Err{s.err};
            }
            known.resize(n);
            if (!arg.ok()) {
                joins.pop_back();
                return Err{arg.err};
            }
            args.push_back(std::move(arg.value));
        }
        joins.pop_back();
        return std::optional<Source>{Source::LanePhi(block, std::move(args))};
    }

    // Whether a path leads from block `from` to block `to` (to `from` itself only through a back
    // edge).
    bool reaches(std::size_t from, std::size_t to) const
    {
        std::vector<bool> seen(p->blocks.size(), false);
        std::vector<std::size_t> stack = p->blocks[from].exit.successors();
        while (!stack.empty()) {
            const std::size_t x = stack.back();
            stack.pop_back();
            if (x == to) return true;
            if (seen[x]) continue;
            seen[x] = true;
            for (const std::size_t s : p->blocks[x].exit.successors()) stack.push_back(s);
        }
        return false;
    }

    // Prove a single-block waterfall at the descriptor access. In each iteration the access serves
    // exactly the matching lanes, and the back edge keeps exactly the other lanes active. Otherwise,
    // why not.
    //
    // shortcut: only the accesses are proven. Any other per-lane work in the loop is taken to run
    // once for each lane, as the access does; a body whose effect adds up over the passes a lane
    // waits through (a counter before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    R<Waterfall> waterfall(std::size_t b, const std::vector<std::size_t>& accesses, const std::vector<Value>& reads) const
    {
        static constexpr const char* LOOP = "a descriptor read from the first lane outside a waterfall loop";
        static constexpr const char* SPLIT = "a waterfall loop nested in another or split over blocks";
        static constexpr const char* SELECT = "a waterfall loop whose compares do not select the access's lanes";
        static constexpr const char* RETRY = "a waterfall loop whose retry does not keep the lanes still waiting";
        const std::size_t access = accesses[0];
        const auto& block = p->blocks[b];
        if (!lanes) return Err{LOOP};
        const Structured& l = *lanes;
        if (block.insts.empty()) return Err{LOOP};
        const Inst& exit = block.insts.back();
        if (!(block.exit.is_branch() && block.exit.taken == b && block.exit.not_taken != b) ||
            exit.name() != "s_cbranch_execnz")
            return Err{LOOP};
        auto mask = [&](std::optional<Value> low, std::optional<Value> high) -> std::optional<Bool> {
            std::optional<Value> h;
            if (p->wave64) {
                if (!high) return std::nullopt;
                h = high;
            }
            if (!low) return std::nullopt;
            return l.mask_bits(*low, h);
        };
        auto exec = [&](const Inst& i) {
            return mask(i.read(Reg::S(ir::EXEC)), i.read(Reg::S(ir::EXEC + 1)));
        };
        // The reads belong to this loop's own block: a loop around this one (one per operand, as LLVM
        // nests them for a per-lane image and sampler) is not followed.
        if (std::any_of(reads.begin(), reads.end(), [&](Value v) { return !defs[v.n] || defs[v.n]->first != b; }))
            return Err{SPLIT};
        const auto selected0 = exec(block.insts[access]);
        if (!selected0) return Err{SELECT};
        const Bool selected = *selected0;
        if (std::any_of(accesses.begin(), accesses.end(), [&](std::size_t i) { return exec(block.insts[i]) != selected; }))
            return Err{SELECT};
        const auto retry0 = exec(exit);
        if (!retry0) return Err{RETRY};
        const Bool retry = *retry0;
        const auto first = defs[reads[0].n];
        if (!first) return Err{LOOP};
        const auto remaining0 = exec(inst(*first));
        if (!remaining0) return Err{SELECT};
        const Bool remaining = *remaining0;
        if (std::any_of(reads.begin(), reads.end(), [&](Value v) {
                const auto d = defs[v.n];
                return !d || d->second >= access || exec(inst(*d)) != remaining;
            }))
            return Err{SELECT};
        const std::pair<Bool, bool> reach{l.reach[b], true};
        auto meet_ = [&](const Known& conditions) { return meet(l, conditions); };
        Waterfall out;
        Known required{{remaining, true}};
        for (std::size_t i = 0; i < access; ++i) {
            const Inst& in = block.insts[i];
            const auto* f = std::get_if<gfx10::Valu>(&in.fields);
            if (!f) continue;
            const std::string_view name = in.name();
            std::string_view ty;
            if (name.starts_with("v_cmp_eq_"))
                ty = name.substr(9);
            else if (name.starts_with("v_cmpx_eq_"))
                ty = name.substr(10);
            else
                continue;
            std::uint8_t dwords;
            if (ty == "u32" || ty == "i32")
                dwords = 1;
            else if (ty == "u64" || ty == "i64")
                dwords = 2;
            else
                continue;
            if (!plain_valu(*f)) continue;
            // The first lane's dwords on one side, the lane's own on the other.
            auto pair = [&](const Operand& fst, const Operand& own) -> std::optional<std::vector<Value>> {
                if (!fst.is_scalar() || !own.is_vector()) return std::nullopt;
                std::vector<Value> xs;
                for (std::uint8_t k = 0; k < dwords; ++k) {
                    const auto sv = in.read(Reg::S(static_cast<std::uint8_t>(fst.code + k)));
                    if (!sv) return std::nullopt;
                    const Term t = origin(*sv);
                    if (!t.is_value()) return std::nullopt;
                    const Value x = t.value();
                    const auto lane = in.read(Reg::V(static_cast<std::uint8_t>(own.reg + k)));
                    if (!lane) return std::nullopt;
                    const auto d = defs[x.n];
                    if (!d) return std::nullopt;
                    const std::size_t read = d->second;
                    bool same = false;
                    if (contains(reads, x) && read < i) {
                        const auto src = first_lane_source(x);
                        if (!src) return std::nullopt;
                        same = lane_term(Term::Of(*src)) == lane_term(Term::Of(*lane));
                    }
                    if (!same) return std::nullopt;
                    xs.push_back(x);
                }
                return xs;
            };
            if (f->src.size() < 2) continue;
            const Operand& a = f->src[0];
            const Operand& c = f->src[1];
            auto xs = pair(a, c);
            if (!xs) xs = pair(c, a);
            if (!xs) continue;
            std::uint8_t code;
            if (name.starts_with("v_cmpx_"))
                code = ir::EXEC;
            else if (f->dst && f->dst->is_scalar())
                code = f->dst->code;
            else
                continue;
            const auto matched0 = mask(in.write(Reg::S(code)), in.write(Reg::S(static_cast<std::uint8_t>(code + 1))));
            if (!matched0) continue;
            const Bool matched = *matched0;
            if (meet_({reach, {selected, true}, {matched, false}})) continue;
            required.push_back({matched, true});
            out.compares.push_back({b, i});
            for (const Value x : *xs)
                if (!contains(out.reads, x)) out.reads.push_back(x);
        }
        if (std::any_of(reads.begin(), reads.end(), [&](Value v) { return !contains(out.reads, v); }) ||
            meet_({reach, {selected, true}, {remaining, false}}))
            return Err{SELECT};
        // The access must take every match.
        required.push_back(reach);
        required.push_back({selected, false});
        if (meet_(required)) return Err{SELECT};
        // The retry must keep exactly the lanes that were eligible but did not perform the access.
        if (meet_({reach, {retry, true}, {remaining, false}}) || meet_({reach, {retry, true}, {selected, true}}) ||
            meet_({reach, {remaining, true}, {selected, false}, {retry, false}}))
            return Err{RETRY};
        return out;
    }

    // Consecutive dwords of one scalar load.
    std::optional<Source> loaded(const std::vector<Term>& words) const
    {
        if (words.empty() || !words[0].is_value()) return std::nullopt;
        const Value v0 = words[0].value();
        const auto load = defs[v0.n];
        if (!load) return std::nullopt;
        const Inst& i = inst(*load);
        if (!i.name().starts_with("s_load_")) return std::nullopt;
        const auto pos = std::find_if(i.writes.begin(), i.writes.end(), [&](const auto& w) { return w.second == v0; });
        if (pos == i.writes.end()) return std::nullopt;
        const std::size_t first = static_cast<std::size_t>(pos - i.writes.begin());
        for (std::size_t k = 0; k < words.size(); ++k)
            if (first + k >= i.writes.size() || words[k] != Term::Of(i.writes[first + k].second)) return std::nullopt;
        return Source::Loaded(*load, static_cast<std::uint8_t>(first));
    }

    // A descriptor the code keeps per lane: every dword of `words` is the first lane
    // (`v_readfirstlane_b32`) of a VGPR that `v_mov_b32` filled from SGPRs. When every scalar those
    // moves copy is the same dword of loads from one constant address, the lanes agree and the
    // descriptor is any of those loads.
    // shortcut: what else the VGPRs held (numbers, on paths or in lanes that never read them as a
    // descriptor) is ignored, and a move is taken to leave either its source or what a move before it
    // wrote; a descriptor read per lane from memory (`global_load_*`) refuses.
    std::optional<Source> lane_copies(const std::vector<Term>& words) const
    {
        std::vector<Site> loads;
        std::optional<std::size_t> first;
        for (std::size_t k = 0; k < words.size(); ++k) {
            if (!words[k].is_value()) return std::nullopt;
            const Value v = words[k].value();
            std::vector<Term> copies;
            const auto fl = first_lane(v);
            if (!fl) return std::nullopt;
            if (!scalar_copies(*fl, copies)) return std::nullopt;
            bool found = false;
            for (const Term c0 : copies) {
                if (!c0.is_value()) continue;
                const Value c = c0.value();
                const auto site = defs[c.n];
                if (!site) continue;
                const Inst& load = inst(*site);
                if (!load.name().starts_with("s_load_")) continue;
                // Dword k of the operand is dword `first + k` of each load.
                const auto pos =
                    std::find_if(load.writes.begin(), load.writes.end(), [&](const auto& x) { return x.second == c; });
                if (pos == load.writes.end()) return std::nullopt;
                const std::size_t dword = static_cast<std::size_t>(pos - load.writes.begin());
                if (dword < k) return std::nullopt;
                const std::size_t f = dword - k;
                if (!first) first = f;
                if (*first != f) return std::nullopt;
                if (!contains(loads, *site)) loads.push_back(*site);
                found = true;
            }
            if (!found) return std::nullopt;
        }
        auto address = [&](Site site) -> std::optional<Address> {
            const Inst& i = inst(site);
            const auto* f = std::get_if<gfx10::Smem>(&i.fields);
            if (!f) return std::nullopt;
            const auto a = smem_address(i, f->base, f->offset, f->soffset);
            if (!a.ok()) return std::nullopt;
            return a.value;
        };
        if (loads.empty()) return std::nullopt;
        const auto a = address(loads.front());
        if (!a) return std::nullopt;
        if (a->dynamic || std::any_of(loads.begin(), loads.end(), [&](Site l) { return address(l) != a; }))
            return std::nullopt;
        // The handle is made where the descriptor is used, from the load's operands: take a load whose
        // base is the root parameter's own SGPR.
        std::sort(loads.begin(), loads.end());
        const auto load = std::find_if(loads.begin(), loads.end(), [&](Site l) { return base_from_entry(l); });
        if (load == loads.end()) return std::nullopt;
        lane_copied.insert(lane_copied.end(), loads.begin(), loads.end());
        if (!first) return std::nullopt;
        return Source::Loaded(*load, static_cast<std::uint8_t>(*first));
    }

    // The scalars (as origins) that `v_mov_b32` copies put in VGPR value `v`, through joins, moves
    // between VGPRs and per-lane selects; a move keeps what it wrote over in the lanes it does not
    // run. False if the VGPR may hold something read per lane from memory.
    bool scalar_copies(Value v0, std::vector<Term>& out) const
    {
        std::vector<Value> seen;
        std::vector<Value> work{v0};
        while (!work.empty()) {
            const Value v = work.back();
            work.pop_back();
            if (contains(seen, v)) continue;
            if (seen.size() == MAX_COPIES) return false;
            seen.push_back(v);
            if (const auto ph = phis[v.n]) {
                const auto& args = p->blocks[ph->first].phis[ph->second].args;
                work.insert(work.end(), args.begin(), args.end());
                continue;
            }
            // An entry value, or a number some instruction computed.
            const Inst* i = def(v);
            if (!i) continue;
            if (i->name().starts_with("global_load_")) return false;
            const auto* f = std::get_if<gfx10::Valu>(&i->fields);
            if (!f) continue;
            std::size_t sources;
            if (i->name() == "v_mov_b32")
                sources = 1;
            else if (i->name() == "v_cndmask_b32")
                sources = 2;
            else
                continue;
            if (!plain_valu(*f)) continue;
            for (std::size_t k = 0; k < std::min(sources, f->src.size()); ++k) {
                const Operand& op = f->src[k];
                if (op.is_vector()) {
                    if (const auto r = i->read(Reg::V(op.reg))) work.push_back(*r);
                } else if (op.is_scalar() && op.code != gfx10::NULL_CODE) {
                    if (const auto r = i->read(Reg::S(op.code))) out.push_back(origin(*r));
                }
            }
            // Lanes outside EXEC keep the old value.
            if (f->dst && f->dst->is_vector()) {
                if (const auto r = i->read(Reg::V(f->dst->reg))) work.push_back(*r);
            }
        }
        return true;
    }

    // The load at `site` has a constant offset and takes its base from a root parameter's SGPR
    // through moves alone: a value that exists wherever the descriptor is used.
    bool base_from_entry(Site site) const
    {
        const Inst& i = inst(site);
        const auto* f = std::get_if<gfx10::Smem>(&i.fields);
        if (!f || !f->base.is_scalar() || !(f->soffset.is_scalar() && f->soffset.code == gfx10::NULL_CODE)) return false;
        auto v = i.read(Reg::S(f->base.code));
        if (!v) return false;
        for (;;) {
            if (inputs[v->n]) return true;
            const Inst* d = def(*v);
            if (!d) return false;
            const auto pos = std::find_if(d->writes.begin(), d->writes.end(), [&](const auto& w) { return w.second == *v; });
            if (pos == d->writes.end()) return false;
            const auto m = copy_source(*d, static_cast<std::size_t>(pos - d->writes.begin()));
            if (!m || m->is_const) return false;
            v = Value{m->n};
        }
    }

    // For `v_readlane_b32` of a constant lane: the SGPR spilled to that lane, found by following the
    // VGPR's `v_writelane_b32` writes.
    std::optional<Moved> restored(const Inst& i) const
    {
        const auto* v = std::get_if<gfx10::Valu>(&i.fields);
        if (!v) return std::nullopt;
        if (i.name() != "v_readlane_b32") return std::nullopt;
        if (v->src.size() != 2 || !v->src[0].is_vector()) return std::nullopt;
        const auto r = i.read(Reg::V(v->src[0].reg));
        if (!r) return std::nullopt;
        const auto lane = lane_of(v->src[1]);
        if (!lane) return std::nullopt;
        std::vector<Value> seen;
        const auto s = spilled(*r, *lane, seen);
        if (!s) return std::nullopt;
        return *s;
    }

    // What lane `lane` of VGPR value `v` holds: an empty inner optional for a phi in `seen` (already being
    // resolved), nullopt if not a spilled SGPR.
    std::optional<std::optional<Moved>> spilled(Value v, std::uint32_t lane, std::vector<Value>& seen) const
    {
        if (const auto ph = phis[v.n]) {
            if (contains(seen, v)) return std::optional<Moved>{};
            seen.push_back(v);
            std::optional<Moved> found;
            for (const Value a : p->blocks[ph->first].phis[ph->second].args) {
                const auto r = spilled(a, lane, seen);
                if (!r) return std::nullopt;
                if (*r) {
                    if (found && *found != **r) return std::nullopt;
                    found = **r;
                }
            }
            return found;
        }
        const Inst* i = def(v);
        if (!i) return std::nullopt;
        const auto* w = std::get_if<gfx10::Valu>(&i->fields);
        if (!w) return std::nullopt;
        if (!w->dst || !w->dst->is_vector() || w->src.size() != 2) return std::nullopt;
        const Operand value = w->src[0];
        const Operand at_ = w->src[1];
        if (i->name() != "v_writelane_b32") return std::nullopt;
        const auto l = lane_of(at_);
        if (!l) return std::nullopt;
        if (*l != lane) {
            const auto r = i->read(Reg::V(w->dst->reg));
            if (!r) return std::nullopt;
            return spilled(*r, lane, seen);
        }
        std::optional<Moved> m;
        if (value.is_scalar() && value.code != gfx10::NULL_CODE) {
            if (const auto r = i->read(Reg::S(value.code))) m = Moved::Of(*r);
        } else if (value.is_literal()) {
            m = Moved::Const(value.literal);
        } else if (value.is_const()) {
            if (const auto c = inline_constant(value.code)) m = Moved::Const(*c);
        }
        if (!m) return std::nullopt;
        return std::optional<Moved>{*m};
    }
};

#undef TRY

}  // namespace

// ---- Source ----

Source Source::Loaded(Site load, std::uint8_t first)
{
    Source s;
    s.kind = Kind::Loaded;
    s.load = load;
    s.first = first;
    return s;
}
Source Source::Built(Address address, std::uint32_t bits, std::array<std::uint32_t, 2> words)
{
    Source s;
    s.kind = Kind::Built;
    s.address = address;
    s.bits = bits;
    s.words = words;
    return s;
}
Source Source::Constant(std::vector<std::uint32_t> words)
{
    Source s;
    s.kind = Kind::Constant;
    s.constant = std::move(words);
    return s;
}
Source Source::Phi(std::size_t block, std::vector<Source> args)
{
    Source s;
    s.kind = Kind::Phi;
    s.block = block;
    s.args = std::move(args);
    return s;
}
Source Source::Select(ir::Value scc, std::vector<Source> args)
{
    Source s;
    s.kind = Kind::Select;
    s.scc = scc;
    s.args = std::move(args);
    return s;
}
Source Source::Fetched(Site load, std::uint8_t first)
{
    Source s;
    s.kind = Kind::Fetched;
    s.load = load;
    s.first = first;
    return s;
}
Source Source::Lanes(ir::Value mask, std::vector<Source> args)
{
    Source s;
    s.kind = Kind::Lanes;
    s.mask = mask;
    s.args = std::move(args);
    return s;
}
Source Source::LanePhi(std::size_t block, std::vector<std::optional<Source>> args)
{
    Source s;
    s.kind = Kind::LanePhi;
    s.block = block;
    s.lane_args = std::move(args);
    return s;
}

std::vector<const Source*> Source::leaves() const
{
    std::vector<const Source*> out;
    switch (kind) {
    case Kind::Phi:
    case Kind::Select:
    case Kind::Lanes:
        for (const Source& a : args)
            for (const Source* l : a.leaves()) out.push_back(l);
        break;
    case Kind::LanePhi:
        for (const auto& a : lane_args)
            if (a)
                for (const Source* l : a->leaves()) out.push_back(l);
        break;
    default: out.push_back(this);
    }
    return out;
}

bool Source::operator==(const Source& o) const
{
    if (kind != o.kind) return false;
    switch (kind) {
    case Kind::Loaded:
    case Kind::Fetched: return load == o.load && first == o.first;
    case Kind::Built: return address == o.address && bits == o.bits && words == o.words;
    case Kind::Constant: return constant == o.constant;
    case Kind::Phi: return block == o.block && args == o.args;
    case Kind::Select: return scc == o.scc && args == o.args;
    case Kind::Lanes: return mask == o.mask && args == o.args;
    case Kind::LanePhi: return block == o.block && lane_args == o.lane_args;
    }
    return false;
}

std::optional<std::uint32_t> inline_constant(std::uint8_t code)
{
    auto fl = [](float f) { return std::bit_cast<std::uint32_t>(f); };
    if (code >= 128 && code <= 192) return std::uint32_t{code} - 128;
    if (code >= 193 && code <= 208) return static_cast<std::uint32_t>(192 - static_cast<std::int32_t>(code));
    switch (code) {
    case 240: return fl(0.5f);
    case 241: return fl(-0.5f);
    case 242: return fl(1.0f);
    case 243: return fl(-1.0f);
    case 244: return fl(2.0f);
    case 245: return fl(-2.0f);
    case 246: return fl(4.0f);
    case 247: return fl(-4.0f);
    case 248: return 0x3E22'F983u;
    default: return std::nullopt;
    }
}

namespace detail {
std::vector<Term> origins(const ir::Program& p, std::span<const native::RootParam> layout, native::HwStage stage)
{
    const Tracer t(p, layout, stage, nullptr);
    return t.origin_;
}
}  // namespace detail

Resources track(const Program& p, std::span<const RootParam> layout, HwStage stage, const Structured* lanes)
{
    const Tracer t(p, layout, stage, lanes);
    Resources out;
    for (const auto& [reg, v] : p.inputs)
        if (const auto pr = t.param_of(reg)) out.params.push_back({v, pr->first, pr->second});
    // Group accesses in the same loop so a shared mask can select several descriptors together.
    std::map<std::size_t, std::pair<std::vector<std::size_t>, std::vector<Value>>> first_lane_at;
    // The console compiler's waterfall reads the first lane's table offset and loads that entry:
    // accesses, offset reads and loads by block.
    std::map<std::size_t, std::tuple<std::vector<std::size_t>, std::vector<Value>, std::vector<Site>>> offset_at;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        t.at = b;
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            const Inst& inst = block.insts[i];
            const Site site{b, i};
            auto fail = [&](const char* what) { return ShaderError::unsupported(inst.machine.offset, what); };
            std::vector<Value> first_lane;
            std::vector<std::pair<Value, Site>> offset;
            auto descriptor = [&](const Operand& op, Role role) {
                auto source = t.descriptor(inst, op, b);
                if (!source.ok()) throw fail(source.err);
                // A descriptor the code keeps per lane (lane_copies) is the same in every lane: its
                // loop is lowered as it stands and needs no proof.
                if (!t.kept_per_lane(inst, op)) {
                    if (const auto w = t.first_lane_words(inst, op)) first_lane.insert(first_lane.end(), w->begin(), w->end());
                }
                if (source.value.kind == Source::Kind::Loaded) {
                    const Site load = source.value.load;
                    if (std::any_of(out.scalar.begin(), out.scalar.end(), [&](const auto& s) { return s.first == load; })) {
                        if (const auto v = t.first_lane_offset(load)) offset.push_back({*v, load});
                    }
                }
                out.descriptors.push_back(Descriptor{site, role, std::move(source.value)});
            };
            const std::string_view name = inst.name();
            if (const auto* f = std::get_if<gfx10::Image>(&inst.fields)) {
                if (name.starts_with("image_bvh")) {
                    // The ray and node test names no texture. A descriptor whose flag dwords the code
                    // sets as numbers and whose type is a ray-tracing structure is a range of title
                    // memory (the title: all of it, or from one structure's address on), and the
                    // node's address is among the operands; any other descriptor must be a resource
                    // of the host.
                    auto word = [&](std::uint8_t k) -> std::optional<Term> {
                        if (!f->rsrc.is_scalar() || k >= f->rsrc.dwords) return std::nullopt;
                        const auto v = inst.read(Reg::S(static_cast<std::uint8_t>(f->rsrc.code + k)));
                        if (!v) return std::nullopt;
                        return t.origin(*v);
                    };
                    const auto w1 = word(1);
                    const auto w3 = word(3);
                    if (w1 && w1->is_const() && w3 && w3->is_const() && (w3->n >> 28) == bvh::DESCRIPTOR_TYPE) {
                        out.bvh.push_back({site, BvhNodes::Memory(w1->n, w3->n)});
                    } else {
                        descriptor(f->rsrc, Role::Resource);
                        out.bvh.push_back({site, BvhNodes::Resource()});
                    }
                } else {
                    descriptor(f->rsrc, Role::Resource);
                    if (f->sampler) descriptor(*f->sampler, Role::Sampler);
                }
            } else if (const auto* fb = std::get_if<gfx10::Buffer>(&inst.fields)) {
                // Cache invalidations name a buffer's address space but do not read its descriptor;
                // the host has no cache to drop.
                if (!name.ends_with("_inv")) descriptor(fb->rsrc, Role::Resource);
            } else if (const auto* fs = std::get_if<gfx10::Smem>(&inst.fields)) {
                if (name.starts_with("s_buffer_")) {
                    descriptor(fs->base, Role::Resource);
                } else if (name.find("_load_") != std::string_view::npos || name.find("_store_") != std::string_view::npos) {
                    auto address = t.smem_address(inst, fs->base, fs->offset, fs->soffset);
                    if (!address.ok()) address = t.by_address(address.err);
                    if (!address.ok()) throw fail(address.err);
                    out.scalar.push_back({site, address.value});
                }
            } else if (const auto* ff = std::get_if<gfx10::Flat>(&inst.fields)) {
                if (name.starts_with("global_")) {
                    auto address = t.flat_address(inst, ff->addr, ff->saddr, ff->offset);
                    if (!address.ok()) address = t.by_address(address.err);
                    if (!address.ok()) throw fail(address.err);
                    out.scalar.push_back({site, address.value});
                }
            }
            if (!first_lane.empty()) {
                auto& [accesses, reads] = first_lane_at[b];
                accesses.push_back(i);
                reads.insert(reads.end(), first_lane.begin(), first_lane.end());
            }
            if (!offset.empty()) {
                auto& [accesses, reads, loads] = offset_at[b];
                accesses.push_back(i);
                for (const auto& [v, load] : offset) {
                    reads.push_back(v);
                    loads.push_back(load);
                }
            }
        }
    }
    // What follows traces no access of one block.
    t.at = std::nullopt;
    // The proof needs the per-lane form. Without it nothing is emitted, and its failure is the one to
    // report, so the reads stay unproven here.
    if (!t.lanes) return out;
    auto sorted = [](std::vector<Value> reads) {
        std::sort(reads.begin(), reads.end());
        reads.erase(std::unique(reads.begin(), reads.end()), reads.end());
        return reads;
    };
    for (auto& [block, entry] : first_lane_at) {
        const auto& accesses = entry.first;
        const auto reads = sorted(entry.second);
        const auto proof = t.waterfall(block, accesses, reads);
        if (!proof.ok()) throw ShaderError::unsupported(t.inst({block, accesses[0]}).machine.offset, proof.err);
        out.waterfall.compares.insert(out.waterfall.compares.end(), proof.value.compares.begin(), proof.value.compares.end());
        out.waterfall.reads.insert(out.waterfall.reads.end(), proof.value.reads.begin(), proof.value.reads.end());
    }
    // Unproven, an offset read stays the first lane's (one uniform entry), which is what the hardware
    // reads outside a waterfall loop.
    for (auto& [block, entry] : offset_at) {
        const auto& accesses = std::get<0>(entry);
        const auto reads = sorted(std::get<1>(entry));
        const auto& loads = std::get<2>(entry);
        const auto proof = t.waterfall(block, accesses, reads);
        if (proof.ok()) {
            out.waterfall.compares.insert(out.waterfall.compares.end(), proof.value.compares.begin(),
                                          proof.value.compares.end());
            out.waterfall.reads.insert(out.waterfall.reads.end(), proof.value.reads.begin(), proof.value.reads.end());
            out.waterfall.loads.insert(out.waterfall.loads.end(), loads.begin(), loads.end());
        }
    }
    out.lane_copied = std::move(t.lane_copied);
    std::sort(out.lane_copied.begin(), out.lane_copied.end());
    out.lane_copied.erase(std::unique(out.lane_copied.begin(), out.lane_copied.end()), out.lane_copied.end());
    for (const Descriptor& d : out.descriptors)
        if (const auto v = t.root_indexed_view(d, out.scalar)) out.root_indexed_views.push_back(*v);
    return out;
}

}  // namespace x2w::vs::resource
