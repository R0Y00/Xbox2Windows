// Whole-wave sections: code between `s_or_saveexec -1` (or `s_nand_saveexec 0`) and the restore of the
// saved mask runs in every lane of the console's wave, the inactive ones included, which is how the
// compiler expands HLSL wave reductions and prefix sums (an identity put into the inactive lanes with
// `v_cndmask`, then DPP, `v_permlanex16` and `v_readlane` steps). A host lane that is elsewhere cannot run
// that code, so each section is simulated on the console's lanes with symbolic values, and a value it
// leaves is lowered by what it turned out to be: the reduction or prefix of one operation over the lanes'
// (masked) inputs, which DXIL's wave operations compute over the host's active lanes. No instruction
// sequence is matched; any step the simulation does not model leaves the values it writes unknown.

#include "x2w_shader/emit/wave.h"
#include "x2w_shader/resource.h"

#include "x2w_shader/emit/bitcode.h"

#include <algorithm>
#include <bit>
#include <utility>
#include <variant>
#include <vector>

namespace x2w::vs::wave {

using gfx10::DppLanes;
using gfx10::DppRow;
using gfx10::Fields;
using gfx10::NULL_CODE;
using gfx10::Operand;
using gfx10::Sop;
using gfx10::Valu;
using ir::EXEC;
using ir::Inst;
using ir::Program;
using ir::Reg;
using ir::Value;

namespace detail {

std::optional<std::uint32_t> inline_constant(std::uint8_t code) { return resource::inline_constant(code); }

}  // namespace detail

using detail::inline_constant;

std::optional<Op> op_of(std::string_view name) {
    // the low dword of a 64-bit sum (`v_add_co_ci_u32` carries into the high one) is the 32-bit
    // sum of the low dwords
    if (name == "v_add_nc_u32" || name == "v_add_co_u32") return Op::AddU;
    if (name == "v_add_f32") return Op::AddF;
    if (name == "v_add_f16") return Op::AddH;
    if (name == "v_min_u32") return Op::MinU;
    if (name == "v_max_u32") return Op::MaxU;
    if (name == "v_min_i32") return Op::MinI;
    if (name == "v_max_i32") return Op::MaxI;
    if (name == "v_min_f32") return Op::MinF;
    if (name == "v_max_f32") return Op::MaxF;
    if (name == "v_or_b32") return Op::Or;
    if (name == "v_and_b32") return Op::And;
    if (name == "v_xor_b32") return Op::Xor;
    return std::nullopt;
}

bool identity(Op op, std::uint32_t k) {
    switch (op) {
        case Op::AddU:
        case Op::AddH:
        case Op::Or:
        case Op::Xor:
        case Op::MaxU: return k == 0;
        case Op::AddF: return k == 0 || k == 0x8000'0000u;
        case Op::MinU:
        case Op::And: return k == 0xFFFF'FFFFu;
        case Op::MinI: return k == 0x7FFF'FFFFu;
        case Op::MaxI: return k == 0x8000'0000u;
        case Op::MinF: return k == 0x7F80'0000u;
        case Op::MaxF: return k == 0xFF80'0000u;
    }
    return false;
}

bool counts(Op op) { return op == Op::AddU || op == Op::AddF || op == Op::AddH || op == Op::Xor; }

// Whether the lane sets `a` and `b` are the two halves of one aligned run of lanes: lanes 2k and
// 2k+1, then two such pairs side by side, and so on up to the two halves of the wave. Sums joined
// only that way form the balanced tree over the lanes in their order, whichever of the two a lane
// adds to the other (the addition is commutative).
bool buddies(std::uint64_t a, std::uint64_t b) {
    const auto n = static_cast<std::uint32_t>(std::popcount(a));
    if (n != static_cast<std::uint32_t>(std::popcount(b)) || !std::has_single_bit(n) || n == 64) return false;
    auto run = [n](std::uint64_t s) -> std::optional<std::uint32_t> {
        const auto first = static_cast<std::uint32_t>(std::countr_zero(s));
        if (first % n == 0 && (s >> first) == (std::uint64_t{1} << n) - 1) return first;
        return std::nullopt;
    };
    const auto x = run(a);
    const auto y = run(b);
    if (!x || !y) return false;
    const std::uint32_t lo = std::min(*x, *y);
    const std::uint32_t diff = *x > *y ? *x - *y : *y - *x;
    return lo % (2 * n) == 0 && diff == n;
}

namespace {

std::uint8_t saved_code(const Inst& inst) {
    if (const auto* s = std::get_if<Sop>(&inst.fields); s && s->dst && s->dst->kind == Operand::Kind::Scalar) {
        return s->dst->code;
    }
    return NULL_CODE;
}

Sym combine(Op op, const Sym& a, const Sym& b) {
    // (Sym::K(k), x) | (x, Sym::K(k)) if op.identity(k) => x: the guard is tried per alternative.
    if (a.kind == Sym::K && identity(op, a.k)) return b;
    if (b.kind == Sym::K && identity(op, b.k)) return a;
    if (a.kind == Sym::Red && b.kind == Sym::Red && a.src == b.src && a.mask == b.mask && a.ident == b.ident &&
        identity(op, a.ident) && (!a.op || *a.op == op) && (!b.op || *b.op == op) &&
        (!counts(op) || (a.lanes & b.lanes) == 0) && (op != Op::AddH || buddies(a.lanes, b.lanes))) {
        return Sym::red(op, a.src, a.mask, a.ident, a.lanes | b.lanes);
    }
    return Sym::unknown();
}

// The lane a `v_readlane_b32` names.
std::optional<std::uint32_t> lane_number(const Operand& op) {
    switch (op.kind) {
        case Operand::Kind::Const: return inline_constant(op.code);
        case Operand::Kind::Literal: return op.literal;
        default: return std::nullopt;
    }
}

// Values of the VGPRs and scalars written by this section, indexed by register.
using VecMap = std::map<Value, std::vector<Sym>>;
using UniMap = std::map<Value, Sym>;

std::uint32_t reverse_bits(std::uint32_t x) {
    std::uint32_t r = 0;
    for (int i = 0; i < 32; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

std::pair<VecMap, UniMap> simulate(std::span<const Inst> insts, std::uint32_t wave, Value saved) {
    const std::uint64_t all = wave == 64 ? ~std::uint64_t{0} : std::uint64_t{0xFFFF'FFFFu};
    std::uint64_t exec = all;
    VecMap vec;
    UniMap uni;
    const std::size_t W = wave;
    // Masks the section derives from the saved EXEC (`s_and` with it).
    std::vector<Value> masked{saved};
    auto is_masked = [&masked](Value v) { return std::find(masked.begin(), masked.end(), v) != masked.end(); };
    for (const Inst& inst : insts) {
        const std::string_view name = inst.name();
        if (inst.write(Reg::S(EXEC))) {
            // Constant lane sets (prefix sums); anything else ends what the simulation can follow.
            std::optional<std::uint64_t> lo;
            if (name.starts_with("s_mov_b")) lo = detail::constant(inst);
            if (!lo) return {std::move(vec), std::move(uni)};
            const std::uint64_t k = *lo;
            exec = (name == "s_mov_b64" ? k : (exec & ~std::uint64_t{0xFFFF'FFFFu}) | (k & 0xFFFF'FFFFu)) & all;
            continue;
        }
        if (inst.write(Reg::S(static_cast<std::uint8_t>(EXEC + 1)))) {
            std::optional<std::uint64_t> hi;
            if (name.starts_with("s_mov_b")) hi = detail::constant(inst);
            if (!hi) return {std::move(vec), std::move(uni)};
            const std::uint64_t k = *hi;
            exec = ((exec & 0xFFFF'FFFFu) | ((k & 0xFFFF'FFFFu) << 32)) & all;
            continue;
        }
        if (const auto* sop = std::get_if<Sop>(&inst.fields)) {
            if (name.starts_with("s_and_b")) {
                bool from_saved = false;
                for (const Operand& o : sop->src) {
                    if (o.kind != Operand::Kind::Scalar) continue;
                    const auto v = inst.read(Reg::S(o.code));
                    if (v && is_masked(*v)) {
                        from_saved = true;
                        break;
                    }
                }
                if (from_saved) {
                    for (const auto& w : inst.writes) {
                        if (w.first.is_s()) {
                            masked.push_back(w.second);
                            break;
                        }
                    }
                }
            }
            if (name == "s_mov_b32" && sop->dst && sop->dst->kind == Operand::Kind::Scalar && !sop->src.empty() &&
                sop->dst->code != NULL_CODE) {
                if (const auto w = inst.write(Reg::S(sop->dst->code))) {
                    const Operand& s0 = sop->src[0];
                    Sym sym = Sym::unknown();
                    if (s0.kind == Operand::Kind::Scalar && s0.code != NULL_CODE) {
                        if (const auto x = inst.read(Reg::S(s0.code))) {
                            const auto it = uni.find(*x);
                            sym = it != uni.end() ? it->second : Sym::uni(*x);
                        }
                    } else if (s0.kind == Operand::Kind::Const) {
                        if (const auto c = inline_constant(s0.code)) sym = Sym::konst(*c);
                    } else if (s0.kind == Operand::Kind::Literal) {
                        sym = Sym::konst(s0.literal);
                    }
                    uni.insert_or_assign(*w, sym);
                }
            }
            for (const auto& w : inst.writes) uni.try_emplace(w.second, Sym::unknown());
            continue;
        }
        const auto* vp = std::get_if<Valu>(&inst.fields);
        if (!vp) {
            for (const auto& w : inst.writes) vec.insert_or_assign(w.second, std::vector<Sym>(W, Sym::unknown()));
            continue;
        }
        const Valu& v = *vp;
        auto lanes = [&](const Operand& op) -> std::vector<Sym> {
            auto one = [W](const Sym& s) { return std::vector<Sym>(W, s); };
            if (op.kind == Operand::Kind::Vector) {
                const auto x = inst.read(Reg::V(op.reg));
                if (!x) return one(Sym::unknown());
                const auto it = vec.find(*x);
                if (it != vec.end()) return it->second;
                return one(Sym::own(*x));
            }
            if (op.kind == Operand::Kind::Scalar && op.code != NULL_CODE) {
                const auto x = inst.read(Reg::S(op.code));
                if (!x) return one(Sym::unknown());
                const auto it = uni.find(*x);
                return one(it != uni.end() ? it->second : Sym::uni(*x));
            }
            if (op.kind == Operand::Kind::Const) {
                const auto c = inline_constant(op.code);
                return one(c ? Sym::konst(*c) : Sym::unknown());
            }
            if (op.kind == Operand::Kind::Literal) return one(Sym::konst(op.literal));
            return one(Sym::unknown());
        };
        const bool modifiers = v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 || v.sdwa.has_value();
        if (name == "v_readlane_b32") {
            if (!(v.dst && v.dst->kind == Operand::Kind::Scalar && v.src.size() >= 2)) continue;
            const std::uint8_t code = v.dst->code;
            const auto l = lane_number(v.src[1]);
            const std::vector<Sym> x = lanes(v.src[0]);
            if (const auto w = inst.write(Reg::S(code))) {
                Sym s = Sym::unknown();
                if (l && *l < x.size()) s = x[*l];
                uni.insert_or_assign(*w, s);
            }
            continue;
        }
        if (name == "v_add_f64" && !modifiers && !v.dpp && v.dst && v.dst->kind == Operand::Kind::Vector) {
            const std::uint8_t dreg = v.dst->reg;
            const std::uint8_t dreg1 = static_cast<std::uint8_t>(dreg + 1);
            const auto dst_lo = inst.write(Reg::V(dreg));
            const auto dst_hi = inst.write(Reg::V(dreg1));
            if (dst_lo && dst_hi) {
                using Pair = std::pair<std::vector<Sym>, std::vector<Sym>>;
                auto pair_lanes = [&](const Operand* op) -> std::optional<Pair> {
                    if (!op) return std::nullopt;
                    if (op->kind == Operand::Kind::Vector) {
                        auto first = lanes(Operand::Vector(op->reg, 1));
                        auto second = lanes(Operand::Vector(static_cast<std::uint8_t>(op->reg + 1), 1));
                        return Pair{std::move(first), std::move(second)};
                    }
                    if (op->kind == Operand::Kind::Scalar && op->code != NULL_CODE) {
                        auto first = lanes(Operand::Scalar(op->code, 1));
                        auto second = lanes(Operand::Scalar(static_cast<std::uint8_t>(op->code + 1), 1));
                        return Pair{std::move(first), std::move(second)};
                    }
                    return std::nullopt;
                };
                const std::vector<Sym> old_lo = lanes(Operand::Vector(dreg, 1));
                const std::vector<Sym> old_hi = lanes(Operand::Vector(dreg1, 1));
                struct Term {
                    Value lo, hi, mask;
                    std::uint64_t lanes;
                };
                auto f64_term = [](const Sym& lo_s, const Sym& hi_s) -> std::optional<Term> {
                    if (lo_s.kind == Sym::Red && !lo_s.op && lo_s.ident == 0 && hi_s.kind == Sym::Red && !hi_s.op &&
                        hi_s.ident == 0 && lo_s.mask == hi_s.mask && lo_s.lanes == hi_s.lanes) {
                        return Term{lo_s.src, hi_s.src, lo_s.mask, lo_s.lanes};
                    }
                    if (lo_s.kind == Sym::Red64F && !lo_s.high && hi_s.kind == Sym::Red64F && hi_s.high &&
                        lo_s.src == hi_s.src && lo_s.hi == hi_s.hi && lo_s.mask == hi_s.mask && lo_s.lanes == hi_s.lanes) {
                        return Term{lo_s.src, lo_s.hi, lo_s.mask, lo_s.lanes};
                    }
                    return std::nullopt;
                };
                std::vector<Sym> out_lo(W, Sym::unknown());
                std::vector<Sym> out_hi(W, Sym::unknown());
                const auto a = pair_lanes(v.src.size() > 0 ? &v.src[0] : nullptr);
                const auto b = pair_lanes(v.src.size() > 1 ? &v.src[1] : nullptr);
                if (a && b) {
                    for (std::size_t l = 0; l < W; ++l) {
                        Sym rl = Sym::unknown();
                        Sym rh = Sym::unknown();
                        const auto ta = f64_term(a->first[l], a->second[l]);
                        const auto tb = f64_term(b->first[l], b->second[l]);
                        if (ta && tb && ta->lo == tb->lo && ta->hi == tb->hi && ta->mask == tb->mask &&
                            (ta->lanes & tb->lanes) == 0) {
                            rl = Sym::red64f(ta->lo, ta->hi, ta->mask, ta->lanes | tb->lanes, false);
                            rh = Sym::red64f(ta->lo, ta->hi, ta->mask, ta->lanes | tb->lanes, true);
                        }
                        const bool on = ((exec >> l) & 1) != 0;
                        out_lo[l] = on ? rl : old_lo[l];
                        out_hi[l] = on ? rh : old_hi[l];
                    }
                }
                vec.insert_or_assign(*dst_lo, std::move(out_lo));
                vec.insert_or_assign(*dst_hi, std::move(out_hi));
                continue;
            }
        }
        if (!(v.dst && v.dst->kind == Operand::Kind::Vector)) {
            for (const auto& w : inst.writes) uni.insert_or_assign(w.second, Sym::unknown());
            continue;
        }
        const std::uint8_t dreg = v.dst->reg;
        const auto dst_opt = inst.write(Reg::V(dreg));
        if (!dst_opt) continue;
        const Value dst = *dst_opt;
        const std::vector<Sym> old = lanes(Operand::Vector(dreg, 1));
        // src0 as each lane fetches it (DPP), with the lanes that write.
        std::optional<std::vector<Sym>> src0;
        if (!v.src.empty()) src0 = lanes(v.src[0]);
        std::uint64_t writes = exec;
        std::optional<std::vector<Sym>> fetched;
        if (v.dpp && src0) {
            if (exec != all) {
                vec.insert_or_assign(dst, std::vector<Sym>(W, Sym::unknown()));
                continue;
            }
            const std::vector<Sym>& s = *src0;
            std::vector<Sym> f;
            f.reserve(W);
            bool ok = true;
            for (std::uint32_t l = 0; l < wave; ++l) {
                const auto d = detail::dpp_lane(*v.dpp, l);
                if (d && std::get<0>(*d) < wave) {
                    const auto [src, zero, on] = *d;
                    f.push_back(zero ? Sym::konst(0) : s[src]);
                    if (!on) writes &= ~(std::uint64_t{1} << l);
                } else {
                    ok = false;
                }
            }
            if (ok) fetched = std::move(f);
        } else if (!v.dpp && src0) {
            fetched = *src0;
        }
        auto src = [&](std::size_t k) -> std::optional<std::vector<Sym>> {
            if (k < v.src.size()) return lanes(v.src[k]);
            return std::nullopt;
        };
        std::vector<Sym> result;
        if (modifiers || !fetched) {
            result.assign(W, Sym::unknown());
        } else if (name == "v_mov_b32") {
            result = std::move(*fetched);
        } else if (name == "v_bfrev_b32") {
            result.reserve(W);
            for (const Sym& s : *fetched) {
                result.push_back(s.kind == Sym::K ? Sym::konst(reverse_bits(s.k)) : Sym::unknown());
            }
        } else if (name == "v_cndmask_b32") {
            const std::vector<Sym>& a = *fetched;
            std::optional<Value> mask;
            if (v.src.size() > 2 && v.src[2].kind == Operand::Kind::Scalar) mask = inst.read(Reg::S(v.src[2].code));
            const std::vector<Sym> b = src(1).value_or(std::vector<Sym>{});
            result.reserve(W);
            for (std::size_t l = 0; l < W; ++l) {
                if (a[l].kind == Sym::K && l < b.size() && b[l].kind == Sym::Own && mask && is_masked(*mask)) {
                    result.push_back(Sym::red(std::nullopt, b[l].src, *mask, a[l].k, std::uint64_t{1} << l));
                } else {
                    result.push_back(Sym::unknown());
                }
            }
        } else if (name == "v_permlanex16_b32") {
            const std::vector<Sym>& a = *fetched;
            // Lanes of each row read the other row of their half; lane i's pick is nibble i of
            // {src2, src1}.
            auto sel = [&](std::size_t k) -> std::optional<std::uint32_t> {
                if (k >= v.src.size()) return std::nullopt;
                const Operand& o = v.src[k];
                if (o.kind == Operand::Kind::Const) return inline_constant(o.code);
                if (o.kind == Operand::Kind::Literal) return o.literal;
                return std::nullopt;
            };
            const auto s1 = sel(1);
            const auto s2 = sel(2);
            if (s1 && s2) {
                const std::uint64_t s = std::uint64_t{*s1} | (std::uint64_t{*s2} << 32);
                result.reserve(W);
                for (std::uint32_t l = 0; l < wave; ++l) {
                    const std::uint32_t other = (l & ~31u) | ((l & 16) ^ 16);
                    const auto pick = static_cast<std::uint32_t>((s >> (4 * (l & 15))) & 15);
                    result.push_back(a.at(other | pick));
                }
            } else {
                result.assign(W, Sym::unknown());
            }
        } else if (name == "v_sub_nc_u32") {
            const std::vector<Sym>& a = *fetched;
            // A prefix sum less the lane's own term: the exclusive one.
            const std::vector<Sym> b = src(1).value_or(std::vector<Sym>{});
            auto addu = [](const Sym& s) {
                return s.kind == Sym::Red && (!s.op || *s.op == Op::AddU) && s.ident == 0;
            };
            result.reserve(W);
            for (std::size_t l = 0; l < W; ++l) {
                if (addu(a[l]) && l < b.size() && addu(b[l]) && a[l].src == b[l].src && a[l].mask == b[l].mask &&
                    (a[l].lanes & b[l].lanes) == b[l].lanes) {
                    const std::uint64_t rest = a[l].lanes & ~b[l].lanes;
                    result.push_back(rest == 0 ? Sym::konst(0) : Sym::red(Op::AddU, a[l].src, a[l].mask, 0, rest));
                } else {
                    result.push_back(Sym::unknown());
                }
            }
        } else {
            const std::vector<Sym>& a = *fetched;
            const auto op = op_of(name);
            const auto b = src(1);
            if (op && b) {
                result.reserve(W);
                for (std::size_t l = 0; l < W; ++l) result.push_back(combine(*op, a.at(l), b->at(l)));
            } else {
                result.assign(W, Sym::unknown());
            }
        }
        std::vector<Sym> merged;
        merged.reserve(W);
        for (std::size_t l = 0; l < W; ++l) merged.push_back(((writes >> l) & 1) != 0 ? result.at(l) : old.at(l));
        vec.insert_or_assign(dst, std::move(merged));
    }
    return {std::move(vec), std::move(uni)};
}

// The shape of a scalar.
std::optional<Shape> single(const Sym& s, std::uint32_t wave) {
    const std::uint64_t all = ~std::uint64_t{0} >> (64 - wave);
    switch (s.kind) {
        case Sym::K: return Shape::konst(s.k);
        case Sym::Own:
        case Sym::Uni: return Shape::value(s.src);
        case Sym::Red:
            if (s.op && s.lanes == all) return Shape::active(*s.op, s.src, s.mask, s.ident);
            return std::nullopt;
        case Sym::Red64F:
            if (s.lanes == all) return Shape::active_f64(s.src, s.hi, s.mask, s.high);
            return std::nullopt;
        default: return std::nullopt;
    }
}

// The shape of each value, where every lane agrees on one.
std::map<Value, Shape> shapes(const VecMap& vec, const UniMap& uni, std::uint32_t wave) {
    const std::uint64_t all = ~std::uint64_t{0} >> (64 - wave);
    std::map<Value, Shape> out;
    for (const auto& [v, s] : uni) {
        if (const auto sh = single(s, wave)) out.insert_or_assign(v, *sh);
    }
    for (const auto& [v, lanes] : vec) {
        if (const auto sh = detail::per_lane(lanes, all)) out.insert_or_assign(v, *sh);
    }
    return out;
}

}  // namespace

namespace detail {

std::optional<std::tuple<std::uint32_t, bool, bool>> dpp_lane(const gfx10::Dpp& d, std::uint32_t l) {
    if (const auto* lanes = std::get_if<DppLanes>(&d)) {
        return std::tuple{(l & ~7u) | ((lanes->sel >> (3 * (l & 7))) & 7), false, true};
    }
    const auto& r = std::get<DppRow>(d);
    const std::uint32_t row = l & ~15u;
    const std::uint32_t pos = l & 15;
    const std::uint32_t n = std::uint32_t(r.ctrl & 15);
    const bool enabled = ((std::uint32_t(r.row_mask) >> ((l >> 4) & 3)) & 1) != 0 &&
                         ((std::uint32_t(r.bank_mask) >> ((l >> 2) & 3)) & 1) != 0;
    std::uint32_t src = 0;
    bool inside = true;
    const std::uint16_t ctrl = r.ctrl;
    if (ctrl <= 0xFF) {
        src = (l & ~3u) | ((std::uint32_t(ctrl) >> (2 * (l & 3))) & 3);
    } else if (ctrl >= 0x101 && ctrl <= 0x10F) {
        src = l + n;
        inside = pos + n < 16;
    } else if (ctrl >= 0x111 && ctrl <= 0x11F) {
        src = l - n;  // wrapping_sub
        inside = pos >= n;
    } else if (ctrl >= 0x121 && ctrl <= 0x12F) {
        src = row | ((pos - n) & 15);
    } else if (ctrl == 0x140) {
        src = l ^ 15;
    } else if (ctrl == 0x141) {
        src = l ^ 7;
    } else if (ctrl >= 0x150 && ctrl <= 0x15F) {
        src = row | n;
    } else if (ctrl >= 0x160 && ctrl <= 0x16F) {
        src = l ^ n;
    } else {
        return std::nullopt;
    }
    if (inside) return std::tuple{src, false, enabled};
    if (r.bound_ctrl) return std::tuple{l, true, enabled};
    return std::tuple{l, false, false};
}

std::optional<std::uint64_t> constant(const Inst& inst) {
    const auto* s = std::get_if<Sop>(&inst.fields);
    if (!s || s->src.empty()) return std::nullopt;
    const bool wide = inst.name() == "s_mov_b64";
    const Operand& o = s->src[0];
    if (o.kind == Operand::Kind::Scalar && o.code == NULL_CODE) return 0;
    if (o.kind == Operand::Kind::Const && o.code >= 128 && o.code <= 208) {
        const auto x = inline_constant(o.code);
        if (!x) return std::nullopt;
        if (wide) return static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(*x)));
        return std::uint64_t{*x};
    }
    if (o.kind == Operand::Kind::Const && !wide) {
        const auto x = inline_constant(o.code);
        if (!x) return std::nullopt;
        return std::uint64_t{*x};
    }
    if (o.kind == Operand::Kind::Literal) return std::uint64_t{o.literal};
    return std::nullopt;
}

std::optional<Shape> per_lane(std::span<const Sym> lanes, std::uint64_t all) {
    if (lanes.empty()) bitcode::bug("wave::per_lane: no lanes");
    const Sym first = lanes[0];
    struct Own {
        Value src, mask;
        std::uint32_t ident;
        bool operator==(const Own&) const = default;
    };
    auto own = [](std::size_t l, const Sym& s) -> std::optional<Own> {
        if (s.kind == Sym::Red && !s.op && s.lanes == (std::uint64_t{1} << l)) return Own{s.src, s.mask, s.ident};
        return std::nullopt;
    };
    if (const auto m = own(0, first)) {
        bool every = true;
        for (std::size_t l = 0; l < lanes.size(); ++l) {
            if (own(l, lanes[l]) != m) {
                every = false;
                break;
            }
        }
        if (every) return Shape::masked(m->src, m->mask, m->ident);
    }
    if (std::all_of(lanes.begin(), lanes.end(), [&](const Sym& s) { return s == first; })) {
        switch (first.kind) {
            case Sym::K: return Shape::konst(first.k);
            case Sym::Uni:
            case Sym::Own: return Shape::value(first.src);
            case Sym::Red:
                if (first.op && first.lanes == all) return Shape::active(*first.op, first.src, first.mask, first.ident);
                return std::nullopt;
            case Sym::Red64F:
                if (first.lanes == all) return Shape::active_f64(first.src, first.hi, first.mask, first.high);
                return std::nullopt;
            default: return std::nullopt;
        }
    }
    // Prefixes: lane l holds lanes 0..l (exclusive) or 0..=l (inclusive).
    struct Meta {
        Value src, mask;
        std::uint32_t ident;
        bool inclusive;
        bool operator==(const Meta&) const = default;
    };
    std::optional<Meta> meta;
    std::optional<Op> op_found;
    for (std::size_t l = 0; l < lanes.size(); ++l) {
        const Sym& s = lanes[l];
        const std::uint64_t bit = std::uint64_t{1} << l;
        const std::uint64_t below = bit - 1;
        if (s.kind != Sym::Red) {
            // Lane 0 of an exclusive prefix holds the identity.
            if (s.kind == Sym::K && l == 0) continue;
            return std::nullopt;
        }
        const std::uint64_t set = s.lanes;
        const bool inclusive = set == (below | bit);
        if (!inclusive && set != below) return std::nullopt;
        if (l == 0 && !inclusive) return std::nullopt;
        if (!s.op && std::popcount(set) != 1) return std::nullopt;
        const Meta lane_meta{s.src, s.mask, s.ident, inclusive};
        if (!meta) {
            meta = lane_meta;
        } else if (!(*meta == lane_meta)) {
            return std::nullopt;
        }
        if (s.op) {
            if (!op_found) {
                op_found = *s.op;
            } else if (*op_found != *s.op) {
                return std::nullopt;
            }
        }
    }
    if (!meta || !op_found) return std::nullopt;
    const Op op = *op_found;
    if (!identity(op, meta->ident)) return std::nullopt;
    const Sym& l0 = lanes[0];
    const bool ok = (l0.kind == Sym::K && !meta->inclusive && l0.k == meta->ident) || (l0.kind == Sym::Red && meta->inclusive);
    if (!ok) return std::nullopt;
    return Shape::prefix(op, meta->src, meta->mask, meta->ident, meta->inclusive);
}

}  // namespace detail

std::unordered_map<std::size_t, Section> sections(const Program& p, std::size_t block) {
    const std::vector<Inst>& insts = p.blocks.at(block).insts;
    const std::uint32_t wave = p.wave64 ? 64 : 32;
    std::unordered_map<std::size_t, Section> out;
    std::size_t i = 0;
    while (i < insts.size()) {
        const Inst& inst = insts[i];
        const std::string_view name = inst.name();
        // EXEC becomes every lane: `s_or_saveexec -1` or `s_nand_saveexec 0`.
        std::optional<Operand> all;
        if (name == "s_or_saveexec_b64" || name == "s_or_saveexec_b32") {
            all = Operand::Const(193);
        } else if (name == "s_nand_saveexec_b64" || name == "s_nand_saveexec_b32") {
            all = Operand::Const(128);
        }
        bool start = false;
        if (all) {
            if (const auto* s = std::get_if<Sop>(&inst.fields)) start = !s->src.empty() && s->src[0] == *all;
        }
        std::optional<Value> saved;
        if (start) {
            if (const auto* s = std::get_if<Sop>(&inst.fields); s && s->dst && s->dst->kind == Operand::Kind::Scalar) {
                saved = inst.write(Reg::S(s->dst->code));
            }
        }
        if (!saved) {
            ++i;
            continue;
        }
        std::optional<std::size_t> end;
        for (std::size_t j = i + 1; j < insts.size(); ++j) {
            if (insts[j].write(Reg::S(EXEC)) && insts[j].read(Reg::S(saved_code(inst))) == saved) {
                end = j;
                break;
            }
        }
        if (!end) break;
        auto [vec, uni] = simulate(std::span<const Inst>(insts).subspan(i + 1, *end - (i + 1)), wave, *saved);
        std::map<Value, Shape> reads;
        for (std::size_t j = *end + 1; j < insts.size(); ++j) {
            const Inst& later = insts[j];
            if (later.name() != "v_readlane_b32") continue;
            const auto* v = std::get_if<Valu>(&later.fields);
            if (!v || !v->dst || v->dst->kind != Operand::Kind::Scalar || v->src.size() != 2 ||
                v->src[0].kind != Operand::Kind::Vector) {
                continue;
            }
            const std::uint8_t code = v->dst->code;
            const auto x = later.read(Reg::V(v->src[0].reg));
            if (!x) continue;
            const auto left = vec.find(*x);
            if (left == vec.end()) continue;
            const auto w = later.write(Reg::S(code));
            if (!w) continue;
            const auto l = lane_number(v->src[1]);
            if (!l || *l >= left->second.size()) continue;
            const Sym sym = left->second[*l];
            // (a lane's own value stays a lane read)
            if (sym.kind == Sym::Own) continue;
            const auto shape = single(sym, wave);
            if (!shape) continue;
            reads.insert_or_assign(*w, *shape);
        }
        auto sh = shapes(vec, uni, wave);
        out.insert_or_assign(i, Section{*end, std::move(sh), std::move(reads)});
        i = *end + 1;
    }
    return out;
}

// `WaveActiveBitOr(v7)` into s7 as the console compiler expands it: every lane takes part, those
// outside the saved EXEC (s[4:5]) with 0.
const std::array<std::uint32_t, 20> WAVE_OR_V7_TO_S7 = {
    0xBE84'25C1,  // s_or_saveexec_b64 s[4:5], -1
    0xD501'0001,
    0x0012'0E80,  // v_cndmask_b32 v1, 0, v7, s[4:5]
    0x3802'02E9,
    0xDE54'C101,  // v_or_b32 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
    0x3802'02E9,
    0x9360'1201,  // v_or_b32 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
    0x3802'02E9,
    0x0009'2401,  // v_or_b32 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
    0x3802'02FA,
    0xFF09'4001,  // v_or_b32 v1, v1, v1 row_mirror bound_ctrl:1
    0xD778'0002,
    0x0201'0101,  // v_permlanex16_b32 v2, v1, 0, 0
    0x3802'0302,  // v_or_b32 v1, v2, v1
    0xD760'0007,
    0x0001'4101,  // v_readlane_b32 s7, v1, 32
    0x3802'0207,  // v_or_b32 v1, s7, v1
    0xD760'0007,
    0x0001'0101,  // v_readlane_b32 s7, v1, 0
    0xBEFE'0404,  // s_mov_b64 exec, s[4:5]
};

}  // namespace x2w::vs::wave
