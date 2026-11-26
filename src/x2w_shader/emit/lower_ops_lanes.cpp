// Lowering helpers for operands, floats, lane booleans, wave exchange, and results.
#include <cstdlib>
#include <stdexcept>

#include "x2w_shader/emit/container.h"  // MESH_OUTPUT_LIMIT, mesh_output_bytes
#include "x2w_shader/emit/lower_impl.h"
#include "x2w_shader/resource.h"  // inline_constant

namespace x2w::vs::lower {

namespace {

[[noreturn]] void unreachable() { throw std::logic_error("unreachable"); }

}  // namespace

V Lower::operand(const Inst& inst, const Operand& op) {
    switch (op.kind) {
        case Operand::Kind::Scalar: {
            if (op.code == gfx10::NULL_CODE) return this->s.m.i32(0);
            auto v = ok_or(inst.read(Reg::S(op.code)), "an unread operand");
            return this->i32_of(v);
        }
        case Operand::Kind::Vector: {
            auto v = ok_or(inst.read(Reg::V(op.reg)), "an unread operand");
            return this->i32_of(v);
        }
        case Operand::Kind::Const: {
            // src_scc reads SCC as zero or one.
            if (op.code == 253) {
                V bit = this->scc_in(inst);
                return this->s.m.cast(this->cur, Cast::ZExt, bit, this->s.t.i32);
            }
            // Nothing to wait for (see `ordered_section`).
            if (op.code == POPS_EXITING && !this->ordered.empty()) return this->s.m.i32(0);
            std::optional<std::uint32_t> x = resource::inline_constant(op.code);
            if (!x) fail("a special operand (vccz, execz, scc, lds_direct)");
            return this->s.m.i32(*x);
        }
        case Operand::Kind::Literal:
            return this->s.m.i32(op.literal);
        case Operand::Kind::Attr:
            fail("an attribute operand");
    }
    unreachable();
}

// A GFX10 inline f16 constant occupies the low half of the operand.
V Lower::f16_operand(const Inst& inst, const Operand& op) {
    if (op.kind == Operand::Kind::Const && op.code >= 240 && op.code <= 248) {
        return this->s.m.i32(inline_f16(op.code));
    }
    return this->operand(inst, op);
}

V Lower::as_float(V x) { return this->s.m.cast(this->cur, Cast::Bitcast, x, this->s.t.f32); }

V Lower::as_int(V f) { return this->s.m.cast(this->cur, Cast::Bitcast, f, this->s.t.i32); }

// The SDWA operand selection of source `k`: a byte or word of the dword, zero- or sign-extended
// (none: the whole dword).
std::optional<std::pair<std::uint8_t, bool>> Lower::sdwa_src(const Inst& inst, std::size_t k) {
    const auto* v = std::get_if<gfx10::Valu>(&inst.fields);
    if (!v) return std::nullopt;
    if (!v->sdwa) return std::nullopt;
    const auto& s = *v->sdwa;
    if (k >= s.src_sel.size()) return std::nullopt;
    std::uint8_t sel = s.src_sel[k];
    bool sext = ((s.sext >> k) & 1) != 0;
    if (sel < 6) return std::pair<std::uint8_t, bool>{sel, sext};
    return std::nullopt;
}

// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
V Lower::select_bits(V x, std::uint8_t sel, bool sext) {
    std::size_t c = this->cur;
    std::uint32_t width, at;
    if (sel < 4) {
        width = 8;
        at = 8u * sel;
    } else {
        width = 16;
        at = 16u * (sel - 4u);
    }
    V left = this->s.m.i32(32 - width - at);
    V t = this->s.m.bin(c, Bin::Shl, x, left);
    V right = this->s.m.i32(32 - width);
    Bin op = sext ? Bin::AShr : Bin::LShr;
    return this->s.m.bin(c, op, t, right);
}

// A float source with its abs/neg modifiers.
V Lower::fsrc(const Inst& inst, std::size_t k) {
    const auto* vp = std::get_if<gfx10::Valu>(&inst.fields);
    if (!vp) unreachable();
    const auto& v = *vp;
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    auto selected = Lower::sdwa_src(inst, k);
    V f;
    if (selected) {
        auto [sel, sext] = *selected;
        V x = this->operand(inst, op);
        x = this->select_bits(x, sel, sext);
        f = this->as_float(x);
    } else if (k == 0 && this->src0) {
        V x = *this->src0;
        f = this->as_float(x);
    } else if ((op.kind == Operand::Kind::Vector || op.kind == Operand::Kind::Scalar) &&
               !(op.kind == Operand::Kind::Scalar && op.code == gfx10::NULL_CODE)) {
        Reg r = op.kind == Operand::Kind::Vector ? Reg::V(op.reg) : Reg::S(op.code);
        Value val = ok_or(inst.read(r), "an unread operand");
        auto it = this->floats.find(val);
        if (it != this->floats.end()) {
            f = it->second;
        } else {
            V x = this->i32_of(val);
            f = this->as_float(x);
        }
    } else {
        V x = this->operand(inst, op);
        f = this->as_float(x);
    }
    if (((v.abs >> k) & 1) != 0) {
        f = this->unary(6, f);
    }
    if (((v.neg >> k) & 1) != 0) {
        V z = this->s.m.f32(0x8000'0000u);
        f = this->s.m.bin(this->cur, Bin::Sub, z, f);
    }
    return f;
}

V Lower::src(const Inst& inst, std::size_t k) {
    const auto* vp = std::get_if<gfx10::Valu>(&inst.fields);
    if (!vp) unreachable();
    const auto& v = *vp;
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    if (((v.abs >> k) & 1) != 0 || ((v.neg >> k) & 1) != 0) {
        fail("integer operand modifiers");
    }
    if (this->src0 && k == 0) return *this->src0;
    V x = this->operand(inst, op);
    if (auto s = Lower::sdwa_src(inst, k)) return this->select_bits(x, s->first, s->second);
    return x;
}

V Lower::unary(std::uint32_t op, V f) {
    return this->s.op(this->cur, "unary.f32", Attr::ReadNone, this->s.t.f32, op, std::vector<V>{f});
}

V Lower::binary(std::uint32_t op, V a, V b) {
    return this->s.op(this->cur, "binary.f32", Attr::ReadNone, this->s.t.f32, op,
                      std::vector<V>{a, b});
}

// The hardware's maximum (`max`) or minimum of two floats: as DXIL's FMax/FMin, but -0 orders
// below +0 (RDNA2 ISA, `v_max_f32`), where the compare and select DXIL specifies gives either zero.
V Lower::max_min(bool max, V a, V b) {
    std::size_t c = this->cur;
    V r = this->binary(max ? 35 : 36, a, b);
    if ((host_rdna() & rdna::MAX_MIN) != 0) {
        return r;
    }
    V ia = this->as_int(a);
    V ib = this->as_int(b);
    V either = this->s.m.bin(c, Bin::Or, ia, ib);
    V magnitude = this->s.m.i32(0x7FFF'FFFFu);
    either = this->s.m.bin(c, Bin::And, either, magnitude);
    V zero = this->s.m.i32(0);
    V zeros = this->s.m.cmp(c, pred::EQ, either, zero);
    // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
    Bin op = max ? Bin::And : Bin::Or;
    V sign = this->s.m.bin(c, op, ia, ib);
    V signed_ = this->as_float(sign);
    return this->s.m.select(c, zeros, signed_, r);
}

V Lower::fmad(V a, V b, V c) {
    return this->s.op(this->cur, "tertiary.f32", Attr::ReadNone, this->s.t.f32, 46,
                      std::vector<V>{a, b, c});
}

// `f` as an f16 in the low bits, rounded toward zero: the nearest-even conversion, one step back
// toward zero where it rounded away (which also turns an overflow to infinity into the largest
// finite half).
V Lower::f16_toward_zero(V f) {
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    V h = this->s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f});
    V back = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, this->s.t.f32, 131, std::vector<V>{h});
    V fa = this->unary(6, f);
    V ba = this->unary(6, back);
    // Ordered: never for NaN.
    V away = this->s.m.cmp(c, pred::OGT, ba, fa);
    V one = this->s.m.i32(1);
    V stepped = this->s.m.bin(c, Bin::Sub, h, one);
    return this->s.m.select(c, away, stepped, h);
}

// `f` as an f16 in the low bits, rounded to nearest, ties to even, as the console's 16-bit float
// operations round, whatever the host's own conversion does (D3D lets `f32tof16` round toward
// zero, and the NVIDIA driver's does). The conversion toward zero, or the next half away from zero
// where that one is nearer, or as near and the first one odd. Past the largest half the next one
// counts as 2^16, so 65520 and up become infinity. (`f` is an f32 result of two halves here;
// rounding it again to 16 bits gives the rounding of the exact result.)
V Lower::f16_nearest(V f) {
    std::size_t c = this->cur;
    Ty f32t = this->s.t.f32;
    V lo = this->f16_toward_zero(f);
    V one = this->s.m.i32(1);
    V zero = this->s.m.i32(0);
    V hi = this->s.m.bin(c, Bin::Add, lo, one);
    auto size = [&](Lower& l, V h) {
        V w = l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{h});
        return l.unary(6, w);
    };
    V below = size(*this, lo);
    V above = size(*this, hi);
    V magnitude = this->s.m.i32(0x7FFF);
    V infinity = this->s.m.i32(0x7C00);
    V m = this->s.m.bin(c, Bin::And, hi, magnitude);
    V top = this->s.m.cmp(c, pred::EQ, m, infinity);
    V bound = this->s.m.f32(f32_bits(65536.0f));
    above = this->s.m.select(c, top, bound, above);
    V fa = this->unary(6, f);
    // (both differences are exact: the three values lie within one half's step)
    V under = this->s.m.bin(c, Bin::Sub, fa, below);
    V over = this->s.m.bin(c, Bin::Sub, above, fa);
    // Ordered: never for NaN, which stays what the conversion made of it.
    V nearer = this->s.m.cmp(c, pred::OLT, over, under);
    V tie = this->s.m.cmp(c, pred::OEQ, over, under);
    V odd = this->s.m.bin(c, Bin::And, lo, one);
    odd = this->s.m.cmp(c, pred::NE, odd, zero);
    tie = this->s.m.bin(c, Bin::And, tie, odd);
    V up = this->s.m.bin(c, Bin::Or, nearer, tie);
    return this->s.m.select(c, up, hi, lo);
}

// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's top bit when `signed`;
// widths from 32 on take every bit. A signed field that runs past bit 31 extends from bit 31, as
// the hardware does: the shift is arithmetic then.
V Lower::bfe(V a, V off, V width, bool signed_) {
    std::size_t c = this->cur;
    V k31 = this->s.m.i32(31);
    V k32 = this->s.m.i32(32);
    V zero = this->s.m.i32(0);
    V one = this->s.m.i32(1);
    V ones = this->s.m.i32(std::numeric_limits<std::uint32_t>::max());
    off = this->s.m.bin(c, Bin::And, off, k31);
    Bin shift = signed_ ? Bin::AShr : Bin::LShr;
    V shifted = this->s.m.bin(c, shift, a, off);
    V full = this->s.m.cmp(c, pred::IUGE, width, k32);
    V w = this->s.m.bin(c, Bin::And, width, k31);
    V mask = this->s.m.bin(c, Bin::Shl, one, w);
    mask = this->s.m.bin(c, Bin::Sub, mask, one);
    mask = this->s.m.select(c, full, ones, mask);
    V r = this->s.m.bin(c, Bin::And, shifted, mask);
    if (!signed_) {
        return r;
    }
    // Sign-extend a field narrower than 32 bits (none for width 0).
    V left = this->s.m.bin(c, Bin::Sub, k32, w);
    left = this->s.m.bin(c, Bin::And, left, k31);
    V t = this->s.m.bin(c, Bin::Shl, r, left);
    t = this->s.m.bin(c, Bin::AShr, t, left);
    V is0 = this->s.m.cmp(c, pred::EQ, w, zero);
    V keep = this->s.m.bin(c, Bin::Or, full, is0);
    return this->s.m.select(c, keep, r, t);
}

// Whether an f32 value is NaN, tested in integer arithmetic so host drivers (such as NVIDIA) do
// not fold `fcmp uno` away under fast-math.
V Lower::is_nan_f32(V f) {
    std::size_t c = this->cur;
    V bits = this->as_int(f);
    V mask = this->s.m.i32(0x7FFF'FFFFu);
    V abs = this->s.m.bin(c, Bin::And, bits, mask);
    V inf = this->s.m.i32(0x7F80'0000u);
    return this->s.m.cmp(c, pred::IUGT, abs, inf);
}

// The median of three. With a NaN among them the hardware gives the minimum of the three instead
// (the host's minimum skips NaNs, so that is the smallest number among them).
V Lower::med3(V a, V b, V third) {
    std::size_t c = this->cur;
    // Between two constants in order: the lower one for a NaN either way.
    if ((host_rdna() & rdna::MED3) != 0) {
        std::optional<float> k0 = this->s.m.float_k(a);
        std::optional<float> k1 = this->s.m.float_k(b);
        std::optional<float> k2 = this->s.m.float_k(third);
        struct Ordered {
            V v, p, q;
            float x, y;
        };
        std::optional<Ordered> ordered;
        if (!k0 && k1 && k2) {
            ordered = Ordered{a, b, third, *k1, *k2};
        } else if (k0 && !k1 && k2) {
            ordered = Ordered{b, a, third, *k0, *k2};
        } else if (k0 && k1 && !k2) {
            ordered = Ordered{third, a, b, *k0, *k1};
        }
        if (ordered && !std::isnan(ordered->x) && !std::isnan(ordered->y) &&
            ordered->x != ordered->y) {
            V low = ordered->x < ordered->y ? ordered->p : ordered->q;
            V high = ordered->x < ordered->y ? ordered->q : ordered->p;
            V t = this->binary(35, ordered->v, low);
            return this->binary(36, t, high);
        }
    }
    V lo = this->binary(36, a, b);
    V hi = this->binary(35, a, b);
    V t = this->binary(36, hi, third);
    V median = this->binary(35, lo, t);
    V least = this->binary(36, lo, third);
    V nan_a = this->is_nan_f32(a);
    V nan_b = this->is_nan_f32(b);
    V nan_3 = this->is_nan_f32(third);
    V nan_ab = this->s.m.bin(c, Bin::Or, nan_a, nan_b);
    V nan = this->s.m.bin(c, Bin::Or, nan_ab, nan_3);
    return this->s.m.select(c, nan, least, median);
}

// `a - floor(a)`, which on the hardware never reaches 1.0: where it would round to it (a tiny
// negative `a`), the result is `below_one`, the largest value under 1.0 in the operation's width.
// Hosts differ here (some give 1.0). Comparing against a `threshold` distinct from `below_one`
// prevents drivers (observed on NVIDIA) from folding `r > k ? k : r` into `minNum` (`FMNMX`) and
// swallowing NaN.
V Lower::fract(V a, float threshold, float below_one) {
    std::size_t c = this->cur;
    V r = this->unary(22, a);
    if ((host_rdna() & rdna::FRACT) != 0 && below_one == f32_from_bits(0x3F7F'FFFFu)) {
        return r;
    }
    V limit = this->s.m.f32(f32_bits(threshold));
    V k = this->s.m.f32(f32_bits(below_one));
    V over = this->s.m.cmp(c, pred::OGE, r, limit);
    return this->s.m.select(c, over, k, r);
}

// Sine, or cosine, of an angle in turns, as the hardware takes it. The whole turns are taken off
// first (`x - round_even(x)`, exact, within half a turn), so a large angle keeps its fraction;
// scaling to radians first rounds it away. A zero passes unreduced (the sine of -0 is -0). The sine
// of a half turn and the cosine of a quarter turn are exactly +0 on the hardware, where the host's
// function of the rounded angle gives about 1e-7. Infinities and NaN give NaN.
// shortcut: below 2^-12 turns the hardware scales by a slightly different constant than 2π (a
// last-place difference).
V Lower::trig_turns(V x, bool cosine) {
    std::size_t c = this->cur;
    V zero = this->s.m.f32(0);
    V whole = this->unary(26, x);
    V part = this->s.m.bin(c, Bin::Sub, x, whole);
    V is_zero = this->s.m.cmp(c, pred::OEQ, x, zero);
    part = this->s.m.select(c, is_zero, x, part);
    V k = this->s.m.f32(f32_bits(TWO_PI));
    V angle = this->s.m.bin(c, Bin::Mul, part, k);
    V r = this->unary(cosine ? 12 : 13, angle);
    float at_f = cosine ? 0.25f : 0.5f;
    V at = this->s.m.f32(f32_bits(at_f));
    V size = this->unary(6, part);
    V exact = this->s.m.cmp(c, pred::OEQ, size, at);
    return this->s.m.select(c, exact, zero, r);
}

// The high dword of a 32 x 32-bit product.
V Lower::mul_hi(V a, V b, bool signed_) {
    std::size_t c = this->cur;
    Ty i64t = this->s.m.int_(64);
    Cast ext = signed_ ? Cast::SExt : Cast::ZExt;
    a = this->s.m.cast(c, ext, a, i64t);
    b = this->s.m.cast(c, ext, b, i64t);
    V m = this->s.m.bin(c, Bin::Mul, a, b);
    V k32 = this->s.m.int_const(i64t, 32);
    V hi = this->s.m.bin(c, Bin::LShr, m, k32);
    this->s.flags |= flags::INT64;
    return this->s.m.cast(c, Cast::Trunc, hi, this->s.t.i32);
}

// `0 x anything = 0`, as in D3D9.
V Lower::legacy_mul(V a, V b) {
    std::size_t c = this->cur;
    if ((host_rdna() & rdna::LEGACY) != 0) {
        V m = this->s.m.bin(c, Bin::Mul, a, b);
        V zero = this->s.m.f32(0);
        // A product is not 0 only where one factor is 0 and the other an infinity or no number:
        // never of a value with itself, nor of two finite ones, and only where the finite one of
        // two is 0.
        bool fa = this->s.m.bounded(a, 12).has_value();
        bool fb = this->s.m.bounded(b, 12).has_value();
        auto not_zero = [](const Lower& l, V x) {
            auto k = l.s.m.float_k(x);
            return k.has_value() && *k != 0.0f && std::isfinite(*k);
        };
        if (a == b || (fa && fb) || not_zero(*this, a) || not_zero(*this, b)) {
            return m;
        }
        V asked;
        if (fa) {
            asked = a;
        } else if (fb) {
            asked = b;
        } else {
            V aa = this->unary(6, a);
            V bb = this->unary(6, b);
            asked = this->binary(36, aa, bb);
        }
        V z = this->s.m.cmp(c, pred::OEQ, asked, zero);
        return this->s.m.select(c, z, zero, m);
    }
    V m = this->s.m.bin(c, Bin::Mul, a, b);
    V zero = this->s.m.f32(0);
    V az = this->s.m.cmp(c, pred::OEQ, a, zero);
    V bz = this->s.m.cmp(c, pred::OEQ, b, zero);
    V z = this->s.m.bin(c, Bin::Or, az, bz);
    return this->s.m.select(c, z, zero, m);
}

// ---- lane booleans ----

V Lower::boolean(Bool b, std::size_t at) {
    if (b == Bool::True || b == Bool::False) {
        return this->s.m.bool_(b == Bool::True);
    }
    auto found = this->bools.find(std::pair<std::size_t, Bool>{at, b});
    if (found != this->bools.end()) {
        return found->second;
    }
    auto node = this->x.st->bools.node(b);
    if (!node) throw std::logic_error("not a constant");
    auto [atom, lo, hi] = *node;
    V a = this->atom(atom, at);
    V v;
    if (lo == Bool::False && hi == Bool::True) {
        v = a;
    } else if (lo == Bool::True && hi == Bool::False) {
        V t = this->s.m.bool_(true);
        v = this->s.m.bin(at, Bin::Xor, a, t);
    } else if (lo == Bool::False) {
        V h = this->boolean(hi, at);
        v = this->s.m.bin(at, Bin::And, a, h);
    } else if (hi == Bool::True) {
        V l = this->boolean(lo, at);
        v = this->s.m.bin(at, Bin::Or, a, l);
    } else {
        V l = this->boolean(lo, at);
        V h = this->boolean(hi, at);
        v = this->s.m.select(at, a, h, l);
    }
    this->bools.insert_or_assign(std::pair<std::size_t, Bool>{at, b}, v);
    return v;
}

V Lower::atom(Atom atom, std::size_t at) {
    switch (atom.kind) {
        case Atom::Kind::Lane: {
            auto it = this->lane_bits.find(atom.value);
            if (it != this->lane_bits.end()) return it->second;
            auto f = this->lane_fail.find(atom.value);
            fail(f != this->lane_fail.end() ? f->second : "a lane mask from a carry");
        }
        case Atom::Kind::Uniform: {
            Val g = this->get(atom.value);
            switch (g.kind) {
                case Val::Kind::Bit:
                    return g.v;
                case Val::Kind::I32: {
                    V z = this->s.m.i32(0);
                    return this->s.m.cmp(at, pred::NE, g.v, z);
                }
                case Val::Kind::Missing:
                    fail(g.why);
            }
            unreachable();
        }
        case Atom::Kind::Phi: {
            Value id = atom.value;
            auto mp = this->mask_phis.find(id);
            if (mp != this->mask_phis.end()) {
                return mp->second;
            }
            auto pa = this->phi_at.find(id);
            if (pa == this->phi_at.end()) fail("a lane-mask join without a phi");
            auto [b, i] = pa->second;
            std::size_t blk = ok_or(this->block.at(b), "an unreachable join");
            V v = this->s.m.phi(blk, this->s.t.i1);
            this->mask_phis.insert_or_assign(id, v);
            this->pending_masks.emplace_back(v, b, i);
            return v;
        }
        // shortcut: "some lane of the quad is set" is taken as true: the lanes it adds only feed
        // derivatives, and exports keep their own masks. Wrong if such a section had a side effect.
        case Atom::Kind::Quad:
            return this->s.m.bool_(true);
        // With console lanes, a mask held as a number is the console's: the lane tests its own bit.
        case Atom::Kind::Opaque: {
            if (!this->console_lanes) fail("an opaque lane mask");
            Value v = atom.value;
            std::optional<Val> hi;
            if (auto h = this->pair_high(v)) {
                hi = this->get(*h);
            }
            Val lo = this->get(v);
            auto word = [](const Val& w) -> V {
                if (w.kind == Val::Kind::I32) return w.v;
                fail("an opaque lane mask");
            };
            V lo_w = word(lo);
            std::optional<V> hi_w;
            if (hi) hi_w = word(*hi);
            return this->lane_bit(at, lo_w, hi_w);
        }
        case Atom::Kind::Constant: {
            if (!this->console_lanes) fail("a constant lane mask");
            std::uint64_t k = atom.constant;
            V lo = this->s.m.i32(static_cast<std::uint32_t>(k));
            V hi = this->s.m.i32(static_cast<std::uint32_t>(k >> 32));
            return this->lane_bit(at, lo, hi);
        }
    }
    unreachable();
}

// A console wave64 program in host waves of 32: the first console thread of this host wave (0, 32, 64,...).
// A GPU need not give a host wave consecutive thread indices (NVIDIA's 8 x 8 groups), so each wave takes
// the next 32 from an LDS counter instead: every thread zeroes it, the group meets, each wave's first lane
// adds 32 and the wave takes what it found. Branch-free, in the entry block, which every thread runs.
std::optional<V> Lower::wave_base() {
    if (!this->halves) return std::nullopt;
    std::uint32_t slot = *this->halves;
    if (this->wave_base_) {
        return this->wave_base_;
    }
    std::size_t e = this->s.entry;
    Ty i32t = this->s.t.i32;
    V lds = this->s.lds();
    V k = this->s.m.i32(slot);
    V ptr = this->s.m.element(e, lds, k);
    V zero = this->s.m.i32(0);
    this->s.m.store(e, ptr, zero);
    this->group_sync(e);
    V lane = this->s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, std::vector<V>{});
    V first = this->s.m.cmp(e, pred::EQ, lane, zero);
    V k32 = this->s.m.i32(32);
    V add = this->s.m.select(e, first, k32, zero);
    ptr = this->s.m.element(e, lds, k);
    V old = this->s.m.atomic(e, Rmw::Add, ptr, add);
    V base = this->s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, std::vector<V>{old});
    this->s.flags |= flags::WAVE_OPS;
    this->wave_base_ = base;
    return base;
}

// The console thread index of this thread in the group: `wave_base` plus the host lane (host waves
// of 32), else the wave's first thread index rounded down plus the lane.
V Lower::console_thread() {
    std::size_t e = this->s.entry;
    Ty i32t = this->s.t.i32;
    V lane = this->s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, std::vector<V>{});
    if (auto base = this->wave_base()) {
        return this->s.m.bin(e, Bin::Add, *base, lane);
    }
    V flat = this->s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, std::vector<V>{});
    V first = this->s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, std::vector<V>{flat});
    std::uint32_t wave = this->s.wave_size.value_or(32);
    V mask = this->s.m.i32(~(wave - 1));
    V base = this->s.m.bin(e, Bin::And, first, mask);
    return this->s.m.bin(e, Bin::Add, base, lane);
}

// This thread's console lane (0..63): its relabelled thread index (`thread_id`), whose host wave
// holds a 32-aligned run of console lanes. Without `exchange`: the host lane.
V Lower::console_lane() {
    if (!this->exchange) {
        return this->lane_index();
    }
    if (this->clane) {
        return *this->clane;
    }
    std::size_t e = this->s.entry;
    V f = this->console_thread();
    V k63 = this->s.m.i32(63);
    V cl = this->s.m.bin(e, Bin::And, f, k63);
    this->s.flags |= flags::WAVE_OPS;
    this->clane = cl;
    return cl;
}

// A group barrier with an LDS fence in `block`.
void Lower::group_sync(std::size_t block) {
    V mode = this->s.m.i32(meeting_mode());
    Ty void_t = this->s.t.void_;
    this->s.op(block, "barrier", Attr::NoDuplicate, void_t, 80, std::vector<V>{mode});
}

// `exchange`: each host wave gives `mine` (the same in all its lanes) for its half; both get
// (lower half's, upper half's). A group with no upper half gets 0 for it.
std::pair<V, V> Lower::exchange_halves(std::size_t at, V mine) {
    if (!this->exchange) {
        V z = this->s.m.i32(0);
        return {mine, z};
    }
    std::uint32_t slot = std::get<0>(*this->exchange);
    bool small = std::get<1>(*this->exchange);
    if (small) {
        V z = this->s.m.i32(0);
        return {mine, z};
    }
    if (this->local) {
        // each host wave on its own: its half holds `mine`, the other nothing
        V cl = this->console_lane();
        V k32 = this->s.m.i32(32);
        V z = this->s.m.i32(0);
        V upper = this->s.m.cmp(at, pred::IUGE, cl, k32);
        V lo = this->s.m.select(at, upper, z, mine);
        V hi = this->s.m.select(at, upper, mine, z);
        return {lo, hi};
    }
    // the pair of dwords of this thread's console wave: its half's, then both
    V pair = this->exchange_pair(slot);
    V cl = this->console_lane();
    V five = this->s.m.i32(5);
    V half = this->s.m.bin(at, Bin::LShr, cl, five);
    V index = this->s.m.bin(at, Bin::Add, pair, half);
    V lds = this->s.lds();
    V ptr = this->s.m.element(at, lds, index);
    this->s.m.store(at, ptr, mine);
    this->group_sync(at);
    V one = this->s.m.i32(1);
    V k1 = this->s.m.bin(at, Bin::Add, pair, one);
    V p0 = this->s.m.element(at, lds, pair);
    V p1 = this->s.m.element(at, lds, k1);
    V lo = this->s.m.load(at, p0);
    V hi = this->s.m.load(at, p1);
    // the slots are written again only after every thread read them
    this->group_sync(at);
    return {lo, hi};
}

// The first of the two exchange dwords of this thread's console wave (made in the entry block): a
// group of several console waves has a pair for each.
V Lower::exchange_pair(std::uint32_t slot) {
    if (this->pair) {
        return *this->pair;
    }
    std::size_t e = this->s.entry;
    V thread = this->console_thread();
    V six = this->s.m.i32(6);
    V wave = this->s.m.bin(e, Bin::LShr, thread, six);
    V one = this->s.m.i32(1);
    V twice = this->s.m.bin(e, Bin::Shl, wave, one);
    V base = this->s.m.i32(slot);
    V v = this->s.m.bin(e, Bin::Add, base, twice);
    this->pair = v;
    return v;
}

// The console's decision of the branch closing IR block `b`, which the wave takes together
// (Structured::wide): see wave_wide. SCC of a mask operation as data (Structured::scc_some):
// whether some lane of the console wave has its bit of the result, asked where the operation
// stands. The lanes that are elsewhere have no bit (the result is ANDed with EXEC or is EXEC), so
// with the console's waves the host wave's answer is the console's; two host waves of 32 exchange
// theirs, which needs every lane there.
V Lower::scc_some(Value scc) {
    auto found = this->x.st->scc_some.find(scc);
    if (found == this->x.st->scc_some.end()) fail(LANE_MASK);
    Bool m = found->second;
    // (not for a console wave cut into host waves that do not exchange)
    if (!this->console_lanes || this->local || (!this->exchange && this->halves)) {
        fail(LANE_MASK);
    }
    std::optional<std::size_t> made;
    if (auto d = this->def_block.find(scc); d != this->def_block.end()) made = d->second;
    std::size_t at;
    if (!made) {
        at = this->s.entry;
    } else if (*made == this->cur_ir) {
        at = this->cur;
    } else {
        at = ok_or(this->tail.at(*made), "a lane mask written after its use");
    }
    V bit = this->boolean(m, at);
    Ty i1t = this->s.t.i1;
    Ty i32t = this->s.t.i32;
    V some = this->s.op(at, "waveAnyTrue", Attr::NoUnwind, i1t, 113, std::vector<V>{bit});
    this->s.flags |= flags::WAVE_OPS;
    if (!(this->exchange && !std::get<1>(*this->exchange))) {
        return some;
    }
    if (!(!made || this->all_lanes(*made))) {
        need(made.value_or(0));
        fail("a lane mask exchanged where not every lane runs");
    }
    some = this->s.m.cast(at, Cast::ZExt, some, i32t);
    auto [lo, hi] = this->exchange_halves(at, some);
    V both = this->s.m.bin(at, Bin::Or, lo, hi);
    V z = this->s.m.i32(0);
    return this->s.m.cmp(at, pred::NE, both, z);
}

V Lower::wave_wide(std::size_t b, Bool when) {
    static constexpr Why WHY = "a wave-wide branch where not every lane runs";
    const auto& wa = this->x.st->wave_any;
    std::optional<bool> any_o = b < wa.size() ? wa[b] : std::nullopt;
    bool any = ok_or(any_o, "a wave-wide branch on a scalar condition");
    if (!this->exchange && !this->halves && this->console_lanes &&
        this->mask_data == MaskData::Console) {
        // The host's waves are the console's: the wave's own answer. The lanes that are elsewhere
        // have no bit set, as on the console, so "some lane" needs no one but the lanes here, and
        // "every lane's bit is clear" is true of the others anyway.
        std::size_t c = this->cur;
        V mine = this->boolean(when, c);
        Ty i1t = this->s.t.i1;
        this->s.flags |= flags::WAVE_OPS;
        const char* name = any ? "waveAnyTrue" : "waveAllTrue";
        std::uint32_t code = any ? 113 : 114;
        return this->s.op(c, name, Attr::NoUnwind, i1t, code, std::vector<V>{mine});
    }
    if (this->local || !(this->exchange && !std::get<1>(*this->exchange))) {
        fail(WHY);
    }
    if (!this->all_lanes(b)) {
        need(b);
        fail(WHY);
    }
    std::size_t c = this->cur;
    V mine = this->boolean(when, c);
    // what the halves exchange is "some lane speaks against" either way (for `any`: some lane's
    // `when` holds; else: some lane's does not), so a half with no lane says nothing
    V set;
    if (any) {
        set = mine;
    } else {
        V t = this->s.m.bool_(true);
        set = this->s.m.bin(c, Bin::Xor, mine, t);
    }
    Ty i1t = this->s.t.i1;
    Ty i32t = this->s.t.i32;
    V some = this->s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, std::vector<V>{set});
    this->s.flags |= flags::WAVE_OPS;
    some = this->s.m.cast(c, Cast::ZExt, some, i32t);
    auto [lo, hi] = this->exchange_halves(c, some);
    c = this->cur;
    V both = this->s.m.bin(c, Bin::Or, lo, hi);
    V z = this->s.m.i32(0);
    return this->s.m.cmp(c, any ? pred::NE : pred::EQ, both, z);
}

// Whether every lane runs IR block `b` (always true without `exchange`).
bool Lower::all_lanes(std::size_t b) const {
    if (!this->exchange) return true;
    const auto& all = std::get<2>(*this->exchange);
    return b < all.size() ? bool(all[b]) : false;
}

// This lane's bit of the mask `lo`, `hi` (wave64).
V Lower::lane_bit(std::size_t at, V lo, std::optional<V> hi) {
    this->s.flags |= flags::WAVE_OPS;
    V lane;
    if (this->exchange) {
        lane = this->console_lane();
    } else {
        lane = this->s.op(at, "waveGetLaneIndex", Attr::ReadOnly, this->s.t.i32, 111, std::vector<V>{});
    }
    V k31 = this->s.m.i32(31);
    V k32 = this->s.m.i32(32);
    V one = this->s.m.i32(1);
    V z = this->s.m.i32(0);
    V word;
    if (hi) {
        V low = this->s.m.cmp(at, pred::IULT, lane, k32);
        word = this->s.m.select(at, low, lo, *hi);
    } else {
        word = lo;
    }
    V sh = this->s.m.bin(at, Bin::And, lane, k31);
    V t = this->s.m.bin(at, Bin::LShr, word, sh);
    t = this->s.m.bin(at, Bin::And, t, one);
    return this->s.m.cmp(at, pred::NE, t, z);
}

// The value of the high dword of the wave64 register pair whose low dword holds `lo` (none in
// wave32).
std::optional<Value> Lower::pair_high(Value lo) const {
    if (!this->x.native->wave64) {
        return std::nullopt;
    }
    const Program& p = *this->x.p;
    auto pick = [](Reg reg) -> std::optional<Reg> {
        if (reg.is_s()) return Reg::S(static_cast<std::uint8_t>(reg.n + 1));
        return std::nullopt;
    };
    for (const auto& in : p.inputs) {
        if (in.second == lo) {
            auto h = pick(in.first);
            if (!h) return std::nullopt;
            for (const auto& j : p.inputs) {
                if (j.first == *h) return j.second;
            }
            return std::nullopt;
        }
    }
    if (auto pa = this->phi_at.find(lo); pa != this->phi_at.end()) {
        auto [b, i] = pa->second;
        auto h = pick(p.blocks[b].phis[i].reg);
        if (!h) return std::nullopt;
        for (const auto& q : p.blocks[b].phis) {
            if (q.reg == *h) return q.value;
        }
        return std::nullopt;
    }
    auto d = this->def_block.find(lo);
    if (d == this->def_block.end()) return std::nullopt;
    std::size_t b = d->second;
    const Inst* inst = nullptr;
    for (const auto& i : p.blocks[b].insts) {
        bool any = false;
        for (const auto& w : i.writes) {
            if (w.second == lo) {
                any = true;
                break;
            }
        }
        if (any) {
            inst = &i;
            break;
        }
    }
    if (!inst) return std::nullopt;
    const std::pair<Reg, Value>* wr = nullptr;
    for (const auto& w : inst->writes) {
        if (w.second == lo) {
            wr = &w;
            break;
        }
    }
    if (!wr) return std::nullopt;
    auto h = pick(wr->first);
    if (!h) return std::nullopt;
    return inst->write(*h);
}

std::optional<V> Lower::guard(std::size_t b, std::size_t i) {
    std::optional<Bool> g = this->x.st->guards[b][i];
    if (!g) return std::nullopt;
    return this->boolean(*g, this->cur);
}

// ---- results ----

void Lower::set(std::optional<Value> v, Val val) {
    if (v) {
        this->vals.insert_or_assign(*v, val);
    }
}

// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
void Lower::write(const Inst& inst, Reg reg, V x, std::optional<V> guard) {
    if (auto d = Lower::sdwa_dst(inst); d && reg.is_v()) {
        x = this->place_bits(inst, reg, x, d->first, d->second);
    }
    this->write_placed(inst, reg, x, guard);
}

// `write` of a dword already placed (SDWA applied).
void Lower::write_placed(const Inst& inst, Reg reg, V x, std::optional<V> guard) {
    auto new_ = inst.write(reg);
    if (!new_) return;
    if (guard && reg.is_v()) {
        V g = *guard;
        std::optional<V> old;
        if (auto o = inst.read(reg)) {
            old = ok([&] { return this->i32_of(*o); });
        }
        // The validator refuses reads of undef.
        if (!old) old = this->s.m.i32(0);
        V v = this->s.m.select(this->cur, g, x, *old);
        this->s.m.tag(v, Tag::Guard);
        x = v;
    }
    this->vals.insert_or_assign(*new_, Val::I32(x));
}

// The SDWA destination selection: a byte or word, and what the other bits get (0 zeros, 1 the sign
// above the field, 2 the old bits).
std::optional<std::pair<std::uint8_t, std::uint8_t>> Lower::sdwa_dst(const Inst& inst) {
    const auto* v = std::get_if<gfx10::Valu>(&inst.fields);
    if (!v) return std::nullopt;
    if (!v->sdwa) return std::nullopt;
    const auto& s = *v->sdwa;
    if (s.dst_sel < 6) return std::pair<std::uint8_t, std::uint8_t>{s.dst_sel, s.dst_unused};
    return std::nullopt;
}

// The low byte or word of `x` placed at `sel` in VGPR `reg`.
V Lower::place_bits(const Inst& inst, Reg reg, V x, std::uint8_t sel, std::uint8_t unused) {
    std::size_t c = this->cur;
    std::uint32_t width, at;
    if (sel < 4) {
        width = 8;
        at = 8u * sel;
    } else {
        width = 16;
        at = 16u * (sel - 4u);
    }
    V left = this->s.m.i32(32 - width);
    V t = this->s.m.bin(c, Bin::Shl, x, left);
    V right = this->s.m.i32(32 - width - at);
    if (unused == 1) {
        // Bits below the field shift in as zeros.
        return this->s.m.bin(c, Bin::AShr, t, right);
    }
    V field = this->s.m.bin(c, Bin::LShr, t, right);
    if (unused != 2) {
        return field;
    }
    std::optional<V> old;
    if (auto o = inst.read(reg)) {
        old = ok([&] { return this->i32_of(*o); });
    }
    if (!old) old = this->s.m.i32(0);
    std::uint32_t bits = static_cast<std::uint32_t>((std::uint64_t{1} << width) - 1);
    V keep = this->s.m.i32(~(bits << at));
    V kept = this->s.m.bin(c, Bin::And, *old, keep);
    return this->s.m.bin(c, Bin::Or, kept, field);
}

void Lower::write_f(const Inst& inst, V f, std::optional<V> guard) {
    const auto* vp = std::get_if<gfx10::Valu>(&inst.fields);
    if (!vp) unreachable();
    const auto& v = *vp;
    if (v.omod != 0) {
        static constexpr float FACTORS[4] = {1.0f, 2.0f, 4.0f, 0.5f};
        float factor = FACTORS[v.omod];  // omod is 2 bits
        V c = this->s.m.f32(f32_bits(factor));
        f = this->s.m.bin(this->cur, Bin::Mul, f, c);
    }
    if (v.clamp) {
        f = this->unary(7, f);
    }
    if (!(v.dst && v.dst->kind == Operand::Kind::Vector)) {
        fail("a float result outside a VGPR");
    }
    std::uint8_t reg = v.dst->reg;
    V x = this->as_int(f);
    this->write(inst, Reg::V(reg), x, guard);
    if (!guard && !Lower::sdwa_dst(inst)) {
        if (auto new_ = inst.write(Reg::V(reg))) {
            this->floats.insert_or_assign(*new_, f);
        }
    }
}

void Lower::write_i(const Inst& inst, V x, std::optional<V> guard) {
    const auto* vp = std::get_if<gfx10::Valu>(&inst.fields);
    if (!vp) unreachable();
    const auto& v = *vp;
    if (v.clamp || v.omod != 0) {
        fail("integer output modifiers");
    }
    if (v.dst && v.dst->kind == Operand::Kind::Vector) {
        this->write(inst, Reg::V(v.dst->reg), x, guard);
    } else if (v.dst && v.dst->kind == Operand::Kind::Scalar) {
        this->write(inst, Reg::S(v.dst->code), x, std::nullopt);
    } else {
        fail("a result without a register");
    }
}

// Starts code only the lanes where `guard` holds run (a store under a lane guard): returns where
// it started and the join block.
std::optional<std::pair<std::size_t, std::size_t>> Lower::begin_guarded(std::optional<V> guard) {
    if (!guard) return std::nullopt;
    V g = *guard;
    std::size_t then = this->s.m.new_block();
    std::size_t join = this->s.m.new_block();
    this->s.m.terminate(this->cur, Term::cond_br(g, then, join));
    std::size_t head = this->cur;
    this->cur = then;
    return std::pair<std::size_t, std::size_t>{head, join};
}

// Ends guarded code; each (value inside, value outside) becomes a phi.
std::vector<V> Lower::end_guarded(std::optional<std::pair<std::size_t, std::size_t>> at,
                                  std::span<const std::pair<V, V>> values) {
    std::vector<V> out;
    if (!at) {
        for (const auto& v : values) out.push_back(v.first);
        return out;
    }
    auto [head, join] = *at;
    std::size_t then = this->cur;
    this->s.m.terminate(then, Term::br(join));
    this->cur = join;
    for (const auto& [inside, outside] : values) {
        Ty ty = this->s.m.type_of(inside);
        V phi = this->s.m.phi(join, ty);
        this->s.m.set_phi(phi, std::vector<std::pair<V, std::size_t>>{{inside, then}, {outside, head}});
        out.push_back(phi);
    }
    return out;
}

// Code that depends on the lane layout: wave operations, and in compute the console's wave size
// (lanes are thread ids in group order there). Pixel lanes match only within a quad.
void Lower::cross_lane(bool quad) {
    if (this->s.mesh) {
        // shortcut: no wave size for mesh shaders in SM 6.6, so host lanes cannot follow the
        // console's; emulate through group-shared memory if mesh programs turn out to need these.
        fail("a cross-lane operation in a mesh shader");
    }
    if (!this->s.threads && !(quad && this->s.pixel)) {
        fail("a cross-lane operation outside compute");
    }
    // debug_assert!(threads.is_none() || console_lanes)
    this->s.flags |= flags::WAVE_OPS;
}

// Component `c` of the thread id in the group. With console lanes, the thread is relabelled so
// that its host lane is its console lane: the flat id is the wave's first lane's, rounded down to
// the wave size, plus the lane index. (A bijection while each host wave holds a wave-aligned run
// of flat ids, as the console's waves do.)
Val Lower::thread_id(std::uint8_t c) {
    if (!this->console_lanes) {
        return this->compute_id("threadIdInGroup.i32", 95, c);
    }
    if (!this->s.threads) throw std::logic_error("compute");
    std::uint32_t x = (*this->s.threads)[0];
    std::uint32_t y = (*this->s.threads)[1];
    std::size_t e = this->s.entry;
    Ty i32t = this->s.t.i32;
    auto split = [&](V f) {
        V xk = this->s.m.i32(x);
        V yk = this->s.m.i32(y);
        V xy = this->s.m.i32(x * y);
        if (c == 0) return this->s.m.bin(e, Bin::URem, f, xk);
        if (c == 1) {
            V t = this->s.m.bin(e, Bin::UDiv, f, xk);
            return this->s.m.bin(e, Bin::URem, t, yk);
        }
        return this->s.m.bin(e, Bin::UDiv, f, xy);
    };
    if (this->halves) {
        V f = this->console_thread();
        this->s.flags |= flags::WAVE_OPS;
        return Val::I32(split(f));
    }
    V flat = this->s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, std::vector<V>{});
    V first = this->s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, std::vector<V>{flat});
    if (!this->s.wave_size) throw std::logic_error("set with console lanes");
    std::uint32_t wave = *this->s.wave_size;
    V mask = this->s.m.i32(~(wave - 1));
    V base = this->s.m.bin(e, Bin::And, first, mask);
    V lane = this->lane_index();
    V f = this->s.m.bin(e, Bin::Add, base, lane);
    this->s.flags |= flags::WAVE_OPS;
    return Val::I32(split(f));
}

// Mesh: the counts and the primitives no lane exports. The console's counts come from one lane's
// allocation request, which DXIL cannot express (one call, dominating every output write).
// shortcut: the counts are the declared maxima and each lane first culls its primitive (all
// indices 0, zero area), so primitives past the program's count are culled and vertices past it
// unused; this costs the host the unused vertex and primitive slots. Pass the program's own counts
// if that shows in frame time.
void Lower::mesh_prologue(Mesh mesh) {
    if (!this->s.threads) throw std::logic_error("mesh");
    std::uint32_t threads = (*this->s.threads)[0];
    if (mesh.topology != 2) {
        fail("a mesh shader not drawing triangles");
    }
    if (threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256) {
        fail("a mesh shader beyond the host's limits");
    }
    // D3D12 caps the outputs of the declared maxima at 32 KiB. Lane n writes only vertex n and
    // primitive n, so maxima past the lanes are never reached: they come down to the lane count
    // when the outputs do not fit.
    const auto& outputs = this->s.outputs;
    const auto& prims = this->s.prims;
    auto fits = [&](const Mesh& m) {
        return container::mesh_output_bytes(outputs, prims, m.max_vertices, m.max_primitives) <=
               container::MESH_OUTPUT_LIMIT;
    };
    if (!fits(mesh)) {
        mesh.max_vertices = std::min(mesh.max_vertices, threads);
        mesh.max_primitives = std::min(mesh.max_primitives, threads);
        // shortcut: outputs past the cap even at the lane count are still emitted with the lower
        // maxima (drivers take them); refuse them here if one hangs.
        // Experiment (`X2W_SHADERCFG_MESH_OUTPUT_STRICT=1`): they are refused.
        if (!fits(mesh) && std::getenv("X2W_SHADERCFG_MESH_OUTPUT_STRICT") != nullptr) {
            fail("a mesh shader whose outputs pass the host's 32 KiB");
        }
        this->s.mesh = mesh;
    }
    std::size_t e = this->s.entry;
    Ty void_t = this->s.t.void_;
    V v = this->s.m.i32(mesh.max_vertices);
    V p = this->s.m.i32(mesh.max_primitives);
    this->s.op(e, "setMeshOutputCounts", Attr::NoUnwind, void_t, 168, std::vector<V>{v, p});
    V flat = this->mesh_thread();
    V z = this->s.m.i32(0);
    // Every primitive of the declared count starts culled, also those past the lanes (a count of
    // 256 with 128 lanes): lane n culls primitives n, n + lanes, ... The host draws all it was
    // promised, and one never written has whatever indices its memory held.
    std::uint32_t passes = div_ceil(mesh.max_primitives, std::max(threads, 1u));
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
        V index;
        if (pass == 0) {
            index = flat;
        } else {
            V past = this->s.m.i32(pass * threads);
            index = this->s.m.bin(this->cur, Bin::Add, flat, past);
        }
        std::optional<V> guard;
        if ((pass + 1) * threads > mesh.max_primitives) {
            guard = this->s.m.cmp(this->cur, pred::IULT, index, p);
        }
        auto at = this->begin_guarded(guard);
        this->s.op(this->cur, "emitIndices", Attr::NoUnwind, void_t, 169, std::vector<V>{index, z, z, z});
        this->end_guarded(at, std::span<const std::pair<V, V>>{});
    }
}

// Mesh: the thread's index in the group, which is its vertex and primitive index.
V Lower::mesh_thread() {
    if (this->flat_) {
        return *this->flat_;
    }
    std::size_t e = this->s.entry;
    V f = this->s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, this->s.t.i32, 96,
                     std::vector<V>{});
    this->flat_ = f;
    return f;
}

// The console wave size.
std::uint32_t Lower::wave() const { return this->x.native->wave64 ? 64 : 32; }

V Lower::lane_index() {
    if (this->s.mesh) {
        // The console's lane: mesh code reads no other lanes, so host waves need not match the
        // console's.
        V f = this->mesh_thread();
        V m = this->s.m.i32(this->wave() - 1);
        return this->s.m.bin(this->s.entry, Bin::And, f, m);
    }
    return this->s.op(this->cur, "waveGetLaneIndex", Attr::ReadOnly, this->s.t.i32, 111,
                      std::vector<V>{});
}

// `x` of lane `src` (in this lane's quad) through quad reads, whose lane must be a constant.
V Lower::quad_pick(V x, V src) {
    this->s.flags |= flags::WAVE_OPS;
    std::size_t c = this->cur;
    V three = this->s.m.i32(3);
    V q = this->s.m.bin(c, Bin::And, src, three);
    V r = x;
    for (std::uint32_t n = 0; n < 4; ++n) {
        V kn = this->s.m.i32(n);
        V read = this->s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, this->s.t.i32, 122,
                            std::vector<V>{x, kn});
        V is = this->s.m.cmp(c, pred::EQ, q, kn);
        r = this->s.m.select(c, is, read, r);
    }
    return r;
}

// Whether a quad-local read takes its lane with the host's quad operations: in a pixel program
// always. Lanes of a quad the primitive does not cover, or that were discarded, are helpers on the
// host; wave reads leave those out (AMD gives 0 from them), quad reads are what reaches them.
bool Lower::helpers_read(bool quad) const { return quad && this->s.pixel; }

// `x` of the lane of this lane's quad that `lanes` names for each of the quad's four, by the
// host's quad operations.
V Lower::quad_read(V x, std::array<std::uint8_t, 4> lanes) {
    this->s.flags |= flags::WAVE_OPS;
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    if (lanes == std::array<std::uint8_t, 4>{0, 1, 2, 3}) {
        return x;
    }
    // One lane for all.
    if (lanes[1] == lanes[0] && lanes[2] == lanes[0] && lanes[3] == lanes[0]) {
        V n = this->s.m.i32(lanes[0]);
        return this->s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, i32t, 122, std::vector<V>{x, n});
    }
    // Else each lane's own, its neighbour across in x (1) or y (2), or the diagonal one (3).
    auto across = [&](std::size_t i) { return static_cast<std::uint8_t>(static_cast<std::uint8_t>(i) ^ lanes[i]); };
    std::optional<V> r;
    for (std::uint8_t by = 0; by < 4; ++by) {
        bool used = false;
        for (std::size_t i = 0; i < 4; ++i) {
            if (across(i) == by) {
                used = true;
                break;
            }
        }
        if (!used) {
            continue;
        }
        V read;
        if (by == 0) {
            read = x;
        } else {
            // QuadReadAcrossX, Y, Diagonal.
            V kind = this->s.m.i8(static_cast<std::uint8_t>(by - 1));
            read = this->s.op(c, "quadOp.i32", Attr::NoUnwind, i32t, 123, std::vector<V>{x, kind});
        }
        if (!r) {
            r = read;
        } else {
            V here = this->s.m.bool_(false);
            V lane = this->lane_index();
            V three = this->s.m.i32(3);
            V at = this->s.m.bin(c, Bin::And, lane, three);
            for (std::size_t i = 0; i < 4; ++i) {
                if (across(i) != by) continue;
                V k = this->s.m.i32(static_cast<std::uint32_t>(i));
                V is = this->s.m.cmp(c, pred::EQ, at, k);
                here = this->s.m.bin(c, Bin::Or, here, is);
            }
            r = this->s.m.select(c, here, read, *r);
        }
    }
    return r.value_or(x);
}

// `x` in the first lane of this host wave where `guard` holds, and whether it has such a lane: the
// console reads the first lane of EXEC, and a lane that got here with EXEC clear (a guard, not a
// branch) is not one. With none, `x` of the wave's first lane here.
//
// A helper lane of a pixel shader takes its own value: the console's whole-quad mode has helper lanes in
// EXEC, so a waterfall loop serves them too, and the host's wave operations leave them out. Once only
// helper lanes are left in such a loop none of them ever matched: they ran to the iteration limit, and left
// with no sample for their quad's derivatives, along every triangle's edges.
std::pair<V, V> Lower::first_lane(V x, std::optional<V> guard) {
    auto [first, any] = this->first_live_lane(x, guard);
    if (!this->s.pixel || !helper_own()) {
        return {first, any};
    }
    V helper = this->helper_lane();
    std::size_t c = this->cur;
    V yes = this->s.m.bool_(true);
    V a = this->s.m.select(c, helper, x, first);
    V b = this->s.m.select(c, helper, yes, any);
    return {a, b};
}

// Whether this lane of a pixel shader is a helper lane.
V Lower::helper_lane() {
    Ty i1t = this->s.t.i1;
    return this->s.op(this->cur, "isHelperLane.i1", Attr::ReadOnly, i1t, 221, std::vector<V>{});
}

std::pair<V, V> Lower::first_live_lane(V x, std::optional<V> guard) {
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    this->s.flags |= flags::WAVE_OPS;
    V first = this->s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, std::vector<V>{x});
    if (!guard) {
        V t = this->s.m.bool_(true);
        return {first, t};
    }
    V g = *guard;
    Ty four = this->s.m.named_struct("dx.types.fouri32", std::vector<Ty>(4, i32t));
    V r = this->s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, std::vector<V>{g});
    V lo = this->s.m.extract(c, r, 0);
    V hi = this->s.m.extract(c, r, 1);
    V z = this->s.m.i32(0);
    V k32 = this->s.m.i32(32);
    // FirstbitLo (dx.op 32): the lowest set bit
    V flo = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, std::vector<V>{lo});
    V fhi = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, std::vector<V>{hi});
    fhi = this->s.m.bin(c, Bin::Add, fhi, k32);
    V in_lo = this->s.m.cmp(c, pred::NE, lo, z);
    V in_hi = this->s.m.cmp(c, pred::NE, hi, z);
    V any = this->s.m.bin(c, Bin::Or, in_lo, in_hi);
    V lane = this->s.m.select(c, in_lo, flo, fhi);
    // no lane in EXEC: read this lane (one that is here); the read is not used
    V me = this->s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, std::vector<V>{});
    lane = this->s.m.select(c, any, lane, me);
    V at = this->read_lane(x, lane);
    V picked = this->s.m.select(c, any, at, first);
    return {picked, any};
}

V Lower::read_lane(V x, V lane) {
    return this->s.op(this->cur, "waveReadLaneAt.i32", Attr::NoUnwind, this->s.t.i32, 117,
                      std::vector<V>{x, lane});
}

// The float value of a register value.
V Lower::float_of(Value v) {
    if (auto it = this->floats.find(v); it != this->floats.end()) {
        return it->second;
    }
    V x = this->i32_of(v);
    return this->as_float(x);
}

// The values of VGPRs `reg..reg + n` as `inst` reads them.
std::vector<V> Lower::vgprs(const Inst& inst, const Operand& op, std::uint8_t n) {
    if (op.kind != Operand::Kind::Vector) {
        fail("a vector operand");
    }
    std::uint8_t reg = op.reg;
    std::vector<V> out;
    for (std::uint8_t k = 0; k < n; ++k) {
        Value v = ok_or(inst.read(Reg::V(static_cast<std::uint8_t>(reg + k))), "an unread operand");
        out.push_back(this->i32_of(v));
    }
    return out;
}

}  // namespace x2w::vs::lower
