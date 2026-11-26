// 64-bit operand helpers, 16-bit and packed 16-bit VALU operations.
#include <algorithm>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

namespace {

bool bit(std::uint32_t mask, std::size_t k) { return ((mask >> k) & 1u) != 0; }

bool in_list(std::span<const std::string_view> l, std::string_view name) {
    return std::find(l.begin(), l.end(), name) != l.end();
}

}  // namespace

V Lower::src64(const Inst& inst, std::size_t k) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    if (bit(v.abs, k) || bit(v.neg, k) || v.sdwa.has_value() || v.dpp.has_value()) {
        fail("64-bit operand modifiers");
    }
    std::size_t c = this->cur;
    Ty i64t = this->s.m.int_(64);
    this->s.flags |= flags::INT64;
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    auto pair = [&](Lower& l, Reg r, Reg r1) -> V {
        auto rv = ok_or(inst.read(r), "an unread operand");
        V lo = l.i32_of(rv);
        auto rv1 = ok_or(inst.read(r1), "an unread operand");
        V hi = l.i32_of(rv1);
        return l.join64(lo, hi);
    };
    if (op.kind == Operand::Kind::Vector) {
        return pair(*this, Reg::V(op.reg), Reg::V(static_cast<std::uint8_t>(op.reg + 1)));
    }
    if (op.kind == Operand::Kind::Scalar && op.code != gfx10::NULL_CODE) {
        return pair(*this, Reg::S(op.code), Reg::S(static_cast<std::uint8_t>(op.code + 1)));
    }
    if (op.kind == Operand::Kind::Const && op.code >= 128 && op.code <= 208) {
        std::uint32_t x = ok_or(resource::inline_constant(op.code), "a special operand");
        return this->s.m.int_const(i64t, static_cast<std::int64_t>(static_cast<std::int32_t>(x)));
    }
    // A signed 64-bit operand sign-extends its 32-bit literal, as the hardware does (LLVM's
    // AMDGPU operand rules).
    if (op.kind == Operand::Kind::Literal && inst.name().ends_with("_i64")) {
        return this->s.m.int_const(i64t, static_cast<std::int64_t>(static_cast<std::int32_t>(op.literal)));
    }
    // Literals (of unsigned and bit operands) and the null register: zero-extended.
    V x = this->operand(inst, op);
    return this->s.m.cast(c, Cast::ZExt, x, i64t);
}

// Reads an IEEE-754 binary64 operand as (lo, hi) 32-bit words, applying abs and neg input
// modifiers on operand k.
std::pair<V, V> Lower::f64_pair(const Inst& inst, std::size_t k) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    if (v.sdwa.has_value() || v.dpp.has_value()) {
        fail("64-bit operand modifiers");
    }
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    auto pair = [&](Lower& l, Reg r, Reg r1) -> std::pair<V, V> {
        auto rv = ok_or(inst.read(r), "an unread operand");
        V lo = l.i32_of(rv);
        auto rv1 = ok_or(inst.read(r1), "an unread operand");
        V hi = l.i32_of(rv1);
        return {lo, hi};
    };
    V lo{};
    V hi{};
    if (op.kind == Operand::Kind::Vector) {
        std::tie(lo, hi) = pair(*this, Reg::V(op.reg), Reg::V(static_cast<std::uint8_t>(op.reg + 1)));
    } else if (op.kind == Operand::Kind::Scalar && op.code != gfx10::NULL_CODE) {
        std::tie(lo, hi) = pair(*this, Reg::S(op.code), Reg::S(static_cast<std::uint8_t>(op.code + 1)));
    } else if (op.kind == Operand::Kind::Const && op.code == 128) {
        lo = this->s.m.i32(0);
        hi = this->s.m.i32(0);
    } else if (op.kind == Operand::Kind::Const && op.code >= 240 && op.code <= 248) {
        static constexpr double INLINE_F64[9] = {
            0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0,
            1.0 / 6.28318530717958647692528676655900577,  // 1 / std::f64::consts::TAU
        };
        std::uint64_t bits = std::bit_cast<std::uint64_t>(INLINE_F64[op.code - 240]);
        lo = this->s.m.i32(static_cast<std::uint32_t>(bits));
        hi = this->s.m.i32(static_cast<std::uint32_t>(bits >> 32));
    } else if (op.kind == Operand::Kind::Literal) {
        // GFX10 places a 32-bit literal in the high 32 bits of a 64-bit float operand.
        lo = this->s.m.i32(0);
        hi = this->s.m.i32(op.literal);
    } else {
        fail("a 64-bit float operand");
    }
    std::size_t c = this->cur;
    if (bit(v.abs, k)) {
        V mask = this->s.m.i32(0x7FFF'FFFF);
        hi = this->s.m.bin(c, Bin::And, hi, mask);
    }
    if (bit(v.neg, k)) {
        V sign = this->s.m.i32(0x8000'0000);
        hi = this->s.m.bin(c, Bin::Xor, hi, sign);
    }
    return {lo, hi};
}

// Converts IEEE-754 binary64 (lo, hi) to binary32 bits with round-to-nearest-even, without
// requiring host DOUBLES (f64) support.
V Lower::f64_to_f32_bits(V lo, V hi) {
    auto& m = this->s.m;
    std::size_t c = this->cur;
    V k0 = m.i32(0);
    V k1 = m.i32(1);
    V sign_mask = m.i32(0x8000'0000);
    V sign = m.bin(c, Bin::And, hi, sign_mask);

    V k20 = m.i32(20);
    V k7ff = m.i32(0x7FF);
    V exp64 = m.bin(c, Bin::LShr, hi, k20);
    exp64 = m.bin(c, Bin::And, exp64, k7ff);

    V frac_mask = m.i32(0x000F'FFFF);
    V frac_hi = m.bin(c, Bin::And, hi, frac_mask);
    V frac_any = m.bin(c, Bin::Or, frac_hi, lo);
    V frac_nz = m.cmp(c, pred::NE, frac_any, k0);

    V k3 = m.i32(3);
    V k29 = m.i32(29);
    V m23_hi = m.bin(c, Bin::Shl, frac_hi, k3);
    V m23_lo = m.bin(c, Bin::LShr, lo, k29);
    V m23 = m.bin(c, Bin::Or, m23_hi, m23_lo);

    V rem_mask = m.i32(0x1FFF'FFFF);
    V rem = m.bin(c, Bin::And, lo, rem_mask);
    V half_rem = m.i32(0x1000'0000);
    V gt_half = m.cmp(c, pred::IUGT, rem, half_rem);
    V eq_half = m.cmp(c, pred::EQ, rem, half_rem);
    V m23_odd_bit = m.bin(c, Bin::And, m23, k1);
    V m23_odd = m.cmp(c, pred::NE, m23_odd_bit, k0);
    V tie_up = m.bin(c, Bin::And, eq_half, m23_odd);
    V norm_round = m.bin(c, Bin::Or, gt_half, tie_up);
    V norm_inc = m.select(c, norm_round, k1, k0);

    // Normal range: exp64 in [897, 1150] -> exp32 in [1, 254].
    V k896 = m.i32(896);
    V k23 = m.i32(23);
    V exp32 = m.bin(c, Bin::Sub, exp64, k896);
    V exp32_sh = m.bin(c, Bin::Shl, exp32, k23);
    V norm_body = m.bin(c, Bin::Add, exp32_sh, m23);
    norm_body = m.bin(c, Bin::Add, norm_body, norm_inc);
    V norm_res = m.bin(c, Bin::Or, sign, norm_body);

    // Subnormal range: exp64 in [874, 896] -> shift in [1, 23].
    V k897 = m.i32(897);
    V k32 = m.i32(32);
    V k31 = m.i32(31);
    V imp = m.i32(0x0080'0000);
    V sig24 = m.bin(c, Bin::Or, imp, m23);
    V raw_shift = m.bin(c, Bin::Sub, k897, exp64);
    // Clamp shift to [1, 23] so shifts never produce poison outside the subnormal branch.
    V lt1 = m.cmp(c, pred::IULT, raw_shift, k1);
    V shift = m.select(c, lt1, k1, raw_shift);
    V gt23 = m.cmp(c, pred::IUGT, shift, k23);
    shift = m.select(c, gt23, k23, shift);
    V sub_m = m.bin(c, Bin::LShr, sig24, shift);
    V inv_shift = m.bin(c, Bin::Sub, k32, shift);
    V lost = m.bin(c, Bin::Shl, sig24, inv_shift);
    V round_bit = m.bin(c, Bin::LShr, lost, k31);
    V round_on = m.cmp(c, pred::NE, round_bit, k0);
    V low31 = m.i32(0x7FFF'FFFF);
    V lost_lo = m.bin(c, Bin::And, lost, low31);
    V sticky = m.bin(c, Bin::Or, lost_lo, rem);
    V sticky_on = m.cmp(c, pred::NE, sticky, k0);
    V sub_odd_bit = m.bin(c, Bin::And, sub_m, k1);
    V sub_odd = m.cmp(c, pred::NE, sub_odd_bit, k0);
    V sub_tie = m.bin(c, Bin::Or, sticky_on, sub_odd);
    V sub_round = m.bin(c, Bin::And, round_on, sub_tie);
    V sub_inc = m.select(c, sub_round, k1, k0);
    V sub_body = m.bin(c, Bin::Add, sub_m, sub_inc);
    V sub_res = m.bin(c, Bin::Or, sign, sub_body);

    // Underflow range: exp64 < 874. Only exp64 == 873 with non-zero fraction rounds up to min
    // subnormal (1).
    V k873 = m.i32(873);
    V eq873 = m.cmp(c, pred::EQ, exp64, k873);
    V round_min = m.bin(c, Bin::And, eq873, frac_nz);
    V min_bit = m.select(c, round_min, k1, k0);
    V under_res = m.bin(c, Bin::Or, sign, min_bit);

    // Overflow / Inf / NaN: exp64 >= 1151.
    V is_exp_all = m.cmp(c, pred::EQ, exp64, k7ff);
    V is_nan = m.bin(c, Bin::And, is_exp_all, frac_nz);
    V inf_bits = m.i32(0x7F80'0000);
    V qnan_bits = m.i32(0x7FC0'0000);
    V nan_body = m.bin(c, Bin::Or, qnan_bits, m23);
    V over_body = m.select(c, is_nan, nan_body, inf_bits);
    V over_res = m.bin(c, Bin::Or, sign, over_body);

    // Select final result by exponent range.
    V k874 = m.i32(874);
    V k1151 = m.i32(1151);
    V is_under = m.cmp(c, pred::IULT, exp64, k874);
    V is_sub = m.cmp(c, pred::IULT, exp64, k897);
    V is_over = m.cmp(c, pred::IUGE, exp64, k1151);
    V r = m.select(c, is_over, over_res, norm_res);
    r = m.select(c, is_sub, sub_res, r);
    return m.select(c, is_under, under_res, r);
}

V Lower::join64(V lo, V hi) {
    std::size_t c = this->cur;
    Ty ty = this->s.m.int_(64);
    lo = this->s.m.cast(c, Cast::ZExt, lo, ty);
    hi = this->s.m.cast(c, Cast::ZExt, hi, ty);
    V shift = this->s.m.int_const(ty, 32);
    hi = this->s.m.bin(c, Bin::Shl, hi, shift);
    return this->s.m.bin(c, Bin::Or, lo, hi);
}

// The bits of the float nearest the double whose dwords are lo and hi, on 32-bit integers: round to nearest
// even, infinities and NaN kept (a NaN stays quiet), a result below the smallest normal float signed zero
// (flushed, as the console's default float mode does). The flush-to-zero form of v_cvt_f32_f64: kept beside
// the IEEE form in use until the console's float mode for this instruction is checked.
V Lower::f64_bits_to_f32_bits(V lo, V hi) {
    std::size_t c = this->cur;
    auto k = [](Lower& l, std::uint32_t x) { return l.s.m.i32(x); };
    V k20 = k(*this, 20);
    V k29 = k(*this, 29);
    V k3 = k(*this, 3);
    V k23 = k(*this, 23);
    V sign_mask = k(*this, 0x8000'0000);
    V sign = this->s.m.bin(c, Bin::And, hi, sign_mask);
    V e = this->s.m.bin(c, Bin::LShr, hi, k20);
    V e_mask = k(*this, 0x7ff);
    e = this->s.m.bin(c, Bin::And, e, e_mask);
    // The top 23 bits of the 52-bit mantissa, and the 29 below them.
    V m_mask = k(*this, 0x000f'ffff);
    V mh = this->s.m.bin(c, Bin::And, hi, m_mask);
    mh = this->s.m.bin(c, Bin::Shl, mh, k3);
    V ml = this->s.m.bin(c, Bin::LShr, lo, k29);
    V m = this->s.m.bin(c, Bin::Or, mh, ml);
    V rest_mask = k(*this, 0x1fff'ffff);
    V rest = this->s.m.bin(c, Bin::And, lo, rest_mask);
    // Round to nearest even: up past half, or at half when odd.
    V half = k(*this, 0x1000'0000);
    V one = k(*this, 1);
    V above = this->s.m.cmp(c, pred::IUGT, rest, half);
    V at = this->s.m.cmp(c, pred::EQ, rest, half);
    V odd = this->s.m.bin(c, Bin::And, m, one);
    V zero = k(*this, 0);
    odd = this->s.m.cmp(c, pred::NE, odd, zero);
    V tie = this->s.m.bin(c, Bin::And, at, odd);
    V up = this->s.m.bin(c, Bin::Or, above, tie);
    up = this->s.m.select(c, up, one, zero);
    // Rebias the exponent (1023 -> 127); a carry out of the mantissa steps the exponent, up to
    // infinity.
    V bias = k(*this, 1023 - 127);
    V e32 = this->s.m.bin(c, Bin::Sub, e, bias);
    V ex = this->s.m.bin(c, Bin::Shl, e32, k23);
    V normal = this->s.m.bin(c, Bin::Add, ex, m);
    normal = this->s.m.bin(c, Bin::Add, normal, up);
    normal = this->s.m.bin(c, Bin::Or, normal, sign);
    // Exponents out of the float's range (signed, after rebiasing).
    V inf = k(*this, 0x7f80'0000);
    inf = this->s.m.bin(c, Bin::Or, inf, sign);
    V k255 = k(*this, 255);
    V big = this->s.m.cmp(c, pred::SGE, e32, k255);
    V small = this->s.m.cmp(c, pred::SLE, e32, zero);
    V r = this->s.m.select(c, big, inf, normal);
    r = this->s.m.select(c, small, sign, r);
    // Infinity and NaN (a NaN with any mantissa bit set stays a NaN).
    V all = k(*this, 0x7ff);
    V special = this->s.m.cmp(c, pred::EQ, e, all);
    V any = this->s.m.bin(c, Bin::Or, mh, ml);
    any = this->s.m.bin(c, Bin::Or, any, rest);
    V nan = this->s.m.cmp(c, pred::NE, any, zero);
    V quiet = k(*this, 0x0040'0000);
    quiet = this->s.m.select(c, nan, quiet, zero);
    V special_bits = this->s.m.bin(c, Bin::Or, inf, quiet);
    return this->s.m.select(c, special, special_bits, r);
}

// The low and high dwords of an i64.
std::pair<V, V> Lower::split64(V x) {
    std::size_t c = this->cur;
    Ty i64t = this->s.m.int_(64);
    V k32 = this->s.m.int_const(i64t, 32);
    V lo = this->s.m.cast(c, Cast::Trunc, x, this->s.t.i32);
    V hi = this->s.m.bin(c, Bin::LShr, x, k32);
    hi = this->s.m.cast(c, Cast::Trunc, hi, this->s.t.i32);
    return {lo, hi};
}

// Source k of a 16-bit operation in the low 16 bits (the rest 0): op_sel (or an SDWA word select)
// picks the high half; inline float constants are f16 values here.
V Lower::half_src(const Inst& inst, std::size_t k) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    bool high = bit(v.opsel, k);
    if (v.sdwa) {
        std::uint8_t sel = k < v.sdwa->src_sel.size() ? v.sdwa->src_sel[k] : 6;
        switch (sel) {
            case 4: break;
            case 5: high = true; break;
            case 6: break;
            default: fail("a byte select on a 16-bit operation");
        }
    }
    V x{};
    if (this->src0 && k == 0) {
        x = *this->src0;
    } else {
        x = this->f16_operand(inst, op);
    }
    std::size_t c = this->cur;
    if (high) {
        V k16 = this->s.m.i32(16);
        x = this->s.m.bin(c, Bin::LShr, x, k16);
    }
    V m = this->s.m.i32(0xFFFF);
    return this->s.m.bin(c, Bin::And, x, m);
}

// An f16 source as an f32 (exact), with its abs/neg modifiers.
V Lower::half_fsrc(const Inst& inst, std::size_t k) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    V h = this->half_src(inst, k);
    Ty f32t = this->s.t.f32;
    V f = this->s.op(this->cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{h});
    if (bit(v.abs, k)) {
        f = this->unary(6, f);
    }
    if (bit(v.neg, k)) {
        V z = this->s.m.f32(0x8000'0000);
        f = this->s.m.bin(this->cur, Bin::Sub, z, f);
    }
    return f;
}

// Writes a 16-bit result (low bits of x) into the destination's low half, or its high half with
// op_sel bit 3 (or an SDWA WORD_1 select); the other half is kept (GFX10).
void Lower::write_half(const Inst& inst, V x, std::optional<V> guard) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    if (!v.dst || v.dst->kind != Operand::Kind::Vector) {
        fail("a 16-bit result outside a VGPR");
    }
    std::uint8_t reg = v.dst->reg;
    bool high = bit(v.opsel, 3);
    // SDWA: the word written, the other one kept (preserve) or zero (pad; a sign extension above
    // the high word adds nothing).
    bool keep = true;
    if (v.sdwa) {
        std::uint8_t sel = v.sdwa->dst_sel;
        std::uint8_t unused = v.sdwa->dst_unused;
        if (sel == 4 && unused == 2) {
        } else if (sel == 5 && unused == 2) {
            high = true;
        } else if (sel == 4 || sel == 6) {
            keep = false;
        } else if (sel == 5) {
            high = true;
            keep = false;
        } else {
            fail("an SDWA destination of a 16-bit operation");
        }
    }
    std::size_t c = this->cur;
    // Convert a register whenever it is read, even if the result is discarded. Use zero if conversion fails.
    std::optional<V> read_old;
    if (auto o = inst.read(Reg::V(reg))) {
        read_old = ok([&] { return this->i32_of(*o); });
    }
    V old{};
    if (read_old && keep) {
        old = *read_old;
    } else {
        old = this->s.m.i32(0);
    }
    V k16 = this->s.m.i32(16);
    V low = this->s.m.i32(0xFFFF);
    x = this->s.m.bin(c, Bin::And, x, low);
    V r{};
    if (high) {
        V o = this->s.m.bin(c, Bin::And, old, low);
        V xs = this->s.m.bin(c, Bin::Shl, x, k16);
        r = this->s.m.bin(c, Bin::Or, o, xs);
    } else {
        V keepk = this->s.m.i32(0xFFFF'0000);
        V o = this->s.m.bin(c, Bin::And, old, keepk);
        r = this->s.m.bin(c, Bin::Or, o, x);
    }
    this->write_placed(inst, Reg::V(reg), r, guard);
}

// 16-bit operations, computed in f32 or i32: f16 sums, differences and products are exact there
// and round once to f16 (f32 has the 2 x 11 + 2 bits that make double rounding harmless); false
// (None) if name is none.
bool Lower::valu16(const Inst& inst, std::string_view name, std::optional<V> guard) {
    static constexpr std::string_view FLOAT2[] = {
        "v_add_f16", "v_sub_f16", "v_subrev_f16", "v_mul_f16", "v_max_f16", "v_min_f16",
    };
    static constexpr std::string_view FLOAT3[] = {
        "v_fma_f16", "v_mad_f16", "v_fmac_f16", "v_mac_f16", "v_max3_f16", "v_min3_f16", "v_med3_f16",
    };
    static constexpr std::string_view FLOAT1[] = {
        "v_rcp_f16",   "v_sqrt_f16",  "v_rsq_f16",  "v_log_f16",     "v_exp_f16",
        "v_fract_f16", "v_trunc_f16", "v_floor_f16", "v_ceil_f16",   "v_rndne_f16",
        "v_sin_f16",   "v_cos_f16",   "v_cvt_f16_u16", "v_cvt_f16_i16",
    };
    static constexpr std::string_view INT[] = {
        "v_add_nc_u16",      "v_sub_nc_u16",      "v_subrev_nc_u16", "v_mul_lo_u16",
        "v_lshlrev_b16",     "v_lshrrev_b16",     "v_ashrrev_i16",   "v_max_u16",
        "v_max_i16",         "v_min_u16",         "v_min_i16",       "v_cvt_u16_f16",
        "v_cvt_i16_f16",     "v_cvt_norm_i16_f16", "v_cvt_norm_u16_f16",
    };
    bool any = in_list(FLOAT1, name) || in_list(FLOAT2, name) || in_list(FLOAT3, name) ||
               in_list(INT, name);
    if (!any && !(name == "v_cvt_f16_f32" || name == "v_cvt_f32_f16")) {
        return false;
    }
    this->valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3);
    return true;
}

void Lower::valu16_op(const Inst& inst, std::string_view name, std::optional<V> guard,
                      std::span<const std::string_view> float1, std::span<const std::string_view> float2,
                      std::span<const std::string_view> float3) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    Ty f32t = this->s.t.f32;
    if (name == "v_cvt_f32_f16") {
        V f = this->half_fsrc(inst, 0);
        this->write_f(inst, f, guard);
        return;
    }
    auto float_result = [&](Lower& l, V f) {
        if (v.omod != 0) {
            static constexpr float FACTORS[4] = {1.0f, 2.0f, 4.0f, 0.5f};
            float factor = FACTORS[v.omod];
            V k = l.s.m.f32(f32_bits(factor));
            f = l.s.m.bin(l.cur, Bin::Mul, f, k);
        }
        if (v.clamp) {
            f = l.unary(7, f);
        }
        // The console rounds to nearest; the host's conversion may round toward zero (NVIDIA's
        // does), one step of the last place lower where the two differ.
        // shortcut: the host's conversion unless `X2W_SHADERCFG_F16_NEAREST=1` (f16_nearest_on), which
        // costs a dozen instructions per result.
        V h{};
        if (f16_nearest_on()) {
            h = l.f16_nearest(f);
        } else {
            h = l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f});
        }
        l.write_half(inst, h, guard);
    };
    if (name == "v_cvt_f16_f32") {
        V f = this->fsrc(inst, 0);
        float_result(*this, f);
        return;
    }
    if (in_list(float2, name)) {
        V a = this->half_fsrc(inst, 0);
        V b = this->half_fsrc(inst, 1);
        V r{};
        if (name == "v_add_f16") {
            r = this->s.m.bin(c, Bin::Add, a, b);
        } else if (name == "v_sub_f16") {
            r = this->s.m.bin(c, Bin::Sub, a, b);
        } else if (name == "v_subrev_f16") {
            r = this->s.m.bin(c, Bin::Sub, b, a);
        } else if (name == "v_mul_f16") {
            r = this->s.m.bin(c, Bin::Mul, a, b);
        } else if (name == "v_max_f16") {
            r = this->max_min(true, a, b);
        } else {
            r = this->max_min(false, a, b);
        }
        float_result(*this, r);
        return;
    }
    if (in_list(float3, name)) {
        V a = this->half_fsrc(inst, 0);
        V b = this->half_fsrc(inst, 1);
        V acc{};
        if (name == "v_fmac_f16" || name == "v_mac_f16") {
            // The accumulator is the destination's half that is written.
            if (!v.dst || v.dst->kind != Operand::Kind::Vector) {
                fail("an accumulator");
            }
            std::uint8_t reg = v.dst->reg;
            Value rv = ok_or(inst.read(Reg::V(reg)), "an accumulator");
            V x = this->i32_of(rv);
            if (bit(v.opsel, 3)) {
                V k16 = this->s.m.i32(16);
                x = this->s.m.bin(c, Bin::LShr, x, k16);
            }
            V m = this->s.m.i32(0xFFFF);
            x = this->s.m.bin(c, Bin::And, x, m);
            acc = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{x});
        } else {
            acc = this->half_fsrc(inst, 2);
        }
        V r{};
        if (name == "v_fma_f16" || name == "v_fmac_f16" || name == "v_mad_f16" || name == "v_mac_f16") {
            // shortcut: fused f16 multiply-adds as FMad in f32 (the product is exact; the sum
            // rounds twice).
            r = this->fmad(a, b, acc);
        } else if (name == "v_max3_f16") {
            V t = this->max_min(true, a, b);
            r = this->max_min(true, t, acc);
        } else if (name == "v_min3_f16") {
            V t = this->max_min(false, a, b);
            r = this->max_min(false, t, acc);
        } else {
            r = this->med3(a, b, acc);
        }
        float_result(*this, r);
        return;
    }
    if (in_list(float1, name)) {
        V r{};
        if (name == "v_cvt_f16_u16" || name == "v_cvt_f16_i16") {
            V x = this->half_src(inst, 0);
            Cast op{};
            if (name == "v_cvt_f16_i16") {
                V k16 = this->s.m.i32(16);
                V t = this->s.m.bin(c, Bin::Shl, x, k16);
                x = this->s.m.bin(c, Bin::AShr, t, k16);
                op = Cast::SiToFp;
            } else {
                op = Cast::UiToFp;
            }
            r = this->s.m.cast(c, op, x, f32t);
        } else if (name == "v_rcp_f16") {
            V a = this->half_fsrc(inst, 0);
            V one = this->s.m.f32(f32_bits(1.0f));
            r = this->s.m.bin(c, Bin::SDiv, one, a);
        } else {
            V a = this->half_fsrc(inst, 0);
            std::uint32_t op = name == "v_sqrt_f16"    ? 24
                               : name == "v_rsq_f16"   ? 25
                               : name == "v_log_f16"   ? 23
                               : name == "v_exp_f16"   ? 21
                               : name == "v_fract_f16" ? 22
                               : name == "v_trunc_f16" ? 29
                               : name == "v_floor_f16" ? 27
                               : name == "v_ceil_f16"  ? 28
                               : name == "v_rndne_f16" ? 26
                               : name == "v_sin_f16"   ? 13
                                                       : 12;
            if (op == 12 || op == 13) {
                r = this->trig_turns(a, op == 12);
            } else if (op == 22) {
                // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                r = this->fract(a, 1.0f - 1.0f / 4096.0f, 1.0f - 1.0f / 2048.0f);
            } else {
                r = this->unary(op, a);
            }
        }
        float_result(*this, r);
        return;
    }
    if (name == "v_cvt_u16_f16" || name == "v_cvt_i16_f16") {
        if (v.clamp || v.omod != 0) {
            fail("modifiers on a 16-bit integer operation");
        }
        // Toward zero, saturating at the 16-bit range (infinities included); NaN gives 0. Clamped in f32 first,
        // where every f16 is exact, so the conversion never leaves i32's range. half_fsrc applies the source
        // modifiers.
        V a = this->half_fsrc(inst, 0);
        float lof = 0.0f;
        float hif = 0.0f;
        if (name == "v_cvt_u16_f16") {
            lof = 0.0f;
            hif = 65535.0f;
        } else {
            lof = -32768.0f;
            hif = 32767.0f;
        }
        V lo = this->s.m.f32(f32_bits(lof));
        V hi = this->s.m.f32(f32_bits(hif));
        V t = this->binary(35, a, lo);
        t = this->binary(36, t, hi);
        V r = this->s.m.cast(c, Cast::FpToSi, t, i32t);
        V nan = this->s.m.cmp(c, pred::UNO, a, a);
        V z = this->s.m.i32(0);
        r = this->s.m.select(c, nan, z, r);
        this->write_half(inst, r, guard);
        return;
    }
    if (name == "v_cvt_norm_i16_f16" || name == "v_cvt_norm_u16_f16") {
        if (v.clamp || v.omod != 0) {
            fail("modifiers on a 16-bit integer operation");
        }
        V r = this->cvt_norm16(inst, name == "v_cvt_norm_i16_f16");
        this->write_half(inst, r, guard);
        return;
    }
    // 16-bit integers: computed on the low halves, the low 16 bits kept.
    if (v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0) {
        fail("modifiers on a 16-bit integer operation");
    }
    V r{};
    {
        V a = this->half_src(inst, 0);
        V b = this->half_src(inst, 1);
        V k16 = this->s.m.i32(16);
        auto sext = [&](Lower& l, V x) {
            V t = l.s.m.bin(c, Bin::Shl, x, k16);
            return l.s.m.bin(c, Bin::AShr, t, k16);
        };
        if (name.ends_with("_i16")) {
            a = sext(*this, a);
            b = sext(*this, b);
        }
        V k15 = this->s.m.i32(15);
        if (name == "v_add_nc_u16") {
            r = this->s.m.bin(c, Bin::Add, a, b);
        } else if (name == "v_sub_nc_u16") {
            r = this->s.m.bin(c, Bin::Sub, a, b);
        } else if (name == "v_subrev_nc_u16") {
            r = this->s.m.bin(c, Bin::Sub, b, a);
        } else if (name == "v_mul_lo_u16") {
            r = this->s.m.bin(c, Bin::Mul, a, b);
        } else if (name == "v_lshlrev_b16" || name == "v_lshrrev_b16" || name == "v_ashrrev_i16") {
            // src0 is the shift (4 bits), src1 the value.
            V sh = this->s.m.bin(c, Bin::And, a, k15);
            V x = name == "v_ashrrev_i16" ? sext(*this, b) : b;
            Bin op = name == "v_lshlrev_b16" ? Bin::Shl : name == "v_lshrrev_b16" ? Bin::LShr : Bin::AShr;
            r = this->s.m.bin(c, op, x, sh);
        } else {
            std::uint8_t p = name == "v_max_u16"   ? pred::IUGT
                             : name == "v_min_u16" ? pred::IULT
                             : name == "v_max_i16" ? pred::SGT
                                                   : pred::SLT;
            V t = this->s.m.cmp(c, p, a, b);
            r = this->s.m.select(c, t, a, b);
        }
    }
    this->write_half(inst, r, guard);
}

// v_cvt_norm_i16_f16 / v_cvt_norm_u16_f16: the f16 source (with its abs/neg modifiers) clamped to
// [-1, 1] or [0, 1], times 32767 or 65535, rounded to nearest even once from the exact product; NaN
// gives 0. Worked on the f16's bits in integers, because in f32 the product (up to 26 significant
// bits) would round before the rounding to an integer.
V Lower::cvt_norm16(const Inst& inst, bool signed_) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    std::size_t c = this->cur;
    auto k = [](Lower& l, std::uint32_t x) { return l.s.m.i32(x); };
    V h = this->half_src(inst, 0);
    if ((v.abs & 1) != 0) {
        V m = k(*this, 0x7FFF);
        h = this->s.m.bin(c, Bin::And, h, m);
    }
    if ((v.neg & 1) != 0) {
        V m = k(*this, 0x8000);
        h = this->s.m.bin(c, Bin::Xor, h, m);
    }
    std::uint32_t scale = signed_ ? 0x7FFF : 0xFFFF;
    V k0 = k(*this, 0);
    V k1 = k(*this, 1);
    V k10 = k(*this, 10);
    V mag_mask = k(*this, 0x7FFF);
    V mag = this->s.m.bin(c, Bin::And, h, mag_mask);
    V k15 = k(*this, 15);
    V neg = this->s.m.bin(c, Bin::LShr, h, k15);
    neg = this->s.m.cmp(c, pred::NE, neg, k0);
    // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
    V exp = this->s.m.bin(c, Bin::LShr, mag, k10);
    V man_mask = k(*this, 0x3FF);
    V man = this->s.m.bin(c, Bin::And, mag, man_mask);
    V implicit = k(*this, 0x400);
    V normal = this->s.m.bin(c, Bin::Or, man, implicit);
    V sub = this->s.m.cmp(c, pred::EQ, exp, k0);
    V m = this->s.m.select(c, sub, man, normal);
    V k25 = k(*this, 25);
    V s_norm = this->s.m.bin(c, Bin::Sub, k25, exp);
    V k24 = k(*this, 24);
    V sh = this->s.m.select(c, sub, k24, s_norm);
    V ks = k(*this, scale);
    V p = this->s.m.bin(c, Bin::Mul, m, ks);
    V q = this->s.m.bin(c, Bin::LShr, p, sh);
    V one_s = this->s.m.bin(c, Bin::Shl, k1, sh);
    V low_mask = this->s.m.bin(c, Bin::Sub, one_s, k1);
    V rem = this->s.m.bin(c, Bin::And, p, low_mask);
    V half = this->s.m.bin(c, Bin::LShr, one_s, k1);
    V above = this->s.m.cmp(c, pred::IUGT, rem, half);
    V tie = this->s.m.cmp(c, pred::EQ, rem, half);
    V odd = this->s.m.bin(c, Bin::And, q, k1);
    odd = this->s.m.cmp(c, pred::NE, odd, k0);
    V tie_up = this->s.m.bin(c, Bin::And, tie, odd);
    V up = this->s.m.bin(c, Bin::Or, above, tie_up);
    V r = this->s.m.select(c, up, k1, k0);
    r = this->s.m.bin(c, Bin::Add, q, r);
    // |x| >= 1 (and infinity) saturates; NaN gives 0.
    V one_h = k(*this, 0x3C00);
    V big = this->s.m.cmp(c, pred::IUGE, mag, one_h);
    r = this->s.m.select(c, big, ks, r);
    V inf = k(*this, 0x7C00);
    V nan = this->s.m.cmp(c, pred::IUGT, mag, inf);
    r = this->s.m.select(c, nan, k0, r);
    if (signed_) {
        V n = this->s.m.bin(c, Bin::Sub, k0, r);
        r = this->s.m.select(c, neg, n, r);
    } else {
        r = this->s.m.select(c, neg, k0, r);
    }
    V m16 = k(*this, 0xFFFF);
    return this->s.m.bin(c, Bin::And, r, m16);
}

// Half `high` of source k of a packed (VOP3P) operation, in the low 16 bits: op_sel picks the half
// the low result reads, op_sel_hi the half the high result reads. Packed integer instructions see
// the f32 bits of an inline float constant.
V Lower::packed_src(const Inst& inst, std::size_t k, bool high, bool float_) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    if (k >= v.src.size()) fail("a missing operand");
    Operand op = v.src[k];
    bool sel = bit(high ? v.opsel_hi : v.opsel, k);
    V x{};
    if (float_) {
        x = this->f16_operand(inst, op);
    } else {
        x = this->operand(inst, op);
    }
    std::size_t c = this->cur;
    if (sel) {
        V k16 = this->s.m.i32(16);
        x = this->s.m.bin(c, Bin::LShr, x, k16);
    }
    V m = this->s.m.i32(0xFFFF);
    return this->s.m.bin(c, Bin::And, x, m);
}

// Half `high` of source k of v_pk_fmac_f16, in the low 16 bits: a VOP2 operation (no op_sel) whose
// third source is its destination.
V Lower::fmac_half(const Inst& inst, std::size_t k, bool high) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    std::optional<Operand> opo;
    if (k == 2) {
        opo = v.dst;
    } else if (k < v.src.size()) {
        opo = v.src[k];
    }
    Operand op = ok_or(opo, "a missing operand");
    if (op.kind == Operand::Kind::Const || op.kind == Operand::Kind::Literal) {
        // shortcut: no constant operand seen; its high half is not known.
        fail("a constant operand of v_pk_fmac_f16");
    }
    V x = this->operand(inst, op);
    std::size_t c = this->cur;
    if (high) {
        V k16 = this->s.m.i32(16);
        x = this->s.m.bin(c, Bin::LShr, x, k16);
    }
    V m = this->s.m.i32(0xFFFF);
    return this->s.m.bin(c, Bin::And, x, m);
}

// Packed 16-bit operations (v_pk_*): each half computed as the 16-bit operation does, f16 through
// f32 as in valu16.
void Lower::packed16(const Inst& inst, std::string_view op, std::optional<V> guard) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    bool float_ = op.ends_with("_f16");
    std::size_t n = 0;
    if (op == "fma_f16" || op == "fmac_f16" || op == "mad_u16" || op == "mad_i16") {
        n = 3;
    } else if (op == "add_f16" || op == "mul_f16" || op == "min_f16" || op == "max_f16" ||
               op == "add_u16" || op == "sub_u16" || op == "add_i16" || op == "sub_i16" ||
               op == "mul_lo_u16" || op == "lshlrev_b16" || op == "lshrrev_b16" ||
               op == "ashrrev_i16" || op == "max_u16" || op == "min_u16" || op == "max_i16" ||
               op == "min_i16") {
        n = 2;
    } else {
        fail("a packed 16-bit operation");
    }
    if (v.omod != 0 || (!float_ && (v.clamp || v.neg != 0 || v.neg_hi != 0))) {
        fail("modifiers on a packed integer operation");
    }
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    Ty f32t = this->s.t.f32;
    std::vector<V> halves;
    for (bool high : {false, true}) {
        std::vector<V> xs;
        for (std::size_t k = 0; k < n; ++k) {
            V h{};
            if (op == "fmac_f16") {
                h = this->fmac_half(inst, k, high);
            } else {
                h = this->packed_src(inst, k, high, float_);
            }
            if (float_) {
                V f = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{h});
                std::uint8_t neg = high ? v.neg_hi : v.neg;
                if (bit(neg, k)) {
                    V z = this->s.m.f32(0x8000'0000);
                    f = this->s.m.bin(c, Bin::Sub, z, f);
                }
                xs.push_back(f);
            } else {
                xs.push_back(h);
            }
        }
        V r{};
        if (float_) {
            V f{};
            if (op == "add_f16") {
                f = this->s.m.bin(c, Bin::Add, xs[0], xs[1]);
            } else if (op == "mul_f16") {
                f = this->s.m.bin(c, Bin::Mul, xs[0], xs[1]);
            } else if (op == "min_f16") {
                f = this->max_min(false, xs[0], xs[1]);
            } else if (op == "max_f16") {
                f = this->max_min(true, xs[0], xs[1]);
            } else {
                // shortcut: FMad in f32, as v_fma_f16.
                f = this->fmad(xs[0], xs[1], xs[2]);
            }
            if (v.clamp) {
                f = this->unary(7, f);
            }
            r = this->s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f});
        } else {
            V k16 = this->s.m.i32(16);
            auto sext = [&](Lower& l, V x) {
                V t = l.s.m.bin(c, Bin::Shl, x, k16);
                return l.s.m.bin(c, Bin::AShr, t, k16);
            };
            if (op.ends_with("_i16")) {
                for (auto& x : xs) {
                    x = sext(*this, x);
                }
            }
            V k15 = this->s.m.i32(15);
            if (op == "add_u16" || op == "add_i16") {
                r = this->s.m.bin(c, Bin::Add, xs[0], xs[1]);
            } else if (op == "sub_u16" || op == "sub_i16") {
                r = this->s.m.bin(c, Bin::Sub, xs[0], xs[1]);
            } else if (op == "mul_lo_u16") {
                r = this->s.m.bin(c, Bin::Mul, xs[0], xs[1]);
            } else if (op == "mad_u16" || op == "mad_i16") {
                V m = this->s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                r = this->s.m.bin(c, Bin::Add, m, xs[2]);
            } else if (op == "lshlrev_b16" || op == "lshrrev_b16" || op == "ashrrev_i16") {
                V sh = this->s.m.bin(c, Bin::And, xs[0], k15);
                Bin o = op == "lshlrev_b16" ? Bin::Shl : op == "lshrrev_b16" ? Bin::LShr : Bin::AShr;
                V x = op == "ashrrev_i16" ? sext(*this, xs[1]) : xs[1];
                r = this->s.m.bin(c, o, x, sh);
            } else {
                std::uint8_t p = op == "max_u16"   ? pred::IUGT
                                 : op == "min_u16" ? pred::IULT
                                 : op == "max_i16" ? pred::SGT
                                                   : pred::SLT;
                V t = this->s.m.cmp(c, p, xs[0], xs[1]);
                r = this->s.m.select(c, t, xs[0], xs[1]);
            }
        }
        V m = this->s.m.i32(0xFFFF);
        halves.push_back(this->s.m.bin(c, Bin::And, r, m));
    }
    V k16 = this->s.m.i32(16);
    V hi = this->s.m.bin(c, Bin::Shl, halves[1], k16);
    V r = this->s.m.bin(c, Bin::Or, halves[0], hi);
    if (!v.dst || v.dst->kind != Operand::Kind::Vector) {
        fail("a packed result outside a VGPR");
    }
    this->write(inst, Reg::V(v.dst->reg), r, guard);
}

}  // namespace x2w::vs::lower
