"""Lift bounded Windows x64 leaf functions using x87 F2XM1, FYL2X, and FYL2XP1 to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat zero-displacement host pointers to caller-owned
buffers (rcx, rdx, and r8 may be read; only r8 may be written).

All x87 values use a portable software 80-bit extended-precision representation
(explicit 64-bit significand + 15-bit biased exponent + sign bit) and pure
integer 128-bit intermediate arithmetic (zero host float/double/long double or
SSE/x87 instructions). This avoids MSVC's 64-bit `long double` truncation and
guarantees complete isolation from host MXCSR (DAZ/FTZ/rounding) without
clobbering MXCSR.

Supported x87 transcendental instructions (docs/coverage/x64_zen2.tsv group x87):
  - F2XM1   : ST(0) <- 2^ST(0) - 1 on the Intel SDM domain [-1, +1]
              (<= 1 ULP vs hardware, <= 0.5 ULP vs exact math in 80-bit extended;
              exact on +-0, +1, -1; preserves signed zero F2XM1(-0.0) == -0.0).
  - FYL2X   : ST(1) <- ST(1) * log2(ST(0)), then pop x87 stack (x > 0;
              #Z (ZE) on x == +-0 with finite nonzero y; #IA (IE) on negative x,
              (+-0, +-0), (+inf, +-0), (1, +-inf)).
  - FYL2XP1 : ST(1) <- ST(1) * log2(ST(0) + 1), then pop x87 stack on the Intel
              SDM domain [-(1 - sqrt(2)/2), +(1 - sqrt(2)/2)] (and models #Z (ZE)
              on x == -1 with finite nonzero y, #IA (IE) on x < -1 or (+-0, +-inf)).

Bounded leaf x87 stack/status/control transport:
  - fld m32 / m64 / m80, fld1, fldz
  - fst m32 / m64
  - fstp m32 / m64 / m80 / st(0)
  - fnstsw ax (paired with mov word ptr [r8], ax when ax is defined)
  - fldcw m16, fnstcw m16, fninit
  - terminal ret (requires an empty x87 stack at ret)

FCW rounding modes (RN/RD/RU/RZ) apply to transcendental results and m32/m64
stores; FCW precision control (PC 24/53/64) is modeled in the control word
(preserved across fldcw/fnstcw) and, matching Intel SDM Vol. 1 8.1.5.2 and x87
hardware, does not truncate F2XM1, FYL2X, or FYL2XP1 results.

FSIN/FCOS/FSINCOS/FPTAN/FPATAN, stack overflow/underflow, non-empty x87 stack
at ret, XMM/YMM/ZMM/VEX/EVEX encodings, prefixes (including WAIT/9B), non-flat,
displaced, or indexed memory, stores outside [r8], branches, undefined ax reads,
missing ret, and trailing instructions after ret are rejected.
"""

import argparse
from pathlib import Path

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    FlowControl,
    MemorySizeExt,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}

NONVOLATILE_GPR = {
    R.RBX, R.EBX, R.BX, R.BL, R.BH,
    R.RBP, R.EBP, R.BP, R.BPL,
    R.RSI, R.ESI, R.SI, R.SIL,
    R.RDI, R.EDI, R.DI, R.DIL,
    R.RSP, R.ESP, R.SP, R.SPL,
    R.R12, R.R12D, R.R12W, R.R12L,
    R.R13, R.R13D, R.R13W, R.R13L,
    R.R14, R.R14D, R.R14W, R.R14L,
    R.R15, R.R15D, R.R15W, R.R15L,
}

TRIG_REJECT_CODES = {
    Code.FSIN,
    Code.FCOS,
    Code.FSINCOS,
    Code.FPTAN,
    Code.FPATAN,
}

FLD_MEM_CODES = {
    Code.FLD_M32FP: (4, "x87_load_f32"),
    Code.FLD_M64FP: (8, "x87_load_f64"),
    Code.FLD_M80FP: (10, "x87_load_m80"),
}

FST_MEM_CODES = {
    Code.FST_M32FP: (4, "x87_store_f32", False),
    Code.FST_M64FP: (8, "x87_store_f64", False),
    Code.FSTP_M32FP: (4, "x87_store_f32", True),
    Code.FSTP_M64FP: (8, "x87_store_f64", True),
    Code.FSTP_M80FP: (10, "x87_store_m80", True),
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.F2XM1,
    Code.FYL2X,
    Code.FYL2XP1,
    *FLD_MEM_CODES,
    Code.FLD1,
    Code.FLDZ,
    *FST_MEM_CODES,
    Code.FSTP_STI,
    Code.FNSTSW_AX,
    Code.FLDCW_M2BYTE,
    Code.FNSTCW_M2BYTE,
    Code.FNINIT,
    Code.MOV_RM16_R16,
}


PRELUDE = r"""#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

#if defined(_MSC_VER)
// Test-only host MXCSR accessors used by tests/cpu/x87_exp_log_aot_check.py to
// verify DAZ/FTZ isolation and non-clobbering; lifted_* functions never touch SSE/MXCSR.
extern "C" __declspec(dllexport) std::uint32_t host_get_mxcsr() {
    return _mm_getcsr();
}

extern "C" __declspec(dllexport) std::uint32_t host_exchange_mxcsr(std::uint32_t next) {
    const std::uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
#endif

using word = std::uint64_t;

// ---- Portable 80-bit extended-precision x87 state (pure integer arithmetic) ----

struct X87Val {
    std::uint64_t sig;  // Explicit integer bit in bit 63
    std::uint16_t se;   // bit 15 = sign, bits 14:0 = biased exponent (bias 16383)
};

static constexpr std::uint64_t X87_JBIT = 0x8000000000000000ull;
static constexpr std::uint64_t X87_QBIT = 0x4000000000000000ull;
static constexpr X87Val X87_INDEF{0xc000000000000000ull, 0xffffu};
static constexpr X87Val X87_POS_ZERO{0ull, 0x0000u};
static constexpr X87Val X87_POS_ONE{X87_JBIT, 0x3fffu};
static constexpr X87Val X87_NEG_ONE{X87_JBIT, 0xbfffu};
static constexpr X87Val X87_NEG_HALF{X87_JBIT, 0xbffeu};
static constexpr X87Val X87_POS_INF{X87_JBIT, 0x7fffu};

static constexpr std::uint16_t SW_IE = 0x0001u;
static constexpr std::uint16_t SW_DE = 0x0002u;
static constexpr std::uint16_t SW_ZE = 0x0004u;
static constexpr std::uint16_t SW_OE = 0x0008u;
static constexpr std::uint16_t SW_UE = 0x0010u;
static constexpr std::uint16_t SW_PE = 0x0020u;
static constexpr std::uint16_t SW_C1 = 0x0200u;

struct X87State {
    X87Val regs[8]{};
    std::uint16_t cw = 0x037fu;
    std::uint16_t sw = 0x0000u;
    unsigned top = 0u;
};

static inline std::uint16_t load16(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint16_t out = 0u;
    std::memcpy(&out, reinterpret_cast<const void*>(address), sizeof(out));
    const volatile std::uint16_t observed = out;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return observed;
}

static inline std::uint32_t load32(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint32_t out = 0u;
    std::memcpy(&out, reinterpret_cast<const void*>(address), sizeof(out));
    const volatile std::uint32_t observed = out;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return observed;
}

static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint64_t out = 0ull;
    std::memcpy(&out, reinterpret_cast<const void*>(address), sizeof(out));
    const volatile std::uint64_t observed = out;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return observed;
}

static inline X87Val load80(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    X87Val out{};
    std::memcpy(&out.sig, reinterpret_cast<const void*>(address), 8);
    std::memcpy(&out.se, reinterpret_cast<const void*>(address + 8ull), 2);
    const volatile std::uint64_t osig = out.sig;
    const volatile std::uint16_t ose = out.se;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {osig, ose};
}

static inline void store16(word address, std::uint16_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store32(word address, std::uint32_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store64(word address, std::uint64_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store80(word address, X87Val value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &value.sig, 8);
    std::memcpy(reinterpret_cast<void*>(address + 8ull), &value.se, 2);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline int clz64(std::uint64_t x) {
    if (x == 0ull) return 64;
    int n = 0;
    if ((x & 0xffffffff00000000ull) == 0ull) { n += 32; x <<= 32; }
    if ((x & 0xffff000000000000ull) == 0ull) { n += 16; x <<= 16; }
    if ((x & 0xff00000000000000ull) == 0ull) { n += 8;  x <<= 8;  }
    if ((x & 0xf000000000000000ull) == 0ull) { n += 4;  x <<= 4;  }
    if ((x & 0xc000000000000000ull) == 0ull) { n += 2;  x <<= 2;  }
    if ((x & 0x8000000000000000ull) == 0ull) { n += 1; }
    return n;
}

static inline void mul64x64(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo) {
    const std::uint64_t a0 = a & 0xffffffffull, a1 = a >> 32;
    const std::uint64_t b0 = b & 0xffffffffull, b1 = b >> 32;
    const std::uint64_t p00 = a0 * b0;
    const std::uint64_t p01 = a0 * b1;
    const std::uint64_t p10 = a1 * b0;
    const std::uint64_t p11 = a1 * b1;
    const std::uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffull) + (p10 & 0xffffffffull);
    lo = (p00 & 0xffffffffull) | (mid << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

static inline void shr128_sticky(std::uint64_t& hi, std::uint64_t& lo, int n) {
    if (n <= 0) return;
    if (n >= 128) {
        lo = (hi | lo) != 0ull ? 1ull : 0ull;
        hi = 0ull;
        return;
    }
    std::uint64_t sticky = 0ull;
    if (n >= 64) {
        const int m = n - 64;
        sticky = (lo != 0ull || (m != 0 && (hi << (64 - m)) != 0ull)) ? 1ull : 0ull;
        lo = (m != 0) ? (hi >> m) : hi;
        hi = 0ull;
    } else {
        sticky = ((lo << (64 - n)) != 0ull) ? 1ull : 0ull;
        lo = (lo >> n) | (hi << (64 - n));
        hi >>= n;
    }
    lo |= sticky;
}

static inline unsigned x87_rc(std::uint16_t cw) { return (cw >> 10) & 3u; }
static inline unsigned x87_exp(X87Val v) { return v.se & 0x7fffu; }
static inline bool x87_sign(X87Val v) { return (v.se & 0x8000u) != 0u; }
static inline bool x87_unsupported(X87Val v) { return x87_exp(v) != 0u && (v.sig & X87_JBIT) == 0ull; }
static inline bool x87_is_nan(X87Val v) { return x87_exp(v) == 0x7fffu && (v.sig & X87_JBIT) != 0ull && (v.sig << 1) != 0ull; }
static inline bool x87_is_snan(X87Val v) { return x87_is_nan(v) && (v.sig & X87_QBIT) == 0ull; }
static inline bool x87_is_qnan(X87Val v) { return x87_is_nan(v) && (v.sig & X87_QBIT) != 0ull; }
static inline bool x87_is_inf(X87Val v) { return x87_exp(v) == 0x7fffu && v.sig == X87_JBIT; }
static inline bool x87_is_zero(X87Val v) { return x87_exp(v) == 0u && v.sig == 0ull; }
static inline bool x87_is_denorm(X87Val v) { return x87_exp(v) == 0u && v.sig != 0ull; }
static inline X87Val x87_quiet(X87Val v) { v.sig |= X87_QBIT; return v; }

static inline X87Val x87_make_zero(bool sign) { return {0ull, sign ? std::uint16_t{0x8000u} : std::uint16_t{0x0000u}}; }
static inline X87Val x87_make_inf(bool sign) { return {X87_JBIT, static_cast<std::uint16_t>((sign ? 0x8000u : 0u) | 0x7fffu)}; }

static inline void x87_set_c1(std::uint16_t& sw, bool up) {
    if (up) sw |= SW_C1;
    else sw &= static_cast<std::uint16_t>(~SW_C1);
}

static inline void x87_push(X87State& st, X87Val v) {
    st.top = (st.top - 1u) & 7u;
    st.regs[st.top] = v;
}

static inline X87Val x87_pop(X87State& st) {
    const X87Val v = st.regs[st.top];
    st.top = (st.top + 1u) & 7u;
    return v;
}

static inline X87Val& x87_st(X87State& st, unsigned i) {
    return st.regs[(st.top + i) & 7u];
}

static inline std::uint16_t x87_status_word(const X87State& st) {
    return static_cast<std::uint16_t>((st.sw & ~0x3800u) | ((st.top & 7u) << 11));
}

static inline void x87_fninit(X87State& st) {
    for (unsigned i = 0u; i < 8u; ++i) st.regs[i] = X87_POS_ZERO;
    st.cw = 0x037fu;
    st.sw = 0x0000u;
    st.top = 0u;
}

// ---- Binary32 / Binary64 / Raw80 load and store conversions ----

static inline X87Val x87_from_binary(std::uint64_t bits, int frac_bits, int exp_bits, int bias, std::uint16_t& sw) {
    const bool sign = ((bits >> (frac_bits + exp_bits)) & 1ull) != 0ull;
    const std::uint16_t s = sign ? 0x8000u : 0u;
    const std::uint64_t exp_mask = (1ull << exp_bits) - 1ull;
    const std::uint64_t frac_mask = (1ull << frac_bits) - 1ull;
    const int exponent = static_cast<int>((bits >> frac_bits) & exp_mask);
    const std::uint64_t fraction = bits & frac_mask;
    if (exponent == static_cast<int>(exp_mask)) {
        if (fraction == 0ull) return {X87_JBIT, static_cast<std::uint16_t>(s | 0x7fffu)};
        X87Val nan{X87_JBIT | (fraction << (63 - frac_bits)), static_cast<std::uint16_t>(s | 0x7fffu)};
        if (x87_is_snan(nan)) sw |= SW_IE;
        return x87_quiet(nan);
    }
    if (exponent == 0) {
        if (fraction == 0ull) return {0ull, s};
        sw |= SW_DE;
        const int shift = clz64(fraction);
        const int e = 1 - bias - frac_bits + (63 - shift);
        return {fraction << shift, static_cast<std::uint16_t>(s | static_cast<std::uint16_t>(e + 16383))};
    }
    return {X87_JBIT | (fraction << (63 - frac_bits)),
            static_cast<std::uint16_t>(s | static_cast<std::uint16_t>(exponent - bias + 16383))};
}

static inline void x87_load_f32(X87State& st, word addr) {
    x87_push(st, x87_from_binary(load32(addr), 23, 8, 127, st.sw));
}

static inline void x87_load_f64(X87State& st, word addr) {
    x87_push(st, x87_from_binary(load64(addr), 52, 11, 1023, st.sw));
}

static inline void x87_load_m80(X87State& st, word addr) {
    x87_push(st, load80(addr));
}

static inline std::uint64_t x87_to_binary(X87Val v, int frac_bits, int exp_bits, int bias, int emax,
                                          std::uint16_t cw, std::uint16_t& sw) {
    const int sign_bit = frac_bits + exp_bits;
    const std::uint64_t sign_mask = static_cast<std::uint64_t>(x87_sign(v) ? 1u : 0u) << sign_bit;
    const std::uint64_t exp_all = static_cast<std::uint64_t>(emax + 1) << frac_bits;
    const std::uint64_t qbit = 1ull << (frac_bits - 1);
    const std::uint64_t frac_mask = (1ull << frac_bits) - 1ull;
    if (x87_unsupported(v)) {
        sw |= SW_IE;
        return (1ull << sign_bit) | exp_all | qbit;
    }
    if (x87_is_nan(v)) {
        if (x87_is_snan(v)) sw |= SW_IE;
        return sign_mask | exp_all | qbit | ((v.sig >> (63 - frac_bits)) & frac_mask);
    }
    if (x87_is_inf(v)) return sign_mask | exp_all;
    if (x87_is_zero(v)) return sign_mask;

    const bool sign = x87_sign(v);
    std::int32_t e = 0;
    std::uint64_t hi = v.sig, lo = 0ull;
    if (x87_exp(v) != 0u) {
        e = static_cast<std::int32_t>(x87_exp(v)) - 16383;
    } else {
        const int shift = clz64(v.sig);
        e = 1 - 16383 - shift;
        hi = v.sig << shift;
    }
    std::int32_t biased = e + bias;
    bool tiny = false;
    if (biased <= 0) {
        shr128_sticky(hi, lo, 1 - biased);
        biased = 0;
        tiny = true;
    }
    const int precision = frac_bits + 1;
    const int drop = 64 - precision;
    const std::uint64_t unit = 1ull << drop;
    const std::uint64_t mask = unit - 1ull;
    const std::uint64_t rest = hi & mask;
    const std::uint64_t middle = unit >> 1;
    const bool above = (rest > middle) || (rest == middle && lo != 0ull);
    const bool half = (rest == middle && lo == 0ull);
    const bool inexact = (rest != 0ull || lo != 0ull);
    hi &= ~mask;
    const unsigned rc = x87_rc(cw);
    bool up = false;
    switch (rc) {
        case 0u: up = above || (half && (hi & unit) != 0ull); break;
        case 1u: up = sign && inexact; break;
        case 2u: up = !sign && inexact; break;
        default: break;
    }
    if (up) {
        hi += unit;
        if (hi == 0ull) { hi = X87_JBIT; ++biased; }
    }
    if (biased == 0 && (hi & X87_JBIT) != 0ull) biased = 1;
    if (biased > emax) {
        sw |= static_cast<std::uint16_t>(SW_OE | SW_PE);
        const bool to_inf = (rc == 0u) || (rc == 1u && sign) || (rc == 2u && !sign);
        x87_set_c1(sw, to_inf);
        if (to_inf) return sign_mask | exp_all;
        return sign_mask | (static_cast<std::uint64_t>(emax) << frac_bits) | frac_mask;
    }
    if (tiny && inexact) sw |= SW_UE;
    if (inexact) sw |= SW_PE;
    x87_set_c1(sw, up);
    return sign_mask | (static_cast<std::uint64_t>(biased) << frac_bits) | ((hi >> (63 - frac_bits)) & frac_mask);
}

static inline void x87_store_f32(X87State& st, word addr, bool pop) {
    const X87Val v = pop ? x87_pop(st) : x87_st(st, 0u);
    store32(addr, static_cast<std::uint32_t>(x87_to_binary(v, 23, 8, 127, 0xfe, st.cw, st.sw)));
}

static inline void x87_store_f64(X87State& st, word addr, bool pop) {
    const X87Val v = pop ? x87_pop(st) : x87_st(st, 0u);
    store64(addr, x87_to_binary(v, 52, 11, 1023, 0x7fe, st.cw, st.sw));
}

static inline void x87_store_m80(X87State& st, word addr, bool pop) {
    const X87Val v = pop ? x87_pop(st) : x87_st(st, 0u);
    store80(addr, v);
}

// ---- 128-bit normalized software floating-point engine (BigF) ----
// Value = (-1)^sign * (hi:lo / 2^127) * 2^exp with bit 63 of hi set when !zero.

struct BigF {
    bool sign;
    bool zero;
    std::int32_t exp;
    std::uint64_t hi;
    std::uint64_t lo;
};

static constexpr BigF BIGF_ZERO{false, true, 0, 0ull, 0ull};
static constexpr BigF BIGF_ONE{false, false, 0, X87_JBIT, 0ull};
static constexpr BigF BIGF_TWO{false, false, 1, X87_JBIT, 0ull};
// Exact 128-bit constants computed to >100 decimal digits:
static constexpr BigF BIGF_LN2{false, false, -1, 0xb17217f7d1cf79abull, 0xc9e3b39803f2f6afull};
static constexpr BigF BIGF_INV_LN2{false, false, 0, 0xb8aa3b295c17f0bbull, 0xbe87fed0691d3e89ull};
static constexpr std::uint64_t SQRT2_HI = 0xb504f333f9de6484ull;
static constexpr std::uint64_t SQRT2_LO = 0x597d89b3754abe9full;

static inline BigF bigf_neg(BigF a) {
    if (!a.zero) a.sign = !a.sign;
    return a;
}

static inline BigF bigf_from_x87(X87Val v) {
    if (x87_is_zero(v)) return {x87_sign(v), true, 0, 0ull, 0ull};
    if (x87_exp(v) != 0u) {
        return {x87_sign(v), false, static_cast<std::int32_t>(x87_exp(v)) - 16383, v.sig, 0ull};
    }
    const int shift = clz64(v.sig);
    return {x87_sign(v), false, 1 - 16383 - shift, v.sig << shift, 0ull};
}

static inline BigF bigf_from_i32(std::int32_t n) {
    if (n == 0) return BIGF_ZERO;
    const bool sign = n < 0;
    const std::uint64_t mag = sign ? (0ull - static_cast<std::uint64_t>(static_cast<std::int64_t>(n)))
                                   : static_cast<std::uint64_t>(n);
    const int shift = clz64(mag);
    return {sign, false, 63 - shift, mag << shift, 0ull};
}

static inline BigF bigf_norm_round(bool sign, std::int32_t exp, std::uint64_t hi, std::uint64_t lo, std::uint64_t guard) {
    if (hi == 0ull && lo == 0ull && guard == 0ull) return BIGF_ZERO;
    while ((hi & X87_JBIT) == 0ull) {
        if (hi == 0ull && lo == 0ull) {
            const int s = clz64(guard);
            hi = guard << s;
            guard = 0ull;
            exp -= 128 + s;
            break;
        }
        if (hi == 0ull) {
            hi = lo;
            lo = guard;
            guard = 0ull;
            exp -= 64;
            continue;
        }
        const int s = clz64(hi);
        hi = (hi << s) | (lo >> (64 - s));
        lo = (lo << s) | (guard >> (64 - s));
        guard <<= s;
        exp -= s;
        break;
    }
    if (guard > X87_JBIT || (guard == X87_JBIT && (lo & 1ull) != 0ull)) {
        ++lo;
        if (lo == 0ull) {
            ++hi;
            if (hi == 0ull) {
                hi = X87_JBIT;
                ++exp;
            }
        }
    }
    return {sign, false, exp, hi, lo};
}

static inline BigF bigf_add(BigF a, BigF b) {
    if (a.zero) return b;
    if (b.zero) return a;
    const bool a_lt_b = (a.exp < b.exp) || (a.exp == b.exp && (a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo)));
    if (a_lt_b) {
        const BigF t = a;
        a = b;
        b = t;
    }
    const std::int32_t diff = a.exp - b.exp;
    if (diff >= 130) return a;
    std::uint64_t bhi = b.hi, blo = b.lo, bg = 0ull;
    if (diff > 0) {
        if (diff >= 128) {
            const int m = diff - 128;
            const std::uint64_t sticky = (blo != 0ull || (m != 0 && (bhi << (64 - m)) != 0ull)) ? 1ull : 0ull;
            bg = ((m != 0) ? (bhi >> m) : bhi) | sticky;
            bhi = 0ull;
            blo = 0ull;
        } else if (diff >= 64) {
            const int m = diff - 64;
            const std::uint64_t sticky = (m != 0 && (blo << (64 - m)) != 0ull) ? 1ull : 0ull;
            bg = ((m != 0) ? ((blo >> m) | (bhi << (64 - m))) : blo) | sticky;
            blo = (m != 0) ? (bhi >> m) : bhi;
            bhi = 0ull;
        } else {
            const std::uint64_t sticky = 0ull;
            bg = (blo << (64 - diff)) | sticky;
            blo = (blo >> diff) | (bhi << (64 - diff));
            bhi >>= diff;
        }
    }
    std::int32_t exp = a.exp;
    if (a.sign == b.sign) {
        const std::uint64_t g = bg;
        const std::uint64_t lo = a.lo + blo;
        const std::uint64_t c0 = (lo < a.lo) ? 1ull : 0ull;
        const std::uint64_t hi_t = a.hi + bhi;
        const std::uint64_t c1 = (hi_t < a.hi) ? 1ull : 0ull;
        const std::uint64_t hi = hi_t + c0;
        const std::uint64_t carry = c1 + ((hi < hi_t) ? 1ull : 0ull);
        if (carry != 0ull) {
            const std::uint64_t ng = (g >> 1) | (lo << 63) | (g & 1ull);
            const std::uint64_t nlo = (lo >> 1) | (hi << 63);
            const std::uint64_t nhi = (hi >> 1) | X87_JBIT;
            return bigf_norm_round(a.sign, exp + 1, nhi, nlo, ng);
        }
        return bigf_norm_round(a.sign, exp, hi, lo, g);
    } else {
        const std::uint64_t borrow_g = (bg != 0ull) ? 1ull : 0ull;
        const std::uint64_t g = 0ull - bg;
        const std::uint64_t blo_b = blo + borrow_g;
        const std::uint64_t c_lo = (borrow_g != 0ull && blo_b == 0ull) ? 1ull : 0ull;
        const std::uint64_t borrow_lo = (c_lo != 0ull || a.lo < blo_b) ? 1ull : 0ull;
        const std::uint64_t lo = a.lo - blo_b;
        const std::uint64_t hi = a.hi - bhi - borrow_lo;
        return bigf_norm_round(a.sign, exp, hi, lo, g);
    }
}

static inline BigF bigf_sub(BigF a, BigF b) {
    return bigf_add(a, bigf_neg(b));
}

static inline BigF bigf_mul(BigF a, BigF b) {
    if (a.zero || b.zero) return BIGF_ZERO;
    std::uint64_t hh_hi, hh_lo, hl_hi, hl_lo, lh_hi, lh_lo, ll_hi, ll_lo;
    mul64x64(a.hi, b.hi, hh_hi, hh_lo);
    mul64x64(a.hi, b.lo, hl_hi, hl_lo);
    mul64x64(a.lo, b.hi, lh_hi, lh_lo);
    mul64x64(a.lo, b.lo, ll_hi, ll_lo);

    std::uint64_t w1 = ll_hi;
    std::uint64_t c1 = 0ull;
    w1 += hl_lo; if (w1 < hl_lo) ++c1;
    w1 += lh_lo; if (w1 < lh_lo) ++c1;

    std::uint64_t w2 = hh_lo;
    std::uint64_t c2 = 0ull;
    w2 += hl_hi; if (w2 < hl_hi) ++c2;
    w2 += lh_hi; if (w2 < lh_hi) ++c2;
    w2 += c1;    if (w2 < c1)    ++c2;

    const std::uint64_t w3 = hh_hi + c2;
    std::uint64_t guard = w1 | (ll_lo != 0ull ? 1ull : 0ull);
    std::int32_t exp = a.exp + b.exp;
    if ((w3 & X87_JBIT) != 0ull) {
        return bigf_norm_round(a.sign != b.sign, exp + 1, w3, w2, guard);
    }
    const std::uint64_t nhi = (w3 << 1) | (w2 >> 63);
    const std::uint64_t nlo = (w2 << 1) | (guard >> 63);
    const std::uint64_t ng = (guard << 1) | (ll_lo != 0ull ? 1ull : 0ull);
    return bigf_norm_round(a.sign != b.sign, exp, nhi, nlo, ng);
}

static inline BigF bigf_div_u32(BigF a, std::uint32_t d) {
    if (a.zero) return BIGF_ZERO;
    const std::uint64_t limbs[6] = {
        a.hi >> 32, a.hi & 0xffffffffull, a.lo >> 32, a.lo & 0xffffffffull, 0ull, 0ull
    };
    std::uint64_t q[6]{};
    std::uint64_t rem = 0ull;
    for (int i = 0; i < 6; ++i) {
        const std::uint64_t cur = (rem << 32) | limbs[i];
        q[i] = cur / d;
        rem = cur % d;
    }
    const std::uint64_t hi = (q[0] << 32) | q[1];
    const std::uint64_t lo = (q[2] << 32) | q[3];
    const std::uint64_t guard = (q[4] << 32) | q[5] | (rem != 0ull ? 1ull : 0ull);
    return bigf_norm_round(a.sign, a.exp, hi, lo, guard);
}

static inline BigF bigf_div(BigF a, BigF b) {
    if (a.zero) return BIGF_ZERO;
    std::uint64_t q_hi = 0ull, q_lo = 0ull;
    std::uint64_t r_top = 0ull, r_hi = a.hi, r_lo = a.lo;
    std::int32_t exp = a.exp - b.exp;

    auto ge_b = [&]() -> bool {
        return r_top != 0ull || r_hi > b.hi || (r_hi == b.hi && r_lo >= b.lo);
    };
    auto sub_b = [&]() {
        const std::uint64_t b0 = (r_lo < b.lo) ? 1ull : 0ull;
        const std::uint64_t nlo = r_lo - b.lo;
        const std::uint64_t bhi_b = b.hi + b0;
        const std::uint64_t b1 = ((b0 != 0ull && bhi_b == 0ull) || r_hi < bhi_b) ? 1ull : 0ull;
        r_hi -= bhi_b;
        r_lo = nlo;
        r_top -= b1;
    };
    auto shl_r = [&]() {
        r_top = r_hi >> 63;
        r_hi = (r_hi << 1) | (r_lo >> 63);
        r_lo <<= 1;
    };

    if (!ge_b()) {
        shl_r();
        --exp;
    }
    for (int bit = 127; bit >= 0; --bit) {
        if (ge_b()) {
            sub_b();
            if (bit >= 64) q_hi |= (1ull << (bit - 64));
            else q_lo |= (1ull << bit);
        }
        shl_r();
    }
    std::uint64_t guard = 0ull;
    if (ge_b()) {
        sub_b();
        guard = X87_JBIT;
    }
    if (r_top != 0ull || r_hi != 0ull || r_lo != 0ull) guard |= 1ull;
    return bigf_norm_round(a.sign != b.sign, exp, q_hi, q_lo, guard);
}

static inline X87Val bigf_to_x87(BigF v, std::uint16_t cw, std::uint16_t& sw) {
    if (v.zero) {
        x87_set_c1(sw, false);
        return x87_make_zero(v.sign);
    }
    std::int32_t biased = v.exp + 16383;
    std::uint64_t sig = v.hi;
    std::uint64_t rem = v.lo;
    bool tiny = false;
    if (biased <= 0) {
        shr128_sticky(sig, rem, 1 - biased);
        biased = 0;
        tiny = true;
    }
    const bool above = rem > X87_JBIT;
    const bool half = rem == X87_JBIT;
    const bool inexact = rem != 0ull;
    const unsigned rc = x87_rc(cw);
    bool up = false;
    switch (rc) {
        case 0u: up = above || (half && (sig & 1ull) != 0ull); break;
        case 1u: up = v.sign && inexact; break;
        case 2u: up = !v.sign && inexact; break;
        default: break;
    }
    if (up) {
        ++sig;
        if (sig == 0ull) {
            sig = X87_JBIT;
            ++biased;
        }
    }
    if (biased == 0 && (sig & X87_JBIT) != 0ull) biased = 1;
    if (biased > 0x7ffe) {
        sw |= static_cast<std::uint16_t>(SW_OE | SW_PE);
        const bool to_inf = (rc == 0u) || (rc == 1u && v.sign) || (rc == 2u && !v.sign);
        x87_set_c1(sw, to_inf);
        if (to_inf) return x87_make_inf(v.sign);
        return {0xffffffffffffffffull, static_cast<std::uint16_t>((v.sign ? 0x8000u : 0u) | 0x7ffeu)};
    }
    if (tiny && inexact) sw |= SW_UE;
    if (inexact) sw |= SW_PE;
    x87_set_c1(sw, up);
    if (biased == 0 && sig == 0ull) return x87_make_zero(v.sign);
    return {sig, static_cast<std::uint16_t>((v.sign ? 0x8000u : 0u) | static_cast<std::uint16_t>(biased))};
}

// ---- Core 128-bit transcendental kernels ----

// expm1(u) = e^u - 1 via argument halving (|z| < 2^-8) + 16-term Taylor series +
// exact doubling identity expm1(2t) = 2*expm1(t) + expm1(t)^2.
static inline BigF bigf_expm1(BigF u) {
    if (u.zero) return u;
    int k = 0;
    BigF z = u;
    while (z.exp >= -8 && k < 24) {
        --z.exp;
        ++k;
    }
    BigF term = z;
    BigF sum = z;
    for (std::uint32_t n = 2u; n <= 18u; ++n) {
        term = bigf_div_u32(bigf_mul(term, z), n);
        if (term.zero) break;
        sum = bigf_add(sum, term);
        if (!sum.zero && term.exp < sum.exp - 130) break;
    }
    for (int i = 0; i < k; ++i) {
        BigF two_sum = sum;
        if (!two_sum.zero) ++two_sum.exp;
        sum = bigf_add(two_sum, bigf_mul(sum, sum));
    }
    return sum;
}

// 2 * atanh(s) = 2 * (s + s^3/3 + s^5/5 + ...) for |s| <= 3 - 2*sqrt(2) (~0.1716).
static inline BigF bigf_two_atanh(BigF s) {
    if (s.zero) return BIGF_ZERO;
    const BigF s2 = bigf_mul(s, s);
    BigF power = s;
    BigF sum = s;
    for (std::uint32_t d = 3u; d <= 57u; d += 2u) {
        power = bigf_mul(power, s2);
        if (power.zero) break;
        const BigF term = bigf_div_u32(power, d);
        sum = bigf_add(sum, term);
        if (!sum.zero && term.exp < sum.exp - 130) break;
    }
    if (!sum.zero) ++sum.exp;  // multiply by 2
    return sum;
}

// log2(m * 2^e) for positive BigF v = m * 2^e.
static inline BigF bigf_log2_pos(BigF v) {
    std::int32_t e = v.exp;
    BigF m = {false, false, 0, v.hi, v.lo};  // m in [1, 2)
    if (m.hi > SQRT2_HI || (m.hi == SQRT2_HI && m.lo > SQRT2_LO)) {
        m.exp = -1;  // m in (sqrt(2)/2, 1)
        ++e;
    }
    if (m.exp == 0 && m.hi == X87_JBIT && m.lo == 0ull) {
        return bigf_from_i32(e);
    }
    const BigF num = bigf_sub(m, BIGF_ONE);
    const BigF den = bigf_add(m, BIGF_ONE);
    const BigF ln_m = bigf_two_atanh(bigf_div(num, den));
    const BigF log2_m = bigf_mul(ln_m, BIGF_INV_LN2);
    if (e == 0) return log2_m;
    return bigf_add(bigf_from_i32(e), log2_m);
}

// ---- x87 NaN/unsupported helper ----

static inline bool x87_binary_special(X87Val a, X87Val b, std::uint16_t& sw, X87Val& out) {
    if (x87_unsupported(a) || x87_unsupported(b)) {
        sw |= SW_IE;
        out = X87_INDEF;
        return true;
    }
    const bool na = x87_is_nan(a);
    const bool nb = x87_is_nan(b);
    if (!na && !nb) return false;
    if (x87_is_snan(a) || x87_is_snan(b)) sw |= SW_IE;
    if (na && nb) {
        if (x87_is_qnan(a) != x87_is_qnan(b)) out = x87_is_qnan(a) ? a : b;
        else out = (b.sig > a.sig || (b.sig == a.sig && !x87_sign(b))) ? b : a;
    } else {
        out = na ? a : b;
    }
    out = x87_quiet(out);
    return true;
}

// ---- F2XM1, FYL2X, FYL2XP1 ----

static inline void x87_f2xm1(X87State& st) {
    X87Val& x = x87_st(st, 0u);
    if (x87_unsupported(x)) {
        st.sw |= SW_IE;
        x = X87_INDEF;
        return;
    }
    if (x87_is_nan(x)) {
        if (x87_is_snan(x)) st.sw |= SW_IE;
        x = x87_quiet(x);
        return;
    }
    if (x87_is_zero(x)) {
        // Preserves signed zero: F2XM1(+0.0) == +0.0, F2XM1(-0.0) == -0.0.
        x87_set_c1(st.sw, false);
        return;
    }
    if (x87_is_inf(x)) {
        x87_set_c1(st.sw, false);
        x = x87_sign(x) ? X87_NEG_ONE : X87_POS_INF;
        return;
    }
    if (x87_is_denorm(x)) st.sw |= SW_DE;
    if (x.se == 0x3fffu && x.sig == X87_JBIT) {
        x87_set_c1(st.sw, false);
        x = X87_POS_ONE;
        return;
    }
    if (x.se == 0xbfffu && x.sig == X87_JBIT) {
        x87_set_c1(st.sw, false);
        x = X87_NEG_HALF;
        return;
    }
    const BigF u = bigf_mul(bigf_from_x87(x), BIGF_LN2);
    const BigF r = bigf_expm1(u);
    x = bigf_to_x87(r, st.cw, st.sw);
}

static inline void x87_fyl2x(X87State& st) {
    const X87Val x = x87_pop(st);  // ST(0)
    X87Val& y = x87_st(st, 0u);    // Former ST(1)
    X87Val special{};
    if (x87_binary_special(y, x, st.sw, special)) {
        y = special;
        return;
    }
    const bool x_zero = x87_is_zero(x);
    const bool y_zero = x87_is_zero(y);
    const bool x_inf = x87_is_inf(x);
    const bool y_inf = x87_is_inf(y);
    if (x87_sign(x) && !x_zero) {
        // Negative base (finite, subnormal, or -inf): #IA -> indefinite.
        st.sw |= SW_IE;
        y = X87_INDEF;
        return;
    }
    if (x_zero) {
        if (y_zero) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (y_inf) {
            x87_set_c1(st.sw, false);
            y = x87_make_inf(!x87_sign(y));
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        st.sw |= SW_ZE;
        x87_set_c1(st.sw, false);
        y = x87_make_inf(!x87_sign(y));
        return;
    }
    if (x_inf) {
        if (y_zero) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        x87_set_c1(st.sw, false);
        y = x87_make_inf(x87_sign(y));
        return;
    }
    if (x.se == 0x3fffu && x.sig == X87_JBIT) {
        // x == +1.0 -> log2(1) == +0.0.
        if (y_inf) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        x87_set_c1(st.sw, false);
        y = x87_make_zero(x87_sign(y));
        return;
    }
    if (x87_is_denorm(x) || x87_is_denorm(y)) st.sw |= SW_DE;
    const bool log_neg = x87_exp(x) < 0x3fffu;
    if (y_zero) {
        x87_set_c1(st.sw, false);
        y = x87_make_zero(x87_sign(y) != log_neg);
        return;
    }
    if (y_inf) {
        x87_set_c1(st.sw, false);
        y = x87_make_inf(x87_sign(y) != log_neg);
        return;
    }
    const BigF log2_x = bigf_log2_pos(bigf_from_x87(x));
    const BigF prod = bigf_mul(bigf_from_x87(y), log2_x);
    y = bigf_to_x87(prod, st.cw, st.sw);
}

static inline void x87_fyl2xp1(X87State& st) {
    const X87Val x = x87_pop(st);  // ST(0)
    X87Val& y = x87_st(st, 0u);    // Former ST(1)
    X87Val special{};
    if (x87_binary_special(y, x, st.sw, special)) {
        y = special;
        return;
    }
    const bool x_zero = x87_is_zero(x);
    const bool y_zero = x87_is_zero(y);
    const bool x_inf = x87_is_inf(x);
    const bool y_inf = x87_is_inf(y);
    if (x_zero) {
        if (y_inf) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        x87_set_c1(st.sw, false);
        y = x87_make_zero(x87_sign(y) != x87_sign(x));
        return;
    }
    if (x.se == 0xbfffu && x.sig == X87_JBIT) {
        // x == -1.0 -> log2(0) == -inf (#Z for finite nonzero y; #IA for y == +-0).
        if (y_zero) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (y_inf) {
            x87_set_c1(st.sw, false);
            y = x87_make_inf(!x87_sign(y));
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        st.sw |= SW_ZE;
        x87_set_c1(st.sw, false);
        y = x87_make_inf(!x87_sign(y));
        return;
    }
    if (x87_sign(x) && (x_inf || x87_exp(x) > 0x3fffu || (x87_exp(x) == 0x3fffu && x.sig > X87_JBIT))) {
        // x < -1.0 -> log2(x + 1) domain error (#IA).
        st.sw |= SW_IE;
        y = X87_INDEF;
        return;
    }
    if (x_inf) {
        if (y_zero) {
            st.sw |= SW_IE;
            y = X87_INDEF;
            return;
        }
        if (x87_is_denorm(y)) st.sw |= SW_DE;
        x87_set_c1(st.sw, false);
        y = x87_make_inf(x87_sign(y));
        return;
    }
    if (x87_is_denorm(x) || x87_is_denorm(y)) st.sw |= SW_DE;
    const bool log_neg = x87_sign(x);
    if (y_zero) {
        x87_set_c1(st.sw, false);
        y = x87_make_zero(x87_sign(y) != log_neg);
        return;
    }
    if (y_inf) {
        x87_set_c1(st.sw, false);
        y = x87_make_inf(x87_sign(y) != log_neg);
        return;
    }
    const BigF bx = bigf_from_x87(x);
    BigF log2_xp1{};
    if (bx.exp <= -2) {
        // |x| < 0.5 (covers entire SDM domain |x| <= 1 - sqrt(2)/2 without cancellation):
        // log2(1 + x) = 2 * atanh(x / (2 + x)) / ln(2).
        const BigF s = bigf_div(bx, bigf_add(BIGF_TWO, bx));
        log2_xp1 = bigf_mul(bigf_two_atanh(s), BIGF_INV_LN2);
    } else {
        log2_xp1 = bigf_log2_pos(bigf_add(BIGF_ONE, bx));
    }
    const BigF prod = bigf_mul(bigf_from_x87(y), log2_xp1);
    y = bigf_to_x87(prod, st.cw, st.sw);
}
"""


def flat_address(ins, *, width, store=False):
    """Lower a flat zero-displacement [rcx|rdx|r8] memory operand of exact byte width."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or not RegisterExt.is_gpr64(base)
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if MemorySizeExt.size(ins.memory_size) != width:
        raise ValueError(f"unsupported memory width at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"store outside output buffer r8 at RVA {ins.ip:#x}: {ins}")
    disp = ins.memory_displacement & ((1 << 64) - 1)
    if disp != 0:
        raise ValueError(f"unsupported displaced memory operand at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def _has_vector_register(ins):
    for i in range(ins.op_count):
        if ins.op_kind(i) == OpKind.REGISTER:
            reg = ins.op_register(i)
            if RegisterExt.is_vector_register(reg):
                return True
    return False


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line x87 exp/log leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    depth = 0
    ax_defined = False
    used_ax = False
    statements = []
    returned = False

    def push_check(ins):
        nonlocal depth
        if depth >= 8:
            raise ValueError(f"x87 stack overflow at RVA {ins.ip:#x}: {ins}")
        depth += 1

    def need_depth(ins, n):
        if depth < n:
            raise ValueError(f"x87 stack underflow at RVA {ins.ip:#x}: {ins}")

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if (
            ins.has_lock_prefix
            or ins.has_rep_prefix
            or ins.has_repne_prefix
            or ins.has_segment_prefix
            or ins.has_xacquire_prefix
            or ins.has_xrelease_prefix
        ):
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        op_code = ins.op_code()
        if (
            op_code.encoding == EncodingKind.EVEX
            or ins.op_mask != R.NONE
            or ins.zeroing_masking
            or ins.is_broadcast
        ):
            raise ValueError(f"unsupported EVEX encoding at RVA {ins.ip:#x}: {ins}")
        if op_code.encoding == EncodingKind.VEX:
            raise ValueError(f"unsupported VEX encoding at RVA {ins.ip:#x}: {ins}")
        if op_code.encoding != EncodingKind.LEGACY:
            raise ValueError(f"unsupported non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if _has_vector_register(ins):
            raise ValueError(f"unsupported vector instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in TRIG_REJECT_CODES:
            raise ValueError(f"unsupported x87 trigonometric instruction at RVA {ins.ip:#x}: {ins}")
        if ins.flow_control not in (FlowControl.NEXT, FlowControl.RETURN):
            raise ValueError(f"unsupported control-flow instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if depth != 0:
                raise ValueError(f"non-empty x87 stack at ret (depth={depth}) at RVA {ins.ip:#x}")
            statements.append("return;")
            returned = True

        elif ins.code == Code.FNINIT:
            depth = 0
            statements.append("x87_fninit(x87);")

        elif ins.code == Code.FLDCW_M2BYTE:
            addr = flat_address(ins, width=2, store=False)
            statements.append(f"x87.cw = load16({addr});")

        elif ins.code == Code.FNSTCW_M2BYTE:
            addr = flat_address(ins, width=2, store=True)
            statements.append(f"store16({addr}, x87.cw);")

        elif ins.code == Code.FNSTSW_AX:
            ax_defined = True
            used_ax = True
            statements.append("ax = x87_status_word(x87);")

        elif ins.code == Code.MOV_RM16_R16:
            if ins.op0_kind != OpKind.MEMORY or ins.op1_kind != OpKind.REGISTER or ins.op1_register != R.AX:
                raise ValueError(f"unsupported mov16 form at RVA {ins.ip:#x}: {ins}")
            if not ax_defined:
                raise ValueError(f"undefined register ax at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, width=2, store=True)
            statements.append(f"store16({addr}, ax);")

        elif ins.code in FLD_MEM_CODES:
            width, helper = FLD_MEM_CODES[ins.code]
            addr = flat_address(ins, width=width, store=False)
            push_check(ins)
            statements.append(f"{helper}(x87, {addr});")

        elif ins.code == Code.FLD1:
            push_check(ins)
            statements.append("x87_push(x87, X87_POS_ONE);")

        elif ins.code == Code.FLDZ:
            push_check(ins)
            statements.append("x87_push(x87, X87_POS_ZERO);")

        elif ins.code in FST_MEM_CODES:
            width, helper, pop = FST_MEM_CODES[ins.code]
            need_depth(ins, 1)
            addr = flat_address(ins, width=width, store=True)
            if pop:
                depth -= 1
            statements.append(f"{helper}(x87, {addr}, {'true' if pop else 'false'});")

        elif ins.code == Code.FSTP_STI:
            if ins.op0_kind != OpKind.REGISTER or ins.op0_register != R.ST0:
                raise ValueError(f"unsupported fstp register operand at RVA {ins.ip:#x}: {ins}")
            need_depth(ins, 1)
            depth -= 1
            statements.append("(void)x87_pop(x87);")

        elif ins.code == Code.F2XM1:
            need_depth(ins, 1)
            statements.append("x87_f2xm1(x87);")

        elif ins.code == Code.FYL2X:
            need_depth(ins, 2)
            depth -= 1
            statements.append("x87_fyl2x(x87);")

        elif ins.code == Code.FYL2XP1:
            need_depth(ins, 2)
            depth -= 1
            statements.append("x87_fyl2xp1(x87);")

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = ["    X87State x87{};"]
    if used_ax:
        decls.append("    std::uint16_t ax = 0u;")
    body = [f"    {stmt}" for stmt in statements]
    lines = [
        f'extern "C" __declspec(dllexport) void lifted_{name}(word rcx, word rdx, word r8) {{',
        "    (void)rcx; (void)rdx; (void)r8;",
        *decls,
        *body,
        "}",
    ]
    return "\n".join(lines), len(instructions)


def lift(pe, name):
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names):
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image, names):
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        source, meta = translate(args.image, args.exports.split(","))
        args.output.write_text(source, encoding="utf-8")
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
