"""Lift bounded Windows x64 leaf functions using x87 environment/state ops to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat zero-displacement host pointers to caller-owned
buffers (rcx, rdx, and r8 may be read; only r8 may be written).

All x87 values use a portable software 80-bit extended-precision representation
(explicit 64-bit significand + 15-bit biased exponent + sign bit) and pure integer
128-bit arithmetic (zero host float/double/long double or SSE/x87 instructions).

Supported x87 environment and state instructions (docs/coverage/x64_zen2.tsv group x87):
  - FDECSTP        : TOP <- (TOP - 1) & 7 and clears FSW.C1; FTW and registers unchanged.
                     Measured x64 status profile: preserves C0/C2/C3 (SDM leaves C0/C2/C3
                     undefined).
  - FNSTENV [r8]   : Store 28-byte 32-bit protected-mode environment, then mask FCW[5:0].
                     Leaves C0..C3 undefined for subsequent readers until re-initialized.
  - FSTENV [r8]    : WAIT immediately followed by FNSTENV [r8].
  - FLDENV [src]   : Restore 28-byte 32-bit protected-mode environment (FCW/FSW/FTW/FIP/FCS/FOP/FDP/FDS).
  - FNSAVE [r8]    : Store 28-byte env + logical ST(0)..ST(7) ((TOP+i)&7), then FNINIT.
  - FSAVE [r8]     : WAIT immediately followed by FNSAVE [r8].
  - FRSTOR [src]   : Restore 28-byte env + logical ST(0)..ST(7) ((TOP+i)&7).

Metadata and safety profile (`preserved-metadata` / masked-exception):
  - `preserved-metadata`: explicit FLDENV/FRSTOR -> FNSTENV/FNSAVE round-trips preserve
    FIP/FCS/FOP/FDP/FDS (with FOP masked to 11 bits); masked non-control/stack ops
    (FDECSTP, FLD, FLD1, FLDZ, FSTP) preserve the loaded/initialized FIP/FCS/FOP/FDP/FDS
    (matching the profile measured on the i7-13620H; hosts with CPUID.(EAX=7,ECX=0):EBX[13]=1
    zero FCS/FDS on hardware, and AMD/other parts may update FOP/FIP/FCS on FDECSTP and
    masked non-control ops).
  - Measured x64 status profile: WAIT, FLDCW, FNSTCW, and FNSTSW AX preserve C0..C3 loaded
    by FLDENV/FRSTOR (whereas SDM leaves C0..C3 undefined after WAIT/FLDCW/FNSTCW/FNSTSW).
  - FNINIT/FNSAVE sets FTW=0xFFFF without clearing physical register contents. FLDENV
    with non-empty FTW bits on physical registers not defined in the current function
    traps via x87_trap() (`__ud2()`).
  - Masked-exception precondition: WAIT, FDECSTP, FLD*, and FSTP* trap via x87_trap()
    when any exception is unmasked (FCW[5:0] != 0x3F) or SW.ES is set; FLDENV/FRSTOR
    also trap eagerly if SW.ES or an unmasked pending exception is loaded (stricter than
    hardware, which defers #MF until the next waiting x87 instruction).
  - Uninitialized reads of FCW (FNSTCW before FNINIT/FLDCW/FLDENV/FRSTOR), FSW/env
    (FNSTSW/FNSTENV/FNSAVE/FDECSTP before FNINIT/FLDENV/FRSTOR), and readers of
    undefined C0..C3 after FNSTENV/FSTENV are statically rejected.

Bounded leaf x87 stack/control/status glue:
  - fninit, fldcw m16, fnstcw m16, fnstsw ax, mov word ptr [r8], ax
  - fld m32 / m64 / m80, fld1, fldz
  - fstp m32 / m64 / m80 / st(0)
  - terminal ret (requires statically known empty x87 stack)

16-bit operand-size env/state forms (o16: 14/94-byte), standalone WAIT or WAIT
before any instruction other than FNSTENV/FNSAVE, stack overflow/underflow,
non-empty or unknown x87 stack at ret, uninitialized CW/SW/env reads, FSW reads
after FNSTENV/FSTENV before re-init, XMM/YMM/ZMM/VEX/EVEX encodings, prefixes,
non-flat, displaced, or indexed memory, stores outside [r8], branches, undefined
ax reads, missing ret, and trailing instructions after ret are rejected.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
import struct
import tempfile

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

O16_ENV_CODES = {
    Code.FLDENV_M14BYTE,
    Code.FNSTENV_M14BYTE,
    Code.FRSTOR_M94BYTE,
    Code.FNSAVE_M94BYTE,
}

FLD_MEM_CODES = {
    Code.FLD_M32FP: (4, "x87_load_f32"),
    Code.FLD_M64FP: (8, "x87_load_f64"),
    Code.FLD_M80FP: (10, "x87_load_m80"),
}

FSTP_MEM_CODES = {
    Code.FSTP_M32FP: (4, "x87_store_f32"),
    Code.FSTP_M64FP: (8, "x87_store_f64"),
    Code.FSTP_M80FP: (10, "x87_store_m80"),
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.WAIT,
    Code.FDECSTP,
    Code.FLDENV_M28BYTE,
    Code.FNSTENV_M28BYTE,
    Code.FRSTOR_M108BYTE,
    Code.FNSAVE_M108BYTE,
    Code.FNINIT,
    Code.FLDCW_M2BYTE,
    Code.FNSTCW_M2BYTE,
    Code.FNSTSW_AX,
    Code.MOV_RM16_R16,
    Code.FLD1,
    Code.FLDZ,
    *FLD_MEM_CODES,
    *FSTP_MEM_CODES,
    Code.FSTP_STI,
}


PRELUDE = r"""#include <atomic>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#include <xmmintrin.h>
#endif

static inline void x87_trap() {
#if defined(_MSC_VER)
    __ud2();
#else
    __builtin_trap();
#endif
}

#if defined(_MSC_VER)
// Test-only host MXCSR accessors used by tests/cpu/x87_env_aot_check.py to
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
static constexpr X87Val X87_POS_ZERO{0ull, 0x0000u};
static constexpr X87Val X87_POS_ONE{X87_JBIT, 0x3fffu};

static constexpr std::uint16_t SW_IE = 0x0001u;
static constexpr std::uint16_t SW_DE = 0x0002u;
static constexpr std::uint16_t SW_OE = 0x0008u;
static constexpr std::uint16_t SW_UE = 0x0010u;
static constexpr std::uint16_t SW_PE = 0x0020u;
static constexpr std::uint16_t SW_C1 = 0x0200u;

struct X87State {
    X87Val regs[8]{};
    std::uint16_t cw = 0x027fu;
    std::uint16_t sw = 0x0000u;
    std::uint16_t tw = 0xffffu;
    std::uint32_t fip = 0u;
    std::uint16_t fcs = 0u;
    std::uint16_t fop = 0u;
    std::uint32_t fdp = 0u;
    std::uint16_t fds = 0u;
    unsigned top = 0u;
    std::uint8_t regs_defined = 0u;
};

static inline void x87_check_mf(const X87State& st) {
    // Masked-exception precondition: non-control ops and WAIT trap if any
    // exception is unmasked (FCW[5:0] != 0x3F) or SW.ES (bit 7) is set.
    if ((st.cw & 0x003fu) != 0x003fu || (st.sw & 0x0080u) != 0u || (st.sw & ~st.cw & 0x003fu) != 0u) {
        x87_trap();
    }
}

static inline void x87_wait(const X87State& st) {
    x87_check_mf(st);
}

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
static inline bool x87_is_inf(X87Val v) { return x87_exp(v) == 0x7fffu && v.sig == X87_JBIT; }
static inline bool x87_is_zero(X87Val v) { return x87_exp(v) == 0u && v.sig == 0ull; }
static inline X87Val x87_quiet(X87Val v) { v.sig |= X87_QBIT; return v; }

static inline unsigned x87_tag(X87Val v) {
    if (x87_is_zero(v)) return 1u;
    if (x87_exp(v) == 0u || x87_exp(v) == 0x7fffu || (v.sig & X87_JBIT) == 0ull) return 2u;
    return 0u;
}

static inline void x87_set_tag(std::uint16_t& tw, unsigned phys, unsigned tag) {
    const unsigned shift = (phys & 7u) * 2u;
    tw = static_cast<std::uint16_t>((tw & ~(3u << shift)) | ((tag & 3u) << shift));
}

static inline void x87_set_c1(std::uint16_t& sw, bool up) {
    if (up) sw |= SW_C1;
    else sw &= static_cast<std::uint16_t>(~SW_C1);
}

static inline void x87_push(X87State& st, X87Val v) {
    x87_check_mf(st);
    st.top = (st.top - 1u) & 7u;
    st.regs[st.top] = v;
    st.regs_defined |= static_cast<std::uint8_t>(1u << st.top);
    x87_set_tag(st.tw, st.top, x87_tag(v));
    x87_set_c1(st.sw, false);
}

static inline X87Val x87_pop(X87State& st) {
    x87_check_mf(st);
    const X87Val v = st.regs[st.top];
    x87_set_tag(st.tw, st.top, 3u);
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
    // SDM: FNINIT sets FTW=0xFFFF without clearing physical register contents.
    st.cw = 0x037fu;
    st.sw = 0x0000u;
    st.tw = 0xffffu;
    st.fip = 0u;
    st.fcs = 0u;
    st.fop = 0u;
    st.fdp = 0u;
    st.fds = 0u;
    st.top = 0u;
}

static inline void x87_fdecstp(X87State& st) {
    x87_check_mf(st);
    st.top = (st.top - 1u) & 7u;
    x87_set_c1(st.sw, false);
}

// ---- 28-byte and 108-byte 32-bit protected-mode environment / state ----

static inline void x87_store_env28(const X87State& st, word addr) {
    store16(addr + 0ull, st.cw);
    store16(addr + 2ull, 0xffffu);
    store16(addr + 4ull, x87_status_word(st));
    store16(addr + 6ull, 0xffffu);
    store16(addr + 8ull, st.tw);
    store16(addr + 10ull, 0xffffu);
    store32(addr + 12ull, st.fip);
    store16(addr + 16ull, st.fcs);
    store16(addr + 18ull, static_cast<std::uint16_t>(st.fop & 0x07ffu));
    store32(addr + 20ull, st.fdp);
    store16(addr + 24ull, st.fds);
    store16(addr + 26ull, 0xffffu);
}

static inline void x87_recompute_tags(X87State& st, std::uint16_t raw_tw) {
    std::uint16_t out = 0xffffu;
    for (unsigned phys = 0u; phys < 8u; ++phys) {
        const unsigned tag = (raw_tw >> (phys * 2u)) & 3u;
        if (tag != 3u) {
            x87_set_tag(out, phys, x87_tag(st.regs[phys]));
        }
    }
    st.tw = out;
}

static inline std::uint16_t x87_load_env28(X87State& st, word addr) {
    st.cw = load16(addr + 0ull);
    const std::uint16_t raw_sw = load16(addr + 4ull);
    st.top = (raw_sw >> 11) & 7u;
    st.sw = static_cast<std::uint16_t>(raw_sw & ~0x3800u);
    const std::uint16_t raw_tw = load16(addr + 8ull);
    st.fip = load32(addr + 12ull);
    st.fcs = load16(addr + 16ull);
    st.fop = static_cast<std::uint16_t>(load16(addr + 18ull) & 0x07ffu);
    st.fdp = load32(addr + 20ull);
    st.fds = load16(addr + 24ull);
    return raw_tw;
}

static inline void x87_fnstenv(X87State& st, word addr) {
    x87_store_env28(st, addr);
    // SDM: FNSTENV/FSTENV masks all 6 exception bits in FCW after saving.
    st.cw |= 0x003fu;
}

static inline void x87_fldenv(X87State& st, word addr) {
    const std::uint16_t raw_tw = x87_load_env28(st, addr);
    if ((st.sw & 0x0080u) != 0u || (st.sw & ~st.cw & 0x003fu) != 0u) {
        x87_trap();
    }
    for (unsigned phys = 0u; phys < 8u; ++phys) {
        const unsigned tag = (raw_tw >> (phys * 2u)) & 3u;
        if (tag != 3u && (st.regs_defined & (1u << phys)) == 0u) {
            // Physical register was not initialized in this function; FNINIT does not clear regs.
            x87_trap();
        }
    }
    x87_recompute_tags(st, raw_tw);
}

static inline void x87_fnsave(X87State& st, word addr) {
    x87_store_env28(st, addr);
    for (unsigned i = 0u; i < 8u; ++i) {
        store80(addr + 28ull + static_cast<word>(i) * 10ull, x87_st(st, i));
    }
    x87_fninit(st);
}

static inline void x87_frstor(X87State& st, word addr) {
    const std::uint16_t raw_tw = x87_load_env28(st, addr);
    if ((st.sw & 0x0080u) != 0u || (st.sw & ~st.cw & 0x003fu) != 0u) {
        x87_trap();
    }
    for (unsigned i = 0u; i < 8u; ++i) {
        x87_st(st, i) = load80(addr + 28ull + static_cast<word>(i) * 10ull);
    }
    st.regs_defined = 0xffu;
    x87_recompute_tags(st, raw_tw);
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
    x87_check_mf(st);
    x87_push(st, x87_from_binary(load32(addr), 23, 8, 127, st.sw));
}

static inline void x87_load_f64(X87State& st, word addr) {
    x87_check_mf(st);
    x87_push(st, x87_from_binary(load64(addr), 52, 11, 1023, st.sw));
}

static inline void x87_load_m80(X87State& st, word addr) {
    x87_check_mf(st);
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

static inline void x87_store_f32(X87State& st, word addr) {
    const X87Val v = x87_pop(st);
    store32(addr, static_cast<std::uint32_t>(x87_to_binary(v, 23, 8, 127, 0xfe, st.cw, st.sw)));
}

static inline void x87_store_f64(X87State& st, word addr) {
    const X87Val v = x87_pop(st);
    store64(addr, x87_to_binary(v, 52, 11, 1023, 0x7fe, st.cw, st.sw));
}

static inline void x87_store_m80(X87State& st, word addr) {
    store80(addr, x87_pop(st));
}
"""


def flat_address(ins, *, width: int, store: bool = False) -> str:
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


def _has_vector_register(ins) -> bool:
    for i in range(ins.op_count):
        if ins.op_kind(i) == OpKind.REGISTER and RegisterExt.is_vector_register(ins.op_register(i)):
            return True
    return False


def lift_code(name: str, start: int, raw: bytes) -> tuple[str, int]:
    """Decode and lift one bounded straight-line x87 environment/state leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    known = True
    top = 0
    occupied = 0
    cw_known = False
    env_known = False
    cc_known = False
    ax_defined = False
    used_ax = False
    statements: list[str] = []
    returned = False

    def push_check(ins) -> None:
        nonlocal top, occupied
        if not known:
            raise ValueError(f"x87 stack overflow possible after FLDENV/FRSTOR at RVA {ins.ip:#x}: {ins}")
        next_top = (top - 1) & 7
        if (occupied >> next_top) & 1:
            raise ValueError(f"x87 stack overflow at RVA {ins.ip:#x}: {ins}")
        top = next_top
        occupied |= 1 << top

    def pop_check(ins) -> None:
        nonlocal top, occupied
        if not known:
            raise ValueError(f"x87 stack underflow possible after FLDENV/FRSTOR at RVA {ins.ip:#x}: {ins}")
        if not ((occupied >> top) & 1):
            raise ValueError(f"x87 stack underflow at RVA {ins.ip:#x}: {ins}")
        occupied &= ~(1 << top)
        top = (top + 1) & 7

    for idx, ins in enumerate(instructions):
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
        if ins.flow_control not in (FlowControl.NEXT, FlowControl.RETURN):
            raise ValueError(f"unsupported control-flow instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in O16_ENV_CODES:
            raise ValueError(f"unsupported 16-bit x87 environment/state operand at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            if not known or occupied != 0:
                raise ValueError(f"non-empty x87 stack at ret at RVA {ins.ip:#x}")
            statements.append("return;")
            returned = True

        elif ins.code == Code.WAIT:
            if idx + 1 >= len(instructions) or instructions[idx + 1].code not in (
                Code.FNSTENV_M28BYTE,
                Code.FNSAVE_M108BYTE,
            ):
                raise ValueError(
                    f"unsupported WAIT not immediately before FNSTENV/FNSAVE at RVA {ins.ip:#x}: {ins}"
                )
            statements.append("x87_wait(x87);")

        elif ins.code == Code.FDECSTP:
            if not env_known:
                raise ValueError(f"read of uninitialized x87 environment before fninit/fldenv/frstor at RVA {ins.ip:#x}: {ins}")
            if not cc_known:
                raise ValueError(f"undefined x87 condition codes after FNSTENV at RVA {ins.ip:#x}: {ins}")
            if known:
                top = (top - 1) & 7
            statements.append("x87_fdecstp(x87);")

        elif ins.code == Code.FNINIT:
            known = True
            top = 0
            occupied = 0
            cw_known = True
            env_known = True
            cc_known = True
            statements.append("x87_fninit(x87);")

        elif ins.code == Code.FLDENV_M28BYTE:
            addr = flat_address(ins, width=28, store=False)
            known = False
            cw_known = True
            env_known = True
            cc_known = True
            statements.append(f"x87_fldenv(x87, {addr});")

        elif ins.code == Code.FNSTENV_M28BYTE:
            if not env_known:
                raise ValueError(f"read of uninitialized x87 environment before fninit/fldenv/frstor at RVA {ins.ip:#x}: {ins}")
            if not cc_known:
                raise ValueError(f"undefined x87 condition codes after FNSTENV at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, width=28, store=True)
            cc_known = False
            statements.append(f"x87_fnstenv(x87, {addr});")

        elif ins.code == Code.FRSTOR_M108BYTE:
            addr = flat_address(ins, width=108, store=False)
            known = False
            cw_known = True
            env_known = True
            cc_known = True
            statements.append(f"x87_frstor(x87, {addr});")

        elif ins.code == Code.FNSAVE_M108BYTE:
            if not env_known:
                raise ValueError(f"read of uninitialized x87 environment before fninit/fldenv/frstor at RVA {ins.ip:#x}: {ins}")
            if not cc_known:
                raise ValueError(f"undefined x87 condition codes after FNSTENV at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, width=108, store=True)
            known = True
            top = 0
            occupied = 0
            cw_known = True
            env_known = True
            cc_known = True
            statements.append(f"x87_fnsave(x87, {addr});")

        elif ins.code == Code.FLDCW_M2BYTE:
            addr = flat_address(ins, width=2, store=False)
            cw_known = True
            statements.append(f"x87.cw = load16({addr});")

        elif ins.code == Code.FNSTCW_M2BYTE:
            if not cw_known:
                raise ValueError(f"read of uninitialized x87 control word at RVA {ins.ip:#x}: {ins}")
            addr = flat_address(ins, width=2, store=True)
            statements.append(f"store16({addr}, x87.cw);")

        elif ins.code == Code.FNSTSW_AX:
            if not env_known:
                raise ValueError(f"read of uninitialized x87 status word before fninit/fldenv/frstor at RVA {ins.ip:#x}: {ins}")
            if not cc_known:
                raise ValueError(f"undefined x87 condition codes after FNSTENV at RVA {ins.ip:#x}: {ins}")
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

        elif ins.code in FSTP_MEM_CODES:
            width, helper = FSTP_MEM_CODES[ins.code]
            pop_check(ins)
            addr = flat_address(ins, width=width, store=True)
            statements.append(f"{helper}(x87, {addr});")

        elif ins.code == Code.FSTP_STI:
            if ins.op0_kind != OpKind.REGISTER or ins.op0_register != R.ST0:
                raise ValueError(f"unsupported fstp register operand at RVA {ins.ip:#x}: {ins}")
            pop_check(ins)
            statements.append("(void)x87_pop(x87);")

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


def lift(pe, name: str) -> tuple[str, int, int]:
    entry = export_rva(pe, name)
    raw = bounded_code(pe, entry)
    source, count = lift_code(name, entry, raw)
    return source, entry, count


def translate_pe(pe, names: Sequence[str]) -> tuple[str, list[tuple[str, int, int]]]:
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.Magic != 0x20B:
        raise ValueError("expected AMD64 PE32+")
    lifted = [lift(pe, name) for name in names]
    source = PRELUDE + "\n".join(f"\n{text}" for text, _, _ in lifted) + "\n"
    meta = [(name, entry, count) for name, (_, entry, count) in zip(names, lifted)]
    return source, meta


def translate(image: Path, names: Sequence[str]) -> tuple[str, list[tuple[str, int, int]]]:
    with pefile.PE(str(image), fast_load=True) as pe:
        return translate_pe(pe, names)


def _validate_paths(image: Path, output: Path, exports: list[str], parser: argparse.ArgumentParser) -> None:
    if not any(exports):
        raise ValueError("exports must not be empty")
    if image.is_symlink():
        raise ValueError(f"image path must not be a symlink: {image}")
    if image.is_dir():
        raise IsADirectoryError(f"image path is a directory: {image}")
    if not image.is_file():
        raise FileNotFoundError(f"image file not found: {image}")
    if output.suffix.lower() != ".cpp":
        parser.error("choose a distinct .cpp output")
    if output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output}")
    if output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output}")
    for parent in output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if output.resolve() == image.resolve() or (output.exists() and image.samefile(output)):
        parser.error("choose a distinct .cpp output")


def _write_atomic(path: Path, content: str) -> None:
    """Write UTF-8 content to a sibling temp file and atomically replace path, leaving 0 temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        tmp_path.replace(path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        exports = [item for item in args.exports.split(",") if item]
        _validate_paths(args.image, args.output, exports, parser)
        source, meta = translate(args.image, exports)
        _write_atomic(args.output, source)
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, struct.error, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
