"""Lift bounded Windows x64 leaf functions with legacy SSE4.1 FP ops to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned buffers. XMM0..XMM5
are modeled as portable 128-bit value structs (two 64-bit lanes); volatile scratch
GPRs (rax, r9, r10, r11) are modeled as 64-bit words for EXTRACTPS and 32/64-bit
mov glue.

Supported legacy SSE4.1 FP encodings (docs/coverage/x64_zen2.tsv group SSE3-4):
  - ROUNDPS, ROUNDPD: imm8[1:0] when imm8[2]==0 else host MXCSR RC; handles -0.0,
    +/-Inf, qNaN/sNaN quieting, and MXCSR DAZ.
  - ROUNDSS, ROUNDSD: scalar rounding preserving destination bits [127:32]/[127:64].
  - DPPS, DPPD: dot product using source mask imm8[7:4]/[5:4] and destination mask
    imm8[3:0]/[1:0] (disabled outputs set to +0.0), executing under host MXCSR
    (rounding mode, DAZ, FTZ).
  - INSERTPS: inserts count_s=imm8[7:6] from xmm (or lane 0 from m32) into
    count_d=imm8[5:4] of dst, then applies zmask=imm8[3:0] last.
  - EXTRACTPS: extracts lane imm8[1:0] to scratch r32/r64 (zero-extending [63:32])
    or flat m32 [r8].

Transport / ABI glue:
  - MOVAPS, MOVAPD, MOVDQA (128-bit reg-reg, 16-byte-aligned flat loads & [r8] stores)
  - MOVUPS, MOVUPD, MOVDQU (128-bit reg-reg, flat loads & [r8] stores)
  - MOVSS, MOVSD (scalar load/store/reg-merge)
  - MOV (32/64-bit scratch GPR reg-reg, flat load, [r8] store) and terminal RET.

VEX/EVEX encodings, nonvolatile XMM6..15 / GPRs, writes to argument GPRs, undefined
reads, unaligned 128-bit aligned-only memory operands, non-flat memory operands,
non-[r8] stores, prefixes, branches, and trailing or missing ret are rejected.
"""

import argparse
from pathlib import Path
import tempfile

import pefile
from iced_x86 import (
    Code,
    Decoder,
    EncodingKind,
    MemorySizeExt,
    OpKind,
    Register as R,
    RegisterExt,
)

from cpu_leaf_aot import bounded_code, export_rva


ARGUMENTS = {R.RCX: "rcx", R.RDX: "rdx", R.R8: "r8"}
ARGUMENT_GPRS = {R.RCX, R.ECX, R.CL, R.RDX, R.EDX, R.DL, R.R8, R.R8D, R.R8L}

VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)

SCRATCH_GPR64 = {R.RAX: "rax", R.R9: "r9", R.R10: "r10", R.R11: "r11"}
SCRATCH_GPR32 = {R.EAX: "rax", R.R9D: "r9", R.R10D: "r10", R.R11D: "r11"}
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

LOAD_ALIGNED_VEC_CODES = {
    Code.MOVAPS_XMM_XMMM128,
    Code.MOVAPD_XMM_XMMM128,
    Code.MOVDQA_XMM_XMMM128,
}
STORE_ALIGNED_VEC_CODES = {
    Code.MOVAPS_XMMM128_XMM,
    Code.MOVAPD_XMMM128_XMM,
    Code.MOVDQA_XMMM128_XMM,
}
LOAD_UNALIGNED_VEC_CODES = {
    Code.MOVUPS_XMM_XMMM128,
    Code.MOVUPD_XMM_XMMM128,
    Code.MOVDQU_XMM_XMMM128,
}
STORE_UNALIGNED_VEC_CODES = {
    Code.MOVUPS_XMMM128_XMM,
    Code.MOVUPD_XMMM128_XMM,
    Code.MOVDQU_XMMM128_XMM,
}
MOVSS_REG_CODES = {Code.MOVSS_XMM_XMMM32, Code.MOVSS_XMMM32_XMM}
MOVSD_REG_CODES = {Code.MOVSD_XMM_XMMM64, Code.MOVSD_XMMM64_XMM}
EXTRACTPS_CODES = {Code.EXTRACTPS_RM32_XMM_IMM8, Code.EXTRACTPS_R64M32_XMM_IMM8}
MOV_GPR_CODES = {
    Code.MOV_R32_RM32: False,
    Code.MOV_RM32_R32: False,
    Code.MOV_R64_RM64: True,
    Code.MOV_RM64_R64: True,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.ROUNDPS_XMM_XMMM128_IMM8,
    Code.ROUNDPD_XMM_XMMM128_IMM8,
    Code.ROUNDSS_XMM_XMMM32_IMM8,
    Code.ROUNDSD_XMM_XMMM64_IMM8,
    Code.DPPS_XMM_XMMM128_IMM8,
    Code.DPPD_XMM_XMMM128_IMM8,
    Code.INSERTPS_XMM_XMMM32_IMM8,
    *EXTRACTPS_CODES,
    *LOAD_ALIGNED_VEC_CODES,
    *STORE_ALIGNED_VEC_CODES,
    *LOAD_UNALIGNED_VEC_CODES,
    *STORE_UNALIGNED_VEC_CODES,
    *MOVSS_REG_CODES,
    *MOVSD_REG_CODES,
    *MOV_GPR_CODES,
}

PRELUDE = """#include <atomic>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define X2W_UNALIGNED __unaligned
#else
#define X2W_UNALIGNED
#endif

using word = std::uint64_t;

struct Vec128 {
    std::uint64_t q[2];
};

static inline std::uint32_t load32(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const std::uint32_t v = *reinterpret_cast<const volatile std::uint32_t X2W_UNALIGNED*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return v;
}

static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const std::uint64_t v = *reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return v;
}

static inline Vec128 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1}};
}

static inline void store32(word address, std::uint32_t val) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &val, 4);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store64(word address, std::uint64_t val) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), &val, 8);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store128(word address, const Vec128& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline std::uint32_t lane32(const Vec128& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static inline void set_lane32(Vec128& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

static inline std::uint32_t round_mode(std::uint8_t imm8) {
    if ((imm8 & 4u) == 0u) {
        return static_cast<std::uint32_t>(imm8 & 3u);
    }
    return (_mm_getcsr() >> 13u) & 3u;
}

static inline std::int64_t round_double_to_int(double val, std::uint32_t rc) {
    const std::int64_t t = static_cast<std::int64_t>(val);
    const double tf = static_cast<double>(t);
    if (rc == 1u) {
        return (tf > val) ? (t - 1) : t;
    }
    if (rc == 2u) {
        return (tf < val) ? (t + 1) : t;
    }
    if (rc == 3u) {
        return t;
    }
    const double rem = val - tf;
    if (val >= 0.0) {
        return (rem > 0.5 || (rem == 0.5 && (t & 1) != 0)) ? (t + 1) : t;
    }
    return (rem < -0.5 || (rem == -0.5 && (t & 1) != 0)) ? (t - 1) : t;
}

static inline std::uint32_t round_f32_bits(std::uint32_t bits, std::uint8_t imm8) {
    const volatile std::uint32_t vb = bits;
    const std::uint32_t raw = vb;
    const std::uint32_t abs_bits = raw & 0x7fffffffu;
    const std::uint32_t sign_bit = raw & 0x80000000u;
    if (abs_bits > 0x7f800000u) {
        return raw | 0x00400000u;
    }
    if (abs_bits == 0x7f800000u || abs_bits == 0u) {
        return raw;
    }
    if ((_mm_getcsr() & 0x0040u) != 0u && abs_bits < 0x00800000u) {
        return sign_bit;
    }
    if (abs_bits >= 0x4b000000u) {
        return raw;
    }
    float f = 0.0f;
    std::memcpy(&f, &raw, 4);
    const std::int64_t n = round_double_to_int(static_cast<double>(f), round_mode(imm8));
    if (n == 0) {
        return sign_bit;
    }
    const float out_f = static_cast<float>(n);
    std::uint32_t out_bits = 0u;
    std::memcpy(&out_bits, &out_f, 4);
    return out_bits;
}

static inline std::uint64_t round_f64_bits(std::uint64_t bits, std::uint8_t imm8) {
    const volatile std::uint64_t vb = bits;
    const std::uint64_t raw = vb;
    const std::uint64_t abs_bits = raw & 0x7fffffffffffffffull;
    const std::uint64_t sign_bit = raw & 0x8000000000000000ull;
    if (abs_bits > 0x7ff0000000000000ull) {
        return raw | 0x0008000000000000ull;
    }
    if (abs_bits == 0x7ff0000000000000ull || abs_bits == 0ull) {
        return raw;
    }
    if ((_mm_getcsr() & 0x0040u) != 0u && abs_bits < 0x0010000000000000ull) {
        return sign_bit;
    }
    if (abs_bits >= 0x4330000000000000ull) {
        return raw;
    }
    double d = 0.0;
    std::memcpy(&d, &raw, 8);
    const std::int64_t n = round_double_to_int(d, round_mode(imm8));
    if (n == 0) {
        return sign_bit;
    }
    const double out_d = static_cast<double>(n);
    std::uint64_t out_bits = 0ull;
    std::memcpy(&out_bits, &out_d, 8);
    return out_bits;
}

static inline Vec128 op_roundps(const Vec128& src, std::uint8_t imm8) {
    Vec128 out{};
    for (unsigned i = 0; i < 4u; ++i) {
        set_lane32(out, i, round_f32_bits(lane32(src, i), imm8));
    }
    return out;
}

static inline Vec128 op_roundpd(const Vec128& src, std::uint8_t imm8) {
    return {{round_f64_bits(src.q[0], imm8), round_f64_bits(src.q[1], imm8)}};
}

static inline Vec128 op_roundss(const Vec128& dst, std::uint32_t src32, std::uint8_t imm8) {
    Vec128 out = dst;
    set_lane32(out, 0u, round_f32_bits(src32, imm8));
    return out;
}

static inline Vec128 op_roundsd(const Vec128& dst, std::uint64_t src64, std::uint8_t imm8) {
    return {{round_f64_bits(src64, imm8), dst.q[1]}};
}

static inline __m128 sse_ss_from_bits(std::uint32_t bits) {
    const volatile std::uint32_t vb = bits;
    return _mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(vb)));
}

static inline std::uint32_t bits_from_sse_ss(__m128 v) {
    return static_cast<std::uint32_t>(_mm_cvtsi128_si32(_mm_castps_si128(v)));
}

static inline __m128d sse_sd_from_bits(std::uint64_t bits) {
    const volatile std::uint64_t vb = bits;
    return _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(vb)));
}

static inline std::uint64_t bits_from_sse_sd(__m128d v) {
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(v)));
}

// DPPS executes products and pairwise sums via host SSE2 so MXCSR RC, DAZ, FTZ,
// +/-0.0, Inf, and NaN quieting match the host processor.
static inline Vec128 op_dpps(const Vec128& a, const Vec128& b, std::uint8_t imm8) {
    __m128 t[4];
    for (unsigned k = 0; k < 4u; ++k) {
        if ((imm8 & (1u << (4u + k))) != 0u) {
            t[k] = _mm_mul_ss(sse_ss_from_bits(lane32(a, k)), sse_ss_from_bits(lane32(b, k)));
        } else {
            t[k] = _mm_setzero_ps();
        }
    }
    const __m128 s01 = _mm_add_ss(t[0], t[1]);
    const __m128 s23 = _mm_add_ss(t[2], t[3]);
    const std::uint32_t sum = bits_from_sse_ss(_mm_add_ss(s01, s23));
    Vec128 out{};
    for (unsigned k = 0; k < 4u; ++k) {
        set_lane32(out, k, ((imm8 >> k) & 1u) != 0u ? sum : 0u);
    }
    return out;
}

static inline Vec128 op_dppd(const Vec128& a, const Vec128& b, std::uint8_t imm8) {
    __m128d t[2];
    for (unsigned k = 0; k < 2u; ++k) {
        if ((imm8 & (1u << (4u + k))) != 0u) {
            t[k] = _mm_mul_sd(sse_sd_from_bits(a.q[k]), sse_sd_from_bits(b.q[k]));
        } else {
            t[k] = _mm_setzero_pd();
        }
    }
    const std::uint64_t sum = bits_from_sse_sd(_mm_add_sd(t[0], t[1]));
    return {{
        (imm8 & 1u) != 0u ? sum : 0ull,
        (imm8 & 2u) != 0u ? sum : 0ull,
    }};
}

static inline Vec128 op_insertps(const Vec128& dst, std::uint32_t src_lane, std::uint8_t imm8) {
    Vec128 out = dst;
    const unsigned count_d = (imm8 >> 4u) & 3u;
    set_lane32(out, count_d, src_lane);
    for (unsigned k = 0; k < 4u; ++k) {
        if (((imm8 >> k) & 1u) != 0u) {
            set_lane32(out, k, 0u);
        }
    }
    return out;
}

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_sse41() {
    int info[4]{};
    __cpuid(info, 1);
    return (info[2] & (1 << 19)) != 0 ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_exchange_mxcsr(std::uint32_t next) {
    const std::uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
#endif
"""


def vec_index(register, ins):
    """Map XMM0..XMM5 to an index, rejecting nonvolatile and unsupported registers."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def gpr_target(register, ins, *, write=False):
    """Map scratch GPR32/GPR64 to its 64-bit variable name, rejecting nonvolatile/arg writes."""
    if register in SCRATCH_GPR64:
        return SCRATCH_GPR64[register]
    if register in SCRATCH_GPR32:
        return SCRATCH_GPR32[register]
    if register in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR {register} at RVA {ins.ip:#x}: {ins}")
    if write and register in ARGUMENT_GPRS:
        raise ValueError(f"write to argument GPR {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported GPR {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, alignment=1, write=False, expected_size=None):
    """Lower a flat zero-displacement [rcx|rdx|r8] pointer operand (stores restricted to [r8])."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or ins.memory_displacement != 0
        or not RegisterExt.is_gpr64(base)
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if base in NONVOLATILE_GPR:
        raise ValueError(f"nonvolatile GPR base {base} in memory operand at RVA {ins.ip:#x}: {ins}")
    if base not in ARGUMENTS:
        raise ValueError(f"unsupported memory base {base} at RVA {ins.ip:#x}: {ins}")
    if write and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    if expected_size is not None and MemorySizeExt.size(ins.memory_size) != expected_size:
        raise ValueError(f"unsupported memory size at RVA {ins.ip:#x}: {ins}")
    if alignment > 1 and (ins.memory_displacement % alignment) != 0:
        raise ValueError(f"unaligned {alignment}-byte memory operand at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line legacy SSE4.1 FP leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_vec = set()
    used_gpr = set()
    defined_gpr = set()
    statements = []
    returned = False

    def read_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register xmm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"xmm{idx}"

    def write_vec(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins)
        used_vec.add(idx)
        return idx, f"xmm{idx}"

    def vec_src128(op_index, ins, *, alignment):
        if ins.op_kind(op_index) == OpKind.MEMORY:
            return f"load128({flat_address(ins, alignment=alignment, expected_size=16)})"
        return read_vec(op_index, ins)

    def read_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR operand {op_index} at RVA {ins.ip:#x}: {ins}")
        gpr = gpr_target(ins.op_register(op_index), ins, write=False)
        if gpr not in defined_gpr:
            raise ValueError(f"undefined GPR {gpr} at RVA {ins.ip:#x}: {ins}")
        used_gpr.add(gpr)
        return gpr

    def write_gpr(op_index, ins):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected GPR destination at RVA {ins.ip:#x}: {ins}")
        gpr = gpr_target(ins.op_register(op_index), ins, write=True)
        used_gpr.add(gpr)
        return gpr

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if ins.op_code().encoding != EncodingKind.LEGACY:
            raise ValueError(f"non-legacy encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code in (Code.ROUNDPS_XMM_XMMM128_IMM8, Code.ROUNDPD_XMM_XMMM128_IMM8):
            helper = "op_roundpd" if ins.code == Code.ROUNDPD_XMM_XMMM128_IMM8 else "op_roundps"
            imm = ins.immediate8 & 0x0F
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = {helper}({src}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif ins.code == Code.ROUNDSS_XMM_XMMM32_IMM8:
            imm = ins.immediate8 & 0x0F
            dst_in = read_vec(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                src32 = f"load32({flat_address(ins, expected_size=4)})"
            else:
                src32 = f"lane32({read_vec(1, ins)}, 0u)"
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = op_roundss({dst_in}, {src32}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif ins.code == Code.ROUNDSD_XMM_XMMM64_IMM8:
            imm = ins.immediate8 & 0x0F
            dst_in = read_vec(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                src64 = f"load64({flat_address(ins, expected_size=8)})"
            else:
                src64 = f"{read_vec(1, ins)}.q[0]"
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = op_roundsd({dst_in}, {src64}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif ins.code in (Code.DPPS_XMM_XMMM128_IMM8, Code.DPPD_XMM_XMMM128_IMM8):
            helper = "op_dppd" if ins.code == Code.DPPD_XMM_XMMM128_IMM8 else "op_dpps"
            imm = ins.immediate8 & 0xFF
            dst_in = read_vec(0, ins)
            src = vec_src128(1, ins, alignment=16)
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = {helper}({dst_in}, {src}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif ins.code == Code.INSERTPS_XMM_XMMM32_IMM8:
            imm = ins.immediate8 & 0xFF
            dst_in = read_vec(0, ins)
            if ins.op1_kind == OpKind.MEMORY:
                src32 = f"load32({flat_address(ins, expected_size=4)})"
            else:
                count_s = (imm >> 6) & 3
                src32 = f"lane32({read_vec(1, ins)}, {count_s}u)"
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = op_insertps({dst_in}, {src32}, {imm:#x}u);")
            defined_vec.add(dst_idx)
        elif ins.code in EXTRACTPS_CODES:
            lane = ins.immediate8 & 3
            src = read_vec(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, write=True, expected_size=4)
                statements.append(f"store32({addr}, lane32({src}, {lane}u));")
            else:
                gpr = write_gpr(0, ins)
                statements.append(f"{gpr} = static_cast<word>(lane32({src}, {lane}u));")
                defined_gpr.add(gpr)
        elif ins.code in (*LOAD_ALIGNED_VEC_CODES, *LOAD_UNALIGNED_VEC_CODES):
            align = 16 if ins.code in LOAD_ALIGNED_VEC_CODES else 1
            src = vec_src128(1, ins, alignment=align)
            dst_idx, dst = write_vec(0, ins)
            statements.append(f"{dst} = {src};")
            defined_vec.add(dst_idx)
        elif ins.code in (*STORE_ALIGNED_VEC_CODES, *STORE_UNALIGNED_VEC_CODES):
            align = 16 if ins.code in STORE_ALIGNED_VEC_CODES else 1
            src = read_vec(1, ins)
            if ins.op0_kind == OpKind.MEMORY:
                addr = flat_address(ins, alignment=align, write=True, expected_size=16)
                statements.append(f"store128({addr}, {src});")
            else:
                dst_idx, dst = write_vec(0, ins)
                statements.append(f"{dst} = {src};")
                defined_vec.add(dst_idx)
        elif ins.code in MOVSS_REG_CODES:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=4)
                dst_idx, dst = write_vec(0, ins)
                statements.append(f"{dst} = Vec128{{static_cast<std::uint64_t>(load32({addr})), 0ull}};")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins)
                addr = flat_address(ins, write=True, expected_size=4)
                statements.append(f"store32({addr}, lane32({src}, 0u));")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                read_vec(0, ins)
                src = read_vec(1, ins)
                dst_idx, dst = write_vec(0, ins)
                statements.append(f"set_lane32({dst}, 0u, lane32({src}, 0u));")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported movss operands at RVA {ins.ip:#x}: {ins}")
        elif ins.code in MOVSD_REG_CODES:
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=8)
                dst_idx, dst = write_vec(0, ins)
                statements.append(f"{dst} = Vec128{{load64({addr}), 0ull}};")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins)
                addr = flat_address(ins, write=True, expected_size=8)
                statements.append(f"store64({addr}, {src}.q[0]);")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                read_vec(0, ins)
                src = read_vec(1, ins)
                dst_idx, dst = write_vec(0, ins)
                statements.append(f"{dst}.q[0] = {src}.q[0];")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported movsd operands at RVA {ins.ip:#x}: {ins}")
        elif ins.code in MOV_GPR_CODES:
            is64 = MOV_GPR_CODES[ins.code]
            sz = 8 if is64 else 4
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=sz)
                dst = write_gpr(0, ins)
                loader = f"load64({addr})" if is64 else f"static_cast<word>(load32({addr}))"
                statements.append(f"{dst} = {loader};")
                defined_gpr.add(dst)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_gpr(1, ins)
                addr = flat_address(ins, write=True, expected_size=sz)
                if is64:
                    statements.append(f"store64({addr}, {src});")
                else:
                    statements.append(f"store32({addr}, static_cast<std::uint32_t>({src}));")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_gpr(1, ins)
                dst = write_gpr(0, ins)
                expr = src if is64 else f"static_cast<word>(static_cast<std::uint32_t>({src}))"
                statements.append(f"{dst} = {expr};")
                defined_gpr.add(dst)
            else:
                raise ValueError(f"unsupported GPR mov operands at RVA {ins.ip:#x}: {ins}")
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec128 xmm{idx}{{}};" for idx in sorted(used_vec)]
    decls.extend(f"    word {gpr} = 0ull;" for gpr in sorted(used_gpr))
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


def _write_atomic(path: Path, content: str) -> None:
    if path.exists() and path.is_dir():
        raise ValueError(f"output path must not be a directory: {path}")
    if path.is_symlink():
        raise ValueError(f"output path must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("exports", help="comma-separated export names")
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise ValueError(f"input image must be a regular file: {args.image}")
        exports = [name for name in args.exports.split(",") if name]
        if not exports:
            raise ValueError("exports list must not be empty")
        if args.output.exists() and args.output.is_dir():
            raise ValueError(f"output path must not be a directory: {args.output}")
        if args.output.is_symlink():
            raise ValueError(f"output path must not be a symlink: {args.output}")
        source, meta = translate(args.image, exports)
        _write_atomic(args.output, source)
        for name, entry, count in meta:
            print(f"{name} at RVA {entry:#x}: {count} instructions")
    except (ValueError, OSError, pefile.PEFormatError) as error:
        parser.exit(1, f"cpu_sse41_fp_aot: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
