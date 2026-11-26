"""Lift bounded Windows x64 leaf functions with AVX VEX FP compare instructions to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned buffers. YMM0..YMM5
are modeled as portable 256-bit value structs (four 64-bit lanes); every VEX.128
write to XMM0..XMM5 zeroes bits [255:128] of the corresponding YMM register.

Supported AVX FP compare encodings (docs/coverage/x64_zen2.tsv group AVX):
  - VCMPPS: VEX.128 (xmm, xmm, xmm/m128, imm8) and VEX.256 (ymm, ymm, ymm/m256, imm8)
  - VCMPPD: VEX.128 (xmm, xmm, xmm/m128, imm8) and VEX.256 (ymm, ymm, ymm/m256, imm8)
  - VCMPSS: VEX.128 scalar f32 (xmm, xmm, xmm/m32, imm8), preserving src1[127:32]
  - VCMPSD: VEX.128 scalar f64 (xmm, xmm, xmm/m64, imm8), preserving src1[127:64]
All 32 5-bit predicates (imm8 & 0x1F) are modeled via baseline SSE2 comparisons
on volatile operand copies (preventing fast-math folding while preserving +0.0 == -0.0,
+/-Inf, qNaN/sNaN, MXCSR DAZ, and quiet vs signaling NaN exception behavior).

Transport / ABI glue:
  - VMOVAPS, VMOVAPD, VMOVUPS, VMOVUPD (128/256-bit reg-reg, flat [rcx|rdx|r8] loads,
    and flat [r8] stores)
  - VMOVSS, VMOVSD (scalar reg-reg merge, flat [rcx|rdx|r8] load, flat [r8] store)
  - VZEROUPPER and terminal RET.

EVEX encodings, legacy SSE forms (CMPPS/CMPPD/CMPSS/CMPSD/MOVAPS/etc.), nonvolatile
vector registers (XMM6..15/YMM6..15), undefined vector reads, displaced or non-flat
memory operands, non-[r8] stores, instruction prefixes, branches, and trailing or
missing ret are rejected.
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
VOLATILE_XMM = {getattr(R, f"XMM{i}"): i for i in range(6)}
VOLATILE_YMM = {getattr(R, f"YMM{i}"): i for i in range(6)}
NONVOLATILE_VEC = (
    {getattr(R, f"XMM{i}") for i in range(6, 32)}
    | {getattr(R, f"YMM{i}") for i in range(6, 32)}
    | {getattr(R, f"ZMM{i}") for i in range(32)}
)

# (helper_name, is_ymm, mem_size, is_scalar, is_double)
VCMP_CODES = {
    Code.VEX_VCMPPS_XMM_XMM_XMMM128_IMM8: ("op_vcmpps", False, 16, False, False),
    Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8: ("op_vcmpps", True, 32, False, False),
    Code.VEX_VCMPPD_XMM_XMM_XMMM128_IMM8: ("op_vcmppd", False, 16, False, True),
    Code.VEX_VCMPPD_YMM_YMM_YMMM256_IMM8: ("op_vcmppd", True, 32, False, True),
    Code.VEX_VCMPSS_XMM_XMM_XMMM32_IMM8: ("op_vcmpss", False, 4, True, False),
    Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8: ("op_vcmpsd", False, 8, True, True),
}

# is_ymm for full-width VEX FP moves
VEC_MOVE_CODES = {
    Code.VEX_VMOVAPS_XMM_XMMM128: False,
    Code.VEX_VMOVAPS_XMMM128_XMM: False,
    Code.VEX_VMOVAPS_YMM_YMMM256: True,
    Code.VEX_VMOVAPS_YMMM256_YMM: True,
    Code.VEX_VMOVAPD_XMM_XMMM128: False,
    Code.VEX_VMOVAPD_XMMM128_XMM: False,
    Code.VEX_VMOVAPD_YMM_YMMM256: True,
    Code.VEX_VMOVAPD_YMMM256_YMM: True,
    Code.VEX_VMOVUPS_XMM_XMMM128: False,
    Code.VEX_VMOVUPS_XMMM128_XMM: False,
    Code.VEX_VMOVUPS_YMM_YMMM256: True,
    Code.VEX_VMOVUPS_YMMM256_YMM: True,
    Code.VEX_VMOVUPD_XMM_XMMM128: False,
    Code.VEX_VMOVUPD_XMMM128_XMM: False,
    Code.VEX_VMOVUPD_YMM_YMMM256: True,
    Code.VEX_VMOVUPD_YMMM256_YMM: True,
}

VMOVSS_REG_CODES = {
    Code.VEX_VMOVSS_XMM_XMM_XMM,
    Code.VEX_VMOVSS_XMM_XMM_XMM_0F11,
}
VMOVSD_REG_CODES = {
    Code.VEX_VMOVSD_XMM_XMM_XMM,
    Code.VEX_VMOVSD_XMM_XMM_XMM_0F11,
}
SCALAR_MOVE_CODES = {
    Code.VEX_VMOVSS_XMM_M32,
    Code.VEX_VMOVSS_M32_XMM,
    *VMOVSS_REG_CODES,
    Code.VEX_VMOVSD_XMM_M64,
    Code.VEX_VMOVSD_M64_XMM,
    *VMOVSD_REG_CODES,
}

LEGACY_SSE_REJECT_CODES = {
    Code.CMPPS_XMM_XMMM128_IMM8,
    Code.CMPPD_XMM_XMMM128_IMM8,
    Code.CMPSS_XMM_XMMM32_IMM8,
    Code.CMPSD_XMM_XMMM64_IMM8,
    Code.MOVAPS_XMM_XMMM128,
    Code.MOVAPS_XMMM128_XMM,
    Code.MOVAPD_XMM_XMMM128,
    Code.MOVAPD_XMMM128_XMM,
    Code.MOVUPS_XMM_XMMM128,
    Code.MOVUPS_XMMM128_XMM,
    Code.MOVUPD_XMM_XMMM128,
    Code.MOVUPD_XMMM128_XMM,
    Code.MOVSS_XMM_XMMM32,
    Code.MOVSS_XMMM32_XMM,
    Code.MOVSD_XMM_XMMM64,
    Code.MOVSD_XMMM64_XMM,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    *VCMP_CODES,
    *VEC_MOVE_CODES,
    *SCALAR_MOVE_CODES,
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

struct Vec256 {
    std::uint64_t q[4];
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

static inline Vec256 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, 0ull, 0ull}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const auto* p = reinterpret_cast<const volatile std::uint64_t X2W_UNALIGNED*>(address);
    const std::uint64_t q0 = p[0];
    const std::uint64_t q1 = p[1];
    const std::uint64_t q2 = p[2];
    const std::uint64_t q3 = p[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
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

static inline void store128(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline void store256(word address, const Vec256& v) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

static inline Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static inline std::uint32_t lane32(const Vec256& v, unsigned i) {
    return static_cast<std::uint32_t>(v.q[i >> 1] >> ((i & 1u) * 32u));
}

static inline void set_lane32(Vec256& v, unsigned i, std::uint32_t val) {
    const unsigned shift = (i & 1u) * 32u;
    v.q[i >> 1] = (v.q[i >> 1] & ~(0xffffffffull << shift)) | (static_cast<std::uint64_t>(val) << shift);
}

// Model all 32 AVX FP compare predicates (imm8 & 0x1F) using baseline SSE2 scalar operations
// on volatile operand copies so /O2 cannot fast-math fold +0/-0, Inf, NaN, DAZ, or IE signaling.
static inline std::uint32_t cmp_f32_scalar(std::uint32_t a, std::uint32_t b, std::uint8_t pred) {
    const volatile std::uint32_t va = a;
    const volatile std::uint32_t vb = b;
    const __m128 x = _mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(va)));
    const __m128 y = _mm_castsi128_ps(_mm_cvtsi32_si128(static_cast<int>(vb)));
    __m128 r;
    switch (pred & 0x1fu) {
        case 0x00u: r = _mm_cmpeq_ss(x, y); break;
        case 0x01u: r = _mm_cmplt_ss(x, y); break;
        case 0x02u: r = _mm_cmple_ss(x, y); break;
        case 0x03u: r = _mm_cmpunord_ss(x, y); break;
        case 0x04u: r = _mm_cmpneq_ss(x, y); break;
        case 0x05u: r = _mm_cmpnlt_ss(x, y); break;
        case 0x06u: r = _mm_cmpnle_ss(x, y); break;
        case 0x07u: r = _mm_cmpord_ss(x, y); break;
        case 0x08u: r = _mm_or_ps(_mm_cmpeq_ss(x, y), _mm_cmpunord_ss(x, y)); break;
        case 0x09u: r = _mm_cmpnle_ss(y, x); break;
        case 0x0au: r = _mm_cmpnlt_ss(y, x); break;
        case 0x0bu: { const volatile __m128 t = _mm_cmpeq_ss(x, y); (void)t; return 0u; }
        case 0x0cu: r = _mm_and_ps(_mm_cmpneq_ss(x, y), _mm_cmpord_ss(x, y)); break;
        case 0x0du: r = _mm_cmple_ss(y, x); break;
        case 0x0eu: r = _mm_cmplt_ss(y, x); break;
        case 0x0fu: { const volatile __m128 t = _mm_cmpeq_ss(x, y); (void)t; return 0xffffffffu; }
        case 0x10u: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_cmpeq_ss(x, y); break; }
        case 0x11u: return _mm_ucomigt_ss(y, x) ? 0xffffffffu : 0u;
        case 0x12u: return _mm_ucomige_ss(y, x) ? 0xffffffffu : 0u;
        case 0x13u: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_cmpunord_ss(x, y); break; }
        case 0x14u: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_cmpneq_ss(x, y); break; }
        case 0x15u: return _mm_ucomigt_ss(y, x) ? 0u : 0xffffffffu;
        case 0x16u: return _mm_ucomige_ss(y, x) ? 0u : 0xffffffffu;
        case 0x17u: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_cmpord_ss(x, y); break; }
        case 0x18u: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_or_ps(_mm_cmpeq_ss(x, y), _mm_cmpunord_ss(x, y)); break; }
        case 0x19u: return _mm_ucomige_ss(x, y) ? 0u : 0xffffffffu;
        case 0x1au: return _mm_ucomigt_ss(x, y) ? 0u : 0xffffffffu;
        case 0x1bu: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; return 0u; }
        case 0x1cu: { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; r = _mm_and_ps(_mm_cmpneq_ss(x, y), _mm_cmpord_ss(x, y)); break; }
        case 0x1du: return _mm_ucomige_ss(x, y) ? 0xffffffffu : 0u;
        case 0x1eu: return _mm_ucomigt_ss(x, y) ? 0xffffffffu : 0u;
        default:    { const volatile __m128 t = _mm_cmplt_ss(x, y); (void)t; return 0xffffffffu; }
    }
    return static_cast<std::uint32_t>(_mm_cvtsi128_si32(_mm_castps_si128(r)));
}

static inline std::uint64_t cmp_f64_scalar(std::uint64_t a, std::uint64_t b, std::uint8_t pred) {
    const volatile std::uint64_t va = a;
    const volatile std::uint64_t vb = b;
    const __m128d x = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(va)));
    const __m128d y = _mm_castsi128_pd(_mm_cvtsi64_si128(static_cast<long long>(vb)));
    __m128d r;
    switch (pred & 0x1fu) {
        case 0x00u: r = _mm_cmpeq_sd(x, y); break;
        case 0x01u: r = _mm_cmplt_sd(x, y); break;
        case 0x02u: r = _mm_cmple_sd(x, y); break;
        case 0x03u: r = _mm_cmpunord_sd(x, y); break;
        case 0x04u: r = _mm_cmpneq_sd(x, y); break;
        case 0x05u: r = _mm_cmpnlt_sd(x, y); break;
        case 0x06u: r = _mm_cmpnle_sd(x, y); break;
        case 0x07u: r = _mm_cmpord_sd(x, y); break;
        case 0x08u: r = _mm_or_pd(_mm_cmpeq_sd(x, y), _mm_cmpunord_sd(x, y)); break;
        case 0x09u: r = _mm_cmpnle_sd(y, x); break;
        case 0x0au: r = _mm_cmpnlt_sd(y, x); break;
        case 0x0bu: { const volatile __m128d t = _mm_cmpeq_sd(x, y); (void)t; return 0ull; }
        case 0x0cu: r = _mm_and_pd(_mm_cmpneq_sd(x, y), _mm_cmpord_sd(x, y)); break;
        case 0x0du: r = _mm_cmple_sd(y, x); break;
        case 0x0eu: r = _mm_cmplt_sd(y, x); break;
        case 0x0fu: { const volatile __m128d t = _mm_cmpeq_sd(x, y); (void)t; return 0xffffffffffffffffull; }
        case 0x10u: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_cmpeq_sd(x, y); break; }
        case 0x11u: return _mm_ucomigt_sd(y, x) ? 0xffffffffffffffffull : 0ull;
        case 0x12u: return _mm_ucomige_sd(y, x) ? 0xffffffffffffffffull : 0ull;
        case 0x13u: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_cmpunord_sd(x, y); break; }
        case 0x14u: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_cmpneq_sd(x, y); break; }
        case 0x15u: return _mm_ucomigt_sd(y, x) ? 0ull : 0xffffffffffffffffull;
        case 0x16u: return _mm_ucomige_sd(y, x) ? 0ull : 0xffffffffffffffffull;
        case 0x17u: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_cmpord_sd(x, y); break; }
        case 0x18u: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_or_pd(_mm_cmpeq_sd(x, y), _mm_cmpunord_sd(x, y)); break; }
        case 0x19u: return _mm_ucomige_sd(x, y) ? 0ull : 0xffffffffffffffffull;
        case 0x1au: return _mm_ucomigt_sd(x, y) ? 0ull : 0xffffffffffffffffull;
        case 0x1bu: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; return 0ull; }
        case 0x1cu: { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; r = _mm_and_pd(_mm_cmpneq_sd(x, y), _mm_cmpord_sd(x, y)); break; }
        case 0x1du: return _mm_ucomige_sd(x, y) ? 0xffffffffffffffffull : 0ull;
        case 0x1eu: return _mm_ucomigt_sd(x, y) ? 0xffffffffffffffffull : 0ull;
        default:    { const volatile __m128d t = _mm_cmplt_sd(x, y); (void)t; return 0xffffffffffffffffull; }
    }
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_castpd_si128(r)));
}

static inline Vec256 op_vcmpps(const Vec256& a, const Vec256& b, std::uint8_t pred, bool wide) {
    Vec256 out{};
    const unsigned count = wide ? 8u : 4u;
    for (unsigned i = 0; i < count; ++i) {
        set_lane32(out, i, cmp_f32_scalar(lane32(a, i), lane32(b, i), pred));
    }
    return out;
}

static inline Vec256 op_vcmppd(const Vec256& a, const Vec256& b, std::uint8_t pred, bool wide) {
    Vec256 out{};
    const unsigned count = wide ? 4u : 2u;
    for (unsigned i = 0; i < count; ++i) {
        out.q[i] = cmp_f64_scalar(a.q[i], b.q[i], pred);
    }
    return out;
}

static inline Vec256 op_vcmpss(const Vec256& a, std::uint32_t b32, std::uint8_t pred) {
    Vec256 out = copy128(a);
    set_lane32(out, 0u, cmp_f32_scalar(lane32(a, 0u), b32, pred));
    return out;
}

static inline Vec256 op_vcmpsd(const Vec256& a, std::uint64_t b64, std::uint8_t pred) {
    Vec256 out = copy128(a);
    out.q[0] = cmp_f64_scalar(a.q[0], b64, pred);
    return out;
}

static inline Vec256 op_vmovss_reg(const Vec256& a, std::uint32_t b32) {
    Vec256 out = copy128(a);
    set_lane32(out, 0u, b32);
    return out;
}

static inline Vec256 op_vmovss_mem(std::uint32_t b32) {
    Vec256 out{};
    set_lane32(out, 0u, b32);
    return out;
}

static inline Vec256 op_vmovsd_reg(const Vec256& a, std::uint64_t b64) {
    Vec256 out = copy128(a);
    out.q[0] = b64;
    return out;
}

static inline Vec256 op_vmovsd_mem(std::uint64_t b64) {
    return {{b64, 0ull, 0ull, 0ull}};
}

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}

extern "C" __declspec(dllexport) std::uint32_t host_exchange_mxcsr(std::uint32_t next) {
    const std::uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
#endif
"""


def vec_index(register, ins, *, expected_wide):
    """Map XMM0..XMM5 or YMM0..YMM5 to register index, enforcing width and rejecting nonvolatile/other."""
    if register in VOLATILE_XMM:
        if expected_wide:
            raise ValueError(f"unsupported vector width for {register} at RVA {ins.ip:#x}: {ins}")
        return VOLATILE_XMM[register]
    if register in VOLATILE_YMM:
        if not expected_wide:
            raise ValueError(f"unsupported vector width for {register} at RVA {ins.ip:#x}: {ins}")
        return VOLATILE_YMM[register]
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, store=False, expected_size=None):
    """Lower a flat zero-displacement [rcx|rdx|r8] pointer operand (stores restricted to [r8])."""
    base = ins.memory_base
    if (
        ins.is_ip_rel_memory_operand
        or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
        or ins.memory_index != R.NONE
        or ins.memory_displacement != 0
        or not RegisterExt.is_gpr64(base)
        or base not in ARGUMENTS
    ):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"non-r8 store memory operand at RVA {ins.ip:#x}: {ins}")
    if expected_size is not None and MemorySizeExt.size(ins.memory_size) != expected_size:
        raise ValueError(f"unsupported memory size at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX VEX FP compare leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used = set()
    defined = set()
    statements = []
    returned = False

    def read_vec(op_index, ins, *, expected_wide):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins, expected_wide=expected_wide)
        if idx not in defined:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used.add(idx)
        return f"ymm{idx}"

    def write_vec(op_index, ins, *, expected_wide):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected register destination at RVA {ins.ip:#x}: {ins}")
        idx = vec_index(ins.op_register(op_index), ins, expected_wide=expected_wide)
        used.add(idx)
        return idx, f"ymm{idx}"

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        if (
            ins.op_code().encoding == EncodingKind.EVEX
            or ins.op_mask != R.NONE
            or ins.zeroing_masking
            or ins.is_broadcast
        ):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in LEGACY_SSE_REJECT_CODES:
            raise ValueError(f"unsupported legacy SSE instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True
        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")
        elif ins.code in VCMP_CODES:
            helper, wide, mem_size, is_scalar, is_double = VCMP_CODES[ins.code]
            if ins.op_count != 4 or ins.op0_kind != OpKind.REGISTER or ins.op1_kind != OpKind.REGISTER:
                raise ValueError(f"unsupported AVX compare operands at RVA {ins.ip:#x}: {ins}")
            pred = ins.immediate8 & 0x1F
            src1 = read_vec(1, ins, expected_wide=wide)
            if ins.op2_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=mem_size)
                if is_scalar:
                    src2 = f"{'load64' if is_double else 'load32'}({addr})"
                else:
                    src2 = f"{'load256' if wide else 'load128'}({addr})"
            elif ins.op2_kind == OpKind.REGISTER:
                reg2 = read_vec(2, ins, expected_wide=wide)
                if is_scalar:
                    src2 = f"{reg2}.q[0]" if is_double else f"lane32({reg2}, 0u)"
                else:
                    src2 = reg2
            else:
                raise ValueError(f"unsupported AVX compare operands at RVA {ins.ip:#x}: {ins}")
            dst_idx, dst = write_vec(0, ins, expected_wide=wide)
            if is_scalar:
                statements.append(f"{dst} = {helper}({src1}, {src2}, {pred:#x}u);")
            else:
                statements.append(f"{dst} = {helper}({src1}, {src2}, {pred:#x}u, {'true' if wide else 'false'});")
            defined.add(dst_idx)
        elif ins.code in VEC_MOVE_CODES:
            wide = VEC_MOVE_CODES[ins.code]
            expected_size = 32 if wide else 16
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_size=expected_size)
                dst_idx, dst = write_vec(0, ins, expected_wide=wide)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expected_wide=wide)
                addr = flat_address(ins, store=True, expected_size=expected_size)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expected_wide=wide)
                dst_idx, dst = write_vec(0, ins, expected_wide=wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined.add(dst_idx)
            else:
                raise ValueError(f"unsupported vector move operands at RVA {ins.ip:#x}: {ins}")
        elif ins.code == Code.VEX_VMOVSS_XMM_M32:
            addr = flat_address(ins, expected_size=4)
            dst_idx, dst = write_vec(0, ins, expected_wide=False)
            statements.append(f"{dst} = op_vmovss_mem(load32({addr}));")
            defined.add(dst_idx)
        elif ins.code == Code.VEX_VMOVSS_M32_XMM:
            src = read_vec(1, ins, expected_wide=False)
            addr = flat_address(ins, store=True, expected_size=4)
            statements.append(f"store32({addr}, lane32({src}, 0u));")
        elif ins.code in VMOVSS_REG_CODES:
            src1 = read_vec(1, ins, expected_wide=False)
            src2 = read_vec(2, ins, expected_wide=False)
            dst_idx, dst = write_vec(0, ins, expected_wide=False)
            statements.append(f"{dst} = op_vmovss_reg({src1}, lane32({src2}, 0u));")
            defined.add(dst_idx)
        elif ins.code == Code.VEX_VMOVSD_XMM_M64:
            addr = flat_address(ins, expected_size=8)
            dst_idx, dst = write_vec(0, ins, expected_wide=False)
            statements.append(f"{dst} = op_vmovsd_mem(load64({addr}));")
            defined.add(dst_idx)
        elif ins.code == Code.VEX_VMOVSD_M64_XMM:
            src = read_vec(1, ins, expected_wide=False)
            addr = flat_address(ins, store=True, expected_size=8)
            statements.append(f"store64({addr}, {src}.q[0]);")
        elif ins.code in VMOVSD_REG_CODES:
            src1 = read_vec(1, ins, expected_wide=False)
            src2 = read_vec(2, ins, expected_wide=False)
            dst_idx, dst = write_vec(0, ins, expected_wide=False)
            statements.append(f"{dst} = op_vmovsd_reg({src1}, {src2}.q[0]);")
            defined.add(dst_idx)
        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used)]
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
        parser.exit(1, f"cpu_avx_cmp_aot: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
