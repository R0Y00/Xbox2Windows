"""Lift bounded Windows x64 leaf functions using AVX VROUNDPS, VROUNDPD, VROUNDSS, and VROUNDSD to C++.

Each export uses the straight-line leaf ABI:
    void lifted_<name>(word rcx, word rdx, word r8)
where rcx, rdx, and r8 carry flat host pointers to caller-owned buffers (rcx/rdx
inputs, r8 output/scratch). YMM0..YMM5 are modeled as portable 256-bit value
structs (four 64-bit lanes); every VEX.128 write to XMM0..XMM5 zeroes [255:128].

Supported AVX rounding encodings (docs/coverage/x64_zen2.tsv group AVX):
  - VROUNDPS xmm, xmm/m128, imm8  |  ymm, ymm/m256, imm8  (packed f32 rounding)
  - VROUNDPD xmm, xmm/m128, imm8  |  ymm, ymm/m256, imm8  (packed f64 rounding)
  - VROUNDSS xmm, xmm, xmm/m32, imm8  (scalar f32 rounding; preserves src1[127:32], zeroes [255:128])
  - VROUNDSD xmm, xmm, xmm/m64, imm8  (scalar f64 rounding; preserves src1[127:64], zeroes [255:128])

Rounding is implemented in pure portable C++ IEEE-754 bit arithmetic (no <immintrin.h>
rounding intrinsics): imm8[1:0] selects nearest-even (0), floor (1), ceil (2), or trunc (3);
imm8[2]=1 reads the live host rounding mode via <cfenv> std::fegetround(); imm8[3]=1
suppresses FE_INEXACT; DAZ (MXCSR[6]) flushes denormal inputs to signed zero before rounding;
sNaN inputs raise FE_INVALID and are quieted preserving sign and payload; -0.0 and
negative zero-results (-0.25 -> -0.0 on nearest/ceil/trunc) preserve sign.

Transport glue:
  - vmovdqu (128/256-bit flat [rcx|rdx|r8] loads, [r8] stores, reg-reg copies)
  - vzeroupper
  - terminal ret.

Legacy SSE4.1 ROUND*, legacy MOVDQU, EVEX VRNDSCALE*, nonvolatile XMM/YMM/ZMM registers,
undefined vector reads, displaced/indexed/non-arg/mis-sized memory operands, stores outside
[r8], instruction prefixes, branches, missing ret, and code after ret are rejected.
"""

import argparse
from pathlib import Path

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

# (helper_name, is_ymm)
PACKED_ROUND_CODES = {
    Code.VEX_VROUNDPS_XMM_XMMM128_IMM8: ("op_vroundps_128", False),
    Code.VEX_VROUNDPS_YMM_YMMM256_IMM8: ("op_vroundps_256", True),
    Code.VEX_VROUNDPD_XMM_XMMM128_IMM8: ("op_vroundpd_128", False),
    Code.VEX_VROUNDPD_YMM_YMMM256_IMM8: ("op_vroundpd_256", True),
}

# (helper_name, mem_bytes, loader_name)
SCALAR_ROUND_CODES = {
    Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8: ("op_vroundss", 4, "load32"),
    Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8: ("op_vroundsd", 8, "load64"),
}

VMOVDQU_CODES = {
    Code.VEX_VMOVDQU_XMM_XMMM128: False,
    Code.VEX_VMOVDQU_XMMM128_XMM: False,
    Code.VEX_VMOVDQU_YMM_YMMM256: True,
    Code.VEX_VMOVDQU_YMMM256_YMM: True,
}

LEGACY_SSE_REJECT_CODES = {
    Code.ROUNDPS_XMM_XMMM128_IMM8,
    Code.ROUNDPD_XMM_XMMM128_IMM8,
    Code.ROUNDSS_XMM_XMMM32_IMM8,
    Code.ROUNDSD_XMM_XMMM64_IMM8,
    Code.MOVDQU_XMM_XMMM128,
    Code.MOVDQU_XMMM128_XMM,
}

ALLOWED_CODES = {
    Code.RETNQ,
    Code.VEX_VZEROUPPER,
    *PACKED_ROUND_CODES,
    *SCALAR_ROUND_CODES,
    *VMOVDQU_CODES,
}


PRELUDE = """#include <atomic>
#include <cfenv>
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if defined(_MSC_VER)
extern "C" __declspec(dllexport) std::uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
#endif

using word = std::uint64_t;

struct Vec256 {
    std::uint64_t q[4];
};

// Guest reads can fault even when the result is dead; signal fences + volatile preserve fault order.
static inline std::uint32_t load32(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint32_t value = 0u;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    const volatile std::uint32_t observed = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return observed;
}

static inline std::uint64_t load64(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint64_t value = 0ull;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    const volatile std::uint64_t observed = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return observed;
}

static inline Vec256 load128(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint64_t tmp[2]{};
    std::memcpy(tmp, reinterpret_cast<const void*>(address), sizeof(tmp));
    const volatile std::uint64_t q0 = tmp[0];
    const volatile std::uint64_t q1 = tmp[1];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, 0ull, 0ull}};
}

static inline Vec256 load256(word address) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    std::uint64_t tmp[4]{};
    std::memcpy(tmp, reinterpret_cast<const void*>(address), sizeof(tmp));
    const volatile std::uint64_t q0 = tmp[0];
    const volatile std::uint64_t q1 = tmp[1];
    const volatile std::uint64_t q2 = tmp[2];
    const volatile std::uint64_t q3 = tmp[3];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return {{q0, q1, q2, q3}};
}

static inline void store128(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 16);
}

static inline void store256(word address, const Vec256& v) {
    std::memcpy(reinterpret_cast<void*>(address), v.q, 32);
}

static inline Vec256 copy128(const Vec256& v) {
    return {{v.q[0], v.q[1], 0ull, 0ull}};
}

static inline bool host_daz_enabled() {
#if defined(_MSC_VER) || defined(__SSE__) || defined(__x86_64__)
    return (_mm_getcsr() & 0x0040u) != 0u;
#else
    return false;
#endif
}

static inline std::uint8_t resolve_round_mode(std::uint8_t imm8) {
    if ((imm8 & 0x04u) == 0u) {
        return imm8 & 0x03u;
    }
    switch (std::fegetround()) {
        case FE_DOWNWARD: return 1u;
        case FE_UPWARD: return 2u;
        case FE_TOWARDZERO: return 3u;
        default: return 0u;
    }
}

static inline std::uint32_t round_f32_bits(std::uint32_t bits, std::uint8_t imm8) {
    const std::uint32_t sign = bits & 0x80000000u;
    const std::uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) {
        if ((abs_bits & 0x00400000u) == 0u) {
            std::feraiseexcept(FE_INVALID);
        }
        return bits | 0x00400000u;
    }
    if (abs_bits == 0u || (abs_bits < 0x00800000u && host_daz_enabled())) {
        return sign;
    }
    if (abs_bits >= 0x4b000000u) {
        return bits;
    }
    const std::uint8_t mode = resolve_round_mode(imm8);
    const int exp = static_cast<int>(abs_bits >> 23) - 127;
    if (exp < 0) {
        bool inc = false;
        if (mode == 0u) {
            inc = (abs_bits > 0x3f000000u);
        } else if (mode == 1u) {
            inc = (sign != 0u);
        } else if (mode == 2u) {
            inc = (sign == 0u);
        }
        if ((imm8 & 0x08u) == 0u) {
            std::feraiseexcept(FE_INEXACT);
        }
        return sign | (inc ? 0x3f800000u : 0u);
    }
    const unsigned frac_bits = static_cast<unsigned>(23 - exp);
    const std::uint32_t frac_mask = (1u << frac_bits) - 1u;
    const std::uint32_t frac = abs_bits & frac_mask;
    if (frac == 0u) {
        return bits;
    }
    const std::uint32_t half = 1u << (frac_bits - 1u);
    const std::uint32_t ulp = 1u << frac_bits;
    const std::uint32_t trunc_abs = abs_bits & ~frac_mask;
    bool inc = false;
    if (mode == 0u) {
        inc = (frac > half) || (frac == half && (trunc_abs & ulp) != 0u);
    } else if (mode == 1u) {
        inc = (sign != 0u);
    } else if (mode == 2u) {
        inc = (sign == 0u);
    }
    if ((imm8 & 0x08u) == 0u) {
        std::feraiseexcept(FE_INEXACT);
    }
    return sign | (inc ? (trunc_abs + ulp) : trunc_abs);
}

static inline std::uint64_t round_f64_bits(std::uint64_t bits, std::uint8_t imm8) {
    const std::uint64_t sign = bits & 0x8000000000000000ull;
    const std::uint64_t abs_bits = bits & 0x7fffffffffffffffull;
    if (abs_bits > 0x7ff0000000000000ull) {
        if ((abs_bits & 0x0008000000000000ull) == 0ull) {
            std::feraiseexcept(FE_INVALID);
        }
        return bits | 0x0008000000000000ull;
    }
    if (abs_bits == 0ull || (abs_bits < 0x0010000000000000ull && host_daz_enabled())) {
        return sign;
    }
    if (abs_bits >= 0x4330000000000000ull) {
        return bits;
    }
    const std::uint8_t mode = resolve_round_mode(imm8);
    const int exp = static_cast<int>(abs_bits >> 52) - 1023;
    if (exp < 0) {
        bool inc = false;
        if (mode == 0u) {
            inc = (abs_bits > 0x3fe0000000000000ull);
        } else if (mode == 1u) {
            inc = (sign != 0ull);
        } else if (mode == 2u) {
            inc = (sign == 0ull);
        }
        if ((imm8 & 0x08u) == 0u) {
            std::feraiseexcept(FE_INEXACT);
        }
        return sign | (inc ? 0x3ff0000000000000ull : 0ull);
    }
    const unsigned frac_bits = static_cast<unsigned>(52 - exp);
    const std::uint64_t frac_mask = (1ull << frac_bits) - 1ull;
    const std::uint64_t frac = abs_bits & frac_mask;
    if (frac == 0ull) {
        return bits;
    }
    const std::uint64_t half = 1ull << (frac_bits - 1u);
    const std::uint64_t ulp = 1ull << frac_bits;
    const std::uint64_t trunc_abs = abs_bits & ~frac_mask;
    bool inc = false;
    if (mode == 0u) {
        inc = (frac > half) || (frac == half && (trunc_abs & ulp) != 0ull);
    } else if (mode == 1u) {
        inc = (sign != 0ull);
    } else if (mode == 2u) {
        inc = (sign == 0ull);
    }
    if ((imm8 & 0x08u) == 0u) {
        std::feraiseexcept(FE_INEXACT);
    }
    return sign | (inc ? (trunc_abs + ulp) : trunc_abs);
}

static inline std::uint64_t round_f32_pair(std::uint64_t q, std::uint8_t imm8) {
    const std::uint32_t lo = round_f32_bits(static_cast<std::uint32_t>(q), imm8);
    const std::uint32_t hi = round_f32_bits(static_cast<std::uint32_t>(q >> 32), imm8);
    return static_cast<std::uint64_t>(lo) | (static_cast<std::uint64_t>(hi) << 32);
}

static inline Vec256 op_vroundps_128(const Vec256& src, std::uint8_t imm8) {
    return {{round_f32_pair(src.q[0], imm8), round_f32_pair(src.q[1], imm8), 0ull, 0ull}};
}

static inline Vec256 op_vroundps_256(const Vec256& src, std::uint8_t imm8) {
    return {{
        round_f32_pair(src.q[0], imm8),
        round_f32_pair(src.q[1], imm8),
        round_f32_pair(src.q[2], imm8),
        round_f32_pair(src.q[3], imm8),
    }};
}

static inline Vec256 op_vroundpd_128(const Vec256& src, std::uint8_t imm8) {
    return {{round_f64_bits(src.q[0], imm8), round_f64_bits(src.q[1], imm8), 0ull, 0ull}};
}

static inline Vec256 op_vroundpd_256(const Vec256& src, std::uint8_t imm8) {
    return {{
        round_f64_bits(src.q[0], imm8),
        round_f64_bits(src.q[1], imm8),
        round_f64_bits(src.q[2], imm8),
        round_f64_bits(src.q[3], imm8),
    }};
}

static inline Vec256 op_vroundss(const Vec256& src1, std::uint32_t src2_bits, std::uint8_t imm8) {
    const std::uint32_t r0 = round_f32_bits(src2_bits, imm8);
    const std::uint64_t q0 = (src1.q[0] & 0xffffffff00000000ull) | static_cast<std::uint64_t>(r0);
    return {{q0, src1.q[1], 0ull, 0ull}};
}

static inline Vec256 op_vroundsd(const Vec256& src1, std::uint64_t src2_bits, std::uint8_t imm8) {
    return {{round_f64_bits(src2_bits, imm8), src1.q[1], 0ull, 0ull}};
}
"""


def vec_index(register, ins):
    """Map volatile XMM0..XMM5 or YMM0..YMM5 to (index, is_ymm)."""
    if register in VOLATILE_XMM:
        return VOLATILE_XMM[register], False
    if register in VOLATILE_YMM:
        return VOLATILE_YMM[register], True
    if register in NONVOLATILE_VEC:
        raise ValueError(f"nonvolatile vector register {register} at RVA {ins.ip:#x}: {ins}")
    raise ValueError(f"unsupported vector register {register} at RVA {ins.ip:#x}: {ins}")


def flat_address(ins, *, expected_bytes, store=False):
    """Lower a flat [rcx|rdx|r8] (zero-displacement, no index, exact width) memory operand."""
    base = ins.memory_base
    if (ins.is_ip_rel_memory_operand or base in (R.NONE, R.RIP, R.EIP, R.RSP, R.ESP)
            or ins.memory_index != R.NONE or ins.memory_displacement != 0
            or not RegisterExt.is_gpr64(base) or base not in ARGUMENTS
            or MemorySizeExt.size(ins.memory_size) != expected_bytes):
        raise ValueError(f"unsupported memory operand at RVA {ins.ip:#x}: {ins}")
    if store and base != R.R8:
        raise ValueError(f"store outside output buffer r8 at RVA {ins.ip:#x}: {ins}")
    return ARGUMENTS[base]


def lift_code(name, start, raw):
    """Decode and lift one bounded straight-line AVX rounding leaf function."""
    instructions = list(Decoder(64, raw, ip=start))
    if not instructions or instructions[-1].next_ip != start + len(raw):
        raise ValueError(f"truncated instruction stream in {name} at RVA {start:#x}")
    if any(ins.is_invalid for ins in instructions):
        bad = next(ins for ins in instructions if ins.is_invalid)
        raise ValueError(f"invalid instruction at RVA {bad.ip:#x}")

    used_vec = set()
    defined_vec = set()
    statements = []
    returned = False

    def read_vec(op_index, ins, *, expect_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector register operand {op_index} at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != expect_ymm:
            raise ValueError(f"mismatched vector width at RVA {ins.ip:#x}: {ins}")
        if idx not in defined_vec:
            raise ValueError(f"undefined vector register ymm{idx} at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return f"ymm{idx}"

    def write_vec(op_index, ins, *, expect_ymm):
        if ins.op_kind(op_index) != OpKind.REGISTER:
            raise ValueError(f"expected vector register destination at RVA {ins.ip:#x}: {ins}")
        idx, wide = vec_index(ins.op_register(op_index), ins)
        if wide != expect_ymm:
            raise ValueError(f"mismatched vector width at RVA {ins.ip:#x}: {ins}")
        used_vec.add(idx)
        return idx, f"ymm{idx}"

    for ins in instructions:
        if returned:
            raise ValueError(f"unreachable instruction after ret at RVA {ins.ip:#x}: {ins}")
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported instruction prefix at RVA {ins.ip:#x}: {ins}")
        op_code = ins.op_code()
        if (op_code.encoding == EncodingKind.EVEX or ins.op_mask != R.NONE
                or ins.zeroing_masking or ins.is_broadcast):
            raise ValueError(f"unsupported EVEX instruction at RVA {ins.ip:#x}: {ins}")
        if ins.code in LEGACY_SSE_REJECT_CODES:
            raise ValueError(f"unsupported legacy SSE encoding at RVA {ins.ip:#x}: {ins}")
        if ins.code not in ALLOWED_CODES:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

        if ins.code == Code.RETNQ:
            statements.append("return;")
            returned = True

        elif ins.code == Code.VEX_VZEROUPPER:
            for idx in sorted(defined_vec):
                statements.append(f"ymm{idx}.q[2] = 0ull; ymm{idx}.q[3] = 0ull;")

        elif ins.code in VMOVDQU_CODES:
            wide = VMOVDQU_CODES[ins.code]
            expected_bytes = 32 if wide else 16
            if ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.MEMORY:
                dst_idx, dst = write_vec(0, ins, expect_ymm=wide)
                addr = flat_address(ins, expected_bytes=expected_bytes, store=False)
                statements.append(f"{dst} = {'load256' if wide else 'load128'}({addr});")
                defined_vec.add(dst_idx)
            elif ins.op0_kind == OpKind.MEMORY and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expect_ymm=wide)
                addr = flat_address(ins, expected_bytes=expected_bytes, store=True)
                statements.append(f"{'store256' if wide else 'store128'}({addr}, {src});")
            elif ins.op0_kind == OpKind.REGISTER and ins.op1_kind == OpKind.REGISTER:
                src = read_vec(1, ins, expect_ymm=wide)
                dst_idx, dst = write_vec(0, ins, expect_ymm=wide)
                statements.append(f"{dst} = {src};" if wide else f"{dst} = copy128({src});")
                defined_vec.add(dst_idx)
            else:
                raise ValueError(f"unsupported vmovdqu operands at RVA {ins.ip:#x}: {ins}")

        elif ins.code in PACKED_ROUND_CODES:
            helper, wide = PACKED_ROUND_CODES[ins.code]
            imm = ins.immediate8 & 0xff
            if ins.op1_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_bytes=32 if wide else 16, store=False)
                src = f"{'load256' if wide else 'load128'}({addr})"
            else:
                src = read_vec(1, ins, expect_ymm=wide)
            dst_idx, dst = write_vec(0, ins, expect_ymm=wide)
            statements.append(f"{dst} = {helper}({src}, {imm:#x}u);")
            defined_vec.add(dst_idx)

        elif ins.code in SCALAR_ROUND_CODES:
            helper, mem_bytes, loader = SCALAR_ROUND_CODES[ins.code]
            imm = ins.immediate8 & 0xff
            src1 = read_vec(1, ins, expect_ymm=False)
            if ins.op2_kind == OpKind.MEMORY:
                addr = flat_address(ins, expected_bytes=mem_bytes, store=False)
                src2 = f"{loader}({addr})"
            else:
                src2_reg = read_vec(2, ins, expect_ymm=False)
                src2 = f"static_cast<std::uint32_t>({src2_reg}.q[0])" if mem_bytes == 4 else f"{src2_reg}.q[0]"
            dst_idx, dst = write_vec(0, ins, expect_ymm=False)
            statements.append(f"{dst} = {helper}({src1}, {src2}, {imm:#x}u);")
            defined_vec.add(dst_idx)

        else:
            raise ValueError(f"unsupported instruction at RVA {ins.ip:#x}: {ins}")

    if not returned:
        raise ValueError(f"missing terminal ret in {name} at RVA {start + len(raw):#x}")

    decls = [f"    Vec256 ymm{idx}{{}};" for idx in sorted(used_vec)]
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
