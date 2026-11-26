"""Differential, MXCSR, refusal, CLI, and coverage checks for legacy SSE4.1 FP AOT."""

import ctypes
from fractions import Fraction
import math
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Instruction,
    MemoryOperand,
    Mnemonic,
    Register as R,
)
import pefile

THIS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = THIS_DIR.parents[1] / "tools"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(TOOLS_DIR))

from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_sse41_fp_aot import lift_code, main as tool_main, translate  # noqa: E402
from sse34_fp import F32, F64, _dp_allowed, _guarded_buf, _round_nearest_even  # noqa: E402


EXPORTS = (
    "op_roundps_reg_0", "op_roundps_reg_1", "op_roundps_reg_2", "op_roundps_reg_3",
    "op_roundps_reg_4", "op_roundps_reg_8",
    "op_roundps_mem_rcx", "op_roundps_mem_rdx", "op_roundps_mem_r8",
    "op_roundpd_reg_0", "op_roundpd_reg_1", "op_roundpd_reg_2", "op_roundpd_reg_3",
    "op_roundpd_reg_4",
    "op_roundpd_mem_rcx", "op_roundpd_mem_rdx", "op_roundpd_mem_r8",
    "op_roundss_reg_0", "op_roundss_reg_1", "op_roundss_reg_2", "op_roundss_reg_3",
    "op_roundss_reg_4",
    "op_roundss_mem_rcx", "op_roundss_mem_rdx", "op_roundss_mem_r8",
    "op_roundsd_reg_0", "op_roundsd_reg_1", "op_roundsd_reg_2", "op_roundsd_reg_3",
    "op_roundsd_reg_4",
    "op_roundsd_mem_rcx", "op_roundsd_mem_rdx", "op_roundsd_mem_r8",
    "op_dpps_reg_f1", "op_dpps_reg_ff", "op_dpps_reg_5a", "op_dpps_reg_0f",
    "op_dpps_mem_rcx", "op_dpps_mem_rdx", "op_dpps_mem_r8",
    "op_dppd_reg_33", "op_dppd_reg_11", "op_dppd_reg_22", "op_dppd_reg_03",
    "op_dppd_mem_rcx", "op_dppd_mem_rdx", "op_dppd_mem_r8",
    "op_insertps_reg_s1_d2_z5", "op_insertps_reg_s3_d0_z0", "op_insertps_reg_s0_d3_z8",
    "op_insertps_mem_rcx", "op_insertps_mem_rdx", "op_insertps_mem_r8",
    "op_extractps_r32_lane2", "op_extractps_r64_lane3", "op_extractps_mem_lane1",
    "op_sse41_glue_pipeline",
)

ALTERED_EXPORTS = (
    "op_roundps_reg_1",
    "op_roundpd_reg_2",
    "op_roundss_reg_0",
    "op_roundsd_reg_3",
    "op_dpps_reg_ff",
    "op_dppd_reg_33",
    "op_insertps_reg_s1_d2_z5",
    "op_extractps_r32_lane2",
)

SEED = bytes(0xB0 + (i % 16) for i in range(16))
MXCSR_MODES = (0x1F80, 0x3F80, 0x5F80, 0x7F80, 0x1FC0, 0x9F80, 0x9FC0)


def encode_seq(*instructions):
    encoder = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += encoder.encode(ins, rip)
    return encoder.take_buffer()


def _u32s(buf: bytes) -> tuple[int, ...]:
    return struct.unpack("<4I", buf[:16])


def _u64s(buf: bytes) -> tuple[int, ...]:
    return struct.unpack("<2Q", buf[:16])


def _p32s(vals) -> bytes:
    return struct.pack("<4I", *vals)


def _p64s(vals) -> bytes:
    return struct.pack("<2Q", *vals)


def _round_ref(fmt, raw: int, imm: int, mxcsr: int) -> int:
    if fmt.is_nan(raw):
        return fmt.quiet(raw)
    if fmt.is_inf(raw) or fmt.is_zero(raw):
        return raw
    sa, exp, _ = fmt.unpack(raw)
    daz = bool(mxcsr & 0x0040)
    if daz and exp == 0:
        return fmt.sign_mask if sa else 0
    mode = ((mxcsr >> 13) & 3) if (imm & 4) else (imm & 3)
    f = fmt.to_fraction(raw)
    if mode == 0:
        n = _round_nearest_even(f)
    elif mode == 1:
        n = math.floor(f)
    elif mode == 2:
        n = math.ceil(f)
    else:
        n = int(f)
    return fmt.from_fraction(Fraction(n, 1), zero_sign=sa)


def _insertps_ref(dst_bytes: bytes, src_bytes: bytes, imm: int, *, mem: bool) -> bytes:
    xa = list(_u32s(dst_bytes))
    val = _u32s(src_bytes)[0 if mem else ((imm >> 6) & 3)]
    xa[(imm >> 4) & 3] = val
    for k in range(4):
        if (imm >> k) & 1:
            xa[k] = 0
    return _p32s(xa)


def ref_export(name: str, a: bytes, b: bytes, seed: bytes, mxcsr: int, *, altered: bool = False) -> bytes:
    if name.startswith("op_roundps_"):
        if name == "op_roundps_mem_rcx":
            src, imm = a, 1
        elif name == "op_roundps_mem_rdx":
            src, imm = b, 2
        elif name == "op_roundps_mem_r8":
            src, imm = seed, 3
        else:
            imm = 2 if (altered and name == "op_roundps_reg_1") else int(name.split("_")[-1])
            src = a
        return _p32s(_round_ref(F32, x, imm, mxcsr) for x in _u32s(src))

    if name.startswith("op_roundpd_"):
        if name == "op_roundpd_mem_rcx":
            src, imm = a, 0
        elif name == "op_roundpd_mem_rdx":
            src, imm = b, 1
        elif name == "op_roundpd_mem_r8":
            src, imm = seed, 2
        else:
            imm = 1 if (altered and name == "op_roundpd_reg_2") else int(name.split("_")[-1])
            src = a
        return _p64s(_round_ref(F64, x, imm, mxcsr) for x in _u64s(src))

    if name.startswith("op_roundss_"):
        if name == "op_roundss_mem_rcx":
            dst, src, imm = b, a, 1
        elif name == "op_roundss_mem_rdx":
            dst, src, imm = a, b, 2
        elif name == "op_roundss_mem_r8":
            dst, src, imm = a, seed, 3
        else:
            imm = 3 if (altered and name == "op_roundss_reg_0") else int(name.split("_")[-1])
            dst, src = a, b
        out = list(_u32s(dst))
        out[0] = _round_ref(F32, _u32s(src)[0], imm, mxcsr)
        return _p32s(out)

    if name.startswith("op_roundsd_"):
        if name == "op_roundsd_mem_rcx":
            dst, src, imm = b, a, 1
        elif name == "op_roundsd_mem_rdx":
            dst, src, imm = a, b, 2
        elif name == "op_roundsd_mem_r8":
            dst, src, imm = a, seed, 3
        else:
            imm = 2 if (altered and name == "op_roundsd_reg_3") else int(name.split("_")[-1])
            dst, src = a, b
        out = list(_u64s(dst))
        out[0] = _round_ref(F64, _u64s(src)[0], imm, mxcsr)
        return _p64s(out)

    if name.startswith("op_insertps_"):
        imms = {
            "op_insertps_reg_s1_d2_z5": (a, b, 0x60 if altered else 0x65, False),
            "op_insertps_reg_s3_d0_z0": (a, b, 0xC0, False),
            "op_insertps_reg_s0_d3_z8": (a, b, 0x38, False),
            "op_insertps_mem_rcx": (b, a, 0xD2, True),
            "op_insertps_mem_rdx": (a, b, 0xE4, True),
            "op_insertps_mem_r8": (a, seed, 0x91, True),
        }
        dst, src, imm, is_mem = imms[name]
        return _insertps_ref(dst, src, imm, mem=is_mem)

    if name == "op_extractps_r32_lane2":
        lane = 1 if altered else 2
        return struct.pack("<QQ", _u32s(a)[lane], _u64s(seed)[1])
    if name == "op_extractps_r64_lane3":
        return struct.pack("<QQ", _u32s(a)[3], _u64s(seed)[1])
    if name == "op_extractps_mem_lane1":
        return struct.pack("<I", _u32s(a)[1]) + seed[4:16]

    if name == "op_sse41_glue_pipeline":
        xmm2 = _p32s((_u32s(b)[0], *_u32s(a)[1:]))
        xmm3 = _p64s((_u64s(a)[0], _u64s(b)[1]))
        xmm4 = _p32s(_round_ref(F32, x, 1, mxcsr) for x in _u32s(xmm2))
        return _insertps_ref(xmm4, xmm3, 0x52, mem=False)

    raise KeyError(name)


def check_dp_result(name: str, a: bytes, b: bytes, seed: bytes, mxcsr: int, got_n: bytes, got_l: bytes, *, altered: bool = False) -> None:
    imms = {
        "op_dpps_reg_f1": (a, b, 0xF1),
        "op_dpps_reg_ff": (a, b, 0x55 if altered else 0xFF),
        "op_dpps_reg_5a": (a, b, 0x5A),
        "op_dpps_reg_0f": (a, b, 0x0F),
        "op_dpps_mem_rcx": (b, a, 0xF3),
        "op_dpps_mem_rdx": (a, b, 0xC5),
        "op_dpps_mem_r8": (a, seed, 0xAA),
        "op_dppd_reg_33": (a, b, 0x11 if altered else 0x33),
        "op_dppd_reg_11": (a, b, 0x11),
        "op_dppd_reg_22": (a, b, 0x22),
        "op_dppd_reg_03": (a, b, 0x03),
        "op_dppd_mem_rcx": (b, a, 0x31),
        "op_dppd_mem_rdx": (a, b, 0x32),
        "op_dppd_mem_r8": (a, seed, 0x23),
    }
    x, y, imm = imms[name]
    is_pd = name.startswith("op_dppd_")
    fmt, unpack, lanes = (F64, _u64s, 2) if is_pd else (F32, _u32s, 4)
    allowed = _dp_allowed(fmt, unpack(x), unpack(y), imm)
    for label, got in (("native", got_n), ("lifted", got_l)):
        words = unpack(got)
        for k in range(lanes):
            if not ((imm >> k) & 1) and words[k] != 0:
                raise AssertionError(f"{name} ({label}): disabled lane {k} is {words[k]:#x}, expected +0.0")
        enabled = {words[k] for k in range(lanes) if (imm >> k) & 1}
        if mxcsr == 0x1F80 and not enabled <= allowed:
            raise AssertionError(f"{name} ({label}): {enabled} not in {sorted(allowed)}")
    # Where the architectural reduction set has a single value (including single-NaN, +/-Inf,
    # signed zero, normal, and non-associative finite cancellation), require bit-for-bit equality.
    if len(allowed) == 1 and got_n != got_l:
        raise AssertionError(f"{name} mxcsr={mxcsr:#x}: native={got_n.hex()} != lifted={got_l.hex()}")


def check_translator(original_path: Path) -> None:
    source_cpp, meta = translate(original_path, EXPORTS)
    assert len(meta) == len(EXPORTS) and "host_has_sse41" in source_cpp, len(meta)
    assert "<smmintrin.h>" not in source_cpp and "<immintrin.h>" not in source_cpp

    ld_xmm0 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_xmm4 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_xmm5 = Instruction.create_reg_mem(Code.MOVDQU_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_xmm4 = Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4)
    ret_ins = Instruction.create(Code.RETNQ)

    accepted = 0
    for code, helper, needs_dst, mem_size in (
        (Code.ROUNDPS_XMM_XMMM128_IMM8, "op_roundps", False, 16),
        (Code.ROUNDPD_XMM_XMMM128_IMM8, "op_roundpd", False, 16),
        (Code.ROUNDSS_XMM_XMMM32_IMM8, "op_roundss", True, 4),
        (Code.ROUNDSD_XMM_XMMM64_IMM8, "op_roundsd", True, 8),
        (Code.DPPS_XMM_XMMM128_IMM8, "op_dpps", True, 16),
        (Code.DPPD_XMM_XMMM128_IMM8, "op_dppd", True, 16),
        (Code.INSERTPS_XMM_XMMM32_IMM8, "op_insertps", True, 4),
    ):
        loads = (ld_xmm4, ld_xmm5) if needs_dst else (ld_xmm5,)
        reg_src, _ = lift_code(
            f"{helper}_reg",
            0x1000,
            encode_seq(
                *loads,
                Instruction.create_reg_reg_u32(code, R.XMM4, R.XMM5, 0x54),
                st_xmm4,
                ret_ins,
            ),
        )
        assert f"{helper}(" in reg_src, reg_src
        accepted += 1

        for base in (R.RCX, R.RDX, R.R8):
            mem_loads = (ld_xmm4,) if needs_dst else ()
            mem_src, _ = lift_code(
                f"{helper}_mem_{base}",
                0x1000,
                encode_seq(
                    *mem_loads,
                    Instruction.create_reg_mem_u32(code, R.XMM4, MemoryOperand(base), 0x11),
                    st_xmm4,
                    ret_ins,
                ),
            )
            loader = {4: "load32", 8: "load64", 16: "load128"}[mem_size]
            assert f"{loader}(" in mem_src, mem_src
            accepted += 1

    # Alternate EXTRACTPS (r32/r64/m32) and MOVSS/MOVSD 0F11 encodings.
    for label, seq, frag in (
        ("ext_r32", (ld_xmm4, Instruction.create_reg_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, R.R10D, R.XMM4, 2), Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.R10), ret_ins), "r10 = static_cast<word>(lane32(xmm4, 2u));"),
        ("ext_r64", (ld_xmm4, Instruction.create_reg_reg_u32(Code.EXTRACTPS_R64M32_XMM_IMM8, R.R11, R.XMM4, 3), Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.R11), ret_ins), "r11 = static_cast<word>(lane32(xmm4, 3u));"),
        ("ext_m32", (ld_xmm4, Instruction.create_mem_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, MemoryOperand(R.R8), R.XMM4, 1), ret_ins), "store32(r8, lane32(xmm4, 1u));"),
        ("movss_0f11", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg(Code.MOVSS_XMMM32_XMM, R.XMM4, R.XMM5), st_xmm4, ret_ins), "set_lane32(xmm4, 0u, lane32(xmm5, 0u));"),
        ("movsd_0f11", (ld_xmm4, ld_xmm5, Instruction.create_reg_reg(Code.MOVSD_XMMM64_XMM, R.XMM4, R.XMM5), st_xmm4, ret_ins), "xmm4.q[0] = xmm5.q[0];"),
    ):
        src, _ = lift_code(label, 0x1000, encode_seq(*seq))
        assert frag in src, src
        accepted += 1

    seg_ins = Instruction.create_reg_mem_u32(Code.ROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX), 0)
    seg_ins.segment_prefix = R.GS

    rejects = [
        ("vex_vroundps", (ld_xmm1, Instruction.create_reg_reg_u32(Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vex_vdpps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(Code.VEX_VDPPS_XMM_XMM_XMMM128_IMM8, R.XMM0, R.XMM0, R.XMM1, 0xFF), ret_ins), "non-legacy encoding"),
        ("vex_vinsertps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_u32(Code.VEX_VINSERTPS_XMM_XMM_XMMM32_IMM8, R.XMM0, R.XMM0, R.XMM1, 0), ret_ins), "non-legacy encoding"),
        ("vex_vextractps", (ld_xmm0, Instruction.create_reg_reg_u32(Code.VEX_VEXTRACTPS_RM32_XMM_IMM8, R.EAX, R.XMM0, 0), ret_ins), "non-legacy encoding"),
        ("evex_vmovaps", (Instruction.create_reg_mem(Code.EVEX_VMOVAPS_XMM_K1Z_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret_ins), "non-legacy encoding"),
        ("unsupported_addps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(Code.ADDPS_XMM_XMMM128, R.XMM0, R.XMM1), ret_ins), "unsupported instruction"),
        ("unsupported_blendps", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_u32(Code.BLENDPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 1), ret_ins), "unsupported instruction"),
        ("nonvol_xmm6_roundps", (ld_xmm1, Instruction.create_reg_reg_u32(Code.ROUNDPS_XMM_XMMM128_IMM8, R.XMM6, R.XMM1, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm15_dpps", (ld_xmm0, Instruction.create_reg_reg_u32(Code.DPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM15, 0xFF), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm8_extractps", (Instruction.create_reg_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, R.EAX, R.XMM8, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_rbx_extractps", (ld_xmm0, Instruction.create_reg_reg_u32(Code.EXTRACTPS_R64M32_XMM_IMM8, R.RBX, R.XMM0, 0), ret_ins), "nonvolatile GPR"),
        ("arg_write_rcx_extractps", (ld_xmm0, Instruction.create_reg_reg_u32(Code.EXTRACTPS_R64M32_XMM_IMM8, R.RCX, R.XMM0, 0), ret_ins), "write to argument GPR"),
        ("arg_write_rdx_mov", (Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RDX, MemoryOperand(R.RCX)), ret_ins), "write to argument GPR"),
        ("undef_roundss_dst", (ld_xmm1, Instruction.create_reg_reg_u32(Code.ROUNDSS_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undef_roundsd_dst", (ld_xmm1, Instruction.create_reg_reg_u32(Code.ROUNDSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 0), ret_ins), "undefined vector register xmm0"),
        ("undef_dpps_dst", (ld_xmm1, Instruction.create_reg_reg_u32(Code.DPPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0xFF), ret_ins), "undefined vector register xmm0"),
        ("undef_insertps_src", (ld_xmm0, Instruction.create_reg_reg_u32(Code.INSERTPS_XMM_XMMM32_IMM8, R.XMM0, R.XMM3, 0), ret_ins), "undefined vector register xmm3"),
        ("undef_extractps_src", (Instruction.create_reg_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, R.EAX, R.XMM2, 0), ret_ins), "undefined vector register xmm2"),
        ("undef_gpr_store", (Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.RAX), ret_ins), "undefined GPR rax"),
        ("disp_mem_rcx", (Instruction.create_reg_mem_u32(Code.ROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX, displ=16, displ_size=1), 0), ret_ins), "unsupported memory operand"),
        ("indexed_mem", (ld_xmm0, Instruction.create_reg_mem_u32(Code.DPPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, R.RCX, 4), 0xFF), ret_ins), "unsupported memory operand"),
        ("rip_rel_mem", (ld_xmm0, Instruction.create_reg_mem_u32(Code.INSERTPS_XMM_XMMM32_IMM8, R.XMM0, MemoryOperand(R.RIP, displ=0x1040, displ_size=4), 0), ret_ins), "unsupported memory operand"),
        ("nonvol_base_rbx", (Instruction.create_reg_mem_u32(Code.ROUNDPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RBX), 0), ret_ins), "nonvolatile GPR base"),
        ("non_arg_base_rax", (Instruction.create_reg_mem_u32(Code.ROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RAX), 0), ret_ins), "unsupported memory base"),
        ("addr32_ecx", (ld_xmm0, Instruction.create_reg_mem_u32(Code.ROUNDSS_XMM_XMMM32_IMM8, R.XMM0, MemoryOperand(R.ECX), 0), ret_ins), "unsupported memory operand"),
        ("non_r8_store_extractps", (ld_xmm0, Instruction.create_mem_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, MemoryOperand(R.RDX), R.XMM0, 0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_movaps", (ld_xmm0, Instruction.create_mem_reg(Code.MOVAPS_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_gpr", (ld_xmm0, Instruction.create_reg_reg_u32(Code.EXTRACTPS_RM32_XMM_IMM8, R.EAX, R.XMM0, 0), Instruction.create_mem_reg(Code.MOV_RM32_R32, MemoryOperand(R.RDX), R.EAX), ret_ins), "non-r8 store memory operand"),
        ("seg_prefix", (seg_ins, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_xmm0, st_xmm0), "missing terminal ret"),
        ("after_ret", (ld_xmm0, ret_ins, st_xmm0), "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejects:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted invalid sequence")

    for label, raw, reason in (
        ("invalid_bytes", b"\xFF\xFE\xC3", "invalid instruction"),
        ("empty_stream", b"", "truncated instruction stream"),
    ):
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: raw stream unexpectedly accepted")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_cpp = root / "out.cpp"
        out_dir = root / "out_dir"
        out_dir.mkdir()
        in_dir = root / "in_dir"
        in_dir.mkdir()
        for argv in (
            [str(root / "missing.dll"), "op_roundps_reg_0", str(out_cpp)],
            [str(in_dir), "op_roundps_reg_0", str(out_cpp)],
            [str(original_path), ",", str(out_cpp)],
            [str(original_path), "missing_export_xyz", str(out_cpp)],
            [str(original_path), "op_roundps_reg_0", str(out_dir)],
        ):
            try:
                tool_main(argv)
            except SystemExit as exc:
                assert exc.code == 1, (argv, exc.code)
            else:
                raise AssertionError(f"CLI did not exit for {argv}")
        assert not out_cpp.exists() and sorted(p.name for p in root.iterdir()) == ["in_dir", "out_dir"]

    print(
        f"translator validated: {accepted} forms accepted, "
        f"{len(rejects) + 2} refusal controls and 5 CLI negative checks passed"
    )


def host_has_sse41(lifted_lib) -> bool:
    if hasattr(lifted_lib, "host_has_sse41"):
        lifted_lib.host_has_sse41.argtypes = []
        lifted_lib.host_has_sse41.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_sse41())
    return False


def check_differential(native_path: Path, lifted_path: Path, altered_path: Path, altered_lifted_path: Path) -> None:
    native = ctypes.CDLL(str(native_path.resolve()))
    lifted = ctypes.CDLL(str(lifted_path.resolve()))
    altered_native = ctypes.CDLL(str(altered_path.resolve()))
    altered_lifted = ctypes.CDLL(str(altered_lifted_path.resolve()))

    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes, exchange.restype = [ctypes.c_uint32], ctypes.c_uint32

    def _f32(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
        return struct.pack("<4I", *words)

    def _f64(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<Q", struct.pack("<d", v))[0] for v in vals]
        return struct.pack("<2Q", *words)

    f32_pairs = (
        (_f32(-3.5, -2.5, -0.25, 0.5), _f32(2.5, 3.5, 0.25, -0.5)),
        (_f32(0x00000000, 0x80000000, 0x00000001, 0x80000001), _f32(0x80000000, 0x00000000, 0x00400001, 0x80400001)),
        (_f32(0x7FC12345, 0x7F800123, 0x7F800000, 0xFF800000), _f32(0xFFC54321, 0xFF800456, 1.5, -2.5)),
        (_f32(0x7FC12345, 1.5, 2.0, -3.0), _f32(1.0, 2.0, -0.5, 0.25)),
        (_f32(16777215.0, -16777215.0, 1.9999999, -1.9999999), _f32(3.5, -0.5, 2.0, -3.0)),
        # Non-associative DPPS + poison upper qword for extractps.
        (_f32(1.0e20, -1.0e20, 1.0, 2.0), _f32(0xCAFEBABE, 0xFEEDFACE, 1.5, -0.25)),
    )

    f64_pairs = (
        (_f64(-3.5, -2.5), _f64(0.25, 0.5)),
        (_f64(0x0000000000000000, 0x8000000000000000), _f64(0x0000000000000001, 0x8000000000000001)),
        (_f64(0x7FF8000000012345, 0x7FF0000000000123), _f64(0xFFF8000000054321, 0xFFF0000000000456)),
        (_f64(0x7FF8000000012345, 1.5), _f64(1.0, -2.5)),
        (_f64(0x7FF0000000000000, 0xFFF0000000000000), _f64(2.5, -1.5)),
    )

    def invoke(dll, prefix: str, name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int, align: int = 0):
        is_rcx_mem = name.endswith("_mem_rcx")
        is_r8_mem = name.endswith("_mem_r8")
        _ra, pa, _ = _guarded_buf(a_bytes, align if is_rcx_mem else 0)
        _rb, pb, _ = _guarded_buf(b_bytes, 0 if (is_rcx_mem or is_r8_mem) else align)
        _rc, pc, _ = _guarded_buf(SEED, align if is_r8_mem else 0)
        fn = getattr(dll, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        previous = exchange(mxcsr)
        try:
            fn(pa, pb, pc)
        finally:
            exchange(previous)
        return ctypes.string_at(pc, 16)

    cases = 0
    for name in EXPORTS:
        is_pd = ("pd_" in name or "roundsd_" in name)
        pairs = f64_pairs if is_pd else f32_pairs
        is_dp = name.startswith(("op_dpps_", "op_dppd_"))
        is_aligned_only = (
            name.startswith(("op_roundps_", "op_roundpd_", "op_dpps_", "op_dppd_"))
            and "_mem_" in name
        )
        for mxcsr in MXCSR_MODES:
            for a_bytes, b_bytes in pairs:
                got_n = invoke(native, "", name, a_bytes, b_bytes, mxcsr, 0)
                got_l = invoke(lifted, "lifted_", name, a_bytes, b_bytes, mxcsr, 0)
                if is_dp:
                    check_dp_result(name, a_bytes, b_bytes, SEED, mxcsr, got_n, got_l)
                else:
                    want = ref_export(name, a_bytes, b_bytes, SEED, mxcsr)
                    if got_n != want or got_l != want:
                        raise AssertionError(
                            f"{name} mxcsr={mxcsr:#x}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
                        )
                cases += 1
        if "_mem_" in name and not is_aligned_only:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3, 7):
                got_n = invoke(native, "", name, a_bytes, b_bytes, 0x1F80, off)
                got_l = invoke(lifted, "lifted_", name, a_bytes, b_bytes, 0x1F80, off)
                want = ref_export(name, a_bytes, b_bytes, SEED, 0x1F80)
                if got_n != want or got_l != want:
                    raise AssertionError(f"{name} off={off}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}")
                cases += 1

    # Altered controls must match altered native and diverge from original on at least one input pair.
    for name in ALTERED_EXPORTS:
        is_pd = ("pd_" in name or "roundsd_" in name)
        pairs = f64_pairs if is_pd else f32_pairs
        diffs = 0
        for a_bytes, b_bytes in pairs:
            got_alt_n = invoke(altered_native, "", name, a_bytes, b_bytes, 0x1F80, 0)
            got_alt_l = invoke(altered_lifted, "lifted_", name, a_bytes, b_bytes, 0x1F80, 0)
            got_orig = invoke(native, "", name, a_bytes, b_bytes, 0x1F80, 0)
            if name.startswith(("op_dpps_", "op_dppd_")):
                check_dp_result(name, a_bytes, b_bytes, SEED, 0x1F80, got_alt_n, got_alt_l, altered=True)
            else:
                want_alt = ref_export(name, a_bytes, b_bytes, SEED, 0x1F80, altered=True)
                assert got_alt_n == want_alt and got_alt_l == want_alt, (
                    name, got_alt_n.hex(), got_alt_l.hex(), want_alt.hex()
                )
            if got_alt_l != got_orig:
                diffs += 1
        assert diffs > 0, f"altered {name} failed to diverge"

    print(
        f"{cases} legacy SSE4.1 FP cases across {len(EXPORTS)} exports and {len(MXCSR_MODES)} MXCSR modes "
        f"matched native DLL and IEEE-754 reference; {len(ALTERED_EXPORTS)} altered controls diverged"
    )


def record_covered_forms(original_path: Path) -> None:
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    mnemonic_names = {
        val: key.lower()
        for key, val in vars(Mnemonic).items()
        if not key.startswith("_") and isinstance(val, int)
    }
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(mnemonic_names[ins.mnemonic])
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main() -> None:
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])
    check_translator(original_path)

    lifted_lib = ctypes.CDLL(str(lifted_path.resolve()))
    if not host_has_sse41(lifted_lib):
        print("Host lacks CPUID.01H:ECX[19] (SSE4.1); skipping native differential check")
        sys.exit(77)

    check_differential(original_path, lifted_path, altered_path, altered_lifted_path)
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
