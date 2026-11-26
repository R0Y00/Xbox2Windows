"""Differential, edge-case, MXCSR, refusal, CLI, and coverage checks for AVX VEX FP compare AOT."""

import ctypes
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

from avx_horiz_fp import _guarded_buf  # noqa: E402
from cpu_avx_cmp_aot import lift_code, main as tool_main, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402


VCMPPS_PRED_EXPORTS = tuple(f"op_vcmpps_ymm_p{p:02x}" for p in range(32))
VCMPPD_PRED_EXPORTS = tuple(f"op_vcmppd_ymm_p{p:02x}" for p in range(32))
VCMPSS_PRED_EXPORTS = tuple(f"op_vcmpss_p{p:02x}" for p in range(32))
VCMPSD_PRED_EXPORTS = tuple(f"op_vcmpsd_p{p:02x}" for p in range(32))

EXTRA_EXPORTS = (
    "op_vcmpps_xmm_reg",
    "op_vcmppd_xmm_reg",
    "op_vcmpps_xmm_mem",
    "op_vcmppd_xmm_mem",
    "op_vcmpps_ymm_mem_rcx",
    "op_vcmpps_ymm_mem_rdx",
    "op_vcmpps_ymm_mem_r8",
    "op_vcmppd_ymm_mem_rcx",
    "op_vcmppd_ymm_mem_rdx",
    "op_vcmppd_ymm_mem_r8",
    "op_vcmpss_mem_rcx",
    "op_vcmpss_mem_rdx",
    "op_vcmpss_mem_r8",
    "op_vcmpsd_mem_rcx",
    "op_vcmpsd_mem_rdx",
    "op_vcmpsd_mem_r8",
    "op_vcmpps_imm_highbits",
    "op_vcmpps_alias_dst_src1",
    "op_vcmppd_alias_dst_src2",
    "op_vcmpps_alias_src1_src2",
    "op_vcmpsd_alias_all",
    "op_vmovss_mem_glue",
    "op_vmovsd_mem_glue",
    "op_vmovss_reg_merge_glue",
    "op_vmovsd_reg_merge_glue",
    "op_vmov_aligned_glue",
)

EXPORTS = (
    VCMPPS_PRED_EXPORTS
    + VCMPPD_PRED_EXPORTS
    + VCMPSS_PRED_EXPORTS
    + VCMPSD_PRED_EXPORTS
    + EXTRA_EXPORTS
)

# (export_name, ref_export_override_or_None)
ALTERED_CASES = (
    ("op_vcmpps_ymm_p00", "op_vcmpps_ymm_p04"),
    ("op_vcmppd_ymm_p01", "op_vcmppd_ymm_p05"),
    ("op_vcmpss_p02", None),
    ("op_vcmpsd_mem_rdx", None),
    ("op_vcmpps_xmm_reg", "op_vcmpps_ymm_p00"),
)

SEED = bytes(0xA0 + (i % 16) for i in range(32))
MXCSR_DEFAULT = 0x1F80
MXCSR_DAZ = 0x1FC0


def encode_seq(*instructions):
    encoder = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += encoder.encode(ins, rip)
    return encoder.take_buffer()


def _is_nan_f32(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) > 0x7F800000


def _is_nan_f64(bits: int) -> bool:
    return (bits & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000


def _is_subnormal_f32(bits: int) -> bool:
    return 0 < (bits & 0x7FFFFFFF) < 0x00800000


def _is_subnormal_f64(bits: int) -> bool:
    return 0 < (bits & 0x7FFFFFFFFFFFFFFF) < 0x0010000000000000


def _daz_f32(bits: int, daz: bool) -> float:
    if daz and _is_subnormal_f32(bits):
        bits &= 0x80000000
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _daz_f64(bits: int, daz: bool) -> float:
    if daz and _is_subnormal_f64(bits):
        bits &= 0x8000000000000000
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFFFFFFFFFFFFFF))[0]


def cmp_pred_bool(x: float, y: float, unord: bool, pred: int) -> bool:
    """Exact boolean semantics for all 32 5-bit AVX FP compare predicates (imm8 & 0x1F)."""
    p = pred & 0x0F
    if p == 0x00:
        return not unord and x == y
    if p == 0x01:
        return not unord and x < y
    if p == 0x02:
        return not unord and x <= y
    if p == 0x03:
        return unord
    if p == 0x04:
        return unord or x != y
    if p == 0x05:
        return unord or not (x < y)
    if p == 0x06:
        return unord or not (x <= y)
    if p == 0x07:
        return not unord
    if p == 0x08:
        return unord or x == y
    if p == 0x09:
        return unord or x < y
    if p == 0x0A:
        return unord or x <= y
    if p == 0x0B:
        return False
    if p == 0x0C:
        return not unord and x != y
    if p == 0x0D:
        return not unord and x >= y
    if p == 0x0E:
        return not unord and x > y
    return True


def ref_vcmpps(a: bytes, b: bytes, pred: int, wide: bool, daz: bool = False) -> bytes:
    count = 8 if wide else 4
    xa = struct.unpack("<8I", a)
    xb = struct.unpack("<8I", b)
    masks = []
    for i in range(count):
        unord = _is_nan_f32(xa[i]) or _is_nan_f32(xb[i])
        fx, fy = _daz_f32(xa[i], daz), _daz_f32(xb[i], daz)
        masks.append(0xFFFFFFFF if cmp_pred_bool(fx, fy, unord, pred) else 0)
    if not wide:
        masks.extend([0, 0, 0, 0])
    return struct.pack("<8I", *masks)


def ref_vcmppd(a: bytes, b: bytes, pred: int, wide: bool, daz: bool = False) -> bytes:
    count = 4 if wide else 2
    xa = struct.unpack("<4Q", a)
    xb = struct.unpack("<4Q", b)
    masks = []
    for i in range(count):
        unord = _is_nan_f64(xa[i]) or _is_nan_f64(xb[i])
        dx, dy = _daz_f64(xa[i], daz), _daz_f64(xb[i], daz)
        masks.append(0xFFFFFFFFFFFFFFFF if cmp_pred_bool(dx, dy, unord, pred) else 0)
    if not wide:
        masks.extend([0, 0])
    return struct.pack("<4Q", *masks)


def ref_vcmpss(a: bytes, b: bytes, pred: int, daz: bool = False) -> bytes:
    xa = list(struct.unpack("<4I", a[:16]))
    xb0 = struct.unpack("<I", b[:4])[0]
    unord = _is_nan_f32(xa[0]) or _is_nan_f32(xb0)
    xa[0] = 0xFFFFFFFF if cmp_pred_bool(_daz_f32(xa[0], daz), _daz_f32(xb0, daz), unord, pred) else 0
    return struct.pack("<4I", *xa) + b"\x00" * 16


def ref_vcmpsd(a: bytes, b: bytes, pred: int, daz: bool = False) -> bytes:
    xa = list(struct.unpack("<2Q", a[:16]))
    xb0 = struct.unpack("<Q", b[:8])[0]
    unord = _is_nan_f64(xa[0]) or _is_nan_f64(xb0)
    xa[0] = 0xFFFFFFFFFFFFFFFF if cmp_pred_bool(_daz_f64(xa[0], daz), _daz_f64(xb0, daz), unord, pred) else 0
    return struct.pack("<2Q", *xa) + b"\x00" * 16


def ref_export(name: str, a: bytes, b: bytes, seed: bytes, daz: bool = False) -> bytes:
    if name.startswith("op_vcmpps_ymm_p"):
        return ref_vcmpps(a, b, int(name[-2:], 16), wide=True, daz=daz)
    if name.startswith("op_vcmppd_ymm_p"):
        return ref_vcmppd(a, b, int(name[-2:], 16), wide=True, daz=daz)
    if name.startswith("op_vcmpss_p"):
        return ref_vcmpss(a, b, int(name[-2:], 16), daz=daz)
    if name.startswith("op_vcmpsd_p"):
        return ref_vcmpsd(a, b, int(name[-2:], 16), daz=daz)
    if name == "op_vcmpps_xmm_reg":
        return ref_vcmpps(a, b, 0x00, wide=False, daz=daz)
    if name == "op_vcmppd_xmm_reg":
        return ref_vcmppd(a, b, 0x0C, wide=False, daz=daz)
    if name == "op_vcmpps_xmm_mem":
        return ref_vcmpps(a, b, 0x02, wide=False, daz=daz)
    if name == "op_vcmppd_xmm_mem":
        return ref_vcmppd(a, b, 0x11, wide=False, daz=daz)
    if name == "op_vcmpps_ymm_mem_rcx":
        return ref_vcmpps(b, a, 0x01, wide=True, daz=daz)
    if name == "op_vcmpps_ymm_mem_rdx":
        return ref_vcmpps(a, b, 0x0E, wide=True, daz=daz)
    if name == "op_vcmpps_ymm_mem_r8":
        return ref_vcmpps(a, seed, 0x12, wide=True, daz=daz)
    if name == "op_vcmppd_ymm_mem_rcx":
        return ref_vcmppd(b, a, 0x0D, wide=True, daz=daz)
    if name == "op_vcmppd_ymm_mem_rdx":
        return ref_vcmppd(a, b, 0x15, wide=True, daz=daz)
    if name == "op_vcmppd_ymm_mem_r8":
        return ref_vcmppd(a, seed, 0x1E, wide=True, daz=daz)
    if name == "op_vcmpss_mem_rcx":
        return ref_vcmpss(b, a, 0x01, daz=daz)
    if name == "op_vcmpss_mem_rdx":
        return ref_vcmpss(a, b, 0x11, daz=daz)
    if name == "op_vcmpss_mem_r8":
        return ref_vcmpss(a, seed, 0x0C, daz=daz)
    if name == "op_vcmpsd_mem_rcx":
        return ref_vcmpsd(b, a, 0x02, daz=daz)
    if name == "op_vcmpsd_mem_rdx":
        return ref_vcmpsd(a, b, 0x0E, daz=daz)
    if name == "op_vcmpsd_mem_r8":
        return ref_vcmpsd(a, seed, 0x1D, daz=daz)
    if name == "op_vcmpps_imm_highbits":
        return ref_vcmpps(a, b, 0x05, wide=True, daz=daz)
    if name == "op_vcmpps_alias_dst_src1":
        return ref_vcmpps(a, b, 0x04, wide=True, daz=daz)
    if name == "op_vcmppd_alias_dst_src2":
        return ref_vcmppd(a, b, 0x06, wide=True, daz=daz)
    if name == "op_vcmpps_alias_src1_src2":
        return ref_vcmpps(a, a, 0x03, wide=True, daz=daz)
    if name == "op_vcmpsd_alias_all":
        return ref_vcmpsd(a, a, 0x00, daz=daz)
    if name == "op_vmovss_mem_glue":
        res = ref_vcmpss(a[:4] + b"\x00" * 12, b[:4] + b"\x00" * 12, 0x01, daz=daz)
        return res[:4] + seed[4:]
    if name == "op_vmovsd_mem_glue":
        res = ref_vcmpsd(a[:8] + b"\x00" * 8, b[:8] + b"\x00" * 8, 0x01, daz=daz)
        return res[:8] + seed[8:]
    if name == "op_vmovss_reg_merge_glue":
        return b[:4] + a[4:16] + b"\x00" * 16
    if name == "op_vmovsd_reg_merge_glue":
        return b[:8] + a[8:16] + b"\x00" * 16
    if name == "op_vmov_aligned_glue":
        return ref_vcmpps(a, b, 0x0C, wide=True, daz=daz)
    raise KeyError(name)


def check_translator(original_path: Path) -> None:
    source_cpp, meta = translate(original_path, EXPORTS)
    assert len(meta) == len(EXPORTS) and "host_has_avx" in source_cpp, len(meta)
    assert "<immintrin.h>" not in source_cpp

    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVUPS_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVUPD_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX))
    ld_xmm2 = Instruction.create_reg_mem(Code.VEX_VMOVAPD_XMM_XMMM128, R.XMM2, MemoryOperand(R.RDX))
    ld_xmm4 = Instruction.create_reg_mem(Code.VEX_VMOVUPS_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_xmm5 = Instruction.create_reg_mem(Code.VEX_VMOVUPS_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX))
    ld_ymm4 = Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM4, MemoryOperand(R.RCX))
    ld_ymm5 = Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM5, MemoryOperand(R.RDX))
    st_xmm0 = Instruction.create_mem_reg(Code.VEX_VMOVUPS_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_ymm0 = Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.R8), R.YMM0)
    vzero = Instruction.create(Code.VEX_VZEROUPPER)
    ret_ins = Instruction.create(Code.RETNQ)

    # Positive lift checks across all 6 VCMP Codes, reg & [rcx|rdx|r8] mem-src, and alternate encodings.
    accepted = 0
    for code, wide, helper, is_scalar, is_double in (
        (Code.VEX_VCMPPS_XMM_XMM_XMMM128_IMM8, False, "op_vcmpps", False, False),
        (Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, True, "op_vcmpps", False, False),
        (Code.VEX_VCMPPD_XMM_XMM_XMMM128_IMM8, False, "op_vcmppd", False, True),
        (Code.VEX_VCMPPD_YMM_YMM_YMMM256_IMM8, True, "op_vcmppd", False, True),
        (Code.VEX_VCMPSS_XMM_XMM_XMMM32_IMM8, False, "op_vcmpss", True, False),
        (Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8, False, "op_vcmpsd", True, True),
    ):
        dst, s1, s2 = (R.YMM4, R.YMM4, R.YMM5) if wide else (R.XMM4, R.XMM4, R.XMM5)
        loads = (ld_ymm4, ld_ymm5) if wide else (ld_xmm4, ld_xmm5)
        st_code = Code.VEX_VMOVUPS_YMMM256_YMM if wide else Code.VEX_VMOVUPS_XMMM128_XMM
        reg_src, _ = lift_code(
            f"{helper}_reg",
            0x1000,
            encode_seq(
                *loads,
                Instruction.create_reg_reg_reg_u32(code, dst, s1, s2, 0x3F),
                Instruction.create_mem_reg(st_code, MemoryOperand(R.R8), dst),
                vzero,
                ret_ins,
            ),
        )
        assert f"{helper}(" in reg_src and "0x1fu" in reg_src, reg_src
        accepted += 1

        for base in (R.RCX, R.RDX, R.R8):
            mem_src, _ = lift_code(
                f"{helper}_mem_{base}",
                0x1000,
                encode_seq(
                    loads[0],
                    Instruction.create_reg_reg_mem_u32(code, dst, s1, MemoryOperand(base), 0x11),
                    Instruction.create_mem_reg(st_code, MemoryOperand(R.R8), dst),
                    ret_ins,
                ),
            )
            loader = (
                ("load64" if is_double else "load32")
                if is_scalar
                else ("load256" if wide else "load128")
            )
            assert f"{loader}(" in mem_src, mem_src
            accepted += 1

    # Alternate VMOVSS/VMOVSD 0F11 encodings.
    for alt_code, helper in (
        (Code.VEX_VMOVSS_XMM_XMM_XMM_0F11, "op_vmovss_reg"),
        (Code.VEX_VMOVSD_XMM_XMM_XMM_0F11, "op_vmovsd_reg"),
    ):
        alt_src, _ = lift_code(
            f"alt_{helper}",
            0x1000,
            encode_seq(
                ld_xmm4,
                ld_xmm5,
                Instruction.create_reg_reg_reg(alt_code, R.XMM0, R.XMM4, R.XMM5),
                st_xmm0,
                ret_ins,
            ),
        )
        assert f"{helper}(ymm4," in alt_src, alt_src
        accepted += 1

    evex_ps = Instruction.create_reg_reg_reg_u32(
        Code.EVEX_VCMPPS_KR_K1_XMM_XMMM128B32_IMM8, R.K1, R.XMM0, R.XMM1, 0
    )
    evex_pd = Instruction.create_reg_reg_reg_u32(
        Code.EVEX_VCMPPD_KR_K1_YMM_YMMM256B64_IMM8, R.K1, R.YMM1, R.YMM2, 0
    )
    evex_ss = Instruction.create_reg_reg_reg_u32(
        Code.EVEX_VCMPSS_KR_K1_XMM_XMMM32_IMM8_SAE, R.K1, R.XMM1, R.XMM2, 0
    )
    evex_sd = Instruction.create_reg_reg_reg_u32(
        Code.EVEX_VCMPSD_KR_K1_XMM_XMMM64_IMM8_SAE, R.K1, R.XMM1, R.XMM2, 0
    )
    evex_mov = Instruction.create_reg_mem(
        Code.EVEX_VMOVAPS_YMM_K1Z_YMMM256, R.YMM0, MemoryOperand(R.RCX)
    )
    seg_ins = Instruction.create_reg_reg_mem_u32(
        Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX), 0
    )
    seg_ins.segment_prefix = R.FS

    rejects = [
        ("evex_vcmpps", (ld_xmm1, evex_ps, ret_ins), "unsupported EVEX instruction"),
        ("evex_vcmppd", (ld_ymm1, ld_ymm2, evex_pd, ret_ins), "unsupported EVEX instruction"),
        ("evex_vcmpss", (ld_xmm1, ld_xmm2, evex_ss, ret_ins), "unsupported EVEX instruction"),
        ("evex_vcmpsd", (ld_xmm1, ld_xmm2, evex_sd, ret_ins), "unsupported EVEX instruction"),
        ("evex_vmovaps", (evex_mov, ret_ins), "unsupported EVEX instruction"),
        ("legacy_cmpps", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(Code.CMPPS_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins), "unsupported legacy SSE instruction"),
        ("legacy_cmppd", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(Code.CMPPD_XMM_XMMM128_IMM8, R.XMM1, R.XMM2, 0), ret_ins), "unsupported legacy SSE instruction"),
        ("legacy_cmpss", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(Code.CMPSS_XMM_XMMM32_IMM8, R.XMM1, R.XMM2, 0), ret_ins), "unsupported legacy SSE instruction"),
        ("legacy_cmpsd", (ld_xmm1, ld_xmm2, Instruction.create_reg_reg_u32(Code.CMPSD_XMM_XMMM64_IMM8, R.XMM1, R.XMM2, 0), ret_ins), "unsupported legacy SSE instruction"),
        ("legacy_movaps", (Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM1, MemoryOperand(R.RCX)), ret_ins), "unsupported legacy SSE instruction"),
        ("other_insn_vaddps", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(Code.VEX_VADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret_ins), "unsupported instruction"),
        ("nonvol_ymm6_dst", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM6, R.YMM1, R.YMM2, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm15_src2", (ld_xmm1, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, R.XMM15, 0), ret_ins), "nonvolatile vector register"),
        ("nonvol_ymm7_mov", (Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM7, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("nonvol_xmm8_movss", (Instruction.create_reg_mem(Code.VEX_VMOVSS_XMM_M32, R.XMM8, MemoryOperand(R.RCX)), ret_ins), "nonvolatile vector register"),
        ("undef_vcmpps_src1", (ld_ymm2, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM3, R.YMM2, 0), ret_ins), "undefined vector register ymm3"),
        ("undef_vcmppd_src2", (ld_ymm1, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPPD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM4, 0), ret_ins), "undefined vector register ymm4"),
        ("undef_vcmpss_alias", (ld_xmm2, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPSS_XMM_XMM_XMMM32_IMM8, R.XMM0, R.XMM0, R.XMM2, 0), ret_ins), "undefined vector register ymm0"),
        ("undef_store", (Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.R8), R.YMM0), ret_ins), "undefined vector register ymm0"),
        ("disp_mem_rcx", (ld_ymm1, Instruction.create_reg_reg_mem_u32(Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RCX, displ=32, displ_size=1), 0), ret_ins), "unsupported memory operand"),
        ("disp_store_r8", (ld_ymm1, Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.R8, displ=32, displ_size=1), R.YMM1), ret_ins), "unsupported memory operand"),
        ("indexed_mem", (ld_ymm1, Instruction.create_reg_reg_mem_u32(Code.VEX_VCMPPD_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RDX, R.RCX, 4), 0), ret_ins), "unsupported memory operand"),
        ("rip_rel_mem", (ld_xmm1, Instruction.create_reg_reg_mem_u32(Code.VEX_VCMPSS_XMM_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.RIP, displ=0x1040, displ_size=4), 0), ret_ins), "unsupported memory operand"),
        ("non_arg_base_rax", (ld_ymm1, Instruction.create_reg_reg_mem_u32(Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, MemoryOperand(R.RAX), 0), ret_ins), "unsupported memory operand"),
        ("nonvol_base_rbx", (Instruction.create_reg_mem(Code.VEX_VMOVAPS_YMM_YMMM256, R.YMM0, MemoryOperand(R.RBX)), ret_ins), "unsupported memory operand"),
        ("addr32_ecx", (ld_xmm1, Instruction.create_reg_reg_mem_u32(Code.VEX_VCMPSD_XMM_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, MemoryOperand(R.ECX), 0), ret_ins), "unsupported memory operand"),
        ("non_r8_store_rcx", (ld_ymm1, Instruction.create_mem_reg(Code.VEX_VMOVUPS_YMMM256_YMM, MemoryOperand(R.RCX), R.YMM1), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_rdx", (ld_xmm1, Instruction.create_mem_reg(Code.VEX_VMOVSS_M32_XMM, MemoryOperand(R.RDX), R.XMM1), ret_ins), "non-r8 store memory operand"),
        ("non_r8_store_sd_rcx", (ld_xmm1, Instruction.create_mem_reg(Code.VEX_VMOVSD_M64_XMM, MemoryOperand(R.RCX), R.XMM1), ret_ins), "non-r8 store memory operand"),
        ("seg_prefix", (ld_ymm1, seg_ins, ret_ins), "unsupported instruction prefix"),
        ("missing_ret", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg_u32(Code.VEX_VCMPPS_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 0), st_ymm0), "missing terminal ret"),
        ("after_ret", (ld_ymm1, ret_ins, st_ymm0), "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejects:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted invalid sequence")

    raw_rejects = [
        ("invalid_bytes", b"\xFF\xFE\xC3", "invalid instruction"),
        ("empty_stream", b"", "truncated instruction stream"),
    ]
    for label, raw, reason in raw_rejects:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: raw stream unexpectedly accepted")

    # CLI failure exit code 1 without traceback and zero leftover temp files.
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_cpp = root / "out.cpp"
        out_dir = root / "out_dir"
        out_dir.mkdir()
        in_dir = root / "in_dir"
        in_dir.mkdir()
        for argv, label in (
            ([str(root / "missing.dll"), "op_vcmpps_ymm_p00", str(out_cpp)], "missing_dll"),
            ([str(in_dir), "op_vcmpps_ymm_p00", str(out_cpp)], "input_dir"),
            ([str(original_path), ",", str(out_cpp)], "empty_exports"),
            ([str(original_path), "missing_export_xyz", str(out_cpp)], "missing_export"),
            ([str(original_path), "op_vcmpps_ymm_p00", str(out_dir)], "output_dir"),
        ):
            try:
                tool_main(argv)
            except SystemExit as exc:
                assert exc.code == 1, (label, exc.code)
            else:
                raise AssertionError(f"{label}: CLI did not exit")
        assert not out_cpp.exists()
        assert sorted(p.name for p in root.iterdir()) == ["in_dir", "out_dir"]

    print(
        f"translator validated: {accepted} forms accepted, "
        f"{len(rejects) + len(raw_rejects)} refusal controls and 5 CLI negative checks passed"
    )


def host_has_avx(lifted_lib) -> bool:
    if hasattr(lifted_lib, "host_has_avx"):
        lifted_lib.host_has_avx.argtypes = []
        lifted_lib.host_has_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx())
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
        return struct.pack("<8I", *words)

    def _f64(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<Q", struct.pack("<d", v))[0] for v in vals]
        return struct.pack("<4Q", *words)

    f32_qnan_pos, f32_qnan_neg = 0x7FC01234, 0xFFC05678
    f32_snan_pos, f32_snan_neg = 0x7F800042, 0xFFA00099
    f32_sub_pos, f32_sub_neg = 0x00400001, 0x80400001

    f32_pairs = (
        (_f32(0x00000000, 0x80000000, 1.5, -2.25, 0x7F800000, 0xFF800000, f32_sub_pos, 0x3F800000),
         _f32(0x80000000, 0x00000000, 1.5, 0.75, 0x7F800000, 0x7F800000, 0x00000000, -1.0)),
        (_f32(f32_qnan_pos, 1.0, f32_snan_pos, f32_qnan_neg, 2.5, -3.5, f32_sub_neg, 0x7F7FFFFF),
         _f32(1.0, f32_qnan_neg, f32_snan_neg, f32_qnan_pos, -4.5, -3.5, 0x80000000, 0xFF7FFFFF)),
        # Scalar lane 0 normal/zero/Inf/NaN + distinct upper src1/src2 lanes (lane 1 has sNaN in src1!)
        (_f32(1.25, f32_snan_pos, 0x3F800000, 0x40000000, 9.0, 8.0, 7.0, 6.0),
         _f32(2.50, 0x11223344, 0x55667788, 0x99AABBCC, -1.0, -2.0, -3.0, -4.0)),
        (_f32(f32_qnan_pos, 0x12345678, 0x9ABCDEF0, 0x0F0E0D0C, 3.0, 4.0, 5.0, 6.0),
         _f32(f32_qnan_neg, 0xAABBCCDD, 0xEEFF0011, 0x22334455, 7.0, 8.0, 9.0, 10.0)),
    )

    f64_qnan_pos, f64_qnan_neg = 0x7FF8000012345678, 0xFFF8000087654321
    f64_snan_pos, f64_snan_neg = 0x7FF0000000000042, 0xFFF4000000000099
    f64_sub_pos, f64_sub_neg = 0x0008000000000001, 0x8008000000000001

    f64_pairs = (
        (_f64(0x0000000000000000, 0x8000000000000000, 1.5, -2.25),
         _f64(0x8000000000000000, 0x0000000000000000, 1.5, 0.75)),
        (_f64(0x7FF0000000000000, 0xFFF0000000000000, f64_sub_pos, f64_sub_neg),
         _f64(0x7FF0000000000000, 0x7FF0000000000000, 0x0000000000000000, 0x8000000000000000)),
        (_f64(f64_qnan_pos, f64_snan_pos, 2.5, f64_qnan_neg),
         _f64(1.0, f64_snan_neg, 2.5, f64_qnan_pos)),
        # Scalar lane 0 normal/qNaN + distinct upper src1/src2 lanes (lane 1 has sNaN in src1!)
        (_f64(1.25, f64_snan_pos, 3.0, 4.0),
         _f64(2.50, 0x1122334455667788, -3.0, -4.0)),
    )

    def invoke(dll, prefix: str, name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int, align: int = 0):
        is_rcx_mem = name.endswith("_mem_rcx")
        is_r8_mem = name.endswith("_mem_r8")
        _ra, pa = _guarded_buf(a_bytes, align if is_rcx_mem else 0)
        _rb, pb = _guarded_buf(b_bytes, 0 if (is_rcx_mem or is_r8_mem) else align)
        _rc, pc = _guarded_buf(SEED, align if is_r8_mem else 0)
        fn = getattr(dll, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = None
        previous = exchange(mxcsr)
        try:
            fn(pa, pb, pc)
        finally:
            status = exchange(previous)
        return ctypes.string_at(pc, 32), status & 0x3F

    cases = 0
    for name in EXPORTS:
        is_pd = ("pd_" in name or "vcmpsd" in name or "vmovsd" in name)
        pairs = f64_pairs if is_pd else f32_pairs
        for mxcsr, daz in ((MXCSR_DEFAULT, False), (MXCSR_DAZ, True)):
            for a_bytes, b_bytes in pairs:
                want = ref_export(name, a_bytes, b_bytes, SEED, daz=daz)
                got_n, _ = invoke(native, "", name, a_bytes, b_bytes, mxcsr, 0)
                got_l, _ = invoke(lifted, "lifted_", name, a_bytes, b_bytes, mxcsr, 0)
                if got_n != want or got_l != want:
                    raise AssertionError(
                        f"{name} mxcsr={mxcsr:#x}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
                    )
                cases += 1
        if "_mem" in name:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3, 7):
                want = ref_export(name, a_bytes, b_bytes, SEED, daz=False)
                got_n, _ = invoke(native, "", name, a_bytes, b_bytes, MXCSR_DEFAULT, off)
                got_l, _ = invoke(lifted, "lifted_", name, a_bytes, b_bytes, MXCSR_DEFAULT, off)
                if got_n != want or got_l != want:
                    raise AssertionError(
                        f"{name} align={off}: want={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
                    )
                cases += 1

    # Verify MXCSR.IE quiet vs signaling distinction on qNaN (pair 3 has lane0 qNaN without sNaN).
    qnan_ps_a, qnan_ps_b = f32_pairs[3]
    qnan_pd_a = _f64(f64_qnan_pos, 1.0, 2.0, 3.0)
    qnan_pd_b = _f64(f64_qnan_neg, 4.0, 5.0, 6.0)
    quiet_sig_checks = (
        ("op_vcmpps_ymm_p00", qnan_ps_a, qnan_ps_b, 0),  # EQ_OQ -> quiet
        ("op_vcmpps_ymm_p10", qnan_ps_a, qnan_ps_b, 1),  # EQ_OS -> signaling
        ("op_vcmpps_ymm_p11", qnan_ps_a, qnan_ps_b, 0),  # LT_OQ -> quiet
        ("op_vcmpps_ymm_p01", qnan_ps_a, qnan_ps_b, 1),  # LT_OS -> signaling
        ("op_vcmpps_ymm_p0f", qnan_ps_a, qnan_ps_b, 0),  # TRUE_UQ -> quiet
        ("op_vcmpps_ymm_p1f", qnan_ps_a, qnan_ps_b, 1),  # TRUE_US -> signaling
        ("op_vcmppd_ymm_p11", qnan_pd_a, qnan_pd_b, 0),  # LT_OQ -> quiet
        ("op_vcmppd_ymm_p01", qnan_pd_a, qnan_pd_b, 1),  # LT_OS -> signaling
        ("op_vcmpss_p11", qnan_ps_a, qnan_ps_b, 0),      # scalar LT_OQ -> quiet (ignores lane1!)
        ("op_vcmpss_p01", qnan_ps_a, qnan_ps_b, 1),      # scalar LT_OS -> signaling
        ("op_vcmpsd_p11", qnan_pd_a, qnan_pd_b, 0),      # scalar LT_OQ -> quiet
        ("op_vcmpsd_p01", qnan_pd_a, qnan_pd_b, 1),      # scalar LT_OS -> signaling
    )
    for name, a_bytes, b_bytes, want_ie in quiet_sig_checks:
        _, st_n = invoke(native, "", name, a_bytes, b_bytes, MXCSR_DEFAULT, 0)
        _, st_l = invoke(lifted, "lifted_", name, a_bytes, b_bytes, MXCSR_DEFAULT, 0)
        if (st_n & 1) != want_ie or (st_l & 1) != want_ie:
            raise AssertionError(
                f"{name} MXCSR.IE mismatch: want={want_ie} native={st_n:#x} lifted={st_l:#x}"
            )
    # Scalar sNaN in upper lane 1 must NOT trigger IE on lane 0 non-NaN operands.
    _, st_scalar_n = invoke(native, "", "op_vcmpss_p01", f32_pairs[2][0], f32_pairs[2][1], MXCSR_DEFAULT, 0)
    _, st_scalar_l = invoke(lifted, "lifted_", "op_vcmpss_p01", f32_pairs[2][0], f32_pairs[2][1], MXCSR_DEFAULT, 0)
    assert (st_scalar_n & 1) == 0 and (st_scalar_l & 1) == 0, (st_scalar_n, st_scalar_l)

    # Altered controls must diverge from original and match altered native DLL.
    for name, alt_ref_name in ALTERED_CASES:
        is_pd = "pd_" in name or "vcmpsd" in name
        a_bytes, b_bytes = (f64_pairs if is_pd else f32_pairs)[0]
        if name == "op_vcmpss_p02":
            a_bytes, b_bytes = f32_pairs[2]
            want_orig = ref_export(name, a_bytes, b_bytes, SEED)
            want_alt = ref_vcmpss(b_bytes, a_bytes, 0x02)
        elif name == "op_vcmpsd_mem_rdx":
            want_orig = ref_export(name, a_bytes, b_bytes, SEED)
            want_alt = ref_vcmpsd(a_bytes, b_bytes, 0x0A)
        else:
            want_orig = ref_export(name, a_bytes, b_bytes, SEED)
            want_alt = ref_export(alt_ref_name, a_bytes, b_bytes, SEED)
        got_alt_n, _ = invoke(altered_native, "", name, a_bytes, b_bytes, MXCSR_DEFAULT, 0)
        got_alt_l, _ = invoke(altered_lifted, "lifted_", name, a_bytes, b_bytes, MXCSR_DEFAULT, 0)
        assert got_alt_n == want_alt and got_alt_l == want_alt, (
            name, got_alt_n.hex(), got_alt_l.hex(), want_alt.hex()
        )
        assert got_alt_l != want_orig, f"altered {name} failed to diverge from original"

    print(
        f"{cases} AVX VEX FP compare cases across {len(EXPORTS)} exports, 32 predicates, "
        f"DAZ/IE modes matched native DLL and IEEE-754 reference; {len(ALTERED_CASES)} altered controls diverged"
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
    if not host_has_avx(lifted_lib):
        print("Host lacks CPUID.01H:ECX[27:28] + XGETBV[2:1]; skipping native differential check")
        sys.exit(77)

    check_differential(original_path, lifted_path, altered_path, altered_lifted_path)
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
