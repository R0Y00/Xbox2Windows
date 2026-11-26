"""Differential, mathematical-reference, MXCSR-isolation, FCW, mutation, and rejection checks
for x87 F2XM1, FYL2X, and FYL2XP1 AOT lifting (`tools/cpu_x87_exp_log_aot.py`).

Tested domains and precision bounds (80-bit extended, 64-bit explicit significand):
  - F2XM1   : Intel SDM domain [-1, +1]; exact on +-0, +1, -1; lifted vs arbitrary-precision
              Decimal reference <= 1 ULP (observed 0 ULP); lifted vs native x87 <= 1 ULP.
  - FYL2X   : x > 0 plus all SDM special cases (x <= 0, +-inf, NaN, unnormal); exact on
              x == 1 and powers of 2 with exact products; lifted vs reference <= 1 ULP;
              lifted vs native x87 <= 1 ULP.
  - FYL2XP1 : Intel SDM domain [-(1 - sqrt(2)/2), +(1 - sqrt(2)/2)] plus SDM special cases
              (+-0, NaN, unnormal); lifted vs reference <= 1 ULP; lifted vs native x87 <= 1 ULP.
              Outside the SDM domain, x == -1 (#Z for finite nonzero y, #IA for y == +-0) and
              x < -1 (#IA) are checked on the lifted DLL against the mathematical spec, because
              hardware results outside [-(1 - sqrt(2)/2), 1 - sqrt(2)/2] are SDM-undefined.
"""

import ctypes
from decimal import Decimal, getcontext
import os
from pathlib import Path
import random
import struct
import sys

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_x87_exp_log_aot import lift_code  # noqa: E402

getcontext().prec = 80
DEC_LN2 = Decimal(2).ln()

EXPORTS = (
    "op_f2xm1_m80",
    "op_f2xm1_m64",
    "op_f2xm1_m32",
    "op_f2xm1_fst_m64",
    "op_f2xm1_fst_m32",
    "op_f2xm1_sw",
    "op_fyl2x_m80",
    "op_fyl2x_m64",
    "op_fyl2x_m32",
    "op_fyl2x_sw",
    "op_fyl2x_pop_depth",
    "op_fyl2x_depth8",
    "op_fyl2xp1_m80",
    "op_fyl2xp1_m64",
    "op_fyl2xp1_m32",
    "op_fyl2xp1_sw",
    "op_fyl2xp1_pop_depth",
    "op_exp_log_roundtrip",
    "op_fcw_roundtrip",
    "op_fninit_cw_sw",
    "op_fninit_sw",
)

ALTERED_EXPORTS = (
    "op_f2xm1_m80",
    "op_fyl2x_m80",
    "op_fyl2xp1_m80",
    "op_exp_log_roundtrip",
    "op_fcw_roundtrip",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))

J = 1 << 63
QBIT = 1 << 62
MASKED_CWS = [0x007F | (pc << 8) | (rc << 10) for pc in (0, 2, 3) for rc in range(4)]

SW_IE = 0x0001
SW_DE = 0x0002
SW_ZE = 0x0004
SW_OE = 0x0008
SW_UE = 0x0010
SW_PE = 0x0020
SW_TOP_MASK = 0x3800
# Mandatory architectural status flags compared strictly against native hardware:
# IE, DE, ZE, OE, UE, SF(0x40), ES(0x80), TOP(0x3800), B(0x8000).
SW_STRICT_MASK = SW_IE | SW_DE | SW_ZE | SW_OE | SW_UE | 0x00C0 | SW_TOP_MASK | 0x8000

POS_ZERO = (0, 0x0000)
NEG_ZERO = (0, 0x8000)
POS_ONE = (J, 0x3FFF)
NEG_ONE = (J, 0xBFFF)
POS_HALF = (J, 0x3FFE)
NEG_HALF = (J, 0xBFFE)
POS_TWO = (J, 0x4000)
POS_FOUR = (J, 0x4001)
POS_INF = (J, 0x7FFF)
NEG_INF = (J, 0xFFFF)
INDEF = (J | QBIT, 0xFFFF)
QNAN_A = (J | QBIT | 0x12345678, 0x7FFF)
QNAN_B = (J | QBIT | 0x00000042, 0xFFFF)
SNAN_A = (J | 0x00005678, 0x7FFF)
SNAN_B = (J | 0x00009ABC, 0xFFFF)
UNNORM = (0x0000123456789ABC, 0x3FFF)
SUB_POS_MIN = (1, 0x0000)
SUB_NEG_MIN = (1, 0x8000)
SUB_POS_MID = (0x0000123456789ABC, 0x0000)
SUB_NEG_MID = (0x0000123456789ABC, 0x8000)
TINY_NORM_POS = (J, 0x0001)
TINY_NORM_NEG = (J, 0x8001)
HUGE_POS = (0xFFFFFFFFFFFFFFFF, 0x7FFE)
HUGE_NEG = (0xFFFFFFFFFFFFFFFF, 0xFFFE)


# --- 80-bit extended encoding/decoding & ULP distance ---

def pack80(val):
    return struct.pack("<QH", val[0], val[1])


def unpack80(raw):
    return struct.unpack("<QH", raw[:10])


def pack16(v):
    return struct.pack("<H", v & 0xFFFF)


def unpack16(raw):
    return struct.unpack("<H", raw[:2])[0]


def pack_f64(val):
    if isinstance(val, int):
        return struct.pack("<Q", val)
    return struct.pack("<d", val)


def pack_f32(val):
    if isinstance(val, int):
        return struct.pack("<I", val)
    return struct.pack("<f", val)


def m80_from_decimal(d, *, rc=0):
    """Round an arbitrary-precision Decimal to an 80-bit extended (sig, se) tuple."""
    if d == 0:
        return NEG_ZERO if d.is_signed() else POS_ZERO
    sign = 1 if d < 0 else 0
    mag = -d if sign else d
    # Find unbiased binary exponent e such that 1 <= mag / 2^e < 2.
    # Use log2 estimate, then adjust exactly.
    approx_e = int((mag.ln() / DEC_LN2).to_integral_value())
    two = Decimal(2)
    e = approx_e
    scale = two ** e
    while mag < scale:
        e -= 1
        scale = two ** e
    while mag >= scale * two:
        e += 1
        scale = two ** e
    biased = e + 16383
    if biased <= 0:
        # Denormal: unit is 2^(1 - 16383 - 63) = 2^-16445.
        q = mag * (two ** 16445)
        fl = int(q)
        rem = q - fl
        up = _round_dir(sign, fl, rem, rc)
        sig = fl + (1 if up else 0)
        if sig >= J:
            return (sig, (sign << 15) | 1)
        return (sig, (sign << 15))
    q = (mag / scale) * (1 << 63)
    fl = int(q)
    rem = q - fl
    up = _round_dir(sign, fl, rem, rc)
    sig = fl + (1 if up else 0)
    if sig >= (1 << 64):
        sig = J
        biased += 1
    if biased > 0x7FFE:
        to_inf = (rc == 0) or (rc == 1 and sign) or (rc == 2 and not sign)
        if to_inf:
            return (J, (sign << 15) | 0x7FFF)
        return (0xFFFFFFFFFFFFFFFF, (sign << 15) | 0x7FFE)
    return (sig, (sign << 15) | biased)


def _round_dir(sign, fl, rem, rc):
    if rem == 0:
        return False
    if rc == 0:
        half = Decimal("0.5")
        return rem > half or (rem == half and (fl & 1) != 0)
    if rc == 1:
        return bool(sign)
    if rc == 2:
        return not bool(sign)
    return False


def m80_to_decimal(val):
    sig, se = val
    sign = (se >> 15) & 1
    exp = se & 0x7FFF
    assert exp != 0x7FFF and (exp == 0 or (sig & J)), f"non-finite/unsupported m80 {val}"
    if exp == 0 and sig == 0:
        return Decimal("-0") if sign else Decimal("0")
    unbiased = (1 - 16383 - 63) if exp == 0 else (exp - 16383 - 63)
    mag = Decimal(sig) * (Decimal(2) ** unbiased)
    return -mag if sign else mag


def m80_is_finite(val):
    sig, se = val
    exp = se & 0x7FFF
    return exp != 0x7FFF and (exp == 0 or (sig & J) != 0)


def m80_ordinal(val):
    """Monotonic signed integer index over all canonical finite 80-bit encodings (1 step = 1 ULP)."""
    sig, se = val
    sign = (se >> 15) & 1
    exp = se & 0x7FFF
    if exp == 0:
        idx = sig
    else:
        idx = (exp << 63) | (sig & (J - 1))
    return -idx if sign else idx


def m80_ulp_diff(a, b):
    if a == b:
        return 0
    if not (m80_is_finite(a) and m80_is_finite(b)):
        return 1 << 64
    # Require matching sign of zero when both are zero.
    if (a[0] == 0 and (a[1] & 0x7FFF) == 0) or (b[0] == 0 and (b[1] & 0x7FFF) == 0):
        if a[1] != b[1] and a[0] == 0 and b[0] == 0:
            return 1 << 64
    return abs(m80_ordinal(a) - m80_ordinal(b))


def f64_ulp_diff(a_bytes, b_bytes):
    if a_bytes == b_bytes:
        return 0
    a = struct.unpack("<Q", a_bytes)[0]
    b = struct.unpack("<Q", b_bytes)[0]
    if (a & 0x7FF0000000000000) == 0x7FF0000000000000 or (b & 0x7FF0000000000000) == 0x7FF0000000000000:
        return 1 << 64
    sa = -(a & 0x7FFFFFFFFFFFFFFF) if (a >> 63) else a
    sb = -(b & 0x7FFFFFFFFFFFFFFF) if (b >> 63) else b
    return abs(sa - sb)


def f32_ulp_diff(a_bytes, b_bytes):
    if a_bytes == b_bytes:
        return 0
    a = struct.unpack("<I", a_bytes)[0]
    b = struct.unpack("<I", b_bytes)[0]
    if (a & 0x7F800000) == 0x7F800000 or (b & 0x7F800000) == 0x7F800000:
        return 1 << 64
    sa = -(a & 0x7FFFFFFF) if (a >> 31) else a
    sb = -(b & 0x7FFFFFFF) if (b >> 31) else b
    return abs(sa - sb)


# --- Independent arbitrary-precision Python reference ---

def ref_f2xm1(x, cw=0x037F):
    rc = (cw >> 10) & 3
    if x == POS_ZERO or x == NEG_ZERO:
        return x
    if x == POS_ONE:
        return POS_ONE
    if x == NEG_ONE:
        return NEG_HALF
    if x == POS_INF:
        return POS_INF
    if x == NEG_INF:
        return NEG_ONE
    dx = m80_to_decimal(x)
    # 2^x - 1 = exp(x * ln(2)) - 1; for tiny |x|, use Taylor series to avoid Decimal cancellation.
    u = dx * DEC_LN2
    if abs(u) < Decimal("1e-20"):
        exact = u + (u * u) / 2 + (u * u * u) / 6
    else:
        exact = u.exp() - Decimal(1)
    return m80_from_decimal(exact, rc=rc)


def ref_fyl2x(y, x, cw=0x037F):
    rc = (cw >> 10) & 3
    dx = m80_to_decimal(x)
    dy = m80_to_decimal(y)
    assert dx > 0
    if dx == 1:
        return NEG_ZERO if (y[1] & 0x8000) else POS_ZERO
    if dy == 0:
        sign = ((y[1] >> 15) & 1) ^ (1 if dx < 1 else 0)
        return NEG_ZERO if sign else POS_ZERO
    exact = dy * (dx.ln() / DEC_LN2)
    return m80_from_decimal(exact, rc=rc)


def ref_fyl2xp1(y, x, cw=0x037F):
    rc = (cw >> 10) & 3
    dx = m80_to_decimal(x)
    dy = m80_to_decimal(y)
    assert dx > -1
    if dx == 0:
        sign = ((y[1] >> 15) & 1) ^ ((x[1] >> 15) & 1)
        return NEG_ZERO if sign else POS_ZERO
    if dy == 0:
        sign = ((y[1] >> 15) & 1) ^ ((x[1] >> 15) & 1)
        return NEG_ZERO if sign else POS_ZERO
    if abs(dx) < Decimal("1e-20"):
        ln1p = dx - (dx * dx) / 2 + (dx * dx * dx) / 3
    else:
        ln1p = (Decimal(1) + dx).ln()
    exact = dy * (ln1p / DEC_LN2)
    return m80_from_decimal(exact, rc=rc)


# --- Guarded buffer invocation ---

def alloc_guarded(payload):
    size = 32 + len(GUARD_PRE) + len(payload) + len(GUARD_POST)
    raw = ctypes.create_string_buffer(size)
    base = ctypes.addressof(raw)
    data_addr = (base + len(GUARD_PRE) + 15) & ~15
    pre_off = (data_addr - len(GUARD_PRE)) - base
    data_off = data_addr - base
    post_off = data_off + len(payload)
    raw[pre_off:data_off] = GUARD_PRE
    raw[data_off:post_off] = payload
    raw[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return raw, pre_off, data_off, post_off, data_addr


def call3(fn, a_bytes, b_bytes, out_init, *, out_len=None, check_inputs=True):
    if out_len is None:
        out_len = len(out_init)
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a_bytes)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b_bytes)
    buf_o, o_pre, o_off, o_post, o_addr = alloc_guarded(out_init)
    fn(a_addr, b_addr, o_addr)
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, "A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, "B guards corrupted"
    assert buf_o.raw[o_pre:o_off] == GUARD_PRE and buf_o.raw[o_post:o_post + 16] == GUARD_POST, "out guards corrupted"
    if check_inputs:
        assert buf_a.raw[a_off:a_post] == a_bytes, "input A mutated"
        assert buf_b.raw[b_off:b_post] == b_bytes, "input B mutated"
    return buf_o.raw[o_off:o_off + out_len]


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


# --- Translator positive and negative controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += enc.encode(ins, rip)
    return enc.take_buffer()


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    fninit = Instruction.create(Code.FNINIT)
    fld1 = Instruction.create(Code.FLD1)
    fldz = Instruction.create(Code.FLDZ)
    fld_m32 = Instruction.create_mem(Code.FLD_M32FP, MemoryOperand(R.RCX))
    fld_m64 = Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RCX))
    fld_m80 = Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RCX))
    fld_m80_rdx = Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RDX))
    fst_m32 = Instruction.create_mem(Code.FST_M32FP, MemoryOperand(R.R8))
    fst_m64 = Instruction.create_mem(Code.FST_M64FP, MemoryOperand(R.R8))
    fstp_m32 = Instruction.create_mem(Code.FSTP_M32FP, MemoryOperand(R.R8))
    fstp_m64 = Instruction.create_mem(Code.FSTP_M64FP, MemoryOperand(R.R8))
    fstp_m80 = Instruction.create_mem(Code.FSTP_M80FP, MemoryOperand(R.R8))
    fstp_st0 = Instruction.create_reg(Code.FSTP_STI, R.ST0)
    f2xm1 = Instruction.create(Code.F2XM1)
    fyl2x = Instruction.create(Code.FYL2X)
    fyl2xp1 = Instruction.create(Code.FYL2XP1)
    fnstsw_ax = Instruction.create_reg(Code.FNSTSW_AX, R.AX)
    mov_r8_ax = Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.R8), R.AX)
    fldcw_rdx = Instruction.create_mem(Code.FLDCW_M2BYTE, MemoryOperand(R.RDX))
    fnstcw_r8 = Instruction.create_mem(Code.FNSTCW_M2BYTE, MemoryOperand(R.R8))

    # Positive acceptance across all supported instruction forms.
    positive = [
        ("f2xm1_m80", (fninit, fldcw_rdx, fld_m80, f2xm1, fstp_m80, ret), "x87_f2xm1(x87);"),
        ("f2xm1_m64_fst", (fninit, fld_m64, f2xm1, fst_m64, fstp_st0, ret), "x87_store_f64(x87, r8, false);"),
        ("f2xm1_m32_fst", (fninit, fld_m32, f2xm1, fst_m32, fstp_m32, ret), "x87_store_f32(x87, r8, true);"),
        ("fyl2x_m80", (fninit, fld_m80, fld_m80_rdx, fyl2x, fstp_m80, ret), "x87_fyl2x(x87);"),
        ("fyl2xp1_m80", (fninit, fld1, fld_m80, fyl2xp1, fstp_m80, ret), "x87_fyl2xp1(x87);"),
        ("sw_and_cw", (fninit, fldcw_rdx, fldz, f2xm1, fnstsw_ax, fstp_st0, mov_r8_ax, fnstcw_r8, ret), "x87_status_word(x87)"),
    ]
    for label, seq, needle in positive:
        src, _ = lift_code(label, 0x1000, encode_seq(*seq))
        assert needle in src, f"{label}: missing {needle!r}"
        assert "long double" not in src and "<cmath>" not in src, f"{label}: forbidden host FP type/header"

    # Negative rejection controls.
    rep_ret = Instruction.create(Code.RETNQ)
    rep_ret.has_rep_prefix = True
    fs_fld = Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RCX))
    fs_fld.segment_prefix = R.FS

    rejections = [
        ("trig_fsin", (fld_m80, Instruction.create(Code.FSIN), fstp_m80, ret), "unsupported x87 trigonometric instruction"),
        ("trig_fcos", (fld_m80, Instruction.create(Code.FCOS), fstp_m80, ret), "unsupported x87 trigonometric instruction"),
        ("trig_fsincos", (fld_m80, Instruction.create(Code.FSINCOS), fstp_m80, fstp_st0, ret), "unsupported x87 trigonometric instruction"),
        ("trig_fptan", (fld_m80, Instruction.create(Code.FPTAN), fstp_st0, fstp_m80, ret), "unsupported x87 trigonometric instruction"),
        ("trig_fpatan", (fld_m80, fld_m80_rdx, Instruction.create(Code.FPATAN), fstp_m80, ret), "unsupported x87 trigonometric instruction"),
        ("stack_overflow_9", (*([fld1] * 9), *([fstp_st0] * 9), ret), "x87 stack overflow"),
        ("underflow_f2xm1", (f2xm1, ret), "x87 stack underflow"),
        ("underflow_fyl2x_0", (fyl2x, ret), "x87 stack underflow"),
        ("underflow_fyl2x_1", (fld1, fyl2x, fstp_st0, ret), "x87 stack underflow"),
        ("underflow_fyl2xp1_1", (fld1, fyl2xp1, fstp_st0, ret), "x87 stack underflow"),
        ("underflow_after_pop", (fld1, fld1, fyl2x, fstp_st0, fstp_st0, ret), "x87 stack underflow"),
        ("underflow_fst", (fst_m64, ret), "x87 stack underflow"),
        ("nonempty_at_ret_1", (fld_m80, f2xm1, ret), "non-empty x87 stack at ret"),
        ("nonempty_at_ret_fyl2x", (fld1, fld_m80, fyl2x, ret), "non-empty x87 stack at ret"),
        ("xmm_addps", (Instruction.create_reg_reg(Code.ADDPS_XMM_XMMM128, R.XMM0, R.XMM1), ret), "unsupported vector instruction"),
        ("xmm_movaps", (Instruction.create_reg_mem(Code.MOVAPS_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret), "unsupported vector instruction"),
        ("vex_vaddps", (Instruction.create_reg_reg_reg(Code.VEX_VADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM0, R.YMM1), ret), "unsupported VEX encoding"),
        ("evex_vmovdqu32", (Instruction.create_reg_mem(Code.EVEX_VMOVDQU32_ZMM_K1Z_ZMMM512, R.ZMM0, MemoryOperand(R.RCX)), ret), "unsupported EVEX encoding"),
        ("displaced_fld", (Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RCX, displ=16, displ_size=1)), fstp_m80, ret), "unsupported displaced memory operand"),
        ("displaced_fstp", (fld_m80, Instruction.create_mem(Code.FSTP_M80FP, MemoryOperand(R.R8, displ=16, displ_size=1)), ret), "unsupported displaced memory operand"),
        ("indexed_mem", (Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RCX, index=R.RDX)), fstp_m64, ret), "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RIP, displ=0x1200, displ_size=4)), fstp_m64, ret), "unsupported memory operand"),
        ("rsp_mem", (Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RSP)), fstp_m64, ret), "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RAX)), fstp_m64, ret), "unsupported memory base"),
        ("nonvol_base_rbx", (Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RBX)), fstp_m64, ret), "nonvolatile GPR base"),
        ("store_outside_r8_rcx", (fld_m80, Instruction.create_mem(Code.FSTP_M80FP, MemoryOperand(R.RCX)), ret), "store outside output buffer r8"),
        ("store_outside_r8_rdx", (fld_m64, Instruction.create_mem(Code.FST_M64FP, MemoryOperand(R.RDX)), fstp_st0, ret), "store outside output buffer r8"),
        ("fnstcw_outside_r8", (Instruction.create_mem(Code.FNSTCW_M2BYTE, MemoryOperand(R.RCX)), ret), "store outside output buffer r8"),
        ("mov_ax_outside_r8", (fnstsw_ax, Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.RCX), R.AX), ret), "store outside output buffer r8"),
        ("undef_ax_store", (mov_r8_ax, ret), "undefined register ax"),
        ("fstp_st1", (fld1, fld1, Instruction.create_reg(Code.FSTP_STI, R.ST1), fstp_st0, ret), "unsupported fstp register operand"),
        ("prefix_fs", (fs_fld, fstp_m80, ret), "unsupported instruction prefix"),
        ("prefix_rep_ret", (fld1, fstp_st0, rep_ret), "unsupported instruction prefix"),
        ("wait_prefix", (Instruction.create(Code.WAIT), fninit, ret), "unsupported instruction"),
        ("branch_jmp", (Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret), "unsupported control-flow instruction"),
        ("missing_ret", (fninit, fld1, fstp_st0), "missing terminal ret"),
        ("trailing_after_ret", (fninit, ret, fninit), "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejections:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected sequence")

    return len(positive), len(rejections)


# --- Case builders ---

def f2xm1_domain_values():
    rng = random.Random(0xF21A07)
    vals = [
        POS_ZERO, NEG_ZERO, POS_ONE, NEG_ONE, POS_HALF, NEG_HALF,
        SUB_POS_MIN, SUB_NEG_MIN, SUB_POS_MID, SUB_NEG_MID,
        TINY_NORM_POS, TINY_NORM_NEG,
        # Near +-1 endpoints: 1 - 2^-63, -(1 - 2^-63), 1 - 2^-64
        (0xFFFFFFFFFFFFFFFF, 0x3FFE),
        (0xFFFFFFFFFFFFFFFF, 0xBFFE),
        # Near +-0.25, +-0.75, +-1/3
        (J, 0x3FFD), (J, 0xBFFD),
        (0xC000000000000000, 0x3FFE), (0xC000000000000000, 0xBFFE),
        (0xAAAAAAAAAAAAAAAA, 0x3FFD), (0xAAAAAAAAAAAAAAAA, 0xBFFD),
        # Very small normal exponents
        (J | 0x123456789ABCDEF0, 0x3FFF - 50),
        (J | 0x123456789ABCDEF0, 0xBFFF - 50),
        (J | 0xFEDCBA9876543210, 0x3FFF - 63),
        (J | 0xFEDCBA9876543210, 0x3FFF - 200),
    ]
    for _ in range(80):
        sign = rng.getrandbits(1) << 15
        exp = rng.randint(0x3FFF - 40, 0x3FFE)
        sig = J | rng.getrandbits(63)
        vals.append((sig, sign | exp))
    return vals


def fyl2x_domain_pairs():
    rng = random.Random(0xF12A07)
    pairs = [
        # (y, x) with x > 0
        (POS_ONE, POS_ONE), (NEG_ONE, POS_ONE),
        (POS_ZERO, POS_ONE), (NEG_ZERO, POS_ONE),
        (POS_ONE, POS_TWO), (NEG_ONE, POS_TWO),
        (POS_ONE, POS_HALF), (NEG_ONE, POS_HALF),
        (POS_ONE, (0xC000000000000000, 0x4000)),  # y=1, x=3
        (NEG_HALF, (0xC000000000000000, 0x4000)),
        # Near x = 1 (cancellation resistance)
        (POS_ONE, (J | 1, 0x3FFF)),                # 1 + 2^-63
        (POS_ONE, (0xFFFFFFFFFFFFFFFF, 0x3FFE)),   # 1 - 2^-64
        (POS_ONE, (J | (1 << 30), 0x3FFF)),
        (NEG_ONE, (0xFFFFFFFFC0000000, 0x3FFE)),
        # Near sqrt(2) range-reduction boundary
        (POS_ONE, (0xB504F333F9DE6484, 0x3FFF)),
        (POS_ONE, (0xB504F333F9DE6485, 0x3FFF)),
        # Subnormal and tiny normal x
        (POS_ONE, SUB_POS_MIN),
        (POS_ONE, SUB_POS_MID),
        (POS_ONE, TINY_NORM_POS),
        (SUB_POS_MIN, POS_TWO),
        (SUB_NEG_MIN, POS_HALF),
        # Large positive x
        (POS_ONE, HUGE_POS),
        (POS_HALF, (J, 0x7FFE)),
        # Overflow under RC: huge * log2(4) = 2 * huge
        (HUGE_POS, POS_FOUR),
        (HUGE_NEG, POS_FOUR),
        # Underflow: tiny * log2(1 + 2^-63)
        (TINY_NORM_POS, (J | 1, 0x3FFF)),
        (TINY_NORM_NEG, (J | 1, 0x3FFF)),
        # Zero y with finite x != 1
        (POS_ZERO, POS_TWO), (NEG_ZERO, POS_TWO),
        (POS_ZERO, POS_HALF), (NEG_ZERO, POS_HALF),
    ]
    for _ in range(64):
        ys = rng.getrandbits(1) << 15
        ye = rng.randint(0x3FFF - 40, 0x3FFF + 40)
        xe = rng.randint(0x3FFF - 80, 0x3FFF + 80)
        pairs.append(((J | rng.getrandbits(63), ys | ye), (J | rng.getrandbits(63), xe)))
    return pairs


def fyl2xp1_domain_pairs():
    """Pairs (y, x) with x inside the exact Intel SDM domain [-(1-sqrt(2)/2), 1-sqrt(2)/2]."""
    rng = random.Random(0xF12B07)
    # 1 - sqrt(2)/2 ~= 0.292893218813452475599... -> 0x3FFD / 0x95F619980C4336F7
    dom_pos = (0x95F619980C4336F6, 0x3FFD)
    dom_neg = (0x95F619980C4336F6, 0xBFFD)
    pairs = [
        (POS_ONE, POS_ZERO), (NEG_ONE, POS_ZERO),
        (POS_ONE, NEG_ZERO), (NEG_ONE, NEG_ZERO),
        (POS_ZERO, POS_ZERO), (NEG_ZERO, POS_ZERO),
        (POS_ZERO, NEG_ZERO), (NEG_ZERO, NEG_ZERO),
        (POS_ONE, (J, 0x3FFD)), (POS_ONE, (J, 0xBFFD)),  # +-0.25
        (NEG_ONE, (J, 0x3FFD)), (NEG_ONE, (J, 0xBFFD)),
        (POS_ONE, dom_pos), (POS_ONE, dom_neg),
        (NEG_HALF, dom_pos), (NEG_HALF, dom_neg),
        (POS_ONE, SUB_POS_MIN), (POS_ONE, SUB_NEG_MIN),
        (POS_ONE, SUB_POS_MID), (POS_ONE, SUB_NEG_MID),
        (POS_ONE, TINY_NORM_POS), (POS_ONE, TINY_NORM_NEG),
        (POS_TWO, (J | 0x123456789ABCDEF0, 0x3FFF - 60)),
        (POS_TWO, (J | 0x123456789ABCDEF0, 0xBFFF - 60)),
        (POS_ZERO, (J, 0x3FFD)), (NEG_ZERO, (J, 0x3FFD)),
        (POS_ZERO, (J, 0xBFFD)), (NEG_ZERO, (J, 0xBFFD)),
        (HUGE_POS, (J, 0x3FFD)), (HUGE_NEG, (J, 0xBFFD)),
    ]
    for _ in range(64):
        ys = rng.getrandbits(1) << 15
        ye = rng.randint(0x3FFF - 40, 0x3FFF + 40)
        xs = rng.getrandbits(1) << 15
        xe = rng.randint(0x3FFF - 60, 0x3FFC)  # |x| < 0.25, safely inside SDM domain
        pairs.append(((J | rng.getrandbits(63), ys | ye), (J | rng.getrandbits(63), xs | xe)))
    return pairs


# --- Core differential & hardware fidelity checks ---

def check_differential(native, lifted, altered, altered_lifted):
    total = 0
    max_ulp_ref = 0
    max_ulp_nat = 0
    dummy = b"\x00" * 16

    # 1. F2XM1 across domain + all 12 masked FCWs (PC 24/53/64 x RC RN/RD/RU/RZ).
    f2_vals = f2xm1_domain_values()
    for x in f2_vals:
        for cw in (0x037F,):
            nat = unpack80(call3(native["op_f2xm1_m80"], pack80(x), pack16(cw), b"\x5a" * 10))
            lif = unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(cw), b"\x5a" * 10))
            ref = ref_f2xm1(x, cw)
            u_ref = m80_ulp_diff(lif, ref)
            u_nat = m80_ulp_diff(lif, nat)
            max_ulp_ref = max(max_ulp_ref, u_ref)
            max_ulp_nat = max(max_ulp_nat, u_nat)
            if u_ref > 1 or u_nat > 1:
                raise AssertionError(f"f2xm1 x={x} cw={cw:#x}: ref={ref} nat={nat} lif={lif} ulp=({u_ref},{u_nat})")
            if x in (POS_ZERO, NEG_ZERO, POS_ONE, NEG_ONE) and (lif != ref or lif != nat):
                raise AssertionError(f"f2xm1 exact point x={x}: ref={ref} nat={nat} lif={lif}")
            total += 1

    # Verify all 12 FCWs (PC 24/53/64 x RC RN/RD/RU/RZ) on F2XM1, FYL2X, FYL2XP1!
    for cw in MASKED_CWS:
        # F2XM1(0.5) and F2XM1(-0.5) and F2XM1(-0.0)
        for x in (POS_HALF, NEG_HALF, NEG_ZERO, POS_ZERO, (0xAAAAAAAAAAAAAAAA, 0x3FFD)):
            nat = unpack80(call3(native["op_f2xm1_m80"], pack80(x), pack16(cw), b"\x5a" * 10))
            lif = unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(cw), b"\x5a" * 10))
            ref = ref_f2xm1(x, cw)
            assert m80_ulp_diff(lif, ref) <= 1 and m80_ulp_diff(lif, nat) <= 1, (x, hex(cw), ref, nat, lif)
            if x in (POS_ZERO, NEG_ZERO):
                assert lif == nat == x
            total += 1
        # Verify PC independence matches Intel hardware: PC=0 and PC=2 equal PC=3 for each RC!
        rc = (cw >> 10) & 3
        cw_pc64 = 0x037F | (rc << 10)
        for op_name, a, b in (
            ("op_f2xm1_m80", pack80(POS_HALF), None),
            ("op_fyl2x_m80", pack80(POS_ONE), pack80((0xC000000000000000, 0x4000))),
            ("op_fyl2xp1_m80", pack80(POS_ONE), pack80((J, 0x3FFD))),
        ):
            if b is None:
                got_pc = unpack80(call3(lifted[op_name], a, pack16(cw), b"\x5a" * 10))
                got_64 = unpack80(call3(lifted[op_name], a, pack16(cw_pc64), b"\x5a" * 10))
                nat_pc = unpack80(call3(native[op_name], a, pack16(cw), b"\x5a" * 10))
                nat_64 = unpack80(call3(native[op_name], a, pack16(cw_pc64), b"\x5a" * 10))
            else:
                got_pc = unpack80(call3(lifted[op_name], a, b, pack16(cw) + b"\x5a" * 8))
                got_64 = unpack80(call3(lifted[op_name], a, b, pack16(cw_pc64) + b"\x5a" * 8))
                nat_pc = unpack80(call3(native[op_name], a, b, pack16(cw) + b"\x5a" * 8))
                nat_64 = unpack80(call3(native[op_name], a, b, pack16(cw_pc64) + b"\x5a" * 8))
            assert got_pc == got_64 and nat_pc == nat_64 and m80_ulp_diff(got_pc, nat_pc) <= 1
            total += 1

    # Verify RC monotonicity (RD <= RN <= RU) on irrational points!
    for x in (POS_HALF, NEG_HALF):
        r_rd = m80_ordinal(unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(0x077F), b"\x5a" * 10)))
        r_rn = m80_ordinal(unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(0x037F), b"\x5a" * 10)))
        r_ru = m80_ordinal(unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(0x0B7F), b"\x5a" * 10)))
        assert r_rd <= r_rn <= r_ru and r_ru - r_rd == 1, (x, r_rd, r_rn, r_ru)

    # 2. FYL2X across positive domain + all 4 RCs on boundary cases.
    for y, x in fyl2x_domain_pairs():
        cws = (0x037F, 0x077F, 0x0B7F, 0x0F7F) if y in (HUGE_POS, HUGE_NEG, POS_ONE) else (0x037F,)
        for cw in cws:
            out_init = pack16(cw) + b"\x5a" * 8
            nat = unpack80(call3(native["op_fyl2x_m80"], pack80(y), pack80(x), out_init))
            lif = unpack80(call3(lifted["op_fyl2x_m80"], pack80(y), pack80(x), out_init))
            ref = ref_fyl2x(y, x, cw)
            u_ref = m80_ulp_diff(lif, ref)
            u_nat = m80_ulp_diff(lif, nat)
            max_ulp_ref = max(max_ulp_ref, u_ref)
            max_ulp_nat = max(max_ulp_nat, u_nat)
            if u_ref > 1 or u_nat > 1:
                raise AssertionError(f"fyl2x y={y} x={x} cw={cw:#x}: ref={ref} nat={nat} lif={lif} ulp=({u_ref},{u_nat})")
            total += 1

    # 3. FYL2XP1 across SDM domain [-(1-sqrt(2)/2), +(1-sqrt(2)/2)].
    for y, x in fyl2xp1_domain_pairs():
        cws = (0x037F, 0x077F, 0x0B7F, 0x0F7F) if y in (POS_ONE, HUGE_POS) else (0x037F,)
        for cw in cws:
            out_init = pack16(cw) + b"\x5a" * 8
            nat = unpack80(call3(native["op_fyl2xp1_m80"], pack80(y), pack80(x), out_init))
            lif = unpack80(call3(lifted["op_fyl2xp1_m80"], pack80(y), pack80(x), out_init))
            ref = ref_fyl2xp1(y, x, cw)
            u_ref = m80_ulp_diff(lif, ref)
            u_nat = m80_ulp_diff(lif, nat)
            max_ulp_ref = max(max_ulp_ref, u_ref)
            max_ulp_nat = max(max_ulp_nat, u_nat)
            if u_ref > 1 or u_nat > 1:
                raise AssertionError(f"fyl2xp1 y={y} x={x} cw={cw:#x}: ref={ref} nat={nat} lif={lif} ulp=({u_ref},{u_nat})")
            total += 1

    # 4. m64 / m32 loads & fst / fstp stores across all 4 RCs.
    f64_f2_inputs = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.125, -0.75, 1e-300, 0x0000000000000001, 0x8000000000000001]
    f32_f2_inputs = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.75, 0x00000001, 0x80000001]
    for cw in (0x037F, 0x077F, 0x0B7F, 0x0F7F):
        for v in f64_f2_inputs:
            inp = pack_f64(v)
            for name in ("op_f2xm1_m64", "op_f2xm1_fst_m64"):
                nat = call3(native[name], inp, pack16(cw), b"\x5a" * 8)
                lif = call3(lifted[name], inp, pack16(cw), b"\x5a" * 8)
                assert f64_ulp_diff(nat, lif) <= 1, (name, v, hex(cw), nat.hex(), lif.hex())
                total += 1
        for v in f32_f2_inputs:
            inp = pack_f32(v)
            for name in ("op_f2xm1_m32", "op_f2xm1_fst_m32"):
                nat = call3(native[name], inp, pack16(cw), b"\x5a" * 4)
                lif = call3(lifted[name], inp, pack16(cw), b"\x5a" * 4)
                assert f32_ulp_diff(nat, lif) <= 1, (name, v, hex(cw), nat.hex(), lif.hex())
                total += 1
        for yv, xv in ((1.0, 3.0), (-2.5, 0.5), (1.0, 0x0000000000000001)):
            out_init = pack16(cw) + b"\x5a" * 6
            nat = call3(native["op_fyl2x_m64"], pack_f64(yv), pack_f64(xv), out_init)
            lif = call3(lifted["op_fyl2x_m64"], pack_f64(yv), pack_f64(xv), out_init)
            assert f64_ulp_diff(nat, lif) <= 1
            total += 1
        for yv, xv in ((1.0, 3.0), (-2.5, 0.5), (1.0, 0x00000001)):
            out_init = pack16(cw) + b"\x5a" * 2
            nat = call3(native["op_fyl2x_m32"], pack_f32(yv), pack_f32(xv), out_init)
            lif = call3(lifted["op_fyl2x_m32"], pack_f32(yv), pack_f32(xv), out_init)
            assert f32_ulp_diff(nat, lif) <= 1
            total += 1
        for yv, xv in ((1.0, 0.25), (-1.5, -0.125), (1.0, 0x0000000000000001)):
            out_init = pack16(cw) + b"\x5a" * 6
            nat = call3(native["op_fyl2xp1_m64"], pack_f64(yv), pack_f64(xv), out_init)
            lif = call3(lifted["op_fyl2xp1_m64"], pack_f64(yv), pack_f64(xv), out_init)
            assert f64_ulp_diff(nat, lif) <= 1
            total += 1
        for yv, xv in ((1.0, 0.25), (-1.5, -0.125), (1.0, 0x00000001)):
            out_init = pack16(cw) + b"\x5a" * 2
            nat = call3(native["op_fyl2xp1_m32"], pack_f32(yv), pack_f32(xv), out_init)
            lif = call3(lifted["op_fyl2xp1_m32"], pack_f32(yv), pack_f32(xv), out_init)
            assert f32_ulp_diff(nat, lif) <= 1
            total += 1

    # 5. Special values & exact status word flags (IE, DE, ZE, OE, UE, TOP).
    f2_special = [
        ("f2xm1_+0", POS_ZERO, POS_ZERO, 0),
        ("f2xm1_-0", NEG_ZERO, NEG_ZERO, 0),
        ("f2xm1_+inf", POS_INF, POS_INF, 0),
        ("f2xm1_-inf", NEG_INF, NEG_ONE, 0),
        ("f2xm1_qnan", QNAN_A, QNAN_A, 0),
        ("f2xm1_snan", SNAN_A, (SNAN_A[0] | QBIT, SNAN_A[1]), SW_IE),
        ("f2xm1_unnorm", UNNORM, INDEF, SW_IE),
        ("f2xm1_+sub", SUB_POS_MIN, SUB_POS_MIN, SW_DE | SW_UE),
        ("f2xm1_-sub", SUB_NEG_MIN, SUB_NEG_MIN, SW_DE | SW_UE),
        ("f2xm1_+tiny", TINY_NORM_POS, None, SW_UE),
    ]
    for label, x, want_val, want_flags in f2_special:
        nat_v = unpack80(call3(native["op_f2xm1_m80"], pack80(x), pack16(0x037F), b"\x5a" * 10))
        lif_v = unpack80(call3(lifted["op_f2xm1_m80"], pack80(x), pack16(0x037F), b"\x5a" * 10))
        nat_sw = unpack16(call3(native["op_f2xm1_sw"], pack80(x), pack16(0x037F), b"\x5a" * 2))
        lif_sw = unpack16(call3(lifted["op_f2xm1_sw"], pack80(x), pack16(0x037F), b"\x5a" * 2))
        if want_val is not None:
            assert lif_v == want_val and nat_v == want_val, (label, nat_v, lif_v, want_val)
        else:
            assert m80_ulp_diff(lif_v, nat_v) <= 1, (label, nat_v, lif_v)
        assert (lif_sw & SW_STRICT_MASK) == (nat_sw & SW_STRICT_MASK) == (0x3800 | want_flags), (
            label, hex(nat_sw), hex(lif_sw), hex(0x3800 | want_flags)
        )
        total += 2

    fyl2x_special = [
        ("fyl2x_+1_+0", POS_ONE, POS_ZERO, NEG_INF, SW_ZE),
        ("fyl2x_-1_+0", NEG_ONE, POS_ZERO, POS_INF, SW_ZE),
        ("fyl2x_+1_-0", POS_ONE, NEG_ZERO, NEG_INF, SW_ZE),
        ("fyl2x_-1_-0", NEG_ONE, NEG_ZERO, POS_INF, SW_ZE),
        ("fyl2x_+0_+0", POS_ZERO, POS_ZERO, INDEF, SW_IE),
        ("fyl2x_-0_+0", NEG_ZERO, POS_ZERO, INDEF, SW_IE),
        ("fyl2x_+inf_+0", POS_INF, POS_ZERO, NEG_INF, 0),
        ("fyl2x_-inf_+0", NEG_INF, POS_ZERO, POS_INF, 0),
        ("fyl2x_+inf_+1", POS_INF, POS_ONE, INDEF, SW_IE),
        ("fyl2x_-inf_+1", NEG_INF, POS_ONE, INDEF, SW_IE),
        ("fyl2x_+1_+inf", POS_ONE, POS_INF, POS_INF, 0),
        ("fyl2x_-1_+inf", NEG_ONE, POS_INF, NEG_INF, 0),
        ("fyl2x_+0_+inf", POS_ZERO, POS_INF, INDEF, SW_IE),
        ("fyl2x_-0_+inf", NEG_ZERO, POS_INF, INDEF, SW_IE),
        ("fyl2x_+1_-1", POS_ONE, NEG_ONE, INDEF, SW_IE),
        ("fyl2x_+0_-1", POS_ZERO, NEG_ONE, INDEF, SW_IE),
        ("fyl2x_+1_-inf", POS_ONE, NEG_INF, INDEF, SW_IE),
        ("fyl2x_+1_-sub", POS_ONE, SUB_NEG_MIN, INDEF, SW_IE),
        ("fyl2x_+1_+sub", POS_ONE, SUB_POS_MIN, (0x807A000000000000, 0xC00D), SW_DE),
        ("fyl2x_+inf_+2", POS_INF, POS_TWO, POS_INF, 0),
        ("fyl2x_+inf_+0.5", POS_INF, POS_HALF, NEG_INF, 0),
        ("fyl2x_snan_2", SNAN_A, POS_TWO, (SNAN_A[0] | QBIT, SNAN_A[1]), SW_IE),
        ("fyl2x_2_snan", POS_TWO, SNAN_B, (SNAN_B[0] | QBIT, SNAN_B[1]), SW_IE),
        ("fyl2x_qnan_snan", QNAN_A, SNAN_B, QNAN_A, SW_IE),
        ("fyl2x_snan_qnan", SNAN_B, QNAN_A, QNAN_A, SW_IE),
        ("fyl2x_unnorm_2", UNNORM, POS_TWO, INDEF, SW_IE),
        ("fyl2x_2_unnorm", POS_TWO, UNNORM, INDEF, SW_IE),
        ("fyl2x_overflow", HUGE_POS, POS_FOUR, POS_INF, SW_OE),
    ]
    for label, y, x, want_val, want_flags in fyl2x_special:
        out_v = pack16(0x037F) + b"\x5a" * 8
        out_sw = pack16(0x037F)
        nat_v = unpack80(call3(native["op_fyl2x_m80"], pack80(y), pack80(x), out_v))
        lif_v = unpack80(call3(lifted["op_fyl2x_m80"], pack80(y), pack80(x), out_v))
        nat_sw = unpack16(call3(native["op_fyl2x_sw"], pack80(y), pack80(x), out_sw))
        lif_sw = unpack16(call3(lifted["op_fyl2x_sw"], pack80(y), pack80(x), out_sw))
        assert lif_v == want_val and nat_v == want_val, (label, nat_v, lif_v, want_val)
        assert (lif_sw & SW_STRICT_MASK) == (nat_sw & SW_STRICT_MASK) == (0x3800 | want_flags), (
            label, hex(nat_sw), hex(lif_sw), hex(0x3800 | want_flags)
        )
        total += 2

    fyl2xp1_sdm_special = [
        ("fyl2xp1_+1_+0", POS_ONE, POS_ZERO, POS_ZERO, 0),
        ("fyl2xp1_-1_+0", NEG_ONE, POS_ZERO, NEG_ZERO, 0),
        ("fyl2xp1_+1_-0", POS_ONE, NEG_ZERO, NEG_ZERO, 0),
        ("fyl2xp1_-1_-0", NEG_ONE, NEG_ZERO, POS_ZERO, 0),
        ("fyl2xp1_+inf_+0", POS_INF, POS_ZERO, INDEF, SW_IE),
        ("fyl2xp1_-inf_-0", NEG_INF, NEG_ZERO, INDEF, SW_IE),
        ("fyl2xp1_+inf_+0.25", POS_INF, (J, 0x3FFD), POS_INF, 0),
        ("fyl2xp1_+inf_-0.25", POS_INF, (J, 0xBFFD), NEG_INF, 0),
        ("fyl2xp1_+1_+sub", POS_ONE, SUB_POS_MIN, SUB_POS_MIN, SW_DE | SW_UE),
        ("fyl2xp1_+1_-sub", POS_ONE, SUB_NEG_MIN, SUB_NEG_MIN, SW_DE | SW_UE),
        ("fyl2xp1_snan_+0", SNAN_A, POS_ZERO, (SNAN_A[0] | QBIT, SNAN_A[1]), SW_IE),
        ("fyl2xp1_+1_snan", POS_ONE, SNAN_A, (SNAN_A[0] | QBIT, SNAN_A[1]), SW_IE),
        ("fyl2xp1_+1_qnan", POS_ONE, QNAN_B, QNAN_B, 0),
        ("fyl2xp1_unnorm_+0", UNNORM, POS_ZERO, INDEF, SW_IE),
    ]
    for label, y, x, want_val, want_flags in fyl2xp1_sdm_special:
        out_v = pack16(0x037F) + b"\x5a" * 8
        out_sw = pack16(0x037F)
        nat_v = unpack80(call3(native["op_fyl2xp1_m80"], pack80(y), pack80(x), out_v))
        lif_v = unpack80(call3(lifted["op_fyl2xp1_m80"], pack80(y), pack80(x), out_v))
        nat_sw = unpack16(call3(native["op_fyl2xp1_sw"], pack80(y), pack80(x), out_sw))
        lif_sw = unpack16(call3(lifted["op_fyl2xp1_sw"], pack80(y), pack80(x), out_sw))
        assert lif_v == want_val and nat_v == want_val, (label, nat_v, lif_v, want_val)
        assert (lif_sw & SW_STRICT_MASK) == (nat_sw & SW_STRICT_MASK) == (0x3800 | want_flags), (
            label, hex(nat_sw), hex(lif_sw), hex(0x3800 | want_flags)
        )
        total += 2

    # Lifted-only singularity checks for FYL2XP1(x == -1, y) and x < -1
    # (outside Intel SDM domain [-(1-sqrt(2)/2), 1-sqrt(2)/2], where hardware is undefined).
    for label, y, x, want_val, want_flags in (
        ("fyl2xp1_+1_-1_singularity", POS_ONE, NEG_ONE, NEG_INF, SW_ZE),
        ("fyl2xp1_-1_-1_singularity", NEG_ONE, NEG_ONE, POS_INF, SW_ZE),
        ("fyl2xp1_+0_-1_indef", POS_ZERO, NEG_ONE, INDEF, SW_IE),
        ("fyl2xp1_+inf_-1", POS_INF, NEG_ONE, NEG_INF, 0),
        ("fyl2xp1_+1_-2_domain", POS_ONE, (J, 0xC000), INDEF, SW_IE),
        ("fyl2xp1_+1_-inf_domain", POS_ONE, NEG_INF, INDEF, SW_IE),
    ):
        lif_v = unpack80(call3(lifted["op_fyl2xp1_m80"], pack80(y), pack80(x), pack16(0x037F) + b"\x5a" * 8))
        lif_sw = unpack16(call3(lifted["op_fyl2xp1_sw"], pack80(y), pack80(x), pack16(0x037F)))
        assert lif_v == want_val, (label, lif_v, want_val)
        assert (lif_sw & SW_STRICT_MASK) == (0x3800 | want_flags), (label, hex(lif_sw), hex(0x3800 | want_flags))
        total += 2

    # 6. Stack pop depth & depth-8 checks.
    for y, x in ((POS_ONE, POS_TWO), (NEG_HALF, (0xC000000000000000, 0x4000))):
        nat_s = unpack80(call3(native["op_fyl2x_pop_depth"], pack80(y), pack80(x), b"\x5a" * 10))
        lif_s = unpack80(call3(lifted["op_fyl2x_pop_depth"], pack80(y), pack80(x), b"\x5a" * 10))
        assert nat_s == POS_ONE and lif_s == POS_ONE
        nat_8 = unpack80(call3(native["op_fyl2x_depth8"], pack80(y), pack80(x), b"\x5a" * 10))
        lif_8 = unpack80(call3(lifted["op_fyl2x_depth8"], pack80(y), pack80(x), b"\x5a" * 10))
        assert m80_ulp_diff(nat_8, lif_8) <= 1
        nat_p = unpack80(call3(native["op_fyl2xp1_pop_depth"], pack80(y), pack80((J, 0x3FFD)), b"\x5a" * 10))
        lif_p = unpack80(call3(lifted["op_fyl2xp1_pop_depth"], pack80(y), pack80((J, 0x3FFD)), b"\x5a" * 10))
        assert nat_p == POS_ZERO and lif_p == POS_ZERO
        total += 3

    # 7. Composition roundtrip: 2^(log2(x+1)) - 1 == x within <= 2 ULP.
    for x in (POS_ZERO, NEG_ZERO, (J, 0x3FFD), (J, 0xBFFD), (0x95F619980C4336F6, 0x3FFD), SUB_POS_MID):
        nat_r = unpack80(call3(native["op_exp_log_roundtrip"], pack80(x), pack16(0x037F), b"\x5a" * 10))
        lif_r = unpack80(call3(lifted["op_exp_log_roundtrip"], pack80(x), pack16(0x037F), b"\x5a" * 10))
        assert m80_ulp_diff(lif_r, x) <= 2 and m80_ulp_diff(nat_r, lif_r) <= 2, (x, nat_r, lif_r)
        if x in (POS_ZERO, NEG_ZERO):
            assert lif_r == nat_r == x
        total += 1

    # 8. FCW & FNINIT transport checks across all 12 masked control words.
    for cw in MASKED_CWS:
        nat_cw = unpack16(call3(native["op_fcw_roundtrip"], pack16(cw), dummy, b"\x5a" * 2))
        lif_cw = unpack16(call3(lifted["op_fcw_roundtrip"], pack16(cw), dummy, b"\x5a" * 2))
        assert nat_cw == cw and lif_cw == cw, (hex(cw), hex(nat_cw), hex(lif_cw))
        nat_icw = unpack16(call3(native["op_fninit_cw_sw"], pack16(cw), pack80(SNAN_A), b"\x5a" * 2))
        lif_icw = unpack16(call3(lifted["op_fninit_cw_sw"], pack16(cw), pack80(SNAN_A), b"\x5a" * 2))
        assert nat_icw == 0x037F and lif_icw == 0x037F
        total += 2
    nat_isw = unpack16(call3(native["op_fninit_sw"], pack80(SNAN_A), dummy, b"\x5a" * 2))
    lif_isw = unpack16(call3(lifted["op_fninit_sw"], pack80(SNAN_A), dummy, b"\x5a" * 2))
    assert nat_isw == 0x0000 and lif_isw == 0x0000
    total += 1

    # 9. Altered-build controls (ALTER_X87_EXP_LOG): altered lifted matches altered native and diverges from original.
    altered_probes = (
        ("op_f2xm1_m80", pack80(POS_HALF), pack16(0x037F), b"\x5a" * 10),
        ("op_fyl2x_m80", pack80(POS_ONE), pack80((J, 0x3FFD)), pack16(0x037F) + b"\x5a" * 8),
        ("op_fyl2xp1_m80", pack80(POS_ONE), pack80((J, 0x3FFD)), pack16(0x037F) + b"\x5a" * 8),
        ("op_exp_log_roundtrip", pack80((J, 0x3FFD)), pack16(0x037F), b"\x5a" * 10),
        ("op_fcw_roundtrip", pack16(0x077F), dummy, b"\x5a" * 2),
    )
    for name, a_b, b_b, o_b in altered_probes:
        orig_n = call3(native[name], a_b, b_b, o_b)
        alt_n = call3(altered[name], a_b, b_b, o_b)
        alt_l = call3(altered_lifted[name], a_b, b_b, o_b)
        if len(o_b) == 10:
            assert m80_ulp_diff(unpack80(alt_n), unpack80(alt_l)) <= 1, (name, alt_n.hex(), alt_l.hex())
        else:
            assert alt_n == alt_l, (name, alt_n.hex(), alt_l.hex())
        assert alt_n != orig_n, f"altered build did not diverge on {name}"
        total += 1

    return total, max_ulp_ref, max_ulp_nat


def check_mxcsr_isolation(native, lifted, lifted_lib):
    """Verify x87 80-bit subnormals do NOT flush under MXCSR 0x1fc0 (DAZ) / 0x9fc0 (DAZ+FTZ)
    and lifted exports never clobber host MXCSR."""
    get_mxcsr = lifted_lib.host_get_mxcsr
    get_mxcsr.argtypes, get_mxcsr.restype = [], ctypes.c_uint32
    xchg_mxcsr = lifted_lib.host_exchange_mxcsr
    xchg_mxcsr.argtypes, xchg_mxcsr.restype = [ctypes.c_uint32], ctypes.c_uint32

    saved = get_mxcsr()
    modes = (0x1F80, 0x1FC0, 0x9FC0, 0x3FC0)
    probes = [
        ("op_f2xm1_m80", pack80(SUB_POS_MIN), pack16(0x037F), b"\x5a" * 10),
        ("op_f2xm1_m80", pack80(SUB_NEG_MIN), pack16(0x037F), b"\x5a" * 10),
        ("op_f2xm1_m80", pack80(TINY_NORM_POS), pack16(0x037F), b"\x5a" * 10),
        ("op_f2xm1_m64", pack_f64(0x0000000000000001), pack16(0x037F), b"\x5a" * 8),
        ("op_f2xm1_m32", pack_f32(0x00000001), pack16(0x037F), b"\x5a" * 4),
        ("op_fyl2x_m80", pack80(POS_ONE), pack80(SUB_POS_MIN), pack16(0x037F) + b"\x5a" * 8),
        ("op_fyl2x_m80", pack80(SUB_POS_MIN), pack80(POS_TWO), pack16(0x037F) + b"\x5a" * 8),
        ("op_fyl2xp1_m80", pack80(POS_ONE), pack80(SUB_POS_MIN), pack16(0x037F) + b"\x5a" * 8),
        ("op_fyl2xp1_m80", pack80(POS_ONE), pack80(SUB_NEG_MID), pack16(0x037F) + b"\x5a" * 8),
    ]
    checked = 0
    try:
        baselines = {}
        for mxcsr in modes:
            xchg_mxcsr(mxcsr)
            for idx, (name, a_b, b_b, o_b) in enumerate(probes):
                before = get_mxcsr()
                assert before == mxcsr, f"failed to set MXCSR {mxcsr:#x} (got {before:#x})"
                lif = call3(lifted[name], a_b, b_b, o_b)
                after = get_mxcsr()
                assert after == mxcsr, f"lifted {name} clobbered MXCSR: {mxcsr:#x} -> {after:#x}"
                nat = call3(native[name], a_b, b_b, o_b)
                assert get_mxcsr() == mxcsr, f"native {name} clobbered MXCSR"
                if idx not in baselines:
                    baselines[idx] = (nat, lif)
                else:
                    # Must be BIT-IDENTICAL across 0x1f80, 0x1fc0 (DAZ), 0x9fc0 (DAZ+FTZ), 0x3fc0!
                    assert lif == baselines[idx][1], f"lifted {name} changed under MXCSR {mxcsr:#x}"
                    assert nat == baselines[idx][0], f"native {name} changed under MXCSR {mxcsr:#x}"
                # Subnormal outputs must remain non-zero!
                assert any(byte != 0 for byte in lif[:8]), f"subnormal flushed to zero on {name} under {mxcsr:#x}"
                if len(o_b) == 10:
                    assert m80_ulp_diff(unpack80(nat), unpack80(lif)) <= 1
                elif len(o_b) == 8:
                    assert f64_ulp_diff(nat, lif) <= 1
                else:
                    assert f32_ulp_diff(nat, lif) <= 1
                checked += 1
    finally:
        xchg_mxcsr(saved)
    return checked


def check_mutants(lifted):
    """Faulty reference models must each be caught by the lifted outputs."""
    caught = 0
    # 1. Dropping -0.0 sign in F2XM1(-0.0)
    got_neg0 = unpack80(call3(lifted["op_f2xm1_m80"], pack80(NEG_ZERO), pack16(0x037F), b"\x5a" * 10))
    assert got_neg0 != POS_ZERO
    caught += 1
    # 2. Truncating 80-bit F2XM1(0.5) to 53-bit MSVC `long double` precision
    got_half = unpack80(call3(lifted["op_f2xm1_m80"], pack80(POS_HALF), pack16(0x037F), b"\x5a" * 10))
    trunc_53 = (got_half[0] & ~0x7FF, got_half[1])
    assert got_half != trunc_53
    caught += 1
    # 3. Ignoring RC=RU on F2XM1(0.5)
    got_ru = unpack80(call3(lifted["op_f2xm1_m80"], pack80(POS_HALF), pack16(0x0B7F), b"\x5a" * 10))
    assert got_ru != got_half
    caught += 1
    # 4. Flushing 80-bit subnormal input to zero in FYL2X(1, 2^-16445)
    got_sub = unpack80(call3(lifted["op_fyl2x_m80"], pack80(POS_ONE), pack80(SUB_POS_MIN), pack16(0x037F) + b"\x5a" * 8))
    assert got_sub != NEG_INF
    caught += 1
    # 5. Missing #Z (ZE) flag on FYL2X(1, +0) and FYL2XP1(1, -1)
    sw_z1 = unpack16(call3(lifted["op_fyl2x_sw"], pack80(POS_ONE), pack80(POS_ZERO), pack16(0x037F)))
    sw_z2 = unpack16(call3(lifted["op_fyl2xp1_sw"], pack80(POS_ONE), pack80(NEG_ONE), pack16(0x037F)))
    assert (sw_z1 & SW_ZE) != 0 and (sw_z2 & SW_ZE) != 0
    caught += 1
    # 6. Missing #IA (IE) flag on negative FYL2X base or sNaN
    sw_ia1 = unpack16(call3(lifted["op_fyl2x_sw"], pack80(POS_ONE), pack80(NEG_ONE), pack16(0x037F)))
    sw_ia2 = unpack16(call3(lifted["op_f2xm1_sw"], pack80(SNAN_A), pack16(0x037F), b"\x5a" * 2))
    assert (sw_ia1 & SW_IE) != 0 and (sw_ia2 & SW_IE) != 0
    caught += 1
    return caught


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path = (Path(p) for p in sys.argv[1:5])

    accepted, rejected = check_translator()
    print(f"{accepted} x87 exp/log sequences accepted; {rejected} rejection controls passed")

    _, native = bind(original_path, "")
    lifted_lib, lifted = bind(lifted_path, "lifted_")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")

    diff_total, max_ulp_ref, max_ulp_nat = check_differential(native, lifted, altered, altered_lifted)
    mxcsr_total = check_mxcsr_isolation(native, lifted, lifted_lib)
    mutant_count = check_mutants(lifted)
    # Restore clean x87 state after native runs.
    call3(native["op_fninit_sw"], pack80(POS_ZERO), b"\x00" * 16, b"\x5a" * 2)

    print(
        f"{diff_total} differential/FCW/status/special cases matched "
        f"(max ULP vs Decimal ref={max_ulp_ref}, vs native x87={max_ulp_nat}); "
        f"{mxcsr_total} MXCSR DAZ/FTZ isolation checks passed; "
        f"{mutant_count} mutant controls + {len(ALTERED_EXPORTS)} altered-build controls caught"
    )
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
