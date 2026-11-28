"""Differential, MXCSR, /O2, mutant, and rejection checks for AVX VROUNDPS, VROUNDPD, VROUNDSS, and VROUNDSD."""

import ctypes
from fractions import Fraction
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_avx_round_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(32))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(32))
FILL = b"\x5A" * 32

EXPORTS = (
    "op_vroundps_xmm_rn", "op_vroundps_xmm_rd", "op_vroundps_xmm_ru", "op_vroundps_xmm_rz",
    "op_vroundps_xmm_mxcsr", "op_vroundps_xmm_suppress", "op_vroundps_xmm_mem",
    "op_vroundps_ymm_rn", "op_vroundps_ymm_rd", "op_vroundps_ymm_ru", "op_vroundps_ymm_rz",
    "op_vroundps_ymm_mxcsr", "op_vroundps_ymm_mem",
    "op_vroundpd_xmm_rn", "op_vroundpd_xmm_rd", "op_vroundpd_xmm_ru", "op_vroundpd_xmm_rz",
    "op_vroundpd_xmm_mxcsr", "op_vroundpd_xmm_mem",
    "op_vroundpd_ymm_rn", "op_vroundpd_ymm_rd", "op_vroundpd_ymm_ru", "op_vroundpd_ymm_rz",
    "op_vroundpd_ymm_mxcsr", "op_vroundpd_ymm_mem",
    "op_vroundss_rn", "op_vroundss_rd", "op_vroundss_ru", "op_vroundss_rz",
    "op_vroundss_mxcsr", "op_vroundss_mem",
    "op_vroundsd_rn", "op_vroundsd_rd", "op_vroundsd_ru", "op_vroundsd_rz",
    "op_vroundsd_mxcsr", "op_vroundsd_mem",
    "op_vroundps_xmm_zero_upper", "op_vroundpd_xmm_zero_upper",
    "op_vroundss_zero_upper", "op_vroundsd_zero_upper",
    "op_vround_r8_reload", "op_vround_discarded_read",
)

MXCSR_EXPORTS = (
    "op_vroundps_xmm_mxcsr",
    "op_vroundps_ymm_mxcsr",
    "op_vroundpd_xmm_mxcsr",
    "op_vroundpd_ymm_mxcsr",
    "op_vroundss_mxcsr",
    "op_vroundsd_mxcsr",
)

DIRECTED_EXPORTS = (
    "op_vroundps_xmm_rd", "op_vroundps_xmm_ru", "op_vroundps_xmm_rz",
    "op_vroundps_ymm_rd", "op_vroundps_ymm_ru", "op_vroundps_ymm_rz",
    "op_vroundpd_xmm_rd", "op_vroundpd_xmm_ru", "op_vroundpd_xmm_rz",
    "op_vroundpd_ymm_rd", "op_vroundpd_ymm_ru", "op_vroundpd_ymm_rz",
    "op_vroundss_rd", "op_vroundss_ru", "op_vroundss_rz",
    "op_vroundsd_rd", "op_vroundsd_ru", "op_vroundsd_rz",
)

ALTERED_IMM = {
    "op_vroundps_xmm_rn": 3, "op_vroundps_xmm_rd": 2, "op_vroundps_xmm_ru": 1, "op_vroundps_xmm_rz": 0,
    "op_vroundps_xmm_mxcsr": 3, "op_vroundps_xmm_suppress": 11, "op_vroundps_xmm_mem": 1,
    "op_vroundps_ymm_rn": 3, "op_vroundps_ymm_rd": 2, "op_vroundps_ymm_ru": 1, "op_vroundps_ymm_rz": 0,
    "op_vroundps_ymm_mxcsr": 1, "op_vroundps_ymm_mem": 2,
    "op_vroundpd_xmm_rn": 3, "op_vroundpd_xmm_rd": 2, "op_vroundpd_xmm_ru": 1, "op_vroundpd_xmm_rz": 0,
    "op_vroundpd_xmm_mxcsr": 2, "op_vroundpd_xmm_mem": 1,
    "op_vroundpd_ymm_rn": 3, "op_vroundpd_ymm_rd": 2, "op_vroundpd_ymm_ru": 1, "op_vroundpd_ymm_rz": 0,
    "op_vroundpd_ymm_mxcsr": 3, "op_vroundpd_ymm_mem": 0,
    "op_vroundss_rn": 3, "op_vroundss_rd": 2, "op_vroundss_ru": 1, "op_vroundss_rz": 0,
    "op_vroundss_mxcsr": 3, "op_vroundss_mem": 1,
    "op_vroundsd_rn": 3, "op_vroundsd_rd": 2, "op_vroundsd_ru": 1, "op_vroundsd_rz": 0,
    "op_vroundsd_mxcsr": 2, "op_vroundsd_mem": 2,
}

# --- Independent exact-rational IEEE-754 rounding reference (fractions.Fraction, zero host FP) ---

def _decode_exact(bits, exp_bits, frac_bits):
    """Return ('nan'|'inf'|'finite', sign, Fraction|None)."""
    bias = (1 << (exp_bits - 1)) - 1
    exp_mask = (1 << exp_bits) - 1
    frac_mask = (1 << frac_bits) - 1
    sign = (bits >> (exp_bits + frac_bits)) & 1
    exp = (bits >> frac_bits) & exp_mask
    frac = bits & frac_mask
    if exp == exp_mask:
        return ("nan" if frac else "inf"), sign, None
    if exp == 0:
        return "finite", sign, Fraction(frac, 1 << (bias - 1 + frac_bits))
    return "finite", sign, Fraction((1 << frac_bits) | frac) * Fraction(2) ** (exp - bias - frac_bits)


def _encode_int_exact(n, sign, exp_bits, frac_bits):
    """Encode a nonnegative exact integer n with sign bit into IEEE-754 binary32/64 bits."""
    total_bits = 1 + exp_bits + frac_bits
    if n == 0:
        return sign << (total_bits - 1)
    bias = (1 << (exp_bits - 1)) - 1
    msb = n.bit_length() - 1
    biased = msb + bias
    if msb <= frac_bits:
        frac = (n << (frac_bits - msb)) & ((1 << frac_bits) - 1)
    else:
        frac = (n >> (msb - frac_bits)) & ((1 << frac_bits) - 1)
    return (sign << (total_bits - 1)) | (biased << frac_bits) | frac


def ref_round_bits(bits, imm8, *, exp_bits, frac_bits, mxcsr_rc=0, daz=False, quirk=None):
    """Exact rational IEEE-754 rounding to integer value, returning exact bit pattern."""
    total_bits = 1 + exp_bits + frac_bits
    quiet_bit = 1 << (frac_bits - 1)
    kind, sign, mag = _decode_exact(bits, exp_bits, frac_bits)
    if kind == "nan":
        return bits if quirk == "no_quieting" else (bits | quiet_bit)
    if kind == "inf":
        return bits
    exp_field = (bits >> frac_bits) & ((1 << exp_bits) - 1)
    if mag == 0 or (daz and exp_field == 0 and quirk != "ignore_daz"):
        return (0 if quirk == "drop_neg_zero" else (sign << (total_bits - 1)))

    if (imm8 & 0x04) != 0 and quirk != "ignore_mxcsr":
        mode = mxcsr_rc & 3
    else:
        mode = imm8 & 3

    whole = mag.numerator // mag.denominator
    rem = mag - whole
    if rem == 0:
        return bits
    if mode == 0:  # nearest-even
        if quirk == "ties_away":
            inc = rem >= Fraction(1, 2)
        else:
            inc = rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (whole & 1) != 0)
    elif mode == 1:  # floor (-inf)
        inc = (sign == 1)
    elif mode == 2:  # ceil (+inf)
        inc = (sign == 0)
    else:  # trunc (0)
        inc = False

    res_int = whole + (1 if inc else 0)
    out_sign = 0 if (res_int == 0 and quirk == "drop_neg_zero") else sign
    return _encode_int_exact(res_int, out_sign, exp_bits, frac_bits)


def ref_f32(bits, imm8, mxcsr_rc=0, daz=False, quirk=None):
    return ref_round_bits(bits, imm8, exp_bits=8, frac_bits=23, mxcsr_rc=mxcsr_rc, daz=daz, quirk=quirk)


def ref_f64(bits, imm8, mxcsr_rc=0, daz=False, quirk=None):
    return ref_round_bits(bits, imm8, exp_bits=11, frac_bits=52, mxcsr_rc=mxcsr_rc, daz=daz, quirk=quirk)


def pk_d(*vals):
    return struct.pack("<" + "I" * len(vals), *(v & 0xFFFFFFFF for v in vals))


def pk_q(*vals):
    return struct.pack("<" + "Q" * len(vals), *(v & 0xFFFFFFFFFFFFFFFF for v in vals))


def u32s(buf):
    return struct.unpack("<" + "I" * (len(buf) // 4), buf)


def u64s(buf):
    return struct.unpack("<" + "Q" * (len(buf) // 8), buf)


def reference(name, a, b, *, mxcsr_rc=0, daz=False, altered=False, quirk=None):
    """Expected 32-byte output buffer; unwritten bytes keep FILL."""
    out = bytearray(FILL)

    def imm_of(default):
        return ALTERED_IMM[name] if (altered and name in ALTERED_IMM) else default

    if name.startswith("op_vroundps_xmm_") and name != "op_vroundps_xmm_zero_upper":
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 4, "suppress": 8, "mem": 0x1A}[name.rsplit("_", 1)[1]]
        src = b[:16] if name.endswith("_mem") else a[:16]
        imm = imm_of(default_imm)
        out[0:16] = pk_d(*(ref_f32(w, imm, mxcsr_rc, daz, quirk) for w in u32s(src)))
    elif name.startswith("op_vroundps_ymm_"):
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 4, "mem": 9}[name.rsplit("_", 1)[1]]
        src = b if name.endswith("_mem") else a
        imm = imm_of(default_imm)
        out[0:32] = pk_d(*(ref_f32(w, imm, mxcsr_rc, daz, quirk) for w in u32s(src)))
    elif name.startswith("op_vroundpd_xmm_") and name != "op_vroundpd_xmm_zero_upper":
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 4, "mem": 10}[name.rsplit("_", 1)[1]]
        src = b[:16] if name.endswith("_mem") else a[:16]
        imm = imm_of(default_imm)
        out[0:16] = pk_q(*(ref_f64(q, imm, mxcsr_rc, daz, quirk) for q in u64s(src)))
    elif name.startswith("op_vroundpd_ymm_"):
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 12, "mem": 11}[name.rsplit("_", 1)[1]]
        src = b if name.endswith("_mem") else a
        imm = imm_of(default_imm)
        out[0:32] = pk_q(*(ref_f64(q, imm, mxcsr_rc, daz, quirk) for q in u64s(src)))
    elif name.startswith("op_vroundss_") and name != "op_vroundss_zero_upper":
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 4, "mem": 10}[name.rsplit("_", 1)[1]]
        imm = imm_of(default_imm)
        ad = list(u32s(a[:16]))
        bd0 = u32s(b[:4])[0]
        ad[0] = ref_f32(bd0, imm, mxcsr_rc, daz, quirk)
        if quirk == "clobber_scalar_upper":
            ad[1] = ad[2] = ad[3] = 0
        out[0:16] = pk_d(*ad)
    elif name.startswith("op_vroundsd_") and name != "op_vroundsd_zero_upper":
        default_imm = {"rn": 0, "rd": 1, "ru": 2, "rz": 3, "mxcsr": 12, "mem": 9}[name.rsplit("_", 1)[1]]
        imm = imm_of(default_imm)
        aq0, aq1 = u64s(a[:16])
        bq0 = u64s(b[:8])[0]
        hi = 0 if quirk == "clobber_scalar_upper" else aq1
        out[0:16] = pk_q(ref_f64(bq0, imm, mxcsr_rc, daz, quirk), hi)
    elif name == "op_vroundps_xmm_zero_upper":
        out[0:16] = pk_d(*(ref_f32(w, 0, mxcsr_rc, daz, quirk) for w in u32s(a[:16])))
        out[16:32] = a[16:32] if quirk == "keep_ymm_upper" else bytes(16)
    elif name == "op_vroundpd_xmm_zero_upper":
        out[0:16] = pk_q(*(ref_f64(q, 1, mxcsr_rc, daz, quirk) for q in u64s(a[:16])))
        out[16:32] = a[16:32] if quirk == "keep_ymm_upper" else bytes(16)
    elif name == "op_vroundss_zero_upper":
        ad = list(u32s(a[:16]))
        ad[0] = ref_f32(u32s(b[:4])[0], 2, mxcsr_rc, daz, quirk)
        out[0:16] = pk_d(*ad)
        out[16:32] = a[16:32] if quirk == "keep_ymm_upper" else bytes(16)
    elif name == "op_vroundsd_zero_upper":
        _, aq1 = u64s(a[:16])
        out[0:16] = pk_q(ref_f64(u64s(b[:8])[0], 3, mxcsr_rc, daz, quirk), aq1)
        out[16:32] = a[16:32] if quirk == "keep_ymm_upper" else bytes(16)
    elif name == "op_vround_r8_reload":
        out[0:32] = pk_q(*(ref_f64(q, 0, mxcsr_rc, daz, quirk) for q in u64s(a)))
    elif name == "op_vround_discarded_read":
        out[0:16] = a[0:16]
    else:
        raise KeyError(name)
    return bytes(out)


# --- Edge-value corpora for f32 and f64 ---

F32_EDGE = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x3E800000,  # +0.25
    0xBE800000,  # -0.25
    0x3F000000,  # +0.5 (tie -> +0.0)
    0xBF000000,  # -0.5 (tie -> -0.0)
    0x3F000001,  # +0.5 + ulp
    0xBF000001,  # -0.5 - ulp
    0x3F400000,  # +0.75
    0xBF400000,  # -0.75
    0x3F800000,  # +1.0
    0xBF800000,  # -1.0
    0x3FC00000,  # +1.5 (tie -> +2.0)
    0xBFC00000,  # -1.5 (tie -> -2.0)
    0x40000000,  # +2.0
    0xC0000000,  # -2.0
    0x40200000,  # +2.5 (tie -> +2.0)
    0xC0200000,  # -2.5 (tie -> -2.0)
    0x40600000,  # +3.5 (tie -> +4.0)
    0xC0600000,  # -3.5 (tie -> -4.0)
    0x4AFFFFFF,  # 2^23 - 0.5 (tie -> 2^23)
    0xCAFFFFFF,  # -(2^23 - 0.5)
    0x4AFFFFFE,  # 2^23 - 1.0 (exact integer)
    0x4B000000,  # 2^23 (boundary >= 2^23)
    0xCB000000,  # -2^23
    0x4B000001,  # 2^23 + 1
    0x7F7FFFFF,  # +max finite
    0xFF7FFFFF,  # -max finite
    0x00000001,  # +min subnormal
    0x80000001,  # -min subnormal
    0x007FFFFF,  # +max subnormal
    0x807FFFFF,  # -max subnormal
    0x00800000,  # +min normal
    0x80800000,  # -min normal
    0x7F800000,  # +Inf
    0xFF800000,  # -Inf
    0x7FC00000,  # +qNaN canonical
    0x7FC01234,  # +qNaN payload
    0xFFC05678,  # -qNaN payload
    0x7F800001,  # +sNaN min payload
    0xFF800001,  # -sNaN min payload
    0x7FBFFFFF,  # +sNaN max payload
)

F64_EDGE = (
    0x0000000000000000,  # +0.0
    0x8000000000000000,  # -0.0
    0x3FD0000000000000,  # +0.25
    0xBFD0000000000000,  # -0.25
    0x3FE0000000000000,  # +0.5 (tie -> +0.0)
    0xBFE0000000000000,  # -0.5 (tie -> -0.0)
    0x3FE0000000000001,  # +0.5 + ulp
    0xBFE0000000000001,  # -0.5 - ulp
    0x3FE8000000000000,  # +0.75
    0xBFE8000000000000,  # -0.75
    0x3FF0000000000000,  # +1.0
    0xBFF0000000000000,  # -1.0
    0x3FF8000000000000,  # +1.5 (tie -> +2.0)
    0xBFF8000000000000,  # -1.5 (tie -> -2.0)
    0x4000000000000000,  # +2.0
    0xC000000000000000,  # -2.0
    0x4004000000000000,  # +2.5 (tie -> +2.0)
    0xC004000000000000,  # -2.5 (tie -> -2.0)
    0x400C000000000000,  # +3.5 (tie -> +4.0)
    0xC00C000000000000,  # -3.5 (tie -> -4.0)
    0x432FFFFFFFFFFFFF,  # 2^52 - 0.5 (tie -> 2^52)
    0xC32FFFFFFFFFFFFF,  # -(2^52 - 0.5)
    0x432FFFFFFFFFFFFE,  # 2^52 - 1.0 (exact integer)
    0x4330000000000000,  # 2^52 (boundary >= 2^52)
    0xC330000000000000,  # -2^52
    0x4330000000000001,  # 2^52 + 1
    0x7FEFFFFFFFFFFFFF,  # +max finite
    0xFFEFFFFFFFFFFFFF,  # -max finite
    0x0000000000000001,  # +min subnormal
    0x8000000000000001,  # -min subnormal
    0x000FFFFFFFFFFFFF,  # +max subnormal
    0x800FFFFFFFFFFFFF,  # -max subnormal
    0x0010000000000000,  # +min normal
    0x8010000000000000,  # -min normal
    0x7FF0000000000000,  # +Inf
    0xFFF0000000000000,  # -Inf
    0x7FF8000000000000,  # +qNaN canonical
    0x7FF8123456789ABC,  # +qNaN payload
    0xFFF8FEDCBA987654,  # -qNaN payload
    0x7FF0000000000001,  # +sNaN min payload
    0xFFF0000000000001,  # -sNaN min payload
    0x7FF7FFFFFFFFFFFF,  # +sNaN max payload
)


def f32_pairs():
    pairs = []
    n = len(F32_EDGE)
    # Sweep every F32_EDGE element in lane 0 of BOTH a and b, with distinct dirty upper lanes.
    for i in range(n):
        a_lanes = [F32_EDGE[(i + k * 3) % n] for k in range(8)]
        b_lanes = [F32_EDGE[(i + k * 5 + 1) % n] for k in range(8)]
        pairs.append((pk_d(*a_lanes), pk_d(*b_lanes)))
    rng = random.Random(0xA032F)
    exps = (0, 1, 125, 126, 127, 128, 129, 130, 148, 149, 150, 151, 254)
    for _ in range(16):
        a = pk_d(*((rng.getrandbits(1) << 31) | (rng.choice(exps) << 23) | rng.getrandbits(23) for _ in range(8)))
        b = pk_d(*((rng.getrandbits(1) << 31) | (rng.choice(exps) << 23) | rng.getrandbits(23) for _ in range(8)))
        pairs.append((a, b))
    return pairs


def f64_pairs():
    pairs = []
    n = len(F64_EDGE)
    for i in range(n):
        a_lanes = [F64_EDGE[(i + k * 3) % n] for k in range(4)]
        b_lanes = [F64_EDGE[(i + k * 5 + 1) % n] for k in range(4)]
        pairs.append((pk_q(*a_lanes), pk_q(*b_lanes)))
    rng = random.Random(0xA064D)
    exps = (0, 1, 1021, 1022, 1023, 1024, 1025, 1026, 1073, 1074, 1075, 1076, 2046)
    for _ in range(16):
        a = pk_q(*((rng.getrandbits(1) << 63) | (rng.choice(exps) << 52) | rng.getrandbits(52) for _ in range(4)))
        b = pk_q(*((rng.getrandbits(1) << 63) | (rng.choice(exps) << 52) | rng.getrandbits(52) for _ in range(4)))
        pairs.append((a, b))
    return pairs


def inputs_for(name):
    if "pd" in name or "sd" in name or name == "op_vround_r8_reload":
        return f64_pairs()
    return f32_pairs()


# --- Guarded invocation & harness helpers ---

def alloc_guarded(payload, align=1, offset=0):
    total = len(GUARD_PRE) + offset + len(payload) + len(GUARD_POST) + align
    buf = ctypes.create_string_buffer(total)
    base_addr = ctypes.addressof(buf)
    start = base_addr + len(GUARD_PRE)
    pad = (align - start % align) % align
    pre_off = pad
    payload_off = pad + len(GUARD_PRE) + offset
    post_off = payload_off + len(payload)
    buf[pre_off:pre_off + len(GUARD_PRE)] = GUARD_PRE
    buf[pre_off + len(GUARD_PRE):payload_off] = b"\x00" * offset
    buf[payload_off:post_off] = payload
    buf[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return buf, base_addr + payload_off, pre_off, payload_off, post_off


def read_guarded(buf, pre_off, payload_off, post_off, label):
    raw = buf.raw
    if (raw[pre_off:pre_off + len(GUARD_PRE)] != GUARD_PRE
            or raw[post_off:post_off + len(GUARD_POST)] != GUARD_POST):
        raise AssertionError(f"guard bytes corrupted around {label}")
    return raw[payload_off:post_off]


def invoke(fn, a, b, *, a_off=0, b_off=0, out_off=0):
    ba, pa, ap, ao, aq = alloc_guarded(a, offset=a_off)
    bb, pb, bp, bo, bq = alloc_guarded(b, offset=b_off)
    bo_buf, po, op, oo, oq = alloc_guarded(FILL, offset=out_off)
    fn(pa, pb, po)
    read_guarded(ba, ap, ao, aq, "input_a")
    read_guarded(bb, bp, bo, bq, "input_b")
    return read_guarded(bo_buf, op, oo, oq, "output")


def bind_mxcsr_driver(harness_path):
    lib = ctypes.CDLL(str(Path(harness_path).resolve()))
    drv = lib.op_call_with_mxcsr
    drv.argtypes = [ctypes.c_uint64] * 4 + [ctypes.c_uint32]
    drv.restype = None
    return lib, drv


def invoke_under(driver, fn, mxcsr, a, b):
    ba, pa, ap, ao, aq = alloc_guarded(a)
    bb, pb, bp, bo, bq = alloc_guarded(b)
    bo_buf, po, op, oo, oq = alloc_guarded(FILL)
    driver(ctypes.cast(fn, ctypes.c_void_p).value, pa, pb, po, mxcsr)
    read_guarded(ba, ap, ao, aq, "input_a")
    read_guarded(bb, bp, bo, bq, "input_b")
    return read_guarded(bo_buf, op, oo, oq, "output")


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_avx(lifted_lib):
    if hasattr(lifted_lib, "host_has_avx"):
        lifted_lib.host_has_avx.argtypes = []
        lifted_lib.host_has_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_avx())
    return False


# --- Positive, MXCSR, inexact-suppress, altered, mutant, and /O2 checks ---

def check_discarded_read(fns, label):
    payload = f32_pairs()[0][0]
    ba, pa, _, _, _ = alloc_guarded(payload)
    bo_ok, po_ok, op, oo, oq = alloc_guarded(FILL)
    fns["op_vround_discarded_read"](pa, pa, po_ok)
    got_ok = read_guarded(bo_ok, op, oo, oq, "output")
    if got_ok != payload[:16] + FILL[16:]:
        raise AssertionError(f"{label} op_vround_discarded_read(valid): {got_ok.hex()}")

    bo_null, po_null, op2, oo2, oq2 = alloc_guarded(FILL)
    try:
        fns["op_vround_discarded_read"](pa, 0, po_null)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_vround_discarded_read(NULL) dropped faulting m32 read")
    if read_guarded(bo_null, op2, oo2, oq2, "output") != FILL:
        raise AssertionError(f"{label} op_vround_discarded_read(NULL) wrote output after fault")
    del ba


def check_inexact_suppress(fns, label):
    """Verify FE_INEXACT (with imm8[3] suppress) and FE_INVALID on sNaN vs qNaN."""
    ucrt = ctypes.CDLL("ucrtbase")
    ucrt.feclearexcept.argtypes = [ctypes.c_int]
    ucrt.feclearexcept.restype = ctypes.c_int
    ucrt.fetestexcept.argtypes = [ctypes.c_int]
    ucrt.fetestexcept.restype = ctypes.c_int
    FE_INEXACT = 0x01  # MSVC UCRT FE_INEXACT (_SW_INEXACT)
    FE_INVALID = 0x10  # MSVC UCRT FE_INVALID (_SW_INVALID)
    FE_ALL = FE_INEXACT | FE_INVALID
    inexact_vec = pk_d(0x3FC00000, 0x3FC00000, 0x3FC00000, 0x3FC00000, 0, 0, 0, 0)  # 1.5f
    exact_vec = pk_d(0x40000000, 0x40000000, 0x40000000, 0x40000000, 0, 0, 0, 0)  # 2.0f
    snan_f32 = pk_d(0x7F800001, 0x40000000, 0x40000000, 0x40000000, 0, 0, 0, 0)
    qnan_f32 = pk_d(0x7FC01234, 0x40000000, 0x40000000, 0x40000000, 0, 0, 0, 0)
    snan_f64 = pk_q(0xFFF0000000000001, 0x4000000000000000, 0, 0)
    qnan_f64 = pk_q(0x7FF8123456789ABC, 0x4000000000000000, 0, 0)

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_rn"], inexact_vec, exact_vec)
    if (ucrt.fetestexcept(FE_INEXACT) & FE_INEXACT) == 0:
        raise AssertionError(f"{label}: imm8[3]=0 failed to raise FE_INEXACT on 1.5f")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_suppress"], inexact_vec, exact_vec)
    if (ucrt.fetestexcept(FE_INEXACT) & FE_INEXACT) != 0:
        raise AssertionError(f"{label}: imm8[3]=1 failed to suppress FE_INEXACT on 1.5f")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_rn"], exact_vec, exact_vec)
    if (ucrt.fetestexcept(FE_ALL)) != 0:
        raise AssertionError(f"{label}: exact integer 2.0f unexpectedly raised FP exceptions")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_rn"], snan_f32, exact_vec)
    got_flags = ucrt.fetestexcept(FE_ALL)
    if (got_flags & FE_INVALID) == 0 or (got_flags & FE_INEXACT) != 0:
        raise AssertionError(f"{label}: sNaN f32 unexpected flags {got_flags:#x}")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_suppress"], snan_f32, exact_vec)
    got_flags = ucrt.fetestexcept(FE_ALL)
    if (got_flags & FE_INVALID) == 0 or (got_flags & FE_INEXACT) != 0:
        raise AssertionError(f"{label}: sNaN f32 with suppress unexpected flags {got_flags:#x}")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundps_xmm_rn"], qnan_f32, exact_vec)
    if (ucrt.fetestexcept(FE_ALL)) != 0:
        raise AssertionError(f"{label}: qNaN f32 unexpectedly raised FP exceptions")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundpd_xmm_rn"], snan_f64, exact_vec)
    got_flags = ucrt.fetestexcept(FE_ALL)
    if (got_flags & FE_INVALID) == 0 or (got_flags & FE_INEXACT) != 0:
        raise AssertionError(f"{label}: sNaN f64 unexpected flags {got_flags:#x}")

    ucrt.feclearexcept(FE_ALL)
    invoke(fns["op_vroundpd_xmm_rn"], qnan_f64, exact_vec)
    if (ucrt.fetestexcept(FE_ALL)) != 0:
        raise AssertionError(f"{label}: qNaN f64 unexpectedly raised FP exceptions")


def verify_against_ref(target_fns, label, *, check_unaligned=True):
    total = 0
    for name in EXPORTS:
        fn = target_fns[name]
        for idx, (a, b) in enumerate(inputs_for(name)):
            want = reference(name, a, b)
            got = invoke(fn, a, b)
            if got != want:
                raise AssertionError(f"{label} {name} case {idx}: got {got.hex()} != exp {want.hex()}")
            total += 1
        if check_unaligned:
            a, b = inputs_for(name)[3]
            want = reference(name, a, b)
            for a_off, b_off, out_off in ((1, 0, 0), (0, 3, 5)):
                got = invoke(fn, a, b, a_off=a_off, b_off=b_off, out_off=out_off)
                if got != want:
                    raise AssertionError(f"{label} {name} unaligned ({a_off},{b_off},{out_off})")
                total += 1
    check_discarded_read(target_fns, label)
    check_inexact_suppress(target_fns, label)
    return total


def verify_mxcsr_modes(target_fns, label, driver):
    runs = 0
    for name in MXCSR_EXPORTS:
        pairs = inputs_for(name)
        by_mode = {}
        for daz in (False, True):
            for rc in range(4):
                mxcsr = 0x1F80 | (0x40 if daz else 0) | (rc << 13)
                results = []
                for idx, (a, b) in enumerate(pairs):
                    want = reference(name, a, b, mxcsr_rc=rc, daz=daz)
                    got = invoke_under(driver, target_fns[name], mxcsr, a, b)
                    if got != want:
                        raise AssertionError(
                            f"{label} {name} [rc={rc},daz={int(daz)}] case {idx}: got {got.hex()} != exp {want.hex()}"
                        )
                    results.append(got)
                    runs += 1
                by_mode[(rc, daz)] = results
        for daz in (False, True):
            for r1 in range(4):
                for r2 in range(r1 + 1, 4):
                    if by_mode[(r1, daz)] == by_mode[(r2, daz)]:
                        raise AssertionError(
                            f"{label} {name}: MXCSR rc={r1} and rc={r2} (daz={int(daz)}) produced identical outputs"
                        )
        for rc in (1, 2):  # RD and RU must separate DAZ=0 from DAZ=1 on subnormals
            if by_mode[(rc, False)] == by_mode[(rc, True)]:
                raise AssertionError(f"{label} {name}: DAZ=0 and DAZ=1 produced identical outputs at rc={rc}")

    # Directed-rounding explicit-imm8 exports under DAZ=0 (0x1F80) and DAZ=1 (0x1FC0), including +/- subnormals.
    for name in DIRECTED_EXPORTS:
        pairs = inputs_for(name)
        for daz in (False, True):
            mxcsr = 0x1FC0 if daz else 0x1F80
            for idx, (a, b) in enumerate(pairs):
                want = reference(name, a, b, daz=daz)
                got = invoke_under(driver, target_fns[name], mxcsr, a, b)
                if got != want:
                    raise AssertionError(
                        f"{label} {name} [daz={int(daz)}] case {idx}: got {got.hex()} != exp {want.hex()}"
                    )
                runs += 1

    # Explicit positive/negative subnormal checks (including the 0x00000001 / 0x1FC0 reproducer).
    sub_f32_a = pk_d(0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF)
    sub_f64_a = pk_q(0x0000000000000001, 0x8000000000000001, 0x000FFFFFFFFFFFFF, 0x800FFFFFFFFFFFFF)
    for name, src_a, src_b in (
        ("op_vroundps_xmm_ru", sub_f32_a, sub_f32_a),
        ("op_vroundps_xmm_rd", sub_f32_a, sub_f32_a),
        ("op_vroundps_ymm_ru", sub_f32_a, sub_f32_a),
        ("op_vroundps_ymm_rd", sub_f32_a, sub_f32_a),
        ("op_vroundss_ru", pk_d(0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0, 0, 0, 0), sub_f32_a),
        ("op_vroundss_rd", pk_d(0x3F800000, 0x40000000, 0x40400000, 0x40800000, 0, 0, 0, 0), pk_d(0x80000001, 0, 0, 0, 0, 0, 0, 0)),
        ("op_vroundpd_xmm_ru", sub_f64_a, sub_f64_a),
        ("op_vroundpd_xmm_rd", sub_f64_a, sub_f64_a),
        ("op_vroundpd_ymm_ru", sub_f64_a, sub_f64_a),
        ("op_vroundpd_ymm_rd", sub_f64_a, sub_f64_a),
        ("op_vroundsd_ru", pk_q(0x3FF0000000000000, 0x4000000000000000, 0, 0), sub_f64_a),
        ("op_vroundsd_rd", pk_q(0x3FF0000000000000, 0x4000000000000000, 0, 0), pk_q(0x8000000000000001, 0, 0, 0)),
    ):
        for daz in (False, True):
            mxcsr = 0x1FC0 if daz else 0x1F80
            want = reference(name, src_a, src_b, daz=daz)
            got = invoke_under(driver, target_fns[name], mxcsr, src_a, src_b)
            if got != want:
                raise AssertionError(f"{label} subnormal {name} [daz={int(daz)}]: got {got.hex()} != exp {want.hex()}")
            runs += 1
    return runs


def check_altered_and_mutants(lifted_fns, altered_lifted_fns, driver):
    diverged = 0
    for name in ALTERED_IMM:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt = reference(name, a, b, altered=True)
            want_orig = reference(name, a, b)
            got_al = invoke(altered_lifted_fns[name], a, b)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name} case {idx}: got {got_al.hex()} != exp {want_alt.hex()}")
            if got_al != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"ALTER_ROUND control {name} did not diverge from original")
        diverged += 1

    # Mutant models on the independent reference must disagree with lifted outputs.
    quirk_checks = (
        ("ties_away", "op_vroundps_xmm_rn", 0, False),
        ("ties_away", "op_vroundpd_ymm_rn", 0, False),
        ("drop_neg_zero", "op_vroundps_xmm_rn", 0, False),
        ("drop_neg_zero", "op_vroundps_xmm_ru", 0, False),
        ("drop_neg_zero", "op_vroundpd_xmm_rz", 0, False),
        ("drop_neg_zero", "op_vroundss_ru", 0, False),
        ("drop_neg_zero", "op_vroundsd_rz", 0, False),
        ("no_quieting", "op_vroundps_ymm_rn", 0, False),
        ("no_quieting", "op_vroundpd_xmm_rd", 0, False),
        ("no_quieting", "op_vroundss_rn", 0, False),
        ("no_quieting", "op_vroundsd_rn", 0, False),
        ("clobber_scalar_upper", "op_vroundss_rn", 0, False),
        ("clobber_scalar_upper", "op_vroundsd_rn", 0, False),
        ("keep_ymm_upper", "op_vroundps_xmm_zero_upper", 0, False),
        ("keep_ymm_upper", "op_vroundpd_xmm_zero_upper", 0, False),
        ("keep_ymm_upper", "op_vroundss_zero_upper", 0, False),
        ("keep_ymm_upper", "op_vroundsd_zero_upper", 0, False),
        ("ignore_mxcsr", "op_vroundps_xmm_mxcsr", 1, False),
        ("ignore_mxcsr", "op_vroundpd_ymm_mxcsr", 2, False),
        ("ignore_mxcsr", "op_vroundss_mxcsr", 3, False),
        ("ignore_mxcsr", "op_vroundsd_mxcsr", 1, False),
        ("ignore_daz", "op_vroundps_xmm_ru", 0, True),
        ("ignore_daz", "op_vroundps_ymm_rd", 0, True),
        ("ignore_daz", "op_vroundpd_xmm_ru", 0, True),
        ("ignore_daz", "op_vroundpd_ymm_rd", 0, True),
        ("ignore_daz", "op_vroundss_ru", 0, True),
        ("ignore_daz", "op_vroundsd_rd", 0, True),
        ("ignore_daz", "op_vroundps_xmm_mxcsr", 2, True),
        ("ignore_daz", "op_vroundpd_ymm_mxcsr", 1, True),
    )
    for quirk, name, rc, daz in quirk_checks:
        mxcsr = 0x1F80 | (0x40 if daz else 0) | (rc << 13)
        pairs = inputs_for(name)
        caught = sum(
            invoke_under(driver, lifted_fns[name], mxcsr, a, b) != reference(name, a, b, mxcsr_rc=rc, daz=daz, quirk=quirk)
            for a, b in pairs
        )
        if not caught:
            raise AssertionError(f"mutant quirk {quirk} on {name} (rc={rc}, daz={int(daz)}) survived")
    return diverged, len(quirk_checks)


def compile_o2_and_verify(original_path, cxx_path, driver):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "avx_round_o2.cpp"
        dll_path = tmp / "avx_round_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'avx_round_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            cases = verify_against_ref(o2_fns, "lifted_o2", check_unaligned=False)
            mx_runs = verify_mxcsr_modes(o2_fns, "lifted_o2", driver)
            return cases, mx_runs
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


# --- Alternate encodings & rejection controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_alt_encodings_and_rejections(original_path):
    ret = Instruction.create(Code.RETNQ)
    ld_x0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_x1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_x4 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM4, MemoryOperand(R.RCX))
    ld_x5 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM5, MemoryOperand(R.RDX))
    ld_y0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_y4 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM4, MemoryOperand(R.RCX))
    st_x0 = Instruction.create_mem_reg(Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)
    st_x4 = Instruction.create_mem_reg(Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4)
    st_y4 = Instruction.create_mem_reg(Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.R8), R.YMM4)
    vzu = Instruction.create(Code.VEX_VZEROUPPER)

    alt_accept = [
        ("vroundps_xmm4_rcx", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM4, MemoryOperand(R.RCX), 0x0C), st_x4, ret),
         "ymm4 = op_vroundps_128(load128(rcx), 0xcu);"),
        ("vroundpd_ymm4_r8", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPD_YMM_YMMM256_IMM8, R.YMM4, MemoryOperand(R.R8), 0x0B), st_y4, vzu, ret),
         "ymm4 = op_vroundpd_256(load256(r8), 0xbu);"),
        ("vroundss_xmm5_xmm4_r8", (ld_x4, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM5, R.XMM4, MemoryOperand(R.R8), 0x09),
            Instruction.create_mem_reg(Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM5), ret),
         "ymm5 = op_vroundss(ymm4, load32(r8), 0x9u);"),
        ("vroundsd_xmm4_xmm5_rcx", (ld_x5, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8, R.XMM4, R.XMM5, MemoryOperand(R.RCX), 0x0A), st_x4, ret),
         "ymm4 = op_vroundsd(ymm5, load64(rcx), 0xau);"),
        ("vmovdqu_reg_copy_xmm", (ld_x4, Instruction.create_reg_reg(
            Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, R.XMM4), st_x0, ret),
         "ymm0 = copy128(ymm4);"),
        ("vmovdqu_reg_copy_ymm", (ld_y4, Instruction.create_reg_reg(
            Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, R.YMM4),
            Instruction.create_mem_reg(Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.R8), R.YMM0), vzu, ret),
         "ymm0 = ymm4;"),
    ]
    for label, seq, expected in alt_accept:
        raw = encode_seq(*seq)
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) == len(seq) and not any(ins.is_invalid for ins in decoded), label
        src, count = lift_code(label, 0x1000, raw)
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label} missing {expected!r}")

    seg_ld = Instruction.create_reg_mem_u32(Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RCX), 0)
    seg_ld.segment_prefix = R.GS
    reject_cases = [
        # Legacy SSE4.1 ROUND* and legacy MOVDQU
        ("legacy_roundps", (ld_x1, Instruction.create_reg_reg_u32(
            Code.ROUNDPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret), "unsupported legacy SSE encoding"),
        ("legacy_roundpd", (ld_x1, Instruction.create_reg_reg_u32(
            Code.ROUNDPD_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 1), ret), "unsupported legacy SSE encoding"),
        ("legacy_roundss", (ld_x0, ld_x1, Instruction.create_reg_reg_u32(
            Code.ROUNDSS_XMM_XMMM32_IMM8, R.XMM0, R.XMM1, 2), ret), "unsupported legacy SSE encoding"),
        ("legacy_roundsd", (ld_x0, ld_x1, Instruction.create_reg_reg_u32(
            Code.ROUNDSD_XMM_XMMM64_IMM8, R.XMM0, R.XMM1, 3), ret), "unsupported legacy SSE encoding"),
        ("legacy_movdqu", (Instruction.create_reg_mem(
            Code.MOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)), ret), "unsupported legacy SSE encoding"),
        # EVEX VRNDSCALE*
        ("evex_vrndscaleps_xmm", (ld_x1, Instruction.create_reg_reg_u32(
            Code.EVEX_VRNDSCALEPS_XMM_K1Z_XMMM128B32_IMM8, R.XMM0, R.XMM1, 0), ret), "unsupported EVEX instruction"),
        ("evex_vrndscalepd_ymm", (ld_y0, Instruction.create_reg_reg_u32(
            Code.EVEX_VRNDSCALEPD_YMM_K1Z_YMMM256B64_IMM8, R.YMM1, R.YMM0, 0), ret), "unsupported EVEX instruction"),
        ("evex_vrndscaless", (ld_x0, ld_x1, Instruction.create_reg_reg_reg_u32(
            Code.EVEX_VRNDSCALESS_XMM_K1Z_XMM_XMMM32_IMM8_SAE, R.XMM2, R.XMM0, R.XMM1, 0), ret),
         "unsupported EVEX instruction"),
        ("evex_vrndscalesd", (ld_x0, ld_x1, Instruction.create_reg_reg_reg_u32(
            Code.EVEX_VRNDSCALESD_XMM_K1Z_XMM_XMMM64_IMM8_SAE, R.XMM2, R.XMM0, R.XMM1, 0), ret),
         "unsupported EVEX instruction"),
        # Nonvolatile XMM6..15 / YMM6..15
        ("nonvol_xmm6_vroundps_dst", (ld_x1, Instruction.create_reg_reg_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM6, R.XMM1, 0), ret), "nonvolatile vector register"),
        ("nonvol_ymm15_vroundpd_src", (Instruction.create_reg_reg_u32(
            Code.VEX_VROUNDPD_YMM_YMMM256_IMM8, R.YMM0, R.YMM15, 0), ret), "nonvolatile vector register"),
        ("nonvol_xmm7_vroundss_src1", (ld_x1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM0, R.XMM7, R.XMM1, 0), ret), "nonvolatile vector register"),
        ("nonvol_xmm9_vroundsd_src2", (ld_x0, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8, R.XMM2, R.XMM0, R.XMM9, 0), ret), "nonvolatile vector register"),
        # Undefined reads
        ("undef_vroundps_src", (Instruction.create_reg_reg_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, R.XMM2, 0), ret), "undefined vector register ymm2"),
        ("undef_vroundpd_ymm_src", (Instruction.create_reg_reg_u32(
            Code.VEX_VROUNDPD_YMM_YMMM256_IMM8, R.YMM0, R.YMM3, 1), ret), "undefined vector register ymm3"),
        ("undef_vroundss_src1", (ld_x1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM2, R.XMM0, R.XMM1, 0), ret), "undefined vector register ymm0"),
        ("undef_vroundss_src2", (ld_x0, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM2, R.XMM0, R.XMM3, 0), ret), "undefined vector register ymm3"),
        ("undef_vroundsd_src1", (ld_x1, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8, R.XMM2, R.XMM4, R.XMM1, 0), ret), "undefined vector register ymm4"),
        ("undef_vroundsd_src2", (ld_x0, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8, R.XMM2, R.XMM0, R.XMM5, 0), ret), "undefined vector register ymm5"),
        ("undef_vmovdqu_store", (Instruction.create_mem_reg(
            Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM2), ret), "undefined vector register ymm2"),
        # Displaced, indexed, RIP-rel, RSP, addr32, non-arg base memory operands
        ("displaced_vroundps", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RDX, displ=16, displ_size=1), 0), ret),
         "unsupported memory operand"),
        ("displaced_vroundss", (ld_x0, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM2, R.XMM0, MemoryOperand(R.RDX, displ=4, displ_size=1), 0), ret),
         "unsupported memory operand"),
        ("displaced_vmovdqu_store", (ld_x0, Instruction.create_mem_reg(
            Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8, displ=16, displ_size=1), R.XMM0), ret),
         "unsupported memory operand"),
        ("indexed_vroundpd", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPD_YMM_YMMM256_IMM8, R.YMM0, MemoryOperand(R.RCX, R.RDX, 4), 0), ret),
         "unsupported memory operand"),
        ("non_arg_base_rax", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RAX), 0), ret),
         "unsupported memory operand"),
        ("nonvol_base_rbx", (ld_x0, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VROUNDSD_XMM_XMM_XMMM64_IMM8, R.XMM2, R.XMM0, MemoryOperand(R.RBX), 0), ret),
         "unsupported memory operand"),
        ("rip_rel_vroundpd", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPD_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RIP, displ=0x1020, displ_size=4), 0), ret),
         "unsupported memory operand"),
        ("rsp_vroundps", (Instruction.create_reg_mem_u32(
            Code.VEX_VROUNDPS_XMM_XMMM128_IMM8, R.XMM0, MemoryOperand(R.RSP), 0), ret),
         "unsupported memory operand"),
        ("addr32_vroundss", (ld_x0, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VROUNDSS_XMM_XMM_XMMM32_IMM8, R.XMM2, R.XMM0, MemoryOperand(R.EDX), 0), ret),
         "unsupported memory operand"),
        # Stores outside [r8]
        ("store_to_rcx", (ld_x0, Instruction.create_mem_reg(
            Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret), "store outside output buffer r8"),
        ("store_to_rdx_ymm", (ld_y0, Instruction.create_mem_reg(
            Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.RDX), R.YMM0), ret), "store outside output buffer r8"),
        # Prefixes, branches, missing/trailing ret
        ("segment_prefix", (seg_ld, st_x0, ret), "unsupported instruction prefix"),
        ("branch_jmp", (ld_x0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret), "unsupported instruction"),
        ("unsupported_vaddps", (ld_x0, ld_x1, Instruction.create_reg_reg_reg(
            Code.VEX_VADDPS_XMM_XMM_XMMM128, R.XMM2, R.XMM0, R.XMM1), ret), "unsupported instruction"),
        ("missing_ret", (ld_x0, st_x0), "missing terminal ret"),
        ("trailing_after_ret", (ld_x0, st_x0, ret, ld_x0), "unreachable instruction after ret"),
    ]

    ret_bytes = encode_seq(ret)
    raw_cases = [
        ("lock_cmpxchg", bytes.fromhex("f00fb109") + ret_bytes, "unsupported instruction prefix"),
        ("rep_movsb", bytes.fromhex("f3a4") + ret_bytes, "unsupported instruction prefix"),
        ("empty_stream", b"", "truncated instruction stream"),
        ("invalid_bytes", bytes.fromhex("fffe") + ret_bytes, "invalid instruction"),
    ]

    for label, seq, reason in reject_cases:
        raw = encode_seq(*seq)
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted raw rejected control")

    # PE-patched rejections & CLI non-traceback rejection.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_vround_r8_reload")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in reject_cases}
    pe_labels = (
        "legacy_roundps",
        "evex_vrndscaleps_xmm",
        "nonvol_xmm6_vroundps_dst",
        "undef_vroundps_src",
        "displaced_vroundps",
        "store_to_rcx",
        "branch_jmp",
        "missing_ret",
        "segment_prefix",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_avx_round_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_labels:
            patch, reason = by_label[label]
            if len(patch) > max_len:
                raise AssertionError(f"pe_{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched.write_bytes(damaged)
            try:
                translate(patched, ["op_vround_r8_reload"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_vround_r8_reload", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists():
            raise AssertionError("CLI failure left behind output file")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(reject_cases)} IR + "
        f"{len(raw_cases)} raw + {len(pe_labels)} PE-patched + CLI"
    )


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    formatter.use_pseudo_ops = False
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

    original_path, lifted_path, altered_path, altered_lifted_path, harness_path, cxx_path = (
        Path(p) for p in sys.argv[1:7]
    )

    # 1) Lift checks, rejections, portable lifted execution, MXCSR, mutants, and /O2 run BEFORE any AVX CPUID skip.
    check_alt_encodings_and_rejections(original_path)

    harness_lib, driver = bind_mxcsr_driver(harness_path)
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_against_ref(lifted_fns, "lifted")
        lifted_mx = verify_mxcsr_modes(lifted_fns, "lifted", driver)
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_lifted_fns, driver)
        o2_cases, o2_mx = compile_o2_and_verify(original_path, cxx_path, driver)

        if not host_has_avx(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{lifted_mx} MXCSR runs, {diverged} ALTER_ROUND + {mutants} mutants); "
                "host lacks AVX, skipping native differential check"
            )
            sys.exit(77)

        # 2) Native AVX differential verification.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_against_ref(orig_fns, "native")
            native_mx = verify_mxcsr_modes(orig_fns, "native", driver)
            for name in ALTERED_IMM:
                for idx, (a, b) in enumerate(inputs_for(name)):
                    got_an = invoke(alt_fns[name], a, b)
                    got_al = invoke(alt_lifted_fns[name], a, b)
                    if got_an != got_al:
                        raise AssertionError(
                            f"altered native vs lifted mismatch on {name} case {idx}: "
                            f"{got_an.hex()} != {got_al.hex()}"
                        )
        finally:
            orig_fns.clear()
            alt_fns.clear()
            unload_lib(orig_lib)
            unload_lib(alt_lib)
    finally:
        lifted_fns.clear()
        alt_lifted_fns.clear()
        unload_lib(lifted_lib)
        unload_lib(alt_lifted_lib)
        unload_lib(harness_lib)

    record_covered_forms(original_path)
    print(
        f"avx_round passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; {native_mx} native + {lifted_mx} lifted + {o2_mx} /O2 MXCSR runs; "
        f"ALTER_ROUND diverged on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
