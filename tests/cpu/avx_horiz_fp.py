"""Differential check for VEX AVX horizontal FP add/subtract lowering (vhaddps, vhaddpd, vhsubps, vhsubpd)."""
import ctypes
from fractions import Fraction
import re
import struct
import sys
from types import SimpleNamespace
from pathlib import Path

from iced_x86 import Code as C, Decoder, EncodingKind, MemorySize, Mnemonic as M, OpKind as K, Register as R
import pefile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

XMM_REG_EXPORTS = (
    "vhaddps_xmm_reg_op", "vhaddpd_xmm_reg_op", "vhsubps_xmm_reg_op", "vhsubpd_xmm_reg_op",
)
XMM_MEM_EXPORTS = (
    "vhaddps_xmm_mem_op", "vhaddpd_xmm_mem_op", "vhsubps_xmm_mem_op", "vhsubpd_xmm_mem_op",
)
YMM_REG_EXPORTS = (
    "vhaddps_ymm_reg_op", "vhaddpd_ymm_reg_op", "vhsubps_ymm_reg_op", "vhsubpd_ymm_reg_op",
)
YMM_MEM_EXPORTS = (
    "vhaddps_ymm_mem_op", "vhaddpd_ymm_mem_op", "vhsubps_ymm_mem_op", "vhsubpd_ymm_mem_op",
)
ALIAS_EXPORTS = (
    "vhaddps_ymm_alias_dst_src1_op",
    "vhaddpd_ymm_alias_dst_src2_op",
    "vhsubps_ymm_alias_src1_src2_op",
    "vhsubpd_ymm_alias_all_op",
    "vhaddps_xmm_alias_dst_src2_op",
    "vhsubps_xmm_alias_dst_src1_op",
)
EXPORTS = XMM_REG_EXPORTS + XMM_MEM_EXPORTS + YMM_REG_EXPORTS + YMM_MEM_EXPORTS + ALIAS_EXPORTS

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
extern "C" __declspec(dllexport) uint32_t host_exchange_mxcsr(uint32_t next) {
    const uint32_t current = _mm_getcsr();
    _mm_setcsr(next);
    return current;
}
"""

MUTATED_CONTROLS = (
    ("vhaddps_ymm_reg_op", "vhaddps_mutated_control",
     re.compile(r"_mm_hadd_ps\("), "_mm_hsub_ps("),
    ("vhaddps_xmm_reg_op", "vhaddps_xmm_no_zero_hi_control",
     re.compile(r"memset\(wide0\.data\(\) \+ 16, 0, 16\);"), ""),
)


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    lowered_by_name = {}
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            lowered_by_name[name] = code
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    for src_name, ctrl_name, pattern, replacement in MUTATED_CONTROLS:
        mutated, count = pattern.subn(replacement, lowered_by_name[src_name])
        if not count:
            raise AssertionError(f"mutated control {ctrl_name} changed nothing")
        parts.append(
            f'extern "C" __declspec(dllexport) uint64_t {ctrl_name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{mutated}\n}}'
        )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _make_synthetic_horiz_insn(
    encoding=EncodingKind.VEX,
    op0=R.XMM0,
    op1=R.XMM1,
    op2_kind=K.REGISTER,
    op2=R.XMM2,
    memory_size=MemorySize.UINT128,
):
    kinds = (K.REGISTER, K.REGISTER, op2_kind)
    regs = (op0, op1, op2)
    return SimpleNamespace(
        ip=0x1000,
        mnemonic=M.VHADDPS,
        code=C.INVALID,
        encoding=encoding,
        op_count=3,
        op0_kind=kinds[0],
        op1_kind=kinds[1],
        op2_kind=kinds[2],
        op0_register=regs[0],
        op1_register=regs[1],
        op2_register=regs[2],
        memory_size=memory_size,
        segment_prefix=R.NONE,
        has_lock_prefix=False,
        op_kind=lambda n: kinds[n],
        op_register=lambda n: regs[n],
    )


def check_controls() -> None:
    rejected_bytes = (
        ("c5f37cda", "VEX vhaddps xmm3, xmm1, xmm2 (untracked dst)"),
        ("c5e37cc2", "VEX vhaddps xmm0, xmm3, xmm2 (untracked src1)"),
        ("c5f37cc3", "VEX vhaddps xmm0, xmm1, xmm3 (untracked src2)"),
        ("c5f77cda", "VEX vhaddps ymm3, ymm1, ymm2 (untracked dst)"),
        ("c5e57cc2", "VEX vhaddpd ymm0, ymm3, ymm2 (untracked src1)"),
        ("c5f77dc3", "VEX vhsubps ymm0, ymm1, ymm3 (untracked src2)"),
        ("c5f57d1a", "VEX vhsubpd ymm3, ymm1, [rdx] (untracked dst with mem)"),
        ("64c5f77c02", "VEX vhaddps ymm0, ymm1, fs:[rdx] (segment prefix)"),
        ("67c5f37c0500000000", "VEX vhaddps xmm0, xmm1, [eip] (EIP-relative)"),
        ("62f1742958c2", "EVEX vaddps ymm0{k1}, ymm1, ymm2"),
        ("62f174287cc2", "EVEX-prefixed invalid vhaddps bytes"),
        ("660f7cc3", "Legacy haddpd xmm0, xmm3 (untracked src)"),
    )
    for encoded, description in rejected_bytes:
        body = list(Decoder(64, bytes.fromhex(encoded + "c3"), ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected {description} to be rejected")

    synthetic_bad = (
        (_make_synthetic_horiz_insn(encoding=EncodingKind.EVEX), "non-VEX encoding"),
        (_make_synthetic_horiz_insn(encoding=EncodingKind.LEGACY), "legacy encoding"),
        (_make_synthetic_horiz_insn(op0=R.YMM0, op1=R.XMM1, op2=R.YMM2), "mixed YMM/XMM width"),
        (_make_synthetic_horiz_insn(op0=R.XMM0, op1=R.XMM1, op2_kind=K.MEMORY, op2=R.NONE,
                                    memory_size=MemorySize.UINT256), "mismatched memory size"),
    )
    for insn, description in synthetic_bad:
        try:
            lower([insn], [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected synthetic {description} to be rejected")


# --- Independent IEEE-754 binary32 / binary64 reference model with MXCSR RC/DAZ/FTZ ---

class _FPFormat:
    def __init__(self, ebits: int, mbits: int):
        self.ebits = ebits
        self.mbits = mbits
        self.bits = 1 + ebits + mbits
        self.bias = (1 << (ebits - 1)) - 1
        self.emax = (1 << ebits) - 1
        self.sign_mask = 1 << (self.bits - 1)
        self.exp_mask = self.emax << mbits
        self.frac_mask = (1 << mbits) - 1
        self.quiet_bit = 1 << (mbits - 1)
        self.indef = self.sign_mask | self.exp_mask | self.quiet_bit
        self.max_finite = ((self.emax - 1) << mbits) | self.frac_mask

    def unpack(self, raw: int):
        sign = (raw >> (self.bits - 1)) & 1
        exp = (raw >> self.mbits) & self.emax
        frac = raw & self.frac_mask
        return sign, exp, frac

    def is_nan(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac != 0

    def is_inf(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac == 0

    def is_zero(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == 0 and frac == 0

    def flush_daz(self, raw: int, daz: bool) -> int:
        if not daz:
            return raw
        sign, exp, frac = self.unpack(raw)
        if exp == 0 and frac != 0:
            return sign << (self.bits - 1)
        return raw

    def quiet(self, raw: int) -> int:
        return raw | self.quiet_bit

    def to_fraction(self, raw: int) -> Fraction:
        sign, exp, frac = self.unpack(raw)
        if exp == 0:
            mag = Fraction(frac, 1) * (Fraction(1, 2) ** (self.bias - 1 + self.mbits))
        else:
            mag = Fraction((1 << self.mbits) | frac, 1) * (Fraction(2, 1) ** (exp - self.bias - self.mbits))
        return -mag if sign else mag

    @staticmethod
    def _round_mag(mag: Fraction, sign: int, rc: int) -> int:
        q, r = divmod(mag.numerator, mag.denominator)
        if rc == 0:  # Round to nearest, ties to even
            twice_r = r * 2
            if twice_r > mag.denominator or (twice_r == mag.denominator and (q & 1)):
                q += 1
        elif rc == 1:  # Round toward -infinity
            if sign and r:
                q += 1
        elif rc == 2:  # Round toward +infinity
            if not sign and r:
                q += 1
        # rc == 3: Round toward zero -> q unchanged
        return q

    def _overflow(self, sign: int, rc: int) -> int:
        sign_bit = sign << (self.bits - 1)
        inf_val = sign_bit | self.exp_mask
        max_val = sign_bit | self.max_finite
        if rc == 0:
            return inf_val
        if rc == 1:
            return inf_val if sign else max_val
        if rc == 2:
            return max_val if sign else inf_val
        return max_val

    def from_fraction(self, val: Fraction, *, rc: int, ftz: bool, cancel_zero: bool = False) -> int:
        if val == 0:
            return self.sign_mask if (cancel_zero and rc == 1) else 0
        sign = 1 if val < 0 else 0
        mag = -val if sign else val
        num, den = mag.numerator, mag.denominator
        e = num.bit_length() - den.bit_length()
        while (num << (-e) if e < 0 else num) < (den << e if e >= 0 else den):
            e -= 1
        while (num << (-(e + 1)) if e + 1 < 0 else num) >= (den << (e + 1) if e + 1 >= 0 else den):
            e += 1
        emin = 1 - self.bias
        if e < emin:
            scaled = mag * (Fraction(2, 1) ** (self.mbits - emin))
            q = self._round_mag(scaled, sign, rc)
            if q == (1 << self.mbits):
                return (sign << (self.bits - 1)) | (1 << self.mbits)
            if ftz or q == 0:
                return sign << (self.bits - 1)
            return (sign << (self.bits - 1)) | q
        scaled = mag * (Fraction(2, 1) ** (self.mbits - e))
        q = self._round_mag(scaled, sign, rc)
        if q == (1 << (self.mbits + 1)):
            q >>= 1
            e += 1
        bexp = e + self.bias
        if bexp >= self.emax:
            return self._overflow(sign, rc)
        return (sign << (self.bits - 1)) | (bexp << self.mbits) | (q & self.frac_mask)

    def add(self, a: int, b: int, mxcsr: int = 0x1f80) -> int:
        rc = (mxcsr >> 13) & 3
        daz = bool(mxcsr & 0x40)
        ftz = bool(mxcsr & 0x8000)
        a = self.flush_daz(a, daz)
        b = self.flush_daz(b, daz)
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_nan(b):
            return self.quiet(b)
        sa, _, _ = self.unpack(a)
        sb, _, _ = self.unpack(b)
        if self.is_inf(a) and self.is_inf(b):
            return self.indef if sa != sb else a
        if self.is_inf(a):
            return a
        if self.is_inf(b):
            return b
        if self.is_zero(a) and self.is_zero(b):
            if sa == sb:
                return self.sign_mask if sa else 0
            return self.sign_mask if rc == 1 else 0
        return self.from_fraction(
            self.to_fraction(a) + self.to_fraction(b), rc=rc, ftz=ftz, cancel_zero=True
        )

    def sub(self, a: int, b: int, mxcsr: int = 0x1f80) -> int:
        rc = (mxcsr >> 13) & 3
        daz = bool(mxcsr & 0x40)
        a = self.flush_daz(a, daz)
        b = self.flush_daz(b, daz)
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_nan(b):
            return self.quiet(b)
        return self.add(a, b ^ self.sign_mask, mxcsr)


F32 = _FPFormat(8, 23)
F64 = _FPFormat(11, 52)


def _lane_hadd_ps(lane_a: bytes, lane_b: bytes, mxcsr: int, sub: bool) -> bytes:
    xa = struct.unpack("<4I", lane_a)
    xb = struct.unpack("<4I", lane_b)
    fn = F32.sub if sub else F32.add
    return struct.pack(
        "<4I",
        fn(xa[0], xa[1], mxcsr),
        fn(xa[2], xa[3], mxcsr),
        fn(xb[0], xb[1], mxcsr),
        fn(xb[2], xb[3], mxcsr),
    )


def _lane_hadd_pd(lane_a: bytes, lane_b: bytes, mxcsr: int, sub: bool) -> bytes:
    xa = struct.unpack("<2Q", lane_a)
    xb = struct.unpack("<2Q", lane_b)
    fn = F64.sub if sub else F64.add
    return struct.pack("<2Q", fn(xa[0], xa[1], mxcsr), fn(xb[0], xb[1], mxcsr))


def ref_op(name: str, a: bytes, b: bytes, mxcsr: int) -> bytes:
    if name == "vhsubps_ymm_alias_src1_src2_op":
        b = a
    elif name == "vhsubpd_ymm_alias_all_op":
        b = a
    is_xmm = "_xmm_" in name
    is_pd = "pd_" in name
    is_sub = "hsub" in name
    lane_fn = _lane_hadd_pd if is_pd else _lane_hadd_ps
    lo = lane_fn(a[0:16], b[0:16], mxcsr, is_sub)
    if is_xmm:
        return lo + b"\x00" * 16
    hi = lane_fn(a[16:32], b[16:32], mxcsr, is_sub)
    return lo + hi


def _guarded_buf(payload: bytes, align_offset: int = 0):
    guard = b"\xCC" * 32
    raw = ctypes.create_string_buffer(guard + payload + guard + bytes(32))
    base = ctypes.addressof(raw) + 32
    mis = base % 32
    aligned = base + ((32 - mis) % 32)
    ptr = aligned + align_offset
    ctypes.memmove(ptr - 32, guard, 32)
    ctypes.memmove(ptr, payload, len(payload))
    ctypes.memmove(ptr + len(payload), guard, 32)
    return raw, ptr


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.argtypes, lifted.host_has_avx.restype = [], ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))
    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes, exchange.restype = [ctypes.c_uint32], ctypes.c_uint32

    def _f32(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in vals]
        return struct.pack("<8I", *words)

    def _f64(*vals):
        words = [v if isinstance(v, int) else struct.unpack("<Q", struct.pack("<d", v))[0] for v in vals]
        return struct.pack("<4Q", *words)

    f32_qnan_pos, f32_qnan_neg = 0x7fc01234, 0xffc05678
    f32_snan_pos, f32_snan_neg = 0x7f800042, 0xffa00099
    f32_sub_pos, f32_sub_neg = 0x00400001, 0x80400001
    f32_min_norm = 0x00800000
    f32_one_half_min_norm = 0x00c00000
    f32_max_fin = 0x7f7fffff

    f32_pairs = (
        # Distinct values across both 128-bit lanes + rounding ties + subnormals
        (_f32(1.5, -2.25, 1.0, 2.0 ** -24, 3.5, 0.125, -1.0, -(2.0 ** -24)),
         _f32(4.25, 0.75, f32_min_norm, f32_sub_neg, -7.5, 2.5, f32_sub_pos, f32_sub_pos)),
        # Signed zeros, exact cancellations, and normal->subnormal underflow (tests FTZ)
        (_f32(0x00000000, 0x00000000, 0x80000000, 0x80000000, f32_one_half_min_norm, f32_min_norm | 0x80000000, 1.5, -1.5),
         _f32(0x00000000, 0x80000000, 0x00000000, 0x80000000, f32_one_half_min_norm | 0x80000000, f32_min_norm, 2.5, 2.5)),
        # Infinities, overflow, and subnormal-to-normal boundary
        (_f32(0x7f800000, 0x7f800000, 0x7f800000, 0xff800000, f32_max_fin, f32_max_fin, f32_sub_pos, f32_sub_pos),
         _f32(0x7f800000, 1.0, 0xff800000, 0xff800000, f32_max_fin | 0x80000000, f32_max_fin | 0x80000000, f32_min_norm, f32_min_norm | 0x80000000)),
        # QNaN and SNaN payloads (single NaN and NaN-vs-NaN operand order)
        (_f32(f32_qnan_pos, 1.0, 2.0, f32_snan_pos, f32_qnan_pos, f32_qnan_neg, f32_snan_pos, f32_snan_neg),
         _f32(f32_snan_neg, f32_qnan_neg, f32_qnan_neg, 3.0, f32_snan_neg, f32_qnan_pos, f32_qnan_pos, f32_snan_pos)),
    )

    f64_qnan_pos, f64_qnan_neg = 0x7ff8000012345678, 0xfff8000087654321
    f64_snan_pos, f64_snan_neg = 0x7ff0000000000042, 0xfff4000000000099
    f64_sub_pos, f64_sub_neg = 0x0008000000000001, 0x8008000000000001
    f64_min_norm = 0x0010000000000000
    f64_one_half_min_norm = 0x0018000000000000
    f64_max_fin = 0x7fefffffffffffff

    f64_pairs = (
        (_f64(1.5, 2.0 ** -53, -3.5, -(2.0 ** -53)),
         _f64(f64_min_norm, f64_sub_neg, f64_sub_pos, f64_sub_pos)),
        (_f64(0x0000000000000000, 0x8000000000000000, f64_one_half_min_norm, f64_min_norm | 0x8000000000000000),
         _f64(2.25, -2.25, 0x0000000000000000, 0x8000000000000000)),
        (_f64(0x7ff0000000000000, 0xfff0000000000000, f64_max_fin, f64_max_fin),
         _f64(0x7ff0000000000000, 0x7ff0000000000000, f64_max_fin | 0x8000000000000000, f64_max_fin | 0x8000000000000000)),
        (_f64(f64_qnan_pos, 1.0, f64_snan_pos, f64_qnan_neg),
         _f64(f64_snan_neg, f64_qnan_neg, 2.0, f64_snan_pos)),
    )

    mxcsr_modes = (
        0x1f80,  # default RNE, DAZ=0, FTZ=0
        0x3f80,  # RC=1 (toward -inf)
        0x5f80,  # RC=2 (toward +inf)
        0x7f80,  # RC=3 (toward 0)
        0x1fc0,  # RNE + DAZ=1
        0x9f80,  # RNE + FTZ=1
        0x9fc0,  # RNE + DAZ=1 + FTZ=1
    )

    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    cases = 0

    def run_case(name: str, a_bytes: bytes, b_bytes: bytes, mxcsr: int, b_align: int = 0) -> None:
        nonlocal cases
        _ra, pa = _guarded_buf(a_bytes, 0)
        _rb, pb = _guarded_buf(b_bytes, b_align)
        want = ref_op(name, a_bytes, b_bytes, mxcsr)
        for dll, out_buf in ((native, out_n), (lifted, out_l)):
            fn = getattr(dll, name)
            fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            fn.restype = ctypes.c_uint64
            ctypes.memset(out_buf, 0xCC, 32)
            previous = exchange(mxcsr)
            try:
                fn(pa, pb, ctypes.addressof(out_buf))
            finally:
                exchange(previous)
        if bytes(out_n) != want or bytes(out_l) != want:
            raise AssertionError(
                f"{name} mxcsr={mxcsr:#x} align={b_align}: expected={want.hex()} "
                f"native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}"
            )
        cases += 1

    for name in EXPORTS:
        pairs = f64_pairs if "pd_" in name else f32_pairs
        for mxcsr in mxcsr_modes:
            for a_bytes, b_bytes in pairs:
                run_case(name, a_bytes, b_bytes, mxcsr, b_align=0)
        # Unaligned memory operand verification for mem exports
        if "_mem_" in name:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3):
                run_case(name, a_bytes, b_bytes, 0x1f80, b_align=off)

    # Mutated negative controls must mismatch the expected output
    def must_mismatch(ctrl_name: str, ref_name: str, a_bytes: bytes, b_bytes: bytes) -> None:
        _ra, pa = _guarded_buf(a_bytes, 0)
        _rb, pb = _guarded_buf(b_bytes, 0)
        want = ref_op(ref_name, a_bytes, b_bytes, 0x1f80)
        fn = getattr(lifted, ctrl_name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        ctypes.memset(out_l, 0xCC, 32)
        fn(pa, pb, ctypes.addressof(out_l))
        if bytes(out_l) == want:
            raise AssertionError(f"mutated control {ctrl_name} unexpectedly matched {want.hex()}")

    must_mismatch("vhaddps_mutated_control", "vhaddps_ymm_reg_op", *f32_pairs[0])
    must_mismatch("vhaddps_xmm_no_zero_hi_control", "vhaddps_xmm_reg_op", *f32_pairs[0])

    print(
        f"{cases} VEX AVX horizontal FP cases across {len(EXPORTS)} exports and {len(mxcsr_modes)} MXCSR modes "
        f"matched native DLL and IEEE-754 reference; {len(MUTATED_CONTROLS)} mutated controls mismatched"
    )


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
