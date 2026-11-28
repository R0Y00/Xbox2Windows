"""Differential check for VEX AVX alternating FP add/subtract lowering (vaddsubps, vaddsubpd).

Even-indexed elements subtract (a - b) and odd-indexed elements add (a + b) in every 128-bit lane.
The IEEE-754 reference model, guarded buffers and host probes are shared with avx_horiz_fp.py.
"""
import ctypes
import re
import struct
import sys
from pathlib import Path

from iced_x86 import Decoder
import pefile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region
from avx_horiz_fp import CPUID_HELPER, F32, F64, _guarded_buf

OPS = ("vaddsubps", "vaddsubpd")
EXPORTS = tuple(
    f"{op}_{width}_{kind}_op" for width in ("xmm", "ymm") for kind in ("reg", "mem") for op in OPS
) + (
    "vaddsubps_ymm_alias_dst_src1_op",
    "vaddsubpd_ymm_alias_dst_src2_op",
    "vaddsubps_ymm_alias_src1_src2_op",
    "vaddsubpd_ymm_alias_all_op",
    "vaddsubps_xmm_alias_dst_src2_op",
    "vaddsubpd_xmm_alias_dst_src1_op",
)
# Exports whose second source is the first source register.
SELF_ALIASED = ("vaddsubps_ymm_alias_src1_src2_op", "vaddsubpd_ymm_alias_all_op")

MUTATED_CONTROLS = (
    # Same op in every element: catches a lowering that ignores the alternating signs.
    ("vaddsubps_ymm_reg_op", "vaddsubps_all_add_control", re.compile(r"_mm_addsub_ps\("), "_mm_add_ps("),
    ("vaddsubpd_ymm_reg_op", "vaddsubpd_all_sub_control", re.compile(r"_mm_addsub_pd\("), "_mm_sub_pd("),
    # VEX.128 must zero bits 255:128 of the destination.
    ("vaddsubpd_xmm_reg_op", "vaddsubpd_xmm_no_zero_hi_control",
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


def check_controls() -> None:
    rejected_bytes = (
        ("c5f3d0da", "VEX vaddsubps xmm3, xmm1, xmm2 (untracked dst)"),
        ("c5e3d0c2", "VEX vaddsubps xmm0, xmm3, xmm2 (untracked src1)"),
        ("c5f1d0c3", "VEX vaddsubpd xmm0, xmm1, xmm3 (untracked src2)"),
        ("c5f7d0da", "VEX vaddsubps ymm3, ymm1, ymm2 (untracked dst)"),
        ("c5e5d0c2", "VEX vaddsubpd ymm0, ymm3, ymm2 (untracked src1)"),
        ("c5f5d01a", "VEX vaddsubpd ymm3, ymm1, [rdx] (untracked dst with mem)"),
        ("64c5f7d002", "VEX vaddsubps ymm0, ymm1, fs:[rdx] (segment prefix)"),
        ("67c5f3d00500000000", "VEX vaddsubps xmm0, xmm1, [eip] (EIP-relative)"),
        ("f20fd0c3", "Legacy addsubps xmm0, xmm3 (untracked src)"),
    )
    for encoded, description in rejected_bytes:
        body = list(Decoder(64, bytes.fromhex(encoded + "c3"), ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected {description} to be rejected")


def _lane_addsub(fmt, count: int, word: str, lane_a: bytes, lane_b: bytes, mxcsr: int) -> bytes:
    xa = struct.unpack(f"<{count}{word}", lane_a)
    xb = struct.unpack(f"<{count}{word}", lane_b)
    return struct.pack(
        f"<{count}{word}",
        *((fmt.sub if k % 2 == 0 else fmt.add)(xa[k], xb[k], mxcsr) for k in range(count)),
    )


def ref_op(name: str, a: bytes, b: bytes, mxcsr: int) -> bytes:
    if name in SELF_ALIASED:
        b = a
    is_pd = name.startswith("vaddsubpd")
    fmt, count, word = (F64, 2, "Q") if is_pd else (F32, 4, "I")
    lo = _lane_addsub(fmt, count, word, a[0:16], b[0:16], mxcsr)
    if "_xmm_" in name:
        return lo + b"\x00" * 16
    return lo + _lane_addsub(fmt, count, word, a[16:32], b[16:32], mxcsr)


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

    # Every pair uses distinct values per element and per lane, so swapped elements,
    # swapped add/sub positions or a dropped upper lane change the result.
    f32_pairs = (
        # Ordinary values, rounding ties and subnormals in both lanes
        (_f32(1.5, -2.25, 1.0, 2.0 ** -24, 3.5, 0.125, -1.0, -(2.0 ** -24)),
         _f32(4.25, 0.75, 2.0 ** -24, f32_sub_neg, -7.5, 2.5, f32_sub_pos, f32_min_norm)),
        # Signed zeros: x - x and x + (-x) cancel in the add and the subtract positions
        (_f32(0x00000000, 0x80000000, 0x80000000, 0x00000000, 1.5, -1.5, f32_one_half_min_norm, f32_min_norm),
         _f32(0x00000000, 0x00000000, 0x80000000, 0x80000000, 1.5, 1.5, f32_min_norm, f32_min_norm | 0x80000000)),
        # Infinities (inf - inf is invalid only in the subtract position), overflow, FTZ boundary
        (_f32(0x7f800000, 0x7f800000, 0xff800000, 0xff800000, f32_max_fin, f32_max_fin, f32_one_half_min_norm, f32_sub_pos),
         _f32(0x7f800000, 0xff800000, 0x7f800000, 0xff800000, f32_max_fin | 0x80000000, f32_max_fin, f32_min_norm, f32_sub_pos)),
        # QNaN and SNaN payloads in either operand and NaN-vs-NaN order
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
        (_f64(1.5, 2.0 ** -53, -3.5, f64_one_half_min_norm),
         _f64(2.0 ** -53, f64_sub_neg, f64_sub_pos, f64_min_norm)),
        (_f64(0x0000000000000000, 0x8000000000000000, 2.25, -2.25),
         _f64(0x0000000000000000, 0x0000000000000000, 2.25, 2.25)),
        (_f64(0x7ff0000000000000, 0x7ff0000000000000, f64_max_fin, f64_max_fin),
         _f64(0x7ff0000000000000, 0xfff0000000000000, f64_max_fin | 0x8000000000000000, f64_max_fin)),
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
        pairs = f64_pairs if name.startswith("vaddsubpd") else f32_pairs
        for mxcsr in mxcsr_modes:
            for a_bytes, b_bytes in pairs:
                run_case(name, a_bytes, b_bytes, mxcsr)
        if "_mem_" in name:
            a_bytes, b_bytes = pairs[0]
            for off in (1, 3):
                run_case(name, a_bytes, b_bytes, 0x1f80, b_align=off)

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

    for src_name, ctrl_name, _pattern, _replacement in MUTATED_CONTROLS:
        pairs = f64_pairs if src_name.startswith("vaddsubpd") else f32_pairs
        must_mismatch(ctrl_name, src_name, *pairs[0])

    print(
        f"{cases} VEX AVX alternating FP add/subtract cases across {len(EXPORTS)} exports and "
        f"{len(mxcsr_modes)} MXCSR modes matched native DLL and IEEE-754 reference; "
        f"{len(MUTATED_CONTROLS)} mutated controls mismatched"
    )


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
