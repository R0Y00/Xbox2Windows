"""Differential check for AVX F16C, masked FP load/store, and non-temporal store lowering."""
import ctypes
import struct
import sys
from pathlib import Path

import pefile
from iced_x86 import Decoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

EXPORTS = (
    "vcvtph2ps_op", "vcvtps2ph_rn_op", "vcvtps2ph_floor_op",
    "vmaskmovps_load_op", "vmaskmovps_store_op", "vmaskmovpd_load_op", "vmaskmovpd_store_op",
    "vmovntps_op", "vmovntpd_op", "vmovntdq_op",
)

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx_f16c() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28) | (1 << 29);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    # vcvtps2ph xmm0, ymm1, 0x04 (MXCSR rounding mode bit 2 set) followed by ret must be rejected.
    mxcsr = list(Decoder(64, b"\xc4\xe3\x7d\x1d\xc8\x04\xc3", ip=0x1000))
    try:
        lower(mxcsr, [], {}, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected MXCSR-mode vcvtps2ph to be rejected")


def aligned32(data: bytes):
    raw = (ctypes.c_ubyte * 64)()
    off = (32 - (ctypes.addressof(raw) & 31)) & 31
    for i, v in enumerate(data):
        raw[off + i] = v
    return raw, ctypes.addressof(raw) + off, off


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx_f16c.restype = ctypes.c_uint32
    if not lifted.host_has_avx_f16c():
        print("host lacks AVX/F16C; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    # Binary16 test lanes: +0.0, -0.0, 1.0, -2.5, subnormal 0x0001, max 0x7bff, +Inf, qNaN
    f16_in = struct.pack("<8H", 0x0000, 0x8000, 0x3c00, 0xc100, 0x0001, 0x7bff, 0x7c00, 0x7e23) + b"\x00" * 16
    # Binary32 test lanes: +0.0, -0.0, 1.5, -2.25, subnormal target 2^-20, overflow 1e5, +Inf, qNaN
    f32_in = struct.pack("<8f", 0.0, -0.0, 1.5, -2.25, 2.0**-20, 100000.0, float("inf"), float("nan"))
    mask_ps = struct.pack("<8I", 0x80000000, 0, 0xffffffff, 0x7fffffff, 0, 0x80000001, 0x80000000, 0)
    mask_pd = struct.pack("<4Q", 0x8000000000000000, 0, 0xffffffffffffffff, 0x7fffffffffffffff)

    for name in EXPORTS:
        if name == "vcvtph2ps_op":
            a, b = f16_in, mask_ps
        elif "pd" in name:
            a, b = f32_in, mask_pd
        else:
            a, b = f32_in, mask_ps
        ra, pa, _ = aligned32(a)
        rb, pb, _ = aligned32(b)
        # Pre-fill output with sentinel 0xCC so masked stores prove unmasked lanes are untouched!
        rn, pn, on = aligned32(b"\xcc" * 32)
        rl, pl, ol = aligned32(b"\xcc" * 32)
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(pa, pb, pn)
        fn_l(pa, pb, pl)
        if bytes(rn[on:on + 32]) != bytes(rl[ol:ol + 32]):
            raise AssertionError(f"{name}: native={bytes(rn[on:on+32]).hex()} lifted={bytes(rl[ol:ol+32]).hex()}")
    print(f"{len(EXPORTS)} AVX F16C/maskmov/movnt cases matched native and translated DLLs")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
