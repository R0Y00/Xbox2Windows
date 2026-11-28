"""Differential check for VEX AVX sign/zero-extend and horizontal integer lowering."""
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
    "vpmovsxbw_op", "vpmovsxbd_op", "vpmovsxbq_op", "vpmovsxwd_op", "vpmovsxwq_op", "vpmovsxdq_op",
    "vpmovzxbw_op", "vpmovzxbd_op", "vpmovzxbq_op", "vpmovzxwd_op", "vpmovzxwq_op", "vpmovzxdq_op",
    "vphaddw_op", "vphaddd_op", "vphaddsw_op", "vphsubw_op", "vphsubd_op", "vphsubsw_op", "vphminposuw_op",
)
AVX2_EXPORTS = ("vpmovsxbd_ymm_mem_op", "vphaddd_ymm_mem_op")

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
extern "C" __declspec(dllexport) uint32_t host_has_avx2() {
    if (!host_has_avx()) return 0u;
    int info[4]{};
    __cpuid(info, 0);
    if (info[0] < 7) return 0u;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in EXPORTS + AVX2_EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    for encoded, description in (
        ("62f27d0920c1", "EVEX vpmovsxbw xmm0{k1}, xmm1"),
        ("c4e27920c3", "VEX vpmovsxbw xmm0, xmm3"),
    ):
        body = list(Decoder(64, bytes.fromhex(encoded + "c3"), ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except ValueError as error:
            if "unsupported packed extension or horizontal operation" not in str(error):
                raise
        else:
            raise AssertionError(f"expected {description} to be rejected")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    a_bytes = bytes([
        0x80, 0x7f, 0xff, 0x00, 0x05, 0x00, 0x05, 0x00,
        0x00, 0x80, 0x00, 0x80, 0x7f, 0x7f, 0x7f, 0x7f,
    ]) + bytes([0xbb] * 16)
    b_bytes = bytes([
        0x01, 0x80, 0xfe, 0x7f, 0x00, 0x80, 0xff, 0x7f,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    ]) + bytes([0xcc] * 16)
    a_buf = (ctypes.c_ubyte * 32)(*a_bytes)
    b_buf = (ctypes.c_ubyte * 32)(*b_bytes)
    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()

    def compare(name, expected=None):
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        if expected is not None and bytes(out_l) != expected:
            raise AssertionError(f"{name}: lifted={bytes(out_l).hex()} expected={expected.hex()}")

    for name in EXPORTS:
        compare(name)
    print(f"{len(EXPORTS)} VEX AVX extend/horizontal/minpos cases matched native and translated DLLs")
    lifted.host_has_avx2.restype = ctypes.c_uint32
    if not lifted.host_has_avx2():
        print("host lacks AVX2; skipping the two YMM memory cases")
        return
    compare(AVX2_EXPORTS[0], struct.pack("<8i", *struct.unpack("<8b", a_bytes[:8])))
    lhs, rhs = struct.unpack("<8I", a_bytes), struct.unpack("<8I", b_bytes)
    pairs = [(source[lane + i] + source[lane + i + 1]) & 0xffffffff
             for lane in (0, 4) for source in (lhs, rhs) for i in (0, 2)]
    compare(AVX2_EXPORTS[1], struct.pack("<8I", *pairs))
    print("AVX2 YMM memory extension and horizontal add matched independent lane references")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
