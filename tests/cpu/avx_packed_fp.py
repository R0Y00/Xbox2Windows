"""Native and translated VEX.256 float32 minimum and maximum, with and without DAZ."""
import ctypes
import sys
from pathlib import Path

import pefile
from iced_x86 import Decoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

EXPORTS = ("vminps_op", "vmaxps_op")

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
    candidates = (
        bytes.fromhex("62 f1 74 29 5d c2 c3"),  # EVEX vminps ymm0{k1}, ymm1, ymm2
        bytes.fromhex("c5 f4 5d 02 c3"),        # VEX vminps ymm0, ymm1, ymmword ptr [rdx]
        bytes.fromhex("c5 f4 5d c3 c3"),        # VEX vminps ymm0, ymm1, ymm3
    )
    for encoded in candidates:
        body = list(Decoder(64, encoded, ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected {body[0]} to be rejected")


def minimum_or_maximum(a, b, *, maximum, daz):
    if daz:
        if a & 0x7fffffff < 0x00800000:
            a &= 0x80000000
        if b & 0x7fffffff < 0x00800000:
            b &= 0x80000000
    if (a & 0x7fffffff > 0x7f800000 or b & 0x7fffffff > 0x7f800000
            or (a | b) & 0x7fffffff == 0):
        return b
    a_key = ~a if a & 0x80000000 else a
    b_key = ~b if b & 0x80000000 else b
    chosen = a_key > b_key if maximum else a_key < b_key
    return a if chosen else b


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
    words = ctypes.c_uint32 * 8
    inputs = (
        (0, 0, 0x80000000, 0x80000000, 0x7fc01234, 0x3f800000, 0xffc07654, 0x80000001),
        (1, 0x80000001, 1, 0x80000001, 0x3f800000, 0x7fc05678, 0x7fc04321, 0),
    )
    count = 0
    for name in EXPORTS:
        maximum = name == "vmaxps_op"
        for mode in (0x1f80, 0x1fc0):
            daz = bool(mode & 0x40)
            for a, b in (inputs, inputs[::-1]):
                expected = tuple(minimum_or_maximum(x, y, maximum=maximum, daz=daz)
                                 for x, y in zip(a, b, strict=True))
                a_ptr, b_ptr = words(*a), words(*b)
                for library in (native, lifted):
                    func = getattr(library, name)
                    func.argtypes = [ctypes.c_uint64] * 3
                    func.restype = ctypes.c_uint64
                    output = words(*([0xcccccccc] * 8))
                    previous = exchange(mode)
                    try:
                        func(ctypes.addressof(a_ptr), ctypes.addressof(b_ptr), ctypes.addressof(output))
                    finally:
                        exchange(previous)
                    if tuple(output) != expected:
                        raise AssertionError(f"{name} MXCSR={mode:#x} {library._name}: "
                                             f"{tuple(output)} != {expected}")
                count += len(expected)
    assert minimum_or_maximum(0, 1, maximum=False, daz=True) != 1  # wrong raw-second source
    print(f"{count} VEX.256 MINPS/MAXPS words matched native, translated and reference results")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
