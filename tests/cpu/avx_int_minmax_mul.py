"""Differential check for VEX AVX packed saturating, min/max, avg, multiply, and sad lowering."""
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

EXPORTS = (
    "vpaddsb_op", "vpaddsw_op", "vpaddusb_op", "vpaddusw_op",
    "vpsubsb_op", "vpsubsw_op", "vpsubusb_op", "vpsubusw_op",
    "vpminsb_op", "vpmaxsb_op", "vpminub_op", "vpmaxub_op",
    "vpminsw_op", "vpmaxsw_op", "vpminuw_op", "vpmaxuw_op",
    "vpminsd_op", "vpmaxsd_op", "vpminud_op", "vpmaxud_op",
    "vpavgb_op", "vpavgw_op",
    "vpmulhw_op", "vpmulhuw_op", "vpmullw_op", "vpmulld_op",
    "vpmuldq_op", "vpmuludq_op", "vpmulhrsw_op", "vpmaddwd_op",
    "vpmaddubsw_op", "vpsadbw_op", "vmpsadbw_op",
)

WIDE_EXPORT = "vmpsadbw256_mem"

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    if ((_xgetbv(0) & 6u) != 6u) return 0u;
    __cpuid(info, 0);
    if (info[0] < 7) return 1u;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) ? 2u : 1u;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in (*EXPORTS, WIDE_EXPORT):
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    # EVEX vpmulld xmm0{k1}, xmm1, xmm2 (62 F2 75 09 40 C2) followed by ret must be rejected.
    evex = list(Decoder(64, b"\x62\xf2\x75\x09\x40\xc2\xc3", ip=0x1000))
    # VEX vpmulld xmm0, xmm1, xmm3 uses an untracked source.
    untracked = list(Decoder(64, bytes.fromhex("c4 e2 71 40 c3 c3"), ip=0x1000))
    for code in (evex, untracked):
        try:
            lower(code, [], {}, {}, {})
        except ValueError:
            continue
        raise AssertionError(f"expected {code[0]} to be rejected")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    capability = lifted.host_has_avx()
    if not capability:
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    a_bytes = bytes([
        0x00, 0x80, 0x00, 0x80, 0x7f, 0x7f, 0xff, 0xff,
        0x00, 0x01, 0x7f, 0x80, 0xfe, 0x55, 0xaa, 0x33,
    ]) + bytes([0xdd] * 16)
    b_bytes = bytes([
        0x00, 0x80, 0x00, 0x80, 0x7f, 0x7f, 0x01, 0x00,
        0xff, 0x02, 0x01, 0x80, 0x03, 0xaa, 0x55, 0xcc,
    ]) + bytes([0xee] * 16)
    a_buf = (ctypes.c_ubyte * 32)(*a_bytes)
    b_buf = (ctypes.c_ubyte * 32)(*b_bytes)
    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    for name in EXPORTS:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        if name == "vpmaddwd_op":
            assert int.from_bytes(bytes(out_n)[:4], "little") == 0x80000000
    print(f"{len(EXPORTS)} VEX.128 cases matched native and translated DLLs")
    if capability < 2:
        print("host lacks AVX2; VEX.256 SAD not exercised")
        raise SystemExit(77)

    a = [(i * i + 3 * i + 5) & 255 for i in range(32)]
    b = [(17 * i + 11 * (i // 16) + 93) & 255 for i in range(32)]
    # Imm 0x15 uses left/right starts (4, 4) and (0, 8) for the two lanes.
    expected = []
    for lane, left, right in ((0, 4, 4), (16, 0, 8)):
        for i in range(8):
            expected.append(sum(abs(a[lane + left + i + k] - b[lane + right + k]) for k in range(4)))
    a_ptr, b_ptr = (ctypes.c_ubyte * 32)(*a), (ctypes.c_ubyte * 32)(*b)
    for library in (native, lifted):
        func = getattr(library, WIDE_EXPORT)
        func.argtypes = [ctypes.c_uint64] * 3
        func.restype = ctypes.c_uint64
        out = (ctypes.c_uint16 * 16)(*([0xffff] * 16))
        func(ctypes.addressof(a_ptr), ctypes.addressof(b_ptr), ctypes.addressof(out))
        assert list(out) == expected, (list(out), expected)
    print("VEX.256 memory MPSADBW matched both separately selected lanes")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
