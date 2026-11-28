"""Differential check for VEX AVX scalar FP, half-register move, extract/insert, and compare lowering."""
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

VEC_EXPORTS = (
    "vaddss_op", "vsubss_op", "vmulss_op", "vdivss_op", "vsqrtss_op", "vminss_op", "vmaxss_op", "vroundss_op",
    "vaddsd_op", "vsubsd_op", "vmulsd_op", "vdivsd_op", "vsqrtsd_op", "vminsd_op", "vmaxsd_op", "vroundsd_op",
    "vmovss_op", "vmovhlps_op", "vmovlhps_op", "vmovlps_hps_op", "vmovlpd_op", "vinsertps_op",
)
CMP_EXPORTS = ("vcomiss_op", "vucomiss_op", "vcomisd_op", "vucomisd_op")
ALL_EXPORTS = VEC_EXPORTS + ("vextractps_op",) + CMP_EXPORTS

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
        for name in ALL_EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    # EVEX vaddss xmm0{k1}, xmm1, xmm2 (62 F1 76 09 58 C2) followed by ret must be rejected.
    evex = list(Decoder(64, b"\x62\xf1\x76\x09\x58\xc2\xc3", ip=0x1000))
    try:
        lower(evex, [], {}, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected EVEX vaddss to be rejected")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    ps_a = struct.pack("<8f", 2.5, -1.25, 3.5, 4.0, 9.0, 8.0, 7.0, 6.0)
    ps_b = struct.pack("<8f", 9.0, 2.0, -5.0, 1.5, 1.0, 2.0, 3.0, 4.0)
    pd_a = struct.pack("<4d", -3.5, 12.25, 9.0, 8.0)
    pd_b = struct.pack("<4d", 20.25, -4.5, 1.0, 2.0)

    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    cases = 0
    for name in VEC_EXPORTS:
        is_pd = "sd" in name or "pd" in name
        a, b = (pd_a, pd_b) if is_pd else (ps_a, ps_b)
        a_buf = (ctypes.c_ubyte * 32)(*a)
        b_buf = (ctypes.c_ubyte * 32)(*b)
        ctypes.memset(out_n, 0xCC, 32)
        ctypes.memset(out_l, 0xCC, 32)
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    # vextractps_op
    a_buf = (ctypes.c_ubyte * 32)(*ps_a)
    native.vextractps_op.argtypes = lifted.vextractps_op.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    native.vextractps_op.restype = lifted.vextractps_op.restype = ctypes.c_uint64
    if native.vextractps_op(ctypes.addressof(a_buf), 0, 0) != lifted.vextractps_op(ctypes.addressof(a_buf), 0, 0):
        raise AssertionError("vextractps_op mismatch")
    cases += 1

    # All 4 ordering outcomes (<, ==, >, unordered) for vcomiss/vucomiss/vcomisd/vucomisd.
    for name in CMP_EXPORTS:
        fmt = "<4d" if "sd" in name else "<8f"
        pad = (0.0,) * (3 if "sd" in name else 7)
        for x, y in ((1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (float("nan"), 1.0)):
            xb = (ctypes.c_ubyte * 32)(*struct.pack(fmt, x, *pad))
            yb = (ctypes.c_ubyte * 32)(*struct.pack(fmt, y, *pad))
            fn_n, fn_l = getattr(native, name), getattr(lifted, name)
            fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            fn_n.restype = fn_l.restype = ctypes.c_uint64
            got_n, got_l = fn_n(ctypes.addressof(xb), ctypes.addressof(yb), 0), fn_l(ctypes.addressof(xb), ctypes.addressof(yb), 0)
            if got_n != got_l:
                raise AssertionError(f"{name}({x}, {y}): native={got_n} lifted={got_l}")
            cases += 1

    check_mxcsr(native, lifted)
    print(f"{cases} VEX AVX scalar FP/move/extract/insert/compare cases matched native and translated DLLs")


def check_mxcsr(native, lifted):
    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes = [ctypes.c_uint32]
    exchange.restype = ctypes.c_uint32

    def invoke(dll, name, fmt, x, y, mode):
        a = ctypes.create_string_buffer(struct.pack(fmt, x), 32)
        b = ctypes.create_string_buffer(struct.pack(fmt, y), 32)
        dst = ctypes.create_string_buffer(32)
        saved = exchange(mode)
        try:
            result = getattr(dll, name)(ctypes.addressof(a), ctypes.addressof(b), ctypes.addressof(dst))
        finally:
            status = exchange(saved) & 0x3f
        return result, dst.raw, status

    for suffix, fmt, signaling_nan, sign, quiet in (
        ("ss", "<I", 0x7f801234, 0x80000000, 0x00400000),
        ("sd", "<Q", 0x7ff0000000001234, 0x8000000000000000, 0x0008000000000000),
    ):
        round_name = f"vround{suffix}_op"
        for name, x, y, mode, expected in (
            (round_name, 0, signaling_nan, 0x1f80, signaling_nan | quiet),
            (f"vmin{suffix}_op", 0, sign | 1, 0x1f80, sign | 1),
            (f"vmin{suffix}_op", 0, sign | 1, 0x1fc0, sign),
            (f"vmax{suffix}_op", 0, 1, 0x1f80, 1),
            (f"vmax{suffix}_op", 0, 1, 0x1fc0, 0),
        ):
            actual = [invoke(dll, name, fmt, x, y, mode) for dll in (native, lifted)]
            assert actual[0][1] == actual[1][1], (name, hex(mode), actual)
            for _, output, _ in actual:
                assert struct.unpack_from(fmt, output)[0] == expected, (name, hex(mode), output.hex())
            if name == round_name:
                assert all(status & 1 for _, _, status in actual), (name, actual)

        for stem in ("vcomi", "vucomi"):
            name = f"{stem}{suffix}_op"
            for nan, invalid in ((signaling_nan, True), (signaling_nan | quiet, stem == "vcomi")):
                actual = [invoke(dll, name, fmt, nan, 0, 0x1f80) for dll in (native, lifted)]
                assert [(value, bool(status & 1)) for value, _, status in actual] == [(3, invalid)] * 2, (name, actual)

    for mode, expected in ((0x1f80, 0xbff0000000000000), (0x1fc0, 0x8000000000000000)):
        for dll in (native, lifted):
            _, output, _ = invoke(dll, "vroundsd_op", "<Q", 0, 0x8000000000000001, mode)
            assert struct.unpack_from("<Q", output)[0] == expected, (hex(mode), output.hex())
    print("scalar rounding quiets signaling NaNs, DAZ affects MIN/MAX and round-down, and COMI/UCOMI invalid flags match")



if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
