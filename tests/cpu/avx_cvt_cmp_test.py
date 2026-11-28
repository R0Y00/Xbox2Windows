"""Differential check for AVX conversion, compare, flag-test, and aligned/streaming move lowering."""
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
    "vcvtdq2ps_op", "vcvtdq2pd_op", "vcvtps2dq_op", "vcvtpd2dq_op",
    "vcvttps2dq_op", "vcvttpd2dq_op", "vcvtps2pd_op", "vcvtpd2ps_op",
    "vcvtsi2ss_op", "vcvtsi2sd_op", "vcvtss2sd_op", "vcvtsd2ss_op",
    "vcmpps_op", "vcmppd_op", "vcmpss_op", "vcmpsd_op", "vlddqu_op", "vcmpps_true_signaling",
)
SCALAR_EXPORTS = ("vcvtss2si_op", "vcvttss2si_op", "vcvtsd2si_op", "vcvttsd2si_op")
FLAG_EXPORTS = ("vptest_op", "vtestps_op", "vtestpd_op")
ALL_EXPORTS = VEC_EXPORTS + SCALAR_EXPORTS + FLAG_EXPORTS

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
    # EVEX vcvtdq2ps ymm0{k1}, ymm1 (62 F1 7C 29 5B C1) followed by ret must be rejected.
    evex = list(Decoder(64, b"\x62\xf1\x7c\x29\x5b\xc1\xc3", ip=0x1000))
    try:
        lower(evex, [], {}, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected EVEX vcvtdq2ps to be rejected")
    streaming = list(Decoder(64, bytes.fromhex("c4 e2 79 2a 09 c3"), ip=0x1000))
    try:
        lower(streaming, [], {}, {}, {})
    except ValueError as error:
        assert "unsupported VMOVNTDQA memory contract" in str(error), error
    else:
        raise AssertionError("accepted an unproven streaming load")


def aligned32(data: bytes):
    raw = (ctypes.c_ubyte * 64)()
    off = (32 - (ctypes.addressof(raw) & 31)) & 31
    for i, v in enumerate(data):
        raw[off + i] = v
    return raw, ctypes.addressof(raw) + off, off


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    i32_vec = struct.pack("<8i", -2147483648, -3, 0, 2147483647, 42, -99, 1000000, -1)
    ps_a = struct.pack("<8f", 2.5, -2.5, 2147483648.0, float("nan"), -0.5, 3.5, 1.25, -4.75)
    ps_b = struct.pack("<8f", 4.0, -2.5, 1.0, 2.0, 0.0, 3.0, 2.0, -5.0)
    pd_a = struct.pack("<4d", -3.5, 2147483647.5, float("nan"), 2.5)
    pd_b = struct.pack("<4d", -3.5, 10.0, 1.0, 2.5)
    int_src = struct.pack("<4q", -123456789, 42, 0, 1)

    cases = 0
    for name in VEC_EXPORTS:
        if name in ("vcvtdq2ps_op", "vcvtdq2pd_op"):
            a, b = i32_vec, i32_vec
        elif name in ("vcvtsi2ss_op", "vcvtsi2sd_op"):
            a, b = (pd_a if "sd" in name else ps_a), int_src
        elif name in ("vcvtpd2dq_op", "vcvttpd2dq_op", "vcvtpd2ps_op", "vcmppd_op", "vcmpsd_op"):
            a, b = pd_a, pd_b
        elif name == "vcvtss2sd_op":
            a, b = pd_a, ps_b
        elif name == "vcvtsd2ss_op":
            a, b = ps_a, pd_b
        else:
            a, b = ps_a, ps_b
        ra, pa, _ = aligned32(a)
        rb, pb, _ = aligned32(b)
        rn, pn, on = aligned32(b"\xcc" * 32)
        rl, pl, ol = aligned32(b"\xcc" * 32)
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(pa, pb, pn)
        fn_l(pa, pb, pl)
        if bytes(rn[on:on + 32]) != bytes(rl[ol:ol + 32]):
            raise AssertionError(f"{name}: native={bytes(rn[on:on+32]).hex()} lifted={bytes(rl[ol:ol+32]).hex()}")
        cases += 1

    for name in SCALAR_EXPORTS:
        vec = pd_a if "sd" in name else ps_a
        ra, pa, _ = aligned32(vec)
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        got_n, got_l = fn_n(pa, 0, 0), fn_l(pa, 0, 0)
        if got_n != got_l:
            raise AssertionError(f"{name}: native={got_n:#x} lifted={got_l:#x}")
        cases += 1

    # Check all 4 (ZF, CF) combinations for vptest, vtestps, vtestpd.
    for name in FLAG_EXPORTS:
        for x, y in (
            (b"\x00" * 32, b"\x00" * 32),
            (b"\x00" * 32, b"\x80" * 32),
            (b"\x80" * 32, b"\x80" * 32),
            (b"\x80" * 16 + b"\x00" * 16, b"\x80" * 32),
        ):
            rx, px, _ = aligned32(x)
            ry, py, _ = aligned32(y)
            fn_n, fn_l = getattr(native, name), getattr(lifted, name)
            fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            fn_n.restype = fn_l.restype = ctypes.c_uint64
            got_n, got_l = fn_n(px, py, 0), fn_l(px, py, 0)
            if got_n != got_l:
                raise AssertionError(f"{name}: native={got_n} lifted={got_l}")
            cases += 1

    check_mxcsr(native, lifted)
    print(f"{cases} AVX conversion/compare/test/move cases matched native and translated DLLs")


def check_mxcsr(native, lifted):
    exchange = lifted.host_exchange_mxcsr
    exchange.argtypes = [ctypes.c_uint32]
    exchange.restype = ctypes.c_uint32

    def run(dll, name, source, mode):
        a, pa, _ = aligned32(source)
        b, pb, _ = aligned32(bytes(32))
        output, dst, offset = aligned32(bytes(32))
        saved = exchange(mode)
        try:
            value = getattr(dll, name)(pa, pb, dst)
        finally:
            status = exchange(saved)
        return value, bytes(output[offset:offset + 32]), status & 0x3f

    inputs = (
        ("vcvtps2dq_op", struct.pack("<8f", 1.75, -1.75, 0, 0, 0, 0, 0, 0)),
        ("vcvtpd2dq_op", struct.pack("<4d", 1.75, -1.75, 0, 0)),
        ("vcvtss2si_op", struct.pack("<8f", 1.75, 0, 0, 0, 0, 0, 0, 0)),
        ("vcvtsd2si_op", struct.pack("<4d", 1.75, 0, 0, 0)),
    )
    for name, source in inputs:
        for mode, expected in ((0x1f80, (2, -2)), (0x3f80, (1, -2)),
                               (0x5f80, (2, -1)), (0x7f80, (1, -1))):
            baseline = run(native, name, source, mode)
            translated = run(lifted, name, source, mode)
            assert baseline[1:] == translated[1:], (name, hex(mode), baseline, translated)
            if "2dq" in name:
                assert struct.unpack_from("<2i", baseline[1]) == expected, (name, hex(mode), baseline)
            else:
                assert baseline[0] == translated[0] == expected[0], (name, hex(mode), baseline, translated)

    qnan = struct.pack("<8I", 0x7fc01234, 0, 0, 0, 0, 0, 0, 0)
    for dll in (native, lifted):
        _, output, status = run(dll, "vcmpps_true_signaling", qnan, 0x1f80)
        assert output == bytes([255] * 32) and status & 1, (output.hex(), status)
    print("MXCSR rounding modes and signaling predicate 31 matched")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
