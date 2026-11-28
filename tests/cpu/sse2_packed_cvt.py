"""Differential check for baseline 128-bit SSE/SSE2 packed conversions, compares, half moves, and scalar sqrt/min/max."""
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
    "cvtdq2ps_op", "cvtdq2pd_op", "cvtps2dq_op", "cvtpd2dq_op",
    "cvttps2dq_op", "cvttpd2dq_op", "cvtps2pd_op", "cvtpd2ps_op",
    "cmpps_lt_op", "cmpps_unord_op", "cmppd_le_op", "cmppd_neq_op",
    "movhlps_op", "movlhps_op", "movlps_hps_op", "movlpd_hpd_op",
    "sqrtss_op", "minss_op", "maxss_op", "sqrtsd_op", "minsd_op", "maxsd_op",
)


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER]
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
    # CMPPD xmm6, xmm1, 2 is outside the shared lowerer's volatile XMM profile.
    nonvolatile = list(Decoder(64, b"\x66\x0f\xc2\xf1\x02\xc3", ip=0x1000))
    try:
        lower(nonvolatile, [], {}, {}, {})
    except ValueError as exc:
        assert "unsupported packed SSE compare destination" in str(exc), exc
    else:
        raise AssertionError("expected CMPPD writing XMM6 to be rejected")


def aligned_buf(data: bytes):
    raw = (ctypes.c_ubyte * 32)()
    off = (16 - (ctypes.addressof(raw) & 15)) & 15
    for i, v in enumerate(data):
        raw[off + i] = v
    return raw, ctypes.addressof(raw) + off, off


def run_export(lib, name, a, b):
    input_a, a_ptr, _ = aligned_buf(a)
    input_b, b_ptr, _ = aligned_buf(b)
    output, out_ptr, offset = aligned_buf(bytes(16))
    function = getattr(lib, name)
    function.argtypes = [ctypes.c_uint64] * 3
    function.restype = ctypes.c_uint64
    function(a_ptr, b_ptr, out_ptr)
    return bytes(output[offset:offset + 16])


def check_mxcsr_modes(native, lifted):
    read_mxcsr, write_mxcsr = native.read_mxcsr, native.write_mxcsr
    for function in (read_mxcsr, write_mxcsr):
        function.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        function.restype = None
    saved = ctypes.c_uint32()
    read_mxcsr(ctypes.byref(saved))
    cases = (
        (0x3f80, "cvtps2dq_op", struct.pack("<4f", 1.75, -1.75, 2.5, -2.5),
         bytes(16), struct.pack("<4i", 1, -2, 2, -3)),
        (0x7f80, "cvtpd2dq_op", struct.pack("<2d", 1.75, -1.75),
         bytes(16), struct.pack("<4i", 1, -1, 0, 0)),
        (0x1fc0, "minss_op", struct.pack("<4I", 1, 2, 3, 4),
         struct.pack("<4f", 1.0, 0.0, 0.0, 0.0), struct.pack("<4I", 0, 2, 3, 4)),
        (0x1fc0, "minsd_op", struct.pack("<2Q", 1, 2),
         struct.pack("<2d", 1.0, 0.0), struct.pack("<2Q", 0, 2)),
    )
    try:
        for config, name, a, b, expected in cases:
            for label, library in (("native", native), ("lifted", lifted)):
                mode = ctypes.c_uint32(config)
                write_mxcsr(ctypes.byref(mode))
                actual = run_export(library, name, a, b)
                if actual != expected:
                    raise AssertionError(f"{name} {label} MXCSR={config:#x}: "
                                         f"actual={actual.hex()} expected={expected.hex()}")
    finally:
        write_mxcsr(ctypes.byref(saved))


def check_unaligned_legacy_load(native, lifted):
    input_a, a_ptr, _ = aligned_buf(bytes(16))
    input_b, b_ptr, _ = aligned_buf(bytes(16))
    output, out_ptr, _ = aligned_buf(bytes(16))
    for label, library in (("native", native), ("lifted", lifted)):
        with_fault = library.cmppd_le_op
        for left, right in ((a_ptr + 1, b_ptr), (a_ptr, b_ptr + 1)):
            try:
                with_fault(left, right, out_ptr)
            except OSError as exc:
                if "access violation" not in str(exc):
                    raise
            else:
                raise AssertionError(f"{label}: misaligned MOVAPD or CMPPD memory read did not fault")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    native = ctypes.CDLL(str(native_dll))
    lifted = ctypes.CDLL(str(lifted_dll))

    i32_vec = struct.pack("<4i", -2147483648, -3, 0, 2147483647)
    # Include ties, negative fraction, out-of-range (>=2^31), and qNaN.
    ps_cvt = struct.pack("<4f", 2.5, -2.5, 2147483648.0, float("nan"))
    pd_cvt = struct.pack("<2d", -3.5, 2147483647.5)
    ps_a = struct.pack("<4f", 1.5, -2.0, float("nan"), 0.0)
    ps_b = struct.pack("<4f", 4.0, -2.0, 1.0, -0.0)
    pd_a = struct.pack("<2d", 2.25, float("nan"))
    pd_b = struct.pack("<2d", 9.0, 2.25)

    for name in EXPORTS:
        if name in ("cvtdq2ps_op", "cvtdq2pd_op"):
            a, b = i32_vec, i32_vec
        elif name in ("cvtps2dq_op", "cvttps2dq_op", "cvtps2pd_op"):
            a, b = ps_cvt, ps_b
        elif name in ("cvtpd2dq_op", "cvttpd2dq_op", "cvtpd2ps_op"):
            a, b = pd_cvt, pd_b
        elif "pd" in name or "sd" in name:
            a, b = pd_a, pd_b
        else:
            a, b = ps_a, ps_b
        got_n = run_export(native, name, a, b)
        got_l = run_export(lifted, name, a, b)
        if got_n != got_l:
            raise AssertionError(f"{name}: native={got_n.hex()} lifted={got_l.hex()}")
    print(f"{len(EXPORTS)} baseline 128-bit SSE/SSE2 conversion/compare/move/scalar cases matched native and translated DLLs")
    check_mxcsr_modes(native, lifted)
    check_unaligned_legacy_load(native, lifted)
    print("MXCSR round-down, round-toward-zero, and DAZ controls passed")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
