"""Compare native and translated packed and scalar SSE/SSE2/AVX float conversion instructions.

MXCSR Configuration Established:
This differential fixture establishes and assumes the default x64 architectural MXCSR state
(0x1F80, standard Windows ABI environment on thread creation):
  - Exception Masks (bits 7-12: PE, UE, OE, ZE, DE, IE) = 111111b (all floating-point
    exceptions masked; operations on NaNs, overflows, underflows, and divisions by zero
    produce IEEE-754 quiet NaNs, infinities, or subnormals without trapping).
  - Rounding Control (bits 13-14: RC) = 00b (Round to nearest / even).
  - Flush-to-Zero (bit 15: FTZ) = 0 (disabled; subnormals preserved in full precision).
  - Denormals-Are-Zeros (bit 6: DAZ) = 0 (disabled; subnormal source operands preserved).
"""
import ctypes
from pathlib import Path
import struct
import sys

from iced_x86 import Decoder
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64

EXPORTS = (
    # Packed XMM-to-XMM
    "cvtdq2ps", "vcvtdq2ps",
    "cvtdq2pd", "vcvtdq2pd",
    "cvtps2dq", "vcvtps2dq",
    "cvttps2dq", "vcvttps2dq",
    "cvtps2pd", "vcvtps2pd",
    "cvtpd2ps", "vcvtpd2ps",
    "cvtpd2dq", "vcvtpd2dq",
    "cvttpd2dq", "vcvttpd2dq",
    # Scalar XMM-to-XMM
    "cvtss2sd", "vcvtss2sd",
    "cvtsd2ss", "vcvtsd2ss",
    # Scalar GPR-to-XMM
    "cvtsi2ss", "cvtsi2ss_qword", "vcvtsi2ss", "vcvtsi2ss_qword",
    "cvtsi2sd", "cvtsi2sd_qword", "vcvtsi2sd", "vcvtsi2sd_qword",
    # Scalar XMM-to-GPR
    "cvtss2si", "cvtss2si_qword", "vcvtss2si", "vcvtss2si_qword",
    "cvttss2si", "cvttss2si_qword", "vcvttss2si", "vcvttss2si_qword",
    "cvtsd2si", "cvtsd2si_qword", "vcvtsd2si", "vcvtsd2si_qword",
    "cvttsd2si", "cvttsd2si_qword", "vcvttsd2si", "vcvttsd2si_qword",
    # F16C Half-Precision
    "vcvtph2ps", "vcvtps2ph",
)
EXPORT_NAMES = tuple(f"{op}_op" for op in EXPORTS)

F16C_EXPORTS = {
    "vcvtph2ps_op", "vcvtps2ph_op",
}

GPR_DEST_EXPORTS = {
    "cvtss2si_op", "cvtss2si_qword_op", "vcvtss2si_op", "vcvtss2si_qword_op",
    "cvttss2si_op", "cvttss2si_qword_op", "vcvttss2si_op", "vcvttss2si_qword_op",
    "cvtsd2si_op", "cvtsd2si_qword_op", "vcvtsd2si_op", "vcvtsd2si_qword_op",
    "cvttsd2si_op", "cvttsd2si_qword_op", "vcvttsd2si_op", "vcvttsd2si_qword_op",
}

GPR_SRC_EXPORTS = {
    "cvtsi2ss_op", "cvtsi2ss_qword_op", "vcvtsi2ss_op", "vcvtsi2ss_qword_op",
    "cvtsi2sd_op", "cvtsi2sd_qword_op", "vcvtsi2sd_op", "vcvtsi2sd_qword_op",
}


def emit(image, output):
    parts = [
        INTEGER_HEADER,
        'extern "C" __declspec(dllexport) int host_has_avx() {\n'
        "    int regs[4]{};\n"
        "    __cpuid(regs, 1);\n"
        "    if ((regs[2] & (1 << 27)) == 0 || (regs[2] & (1 << 28)) == 0) return 0;\n"
        "    return (_xgetbv(0) & 6ull) == 6ull;\n"
        "}\n",
        'extern "C" __declspec(dllexport) int host_has_f16c() {\n'
        "    int regs[4]{};\n"
        "    __cpuid(regs, 1);\n"
        "    if ((regs[2] & (1 << 27)) == 0 || (regs[2] & (1 << 28)) == 0 || (regs[2] & (1 << 29)) == 0) return 0;\n"
        "    return (_xgetbv(0) & 6ull) == 6ull;\n"
        "}\n",
    ]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORT_NAMES:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word dst) {{\n{code}\nreturn rax;\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORT_NAMES:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def aligned_buffer(data_bytes):
    raw = ctypes.create_string_buffer(len(data_bytes) + 32)
    addr = ctypes.addressof(raw)
    off = (16 - (addr % 16)) % 16
    ptr = addr + off
    ctypes.memmove(ptr, data_bytes, len(data_bytes))
    return raw, ptr


def check(native_path, lifted_path):
    # Keep EVEX forms, legacy memory operands, untracked registers, and unsupported rounding immediates refused.
    unsupported_hex = (
        "62 f1 7c 29 5b c1 c3",  # EVEX vcvtdq2ps ymm0{k1}, ymm1; ret
        "0f 5b 01",        # cvtdq2ps xmm0, [rcx] (memory operand)
        "0f 5b c3",        # cvtdq2ps xmm0, xmm3 (untracked XMM register operand 1)
        "0f 5b d8",        # cvtdq2ps xmm3, xmm0 (untracked XMM register destination)
        "f3 0f 2d 01",     # cvtss2si eax, dword ptr [rcx] (memory operand on GPR dest)
        "c4 e3 79 1d c8 01",  # vcvtps2ph xmm0, xmm1, 1 (non-zero rounding mode immediate rejected in first profile)
    )
    for raw in unsupported_hex:
        try:
            lower(list(Decoder(64, bytes.fromhex(raw))), [], {}, {}, {})
        except ValueError as error:
            assert "unsupported" in str(error), error
        else:
            raise AssertionError(f"accepted unsupported instruction {raw}")

    lifted = load_lib(lifted_path)
    lifted.host_has_avx.argtypes, lifted.host_has_avx.restype = [], ctypes.c_int
    lifted.host_has_f16c.argtypes, lifted.host_has_f16c.restype = [], ctypes.c_int
    if not lifted.host_has_avx():
        print("Host CPU/OS lacks AVX (CPUID leaf 1 ECX bit 28 / XCR0 YMM); skipping native differential check.")
        sys.exit(77)

    has_f16c = bool(lifted.host_has_f16c())
    if not has_f16c:
        print("Host CPU lacks F16C (CPUID leaf 1 ECX bit 29); gating F16C exports, testing other conversions.")

    native = load_lib(native_path)

    sentinel_a = b"\x5a" * 16
    sentinel_b = b"\xa5" * 16

    # Test patterns for XMM vectors (32 bytes = 16 bytes payload + 16 bytes sentinel)
    # Floats with distinct rounding vs truncation targets
    p_f1 = struct.pack("<4f", 1.25, 2.75, -1.25, -2.75) + sentinel_a
    p_f2 = struct.pack("<4f", 0.0, -0.0, 100.5, -100.5) + sentinel_b
    p_f3 = struct.pack("<4f", 1e20, -1e20, 1e-20, -1e-20) + sentinel_a
    # Integers packed into XMM (for cvtdq2ps / cvtdq2pd)
    p_i1 = struct.pack("<4i", 1, -2, 1000000, -1000000) + sentinel_a
    p_i2 = struct.pack("<4i", 0, 2147483647, -2147483648, 42) + sentinel_b
    # Doubles (for cvtpd2ps, cvtpd2dq, cvtsd2ss, cvtsd2si, etc.)
    p_d1 = struct.pack("<2d", 2.25, -7.75) + sentinel_a
    p_d2 = struct.pack("<2d", 0.0, 1e12) + sentinel_b
    p_d3 = struct.pack("<2d", 1e20, -1e20) + sentinel_a
    # Half-precision F16C bit patterns (stored in low 64 bits)
    # 0x3c00 = 1.0, 0x4000 = 2.0, 0xc000 = -2.0, 0x3555 = ~0.333
    p_h1 = struct.pack("<4H4H", 0x3c00, 0x4000, 0xc000, 0x3555, 0, 0, 0, 0) + sentinel_a

    patterns = [
        (p_f1, p_i1),
        (p_f2, p_i2),
        (p_f3, p_i1),
        (p_d1, p_d2),
        (p_d3, p_d1),
        (p_h1, p_f1),
    ]

    # Scalar GPR test integers (passed in RDX)
    test_integers = [
        0, 1, -1, 42, -42, 1000000, -1000000,
        0x7fffffff, -0x80000000,
        0x7fffffffffffffff, -0x8000000000000000,
    ]

    comparisons = 0
    # 1. Test XMM-destination and GPR-destination vector conversions
    for a_bytes, b_bytes in patterns:
        raw_a, ptr_a = aligned_buffer(a_bytes)
        raw_b, ptr_b = aligned_buffer(b_bytes)
        raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
        raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)

        for name in EXPORT_NAMES:
            if name in GPR_SRC_EXPORTS:
                continue
            if name in F16C_EXPORTS and not has_f16c:
                continue
            ctypes.memset(ptr_out_n, 0xaa, 32)
            ctypes.memset(ptr_out_l, 0x55, 32)

            res_n = getattr(native, name)(ptr_a, ptr_b, ptr_out_n)
            res_l = getattr(lifted, name)(ptr_a, ptr_b, ptr_out_l)

            if name in GPR_DEST_EXPORTS:
                if res_n != res_l:
                    raise AssertionError(f"{name}: returned GPR mismatch native {hex(res_n)} != lifted {hex(res_l)}")
            else:
                out_n = ctypes.string_at(ptr_out_n, 32)
                out_l = ctypes.string_at(ptr_out_l, 32)
                if out_n != out_l:
                    raise AssertionError(f"{name}: full 256-bit vector mismatch native {out_n.hex()} != lifted {out_l.hex()}")

                # Verify architectural high-half behavior: legacy XMM preserves, VEX.128 clears
                if name.startswith("v"):
                    if out_l[16:32] != b"\x00" * 16:
                        raise AssertionError(f"{name}: VEX.128 write did not clear high vector bits: {out_l[16:32].hex()}")
                else:
                    if out_l[16:32] != a_bytes[16:32]:
                        raise AssertionError(f"{name}: legacy XMM write did not preserve high vector bits: {out_l[16:32].hex()}")

            comparisons += 1

    # 2. Test GPR-to-XMM scalar conversions with diverse integer values
    for val in test_integers:
        u_val = ctypes.c_uint64(val).value
        raw_a, ptr_a = aligned_buffer(p_f1)
        raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
        raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)

        for name in GPR_SRC_EXPORTS:
            ctypes.memset(ptr_out_n, 0xaa, 32)
            ctypes.memset(ptr_out_l, 0x55, 32)

            getattr(native, name)(ptr_a, u_val, ptr_out_n)
            getattr(lifted, name)(ptr_a, u_val, ptr_out_l)

            out_n = ctypes.string_at(ptr_out_n, 32)
            out_l = ctypes.string_at(ptr_out_l, 32)
            if out_n != out_l:
                raise AssertionError(f"{name} with int {val}: full 256-bit vector mismatch native {out_n.hex()} != lifted {out_l.hex()}")

            if name.startswith("v"):
                if out_l[16:32] != b"\x00" * 16:
                    raise AssertionError(f"{name}: VEX.128 did not clear high vector bits: {out_l[16:32].hex()}")
            else:
                if out_l[16:32] != p_f1[16:32]:
                    raise AssertionError(f"{name}: legacy XMM did not preserve high vector bits: {out_l[16:32].hex()}")

            comparisons += 1

    # 3. Aliased operand testing: same buffer passed as both source operands
    raw_aliased, ptr_aliased = aligned_buffer(p_f1)
    raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
    raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)
    for name in EXPORT_NAMES:
        if name in GPR_SRC_EXPORTS:
            continue
        if name in F16C_EXPORTS and not has_f16c:
            continue
        ctypes.memset(ptr_out_n, 0xaa, 32)
        ctypes.memset(ptr_out_l, 0x55, 32)
        res_n = getattr(native, name)(ptr_aliased, ptr_aliased, ptr_out_n)
        res_l = getattr(lifted, name)(ptr_aliased, ptr_aliased, ptr_out_l)
        if name in GPR_DEST_EXPORTS:
            if res_n != res_l:
                raise AssertionError(f"{name}: aliased GPR mismatch")
        else:
            if ctypes.string_at(ptr_out_n, 32) != ctypes.string_at(ptr_out_l, 32):
                raise AssertionError(f"{name}: aliased 256-bit vector mismatch")
        comparisons += 1

    print(f"All {comparisons} float conversion operations verified across full 256-bit vector state, GPR returns, and aliased operands")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
    else:
        raise ValueError(mode)
