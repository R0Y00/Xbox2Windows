"""Differential check for VEX AVX packed shifts, packs, unpacks, shuffles, blends, movemask, and insert/extract."""
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
    "vpsllw_imm_op", "vpslld_imm_op", "vpsllq_imm_op", "vpsrlw_reg_op", "vpsrlq_reg_op",
    "vpsraw_imm_op", "vpsrad_reg_op", "vpslldq_op",
    "vpacksswb_op", "vpackssdw_op", "vpackuswb_op", "vpackusdw_op",
    "vpunpcklbw_op", "vpunpcklwd_op", "vpunpckldq_op", "vpunpcklqdq_op",
    "vpunpckhbw_op", "vpunpckhwd_op", "vpunpckhdq_op", "vpunpckhqdq_op",
    "vpshufd_op", "vpshuflw_op", "vpshufhw_op", "vpblendw_op", "vpblendvb_op",
    "vpinsrb_op", "vpinsrw_op", "vpinsrd_op", "vpinsrq_op",
)
SCALAR_EXPORTS = ("vpmovmskb_op", "vpextrb_op", "vpextrw_op", "vpextrd_op", "vpextrq_op")
ALL_EXPORTS = VEC_EXPORTS + SCALAR_EXPORTS + ("vpsrlw_mem_count_op", "vpshufd_ymm_mem_op")

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
        for name in ALL_EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst; r9 = mask;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst, uint64_t mask) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    for encoded in ("62f17d0970c11b", "c5f970c31b"):
        body = list(Decoder(64, bytes.fromhex(encoded + "c3"), ip=0x1000))
        try:
            lower(body, [], {}, {}, {})
        except ValueError as error:
            if "unsupported packed shift, pack or lane operation" not in str(error):
                raise
        else:
            raise AssertionError(f"expected {body[0]} to be rejected")


def packed_function(library, name):
    function = getattr(library, name)
    function.argtypes = [ctypes.c_void_p] * 4
    function.restype = ctypes.c_uint64
    return function


def check_memory_count(native, lifted, source, expected):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_void_p]
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    page = 4096
    base = kernel32.VirtualAlloc(None, 2 * page, 0x3000, 0x04)
    if not base:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        old = ctypes.c_uint32()
        if not kernel32.VirtualProtect(base + page, page, 0x01, ctypes.byref(old)):
            raise ctypes.WinError(ctypes.get_last_error())
        pointer = base + page - 16
        ctypes.memmove(pointer, struct.pack("<QQ", 5, 0xffffffffffffffff), 16)
        output = ctypes.create_string_buffer(32)
        for library, label in ((native, "native"), (lifted, "lifted")):
            run = packed_function(library, "vpsrlw_mem_count_op")
            run(source, pointer, output, None)
            if output.raw != expected:
                raise AssertionError(f"{label}: count must use the low 64 bits")
            # The low count still fits. The rest of the m128 operand is inaccessible.
            ctypes.memmove(pointer + 8, struct.pack("<Q", 5), 8)
            ctypes.memset(output, 0xa5, 32)
            try:
                run(source, pointer + 8, output, None)
            except OSError as error:
                if "access violation reading" not in str(error):
                    raise
            else:
                raise AssertionError(f"{label}: incomplete m128 count did not fault")
            if output.raw != b"\xa5" * 32:
                raise AssertionError(f"{label}: result was stored before the fault")
            ctypes.memmove(pointer + 8, struct.pack("<Q", 0xffffffffffffffff), 8)
    finally:
        if not kernel32.VirtualFree(base, 0, 0x8000):
            raise ctypes.WinError(ctypes.get_last_error())
    print("native and lifted shift read the full m128 count before storing")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    a_bytes = bytes([
        0x00, 0x80, 0x7f, 0x01, 0xff, 0x7f, 0x00, 0xfe,
        0x81, 0x00, 0x12, 0x34, 0xab, 0xcd, 0xef, 0x90,
    ]) + bytes([0xdd] * 16)
    b_bytes = bytes([
        0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x80, 0xff, 0x01, 0x02, 0x7f, 0x80, 0x00, 0x7f,
    ]) + bytes([0xee] * 16)
    m_bytes = bytes([
        0x00, 0x80, 0x7f, 0xff, 0x80, 0x00, 0xff, 0x01,
        0x80, 0x80, 0x00, 0x00, 0x7f, 0x80, 0x01, 0xff,
    ]) + bytes([0x11] * 16)
    a_buf = (ctypes.c_ubyte * 32)(*a_bytes)
    b_buf = (ctypes.c_ubyte * 32)(*b_bytes)
    m_buf = (ctypes.c_ubyte * 32)(*m_bytes)
    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()

    cases = 0
    for name in VEC_EXPORTS:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n), ctypes.addressof(m_buf))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l), ctypes.addressof(m_buf))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    for name in SCALAR_EXPORTS:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        got_n = fn_n(ctypes.addressof(a_buf), 0, 0, 0)
        got_l = fn_l(ctypes.addressof(a_buf), 0, 0, 0)
        if got_n != got_l:
            raise AssertionError(f"{name}: native={got_n:#x} lifted={got_l:#x}")
        cases += 1

    print(f"{cases} VEX AVX shift/pack/unpack/shuffle/blend/movemask/insert/extract cases matched native and translated DLLs")
    expected_count = struct.pack("<8H", *(word >> 5 for word in struct.unpack("<8H", a_bytes[:16]))) + bytes(16)
    check_memory_count(native, lifted, a_buf, expected_count)
    lifted.host_has_avx2.restype = ctypes.c_uint32
    if not lifted.host_has_avx2():
        print("host lacks AVX2; skipping the YMM memory shuffle")
        return
    source = ctypes.create_string_buffer(bytes(range(32)), 32)
    words = struct.unpack("<8I", source.raw)
    expected_y = struct.pack("<8I", *(words[lane + i] for lane in (0, 4) for i in (1, 0, 3, 2)))
    for library, label in ((native, "native"), (lifted, "lifted")):
        packed_function(library, "vpshufd_ymm_mem_op")(source, None, out_n, None)
        if bytes(out_n) != expected_y:
            raise AssertionError(f"{label}: YMM shuffle disagreed with the lane reference")
    print("AVX2 YMM memory shuffle matched independent lane positions")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
