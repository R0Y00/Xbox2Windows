"""Differential check for AVX broadcast, extract/insert, permute, shuffle, blend, dot, round, and zeroall."""
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
    "vbroadcastss_op", "vbroadcastsd_op", "vbroadcastf128_op",
    "vextractf128_op", "vinsertf128_op", "vperm2f128_op",
    "vpermilps_imm_op", "vpermilpd_imm_op", "vshufps_op", "vshufpd_op",
    "vunpcklps_op", "vunpckhps_op", "vunpcklpd_op", "vunpckhpd_op",
    "vmovddup_op", "vmovddup_m64_op", "vmovsldup_op", "vmovshdup_op",
    "vblendps_op", "vblendpd_op", "vblendvps_op", "vblendvpd_op",
    "vdpps_op", "vdppd_op", "vroundps_op", "vroundpd_op", "vzeroall_op",
)

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in EXPORTS:
            entry = "vzeroall_lift_entry" if name == "vzeroall_op" else name
            ranges = function_ranges(pe, export_rva(pe, entry))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst; r9 = mask;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst, uint64_t mask) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    # vroundps ymm0, ymm1, 0x04 (MXCSR rounding mode bit 2 set) followed by ret must be rejected.
    mxcsr = list(Decoder(64, b"\xc4\xe3\x7d\x08\xc1\x04\xc3", ip=0x1000))
    try:
        lower(mxcsr, [], {}, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected MXCSR-mode vroundps to be rejected")


def invoke(library, name, a, b=0, mask=0):
    out = (ctypes.c_ubyte * 32)()
    fn = getattr(library, name)
    fn.argtypes = [ctypes.c_uint64] * 4
    fn.restype = ctypes.c_uint64
    fn(a, b, ctypes.addressof(out), mask)
    return bytes(out)


def check_m64(native, lifted):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel.VirtualAlloc.restype = ctypes.c_void_p
    kernel.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_uint32)]
    kernel.VirtualProtect.restype = ctypes.c_int
    kernel.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    kernel.VirtualFree.restype = ctypes.c_int
    page_size = 4096
    base = kernel.VirtualAlloc(None, 2 * page_size, 0x3000, 4)
    if not base:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        end = base + page_size
        if not kernel.VirtualProtect(end, page_size, 1, ctypes.byref(ctypes.c_uint32())):
            raise ctypes.WinError(ctypes.get_last_error())
        value = bytes.fromhex("2143fedc88776655")
        ctypes.memmove(end - len(value), value, len(value))
        expected = value * 2 + bytes(16)
        for library in (native, lifted):
            assert invoke(library, "vmovddup_m64_op", end - len(value)) == expected
    finally:
        if not kernel.VirtualFree(base, 0, 0x8000):
            raise ctypes.WinError(ctypes.get_last_error())
    print("VMOVDDUP XMM reads eight bytes at a guard-page boundary")


def check_cancellation(native, lifted):
    # A left-to-right DPPS sum yields 1 in the first quad on this case; paired
    # addition loses the 1 and cancels to zero.
    a = ctypes.create_string_buffer(struct.pack("<8f", 1e20, 0, -1e20, 1, 1, 1, 1, 1))
    b = ctypes.create_string_buffer(struct.pack("<8f", *([1] * 8)))
    args = (ctypes.addressof(a), ctypes.addressof(b))
    expected = struct.pack("<8f", 0, 0, 0, 0, 3, 0, 3, 0)
    for library in (native, lifted):
        assert invoke(library, "vdpps_op", *args) == expected
    print("VDPPS pairwise cancellation matches the native result")

    a = ctypes.create_string_buffer(struct.pack("<8I", 0x7fc01234, 0, 0xffc05678, 0x3f800000,
                                                *([0x3f800000] * 4)))
    args = (ctypes.addressof(a), ctypes.addressof(b))
    for library in (native, lifted):
        lanes = struct.unpack("<8I", invoke(library, "vdpps_op", *args))
        # The horizontal sum does not guarantee a priority among NaN inputs.
        for index in (0, 2):
            assert lanes[index] & 0x7f800000 == 0x7f800000 and lanes[index] & 0x007fffff
        assert lanes[1] == lanes[3] == 0
        assert lanes[4:] == (0x40400000, 0, 0x40400000, 0)
    print("VDPPS multiple NaNs stay in selected lanes; other lanes match exactly")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    if not lifted.host_has_avx():
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    ps_a = struct.pack("<8f", 2.5, -2.5, 3.5, -0.25, 1.25, -4.5, 0.5, -0.5)
    ps_b = struct.pack("<8f", 1.5, 2.0, -1.0, 4.0, -2.0, 3.0, 6.0, -8.0)
    ps_mask = struct.pack("<8I", 0, 0x80000000, 0x7fffffff, 0xffffffff, 0x80000001, 0, 0x80000000, 1)

    pd_a = struct.pack("<4d", -3.5, 2.5, 1.25, -0.25)
    pd_b = struct.pack("<4d", 1.5, -4.25, 2.0, 8.0)
    pd_mask = struct.pack("<4Q", 0x8000000000000000, 0x7fffffffffffffff, 0, 0xffffffffffffffff)

    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    for name in EXPORTS:
        is_pd = "pd" in name or "sd" in name or "ddup" in name
        a, b, m = (pd_a, pd_b, pd_mask) if is_pd else (ps_a, ps_b, ps_mask)
        a_buf = (ctypes.c_ubyte * 32)(*a)
        b_buf = (ctypes.c_ubyte * 32)(*b)
        m_buf = (ctypes.c_ubyte * 32)(*m)
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n), ctypes.addressof(m_buf))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l), ctypes.addressof(m_buf))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
    probe = native.vzeroall_abi
    probe.argtypes = [ctypes.c_uint64] * 4
    probe.restype = ctypes.c_uint32
    assert probe(ctypes.addressof(a_buf), 0, ctypes.addressof(out_n), 0) == 1
    print("VZEROALL native wrapper preserves a caller XMM6 sentinel")
    for library in (native, lifted):
        f32 = ctypes.create_string_buffer(struct.pack("<8I", 0x7f800001, *([0] * 7)))
        f64 = ctypes.create_string_buffer(struct.pack("<4Q", 0xfff0000000000001, *([0] * 3)))
        assert invoke(library, "vroundps_op", ctypes.addressof(f32)) == struct.pack("<8I", 0x7fc00001, *([0] * 7))
        assert invoke(library, "vroundpd_op", ctypes.addressof(f64)) == struct.pack("<4Q", 0xfff8000000000001, *([0] * 3))
    print("VROUNDPS and VROUNDPD quiet signaling NaNs with the same payload")
    check_m64(native, lifted)
    check_cancellation(native, lifted)
    print(f"{len(EXPORTS)} AVX broadcast/permute/shuffle/blend/dot/round/zeroall cases matched native and translated DLLs")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
