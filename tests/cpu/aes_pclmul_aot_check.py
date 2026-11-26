"""Differential, FIPS-197, YMM[255:128], /O2, mutant, and rejection checks for cpu_aes_pclmul_aot."""

import ctypes
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from iced_x86 import Code, Decoder, Encoder, Formatter, FormatterSyntax, Instruction, MemoryOperand, Register as R
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_aes_pclmul_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_aesenc_reg", "op_aesenclast_reg", "op_aesdec_reg", "op_aesdeclast_reg",
    "op_aesimc_reg", "op_aeskeygenassist_reg",
    "op_pclmulqdq_00_reg", "op_pclmulqdq_11_reg", "op_pclmulqdq_ignore_bits_reg",
    "op_aesenc_mem", "op_aesenclast_mem", "op_aesdec_mem", "op_aesdeclast_mem",
    "op_aesimc_mem", "op_aeskeygenassist_mem", "op_pclmulqdq_01_mem",
    "op_vaesenc_reg", "op_vaesenclast_reg", "op_vaesdec_reg", "op_vaesdeclast_reg",
    "op_vaesimc_reg", "op_vaeskeygenassist_reg",
    "op_vpclmulqdq_10_reg", "op_vpclmulqdq_ignore_bits_reg",
    "op_vaesenc_mem", "op_vaesenclast_mem", "op_vaesdec_mem", "op_vaesdeclast_mem",
    "op_vaesimc_mem", "op_vaeskeygenassist_mem", "op_vpclmulqdq_11_mem",
    "op_sse_preserves_ymm_hi", "op_vex_zeroes_ymm_hi", "op_vzeroupper_clears_ymm_hi",
    "op_r8_source_then_store", "op_dst_alias_src", "op_aes_pclmul_discarded_read",
)

ALTERED_EXPORTS = (
    "op_aesenc_reg", "op_aesenclast_reg", "op_aesdec_reg", "op_aesdeclast_reg",
    "op_aesimc_reg", "op_aeskeygenassist_reg", "op_pclmulqdq_00_reg", "op_pclmulqdq_11_reg",
    "op_vaesenc_reg", "op_vaesenclast_reg", "op_vaesdec_reg", "op_vaesdeclast_reg",
    "op_vaesimc_reg", "op_vaeskeygenassist_reg", "op_vpclmulqdq_10_reg",
    "op_vaesenc_mem", "op_vpclmulqdq_11_mem", "op_sse_preserves_ymm_hi",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python GF(2^8) AES & GF(2) PCLMULQDQ reference model ---

def _gf_mul(a, b):
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return p


def _gf_inv(a):
    if a == 0:
        return 0
    for x in range(1, 256):
        if _gf_mul(a, x) == 1:
            return x
    raise ValueError(a)


def _rotl8(x, n):
    return ((x << n) | (x >> (8 - n))) & 0xFF


SBOX = tuple(_gf_inv(i) ^ _rotl8(_gf_inv(i), 1) ^ _rotl8(_gf_inv(i), 2) ^ _rotl8(_gf_inv(i), 3) ^ _rotl8(_gf_inv(i), 4) ^ 0x63 for i in range(256))
INV_SBOX = tuple(SBOX.index(i) for i in range(256))


def _shift_rows(b, *, inv=False):
    idx = (0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3) if inv else (0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11)
    return bytes(b[i] for i in idx)


def _mix_columns(b, *, inv=False):
    out = bytearray(16)
    coeffs = (0x0E, 0x0B, 0x0D, 0x09) if inv else (2, 3, 1, 1)
    for c in range(4):
        col = b[c * 4:(c + 1) * 4]
        for r in range(4):
            val = 0
            for k in range(4):
                coef = coeffs[(k - r) % 4]
                val ^= _gf_mul(coef, col[k])
            out[c * 4 + r] = val
    return bytes(out)


def _xor16(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def ref_aesenc(state, key, *, quirk=None):
    if quirk == "aes_no_shiftrows":
        sb = bytes(SBOX[x] for x in state)
        return _xor16(_mix_columns(sb), key)
    if quirk == "aes_no_mixcols":
        return ref_aesenclast(state, key)
    sr = _shift_rows(state)
    sb = bytes(SBOX[x] for x in sr)
    return _xor16(_mix_columns(sb), key)


def ref_aesenclast(state, key):
    sb = bytes(SBOX[x] for x in _shift_rows(state))
    return _xor16(sb, key)


def ref_aesdec(state, key):
    sb = bytes(INV_SBOX[x] for x in _shift_rows(state, inv=True))
    return _xor16(_mix_columns(sb, inv=True), key)


def ref_aesdeclast(state, key):
    sb = bytes(INV_SBOX[x] for x in _shift_rows(state, inv=True))
    return _xor16(sb, key)


def ref_aesimc(src):
    return _mix_columns(src, inv=True)


def ref_aeskeygenassist(src, imm8, *, quirk=None):
    words = list(struct.unpack("<4I", src))
    def sub_w(w):
        return (SBOX[w & 0xFF] | (SBOX[(w >> 8) & 0xFF] << 8)
                | (SBOX[(w >> 16) & 0xFF] << 16) | (SBOX[(w >> 24) & 0xFF] << 24))
    def rot_w(w):
        return ((w >> 8) | (w << 24)) & 0xFFFFFFFF
    s1, s3 = sub_w(words[1]), sub_w(words[3])
    rcon = 0 if quirk == "keygen_drop_rcon" else (imm8 & 0xFF)
    if quirk == "keygen_wrong_words":
        s1, s3 = sub_w(words[0]), sub_w(words[2])
    return struct.pack("<4I", s1, rot_w(s1) ^ rcon, s3, rot_w(s3) ^ rcon)


def ref_pclmulqdq(src1, src2, imm8, *, quirk=None):
    a0, a1 = struct.unpack("<2Q", src1)
    b0, b1 = struct.unpack("<2Q", src2)
    a = a1 if (imm8 & 0x01) else a0
    if quirk == "pclmul_wrong_op2_bit":
        b = b1 if (imm8 & 0x02) else b0
    elif quirk == "pclmul_reject_other_bits":
        b = 0 if (imm8 & 0xEE) else (b1 if (imm8 & 0x10) else b0)
    else:
        b = b1 if (imm8 & 0x10) else b0
    res = 0
    for i in range(64):
        if (b >> i) & 1:
            res ^= (a << i)
    return struct.pack("<2Q", res & 0xFFFFFFFFFFFFFFFF, (res >> 64) & 0xFFFFFFFFFFFFFFFF)


# --- Guarded aligned memory & DLL helpers ---

def alloc_guarded(payload, *, align=32):
    total = len(GUARD_PRE) + len(payload) + len(GUARD_POST) + align
    raw = ctypes.create_string_buffer(total)
    base = ctypes.addressof(raw)
    aligned_ptr = (base + len(GUARD_PRE) + align - 1) & ~(align - 1)
    off = aligned_ptr - base
    ctypes.memmove(aligned_ptr - len(GUARD_PRE), GUARD_PRE, len(GUARD_PRE))
    ctypes.memmove(aligned_ptr, payload, len(payload))
    ctypes.memmove(aligned_ptr + len(payload), GUARD_POST, len(GUARD_POST))
    return raw, aligned_ptr, off, len(payload)


def read_guarded(buf, off, length, label):
    raw = buf.raw
    if raw[off - len(GUARD_PRE):off] != GUARD_PRE or raw[off + length:off + length + len(GUARD_POST)] != GUARD_POST:
        raise AssertionError(f"guard bytes corrupted around {label}")
    return raw[off:off + length]


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_aes_pclmul_avx(lifted_lib):
    if hasattr(lifted_lib, "host_has_aes_pclmul_avx"):
        lifted_lib.host_has_aes_pclmul_avx.argtypes = []
        lifted_lib.host_has_aes_pclmul_avx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_aes_pclmul_avx())
    return False


def test_vectors():
    fips_a = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    fips_b = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    vectors = [
        (b"\x00" * 16, b"\x00" * 16),
        (b"\xff" * 16, b"\xff" * 16),
        (b"\x00" * 16, b"\xff" * 16),
        (b"\xff" * 16, b"\x00" * 16),
        (fips_a, fips_b),
        (fips_b, fips_a),
        (bytes(range(16)), bytes(range(16, 32))),
        (struct.pack("<2Q", 0x8000000000000001, 0xFFFFFFFFFFFFFFFF),
         struct.pack("<2Q", 0x8000000000000001, 0x0123456789ABCDEF)),
    ]
    rng = random.Random(0xAE59C1)
    for _ in range(16):
        vectors.append((rng.randbytes(16), rng.randbytes(16)))
    return vectors


# --- Rejection & alternate-encoding controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ret = Instruction.create(Code.RETNQ)
    ld_xmm0 = Instruction.create_reg_mem(Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX))
    ld_xmm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX))
    ld_ymm0 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX))
    ld_ymm1 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX))
    ld_ymm2 = Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX))
    st_xmm0 = Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM0)

    # Alternate encodings accepted cleanly.
    alt_accept = [
        ("vaesenc_xmm5_xmm1_r8mem", (ld_xmm1, Instruction.create_reg_reg_mem(
            Code.VEX_VAESENC_XMM_XMM_XMMM128, R.XMM5, R.XMM1, MemoryOperand(R.R8)),
            Instruction.create_mem_reg(Code.VEX_VMOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM5), ret),
         "ymm5 = zext256(op_aesenc(low128(ymm1), ld_1004_2));"),
        ("aesimc_xmm3_xmm1", (ld_xmm1, Instruction.create_reg_reg(Code.AESIMC_XMM_XMMM128, R.XMM3, R.XMM1),
            Instruction.create_mem_reg(Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8), R.XMM3), ret),
         "set_low128(ymm3, op_aesimc(low128(ymm1)));"),
        ("vpclmulqdq_xmm4_xmm1_r8mem", (ld_xmm1, Instruction.create_reg_reg_mem_u32(
            Code.VEX_VPCLMULQDQ_XMM_XMM_XMMM128_IMM8, R.XMM4, R.XMM1, MemoryOperand(R.R8), 0x11),
            Instruction.create_mem_reg(Code.VEX_VMOVDQA_XMMM128_XMM, MemoryOperand(R.R8), R.XMM4), ret),
         "ymm4 = zext256(op_pclmulqdq(low128(ymm1), ld_1004_2, 0x11u));"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r}")

    seg_aes = Instruction.create_reg_mem(Code.AESENC_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX))
    seg_aes.segment_prefix = R.GS
    evex_vaes = Instruction.create_reg_reg_reg(Code.EVEX_VAESENC_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2)

    unit_cases = [
        # VEX.256 compute rejections
        ("vex256_vaesenc", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(
            Code.VEX_VAESENC_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret), "unsupported VEX.256 instruction"),
        ("vex256_vaesdeclast", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(
            Code.VEX_VAESDECLAST_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret), "unsupported VEX.256 instruction"),
        ("vex256_vpclmulqdq", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg_u32(
            Code.VEX_VPCLMULQDQ_YMM_YMM_YMMM256_IMM8, R.YMM0, R.YMM1, R.YMM2, 0), ret), "unsupported VEX.256 instruction"),
        ("vex256_vpaddq", (ld_ymm1, ld_ymm2, Instruction.create_reg_reg_reg(
            Code.VEX_VPADDQ_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2), ret), "unsupported VEX.256 instruction"),
        # EVEX / KeyLocker / out-of-scope
        ("evex_vaesenc", (ld_xmm0, ld_xmm1, evex_vaes, ret), "unsupported EVEX instruction"),
        ("aes128kl", (ld_xmm0, Instruction.create_reg_mem(
            Code.AESENC128KL_XMM_M384, R.XMM0, MemoryOperand(R.RCX)), ret), "unsupported instruction"),
        ("xop_vpcmov", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg_reg_reg(
            Code.XOP_VPCMOV_XMM_XMM_XMMM128_XMM, R.XMM0, R.XMM1, R.XMM1, R.XMM0), ret), "unsupported XOP instruction"),
        # Nonvolatile XMM6..15 / YMM6..15
        ("nonvol_xmm6", (Instruction.create_reg_mem(
            Code.MOVDQU_XMM_XMMM128, R.XMM6, MemoryOperand(R.RCX)), ret), "nonvolatile vector register"),
        ("nonvol_ymm7", (Instruction.create_reg_mem(
            Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM7, MemoryOperand(R.RCX)), ret), "nonvolatile vector register"),
        ("nonvol_xmm15_aes", (ld_xmm0, Instruction.create_reg_reg(
            Code.AESENC_XMM_XMMM128, R.XMM0, R.XMM15), ret), "nonvolatile vector register"),
        # Undefined XMM / YMM reads
        ("undef_xmm2_vaesenc", (ld_xmm1, Instruction.create_reg_reg_reg(
            Code.VEX_VAESENC_XMM_XMM_XMMM128, R.XMM0, R.XMM1, R.XMM2), ret), "undefined vector register xmm2"),
        ("undef_xmm0_aesenc_dst", (ld_xmm1, Instruction.create_reg_reg(
            Code.AESENC_XMM_XMMM128, R.XMM0, R.XMM1), ret), "undefined vector register xmm0"),
        ("undef_xmm0_pclmul_dst", (ld_xmm1, Instruction.create_reg_reg_u32(
            Code.PCLMULQDQ_XMM_XMMM128_IMM8, R.XMM0, R.XMM1, 0), ret), "undefined vector register xmm0"),
        ("undef_ymm0_hi_after_sse", (ld_xmm0, ld_xmm1, Instruction.create_reg_reg(
            Code.AESENC_XMM_XMMM128, R.XMM0, R.XMM1),
            Instruction.create_mem_reg(Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.R8), R.YMM0), ret),
         "undefined vector register ymm0"),
        # Bad memory operands & non-[r8] stores
        ("displaced_aesenc_rdx", (ld_xmm0, Instruction.create_reg_mem(
            Code.AESENC_XMM_XMMM128, R.XMM0, MemoryOperand(R.RDX, displ=16, displ_size=1)), ret),
         "displaced memory operand"),
        ("displaced_store_r8", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.R8, displ=16, displ_size=1), R.XMM0), ret),
         "displaced memory operand"),
        ("indexed_vaesdec", (ld_xmm1, Instruction.create_reg_reg_mem(
            Code.VEX_VAESDEC_XMM_XMM_XMMM128, R.XMM0, R.XMM1, MemoryOperand(R.RCX, R.RDX, 1)), ret),
         "SIB index memory operand"),
        ("rip_rel_aesimc", (Instruction.create_reg_mem(
            Code.AESIMC_XMM_XMMM128, R.XMM0, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.RBX)), ret), "nonvolatile GPR base"),
        ("bad_base_r9", (Instruction.create_reg_mem(
            Code.MOVDQA_XMM_XMMM128, R.XMM0, MemoryOperand(R.R9)), ret), "bad base register"),
        ("non_r8_store_rcx", (ld_xmm0, Instruction.create_mem_reg(
            Code.MOVDQU_XMMM128_XMM, MemoryOperand(R.RCX), R.XMM0), ret), "non-r8 store memory operand"),
        ("non_r8_store_ymm_rdx", (ld_ymm0, Instruction.create_mem_reg(
            Code.VEX_VMOVDQU_YMMM256_YMM, MemoryOperand(R.RDX), R.YMM0), ret), "non-r8 store memory operand"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_aesenc", (ld_xmm0, seg_aes, ret), "unsupported instruction prefix"),
        ("branch_jmp", (ld_xmm0, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (ld_xmm0, st_xmm0), "missing terminal ret"),
        ("trailing_after_ret", (ld_xmm0, st_xmm0, ret, ld_xmm1), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(ret), "unsupported instruction prefix"),
        ("empty_stream", b"", "truncated instruction stream"),
        ("invalid_bytes", bytes.fromhex("fffe") + encode_seq(ret), "invalid instruction"),
    ]

    for label, seq, reason in unit_cases:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected control")

    for label, raw, reason in raw_cases:
        try:
            lift_code(label, 0x1000, raw)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted raw rejected control")

    # End-to-end PE-patched rejections & CLI non-traceback / temp-cleanup rejections.
    pe_bytes = Path(original_path).read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_dst_alias_src")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in unit_cases}
    pe_labels = (
        "vex256_vaesenc",
        "vex256_vpclmulqdq",
        "evex_vaesenc",
        "nonvol_xmm6",
        "undef_xmm2_vaesenc",
        "displaced_aesenc_rdx",
        "non_r8_store_rcx",
        "branch_jmp",
        "seg_prefix_aesenc",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_aes_pclmul_aot.py"
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        patched = tmp / "patched.dll"
        out_cpp = tmp / "rejected.cpp"
        for label in pe_labels:
            patch, reason = by_label[label]
            if len(patch) > max_len:
                raise AssertionError(f"pe_{label}: patch ({len(patch)} bytes) exceeds max_len ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off:file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched.write_bytes(damaged)
            try:
                translate(patched, ["op_dst_alias_src"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_dst_alias_src", str(out_cpp)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 1 or "Traceback" in proc.stderr or "unsupported instruction prefix" not in proc.stderr:
            raise AssertionError(f"CLI rejection unexpected: rc={proc.returncode}, stderr={proc.stderr!r}")
        if out_cpp.exists() or list(tmp.glob("*.tmp")):
            raise AssertionError("CLI failure left behind output or temp file")

        bad_dir = tmp / "dir_out"
        bad_dir.mkdir()
        proc_dir = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(original_path), "op_aesenc_reg", str(bad_dir)],
            capture_output=True,
            text=True,
        )
        if proc_dir.returncode != 1 or "Traceback" in proc_dir.stderr or "directory" not in proc_dir.stderr:
            raise AssertionError(f"CLI dir rejection unexpected: rc={proc_dir.returncode}, stderr={proc_dir.stderr!r}")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(unit_cases)} IR + "
        f"{len(raw_cases)} raw + {len(pe_labels)} PE-patched + CLI"
    )


# --- Positive, FIPS-197, YMM[255:128], altered, mutant, and /O2 verification ---

def _expected_16(name, a, b, c=b"\x00" * 16):
    if name in ("op_aesenc_reg", "op_aesenc_mem", "op_vaesenc_reg", "op_vaesenc_mem"):
        return ref_aesenc(a, b)
    if name in ("op_aesenclast_reg", "op_aesenclast_mem", "op_vaesenclast_reg", "op_vaesenclast_mem"):
        return ref_aesenclast(a, b)
    if name in ("op_aesdec_reg", "op_aesdec_mem", "op_vaesdec_reg", "op_vaesdec_mem"):
        return ref_aesdec(a, b)
    if name in ("op_aesdeclast_reg", "op_aesdeclast_mem", "op_vaesdeclast_reg", "op_vaesdeclast_mem"):
        return ref_aesdeclast(a, b)
    if name in ("op_aesimc_reg", "op_aesimc_mem", "op_vaesimc_reg", "op_vaesimc_mem"):
        return ref_aesimc(a)
    if name == "op_aeskeygenassist_reg":
        return ref_aeskeygenassist(a, 0x1B)
    if name == "op_aeskeygenassist_mem":
        return ref_aeskeygenassist(a, 0x80)
    if name == "op_vaeskeygenassist_reg":
        return ref_aeskeygenassist(a, 0x36)
    if name == "op_vaeskeygenassist_mem":
        return ref_aeskeygenassist(a, 0xFF)
    if name == "op_pclmulqdq_00_reg":
        return ref_pclmulqdq(a, b, 0x00)
    if name in ("op_pclmulqdq_11_reg", "op_vpclmulqdq_11_mem"):
        return ref_pclmulqdq(a, b, 0x11)
    if name in ("op_pclmulqdq_ignore_bits_reg", "op_pclmulqdq_01_mem"):
        return ref_pclmulqdq(a, b, 0x01)
    if name in ("op_vpclmulqdq_10_reg", "op_vpclmulqdq_ignore_bits_reg"):
        return ref_pclmulqdq(a, b, 0x10)
    if name == "op_r8_source_then_store":
        x0 = ref_aesenc(a, c)
        return ref_pclmulqdq(x0, b, 0x00)
    if name == "op_dst_alias_src":
        x0 = ref_aesenc(a, a)
        x0 = ref_aesimc(x0)
        x0 = ref_aeskeygenassist(x0, 0x01)
        return ref_pclmulqdq(x0, x0, 0x01)
    raise KeyError(name)


def verify_suite(fns, label, vectors):
    total = 0

    # 1) All 16-byte AES-NI & PCLMULQDQ register and memory forms.
    ops_16 = (
        "op_aesenc_reg", "op_aesenclast_reg", "op_aesdec_reg", "op_aesdeclast_reg",
        "op_aesimc_reg", "op_aeskeygenassist_reg",
        "op_pclmulqdq_00_reg", "op_pclmulqdq_11_reg", "op_pclmulqdq_ignore_bits_reg",
        "op_aesenc_mem", "op_aesenclast_mem", "op_aesdec_mem", "op_aesdeclast_mem",
        "op_aesimc_mem", "op_aeskeygenassist_mem", "op_pclmulqdq_01_mem",
        "op_vaesenc_reg", "op_vaesenclast_reg", "op_vaesdec_reg", "op_vaesdeclast_reg",
        "op_vaesimc_reg", "op_vaeskeygenassist_reg",
        "op_vpclmulqdq_10_reg", "op_vpclmulqdq_ignore_bits_reg",
        "op_vaesenc_mem", "op_vaesenclast_mem", "op_vaesdec_mem", "op_vaesdeclast_mem",
        "op_vaesimc_mem", "op_vaeskeygenassist_mem", "op_vpclmulqdq_11_mem",
        "op_dst_alias_src",
    )
    for name in ops_16:
        fn = fns[name]
        for a, b in vectors:
            ba, pa, oa, la = alloc_guarded(a)
            bb, pb, ob, lb = alloc_guarded(b)
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 16)
            fn(pa, pb, po)
            read_guarded(ba, oa, la, f"{name}_a")
            read_guarded(bb, ob, lb, f"{name}_b")
            got = read_guarded(bo, oo, lo, f"{name}_out")
            want = _expected_16(name, a, b)
            if got != want:
                raise AssertionError(f"{label} {name}: got={got.hex()} != want={want.hex()}")
            total += 1

    # 2) FIPS-197 Appendix B full AES-128 cipher & inverse cipher round check.
    fips_pt = bytes.fromhex("3243f6a8885a308d313198a2e0370734")
    fips_rk0 = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    fips_ct = bytes.fromhex("3925841d02dc09fbdc118597196a0b32")
    rcons = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)
    rks = [fips_rk0]
    for rcon in rcons:
        prev = list(struct.unpack("<4I", rks[-1]))
        kg = struct.unpack("<4I", ref_aeskeygenassist(rks[-1], rcon))[3]
        k0 = prev[0] ^ kg
        k1 = prev[1] ^ k0
        k2 = prev[2] ^ k1
        k3 = prev[3] ^ k2
        rks.append(struct.pack("<4I", k0, k1, k2, k3))
    state = _xor16(fips_pt, rks[0])
    for r in range(1, 10):
        ba, pa, oa, la = alloc_guarded(state)
        bb, pb, ob, lb = alloc_guarded(rks[r])
        bo, po, oo, lo = alloc_guarded(b"\x00" * 16)
        fns["op_vaesenc_reg"](pa, pb, po)
        state = read_guarded(bo, oo, lo, f"fips_enc_{r}")
    ba, pa, oa, la = alloc_guarded(state)
    bb, pb, ob, lb = alloc_guarded(rks[10])
    bo, po, oo, lo = alloc_guarded(b"\x00" * 16)
    fns["op_vaesenclast_reg"](pa, pb, po)
    ct = read_guarded(bo, oo, lo, "fips_enclast")
    if ct != fips_ct:
        raise AssertionError(f"{label} FIPS-197 encrypt mismatch: {ct.hex()} != {fips_ct.hex()}")
    total += 1

    # Inverse cipher round-trip using AESIMC + VAESDEC + VAESDECLAST.
    dec_state = _xor16(ct, rks[10])
    for r in range(9, 0, -1):
        bi, pi, oi, li = alloc_guarded(rks[r])
        bik, pik, oik, lik = alloc_guarded(b"\x00" * 16)
        fns["op_vaesimc_reg"](pi, 0, pik)
        inv_rk = read_guarded(bik, oik, lik, f"fips_imc_{r}")
        ba, pa, oa, la = alloc_guarded(dec_state)
        bb, pb, ob, lb = alloc_guarded(inv_rk)
        bo, po, oo, lo = alloc_guarded(b"\x00" * 16)
        fns["op_vaesdec_reg"](pa, pb, po)
        dec_state = read_guarded(bo, oo, lo, f"fips_dec_{r}")
    ba, pa, oa, la = alloc_guarded(dec_state)
    bb, pb, ob, lb = alloc_guarded(rks[0])
    bo, po, oo, lo = alloc_guarded(b"\x00" * 16)
    fns["op_vaesdeclast_reg"](pa, pb, po)
    pt = read_guarded(bo, oo, lo, "fips_declast")
    if pt != fips_pt:
        raise AssertionError(f"{label} FIPS-197 decrypt mismatch: {pt.hex()} != {fips_pt.hex()}")
    total += 1

    # 3) YMM[255:128] preservation (SSE), zeroing (VEX.128), and vzeroupper.
    hi_sentinel = bytes(range(0x10, 0x20))
    for a, b in vectors[:12]:
        ymm_in = a + hi_sentinel
        ba, pa, oa, la = alloc_guarded(ymm_in)
        bb, pb, ob, lb = alloc_guarded(b)
        bo1, po1, oo1, lo1 = alloc_guarded(b"\x5a" * 32)
        fns["op_sse_preserves_ymm_hi"](pa, pb, po1)
        read_guarded(ba, oa, la, "sse_hi_a")
        read_guarded(bb, ob, lb, "sse_hi_b")
        got_sse = read_guarded(bo1, oo1, lo1, "sse_hi_out")
        want_sse = ref_aesenc(a, b) + hi_sentinel
        if got_sse != want_sse:
            raise AssertionError(f"{label} op_sse_preserves_ymm_hi: {got_sse.hex()} != {want_sse.hex()}")
        total += 1

        bo2, po2, oo2, lo2 = alloc_guarded(b"\x5a" * 32)
        fns["op_vex_zeroes_ymm_hi"](pa, pb, po2)
        got_vex = read_guarded(bo2, oo2, lo2, "vex_hi_out")
        want_vex = ref_aesenc(a, b) + (b"\x00" * 16)
        if got_vex != want_vex:
            raise AssertionError(f"{label} op_vex_zeroes_ymm_hi: {got_vex.hex()} != {want_vex.hex()}")
        total += 1

        bo3, po3, oo3, lo3 = alloc_guarded(b"\x5a" * 32)
        fns["op_vzeroupper_clears_ymm_hi"](pa, 0, po3)
        got_vz = read_guarded(bo3, oo3, lo3, "vzeroupper_out")
        want_vz = a + (b"\x00" * 16)
        if got_vz != want_vz:
            raise AssertionError(f"{label} op_vzeroupper_clears_ymm_hi: {got_vz.hex()} != {want_vz.hex()}")
        total += 1

    # 4) [r8] source-then-store.
    for i in range(len(vectors)):
        a, b = vectors[i]
        c = vectors[(i + 5) % len(vectors)][0]
        ba, pa, oa, la = alloc_guarded(a)
        bb, pb, ob, lb = alloc_guarded(b)
        br, pr, or_, lr = alloc_guarded(c)
        fns["op_r8_source_then_store"](pa, pb, pr)
        read_guarded(ba, oa, la, "r8_src_a")
        read_guarded(bb, ob, lb, "r8_src_b")
        got_mem = read_guarded(br, or_, lr, "r8_src_out")
        want = _expected_16("op_r8_source_then_store", a, b, c)
        if got_mem != want:
            raise AssertionError(f"{label} op_r8_source_then_store: got={got_mem.hex()} != exp={want.hex()}")
        total += 1

    # 5) Discarded faulting m128 read must fault on NULL rdx even under /O2.
    a0, _ = vectors[0]
    ba, pa, oa, la = alloc_guarded(a0)
    bd, pd, od, ld = alloc_guarded(vectors[1][0])
    bo, po, oo, lo = alloc_guarded(b"\x5a" * 16)
    fns["op_aes_pclmul_discarded_read"](pa, pd, po)
    read_guarded(bd, od, ld, "discarded_valid")
    assert read_guarded(bo, oo, lo, "discarded_out") == a0
    try:
        fns["op_aes_pclmul_discarded_read"](pa, 0, po)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_aes_pclmul_discarded_read(NULL) dropped faulting m128 read")

    return total


def _altered_expected(name, a, b):
    if name in ("op_aesenc_reg", "op_vaesenclast_reg"):
        return ref_aesenclast(a, b) if name == "op_aesenc_reg" else ref_aesdeclast(a, b)
    if name == "op_aesenclast_reg":
        return ref_aesenc(a, b)
    if name == "op_aesdec_reg":
        return ref_aesdeclast(a, b)
    if name == "op_aesdeclast_reg":
        return ref_aesdec(a, b)
    if name in ("op_aesimc_reg", "op_vaesimc_reg"):
        return a
    if name == "op_aeskeygenassist_reg":
        return ref_aeskeygenassist(a, 0x1C)
    if name == "op_pclmulqdq_00_reg":
        return ref_pclmulqdq(a, b, 0x11)
    if name == "op_pclmulqdq_11_reg":
        return ref_pclmulqdq(a, b, 0x00)
    if name in ("op_vaesenc_reg", "op_vaesenc_mem"):
        return ref_aesdec(a, b)
    if name == "op_vaesdec_reg":
        return ref_aesenc(a, b)
    if name == "op_vaesdeclast_reg":
        return ref_aesenclast(a, b)
    if name == "op_vaeskeygenassist_reg":
        return ref_aeskeygenassist(a, 0x37)
    if name == "op_vpclmulqdq_10_reg":
        return ref_pclmulqdq(a, b, 0x01)
    if name == "op_vpclmulqdq_11_mem":
        return ref_pclmulqdq(a, b, 0x00)
    if name == "op_sse_preserves_ymm_hi":
        return ref_aesenc(a, b) + (b"\x00" * 16)
    raise KeyError(name)


def check_altered_and_mutants(lifted_fns, alt_lifted_fns, vectors):
    diverged = 0
    hi_sentinel = bytes(range(0x10, 0x20))
    for name in ALTERED_EXPORTS:
        is_ymm = name == "op_sse_preserves_ymm_hi"
        diffs = 0
        for a, b in vectors[:8]:
            buf_a = (a + hi_sentinel) if is_ymm else a
            ba, pa, oa, la = alloc_guarded(buf_a)
            bb, pb, ob, lb = alloc_guarded(b)
            out_len = 32 if is_ymm else 16
            bo_a, po_a, oo_a, lo_a = alloc_guarded(b"\x5a" * out_len)
            bo_o, po_o, oo_o, lo_o = alloc_guarded(b"\x5a" * out_len)
            alt_lifted_fns[name](pa, pb, po_a)
            lifted_fns[name](pa, pb, po_o)
            got_al = read_guarded(bo_a, oo_a, lo_a, f"alt_{name}")
            got_ol = read_guarded(bo_o, oo_o, lo_o, f"orig_{name}")
            want_alt = _altered_expected(name, a, b)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}: {got_al.hex()} != {want_alt.hex()}")
            diffs += got_al != got_ol
        if not diffs:
            raise AssertionError(f"ALTER_AES_PCLMUL failed to diverge on {name}")
        diverged += 1

    # Mutant models targeting critical AES/PCLMUL SDM details.
    mutant_specs = (
        ("op_aesenc_reg", lambda a, b: ref_aesenc(a, b, quirk="aes_no_shiftrows")),
        ("op_aesenc_reg", lambda a, b: ref_aesenc(a, b, quirk="aes_no_mixcols")),
        ("op_aeskeygenassist_reg", lambda a, b: ref_aeskeygenassist(a, 0x1B, quirk="keygen_drop_rcon")),
        ("op_aeskeygenassist_reg", lambda a, b: ref_aeskeygenassist(a, 0x1B, quirk="keygen_wrong_words")),
        ("op_vpclmulqdq_ignore_bits_reg", lambda a, b: ref_pclmulqdq(a, b, 0xFC, quirk="pclmul_wrong_op2_bit")),
        ("op_pclmulqdq_ignore_bits_reg", lambda a, b: ref_pclmulqdq(a, b, 0xEF, quirk="pclmul_reject_other_bits")),
    )
    for name, mut_fn in mutant_specs:
        fn = lifted_fns[name]
        caught = 0
        for a, b in vectors[:8]:
            ba, pa, oa, la = alloc_guarded(a)
            bb, pb, ob, lb = alloc_guarded(b)
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 16)
            fn(pa, pb, po)
            got = read_guarded(bo, oo, lo, f"mut_{name}")
            if got != mut_fn(a, b):
                caught += 1
        if not caught:
            raise AssertionError(f"mutant on {name} survived")

    return diverged, len(mutant_specs)


def compile_o2_and_verify(original_path, cxx_path, vectors):
    source_text, _ = translate(original_path, list(EXPORTS))
    if "<wmmintrin.h>" in source_text or "_mm_aes" in source_text or "_mm_clmul" in source_text:
        raise AssertionError("translated C++ must be pure portable C++ without AES/PCLMUL intrinsics")
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "aes_pclmul_aot_o2.cpp"
        dll_path = tmp / "aes_pclmul_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'aes_pclmul_aot_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_suite(o2_fns, "lifted_o2", vectors)
        finally:
            o2_fns.clear()
            unload_lib(o2_lib)


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path, cxx_path = (
        Path(p) for p in sys.argv[1:6]
    )

    # 1) Lift + rejections + portable lifted + mutants + /O2 run unconditionally BEFORE CPUID skip.
    check_rejections(original_path)

    vectors = test_vectors()
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_suite(lifted_fns, "lifted", vectors)
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_lifted_fns, vectors)
        o2_cases = compile_o2_and_verify(original_path, cxx_path, vectors)

        if not host_has_aes_pclmul_avx(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{diverged} ALTER_AES_PCLMUL + {mutants} mutants); "
                "Host lacks CPUID.01H:ECX[1,25,27,28]/XGETBV[2:1] (PCLMULQDQ+AES+AVX), skipping native differential check"
            )
            sys.exit(77)

        # 2) Native AES-NI + PCLMULQDQ + AVX hardware differential checks.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_suite(orig_fns, "native", vectors)
            hi_sentinel = bytes(range(0x10, 0x20))
            for name in ALTERED_EXPORTS:
                is_ymm = name == "op_sse_preserves_ymm_hi"
                out_len = 32 if is_ymm else 16
                for a, b in vectors[:8]:
                    buf_a = (a + hi_sentinel) if is_ymm else a
                    ba, pa, oa, la = alloc_guarded(buf_a)
                    bb, pb, ob, lb = alloc_guarded(b)
                    bn, pn, on, ln = alloc_guarded(b"\x5a" * out_len)
                    bl, pl, ol, ll = alloc_guarded(b"\x5a" * out_len)
                    alt_fns[name](pa, pb, pn)
                    alt_lifted_fns[name](pa, pb, pl)
                    got_an = read_guarded(bn, on, ln, f"alt_nat_{name}")
                    got_al = read_guarded(bl, ol, ll, f"alt_lift_{name}")
                    if got_an != got_al:
                        raise AssertionError(f"altered native vs lifted {name}: {got_an.hex()} != {got_al.hex()}")
        finally:
            orig_fns.clear()
            alt_fns.clear()
            unload_lib(orig_lib)
            unload_lib(alt_lib)
    finally:
        lifted_fns.clear()
        alt_lifted_fns.clear()
        unload_lib(lifted_lib)
        unload_lib(alt_lifted_lib)

    record_covered_forms(original_path)
    print(
        f"aes_pclmul_aot passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; ALTER_AES_PCLMUL diverged on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
