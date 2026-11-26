"""Differential, /O2, flag-modeling, mutant, and rejection checks for cpu_bmi1_logic_aot (ANDN, BEXTR, TZCNT)."""

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
from cpu_bmi1_logic_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_andn32", "op_andn64",
    "op_bextr32", "op_bextr64",
    "op_tzcnt32", "op_tzcnt64",
    "op_andn32_mem", "op_andn64_mem",
    "op_bextr32_mem", "op_bextr64_mem",
    "op_tzcnt32_mem", "op_tzcnt64_mem",
    "op_andn32_flags", "op_andn64_flags",
    "op_bextr32_flags", "op_bextr64_flags",
    "op_tzcnt32_flags", "op_tzcnt64_flags",
    "op_bextr64_mem_flags",
    "op_tzcnt64_mem_flags",
    "op_r8_source_then_store",
    "op_dst_alias_src",
    "op_bmi1_logic_discarded_read",
)

ALTERED_REG_EXPORTS = (
    "op_andn32", "op_andn64",
    "op_bextr32", "op_bextr64",
    "op_tzcnt32", "op_tzcnt64",
)
ALTERED_MEM_EXPORTS = (
    "op_andn32_mem", "op_andn64_mem",
    "op_bextr32_mem", "op_bextr64_mem",
    "op_tzcnt32_mem", "op_tzcnt64_mem",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference model ---

def ref_bmi1_logic(op, a, b, bits, *, quirk=None):
    """Return (dst, packed_flags) matching SDM flags per instruction."""
    mask = MASK32 if bits == 32 else MASK64
    sign = 1 << (bits - 1)
    s1 = a & mask
    s2 = b & mask

    if op == "andn":
        if quirk == "andn_swap":
            d = ((~s2) & s1) & mask
        elif quirk == "andn_plain_and":
            d = (s1 & s2) & mask
        else:
            d = ((~s1) & s2) & mask
        cf = True if quirk == "set_cf" else False
        of = True if quirk == "set_of" else False
        zf = (d == 0)
        sf = False if quirk == "drop_sf" else ((d & sign) != 0)
    elif op == "bextr":
        if quirk == "bextr_swap_start_len":
            start = (s2 >> 8) & 0xFF
            length = s2 & 0xFF
        else:
            start = s2 & 0xFF
            length = (s2 >> 8) & 0xFF
        if quirk == "bextr_mask_start_bits":
            start &= (bits - 1)
        if quirk == "bextr_len_zero_all" and length == 0:
            length = bits
        if length == 0 or start >= bits:
            d = 0
        else:
            bit_mask = mask if length >= bits else ((1 << length) - 1)
            d = ((s1 >> start) & bit_mask) & mask
        cf = True if quirk == "set_cf" else False
        of = True if quirk == "set_of" else False
        zf = False if quirk == "bextr_zf_false" else (d == 0)
        sf = False
    elif op == "tzcnt":
        if s1 == 0:
            if quirk == "tzcnt_zero_0":
                d = 0
            elif quirk == "tzcnt_zero_width_m1":
                d = bits - 1
            else:
                d = bits
            cf = False if quirk == "tzcnt_invert_cf" else True
        else:
            d = (s1 & -s1).bit_length() - 1
            cf = True if quirk == "tzcnt_invert_cf" else False
        zf = (s1 == 0) if quirk == "tzcnt_zf_from_src" else (d == 0)
        sf = False
        of = False
    else:
        raise ValueError(op)

    if quirk == "no_zext32" and bits == 32:
        d |= 0xFFFFFFFF00000000
    packed = (int(cf) << 0) | (int(zf) << 1) | (int(sf) << 2) | (int(of) << 3)
    return d, packed


def test_pairs():
    singles = [
        0x0000000000000000,
        0x0000000000000001,
        0x0000000000000002,
        0x0000000000000010,
        0x000000007FFFFFFF,
        0x0000000080000000,
        0x0000000080000001,
        0x00000000FFFFFFFE,
        0x00000000FFFFFFFF,
        0xDEADBEEF00000000,
        0xDEADBEEF80000000,
        0x0000000100000000,
        0x7FFFFFFFFFFFFFFF,
        0x8000000000000000,
        0x8000000000000001,
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
    ]
    singles.extend(1 << b for b in range(64))
    bextr_ctrls = [
        0x0000000000000000,  # start=0, len=0
        0x0000000000000005,  # start=5, len=0
        0x0000000000000100,  # start=0, len=1
        0x000000000000011F,  # start=31, len=1
        0x0000000000000120,  # start=32, len=1 (out of 32b range, valid in 64b)
        0x000000000000013F,  # start=63, len=1
        0x0000000000000140,  # start=64, len=1 (out of 64b range)
        0x0000000000001418,  # start=24, len=20 (crosses 32b boundary)
        0x0000000000002000,  # start=0, len=32
        0x0000000000002010,  # start=16, len=32
        0x0000000000004000,  # start=0, len=64
        0x0000000000004020,  # start=32, len=64 (crosses 64b boundary)
        0x000000000000FF00,  # start=0, len=255
        0x000000000000FF1F,  # start=31, len=255
        0x000000000000FF3F,  # start=63, len=255
        0x000000000000FFFF,  # start=255, len=255
        0xABCD123400000804,  # dirty upper bits [31:16]/[63:16]
        0xCAFE000000002018,  # dirty upper bits
    ]
    pairs = [
        (0x0000000000000000, 0x0000000080000000),
        (0x0000000000000000, 0x8000000000000000),
        (0x0000000080000000, 0x00000000FFFFFFFF),
        (0x8000000000000000, 0xFFFFFFFFFFFFFFFF),
    ]
    for idx, a in enumerate(singles):
        b = bextr_ctrls[idx % len(bextr_ctrls)] if idx < len(bextr_ctrls) * 2 else singles[(idx * 7 + 3) % len(singles)]
        pairs.append((a, b))
    for ctrl in bextr_ctrls:
        pairs.append((0xF0E1D2C3B4A59687, ctrl))
        pairs.append((0x8000000080000001, ctrl))
    rng = random.Random(0xB11101C)
    for _ in range(32):
        pairs.append((rng.getrandbits(64), rng.getrandbits(64)))
    return pairs


# --- Guarded memory & DLL helpers ---

def alloc_guarded(payload):
    raw = ctypes.create_string_buffer(GUARD_PRE + payload + GUARD_POST, len(GUARD_PRE) + len(payload) + len(GUARD_POST))
    base = ctypes.addressof(raw)
    return raw, base + len(GUARD_PRE), len(GUARD_PRE), len(payload)


def read_guarded(buf, off, length, label):
    raw = buf.raw
    if raw[:off] != GUARD_PRE or raw[off + length:] != GUARD_POST:
        raise AssertionError(f"guard bytes corrupted around {label}")
    return raw[off:off + length]


def bind_exports(dll_path, prefix=""):
    lib = ctypes.CDLL(str(Path(dll_path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def unload_lib(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.FreeLibrary.restype = ctypes.c_int
    handle = getattr(lib, "_handle", None)
    if handle:
        kernel32.FreeLibrary(handle)


def host_has_bmi1(lifted_lib):
    if hasattr(lifted_lib, "host_has_bmi1"):
        lifted_lib.host_has_bmi1.argtypes = []
        lifted_lib.host_has_bmi1.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_bmi1())
    return False


# --- Rejection & alternate-encoding controls ---

def encode_seq(*instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections(original_path):
    ret = Instruction.create(Code.RETNQ)
    mov_rax_rcx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.RCX)
    xor_r9 = Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R9D, R.R9D)
    andn_eax_ecx_edx = Instruction.create_reg_reg_reg(Code.VEX_ANDN_R32_R32_RM32, R.EAX, R.ECX, R.EDX)
    bextr_rax_rcx_rdx = Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R64_RM64_R64, R.RAX, R.RCX, R.RDX)
    tzcnt_rax_rcx = Instruction.create_reg_reg(Code.TZCNT_R64_RM64, R.RAX, R.RCX)

    # Alternate encodings accepted cleanly.
    alt_accept = [
        ("andn32_r11_rdx_r8mem", (Instruction.create_reg_reg_mem(Code.VEX_ANDN_R32_R32_RM32, R.R11D, R.EDX, MemoryOperand(R.R8)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11), ret),
         "r11 = op_andn32((rdx & 0xFFFFFFFFull), ld_1000_2, flags);"),
        ("bextr64_r10_rdxmem_rcx", (Instruction.create_reg_mem_reg(Code.VEX_BEXTR_R64_RM64_R64, R.R10, MemoryOperand(R.RDX), R.RCX),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R10), ret),
         "r10 = op_bextr64(ld_1000_1, rcx, flags);"),
        ("tzcnt32_r9_r8d", (Instruction.create_reg_reg(Code.TZCNT_R32_RM32, R.R9D, R.R8D),
            Instruction.create_reg_reg(Code.MOV_R32_RM32, R.EAX, R.R9D), ret),
         "r9 = op_tzcnt32((r8 & 0xFFFFFFFFull), flags);"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r}")

    seg_andn = Instruction.create_reg_reg_mem(Code.VEX_ANDN_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.RDX))
    seg_andn.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # Other BMI1 instructions (BLSI / BLSMSK / BLSR) & 16-bit TZCNT & BSF/LZCNT
        ("bmi1_blsi", (Instruction.create_reg_reg(Code.VEX_BLSI_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported VEX instruction"),
        ("bmi1_blsmsk", (Instruction.create_reg_reg(Code.VEX_BLSMSK_R32_RM32, R.EAX, R.ECX), ret),
         "unsupported VEX instruction"),
        ("bmi1_blsr", (Instruction.create_reg_reg(Code.VEX_BLSR_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported VEX instruction"),
        ("tzcnt_16bit", (Instruction.create_reg_reg(Code.TZCNT_R16_RM16, R.AX, R.CX), mov_rax_rcx, ret),
         "unsupported instruction"),
        ("bsf_32bit", (Instruction.create_reg_reg(Code.BSF_R32_RM32, R.EAX, R.ECX), ret),
         "unsupported instruction"),
        ("lzcnt_64bit", (Instruction.create_reg_reg(Code.LZCNT_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported instruction"),
        # BMI2 rejections
        ("bmi2_rorx", (Instruction.create_reg_reg_u32(Code.VEX_RORX_R64_RM64_IMM8, R.RAX, R.RCX, 5), ret),
         "unsupported VEX instruction"),
        ("bmi2_sarx", (Instruction.create_reg_reg_reg(Code.VEX_SARX_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_shlx", (Instruction.create_reg_reg_reg(Code.VEX_SHLX_R32_RM32_R32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("bmi2_bzhi", (Instruction.create_reg_reg_reg(Code.VEX_BZHI_R64_RM64_R64, R.RAX, R.RCX, R.RDX), ret),
         "unsupported VEX instruction"),
        # EVEX / XOP
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_bextr32", (Instruction.create_reg_reg_u32(Code.XOP_BEXTR_R32_RM32_IMM32, R.EAX, R.ECX, 0x0804), ret),
         "unsupported XOP instruction"),
        ("xop_bextr64", (Instruction.create_reg_reg_u32(Code.XOP_BEXTR_R64_RM64_IMM32, R.RAX, R.RCX, 0x0804), ret),
         "unsupported XOP instruction"),
        # Undefined flag reads per instruction (including after prior XOR defining all flags!)
        ("undef_pf_after_andn", (xor_r9, andn_eax_ecx_edx, Instruction.create_reg(Code.SETP_RM8, R.R9L), ret),
         "undefined flag pf read"),
        ("undef_sf_after_bextr", (xor_r9, bextr_rax_rcx_rdx, Instruction.create_reg(Code.SETS_RM8, R.R9L), ret),
         "undefined flag sf read"),
        ("undef_pf_after_bextr", (xor_r9, bextr_rax_rcx_rdx, Instruction.create_mem(Code.SETNP_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag pf read"),
        ("undef_sf_after_tzcnt", (xor_r9, tzcnt_rax_rcx, Instruction.create_reg(Code.SETNS_RM8, R.R9L), ret),
         "undefined flag sf read"),
        ("undef_of_after_tzcnt", (xor_r9, tzcnt_rax_rcx, Instruction.create_mem(Code.SETO_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag of read"),
        ("undef_pf_after_tzcnt", (xor_r9, tzcnt_rax_rcx, Instruction.create_reg(Code.SETP_RM8, R.R9L), ret),
         "undefined flag pf read"),
        ("undef_af_lahf_after_andn", (andn_eax_ecx_edx, Instruction.create(Code.LAHF), ret),
         "unsupported instruction"),
        ("undef_flags_pushfq", (tzcnt_rax_rcx, Instruction.create(Code.PUSHFQ), ret),
         "unsupported instruction"),
        ("undef_flags_before_op", (mov_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag cf read"),
        # Nonvolatile GPRs, arg GPR writes, high-byte GPRs
        ("nonvol_rbx_dst", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R64_R64_RM64, R.RBX, R.RCX, R.RDX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R64_RM64_R64, R.RAX, R.RSI, R.RDX), ret),
         "nonvolatile GPR"),
        ("nonvol_r12_dst", (Instruction.create_reg_reg(Code.TZCNT_R32_RM32, R.R12D, R.ECX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("write_arg_rcx", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R64_R64_RM64, R.RCX, R.RDX, R.R8), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_edx", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R32_RM32_R32, R.EDX, R.ECX, R.R8D), mov_rax_rcx, ret),
         "cannot write argument GPR rdx"),
        ("write_arg_r8", (Instruction.create_reg_reg(Code.TZCNT_R64_RM64, R.R8, R.RCX), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        ("high_byte_ah", (mov_rax_rcx, Instruction.create_reg_u32(Code.MOV_R8_IMM8, R.AH, 1), ret),
         "unsupported high-byte GPR"),
        # Undefined GPRs
        ("undef_gpr_r9_src", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R64_R64_RM64, R.RAX, R.R9, R.RCX), ret),
         "undefined GPR r9"),
        ("undef_gpr_r10_bextr_ctrl", (Instruction.create_reg_reg_reg(Code.VEX_BEXTR_R64_RM64_R64, R.RAX, R.RCX, R.R10), ret),
         "undefined GPR r10"),
        ("undef_rax_ret", (Instruction.create_reg_reg(Code.TZCNT_R64_RM64, R.R9, R.RCX), ret),
         "return with undefined RAX"),
        # Displaced / indexed / non-flat memory & non-r8 stores
        ("displaced_andn_rcx", (Instruction.create_reg_reg_mem(
            Code.VEX_ANDN_R64_R64_RM64, R.RAX, R.RDX, MemoryOperand(R.RCX, displ=8, displ_size=1)), ret),
         "displaced memory operand"),
        ("displaced_bextr_rdx", (Instruction.create_reg_mem_reg(
            Code.VEX_BEXTR_R32_RM32_R32, R.EAX, MemoryOperand(R.RDX, displ=4, displ_size=1), R.ECX), ret),
         "displaced memory operand"),
        ("displaced_store_r8", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.R8, displ=8, displ_size=1), R.RAX), ret),
         "displaced memory operand"),
        ("indexed_tzcnt", (Instruction.create_reg_mem(
            Code.TZCNT_R64_RM64, R.RAX, MemoryOperand(R.RCX, R.RDX, 8)), ret),
         "SIB index memory operand"),
        ("rip_rel_andn", (Instruction.create_reg_reg_mem(
            Code.VEX_ANDN_R64_R64_RM64, R.RAX, R.RCX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_mem(Code.TZCNT_R64_RM64, R.RAX, MemoryOperand(R.RBX)), ret),
         "bad base register in memory operand"),
        ("bad_base_rsp", (Instruction.create_reg_mem_reg(
            Code.VEX_BEXTR_R32_RM32_R32, R.EAX, MemoryOperand(R.RSP), R.ECX), ret),
         "bad base register in memory operand"),
        ("non_r8_store_rcx", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret), "non-r8 store memory operand"),
        ("non_r8_store_setb", (tzcnt_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.RDX)), ret),
         "non-r8 store memory operand"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_andn", (seg_andn, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (andn_eax_ecx_edx,), "missing terminal ret"),
        ("trailing_after_ret", (tzcnt_rax_rcx, ret, mov_rax_rcx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("repne_bsf", bytes.fromhex("f20fbcc1") + encode_seq(ret), "unsupported instruction prefix"),
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
        entry = export_rva(pe, "op_andn64_flags")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in unit_cases}
    pe_labels = (
        "bmi1_blsi",
        "bmi2_rorx",
        "xop_bextr32",
        "evex_vpaddd",
        "undef_sf_after_bextr",
        "undef_of_after_tzcnt",
        "nonvol_rbx_dst",
        "write_arg_rcx",
        "displaced_andn_rcx",
        "non_r8_store_rcx",
        "branch_jmp",
        "seg_prefix_andn",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_bmi1_logic_aot.py"
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
                translate(patched, ["op_andn64_flags"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_andn64_flags", str(out_cpp)],
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
            [sys.executable, "-B", str(lifter_script), str(original_path), "op_andn32", str(bad_dir)],
            capture_output=True,
            text=True,
        )
        if proc_dir.returncode != 1 or "Traceback" in proc_dir.stderr or "directory" not in proc_dir.stderr:
            raise AssertionError(f"CLI dir rejection unexpected: rc={proc_dir.returncode}, stderr={proc_dir.stderr!r}")

    print(
        f"rejections passed: {len(alt_accept)} alt + {len(unit_cases)} IR + "
        f"{len(raw_cases)} raw + {len(pe_labels)} PE-patched + CLI"
    )


# --- Positive, altered, mutant, and /O2 verification ---

def verify_suite(fns, label, pairs):
    total = 0

    # 1) Register forms (32-bit zero-ext + 64-bit).
    for op in ("andn", "bextr", "tzcnt"):
        for bits in (32, 64):
            fn = fns[f"op_{op}{bits}"]
            for a, b in pairs:
                want, _ = ref_bmi1_logic(op, a, b, bits)
                got = fn(a, b, 0)
                if got != want:
                    raise AssertionError(f"{label} op_{op}{bits}({a:#x}, {b:#x}): got {got:#x} != exp {want:#x}")
                total += 1

    # 2) Flat [rcx|rdx] memory forms.
    mem_specs = (
        ("op_andn32_mem", "andn", 32, "rcx"),
        ("op_andn64_mem", "andn", 64, "rcx"),
        ("op_bextr32_mem", "bextr", 32, "rcx"),
        ("op_bextr64_mem", "bextr", 64, "rcx"),
        ("op_tzcnt32_mem", "tzcnt", 32, "rdx"),
        ("op_tzcnt64_mem", "tzcnt", 64, "rdx"),
    )
    for name, op, bits, mem_reg in mem_specs:
        fn = fns[name]
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        for a, b in pairs:
            if op == "andn":
                # andn dst, rdx, [rcx] -> src1=rdx (b), src2=[rcx] (a)
                buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, a & mask))
                want, _ = ref_bmi1_logic(op, b, a, bits)
                got = fn(ptr, b, 0)
            elif op == "bextr":
                # bextr dst, [rcx], rdx -> src1=[rcx] (a), src2=rdx (b)
                buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, a & mask))
                want, _ = ref_bmi1_logic(op, a, b, bits)
                got = fn(ptr, b, 0)
            else:
                # tzcnt dst, [rdx] -> src=[rdx] (a)
                buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, a & mask))
                want, _ = ref_bmi1_logic(op, a, 0, bits)
                got = fn(0, ptr, 0) if mem_reg == "rdx" else fn(ptr, 0, 0)
            read_guarded(buf, off, ln, name)
            if got != want:
                raise AssertionError(f"{label} {name}({a:#x}, {b:#x}): got {got:#x} != exp {want:#x}")
            total += 1

    # 3) Flag-packing exports.
    for op in ("andn", "bextr", "tzcnt"):
        for bits in (32, 64):
            name = f"op_{op}{bits}_flags"
            fn = fns[name]
            for a, b in pairs:
                want_val, want_flags = ref_bmi1_logic(op, a, b, bits)
                bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
                got_val = fn(a, b, po)
                got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, name))[0]
                if got_val != want_val or got_flags != want_flags:
                    raise AssertionError(
                        f"{label} {name}({a:#x}, {b:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
                    )
                total += 1

    # Memory flag-packing exports: op_bextr64_mem_flags & op_tzcnt64_mem_flags.
    for a, b in pairs:
        want_val, want_flags = ref_bmi1_logic("bextr", a, b, 64)
        ba, pa, oa, la = alloc_guarded(struct.pack("<Q", a))
        bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
        got_val = fns["op_bextr64_mem_flags"](pa, b, po)
        read_guarded(ba, oa, la, "op_bextr64_mem_flags_a")
        got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, "op_bextr64_mem_flags_out"))[0]
        if got_val != want_val or got_flags != want_flags:
            raise AssertionError(
                f"{label} op_bextr64_mem_flags({a:#x}, {b:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
            )
        total += 1

        want_val_t, want_flags_t = ref_bmi1_logic("tzcnt", a, 0, 64)
        bt, pt, ot, lt = alloc_guarded(struct.pack("<Q", a))
        bo2, po2, oo2, lo2 = alloc_guarded(b"\x5a" * 4)
        got_val_t = fns["op_tzcnt64_mem_flags"](pt, 0, po2)
        read_guarded(bt, ot, lt, "op_tzcnt64_mem_flags_a")
        got_flags_t = struct.unpack("<I", read_guarded(bo2, oo2, lo2, "op_tzcnt64_mem_flags_out"))[0]
        if got_val_t != want_val_t or got_flags_t != want_flags_t:
            raise AssertionError(
                f"{label} op_tzcnt64_mem_flags({a:#x}): val={got_val_t:#x}/{want_val_t:#x} flags={got_flags_t:#x}/{want_flags_t:#x}"
            )
        total += 1

    # 4) [r8] source-then-store and dst aliasing src.
    for i in range(len(pairs) - 1):
        a, b = pairs[i]
        c, _ = pairs[i + 1]
        ba, pa, oa, la = alloc_guarded(struct.pack("<Q", a))
        bb, pb, ob, lb = alloc_guarded(struct.pack("<Q", b))
        br, pr, or_, lr = alloc_guarded(struct.pack("<Q", c))
        got_rax = fns["op_r8_source_then_store"](pa, pb, pr)
        read_guarded(ba, oa, la, "r8_src_a")
        read_guarded(bb, ob, lb, "r8_src_b")
        got_mem = struct.unpack("<Q", read_guarded(br, or_, lr, "r8_src_out"))[0]
        v_andn = ref_bmi1_logic("andn", b, c, 64)[0]
        v_bextr = ref_bmi1_logic("bextr", a, b, 64)[0]
        v_tzcnt = ref_bmi1_logic("tzcnt", b, 0, 64)[0]
        want = (v_andn | v_bextr) ^ v_tzcnt
        if got_rax != want or got_mem != want:
            raise AssertionError(f"{label} op_r8_source_then_store: rax={got_rax:#x} mem={got_mem:#x} exp={want:#x}")
        total += 1

        got_alias = fns["op_dst_alias_src"](a, b, 0)
        r9 = ref_bmi1_logic("andn", a, b, 64)[0]
        r10 = ref_bmi1_logic("bextr", b, r9, 64)[0]
        r10 = ref_bmi1_logic("tzcnt", r10, 0, 64)[0]
        want_alias = r9 | r10
        if got_alias != want_alias:
            raise AssertionError(f"{label} op_dst_alias_src({a:#x}, {b:#x}): got={got_alias:#x} exp={want_alias:#x}")
        total += 1

    # 5) Discarded faulting m64 read must fault on NULL rdx even under /O2.
    bd, pd, od, ld = alloc_guarded(struct.pack("<Q", 0x123456789ABCDEF0))
    assert fns["op_bmi1_logic_discarded_read"](0xCAFEBABE, pd, 0) == 0xCAFEBABE
    read_guarded(bd, od, ld, "discarded_valid")
    try:
        fns["op_bmi1_logic_discarded_read"](0xCAFEBABE, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_bmi1_logic_discarded_read(NULL) dropped faulting m64 read")

    return total


def _altered_reg_expected(name, a, b):
    if name == "op_andn32":
        return ref_bmi1_logic("andn", b, a, 32)[0]
    if name == "op_andn64":
        return ref_bmi1_logic("andn", b, a, 64)[0]
    if name == "op_bextr32":
        return ref_bmi1_logic("bextr", b, a, 32)[0]
    if name == "op_bextr64":
        return ref_bmi1_logic("bextr", b, a, 64)[0]
    if name == "op_tzcnt32":
        return ref_bmi1_logic("bextr", a, a, 32)[0]
    if name == "op_tzcnt64":
        return ref_bmi1_logic("bextr", a, a, 64)[0]
    raise ValueError(name)


def _altered_mem_expected(name, a, b):
    if name == "op_andn32_mem":
        return ref_bmi1_logic("bextr", a, b, 32)[0]
    if name == "op_andn64_mem":
        return ref_bmi1_logic("bextr", a, b, 64)[0]
    if name == "op_bextr32_mem":
        return ref_bmi1_logic("andn", b, a, 32)[0]
    if name == "op_bextr64_mem":
        return ref_bmi1_logic("andn", b, a, 64)[0]
    if name == "op_tzcnt32_mem":
        return ref_bmi1_logic("bextr", a, b, 32)[0]
    if name == "op_tzcnt64_mem":
        return ref_bmi1_logic("bextr", a, b, 64)[0]
    raise ValueError(name)


def check_altered_and_mutants(lifted_fns, alt_lifted_fns, pairs):
    diverged = 0
    sample = pairs
    for name in ALTERED_REG_EXPORTS:
        diffs = 0
        for a, b in sample:
            want_alt = _altered_reg_expected(name, a, b)
            got_al = alt_lifted_fns[name](a, b, 0)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({a:#x}, {b:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != lifted_fns[name](a, b, 0)
        if not diffs:
            raise AssertionError(f"ALTER_BMI1_LOGIC failed to diverge on {name}")
        diverged += 1

    for name in ALTERED_MEM_EXPORTS:
        bits = 32 if "32" in name else 64
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        is_tz = name.startswith("op_tzcnt")
        diffs = 0
        for a, b in sample:
            buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, a & mask))
            want_alt = _altered_mem_expected(name, a, b)
            if is_tz:
                got_al = alt_lifted_fns[name](b, ptr, 0)
                got_ol = lifted_fns[name](b, ptr, 0)
            else:
                got_al = alt_lifted_fns[name](ptr, b, 0)
                got_ol = lifted_fns[name](ptr, b, 0)
            read_guarded(buf, off, ln, name)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({a:#x}, {b:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != got_ol
        if not diffs:
            raise AssertionError(f"ALTER_BMI1_LOGIC failed to diverge on {name}")
        diverged += 1

    # Mutant models targeting SDM edge cases across ANDN, BEXTR, TZCNT.
    mutant_specs = (
        ("andn", 32, "andn_swap"),
        ("andn", 64, "andn_plain_and"),
        ("andn", 32, "drop_sf"),
        ("andn", 64, "set_of"),
        ("andn", 64, "set_cf"),
        ("andn", 32, "no_zext32"),
        ("bextr", 32, "bextr_swap_start_len"),
        ("bextr", 32, "bextr_mask_start_bits"),
        ("bextr", 64, "bextr_len_zero_all"),
        ("bextr", 64, "bextr_zf_false"),
        ("bextr", 64, "set_cf"),
        ("bextr", 32, "no_zext32"),
        ("tzcnt", 32, "tzcnt_zero_0"),
        ("tzcnt", 64, "tzcnt_zero_width_m1"),
        ("tzcnt", 64, "tzcnt_invert_cf"),
        ("tzcnt", 32, "tzcnt_zf_from_src"),
        ("tzcnt", 32, "no_zext32"),
    )
    for op, bits, quirk in mutant_specs:
        fn = lifted_fns[f"op_{op}{bits}_flags"]
        caught = 0
        for a, b in sample:
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
            got_val = fn(a, b, po)
            got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, f"mut_{op}"))[0]
            mut_val, mut_flags = ref_bmi1_logic(op, a, b, bits, quirk=quirk)
            if got_val != mut_val or got_flags != mut_flags:
                caught += 1
        if not caught:
            raise AssertionError(f"mutant {quirk} on {op}{bits} survived")

    return diverged, len(mutant_specs)


def compile_o2_and_verify(original_path, cxx_path, pairs):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "bmi1_logic_aot_o2.cpp"
        dll_path = tmp / "bmi1_logic_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'bmi1_logic_aot_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_suite(o2_fns, "lifted_o2", pairs)
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

    pairs = test_pairs()
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_suite(lifted_fns, "lifted", pairs)
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_lifted_fns, pairs)
        o2_cases = compile_o2_and_verify(original_path, cxx_path, pairs)

        if not host_has_bmi1(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{diverged} ALTER_BMI1_LOGIC + {mutants} mutants); "
                "Host lacks CPUID.07H.00H:EBX[3] (BMI1), skipping native differential check"
            )
            sys.exit(77)

        # 2) Native BMI1 hardware differential checks.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_suite(orig_fns, "native", pairs)
            for name in ALTERED_REG_EXPORTS:
                for a, b in pairs[:32]:
                    got_an = alt_fns[name](a, b, 0)
                    got_al = alt_lifted_fns[name](a, b, 0)
                    if got_an != got_al:
                        raise AssertionError(f"altered native vs lifted {name}({a:#x}, {b:#x}): {got_an:#x} != {got_al:#x}")
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
        f"bmi1_logic_aot passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; ALTER_BMI1_LOGIC diverged on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
