"""Differential, /O2, dual-carry flag-isolation, mutant, and rejection checks for cpu_adx_aot (ADCX, ADOX)."""

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
from cpu_adx_aot import lift_code, translate  # noqa: E402
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402

EXPORTS = (
    "op_adcx32_c0", "op_adcx32_c1", "op_adcx64_c0", "op_adcx64_c1",
    "op_adox32_o0", "op_adox32_o1", "op_adox64_o0", "op_adox64_o1",
    "op_adcx32_mem", "op_adcx64_mem", "op_adox32_mem", "op_adox64_mem",
    "op_adcx32_flags", "op_adcx64_flags", "op_adox32_flags", "op_adox64_flags",
    "op_adcx64_mem_flags", "op_adox64_mem_flags",
    "op_adx_dual64_adcx", "op_adx_dual64_adox",
    "op_adx_dst_alias", "op_adx_discarded_read",
)

ALTERED_REG_EXPORTS = (
    "op_adcx32_c0", "op_adcx32_c1", "op_adcx64_c0", "op_adcx64_c1",
    "op_adox32_o0", "op_adox32_o1", "op_adox64_o0", "op_adox64_o1",
)
ALTERED_MEM_EXPORTS = (
    "op_adcx32_mem", "op_adcx64_mem", "op_adox32_mem", "op_adox64_mem",
)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference model ---

def ref_adx(op, a, b, cin, bits, *, init_cf=0, init_of=0, init_zf=0, init_sf=0, quirk=None):
    """Return (dst, packed_flags) where packed_flags = (CF<<0)|(OF<<1)|(ZF<<2)|(SF<<3)."""
    mask = MASK32 if bits == 32 else MASK64
    sa = a & mask
    sb = b & mask
    carry_in = cin & 1
    if quirk == "invert_cin":
        carry_in ^= 1
    elif quirk == "drop_cin":
        carry_in = 0
    total = sa + sb + carry_in
    val = total & mask
    cout = 1 if total > mask else 0
    if quirk == "invert_cout":
        cout ^= 1

    cf, of, zf, sf = init_cf, init_of, init_zf, init_sf
    if op == "adcx":
        cf = cout
        if quirk == "clobber_of":
            of = 0
        if quirk == "clobber_zf":
            zf = int(val == 0)
        if quirk == "clobber_sf":
            sf = int((val >> (bits - 1)) & 1)
    elif op == "adox":
        of = cout
        if quirk == "clobber_cf":
            cf = 0
        if quirk == "clobber_zf":
            zf = int(val == 0)
        if quirk == "clobber_sf":
            sf = int((val >> (bits - 1)) & 1)
    else:
        raise ValueError(op)

    if quirk == "no_zext32" and bits == 32:
        val |= 0xFFFFFFFF00000000
    packed = (cf << 0) | (of << 1) | (zf << 2) | (sf << 3)
    return val, packed


def test_pairs():
    singles = [
        0x0000000000000000,
        0x0000000000000001,
        0x000000007FFFFFFF,
        0x0000000080000000,
        0x00000000FFFFFFFE,
        0x00000000FFFFFFFF,
        0xDEADBEEF00000000,
        0xDEADBEEF7FFFFFFF,
        0xDEADBEEF80000000,
        0xDEADBEEFFFFFFFFF,
        0x0000000100000000,
        0x7FFFFFFFFFFFFFFF,
        0x8000000000000000,
        0xFFFFFFFFFFFFFFFE,
        0xFFFFFFFFFFFFFFFF,
        0x0123456789ABCDEF,
        0xFEDCBA9876543210,
    ]
    pairs = [
        (0x0000000000000000, 0x0000000000000000),
        (0x00000000FFFFFFFF, 0x0000000000000000),
        (0x00000000FFFFFFFF, 0x0000000000000001),
        (0x00000000FFFFFFFF, 0x00000000FFFFFFFF),
        (0x000000007FFFFFFF, 0x0000000000000001),
        (0x0000000080000000, 0x0000000080000000),
        (0xFFFFFFFFFFFFFFFF, 0x0000000000000000),
        (0xFFFFFFFFFFFFFFFF, 0x0000000000000001),
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0x7FFFFFFFFFFFFFFF, 0x0000000000000001),
        (0x8000000000000000, 0x8000000000000000),
    ]
    for i, a in enumerate(singles):
        pairs.append((a, singles[(i * 5 + 3) % len(singles)]))
    rng = random.Random(0xAD70A07)
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


def host_has_adx(lifted_lib):
    if hasattr(lifted_lib, "host_has_adx"):
        lifted_lib.host_has_adx.argtypes = []
        lifted_lib.host_has_adx.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_adx())
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
    stc = Instruction.create(Code.STC)
    adcx_rax_rdx = Instruction.create_reg_reg(Code.ADCX_R64_RM64, R.RAX, R.RDX)
    adox_rax_rdx = Instruction.create_reg_reg(Code.ADOX_R64_RM64, R.RAX, R.RDX)

    # Alternate encodings accepted cleanly.
    alt_accept = [
        ("adcx32_r11_r8mem", (xor_r9, Instruction.create_reg_reg(Code.MOV_R32_RM32, R.R11D, R.EDX),
            Instruction.create_reg_mem(Code.ADCX_R32_RM32, R.R11D, MemoryOperand(R.R8)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11), ret),
         "r11 = op_adcx32((r11 & 0xFFFFFFFFull), ld_"),
        ("adox64_r10_rcxmem", (xor_r9, Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R10, R.RDX),
            Instruction.create_reg_mem(Code.ADOX_R64_RM64, R.R10, MemoryOperand(R.RCX)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R10), ret),
         "r10 = op_adox64(r10, ld_"),
        ("adcx64_rax_rdx_r8", (stc, mov_rax_rcx, adcx_rax_rdx, ret),
         "rax = op_adcx64(rax, rdx, flags);"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r} in {src}")

    seg_adcx = Instruction.create_reg_mem(Code.ADCX_R64_RM64, R.RAX, MemoryOperand(R.RDX))
    seg_adcx.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # Undefined CF / OF reads
        ("undef_cf_adcx_entry", (mov_rax_rcx, adcx_rax_rdx, ret), "undefined flag cf read"),
        ("undef_of_adox_entry", (mov_rax_rcx, adox_rax_rdx, ret), "undefined flag of read"),
        ("undef_of_after_stc", (stc, mov_rax_rcx, adox_rax_rdx, ret), "undefined flag of read"),
        ("undef_of_after_shl2", (xor_r9, Instruction.create_reg_u32(Code.SHL_RM32_IMM8, R.R9D, 2),
            mov_rax_rcx, adox_rax_rdx, ret), "undefined flag of read"),
        ("undef_flags_seto_before", (mov_rax_rcx, Instruction.create_reg(Code.SETO_RM8, R.R9L), ret),
         "undefined flag of read"),
        # Unsupported instructions (ADC, SBB, MULX, BMI1, VEX/EVEX/XOP, LAHF/PUSHFQ)
        ("adc_r64_r64", (xor_r9, mov_rax_rcx, Instruction.create_reg_reg(Code.ADC_R64_RM64, R.RAX, R.RDX), ret),
         "unsupported instruction"),
        ("sbb_r32_r32", (xor_r9, mov_rax_rcx, Instruction.create_reg_reg(Code.SBB_R32_RM32, R.EAX, R.EDX), ret),
         "unsupported instruction"),
        ("vex_mulx64", (Instruction.create_reg_reg_reg(Code.VEX_MULX_R64_R64_RM64, R.RAX, R.R9, R.RCX), ret),
         "unsupported VEX instruction"),
        ("vex_andn32", (Instruction.create_reg_reg_reg(Code.VEX_ANDN_R32_R32_RM32, R.EAX, R.ECX, R.EDX), ret),
         "unsupported VEX instruction"),
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_bextr32", (Instruction.create_reg_reg_u32(Code.XOP_BEXTR_R32_RM32_IMM32, R.EAX, R.ECX, 0x0804), ret),
         "unsupported XOP instruction"),
        ("lahf_after_adcx", (xor_r9, mov_rax_rcx, adcx_rax_rdx, Instruction.create(Code.LAHF), ret),
         "unsupported instruction"),
        ("pushfq", (xor_r9, mov_rax_rcx, Instruction.create(Code.PUSHFQ), ret), "unsupported instruction"),
        # Nonvolatile GPRs, arg GPR writes, high-byte GPRs, undefined GPRs
        ("nonvol_rbx_dst", (xor_r9, Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RBX, R.RCX),
            Instruction.create_reg_reg(Code.ADCX_R64_RM64, R.RBX, R.RDX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (xor_r9, mov_rax_rcx, Instruction.create_reg_reg(Code.ADOX_R64_RM64, R.RAX, R.RSI), ret),
         "nonvolatile GPR"),
        ("nonvol_r12_dst", (xor_r9, Instruction.create_reg_reg(Code.ADOX_R32_RM32, R.R12D, R.ECX), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("write_arg_rcx", (xor_r9, Instruction.create_reg_reg(Code.ADCX_R64_RM64, R.RCX, R.RDX), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_edx", (xor_r9, Instruction.create_reg_reg(Code.ADOX_R32_RM32, R.EDX, R.ECX), mov_rax_rcx, ret),
         "cannot write argument GPR rdx"),
        ("write_arg_r8", (xor_r9, Instruction.create_reg_reg(Code.ADCX_R64_RM64, R.R8, R.RCX), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        ("high_byte_ah", (mov_rax_rcx, Instruction.create_reg_u32(Code.MOV_R8_IMM8, R.AH, 1), ret),
         "unsupported high-byte GPR"),
        ("undef_gpr_r9_dst", (xor_r9, Instruction.create_reg_reg(Code.ADCX_R64_RM64, R.R10, R.RCX), mov_rax_rcx, ret),
         "undefined GPR r10"),
        ("undef_gpr_r11_src", (xor_r9, mov_rax_rcx, Instruction.create_reg_reg(Code.ADOX_R64_RM64, R.RAX, R.R11), ret),
         "undefined GPR r11"),
        ("undef_rax_ret", (xor_r9, Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RCX), ret),
         "return with undefined RAX"),
        # Displaced / indexed / RIP-relative / bad-base memory & stores outside [r8]
        ("displaced_adcx_rcx", (xor_r9, mov_rax_rcx, Instruction.create_reg_mem(
            Code.ADCX_R64_RM64, R.RAX, MemoryOperand(R.RCX, displ=8, displ_size=1)), ret),
         "displaced memory operand"),
        ("displaced_adox_rdx", (xor_r9, mov_rax_rcx, Instruction.create_reg_mem(
            Code.ADOX_R32_RM32, R.EAX, MemoryOperand(R.RDX, displ=4, displ_size=1)), ret),
         "displaced memory operand"),
        ("displaced_store_r8", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.R8, displ=8, displ_size=1), R.RAX), ret),
         "displaced memory operand"),
        ("indexed_adcx", (xor_r9, mov_rax_rcx, Instruction.create_reg_mem(
            Code.ADCX_R64_RM64, R.RAX, MemoryOperand(R.RCX, R.RDX, 8)), ret),
         "SIB index memory operand"),
        ("rip_rel_adox", (xor_r9, mov_rax_rcx, Instruction.create_reg_mem(
            Code.ADOX_R64_RM64, R.RAX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("bad_base_rbx", (xor_r9, mov_rax_rcx, Instruction.create_reg_mem(
            Code.ADCX_R64_RM64, R.RAX, MemoryOperand(R.RBX)), ret),
         "bad base register in memory operand"),
        ("non_r8_store_rcx", (mov_rax_rcx, Instruction.create_mem_reg(
            Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret),
         "non-r8 store memory operand"),
        ("non_r8_store_setb", (xor_r9, mov_rax_rcx, Instruction.create_mem(
            Code.SETB_RM8, MemoryOperand(R.RDX)), ret),
         "non-r8 store memory operand"),
        # Prefixes, branches, missing/trailing ret
        ("seg_prefix_adcx", (xor_r9, mov_rax_rcx, seg_adcx, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (xor_r9, mov_rax_rcx, adcx_rax_rdx), "missing terminal ret"),
        ("trailing_after_ret", (mov_rax_rcx, ret, mov_rax_rcx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("lock_prefix", bytes.fromhex("f00fb109") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
        ("rep_prefix", bytes.fromhex("f3a4") + encode_seq(mov_rax_rcx, ret), "unsupported instruction prefix"),
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
        entry = export_rva(pe, "op_adx_dual64_adcx")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in unit_cases}
    pe_labels = (
        "undef_cf_adcx_entry",
        "undef_of_adox_entry",
        "adc_r64_r64",
        "vex_mulx64",
        "evex_vpaddd",
        "xop_bextr32",
        "nonvol_rbx_dst",
        "write_arg_rcx",
        "displaced_adcx_rcx",
        "non_r8_store_rcx",
        "branch_jmp",
        "seg_prefix_adcx",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_adx_aot.py"
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
                translate(patched, ["op_adx_dual64_adcx"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_adx_dual64_adcx", str(out_cpp)],
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
            [sys.executable, "-B", str(lifter_script), str(original_path), "op_adcx32_c0", str(bad_dir)],
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

def _ref_dual64(a, b, c):
    cf = 1
    of = 1
    t1 = a + c + cf
    cf = 1 if t1 > MASK64 else 0
    rax = t1 & MASK64
    t2 = b + c + of
    of = 1 if t2 > MASK64 else 0
    r11 = t2 & MASK64
    t3 = rax + b + cf
    cf = 1 if t3 > MASK64 else 0
    rax = t3 & MASK64
    t4 = r11 + a + of
    of = 1 if t4 > MASK64 else 0
    r11 = t4 & MASK64
    return rax, r11, (of << 1) | cf


def verify_suite(fns, label, pairs):
    total = 0

    # 1) Register forms (32-bit zero-ext + 64-bit, carry-in 0 and 1).
    reg_specs = (
        ("op_adcx32_c0", "adcx", 32, 0),
        ("op_adcx32_c1", "adcx", 32, 1),
        ("op_adcx64_c0", "adcx", 64, 0),
        ("op_adcx64_c1", "adcx", 64, 1),
        ("op_adox32_o0", "adox", 32, 0),
        ("op_adox32_o1", "adox", 32, 1),
        ("op_adox64_o0", "adox", 64, 0),
        ("op_adox64_o1", "adox", 64, 1),
    )
    for name, op, bits, cin in reg_specs:
        fn = fns[name]
        for a, b in pairs:
            want, _ = ref_adx(op, a, b, cin, bits)
            got = fn(a, b, 0)
            if got != want:
                raise AssertionError(f"{label} {name}({a:#x}, {b:#x}): got {got:#x} != exp {want:#x}")
            total += 1

    # 2) Flat [rdx] m32 / m64 source forms with carry-in 0 and 1.
    mem_specs = (
        ("op_adcx32_mem", "adcx", 32),
        ("op_adcx64_mem", "adcx", 64),
        ("op_adox32_mem", "adox", 32),
        ("op_adox64_mem", "adox", 64),
    )
    for name, op, bits in mem_specs:
        fn = fns[name]
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        for a, b in pairs:
            for cin in (0, 1):
                buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, b & mask))
                want, _ = ref_adx(op, a, b, cin, bits)
                got = fn(a, ptr, cin)
                read_guarded(buf, off, ln, name)
                if got != want:
                    raise AssertionError(f"{label} {name}({a:#x}, {b:#x}, {cin}): got {got:#x} != exp {want:#x}")
                total += 1

    # 3) Flag-isolation exports: verify ONLY CF (for ADCX) or ONLY OF (for ADOX) updates, others preserved.
    flag_specs = (
        ("op_adcx32_flags", "adcx", 32, 0, 1, 0, 1),  # init: CF=0, OF=1, ZF=0, SF=1
        ("op_adcx64_flags", "adcx", 64, 1, 1, 0, 1),  # init: CF=1, OF=1, ZF=0, SF=1
        ("op_adox32_flags", "adox", 32, 1, 0, 1, 0),  # init: CF=1, OF=0, ZF=1, SF=0
        ("op_adox64_flags", "adox", 64, 1, 1, 1, 0),  # init: CF=1, OF=1, ZF=1, SF=0
    )
    for name, op, bits, icf, iof, izf, isf in flag_specs:
        fn = fns[name]
        cin = icf if op == "adcx" else iof
        for a, b in pairs:
            want_val, want_flags = ref_adx(op, a, b, cin, bits, init_cf=icf, init_of=iof, init_zf=izf, init_sf=isf)
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
            got_val = fn(a, b, po)
            got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, name))[0]
            if got_val != want_val or got_flags != want_flags:
                raise AssertionError(
                    f"{label} {name}({a:#x}, {b:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
                )
            total += 1

    # [rcx] memory flag-isolation exports.
    for name, op, icf, iof, izf, isf in (
        ("op_adcx64_mem_flags", "adcx", 0, 1, 0, 1),
        ("op_adox64_mem_flags", "adox", 1, 0, 1, 0),
    ):
        fn = fns[name]
        cin = icf if op == "adcx" else iof
        for a, b in pairs:
            want_val, want_flags = ref_adx(op, b, a, cin, 64, init_cf=icf, init_of=iof, init_zf=izf, init_sf=isf)
            ba, pa, oa, la = alloc_guarded(struct.pack("<Q", a))
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
            got_val = fn(pa, b, po)
            read_guarded(ba, oa, la, f"{name}_src")
            got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, f"{name}_out"))[0]
            if got_val != want_val or got_flags != want_flags:
                raise AssertionError(
                    f"{label} {name}({a:#x}, {b:#x}): val={got_val:#x}/{want_val:#x} flags={got_flags:#x}/{want_flags:#x}"
                )
            total += 1

    # 4) Interleaved dual-carry chains ([r8] source-then-store), dst aliasing, discarded faulting read.
    for i in range(len(pairs) - 1):
        a, b = pairs[i]
        c, _ = pairs[i + 1]
        want_adcx, want_adox, want_flags = _ref_dual64(a, b, c)

        br1, pr1, or1, lr1 = alloc_guarded(struct.pack("<Q", c))
        got_adcx = fns["op_adx_dual64_adcx"](a, b, pr1)
        got_packed = struct.unpack("<Q", read_guarded(br1, or1, lr1, "dual64_adcx"))[0]
        want_packed = ((want_adox << 2) & MASK64) | want_flags
        if got_adcx != want_adcx or got_packed != want_packed:
            raise AssertionError(
                f"{label} op_adx_dual64_adcx({a:#x}, {b:#x}, {c:#x}): "
                f"adcx={got_adcx:#x}/{want_adcx:#x} packed={got_packed:#x}/{want_packed:#x}"
            )

        br2, pr2, or2, lr2 = alloc_guarded(struct.pack("<Q", c))
        got_adox = fns["op_adx_dual64_adox"](a, b, pr2)
        got_f = struct.unpack("<Q", read_guarded(br2, or2, lr2, "dual64_adox"))[0] & 0xFFFFFFFF
        if got_adox != want_adox or got_f != want_flags:
            raise AssertionError(
                f"{label} op_adx_dual64_adox({a:#x}, {b:#x}, {c:#x}): "
                f"adox={got_adox:#x}/{want_adox:#x} flags={got_f:#x}/{want_flags:#x}"
            )
        total += 2

        got_alias = fns["op_adx_dst_alias"](a, b, 0)
        want_alias = ref_adx("adcx", a, a, 1, 64)[0] ^ ref_adx("adox", b, b, 0, 64)[0]
        if got_alias != want_alias:
            raise AssertionError(f"{label} op_adx_dst_alias({a:#x}, {b:#x}): got={got_alias:#x} exp={want_alias:#x}")
        total += 1

    bd, pd, od, ld = alloc_guarded(struct.pack("<Q", 0x123456789ABCDEF0))
    assert fns["op_adx_discarded_read"](0xCAFEBABE, pd, 0) == 0xCAFEBABE
    read_guarded(bd, od, ld, "discarded_valid")
    try:
        fns["op_adx_discarded_read"](0xCAFEBABE, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_adx_discarded_read(NULL) dropped faulting m64 read")

    return total


def _altered_expected(name, a, b, cin=0):
    if name == "op_adcx32_c0":
        return ref_adx("adox", a, b, 0, 32)[0]
    if name == "op_adcx32_c1":
        return ref_adx("adox", a, b, 0, 32)[0]
    if name == "op_adcx64_c0":
        return ref_adx("adox", a, b, 0, 64)[0]
    if name == "op_adcx64_c1":
        return ref_adx("adox", a, b, 0, 64)[0]
    if name == "op_adox32_o0":
        return ref_adx("adcx", a, b, 0, 32)[0]
    if name == "op_adox32_o1":
        return ref_adx("adcx", a, b, 0, 32)[0]
    if name == "op_adox64_o0":
        return ref_adx("adcx", a, b, 0, 64)[0]
    if name == "op_adox64_o1":
        return ref_adx("adcx", a, b, 0, 64)[0]
    if name == "op_adcx32_mem":
        return ref_adx("adox", a, b, 0, 32)[0]
    if name == "op_adcx64_mem":
        return ref_adx("adox", a, b, 0, 64)[0]
    if name == "op_adox32_mem":
        return ref_adx("adcx", a, b, 0, 32)[0]
    if name == "op_adox64_mem":
        return ref_adx("adcx", a, b, 0, 64)[0]
    raise ValueError(name)


def check_altered_and_mutants(lifted_fns, alt_lifted_fns, pairs):
    diverged = 0
    for name in ALTERED_REG_EXPORTS:
        diffs = 0
        for a, b in pairs:
            want_alt = _altered_expected(name, a, b)
            got_al = alt_lifted_fns[name](a, b, 0)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({a:#x}, {b:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != lifted_fns[name](a, b, 0)
        if name.endswith(("_c1", "_o1")) and not diffs:
            raise AssertionError(f"ALTER_ADX failed to diverge on {name}")
        diverged += 1

    for name in ALTERED_MEM_EXPORTS:
        bits = 32 if "32" in name else 64
        fmt = "<I" if bits == 32 else "<Q"
        mask = MASK32 if bits == 32 else MASK64
        diffs = 0
        for a, b in pairs:
            buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, b & mask))
            want_alt = _altered_expected(name, a, b, 1)
            got_al = alt_lifted_fns[name](a, ptr, 1)
            got_ol = lifted_fns[name](a, ptr, 1)
            read_guarded(buf, off, ln, name)
            if got_al != want_alt:
                raise AssertionError(f"altered_lifted {name}({a:#x}, {b:#x}): {got_al:#x} != {want_alt:#x}")
            diffs += got_al != got_ol
        if not diffs:
            raise AssertionError(f"ALTER_ADX failed to diverge on {name}")
        diverged += 1

    mutant_specs = (
        ("adcx", 32, 0, 1, 0, 1, "invert_cin"),
        ("adcx", 64, 1, 1, 0, 1, "drop_cin"),
        ("adcx", 64, 1, 1, 0, 1, "invert_cout"),
        ("adcx", 32, 0, 1, 0, 1, "clobber_of"),
        ("adcx", 32, 0, 1, 0, 1, "clobber_zf"),
        ("adcx", 64, 1, 1, 0, 1, "clobber_sf"),
        ("adcx", 32, 0, 1, 0, 1, "no_zext32"),
        ("adox", 32, 1, 0, 1, 0, "invert_cin"),
        ("adox", 64, 1, 1, 1, 0, "drop_cin"),
        ("adox", 64, 1, 1, 1, 0, "invert_cout"),
        ("adox", 32, 1, 0, 1, 0, "clobber_cf"),
        ("adox", 32, 1, 0, 1, 0, "clobber_zf"),
        ("adox", 64, 1, 1, 1, 0, "clobber_sf"),
        ("adox", 32, 1, 0, 1, 0, "no_zext32"),
    )
    for op, bits, icf, iof, izf, isf, quirk in mutant_specs:
        fn = lifted_fns[f"op_{op}{bits}_flags"]
        cin = icf if op == "adcx" else iof
        caught = 0
        for a, b in pairs:
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 4)
            got_val = fn(a, b, po)
            got_flags = struct.unpack("<I", read_guarded(bo, oo, lo, f"mut_{op}"))[0]
            mut_val, mut_flags = ref_adx(op, a, b, cin, bits, init_cf=icf, init_of=iof, init_zf=izf, init_sf=isf, quirk=quirk)
            if got_val != mut_val or got_flags != mut_flags:
                caught += 1
        if not caught:
            raise AssertionError(f"mutant {quirk} on {op}{bits} survived")

    return diverged, len(mutant_specs)


def compile_o2_and_verify(original_path, cxx_path, pairs):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "adx_aot_o2.cpp"
        dll_path = tmp / "adx_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'adx_aot_o2.obj'}", f"/Fe{dll_path}"],
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

        if not host_has_adx(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{diverged} ALTER_ADX + {mutants} mutants); "
                "Host lacks CPUID.07H.00H:EBX[19] (ADX), skipping native differential check"
            )
            sys.exit(77)

        # 2) Native ADX hardware differential checks.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_suite(orig_fns, "native", pairs)
            for name in ALTERED_REG_EXPORTS + ALTERED_MEM_EXPORTS:
                is_mem = name.endswith("_mem")
                bits = 32 if "32" in name else 64
                fmt = "<I" if bits == 32 else "<Q"
                mask = MASK32 if bits == 32 else MASK64
                for a, b in pairs[:24]:
                    if is_mem:
                        buf, ptr, off, ln = alloc_guarded(struct.pack(fmt, b & mask))
                        got_an = alt_fns[name](a, ptr, 1)
                        got_al = alt_lifted_fns[name](a, ptr, 1)
                        read_guarded(buf, off, ln, f"alt_{name}")
                    else:
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
        f"adx_aot passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; ALTER_ADX checked on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
