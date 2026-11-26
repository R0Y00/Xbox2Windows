"""Differential, /O2, flag-preservation, mutant, and rejection checks for cpu_movbe_aot."""

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
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
from cpu_movbe_aot import lift_code, translate  # noqa: E402

EXPORTS = (
    "op_movbe_load16", "op_movbe_load16_disp",
    "op_movbe_load32", "op_movbe_load32_disp",
    "op_movbe_load64", "op_movbe_load64_disp",
    "op_movbe_store16", "op_movbe_store16_disp",
    "op_movbe_store32", "op_movbe_store32_disp",
    "op_movbe_store64", "op_movbe_store64_disp",
    "op_movbe_flags_cf_zf_of_pf",
    "op_movbe_flags_sf_of_pf",
    "op_movbe_flags_all_clear",
    "op_r8_source_then_store",
    "op_movbe_discarded_read",
)

ALTERED_EXPORTS = (
    "op_movbe_load16", "op_movbe_load16_disp",
    "op_movbe_load32", "op_movbe_load32_disp",
    "op_movbe_load64", "op_movbe_load64_disp",
    "op_movbe_store16", "op_movbe_store16_disp",
    "op_movbe_store32", "op_movbe_store32_disp",
    "op_movbe_store64", "op_movbe_store64_disp",
)

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))


# --- Independent Python reference model ---

def ref_movbe_load(buf, offset, width, prior_reg, *, quirk=None):
    raw = buf[offset:offset + width // 8]
    val = int.from_bytes(raw, "little" if quirk == "noswap" else "big")
    if width == 16:
        if quirk == "clear16":
            return val
        return (prior_reg & ~MASK16) | val
    if width == 32:
        if quirk == "no_zext32":
            return val | 0xFFFFFFFF00000000
        return val
    return val


def ref_movbe_store(val, width, *, quirk=None):
    mask = (1 << width) - 1
    return (val & mask).to_bytes(width // 8, "little" if quirk == "noswap" else "big")


def test_inputs():
    mem_blocks = [
        bytes(range(0x10, 0x30)),
        b"\x00" * 32,
        b"\xff" * 32,
        bytes((0x11 * k + 0x5A) & 0xFF for k in range(32)),
        struct.pack("<4Q", 0x0102030405060708, 0x8000000000000001, 0xDEADBEEFCAFEF00D, 0x00FF00FF00FF00FF),
    ]
    regs = [
        0x0000000000000000,
        0xFFFFFFFFFFFFFFFF,
        0xDEADBEEF12345678,
        0x0102030405060708,
        0x8000000000000001,
        0x00000000CAFEBABE,
    ]
    rng = random.Random(0x40DBE07)
    for _ in range(8):
        mem_blocks.append(rng.randbytes(32))
        regs.append(rng.getrandbits(64))
    return mem_blocks, regs


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


def host_has_movbe(lifted_lib):
    if hasattr(lifted_lib, "host_has_movbe"):
        lifted_lib.host_has_movbe.argtypes = []
        lifted_lib.host_has_movbe.restype = ctypes.c_uint32
        return bool(lifted_lib.host_has_movbe())
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
    mov_r9_rdx = Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R9, R.RDX)

    # Alternate encodings accepted cleanly.
    alt_accept = [
        ("movbe_r11w_r8_disp12", (Instruction.create_reg_reg(Code.MOV_R64_RM64, R.R11, R.RDX),
            Instruction.create_reg_mem(Code.MOVBE_R16_M16, R.R11W, MemoryOperand(R.R8, displ=12, displ_size=1)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R11), ret),
         "const word ld_1003_1 = static_cast<word>(bswap16(load_always<std::uint16_t>((r8 + 0xcull))));"),
        ("movbe_m32_r10d_disp16", (Instruction.create_reg_reg(Code.MOV_R32_RM32, R.R10D, R.ECX),
            Instruction.create_mem_reg(Code.MOVBE_M32_R32, MemoryOperand(R.R8, displ=16, displ_size=1), R.R10D),
            mov_rax_rcx, ret),
         "store_always<std::uint32_t>((r8 + 0x10ull), bswap32(static_cast<std::uint32_t>((r10 & 0xFFFFFFFFull))));"),
        ("movbe_r9_rdx_disp64", (Instruction.create_reg_mem(
            Code.MOVBE_R64_M64, R.R9, MemoryOperand(R.RDX, displ=64, displ_size=1)),
            Instruction.create_reg_reg(Code.MOV_R64_RM64, R.RAX, R.R9), ret),
         "const word ld_1000_1 = static_cast<word>(bswap64(load_always<std::uint64_t>((rdx + 0x40ull))));"),
    ]
    for label, seq, expected in alt_accept:
        src, count = lift_code(label, 0x1000, encode_seq(*seq))
        assert f"lifted_{label}" in src and count == len(seq), (label, count)
        if expected not in src:
            raise AssertionError(f"alt {label}: expected {expected!r}")

    seg_movbe = Instruction.create_reg_mem(Code.MOVBE_R64_M64, R.RAX, MemoryOperand(R.RCX))
    seg_movbe.segment_prefix = R.GS
    evex_ins = Instruction.create_reg_reg_reg(Code.EVEX_VPADDD_XMM_K1Z_XMM_XMMM128B32, R.XMM0, R.XMM1, R.XMM2)
    evex_ins.op_mask = R.K1

    unit_cases = [
        # Stores outside [r8]
        ("store_rcx_movbe16", (Instruction.create_mem_reg(Code.MOVBE_M16_R16, MemoryOperand(R.RCX), R.DX), mov_rax_rcx, ret),
         "non-r8 store memory operand"),
        ("store_rdx_movbe32", (Instruction.create_mem_reg(Code.MOVBE_M32_R32, MemoryOperand(R.RDX, displ=4, displ_size=1), R.ECX), mov_rax_rcx, ret),
         "non-r8 store memory operand"),
        ("store_rcx_movbe64", (Instruction.create_mem_reg(Code.MOVBE_M64_R64, MemoryOperand(R.RCX), R.RDX), mov_rax_rcx, ret),
         "non-r8 store memory operand"),
        ("store_rcx_mov64", (mov_rax_rcx, Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RCX), R.RAX), ret),
         "non-r8 store memory operand"),
        ("store_rdx_setb", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R9D, R.R9D),
            Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.RDX)), mov_rax_rcx, ret),
         "non-r8 store memory operand"),
        # Indexed / scaled / RIP-relative / bad base / out-of-range displacement
        ("indexed_movbe64", (Instruction.create_reg_mem(
            Code.MOVBE_R64_M64, R.RAX, MemoryOperand(R.RCX, R.RDX, 4)), ret),
         "SIB index memory operand"),
        ("scaled_store_movbe", (Instruction.create_mem_reg(
            Code.MOVBE_M32_R32, MemoryOperand(R.R8, R.RCX, 2), R.EDX), mov_rax_rcx, ret),
         "SIB index memory operand"),
        ("rip_rel_movbe", (Instruction.create_reg_mem(
            Code.MOVBE_R64_M64, R.RAX, MemoryOperand(R.RIP, displ=0x1020, displ_size=4)), ret),
         "RIP-relative memory operand"),
        ("bad_base_rbx", (Instruction.create_reg_mem(Code.MOVBE_R64_M64, R.RAX, MemoryOperand(R.RBX)), ret),
         "bad base register in memory operand"),
        ("bad_base_rsp", (Instruction.create_reg_mem(Code.MOVBE_R32_M32, R.EAX, MemoryOperand(R.RSP)), ret),
         "bad base register in memory operand"),
        ("neg_disp_movbe", (Instruction.create_reg_mem(
            Code.MOVBE_R32_M32, R.EAX, MemoryOperand(R.RCX, displ=-4, displ_size=1)), ret),
         "out-of-range displaced memory operand"),
        ("large_disp_movbe", (Instruction.create_reg_mem(
            Code.MOVBE_R64_M64, R.RAX, MemoryOperand(R.RCX, displ=128, displ_size=8)), ret),
         "out-of-range displaced memory operand"),
        ("large_disp_store", (Instruction.create_mem_reg(
            Code.MOVBE_M64_R64, MemoryOperand(R.R8, displ=65, displ_size=1), R.RDX), mov_rax_rcx, ret),
         "out-of-range displaced memory operand"),
        # Nonvolatile GPRs, arg GPR writes, high-byte GPRs
        ("nonvol_rbx_dst", (Instruction.create_reg_mem(Code.MOVBE_R64_M64, R.RBX, MemoryOperand(R.RCX)), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_rsi_src", (Instruction.create_mem_reg(Code.MOVBE_M64_R64, MemoryOperand(R.R8), R.RSI), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("nonvol_r12_dst", (Instruction.create_reg_mem(Code.MOVBE_R32_M32, R.R12D, MemoryOperand(R.RCX)), mov_rax_rcx, ret),
         "nonvolatile GPR"),
        ("write_arg_rcx", (Instruction.create_reg_mem(Code.MOVBE_R64_M64, R.RCX, MemoryOperand(R.RDX)), mov_rax_rcx, ret),
         "cannot write argument GPR rcx"),
        ("write_arg_edx", (Instruction.create_reg_mem(Code.MOVBE_R32_M32, R.EDX, MemoryOperand(R.RCX)), mov_rax_rcx, ret),
         "cannot write argument GPR rdx"),
        ("write_arg_r8w", (Instruction.create_reg_mem(Code.MOVBE_R16_M16, R.R8W, MemoryOperand(R.RCX)), mov_rax_rcx, ret),
         "cannot write argument GPR r8"),
        ("high_byte_ah", (mov_rax_rcx, Instruction.create_reg_u32(Code.MOV_R8_IMM8, R.AH, 1), ret),
         "unsupported high-byte GPR"),
        # Undefined GPRs / flags
        ("undef_r9_movbe16_dst", (Instruction.create_reg_mem(Code.MOVBE_R16_M16, R.R9W, MemoryOperand(R.RCX)), mov_rax_rcx, ret),
         "undefined GPR r9"),
        ("undef_r10_movbe_store", (Instruction.create_mem_reg(Code.MOVBE_M64_R64, MemoryOperand(R.R8), R.R10), mov_rax_rcx, ret),
         "undefined GPR r10"),
        ("undef_rax_ret", (mov_r9_rdx, ret), "return with undefined RAX"),
        ("undef_flags_setb", (mov_rax_rcx, Instruction.create_mem(Code.SETB_RM8, MemoryOperand(R.R8)), ret),
         "undefined flag cf read"),
        # Out-of-scope / VEX / EVEX / XOP / prefixes / branches / ret
        ("bswap_rejected", (mov_rax_rcx, Instruction.create_reg(Code.BSWAP_R64, R.RAX), ret),
         "unsupported instruction"),
        ("cmovbe_rejected", (Instruction.create_reg_reg(Code.XOR_R32_RM32, R.R9D, R.R9D),
            Instruction.create_reg_reg(Code.CMOVBE_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported instruction"),
        ("vex_blsi", (Instruction.create_reg_reg(Code.VEX_BLSI_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported VEX instruction"),
        ("evex_vpaddd", (evex_ins, mov_rax_rcx, ret), "unsupported EVEX instruction"),
        ("xop_blsic", (Instruction.create_reg_reg(Code.XOP_BLSIC_R64_RM64, R.RAX, R.RCX), ret),
         "unsupported XOP instruction"),
        ("seg_prefix_movbe", (seg_movbe, ret), "unsupported instruction prefix"),
        ("branch_jmp", (mov_rax_rcx, Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret),
         "non-leaf branch or call"),
        ("missing_ret", (mov_rax_rcx,), "missing terminal ret"),
        ("trailing_after_ret", (mov_rax_rcx, ret, mov_r9_rdx), "unreachable instruction after ret"),
    ]

    raw_cases = [
        ("movbe_reg_reg_ud", bytes.fromhex("0f38f0c1") + encode_seq(ret), "invalid instruction"),
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
        entry = export_rva(pe, "op_movbe_flags_cf_zf_of_pf")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(i for i in range(0, len(records), 12) if struct.unpack_from("<I", records, i)[0] == entry)
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        max_len = struct.unpack_from("<I", records, rec_idx + 4)[0] - entry

    by_label = {label: (encode_seq(*seq), reason) for label, seq, reason in unit_cases}
    pe_labels = (
        "store_rcx_movbe16",
        "store_rcx_movbe64",
        "indexed_movbe64",
        "rip_rel_movbe",
        "neg_disp_movbe",
        "large_disp_movbe",
        "nonvol_rbx_dst",
        "write_arg_rcx",
        "bswap_rejected",
        "branch_jmp",
        "seg_prefix_movbe",
    )
    lifter_script = Path(__file__).resolve().parents[2] / "tools" / "cpu_movbe_aot.py"
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
                translate(patched, ["op_movbe_flags_cf_zf_of_pf"])
            except ValueError as exc:
                if reason not in str(exc):
                    raise AssertionError(f"pe_{label}: expected {reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"pe_{label}: PE translate accepted rejected patch")

        proc = subprocess.run(
            [sys.executable, "-B", str(lifter_script), str(patched), "op_movbe_flags_cf_zf_of_pf", str(out_cpp)],
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
            [sys.executable, "-B", str(lifter_script), str(original_path), "op_movbe_load64", str(bad_dir)],
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

def verify_suite(fns, label, mem_blocks, regs):
    total = 0

    # 1) Load forms (16-bit preserves [63:16], 32-bit zero-extends, 64-bit full, + disp).
    load_specs = (
        ("op_movbe_load16", 16, 0, "rcx"),
        ("op_movbe_load16_disp", 16, 3, "rcx"),
        ("op_movbe_load32", 32, 0, "rcx"),
        ("op_movbe_load32_disp", 32, 5, "rdx"),
        ("op_movbe_load64", 64, 0, "rcx"),
        ("op_movbe_load64_disp", 64, 8, "rdx"),
    )
    for name, width, disp, reg in load_specs:
        fn = fns[name]
        for blk in mem_blocks:
            for r in regs:
                buf, ptr, off, ln = alloc_guarded(blk)
                got = fn(ptr, r, 0) if reg == "rcx" else fn(r, ptr, 0)
                read_guarded(buf, off, ln, name)
                want = ref_movbe_load(blk, disp, width, r)
                if got != want:
                    raise AssertionError(f"{label} {name}: got={got:#x} != exp={want:#x}")
                total += 1

    # 2) Store forms strictly to [r8] / [r8+disp].
    store_specs = (
        ("op_movbe_store16", 16, 0),
        ("op_movbe_store16_disp", 16, 2),
        ("op_movbe_store32", 32, 0),
        ("op_movbe_store32_disp", 32, 4),
        ("op_movbe_store64", 64, 0),
        ("op_movbe_store64_disp", 64, 8),
    )
    for name, width, disp in store_specs:
        fn = fns[name]
        nbytes = width // 8
        for r in regs:
            fill = b"\x5a" * 32
            bo, po, oo, lo = alloc_guarded(fill)
            got_rax = fn(0, r, po)
            out_mem = read_guarded(bo, oo, lo, name)
            want_rax = (r & MASK32) if width == 32 else r
            want_bytes = ref_movbe_store(r, width)
            expected_buf = fill[:disp] + want_bytes + fill[disp + nbytes:]
            if got_rax != want_rax or out_mem != expected_buf:
                raise AssertionError(f"{label} {name}({r:#x}): rax={got_rax:#x}/{want_rax:#x} mem={out_mem.hex()}")
            total += 1

    # 3) Flag preservation checks (CF, ZF, SF, OF, PF untouched by MOVBE).
    flag_specs = (
        ("op_movbe_flags_cf_zf_of_pf", 32, 0, (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4)),
        ("op_movbe_flags_sf_of_pf", 64, 8, (1 << 2) | (1 << 3) | (1 << 4)),
        ("op_movbe_flags_all_clear", 16, 0, 0),
    )
    for name, width, disp, want_flags in flag_specs:
        fn = fns[name]
        nbytes = width // 8
        for blk in mem_blocks[:5]:
            ba, pa, oa, la = alloc_guarded(blk)
            bo, po, oo, lo = alloc_guarded(b"\x5a" * 24)
            got_rax = fn(pa, 0, po)
            read_guarded(ba, oa, la, f"{name}_in")
            out_mem = read_guarded(bo, oo, lo, f"{name}_out")
            got_flags = struct.unpack_from("<I", out_mem, 16)[0]
            want_rax = ref_movbe_load(blk, 0, width, MASK64)
            if got_rax != want_rax or got_flags != want_flags or out_mem[disp:disp + nbytes] != blk[:nbytes]:
                raise AssertionError(
                    f"{label} {name}: rax={got_rax:#x}/{want_rax:#x} flags={got_flags:#x}/{want_flags:#x}"
                )
            total += 1

    # 4) [r8] source-then-store.
    for i in range(len(mem_blocks) - 1):
        a = mem_blocks[i]
        b = mem_blocks[i + 1]
        c = mem_blocks[(i + 2) % len(mem_blocks)]
        ba, pa, oa, la = alloc_guarded(a)
        bb, pb, ob, lb = alloc_guarded(b)
        br, pr, or_, lr = alloc_guarded(c)
        got_rax = fns["op_r8_source_then_store"](pa, pb, pr)
        read_guarded(ba, oa, la, "r8_src_a")
        read_guarded(bb, ob, lb, "r8_src_b")
        out_r8 = read_guarded(br, or_, lr, "r8_src_out")
        want_rax = ref_movbe_load(b, 0, 16, pb)
        if got_rax != want_rax or out_r8[0:4] != a[0:4] or out_r8[8:16] != c[0:8]:
            raise AssertionError(f"{label} op_r8_source_then_store mismatch")
        total += 1

    # 5) Discarded faulting m64 read must fault on NULL rdx even under /O2.
    bd, pd, od, ld = alloc_guarded(mem_blocks[0])
    assert fns["op_movbe_discarded_read"](0xCAFEBABE, pd, 0) == 0xCAFEBABE
    read_guarded(bd, od, ld, "discarded_valid")
    try:
        fns["op_movbe_discarded_read"](0xCAFEBABE, 0, 0)
    except OSError as exc:
        if "access violation reading" not in str(exc).lower():
            raise AssertionError(f"{label} discarded_read(NULL) unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"{label} op_movbe_discarded_read(NULL) dropped faulting m64 read")

    return total


def _altered_load_expected(name, blk, r):
    if name == "op_movbe_load16":
        return ref_movbe_load(blk, 0, 16, r, quirk="noswap")
    if name == "op_movbe_load16_disp":
        return ref_movbe_load(blk, 1, 16, r)
    if name == "op_movbe_load32":
        return ref_movbe_load(blk, 0, 32, r, quirk="noswap")
    if name == "op_movbe_load32_disp":
        return ref_movbe_load(blk, 1, 32, r)
    if name == "op_movbe_load64":
        return ref_movbe_load(blk, 0, 64, r, quirk="noswap")
    if name == "op_movbe_load64_disp":
        return ref_movbe_load(blk, 4, 64, r)
    raise KeyError(name)


def check_altered_and_mutants(lifted_fns, alt_lifted_fns, mem_blocks, regs):
    diverged = 0
    for name in ALTERED_EXPORTS:
        diffs = 0
        if "load" in name:
            is_rdx = name in ("op_movbe_load32_disp", "op_movbe_load64_disp")
            for blk in mem_blocks[:4]:
                for r in regs[:4]:
                    buf, ptr, off, ln = alloc_guarded(blk)
                    got_al = alt_lifted_fns[name](r if is_rdx else ptr, ptr if is_rdx else r, 0)
                    got_ol = lifted_fns[name](r if is_rdx else ptr, ptr if is_rdx else r, 0)
                    read_guarded(buf, off, ln, name)
                    want_alt = _altered_load_expected(name, blk, r)
                    if got_al != want_alt:
                        raise AssertionError(f"altered_lifted {name}: {got_al:#x} != {want_alt:#x}")
                    diffs += got_al != got_ol
        else:
            width = 16 if "16" in name else (32 if "32" in name else 64)
            disp = {"op_movbe_store16": 0, "op_movbe_store16_disp": 4,
                    "op_movbe_store32": 0, "op_movbe_store32_disp": 8,
                    "op_movbe_store64": 0, "op_movbe_store64_disp": 16}[name]
            quirk = None if "disp" in name else "noswap"
            for r in regs[:6]:
                fill = b"\x5a" * 32
                bo_a, po_a, oo_a, lo_a = alloc_guarded(fill)
                bo_o, po_o, oo_o, lo_o = alloc_guarded(fill)
                alt_lifted_fns[name](0, r, po_a)
                lifted_fns[name](0, r, po_o)
                mem_a = read_guarded(bo_a, oo_a, lo_a, f"alt_{name}")
                mem_o = read_guarded(bo_o, oo_o, lo_o, f"orig_{name}")
                want_bytes = ref_movbe_store(r, width, quirk=quirk)
                expected = fill[:disp] + want_bytes + fill[disp + width // 8:]
                if mem_a != expected:
                    raise AssertionError(f"altered_lifted {name}: {mem_a.hex()} != {expected.hex()}")
                diffs += mem_a != mem_o
        if not diffs:
            raise AssertionError(f"ALTER_MOVBE failed to diverge on {name}")
        diverged += 1

    # Mutant models (noswap, clear16, drop_zext32, clobber_flags).
    mutants = 0
    blk0 = mem_blocks[0]
    ba, pa, oa, la = alloc_guarded(blk0)
    assert lifted_fns["op_movbe_load16"](pa, 0xDEADBEEF12345678, 0) != ref_movbe_load(blk0, 0, 16, 0xDEADBEEF12345678, quirk="noswap")
    mutants += 1
    assert lifted_fns["op_movbe_load16"](pa, 0xDEADBEEF12345678, 0) != ref_movbe_load(blk0, 0, 16, 0xDEADBEEF12345678, quirk="clear16")
    mutants += 1
    assert lifted_fns["op_movbe_load32"](pa, 0, 0) != ref_movbe_load(blk0, 0, 32, 0, quirk="no_zext32")
    mutants += 1
    bo, po, oo, lo = alloc_guarded(b"\x5a" * 24)
    lifted_fns["op_movbe_flags_cf_zf_of_pf"](pa, 0, po)
    got_flags = struct.unpack_from("<I", read_guarded(bo, oo, lo, "mut_flags"), 16)[0]
    if got_flags == 0:
        raise AssertionError("mutant clobber_flags survived")
    mutants += 1

    return diverged, mutants


def compile_o2_and_verify(original_path, cxx_path, mem_blocks, regs):
    source_text, _ = translate(original_path, list(EXPORTS))
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        cpp_path = tmp / "movbe_aot_o2.cpp"
        dll_path = tmp / "movbe_aot_o2.dll"
        cpp_path.write_text(source_text, encoding="utf-8")
        proc = subprocess.run(
            [str(cxx_path), "/nologo", "/O2", "/W4", "/WX", "/LD", str(cpp_path),
             f"/Fo{tmp / 'movbe_aot_o2.obj'}", f"/Fe{dll_path}"],
            capture_output=True,
            text=True,
            errors="replace",
        )
        if proc.returncode != 0:
            raise AssertionError(f"/O2 compile failed:\n{proc.stdout}\n{proc.stderr}")
        o2_lib, o2_fns = bind_exports(dll_path, "lifted_")
        try:
            return verify_suite(o2_fns, "lifted_o2", mem_blocks, regs)
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

    mem_blocks, regs = test_inputs()
    lifted_lib, lifted_fns = bind_exports(lifted_path, "lifted_")
    alt_lifted_lib, alt_lifted_fns = bind_exports(altered_lifted_path, "lifted_")
    try:
        lifted_cases = verify_suite(lifted_fns, "lifted", mem_blocks, regs)
        diverged, mutants = check_altered_and_mutants(lifted_fns, alt_lifted_fns, mem_blocks, regs)
        o2_cases = compile_o2_and_verify(original_path, cxx_path, mem_blocks, regs)

        if not host_has_movbe(lifted_lib):
            print(
                f"portable lifted passed ({lifted_cases} lifted + {o2_cases} /O2 cases, "
                f"{diverged} ALTER_MOVBE + {mutants} mutants); "
                "Host lacks CPUID.01H:ECX[22] (MOVBE), skipping native differential check"
            )
            sys.exit(77)

        # 2) Native MOVBE hardware differential checks.
        orig_lib, orig_fns = bind_exports(original_path)
        alt_lib, alt_fns = bind_exports(altered_path)
        try:
            native_cases = verify_suite(orig_fns, "native", mem_blocks, regs)
            for name in ALTERED_EXPORTS:
                if "load" in name:
                    is_rdx = name in ("op_movbe_load32_disp", "op_movbe_load64_disp")
                    for blk in mem_blocks[:4]:
                        for r in regs[:4]:
                            buf, ptr, off, ln = alloc_guarded(blk)
                            got_an = alt_fns[name](r if is_rdx else ptr, ptr if is_rdx else r, 0)
                            got_al = alt_lifted_fns[name](r if is_rdx else ptr, ptr if is_rdx else r, 0)
                            read_guarded(buf, off, ln, f"alt_nat_{name}")
                            if got_an != got_al:
                                raise AssertionError(f"altered native vs lifted {name}: {got_an:#x} != {got_al:#x}")
                else:
                    for r in regs[:4]:
                        bn, pn, on, ln = alloc_guarded(b"\x5a" * 32)
                        bl, pl, ol, ll = alloc_guarded(b"\x5a" * 32)
                        alt_fns[name](0, r, pn)
                        alt_lifted_fns[name](0, r, pl)
                        if read_guarded(bn, on, ln, f"alt_nat_{name}") != read_guarded(bl, ol, ll, f"alt_lift_{name}"):
                            raise AssertionError(f"altered native vs lifted store {name}")
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
        f"movbe_aot passed: {native_cases} native + {lifted_cases} lifted + {o2_cases} /O2 cases "
        f"across {len(EXPORTS)} exports; ALTER_MOVBE diverged on {diverged} exports + {mutants} mutants caught"
    )


if __name__ == "__main__":
    main()
