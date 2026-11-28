"""Differential checks, mutated-C++ controls, ALTER_IMM controls, and rejection controls for PUSHF/PUSHFQ/POPF/POPFQ via ffx_version_query."""

import ctypes
from pathlib import Path
import random
import re
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import (
    FLAG_STACK_LIMIT,
    INTEGER_HEADER,
    flag_definedness,
    flag_stack_depths,
    lower,
    region,
)

MASK64 = 0xFFFFFFFFFFFFFFFF
GUARD_PRE = bytes(0xA0 + (i % 16) for i in range(32))
GUARD_POST = bytes(0xC0 + (i % 16) for i in range(32))
FILL = b"\x5A" * 32

# (op_kind, width_bits, push_width_bits, imm_spec)
ARITH_SPECS = {
    "op_add8_pushf": ("add", 8, 16, None),
    "op_add16_pushfq": ("add", 16, 64, None),
    "op_add32_pushfq": ("add", 32, 64, None),
    "op_add64_pushfq": ("add", 64, 64, None),
    "op_sub8_pushfq": ("sub", 8, 64, None),
    "op_sub16_pushf": ("sub", 16, 16, None),
    "op_sub32_pushfq": ("sub", 32, 64, None),
    "op_sub64_pushfq": ("sub", 64, 64, None),
    "op_cmp8_pushf": ("cmp", 8, 16, None),
    "op_cmp16_pushfq": ("cmp", 16, 64, None),
    "op_cmp32_pushf": ("cmp", 32, 16, None),
    "op_cmp64_pushfq": ("cmp", 64, 64, None),
    "op_add8_al_pushf": ("add_al", 8, 16, 5),
    "op_add32_imm_pushfq": ("add", 32, 64, (17, 1)),
    "op_sub32_imm_pushf": ("sub", 32, 16, (15, 1)),
    "op_cmp64_imm_pushfq": ("cmp", 64, 64, (17, 1)),
}

SPECIAL_EXPORTS = (
    "op_popf_lahf_setcc",
    "op_popfq_df_movsb",
    "op_pushfq_std_movsb_popfq",
    "op_nested_push_pop",
    "op_popfq_normalize",
    "op_popf_normalize",
    "op_pushf_df_nt",
)

EXPORTS = tuple(ARITH_SPECS.keys()) + SPECIAL_EXPORTS
ALL_EXPORTS = EXPORTS + ("op_discarded_read",)
ALTERED_EXPORTS = (
    "op_add8_al_pushf",
    "op_add32_imm_pushfq",
    "op_sub32_imm_pushf",
    "op_cmp64_imm_pushfq",
)

EXTRA_CONTROL_HELPERS = """
static inline bool calc_pf_full(uint64_t val) {
    val ^= val >> 32u;
    val ^= val >> 16u;
    val ^= val >> 8u;
    val ^= val >> 4u;
    val ^= val >> 2u;
    val ^= val >> 1u;
    return (val & 1ull) == 0ull;
}
"""

PARITY_V_RE = re.compile(
    r"parity = \(\(0x6996u >> \(\(static_cast<unsigned>\(static_cast<uint8_t>\(v\)\) \^ "
    r"\(static_cast<unsigned>\(static_cast<uint8_t>\(v\)\) >> 4\)\) & 0xfu\)\) & 1u\) == 0;"
)

# Mutated-C++ negative controls: (base_export, control_name, regex, replacement)
CPP_MUTATIONS = (
    ("op_add32_pushfq", "ctrl_bit1_cleared", re.compile(r"0x0202ull"), "0x0200ull"),
    ("op_cmp64_pushfq", "ctrl_if_cleared", re.compile(r"0x0202ull"), "0x0002ull"),
    ("op_add32_pushfq", "ctrl_af_bit5", re.compile(r"& 0x10u\) != 0;"), "& 0x20u) != 0;"),
    ("op_sub32_pushfq", "ctrl_pf_full_width", PARITY_V_RE, "parity = calc_pf_full(v);"),
    ("op_popfq_df_movsb", "ctrl_df_ignored", re.compile(r"direction = \(flags & 0x0400ull\) != 0;"), "direction = false;"),
    ("op_sub64_pushfq", "ctrl_of_dropped", re.compile(r"\(word\(overflow\) << 11\)"), "(word(false) << 11)"),
    ("op_popfq_normalize", "ctrl_id_dropped", re.compile(r"\(word\(id_flag\) << 21\)"), "(word(false) << 21)"),
    ("op_nested_push_pop", "ctrl_rsp_step_wrong", re.compile(r"rsp -= 2ull;"), "rsp -= 8ull;"),
    ("op_pushf_df_nt", "ctrl_pushf_df_dropped", re.compile(r"\(static_cast<uint16_t>\(direction\) << 10\)"),
     "(static_cast<uint16_t>(false) << 10)"),
    ("op_pushf_df_nt", "ctrl_pushf_nt_dropped", re.compile(r"\(static_cast<uint16_t>\(nt_flag\) << 14\)"),
     "(static_cast<uint16_t>(false) << 14)"),
)


def _wrap(symbol: str, code: str) -> str:
    return f'extern "C" __declspec(dllexport) word {symbol}(word a, word b, word c) {{\n{code}\n}}'


def _lower_export(pe: pefile.PE, name: str) -> str:
    ranges = function_ranges(pe, export_rva(pe, name))
    body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
    depths = flag_stack_depths(body, {})
    assert depths[body[0].ip] == 0
    flag_definedness(body, {}, flag_depths=depths)
    return lower(body, ["rcx = a; rdx = b; r8 = c;"], {}, {}, {})


def emit(native_path: Path, output_path: Path) -> None:
    """Lower all fixture exports through tools/ffx_version_query.py."""
    parts = [INTEGER_HEADER]
    with pefile.PE(str(native_path), fast_load=True) as pe:
        for name in ALL_EXPORTS:
            parts.append(_wrap(f"lifted_{name}", _lower_export(pe, name)))
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def emit_controls(original_path: Path, output_path: Path) -> None:
    """Emit mutated-C++ negative control exports into output_path."""
    parts = [INTEGER_HEADER, EXTRA_CONTROL_HELPERS]
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for base_name, ctrl_name, pattern, replacement in CPP_MUTATIONS:
            src = _lower_export(pe, base_name)
            mutated, count = pattern.subn(replacement, src)
            if count < 1:
                raise AssertionError(f"mutation {ctrl_name} changed {count} occurrences in {base_name}, expected >= 1")
            parts.append(_wrap(ctrl_name, mutated))
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# --- Independent Python RFLAGS reference model ---

MOD_FLAGS_16 = 0x4CD5  # CF(0), PF(2), AF(4), ZF(6), SF(7), DF(10), OF(11), NT(14)
MOD_FLAGS_64 = 0x204CD5  # + ID(21)
FIXED_FLAGS = 0x0202  # bit 1 (0x0002) | IF (0x0200)


def calc_pf(val: int) -> int:
    return int(((val & 0xFF).bit_count() % 2) == 0)


def add_flags(bits: int, a: int, b: int):
    mask = (1 << bits) - 1
    ua = a & mask
    ub = b & mask
    total = ua + ub
    res = total & mask
    cf = int((total >> bits) != 0)
    af = int(((ua ^ ub ^ res) & 0x10) != 0)
    pf = calc_pf(res)
    zf = int(res == 0)
    sf = int(((res >> (bits - 1)) & 1) != 0)
    of = int((((ua ^ res) & (ub ^ res)) >> (bits - 1)) & 1)
    return res, (cf, pf, af, zf, sf, of)


def sub_flags(bits: int, a: int, b: int):
    mask = (1 << bits) - 1
    ua = a & mask
    ub = b & mask
    diff = ua - ub
    res = diff & mask
    cf = int(diff < 0)
    af = int(((ua ^ ub ^ res) & 0x10) != 0)
    pf = calc_pf(res)
    zf = int(res == 0)
    sf = int(((res >> (bits - 1)) & 1) != 0)
    of = int((((ua ^ ub) & (ua ^ res)) >> (bits - 1)) & 1)
    return res, (cf, pf, af, zf, sf, of)


def pack_rflags(flags, df: int = 0, nt: int = 0, id_bit: int = 0) -> int:
    cf, pf, af, zf, sf, of = flags
    return (
        FIXED_FLAGS
        | (cf << 0)
        | (pf << 2)
        | (af << 4)
        | (zf << 6)
        | (sf << 7)
        | (df << 10)
        | (of << 11)
        | (nt << 14)
        | (id_bit << 21)
    )


def unpack_flags(rflags: int):
    cf = (rflags >> 0) & 1
    pf = (rflags >> 2) & 1
    af = (rflags >> 4) & 1
    zf = (rflags >> 6) & 1
    sf = (rflags >> 7) & 1
    of = (rflags >> 11) & 1
    return (cf, pf, af, zf, sf, of)


def pack_lahf_byte(flags) -> int:
    cf, pf, af, zf, sf, _of = flags
    return (sf << 7) | (zf << 6) | (af << 4) | (pf << 2) | 0x02 | cf


def apply_lahf(rax: int, flags) -> int:
    return (rax & ~0xFF00) | (pack_lahf_byte(flags) << 8)


def ring3_popf_normalize(val: int, prev_rflags: int = FIXED_FLAGS, width: int = 64) -> int:
    if width == 16:
        return (prev_rflags & ~0xFFFF) | FIXED_FLAGS | (val & MOD_FLAGS_16)
    return FIXED_FLAGS | (val & MOD_FLAGS_64)


def pack_pushf_out(rax: int, flags, push_bits: int) -> bytes:
    rflags = pack_rflags(flags)
    if push_bits == 16:
        new_rax = (rax & ~0xFFFF) | (rflags & 0xFFFF)
    else:
        new_rax = rflags
    cf, pf, _af, zf, sf, of = flags
    out = bytearray(FILL)
    struct.pack_into("<Q", out, 0, new_rax & MASK64)
    out[8:14] = bytes((cf, int(sf != of), zf, sf, of, pf))
    return bytes(out)


def u64s(buf: bytes):
    return struct.unpack("<4Q", buf)


def pack_u64s(*vals) -> bytes:
    return struct.pack("<4Q", *(v & MASK64 for v in vals))


def pack_out(*qwords) -> bytes:
    out = bytearray(FILL)
    for i, val in enumerate(qwords):
        struct.pack_into("<Q", out, i * 8, val & MASK64)
    return bytes(out)


def reference(name: str, a_bytes: bytes, b_bytes: bytes, altered: bool = False):
    a0, a1, a2, _a3 = u64s(a_bytes)
    b0, b1, b2, b3 = u64s(b_bytes)

    if name in ARITH_SPECS:
        kind, bits, push_bits, imm_spec = ARITH_SPECS[name]
        if kind == "add_al":
            rhs = imm_spec if altered else (b0 & 0xFF)
            al_res, flags = add_flags(8, a0, rhs)
            rax = (a0 & ~0xFF) | al_res
            return pack_pushf_out(rax, flags, push_bits), pack_rflags(flags)
        rhs = (imm_spec[1] if altered else imm_spec[0]) if imm_spec is not None else b1
        if kind == "add":
            _, flags = add_flags(bits, b0, rhs)
        else:
            _, flags = sub_flags(bits, b0, rhs)
        return pack_pushf_out(a0, flags, push_bits), pack_rflags(flags)

    if name == "op_popf_lahf_setcc":
        norm = ring3_popf_normalize(b0 & 0xFFFF, width=16)
        flags = unpack_flags(norm)
        rax = apply_lahf(a0, flags)
        cf, pf, _af, zf, sf, of = flags
        out = bytearray(FILL)
        struct.pack_into("<Q", out, 0, rax & MASK64)
        out[8:14] = bytes((cf, int(sf != of), zf, sf, of, pf))
        return bytes(out), norm

    if name == "op_popfq_df_movsb":
        norm = ring3_popf_normalize(a0, width=64)
        df = (norm >> 10) & 1
        flags = unpack_flags(norm)
        cf, pf, _af, _zf, _sf, _of = flags
        out = bytearray(FILL)
        step = -1 if df else 1
        out[16] = b_bytes[16]
        out[16 + step] = b_bytes[16 + step]
        end_off = 16 + 2 * step
        struct.pack_into("<Q", out, 0, end_off)
        out[8] = cf
        out[9] = pf
        struct.pack_into("<Q", out, 24, apply_lahf(a1, flags) & MASK64)
        return bytes(out), norm

    if name == "op_pushfq_std_movsb_popfq":
        _, flags = add_flags(64, a1, a2)
        cf, pf, _af, zf, sf, of = flags
        out = bytearray(FILL)
        out[16] = b_bytes[16]
        out[19] = b_bytes[15]
        struct.pack_into("<Q", out, 0, apply_lahf(a0, flags) & MASK64)
        out[8:14] = bytes((cf, int(sf != of), zf, sf, of, pf))
        struct.pack_into("<Q", out, 24, 20)
        return bytes(out), pack_rflags(flags)

    if name == "op_nested_push_pop":
        _, f1 = add_flags(64, b0, b1)
        _, f2 = sub_flags(16, b2, b1)
        _, f3 = sub_flags(32, b3, b1)
        out = bytearray(FILL)
        r3 = apply_lahf(a0, f3)
        r2 = apply_lahf(r3, f2)
        r1 = apply_lahf(r2, f1)
        struct.pack_into("<3Q", out, 0, r3 & MASK64, r2 & MASK64, r1 & MASK64)
        out[24:31] = bytes((f3[0], f3[5], f2[0], f2[5], f1[0], f1[5], f1[1]))
        return bytes(out), pack_rflags(f1)

    if name == "op_popfq_normalize":
        norm = ring3_popf_normalize(b0, width=64)
        flags = unpack_flags(norm)
        cf, pf, _af, zf, sf, of = flags
        out = bytearray(FILL)
        struct.pack_into("<Q", out, 0, norm & MASK64)
        out[8:14] = bytes((cf, int(sf != of), zf, sf, of, pf))
        struct.pack_into("<Q", out, 16, apply_lahf(norm, flags) & MASK64)
        return bytes(out), norm

    if name == "op_popf_normalize":
        pre = ring3_popf_normalize(b1, width=64)
        norm = ring3_popf_normalize(b0 & 0xFFFF, prev_rflags=pre, width=16)
        flags = unpack_flags(norm)
        cf, pf, _af, zf, sf, of = flags
        out = bytearray(FILL)
        struct.pack_into("<Q", out, 0, norm & MASK64)
        out[8:14] = bytes((cf, int(sf != of), zf, sf, of, pf))
        return bytes(out), norm

    if name == "op_pushf_df_nt":
        norm = ring3_popf_normalize(a0, width=64)
        word16 = norm & 0xFFFF
        out = bytearray(FILL)
        struct.pack_into("<Q", out, 0, word16)
        return bytes(out), norm & ~0x400

    raise AssertionError(f"unknown export {name}")


# --- Test vectors ---

RAX_PATTERNS = (
    0x123456789ABCDEF0,
    0xFFFFFFFFFFFF000F,
    0x000000000000FF80,
    0xDEADBEEFCAFE2855,
)


def pairs_for_width(bits: int):
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    edges = [
        (0x0F, 0x01),
        (0x08, 0x08),
        (0x10, 0x01),
        (0x20, 0x01),
        (mask, 0x01),
        (mask - 7, 0x08),
        (0x00, 0x01),
        (0x0F, mask),
        (0x10, 0x10),
        (0x11, 0x11),
        (0x20, 0x10),
        (0x30, 0x10),
        (0x00, 0x00),
        (0x01, 0x01),
        (sign - 1, 0x01),
        (sign, 0x01),
        (sign, sign),
        (mask, 0x00),
        (mask, mask),
    ]
    if bits >= 16:
        edges.extend([
            (0x0100, 0x0000),
            (0x0103, 0x0000),
            (0x0101, 0x0000),
            (0x0100, 0x0001),
            (0x0301, 0x0100),
        ])
    rng = random.Random(0x9C9D00 + bits)
    for _ in range(6):
        edges.append((rng.getrandbits(bits), rng.getrandbits(bits)))
    return [
        (pack_u64s(RAX_PATTERNS[i % len(RAX_PATTERNS)], 0, 0, 0), pack_u64s(a, b, 0, 0))
        for i, (a, b) in enumerate(edges)
    ]


def special_inputs(name: str):
    # Safe ring-3 mask: never sets TF (0x100) or AC (0x40000).
    trap_free = ~(0x100 | 0x40000) & MASK64
    norm_candidates = [
        0x0000000000000000,                                  # bit1=0, IF=0 -> normalized to 0x0202
        0x00000000000008D5,                                  # all status flags set + OF, DF=0, bit1=0, IF=0
        0x0000000000000CD5,                                  # all status flags + DF=1, bit1=0, IF=0
        0x0000000000003000,                                  # IOPL=3 (ignored in ring 3)
        0x0000000000004000,                                  # NT=1
        0x0000000000200000,                                  # ID=1
        0x0000000000204CD5,                                  # ID=1, NT=1, DF=1, all status flags
        0xFFFFFFFFFFFFFFFF & trap_free & ~0x400,             # RF, VM, VIF, VIP, IOPL, upper reserved all 1 -> cleared
        0xFFFFFFFFFFFBFFFF & trap_free,                      # all non-trapping bits including DF=1
        0x00000000001B8028,                                  # reserved/system bits (3, 5, 15, RF, VM, VIF, VIP) all 1 -> 0
    ]

    if name == "op_popf_lahf_setcc":
        return [
            (pack_u64s(RAX_PATTERNS[i % len(RAX_PATTERNS)], 0, 0, 0), pack_u64s(cand & 0xFFFF, i, 0, 0))
            for i, cand in enumerate(norm_candidates)
        ]

    if name == "op_popfq_df_movsb":
        byte_payload = bytes(0x10 + i for i in range(32))
        cases = []
        for i, cand in enumerate(norm_candidates):
            cases.append((pack_u64s(cand, RAX_PATTERNS[i % len(RAX_PATTERNS)], 0, 0), byte_payload))
        return cases

    if name == "op_pushfq_std_movsb_popfq":
        byte_payload = bytes(0x30 + i for i in range(32))
        return [
            (pack_u64s(RAX_PATTERNS[i % len(RAX_PATTERNS)], a, b, 0), byte_payload)
            for i, (a, b) in enumerate((
                (0, 0),
                (0xFFFFFFFFFFFFFFFF, 1),
                (0x7FFFFFFFFFFFFFFF, 1),
                (0x0F, 0x01),
                (0x10, 0x10),
                (0x0101, 0x0000),
                (0x8000000000000000, 0x8000000000000000),
            ))
        ]

    if name == "op_nested_push_pop":
        return [
            (pack_u64s(RAX_PATTERNS[0], 0, 0, 0), pack_u64s(0x0F, 0x01, 0x8000, 0x7FFFFFFF)),
            (pack_u64s(RAX_PATTERNS[1], 0, 0, 0), pack_u64s(0xFFFFFFFFFFFFFFFF, 0x01, 0x0000, 0x00000000)),
            (pack_u64s(RAX_PATTERNS[2], 0, 0, 0), pack_u64s(0x7FFFFFFFFFFFFFFF, 0x01, 0x0100, 0x80000000)),
            (pack_u64s(RAX_PATTERNS[3], 0, 0, 0), pack_u64s(0x10, 0x10, 0x0010, 0x00000010)),
            (pack_u64s(RAX_PATTERNS[0], 0, 0, 0), pack_u64s(0x00, 0x00, 0x0001, 0x00000002)),
            (pack_u64s(RAX_PATTERNS[2], 0, 0, 0), pack_u64s(0x123456789ABCDEF0, 0x00FF, 0x8001, 0x80000005)),
        ]

    if name == "op_popfq_normalize":
        return [
            (pack_u64s(0x0F, 0x01, 0, 0), pack_u64s(cand, 0, 0, 0))
            for cand in norm_candidates
        ]

    if name == "op_popf_normalize":
        cases = []
        for i, cand in enumerate(norm_candidates):
            pre = 0x00200202 if (i % 2 == 0) else 0x00000202
            cases.append((pack_u64s(0x0F, 0x01, 0, 0), pack_u64s(cand & 0xFFFF, pre, 0, 0)))
        return cases

    if name == "op_pushf_df_nt":
        return [
            (pack_u64s(cand, 0x0F, 0, 0), pack_u64s(0, 0, 0, 0))
            for cand in (0x0000, 0x0400, 0x4000, 0x4400, 0x08D5, 0x204400)
        ]

    raise AssertionError(f"no inputs defined for {name}")


def inputs_for(name: str):
    if name in ARITH_SPECS:
        _, bits, _, _ = ARITH_SPECS[name]
        return pairs_for_width(bits)
    return special_inputs(name)


# --- 32-byte guarded buffer helpers ---

def alloc_aligned_buffer(payload: bytes, align: int = 32):
    total = len(GUARD_PRE) + len(payload) + len(GUARD_POST) + align
    buf = ctypes.create_string_buffer(total)
    base_addr = ctypes.addressof(buf)
    start_after_guard = base_addr + len(GUARD_PRE)
    pad = (align - start_after_guard % align) % align
    payload_off = len(GUARD_PRE) + pad
    pre_off = payload_off - len(GUARD_PRE)
    post_off = payload_off + len(payload)
    buf[pre_off:payload_off] = GUARD_PRE
    buf[payload_off:post_off] = payload
    buf[post_off : post_off + len(GUARD_POST)] = GUARD_POST
    return buf, base_addr + payload_off, pre_off, payload_off, post_off


def check_guards_and_read(buf, pre_off: int, payload_off: int, post_off: int, label: str) -> bytes:
    raw = buf.raw
    if raw[pre_off:payload_off] != GUARD_PRE or raw[post_off : post_off + len(GUARD_POST)] != GUARD_POST:
        raise AssertionError(f"32-byte guard bytes corrupted around {label}")
    return raw[payload_off:post_off]


def invoke(fn, a: bytes, b: bytes):
    buf_a, ptr_a, a_pre, a_off, a_post = alloc_aligned_buffer(a)
    buf_b, ptr_b, b_pre, b_off, b_post = alloc_aligned_buffer(b)
    buf_out, ptr_out, o_pre, o_off, o_post = alloc_aligned_buffer(FILL)
    ret_rax = fn(ptr_a, ptr_b, ptr_out)
    check_guards_and_read(buf_a, a_pre, a_off, a_post, "input_a")
    check_guards_and_read(buf_b, b_pre, b_off, b_post, "input_b")
    return ret_rax, check_guards_and_read(buf_out, o_pre, o_off, o_post, "output")


def assert_equal_output(label: str, name: str, idx, got: bytes, want: bytes, got_rax: int | None = None) -> None:
    if got != want:
        raise AssertionError(f"{label} {name} case {idx}: got {got.hex()} != expected {want.hex()}")
    if got_rax is not None:
        want_rax = struct.unpack_from("<Q", want, 0)[0]
        if got_rax != want_rax:
            raise AssertionError(f"{label} {name} case {idx}: returned RAX {got_rax:#018x} != {want_rax:#018x}")


def bind_exports(dll_path: Path, prefix: str = "", names=EXPORTS):
    lib = ctypes.CDLL(str(dll_path.resolve()))
    funcs = {}
    for name in names:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        funcs[name] = fn
    return lib, funcs


def check_discarded_read(native_path: Path, lifted_path: Path) -> None:
    payload = pack_u64s(0x1122334455667788, 0, 0, 0)
    for label, dll, prefix in (("native", native_path, ""), ("lifted", lifted_path, "lifted_")):
        _, fns = bind_exports(dll, prefix, ("op_discarded_read",))
        fn = fns["op_discarded_read"]
        source, ptr, _, _, _ = alloc_aligned_buffer(payload)
        for readable in (True, False):
            result, result_ptr, pre, off, post = alloc_aligned_buffer(FILL)
            ret_rax = None
            try:
                ret_rax = fn(ptr, ptr if readable else 0, result_ptr)
            except OSError as error:
                assert not readable and "access violation reading" in str(error).lower(), (label, error)
            else:
                assert readable, f"{label} op_discarded_read discarded a faulting read"
            expected = pack_out(0x1122334455667788, 0x1122334455667788) if readable else pack_out(0x1122334455667788)
            assert_equal_output(
                label,
                "op_discarded_read",
                readable,
                check_guards_and_read(result, pre, off, post, label),
                expected,
                ret_rax if readable else None,
            )
        _ = source


def check_differential(native_path: Path, lifted_path: Path, altered_path: Path, controls_path: Path) -> None:
    check_discarded_read(native_path, lifted_path)

    _, native = bind_exports(native_path)
    _, lifted = bind_exports(lifted_path, "lifted_")
    _, altered = bind_exports(altered_path, "lifted_")
    _, controls = bind_exports(controls_path, "", tuple(c[1] for c in CPP_MUTATIONS))

    total = 0
    seen = {k: set() for k in ("CF", "PF", "AF", "ZF", "SF", "DF", "OF", "NT", "ID")}
    bit_positions = (
        ("CF", 0), ("PF", 2), ("AF", 4), ("ZF", 6), ("SF", 7),
        ("DF", 10), ("OF", 11), ("NT", 14), ("ID", 21),
    )
    for name in EXPORTS:
        for idx, (a, b) in enumerate(inputs_for(name)):
            want, rflags = reference(name, a, b)
            rax_n, got_n = invoke(native[name], a, b)
            rax_l, got_l = invoke(lifted[name], a, b)
            assert_equal_output("native", name, idx, got_n, want, rax_n)
            assert_equal_output("lifted", name, idx, got_l, want, rax_l)
            for key, bit in bit_positions:
                seen[key].add(bool((rflags >> bit) & 1))
            total += 1

    for key, vals in seen.items():
        if vals != {False, True}:
            raise AssertionError(f"flag {key} never took both values: {vals}")

    # ALTER_IMM controls
    diverged = {}
    for name in ALTERED_EXPORTS:
        diffs = 0
        for idx, (a, b) in enumerate(inputs_for(name)):
            want_alt, _ = reference(name, a, b, altered=True)
            want_orig, _ = reference(name, a, b)
            rax_alt, got_alt = invoke(altered[name], a, b)
            assert_equal_output("altered", name, idx, got_alt, want_alt, rax_alt)
            if got_alt != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"altered control {name} did not diverge from original")
        diverged[name] = diffs

    # Mutated-C++ controls (bit1, IF, AF, PF, DF, OF, ID, rsp-step)
    ctrl_diverged = {}
    for base_name, ctrl_name, _, _ in CPP_MUTATIONS:
        diffs = 0
        for a, b in inputs_for(base_name):
            want_orig, _ = reference(base_name, a, b)
            _, got_ctrl = invoke(controls[ctrl_name], a, b)
            if got_ctrl != want_orig:
                diffs += 1
        if not diffs:
            raise AssertionError(f"mutated-C++ control {ctrl_name} did not diverge from {base_name}")
        ctrl_diverged[ctrl_name] = diffs

    # Output mutation negative control
    sample_a, sample_b = inputs_for("op_add32_pushfq")[0]
    expected, _ = reference("op_add32_pushfq", sample_a, sample_b)
    mutated = bytearray(expected)
    mutated[0] ^= 0x02
    try:
        assert_equal_output("mutation", "op_add32_pushfq", 0, bytes(mutated), expected)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated output was not caught by comparator")

    print(
        f"{total} native/lifted/reference PUSHF/PUSHFQ/POPF/POPFQ cases across {len(EXPORTS)} exports matched bit-for-bit "
        f"(with CF/PF/AF/ZF/SF/DF/OF/NT/ID each seen both ways); "
        f"ALTER_IMM diverged on {diverged}; C++ mutations diverged on {ctrl_diverged}; "
        f"dead-read and output mutation controls passed"
    )


# --- Alternate-encoding acceptance & rejection controls ---

def encode_seq(*instructions) -> bytes:
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_alt_encodings_and_rejections(original_path: Path) -> None:
    ld_rax = Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RCX))
    ld_r9 = Instruction.create_reg_mem(Code.MOV_R64_RM64, R.R9, MemoryOperand(R.RDX))
    add_rax_r9 = Instruction.create_reg_reg(Code.ADD_R64_RM64, R.RAX, R.R9)
    add_eax_1 = Instruction.create_reg_i32(Code.ADD_EAX_IMM32, R.EAX, 1)
    pushf_ins = Instruction.create(Code.PUSHFW)
    pushfq_ins = Instruction.create(Code.PUSHFQ)
    popf_ins = Instruction.create(Code.POPFW)
    popfq_ins = Instruction.create(Code.POPFQ)
    ret_ins = Instruction.create(Code.RETNQ)
    st_r8_rax = Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.R8), R.RAX)

    # Alternate valid encodings & entry_state="canonical" support
    alt_accept = [
        ("rex_w_pushfq_popfq", bytes.fromhex("489c489dc3"), "rsp -= 8ull;", "canonical"),
        ("add_mem_then_pushfq_popfq",
         encode_seq(Instruction.create_reg_mem(Code.ADD_R64_RM64, R.RAX, MemoryOperand(R.RCX)),
                    pushfq_ins, popfq_ins, st_r8_rax, ret_ins),
         "aux = ((a ^ b ^ v) & 0x10u) != 0;", "defined"),
        ("branch_join_add_sub_pushfq",
         bytes.fromhex("83f900740583c001eb0383e8019c9dc3"),
         "rsp -= 8ull;", "defined"),
        ("pushfq_std_popfq_restores_df",
         encode_seq(ld_rax, add_eax_1, pushfq_ins, Instruction.create(Code.STD), popfq_ins, ret_ins),
         "direction = (flags & 0x0400ull) != 0;", "defined"),
        ("popf_sets_lahf_and_setp",
         encode_seq(ld_rax, add_eax_1, pushf_ins, popf_ins, Instruction.create(Code.LAHF),
                    Instruction.create_mem(Code.SETP_RM8, MemoryOperand(R.R8)), st_r8_rax, ret_ins),
         "store<uint8_t>(r8 + 0ull, (parity));", "defined"),
        ("const_imm_write_popf",
         encode_seq(ld_rax, add_eax_1, pushf_ins,
                    Instruction.create_mem_i32(Code.MOV_RM16_IMM16, MemoryOperand(R.RSP), 0x0246),
                    popf_ins, ret_ins),
         "flags = load<uint16_t>(rsp);", "defined"),
        ("masked_reg_write_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "direction = false;", "defined"),
        ("masked_stack_write_popf",
         encode_seq(ld_rax, add_eax_1, pushf_ins,
                    Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.RSP), R.CX),
                    Instruction.create_mem_i32(Code.AND_RM16_IMM16, MemoryOperand(R.RSP), 0xFEFF),
                    popf_ins, Instruction.create(Code.CLD), ret_ins),
         "carry = (flags & 0x0001u) != 0;", "defined"),
        ("low_byte_write_popfq",
         encode_seq(ld_rax, add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM8_R8, MemoryOperand(R.RSP), R.CL),
                    popfq_ins, ret_ins),
         "id_flag = (flags & 0x200000ull) != 0;", "defined"),
        ("high_byte_ch_masked_popf",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM32_IMM32, R.ECX, -0x101),
                    add_eax_1, pushf_ins,
                    Instruction.create_mem_reg(Code.MOV_RM8_R8, MemoryOperand(R.RSP, displ=1, displ_size=1), R.CH),
                    popf_ins, Instruction.create(Code.CLD), ret_ins),
         "direction = (flags & 0x0400u) != 0;", "defined"),
        ("or_safe_reg_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
                    Instruction.create_reg_i32(Code.OR_RM64_IMM32, R.RCX, 0x400),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "direction = false;", "defined"),
        ("xor_safe_reg_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
                    Instruction.create_reg_i32(Code.XOR_RM64_IMM32, R.RCX, 0x400),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "direction = false;", "defined"),
        ("shl_safe_reg_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, 0xFF),
                    Instruction.create_reg_i32(Code.SHL_RM64_IMM8, R.RCX, 20),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "id_flag = (flags & 0x200000ull) != 0;", "defined"),
        ("shr_safe_reg_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM32_IMM32, R.ECX, 0xFF00),
                    Instruction.create_reg_i32(Code.SHR_RM32_IMM8, R.ECX, 8),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "carry = (flags & 0x0001ull) != 0;", "defined"),
        ("btr_reg_tf_cleared_popfq",
         encode_seq(ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40001),
                    Instruction.create_reg_i32(Code.BTR_RM64_IMM8, R.RCX, 8),
                    add_eax_1, pushfq_ins,
                    Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
                    popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "direction = false;", "defined"),
    ]
    for label, raw, expected_snippet, entry_state in alt_accept:
        decoded = list(Decoder(64, raw, ip=0x1000))
        assert len(decoded) > 0 and not any(ins.is_invalid for ins in decoded), label
        depths = flag_stack_depths(decoded, {})
        assert depths[decoded[0].ip] == 0
        flag_definedness(decoded, {}, flag_depths=depths, entry_state=entry_state)
        src = lower(decoded, ["rcx = a; rdx = b; r8 = c;"], {}, {}, {}, entry_state=entry_state)
        if expected_snippet not in src:
            raise AssertionError(f"{label}: expected {expected_snippet!r} in generated C++:\n{src}")

    # Direct flag_definedness pair for BTR [rsp], imm (stack-edit may-be-1 tracking)
    btr_stack_ok = list(Decoder(64, encode_seq(
        ld_rax, add_eax_1, pushf_ins,
        Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.RSP), R.CX),
        Instruction.create_mem_i32(Code.BTR_RM16_IMM8, MemoryOperand(R.RSP), 8),
        popf_ins, Instruction.create(Code.CLD), ret_ins), ip=0x1000))
    flag_definedness(btr_stack_ok, {}, flag_depths=flag_stack_depths(btr_stack_ok, {}))
    btr_stack_bad = list(Decoder(64, encode_seq(
        ld_rax, add_eax_1, pushf_ins,
        Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.RSP), R.CX),
        Instruction.create_mem_i32(Code.BTR_RM16_IMM8, MemoryOperand(R.RSP), 9),
        popf_ins, Instruction.create(Code.CLD), ret_ins), ip=0x1000))
    try:
        flag_definedness(btr_stack_bad, {}, flag_depths=flag_stack_depths(btr_stack_bad, {}))
    except ValueError as exc:
        if "may set unmodeled control flag(s) TF" not in str(exc):
            raise AssertionError(f"btr_stack_bad: unexpected {exc!r}") from exc
    else:
        raise AssertionError("btr_stack_bad was unexpectedly accepted by flag_definedness")

    # Direct unit test on stack bound limit (128 bytes = 16 PUSHFQs allowed, 17 rejected)
    max_seq = encode_seq(ld_rax, add_eax_1, *([pushfq_ins] * (FLAG_STACK_LIMIT // 8)),
                         *([popfq_ins] * (FLAG_STACK_LIMIT // 8)), ret_ins)
    max_decoded = list(Decoder(64, max_seq, ip=0x1000))
    max_depths = flag_stack_depths(max_decoded, {})
    assert max(max_depths.values()) == FLAG_STACK_LIMIT
    lower(max_decoded, [], {}, {}, {})

    overflow_seq = encode_seq(ld_rax, add_eax_1, *([pushfq_ins] * (FLAG_STACK_LIMIT // 8 + 1)),
                              *([popfq_ins] * (FLAG_STACK_LIMIT // 8 + 1)), ret_ins)

    seg_pushfq = Instruction.create(Code.PUSHFQ)
    seg_pushfq.segment_prefix = R.FS
    seg_popfq = Instruction.create(Code.POPFQ)
    seg_popfq.segment_prefix = R.GS

    unit_cases = [
        # Undefined flags at entry
        ("undef_entry_pushf", encode_seq(pushf_ins, popf_ins, ret_ins), "reads undefined flag"),
        ("undef_entry_pushfq", encode_seq(pushfq_ins, popfq_ins, ret_ins), "reads undefined flag"),
        ("undef_entry_mov_pushfq", encode_seq(ld_rax, pushfq_ins, popfq_ins, ret_ins), "reads undefined flag"),
        # Undefined flags after XOR / TEST / INC / CLC / CALL / join
        ("undef_after_xor", encode_seq(
            Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EAX, R.EAX), pushfq_ins, popfq_ins, ret_ins),
         "reads undefined flag(s) PF, AF"),
        ("undef_add_then_xor", encode_seq(
            ld_rax, ld_r9, add_rax_r9,
            Instruction.create_reg_reg(Code.XOR_R32_RM32, R.EDX, R.EDX),
            pushfq_ins, popfq_ins, ret_ins),
         "reads undefined flag(s) PF, AF"),
        ("undef_after_test", encode_seq(
            ld_rax, Instruction.create_reg_reg(Code.TEST_RM64_R64, R.RAX, R.RAX),
            pushfq_ins, popfq_ins, ret_ins),
         "reads undefined flag(s) PF, AF"),
        ("undef_after_inc", encode_seq(
            ld_rax, Instruction.create_reg(Code.INC_RM32, R.EAX), pushfq_ins, popfq_ins, ret_ins),
         "reads undefined flag(s) CF, PF, AF"),
        ("undef_after_clc_only", encode_seq(
            Instruction.create(Code.CLC), pushfq_ins, popfq_ins, ret_ins),
         "reads undefined flag(s) PF, AF, ZF, SF, OF"),
        ("undef_across_call", bytes.fromhex("4801c8e8000000009c9dc3"), "reads undefined flag"),
        ("undef_branch_join_one_arm", bytes.fromhex("85c9740383c0019c9dc3"), "reads undefined flag(s) PF, AF"),
        # Unbalanced RSP at RET / CALL / join / underflow
        ("unbalanced_ret_pushfq", encode_seq(ld_rax, add_eax_1, pushfq_ins, ret_ins), "unbalanced flag stack depth 8"),
        ("unbalanced_ret_pushf", encode_seq(ld_rax, add_eax_1, pushf_ins, ret_ins), "unbalanced flag stack depth 2"),
        ("underflow_popfq_entry", encode_seq(popfq_ins, ret_ins), "underflows flag stack"),
        ("underflow_popf_entry", encode_seq(popf_ins, ret_ins), "underflows flag stack"),
        ("underflow_pushf_then_popfq", encode_seq(ld_rax, add_eax_1, pushf_ins, popfq_ins, ret_ins), "underflows flag stack"),
        ("unbalanced_join_depths", bytes.fromhex("4801c89c85c974019c909dc3"), "flag stack depths"),
        ("unbalanced_call_pushfq", bytes.fromhex("4801c89ce8000000009dc3"), "calls with unbalanced flag stack depth 8"),
        # Stack bounds & RSP tampering / escaping
        ("stack_overflow_bound", overflow_seq, f"exceeds flag stack bound {FLAG_STACK_LIMIT}"),
        ("rsp_reg_modify_with_pushfq", encode_seq(
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_reg_i32(Code.ADD_RM64_IMM8, R.RSP, 8), ret_ins),
         "modifies or reads RSP alongside flag stack"),
        ("rsp_oob_read_before_push", encode_seq(
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RSP)),
            add_eax_1, pushfq_ins, popfq_ins, ret_ins),
         "accesses outside pushed flag stack"),
        ("rsp_oob_read_past_slot", encode_seq(
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.RSP, displ=8, displ_size=1)),
            popfq_ins, ret_ins),
         "accesses outside pushed flag stack"),
        ("rsp_lea_alias", encode_seq(
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_reg_mem(Code.LEA_R64_M, R.RAX, MemoryOperand(R.RSP)),
            popfq_ins, ret_ins),
         "aliases RSP alongside flag stack"),
        # DF=1 leak at RET / CALL
        ("df_leak_std_pushfq_popfq_ret", encode_seq(
            ld_rax, add_eax_1, Instruction.create(Code.STD), pushfq_ins, popfq_ins, ret_ins),
         "direction flag may be set"),
        ("df_leak_std_pushfq_cld_popfq_ret", encode_seq(
            ld_rax, add_eax_1, Instruction.create(Code.STD), pushfq_ins,
            Instruction.create(Code.CLD), popfq_ins, ret_ins),
         "direction flag may be set"),
        ("df_leak_mem_write_popfq_ret", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, ret_ins),
         "direction flag may be set"),
        ("df_leak_after_popfq_std_ret", encode_seq(
            ld_rax, add_eax_1, pushfq_ins, popfq_ins, Instruction.create(Code.STD), ret_ins),
         "direction flag may be set"),
        ("df_leak_at_call", bytes.fromhex("83c0019c9dfde800000000fcc3"), "direction flag may be set"),
        # Unconstrained / unmodeled control bits (TF / AC) on flag-stack edits
        ("unconstrained_input_mem_write_popfq", encode_seq(
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF, AC"),
        ("unconstrained_input_mem_write_popf", encode_seq(
            ld_rax, add_eax_1, pushf_ins,
            Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.RSP), R.CX),
            popf_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("ac_unconstrained_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x101),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) AC"),
        ("tf_imm_write_popf", encode_seq(
            ld_rax, add_eax_1, pushf_ins,
            Instruction.create_mem_i32(Code.MOV_RM16_IMM16, MemoryOperand(R.RSP), 0x0100),
            popf_ins, ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("ac_or_stack_popfq", encode_seq(
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_mem_i32(Code.OR_RM64_IMM32, MemoryOperand(R.RSP), 0x40000),
            popfq_ins, ret_ins),
         "may set unmodeled control flag(s) AC"),
        ("unconstrained_join_one_arm_popfq", bytes.fromhex("85c974074881e1fffefbff83c0019c48890c249dfcc3"),
         "may set unmodeled control flag(s) TF, AC"),
        ("non_rsp_mem_load_popfq", encode_seq(
            Instruction.create_reg_mem(Code.MOV_R64_RM64, R.RCX, MemoryOperand(R.RDX)),
            ld_rax, add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF, AC"),
        ("high_byte_ch_unmasked_popf", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM32_IMM32, R.ECX, -2),
            add_eax_1, pushf_ins,
            Instruction.create_mem_reg(Code.MOV_RM8_R8, MemoryOperand(R.RSP, displ=1, displ_size=1), R.CH),
            popf_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("or_tf_reg_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
            Instruction.create_reg_i32(Code.OR_RM64_IMM32, R.RCX, 0x100),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("xor_tf_reg_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x40101),
            Instruction.create_reg_i32(Code.XOR_RM64_IMM32, R.RCX, 0x100),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("shl_tf_reg_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, 0xFF),
            Instruction.create_reg_i32(Code.SHL_RM64_IMM8, R.RCX, 8),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("shr_tf_reg_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, 0x10000),
            Instruction.create_reg_i32(Code.SHR_RM64_IMM8, R.RCX, 8),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) TF"),
        ("btr_reg_wrong_bit_popfq", encode_seq(
            ld_rax, Instruction.create_reg_i32(Code.AND_RM64_IMM32, R.RCX, -0x101),
            Instruction.create_reg_i32(Code.BTR_RM64_IMM8, R.RCX, 9),
            add_eax_1, pushfq_ins,
            Instruction.create_mem_reg(Code.MOV_RM64_R64, MemoryOperand(R.RSP), R.RCX),
            popfq_ins, Instruction.create(Code.CLD), ret_ins),
         "may set unmodeled control flag(s) AC"),
        ("join_stack_one_arm_ac_popfq", bytes.fromhex("4801c89c85c9740848810c24000004009dfcc3"),
         "may set unmodeled control flag(s) AC"),
        ("join_stack_taken_arm_ac_popfq", bytes.fromhex("4801c89c85c97402eb0848810c24000004009dfcc3"),
         "may set unmodeled control flag(s) AC"),
        ("df_join_one_arm_std_popfq", bytes.fromhex("83c0019c9d85c97403fdeb01fcc3"),
         "direction flag may be set"),
        # Disallowed prefixes, out-of-scope ops, addressing & stream boundaries
        ("prefix_rep_pushfq", bytes.fromhex("f39c9dc3"), "unsupported instruction prefix"),
        ("prefix_repne_pushfq", bytes.fromhex("f29c9dc3"), "unsupported instruction prefix"),
        ("prefix_lock_with_pushfq", bytes.fromhex("f00fb10983c0019c9dc3"), "unsupported instruction prefix"),
        ("prefix_fs_pushfq", encode_seq(seg_pushfq, popfq_ins, ret_ins), "unsupported instruction prefix"),
        ("prefix_gs_popfq", encode_seq(ld_rax, add_eax_1, pushfq_ins, seg_popfq, ret_ins), "unsupported instruction prefix"),
        ("prefix_repne_movsb", bytes.fromhex("83c0019c9df2a4c3"), "unsupported REPNE string prefix"),
        ("out_of_scope_push_rax", encode_seq(ld_rax, Instruction.create_reg(Code.PUSH_R64, R.RAX),
                                             Instruction.create_reg(Code.POP_R64, R.RAX), ret_ins),
         "unsupported query operation"),
        ("eip_rel_mem", encode_seq(Instruction.create_reg_mem(
            Code.MOV_R64_RM64, R.RAX, MemoryOperand(R.EIP, displ=0x1020, displ_size=4)), ret_ins),
         "unsupported EIP-relative operand"),
        ("missing_ret", encode_seq(ld_rax, add_eax_1, pushfq_ins, popfq_ins), "unbound fallthrough"),
    ]

    for label, raw, expected_reason in unit_cases:
        body = list(Decoder(64, raw, ip=0x1000))
        if not body or any(ins.is_invalid for ins in body):
            raise AssertionError(f"{label}: invalid test encoding {raw.hex()}")
        try:
            lower(body, [], {}, {}, {})
        except ValueError as exc:
            if expected_reason not in str(exc):
                raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator unexpectedly accepted rejected control")

    # Invalid entry_state parameter rejection
    try:
        lower(list(Decoder(64, encode_seq(ret_ins), ip=0x1000)), [], {}, {}, {}, entry_state="bogus")
    except ValueError as exc:
        assert "unsupported entry-state profile" in str(exc)
    else:
        raise AssertionError("invalid entry_state was accepted")

    # Host-op replacing flag producer leaves flags undefined for PUSHFQ
    bound_writer = list(Decoder(64, bytes.fromhex("83c0019c9dc3"), ip=0x1000))
    try:
        lower(bound_writer, ["rax = 0xf;"], {}, {}, {}, host_ops={0x1000: ";"})
    except ValueError as exc:
        if "reads undefined flag" not in str(exc):
            raise AssertionError(f"wrong diagnostic for replaced PUSHFQ flag writer: {exc}") from exc
    else:
        raise AssertionError("PUSHFQ accepted flags from a replaced ADD")

    # Host-op writing a GP register must reset that register's may-be-1 mask
    host_op_reg_reset = list(Decoder(64, bytes.fromhex("31c0488b014183c1019c488904249dc3"), ip=0x1000))
    try:
        lower(host_op_reg_reset, ["rcx = a;"], {}, {}, {}, host_ops={0x1002: "rax = load<word>(rcx);"})
    except ValueError as exc:
        if "may set unmodeled control flag(s) TF, AC" not in str(exc):
            raise AssertionError(f"wrong diagnostic for host-op register reset: {exc}") from exc
    else:
        raise AssertionError("POPFQ accepted RAX after host-op overwrite")

    # x87 profile alongside flag-stack instructions must be rejected
    x87_with_pushfq = list(Decoder(64, encode_seq(
        Instruction.create(Code.FLD1),
        Instruction.create_reg(Code.FSTP_STI, R.ST0),
        add_eax_1, pushfq_ins, popfq_ins, ret_ins), ip=0x1000))
    try:
        lower(x87_with_pushfq, [], {}, {}, {}, x87=True)
    except ValueError as exc:
        if "unsupported query operation" not in str(exc) or "pushfq" not in str(exc):
            raise AssertionError(f"wrong diagnostic for x87 alongside flag stack: {exc}") from exc
    else:
        raise AssertionError("flag-stack sequence accepted alongside x87")

    # Explicit RSP binding alongside flag-stack instructions must be rejected
    rsp_bind_seq = list(Decoder(64, encode_seq(ld_rax, add_eax_1, pushfq_ins, popfq_ins, ret_ins), ip=0x1000))
    try:
        lower(rsp_bind_seq, ["rsp = a;"], {}, {}, {})
    except ValueError as exc:
        if "RSP binding is unsupported alongside flag stack" not in str(exc):
            raise AssertionError(f"wrong diagnostic for RSP binding alongside flag stack: {exc}") from exc
    else:
        raise AssertionError("flag-stack sequence accepted with explicit RSP binding")

    # End-to-end PE patching controls on a copy of the built native DLL.
    pe_bytes = original_path.read_bytes()
    with pefile.PE(data=pe_bytes, fast_load=True) as pe:
        entry = export_rva(pe, "op_nested_push_pop")
        file_off = pe.get_offset_from_rva(entry)
        pdata = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
        records = pe.get_data(pdata.VirtualAddress, pdata.Size)
        rec_idx = next(
            i for i in range(0, len(records), 12)
            if struct.unpack_from("<I", records, i)[0] == entry
        )
        pdata_end_off = pe.get_offset_from_rva(pdata.VirtualAddress + rec_idx + 4)
        orig_end = struct.unpack_from("<I", records, rec_idx + 4)[0]
        max_len = orig_end - entry

    unit_map = {lbl: raw for lbl, raw, _ in unit_cases}
    pe_patches = [
        ("pe_undef_entry_pushfq", unit_map["undef_entry_pushfq"], "reads undefined flag"),
        ("pe_undef_after_xor", unit_map["undef_after_xor"], "reads undefined flag(s) PF, AF"),
        ("pe_unbalanced_ret_pushfq", unit_map["unbalanced_ret_pushfq"], "unbalanced flag stack depth 8"),
        ("pe_underflow_popfq_entry", unit_map["underflow_popfq_entry"], "underflows flag stack"),
        ("pe_unbalanced_join_depths", unit_map["unbalanced_join_depths"], "flag stack depths"),
        ("pe_stack_overflow_bound", unit_map["stack_overflow_bound"], f"exceeds flag stack bound {FLAG_STACK_LIMIT}"),
        ("pe_df_leak_std_popfq", unit_map["df_leak_std_pushfq_cld_popfq_ret"], "direction flag may be set"),
        ("pe_df_leak_mem_popfq", unit_map["df_leak_mem_write_popfq_ret"], "direction flag may be set"),
        ("pe_unconstrained_input_popfq", unit_map["unconstrained_input_mem_write_popfq"],
         "may set unmodeled control flag(s) TF, AC"),
        ("pe_unconstrained_input_popf", unit_map["unconstrained_input_mem_write_popf"],
         "may set unmodeled control flag(s) TF"),
        ("pe_prefix_rep_pushfq", unit_map["prefix_rep_pushfq"], "unsupported instruction prefix"),
        ("pe_prefix_fs_pushfq", unit_map["prefix_fs_pushfq"], "unsupported instruction prefix"),
        ("pe_invalid_bytes", bytes.fromhex("fffe") + encode_seq(ret_ins), "instruction boundary"),
    ]
    with TemporaryDirectory() as folder:
        patched_path = Path(folder) / "patched.dll"
        for label, patch, expected_reason in pe_patches:
            if len(patch) > max_len:
                raise AssertionError(f"{label}: patch ({len(patch)} bytes) exceeds op_nested_push_pop ({max_len})")
            damaged = bytearray(pe_bytes)
            damaged[file_off : file_off + len(patch)] = patch
            struct.pack_into("<I", damaged, pdata_end_off, entry + len(patch))
            patched_path.write_bytes(damaged)
            try:
                with pefile.PE(str(patched_path), fast_load=True) as pe:
                    _lower_export(pe, "op_nested_push_pop")
            except ValueError as exc:
                if expected_reason not in str(exc):
                    raise AssertionError(f"{label}: expected {expected_reason!r}, got {exc!r}") from exc
            else:
                raise AssertionError(f"{label}: PE translate accepted rejected patch")

    print(
        f"{len(alt_accept)} alternate-encoding/entry-state checks, {len(unit_cases)} unit rejection controls, "
        f"and {len(pe_patches)} PE-patched rejection controls passed"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "emit":
        emit(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "emit-controls":
        emit_controls(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "check":
        orig, lifted, alt, ctrls = (Path(p) for p in sys.argv[2:6])
        check_alt_encodings_and_rejections(orig)
        check_differential(orig, lifted, alt, ctrls)
    else:
        raise SystemExit(f"unknown mode {mode}")
