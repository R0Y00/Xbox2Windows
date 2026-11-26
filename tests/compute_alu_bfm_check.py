"""Check the compute lowering of v_bfm_b32 between a guarded read and its store (#1546).

Modes:
  controls                          translator output and rejections only
  draw <dxc> <renderer> <adapter>   also dispatch the fixture and compare the UAV with an independent reference
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base

# v_bfm_b32 is VOP3 only on gfx10: D = ((1 << S0[4:0]) - 1) << S1[4:0], modulo 2^32. src0 is the width and
# src1 the offset. A width field of 0 gives an empty mask; a width of 32 cannot be encoded, so the widest mask is
# 0x7fffffff. Mask bits shifted past bit 31 are dropped, not rotated.
#
# The read-compute-write fixture of compute_store_check.py with its ALU lines replaced, by hand, in LLVM 15 gfx1013
# spelling. With w = source word gid (byte offset gid * 4) and cap = params[0].y:
#   a = bfm(w, w >> 5)     VGPR sources; both fields come from words with higher bits set
#   b = bfm(cap, 0x7b)     SGPR width, literal offset 0x7b & 31 = 27
#   c = bfm(44, gid)       inline width 44 & 31 = 12, offset from the dispatch index
#   store a ^ b ^ c at element gid
# The low five bits of source_word(gid) are 17 * (gid + 1) mod 32, so lanes 0..31 see every width, 0 included.
READ_BFM = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", ""),
    ("v_lshrrev_b32_e32 v1, 16, v1", "v_lshrrev_b32_e32 v2, 5, v1"),
    ("v_mad_u32_u24 v1, v2, v1, v0", "v_bfm_b32 v2, v1, v2\nv_bfm_b32 v3, s0, 0x7b\nv_bfm_b32 v4, 44, v0"),
    ("v_min_u32_e32 v1, s0, v1", "v_xor_b32_e32 v2, v3, v2\nv_xor_b32_e32 v1, v4, v2"),
)
BFM_CASE = ("read_bfm", READ_BFM, 32, base.READ_BINDING)
# Width field 0x29 & 31 = 9 at offset 27: only bits 27..31 survive, b = 0xf8000000.
BFM_CAP = 0xc0de0029
LOAD = base.READ_LOADS["read_copy"]
BFM_STORE = ("        outputData.Store(tid.x * 4u + 0u, ((((1u << (0x0000002cu & 31u)) - 1u) << (tid.x & 31u)) ^ "
             "((((1u << (params[0].y & 31u)) - 1u) << (0x0000007bu & 31u)) ^ (((1u << ({0} & 31u)) - 1u) << "
             "(({0} >> (0x00000005u & 31u)) & 31u)))));").format(LOAD)
# Lanes worked out by hand from source_word(gid) = 0x8b02a7f1 + gid * 0x413b1 (gid: width, offset of a):
#   0: 17, 31  a = 0x80000000                 c = 0x00000fff  -> 0x78000fff
#  14: 31, 28  a = 0xf0000000 (widest field)  c = 0x03ffc000  -> 0x0bffc000
#  31:  0, 19  a = 0 (empty mask)             c = 0x80000000  -> 0x78000000
#  32: 17, 16  a = 0xffff0000                 c = 0x00000fff  -> 0x07ff0fff (c offset 32 & 31 = 0)
# 149: 22, 15  a = 0xffff8000                 c = 0xffe00000  -> 0xf81f8000
HAND_WORDS = {0: 0x78000fff, 14: 0x0bffc000, 31: 0x78000000, 32: 0x07ff0fff, 149: 0xf81f8000}


def bit_mask(width, offset, *, rotate=False):
    """A mask built bit by bit: bits offset .. offset + width - 1, dropping (or, if rotate, wrapping) bits past 31."""
    bits = range(offset, offset + min(width, 32))
    return sum(1 << (bit % 32) for bit in bits) if rotate else sum(1 << bit for bit in bits if bit < 32)


def bfm_word(gid, *, swap=False, width_field=31, offset_field=31, zero_full=False, rotate=False):
    """Independent CPU reference of read_bfm for one lane. The keywords select a wrong model of the instruction."""
    word = base.source_word(gid)

    def bfm(src0, src1):
        width, offset = (src1, src0) if swap else (src0, src1)
        width, offset = width & width_field, offset & offset_field
        return bit_mask(32 if zero_full and width == 0 else width, offset, rotate=rotate)

    return bfm(word, word >> 5) ^ bfm(BFM_CAP, 0x7b) ^ bfm(44, gid)


def bfm_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = bfm_word(gid, **reference)
    return words


# Wrong models of the instruction; each must give other words for the largest dispatch and fail the readback.
WRONG = (("operands swapped", dict(swap=True)), ("6-bit width field", dict(width_field=63)),
         ("6-bit offset field", dict(offset_field=63)), ("width 0 as 32", dict(zero_full=True)),
         ("mask rotated", dict(rotate=True)))


def controls():
    """Translator output, the independent reference against hand-worked words, and the refused forms."""
    def bfm(*changes):
        _, lines, width, binding = BFM_CASE
        return base.aot.lower(base.variant(lines, *changes), width, **binding).splitlines()

    for gid, word in HAND_WORDS.items():
        assert bfm_word(gid) == word, (gid, hex(bfm_word(gid)), hex(word))
    widths = [base.source_word(gid) & 31 for gid in range(32)]
    assert sorted(widths) == list(range(32)), widths
    lines = bfm()
    assert lines[-4:] == [base.GUARD_IF, BFM_STORE, "    }", "}"], lines
    # A copy of a result is the same program; exchanging the sources is another.
    assert bfm(("v_bfm_b32 v4, 44, v0", "v_bfm_b32 v5, 44, v0\nv_mov_b32_e32 v4, v5")) == lines
    assert bfm(("v_bfm_b32 v3, s0, 0x7b", "v_bfm_b32 v3, 0x7b, s0"))[-3] != lines[-3]
    assert "(((1u << (0x0000007bu & 31u)) - 1u) << (params[0].y & 31u))" in bfm(
        ("v_bfm_b32 v3, s0, 0x7b", "v_bfm_b32 v3, 0x7b, s0"))[-3]
    # A negative inline constant is its 32-bit pattern: -1 & 31 = 31.
    assert "(((1u << (0xffffffffu & 31u)) - 1u) << (tid.x & 31u))" in bfm(("44, v0", "-1, v0"))[-3]
    bfm_a = "v_bfm_b32 v2, v1, v2"
    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    for message, changes in (
        # Source modifiers: LLVM refuses them for this integer opcode, and so does the lowering.
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, |v1|, v2"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, -v1, v2"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, v1, -v2"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, v1, v2 clamp"),)),
        # Operand spellings outside ALU_SOURCE: a register range, vcc_lo, exec_lo and a float inline constant.
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, v[1:2], v2"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, vcc_lo, v2"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, v1, exec_lo"),)),
        (unsupported + "v_bfm_b32", ((bfm_a, "v_bfm_b32 v2, v1, 0.5"),)),
        # Spellings and neighbours that are not this lowering: the _e64 suffix LLVM 15 does not print, the scalar
        # and 64-bit forms, and the bitfield extract and insert opcodes.
        (unsupported + "v_bfm_b32_e64", ((bfm_a, "v_bfm_b32_e64 v2, v1, v2"),)),
        (unsupported + "s_bfm_b32", (("v_bfm_b32 v3, s0, 0x7b", "s_bfm_b32 s16, s0, 27\nv_mov_b32_e32 v3, s16"),)),
        (unsupported + "s_bfm_b64", (("v_bfm_b32 v3, s0, 0x7b", "s_bfm_b64 s[16:17], s0, 27\nv_mov_b32_e32 v3, s16"),)),
        (unsupported + "v_bfe_u32", ((bfm_a, "v_bfe_u32 v2, v1, v2, 5"),)),
        (unsupported + "v_bfi_b32", ((bfm_a, "v_bfi_b32 v2, v1, v2, v0"),)),
        # Sources that are not known words: an unwritten, a descriptor and the saved exec-mask SGPR, an unwritten
        # VGPR, a literal wider than 32 bits and a constant word still waiting for its s_waitcnt. Also a
        # destination past v255.
        (operand, (("s0, 0x7b", "s20, 0x7b"),)),
        (operand, (("s0, 0x7b", "s8, 0x7b"),)),
        (operand, (("s0, 0x7b", "s2, 0x7b"),)),
        (operand, ((bfm_a, "v_bfm_b32 v2, v1, v9"),)),
        (operand, (("s0, 0x7b", "s0, 0x1ffffffff"),)),
        (operand, (("s_buffer_load_dword s0, s[4:7], 0x4", ""),
                   ("v_lshrrev_b32_e32 v2, 5, v1", "s_buffer_load_dword s0, s[4:7], 0x4\nv_lshrrev_b32_e32 v2, 5, v1"))),
        (operand, ((bfm_a, "v_bfm_b32 v256, v1, v2"),)),
        ("Vector load result used or overwritten before s_waitcnt vmcnt(0)", (("s_waitcnt vmcnt(0)", ""),)),
        # A result is data only: not an address and not a lane-guard operand, and a doubling chain is refused.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v4, s[8:11]"),)),
        ("Unverified lane guard compare", (("v_xor_b32_e32 v1, v4, v2",
                                            "v_xor_b32_e32 v1, v4, v2\nv_cmp_gt_u32_e32 vcc_lo, s3, v4"),)),
        ("ALU expression too large", (("v_xor_b32_e32 v1, v4, v2",
                                       "v_xor_b32_e32 v1, v4, v2" + "\nv_bfm_b32 v1, v1, v1" * 4),)),
    ):
        base.rejects(message, bfm, *changes)


def draw(directory, compiler, renderer, backend):
    """Dispatch read_bfm on the renderer and compare the UAV with bfm_words, with wrong references and mutants."""
    name, lines, width, binding = BFM_CASE
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    src.write_text(base.aot.lower(lines, width, **binding), encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = BFM_CAP

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, bfm_words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = bfm_words(groups, bound)
    assert all(expected[gid] == word for gid, word in HAND_WORDS.items() if gid < bound), "hand-worked words"
    fields = [(base.source_word(g) & 31, (base.source_word(g) >> 5) & 31) for g in range(bound)]
    # The inputs must show an empty mask, the widest mask, masks cut at bit 31 and masks that fit.
    assert any(w == 0 for w, _ in fields) and any(w == 31 for w, _ in fields), "width extremes"
    assert any(w + o > 32 for w, o in fields) and any(0 < w and w + o <= 32 for w, o in fields), "cut and whole masks"
    off = list(expected)
    off[bound - 1] ^= 1
    wrong = [("one word off", off), ("lane past the bound", bfm_words(groups, bound + 1))]
    wrong += [(label, bfm_words(groups, bound, **reference)) for label, reference in WRONG]
    for label, reference in wrong:
        assert reference != expected, f"{label} would go unseen"
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"{name}: reference '{label}' escaped detection\n{bad.stderr}")
    write(groups, bound, expected)
    source = src.read_text(encoding="ascii")
    cap_term = "(((1u << (params[0].y & 31u)) - 1u) << (0x0000007bu & 31u))"
    for label, old, new in (("mask without the minus one", " - 1u)", " - 0u)"),
                            ("inline width 43", "(0x0000002cu & 31u)", "(0x0000002bu & 31u)"),
                            ("cap term operands swapped", cap_term,
                             "(((1u << (0x0000007bu & 31u)) - 1u) << (params[0].y & 31u))"),
                            ("cap from another word", "params[0].y", "params[0].z")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"{name}: mutation '{label}' did not change the source")
        mutant, mutant_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        mutant.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mutant_cso, mutant], check=True)
        bad = subprocess.run([*cmd[:3], mutant_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{name}: mutation '{label}' escaped detection")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    if mode == "controls" and not args:
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
