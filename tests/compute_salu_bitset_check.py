"""Check the compute lowering of s_bitset0_b32 and s_bitset1_b32 on known scalar words."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base

aot = base.aot

# One invocation reads eight root constant words c0..c7 and stores eight words at u0 words 4..11. Each SOP1 bitset
# reads and writes its destination SGPR: it clears (bitset0) or sets (bitset1) bit src0[4:0] and keeps the rest.
#   w4  = c0 with bit c1[4:0] set            (runtime destination and index)
#   w5  = c2 with bit c3[4:0] cleared
#   w6  = c4 with bit c5[4:0] set            (c5 is above 31 in both runs)
#   w7  = w4 with bit 4 set again            (a copy of a bitset result; bit 4 is already set)
#   w8  = c6 with bit 31 cleared             (inline 63: only bits [4:0] count)
#   w9  = c7 with bit 31 set                 (inline -1, 0xffffffff)
#   w10 = 0x0f0f0f0f, bit 1 cleared (literal 0xffffffe1), then bit c5[4:0] set
#   w11 = c3 with bit c3[4:0] cleared        (destination and source are one SGPR)
BITSET = [
    "s_mov_b32 s9, 0",
    "s_load_dwordx4 s[12:15], s[8:9], null",
    "s_load_dwordx4 s[16:19], s[8:9], 0x20",
    "s_waitcnt lgkmcnt(0)",
    "s_buffer_load_dwordx4 s[20:23], s[16:19], null",
    "s_buffer_load_dwordx4 s[24:27], s[16:19], 0x10",
    "s_waitcnt lgkmcnt(0)",
    "s_bitset1_b32 s20, s21",
    "s_bitset0_b32 s22, s23",
    "s_bitset1_b32 s24, s25",
    "s_mov_b32 s28, s20",
    "s_bitset1_b32 s28, 4",
    "s_bitset0_b32 s26, 63",
    "s_bitset1_b32 s27, -1",
    "s_mov_b32 s29, 0x0f0f0f0f",
    "s_bitset0_b32 s29, 0xffffffe1",
    "s_bitset1_b32 s29, s25",
    "s_bitset0_b32 s23, s23",
    "v_mov_b32_e32 v4, s20", "v_mov_b32_e32 v5, s22", "v_mov_b32_e32 v6, s24", "v_mov_b32_e32 v7, s28",
    "v_mov_b32_e32 v8, s26", "v_mov_b32_e32 v9, s27", "v_mov_b32_e32 v10, s29", "v_mov_b32_e32 v11, s23",
    "s_clause 0x1",
    "buffer_store_dwordx4 v[4:7], off, s[12:15], 0 offset:16",
    "buffer_store_dwordx4 v[8:11], off, s[12:15], 0 offset:32",
    "s_endpgm",
]
BINDING = base.RUNTIME_BINDING
FIRST_STORED = 4
# Two runs of root constants (c8..c15 are not read). In the first run c1 = 0 sets bit 0, c3 = 31 clears bit 31 and
# c5 = 0x47 (71) selects bit 7. In the second c1 = 0x20 (32) selects bit 0 again, already set, c3 = 0x3e selects bit
# 30, c5 = 0xffffffe3 selects bit 3, c6 already has bit 31 clear and c7 already has it set.
RUNS = (
    [0x12345670, 0x00000000, 0x8000ffff, 0x0000001f, 0x00000100, 0x00000047, 0xfedcba98, 0x00000001,
     *(0x5eed0000 + i for i in range(8, 16))],
    [0xffffffff, 0x00000020, 0xffffffff, 0x0000003e, 0x80000000, 0xffffffe3, 0x7fffffff, 0x80000000,
     *(0x5eed0000 + i for i in range(8, 16))],
)
# Worked by hand from the ISA definition, D.u32[S0.u32[4:0]] = 0 or 1, for the two runs above.
HAND = (
    [0x12345671, 0x0000ffff, 0x00000180, 0x12345671, 0x7edcba98, 0x80000001, 0x0f0f0f8d, 0x0000001f],
    [0xffffffff, 0xbfffffff, 0x80000008, 0xffffffff, 0x7fffffff, 0x80000000, 0x0f0f0f0d, 0x0000003e],
)


def bitset(word, index, value, *, mask=True):
    """Independent reference: rewrite one entry of the word's bit list. Unmasked, an index past 31 changes nothing."""
    bits = [(word >> i) & 1 for i in range(32)]
    if mask or index < 32:
        bits[index % 32 if mask else index] = value
    return sum(bit << i for i, bit in enumerate(bits))


def reference(c, *, swap=False, mask=True, keep=True):
    """Expected u0 words. swap exchanges set and clear, mask=False ignores src0[4:0], keep=False drops sD's old bits."""
    def op(word, index, value):
        return bitset(word if keep else (0 if value else 0xffffffff), index & 0xffffffff, value ^ swap, mask=mask)

    w4 = op(c[0], c[1], 1)
    w10 = op(op(0x0f0f0f0f, 0xffffffe1, 0), c[5], 1)
    stored = [w4, op(c[2], c[3], 0), op(c[4], c[5], 1), op(w4, 4, 1), op(c[6], 63, 0), op(c[7], -1, 1), w10,
              op(c[3], c[3], 0)]
    words = [base.SENTINEL] * 320
    words[FIRST_STORED:FIRST_STORED + len(stored)] = stored
    return words


def lowered(*changes):
    return aot.lower(base.variant(BITSET, *changes), 1, **BINDING)


def controls():
    """Exact HLSL for each form, then one rejection per operand form outside the subset."""
    for constants, hand in zip(RUNS, HAND, strict=True):
        stored = reference(constants)[FIRST_STORED:FIRST_STORED + len(hand)]
        assert stored == hand, [hex(word) for word in stored]
    # The reference controls the draw uses must each differ from the expected words.
    for wrong in (dict(swap=True), dict(mask=False), dict(keep=False)):
        assert reference(RUNS[0], **wrong) != reference(RUNS[0]), wrong
    source = lowered()
    w4 = "(params[0].x | (1u << (params[0].y & 31u)))"
    for expression in (
        w4,
        "(params[0].z & ~(1u << (params[0].w & 31u)))",
        "(params[1].x | (1u << (params[1].y & 31u)))",
        f"({w4} | (1u << (0x00000004u & 31u)))",
        "(params[1].z & ~(1u << (0x0000003fu & 31u)))",
        "(params[1].w | (1u << (0xffffffffu & 31u)))",
        "((0x0f0f0f0fu & ~(1u << (0xffffffe1u & 31u))) | (1u << (params[1].y & 31u)))",
        "(params[0].w & ~(1u << (params[0].w & 31u)))",
    ):
        assert expression in source, (expression, source)
    # w4 appears in its own store and inside w7, the bitset of its copy in s28.
    assert source.count(w4) == 2, source
    # A bitset on the copy leaves s20 alone: storing s20 after it is still w4.
    assert w4 + ")" not in source and w4 + ")" in lowered(("v_mov_b32_e32 v7, s28", "v_mov_b32_e32 v7, s20"))
    unsupported, operand = "Unsupported compute instruction: ", "Unverified scalar ALU operand"
    for message, changes in (
        # A destination that holds no known word: an unwritten SGPR, a descriptor SGPR, the table pointer and an SGPR
        # whose load still waits for its s_waitcnt.
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s30, s21"),)),
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s12, s21"),)),
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s8, s21"),)),
        (operand, (("s_bitset1_b32 s20, s21", "s_buffer_load_dword s30, s[16:19], 0x20\ns_bitset1_b32 s30, 0"),)),
        # A source that is not a known word: unwritten, descriptor, pending, or a literal wider than 32 bits.
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s20, s31"),)),
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s20, s13"),)),
        (operand, (("s_bitset1_b32 s20, s21", "s_buffer_load_dword s30, s[16:19], 0x20\ns_bitset1_b32 s20, s30"),)),
        (operand, (("s_bitset0_b32 s29, 0xffffffe1", "s_bitset0_b32 s29, 0x1ffffffe1"),)),
        # A destination past s105.
        (operand, (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s106, s21"),)),
        # A fixture register used before its s_waitcnt lgkmcnt(0): its loaded word is not yet known.
        (operand, (("s_buffer_load_dwordx4 s[24:27], s[16:19], 0x10",
                    "s_buffer_load_dwordx4 s[24:27], s[16:19], 0x10\ns_bitset1_b32 s24, 0"),)),
        # A vector source, and the 64-bit forms.
        (unsupported + "s_bitset1_b32", (("s_bitset1_b32 s20, s21", "s_bitset1_b32 s20, v4"),)),
        (unsupported + "s_bitset0_b64", (("s_bitset0_b32 s22, s23", "s_bitset0_b64 s[22:23], s21"),)),
        (unsupported + "s_bitset1_b64", (("s_bitset1_b32 s24, s25", "s_bitset1_b64 s[24:25], s21"),)),
        # Each self-referencing bitset doubles the inline expression, so a long chain is rejected.
        ("ALU expression too large", (("s_bitset0_b32 s23, s23",
                                       "s_bitset0_b32 s23, s23" + "\ns_bitset1_b32 s23, s23" * 6),)),
        # A bitset result is data only, not a store address.
        ("Unsupported store address", (("v_mov_b32_e32 v4, s20", "v_mov_b32_e32 v4, s20\nv_mov_b32_e32 v12, s20"),
                                       ("v[4:7], off, s[12:15], 0 offset:16", "v[4:7], v12, s[12:15], 0 offen"))),
    ):
        base.rejects(message, lowered, *changes)
    # In the guarded read fixture, the group-ID SGPR and the saved exec mask are not known words either.
    def guarded(*changes):
        return aot.lower(base.variant(base.READ_SCALAR, *changes), 32, **base.READ_BINDING)

    guarded()
    base.rejects(operand, guarded, ("s_mov_b32 s2, s1", "s_bitset1_b32 s1, 0\ns_mov_b32 s2, s1"))
    base.rejects(operand, guarded, ("s_lshr_b32 s21, s20, 3", "s_lshr_b32 s21, s20, 3\ns_bitset0_b32 s2, 0"))
    # Nor is the saved exec mask a known bit index.
    base.rejects(operand, guarded, ("s_lshr_b32 s21, s20, 3", "s_lshr_b32 s21, s20, 3\ns_bitset1_b32 s21, s2"))


def run(cmd, case, constants, words, *, expect=None):
    case.write_bytes(struct.pack("<337I", 1, *words, *constants))
    if expect is None:
        return subprocess.run(cmd).returncode
    bad = subprocess.run(cmd, capture_output=True, text=True)
    if bad.returncode != 1 or expect not in bad.stderr:
        raise AssertionError(f"Expected '{expect}', got {bad.returncode}\n{bad.stdout}\n{bad.stderr}")
    return 0


def draw(directory, compiler, renderer, backend):
    """Dispatch the fixture for both runs, then require wrong references and mutated translations to fail."""
    src, cso, case = [directory / x for x in ("bitset.hlsl", "bitset.cso", "bitset.case")]
    source = lowered()
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    for constants in RUNS:
        if result := run(cmd, case, constants, reference(constants)):
            return result
    # Wrong references: set and clear exchanged, src0[4:0] ignored, sD's other bits dropped, and the other run's
    # words (a translation that folded the constants).
    constants = RUNS[0]
    expected = reference(constants)
    for wrong in (reference(constants, swap=True), reference(constants, mask=False),
                  reference(constants, keep=False), reference(RUNS[1])):
        word = next(i for i, (a, b) in enumerate(zip(expected, wrong)) if a != b)
        run(cmd, case, constants, wrong, expect=f"word={word} expected=")
    run(cmd, case, constants, expected)
    w4 = "(params[0].x | (1u << (params[0].y & 31u)))"
    for label, old, new in (("set for clear", "(params[0].z & ~(1u << ", "(params[0].z | (1u << "),
                            ("clear for set", w4, "(params[0].x & ~(1u << (params[0].y & 31u)))"),
                            ("destination dropped", w4, "(0x00000000u | (1u << (params[0].y & 31u)))"),
                            ("next bit", "(1u << (params[1].y & 31u))", "(2u << (params[1].y & 31u))"),
                            ("index from c1", "(params[1].x | (1u << (params[1].y",
                             "(params[1].x | (1u << (params[0].y")):
        mutated = source.replace(old, new)
        if mutated == source:
            raise AssertionError(f"mutation '{label}' did not change the source")
        msrc, mcso = directory / "mutated.hlsl", directory / "mutated.cso"
        msrc.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mcso, msrc], check=True)
        case.write_bytes(struct.pack("<337I", 1, *expected, *constants))
        bad = subprocess.run([*cmd[:3], mcso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"mutation '{label}' escaped detection\n{bad.stdout}\n{bad.stderr}")
    return 0


if __name__ == "__main__":
    controls()
    mode, *args = sys.argv[1:]
    if mode == "controls":
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_salu_bitset_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
