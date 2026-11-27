"""Check the compute VOP1 bit forms v_bfrev_b32 and v_ffbh_i32 between a guarded read and its store.

Modes:
    controls                                    translator only: exact output and rejections
    draw <dxc> <renderer> <hardware|warp>       translate, dispatch and compare the full UAV readback
"""
from pathlib import Path
from unittest.mock import patch
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compute_store_check as base  # noqa: E402

aot = base.aot
MASK = 0xffffffff

# The read-compute-write fixture of compute_store_check.py (READ_COMPUTE) with its four ALU lines replaced by the
# two VOP1 forms, written by hand in the LLVM 15 spelling, and a 16-byte element per lane. With w the source word:
#   out[4 * gid + 0] = bfrev(w)
#   out[4 * gid + 1] = ffbh_i32(w << (w & 31))
#   out[4 * gid + 2] = ffbh_i32(ashr(bfrev(w), (w >> 5) & 31))
#   out[4 * gid + 3] = ((bfrev(gid) ^ ffbh_i32(cap)) + ffbh_i32(-1)) ^ bfrev(0x12345678)
# The shifted words give both signs and leading runs of 1 to 30 bits (the cap gives 31), and the arithmetic shift
# also gives 0 and 0xffffffff. The sources cover a VGPR read from t0, a computed VGPR, the dispatch index, an SGPR (cap =
# params[0].y), an inline constant and a literal.
VOP1_CHAIN = ("v_bfrev_b32_e32 v2, v1\n"
              "v_lshlrev_b32_e32 v3, v1, v1\nv_ffbh_i32_e32 v3, v3\n"
              "v_lshrrev_b32_e32 v4, 5, v1\nv_ashrrev_i32_e32 v4, v4, v2\nv_ffbh_i32_e32 v4, v4\n"
              "v_bfrev_b32_e32 v5, v0\nv_ffbh_i32_e32 v6, s0\nv_xor_b32_e32 v5, v6, v5\n"
              "v_ffbh_i32_e32 v6, -1\nv_add_nc_u32_e32 v5, v6, v5\n"
              "v_bfrev_b32_e32 v6, 0x12345678\nv_xor_b32_e32 v5, v6, v5")
READ_VOP1 = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", VOP1_CHAIN),
    ("v_lshrrev_b32_e32 v1, 16, v1", ""),
    ("v_mad_u32_u24 v1, v2, v1, v0", ""),
    ("v_min_u32_e32 v1, s0, v1", ""),
    ("buffer_store_dword v1, v0, s[8:11], 0 idxen", "buffer_store_dwordx4 v[2:5], v0, s[8:11], 0 idxen"),
)
VOP1_BINDING = base.READ_BINDING | dict(stride=16)
LOAD = base.READ_LOADS["read_copy"]


def bfrev_text(x):
    return f"reversebits({x})"


def ffbh_text(x):
    return f"firstbitlow(reversebits({x} ^ asuint(asint({x}) >> 31u)))"


SHIFTED = f"({LOAD} << ({LOAD} & 31u))"
SPREAD = f"asuint(asint({bfrev_text(LOAD)}) >> (({LOAD} >> (0x00000005u & 31u)) & 31u))"


def vop1_store(index="tid.x", cap="params[0].y", inline="0xffffffffu"):
    """The guarded store, with the sources of the three word-3 instructions that take a known word."""
    constant = f"({bfrev_text('0x12345678u')} ^ ({ffbh_text(inline)} + ({ffbh_text(cap)} ^ {bfrev_text(index)})))"
    return (f"        outputData.Store4(tid.x * 16u + 0u, uint4({bfrev_text(LOAD)}, {ffbh_text(SHIFTED)}, "
            f"{ffbh_text(SPREAD)}, {constant}));")


def bfrev(x):
    """Bit i of the result is bit 31 - i of x."""
    return int(f"{x:032b}"[::-1], 2)


def ffbh_i32(x):
    """AMD RDNA 2 ISA, V_FFBH_I32: D = -1; for i in 1..31, the first i with S0[31 - i] != S0[31] is D."""
    for i in range(1, 32):
        if (x >> (31 - i)) & 1 != x >> 31:
            return i
    return MASK


def ffbh_u32(x):
    """V_FFBH_U32, the unsigned scan: the MSB-first index of the first set bit, -1 for zero."""
    return next((i for i in range(32) if (x >> (31 - i)) & 1), MASK)


def ashr(x, n):
    return ((x - (1 << 32) if x >> 31 else x) >> (n & 31)) & MASK


# Wrong models of the two forms for the reference controls. Each must differ from the right one on the fixture.
WRONG_MODELS = {
    "ffbh_u32 for ffbh_i32": dict(ffbh=ffbh_u32),
    "ffbh counting after the sign bit": dict(ffbh=lambda x: ffbh_i32(x) - 1 if ffbh_i32(x) != MASK else MASK),
    "ffbh as an LSB-first index": dict(ffbh=lambda x: 31 - ffbh_i32(x) if ffbh_i32(x) != MASK else MASK),
    "ffbh 32 for 0 and -1": dict(ffbh=lambda x: 32 if ffbh_i32(x) == MASK else ffbh_i32(x)),
    "bfrev as a copy": dict(reverse=lambda x: x),
}
VOP1_CAP = 0xfffffffe  # ffbh_i32 = 31: only bit 0 differs from the sign bit
# (groups, bound): no lane passes, a split wave, and three groups with 70 active lanes, 280 of the 320 words.
VOP1_RUNS = ((1, 0), (1, 17), (3, 70))


def lane_words(gid, *, reverse=bfrev, ffbh=ffbh_i32):
    """Independent CPU reference of one lane's four words."""
    w = base.source_word(gid)
    shifted = (w << (w & 31)) & MASK
    spread = ashr(reverse(w), w >> 5)
    constant = (((reverse(gid) ^ ffbh(VOP1_CAP)) + ffbh(MASK)) & MASK) ^ reverse(0x12345678)
    return [reverse(w), ffbh(shifted), ffbh(spread), constant]


def vop1_words(groups, bound, **model):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[4 * gid:4 * gid + 4] = lane_words(gid, **model)
    return words


def reference_controls():
    """The CPU model against hand-worked ISA cases, and the fixture's coverage of them."""
    for x, expected in ((0, MASK), (MASK, MASK), (1, 31), (0xfffffffe, 31), (0x80000000, 1), (0x7fffffff, 1),
                        (0x40000000, 1), (0xbfffffff, 1), (0x3fffffff, 2), (0xc0000000, 2), (0x0000ffff, 16),
                        (0xffe0c0de, 11)):
        assert ffbh_i32(x) == expected, (hex(x), ffbh_i32(x))
    for x, expected in ((0, 0), (1, 0x80000000), (0x80000000, 1), (0x12345678, 0x1e6a2c48), (0xf0000000, 0xf)):
        assert bfrev(x) == expected, (hex(x), hex(bfrev(x)))
    groups, bound = VOP1_RUNS[-1]
    lanes = [base.source_word(gid) for gid in range(bound)]
    inputs = [(w << (w & 31)) & MASK for w in lanes] + [ashr(bfrev(w), w >> 5) for w in lanes] + [VOP1_CAP]
    # Both signs, both all-equal words, the shortest and the longest run.
    assert {0, MASK} <= set(inputs) and any(x >> 31 and x != MASK for x in inputs)
    assert any(0 < x < 1 << 31 for x in inputs)
    assert {1, 31} <= {ffbh_i32(x) for x in inputs}
    expected = vop1_words(groups, bound)
    for label, model in WRONG_MODELS.items():
        assert vop1_words(groups, bound, **model) != expected, f"{label} would go unseen"


def controls():
    """Translator only: exact output, then one rejection per form or operand outside the subset."""
    reference_controls()

    def vop1(*changes, **binding):
        return aot.lower(base.variant(READ_VOP1, *changes), 32, **(VOP1_BINDING | binding)).splitlines()

    lines = vop1()
    assert lines[:3] == ["cbuffer Constants : register(b0) { uint4 params[4]; };",
                         "ByteAddressBuffer inputData : register(t0);",
                         "RWByteAddressBuffer outputData : register(u0);"], lines
    assert lines[-4:] == [base.GUARD_IF, vop1_store(), "    }", "}"], lines[-4:]
    # A copy of a result is the same program.
    assert vop1(("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e32 v7, v1\nv_mov_b32_e32 v2, v7")) == lines
    # Each source kind prints as its own word.
    for source, word in (("s0", "params[0].y"), ("-16", "0xfffffff0u"), ("0x80000000", "0x80000000u"),
                         ("v0", "tid.x"), ("v1", LOAD)):
        assert vop1(("v_ffbh_i32_e32 v6, -1", f"v_ffbh_i32_e32 v6, {source}"))[-3] == vop1_store(inline=word)
        assert vop1(("v_ffbh_i32_e32 v6, s0", f"v_ffbh_i32_e32 v6, {source}"))[-3] == vop1_store(cap=word)
        assert vop1(("v_bfrev_b32_e32 v5, v0", f"v_bfrev_b32_e32 v5, {source}"))[-3] == vop1_store(index=word)
    unsupported, operand = "Unsupported compute instruction: ", "Unverified ALU operand"
    for message, changes in (
        # The VOP3, SDWA and DPP encodings of both forms, source and output modifiers, and a register range.
        (unsupported + "v_bfrev_b32_e64", (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e64 v2, v1"),)),
        (unsupported + "v_ffbh_i32_e64", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_e64 v3, v3"),)),
        (unsupported + "v_ffbh_i32_e64", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_e64 v3, -v3"),)),
        (unsupported + "v_bfrev_b32_sdwa", (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_sdwa v2, v1 dst_sel:DWORD "
                                             "dst_unused:UNUSED_PAD src0_sel:WORD_1"),)),
        (unsupported + "v_ffbh_i32_sdwa", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_sdwa v3, v3 dst_sel:DWORD "
                                            "dst_unused:UNUSED_PAD src0_sel:BYTE_0"),)),
        (unsupported + "v_ffbh_i32_dpp", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_dpp v3, v3 quad_perm:[1,0,3,2] "
                                           "row_mask:0xf bank_mask:0xf"),)),
        (unsupported + "v_bfrev_b32_e32", (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e32 v2, -v1"),)),
        (unsupported + "v_bfrev_b32_e32", (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e32 v2, |v1|"),)),
        (unsupported + "v_ffbh_i32_e32", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_e32 v3, v3 clamp"),)),
        (unsupported + "v_bfrev_b32_e32", (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e32 v2, v[1:2]"),)),
        # Neighbouring opcodes outside the family, and the scalar forms.
        (unsupported + "v_ffbh_u32_e32", (("v_ffbh_i32_e32 v3, v3", "v_ffbh_u32_e32 v3, v3"),)),
        (unsupported + "v_ffbl_b32_e32", (("v_ffbh_i32_e32 v3, v3", "v_ffbl_b32_e32 v3, v3"),)),
        (unsupported + "v_not_b32_e32", (("v_bfrev_b32_e32 v2, v1", "v_not_b32_e32 v2, v1"),)),
        (unsupported + "s_brev_b32", (("v_ffbh_i32_e32 v6, s0", "s_brev_b32 s16, s0\nv_ffbh_i32_e32 v6, s16"),)),
        (unsupported + "s_flbit_i32", (("v_ffbh_i32_e32 v6, s0", "s_flbit_i32 s16, s0\nv_ffbh_i32_e32 v6, s16"),)),
        # Sources that are not known words: an unknown VGPR, an unwritten, descriptor, saved exec-mask, pending or
        # out-of-range SGPR, a literal wider than 32 bits and a byte offset. Also a destination past v255.
        (operand, (("v_ffbh_i32_e32 v3, v3", "v_ffbh_i32_e32 v3, v9"),)),
        (operand, (("v_ffbh_i32_e32 v6, s0", "v_ffbh_i32_e32 v6, s20"),)),
        (operand, (("v_ffbh_i32_e32 v6, s0", "v_ffbh_i32_e32 v6, s8"),)),
        (operand, (("v_ffbh_i32_e32 v6, s0", "v_ffbh_i32_e32 v6, s2"),)),
        (operand, (("v_ffbh_i32_e32 v6, s0", "s_buffer_load_dword s20, s[4:7], 0x8\nv_ffbh_i32_e32 v6, s20"),)),
        (operand, (("v_ffbh_i32_e32 v6, s0", "v_ffbh_i32_e32 v6, s106"),)),
        (operand, (("v_bfrev_b32_e32 v6, 0x12345678", "v_bfrev_b32_e32 v6, 0x112345678"),)),
        (operand, (("v_bfrev_b32_e32 v5, v0", "v_lshlrev_b32_e32 v7, 2, v0\nv_bfrev_b32_e32 v5, v7"),)),
        (operand, (("v_bfrev_b32_e32 v2, v1", "v_bfrev_b32_e32 v256, v1"),)),
        ("Vector load result used or overwritten before s_waitcnt vmcnt(0)", (("s_waitcnt vmcnt(0)", ""),)),
        # A result is data only: not a store or read address and not a lane-guard operand.
        ("Unsupported store address", (("v[2:5], v0, s[8:11]", "v[2:5], v3, s[8:11]"),)),
        ("Unsupported read address", (("buffer_store_dwordx4", "buffer_load_dword v7, v4, s[12:15], 0 offen\n"
                                       "buffer_store_dwordx4"),)),
        ("Unverified lane guard compare", (("buffer_store_dwordx4", "v_cmp_gt_u32_e32 vcc_lo, s3, v3\n"
                                            "buffer_store_dwordx4"),)),
        # ffbh_i32 names its source twice, so a chain of them is refused once it passes the limit.
        ("ALU expression too large", (("v_ffbh_i32_e32 v4, v4", "v_ffbh_i32_e32 v4, v4"
                                       + "\nv_ffbh_i32_e32 v4, v4" * 2),)),
    ):
        base.rejects(message, vop1, *changes)
    # With the x8 descriptor load moved before the guard, a result computed under the guard is unknown after the
    # restore, and one computed after it is known.
    restored = (("s_load_dwordx8 s[8:15], s[0:1], 0x0", ""),
                ("s_and_saveexec_b32 s2, vcc_lo", "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
                                                  "s_and_saveexec_b32 s2, vcc_lo"),
                ("s_cbranch_execz .LBB0_2", ""), (".LBB0_2:", ""),
                ("buffer_store_dwordx4 v[2:5], v0, s[8:11], 0 idxen",
                 "buffer_store_dwordx3 v[2:4], v0, s[8:11], 0 idxen\ns_or_b32 exec_lo, exec_lo, s2\n"
                 "buffer_store_dword v5, v0, s[8:11], 0 idxen offset:12"))
    base.rejects("Store data is not a known word", vop1, *restored)
    after = vop1(*restored[:-1], (restored[-1][0], restored[-1][1].replace(
        "\nbuffer_store_dword v5", "\nv_bfrev_b32_e32 v5, 1\nbuffer_store_dword v5")))
    assert after[-3:] == ["    }", "    outputData.Store(tid.x * 16u + 12u, reversebits(0x00000001u));", "}"], after
    return 0


# Translations that a wrong template would produce; each must fail on the GPU.
TEMPLATE_MUTATIONS = (
    ("bfrev as a copy", "v_bfrev_b32_e32", "({0})"),
    ("bfrev as a complement", "v_bfrev_b32_e32", "(~{0})"),
    ("ffbh_i32 as ffbh_u32", "v_ffbh_i32_e32", "firstbitlow(reversebits({0}))"),
    ("logical sign fold", "v_ffbh_i32_e32", "firstbitlow(reversebits({0} ^ ({0} >> 31u)))"),
    ("LSB-first index", "v_ffbh_i32_e32", "firstbithigh({0} ^ asuint(asint({0}) >> 31u))"),
    ("32 for 0 and -1", "v_ffbh_i32_e32", "(31u - firstbithigh({0} ^ asuint(asint({0}) >> 31u)))"),
    ("count after the sign bit", "v_ffbh_i32_e32", "(firstbitlow(reversebits({0} ^ asuint(asint({0}) >> 31u))) - 1u)"),
)


def draw(directory, compiler, renderer, backend):
    controls()
    src, cso, case = (directory / f"vop1.{ext}" for ext in ("hlsl", "cso", "case"))
    compile_hlsl = [compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src]
    run = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = VOP1_CAP

    def write(groups, bound, words):
        case.write_bytes(struct.pack("<337I", groups, *words, bound, *constants[1:]))

    source = aot.lower(READ_VOP1, 32, **VOP1_BINDING)
    src.write_text(source, encoding="ascii")
    subprocess.run(compile_hlsl, check=True)
    for groups, bound in VOP1_RUNS:
        print(f"read_vop1: groups={groups} bound={bound}", flush=True)
        write(groups, bound, vop1_words(groups, bound))
        result = subprocess.run(run)
        if result.returncode:
            return result.returncode
    groups, bound = VOP1_RUNS[-1]
    expected = vop1_words(groups, bound)
    # Reference controls: one word off, the first lane past the bound writing, and each wrong model.
    off = list(expected)
    off[4 * bound - 1] ^= 1
    for label, reference in (("one word off", off), ("lane past the bound", vop1_words(groups, bound + 1)),
                             *((label, vop1_words(groups, bound, **model)) for label, model in WRONG_MODELS.items())):
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(run, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected=" not in bad.stderr:
            raise AssertionError(f"read_vop1: reference '{label}' escaped detection\n{bad.stdout}\n{bad.stderr}")
        print(f"read_vop1: reference '{label}' fails at word {word}", flush=True)
    write(groups, bound, expected)
    mutated_src, mutated_cso = directory / "mutated.hlsl", directory / "mutated.cso"
    for label, op, template in TEMPLATE_MUTATIONS:
        with patch.dict(aot.VOP1_ALU, {op: template}):
            mutated = aot.lower(READ_VOP1, 32, **VOP1_BINDING)
        if mutated == source:
            raise AssertionError(f"read_vop1: mutation '{label}' did not change the source")
        mutated_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", mutated_cso, mutated_src], check=True)
        bad = subprocess.run([*run[:3], mutated_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"read_vop1: mutation '{label}' escaped detection\n{bad.stdout}\n{bad.stderr}")
        print(f"read_vop1: mutation '{label}' fails", flush=True)
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("v_bfrev_b32_e32\nv_ffbh_i32_e32\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "controls" and not args:
        sys.exit(controls())
    if mode == "draw" and len(args) == 3:
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(draw(Path(tmp), *args))
    sys.exit(f"usage: {Path(__file__).name} controls | draw <dxc> <renderer> <hardware|warp>")
