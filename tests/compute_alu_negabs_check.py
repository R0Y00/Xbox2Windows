"""Check -v, |v| and -|v| sources of the VOP3 binary32 forms in the compute store profile: exact lowering, refusals,
an LLVM 15 round trip and a WARP or GPU readback."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402

aot = base.aot
NAME = "read_f32_negabs"
# cap = params[0].y is the float -0.75, so -|s0| differs from -s0 and from |s0| on every lane.
CAP = 0xbf400000
HALF_WAY = 0x3fe00000  # the float 1.75, a literal: it is not an inline constant
# Binary32 with VOP3 source modifiers, hand-edited in LLVM 15 spelling into the read-compute-write fixture. As in
# compute_store_check.READ_F32, the source word w gives x = 1.5 + (w & 0x3fffff) / 2^23 in [1.5, 2) and
# y = 3.0 + (w >> 10) / 2^22 in [3, 4). Then
#   d = x - 1.75        in [-0.25, 0.25), negative on some lanes and positive on others
#   z = 0 * d           +0.0 where d > 0 and -0.0 where d < 0
#   a = |d| + y         |v| on add, src0
#   b = -d * a          -v on mul, src0
#   c = b - (-|d|)      -|v| on sub, src1
#   e = (-c) - (-|cap|) -|v| on an SGPR and -v on src1 of subrev
#   f = |z| * y         |v| on a signed zero: +0.0 on every lane
#   out = e ^ f         bitwise, so a sign bit left on f reaches the readback
# Each modifier appears on two operations. Dropping an abs changes the lanes with d < 0 (and the cap); exchanging an
# abs for a negation changes the lanes with d > 0; dropping a negation changes every lane. The cap is negative, so its abs\n# and its negation each change every lane.
MODIFIED = ("v_add_f32_e64 v6, |v4|, v3", "v_mul_f32_e64 v7, -v4, v6", "v_sub_f32_e64 v8, v7, -|v4|",
            "v_subrev_f32_e64 v9, -|s0|, -v8", "v_mul_f32_e64 v10, |v5|, v3")
LINES = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", "v_and_b32_e32 v2, 0x3fffff, v1\nv_or_b32_e32 v2, 0x3fc00000, v2\n"
                                     "v_lshrrev_b32_e32 v3, 10, v1\nv_or_b32_e32 v3, 0x40400000, v3"),
    ("v_lshrrev_b32_e32 v1, 16, v1", f"v_subrev_f32_e32 v4, 0x{HALF_WAY:08x}, v2\nv_mul_f32_e32 v5, 0, v4\n"
                                     + "\n".join(MODIFIED)),
    ("v_mad_u32_u24 v1, v2, v1, v0", "v_xor_b32_e32 v1, v10, v9"),
    ("v_min_u32_e32 v1, s0, v1", ""),
)
BINDING = base.READ_BINDING
# The modifier of each modified source, by name: a0 is src0 of a, c1 src1 of c, and so on.
SITES = {"a0": "|", "b0": "-", "c1": "-|", "e0": "-|", "e1": "-", "f0": "|"}
ABS, NEG = "& 0x7fffffffu)", " ^ 0x80000000u)"
# The expected store, built from the step names above: a modified source is its word with bit 31 cleared or flipped.
LOAD = base.READ_LOADS["read_copy"]
X = f"(0x3fc00000u | (0x003fffffu & {LOAD}))"
Y = f"(0x40400000u | ({LOAD} >> (0x0000000au & 31u)))"
D = f"f32_sub({X}, 0x{HALF_WAY:08x}u)"
Z = f"f32_mul(0x00000000u, {D})"
A = f"f32_add(({D} & 0x7fffffffu), {Y})"
B = f"f32_mul(({D} ^ 0x80000000u), {A})"
C = f"f32_sub({B}, (({D} & 0x7fffffffu) ^ 0x80000000u))"
E = f"f32_sub(({C} ^ 0x80000000u), ((params[0].y & 0x7fffffffu) ^ 0x80000000u))"
F = f"f32_mul(({Z} & 0x7fffffffu), {Y})"
STORE = f"        outputData.Store(tid.x * 4u + 0u, ({F} ^ {E}));"


def modify(bits, modifier):
    """VOP3 source modifiers on the 32-bit word: abs clears bit 31, then neg flips it."""
    if "|" in modifier:
        bits &= 0x7fffffff
    if "-" in modifier:
        bits ^= 0x80000000
    return bits


def steps(gid, sites=SITES):
    """Independent CPU model of one lane as binary32 bit patterns, every intermediate value; sites may change one."""
    value, bits = base.f32_value, base.f32_bits

    def op(fn, a, b):
        return bits(base.f32_round(fn(value(a), value(b))))

    word = base.source_word(gid)
    x, y = 0x3fc00000 | word & 0x3fffff, 0x40400000 | word >> 10
    d = op(lambda p, q: p - q, x, HALF_WAY)
    z = op(lambda p, q: p * q, 0, d)
    a = op(lambda p, q: p + q, modify(d, sites["a0"]), y)
    b = op(lambda p, q: p * q, modify(d, sites["b0"]), a)
    c = op(lambda p, q: p - q, b, modify(d, sites["c1"]))
    e = op(lambda p, q: p - q, modify(c, sites["e1"]), modify(CAP, sites["e0"]))
    f = op(lambda p, q: p * q, modify(z, sites["f0"]), y)
    return x, y, d, z, a, b, c, e, f, e ^ f


def negabs_words(groups, bound, **reference):
    words = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        words[gid] = steps(gid, **reference)[-1]
    return words


def wrong_sites():
    """One reference per modified source with its negation dropped, its abs dropped, or abs and negation exchanged."""
    for site, modifier in SITES.items():
        changes = {"-": ("neg dropped", "", "abs for neg", "|"), "|": ("abs dropped", "", "neg for abs", "-"),
                   "-|": ("neg dropped", "|", "abs dropped", "-")}[modifier]
        for label, new in (changes[:2], changes[2:]):
            yield f"{site} {label}", SITES | {site: new}


def lower(*changes, **binding):
    return aot.lower(base.variant(LINES, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Exact modifier words, the sign-bit form on zeros and NaNs, then one rejection per form outside the subset."""
    lines = lower()
    store = lines[-3]
    assert lines[-4:] == [base.GUARD_IF, STORE, "    }", "}"], lines
    # Each modifier is an integer operation on the word, once per modified source, and never a float abs or negation.
    assert store.count(ABS) == 4 and store.count(NEG) == 4, store
    assert "abs(" not in "\n".join(lines) and "-asfloat" not in "\n".join(lines), lines

    # The sign-bit form keeps every other bit, also for signed zeros, infinities and NaN payloads, which the float
    # forms abs() and unary minus need not do. The expression is evaluated here on the word.
    vector = {2: ("alu", "W")}
    forms = {token: aot.float_e64_source(token, {}, vector, set()) for token in ("-v2", "|v2|", "-|v2|")}
    assert forms == {"-v2": "(W ^ 0x80000000u)", "|v2|": "(W & 0x7fffffffu)", "-|v2|": "((W & 0x7fffffffu) ^ 0x80000000u)"}
    for word in (0x00000000, 0x80000000, 0x7fc12345, 0xffc00001, 0x7f800001, 0xff800000, 0xbf400000, 0x00000001):
        for token, expression in forms.items():
            got = eval(re.sub(r"(0x[0-9a-f]+)u", r"\1", expression).replace("W", str(word)))
            assert got == modify(word, token.replace("v2", "")), (token, hex(word), hex(got))

    # A spelling of the same program: a source without modifier may stay plain in a modified _e64 form.
    plain = "v_add_f32_e64 v6, |v4|, v3"
    assert lower((plain, "v_add_f32_e64 v6, v3, |v4|"))[-3] != store
    assert lower((plain, "v_add_f32_e64 v6, |v4|, s0"))[-3].count("params[0].y") == 2
    # A source modifier changes the program: each one dropped (as the _e32 form where that is all) differs.
    assert lower(("v_mul_f32_e64 v7, -v4, v6", "v_mul_f32_e32 v7, v4, v6"))[-3] != store
    assert lower(("v_sub_f32_e64 v8, v7, -|v4|", "v_sub_f32_e64 v8, v7, |v4|"))[-3] != store
    assert lower(("v_sub_f32_e64 v8, v7, -|v4|", "v_sub_f32_e64 v8, v7, -v4"))[-3] != store
    assert lower(("v_subrev_f32_e64 v9, -|s0|, -v8", "v_subrev_f32_e64 v9, |s0|, -v8"))[-3] != store

    unsupported, modifier = "Unsupported compute instruction: ", "Unsupported float source modifier"
    output, operand = "Unsupported float output modifier", "Unverified float operand"
    add = "v_add_f32_e64 v6, |v4|, v3"
    for message, changes in (
        # The unmodified _e64 spelling stays refused, as does every opcode outside the four binary32 forms.
        (unsupported + "v_add_f32_e64", ((add, "v_add_f32_e64 v6, v4, v3"),)),
        (unsupported + "v_max_f32_e64", ((add, "v_max_f32_e64 v6, |v4|, v3"),)),
        (unsupported + "v_fma_f32", ((add, "v_fma_f32 v6, |v4|, v3, v2"),)),
        (unsupported + "v_add_f16_e64", ((add, "v_add_f16_e64 v6, |v4|, v3"),)),
        # Modifiers on the VOP2 spelling, which has no modifier field.
        (unsupported + "v_add_f32_e32", ((add, "v_add_f32_e32 v6, |v4|, v3"),)),
        (unsupported + "v_add_f32_e32", ((add, "v_add_f32_e32 v6, -v4, v3"),)),
        (unsupported + "v_add_f32_e32", ((add, "v_add_f32_e32 v6, -|v4|, v3"),)),
        # Output modifiers, a third source and a register range.
        (output, ((add, add + " clamp"),)),
        (output, ((add, add + " mul:2"),)),
        (output, ((add, add + " mul:4"),)),
        (output, ((add, add + " div:2"),)),
        (unsupported + "v_add_f32_e64", ((add, add + ", v2"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v[4:5]|, v3"),)),
        # Spellings LLVM 15 does not print for a register, unbalanced bars, and modifiers on constants.
        (modifier, ((add, "v_add_f32_e64 v6, abs(v4), v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, neg(v4), v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, --v4, v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |-v4|, v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4, v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, -|v4, v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, -v4|, v3"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4|, |1.0|"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4|, -|0.5|"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4|, neg(0x3f800000)"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4|, -0x3f800000"),)),
        (modifier, ((add, "v_add_f32_e64 v6, |v4|, |0x3f800000|"),)),
        # Sources that are not known words or not finite normal constants, with and without a modifier.
        (operand, ((add, "v_add_f32_e64 v6, |v40|, v3"),)),
        (operand, ((add, "v_add_f32_e64 v6, |v4|, v40"),)),
        (operand, ((add, "v_add_f32_e64 v6, -|s20|, v3"),)),
        (operand, ((add, "v_add_f32_e64 v6, -|s8|, v3"),)),
        (operand, ((add, "s_mov_b32 s20, 0x7fc00000\nv_add_f32_e64 v6, |s20|, v3"),)),
        (operand, ((add, "s_mov_b32 s20, 0x80000001\nv_add_f32_e64 v6, -s20, v3"),)),
        (operand, ((add, "v_add_f32_e64 v6, |v4|, 0x7f800000"),)),
        (operand, ((add, "v_add_f32_e64 v6, |v4|, -1"),)),
        (operand, ((add, "v_add_f32_e64 v256, |v4|, v3"),)),
        # A result is data only: not a store address and not a lane-guard operand, and a long chain is refused.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", (("v_xor_b32_e32 v1, v10, v9",
                                            "v_xor_b32_e32 v1, v10, v9\nv_cmp_gt_u32_e32 vcc_lo, s3, v6"),)),
        ("ALU expression too large", (("v_xor_b32_e32 v1, v10, v9",
                                       "v_mul_f32_e64 v9, -v9, v9\n" * 3 + "v_xor_b32_e32 v1, v10, v9"),)),
    ):
        base.rejects(message, lower, *changes)


def binary(directory, assembler, objcopy, library):
    """Assemble the fixture with LLVM 15, disassemble it and require the same spelling and the same translation."""
    src, obj, raw = [directory / x for x in ("negabs.s", "negabs.o", "negabs.bin")]
    src.write_text("\n".join(LINES) + "\n", encoding="ascii")
    subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", src, "-o", obj], check=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
    decoded = base.disassemble(raw.read_bytes(), library, offsets=True)
    printed = [line for _, line in decoded]
    missing = [line for line in MODIFIED if line not in printed]
    assert not missing, (missing, printed)
    assert aot.lower(aot.labelled(decoded), 32, **BINDING) == aot.lower(LINES, 32, **BINDING)
    print(f"{NAME}: LLVM round trip prints the {len(MODIFIED)} modified forms and translates identically")
    return 0


def draw(directory, compiler, renderer, backend):
    """Translate, compile and dispatch the fixture, then require wrong references and mutated sources to fail."""
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    source = aot.lower(LINES, 32, **BINDING)
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = CAP
    groups, bound = base.READ_RUNS[-1]

    def write(run_groups, run_bound, words):
        case.write_bytes(struct.pack("<337I", run_groups, *words, run_bound, *constants[1:]))

    # Every operand and result of an active lane is finite and normal, except z and f, which are signed zeros, and d
    # takes both signs, so each modifier is observed. f is +0.0 on every lane, z is -0.0 on some lanes.
    lanes = [steps(gid) for gid in range(bound)]
    for gid, lane in enumerate(lanes):
        x, y, d, z, a, b, c, e, f, _ = lane
        for value in (x, y, d, a, b, c, e):
            assert 0x00800000 <= value & 0x7fffffff < 0x7f800000, (gid, hex(value))
        assert z == d & 0x80000000 and f == 0, (gid, hex(z), hex(f))
    assert {lane[2] >> 31 for lane in lanes} == {0, 1}, "d has one sign on every lane"
    for run_groups, run_bound in base.READ_RUNS:
        write(run_groups, run_bound, negabs_words(run_groups, run_bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    expected = negabs_words(groups, bound)
    wrong = list(wrong_sites())
    for label, sites in wrong:
        reference = negabs_words(groups, bound, sites=sites)
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected={reference[word]:08x}" not in bad.stderr:
            raise AssertionError(f"{NAME}: reference '{label}' escaped detection\n{bad.stderr}")
    write(groups, bound, expected)
    # Each modifier occurrence in the HLSL, dropped alone or exchanged for the other modifier, must fail.
    mutations = []
    for old, labels in ((ABS, (("abs dropped", "& 0xffffffffu)"),)),
                        (NEG, (("neg dropped", " ^ 0x00000000u)"), ("neg as abs", " & 0x7fffffffu)")))):
        starts = [m.start() for m in re.finditer(re.escape(old), source)]
        assert len(starts) == 4, (old, len(starts))
        for n, start in enumerate(starts):
            for label, new in labels:
                mutations.append((f"{label} at occurrence {n}", source[:start] + new + source[start + len(old):]))
    for label, mutated in mutations:
        bad_src, bad_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        bad_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", bad_cso, bad_src], check=True)
        bad = subprocess.run([*cmd[:3], bad_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{bad.stderr}")
    print(f"{NAME}: {len(base.READ_RUNS)} dispatches, {len(wrong)} wrong references and "
          f"{len(mutations)} mutated sources checked")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls":
        sys.exit(0)
    if mode not in ("draw", "binary") or len(args) != 3:
        sys.exit("usage: compute_alu_negabs_check.py controls | draw <dxc> <renderer> <adapter>"
                 " | binary <llvm-mc> <llvm-objcopy> <libLLVM>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit((draw if mode == "draw" else binary)(Path(tmp), *args))
