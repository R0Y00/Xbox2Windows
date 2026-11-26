"""High-half and masked unsigned sum-of-absolute-differences in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls: on-demand helpers, value evaluation against the
                                       tools/pixel_aot.py formulas, the zero-src1-byte and 2^32 carry cases, and
                                       refusals; pure Python

`v_sad_hi_u8` (VOP3A opcode 0x15b, 347) and `v_msad_u8` (0x171, 369) are lowered like `v_sad_u8`, but the reference
is the AOT table in `tools/pixel_aot.py` so the two translators agree:

    "sad_hi_u8": (3, "(sad4_u8({0}, {1}, false) << 16u) + {2}"),
    "msad_u8":   (3, "sad4_u8({0}, {1}, true) + {2}"),

where `sad4_u8(a, b, masked)` sums ABSDIFF over the four byte lanes and, when masked, leaves out each byte lane whose
src1 byte is zero ("Components where the reference value in S1 are zero are not included in the sum").  Both results
are modulo 2^32.  The fixture decodes synthetic gfx1013 VOP3A bytes via G.decode() only (no game data) and evaluates
the emitted HLSL with the evaluator of tests/startup_gfx2hlsl_sad_check.py.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK,
    POST_WORDS,
    PRE_WORDS,
    Probe,
    check,
    refused,
    vs_words,
    words_text,
)
from startup_gfx2hlsl_sad_check import HlslExpr, vop3a  # noqa: E402

OPCODES = {
    'v_sad_hi_u8': 0x15b,
    'v_msad_u8': 0x171,
}

# The helper names the two instructions ask Pipeline.common() for.
HELPERS = {
    'v_sad_hi_u8': 'sad_hi_u8',
    'v_msad_u8': 'msad_u8',
}

BODY_WORDS = {
    op: b''.join(vop3a(code, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))
    for op, code in OPCODES.items()
}
# The unchanged v_sad_u8 body of tests/startup_gfx2hlsl_sad_check.py, used only for the no-regression text check.
BODY_WORDS_U8 = b''.join(vop3a(0x15a, 12 + lane, 256 + lane, 256 + 4 + lane, 256 + 8 + lane) for lane in range(4))


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


# ---- the AOT reference (tools/pixel_aot.py SAD FORMS), evaluated in Python ------------------------------------
def sad4_u8(a, b, masked):
    """Sums |x - y| over the four byte lanes; a masked lane with a zero src1 byte contributes nothing."""
    total = 0
    for shift in (0, 8, 16, 24):
        x, y = (a >> shift) & 0xff, (b >> shift) & 0xff
        if masked and y == 0:
            continue
        total += abs(x - y)
    return total


# The two formula strings above, with {0} = a, {1} = b, {2} = c and the 32-bit wraparound of the destination.
AOT_REFERENCE = {
    'v_sad_hi_u8': lambda a, b, c: ((sad4_u8(a, b, False) << 16) + c) & MASK,
    'v_msad_u8': lambda a, b, c: (sad4_u8(a, b, True) + c) & MASK,
}

# Word grids with every interesting byte boundary. 1,728 grid triples in total; src1 = 0 and the words with zero
# bytes exercise the masking rule on msad_u8, and S2 = 0xffffffff / 0xfffffc04 exercise the << 16 + S2 wrap.
U8_WORDS = (
    0x00000000, 0x01020304, 0x04030201, 0x7f80ff00, 0x807f00ff,
    0x00010000, 0x0000ffff, 0x7fffffff, 0x80000000, 0xffffffff,
    0x00ff00ff, 0xff00ff00,
)
U8_ADDENDS = (0, 1, 0x7fffffff, 0x80000000, 0xfffffc04, 0xffffffff)
GRID = {op: [(a, b, c) for a in U8_WORDS for b in U8_WORDS for c in U8_ADDENDS] for op in OPCODES}

# Explicit cases with hand-computed results: (a, b, c, v_sad_hi_u8, v_msad_u8).  The msad_u8 entries are where the
# skip-zero-src1-byte rule changes the sum; the carry rows are where (sum << 16) + S2 wraps 2^32.
CASES = [
    # normal: byte lane 3 differs by 0xe0 and the others match; both forms add c unshifted (msad does not shift).
    (0x10203040, 0xf0203040, 5, 0x00e00005, 0x000000e5),
    # every src1 byte is zero: v_msad_u8 adds only c, v_sad_hi_u8 keeps all four lanes.
    (0x01020304, 0x00000000, 5, 0x000a0005, 0x00000005),
    # lanes 0, 2 and 3 are skipped and lane 1 contributes |0x30 - 0x20| = 0x10.
    (0x10203040, 0x00002000, 0, 0x00800000, 0x00000010),
    # b = 0x000000ff: lane 0 matches and contributes nothing, lanes 1 to 3 are skipped (each would add 0xff).
    (0xffffffff, 0x000000ff, 0, 0x02fd0000, 0x00000000),
    # 1020 << 16 = 0x03fc0000; adding 0xffffffff wraps 2^32.  msad skips every lane and adds c.
    (0xffffffff, 0x00000000, 0xffffffff, 0x03fbffff, 0xffffffff),
    # (0xfe << 16) + 0xffffffff wraps as well; lane 3 is the only lane not skipped by msad (0xfe + 0xffffffff wraps).
    (0xff000000, 0x01000000, 0xffffffff, 0x00fdffff, 0x000000fd),
    # The addend of the v_sad_u8 carry control; msad adds it to a skipped-lane sum of zero.
    (0xffffffff, 0x00000000, 0xfffffc04, 0x03fbfc04, 0xfffffc04),
]

# The emitted helper text, pinned exactly (the same style as the inlined sad_u8 helper: a single return statement).
SAD4_U8_HELPER = (
    'uint sad4_u8(uint a, uint b, bool masked) {'
    ' return (masked && ((b & 0xffu) == 0u) ? 0u : uint(abs(int(a & 0xffu) - int(b & 0xffu))))'
    ' + (masked && (((b >> 8) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu))))'
    ' + (masked && (((b >> 16) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu))))'
    ' + (masked && (((b >> 24) & 0xffu) == 0u) ? 0u : uint(abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu)))); }')
EXPECTED_HELPER = {
    'v_sad_hi_u8': 'uint sad_hi_u8(uint a, uint b, uint c) { return (sad4_u8(a, b, false) << 16u) + c; }',
    'v_msad_u8': 'uint msad_u8(uint a, uint b, uint c) { return sad4_u8(a, b, true) + c; }',
}

# The unchanged inline helper of the existing v_sad_u8 lowering, which must stay byte for byte as it was.
SAD_U8_HELPER = (
    'uint sad_u8(uint a, uint b, uint c) {'
    ' return uint(abs(int(a & 0xffu) - int(b & 0xffu)))'
    ' + uint(abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu)))'
    ' + uint(abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu)))'
    ' + uint(abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu))) + c; }')


def translate(op):
    pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob())
    return pipe, pipe.hlsl_ps()


def extract_helpers(hlsl_text):
    """Parse emitted helpers into 32-bit evaluators.  sad4_u8 has the bool `masked` argument of the AOT helper."""
    helpers = {}
    m = re.search(r'^uint sad4_u8\(uint a, uint b, bool masked\) \{ return (.*); \}$', hlsl_text, re.M)
    if m:
        expr = HlslExpr(m.group(1))
        helpers['sad4_u8'] = lambda a, b, masked, e=expr: e.run({'a': a, 'b': b, 'masked': 1 if masked else 0})[0]
    for name in ('sad_hi_u8', 'msad_u8'):
        m = re.search(r'^uint %s\(uint a, uint b, uint c\) \{ return (.*); \}$' % name, hlsl_text, re.M)
        if m:
            expr = HlslExpr(m.group(1), helpers)
            helpers[name] = lambda a, b, c, e=expr: e.run({'a': a, 'b': b, 'c': c})[0]
    return helpers


def lane_evaluators(hlsl_text):
    helpers = extract_helpers(hlsl_text)
    found = re.findall(r'^\s*v(1[2-5]) = (.*);$', hlsl_text, re.M)
    assert [r for r, _ in found] == ['12', '13', '14', '15'], found
    return helpers, [HlslExpr(rhs, helpers) for _, rhs in found]


def eval_lane(expr, lane, triple):
    a, b, c = triple
    return expr.run({'v%d' % lane: a, 'v%d' % (4 + lane): b, 'v%d' % (8 + lane): c})[0]


def mentions(hlsl_text, name):
    """True when `name` is defined or called in the text, without matching another name's suffix (msad_u8/sad_u8)."""
    return re.search(r'(?<![A-Za-z0-9_])%s\(' % re.escape(name), hlsl_text) is not None


def controls():
    # 1. On-demand helpers: none without a SAD op, and the existing v_sad_u8 lowering is unchanged (it keeps its
    # inlined four-term helper and must not pull in the shared sad4_u8 helper).
    base_pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + POST_WORDS, root_blob())
    base_text = base_pipe.hlsl_ps() + base_pipe.hlsl_vs()
    check('on-demand: baseline pipeline without v_sad_* emits no sad4_u8/sad_hi_u8/msad_u8 helper',
          not any(mentions(base_text, h) for h in ('sad4_u8', 'sad_hi_u8', 'msad_u8')))

    u8_pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + BODY_WORDS_U8 + POST_WORDS, root_blob())
    u8_text = u8_pipe.hlsl_ps()
    check('existing v_sad_u8: helper text and call sites are unchanged and no new SAD helper is emitted',
          SAD_U8_HELPER in u8_text and 'v12 = sad_u8(v0, v4, v8);' in u8_text
          and not any(mentions(u8_text, h) for h in ('sad4_u8', 'sad_hi_u8', 'msad_u8')))

    for op, helper in HELPERS.items():
        pipe, text = translate(op)
        helpers, exprs = lane_evaluators(text)

        # 1a. Four lane calls in the emitted HLSL.
        found = re.findall(r'^\s*v(1[2-5]) = %s\(v(\d+), v(\d+), v(\d+)\);$' % helper, text, re.M)
        check('%s: four lane calls emitted through %s' % (op, helper),
              found == [('12', '0', '4', '8'), ('13', '1', '5', '9'),
                        ('14', '2', '6', '10'), ('15', '3', '7', '11')])

        # 1b. Value-level evaluation of the emitted HLSL against the AOT formula over 864 edge/grid triples.
        grid = GRID[op]
        mismatch = next(((i % 4, triple, eval_lane(exprs[i % 4], i % 4, triple), AOT_REFERENCE[op](*triple))
                         for i, triple in enumerate(grid)
                         if eval_lane(exprs[i % 4], i % 4, triple) != AOT_REFERENCE[op](*triple)), None)
        check('%s: emitted HLSL matches the AOT formula on %d edge/grid triples (first mismatch: %r)'
              % (op, len(grid), mismatch), mismatch is None)

        # 1c. The explicit normal, zero-src1-byte and carry cases, against the hand-computed results.  The same
        # table is checked against the AOT formula, so the pin and the formula cannot drift together.
        op_index = 3 if op == 'v_sad_hi_u8' else 4
        rows = [(case[:3], case[op_index], AOT_REFERENCE[op](*case[:3]), eval_lane(exprs[0], 0, case[:3]))
                for case in CASES]
        bad = [(triple, pinned, formula, got) for triple, pinned, formula, got in rows
               if not (pinned == formula == got)]
        check('%s: pinned normal/zero-lane/carry rows all match (%s; first bad: %r)'
              % (op, words_text(r[3] for r in rows), bad[0] if bad else None), not bad)
        # The zero-src1-byte rule is what separates the two forms on these rows; require the cases to show it.
        if op == 'v_msad_u8':
            zero_lane = next(((case[0], case[1], case[2]) for case in CASES
                              if sad4_u8(case[0], case[1], False) != sad4_u8(case[0], case[1], True)), None)
            check('%s: a row with skipped src1 byte lanes separates msad_u8 from the unmasked sum (%r)'
                  % (op, zero_lane), zero_lane is not None)

        # 1d. Exact emitted helper text, on demand only: the shared sad4_u8 plus the thin wrapper, and neither the
        # other new helper nor the old sad_u8/sad_u16/sad_u32 helpers.
        others = [h for o, h in HELPERS.items() if o != op]
        check('%s: on-demand emits exactly sad4_u8 plus %s and no other SAD helper' % (op, helper),
              SAD4_U8_HELPER in text and EXPECTED_HELPER[op] in text
              and not any(mentions(text, h) for h in others + ['sad_u8', 'sad_u16', 'sad_u32'])
              and text.count('uint sad4_u8(') == 1)

        # 1e. The pipeline records the op for the helper emission, and both stages share the common preamble, so
        # both emitted shaders carry the same two helpers and nothing else from the SAD family.
        check('%s: uses_sad accounting names only %s and the shared preamble carries only its two helpers'
              % (op, op),
              pipe.uses_sad == {op}
              and all(SAD4_U8_HELPER in t and EXPECTED_HELPER[op] in t for t in (text, pipe.hlsl_vs()))
              and not any(mentions(pipe.hlsl_vs(), h) for h in others + ['sad_u8', 'sad_u16', 'sad_u32']))

    # Both new forms in one pixel stage: the shared sad4_u8 helper is emitted once and each wrapper once.
    both_pipe = G.Pipeline(vs_words([(0, 0, 0)] * 4),
                           PRE_WORDS + BODY_WORDS['v_sad_hi_u8'] + BODY_WORDS['v_msad_u8'] + POST_WORDS,
                           root_blob())
    both_text = both_pipe.hlsl_ps()
    check('both forms in one stage: sad4_u8 once, sad_hi_u8 once, msad_u8 once',
          both_text.count('uint sad4_u8(') == 1 and both_text.count('uint sad_hi_u8(') == 1
          and both_text.count('uint msad_u8(') == 1 and both_pipe.uses_sad == set(OPCODES))

    # 2. ComputePipeline on-demand helper check + SGPR / inline constant / literal encodings.
    for op, code in OPCODES.items():
        cs_words = vop3a(code, 1, 4, 129, 255, lit=0x12345678) + struct.pack('<I', 0xbf810000)
        cs_pipe = G.ComputePipeline(cs_words, root_blob(), (4 << 1) | (1 << 7), (1, 1, 1))
        cs_hlsl = cs_pipe.hlsl_cs()
        check('ComputePipeline %s: SGPR/inline/literal emits sad4_u8 and %s only' % (op, HELPERS[op]),
              '%s(s4, 0x00000001u, 0x12345678u)' % HELPERS[op] in cs_hlsl
              and SAD4_U8_HELPER in cs_hlsl and EXPECTED_HELPER[op] in cs_hlsl
              and not any(mentions(cs_hlsl, h) for o, h in HELPERS.items() if o != op))

        # Genuine inline -1 (src=193, neg=0) and literal (src=255) decoded via G.decode().
        bad_word = vop3a(code, 12, 256, 193, 255, lit=0xabcdef01)
        ins = G.decode(bad_word)
        assert len(ins) == 1, ins
        _, _, name, asm = ins[0]
        stage = Probe()
        stage.one(name, asm)
        check('%s: decoded inline -1 and literal lower cleanly' % op,
              stage.lines == ['v12 = %s(v0, 0xffffffffu, 0xabcdef01u);' % HELPERS[op]])

        # Exec-masked emission.
        masked = Probe()
        masked.exec_known = False
        masked.one(op, '%s v1, v2, v3, v4' % op)
        check('%s: exec-masked emission guards assignment' % op,
              masked.lines == ['if (exec & 1) v1 = %s(v2, v3, v4);' % HELPERS[op]])

    # 3. Refusal controls (binary VOP3A decoded via G.decode() + Stage.one assembly/arity/unlowered).
    for op, code in OPCODES.items():
        modifiers = [
            ('clamp', vop3a(code, 12, 256, 260, 264, clamp=1)),
            ('omod=mul:2', vop3a(code, 12, 256, 260, 264, omod=1)),
            ('omod=mul:4', vop3a(code, 12, 256, 260, 264, omod=2)),
            ('omod=div:2', vop3a(code, 12, 256, 260, 264, omod=3)),
            ('op_sel=[1,0,0]', vop3a(code, 12, 256, 260, 264, op_sel=1)),
            ('op_sel=[0,0,0,1]', vop3a(code, 12, 256, 260, 264, op_sel=8)),
        ]
        abs_neg = [
            ('abs src0 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=1)),
            ('abs src1 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=2)),
            ('abs src2 VGPR', vop3a(code, 12, 256, 260, 264, abs_bits=4)),
            ('abs src0 inline 1', vop3a(code, 12, 129, 260, 264, abs_bits=1)),
            ('abs src1 inline 1', vop3a(code, 12, 256, 129, 264, abs_bits=2)),
            ('abs src2 inline 1', vop3a(code, 12, 256, 260, 129, abs_bits=4)),
            ('neg src0 VGPR', vop3a(code, 12, 256, 260, 264, neg=1)),
            ('neg src1 VGPR', vop3a(code, 12, 256, 260, 264, neg=2)),
            ('neg src2 VGPR', vop3a(code, 12, 256, 260, 264, neg=4)),
            ('neg src0 inline 1', vop3a(code, 12, 129, 260, 264, neg=1)),
            ('neg src1 inline 1', vop3a(code, 12, 256, 129, 264, neg=2)),
            ('neg src2 inline 1', vop3a(code, 12, 256, 260, 129, neg=4)),
        ]
        for label, bad_word in modifiers + abs_neg:
            try:
                G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + bad_word + POST_WORDS, root_blob())
            except G.Unsupported:
                pass
            else:
                raise AssertionError('%s: accepted binary VOP3A with %s' % (op, label))
        for label, bad_word in abs_neg:
            reasons = []
            G.decode(bad_word, reasons)
            check('%s: encoding guard refuses %s' % (op, label),
                  [reason for reason, where in reasons] == ['%s with neg/abs' % op])

        # Direct Stage.one refusals: modifiers, abs/neg, arity.
        check('%s: Stage refuses modifiers, abs/neg, and wrong operand counts' % op, all(refused(asm) for asm in (
            '%s v1, v2, v3, v4 clamp' % op,
            '%s v1, v2, v3, v4 mul:2' % op,
            '%s v1, v2, v3, v4 op_sel:[1,0,0]' % op,
            '%s v1, |v2|, v3, v4' % op,
            '%s v1, v2, |v3|, v4' % op,
            '%s v1, v2, v3, |v4|' % op,
            '%s v1, -v2, v3, v4' % op,
            '%s v1, v2, -v3, v4' % op,
            '%s v1, v2, v3, -v4' % op,
            '%s v1' % op,
            '%s v1, v2' % op,
            '%s v1, v2, v3' % op,
            '%s v1, v2, v3, v4, v5' % op,
        )))

    # 4. Unlowered control: removing the entries from VEC_U must cause G.Pipeline and Stage.one to refuse.
    saved = {op: G.VEC_U.pop(op) for op in OPCODES}
    try:
        for op in OPCODES:
            try:
                translate(op)
            except G.Unsupported:
                pass
            else:
                raise AssertionError('%s: translated after removal from VEC_U' % op)
            assert refused('%s v1, v2, v3, v4' % op), op
    finally:
        G.VEC_U.update(saved)
    check('unlowered control: removing v_sad_hi_u8/v_msad_u8 from VEC_U restores refusals', True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    raise SystemExit(__doc__)


if __name__ == '__main__':
    main()