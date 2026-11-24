"""Negated sources on the vector select in tools/startup/gfx2hlsl.py (issue #1709).

usage: startup_gfx2hlsl_cndmask_neg_check.py

A VOP3 `v_cndmask_b32` can carry the `-` source modifier, which in an integer instruction is the source's
sign bit flipped. The select handler passed the printed operand straight to the uint helper, so
`v_cndmask_b32_e64 v31, v30, -v30` was refused with `Unsupported: operand '-v30'` - 2 of the 29 refused
shaders in the first main-game loading-screen cache (#870).

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch):
  w0: [31:26] = 0b110101, [25:16] opcode (VOP2 e64 = 0x100 + VOP2 opcode), [7:0] vdst
  w1: [8:0] src0, [17:9] src1, [28:27] omod, [31:29] neg
"""
import os
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


VOP2_E64_BASE = 0x100
CNDMASK = 0x01
AND_B32 = 0x1b
V30 = 256 + 30  # VGPR operands are encoded from 256 upwards


def encode_vop2_e64(op, vdst, src0, src1, *, neg0=0, neg1=0, abs0=0, abs1=0, clamp=0, omod=0, neg_mask=0):
    """Encode a VOP2 instruction in its VOP3A (e64) form."""
    w0 = ((0b110101 << 26) | ((VOP2_E64_BASE + op) << 16) | ((clamp & 1) << 15)
          | ((abs1 & 1) << 9) | ((abs0 & 1) << 8) | (vdst & 0xff))
    w1 = ((src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((omod & 0x3) << 27) | ((neg0 & 1) << 29)
          | ((neg1 & 1) << 30) | ((neg_mask & 1) << 31))
    return struct.pack('<II', w0, w1)


class Pipeline:
    """Just enough pipeline state for one vector instruction."""

    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(code):
    try:
        translate(code)
    except G.Unsupported:
        return True
    return False


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, neg1=1))
check('encodes and decodes the negated select',
      asm == ['v_cndmask_b32_e64 v31, v30, -v30, s0'], repr(asm))

# 2. A negation is the source's sign bit, and it applies to the source it is printed on.
check('negated source flips its sign bit',
      emitted == 'v31 = (s0 & 1) ? (v30 ^ 0x80000000u) : v30;', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, neg0=1))
check('negation on the other source',
      'v31 = (s0 & 1) ? v30 : (v30 ^ 0x80000000u);' in emitted, emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, neg0=1, neg1=1))
check('both sources negated', emitted.count('0x80000000u') == 2, emitted)

# 3. A select without a negation emits exactly what it did before.
asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30))
check('no negation is unchanged', emitted == 'v31 = (s0 & 1) ? v30 : v30;', emitted)

# 4. The negation is only modelled where the handler asks for it: another integer instruction carrying the
#    same modifier stays refused rather than silently flipping a bit. (`llvm-mc` rejects `-v30` on
#    `v_and_b32_e64`, so this encoding is not one an assembler produces; the property under test is that the
#    decoder's negated text is refused by a handler that does not model it.)
check('a negation on another integer op stays refused',
      refused(encode_vop2_e64(AND_B32, 1, V30, V30, neg1=1)))

# Plain negative inline constants keep their bit patterns even when the printed spelling is identical.
asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, 193, V30))
check('plain inline -1 still translates',
      asm == ['v_cndmask_b32_e64 v31, -1, v30, s0'] and
      'v31 = (s0 & 1) ? v30 : 0xffffffffu;' in emitted, emitted)
asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, 243, V30))
check('plain inline -1.0 still translates',
      asm == ['v_cndmask_b32_e64 v31, -1.0, v30, s0'] and
      'v31 = (s0 & 1) ? v30 : 0xbf800000u;' in emitted, emitted)
check('NEG on inline integer remains refused', refused(encode_vop2_e64(CNDMASK, 31, 129, V30, neg0=1)))
check('NEG on inline float remains refused', refused(encode_vop2_e64(CNDMASK, 31, V30, 242, neg1=1)))
check('NEG on literal remains refused', refused(encode_vop2_e64(CNDMASK, 31, V30, 255, neg1=1)
                                              + struct.pack('<I', 0x12345678)))
# 6. Every edge of the refused immediate ranges, plus the constants just inside them: a mutation moving a
#    range boundary by one has to fail one of these.
for edge in (128, 208, 240, 248):
    check('NEG on the immediate at %d is refused' % edge,
          refused(encode_vop2_e64(CNDMASK, 31, edge, V30, neg0=1)))
asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, 240, V30))
check('plain 0.5 still translates',
      asm == ['v_cndmask_b32_e64 v31, 0.5, v30, s0'] and '0x3f000000u' in emitted, emitted)

# 7. The select takes no output modifier; one that reaches it is refused rather than dropped from the text.
check('a clamp on the select is refused', refused(encode_vop2_e64(CNDMASK, 31, V30, V30, clamp=1)))
check('a mul:2 on the select is refused', refused(encode_vop2_e64(CNDMASK, 31, V30, V30, omod=1)))

# 8. Source absolute values and a negated mask reach `u()`/`u_src()` as text they cannot model, and are
#    refused. llvm-mc rejects all three spellings, so they are decoder-produced words only; the property
#    under test is that production refuses the printed operand rather than translating it.
check('an absolute value on the first source is refused',
      refused(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1)))
check('an absolute value on the second source is refused',
      refused(encode_vop2_e64(CNDMASK, 31, V30, V30, abs1=1)))
check('absolute values on both sources are refused',
      refused(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1, abs1=1)))
check('a negated mask is refused', refused(encode_vop2_e64(CNDMASK, 31, V30, V30, neg_mask=1)))


if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
