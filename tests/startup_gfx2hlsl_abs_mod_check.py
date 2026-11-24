"""Absolute-value source modifier `|vN|` on the vector select in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_abs_mod_check.py

A VOP3 `v_cndmask_b32` can carry the abs source modifier, which clears the source's sign bit before any
negation. The select handler passed the printed `|v30|` to the uint helper and was refused with
`Unsupported: operand '|v30|'`.

Instruction bytes are constructed from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch): w0 [9:8] abs for src1 and src0; w1 [31:29] neg.
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


def encode_vop2_e64(op, vdst, src0, src1, *, neg0=0, neg1=0, abs0=0, abs1=0):
    """Encode a VOP2 instruction in its VOP3A (e64) form."""
    w0 = (0b110101 << 26) | ((VOP2_E64_BASE + op) << 16) | ((abs1 & 1) << 9) | ((abs0 & 1) << 8) | (vdst & 0xff)
    w1 = (src0 & 0x1ff) | ((src1 & 0x1ff) << 9) | ((neg0 & 1) << 29) | ((neg1 & 1) << 30)
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


asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1))
check('abs on the first source decodes as printed', asm == ['v_cndmask_b32_e64 v31, |v30|, v30, s0'], repr(asm))
check('abs on the first source clears the sign bit',
      emitted == 'v31 = (s0 & 1) ? v30 : (v30 & 0x7fffffffu);', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs1=1))
check('abs on the second source clears the sign bit',
      emitted == 'v31 = (s0 & 1) ? (v30 & 0x7fffffffu) : v30;', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1, abs1=1))
check('abs on both sources', emitted.count('0x7fffffffu') == 2, emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30, abs0=1, neg0=1))
check('neg after abs gives a negative value',
      emitted == 'v31 = (s0 & 1) ? v30 : ((v30 & 0x7fffffffu) ^ 0x80000000u);', emitted)

asm, emitted = translate(encode_vop2_e64(CNDMASK, 31, V30, V30))
check('no modifier is unchanged', emitted == 'v31 = (s0 & 1) ? v30 : v30;', emitted)

check('abs on an inline immediate stays refused', refused(encode_vop2_e64(CNDMASK, 31, 193, V30, abs0=1)))
check('abs on another integer op stays refused', refused(encode_vop2_e64(AND_B32, 1, V30, V30, abs1=1)))

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('PASS')
