"""`v_mul_hi_u32` in tools/startup/gfx2hlsl.py (issue #1792).

usage: startup_gfx2hlsl_mul_hi_check.py

The high half of an unsigned 32x32 product had no lowering, so `v_mul_hi_u32` was refused - one of the
remaining refused shaders in the main-game loading-screen cache (#870). `tools/pixel_aot.py` already had the
halfword-product helper; this port lowers the VOP3A form to it.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch): prefix `0b110101`, opcode `0x16a` in `w0[25:16]`, vdst in `w0[7:0]`,
src0 in `w1[8:0]`, src1 in `w1[17:9]`, and source 255 followed by a literal dword.
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


VOP3A_MUL_HI_U32 = 0x16A
V0, V1, V2 = 0x100, 0x101, 0x102
LITERAL_SOURCE = 255


def encode_vop3a(op, vdst, srcs, literal=None):
    """Encode a native VOP3A instruction; a source of 255 is followed by its literal dword."""
    w0 = (0b110101 << 26) | ((op & 0x3FF) << 16) | (vdst & 0xFF)
    w1 = 0
    for i, src in enumerate(srcs):
        w1 |= (src & 0x1FF) << (9 * i)
    code = struct.pack('<II', w0, w1)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


class Pipeline:
    """Just enough pipeline state for one scalarized multiply."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


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


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, [V1, V2]))
check('encodes and decodes the register form', asm == ['v_mul_hi_u32 v1, v1, v2'], repr(asm))

# 2. The lowering calls the halfword-product helper rather than a plain multiply (the low half of the product
#    is not the high half, so a `*` here would be silently wrong).
check('the register form uses the helper', emitted == 'v1 = mul_hi_u32(v1, v2);', emitted)

# 3. The census form carries a literal operand, which reaches the helper with the same suffix the other
#    uint operations use.
asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, [V0, LITERAL_SOURCE],
                                      literal=0xAAAAAAAB))
check('encodes and decodes the literal form', asm == ['v_mul_hi_u32 v1, v0, 0xaaaaaaab'], repr(asm))
check('the literal form uses the helper', emitted == 'v1 = mul_hi_u32(v0, 0xaaaaaaabu);', emitted)

# 4. The helper is in the preamble the generated shader is built with; without it the lowering would not
#    compile even though its text looks right.
pipe = object.__new__(G.Pipeline)
pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
common = ' '.join(pipe.common())
check('the helper is emitted in the common preamble', 'uint mul_hi_u32(uint a, uint b)' in common, common[:200])
check('the helper uses halfword products, not a 64-bit multiply',
      'a & 0xffffu' in common and 'b >> 16' in common, common[:200])

# 5. A neighbouring operation is untouched: the low-half multiply still lowers to a plain product.
VOP3A_MUL_LO_U32 = 0x169
asm, emitted = translate(encode_vop3a(VOP3A_MUL_LO_U32, 1, [V1, V2]))
check('the low multiply is unchanged', emitted == 'v1 = v1 * v2;', emitted)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')