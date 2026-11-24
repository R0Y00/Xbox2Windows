"""SDWA forms of the float-result conversions in tools/startup/gfx2hlsl.py (issue #1727).

usage: startup_gfx2hlsl_sdwa_cvt_check.py

Moving the float-result conversions into their own table (the output-modifier change) left `sdwa()` looking only
at the uint-result table, so every `v_cvt_f32_*_sdwa` stopped translating - a regression in the merged tree that
review found after the merge. This pins the restored behaviour.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP1 branch): a VOP1 word with SRC0 = 249 marks SDWA, and the next dword holds the
control fields:
  w0: [31:25] = 0b0111111, [24:17] vdst, [16:9] opcode, [8:0] = 249 (SDWA)
  w1: [7:0] src0, [10:8] dst_sel, [12:11] dst_unused, [13] clamp, [15:14] omod, [18:16] src0_sel
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


VOP1_CVT_F32_I32 = 0x05
VOP1_CVT_F32_U32 = 0x06
VOP1_CVT_F16_F32 = 0x0A
VOP1_CVT_F32_F16 = 0x0B
SEL = {'BYTE_0': 0, 'BYTE_1': 1, 'BYTE_2': 2, 'BYTE_3': 3, 'WORD_0': 4, 'WORD_1': 5, 'DWORD': 6}
UNUSED = {'UNUSED_PAD': 0, 'UNUSED_SEXT': 1, 'UNUSED_PRESERVE': 2}
OMOD = {'': 0, 'mul:2': 1, 'mul:4': 2, 'div:2': 3}
SDWA_SRC0 = 249


def encode_vop1_sdwa(op, vdst, src0, *, dst_sel='DWORD', dst_unused='UNUSED_PAD', src0_sel='WORD_0', omod=''):
    """Encode a VOP1 instruction in its SDWA form."""
    w0 = ((0b0111111 << 25) | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | SDWA_SRC0)
    w1 = ((src0 & 0xFF) | (SEL[dst_sel] << 8) | (UNUSED[dst_unused] << 11) | (OMOD[omod] << 14)
          | (SEL[src0_sel] << 16))
    return struct.pack('<II', w0, w1)


class Pipeline:
    """Just enough pipeline state for one conversion."""

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


def refused(code):
    try:
        translate(code)
    except G.Unsupported:
        return True
    return False


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_0'))
check('encodes and decodes a conversion SDWA',
      asm == ['v_cvt_f32_u32_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0'], repr(asm))

# 2. Each float-result conversion translates again, reading the selected bits and converting them.
check('u32 conversion SDWA translates',
      emitted == 'v0 = U(float(((v1 >> 0u) & 0xffffu)));', emitted)
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_I32, 0, 1, src0_sel='BYTE_0'))
check('i32 conversion SDWA translates',
      emitted == 'v0 = U(float(int(((v1 >> 0u) & 0xffu))));', emitted)
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_F16, 0, 1, src0_sel='WORD_0'))
check('f16 conversion SDWA translates',
      emitted == 'v0 = U(f16tof32(((v1 >> 0u) & 0xffffu)));', emitted)
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='DWORD'))
check('DWORD selection reads the whole register', emitted == 'v0 = U(float(v1));', emitted)
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='BYTE_1'))
check('BYTE_1 selection shifts to the byte', '(v1 >> 8u) & 0xffu' in emitted, emitted)

# 3. A destination selection still writes the converted value's bit pattern.
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, dst_sel='BYTE_1'))
check('destination selection masks the result', '& 0xffu) << 8u' in emitted, emitted)
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, dst_sel='BYTE_0',
                                          dst_unused='UNUSED_PRESERVE'))
check('UNUSED_PRESERVE keeps the other destination bits', 'asuint(v0)' in emitted, emitted)

# 4. A uint-result conversion takes the uint path and is not wrapped as a float.
asm, emitted = translate(encode_vop1_sdwa(VOP1_CVT_F16_F32, 0, 1, src0_sel='WORD_0'))
check('uint-result conversion stays on the uint path',
      emitted == 'v0 = f32tof16(F(((v1 >> 0u) & 0xffffu)));', emitted)

# 5. Boundaries kept: an output modifier on an SDWA conversion is still refused rather than dropped, and an
#    operation outside both tables is still refused as an SDWA form.
check('an omod on an SDWA conversion stays refused',
      refused(encode_vop1_sdwa(VOP1_CVT_F32_U32, 0, 1, src0_sel='WORD_0', omod='mul:2')))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')