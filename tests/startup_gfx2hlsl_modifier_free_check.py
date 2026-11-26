"""Handlers whose lowering does not model a modifier they can receive (issue #1771).

usage: startup_gfx2hlsl_modifier_free_check.py

Several handlers take the modifier list and never read it, so a modifier their encoding can carry was dropped
from the emitted HLSL - the same class as the `op_sel`/`clamp` cases fixed one at a time earlier
(`set_vf`, `op_v_cndmask_b32`, `compare`). The dispatch now refuses a modifier for the handlers declared in
`MODIFIER_FREE_HANDLERS`, before the handler can ignore it.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`): VOP1/VOP2-as-VOP3 use prefix `0b110101` with opcode `0x180 + n` / `0x100 + n`,
modifiers at `w0[15]` (clamp), `w0[14:11]` (op_sel) and `w1[28:27]` (omod); VOP3A uses the opcode in `w0[25:16]`.
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


VOP1_E64_BASE = 0x180
VOP2_E64_BASE = 0x100
VOP1_MOV_B32 = 0x01
VOP2_CVT_PKRTZ_F16_F32 = 0x2F
VOP3A_INTERP_P1 = 0x200
VOP3A_INTERP_P2 = 0x201
VOP3A_INTERP_MOV = 0x202
V1, V2, V3 = 0x101, 0x102, 0x103


def _word0(opcode, vdst, *, clamp=0, op_sel=(0, 0, 0, 0)):
    w0 = (0b110101 << 26) | ((opcode & 0x3FF) << 16) | ((clamp & 1) << 15) | (vdst & 0xFF)
    for i, bit in enumerate(op_sel):
        w0 |= (bit & 1) << (11 + i)
    return w0


def encode_vop1(op, vdst, src0, *, clamp=0):
    return struct.pack('<II', _word0(VOP1_E64_BASE + op, vdst, clamp=clamp), src0 & 0x1FF)


def encode_vop2(op, vdst, src0, src1, *, clamp=0, op_sel=(0, 0, 0, 0), omod=0):
    w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((omod & 0x3) << 27)
    return struct.pack('<II', _word0(VOP2_E64_BASE + op, vdst, clamp=clamp, op_sel=op_sel), w1)


def encode_interp(op, vdst, source, *, attr=0, chan=0, abs_source=0, neg_source=0, clamp=0):
    """Encode the gfx10 VOP3 spelling of v_interp_p1_f32 / v_interp_p2_f32."""
    w0 = ((0b110101 << 26) | ((op & 0x3FF) << 16) | ((clamp & 1) << 15)
          | ((abs_source & 1) << 9) | (vdst & 0xFF))
    w1 = ((attr & 0x3F) | ((chan & 0x3) << 6) | ((source & 0x1FF) << 9) | ((neg_source & 1) << 30))
    return struct.pack('<II', w0, w1)


def encode_vop3a(op, vdst, srcs, *, clamp=0):
    w1 = 0
    for i, src in enumerate(srcs):
        w1 |= (src & 0x1FF) << (9 * i)
    return struct.pack('<II', _word0(op, vdst, clamp=clamp), w1)


class Pipeline:
    """Just enough pipeline state for one instruction."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()

    def ud(self, key):
        return 0


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.attrs = {}
        self.pipe = Pipeline()
        self.kind = 'ps'
        self.blocks = []


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


# 1. The declared set is what we expect; a handler silently added to or dropped from it shows up here.
check('the declared modifier-free handlers are exactly these five',
      set(G.MODIFIER_FREE_HANDLERS) == {
          'op_v_mov_b32', 'op_v_cvt_pkrtz_f16_f32', 'op_v_interp_p1_f32',
          'op_v_interp_p2_f32', 'op_v_interp_mov_f32'}, ', '.join(sorted(G.MODIFIER_FREE_HANDLERS)))

# 2. v_mov_b32: a plain move translates, and a modifier it never modelled is refused.
asm, emitted = translate(encode_vop1(VOP1_MOV_B32, 1, V2))
check('a plain move still translates', emitted == 'v1 = v2;', emitted)
check('a clamp on the move is refused', refused(encode_vop1(VOP1_MOV_B32, 1, V2, clamp=1)))
check('a mul:2 on the move is refused',
      refused(struct.pack('<II', _word0(VOP1_E64_BASE + VOP1_MOV_B32, 1),
                          (V2 & 0x1FF) | (1 << 27))))

# 3. v_cvt_pkrtz_f16_f32: a plain pack translates, and op_sel/clamp are refused.
asm, emitted = translate(encode_vop2(VOP2_CVT_PKRTZ_F16_F32, 1, V2, V3))
check('a plain packed conversion still translates',
      emitted == 'v1 = f32tof16(F(v2)) | (f32tof16(F(v3)) << 16);', emitted)
check('an op_sel on the packed conversion is refused',
      refused(encode_vop2(VOP2_CVT_PKRTZ_F16_F32, 1, V2, V3, op_sel=(0, 0, 1, 0))))
check('a clamp on the packed conversion is refused',
      refused(encode_vop2(VOP2_CVT_PKRTZ_F16_F32, 1, V2, V3, clamp=1)))

# 4. v_interp_mov_f32: the parameter selector is the VINTRP field, which decodes as p10, p20, p0 in that
#    order. Only p0 is the flat copy; p10 and p20 need the per-vertex attribute values, so they are
#    refused by name rather than read as p0. A plain p0 read translates, and a modifier on it is refused.
_p10 = encode_vop3a(VOP3A_INTERP_MOV, 1, [0, 0])
asm, emitted = translate(encode_vop3a(VOP3A_INTERP_MOV, 1, [0, 2]))
check('encodes and decodes the flat p0 read', asm == ['v_interp_mov_f32_e64 v1, p0, attr0.x'], repr(asm))
check('a plain flat interpolation still translates', emitted == 'v1 = i.fparam0.x;', emitted)
check('a clamp on the flat interpolation is refused', refused(encode_vop3a(VOP3A_INTERP_MOV, 1, [0, 2], clamp=1)))
check('a p10 selector is refused rather than read as p0', refused(_p10))
check('a p20 selector is refused rather than read as p0', refused(encode_vop3a(VOP3A_INTERP_MOV, 1, [0, 1])))

# 5. The p1/p2 interpolations have their own operand profile. A valid form carries a data-source VGPR in
#    w1[16:9], and the decoder prints an abs/neg on that source as `|v2|` / `-v2` - inside the operand
#    text, not as a mod - so the dispatch guard does not see it. The lowering never reads that operand
#    (the value comes from the interpolator), so the handler refuses a modified source itself.
_asm, _emitted = translate(encode_interp(VOP3A_INTERP_P1, 1, V2))
check('a plain p1 interpolation translates', _emitted == '', _emitted)
_asm, emitted = translate(encode_interp(VOP3A_INTERP_P2, 1, V2))
check('a plain p2 interpolation translates', emitted == 'v1 = U(i.param0.x);', emitted)
_asm, emitted = translate(encode_interp(VOP3A_INTERP_P2, 1, V2, attr=1, chan=1))
check('the attribute channel is read from the operand', emitted == 'v1 = U(i.param1.y);', emitted)
check('an absolutely-modified p1 source is refused',
      refused(encode_interp(VOP3A_INTERP_P1, 1, V2, abs_source=1)))
check('a negated p1 source is refused',
      refused(encode_interp(VOP3A_INTERP_P1, 1, V2, neg_source=1)))
check('an absolutely-modified p2 source is refused',
      refused(encode_interp(VOP3A_INTERP_P2, 1, V2, abs_source=1)))
check('a clamp on the p1 interpolation is refused',
      refused(encode_interp(VOP3A_INTERP_P1, 1, V2, clamp=1)))

# 6. No scalar-ALU handler is in the set: their encoding has no modifier field, so there is nothing for the
#    dispatch to refuse and they must keep translating whatever the decoder prints.
check('no scalar handler is declared modifier-free',
      not [name for name in G.MODIFIER_FREE_HANDLERS if name.startswith('op_s_')],
      ', '.join(sorted(G.MODIFIER_FREE_HANDLERS)))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')