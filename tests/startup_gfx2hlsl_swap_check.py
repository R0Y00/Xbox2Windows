"""v_swap_b32 in tools/startup/gfx2hlsl.py (found by the offline shader census in #1723).

usage: startup_gfx2hlsl_swap_check.py

The VOP1 `v_swap_b32 vdst, vsrc` exchanges two VGPRs. The translator refused it with
`Unsupported: v_swap_b32`; 2 of the 1,171 pixel and vertex program pairs of the census stop there.

The instruction words are **constructed here** from the public VOP1 field layout (AMD RDNA 2 ISA, VOP1:
[8:0] SRC0, [16:9] OP, [24:17] VDST, [31:25] 0b0111111) with OP = 101 for V_SWAP_B32, and are also checked
against the bytes LLVM 19 assembles for `v_swap_b32 v1, v2` (02 cb 02 7e, gfx1010 and gfx1030).
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


V_SWAP_B32 = 101
VOP1_PREFIX = 0b0111111


def encode_vop1(op, vdst, src0):
    """A VOP1 word; src0 uses the 9-bit source field (VGPRs from 256)."""
    return struct.pack('<I', (VOP1_PREFIX << 25) | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff))


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified, self.sampler_heap = {}, set(), set(), False


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self, exec_known=True):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = exec_known
        self.exports = {}
        self.pipe = Pipeline()


def translate(code, exec_known=True):
    stage = Stage(exec_known)
    for _offset, _size, name, asm in G.decode(code):
        stage.one(name, asm)
    return stage


def refused(code):
    try:
        translate(code)
    except G.Unsupported as error:
        return str(error)
    return None


def main():
    word = encode_vop1(V_SWAP_B32, 1, 256 + 2)
    check('encoding matches the bytes LLVM assembles', word == bytes([0x02, 0xcb, 0x02, 0x7e]), repr(word))

    stage = translate(word)
    check('swap exchanges the two registers through a temporary',
          stage.lines == ['{ uint swap_tmp = v1; v1 = v2; v2 = swap_tmp; }'], repr(stage.lines))
    check('both registers are written', {1, 2} <= stage.used_v)

    stage = translate(encode_vop1(V_SWAP_B32, 255, 256), exec_known=False)
    check('an unknown exec mask guards the whole exchange',
          stage.lines == ['if (exec & 1) { uint swap_tmp = v255; v255 = v0; v0 = swap_tmp; }'], repr(stage.lines))

    stage = translate(encode_vop1(V_SWAP_B32, 3, 256 + 3))
    check('swapping a register with itself emits nothing', stage.lines == [], repr(stage.lines))

    # Chained with an arithmetic use: the swapped values are what later instructions read.
    code = encode_vop1(V_SWAP_B32, 7, 256 + 8) + struct.pack('<I', (0b0111111 << 25) | (9 << 17) | (1 << 9) | (256 + 7))
    stage = translate(code)
    check('a later read sees the swapped register',
          stage.lines == ['{ uint swap_tmp = v7; v7 = v8; v8 = swap_tmp; }', 'v9 = v7;'], repr(stage.lines))

    # A swap invalidates the proven-uniform status of both registers.
    stage = Stage()
    stage.uniform_v.update({4, 5, 6})
    for _o, _s, name, asm in G.decode(encode_vop1(V_SWAP_B32, 4, 256 + 5)):
        stage.one(name, asm)
    check('uniform status is dropped for both swapped registers', stage.uniform_v == {6}, repr(stage.uniform_v))

    # Refusals: a scalar or constant source cannot be encoded by the hardware, so a hand-written line
    # must not be accepted either; modifiers and extra operands are refused, not dropped.
    for label, asm in (('scalar source', 'v_swap_b32 v1, s2'), ('constant source', 'v_swap_b32 v1, 0'),
                       ('scalar destination', 'v_swap_b32 s1, v2'), ('missing source', 'v_swap_b32 v1'),
                       ('third operand', 'v_swap_b32 v1, v2, v3')):
        stage = Stage()
        try:
            stage.one('v_swap_b32', asm)
            ok = False
        except G.Unsupported:
            ok = True
        check('%s is refused' % label, ok and stage.lines == [], asm)
    stage = Stage()
    try:
        stage.one('v_swap_b32', 'v_swap_b32 v1, v2 clamp')
        ok = False
    except G.Unsupported:
        ok = True
    check('a modifier is refused', ok and stage.lines == [])

    # The SDWA spelling (src0 field 249 plus a second dword) is not a swap the translator may run.
    sdwa = struct.pack('<II', (VOP1_PREFIX << 25) | (1 << 17) | (V_SWAP_B32 << 9) | 249, 2 | (6 << 8) | (6 << 16) | (6 << 24))
    reason = refused(sdwa)
    check('the SDWA spelling is refused', reason is not None, repr(reason))

    if failures:
        print('\nFAILED: ' + ', '.join(failures))
        sys.exit(1)
    print('PASS')


main()
