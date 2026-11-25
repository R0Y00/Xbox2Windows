"""Translation controls for v_fma_mix_f32 and op_sel on v_fma_mix{lo,hi}_f16 in gfx2hlsl.

usage: startup_gfx2hlsl_fma_mix_check.py

Synthetic VOP3P words only; spellings are asserted from the translator's decoder. Covers f32/f16
operand widening, op_sel high-half extraction, abs/neg after widening, clamp, untouched-half
preservation for f16 destinations, odd-d16 low16 tracking, refusals, and a negative control.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = 0


def check(what, ok, detail=''):
    global failures
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', what, '' if ok or not detail else ': %s' % (detail,)))
    if not ok:
        failures += 1


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


def vop3p_fma_mix(op, vdst, src0, src1, src2, op_sel=(0, 0, 0), op_sel_hi=(0, 0, 0),
                  neg=(0, 0, 0), abs_=(0, 0, 0), clamp=False):
    w0 = (
        0xcc000000
        | (op << 16)
        | (clamp << 15)
        | (op_sel_hi[2] << 14)
        | (op_sel[2] << 13)
        | (op_sel[1] << 12)
        | (op_sel[0] << 11)
        | (abs_[2] << 10)
        | (abs_[1] << 9)
        | (abs_[0] << 8)
        | vdst
    )
    w1 = (
        (neg[2] << 31)
        | (neg[1] << 30)
        | (neg[0] << 29)
        | (op_sel_hi[1] << 28)
        | (op_sel_hi[0] << 27)
        | (src2 << 18)
        | (src1 << 9)
        | src0
    )
    return [w0, w1]


def stage(low16=()):
    s = G.Stage(Pipe(), 'ps', struct.pack('<I', 0xbf810000))
    s.lines = []
    s.pc = 0
    s.exec_known = True
    s.low16_only_v = set(low16)
    s.whole_read_v = set()
    return s


def translate(asm_or_words, low16=()):
    s = stage(low16=low16)
    if isinstance(asm_or_words, str):
        asm = asm_or_words
        name = asm.split()[0]
    else:
        _, _, name, asm = G.decode(struct.pack('<%dI' % len(asm_or_words), *asm_or_words))[0]
    s.lines = []
    s.one(name, asm)
    return s, asm, s.lines


def main():
    # Mixed f16/f32 with op_sel high-half selection, |.| and - widened from f16, f32 clamp, and inline f16 const.
    s, text, lines = translate(vop3p_fma_mix(0x20, 12, 44, 256 + 19, 256 + 20,
                                             op_sel=(1, 0, 0), op_sel_hi=(1, 1, 0),
                                             neg=(0, 1, 0), abs_=(1, 0, 0), clamp=True))
    check('decodes v_fma_mix_f32', text == 'v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,0,0] op_sel_hi:[1,1,0] clamp', text)
    check('v_fma_mix_f32 widens f16 before abs/neg and extracts high half',
          lines == ['v12 = U(saturate(abs(f16tof32(s44 >> 16)) * (-f16tof32(v19)) + F(v20)));'], lines)

    # Inline f16 constant with op_sel_hi=1 and high-half selection on mixlo/mixhi_f16.
    _, text, lines = translate(vop3p_fma_mix(0x21, 22, 242, 240, 256 + 21,
                                             op_sel=(0, 1, 0), op_sel_hi=(1, 1, 0)))  # 242=1.0, 240=0.5
    check('decodes v_fma_mixlo_f16 with op_sel', text == 'v_fma_mixlo_f16 v22, 1.0, 0.5, v21 op_sel:[0,1,0] op_sel_hi:[1,1,0]', text)
    check('v_fma_mixlo_f16 preserves upper half and shifts inline f16 op_sel=1',
          lines == ['v22 = (v22 & 0xffff0000u) | f32tof16(f16tof32(0x3c00u) * f16tof32(0x3800u >> 16) + F(v21));'], lines)

    _, text, lines = translate(vop3p_fma_mix(0x22, 22, 256 + 19, 256 + 23, 256 + 20,
                                             op_sel=(1, 0, 0), op_sel_hi=(1, 0, 0),
                                             neg=(1, 0, 0), abs_=(1, 0, 0), clamp=True))
    check('decodes v_fma_mixhi_f16 with op_sel and clamp',
          text == 'v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp', text)
    check('v_fma_mixhi_f16 preserves lower half and saturates',
          lines == ['v22 = (v22 & 0xffffu) | (f32tof16(saturate((-abs(f16tof32(v19 >> 16))) * F(v23) + F(v20))) << 16);'], lines)

    # Zero literal and inline negative zero remain valid even when op_sel_hi bit is set.
    _, _, lines = translate('v_fma_mix_f32 v2, 1.0, s38, -0 op_sel:[0,1,0] op_sel_hi:[0,1,0]')
    check('-0 under f32 op_sel_hi=0 lowers cleanly', lines == ['v2 = U(1.0 * f16tof32(s38 >> 16) + 0.0);'], lines)
    s_low, _, lines = translate('v_fma_mix_f32 v4, v4, 1.0, 0 op_sel_hi:[1,0,0]', low16=(4,))
    check('v_fma_mix_f32 reads low16 safely when op_sel=0 and clears low16_only on f32 write',
          4 not in s_low.whole_read_v and 4 not in s_low.low16_only_v and
          lines == ['v4 = U(f16tof32(v4) * 1.0 + F(0x00000000u));'], (s_low.whole_read_v, s_low.low16_only_v, lines))

    # Reading high half (op_sel=1) on an odd-d16 low16-only VGPR must fail whole-word read check.
    for label, fn in (
        ('high-half read of low16-only v4 refused',
         lambda: translate('v_fma_mix_f32 v5, v4, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]', low16=(4,))),
        ('op_sel=1 without op_sel_hi refused',
         lambda: translate('v_fma_mix_f32 v5, v4, 1.0, 0 op_sel:[1,0,0]')),
        ('output modifier mul:2 refused',
         lambda: translate('v_fma_mix_f32 v5, v4, 1.0, 0 mul:2')),
        ('SGPR destination refused',
         lambda: translate('v_fma_mix_f32 s5, v4, 1.0, 0')),
        ('wrong operand count refused',
         lambda: translate('v_fma_mix_f32 v5, v4, 1.0')),
    ):
        try:
            fn()
        except G.Unsupported:
            check(label, True)
        else:
            check(label, False, 'accepted')

    # Negative control: removing op_v_fma_mix_f32 from Stage must refuse translation.
    saved = G.Stage.op_v_fma_mix_f32
    delattr(G.Stage, 'op_v_fma_mix_f32')
    try:
        translate('v_fma_mix_f32 v12, v1, v2, v3')
    except G.Unsupported:
        check('negative control: removed handler refuses v_fma_mix_f32', True)
    else:
        check('negative control: removed handler refuses v_fma_mix_f32', False, 'accepted')
    finally:
        G.Stage.op_v_fma_mix_f32 = saved

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
