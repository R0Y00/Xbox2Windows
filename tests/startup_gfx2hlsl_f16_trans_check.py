"""Binary16 rounding and transcendental operations in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_f16_trans_check.py

v_ceil/floor/trunc/rndne/fract/rcp/rsq/sqrt/sin/cos/exp/log_f16 read the binary16 value in the low half of their
source, compute it in binary32, round the result to binary16 and write it to the low half with the high half
cleared. Each must translate in both the VOP1 and the VOP3 spelling, take inline constants and the neg/abs source
modifiers, and refuse a modifier or a wrong operand count instead of dropping it.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
import gfx2hlsl as G  # noqa: E402

OPS = {
    'v_ceil_f16': 'ceil(%s)', 'v_floor_f16': 'floor(%s)', 'v_trunc_f16': 'trunc(%s)', 'v_rndne_f16': 'round(%s)',
    'v_fract_f16': 'min(frac(%s), 0.99951171875)', 'v_rcp_f16': '1.0 / %s', 'v_rsq_f16': 'rsqrt(%s)',
    'v_sqrt_f16': 'sqrt(%s)', 'v_sin_f16': 'sin(%s * 6.283185307)', 'v_cos_f16': 'cos(%s * 6.283185307)',
    'v_exp_f16': 'exp2(%s)', 'v_log_f16': 'log2(%s)',
}
failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH  ' + detail))
    if not condition:
        failures.append(label)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def translate(line):
    """The emitted lines for one instruction, or 'refused' / the name of another exception."""
    stage = Probe()
    try:
        stage.one(line.split(' ', 1)[0], line)
    except G.Unsupported:
        return 'refused'
    except Exception as error:  # noqa: BLE001  the point is to see which exception a malformed form raises
        return type(error).__name__
    return stage.lines


for op, template in OPS.items():
    expected = ['v1 = f32tof16(%s);' % (template % 'f16tof32(v2 & 0xffffu)')]
    for spelling in (op, op + '_e64'):
        got = translate('%s v1, v2' % spelling)
        check('%s translates to a binary16 round of the low half' % spelling, got == expected, repr(got))
    check('%s with a constant source' % op,
          translate('%s v1, 0.5' % op) == ['v1 = f32tof16(%s);' % (template % '0.5')], repr(translate('%s v1, 0.5' % op)))
    check('%s keeps neg and abs on its source' % op,
          translate('%s v1, -|v2|' % op) ==
          ['v1 = f32tof16(%s);' % (template % '(-abs(f16tof32(v2 & 0xffffu)))')], repr(translate('%s v1, -|v2|' % op)))
    check('%s refuses a clamp' % op, translate('%s_e64 v1, v2 clamp' % op) == 'refused')
    check('%s refuses a missing source' % op, translate('%s v1' % op) == 'refused')
    check('%s refuses an extra source' % op, translate('%s v1, v2, v3' % op) == 'refused')

print('f16 transcendental checks: %d failed' % len(failures))
sys.exit(1 if failures else 0)
