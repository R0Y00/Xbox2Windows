"""A 32-bit literal as S1 of v_lshrrev_b64 / v_lshlrev_b64 is zero-extended (S1.u64, RDNA 2 ISA 6.2.1 and 12.9).

usage: startup_gfx2hlsl_shift64_literal_check.py

Vertex programs materialize a 64-bit pair with `v_lshrrev_b64 v[a:b], 0, 0xffffffff` (and, next to it, all ones
with the inline `-1`); the literal was refused as a 64-bit literal operand. Signed or other 64-bit operand uses
keep the old rule (bit 31 set is refused).
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def lower(line):
    stage = Probe()
    try:
        stage.one(line.split(' ', 1)[0], line)
    except G.Unsupported as reason:
        return 'refused: %s' % reason
    return ' '.join(stage.lines)


for line, lo, hi in (('v_lshrrev_b64 v[5:6], 0, 0xffffffff', '0xffffffffu', '0x00000000u'),
                     ('v_lshlrev_b64 v[5:6], 4, 0x80000001', '0x80000001u', '0x00000000u'),
                     ('v_lshrrev_b64 v[5:6], 0, -1', '0xffffffffu', '0xffffffffu'),
                     ('v_lshrrev_b64 v[5:6], 0, 0x7fffffff', '0x7fffffffu', '0x00000000u')):
    text = lower(line)
    check('%s reads (%s, %s)' % (line, lo, hi), ('(%s, %s, ' % (lo, hi)) in text, text)
try:
    Probe().pair('0xffffffff')
    kept = False
except G.Unsupported:
    kept = True
check('the default 64-bit operand rule still refuses a literal with bit 31 set', kept)

if failures:
    print('FAIL')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print('PASS')
