"""A scalar ALU op whose destination is a named register half the translator does not model (vcc_hi) refuses with
Unsupported. The same operation into an SGPR still lowers."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402


def run(line):
    stage = Probe()
    stage.masks = set()
    stage.one(line.split(' ', 1)[0], line)
    return stage.lines


for line in ('s_lshl_b32 vcc_hi, vcc_lo, 7', 's_bfe_u32 vcc_hi, s2, 0x50010'):
    try:
        run(line)
    except G.Unsupported as error:
        assert 'vcc_hi' in str(error), error
        print('%s: refused (%s)' % (line, error))
    else:
        raise AssertionError('%s accepted' % line)
lines = run('s_lshl_b32 s4, s2, 7')
assert any(l.startswith('s4 = ') for l in lines), lines
print('control s_lshl_b32 s4, s2, 7: %s' % lines[0])
print('PASS')
