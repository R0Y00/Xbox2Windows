"""`v_fract_f32` lowers through the clamped helper `fract32` in tools/startup/gfx2hlsl.py (#1741).

usage: startup_gfx2hlsl_fract_check.py

Two NVIDIA reports fail `startup_gfx2hlsl_float_round_hardware` on the `v_fract_f32` input `0xb0800000`
(-2^-30): expected `0x3f7fffff`, actual `0x3f800000` (RTX 3070 Ti, driver 32.0.16.2101; Quadro M1000M
Maxwell, driver 32.0.15.7371). The same draw passes on WARP, on Intel HD Graphics 530 and on the RDNA 4
card used here. The table used to emit a bare D3D `frac`, which has no upper cap and differs between
adapters, while `tools/pixel_aot.py` has always capped `x - floor(x)` at `0x3f7fffff` through `fract32`.

This check reads the emitted text only. It cannot reproduce the NVIDIA result: that needs one of those
adapters, and no such GPU is available here. What it pins is the lowering the adapters disagreed on.
"""
import os
import re
import sys
from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def lower(line):
    """The translated statement of one assembly line, as the caller sees it."""
    stage = Probe()
    stage.one(line.split(' ', 1)[0], line)
    return ' '.join(stage.lines)


HELPER = re.compile(r'float fract32\(float value\) \{.*?\}')
BARE_FRAC = re.compile(r'(?<![\w.])frac\(')


def common():
    pipe = object.__new__(G.Pipeline)
    pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
    return ' '.join(pipe.common())


def problems(statement, helper):
    """Everything wrong with one lowered statement and the helper it calls; empty when both are right."""
    found = []
    if 'fract32(' not in statement:
        found.append('the instruction does not call fract32')
    if BARE_FRAC.search(statement):
        found.append('the instruction calls a bare frac(')
    if '0x3f7fffff' not in helper:
        found.append('the helper does not clamp at 0x3f7fffff')
    if 'f >= 1.0f ? asfloat(0x3f7fffffu) : f' not in helper:
        found.append('the clamp is not the comparison form pixel_aot writes')
    if 'min(' in helper:
        found.append('the clamp goes through min, which loses a NaN')
    return found


preamble = common()
match = HELPER.search(preamble)
check('the fract32 helper is emitted in the common preamble', match is not None, preamble[:240])
helper = match.group(0) if match else ''
check('the helper is defined once', preamble.count('float fract32(') == 1, preamble[:240])
source = Path(G.__file__).read_text(encoding='utf-8')
check('the helper carries the pixel_aot comment',
      '# Match pixel_aot: x - floor(x) capped at 0x3f7fffff' in source)

# 1. The table entry itself: one source, through the helper.
check('the VEC_F entry goes through fract32', G.VEC_F.get('v_fract_f32') == (1, 'fract32({0})'),
      repr(G.VEC_F.get('v_fract_f32')))
users = sorted(op for op, (_arity, text) in G.VEC_F.items() if 'fract32(' in text)
check('fract32 is used by v_fract_f32 alone and no other entry', users == ['v_fract_f32'], repr(users))

# 2. The emitted statement of each spelling, with the source modifiers the form accepts.
for line, expected in (
        ('v_fract_f32 v12, v0', 'v12 = U(fract32(F(v0)));'),
        ('v_fract_f32 v0, v0', 'v0 = U(fract32(F(v0)));'),
        ('v_fract_f32_e64 v13, |v1|', 'v13 = U(fract32(abs(F(v1))));'),
        ('v_fract_f32_e64 v14, -|v2|', 'v14 = U(fract32((-abs(F(v2)))));'),
):
    statement = lower(line)
    found = problems(statement, helper)
    check(f'{line} lowers through the clamp', not found, '%r: %s' % (statement, '; '.join(found)))
    check(f'{line} emits the expected statement', statement == expected, statement)

# 3. Negative controls: the checks above must reject the two lowerings that caused the report, an uncapped
#    frac and a cap written with min. The same problem list is used, so the controls cannot drift from it.
uncapped = problems(lower('v_fract_f32 v12, v0').replace('fract32(', 'frac('), helper)
check('control: a bare frac is rejected', any('bare frac' in text for text in uncapped), repr(uncapped))
capped_with_min = problems(lower('v_fract_f32 v12, v0'),
                           helper.replace('f >= 1.0f ? asfloat(0x3f7fffffu) : f',
                                          'min(frac(value), asfloat(0x3f7fffffu))'))
check('control: a min-form clamp is rejected', bool(capped_with_min), repr(capped_with_min))
raised_cap = problems(lower('v_fract_f32 v12, v0'), helper.replace('0x3f7fffffu', '0x3f800000u'))
check('control: a cap other than 0x3f7fffff is rejected', bool(raised_cap), repr(raised_cap))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
