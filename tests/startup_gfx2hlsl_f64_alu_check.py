"""The 64-bit float ALU slice in tools/startup/gfx2hlsl.py: v_add_f64, v_mul_f64, v_fma_f64,
v_cvt_f32_f64 and v_cvt_f64_f32 (Part of #1723, frontier batch 3 item 10).

A binary64 value is a VGPR/SGPR pair. The lowering reads both words into one `double` with asdouble(),
computes, and writes both result words from one `asuint(double, lo, hi)` call, so an in-place destination
(`v_add_f64 v[2:3], v[2:3], v[4:5]`) keeps the original operands until the result is complete.
Output modifiers, op_sel, literals and a destination that is not a VGPR pair are refused, not dropped.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


class Stage(G.Stage):
    """A Stage with no pipeline: enough state to translate one instruction."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def translate(name, asm):
    stage = Stage()
    stage.one(name, asm)
    return ' '.join(stage.lines)


def refused(name, asm, needle):
    try:
        translate(name, asm)
    except G.Unsupported as error:
        check('refuses ' + asm, needle in str(error), str(error))
        return
    check('refuses ' + asm, False, 'accepted')


text = translate('v_add_f64', 'v_add_f64 v[2:3], v[4:5], v[6:7]')
check('add reads both pairs and writes both words',
      'asdouble(v4, v5) + asdouble(v6, v7)' in text and 'v2 = lo;' in text and 'v3 = hi;' in text, text)

text = translate('v_add_f64', 'v_add_f64 v[2:3], v[2:3], v[4:5]')
check('in-place add computes before writing', text.index('asuint(') < text.index('v2 = lo;'), text)

text = translate('v_mul_f64', 'v_mul_f64 v[2:3], -|v[4:5]|, 2.0')
check('mul with neg/abs and an inline constant', '(-abs(asdouble(v4, v5))) * 2.0' in text, text)

text = translate('v_fma_f64', 'v_fma_f64 v[2:3], v[4:5], v[6:7], s[8:9]')
check('fma takes three pairs', 'fma(asdouble(v4, v5), asdouble(v6, v7), asdouble(s8, s9))' in text, text)

text = translate('v_cvt_f32_f64_e32', 'v_cvt_f32_f64 v1, v[4:5]')
check('narrowing conversion writes one word', 'v1 = asuint((float)asdouble(v4, v5))' in text, text)

text = translate('v_cvt_f64_f32_e32', 'v_cvt_f64_f32 v[2:3], v1')
check('widening conversion writes a pair', 'double(' in text and 'v2 = lo;' in text and 'v3 = hi;' in text, text)

refused('v_add_f64', 'v_add_f64 v[2:3], v[4:5], v[6:7] clamp', 'v_add_f64 with')
refused('v_add_f64', 'v_add_f64 v[2:3], v[4:5], v[6:7] mul:2', 'v_add_f64 with')
refused('v_add_f64', 'v_add_f64 v2, v[4:5], v[6:7]', 'destination')
refused('v_mul_f64', 'v_mul_f64 v[2:3], v[4:5]', 'expected 3')
refused('v_fma_f64', 'v_fma_f64 v[2:3], v[4:5], v[6:7], 0.15915494', '64-bit float source')
refused('v_cvt_f32_f64', 'v_cvt_f32_f64 v[1:2], v[4:5]', 'destination')
refused('v_cvt_f64_f32', 'v_cvt_f64_f32 v[2:3], v1 clamp', 'with')

if failures:
    print('%d failures' % len(failures))
    sys.exit(1)
print('gfx2hlsl f64 alu: all checks passed')
