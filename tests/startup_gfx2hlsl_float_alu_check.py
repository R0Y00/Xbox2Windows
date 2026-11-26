"""Binary32 arithmetic of tools/startup/gfx2hlsl.py (its VEC_F table).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_add_f32, v_sub_f32, v_subrev_f32, v_mul_f32, v_mul_legacy_f32, v_min_f32, v_max_f32, v_fma_f32, v_mad_f32,
v_mad_legacy_f32, v_med3_f32, v_min3_f32 and v_max3_f32, which no check ran before. The fixture and runner are
startup_gfx2hlsl_lane_alu.py: one operation per lane on flat parameters the compilers cannot fold, alternating the
e32 and e64 spellings of the VOP2 forms. The references follow the AMD RDNA 2 ISA reference: v_subrev is S1 - S0;
the legacy multiply reads 0 when either factor is 0, even against an infinity or a NaN; v_med3_f32 is the middle of
three values.

Every input set has an exact result, so the operations are checked independently of rounding and of whether a
compiler fuses a multiply and an add. Not covered: inexact products (v_fma_f32 rounds once and v_mad_f32 twice, while
the translation writes both as a * b + c and leaves the choice to the compiler), NaN and signed-zero ordering in min,
max and med3, and denormals.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import main, words  # noqa: E402


def f(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def v(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def op2(function):
    return lambda a, b: f(function(v(a), v(b)))


def op3(function):
    return lambda a, b, c: f(function(v(a), v(b), v(c)))


def legacy(a, b):
    return 0.0 if a == 0.0 or b == 0.0 else a * b


INF, NAN = 0x7f800000, 0x7fc00000
OPS = {
    'v_add_f32': (2, op2(lambda a, b: a + b), True),
    'v_sub_f32': (2, op2(lambda a, b: a - b), True),
    'v_subrev_f32': (2, op2(lambda a, b: b - a), True),
    'v_mul_f32': (2, op2(lambda a, b: a * b), True),
    'v_mul_legacy_f32': (2, op2(legacy), True),
    'v_min_f32': (2, op2(min), True),
    'v_max_f32': (2, op2(max), True),
    'v_fma_f32': (3, op3(lambda a, b, c: a * b + c), False),
    'v_mad_f32': (3, op3(lambda a, b, c: a * b + c), False),
    'v_mad_legacy_f32': (3, op3(lambda a, b, c: legacy(a, b) + c), False),
    'v_med3_f32': (3, op3(lambda a, b, c: sorted((a, b, c))[1]), False),
    'v_min3_f32': (3, op3(lambda a, b, c: min(a, b, c)), False),
    'v_max3_f32': (3, op3(lambda a, b, c: max(a, b, c)), False),
}


def sets(*rows):
    return [tuple(x if isinstance(x, int) else f(x) for x in row) for row in rows]


# Exact inputs and results: dyadic values whose sums, differences and products fit in 24 bits.
TWO = sets((1.5, 2.25, 0), (-3.0, 0.5, 0), (1024.0, -0.125, 0), (7.0, -7.5, 0),
           (0.375, 8.0, 0), (-2.5, -4.0, 0), (65536.0, 3.0, 0), (0.0, 5.0, 0))
LEGACY = sets((0.0, INF, 0), (INF, 0.0, 0), (2.0, 3.0, 0), (0.0, NAN, 0),
              (-1.5, 4.0, 0), (NAN, 0.0, 0), (0.25, -0.5, 0), (3.0, 3.0, 0))
THREE = sets((1.5, 2.0, 0.25), (-3.0, 0.5, 10.0), (4.0, 4.0, -16.0), (0.75, -8.0, 1.0),
             (2.0, 2.0, 2.0), (-1.0, 5.0, 3.0), (100.0, 0.5, -50.0), (9.0, -1.0, 2.5))
LEGACY3 = sets((0.0, INF, 1.0), (INF, 0.0, -2.0), (2.0, 3.0, 0.5), (0.0, NAN, 4.0),
               (-1.5, 4.0, 6.0), (NAN, 0.0, -0.25), (0.25, -0.5, 1.0), (3.0, 3.0, -9.0))
DRAW = {op: TWO for op in ('v_add_f32', 'v_sub_f32', 'v_subrev_f32', 'v_mul_f32', 'v_min_f32', 'v_max_f32')}
DRAW.update({op: THREE for op in ('v_fma_f32', 'v_mad_f32', 'v_med3_f32', 'v_min3_f32', 'v_max3_f32')})
DRAW.update({'v_mul_legacy_f32': LEGACY, 'v_mad_legacy_f32': LEGACY3})
F = r'F\((v\d+)\)'
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_add_f32': (r'U\(%s \+ %s\)' % (F, F), r'U(F(\1) - F(\2))', op2(lambda a, b: a - b)),
    'v_sub_f32': (r'U\(%s - %s\)' % (F, F), r'U(F(\2) - F(\1))', op2(lambda a, b: b - a)),
    'v_subrev_f32': (r'U\(%s - %s\)' % (F, F), r'U(F(\2) - F(\1))', op2(lambda a, b: a - b)),
    'v_mul_f32': (r'U\(%s \* %s\)' % (F, F), r'U(F(\1) + F(\2))', op2(lambda a, b: a + b)),
    'v_mul_legacy_f32': (r'U\(mul_legacy\(', r'U(2.0 * mul_legacy(', op2(lambda a, b: 2.0 * legacy(a, b))),
    'v_min_f32': (r'U\(min\(', r'U(max(', op2(max)),
    'v_max_f32': (r'U\(max\(', r'U(min(', op2(min)),
    'v_fma_f32': (r' \+ (F\(v\d+\))\)', r' - \1)', op3(lambda a, b, c: a * b - c)),
    'v_mad_f32': (r'mad32\(%s, %s, (F\(v\d+\)), false\)' % (F, F), r'mad32(F(\1), F(\2), -\3, false)',
                  op3(lambda a, b, c: a * b - c)),
    'v_mad_legacy_f32': (r'mad32\(%s, %s, (F\(v\d+\)), true\)' % (F, F), r'mad32(2.0 * F(\1), F(\2), \3, true)',
                         op3(lambda a, b, c: 2.0 * legacy(a, b) + c)),
    'v_med3_f32': (r'med3_32\(%s, %s, %s\)' % (F, F, F), r'max(F(\1), max(F(\2), F(\3)))',
                   op3(lambda a, b, c: max(a, b, c))),
    'v_min3_f32': (r'U\(min\(%s, min\(' % F, r'U(max(F(\1), min(', op3(lambda a, b, c: max(a, min(b, c)))),
    'v_max3_f32': (r'U\(max\(%s, max\(' % F, r'U(min(F(\1), max(', op3(lambda a, b, c: min(a, max(b, c)))),
}
BODY = {
    'v_add_f32': words('000918060d0003d5010b0200020d1c060f0003d5030f0200'),
    'v_sub_f32': words('000918080d0004d5010b0200020d1c080f0004d5030f0200'),
    'v_subrev_f32': words('0009180a0d0005d5010b0200020d1c0a0f0005d5030f0200'),
    'v_mul_f32': words('000918100d0008d5010b0200020d1c100f0008d5030f0200'),
    'v_mul_legacy_f32': words('0009180e0d0007d5010b0200020d1c0e0f0007d5030f0200'),
    'v_min_f32': words('0009181e0d000fd5010b0200020d1c1e0f000fd5030f0200'),
    'v_max_f32': words('000918200d0010d5010b0200020d1c200f0010d5030f0200'),
    'v_fma_f32': words('0c004bd5000922040d004bd5010b26040e004bd5020d2a040f004bd5030f2e04'),
    'v_mad_f32': words('0c0041d5000922040d0041d5010b26040e0041d5020d2a040f0041d5030f2e04'),
    'v_mad_legacy_f32': words('0c0040d5000922040d0040d5010b26040e0040d5020d2a040f0040d5030f2e04'),
    'v_med3_f32': words('0c0057d5000922040d0057d5010b26040e0057d5020d2a040f0057d5030f2e04'),
    'v_min3_f32': words('0c0051d5000922040d0051d5010b26040e0051d5020d2a040f0051d5030f2e04'),
    'v_max3_f32': words('0c0054d5000922040d0054d5010b26040e0054d5020d2a040f0054d5030f2e04'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_add_f32 v1, v2, v3', 'v_mul_f32_e64 v1, -v2, |v3|', 'v_fma_f32 v1, v2, v3, v4',
                 'v_max3_f32 v1, v2, 1.0, v4', 'v_min_f32_e64 v1, v2, v3 clamp'],
    'refused': ['v_add_f32_e64 v1, v2, v3 neg_lo:[1,0,0]', 'v_mul_f32_e64 v1, v2, v3 neg_hi:[0,1,0]'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
