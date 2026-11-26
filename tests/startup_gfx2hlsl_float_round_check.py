"""Binary32 rounding operations of tools/startup/gfx2hlsl.py (its VEC_F table).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_floor_f32, v_ceil_f32, v_trunc_f32, v_fract_f32 and v_rndne_f32, which no check ran before. The fixture and
runner are startup_gfx2hlsl_lane_alu.py: one operation per lane, alternating the e32 and e64 spellings, on flat
parameters the compilers cannot fold. The references follow the AMD RDNA 2 ISA reference: a zero result keeps the sign
of the source (ceil(-0.75) and trunc(-0.75) are -0), v_rndne_f32 rounds halves to even, and v_fract_f32 is
x - floor(x) capped at the largest binary32 below 1.0 (0x3f7fffff), so a tiny negative source does not give 1.0.
The inputs are exact halves and quarters, 1000000.5, the binary32 just below 0.5, 8388607.5 and -2^-30.
"""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import main, words  # noqa: E402


def f(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def v(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def signed_zero(function):
    """function on the binary32 value of a word; a zero result keeps the sign of the source."""
    def apply(a):
        x = v(a)
        result = float(function(x))
        return f(math.copysign(0.0, x) if result == 0 else result)
    return apply


def fract(a):
    x = v(a)
    return min(f(x - math.floor(x)), 0x3f7fffff)


ONE_BELOW_HALF = 0x3effffff
TINY_NEGATIVE = 0xb0800000  # -2^-30
OPS = {
    'v_floor_f32': (1, signed_zero(math.floor), True),
    'v_ceil_f32': (1, signed_zero(math.ceil), True),
    'v_trunc_f32': (1, signed_zero(math.trunc), True),
    'v_fract_f32': (1, fract, True),
    'v_rndne_f32': (1, signed_zero(round), True),
}
SETS = [(f(2.5), 0, 0), (f(-0.75), 0, 0), (f(3.5), 0, 0), (f(-2.5), 0, 0),
        (f(1000000.5), 0, 0), (ONE_BELOW_HALF, 0, 0), (TINY_NEGATIVE, 0, 0), (f(8388607.5), 0, 0)]
DRAW = {op: SETS for op in OPS}
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_floor_f32': (r'U\(floor\(', r'U(ceil(', signed_zero(math.ceil)),
    'v_ceil_f32': (r'U\(ceil\(', r'U(floor(', signed_zero(math.floor)),
    'v_trunc_f32': (r'U\(trunc\(', r'U(floor(', signed_zero(math.floor)),
    'v_fract_f32': (r'U\(fract32\(F\((v\d+)\)\)\)', r'U(F(\1) - trunc(F(\1)))', lambda a: f(v(a) - math.trunc(v(a)))),
    'v_rndne_f32': (r'U\(round\(', r'U(trunc(', signed_zero(math.trunc)),
}
BODY = {
    'v_floor_f32': words('0049187e0d00a4d50101000002491c7e0f00a4d503010000'),
    'v_ceil_f32': words('0045187e0d00a2d50101000002451c7e0f00a2d503010000'),
    'v_trunc_f32': words('0043187e0d00a1d50101000002431c7e0f00a1d503010000'),
    'v_fract_f32': words('0041187e0d00a0d50101000002411c7e0f00a0d503010000'),
    'v_rndne_f32': words('0047187e0d00a3d50101000002471c7e0f00a3d503010000'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_floor_f32 v1, v2', 'v_ceil_f32_e64 v1, -v2', 'v_trunc_f32_e64 v1, |v2| clamp',
                 'v_fract_f32 v1, s2', 'v_rndne_f32_e64 v1, v2 mul:2'],
    'refused': ['v_floor_f32_e64 v1, v2 neg_lo:[1]', 'v_rndne_f32_e64 v1, v2 neg_hi:[1]'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
