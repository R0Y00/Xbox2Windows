"""Binary32 reciprocal, root, exponential and logarithm of tools/startup/gfx2hlsl.py (its VEC_F table).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_rcp_f32, v_rcp_iflag_f32, v_sqrt_f32, v_rsq_f32, v_exp_f32 and v_log_f32, which no check ran before. They
lower to 1.0 / x, sqrt, rsqrt, exp2 and log2. The fixture and runner are startup_gfx2hlsl_lane_alu.py: one operation
per lane, alternating the e32 and e64 spellings, on flat parameters the compilers cannot fold. The references follow
the AMD RDNA 2 ISA reference (v_exp_f32 and v_log_f32 are base 2). Every input is a power of two, or 0 and 9 for the
square root, so every result is exactly representable: the check tests which function each operation is lowered to,
not its precision. Adapters need not return those results exactly. An NVIDIA GeForce RTX 4090 reads log2(0.125) as
c03fffff, 1 ulp above -3.0 (#2797), so v_log_f32 accepts 4 ulps: the Direct3D 11.3 functional specification bounds
log to 2^-21 relative error, or 2^-21 absolute for inputs in [0.5, 2], and both are at least 4 ulps of a binary32
result in this fixture. The other operations stay exact. The mutation (the logarithm plus one) is off by far more.
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


def op(function):
    return lambda a: f(function(v(a)))


OPS = {
    'v_rcp_f32': (1, op(lambda x: 1.0 / x), True),
    'v_rcp_iflag_f32': (1, op(lambda x: 1.0 / x), True),
    'v_sqrt_f32': (1, op(math.sqrt), True),
    'v_rsq_f32': (1, op(lambda x: 1.0 / math.sqrt(x)), True),
    'v_exp_f32': (1, op(lambda x: 2.0 ** x), True),
    'v_log_f32': (1, op(math.log2), True),
}


def sets(*values):
    return [(f(x), 0, 0) for x in values]


RCP = sets(4.0, -0.5, 1.0, 0.125, 2.0 ** 20, -8.0, 256.0, 2.0 ** -10)
DRAW = {
    'v_rcp_f32': RCP, 'v_rcp_iflag_f32': RCP,
    'v_sqrt_f32': sets(16.0, 0.25, 1.0, 4096.0, 2.0 ** -20, 64.0, 0.0, 9.0),
    'v_rsq_f32': sets(4.0, 0.25, 1.0, 16.0, 2.0 ** 20, 64.0, 2.0 ** -20, 0.0625),
    'v_exp_f32': sets(3.0, -2.0, 0.0, 10.0, -10.0, 1.0, 20.0, -1.0),
    'v_log_f32': sets(8.0, 0.125, 1.0, 1024.0, 2.0 ** -20, 2.0, 2.0 ** 30, 0.5),
}
F = r'F\((v\d+)\)'
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_rcp_f32': (r'U\(1\.0 / ', r'U(2.0 / ', op(lambda x: 2.0 / x)),
    'v_rcp_iflag_f32': (r'U\(1\.0 / ', r'U(2.0 / ', op(lambda x: 2.0 / x)),
    'v_sqrt_f32': (r'U\(sqrt\(', r'U(rsqrt(', op(lambda x: 1.0 / math.sqrt(x))),
    'v_rsq_f32': (r'U\(rsqrt\(', r'U(sqrt(', op(math.sqrt)),
    'v_exp_f32': (r'U\(exp2\(%s\)\)' % F, r'U(exp2(F(\1)) * 2.0)', op(lambda x: 2.0 ** (x + 1))),
    'v_log_f32': (r'U\(log2\(%s\)\)' % F, r'U(log2(F(\1)) + 1.0)', op(lambda x: math.log2(x) + 1)),
}
BODY = {
    'v_rcp_f32': words('0055187e0d00aad50101000002551c7e0f00aad503010000'),
    'v_rcp_iflag_f32': words('0057187e0d00abd50101000002571c7e0f00abd503010000'),
    'v_sqrt_f32': words('0067187e0d00b3d50101000002671c7e0f00b3d503010000'),
    'v_rsq_f32': words('005d187e0d00aed501010000025d1c7e0f00aed503010000'),
    'v_exp_f32': words('004b187e0d00a5d501010000024b1c7e0f00a5d503010000'),
    'v_log_f32': words('004f187e0d00a7d501010000024f1c7e0f00a7d503010000'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_rcp_f32 v1, v2', 'v_sqrt_f32_e64 v1, |v2|', 'v_rsq_f32_e64 v1, v2 clamp', 'v_exp_f32 v1, 1.0',
                 'v_log_f32_e64 v1, -v2 mul:2'],
    'refused': ['v_rcp_f32_e64 v1, v2 neg_lo:[1]', 'v_log_f32_e64 v1, v2 neg_hi:[1]'],
    'ulps': {'v_log_f32': 4},
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
