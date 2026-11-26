"""Two- and one-source integer vector operations of tools/startup/gfx2hlsl.py (its VEC_U table).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_add_nc_u32, v_sub_nc_u32, v_subrev_nc_u32, v_mul_lo_u32, v_and_b32, v_or_b32, v_xor_b32, v_not_b32,
v_lshlrev_b32, v_lshrrev_b32, v_ashrrev_i32, v_min_u32, v_max_u32, v_min_i32, v_max_i32 and v_mul_u32_u24, which
no check ran before. The fixture and runner are startup_gfx2hlsl_lane_alu.py: one operation per lane, alternating
the e32 and e64 spellings where both exist (v_mul_lo_u32 is VOP3 only), on flat parameters the compilers cannot fold.
The references follow the AMD RDNA 2 ISA reference: 32-bit wrapping arithmetic, shifts by S0[4:0] of S1 (the
"rev" forms), an arithmetic shift for v_ashrrev_i32, signed or unsigned comparisons by name, and the 24-bit
multiply of the low 24 bits of each source. The inputs include shift counts above 31, values with the sign bit
set for the signed and unsigned pairs, products that wrap and sources with bits above bit 23.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import MASK, main, signed, words  # noqa: E402

M24 = 0xffffff
OPS = {
    'v_add_nc_u32': (2, lambda a, b: a + b, True),
    'v_sub_nc_u32': (2, lambda a, b: a - b, True),
    'v_subrev_nc_u32': (2, lambda a, b: b - a, True),
    'v_mul_lo_u32': (2, lambda a, b: a * b, False),
    'v_and_b32': (2, lambda a, b: a & b, True),
    'v_or_b32': (2, lambda a, b: a | b, True),
    'v_xor_b32': (2, lambda a, b: a ^ b, True),
    'v_not_b32': (1, lambda a: ~a, True),
    'v_lshlrev_b32': (2, lambda a, b: b << (a & 31), True),
    'v_lshrrev_b32': (2, lambda a, b: b >> (a & 31), True),
    'v_ashrrev_i32': (2, lambda a, b: signed(b) >> (a & 31), True),
    'v_min_u32': (2, min, True),
    'v_max_u32': (2, max, True),
    'v_min_i32': (2, lambda a, b: min(signed(a), signed(b)), True),
    'v_max_i32': (2, lambda a, b: max(signed(a), signed(b)), True),
    'v_mul_u32_u24': (2, lambda a, b: (a & M24) * (b & M24), True),
}
# Input sets (src0, src1, unused), eight per operation: two draws of four pixels.
ARITH = [(1, MASK, 0), (0x80000000, 0x80000000, 0), (0x12345678, 0x9abcdef0, 0), (0, 1, 0),
         (0x7fffffff, 1, 0), (0xdeadbeef, 0xfeedface, 0), (5, 3, 0), (MASK, MASK, 0)]
LOGIC = [(0xf0f0f0f0, 0x0ff00ff0, 0), (0, MASK, 0), (0x12345678, 0x9abcdef0, 0), (MASK, 0x80000001, 0),
         (0xaaaaaaaa, 0x55555555, 0), (0xdeadbeef, 0, 0), (1, 3, 0), (0x00ff0000, 0x0f0f0f0f, 0)]
SHIFT = [(4, 0x80000001, 0), (31, 0xc0000000, 0), (33, 0x80000010, 0), (0, 0xdeadbeef, 0),
         (16, 0x12345678, 0), (0xffffffe1, 0xf0000000, 0), (32, 0x7fffffff, 0), (7, 0xffffff80, 0)]
ORDER = [(1, 0x80000000, 0), (MASK, 0, 0), (0x7fffffff, 0x80000000, 0), (5, 5, 0),
         (0xfffffffe, 0xffffffff, 0), (100, 0x80000001, 0), (0, 0x7fffffff, 0), (0x80000000, 0x7fffffff, 0)]
MUL24 = [(0x01000003, 0x02000005, 0), (M24, M24, 0), (0x12345678, 0x9abcdef0, 0), (0x00800000, 2, 0),
         (MASK, 1, 0), (0x00010000, 0x00010000, 0), (0xff000000, 0xff123456, 0), (7, 0x00ffffff, 0)]
DRAW = {op: ARITH for op in ('v_add_nc_u32', 'v_sub_nc_u32', 'v_subrev_nc_u32', 'v_mul_lo_u32')}
DRAW.update({op: LOGIC for op in ('v_and_b32', 'v_or_b32', 'v_xor_b32', 'v_not_b32')})
DRAW.update({op: SHIFT for op in ('v_lshlrev_b32', 'v_lshrrev_b32', 'v_ashrrev_i32')})
DRAW.update({op: ORDER for op in ('v_min_u32', 'v_max_u32', 'v_min_i32', 'v_max_i32')})
DRAW['v_mul_u32_u24'] = MUL24
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_add_nc_u32': (r'= (v\d+) \+ (v\d+);', r'= \1 - \2;', lambda a, b: a - b),
    'v_sub_nc_u32': (r'= (v\d+) - (v\d+);', r'= \2 - \1;', lambda a, b: b - a),
    'v_subrev_nc_u32': (r'= (v\d+) - (v\d+);', r'= \2 - \1;', lambda a, b: a - b),
    'v_mul_lo_u32': (r'= (v\d+) \* (v\d+);', r'= \1 * (\2 & 0xffffffu);', lambda a, b: a * (b & M24)),
    'v_and_b32': (r'= (v\d+) & (v\d+);', r'= \1 | \2;', lambda a, b: a | b),
    'v_or_b32': (r'= (v\d+) \| (v\d+);', r'= \1 ^ \2;', lambda a, b: a ^ b),
    'v_xor_b32': (r'= (v\d+) \^ (v\d+);', r'= \1 | \2;', lambda a, b: a | b),
    'v_not_b32': (r'= ~(v\d+);', r'= \1;', lambda a: a),
    'v_lshlrev_b32': (r'= (v\d+) << \((v\d+) & 31\);', r'= \2 << (\1 & 31);', lambda a, b: a << (b & 31)),
    'v_lshrrev_b32': (r'>> \((v\d+) & 31\)', r'>> (\1 & 15)', lambda a, b: b >> (a & 15)),
    'v_ashrrev_i32': (r'uint\(int\((v\d+)\) >> ', r'(\1 >> ', lambda a, b: b >> (a & 31)),
    'v_min_u32': (r'min\((v\d+), (v\d+)\)', r'max(\1, \2)', max),
    'v_max_u32': (r'max\((v\d+), (v\d+)\)', r'min(\1, \2)', min),
    'v_min_i32': (r'uint\(min\(int\((v\d+)\), int\((v\d+)\)\)\)', r'min(\1, \2)', min),
    'v_max_i32': (r'uint\(max\(int\((v\d+)\), int\((v\d+)\)\)\)', r'max(\1, \2)', max),
    'v_mul_u32_u24': (r'\((v\d+) & 0xffffffu\) \* \((v\d+) & 0xffffffu\)', r'\1 * \2', lambda a, b: a * b),
}
BODY = {
    'v_add_nc_u32': words('0009184a0d0025d5010b0200020d1c4a0f0025d5030f0200'),
    'v_sub_nc_u32': words('0009184c0d0026d5010b0200020d1c4c0f0026d5030f0200'),
    'v_subrev_nc_u32': words('0009184e0d0027d5010b0200020d1c4e0f0027d5030f0200'),
    'v_mul_lo_u32': words('0c0069d5000902000d0069d5010b02000e0069d5020d02000f0069d5030f0200'),
    'v_and_b32': words('000918360d001bd5010b0200020d1c360f001bd5030f0200'),
    'v_or_b32': words('000918380d001cd5010b0200020d1c380f001cd5030f0200'),
    'v_xor_b32': words('0009183a0d001dd5010b0200020d1c3a0f001dd5030f0200'),
    'v_not_b32': words('006f187e0d00b7d501010000026f1c7e0f00b7d503010000'),
    'v_lshlrev_b32': words('000918340d001ad5010b0200020d1c340f001ad5030f0200'),
    'v_lshrrev_b32': words('0009182c0d0016d5010b0200020d1c2c0f0016d5030f0200'),
    'v_ashrrev_i32': words('000918300d0018d5010b0200020d1c300f0018d5030f0200'),
    'v_min_u32': words('000918260d0013d5010b0200020d1c260f0013d5030f0200'),
    'v_max_u32': words('000918280d0014d5010b0200020d1c280f0014d5030f0200'),
    'v_min_i32': words('000918220d0011d5010b0200020d1c220f0011d5030f0200'),
    'v_max_i32': words('000918240d0012d5010b0200020d1c240f0012d5030f0200'),
    'v_mul_u32_u24': words('000918160d000bd5010b0200020d1c160f000bd5030f0200'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_add_nc_u32 v1, v2, v3', 'v_mul_lo_u32 v1, v2, v3', 'v_not_b32_e64 v1, v2',
                 'v_ashrrev_i32_e64 v1, v2, v3', 'v_min_i32 v1, 5, v3', 'v_mul_u32_u24_e64 v1, v2, v3'],
    'refused': ['v_add_nc_u32 v1, v2', 'v_mul_lo_u32 v1, v2, v3, v4', 'v_not_b32 v1, v2, v3',
                'v_add_nc_u32_e64 v1, v2, v3 clamp', 'v_and_b32_e64 v1, -v2, v3', 'v_max_i32_e64 v1, |v2|, v3',
                'v_lshlrev_b32_e64 v1, v2, v3 mul:2', 'v_xor_b32 v1, v2'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
