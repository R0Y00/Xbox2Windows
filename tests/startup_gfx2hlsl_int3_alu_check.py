"""Three-source integer vector operations of tools/startup/gfx2hlsl.py (its VEC_U table).

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the pixel bodies and compare them with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       draw one pixel per four input sets and compare the four result words

Covers v_bfe_u32, v_mad_u32_u24, v_lshl_add_u32, v_add_lshl_u32, v_add3_u32, v_or3_b32, v_and_or_b32,
v_lshl_or_b32, v_xad_u32 and v_bfi_b32, which no check ran before. All ten are VOP3 only. The fixture and runner
are startup_gfx2hlsl_lane_alu.py: one operation per lane on flat parameters the compilers cannot fold. The references
follow the AMD RDNA 2 ISA reference with plain integer arithmetic: an unsigned field of width S2[4:0] at offset
S1[4:0] for v_bfe_u32, the low 24 bits of S0 and S1 for v_mad_u32_u24, shift counts taken from bits 4:0, 32-bit
wrapping sums and the bitfield insert (S0 & S1) | (~S0 & S2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_lane_alu import MASK, main, words  # noqa: E402

M24 = 0xffffff
OPS = {
    'v_bfe_u32': (3, lambda a, b, c: (a >> (b & 31)) & ((1 << (c & 31)) - 1), False),
    'v_mad_u32_u24': (3, lambda a, b, c: (a & M24) * (b & M24) + c, False),
    'v_lshl_add_u32': (3, lambda a, b, c: (a << (b & 31)) + c, False),
    'v_add_lshl_u32': (3, lambda a, b, c: ((a + b) & MASK) << (c & 31), False),
    'v_add3_u32': (3, lambda a, b, c: a + b + c, False),
    'v_or3_b32': (3, lambda a, b, c: a | b | c, False),
    'v_and_or_b32': (3, lambda a, b, c: (a & b) | c, False),
    'v_lshl_or_b32': (3, lambda a, b, c: (a << (b & 31)) | c, False),
    'v_xad_u32': (3, lambda a, b, c: (a ^ b) + c, False),
    'v_bfi_b32': (3, lambda a, b, c: (a & b) | (~a & c), False),
}
# Input sets (src0, src1, src2), eight per operation: two draws of four pixels. Shift counts and widths above 31,
# sources with bits above bit 23, sums that wrap and masks with every bit pattern.
SETS = [(0x12345678, 4, 8), (0xdeadbeef, 31, 1), (0x80000000, 0, 0), (MASK, 33, 31),
        (0xf0f0f0f0, 16, 0x0f0f0f0f), (0x01000003, 0x02000005, 7), (0xaaaaaaaa, 0x55555555, 0xffff0000),
        (0x7fffffff, 1, 0x80000001)]
BFE = [(0x12345678, 4, 8), (0xdeadbeef, 31, 1), (0x80000000, 0, 0), (MASK, 33, 31),
       (0xf0f0f0f0, 4, 8), (0x89abcdef, 28, 4), (0x0000ff00, 8, 40), (0x80000001, 0, 31)]
DRAW = {op: SETS for op in OPS}
DRAW['v_bfe_u32'] = BFE
# A wrong lowering of each operation: (pattern in the translated text, replacement, reference of the mutation).
MUTATIONS = {
    'v_bfe_u32': (r'\(\(1u << \((v\d+) & 31\)\) - 1\)', r'(1u << (\1 & 31))',
                  lambda a, b, c: (a >> (b & 31)) & (1 << (c & 31))),
    'v_mad_u32_u24': (r'= \((v\d+) & 0xffffffu\) \* ', r'= \1 * ', lambda a, b, c: a * (b & M24) + c),
    'v_lshl_add_u32': (r'(\(v\d+ << \(v\d+ & 31\)\)) \+ ', r'\1 | ', lambda a, b, c: (a << (b & 31)) | c),
    'v_add_lshl_u32': (r'<< \((v\d+) & 31\)', r'<< (\1 & 15)', lambda a, b, c: ((a + b) & MASK) << (c & 15)),
    'v_add3_u32': (r'(v\d+) \+ (v\d+) \+ (v\d+)', r'\1 + \2 - \3', lambda a, b, c: a + b - c),
    'v_or3_b32': (r'(v\d+) \| (v\d+) \| (v\d+)', r'\1 ^ \2 ^ \3', lambda a, b, c: a ^ b ^ c),
    'v_and_or_b32': (r'\) \| (v\d+)', r') ^ \1', lambda a, b, c: (a & b) ^ c),
    'v_lshl_or_b32': (r'(\(v\d+ << \(v\d+ & 31\)\)) \| ', r'\1 + ', lambda a, b, c: (a << (b & 31)) + c),
    'v_xad_u32': (r'\((v\d+) \^ (v\d+)\)', r'(\1 | \2)', lambda a, b, c: (a | b) + c),
    'v_bfi_b32': (r'\(~(v\d+) & (v\d+)\)', r'(\1 & \2)', lambda a, b, c: (a & b) | (a & c)),
}
BODY = {
    'v_bfe_u32': words('0c0048d5000922040d0048d5010b26040e0048d5020d2a040f0048d5030f2e04'),
    'v_mad_u32_u24': words('0c0043d5000922040d0043d5010b26040e0043d5020d2a040f0043d5030f2e04'),
    'v_lshl_add_u32': words('0c0046d7000922040d0046d7010b26040e0046d7020d2a040f0046d7030f2e04'),
    'v_add_lshl_u32': words('0c0047d7000922040d0047d7010b26040e0047d7020d2a040f0047d7030f2e04'),
    'v_add3_u32': words('0c006dd7000922040d006dd7010b26040e006dd7020d2a040f006dd7030f2e04'),
    'v_or3_b32': words('0c0072d7000922040d0072d7010b26040e0072d7020d2a040f0072d7030f2e04'),
    'v_and_or_b32': words('0c0071d7000922040d0071d7010b26040e0071d7020d2a040f0071d7030f2e04'),
    'v_lshl_or_b32': words('0c006fd7000922040d006fd7010b26040e006fd7020d2a040f006fd7030f2e04'),
    'v_xad_u32': words('0c0045d7000922040d0045d7010b26040e0045d7020d2a040f0045d7030f2e04'),
    'v_bfi_b32': words('0c004ad5000922040d004ad5010b26040e004ad5020d2a040f004ad5030f2e04'),
}
FAMILY = {
    'ops': OPS, 'draw': DRAW, 'mutations': MUTATIONS, 'body': BODY,
    'accepted': ['v_bfe_u32 v1, v2, 8, 4', 'v_mad_u32_u24 v1, v2, v3, v4', 'v_add3_u32 v1, v2, s3, 1',
                 'v_bfi_b32 v1, v2, v3, v4', 'v_lshl_or_b32 v1, v2, 16, v3'],
    'refused': ['v_bfe_u32 v1, v2, v3', 'v_add3_u32 v1, v2, v3, v4, v5', 'v_or3_b32 v1, v2',
                'v_mad_u32_u24 v1, v2, v3, v4 clamp', 'v_xad_u32 v1, -v2, v3, v4', 'v_bfi_b32 v1, |v2|, v3, v4',
                'v_and_or_b32 v1, v2, v3, v4 op_sel:[1,0,0,0]'],
}

if __name__ == '__main__':
    main(FAMILY, __doc__)
