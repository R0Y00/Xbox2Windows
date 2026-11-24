"""s_bfm_b64 in tools/startup/gfx2hlsl.py (Part of #1723), the 64-bit twin of s_bfm_b32.

usage: startup_gfx2hlsl_bfm_b64_check.py

Pure Python, no GPU. Constant operands fold to two exact 32-bit words; run-time operands emit one block
that reads both sources before writing either destination register; unsupported shapes are refused.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_scalar_m0_check as M  # noqa: E402

G = M.G
SOP2_BFM_B64 = 0x25
DST = 20          # s[20:21]
CONST_W, CONST_O = 6, 7


def run(width_sym, offset_sym, *, dst=DST, w=CONST_W, o=CONST_O):
    stage = M.ProbeStage()
    stage.sym[CONST_W], stage.sym[CONST_O] = ('const', width_sym), ('const', offset_sym)
    code = M.encode_sop2(SOP2_BFM_B64, dst, w, o)
    _, name, asm, _ = M.D.decode_instruction(code, 0)
    assert name == 's_bfm_b64', name
    stage.one(name, asm)
    return stage


def words(stage):
    lo, hi = stage.sym[DST], stage.sym[DST + 1]
    assert lo[0] == hi[0] == 'const', (lo, hi)
    assert stage.lines == ['s%d = 0x%08xu;' % (DST, lo[1]), 's%d = 0x%08xu;' % (DST + 1, hi[1])], stage.lines
    return lo[1], hi[1]


def controls():
    # (width, offset) -> (low word, high word), written out by hand from the 64-bit definition.
    cases = (
        (0, 0, 0x00000000, 0x00000000),
        (1, 0, 0x00000001, 0x00000000),
        (16, 0, 0x0000ffff, 0x00000000),
        (32, 0, 0xffffffff, 0x00000000),      # the width the 32-bit op cannot express
        (33, 0, 0xffffffff, 0x00000001),
        (63, 0, 0xffffffff, 0x7fffffff),
        (64, 0, 0x00000000, 0x00000000),      # only bits [5:0] count: width 64 is width 0
        (1, 63, 0x00000000, 0x80000000),
        (4, 30, 0xc0000000, 0x00000003),      # the mask crosses the word boundary
        (8, 32, 0x00000000, 0x000000ff),
        (32, 16, 0xffff0000, 0x0000ffff),
        (16, 100, 0x00000000, 0x000ffff0),  # offset 100 -> 36: bits 36..51
        (1, 64, 0x00000001, 0x00000000),      # only bits [5:0] of the offset count: offset 64 is offset 0
    )
    for width, offset, lo, hi in cases:
        got = words(run(width, offset))
        assert got == (lo, hi), (width, offset, '%08x %08x' % got)

    # A run-time source keeps the mask in run-time code: both sources are read inside one block, so a
    # destination that overlaps a source (s[6:7] = bfm(s6, s7)) cannot feed a half-written value back.
    stage = M.ProbeStage()
    _, name, asm, _ = M.D.decode_instruction(M.encode_sop2(SOP2_BFM_B64, 8, 8, 128 + 2), 0)
    stage.one(name, asm)
    assert stage.sym.get(8) is None and stage.sym.get(9) is None, stage.sym
    block = [line for line in stage.lines if 'shl64' in line]
    assert len(block) == 1 and block[0].count('s8 = ') == 1 and block[0].count('s9 = ') == 1, stage.lines
    assert block[0].index('ud') < block[0].index('s8 = '), block[0]
    assert {8, 9} <= stage.used_s, stage.used_s

    # SCC is not touched.
    assert not any('scc' in line for line in stage.lines), stage.lines

    for dst in (M.M0, 106):
        try:
            run(1, 1, dst=dst)
        except G.Unsupported:
            pass
        else:
            raise AssertionError('destination %d should be refused' % dst)


def main():
    controls()
    print('startup s_bfm_b64 controls passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
