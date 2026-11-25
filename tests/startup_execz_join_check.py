"""A join where one register is symbolic only on the s_cbranch_execz edge, then ANDed with exec in place."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_scalar_bits_check import sop1, stage  # noqa: E402

CMP_VCC = 0x7c000000 | 0xc1 << 17 | 0 << 9 | 129      # v_cmp_lt_u32 vcc, 1, v0
SAVE32 = sop1(0x3c, 20, 106)                           # s_and_saveexec_b32 s20, vcc_lo
NOP = 0xbf800000


def sop2(op, dst, a, b):
    return 0x80000000 | op << 23 | dst << 16 | b << 8 | a


def program(branch_op=8, before_use=(), use=None):
    use = use if use is not None else sop2(0x0e, 13, 13, 126)        # s_and_b32 s13, s13, exec_lo
    return [sop1(3, 13, 0), CMP_VCC, SAVE32, 0xbf800000 | branch_op << 16 | 2,
            sop2(0x0e, 13, 106, 126), NOP] + list(before_use) + [use]


def refused(words, label):
    try:
        stage(words)
    except G.Unsupported as error:
        assert 'conflict' in str(error) or 'execzero' in str(error), (label, str(error))
        return 1
    raise AssertionError('accepted: ' + label)


def main():
    s = stage(program())
    body = [line for _, lines, _ in s.blocks for line in lines]
    assert 's13 = (s13 & exec) & 1;' in body, body
    for v0, want in ((0, 0), (2, 1)):
        exec_bit = int(1 < v0)
        s13 = 0x5678 if not exec_bit else 1
        assert (s13 & exec_bit) & 1 == want
    controls = refused(program(branch_op=4), 's_cbranch_scc0 edge')
    controls += refused(program(before_use=[sop1(3, 126, 20)]), 'exec written before the AND')
    controls += refused(program(use=sop2(0x1e, 5, 13, 0x81)), 's_lshl_b32 use')   # s_lshl_b32 s5, s13, 1
    print('execz_join: in-place AND with exec after an execz-only conflict reads the run-time variable (0 on the '
          'taken path, the mask on the other); %d controls refused' % controls)


if __name__ == '__main__':
    main()
