"""32-bit hex literal as the 64-bit source of v_lshlrev_b64, v_lshrrev_b64 and v_ashrrev_i64 (Refs #1723).

A literal fills the low word. The high word is zero for the unsigned shifts and copies bit 31 for the signed
shift (the rule of s_mov_b64; the scalar shifts and bitfield extracts take it from this change). Pure Python; run with no arguments.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_scalar_m0_check as M  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe, check, refused  # noqa: E402

# (operation, helper, literal, expected low word, expected high word)
CASES = (
    ('v_lshlrev_b64', 'shl64', '0x41', '65u', '0u'),
    ('v_lshrrev_b64', 'shr64u', '0x80000000', '2147483648u', '0u'),
    ('v_lshrrev_b64', 'shr64u', '0xffffffff', '4294967295u', '0u'),
    ('v_ashrrev_i64', 'shr64i', '0x7fffffff', '2147483647u', '0u'),
    ('v_ashrrev_i64', 'shr64i', '0x80000000', '2147483648u', '4294967295u'),
    ('v_ashrrev_i64', 'shr64i', '0xffffffff', '4294967295u', '4294967295u'),
)


def lowered(op, literal):
    probe = Probe()
    probe.one(op, '%s v[0:1], v2, %s' % (op, literal))
    return probe.lines


def scalar_lowered(opcode):
    """s_<op> s[20:21], 0xffffffff, s9 as encoded (SOP2, source 255 plus a literal dword)."""
    stage = M.run_seq(M.encode_sop2(opcode, 20, 255, 9) + struct.pack('<I', 0xffffffff))
    return [line for line in stage.lines if 'shr64' in line or 'bfe64' in line]


def main():
    check('s_ashr_i64 literal 0xffffffff is sign-extended',
          scalar_lowered(0x23)[0].startswith('{ uint2 t = shr64i(4294967295u, 4294967295u, s9);'))
    check('s_lshr_b64 literal 0xffffffff is zero-extended',
          scalar_lowered(0x21)[0].startswith('{ uint2 t = shr64u(4294967295u, 0u, s9);'))
    check('s_bfe_i64 literal 0xffffffff is sign-extended',
          'bfe64(4294967295u, 4294967295u, s9, 1u)' in scalar_lowered(0x2a)[0])
    check('s_bfe_u64 literal 0xffffffff is zero-extended',
          'bfe64(4294967295u, 0u, s9, 0u)' in scalar_lowered(0x29)[0])
    for op, helper, literal, low, high in CASES:
        want = ['{ uint2 t = %s(%s, %s, v2); v0 = t.x; v1 = t.y; }' % (helper, low, high)]
        check('%s with literal %s' % (op, literal), lowered(op, literal) == want)
    check('literals wider than 32 bits stay refused',
          all(refused('%s v[0:1], v2, 0x100000000' % op) for op in ('v_lshlrev_b64', 'v_ashrrev_i64')))
    check('a hex literal stays refused as the addend of v_mad_u64_u32 and v_mad_i64_i32',
          all(refused('%s v[0:1], vcc_lo, v2, v3, 0x41' % op) for op in ('v_mad_u64_u32', 'v_mad_i64_i32')))
    check('a literal is not accepted as a destination or in the wrong operand count',
          refused('v_lshrrev_b64 0x41, v2, v[4:5]') and refused('v_lshrrev_b64 v[0:1], 0x41'))


if __name__ == '__main__':
    main()
