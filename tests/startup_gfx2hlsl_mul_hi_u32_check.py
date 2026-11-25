"""Translation controls for unsigned 32-bit high-half multiply (v_mul_hi_u32) in gfx2hlsl.

  (no arguments)                       translation controls, pure Python
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl  # noqa: E402


MASK = 0xffffffff
PAIRS = [
    (0, 0),
    (0, MASK),
    (1, MASK),
    (0x80000000, 2),
    (0x10000, 0x10000),
    (0xffff, 0xffff),
    (0x8000ffff, 0x8000ffff),
    (0xffffffff, 0xffffffff),
    (0x12345678, 0x9abcdef0),
    (0x7fffffff, 0x80000001),
]


def hlsl_mul_hi_u32(a, b):
    """Exact 32-bit evaluation of the mul_hi_u32 HLSL helper emitted by gfx2hlsl."""
    la = a & 0xffff
    lb = b & 0xffff
    middle = (((a >> 16) * lb) & MASK) + (((la * lb) & MASK) >> 16)
    assert middle <= MASK, (a, b, middle)
    mid_lo = (middle & 0xffff) + ((la * (b >> 16)) & MASK)
    assert mid_lo <= MASK, (a, b, mid_lo)
    result = (((a >> 16) * (b >> 16)) & MASK) + (middle >> 16) + (mid_lo >> 16)
    assert result <= MASK, (a, b, result)
    return result


def translate(asm):
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
    stage.lines = []
    stage.used_v = set()
    stage.uniform_v = set()
    stage.exec_known = True
    stage.one(asm.split()[0], asm)
    return stage.lines


def refused(asm, reason):
    try:
        translate(asm)
    except gfx2hlsl.Unsupported as exc:
        assert str(exc) == reason, (asm, str(exc))
    else:
        raise AssertionError('accepted %s' % asm)


def main():
    for suffix in ('', '_e64'):
        asm = 'v_mul_hi_u32%s v1, v2, v3' % suffix
        lines = translate(asm)
        assert lines == ['v1 = mul_hi_u32(v2, v3);'], (asm, lines)

    # Inline-constant and literal sources go through the shared u() path as 32-bit words.
    lines = translate('v_mul_hi_u32 v1, v2, 0xaaaaaaab')
    assert lines == ['v1 = mul_hi_u32(v2, 0xaaaaaaabu);'], lines
    lines = translate('v_mul_hi_u32_e64 v1, -1, v2')
    assert lines == ['v1 = mul_hi_u32(0xffffffffu, v2);'], lines

    # Verify the emitted mul_hi_u32 HLSL helper matches (a * b) >> 32 without 32-bit intermediate overflow.
    for a, b in PAIRS:
        for x, y in ((a, b), (b, a)):
            expected = (x * y) >> 32
            actual = hlsl_mul_hi_u32(x, y)
            assert actual == expected, (hex(x), hex(y), hex(actual), hex(expected))

    # Modifiers, float-style operand modifiers, and wrong operand counts are refused.
    refused('v_mul_hi_u32_e64 v1, v2, v3 clamp', "v_mul_hi_u32 with ['clamp']")
    refused('v_mul_hi_u32_e64 v1, -v2, v3', "operand '-v2'")
    refused('v_mul_hi_u32 v1, v2', 'v_mul_hi_u32 with 2 operands, expected 3')

    # Negative control: removing the entry from VEC_U must refuse translation.
    entry = gfx2hlsl.VEC_U.pop('v_mul_hi_u32')
    try:
        refused('v_mul_hi_u32 v1, v2, v3', 'v_mul_hi_u32')
    finally:
        gfx2hlsl.VEC_U['v_mul_hi_u32'] = entry

    print('PASS: startup_gfx2hlsl_mul_hi_u32_check')


if __name__ == '__main__':
    main()
