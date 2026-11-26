"""Arithmetic and refusal controls for signed multiply-high and selected halfword multiply-add."""
from itertools import product
import random
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import gfx2hlsl


def translate(asm):
    stage = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)
    stage.lines = []
    stage.used_v = set()
    stage.uniform_v = set()
    stage.exec_known = True
    stage.one(asm.split()[0], asm)
    return stage.lines

MASK = 0xffffffff


def expression(instruction):
    assignment, = translate(instruction)
    assert assignment.startswith('v1 = ') and assignment.endswith(';')
    return re.sub(r'\b(0x[0-9a-f]+|\d+)u\b', r'\1', assignment[5:-1])


def signed(value):
    return value - (1 << 32) if value >> 31 else value


def check():
    rng = random.Random(8614)
    edges = [0, 1, 0xffff, 0x10000, 0x7fffffff, 0x80000000, MASK]
    cases = list(product(edges, repeat=2))
    cases.extend((rng.getrandbits(32), rng.getrandbits(32)) for _ in range(1024))
    hi = compile(expression('v_mul_hi_i32 v1, v2, v3'), '<multiply-high>', 'eval')
    functions = {'__builtins__': {}, 'x2w_mul_hi_u32': lambda a, b: a * b >> 32}
    for a, b in cases:
        got = eval(hi, functions, {'v2': a, 'v3': b}) & MASK
        assert got == (signed(a) * signed(b) >> 32) & MASK
    for left, right in product(range(2), repeat=2):
        asm = 'v_mad_u32_u16 v1, v2, v3, v4 op_sel:[%d,%d,0]' % (left, right)
        mad = compile(expression(asm), '<multiply-add>', 'eval')
        for a, b in cases:
            c = rng.getrandbits(32)
            got = eval(mad, {'__builtins__': {}}, {'v2': a, 'v3': b, 'v4': c}) & MASK
            assert got == (((a >> (16 * left)) & 0xffff) * ((b >> (16 * right)) & 0xffff) + c) & MASK
    for operands in ('v2, v3, v4 op_sel:[0,0,1]', 'v2, v3, v4 op_sel:[0,0,0,1]',
                     'v2, 12, v4 op_sel:[0,1,0]', 'v2, v3, v4 clamp'):
        try:
            translate('v_mad_u32_u16 v1, ' + operands)
        except gfx2hlsl.Unsupported:
            continue
        raise AssertionError('unexpectedly accepted ' + operands)
    print('PASS: signed high word and four unsigned half selections, %d operand pairs each' % len(cases))


if __name__ == '__main__':
    check()
