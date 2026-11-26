"""The SCC carry chain of s_add_u32 / s_addc_u32 and s_sub_u32 / s_subb_u32 in tools/startup/gfx2hlsl.py.

usage: startup_gfx2hlsl_scc_carry_check.py

Synthetic SOP2 words only; the spellings are asserted from the translator's decoder.
AMD RDNA 2 ISA SOP2: S_ADD_U32 D = S0 + S1, SCC = carry out; S_ADDC_U32 D = S0 + S1 + SCC, SCC = carry out;
S_SUB_U32 D = S0 - S1, SCC = borrow; S_SUBB_U32 D = S0 - S1 - SCC, SCC = borrow (all unsigned, 33-bit).
Checks:
  values:   the emitted lines, evaluated on a grid of words and both carry-in values (0xffffffff + 0 + 1 carries,
            0 - 0 - 1 borrows), equal a 64-bit reference for D and SCC, also with D equal to a source;
            a 64-bit add and subtract built from the pairs equal the 64-bit result.
  pre-pass: carry_scc_writers lists an add only where its SCC can reach a reader (straight line and through a
            branch), not past an s_cmp, not for a carry nobody reads; those programs' lines carry no SCC (the HLSL
            of programs whose carry is never read does not change).
  refused:  s_add_i32's SCC (signed overflow, not modelled) read by s_addc_u32; s_addc_u32 on exec or a lane mask.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402
from startup_scalar_bits_check import stage  # noqa: E402

MASK = 0xffffffff
GRID = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, MASK, 0x12345678]
OPS = {'s_add_u32': 0, 's_sub_u32': 1, 's_add_i32': 2, 's_addc_u32': 4, 's_subb_u32': 5}


def sop2(name, d, a, b):
    return 0x80000000 | OPS[name] << 23 | d << 16 | b << 8 | a


def decode(words):
    return G.decode(struct.pack('<%dI' % len(words), *words))


def split_top(text):
    depth, parts, cur = 0, [], ''
    for ch in text:
        depth += ch == '('
        depth -= ch == ')'
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    return parts + [cur]


def evaluate(lines, env):
    env = dict(env)
    for line in lines:
        target, expr = line.rstrip(';').split(' = ', 1)
        expr = expr.replace('q.x', 'qx').replace('q.y', 'qy')
        if target == 'q':
            inner = re.fullmatch(r'uint2\((.*)\)', expr).group(1)
            env['qx'], env['qy'] = (Expr(p).run(env)[0] & MASK for p in split_top(inner))
        else:
            env[target] = Expr(expr).run(env)[0] & (1 if target == 'scc' else MASK)
    return env


def lines_of(name, text, carry=True):
    s = stage([])
    s.lines, s.pc = [], 0
    s.carry_writes = {0} if carry else set()
    s.one(name, text)
    return s.lines


def ref(name, a, b, c):
    if name.startswith('s_add'):
        full = a + b + c
        return full & MASK, int(full >> 32 != 0)
    full = a - b - c
    return full & MASK, int(full < 0)


def values():
    checked = 0
    for name in ('s_add_u32', 's_addc_u32', 's_sub_u32', 's_subb_u32'):
        (_, _, dname, text), = decode([sop2(name, 24, 20, 22)])
        assert dname == name and text == '%s s24, s20, s22' % name, text
        same = decode([sop2(name, 20, 20, 22)])[0][3]       # D = S0
        same1 = decode([sop2(name, 22, 20, 22)])[0][3]      # D = S1
        for form, dst in ((text, 's24'), (same, 's20'), (same1, 's22')):
            lines = lines_of(name, form)
            for a in GRID:
                for b in GRID:
                    for c in (0, 1):  # the incoming SCC; only addc / subb read it
                        cin = c if name in ('s_addc_u32', 's_subb_u32') else 0
                        env = evaluate(lines, {'s20': a, 's22': b, 'scc': c})
                        want = ref(name, a, b, cin)
                        assert (env[dst], env['scc']) == want, (name, form, a, b, c, env[dst], env['scc'], want)
                        checked += 1
    # 64-bit chains: s[20:21] + s[22:23] -> s[24:25], and the subtract.
    for lo, hi in (('s_add_u32', 's_addc_u32'), ('s_sub_u32', 's_subb_u32')):
        lines = lines_of(lo, decode([sop2(lo, 24, 20, 22)])[0][3]) + lines_of(hi, decode([sop2(hi, 25, 21, 23)])[0][3])
        for x in (0, 1, 0xffffffff, 0x100000000, 0xffffffffffffffff, 0x123456789abcdef0):
            for y in (0, 1, 0xffffffff, 0xffffffffffffffff, 0x0fedcba987654321):
                env = evaluate(lines, {'s20': x & MASK, 's21': x >> 32, 's22': y & MASK, 's23': y >> 32, 'scc': 0})
                full = x + y if lo == 's_add_u32' else x - y
                assert env['s24'] | env['s25'] << 32 == full & 0xffffffffffffffff, (lo, hex(x), hex(y))
                assert env['scc'] == int(not 0 <= full < 1 << 64), (lo, hex(x), hex(y))
                checked += 1
    print('values: %d (D, SCC) results equal the 33-bit reference, incl. 0xffffffff + 0 + 1 and 0 - 0 - 1; '
          '64-bit add/sub chains exact' % checked)


def prepass():
    end = 0xbf810000
    add, addc = sop2('s_add_u32', 24, 20, 22), sop2('s_addc_u32', 25, 21, 23)
    cmp = 0xbf000000 | 21 << 8 | 20                      # s_cmp_eq_i32 s20, s21
    ins = decode([add, addc, end])
    assert ins[2][2] == 's_endpgm'
    assert G.carry_scc_writers(ins, {}) == {0}, G.carry_scc_writers(ins, {})
    ins = decode([add, cmp, addc, end])
    assert ins[1][3] == 's_cmp_eq_i32 s20, s21', ins[1]
    assert G.carry_scc_writers(ins, {}) == set()
    ins = decode([add, end])
    assert G.carry_scc_writers(ins, {}) == set()
    # through a branch: s_add_u32; s_cbranch_execz +1 (skips the s_cmp); s_cmp; s_addc_u32
    branch = 0xbf880001
    ins = decode([add, branch, cmp, addc, end])
    assert ins[1][3].startswith('s_cbranch_execz'), ins[1]
    assert G.carry_scc_writers(ins, {}) == {0}
    # the calculated-PC proof's arithmetic is neither writer nor reader here
    ins = decode([add, addc, end])
    assert G.carry_scc_writers(ins, {0: None, 4: None}) == set()
    # programs: the add whose carry is read emits scc, the one whose carry is not read does not
    s = stage([add, addc])
    text = '\n'.join(line for _, lines, _ in s.blocks for line in lines)
    assert 'scc = q.x < q.y;' in text and 'q.x <= q.y' in text, text
    s = stage([add])
    text = '\n'.join(line for _, lines, _ in s.blocks for line in lines)
    assert 'scc' not in text and 's24 = s20 + s22;' in text, text
    print('pre-pass: carry emitted only where read (straight, through a branch); not past s_cmp, not unread, '
          'not for calculated-PC arithmetic; an unread add keeps its old line')


def refused(words, why):
    try:
        stage(words)
    except G.Unsupported as error:
        assert why in str(error), (why, str(error))
        return
    raise AssertionError('accepted: ' + why)


def refusals():
    refused([sop2('s_add_i32', 24, 20, 22), sop2('s_addc_u32', 25, 21, 23)],
            'SCC of s_add_i32 at 0x0 (not modelled) is read by s_addc_u32 at 0x4')
    for text, why in (('s_addc_u32 s25, exec_lo, s23', 'not a plain number'),
                      ('s_addc_u32 exec_lo, s21, s23', 'destination exec_lo')):
        try:
            lines_of('s_addc_u32', text)
        except G.Unsupported as error:
            assert why in str(error), (why, str(error))
            continue
        raise AssertionError('accepted ' + text)
    print('refused: s_add_i32 SCC read by s_addc_u32; s_addc_u32 on exec')


if __name__ == '__main__':
    values()
    prepass()
    refusals()
