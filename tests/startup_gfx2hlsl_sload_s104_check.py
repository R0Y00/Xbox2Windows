"""Check s_load into s[104:107] in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_sload_s104_check.py

s106 and s107 are vcc. A descriptor load (a table pointer plus an immediate offset) emits no code and only
records which table bytes the destination registers name, so it may end in s[104:107]. A load that must
write the registers (a root constant buffer) or that starts above s105 is still refused.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    sampler_heap = False

    def resource(self, key, kind):
        return 'r0'


def run(symbols, op, operands):
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.sym, s.lines = Pipe(), dict(symbols), []
    s.used_s, s.used_v, s.uniform_v, s.exec_known = set(), set(), set(), True
    try:
        getattr(s, 'op_' + op)(operands.split(', '), [])
    except G.Unsupported as error:
        return str(error)
    return s


TABLE = {4: ('ptr', ('table', 0))}
ROOT = {4: ('ptr', ('root', 1))}


def main():
    s = run(TABLE, 's_load_dwordx4', 's[104:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    assert [s.sym[104 + i] for i in range(4)] == [('desc', ('table', 0), 0x20, i) for i in range(4)], s.sym
    assert s.lines == [], s.lines
    s = run(TABLE, 's_load_dwordx8', 's[100:107], s[4:5], 0x20')
    assert not isinstance(s, str), s
    for bad, symbols in (('s_load_dwordx4 s[104:107], s[4:5], 0x20', ROOT),
                         ('s_load_dwordx2 s[106:107], s[4:5], 0x20', TABLE),
                         ('s_load_dwordx4 s[105:108], s[4:5], 0x20', TABLE)):
        op, operands = bad.split(' ', 1)
        message = run(symbols, op, operands)
        assert isinstance(message, str) and 's_load destination' in message, (bad, message)
    print('ok')


main()
