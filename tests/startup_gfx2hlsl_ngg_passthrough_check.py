"""Check the NGG vertex pass-through lowering in tools/startup/gfx2hlsl.py without shader machine code.

A pass-through NGG vertex shader asks for its allocation, parks per-lane values in LDS, waits at a barrier, reads
them back and exports position, parameters and the primitive word. The lowering drops the LDS round trip (the read
returns the register that was written), keeps the position and parameter exports, and accepts the primitive export
only when it hands on the entry VGPR0 word unchanged. Everything else stays refused.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


class Pipe:
    def resource(self, key, kind):
        return 'r0'

    def mrt_kind(self, index):
        return 'float'


ALLOC = 's_sendmsg 0x9'
BARRIER = 's_barrier 0x0'
PRIM = 'exp target_20 v0, off, off, off done'
POS = 'exp pos0 v4, v5, v6, v7 done'


def run(*instructions, kind='vs', branch=False):
    """Translate disassembly text with the same dispatch and branch scan as a real stage."""
    s = G.Stage.__new__(G.Stage)
    s.pipe, s.lines, s.kind = Pipe(), [], kind
    s.attrs, s.sym, s.exports = {}, {}, {}
    s.used_s, s.used_v, s.masks = set(), set(), set()
    s.exec_known = True
    asm = list(instructions)
    if branch:
        asm = ['s_cbranch_scc0 label_0008'] + asm
    s.ins = [(4 * i, 4, text.split()[0], text) for i, text in enumerate(asm)]
    try:
        s.translate()
    except G.Unsupported as error:
        return str(error)
    return [line for _, lines, _ in s.blocks for line in lines]


ROUND_TRIP = ('v_mov_b32_e32 v1, v0', 'ds_write_b32 v1, v2 offset:16', BARRIER, 'ds_read_b32 v3, v1 offset:16')

got = run(ALLOC, *ROUND_TRIP, PRIM)
check('round trip becomes a register copy', got == ['v1 = v0;', 'v3 = v2;'], repr(got))

got = run(ALLOC, 'ds_write_b32 v1, v2 offset:16', BARRIER, 'ds_read_b32 v3, v1 offset:16', POS, PRIM)
check('position export survives the pass-through', got[-1] == 'o.pos = float4(F(v4), F(v5), F(v6), F(v7));', repr(got))

got = run(ALLOC, 'ds_write_b32 v1, v2 offset:16', 'ds_write_b32 v1, v8 offset:20', BARRIER,
          'ds_read_b32 v3, v1 offset:20', 'ds_read_b32 v9, v1 offset:16', PRIM)
check('slots are kept apart by offset', got == ['v3 = v8;', 'v9 = v2;'], repr(got))

# Controls that must stay refused.
REFUSED = [
    ('primitive word computed from v0', (ALLOC, 'v_add_nc_u32_e32 v0, 1, v0', PRIM), 'non-identity primitive export'),
    ('primitive word from another register', (ALLOC, 'exp target_20 v1, off, off, off done'),
     'non-identity primitive export'),
    ('primitive export with extra channels', (ALLOC, 'exp target_20 v0, v1, off, off done'),
     'non-identity primitive export'),
    ('compressed primitive export', (ALLOC, 'exp target_20 v0, off, off, off compr done'),
     'non-identity primitive export'),
    ('primitive export after a branch', (PRIM,), 'non-identity primitive export', True),
    ('slot read without a write', (ALLOC, 'ds_read_b32 v3, v1 offset:16'), 'did not write'),
    ('other offset read', (ALLOC, 'ds_write_b32 v1, v2 offset:16', 'ds_read_b32 v3, v1 offset:20'), 'did not write'),
    ('address changed after the write',
     (ALLOC, 'ds_write_b32 v1, v2 offset:16', 'v_add_nc_u32_e32 v1, 1, v1', 'ds_read_b32 v3, v1 offset:16'),
     'did not write'),
    ('data changed after the write',
     (ALLOC, 'ds_write_b32 v1, v2 offset:16', 'v_mov_b32_e32 v2, 0', 'ds_read_b32 v3, v1 offset:16'),
     'did not write'),
    ('unknown message', ('s_sendmsg 0x2',), 's_sendmsg'),
    ('LDS write in a pixel shader', ('ds_write_b32 v1, v2 offset:16',), 'straight-line vertex shader', False, 'ps'),
    ('LDS after a branch', ('ds_write_b32 v1, v2 offset:16',), 'straight-line vertex shader', True),
]
for label, instructions, reason, *extra in REFUSED:
    branch = bool(extra and extra[0])
    kind = extra[1] if len(extra) > 1 else 'vs'
    got = run(*instructions, branch=branch, kind=kind)
    check('refused: ' + label, isinstance(got, str) and reason in got, repr(got))

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all ok')
