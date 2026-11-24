"""SMEM 'off' offset operand in tools/startup/gfx2hlsl.py (#1674): s_buffer_load_* and s_load_dword*
accept 'off' as the zero byte offset, the spelling with no m0 register and no immediate at all.

usage: startup_gfx2hlsl_smem_off_check.py

The probe feeds source text to Stage.one, as startup_gfx2hlsl_bit_alu_check.py does, so no binary
fixture is needed. Before the fix 'off' was refused: 'operand 'off'' for the buffer forms and the
misleading 's_load with a register offset through ...' for the scalar loads.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    """Enough pipeline state for the SMEM loads: the resource registry and the flags they read."""

    def __init__(self):
        self.resources = {}
        self.modified = set()
        self.notes = set()
        self.sampler_heap = False
        self.uses_sampler_heap = False

    def resource(self, key, kind):
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise G.Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']


class Probe(G.Stage):
    """A Stage with no real pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.pipe = Pipe()
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        # s[4:5] holds a root-table pointer, the state a bound descriptor read leaves.
        self.sym[4] = ('ptr', ('root', 0))
        self.sym[5] = ('ptrhi', ('root', 0))


def run(assembly):
    probe = Probe()
    probe.one(assembly.split(' ', 1)[0], assembly)
    return probe.lines


def refused(assembly):
    try:
        run(assembly)
    except G.Unsupported:
        return True
    return False


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


# 'off' is the SMEM encoding of "no offset": a zero byte offset, not a register.
check('s_buffer_load_dword off loads word 0',
      run('s_buffer_load_dword s0, s[4:7], off') == ['s0 = r0.Load(0u + 0);'])
check('s_buffer_load_dwordx2 off loads words 0 and 4',
      run('s_buffer_load_dwordx2 s[2:3], s[4:7], off') == ['s2 = r0.Load(0u + 0);', 's3 = r0.Load(0u + 4);'])
check('s_load_dword off loads at offset 0',
      run('s_load_dword s0, s[4:5], off') == ['s0 = r0.Load(0);'])
check('s_load_dwordx2 off loads at offsets 0 and 4',
      run('s_load_dwordx2 s[2:3], s[4:5], off') == ['s2 = r0.Load(0);', 's3 = r0.Load(4);'])

# Existing forms keep their behaviour: a literal, m0 as a buffer register offset and a root-table
# scalar load read the same as before.
check('literal buffer offset unchanged',
      run('s_buffer_load_dword s0, s[4:7], 0x4') == ['s0 = r0.Load(0x00000004u + 0);'])
check('m0 keeps the buffer register offset',
      run('s_buffer_load_dword s0, s[4:7], m0') == ['s0 = r0.Load(m0 + 0);'])
check('literal s_load offset unchanged',
      run('s_load_dword s0, s[4:5], 8') == ['s0 = r0.Load(8);'])

# Both refusal kinds stay refusals: a register offset through s_load and an operand that is none of
# literal, m0 or off.
check('s_load register offset refused', refused('s_load_dword s0, s[4:5], s2'))
check('s_load m0 refused', refused('s_load_dword s0, s[4:5], m0'))
check('unknown buffer offset refused', refused('s_buffer_load_dword s0, s[4:7], bogus'))
check('unknown s_load offset refused', refused('s_load_dword s0, s[4:5], bogus'))

print('startup_gfx2hlsl_smem_off_check passed')
