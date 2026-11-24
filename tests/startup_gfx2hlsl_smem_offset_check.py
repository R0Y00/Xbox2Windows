"""SMEM offset operand `off` in tools/startup/gfx2hlsl.py (issue #1674).

usage: startup_gfx2hlsl_smem_offset_check.py

The SMEM offset operand is a literal, m0 (a register offset) or `off`. `off` is the form with no m0 offset
at all, so its byte offset is zero; the disassembler emits it and the translator used to pass it to the uint
operand helper, which refused the whole translation with `Unsupported: operand 'off'`. That refusal accounted
for the largest group (7 of 29) of refused shaders in the first main-game loading-screen cache.

This checks both decoded SMEM forms - `s_load_dword*` and `s_buffer_load_dword*` - plus the controls that the
literal and the register forms keep their existing behaviour.
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


class Pipeline:
    """Just enough pipeline state for the SMEM operand helpers."""

    def __init__(self):
        self.resources, self.notes, self.sampler_heap = {}, set(), False

    def resource(self, key, kind):
        return 'r%d' % self.resources.setdefault(key, len(self.resources))

    def ud(self, source):
        return 0


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.pipe = Pipeline()

    def descriptor(self, tok):
        return ('buffer', tok)


def translate(assembly, symbols=None):
    stage = Probe()
    for register, value in (symbols or {}).items():
        stage.sym[register] = value
    stage.one(assembly.split(' ', 1)[0], assembly)
    return ' '.join(stage.lines)


def refused(assembly, symbols=None):
    try:
        translate(assembly, symbols)
    except G.Unsupported:
        return True
    return False


# 1. `off` is a zero byte offset, not an unknown operand.
emitted = translate('s_buffer_load_dword s0, s[4:7], off')
check('s_buffer_load off emits offset 0', 'r0.Load(0u + 0)' in emitted, emitted)

emitted = translate('s_buffer_load_dwordx2 s[2:3], s[4:7], off')
check('s_buffer_loadx2 off loads 0 and 4', 'r0.Load(0u + 0)' in emitted and 'r0.Load(0u + 4)' in emitted, emitted)

emitted = translate('s_load_dword s0, s[4:5], off', {4: ('ptr', ('root',))})
check('s_load off emits offset 0', 'r0.Load(0)' in emitted, emitted)

# 2. The literal and register forms are unchanged.
emitted = translate('s_buffer_load_dword s0, s[4:7], 0x4')
check('s_buffer_load literal offset unchanged', 'r0.Load(0x00000004u + 0)' in emitted, emitted)

emitted = translate('s_load_dword s0, s[4:5], 0x8', {4: ('ptr', ('root',))})
check('s_load literal offset unchanged', 'r0.Load(8)' in emitted, emitted)

# 3. m0 stays a register offset for the s_buffer form and stays refused for s_load, as before.
emitted = translate('s_buffer_load_dword s0, s[4:7], m0')
check('s_buffer_load m0 still a register offset', 'm0' in emitted, emitted)
check('s_load register offset still refused', refused('s_load_dword s0, s[4:5], m0', {4: ('ptr', ('root',))}))

# 4. An operand that is neither a literal, m0 nor off is still refused rather than treated as zero.
check('unknown s_buffer_load operand refused', refused('s_buffer_load_dword s0, s[4:7], bogus'))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')