"""Four-channel half-float typed buffer loads in tools/startup/gfx2hlsl.py (issue #1675).

usage: startup_gfx2hlsl_tbuffer_16f_check.py

`BUF_FMT_16_16_16_16_FLOAT` stores four binary16 channels in one 8-byte element: x and y in the first dword,
z and w in the second. The format table had the two-channel form only, so a typed load of this format was
refused with `buffer format BUF_FMT_16_16_16_16_FLOAT`; that refusal accounted for 5 of the 29 refused shaders
in the first main-game loading-screen cache (#870).

The cases start from encoded instructions and go through `decode()`, then the translator.
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
    """Just enough pipeline state for a typed buffer load."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']

    def ud(self, source):
        return 0


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()

    def descriptor(self, tok):
        return ('buffer', tok)


def translate_bytes(hexbytes):
    """Decode the bytes the way the pipeline does, then translate every decoded instruction."""
    decoded = G.decode(bytes.fromhex(hexbytes))
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


def refused(hexbytes):
    try:
        translate_bytes(hexbytes)
    except G.Unsupported:
        return True
    return False


# 1. The decoded text is what the rest of this test assumes; a decoder change shows up here first.
#    00203aea00000580 is a real encoding from the loading-screen cache.
asm, emitted = translate_bytes("00203aea00000580")
check('decodes to a typed xyz load of the four-half format',
      asm == ['tbuffer_load_format_xyz v[0:2], v0, s[20:23], 0 format:[BUF_FMT_16_16_16_16_FLOAT] idxen'],
      repr(asm))

# 2. x and y come from the first dword, z from the second. All three are converted from binary16.
check('x from the first dword', 'v0 = U(f16tof32(r0.Load(a)))' in emitted, emitted)
check('y from the high half of the first dword', 'v1 = U(f16tof32(r0.Load(a) >> 16))' in emitted, emitted)
check('z from the second dword', 'v2 = U(f16tof32(r0.Load(a + 4)))' in emitted, emitted)

# 3. The four-channel form also lowers, and w comes from the high half of the second dword.
asm, emitted = translate_bytes("00203bea00000580")
check('decodes to the xyzw form',
      asm == ['tbuffer_load_format_xyzw v[0:3], v0, s[20:23], 0 format:[BUF_FMT_16_16_16_16_FLOAT] idxen'],
      repr(asm))
check('w from the high half of the second dword', 'v3 = U(f16tof32(r0.Load(a + 4) >> 16))' in emitted, emitted)

# 4. The one-channel form lowers and reads only the first half.
asm, emitted = translate_bytes("002038ea00000580")
check('decodes to the x form', asm[0].startswith('tbuffer_load_format_x '), repr(asm))
check('x form reads one channel', 'v0 = U(f16tof32(r0.Load(a)))' in emitted and 'v1 =' not in emitted, emitted)


# 5. Refusals that must not widen. A numeric format the table does not name stays refused, and the d16 form
#    (which the translator has no lowering for at all) stays refused.
check('an unnamed numeric format stays refused', refused("00207aea00000580"))
check('the d16 form stays refused', refused("00203aea00002580"))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')