"""Compressed export with `off` channels in tools/startup/gfx2hlsl.py (issue #1674).

usage: startup_gfx2hlsl_export_off_check.py

`compr` packs two channels into each source register. An export that writes only some channels spells the
unwritten pair `off`, and the translator used to evaluate that token as an operand, refusing the whole
translation with `Unsupported: operand 'off'`. That refusal accounted for the largest single group - 7 of 29 -
of refused shaders in the first main-game loading-screen cache (#870).

The cases start from encoded instructions and go through `decode()`, then the translator, because an
injected-assembly test is exactly what made an earlier, wrong fix for this group look correct.
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
    """Just enough pipeline state for the export path: the MRT format and the flat-parameter offset."""

    def __init__(self, kind='float'):
        self.notes = set()
        self.kind = kind

    def mrt_kind(self, index):
        return self.kind

    def ud(self, source):
        return 0


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self, kind='float'):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline(kind)

    def descriptor(self, tok):
        return ('buffer', tok)


def translate_bytes(hexbytes, kind='float'):
    """Decode the bytes the way the pipeline does, then translate every decoded instruction."""
    decoded = G.decode(bytes.fromhex(hexbytes))
    stage = Stage(kind)
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


# 1. The decoded text is what the rest of this test assumes; a decoder change should show up here first.
asm, emitted = translate_bytes("031c00f800808080")
check('decodes to a compressed export with off channels', asm == ['exp mrt0 v0, v0, off, off done compr vm'],
      repr(asm))

# 2. The off pair contributes zero and the written pair still comes from its register. Channels 0/1 are the
#    two halves of v0; channels 2/3 are the unwritten pair.
check('written pair comes from v0', 'f16tof32(v0)' in emitted and 'f16tof32(v0 >> 16)' in emitted, emitted)
check('unwritten pair contributes zero', 'f16tof32(0u)' in emitted and 'f16tof32(0u >> 16)' in emitted, emitted)
check('export target still written', 'o.mrt0' in emitted, emitted)

# 3. The mirror case: off in the first pair and real registers in the second (a real encoding from the same
#    cache). The fix must handle either pair, not just the one the common case uses.
asm, emitted = translate_bytes("4cccb1f9d1537dd9")
check('decodes to compr with off in the first pair',
      asm == ['exp mrt4 off, off, v83, v83 done compr'], repr(asm))
check('first pair zero, second pair from v83',
      'f16tof32(v83)' in emitted and 'f16tof32(0u)' in emitted, emitted)

# 4. Another real encoding from the same cache: a different MRT target with the same off pair.
asm, emitted = translate_bytes("131c00f81b808080")
check('second MRT target decodes', asm == ['exp mrt1 v27, v27, off, off done compr vm'], repr(asm))
check('second MRT target exports to mrt1', 'o.mrt1' in emitted and 'f16tof32(0u)' in emitted, emitted)


# 5. Refusals that must not widen.
def refused(hexbytes, kind='float'):
    try:
        translate_bytes(hexbytes, kind)
    except G.Unsupported:
        return True
    return False


# A compressed export into an integer MRT is packed-integer data and stays refused.
check('packed-integer compressed export still refused', refused("031c00f800808080", kind='uint'))
# An explicit en mask is a separate form: the export parser does not read it yet, so it stays refused
# loudly rather than exporting all four channels and silently disagreeing with the mask.
check('explicit en mask stays refused (separate slice)', refused("0b1c00f807808080"))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')