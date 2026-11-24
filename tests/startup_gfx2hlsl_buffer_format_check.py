"""Untyped-format buffer loads in tools/startup/gfx2hlsl.py (issue #1809).

usage: startup_gfx2hlsl_buffer_format_check.py

`buffer_load_format_*` (MUBUF) has no format field: the element format is the DATA_FORMAT of the resource
descriptor, which the translator does not track (`Stage.descriptor()` records only where the words came from).
Choosing a format from the destination width would load silently wrong data, so each form must be refused by
name, with the reason, while `tbuffer_load_format_*` (format in the instruction) keeps translating.

The instruction bytes are constructed here from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, the MUBUF branch); no game-derived bytes are used.
"""
import os
import struct
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


# MUBUF: w0 [31:26]=0b111000, [25:18] opcode, [15:0] flags and offset; w1 as in MTBUF (soffset 128 = 0).
MUBUF_LOAD_FORMAT = {'x': 0, 'xy': 1, 'xyz': 2, 'xyzw': 3}
MUBUF_LOAD_DWORD = 12


def encode_mubuf(op, *, vdata=6, srsrc=32, soffset=128):
    w0 = (0b111000 << 26) | (op << 18)
    w1 = ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16) | ((soffset & 0xff) << 24)
    return struct.pack('<II', w0, w1)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']

    def ud(self, source):
        return 0


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()

    def descriptor(self, tok):
        return ('buffer', tok)


def translate(code):
    stage = Stage()
    decoded = G.decode(code)
    for _o, _s, name, asm in decoded:
        stage.one(name, asm)
    return [a for _o, _s, _n, a in decoded], stage.lines


# 1. Every width is refused by its own name and says why; nothing is emitted.
for suffix, op in MUBUF_LOAD_FORMAT.items():
    code = encode_mubuf(op)
    try:
        asm, lines = translate(code)
        check('refuses buffer_load_format_' + suffix, False, 'translated: %r' % lines)
    except G.Unsupported as e:
        text = str(e)
        check('refuses buffer_load_format_' + suffix, text.startswith('buffer_load_format_%s ' % suffix), text)
        check('refusal names DATA_FORMAT for ' + suffix, 'DATA_FORMAT' in text, text)

# 2. The census form decodes as written in the issue.
asm, _ = (lambda d: ([a for _o, _s, _n, a in d], None))(G.decode(encode_mubuf(MUBUF_LOAD_FORMAT['xyz'])))
check('census form decodes', asm == ['buffer_load_format_xyz v[6:8], off, s[32:35], 0'], repr(asm))

# 3. Plain dword loads are unaffected: they read raw dwords and need no format.
asm, lines = translate(encode_mubuf(MUBUF_LOAD_DWORD))
check('buffer_load_dword still translates', any('r0.Load(a' in l for l in lines), repr(lines))

if failures:
    print('FAIL')
    for t in failures:
        print('  ' + t)
    sys.exit(1)
print('PASS')
