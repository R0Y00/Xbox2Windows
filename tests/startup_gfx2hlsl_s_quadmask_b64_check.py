"""s_quadmask_b64 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_s_quadmask_b64_check.py

RDNA 2 ISA SOP1: each output bit is the OR of one four-bit group of the 64-bit
source. Sixteen nibbles collapse into bits 0..15 of D.lo; D.hi is 0.
SCC = (D.lo | D.hi) != 0. Matches pixel_aot. Negative integer inline constants
extend to 64 bits and 32-bit literals zero-extend. Lane masks and descriptor sources are refused.
The reference is plain Python; the translated text is evaluated on the CPU.
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Expr  # noqa: E402

MASK = 0xffffffff
SOP1 = {name: op for op, (name, _) in D.SOP1_OPS.items()}
# (lo, hi) pairs: null, low nibble, high bit of low word, high-word nibble, both halves, all ones.
PAIRS = (
    (0x00000000, 0x00000000),
    (0x0000000f, 0x00000000),
    (0x80000000, 0x00000000),
    (0x00000000, 0x0000000f),
    (0x000f0000, 0x00f00000),
    (0xffffffff, 0xffffffff),
    (0x12345678, 0x9abcdef0),
    (0xf0f0f0f0, 0x0f0f0f0f),
)
VCC_LO, EXEC_LO = 106, 126

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc, literal=None):
    words = struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)
    if literal is not None:
        words += struct.pack('<I', literal & MASK)
    return words


def quadmask32(word):
    return sum((((word >> (4 * group)) & 0xf) != 0) << group for group in range(8))


def quadmask64(lo, hi):
    return quadmask32(lo) | (quadmask32(hi) << 8)


class Probe(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.masks = set()
        self.exec_known = True


def translate(words, masks=(), sym=None):
    decoded = G.decode(words)
    assert len(decoded) == 1 and decoded[0][1] == len(words), decoded
    stage = Probe()
    stage.masks = set(masks)
    if sym:
        stage.sym.update(sym)
    try:
        stage.one(decoded[0][2], decoded[0][3])
    except G.Unsupported as error:
        stage.lines = ['refused: %s' % error]
    return stage.lines, stage.sym


def run(lines, env):
    env = dict(env)
    for line in lines:
        m = re.fullmatch(r'(\w+) = (.*);', line)
        assert m, line
        env[m.group(1)] = Expr(m.group(2)).run(env)[0]
    return env


def controls():
    bad = None
    lines, _ = translate(sop1('s_quadmask_b64', 4, 2))
    assert lines and not lines[0].startswith('refused'), lines
    for lo, hi in PAIRS:
        out = run(lines, {'s2': lo, 's3': hi, 'scc': 0})
        want_lo = quadmask64(lo, hi)
        want = (want_lo, 0, int(want_lo != 0))
        if (out['s4'], out['s5'], int(bool(out['scc']))) != want:
            bad = ((hex(lo), hex(hi)), (out['s4'], out['s5'], out['scc']), want)
            break
    check('s_quadmask_b64 SGPR pair source', bad is None, str(bad))

    bad = None
    for word in (0x00000000, 0x0000000f, 0x80000000, 0x000f0000, 0xffffffff, 0x12345678):
        lines, _ = translate(sop1('s_quadmask_b64', 4, 255, word))
        if lines[0].startswith('refused'):
            bad = ('refused %s for %s' % (lines[0], hex(word)),)
            break
        out = run(lines, {'scc': 0})
        want_lo = quadmask64(word, 0)
        if (out['s4'], out['s5'], int(bool(out['scc']))) != (want_lo, 0, int(want_lo != 0)):
            bad = (hex(word), (out['s4'], out['s5'], out['scc']), (want_lo, 0, int(want_lo != 0)))
            break
    check('s_quadmask_b64 zero-extended literal', bad is None, str(bad))

    # SSRC 193 and 208 decode as the inline integers -1 and -16. The same
    # low DWORD encoded as a literal must leave the upper eight result bits clear.
    for ssrc, want in ((193, 0xffff), (208, 0xfffe)):
        lines, _ = translate(sop1('s_quadmask_b64', 4, ssrc))
        out = run(lines, {'scc': 0})
        actual = (out['s4'], out['s5'], int(bool(out['scc'])))
        check('s_quadmask_b64 negative inline SSRC %d' % ssrc, actual == (want, 0, 1), str(actual))

    lines, sym = translate(sop1('s_quadmask_b64', 4, 2),
                           sym={2: ('const', 0x000f0000), 3: ('const', 0x00f00000)})
    # nibble 4 of lo -> bit 4; nibble 5 of hi -> bit 8+5 = bit 13
    want = quadmask64(0x000f0000, 0x00f00000)
    check('s_quadmask_b64 folds a constant pair',
          lines[0].startswith('s4 = 0x%08xu;' % want)
          and any(line.startswith('s5 = 0u;') or line.startswith('s5 = 0x00000000u;') for line in lines)
          and sym.get(4) == ('const', want)
          and sym.get(5) == ('const', 0),
          str((lines, sym.get(4), sym.get(5), want)))

    refused = []
    for words, masks, sym in (
            (sop1('s_quadmask_b64', VCC_LO, 2), (), None),
            (sop1('s_quadmask_b64', 4, VCC_LO), (), None),
            (sop1('s_quadmask_b64', 4, 2), (2,), None),
            (sop1('s_quadmask_b64', 4, 2), (), {2: ('desc', 0, 0, 0)}),
            (sop1('s_quadmask_b64', 3, 2), (), None),  # odd destination
            (sop1('s_quadmask_b64', 4, 1), (), None),  # odd source
    ):
        lines, _ = translate(words, masks=masks, sym=sym)
        refused.append(lines[0])
    check('s_quadmask_b64 refuses lane masks, descriptors, and odd pairs',
          all(r.startswith('refused:') for r in refused), str(refused))

    # Negative control: leave the high-word nibble bits unshifted (<< 0 instead of << 8).
    lines, _ = translate(sop1('s_quadmask_b64', 4, 2))
    mutated = [lines[0].replace('<< 8u', '<< 0u', 1)] + lines[1:]
    diverged = False
    for lo, hi in ((0x00000000, 0x0000000f), (0x000f0000, 0x00f00000)):
        good = run(lines, {'s2': lo, 's3': hi, 'scc': 0})['s4']
        bad_out = run(mutated, {'s2': lo, 's3': hi, 'scc': 0})['s4']
        if good != bad_out:
            diverged = True
            break
    check('mutation that drops the high-word shift diverges', diverged, str(mutated[0]))


if __name__ == '__main__':
    controls()
    if failures:
        print('FAILED: %s' % ', '.join(failures))
        sys.exit(1)
    print('all ok')
