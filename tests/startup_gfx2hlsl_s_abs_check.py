"""s_abs_i32 in tools/startup/gfx2hlsl.py (#1723).

usage: startup_gfx2hlsl_s_abs_check.py

RDNA 2 ISA SOP1: S_ABS_I32 sets D to the absolute value of S0 as a signed 32-bit word
(INT_MIN stays 0x80000000) and SCC = (D != 0). Matches pixel_aot. Lane masks and
descriptor sources are refused. The reference is plain Python; the translated text is
evaluated on the CPU.
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
WORDS = (0x00000000, 0x00000001, 0xffffffff, 0x80000000, 0x7fffffff, 0xfffffff1, 0x12345678)
VCC_LO, EXEC_LO = 106, 126

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def sop1(name, sdst, ssrc):
    return struct.pack('<I', (0b101111101 << 23) | (sdst << 16) | (SOP1[name] << 8) | ssrc)


def ref_abs(word):
    word &= MASK
    return (-word if word & 0x80000000 else word) & MASK


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


bad = None
lines, _ = translate(sop1('s_abs_i32', 3, 1))
assert lines and not lines[0].startswith('refused'), lines
for word in WORDS:
    out = run(lines, {'s1': word, 'scc': 0})
    want = (ref_abs(word), int(ref_abs(word) != 0))
    if (out['s3'], int(bool(out['scc']))) != want:
        bad = (hex(word), (out['s3'], out['scc']), want)
        break
check('s_abs_i32 SGPR source', bad is None, str(bad))

bad = None
for word in WORDS:
    stage = Probe()
    try:
        stage.one('s_abs_i32', 's_abs_i32 s3, %d' % (word if word < 0x80000000 else word - (1 << 32)))
    except G.Unsupported as error:
        bad = ('refused %s' % error,)
        break
    out = run(stage.lines, {'scc': 0})
    want = (ref_abs(word), int(ref_abs(word) != 0))
    # Constant fold emits a hex literal and SCC from the folded value.
    if 's3 = 0x' in stage.lines[0]:
        if (out['s3'], int(bool(out['scc']))) != want:
            bad = (hex(word), stage.lines, (out['s3'], out['scc']), want)
            break
    elif (out['s3'], int(bool(out['scc']))) != want:
        bad = (hex(word), stage.lines, (out['s3'], out['scc']), want)
        break
check('s_abs_i32 immediate source', bad is None, str(bad))

lines, sym = translate(sop1('s_abs_i32', 3, 1), sym={1: ('const', 0xfffffff1)})
# 0xfffffff1 = -15; abs = 15
check('s_abs_i32 folds a constant source',
      lines[0].startswith('s3 = 0x0000000fu;') and sym.get(3) == ('const', 15)
      and any(l.startswith('scc = ') for l in lines),
      str((lines, sym.get(3))))

# INT_MIN stays.
lines, sym = translate(sop1('s_abs_i32', 3, 1), sym={1: ('const', 0x80000000)})
check('s_abs_i32 keeps INT_MIN',
      lines[0].startswith('s3 = 0x80000000u;') and sym.get(3) == ('const', 0x80000000),
      str((lines, sym.get(3))))

refused = [translate(sop1('s_abs_i32', d, s))[0][0] for d, s in (
    (3, VCC_LO), (3, EXEC_LO), (VCC_LO, 1), (EXEC_LO, 1))]
check('s_abs_i32 refuses lane-mask destinations/sources',
      all(r.startswith('refused: ') for r in refused), str(refused))

stage = Probe()
stage.masks.add(1)
try:
    stage.one('s_abs_i32', 's_abs_i32 s3, s1')
    check('s_abs_i32 refuses a tracked lane-mask source', False, stage.lines)
except G.Unsupported as error:
    check('s_abs_i32 refuses a tracked lane-mask source', 'lane mask' in str(error), str(error))

stage = Probe()
stage.sym[1] = ('desc', ('root', 0), 0, 0)
try:
    stage.one('s_abs_i32', 's_abs_i32 s3, s1')
    check('s_abs_i32 refuses a descriptor source', False, stage.lines)
except G.Unsupported as error:
    check('s_abs_i32 refuses a descriptor source', 'desc' in str(error), str(error))

# Negative control: drop the abs; evaluation must diverge on a negative word.
stage = Probe()
stage.one('s_abs_i32', 's_abs_i32 s3, s1')
mutated = [re.sub(r'\(asint\(s1\) < 0 \? \(0u - s1\) : s1\)', 's1', stage.lines[0])]
assert mutated != stage.lines, stage.lines
out_ok = run(stage.lines, {'s1': 0xfffffff1, 'scc': 0})
out_bad = run(mutated + stage.lines[1:], {'s1': 0xfffffff1, 'scc': 0})
check('mutation that drops abs diverges',
      out_ok['s3'] == 15 and out_bad['s3'] == 0xfffffff1,
      str((out_ok['s3'], out_bad['s3'])))

if failures:
    print('FAILED: %s' % ', '.join(failures))
    sys.exit(1)
print('all ok')
