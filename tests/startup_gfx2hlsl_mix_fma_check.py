"""v_fma_mix_f32 under X2W_DRAW_FEATURES=mixfma (tools/startup/gfx2hlsl.py Stage.mix).

usage: startup_gfx2hlsl_mix_fma_check.py [fxc]

v_fma_mix_f32 rounds once. The plain lowering, a * b + c, rounds the product and lets fxc reassociate a chain of
them. With the feature, a draw stage lowers v_fma_mix_f32 through x2w_fma_exact (the double fma behind the x2w_fz guard), except
that a signed-zero addend goes through x2w_mad_precise: fxc folds that zero into a d(...) immediate even behind the
guard, and fma(a, b, +-0) is the product rounded once plus that zero, which the precise multiply and add give.

Pure Python: the emitted statements with and without the feature, the f16 results (mixlo/mixhi) unchanged, mesh and
compute pipelines unchanged.
(Why +-0 is exact: round(a * b) is a binary32 value; adding a signed zero to a nonzero value is exact, and for a zero
product IEEE 754 6.3 gives the same signed zero either way.) With fxc given: a vs_5_1 stage with every operand shape the
draw uses compiles with no d(...) immediate (the control, x2w_fma_exact with the -0.0 addend, has one).
"""
import os
import re
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + str(detail)[:1500])


class Pipe:
    def __init__(self, features):
        self.features = frozenset(features)
        self.fma_rule = None


def lower(line, features, pipe=None):
    stage = Probe()
    stage.pipe = pipe or Pipe(features)
    stage.one(line.split(' ', 1)[0], line)
    return '\n'.join(stage.lines), stage.pipe


GENERAL = 'v_fma_mix_f32 v4, v1, v2, v3 op_sel_hi:[0,1,0]'
ONE = 'v_fma_mix_f32 v4, 1.0, v2, v3 op_sel_hi:[0,1,0]'
NEGZERO = 'v_fma_mix_f32 v4, v1, v2, -0 op_sel:[0,1,0] op_sel_hi:[0,1,0]'

if G.DRAW_FMA != 'exact':
    sys.exit('this check expects the default draw FMA rule (exact), got %s' % G.DRAW_FMA)
for line in (GENERAL, ONE):
    off, _ = lower(line, ())
    on, pipe = lower(line, ('mixfma',))
    check('%s: plain without the feature' % line, 'x2w_fma' not in off and ' * ' in off, off)
    check('%s: x2w_fma_exact with the feature' % line, re.search(r'v4 = U\(x2w_fma_exact\(', on) is not None, on)
    check('%s: the feature sets the manifest fma rule' % line, pipe.fma_rule == 'exact', pipe.fma_rule)
on, _ = lower(NEGZERO, ('mixfma',))
check('signed-zero addend goes through x2w_mad_precise',
      'x2w_mad_precise(F(v1), f16tof32(v2 >> 16), (-0.0))' in on, on)
for line in ('v_fma_mixlo_f16 v4, v1, v2, v3 op_sel_hi:[0,1,0]', 'v_fma_mixhi_f16 v4, v1, v2, v3 op_sel_hi:[0,1,0]'):
    on, _ = lower(line, ('mixfma',))
    check('%s keeps the plain form' % line.split()[0], 'x2w_fma' not in on and 'x2w_mad' not in on, on)
mesh = G.MeshPipeline.__new__(G.MeshPipeline)
mesh.features = frozenset(('mixfma',))
on, _ = lower(GENERAL, (), pipe=mesh)
check('mesh pipeline keeps the plain form', 'x2w_fma' not in on, on)
# v_fma_f32 with a zero constant operand (corpus 09f13eef69a075bb: x2w_fma_exact(F(0x00000000u), ...) compiled to
# dfma d(0, 0), ...): x2w_mad_precise, which is exact there; a nonzero constant keeps x2w_fma_exact.
for line, helper in (('v_fma_f32 v4, 0, v2, v3', 'x2w_mad_precise'), ('v_fma_f32 v4, v1, v2, 0', 'x2w_fma_exact'), ('v_fma_f32 v4, v1, v2, 0x80000000', 'x2w_mad_precise'),
                     ('v_fma_f32 v4, v1, -0, v3', 'x2w_mad_precise'), ('v_fma_f32 v4, v1, 0x80000000, v3', 'x2w_mad_precise'),
                     ('v_fma_f32 v4, 2.0, v2, -1.0', 'x2w_fma_exact'), ('v_fma_f32 v4, v1, 0x00000001, v3', 'x2w_fma_exact')):
    on, _ = lower(line, ())
    check('%s -> %s' % (line, helper), helper + '(' in on, on)
for text, zero in (('0.0', True), ('(-0.0)', True), ('F(0x00000000u)', True), ('(-abs(F(0x80000000u)))', True),
                   ('F(0x00000001u)', False), ('F(v0)', False), ('1.0', False)):
    check('zero_operand(%s) is %s' % (text, zero), G.zero_operand(text) == zero)
check('unknown feature names still refuse', 'mixfma' in G.KNOWN_DRAW_FEATURES)

if len(sys.argv) > 1:
    shapes = ['x2w_fma_exact(F(a), F(b), F(c))', 'x2w_fma_exact(1.0, f16tof32(b), F(c))',
              'x2w_fma_exact(1.0, (-f16tof32(b)), (-F(c)))', 'x2w_mad_precise(F(a), f16tof32(b >> 16), (-0.0))',
              'x2w_mad_precise(F(0x00000000u), F(b), F(c))']
    control = 'x2w_fma_exact(F(a), f16tof32(b >> 16), (-0.0))'
    helpers = G.FMA_EXACT_FXC + '\n' + G.HELPERS['x2w_mad_precise']

    def immediates(exprs):
        body = ' '.join('r += asuint(%s);' % e for e in exprs)
        src = ('#define F(x) asfloat(x)\n%s\ncbuffer cb : register(b0) { uint4 q; };\n'
               'uint4 main(uint vid : SV_VertexID) : P { x2w_fz = (vid * vid + vid) & 1u; uint a = q.x + vid, '
               'b = q.y ^ vid, c = q.z * vid, r = 0; %s return uint4(r, a, b, c); }\n' % (helpers, body))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mix.hlsl')
            open(path, 'w').write(src)
            r = subprocess.run([sys.argv[1], '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', path + '.cso', path],
                               capture_output=True, text=True, timeout=120)
            if r.returncode:
                return None, r.stdout + r.stderr
            d = subprocess.run([sys.argv[1], '/nologo', '/dumpbin', path + '.cso'], capture_output=True, text=True)
            return len([l for l in d.stdout.splitlines() if not l.lstrip().startswith('//') and 'd(' in l]), d.stdout
    n, text = immediates(shapes)
    check('fxc vs_5_1: the lowered shapes compile with no d(...) immediate', n == 0, text)
    n, text = immediates([control])
    check('control: x2w_fma_exact with a -0.0 addend gets a d(...) immediate', n is not None and n > 0, text)

if failures:
    print('FAIL')
    sys.exit(1)
print('PASS')
