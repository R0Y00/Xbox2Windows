"""Exact FMA (x2w_fma_exact) in fxc-compiled draw stages: no double immediates in the bytecode.

  (no arguments)                       translation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> <fxc>
                                       compile probe pixel stages with FXC ps_5_1 and DXC ps_6_0, check the FXC
                                       bytecode for d(...) immediates and draw one pixel per probe

Defect (measured on WARP and on an RX 9070 XT, driver 32.0.32015.7029): when fxc 10.0.26100 folds a constant
operand of a double op into a d(a, b) immediate and the instruction writes .zw, the second half of the immediate
is not what the GPU uses. Two calls packed into one register then lose the second result:
`dfma r0.zw, r1.xyxy, r1.zwzw, d(0.0, 1.5)` adds 0.0, and an all-constant `dfma r0.xyzw` returns 0 in .zw.
gfx2hlsl's draw stages therefore XOR every helper operand with x2w_fz, a static that main() sets to p * p + p & 1
of an input word (always 0; fxc does not fold it), so the double ops only ever read registers (FMA_EXACT_FXC).

Probes: four helper calls written to mrt0.xyzw, so two land in .zw halves, in both orders (fwd / rev), with
operands that are literals ('c'), run-time values ('rt': z + literal, z = 0.0 from root constant 1), or run-time
a, b with a literal c ('rtc'), on two value sets (plain values and rounding cases). Expected words: the exact
rational a * b + c rounded to binary64, then to binary32 (the helper's definition), computed here in Python.
Every FXC and DXC result must match it bit for bit, and the FXC bytecode must have no d(...) immediate.
Negative control: the plain helper (HELPERS['x2w_fma_exact'], the form before the guard) must fail the four
literal probes and a run-time probe (a, b from z, literal c, second call first) under FXC and show the immediates.
"""
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_draw_fma_check as D  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS  # noqa: E402
from startup_sampler_heap_check import root_blob  # noqa: E402


def compile_hlsl(compiler, kind, source, out, profile, directory):
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    target = directory / ('%s.cso' % out)
    if kind == 'fxc':
        args = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(target), str(path)]
    else:
        args = [compiler, '-T', profile, '-E', 'main', '-Fo', str(target), str(path)]
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
    return target


def fma_pipeline():
    """startup_draw_fma_check's vertex stage with v_fma_f32 v4, v1, v2, v3 (exact FMA) and its plain pixel stage."""
    words = list(D.PROLOGUE) + [w for w in D.vop3(0x14b, 4, 1, 2, 3)] + list(D.EXPORT)
    return G.Pipeline(struct.pack('<%dI' % len(words), *words), D.PS, root_blob(flags=0))

HEAD = '''#define F(x) asfloat(x)
#define U(x) asuint(x)
%s
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; uint ud3; };
struct PSIn { float4 pos : SV_Position; };
struct PSOut { float4 mrt0 : SV_Target0; };
PSOut main(PSIn i) {
    PSOut o = (PSOut)0;
%s
    float z = F(ud1);
%s
    return o;
}
'''
SETS = {
    'plain': [(0.25, 0.5, 1.5), (-0.25, 0.5, 1.5), (3.0, -0.75, 0.125), (1.0e-3, 7.0, -1.0)],
    'rounding': [(1.0 + 2 ** -23, 1.0 - 2 ** -24, -1.0), (16777215.0, 3.0, 1.0), (0.1, 0.2, 0.3), (-1.5, 2 ** -20, 2 ** 30)],
}


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def expect(a, b, c):
    a, b, c = f32(a), f32(b), f32(c)
    return f32(float(Fraction(a) * Fraction(b) + Fraction(c)))  # float(Fraction) rounds once to binary64


def lit(x):
    return 'F(0x%08xu)' % bits(f32(x))


def body(quads, mode, order):
    lines = []
    for k, (a, b, c) in enumerate(quads):
        run_ab, run_c = mode in ('rt', 'rtc'), mode == 'rt'
        lines.append('    float r%d = x2w_fma_exact(%s, %s, %s);' % (
            k, '(z + %s)' % lit(a) if run_ab else lit(a), '(z + %s)' % lit(b) if run_ab else lit(b),
            '(z + %s)' % lit(c) if run_c else lit(c)))
    if order == 'rev':
        lines.reverse()
    return '\n'.join(lines + ['    o.mrt0 = float4(r0, r1, r2, r3);'])


class _Guarded:
    def fma_guard(self):
        return True


def shader(guarded, text):
    helper = G.FMA_EXACT_FXC if guarded else G.HELPERS['x2w_fma_exact']
    init = '\n'.join(G.Pipeline.fma_guard_init(_Guarded(), 'asuint(i.pos.x)')) if guarded else ''
    return HEAD % (helper, init, text)


def controls():
    saved = G.DRAW_FMA
    G.DRAW_FMA = 'exact'
    try:
        pipe = fma_pipeline()
        vs, ps = pipe.hlsl_vs(), pipe.hlsl_ps()
        mesh = fma_pipeline()
        mesh.hlsl_ms = None  # a mesh pipeline (dxc, its mesh stage has no main() here to set x2w_fz)
        mesh_common = '\n'.join(mesh.common())
        G.DRAW_FMA = 'legacy'
        plain_vs, plain_ps = fma_pipeline().hlsl_vs(), fma_pipeline().hlsl_ps()
    finally:
        G.DRAW_FMA = saved
    vinit = '    x2w_fz = (vid * vid + vid) & 1u;  // always 0, see FMA_EXACT_FXC'
    assert 'v4 = U(x2w_fma_exact(F(v1), F(v2), F(v3)));' in vs, vs
    assert G.FMA_EXACT_FXC in vs and G.HELPERS['x2w_fma_exact'] not in vs and vs.count(vinit) == 1, vs
    # The fixture reads v3 at entry, so its stage also takes SV_InstanceID (startup_gfx2hlsl_vs_instance_id_check.py).
    assert vs.index('VSOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {') < vs.index(vinit) < \
        vs.index('return body(vid, iid);'), vs
    # Helpers are emitted for the pipeline, so the pixel stage declares the guard too and its main() sets it.
    pinit = '    x2w_fz = (asuint(i.pos.x) * asuint(i.pos.x) + asuint(i.pos.x)) & 1u;  // always 0, see FMA_EXACT_FXC'
    assert G.FMA_EXACT_FXC in ps and ps.count(pinit) == 1, ps
    assert not mesh.fma_guard() and G.FMA_EXACT_FXC not in mesh_common and G.HELPERS['x2w_fma_exact'] in mesh_common
    assert 'x2w_fz' not in plain_vs + plain_ps and 'x2w_fma_exact' not in plain_vs + plain_ps
    # The guard word is 0 for every input word (p (p + 1) is even, also mod 2^32).
    for p in (0, 1, 2, 3, 0x3f000000, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff):
        assert (p * p + p) & 0xffffffff & 1 == 0
    for name, quads in SETS.items():
        for q in quads:
            assert expect(*q) == f32(expect(*q))
    print('x2w_fma_exact: draw stages use FMA_EXACT_FXC with the x2w_fz init in main(); mesh and legacy keep none')


def double_immediates(fxc, cso):
    out = subprocess.run([fxc, '/nologo', '/dumpbin', str(cso)], capture_output=True, text=True, timeout=60).stdout
    return [l.strip() for l in out.splitlines() if not l.lstrip().startswith('//') and re.search(r'\bd\(', l)]


def draw(renderer, adapter, vs, ps, want):
    hexw = ','.join('%08x' % bits(w) for w in want)
    run = subprocess.run([renderer, adapter, '--startup-image-3d', str(vs), str(ps), '0', hexw],
                         capture_output=True, text=True, timeout=90)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(77)
    if run.returncode == 0:
        return None
    m = re.search(r'actual=([0-9a-f,]+)', run.stderr)
    if not m:
        raise AssertionError(run.stdout + run.stderr)
    return m.group(1)


def check_draw(dxc, renderer, adapter, fxc):
    failed_control = 0
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        vss = {'fxc': compile_hlsl(fxc, 'fxc', FULLSCREEN_VS, 'vs5', 'vs_5_1', d),
               'dxc': compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs6', 'vs_6_0', d)}
        for name, quads in SETS.items():
            want = [expect(*q) for q in quads]
            for mode in ('c', 'rt', 'rtc'):
                for order in ('fwd', 'rev'):
                    for guarded in (True, False):
                        text = shader(guarded, body(quads, mode, order))
                        for kind, comp, prof in (('fxc', fxc, 'ps_5_1'), ('dxc', dxc, 'ps_6_0')):
                            label = '%s %s %s %s %s' % (name, mode, order, kind, 'guard' if guarded else 'plain')
                            ps = compile_hlsl(comp, kind, text, 'p', prof, d)
                            imm = double_immediates(fxc, ps) if kind == 'fxc' else []
                            bad = draw(renderer, adapter, vss[kind], ps, want)
                            if guarded or kind == 'dxc':
                                assert not imm, (label, imm)
                                assert bad is None, (label, bad, ['%08x' % bits(w) for w in want])
                                print('%-40s ok' % label, flush=True)
                            elif mode == 'c':
                                # negative control: the plain helper with literal operands, FXC
                                assert imm and bad is not None, (label, imm, bad)
                                failed_control += 1
                                print('%-40s fails as expected: %s; %s' % (label, bad, imm[0]), flush=True)
        # Run-time a and b, literal c, the second call first: plain fxc writes `dfma r0.zw, ..., d(0.0, 1.5)`.
        text = ('    float a4 = z + 0.25, a5 = z - 0.25, h = z + 0.5;\n    float v5 = x2w_fma_exact(a5, h, 1.5);\n'
                '    float v4 = x2w_fma_exact(a4, h, 1.5);\n    o.mrt0 = float4(h, v4, v5, 0);')
        want = [0.5, expect(0.25, 0.5, 1.5), expect(-0.25, 0.5, 1.5), 0.0]
        for guarded in (True, False):
            for kind, comp, prof in (('fxc', fxc, 'ps_5_1'), ('dxc', dxc, 'ps_6_0')):
                label = 'runtime swapped %s %s' % (kind, 'guard' if guarded else 'plain')
                ps = compile_hlsl(comp, kind, shader(guarded, text), 'p', prof, d)
                imm = double_immediates(fxc, ps) if kind == 'fxc' else []
                bad = draw(renderer, adapter, vss[kind], ps, want)
                if guarded or kind == 'dxc':
                    assert not imm and bad is None, (label, imm, bad)
                    print('%-40s ok' % label, flush=True)
                else:
                    assert imm and bad is not None, (label, imm, bad)
                    failed_control += 1
                    print('%-40s fails as expected: %s; %s' % (label, bad, imm[0]), flush=True)
    assert failed_control == 5, failed_control
    print('%s: guarded helper matches the binary64 reference under FXC and DXC in every probe, no d(...) '
          'immediates; plain helper fails all %d control probes under FXC (4 literal, 1 run-time)'
          % (adapter, failed_control))


def main():
    controls()
    if len(sys.argv) > 1 and sys.argv[1] == 'draw':
        check_draw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])


if __name__ == '__main__':
    main()
