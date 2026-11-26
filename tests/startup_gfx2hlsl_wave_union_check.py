"""X2W_PS_WAVE_UNION=1: wave64 reductions in a pixel stage over the host wave (tools/startup/gfx2hlsl.py,
wave_union_plan).

  (no arguments)                       translation controls, pure Python
  draw <dxc> <dump.exe> <warp|hardware>
                                       compile the translated pixel stage with DXC ps_6_0, draw 16 x 16 pixels with
                                       a two-target dump tool (<adapter> <vs> <ps> <w> <h> <out.bin>: the words of
                                       both R32G32B32A32 targets, row by row) and compare every pixel with a CPU
                                       reference that takes the wave membership as a parameter

The guest computes OP over the active lanes of its wave64 (s_or_saveexec -1, v_cndmask with the identity, four DPP
row steps, v_permlanex16, readlane 0, readlane 63). Which pixels share a wave is the hardware's choice, so the
guest result already depends on it; with the gate the HLSL takes OP over the active lanes of the host wave
(WaveActiveMin / WaveActiveMax / WaveActiveBitOr) and the manifest says `wavemembership host`.

The fixture is built here from the instruction formats (checked against the decoder): v0 is the packed screen
position x | y << 16; v_cmpx_ne_u32 0, (v0 & 5) leaves the pixels with x & 5 != 0 active; three chains reduce v0
with min (identity 0xffffffff), max and or (identity 0); exec is restored and mrt0 = (min, max, or, own position),
each masked to 21 bits and ORed with 0x40000000 so the words are normal binary32 values.

A program that took a chain decides every branch per host wave. The branch fixture (branch_fixture) checks it: after
a min chain, exec is narrowed to the odd-x pixels and s_cbranch_execz skips a block that writes s20 = 1 (a scalar
write, run by the guest when any lane of the wave is active) and s21 = v_readfirstlane of (min + 7) computed in
that block from the uniform s10. mrt0 = (min, s20, s21, own position). The reference: s20 = 1 and s21 = min + 7 in
every pixel of a wave with an active odd-x pixel, else 0. An SCC branch on a mask (s_and_b64 with exec) and a
readfirstlane of a per-lane VGPR are refused.

The draw mode adds an instrumentation output to the translated text (mrt1 = WaveReadLaneFirst of the position,
the lane index and the lane count, at the end of main) to learn the host's actual lane grouping, computes the
reference from that grouping, and requires every pixel to match. Negative controls: the reference computed with
each pixel in a wave of its own must differ, as must translations with the min chain lowered as WaveActiveMax,
the execz branch decided per lane, and the readfirstlane replaced by the lane's own value. WARP runs 4-lane waves
(2x2 quads, split where the full-screen triangle's edge cuts a quad), so on WARP the reduction does not cross a quad;
the hardware adapter mode runs the same checks on 32/64-lane waves.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

WIDTH = HEIGHT = 16
FLAG = 0x40000000
KEEP = 0x001fffff
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
OPS = {'min': (0x13, -1, 0xffffffff), 'max': (0x14, 0, 0), 'or': (0x1c, 0, 0)}  # VOP2 opcode, identity token, value
CTRL = (0xb1, 0x4e, 0x141, 0x140)  # quad_perm:[1,0,3,2], quad_perm:[2,3,0,1], row_half_mirror, row_mirror
failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + str(detail)[:3000])


def inline(token):
    return {-1: 193, 0: 128, 63: 191, 5: 133}[token]


def sop1(op, sdst, src):
    return [0xbe800000 | sdst << 16 | op << 8 | src]


def vop2(op, vdst, src0, vsrc1, literal=None):
    return [op << 25 | vdst << 17 | vsrc1 << 9 | src0] + ([literal] if literal is not None else [])


def vop2_dpp(op, v, ctrl, fi=1):
    return [op << 25 | v << 17 | v << 9 | 250, v | ctrl << 8 | fi << 18 | 0xf << 24 | 0xf << 28]


def vop3(op, vdst, src0, src1=0, src2=0):
    return [0xd4000000 | op << 16 | vdst, src0 | src1 << 9 | src2 << 18]


def chain(op, mask, k, src=0, controls=CTRL, ident=None, tail=True):
    """s_or_saveexec_b64 s[mask], -1 ... v_readlane_b32 sK, v94, 63; v_mov_b32 v(dst), sK is added by fixture()."""
    code, token, _ = OPS[op]
    ident = token if ident is None else ident
    w = sop1(0x25, mask, inline(-1))                                   # s_or_saveexec_b64 s[mask:mask+1], -1
    w += vop3(0x101, 94, inline(ident), 256 + src, mask)               # v_cndmask_b32_e64 v94, IDENT, vS, s[mask]
    for c in controls:
        w += vop2_dpp(code, 94, c)                                     # v_OP_dpp v94, v94, v94 <ctrl> fi:1
    w += vop3(0x378, 95, 256 + 94, 128, 128)                           # v_permlanex16_b32 v95, v94, 0, 0
    w += vop2(code, 94, 256 + 94, 95)                                  # v_OP_e32 v94, v94, v95
    w += vop3(0x360, k, 256 + 94, 128)                                 # v_readlane_b32 sK, v94, 0
    w += vop2(code, 94, k, 94)                                         # v_OP_e32 v94, sK, v94
    w += sop1(0x04, 126, mask)                                         # s_mov_b64 exec, s[mask:mask+1]
    if tail:
        w += vop3(0x360, k, 256 + 94, inline(63))                      # v_readlane_b32 sK, v94, 63
    return w


def fixture(mutate=None):
    w = sop1(0x04, 0, 126)                                             # s_mov_b64 s[0:1], exec
    w += [0x7e000000 | 9 << 17 | 1 << 9 | 244]                         # v_mov_b32 v9, 2.0 (0x40000000)
    w += vop2(0x1b, 6, inline(5), 0)                                   # v_and_b32 v6, 5, v0
    w += [0x7c000000 | 0xd5 << 17 | 6 << 9 | 128]                      # v_cmpx_ne_u32 0, v6
    for i, op in enumerate(('min', 'max', 'or')):
        kwargs = (mutate or {}).get(op, {})
        w += chain(op, 2 + 2 * i, 10 + i, **kwargs)
        w += [0x7e000000 | (1 + i) << 17 | 1 << 9 | (10 + i)]          # v_mov_b32 v(1+i), sK
    w += sop1(0x04, 126, 0)                                            # s_mov_b64 exec, s[0:1]
    for v in (1, 2, 3):
        w += vop2(0x1b, v, 255, v, KEEP)                               # v_and_b32 vN, 0x1fffff, vN
        w += vop2(0x1c, v, 256 + v, 9)                                 # v_or_b32 vN, vN, v9
    w += vop2(0x1c, 4, 256 + 0, 9)                                     # v_or_b32 v4, v0, v9
    w += [0xf800180f, 0x04030201, 0xbf810000]                          # exp mrt0 v1, v2, v3, v4 done vm; s_endpgm
    return struct.pack('<%dI' % len(w), *w)


def branch_fixture(variant=None):
    """See the module text. variant 'scc_mask': an s_and_b64 of exec and an s_cbranch_scc0 before the execz branch;
    'scc_vcmp': the same on an s_and_b32 of a v_cmp_e64 mask of a uniform value (a VALU result is per lane);
    'rfl_lane': the readfirstlane source is the lane's own position."""
    w = sop1(0x04, 0, 126)                                             # s_mov_b64 s[0:1], exec
    w += [0x7e000000 | 9 << 17 | 1 << 9 | 244]                         # v_mov_b32 v9, 2.0
    w += sop1(0x03, 20, 128) + sop1(0x03, 21, 128)                     # s_mov_b32 s20, 0; s_mov_b32 s21, 0
    w += vop2(0x1b, 6, inline(5), 0)                                   # v_and_b32 v6, 5, v0
    w += [0x7c000000 | 0xd5 << 17 | 6 << 9 | 128]                      # v_cmpx_ne_u32 0, v6
    w += chain('min', 2, 10)
    w += [0x7e000000 | 1 << 17 | 1 << 9 | 10]                          # v_mov_b32 v1, s10
    w += vop2(0x1b, 7, 129, 0)                                         # v_and_b32 v7, 1, v0
    w += [0x7c000000 | 0xd5 << 17 | 7 << 9 | 128]                      # v_cmpx_ne_u32 0, v7
    if variant == 'scc_mask':
        w += [0x80000000 | 0x0f << 23 | 30 << 16 | 0 << 8 | 126]       # s_and_b64 s[30:31], exec, s[0:1]
        w += [0xbf840000]                                              # s_cbranch_scc0 (next instruction)
    if variant == 'scc_vcmp':
        w += [0x7e000000 | 8 << 17 | 1 << 9 | 10]                      # v_mov_b32 v8, s10 (uniform value)
        w += [0xd4c5001e, 128 | (256 + 8) << 9]                        # v_cmp_ne_u32_e64 s30, 0, v8 (per-lane mask)
        w += [0x80000000 | 0x0e << 23 | 31 << 16 | 30 << 8 | 30]       # s_and_b32 s31, s30, s30
        w += [0xbf840000]                                              # s_cbranch_scc0 (next instruction)
    w += [0xbf880004]                                                  # s_cbranch_execz over the next 4 dwords
    w += sop1(0x03, 20, 129)                                           # s_mov_b32 s20, 1
    w += [0x7e000000 | 8 << 17 | 1 << 9 | (256 if variant == 'rfl_lane' else 10)]  # v_mov_b32 v8, s10 (or v0)
    w += vop2(0x25, 8, 135, 8)                                         # v_add_nc_u32 v8, 7, v8
    w += [0x7e000000 | 21 << 17 | 2 << 9 | (256 + 8)]                  # v_readfirstlane_b32 s21, v8
    w += sop1(0x04, 126, 0)                                            # s_mov_b64 exec, s[0:1]
    w += [0x7e000000 | 2 << 17 | 1 << 9 | 20, 0x7e000000 | 3 << 17 | 1 << 9 | 21]  # v_mov v2, s20; v_mov v3, s21
    for v in (1, 2, 3):
        w += vop2(0x1b, v, 255, v, KEEP)
        w += vop2(0x1c, v, 256 + v, 9)
    w += vop2(0x1c, 4, 256 + 0, 9)
    w += [0xf800180f, 0x04030201, 0xbf810000]
    return struct.pack('<%dI' % len(w), *w)


def translate(words, gate=True):
    saved = os.environ.get('X2W_PS_WAVE_UNION')
    os.environ['X2W_PS_WAVE_UNION'] = '1' if gate else '0'
    try:
        return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED)
    finally:
        if saved is None:
            os.environ.pop('X2W_PS_WAVE_UNION', None)
        else:
            os.environ['X2W_PS_WAVE_UNION'] = saved


def refused(words, gate=True):
    try:
        pipe = translate(words, gate)
        pipe.hlsl_ps()
        pipe.manifest()
    except G.Unsupported as reason:
        return str(reason)
    return None


def controls():
    text = [a for _, _, _, a in G.decode(fixture())]
    chain_text = [a for a in text if a.startswith(('s_or_saveexec', 'v_cndmask', 'v_permlane')) or '_dpp' in a]
    check('the fixture decodes to three chains of the 14bb form', chain_text[:7] == [
        's_or_saveexec_b64 s[2:3], -1', 'v_cndmask_b32_e64 v94, -1, v0, s2',
        'v_min_u32_dpp v94, v94, v94 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1',
        'v_min_u32_dpp v94, v94, v94 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1',
        'v_min_u32_dpp v94, v94, v94 row_half_mirror row_mask:0xf bank_mask:0xf fi:1',
        'v_min_u32_dpp v94, v94, v94 row_mirror row_mask:0xf bank_mask:0xf fi:1',
        'v_permlanex16_b32 v95, v94, 0, 0'], '\n'.join(text))
    pipe = translate(fixture())
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    for op, intrinsic, ident in (('min', 'WaveActiveMin', '0xffffffffu'), ('max', 'WaveActiveMax', '0x00000000u'),
                                 ('or', 'WaveActiveBitOr', '0x00000000u')):
        check('%s chain -> %s over the saved exec bit with identity %s' % (op, intrinsic, ident),
              re.search(r'wu\d = %s\(\(s\d+ & 1u\) \? v0 : %s\);' % (intrinsic, ident), ps) is not None, ps)
    check('no DPP, permlane or lane-0 readlane text is left', not re.search(r'permlane|QuadRead|_dpp', ps), ps)
    check('the stage needs SM 6.0 and the manifest says wavemembership host',
          pipe.ps.needs_sm6 and 'wavemembership host' in manifest.splitlines(), manifest)
    check('gate off: the chain stays refused at the DPP', 'v_min_u32_dpp' in (refused(fixture(), gate=False) or ''),
          refused(fixture(), gate=False))
    # Anything but the exact chain is left to the normal lowering, which refuses the DPP.
    mutants = {
        'a row step out of order': {'min': {'controls': (0x4e, 0xb1, 0x141, 0x140)}},
        'a missing row step': {'min': {'controls': (0xb1, 0x4e, 0x141, 0x141)}},
        'the wrong identity (0 for min)': {'min': {'ident': 0}},
        'no readlane of lane 63': {'min': {'tail': False}},
    }
    for label, mutate in mutants.items():
        reason = refused(fixture(mutate))
        check('refused: %s (%s)' % (label, reason), reason is not None)
    # v94 read outside the chains drops the whole plan
    extra = fixture()[:-12] + struct.pack('<I', 0x7e000000 | 5 << 17 | 1 << 9 | (256 + 94)) + fixture()[-12:]
    reason = refused(extra)
    check('refused: v94 read outside the chains (%s)' % reason, reason is not None)
    # The branch fixture: wave-level execz, a scalar write in the skipped block, readfirstlane of a uniform value.
    pipe = translate(branch_fixture())
    ps = pipe.hlsl_ps()
    check('branch fixture: the execz branch is decided per host wave',
          'pc = (!WaveActiveAnyTrue((exec & 1) != 0)) ?' in ps, ps)
    check('branch fixture: readfirstlane of the uniform min + 7 -> x2w_readfirstlane',
          re.search(r's21 = x2w_readfirstlane\(v8, \(exec & 1u\) != 0\);', ps) is not None, ps)
    check('branch fixture: the scalar write in the block is unguarded', '\n        s20 = 0x00000001u;' in ps, ps)
    for variant, want in (('scc_mask', 'SCC not host-uniform'), ('scc_vcmp', 'SCC not host-uniform'),
                          ('rfl_lane', 'v_readfirstlane_b32')):
        reason = refused(branch_fixture(variant))
        check('branch fixture refused: %s (%s)' % (variant, reason), reason is not None and want in reason)
    check('a stage without such a chain is unchanged by the gate',
          translate(VS_WORDS[:0] + struct.pack('<3I', 0xf800180f, 0, 0xbf810000), True).hlsl_ps() ==
          translate(struct.pack('<3I', 0xf800180f, 0, 0xbf810000), False).hlsl_ps())


# ---- draw mode -------------------------------------------------------------------------------------------------
INSTRUMENT = ('    { uint x2w_pid = uint(i.pos.x) | (uint(i.pos.y) << 16); '
              'o.mrt1 = float4(asfloat(0x40000000u | WaveReadLaneFirst(x2w_pid)), asfloat(0x40000000u | WaveGetLaneIndex()), '
              'asfloat(0x40000000u | WaveGetLaneCount()), 2.0); }')


def instrument(ps):
    out = ps.replace('    float4 mrt0 : SV_Target0;', '    float4 mrt0 : SV_Target0;\n    float4 mrt1 : SV_Target1;', 1)
    assert out != ps
    last = out.rindex('    return o;')
    return out[:last] + INSTRUMENT + '\n' + out[last:]


def reference(groups):
    """Pixel words from a grouping: groups[(x, y)] = wave key; the reduction is over the active pixels of a group."""
    members = {}
    for p, key in groups.items():
        members.setdefault(key, []).append(p)
    out = {}
    for p, key in groups.items():
        active = [q[0] | q[1] << 16 for q in members[key] if q[0] & 5]
        mn = min(active) if active else 0xffffffff
        mx = max(active) if active else 0
        orr = 0
        for a in active:
            orr |= a
        if not p[0] & 5:
            mn = mx = orr = 0  # v1..v3 are written under the narrowed exec only: an inactive pixel keeps 0
        out[p] = [(v & KEEP) | FLAG for v in (mn, mx, orr)] + [(p[0] | p[1] << 16) | FLAG]
    return out


def branch_reference(groups):
    """branch_fixture words from a grouping: (min over the active pixels, s20, s21, own position)."""
    members = {}
    for p, key in groups.items():
        members.setdefault(key, []).append(p)
    out = {}
    for p, key in groups.items():
        active = [q[0] | q[1] << 16 for q in members[key] if q[0] & 5]
        mn = min(active) if active else 0xffffffff
        odd = any(q[0] & 1 for q in members[key])
        s20, s21 = (1, (mn + 7) & 0xffffffff) if odd else (0, 0)
        v1 = mn if p[0] & 5 else 0
        out[p] = [(v & KEEP) | FLAG for v in (v1, s20, s21)] + [(p[0] | p[1] << 16) | FLAG]
    return out


def check_draw(dxc, dump, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'vs', 'vs_6_0', d)
        text = translate(fixture()).hlsl_ps()

        def run(source, name):
            ps = compile_hlsl(dxc, 'dxc', instrument(source), name, 'ps_6_0', d)
            out = d / (name + '.bin')
            r = subprocess.run([dump, adapter, str(vs), str(ps), str(WIDTH), str(HEIGHT), str(out)],
                               capture_output=True, text=True)
            if r.returncode == 77:
                print(r.stdout + r.stderr)
                sys.exit(77)
            if r.returncode:
                raise AssertionError('%s\n%s' % (r.stdout, r.stderr))
            words = struct.unpack('<%dI' % (WIDTH * HEIGHT * 8), out.read_bytes())
            rt0 = {(x, y): list(words[(y * WIDTH + x) * 4:(y * WIDTH + x) * 4 + 4])
                   for y in range(HEIGHT) for x in range(WIDTH)}
            base = WIDTH * HEIGHT * 4
            rt1 = {(x, y): list(words[base + (y * WIDTH + x) * 4:base + (y * WIDTH + x) * 4 + 4])
                   for y in range(HEIGHT) for x in range(WIDTH)}
            return rt0, rt1

        rt0, rt1 = run(text, 'ps')
        groups = {p: w[0] & ~FLAG for p, w in rt1.items()}
        sizes = sorted({sum(1 for g in groups.values() if g == k) for k in set(groups.values())})
        lanes = sorted({w[2] & ~FLAG for w in rt1.values()})
        print('%s: host waves of %s pixels (WaveGetLaneCount %s), %d waves' % (adapter, sizes, lanes,
                                                                                len(set(groups.values()))))
        ref = reference(groups)
        bad = [p for p in rt0 if rt0[p] != ref[p]]
        check('%s: all %d pixels equal the reference over the host grouping' % (adapter, WIDTH * HEIGHT), not bad,
              bad[:4] and ['%s got %s want %s' % (p, ['%08x' % v for v in rt0[p]], ['%08x' % v for v in ref[p]])
                           for p in bad[:4]])
        solo = reference({p: p for p in rt0})
        check('%s: control: one pixel per wave would read a different image' % adapter,
              any(rt0[p] != solo[p] for p in rt0))
        mutated = re.sub(r'WaveActiveMin\(', 'WaveActiveMax(', text)
        assert mutated != text
        rt0m, _ = run(mutated, 'ps_mutated')
        check('%s: control: the min chain lowered as a max fails the reference' % adapter,
              any(rt0m[p] != ref[p] for p in rt0m))

        # branch fixture: same draw, its own grouping (measured again: a different shader may pack differently)
        btext = translate(branch_fixture()).hlsl_ps()
        bt0, bt1 = run(btext, 'ps_branch')
        bgroups = {p: w[0] & ~FLAG for p, w in bt1.items()}
        bref = branch_reference(bgroups)
        bad = [p for p in bt0 if bt0[p] != bref[p]]
        check('%s: branch fixture: all %d pixels equal the reference over the host grouping' % (adapter, WIDTH * HEIGHT),
              not bad, bad[:4] and ['%s got %s want %s' % (p, ['%08x' % v for v in bt0[p]],
                                                            ['%08x' % v for v in bref[p]]) for p in bad[:4]])
        check('%s: branch fixture: some wave mixes taken and not-taken pixels (the per-lane rule would differ)'
              % adapter, any(bref[p] != branch_reference({q: q for q in bt0})[p] for p in bt0))
        mutants = {
            'the execz branch decided per lane': ('!WaveActiveAnyTrue((exec & 1) != 0)', '(exec & 1) == 0'),
            'readfirstlane as the lane\'s own value': ('x2w_readfirstlane(v8, (exec & 1u) != 0)', 'v8'),
        }
        for n, (label, (old, new)) in enumerate(mutants.items()):
            assert old in btext, old
            m0, _ = run(btext.replace(old, new), 'ps_branch_m%d' % n)
            check('%s: branch control: %s fails the reference' % (adapter, label), any(m0[p] != bref[p] for p in m0))


def main():
    controls()
    if len(sys.argv) > 1 and sys.argv[1] == 'draw':
        check_draw(*sys.argv[2:5])
    if failures:
        print('FAIL')
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
