"""SGPR spill to VGPR lanes across branches in the pixel stage of tools/startup/gfx2hlsl.py (#2125).

`v_writelane_b32 vN, sM, k` parks a scalar in lane k of vN and `v_readlane_b32 sD, vN, k` reads it back; both ignore
EXEC (AMD RDNA 2 ISA reference, VOP3 lane operations). The translator keeps one scalar per (vN, k). In a program with
branches that scalar is assigned on the pixel's own path, while the hardware slot holds the last write on the wave's
path. A read is accepted when a write of its (VGPR, lane) reaches it on every path (forward must-dataflow:
intersection at joins, fixed point through loops) and no write that can supply it lies where the wave can run it
while this pixel skips it: inside the region of a branch not proven uniform that the read is outside, or, when several
writes can supply the read, inside any such region.

Synthetic programs only, translated through Stage.translate(). Every refusal has a negative control, and the
translator mutations at the end (union instead of intersection, one pass instead of the fixed point, each region rule
removed) must each turn a control red.

  (no arguments)   pure Python
  binary <llvm-mc>  the draw fixtures assemble to their words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]   both uniform choices drawn, read back, opposite arm refused
"""
import re
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / 'tools'
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / 'startup'))
import gfx2hlsl as G  # noqa: E402
sys.path.insert(0, str(HERE))
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

failures = []


def check(label, ok, detail=''):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        failures.append(label)
        if detail:
            print('  ' + detail)


def program(text):
    """Assembly text with `Name:` labels -> decoded (offset, size, name, text) tuples, four bytes each."""
    lines = [line.strip() for line in text.strip().splitlines()]
    labels, offset = {}, 0
    for line in lines:
        if line.endswith(':'):
            labels[line[:-1]] = offset
        else:
            offset += 4
    decoded, offset = [], 0
    for line in lines:
        if not line.endswith(':'):
            line = re.sub(r'\bL\w+\b', lambda m: 'label_%x' % labels[m[0]], line)
            decoded.append((offset, 4, line.split(' ', 1)[0], line))
            offset += 4
    return decoded


def translate(text, module=G):
    """-> (refusal reasons, emitted lines in block order)."""
    refusals = []
    stage = module.Stage(SimpleNamespace(root={'params': [], 'flags': 0}), 'ps', b'', refusals=refusals,
                         decoded=program(text))
    return [reason for reason, _ in refusals], [line for _, lines, _ in stage.blocks for line in lines]


# s3 is an entry scalar, so `s_cmp_eq_u32 s2, s3` is a uniform SCC branch; s_cbranch_execz is never proven uniform.
ACCEPTED = {
    'two arms of a uniform branch, each writing a different constant': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Lelse
        v_writelane_b32 v10, 7, 3
        s_branch Ljoin
        Lelse:
        v_writelane_b32 v10, 9, 3
        Ljoin:
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        s_endpgm''', ['sp10_3 = 0x00000007u;', 'sp10_3 = 0x00000009u;', 's1 = sp10_3;', 'v1 = s1;']),
    'two arms of a uniform branch writing the same constant keep it': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Lelse
        v_writelane_b32 v10, 7, 3
        s_branch Ljoin
        Lelse:
        v_writelane_b32 v10, 7, 3
        Ljoin:
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        s_endpgm''', ['sp10_3 = 0x00000007u;', 'sp10_3 = 0x00000007u;', 's1 = sp10_3;', 'v1 = 0x00000007u;']),
    'write before a non-uniform branch, read inside its region and after the join': ('''
        v_writelane_b32 v10, s3, 3
        s_and_saveexec_b32 s4, vcc_lo
        s_cbranch_execz Ljoin
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        Ljoin:
        s_or_b32 exec_lo, exec_lo, s4
        v_readlane_b32 s5, v10, 3
        v_mov_b32 v2, s5
        s_endpgm''', ['sp10_3 = s3;', 's1 = sp10_3;', 'if (exec & 1) v1 = s1;', 's5 = sp10_3;',
                      'if (exec & 1) v2 = s5;']),
    'write and read inside the same non-uniform region': ('''
        s_and_saveexec_b32 s4, vcc_lo
        s_cbranch_execz Ljoin
        v_writelane_b32 v10, s3, 3
        s_mov_b32 s3, 0
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        Ljoin:
        s_or_b32 exec_lo, exec_lo, s4
        s_endpgm''', ['sp10_3 = s3;', 's1 = sp10_3;', 'if (exec & 1) v1 = s1;']),
    'a lane spelled 0x3 overwrites lane 3': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Ljoin
        Ljoin:
        v_writelane_b32 v10, 7, 3
        v_writelane_b32 v10, 9, 0x3
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        s_endpgm''', ['sp10_3 = 0x00000007u;', 'sp10_3 = 0x00000009u;', 's1 = sp10_3;', 'v1 = 0x00000009u;']),
    'write before a uniform loop, read in the body and after it': ('''
        v_writelane_b32 v10, s3, 3
        s_mov_b32 s2, 0
        Lloop:
        v_readlane_b32 s1, v10, 3
        s_add_u32 s2, s2, 1
        s_cmp_lt_u32 s2, 4
        s_cbranch_scc1 Lloop
        v_readlane_b32 s5, v10, 3
        v_mov_b32 v1, s5
        s_endpgm''', ['sp10_3 = s3;', 's1 = sp10_3;', 's5 = sp10_3;', 'v1 = s5;']),
}

REFUSED = {
    'a write on one arm only (no write on some path)': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Ljoin
        v_writelane_b32 v10, 7, 3
        Ljoin:
        v_readlane_b32 s1, v10, 3
        s_endpgm''', 'v_readlane_b32 reads v10 lane 3 with no v_writelane_b32 of it on some path'),
    'another lane on the other arm': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Lelse
        v_writelane_b32 v10, 7, 3
        s_branch Ljoin
        Lelse:
        v_writelane_b32 v10, 7, 4
        Ljoin:
        v_readlane_b32 s1, v10, 3
        s_endpgm''', 'v_readlane_b32 reads v10 lane 3 with no v_writelane_b32 of it on some path'),
    'a later write in a uniform loop reaches the loop head (loop-carried)': ('''
        v_writelane_b32 v10, 7, 3
        s_mov_b32 s2, 0
        Lloop:
        v_readlane_b32 s1, v10, 3
        v_mov_b32 v1, s1
        v_writelane_b32 v10, 9, 3
        s_add_u32 s2, s2, 1
        s_cmp_lt_u32 s2, 4
        s_cbranch_scc1 Lloop
        s_endpgm''', 'v_readlane_b32 reads v10 lane 3 written later in a loop (loop-carried)'),
    'the only write inside a non-uniform loop, read after it': ('''
        s_mov_b32 s2, 0
        Lloop:
        v_writelane_b32 v10, s2, 3
        s_add_u32 s2, s2, 1
        v_cmp_gt_u32 vcc_lo, s2, v0
        s_cbranch_vccnz Lloop
        v_readlane_b32 s1, v10, 3
        s_endpgm''', 'v_readlane_b32 reads v10 lane 3 outside the region of the branch at 0x10 that writes it, '
                     'which is not proven uniform'),
    'a write before and a write inside a non-uniform region, read after the join': ('''
        v_writelane_b32 v10, s3, 3
        s_and_saveexec_b32 s4, vcc_lo
        s_cbranch_execz Ljoin
        v_writelane_b32 v10, 9, 3
        Ljoin:
        s_or_b32 exec_lo, exec_lo, s4
        v_readlane_b32 s1, v10, 3
        s_endpgm''', 'v_readlane_b32 joins 2 writes of v10 lane 3, one inside the region of the branch at 0x8, which '
                     'is not proven uniform'),
    'two arms of a non-uniform branch rejoining inside an outer non-uniform region': ('''
        s_and_saveexec_b32 s4, vcc_lo
        s_cbranch_execz Lend
        v_writelane_b32 v10, 7, 3
        s_cbranch_vccz Lelse
        v_writelane_b32 v10, 9, 3
        Lelse:
        v_readlane_b32 s1, v10, 3
        Lend:
        s_or_b32 exec_lo, exec_lo, s4
        s_endpgm''', 'v_readlane_b32 joins 2 writes of v10 lane 3, one inside the region of the branch at 0x4, which '
                     'is not proven uniform'),
    'the reaching write was refused': ('''
        s_mov_b32 s2, 5
        s_cmp_eq_u32 s2, s3
        s_cbranch_scc0 Ljoin
        Ljoin:
        v_writelane_b32 v10, vcc_lo, 3
        v_readlane_b32 s1, v10, 3
        s_endpgm''', 'v_readlane_b32 reads v10 lane 3 from a v_writelane_b32 that was not translated'),
}


def controls(module=G, quiet=False):
    """-> labels of the controls that failed against `module`."""
    failed = []
    for label, (text, want) in ACCEPTED.items():
        reasons, lines = translate(text, module)
        ok = not reasons and [line for line in lines if re.match(r'(if \(exec & 1\) )?(sp\d|s1 |s5 |v[12] )', line)] == want
        failed += [] if ok else ['accepted: ' + label]
        if not quiet:
            check('accepted: %s' % label, ok, '%s %s' % (reasons, lines))
    for label, (text, want) in REFUSED.items():
        reasons, _ = translate(text, module)
        ok = want in reasons
        failed += [] if ok else ['refused: ' + label]
        if not quiet:
            check('refused: %s' % label, ok, str(reasons))
    return failed


MUTATIONS = {
    'union instead of intersection at joins': (
        '{k: into[t][k] | state[k] for k in into[t].keys() & state.keys()}',
        '{k: into[t].get(k, frozenset()) | state.get(k, frozenset()) for k in into[t].keys() | state.keys()}'),
    'one pass instead of the fixed point': (
        '        while changed:\n            changed = False\n            for b, (lo, hi) in enumerate(bounds):\n',
        '        for changed in [False]:\n            for b, (lo, hi) in enumerate(bounds):\n'),
    'lanes keyed by their spelling': ('lane = int(ops[2], 0)  # 3 and 0x3', 'lane = ops[2]  # 3 and 0x3'),
    'no region rule for a single write': ('            if here not in blocks:\n', '            if False:\n'),
    'no region rule for joined writes': ('            if len(writes) > 1:\n', '            if False:\n'),
    'every conditional branch counted as uniform': ('            if off in self.uniform_branches or',
                                                    '            if True or'),
}


def mutated(old, new):
    source = (TOOLS / 'startup' / 'gfx2hlsl.py').read_text(encoding='utf-8').replace('\r\n', '\n')
    assert source.count(old) == 1, old
    module = types.ModuleType('gfx2hlsl_mutant')
    module.__file__ = str(TOOLS / 'startup' / 'gfx2hlsl.py')
    exec(compile(source.replace(old, new), module.__file__, 'exec'), module.__dict__)
    return module


# The same slot drawn on D3D12: two uniform branch choices write different words to v10 lane 3, the join reloads it and
# exports it OR-ed with the packed screen position. s3 is 5 (the compare s2 == s3 holds: first arm) or 6 (second arm).
WIDTH = HEIGHT = 16
PACKED = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: the packed screen position only, in v0
ARM_WORDS = (0x40200000, 0x40400000)  # binary32 words with no bit below 21, clear of x | y << 16
FIXTURE_TEXT = '''s_mov_b32 s0, 0x40200000
s_mov_b32 s1, 0x40400000
s_mov_b32 s2, 5
s_mov_b32 s3, %d
s_cmp_eq_u32 s2, s3
s_cbranch_scc0 3
v_writelane_b32 v10, s0, 3
s_branch 2
v_writelane_b32 v10, s1, 3
v_readlane_b32 s5, v10, 3
v_or_b32_e32 v1, s5, v0
exp mrt0 v1, v1, v1, v1 done vm
s_endpgm
'''
_COMMON = (0xbe8003ff, 0x40200000, 0xbe8103ff, 0x40400000, 0xbe820385)
_TAIL = (0xbf060302, 0xbf840003, 0xd761000a, 0x00010600, 0xbf820002, 0xd761000a, 0x00010601, 0xd7600005,
         0x0001070a, 0x38020005, 0xf800180f, 0x01010101, 0xbf810000)
CHOICES = {5: ARM_WORDS[0], 6: ARM_WORDS[1]}
CHOICE_WORDS = {s3: struct.pack('<19I', *_COMMON, 0xbe830380 | s3, *_TAIL) for s3 in CHOICES}


def pixel_image(arm):
    return [(x | y << 16) | arm for y in range(HEIGHT) for x in range(WIDTH) for _ in range(4)]


def check_binary(llvm_mc):
    for s3 in CHOICES:
        check('llvm-mc assembles the s3 == %d fixture to the same words' % s3,
              assemble(llvm_mc, FIXTURE_TEXT % s3) == CHOICE_WORDS[s3])


def words_file(directory, name, words):
    path = directory / ('%s.bin' % name)
    path.write_bytes(struct.pack('<%dI' % len(words), *words))
    return path


def draw(renderer, adapter, vs, ps, expected):
    run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT),
                          str(expected)], capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        sys.exit(1 if failures else 77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    texts = {}
    for s3, words in CHOICE_WORDS.items():
        refusals = []
        texts[s3] = G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED, refusals=refusals).hlsl_ps()
        check('s3 == %d: the pixel stage translates with no refusal' % s3, not refusals, str(refusals))
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        images = {s3: words_file(directory, 'image%d' % s3, pixel_image(arm)) for s3, arm in CHOICES.items()}
        for name, compiler, profile_vs, profile_ps in (
                [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])):
            vs = compile_hlsl(compiler, name, FULLSCREEN_VS, 'vs_' + name, profile_vs, directory)
            for s3, text in texts.items():
                ps = compile_hlsl(compiler, name, text, 'ps_%s_%d' % (name, s3), profile_ps, directory)
                run = draw(renderer, adapter, vs, ps, images[s3])
                check('%s %s s3 == %d: all %d pixels equal the selected arm' % (adapter, name, s3, WIDTH * HEIGHT),
                      run.returncode == 0 and 'all %d pixels matched' % (WIDTH * HEIGHT) in run.stdout,
                      run.stdout + run.stderr)
                # Negative control: the opposite arm's reference must differ from what this choice draws.
                run = draw(renderer, adapter, vs, ps, images[11 - s3])
                check('%s %s s3 == %d: the opposite arm as reference fails' % (adapter, name, s3),
                      run.returncode != 0 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)


def main():
    if len(sys.argv) != 1:
        mode, *args = sys.argv[1:]
        if mode == 'binary':
            check_binary(*args)
        elif mode == 'draw':
            controls()
            check_draw(*args)
        else:
            raise SystemExit(__doc__)
        if failures:
            print('FAIL')
            for text in failures:
                print('  ' + text)
            sys.exit(1)
        print('PASS')
        return
    controls()
    # The straight-line requirement stays for a stage with no block graph (a unit probe that skips translate()).
    probe = G.Stage.__new__(G.Stage)
    probe.kind, probe.lines, probe.sym, probe.masks = 'ps', [], {}, set()
    probe.used_s, probe.used_v, probe.uniform_v, probe.varying_s = set(), set(), set(), set()
    probe.scc_uniform, probe.has_branches, probe.ins = True, True, []
    try:
        probe.one('v_writelane_b32', 'v_writelane_b32 v10, s0, 3')
        reason = None
    except G.Unsupported as error:
        reason = str(error)
    check('a stage with branches and no block graph keeps the straight-line refusal (%s)' % reason,
          reason == 'v_writelane_b32 needs straight-line code to prove write-before-read')
    # The whole pixel stage: the slot is declared once before the blocks and assigned on each arm.
    from startup_image_3d_check import VS_WORDS, root_blob
    text = ACCEPTED['two arms of a uniform branch, each writing a different constant'][0].replace(
        'v_mov_b32 v1, s1', 'v_mov_b32 v1, s1\nexp mrt0 v1, v1, v1, v1 done vm')
    refusals = []
    ps = G.Pipeline(VS_WORDS, b'', root_blob(), refusals=refusals, decoded={'ps': program(text)}).hlsl_ps()
    want = ['    uint sp10_3 = 0;', '    if (pc == 1) {', '        sp10_3 = 0x00000007u;', '    if (pc == 2) {',
            '        sp10_3 = 0x00000009u;', '    if (pc == 3) {', '        s1 = sp10_3;']
    lines = ps.splitlines()
    at = [lines.index(line) if line in lines else -1 for line in want]
    check('pixel stage: slot declared once, assigned on both arms, read after the join',
          not refusals and -1 not in at and at == sorted(at) and ps.count('uint sp10_3') == 1, '%s\n%s' % (refusals, ps))
    for label, (old, new) in MUTATIONS.items():
        red = controls(mutated(old, new), quiet=True)
        check('mutation "%s" turns %d control(s) red: %s' % (label, len(red), red), bool(red))
    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
