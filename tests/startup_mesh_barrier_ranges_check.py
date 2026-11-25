"""Check that tools/startup/mesh2hlsl.py runs a group-uniform execz range that holds a phase barrier instead of"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import startup_mesh_exec_restore_check as E  # noqa: E402  (encoder, prologue, assembler)

I, ins, exp = E.I, E.ins, E.exp

PROGRAM = E.PROLOGUE + [
    I('s_mov_b32 s40, 7', 'sop1', 's_mov_b32', d=40, s0='7'),
    I('s_mov_b32 s41, 3', 'sop1', 's_mov_b32', d=41, s0='3'),
    E.CMPX_N,
    ('s_cbranch_execz AFTER', 'execz'),
    I('s_mov_b32 s40, 11', 'sop1', 's_mov_b32', d=40, s0='11'),
    E.CMPX_T,
    ('s_cbranch_execz INNER', 'execz'),
    I('s_mov_b32 s41, 13', 'sop1', 's_mov_b32', d=41, s0='13'),
    E.V1_9,
    ('INNER:', None),
    ('AFTER:', None),
    I('s_mov_b64 exec, -1', 'sop1', 's_mov_b64', d=126, s0='-1'),
    I('s_mov_b32 m0, 0x80080', 'sop1', 's_mov_b32', d=124, s0='0x80080'),
    ('s_sendmsg 0x9', lambda: ins('sopp', 's_sendmsg', imm=9)),
    I('v_mov_b32_e32 v4, s40', 'vop1', 'v_mov_b32_e32', d=4, s0='s40'),
    I('v_mov_b32_e32 v6, s41', 'vop1', 'v_mov_b32_e32', d=6, s0='s41'),
    I('v_cvt_f32_u32_e32 v2, v0', 'vop1', 'v_cvt_f32_u32_e32', d=2, s0='v0'),
    I('v_mov_b32_e32 v3, 0', 'vop1', 'v_mov_b32_e32', d=3, s0='0'),
    ('exp pos0 v2, v2, v2, v2 done', lambda: exp(12, ['v2', 'v2', 'v2', 'v2'], True)),
    ('exp param0 v1, v0, v4, v6', lambda: exp(32, ['v1', 'v0', 'v4', 'v6'])),
    ('exp target_20 v3, off, off, off done', lambda: exp(20, ['v3', None, None, None], True)),
    ('s_endpgm', lambda: ins('sopp', 's_endpgm', imm=0)),
]


def translate(path, env):
    return E.translate(path, dict({'X2W_MESH_BARRIER_RANGES': '1'}, **env))


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_barrier_ranges')
    os.makedirs(out_dir, exist_ok=True)
    code, texts = E.assemble(PROGRAM)
    path = os.path.join(out_dir, 'nested.bin')
    with open(path, 'wb') as f:
        f.write(code)
    got = E.decoded(code)
    E.check('decodes to the intended %d instructions' % len(texts), [t for _, t in got] == [t for _, t in texts],
            '\n'.join('%5x  %-40s %s' % (o, t, g[1]) for (o, t), g in zip(texts, got) if g[1] != t))
    on = translate(path, {})
    E.check('translates with the rule on', on.startswith('OK'), on[:300])
    skips = [k for k in range(4) if '(x2w_any[%d][0] | x2w_any[%d][1]) == 0u' % (k, k) in on]
    E.check('two group-uniform ranges published', 'x2w_any[1][wave] = 1u' in on and 'x2w_any[2][wave]' not in on, '')
    E.check('outer range (k 0, holds the inner publish barrier) is not skipped; inner (k 1) keeps its skip',
            skips == [1], 'skip branches for k = %s' % skips)
    E.check('s40 restored for a wave with no lane in the outer range', 'x2w_any[0][wave] == 0u) s40 = ' in on, '')
    E.check('s41 restored for a wave with no lane in the inner range', 'x2w_any[1][wave] == 0u) s41 = ' in on, '')
    off = translate(path, {'X2W_MESH_BARRIER_RANGES': '0'})
    E.check('negative control: X2W_MESH_BARRIER_RANGES=0 refuses again',
            # f37ad9f8 refused in barrier_count; since 28020dea (uniform skips over translator barriers) the skip
            # check in body() refuses first, because the inner range's publish barrier is not a lane-SCC one
            off.startswith(('REFUSED paths cross different numbers of barriers',
                            'REFUSED s_barrier inside a group-uniform execz range')), off[:200])
    print('program written to %s (difftest: python -B difftest.py %s <harness> --groups 4 --seed 1)' % (path, path))
    if E.failures:
        print('%d check(s) failed' % len(E.failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
