"""Check the path-aware exec walk of tools/startup/mesh2hlsl.py (MeshStage.exec_walk, with exec_copies_at tracking"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import startup_mesh_exec_restore_check as E  # noqa: E402
from startup_mesh_exec_restore_check import I  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


def MOV32(d, s):
    return I('s_mov_b32 s%d, %s' % (d, s), 'sop1', 's_mov_b32', d=d, s0=s)


def XOR(width, d, s0, s1):
    """s_xor_b32 / s_xor_b64 on SGPRs (bases d, s0, s1)."""
    t = (lambda r: 's%d' % r) if width == 32 else (lambda r: 's[%d:%d]' % (r, r + 1))
    return I('s_xor_b%d %s, %s, %s' % (width, t(d), t(s0), t(s1)), 'sop2', 's_xor_b%d' % width, d=d, s0='s%d' % s0,
             s1='s%d' % s1)


def SWAP(width, a, b):
    return [XOR(width, a, a, b), XOR(width, b, a, b), XOR(width, a, a, b)]


COPY34 = I('s_mov_b64 s[34:35], s[30:31]', 'sop1', 's_mov_b64', d=34, s0='s30')
CLEAR34 = I('s_mov_b64 s[34:35], 0', 'sop1', 's_mov_b64', d=34, s0='0')
REST34 = I('s_mov_b64 exec, s[34:35]', 'sop1', 's_mov_b64', d=126, s0='s34')
CMP_T0 = I('s_cmp_eq_u32 s22, 0', 'sopc', 's_cmp_eq_u32', s0='s22', s1='0')
HEAD = E.PROLOGUE + [E.CMPX_N, E.SAVE, ('s_cbranch_execz AFTER', 'execz')]
TAIL = [E.V4_7] + E.EPILOGUE
REFUSE = 'exec written'
PROGRAMS = {
    'clobberskip': ('ok', HEAD + [COPY34, CMP_T0, ('s_cbranch_scc1 R', 'scc1'), CLEAR34, E.CMPX_T, E.V1_9, COPY34,
                                  ('R:', None), REST34] + TAIL),
    'halves': ('ok', HEAD + [MOV32(34, 's30'), MOV32(35, 's31'), E.CMPX_T, E.V1_9, REST34] + TAIL),
    'onepath': (REFUSE, HEAD + [COPY34, CMP_T0, ('s_cbranch_scc1 R', 'scc1'), CLEAR34, E.CMPX_T, E.V1_9,
                                ('R:', None), REST34] + TAIL),
    'swapped': (REFUSE, HEAD + [MOV32(34, 's31'), MOV32(35, 's30'), E.CMPX_T, E.V1_9, REST34] + TAIL),
    'mixed': (REFUSE, HEAD + [MOV32(34, 's30'), MOV32(35, 's22'), E.CMPX_T, E.V1_9, REST34] + TAIL),
    # XOR swaps: s[34:35] = (T, 0) is data; the swap moves the exec copy from s[30:31] into s[34:35]
    'xorswap64': ('ok', HEAD + [MOV32(34, 's22'), MOV32(35, '0')] + SWAP(64, 30, 34) +
                  [E.CMPX_T, E.V1_9, REST34] + TAIL),
    'xorswap32': ('ok', HEAD + [MOV32(34, 's22'), MOV32(35, '0'), XOR(32, 30, 30, 34), E.V1_9,
                                XOR(32, 34, 30, 34), XOR(32, 30, 30, 34)] + SWAP(32, 31, 35) +
                  [E.CMPX_T, E.V1_9, REST34] + TAIL),
    'xorwrong': (REFUSE, HEAD + [MOV32(34, 's22'), MOV32(35, '0')] + SWAP(64, 30, 34) +
                 [E.CMPX_T, E.V1_9, I('s_mov_b64 exec, s[30:31]', 'sop1', 's_mov_b64', d=126, s0='s30')] + TAIL),
    'xorbroken': (REFUSE, HEAD + [MOV32(34, 's22'), MOV32(35, '0'), XOR(64, 30, 30, 34), MOV32(35, '1'),
                                  XOR(64, 34, 30, 34), XOR(64, 30, 30, 34), E.CMPX_T, E.V1_9, REST34] + TAIL),
}


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_exec_walk')
    os.makedirs(out_dir, exist_ok=True)
    for name, (want, items) in PROGRAMS.items():
        code, texts = E.assemble(items)
        path = os.path.join(out_dir, name + '.bin')
        with open(path, 'wb') as f:
            f.write(code)
        got = E.decoded(code)
        check('%s: decodes to the intended %d instructions' % (name, len(texts)),
              [t for _, t in got] == [t for _, t in texts],
              '\n'.join('%5x  %-40s %s' % (o, t, g[1]) for (o, t), g in zip(texts, got) if g[1] != t))
        on = E.translate(path, {})
        if want == 'ok':
            check('%s: translates' % name, on.startswith('OK'), on[:300])
            if name == 'clobberskip':
                off = E.translate(path, {'X2W_MESH_EXEC_COPIES': '0'})
                check('clobberskip: negative control X2W_MESH_EXEC_COPIES=0 refuses', off.startswith('REFUSED'),
                      off[:300])
        else:
            check('%s: refused (%s)' % (name, want), on.startswith('REFUSED') and want in on.splitlines()[0],
                  on[:200])
    print('programs written to %s' % out_dir)
    if failures:
        print('%d check(s) failed' % len(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
