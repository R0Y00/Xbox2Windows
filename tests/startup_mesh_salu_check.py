"""Check plain scalar ALU forms of tools/startup/gfx2hlsl.py (SCA) on the mesh stage: s_sext_i32_i8 / s_sext_i32_i16,"""
import os
import re
import subprocess
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


def translate(path, prelude=''):
    child = ('import sys; sys.path[:0] = [%r, %r]\n'
             'import gfx2hlsl as G, mesh2hlsl as M\n%s\n'
             'code = open(sys.argv[1], "rb").read()\n'
             'try:\n'
             '    p = M.MeshPipeline(code, None, M.synth_root(code), user_sgpr=16)\n'
             '    sys.stdout.write("OK\\n" + p.hlsl_ms())\n'
             'except G.Unsupported as e:\n'
             '    sys.stdout.write("REFUSED %%s\\n" %% e)\n'
             % (os.path.join(HERE, '..', 'tools'), os.path.join(HERE, '..', 'tools', 'startup'), prelude))
    r = subprocess.run([sys.executable, '-B', '-c', child, path], capture_output=True, text=True)
    return 'CRASH ' + (r.stderr.strip().splitlines() or ['?'])[-1] if r.returncode else r.stdout


def hlsl_eval(expr, x):
    """Evaluate the emitted `uint(int(sN << k) >> k)` for the 32-bit value x (int shifts are arithmetic in HLSL)."""
    m = re.fullmatch(r'uint\(int\(\(?s\d+\)? << (\d+)\) >> (\d+)\)', expr)
    if not m or m.group(1) != m.group(2):
        return None
    k = int(m.group(1))
    v = (x << k) & 0xffffffff
    v = v - (1 << 32) if v & 0x80000000 else v
    return (v >> k) & 0xffffffff


PROGRAMS = {
    'sext': E.PROLOGUE + [
        I('s_add_u32 s24, s22, 0x7ff0', 'sop2', 's_add_u32', d=24, s0='s22', s1='0x7ff0'),
        I('s_sext_i32_i16 s25, s24', 'sop1', 's_sext_i32_i16', d=25, s0='s24'),
        I('s_add_u32 s26, s22, 0x70', 'sop2', 's_add_u32', d=26, s0='s22', s1='0x70'),
        I('s_sext_i32_i8 s27, s26', 'sop1', 's_sext_i32_i8', d=27, s0='s26'),
        I('v_mov_b32_e32 v1, s25', 'vop1', 'v_mov_b32_e32', d=1, s0='s25'),
        I('v_mov_b32_e32 v4, s27', 'vop1', 'v_mov_b32_e32', d=4, s0='s27')] + E.EPILOGUE,
    'bcnt': E.PROLOGUE + [
        I('s_or_b32 s24, s22, 0xf00', 'sop2', 's_or_b32', d=24, s0='s22', s1='0xf00'),
        I('s_bcnt1_i32_b32 s25, s24', 'sop1', 's_bcnt1_i32_b32', d=25, s0='s24'),
        I('s_and_b32 s27, s22, 0x100', 'sop2', 's_and_b32', d=27, s0='s22', s1='0x100'),
        I('s_bcnt1_i32_b32 s28, s27', 'sop1', 's_bcnt1_i32_b32', d=28, s0='s27'),
        I('s_cselect_b32 s26, 7, 9', 'sop2', 's_cselect_b32', d=26, s0='7', s1='9'),
        I('s_lshl_b32 s29, s25, 16', 'sop2', 's_lshl_b32', d=29, s0='s25', s1='16'),
        I('v_mov_b32_e32 v1, s29', 'vop1', 'v_mov_b32_e32', d=1, s0='s29'),
        I('v_mov_b32_e32 v4, s26', 'vop1', 'v_mov_b32_e32', d=4, s0='s26')] + E.EPILOGUE,
    'bcntmask': E.PROLOGUE + [
        I('v_cmp_gt_u32_e64 s30, s21, v0', 'vop3', 'v_cmp_gt_u32_e64', d=30, s0='s21', s1='v0'),
        I('s_bcnt1_i32_b32 s25, s30', 'sop1', 's_bcnt1_i32_b32', d=25, s0='s30'),
        I('v_mov_b32_e32 v1, s25', 'vop1', 'v_mov_b32_e32', d=1, s0='s25')] + E.EPILOGUE,
}


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_salu')
    os.makedirs(out_dir, exist_ok=True)
    for name, items in PROGRAMS.items():
        code, texts = E.assemble(items)
        path = os.path.join(out_dir, name + '.bin')
        with open(path, 'wb') as f:
            f.write(code)
        got = E.decoded(code)
        check('%s: decodes to the intended %d instructions' % (name, len(texts)),
              [t for _, t in got] == [t for _, t in texts],
              '\n'.join('%5x  %-40s %s' % (o, t, g[1]) for (o, t), g in zip(texts, got) if g[1] != t))
        on = translate(path)
        if name == 'bcnt':
            line = re.search(r'^\s*s25 = (.*);', on, re.M) if on.startswith('OK') else None
            check('bcnt: translates, s25 = countbits(...)', bool(line) and line.group(1).startswith('countbits('), on[:3000])
            check('bcnt: SCC of the second count is set from its result', 'scc = (s28 != 0);' in on or 'scc = s28 != 0;' in on
                  or re.search(r'scc = .*countbits', on) is not None, on[:3000])
            off = translate(path, "del G.SCA['s_bcnt1_i32_b32']")
            check('bcnt: negative control (no SCA entry) refuses s_bcnt1_i32_b32',
                  off.startswith('REFUSED') and 's_bcnt1_i32_b32' in off.splitlines()[0], off[:200])
            continue
        if name == 'bcntmask':
            check('bcntmask: refused (a count of a lane mask)', on.startswith('REFUSED') and 'lane mask' in on.splitlines()[0],
                  on[:200])
            continue
        check('sext: translates', on.startswith('OK'), on[:300])
        e16 = re.search(r'^\s*s25 = (uint\(.*\));', on, re.M)
        e8 = re.search(r'^\s*s27 = (uint\(.*\));', on, re.M)
        check('sext: s25 / s27 are the arithmetic-shift forms', bool(e16 and e8), on[:3000])
        if e16 and e8:
            want16 = [0x7ff0, 0xffff8004, 0xffff8018, 0xffff802c]
            want8 = [0x70, 0xffffff84, 0xffffff98, 0xffffffac]
            got16 = [hlsl_eval(e16.group(1), t + 0x7ff0) for t in (0, 20, 40, 60)]
            got8 = [hlsl_eval(e8.group(1), t + 0x70) for t in (0, 20, 40, 60)]
            check('sext: s_sext_i32_i16 gives %s' % [hex(w) for w in want16], got16 == want16,
                  '%s from %s' % (got16, e16.group(1)))
            check('sext: s_sext_i32_i8 gives %s' % [hex(w) for w in want8], got8 == want8,
                  '%s from %s' % (got8, e8.group(1)))
        off = translate(path, "del G.SCA['s_sext_i32_i16'], G.SCA['s_sext_i32_i8']")
        check('sext: negative control (no SCA entries) refuses s_sext_i32_i16',
              off.startswith('REFUSED') and 's_sext_i32_i16' in off.splitlines()[0], off[:200])
    print('programs written to %s (difftest: python -B difftest.py --program <dir>/sext.bin)' % out_dir)
    if failures:
        print('%d check(s) failed' % len(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
