"""Check the first-active-lane rule of tools/startup/mesh2hlsl.py (MeshStage.first_active_broadcast, NeedGroupRange):
v_readfirstlane_b32 of a lane-dependent VGPR under a partial exec reads the wave's lowest ACTIVE lane.

usage: startup_mesh_first_lane_check.py [--write <dir>]

Hand-assembled gfx1013 NGG programs (no title bytes), built with the encoder of tests/startup_mesh_exec_restore_check.py.
Prologue as there (T = 20 * (g & 3) in s22, N = 40 + 60 * (g & 1) in s21, v1 = 5, v4 = 0; v0 = the thread's index in
the group, 0..127); the epilogue sets exec = -1 and exports (v1, v0, v4, v0) as param0.

Hardware meaning (RDNA 2 ISA V_READFIRSTLANE_B32: the value of the lowest lane whose EXEC bit is set; V_CMPX writes
EXEC = EXEC & compare; S_CBRANCH_EXECZ is taken by a wave whose EXEC is 0):
  PARTIAL (accept): exec = (T <= v0); v7 = 2 * v0; s24 = readfirstlane(v7); v1 = s24.
            expected: wave 0: active lanes v0 >= T, first = T, v1 = 2T; wave 1: first = lane 0 (v0 = 64), v1 = 128;
            v1 = 5 in the inactive lanes.
  RANGE   (accept): exec = (v0 < N); execz -> AFTER; exec &= (T <= v0); v7 = 2 * v0; s24 = readfirstlane(v7); v1 = s24.
            The read is inside a per-lane execz range: the range is run by the whole group (NeedGroupRange).
            expected: active lanes T <= v0 < N; wave 0: v1 = 2T; wave 1 (v0 >= 64): v1 = 128 when N = 100,
            the wave skips the range when N = 40; else 5.
  LOOP    (refuse): as PARTIAL with the read inside a two-trip loop (s_cbranch_scc1 back): a group barrier in a loop
            is not reached the same number of times by every thread when the waves' trip counts differ (the
            waterfall form is mesh4's lowering).
  VCCZ    (refuse): exec = (v0 < N); vcc = (30 < T) (uniform over the active lanes); vccz -> AFTER; then as RANGE:
            only execz ranges are run by the whole group.

Checks: decode; PARTIAL / RANGE translate with x2w_rfirst; negative control X2W_MESH_RFL_FIRST=0 refuses both
("lanes off"); LOOP and VCCZ refused. --write keeps the programs for mesh_work/difftest.py.
"""
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


CMPX_LE_T = I('v_cmpx_le_u32_e32 s22, v0', 'vopc', 'v_cmpx_le_u32_e32', s0='s22', s1='v0')
V7 = I('v_lshlrev_b32_e64 v7, 1, v0', 'vop3', 'v_lshlrev_b32_e64', d=7, s0='1', s1='v0')
RFL = I('v_readfirstlane_b32 s24, v7', 'vop1', 'v_readfirstlane_b32', d=24, s0='v7')
V1 = I('v_mov_b32_e32 v1, s24', 'vop1', 'v_mov_b32_e32', d=1, s0='s24')
P, EP = E.PROLOGUE, E.EPILOGUE
PROGRAMS = {
    'rflpartial': ('ok', P + [CMPX_LE_T, V7, RFL, V1] + EP),
    'rflrange': ('ok', P + [E.CMPX_N, ('s_cbranch_execz AFTER', 'execz'), CMPX_LE_T, V7, RFL, V1] + EP),
    'rflloop': ('lanes off inside a loop', P + [CMPX_LE_T, I('s_mov_b32 s30, 0', 'sop1', 's_mov_b32', d=30, s0='0'), ('H:', None),
                                              V7, RFL, I('s_add_u32 s30, s30, 1', 'sop2', 's_add_u32', d=30, s0='s30', s1='1'),
                                              I('s_cmp_lt_u32 s30, 2', 'sopc', 's_cmp_lt_u32', s0='s30', s1='2'),
                                              ('s_cbranch_scc1 H', 'scc1'), V1] + EP),
    'rflvccz': ('lanes off', P + [E.CMPX_N, I('v_mov_b32_e32 v6, s22', 'vop1', 'v_mov_b32_e32', d=6, s0='s22'),
                                  I('v_cmp_lt_u32_e32 vcc_lo, 30, v6', 'vopc', 'v_cmp_lt_u32_e32', s0='30', s1='v6'),
                                  ('s_cbranch_vccz AFTER', 'vccz'), CMPX_LE_T, V7, RFL, V1] + EP),
}


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_first_lane')
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
            check('%s: translates with the first-active-lane read' % name,
                  on.startswith('OK') and 's24 = x2w_rfirst(0u, wbase);' in on and 'x2w_rfe[0][tid] = exec & 1u;' in on,
                  on[:3000])
            if name == 'rflrange':
                check('rflrange: the execz range is run by the group (x2w_any publish)', 'x2w_any[0][wave] = 1u' in on,
                      on[:3000])
            off = E.translate(path, {'X2W_MESH_RFL_FIRST': '0'})
            check('%s: negative control X2W_MESH_RFL_FIRST=0 refuses (lanes off)' % name,
                  off.startswith('REFUSED') and 'lanes off' in off.splitlines()[0], off[:200])
        else:
            check('%s: refused (%s)' % (name, want), on.startswith('REFUSED') and want in on.splitlines()[0], on[:200])
    print('programs written to %s' % out_dir)
    if failures:
        print('%d check(s) failed' % len(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
