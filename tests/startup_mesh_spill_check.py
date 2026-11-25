"""Check the SGPR spill lowering of tools/startup/mesh2hlsl.py (MeshStage.lower_spills): v_writelane_b32 /"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import startup_mesh_exec_restore_check as E  # noqa: E402  (encoder, prologue, epilogue, translate())
from startup_mesh_exec_restore_check import I  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH  ' + detail}")
    if not condition:
        failures.append(label)


def WL(v, s, lane):
    return I('v_writelane_b32 v%d, %s, %s' % (v, s, lane), 'vop3', 'v_writelane_b32', d=v, s0=s, s1=str(lane))


def RL(s, v, lane):
    return I('v_readlane_b32 s%d, v%d, %s' % (s, v, lane), 'vop3', 'v_readlane_b32', d=s, s0='v%d' % v, s1=str(lane))


def MOVS(d, s):
    return I('s_mov_b32 s%d, %s' % (d, s), 'sop1', 's_mov_b32', d=d, s0=s)


def VMOV(d, s):
    return I('v_mov_b32_e32 v%d, %s' % (d, s), 'vop1', 'v_mov_b32_e32', d=d, s0=s)


EXECZ_AFTER = ('s_cbranch_execz AFTER', 'execz')
CLEAR30 = I('s_mov_b64 s[30:31], 0', 'sop1', 's_mov_b64', d=30, s0='0')
P, EP = E.PROLOGUE, E.EPILOGUE
PROGRAMS = {
    'spill32': ('ok', P + [I('s_mul_i32 s24, s22, 3', 'sop2', 's_mul_i32', d=24, s0='s22', s1='3'),
                           WL(40, 's24', 5), WL(40, '7', 6), MOVS(24, '0'), RL(25, 40, 5), RL(26, 40, 6),
                           VMOV(1, 's25'), VMOV(4, 's26')] + EP),
    'spillpair': ('ok', P + [I('s_mul_i32 s24, s22, 3', 'sop2', 's_mul_i32', d=24, s0='s22', s1='3'),
                             I('s_add_u32 s25, s22, 1', 'sop2', 's_add_u32', d=25, s0='s22', s1='1'),
                             WL(40, 's24', 2), WL(40, 's25', 3), I('s_mov_b64 s[24:25], 0', 'sop1', 's_mov_b64', d=24,
                                                                  s0='0'),
                             RL(26, 40, 2), RL(27, 40, 3), VMOV(1, 's26'), VMOV(4, 's27')] + EP),
    'spillexec': ('ok', P + [E.CMPX_N, E.SAVE, EXECZ_AFTER, WL(40, 's30', 0), WL(40, 's31', 1), CLEAR30, E.CMPX_T,
                             E.V1_9, RL(30, 40, 0), RL(31, 40, 1), E.RESTORE, E.V4_7] + EP),
    'spillwave': ('ok', P + [WL(40, 's23', 2), E.CMPX_N, ('s_cbranch_execz J', 'execz'), WL(40, '11', 2), ('J:', None),
                             I('s_mov_b64 exec, -1', 'sop1', 's_mov_b64', d=126, s0='-1'), RL(25, 40, 2),
                             VMOV(1, 's25')] + EP),
    'datause': ('also used by v_mov_b32', P + [WL(40, 's23', 0), VMOV(1, 'v40')] + EP),
    'lanesgpr': ('is not a constant lane', P + [WL(40, 's23', 's22'), RL(25, 40, 0), VMOV(1, 's25')] + EP),
    'unwritten': ('which no v_writelane_b32 writes', P + [
        I('s_cmp_eq_u32 s22, 0', 'sopc', 's_cmp_eq_u32', s0='s22', s1='0'), ('s_cbranch_scc1 J', 'scc1'),
        WL(40, 's23', 3), ('J:', None), RL(25, 40, 3), VMOV(1, 's25')] + EP),
    'vcchalf': ('v_writelane_b32 of vcc_lo alone', P + [
        VMOV(6, 's22'), I('v_cmp_lt_u32_e32 vcc_lo, 30, v6', 'vopc', 'v_cmp_lt_u32_e32', s0='30', s1='v6'),
        WL(40, 'vcc_lo', 0), RL(25, 40, 0), VMOV(1, 's25')] + EP),
    'spillhalves': ('ok', P + [E.CMPX_N, E.SAVE, EXECZ_AFTER, WL(40, 's30', 0), WL(40, 's31', 1), CLEAR30, E.CMPX_T,
                               E.V1_9, RL(30, 40, 0), ('K:', None), RL(31, 40, 1), E.RESTORE, E.V4_7] + EP),
    'splitpair': ('exec written by s_mov_b64 inside the per-lane skipped range', P + [
        E.CMPX_N, E.SAVE, EXECZ_AFTER, WL(40, 's30', 0), WL(40, 's31', 1), CLEAR30, E.CMPX_T, E.V1_9,
        RL(30, 40, 0), RL(33, 40, 1), E.RESTORE, E.V4_7] + EP),
}
# the reason the translator gives without the lowering (the base has no v_writelane_b32 / v_readlane_b32)
NO_LOWERING = 'REFUSED v_writelane_b32'
NO_EXEC = 'REFUSED exec written by s_mov_b64 inside the per-lane skipped range'


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_spill')
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
        on = E.translate(path, {'X2W_MESH_SPILLS': '1'})
        if want == 'ok':
            check('%s: translates with the lowering on' % name, on.startswith('OK'), on[:300])
            check('%s: no v_writelane / v_readlane left in the HLSL' % name, 'lane_b32' not in on, '')
            if name == 'spill32':
                # (lane 6 holds a constant: the base translator's constant tracking carries it through the slot)
                check('spill32: slots s1005 / s1006 assigned and read back',
                      's1005 = s24;' in on and 's25 = s1005;' in on and 's1006 = 0x00000007u;' in on
                      and 's26 = 0x00000007u;' in on, on[:2000])
            if name == 'spillpair':
                # a plain data pair, fused into one 64-bit spill and reload: both words survive (MeshStage.scalar_inner
                # with X2W_MESH_DATA_PAIRS on; off, the high word is lost and the low word cut to bit 0)
                check('spillpair: s[24:25] spilled to s[1002:1003] and reloaded into s[26:27] as two words',
                      's1002 = s24;' in on and 's1003 = s25;' in on and 's26 = s1002;' in on and 's27 = s1003;' in on,
                      on[:3000])
                poff = E.translate(path, {'X2W_MESH_DATA_PAIRS': '0'})
                check('spillpair: negative control X2W_MESH_DATA_PAIRS=0 cuts the pair to the lane bit',
                      's1002 = (s24) & 1;' in poff, poff[:300])
            if name == 'spillexec':
                check('spillexec: the 64-bit spill and reload are one move each, exec restored from s30',
                      's1000 = (s30) & 1;' in on and 's30 = (s1000) & 1;' in on and 'exec = (s30) & 1;' in on,
                      on[:3000])
            if name == 'spillwave':
                check('spillwave: the slot written in the skipped range goes through the group-uniform branch',
                      'x2w_any' in on and 's1002 = ' in on, on[:3000])
            off = E.translate(path, {'X2W_MESH_SPILLS': '0'})
            # without the lowering SPILLEXEC stops where the class's programs stopped: the skipped-range check sees
            # the restore from s[30:31], which the unlowered v_readlane does not make a copy of the saved mask
            expect = NO_EXEC if name in ('spillexec', 'spillhalves') else NO_LOWERING
            check('%s: negative control X2W_MESH_SPILLS=0 refuses again (%s)' % (name, expect),
                  off.splitlines()[0].startswith(expect), off[:200])
        else:
            check('%s: refused (%s) with the lowering on' % (name, want),
                  on.startswith('REFUSED') and want in on.splitlines()[0], on[:200])
    print('programs written to %s (difftest: python -B difftest.py <dir>/spillexec.bin <harness> --groups 4 --seed 1)'
          % out_dir)
    if failures:
        print('%d check(s) failed' % len(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
