"""Check the vcc-as-data rule of tools/startup/mesh2hlsl.py (MeshStage.lower_vcc_data): where vcc holds no lane mask,"""
import os
import struct
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


def VMOV(d, s):
    return I('v_mov_b32_e32 v%d, %s' % (d, s), 'vop1', 'v_mov_b32_e32', d=d, s0=s)


VCCLO_T7 = I('s_add_u32 vcc_lo, s22, 7', 'sop2', 's_add_u32', d=106, s0='s22', s1='7')
VCCLO_T = I('s_mov_b32 vcc_lo, s22', 'sop1', 's_mov_b32', d=106, s0='s22')
CMP_N = I('v_cmp_gt_u32_e32 vcc_lo, s21, v0', 'vopc', 'v_cmp_gt_u32_e32', s0='s21', s1='v0')
P, EP = E.PROLOGUE, E.EPILOGUE


def vop2(text, mnemonic, d, s0, s1):
    """VOP2 (v_cndmask_b32_e32: vdst = vcc ? vsrc1 : src0)."""
    def make():
        fields = {'d': d, 's0': E.src(s0)[0], 's1': E.src(s1)[0]}
        for op in range(64):
            w = (op << 25) | (fields['d'] << 17) | ((fields['s1'] - 256) << 9) | fields['s0']
            code = struct.pack('<I', w) + b'\0' * 8
            try:
                _, name, _, _ = E.D.decode_instruction(code, 0)
            except Exception:
                continue
            if name == mnemonic:
                return [w]
        raise KeyError(mnemonic)
    return (text, make)


V7 = VMOV(7, '7')
CND_V4 = vop2('v_cndmask_b32_e32 v4, 9, v4, vcc_lo', 'v_cndmask_b32_e32', 4, '9', 'v4')
CND_V4_7 = vop2('v_cndmask_b32_e32 v4, 0, v7, vcc_lo', 'v_cndmask_b32_e32', 4, '0', 'v7')
LSHL27 = I('s_lshl_b32 s27, vcc_lo, 2', 'sop2', 's_lshl_b32', d=27, s0='vcc_lo', s1='2')
CMP_10 = I('v_cmp_gt_u32_e32 vcc_lo, s23, v0', 'vopc', 'v_cmp_gt_u32_e32', s0='s23', s1='v0')
PROGRAMS = {
    'vccdata': ('ok', P + [VCCLO_T7, I('s_mov_b32 vcc_hi, 3', 'sop1', 's_mov_b32', d=107, s0='3'),
                           VMOV(1, 'vcc_lo'), VMOV(4, 'vcc_hi'), CMP_10, CND_V4] + EP),
    'vccspill': ('ok', P + [VCCLO_T, I('s_mov_b32 vcc_hi, 5', 'sop1', 's_mov_b32', d=107, s0='5'),
                            WL(40, 'vcc_lo', 0), WL(40, 'vcc_hi', 1),
                            I('s_mov_b64 vcc, 0', 'sop1', 's_mov_b64', d=106, s0='0'),
                            RL(25, 40, 0), RL(26, 40, 1), VMOV(1, 's25'), VMOV(4, 's26')] + EP),
    'vcczero': ('ok', P + [CMP_N, V7, CND_V4_7, E.CMPX_N, ('s_cbranch_execz J', 'execz'), VCCLO_T, ('J:', None),
                           VMOV(1, 'vcc_lo')] + EP),
    'vcclive': ('vcc used as a lane mask', P + [CMP_N, V7, CND_V4_7,
                                                I('s_cmp_eq_u32 s22, 0', 'sopc', 's_cmp_eq_u32', s0='s22', s1='0'),
                                                ('s_cbranch_scc1 J', 'scc1'), VCCLO_T, ('J:', None),
                                                VMOV(1, 'vcc_lo')] + EP),
    'vccmix': ('vcc used as a lane mask', P + [CMP_N, V7, CND_V4_7, VMOV(1, 'vcc_hi')] + EP),
    # a lane mask read as a 32-bit number by a scalar op whose result nothing reads (s27 and its SCC are dead)
    'vccdead': ('ok', P + [CMP_N, V7, CND_V4_7, LSHL27] + EP),
    'vccdeadreg': ('vcc used as a lane mask', P + [CMP_N, V7, CND_V4_7, LSHL27, VMOV(1, 's27')] + EP),
    'vccdeadscc': ('vcc used as a lane mask', P + [CMP_N, V7, CND_V4_7, LSHL27,
                                                   I('s_cselect_b32 s28, 1, 0', 'sop2', 's_cselect_b32', d=28,
                                                     s0='1', s1='0'), VMOV(1, 's28')] + EP),
}


def main():
    out_dir = sys.argv[sys.argv.index('--write') + 1] if '--write' in sys.argv else os.path.join(
        HERE, '..', 'out', 'startup_mesh_vcc_data')
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
        on = E.translate(path, {'X2W_MESH_VCC_DATA': '1'})
        off = E.translate(path, {'X2W_MESH_VCC_DATA': '0'})
        if want == 'ok':
            check('%s: translates with the rule on' % name, on.startswith('OK'), on[:300])
            if name == 'vccdata':
                check('vccdata: vcc_lo / vcc_hi are s106 / s107, the later lane mask stays vcc',
                      's106 = s22 + 0x00000007u;' in on and 'v1 = s106;' in on and 's107 = 0x00000003u;' in on
                      and 'v4 = 0x00000003u;' in on and 'vcc = ((0x0000000au > v0)' in on, on[:4000])
                check('vccdata: negative control X2W_MESH_VCC_DATA=0 gives the base form (refused or cut to the bit)',
                      off.startswith('REFUSED') or off.startswith('CRASH') or 'vcc = (s22 + 0x00000007u) & 1;' in off,
                      off[:300])
            elif name == 'vccspill':
                # the fused 64-bit spill of s[106:107] copies both words (not a lane mask: no `& 1`)
                check('vccspill: the vcc pair is spilled and reloaded through s[1000:1001] as two words',
                      's1000 = s106;' in on and 's1001 = 0x00000005u;' in on and 's25 = s1000;' in on
                      and 's26 = s1001;' in on, on[:4000])
                pairs_off = E.translate(path, {'X2W_MESH_DATA_PAIRS': '0'})
                check('vccspill: negative control X2W_MESH_DATA_PAIRS=0 cuts the 64-bit data move to the lane bit',
                      's1000 = (s106) & 1;' in pairs_off, pairs_off[:300])
                check('vccspill: negative control X2W_MESH_VCC_DATA=0 refuses (vcc_hi as data)',
                      off.startswith('REFUSED') or off.startswith('CRASH'), off[:300])
            elif name == 'vccdead':
                check('vccdead: the mask keeps vcc (no s106 rename)', 's106' not in on and 'vcc = ' in on, on[:4000])
            else:
                check('vcczero: the data def is s106, the mask stays vcc', 's106 = s22;' in on and 'v1 = s106;' in on,
                      on[:4000])
                check('vcczero: negative control X2W_MESH_VCC_DATA=0 cuts the value to the lane bit',
                      'vcc = (s22) & 1;' in off, off[:300])
        else:
            check('%s: refused (%s) with the rule on' % (name, want),
                  on.startswith('REFUSED') and want in on.splitlines()[0], on[:200])
    print('programs written to %s (difftest: python -B difftest.py --program <dir>/vccdata.bin)' % out_dir)
    if failures:
        print('%d check(s) failed' % len(failures))
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
