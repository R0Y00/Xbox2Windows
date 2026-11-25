"""Offline proofs for scalar literal addressing and SCC-dead SGPR exchange lowering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import mesh2hlsl as M


def stage(*asm):
    s = M.MeshStage.__new__(M.MeshStage)
    s.ins = [(4 * i, 4, line.split()[0], line) for i, line in enumerate(asm)]
    s.pc = s.ins[-1][0]
    s.sym = {}
    return s


def check_literals():
    prefix = ['s_movk_i32 s8, 0xffff', 's_mov_b32 s12, s8']
    end = 's_endpgm'
    assert stage(*prefix, end).copied_constant(12) == 0xffffffff
    assert stage(*prefix, 's_addk_i32 s12, 1', end).copied_constant(12) is None
    assert stage('s_cbranch_scc0 label_0xc', *prefix, end).copied_constant(12) is None
    assert stage('s_mov_b32 s8, 24', 's_cbranch_scc0 label_0x10', 's_mov_b32 s8, 16',
                 's_branch label_0x10', end).copied_constant(8) is None
    assert stage('s_mov_b32 s8, 24', 'v_cmp_gt_f32_e64 s7, 0, v5', end).copied_constant(8) is None

    extract = 's_bfe_u32 s8, s2, 0x30004'
    shift = 's_lshl_b32 s8, s8, 2'
    load = 's_buffer_load_dword s4, s[12:15], s8'
    assert stage(extract, shift, load, load).scalar_literal_index_bits('s8') == 3
    assert stage('s_branch label_0x8', extract, shift, load).scalar_literal_index_bits('s8') is None
    assert stage(extract, shift, load.replace('s4,', 's8,'), load).scalar_literal_index_bits('s8') is None
    assert stage(extract, shift.replace(', 2', ', 4'), load).scalar_literal_index_bits('s8') is None


def check_exchanges():
    a = 's_xor_b32 s4, s4, s7'
    b = 's_xor_b32 s7, s7, s4'
    neutral = 'v_mul_f32_e32 v1, v3, v2'
    s = stage(a, neutral, b, a, 's_endpgm')
    assert s.lower_pair_swaps() == {12}
    regs = {'s4': 0xcabb1e, 's7': 0xabcdef}
    for _, _, name, asm in s.ins:
        if name == 's_mov_b32':
            d, source = M.parse(asm)[1]
            regs[d] = regs[source]
    assert regs['s4'] == 0xabcdef and regs['s7'] == 0xcabb1e
    assert not stage(a, 's_mov_b32 s6, s7', b, a).lower_pair_swaps()
    assert not stage('s_branch label_0x8', a, b, a).lower_pair_swaps()
    assert not stage(a, 's_mov_b32 s5, src_scc', b, a).lower_pair_swaps()
    s = stage(a, b, a, 's_cbranch_scc1 label_0x10', 's_endpgm')
    s.swap_scc = s.lower_pair_swaps()
    try:
        s.check_scc(set(), set())
    except M.Unsupported:
        pass
    else:
        raise AssertionError('exchange with live SCC was accepted')


def check_conflict_copy():
    s = stage('s_mov_b64 s[16:17], s[64:65]')
    s.sym = {64: ('conflict',), 65: ('conflict',)}
    s.masks, s.used_s, s.exec_off = set(), set(), {16, 17}
    s.execzero = {16: 1, 17: 1}
    lines = []
    s.emit = lines.append
    s.scalar_inner('s_mov_b64', ['s[16:17]', 's[64:65]'])
    assert lines == ['s16 = s64;', 's17 = s65;']
    assert not s.exec_off and not s.execzero
    for reg in (16, 17):
        assert s.sym[reg] == ('conflict',)
        try:
            s.S(reg)
        except M.Unsupported as refusal:
            assert 'holding conflict' in str(refusal)
        else:
            raise AssertionError('unresolved SGPR copy accepted as a number')

    s.sym = {65: ('conflict',)}
    s.masks = {64}
    lines.clear()
    s.scalar_inner('s_mov_b64', ['s[16:17]', 's[64:65]'])
    assert 16 in s.masks and s.sym[17] == ('conflict',)
    assert lines[-1] == 's17 = s65;'
    assert 's64' in lines[0] and '& 1' in lines[0]

    s.sym = {64: ('conflict',), 65: ('conflict',)}
    s.masks.clear()
    lines.clear()
    s.scalar_inner('s_mov_b64', ['s[65:66]', 's[64:65]'])
    assert lines == ['s66 = s65;', 's65 = s64;']


if __name__ == '__main__':
    check_literals()
    check_exchanges()
    check_conflict_copy()
    print('Scalar literal copy/bounds and XOR exchange controls passed')
