from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import mesh2hlsl as M


def accepts(tail):
    code = ['s_bfe_u32 s67, s67, 0x10000e', 's_add_u32 s67, s67, 1', 's_max_u32 s67, s67, 1',
            's_cbranch_scc0 label_0x14', 'v_cvt_f32_i32_e32 v1, s67', tail, 's_endpgm']
    stage = M.MeshStage.__new__(M.MeshStage)
    stage.pc = 0
    stage.ins = [(i * 4, 4, line.split()[0], line) for i, line in enumerate(code)]
    return stage.descriptor_numeric_field('s_bfe_u32', ['s67', 's67', '0x10000e'])


if __name__ == '__main__':
    assert accepts('s_mov_b32 s67, 0')
    assert not accepts('s_mov_b32 s11, s67')
    assert not accepts('s_buffer_load_dword s12, s[64:67], 0')
    print('PASS: numeric field use, escaped copy and descriptor use on another path')
