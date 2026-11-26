from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import mesh2hlsl as M


def proof(change=None):
    assembly = ['s_nop 0', 's_load_dwordx8 s[88:95], s[12:13], s22',
                's_mov_b64 s[70:71], s[90:91]', 's_mov_b64 s[68:69], s[88:89]',
                's_mov_b64 s[72:73], s[92:93]', 's_mov_b64 s[74:75], s[94:95]',
                's_nop 0', 'image_load v0, v1, s[68:75] dmask:0x1 dim:SQ_RSRC_IMG_1D']
    if change:
        assembly[change[0]] = change[1]
    stage = M.MeshStage.__new__(M.MeshStage)
    stage.ins = [(4*i,4,s.split()[0],s) for i,s in enumerate(assembly)]
    stage.pc = 28
    stage._descriptor_loads = {4: 3}
    return stage.copied_heap_descriptor(('s',68,75))


if __name__ == '__main__':
    assert proof() == (('heapdyn',3),0)
    assert proof((6,'s_and_b32 s74, s74, 0xff')) is None
    assert proof((4,'s_mov_b64 s[72:73], s[90:91]')) is None
    assert proof((0,'s_cbranch_scc0 label_0xc')) is None
    print('PASS: exact copied descriptor; edits, reordered words and alternate entries rejected')
