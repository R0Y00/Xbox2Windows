"""Check scalar values carried out of per-lane mesh loops."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import mesh2hlsl as M


PROGRAM = (
    's_branch label_0xc',
    's_mov_b64 exec, s[8:9]',
    's_mov_b32 s5, s4',
    's_mov_b32 s4, 1',
    's_mov_b32 s4, s5',
    's_and_b64 exec, s[12:13], exec',
    's_andn2_b64 s[8:9], s[8:9], exec',
    's_cbranch_scc1 label_0x4',
    's_mov_b32 s2, s4',
    's_mov_b64 exec, s[6:7]',
    'v_mov_b32_e32 v2, s2',
    's_endpgm',
)


def proof(changes=None):
    scene = M.MeshStage.__new__(M.MeshStage)
    instructions = [(changes or {}).get(i, asm) for i, asm in enumerate(PROGRAM)]
    scene.ins = [(i * 4, 4, asm.split()[0], asm) for i, asm in enumerate(instructions)]
    return scene.perlane_loop(4, 28)


if __name__ == '__main__':
    assert proof() is None
    assert 's4 written' in proof({0: 's_cbranch_scc0 label_0xc'})
    assert 's4 written' in proof({4: 's_mov_b32 s4, s3'})
    assert 's4 written' in proof({3: 's_xor_b32 s5, s5, 1'})
    assert 'no s_mov_b64 exec restore' in proof({8: 'v_mov_b32_e32 v4, s4'})
    assert 'no s_mov_b64 exec restore' in proof({8: 's_mov_b32 s2, exec_lo'})
    assert 'restored exec copy' in proof({8: 's_mov_b32 s6, s4'})
    print('Scalar-only loop tail and saved scalar checks passed.')
