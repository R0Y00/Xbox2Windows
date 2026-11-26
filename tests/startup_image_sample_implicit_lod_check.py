"""An implicit-LOD image_sample in branched pixel-stage code (X2W_PS_IMPLICIT_LOD=1).

image_sample takes its LOD from derivatives of the quad's address VGPRs. Without the gate, a sample after a branch
(or with exec modified) is lowered as SampleLevel(..., 0): mip 0 whatever the derivatives say. With the gate it is
Sample in a pixel stage, never level 0: before the first branch as before; in branched code on the assumption
that the four lanes of each quad take one path (manifest `quaddivergent 1`, as image_sample_b under
X2W_PS_QUAD_DIVERGENT); in a loop that is not a proven waterfall it is refused (lanes of a quad can sit in
different iterations of the dispatch loop).

Programs: tests/startup_image_3d_check.py WORDS_2D (straight line), the same with `s_cbranch_execz` to the next
instruction before the sample (branched, same result), and with `s_cmp_eq_u32 s9, 0; s_cbranch_scc0` back to the
sample (a loop). Negative control: gate off keeps every output as it was.
"""
import os
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import WORDS_2D, VS_WORDS, root_blob  # noqa: E402

SAMPLE = bytes.fromhex('080f80f00000c400')  # image_sample v[0:3], v[0:1], s[16:23], s[24:27] dmask:0xf dim:2D
EXECZ = struct.pack('<I', 0xbf880000)       # s_cbranch_execz to the next instruction
LOOPBACK = struct.pack('<II', 0xbf060000 | 128 << 8 | 9, 0xbf840000 | 0xfffc)  # s_cmp_eq_u32 s9, 0; s_cbranch_scc0 back
BRANCH = WORDS_2D.replace(SAMPLE, EXECZ + SAMPLE)
LOOP = WORDS_2D.replace(SAMPLE, EXECZ + SAMPLE + LOOPBACK)
IMPLICIT = 't = r0.Sample(q0, float2(F(v0), F(v1)));'
LEVEL0 = 't = r0.SampleLevel(q0, float2(F(v0), F(v1)), 0);'


def translate(words, gate):
    os.environ['X2W_PS_IMPLICIT_LOD'] = '1' if gate else '0'
    try:
        pipe = G.Pipeline(VS_WORDS, words, root_blob())
        return [line.strip() for line in pipe.hlsl_ps().splitlines() if line.strip().startswith('t = ')], \
            pipe.manifest().splitlines()
    finally:
        os.environ.pop('X2W_PS_IMPLICIT_LOD', None)


def main():
    text = [a for _, _, _, a in G.decode(LOOP)]
    assert text[-6:-2] == ['image_sample v[0:3], v[0:1], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
                           's_cmp_eq_u32 s9, 0', 's_cbranch_scc0 label_0x34', 's_waitcnt vmcnt(0)'], text
    assert text[-7] == 's_cbranch_execz label_0x34', text
    # gate off: unchanged (the level-0 fallback after a branch)
    assert translate(WORDS_2D, False)[0] == [IMPLICIT]
    assert translate(BRANCH, False)[0] == [LEVEL0]
    assert translate(LOOP, False)[0] == [LEVEL0]
    # gate on
    calls, manifest = translate(WORDS_2D, True)
    assert calls == [IMPLICIT] and 'quaddivergent 1' not in manifest, manifest
    calls, manifest = translate(BRANCH, True)
    assert calls == [IMPLICIT] and 'quaddivergent 1' in manifest, (calls, manifest)
    try:
        translate(LOOP, True)
    except G.Unsupported as error:
        assert 'image_sample at 0x34 with implicit derivatives in a loop' in str(error), str(error)
    else:
        raise AssertionError('implicit sample in a loop accepted')
    print('implicit LOD: gate off unchanged (level 0 after a branch); gate on: Sample before and after a branch '
          '(quaddivergent 1 after it), refused in a loop')


if __name__ == '__main__':
    main()
