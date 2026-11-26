"""The vertex stage's entry v3 is the instance id (SV_InstanceID); v1 and v2 have no lowering.

Synthetic code: v1 = float(v3), v2 = float(v0), v4 = 1.0; exp pos0 v1, v2, v4, v4 done. The title's vertex stages
read v3 before any write exactly when their input signature declares SV_INSTANCEID (measured on the archive's
distinct vertex containers: 441 of 441 with it, none without) and never read v1 or v2; v3 is the instance index
without the start instance (D3D SV_InstanceID; the title adds its own offsets from constants).
Checks:
  v3 read:   body(vid, iid) with v3 = iid, main takes SV_InstanceID and passes it on every rectangle-list path,
             and FXC compiles the pair;
  no v3:     a stage that reads only v0 keeps the old text (body(uint vid), no SV_InstanceID);
  refused:   a read of entry v1 or v2.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import PS_WORDS, root_blob  # noqa: E402


def cvt(dst, src):
    return 0x7e000000 | dst << 17 | 6 << 9 | src    # v_cvt_f32_u32_e32 v<dst>, <src>


def program(first_src):
    words = [cvt(1, first_src), cvt(2, 256 + 0), 0x7e0802f2, 0xf80008cf, 0x04040201, 0xbf810000]
    return struct.pack('<%dI' % len(words), *words)


def main():
    vs = program(256 + 3)
    text = [x[3] for x in G.decode(vs)]
    assert text[0] == 'v_cvt_f32_u32_e32 v1, v3' and text[3] == 'exp pos0 v1, v2, v4, v4 done', text
    hlsl = G.Pipeline(vs, PS_WORDS, root_blob()).hlsl_vs()
    for line in ('VSOut body(uint vid, uint iid) {', '    v3 = iid;', 'VSOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {',
                 'return body(vid, iid);', 'return body(first + corner, iid);', 'return body(first, iid);',
                 'return body(first + 2, iid);', 'VSOut p = body(first, iid), q = body(first + 1, iid), r = body(first + 2, iid), o;'):
        assert line in hlsl, (line, hlsl)
    assert hlsl.index('    v0 = vid;') < hlsl.index('    v3 = iid;') < hlsl.index('float(v3)'), hlsl
    with tempfile.TemporaryDirectory() as d:
        result = G.translate_to(d, 1, vs, PS_WORDS, root_blob())
        assert result == 'ok', result
        assert os.path.getsize(os.path.join(d, '%016x.vs.cso' % 1)) > 0
    plain = G.Pipeline(program(256 + 0), PS_WORDS, root_blob()).hlsl_vs()
    assert 'VSOut body(uint vid) {' in plain and 'SV_InstanceID' not in plain and 'iid' not in plain, plain
    for reg in (1, 2):
        words = program(256 + reg)
        try:
            G.Pipeline(words, PS_WORDS, root_blob()).hlsl_vs()
        except G.Unsupported as error:
            assert 'vertex stage reads entry v%d' % reg in str(error), str(error)
        else:
            raise AssertionError('accepted an entry read of v%d' % reg)
    print('vs instance id: v3 = SV_InstanceID through every rectangle-list path, FXC ok; a v0-only stage unchanged; '
          'entry v1 and v2 refused')


if __name__ == '__main__':
    main()
