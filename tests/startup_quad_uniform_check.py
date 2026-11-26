"""Forward pixel-quad dispatch. Optional DXC and offscreen_check paths run partial-quad WARP draws."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import startup_quad_dpp_check as Q

G = Q.G


def reference(x, y, skipped):
    lane = (x & 1) + 2 * (y & 1)
    if lane == skipped:
        return 2000 + 8 * y + x
    # On entered lanes write a new VGPR value before reading three neighbors.
    # The skipped lane (including a helper across the triangle edge) retains its old value.
    def source(neighbor):
        px = (x & ~1) + (neighbor & 1)
        py = (y & ~1) + (neighbor >> 1)
        return 8 * py + px + (1000 if neighbor != skipped else 0)
    return source(lane ^ 1) + 2 * source(lane ^ 2) + 4 * source(skipped)


def draw_fixture(directory, compiler, renderer, skipped, wrong):
    s = G.Stage.__new__(G.Stage)
    reads = ['uint a = QuadReadAcrossX(v);', 'uint b = QuadReadAcrossY(v);',
             'uint c = QuadReadLaneAt(v, %du);' % skipped]
    s.uniform_dpp = set(reads)
    s.blocks = [(0, [], ('branch', 'lane == %du' % skipped, 2)),
                (1, ['v += 1000u;'] + reads + ['result = a + 2u * b + 4u * c;'], None),
                (2, ['++visits;'], ('branch', 'visits < 1u + lane', 2)),
                (3, [], ('end',))]
    expected = [reference(x, y, skipped) for y in range(8) for x in range(8)]
    src = directory / 'test.hlsl'
    obj = directory / 'test.cso'
    raw = directory / 'test.raw'
    src.write_text('static const uint ref[64] = {%s};\n' % ','.join(map(str, expected)) +
                   'float4 main(float4 uv : TEXCOORD0, float4 pos : SV_Position) : SV_Target {\n'
                   'uint x = uint(pos.x), y = uint(pos.y), p = 8*y + x;\n'
                   'uint lane = (x & 1u) + 2u*(y & 1u), v = p, result = 2000u + p, visits = 0u;\n' +
                   ('if (lane == %du) v += 3000u;\n' % skipped if wrong else '') +
                   '\n'.join(s.body()) +
                   '\nreturn result == ref[p] ? float4(0,1,0,1) : float4(1,0,0,1); }\n')
    subprocess.run([compiler, '-nologo', '-T', 'ps_6_0', '-E', 'main', '-Fo', obj, src], check=True)
    raw.unlink(missing_ok=True)
    run = subprocess.run([renderer, 'warp', '--check-pixel-quad-lod', obj, raw], capture_output=True)
    assert raw.exists(), (run.returncode, run.stdout, run.stderr)
    pixels = list(struct.iter_unpack('<4f', raw.read_bytes()))
    assert len(pixels) == 64
    for y in range(8):
        for x in range(8):
            lane = (x & 1) + 2 * (y & 1)
            expected_pixel = (-1,-1,-1,-1) if x+y > 6 else (1,0,0,1) if wrong and lane != skipped else (0,1,0,1)
            assert pixels[8*y+x] == expected_pixel, (skipped, wrong, x, y, pixels[8*y+x], expected_pixel)
    assert bool(run.returncode) == wrong, (run.stdout, run.stderr)


def main():
    os.environ.pop('X2W_PS_QUAD_DIVERGENT', None)
    os.environ['X2W_PS_QUAD_UNIFORM'] = '1'
    stage = Q.mesh_pipeline(Q.program(prefix=struct.pack('<I', 0xbf880002))).ps
    assert not stage.quad_divergent_used
    body = stage.body()
    assert all('    ' + stmt in body for stmt in stage.dpp_statements)
    for stmt in stage.dpp_statements:
        i = body.index('    ' + stmt)
        assert body[i-1] == '    }' and body[i+1].startswith('    if (pc == ')
    # Pixel termination before a shuffle can remove its source lane, even if all
    # remaining lanes reach the read. Reject both this and a locally nested read.
    for prefix in ('if (kill) discard;', 'if (mask) {'):
        old = stage.blocks
        stage.blocks = [(n, [prefix] + lines, term) for n, lines, term in old]
        try:
            stage.check_quad_ops()
        except G.Unsupported as error:
            assert 'after discard or inside a nested scope' in str(error), error
        else:
            raise AssertionError('accepted quad read after ' + prefix)
        finally:
            stage.blocks = old
    end = stage.ins[-1][0] + 8
    last_quad = max(stage.dpp_at)
    old_ins = stage.ins
    for target in (end, last_quad):
        stage.ins = old_ins + [(end, 4, 's_cbranch_scc1', 's_cbranch_scc1 label_0x%x' % target)]
        try:
            stage.check_quad_ops()
        except G.Unsupported as error:
            assert target == last_quad and 'loop' in str(error), error
        else:
            assert target == end
    stage.ins = old_ins
    if len(sys.argv) == 3:
        with tempfile.TemporaryDirectory() as directory:
            for skipped in range(4):
                for wrong in (False, True):
                    draw_fixture(Path(directory), *sys.argv[1:], skipped, wrong)
        print('Four lane masks, 28 covered and 36 clear pixels each, and four clobbered-lane controls passed.')


if __name__ == '__main__':
    main()
