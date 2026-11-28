"""A published native compute program sampling a cube texture through the stand-in's Dispatch (list slot 14).

usage: startup_compute_dispatch_cube_check.py <compile_dxil> <startup_stream_consumer> <offscreen_check> <dxc dir> <warp|hardware>
       startup_compute_dispatch_cube_check.py compile <compile_dxil> <dxc dir>

The program is synthetic gfx1013 code: 24 lanes per group, lane L sampling (image_sample_lz, dim cube, NSA form with
S = v5, T = v6, face = v4) the face L >> 2 at s = 1.25 + 0.5 (L & 1) and t = 1.25 + 0.5 ((L >> 1) & 1), that is the face
coordinates (2 s - 3, 2 t - 3) = (+-0.5, +-0.5), under an all-zero (point, wrap) sampler. Each of the 12 groups stores
its 24 results at word 24 * group + lane; the last 32 of the 320 output words keep the sentinel. The translator declares
the resource as a TextureCubeArray (DXIL kind 9), so the published manifest names the input `srv 0 texcubearr table 1 64`
and the output `uav 0 table 1 32`. The consumer (dispatch-cube) creates a 2x2 R32_FLOAT texture of six slices, texel
(x, y) of slice f = 100 + 4 f + x + 2 y, and a TEXTURECUBE view of it at table byte 64 (device slot 18).

The expected word of a lane is the texel its face coordinates name: face L >> 2, texel x from the sign of the first
coordinate and y from the sign of the second (the D3D sc/tc convention). The direction tables here only round-trip
that choice; what checks the orientation is the GPU: the translator's cube_axes on hardware (CUBE_HLSL's directions on
WARP) must send each lane to that texel, so an s/t swap, a sign error on a minor axis or a wrong face order change words.

Positive run: before the manifest exists the Dispatch is refused with "no translation" and the output keeps the
sentinel; after `compile_dxil --requests` all 320 words equal the reference, the debug layer reports no error or
corruption message, and exactly one Dispatch translates. Controls, each refused with one logged reason and an
untouched output: the title's view is a TEXTURE2DARRAY of the six slices (-array), the view has no description
(-undescribed), the TEXTURECUBE view is on a twelve-slice texture (-twelve), and the published line rewritten to
tex2da, tex2d (a manifest dimension the title's view is not) or texcube (a name the compute reader does not know).
Exit 77 when the adapter is not available, or, on hardware, when it has no wave32.

The translator's DXIL is [WaveSize(32)] (a 32-lane guest wave), which WARP cannot create. On warp the published
manifest is therefore kept (compile_dxil still publishes it from the request) and the program's <key>.cs.cso is replaced
by CUBE_HLSL below, the same program in HLSL for cs_6_0 with the face directions of direction(): that run covers the
manifest line, the reader and the debug layer on WARP, not the translator (hardware covers it). The TEXTURECUBEARRAY
rewrite of the view is not discriminated by any run here: with the rewrite removed, WARP and hardware gave the same
words and no debug-layer message.
"""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_dpp_mirror_check import vop2
from native_gather_lod_check import render
from native_scalar_lds_check import scalar, sop1
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a, write_request
from startup_gfx2hlsl_compute_check import SENTINEL, TGID_X, TGID_Y, TGID_Z, USER7, WORDS, root_blob

GROUP = (24, 1, 1)
STATE = state_a(GROUP, USER7 | TGID_X | TGID_Y | TGID_Z)
GROUPS = 12  # the consumer's Dispatch(3, 2, 2)
LANES = 24
MODE = 'dispatch-cube'
INPUT_LINE = 'srv 0 texcubearr table 1 64'
OUTPUT_LINE = 'uav 0 table 1 32'
REJECTED = 'an input texture view is not cube'


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def program():
    words = [
        sop1(3, 20, 6), sop1(3, 21, 128),          # s[20:21] = the descriptor table
        0xf40c060a, 0xfa000040,                    # s_load_dwordx8 s[24:31], s[20:21], 0x40: the cube SRV
        0xf408080a, 0xfa000020,                    # s_load_dwordx4 s[32:35], s[20:21], 0x20: the output
        *[sop1(3, 40 + i, 128) for i in range(4)],  # s[40:43] = 0: point filter, wrap
        vop2(27, 1, 129, 0),                       # v1 = lane & 1
        vop2(22, 2, 129, 0),                       # v2 = lane >> 1
        vop2(27, 2, 129, 2),                       # v2 &= 1
        vop2(22, 4, 130, 0),                       # v4 = lane >> 2, the face
        0x7e020d01, 0x7e040d02, 0x7e080d04,        # v_cvt_f32_u32 of v1, v2, v4
        vop2(8, 1, 240, 1), vop2(8, 2, 240, 2),    # v1, v2 *= 0.5
        vop2(3, 5, 255, 1), bits(1.25),            # v5 = S = 1.25 + v1
        vop2(3, 6, 255, 2), bits(1.25),            # v6 = T = 1.25 + v2
        scalar(0x26, 8, 8, 131),                   # s8 = 3 * group.y
        scalar(0x26, 9, 9, 134),                   # s9 = 6 * group.z
        scalar(0, 11, 7, 8), scalar(0, 11, 11, 9),  # s11 = flat group index
        scalar(0x26, 11, 11, 128 + LANES),         # s11 *= 24
        vop2(0x25, 8, 11, 0),                      # v8 = 24 * group + lane
        vop2(26, 9, 130, 8),                       # v9 = v8 << 2
        0xbf8cc07f,                                # s_waitcnt lgkmcnt(0)
        0xf09c011a, 5 | 7 << 8 | 6 << 16 | 10 << 21, 0x00000406,  # image_sample_lz v7, v5, s[24:31], s[40:43] cube, T v6, face v4
        0xbf8c3f70,                                # s_waitcnt vmcnt(0)
        0xe0701000, 0x80080709,                    # buffer_store_dword v7, v9, s[32:35], 0 offen
        0xbf810000,
    ]
    return struct.pack('<%dI' % len(words), *words)


# The program as HLSL: lane L samples face L >> 2 of the cube array at the direction of face coordinates (2 s - 3, 2 t - 3).
CUBE_HLSL = """TextureCubeArray<float4> r0 : register(t0);
RWByteAddressBuffer w0 : register(u0);
SamplerState q0 : register(s0);
[numthreads(24, 1, 1)] void main(uint3 g : SV_GroupID, uint lane : SV_GroupIndex)
{
    float sc = 2.0 * (1.25 + 0.5 * (lane & 1)) - 3.0, tc = 2.0 * (1.25 + 0.5 * ((lane >> 1) & 1)) - 3.0;
    static const float3 scale[6] = {float3(0, 0, -1), float3(0, 0, 1), float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0),
                                    float3(-1, 0, 0)};
    static const float3 down[6] = {float3(0, -1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1), float3(0, -1, 0),
                                   float3(0, -1, 0)};
    static const float3 major[6] = {float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1),
                                    float3(0, 0, -1)};
    const uint face = lane >> 2;
    const float3 direction = major[face] + sc * scale[face] + tc * down[face];
    const uint flat = g.x + 3 * g.y + 6 * g.z;
    w0.Store((flat * 24 + lane) * 4, asuint(r0.SampleLevel(q0, float4(direction, 0), 0).x));
}
"""


def texel_value(face, x, y):
    return 100 + 4 * face + x + 2 * y


def project(direction):
    """D3D cube mapping of a direction: (face, texel x, texel y) of a 2x2 face."""
    x, y, z = direction
    axis = max(range(3), key=lambda i: abs(direction[i]))
    major = direction[axis]
    face = 2 * axis + (major < 0)
    sc, tc = {0: (-z, -y), 1: (z, -y), 2: (x, z), 3: (x, -z), 4: (x, -y), 5: (-x, -y)}[face]
    u, v = (sc / abs(major) + 1) / 2, (tc / abs(major) + 1) / 2
    return face, min(int(u * 2), 1), min(int(v * 2), 1)


def direction(face, sc, tc):
    """The direction of face coordinates sc, tc (ma = 1): the inverse of the table in project()."""
    return {0: (1, -tc, -sc), 1: (-1, -tc, sc), 2: (sc, 1, tc), 3: (sc, -1, -tc), 4: (sc, -tc, 1), 5: (-sc, -tc, -1)}[face]


def reference():
    values = []
    for lane in range(LANES):
        s, t, face = 1.25 + 0.5 * (lane & 1), 1.25 + 0.5 * ((lane >> 1) & 1), lane >> 2
        values.append(texel_value(*project(direction(face, 2 * s - 3, 2 * t - 3))))
    assert len(set(values)) == LANES, values  # a swapped axis or face must change a word
    return values


def expected():
    values = [bits(float(v)) for v in reference()]
    return [values[i % LANES] for i in range(GROUPS * LANES)] + [SENTINEL] * (WORDS - GROUPS * LANES)


def check_words(got, want, label):
    first = next(((i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is not None:
        word, actual, wanted = first
        raise AssertionError(f'{label}: word {word} expected {wanted:08x}, read {actual:08x} '
                             f'({sum(a != b for a, b in zip(got, want))} words differ)')


def hlsl_program(sdk, cache, key):
    source = cache / 'cube.hlsl'
    source.write_text(CUBE_HLSL)
    result = subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'cs_6_0', '-E', 'main', '-Fo', cache / f'{key}.cs.cso', source],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def publish(compile_dxil, sdk, cache, key):
    result = subprocess.run([compile_dxil, sdk, cache.parent / 'native', 'gfx1013-direct-cs@0',
                             '--requests', cache, cache], capture_output=True, text=True, timeout=30)
    manifest = cache / f'{key}.txt'
    assert result.returncode == 0 and manifest.is_file(), (result.stdout, result.stderr)
    text = manifest.read_text()
    assert INPUT_LINE in text.splitlines() and OUTPUT_LINE in text.splitlines(), text
    assert not (cache / f'{key}.cs.hlsl').exists(), 'expected native publication'
    return manifest, text


def check_compiler(compile_dxil, sdk):
    with TemporaryDirectory() as directory:
        cache = Path(directory) / 'cache'
        cache.mkdir()
        key = write_request(cache, program(), STATE, root=root_blob(table_range=True))
        _, text = publish(compile_dxil, sdk, cache, f'{key:016x}')
        assert 'srv 0 texcube table' not in text, text
    print('startup_compute_dispatch_cube compile: PASS')


def check(compile_dxil, consumer, sdk, work, adapter):
    cache, env = prepare_case(work, adapter)
    env.pop('X2W_NATIVE_SHADERS', None)
    env.pop('X2W_NATIVE_SHADERS_ONLY', None)
    (work / 'program').write_bytes(program())
    (work / 'state').write_bytes(STATE)
    untouched, want = [SENTINEL] * WORDS, expected()
    assert want != untouched

    got, result = dispatch(work, consumer, env, MODE, probe=True)
    check_words(got, untouched, 'before translation')
    assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
    [request] = cache.glob('req_*.c67')
    key = request.stem.removeprefix('req_')
    manifest, original = publish(compile_dxil, sdk, cache, key)
    if adapter == 'warp':
        hlsl_program(sdk, cache, key)

    got, result = dispatch(work, consumer, env, MODE)
    check_words(got, want, 'cube samples')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
    errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    assert not errors, errors[0]
    layers = lines(result, 'x2w-d3d12 debug layer: ')
    assert layers and '0x00000000' in layers[0], ('debug layer not enabled', layers)
    print(f'{adapter}: {GROUPS * LANES} cube samples (24 distinct texels) match the D3D face mapping; '
          'debug layer: enabled, no error message', flush=True)

    def refused(label, mode, reason):
        got, result = dispatch(work, consumer, env, mode)
        logged = lines(result, 'refused: ')
        check_words(got, untouched, label)
        assert len(logged) == 1 and logged[0].endswith('refused: ' + reason), (label, logged, result.stderr[-3000:])
        assert not lines(result, 'Dispatch(3, 2, 2) translated'), label
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, (label, errors[0])

    refused('2D array view', MODE + '-array', REJECTED)
    refused('no description', MODE + '-undescribed', 'an input cube texture has no recorded view description')
    refused('view does not fit', MODE + '-twelve', 'an input texture view does not fit its texture')

    # Mutants: the dimension is the manifest's. The title's view is a cube, so another name must be refused.
    assert original.count(INPUT_LINE) == 1
    for name, reason in (
            ('tex2da', 'an input texture view is not 2D array'),
            ('tex2d', 'an input texture view is not 2D'),
            ('texcube', 'a resource that is not a raw table buffer or a 1D/2D/2D-array/3D/cube texture')):
        manifest.write_bytes(original.replace(INPUT_LINE, INPUT_LINE.replace('texcubearr', name)).encode())
        refused('manifest ' + name, MODE, reason)
    manifest.write_bytes(original.encode())
    got, result = dispatch(work, consumer, env, MODE)
    check_words(got, want, 'restored manifest')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
    print(f'{adapter}: a TEXTURE2DARRAY view, no description, a view on twelve slices and the line rewritten to tex2da, '
          f'tex2d or texcube are each refused with the output untouched', flush=True)


def main(compile_dxil, consumer, renderer, sdk, adapter):
    if adapter == 'hardware':
        wave = render(renderer, adapter, '--shader-wave32')
        assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as directory:
        check(compile_dxil, consumer, sdk, Path(directory) / adapter, adapter)
    print(f'startup_compute_dispatch_cube: PASS on {adapter}')


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'compile':
        check_compiler(*sys.argv[2:])
    elif len(sys.argv) == 6 and sys.argv[5] in ('warp', 'hardware'):
        main(*sys.argv[1:])
    else:
        sys.exit(__doc__)
