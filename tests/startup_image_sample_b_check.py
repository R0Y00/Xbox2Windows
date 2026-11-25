"""Biased image_sample_b in tools/startup/gfx2hlsl.py (pure Python, no arguments).

image_sample_b addresses the bias and then the coordinates (image_sample_l puts its level after them). The bias
is added to the implicit level, so the lowering must be Texture.SampleBias(sampler, uv, bias) with the bias read from
the FIRST address register. Sampling at level zero or at the bias as a level would be a different picture, and
forms the lowering does not model are refused by name.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the draw fixture and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       GPU readback of the lowered SampleBias against the mip it must select

The fixtures reuse the recorded image_sample_l stages of startup_image_sample_l_check.py and change only the MIMG
opcode field (bits 18..24 of the first word: image_sample_l is 0x24, image_sample_b is 0x25).
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_sample_l_check import PLAIN_INSTRUCTION, WORDS as L_WORDS  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl  # noqa: E402
from startup_image_3d_check import root_blob as draw_root_blob  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, assemble, root_blob  # noqa: E402

OPCODE_L, OPCODE_B = 0x24, 0x25


def as_bias(name):
    """The recorded image_sample_l stage with its single MIMG instruction turned into image_sample_b."""
    words = L_WORDS[name]
    # The NSA form is a longer instruction, so find the opcode word by its image_sample_l opcode field.
    for offset in range(0, len(words) - 7, 4):
        word0 = struct.unpack_from('<I', words, offset)[0]
        if word0 >> 26 == 0x3c and (word0 >> 18) & 0x7f == OPCODE_L and word0 & 1 == 0:
            patched = word0 ^ ((OPCODE_L ^ OPCODE_B) << 18)
            return words[:offset] + struct.pack('<I', patched) + words[offset + 4:]
    raise AssertionError('no image_sample_l instruction in %s' % name)


# GPU readback fixture: a 2D array pixel stage that samples (0.75, 0.25) of the slice in root constant 1 with a
# bias held in v0. The bias literal below is 3.0 and is patched per draw.
DRAW_SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_mov_b32 s24, 0
s_mov_b32 s25, 0
s_mov_b32 s26, 0
s_mov_b32 s27, 0
v_mov_b32 v0, 0x40400000
v_mov_b32 v1, 0x3f400000
v_mov_b32 v2, 0x3e800000
v_mov_b32 v3, s9
s_waitcnt lgkmcnt(0)
image_sample_b v[4:7], v[0:3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
BIAS_LITERAL = bytes.fromhex('ff02007e00004040')
DRAW_WORDS = bytes.fromhex(
    '04040cf4000000fa' '800398be' '800399be' '80039abe' '80039bbe' 'ff02007e00004040' 'ff02027e0000403f'
    'ff02047e0000803e' '0902067e' '7fc08cbf' '280f94f00004c400' '703f8cbf' '0f1800f804050607' '000081bf')


def stage(name):
    return G.Pipeline(VS_WORDS, as_bias(name), root_blob(flags=0)).hlsl_ps()


def refused(name, text):
    try:
        stage(name)
    except G.Unsupported as error:
        assert text in str(error), (name, str(error))
    else:
        raise AssertionError('%s was accepted' % name)


def bias_register(ps, width):
    m = re.search(r'\w+\.SampleBias\(\w+, float%d\(F\((v\d+)\), F\((v\d+)\)(?:, F\((v\d+)\))?\), F\((v\d+)\)\);' % width, ps)
    assert m, ps
    return m.groups()


def controls():
    assert PLAIN_INSTRUCTION in L_WORDS['plain']
    plain = stage('plain')
    assert bias_register(plain, 2) == ('v1', 'v2', None, 'v0'), plain
    assert 'SampleLevel' not in plain  # not the explicit-level or level-zero lowering
    assert bias_register(stage('nsa'), 2) == ('v1', 'v3', None, 'v0')  # the bias is the first register
    assert bias_register(stage('three'), 3) == ('v1', 'v2', 'v3', 'v0')
    array = stage('array')
    assert bias_register(array, 3) == ('v1', 'v2', 'v3', 'v0') and 'Texture2DArray' in array
    assert stage('glc') == plain and stage('dlc') == plain and stage('slc') == plain
    assert bias_register(stage('dmask3'), 2) == ('v1', 'v2', None, 'v0')
    assert plain != G.Pipeline(VS_WORDS, L_WORDS['plain'], root_blob(flags=0)).hlsl_ps()

    refused('unorm', 'image_sample_b modifier unorm')
    refused('tfe', 'image_sample_b modifier tfe')
    refused('a16', 'image_sample_b modifier a16')
    refused('lwe', 'image_sample_b modifier lwe')
    refused('d16', 'image_sample_b modifier d16')
    refused('r128', 'image_sample_b modifier r128')
    refused('dmask0', 'image_sample_b with an empty dmask')
    refused('dim1d', 'unverified 1D image sample address or modifiers')
    refused('cube', 'image dimension SQ_RSRC_IMG_CUBE')
    assert DRAW_WORDS.count(BIAS_LITERAL) == 1
    print('startup_image_sample_b_check: ok')


def bias_words(bias):
    """The draw fixture with its bias register loaded with the binary32 value bias."""
    return DRAW_WORDS.replace(BIAS_LITERAL, bytes.fromhex('ff02007e') + struct.pack('<f', bias))


def cell(x, y, slice_index, mip):
    """The mip fixture's texel (x, y) of slice slice_index at level mip (tests/startup_image_3d.cpp), as hex words."""
    red = (1.0 + x + 4.0 * y + 16.0 * slice_index + 32.0 * mip) / 128.0
    return ','.join('%08x' % struct.unpack('<I', struct.pack('<f', c))[0]
                    for c in (red, float(x), float(y), float(slice_index + 2 * mip)))


def expected_cell(slice_index, mip):
    """(0.75, 0.25) at a mip level of the 4x4 fixture: texel (floor(0.75 * e), floor(0.25 * e)), e = 4 >> mip."""
    extent = 4 >> mip
    return cell(int(0.75 * extent), int(0.25 * extent), slice_index, mip)


def check_binary(llvm_mc):
    built = assemble(llvm_mc, DRAW_SOURCE)
    if built != DRAW_WORDS:
        raise AssertionError('draw fixture: assembled %s, recorded %s' % (built.hex(), DRAW_WORDS.hex()))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    """GPU readback against a CPU reference. The coordinates are constants, so the derivatives are zero and the
    implicit level is exactly the bias: bias n must read mip n of the slice the root constant names."""
    operation = '--startup-image-2d-array-mips'
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    def draw(vs, shader, slice_index, expected):
        run = subprocess.run([renderer, adapter, operation, str(vs), str(shader), str(slice_index), expected],
                             capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            sys.exit(77)
        return run

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'fullscreen_bias_%s' % kind, vs_profile, directory)
            for bias in (0.0, 1.0, 2.0):
                text = G.Pipeline(VS_WORDS, bias_words(bias), draw_root_blob()).hlsl_ps()
                assert 'SampleBias(' in text, text
                shader = compile_hlsl(compiler, kind, text, 'ps_bias%d_%s' % (bias, kind), ps_profile, directory)
                for slice_index in (0, 1):
                    expected = expected_cell(slice_index, int(bias))
                    run = draw(vs, shader, slice_index, expected)
                    if run.returncode:
                        raise AssertionError('%s bias=%g slice=%d\n%s\n%s' % (kind, bias, slice_index, run.stdout, run.stderr))
                    print('%s bias=%g slice=%d: %s' % (kind, bias, slice_index, expected), flush=True)
                if bias == 0.0:
                    continue
                # Controls: dropping the bias (level zero) and reading the y coordinate (0.25) as the bias must both
                # read a different cell than mip `bias` does.
                call = re.search(r'\w+\.SampleBias\([^;]*\)', text).group(0)
                for label, mutated, actual_mip in (
                        ('bias dropped', re.sub(r'SampleBias\((.*), [^,]*\)$', r'SampleLevel(\1, 0)', call), 0),
                        ('y as bias', re.sub(r'(\w+)\.SampleBias\((\w+), float3\((F\(v\d+\)), (F\(v\d+\)), ([^)]*\))\), F\(v\d+\)\)',
                                             r'\1.SampleBias(\2, float3(\3, \4, \5), \4)', call), 0)):
                    source = text.replace(call, mutated)
                    assert source != text, label
                    mutant = compile_hlsl(compiler, kind, source, 'ps_bias_mutant_%s' % kind, ps_profile, directory)
                    run = draw(vs, mutant, 1, expected_cell(1, int(bias)))
                    actual = expected_cell(1, actual_mip)
                    if run.returncode == 0 or 'actual=%s' % actual not in run.stderr:
                        raise AssertionError('%s %s bias=%g: control did not read cell %s\n%s\n%s'
                                             % (kind, label, bias, actual, run.stdout, run.stderr))
                    print('%s %s bias=%g: fails as expected, reads %s' % (kind, label, bias, actual), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
