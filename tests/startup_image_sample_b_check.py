"""Biased image_sample_b in tools/startup/gfx2hlsl.py (pure Python, no arguments).

image_sample_b addresses the coordinates and then the bias, exactly where image_sample_l puts its level. The bias
is added to the implicit level, so the lowering must be Texture.SampleBias(sampler, uv, bias) with the bias read from
the LAST address register. Sampling at level zero or at the bias as a level would be a different picture, and
forms the lowering does not model are refused by name.

The fixtures reuse the recorded image_sample_l stages of startup_image_sample_l_check.py and change only the MIMG
opcode field (bits 18..24 of the first word: image_sample_l is 0x24, image_sample_b is 0x25).
"""
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_sample_l_check import PLAIN_INSTRUCTION, WORDS as L_WORDS  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

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


def main():
    assert PLAIN_INSTRUCTION in L_WORDS['plain']
    plain = stage('plain')
    assert bias_register(plain, 2) == ('v0', 'v1', None, 'v2'), plain
    assert 'SampleLevel' not in plain  # not the explicit-level or level-zero lowering
    assert bias_register(stage('nsa'), 2) == ('v0', 'v1', None, 'v3')  # the bias is the last register
    assert bias_register(stage('three'), 3) == ('v0', 'v1', 'v2', 'v3')
    array = stage('array')
    assert bias_register(array, 3) == ('v0', 'v1', 'v2', 'v3') and 'Texture2DArray' in array
    assert stage('glc') == plain and stage('dlc') == plain and stage('slc') == plain
    assert bias_register(stage('dmask3'), 2) == ('v0', 'v1', None, 'v2')
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
    print('startup_image_sample_b_check: ok')


if __name__ == '__main__':
    sys.exit(main())
