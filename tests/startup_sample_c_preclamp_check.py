"""Check the preclamp flag on a synthetic point-filtered comparison sampler.

An explicit level-zero sample may ignore preclamp. Implicit LOD and unsupported
filter flags must retain the decoder's refusal.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import pixel_aot  # noqa: E402
import startup_image_sample_c_check as C  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

# CLAMP_LAST_TEXEL in X and Y, point-filter truncation, GREATER comparison.
SYNTHETIC = (C.POINT | (4 << 12), 0, (1 << 26) | (1 << 28), 0)
S24 = bytes.fromhex('ff0398be')   # s_mov_b32 s24, literal
S26_ZERO = bytes.fromhex('80039abe')  # s_mov_b32 s26, 0


def patched(name, w0, w2):
    words = C.WORDS[name]
    i = words.index(S24)
    words = words[:i + 4] + struct.pack('<I', w0) + words[i + 8:]
    assert words.count(S26_ZERO) == 1, name
    return words.replace(S26_ZERO, bytes.fromhex('ff039abe') + struct.pack('<I', w2))


def ps(name, w0, w2):
    pipe = G.Pipeline(VS_WORDS, patched(name, w0, w2), root_blob(flags=0))
    return pipe, pipe.hlsl_ps()


def refused(name, w0, w2, text):
    try:
        ps(name, w0, w2)
    except G.Unsupported as error:
        assert text in str(error), (name, str(error))
    else:
        raise AssertionError('accepted: %s %#x' % (name, w2))


def main():
    pipe, text = ps('lz', SYNTHETIC[0], SYNTHETIC[2])
    method, u, v, ref, _, _ = C.lowered(text)
    assert (method, u, v, ref) == ('SampleCmpLevelZero', 'v1', 'v2', 'v0'), (method, u, v, ref)
    assert 'SamplerComparisonState q0 : register(s0);' in text, text
    assert C.manifest_samplers(pipe) == ['smpc 0 ' + ' '.join(hex(word) for word in SYNTHETIC)], C.manifest_samplers(pipe)
    refused('lz', SYNTHETIC[0], SYNTHETIC[2] | 1 << 29, 'word 2 bits 28-31')
    refused('c', SYNTHETIC[0], SYNTHETIC[2], 'word 2 bits 28-31')
    try:
        pixel_aot.comparison_sampler_state(SYNTHETIC)
    except ValueError as error:
        assert 'word 2 bits 28-31' in str(error)
    else:
        raise AssertionError('pixel_aot accepts bit 28: the negative control no longer shows the gap')
    print('sample_c_preclamp: synthetic comparison sampler (MIP_POINT_PRECLAMP) lowered on c_lz with its words kept; '
          'bit 29 and implicit-LOD controls refused; pixel_aot still refuses bit 28 (negative control)')


if __name__ == '__main__':
    main()
