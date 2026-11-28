"""Publish a synthetic slot 67 program with buffer_load_format_* and check its typed buffer input (#3224).

The program reads input element `index` with buffer_load_format_<n> (IDXEN) from table byte 0 and stores the n
dwords raw at output byte 4 * n * index (table byte 32). compile_dxil publishes `srv 0 typed table 1 0 bytes 4n`;
the stand-in binds a UINT view of the title's 32-bit components. Fewer records than lanes checks the zeros past
the view's last element.

usage:
  startup_compute_typed_buffer_srv_check.py compile <compile_dxil> <sdk>
  startup_compute_typed_buffer_srv_check.py <compile_dxil> <startup_stream_consumer> <offscreen_check> <sdk> <warp|hardware>
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

STATE = state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z)
FLOAT_FORMATS = {1: 22, 2: 64, 3: 74, 4: 77}  # gfx10 BUF_FMT_32[_32[_32[_32]]]_FLOAT
STORES = {1: 0x1c, 2: 0x1d, 3: 0x1f, 4: 0x1e}  # buffer_store_dword[x2|x3|x4]
RECORDS = 50
INPUT = [(0x8b02a7f1 + i * 0x413b1) & 0xffffffff for i in range(WORDS)]  # the consumer's input words
DXGI_R32G32B32A32_FLOAT, DXGI_R8G8B8A8_UNORM = 2, 28


def program(channels):
    offset = (vop2(0x0b, 9, 128 + 12, 8) if channels == 3 else           # v9 = 12 * v8
              vop2(0x1a, 9, 128 + {1: 2, 2: 3, 4: 4}[channels], 8))     # v9 = v8 << log2(4n)
    words = [
        sop1(3, 20, 6), sop1(3, 21, 128),
        0xf408080a, 0xfa000020,                 # output descriptor s[32:35], table byte 32
        0xf408090a, 0xfa000000,                 # input descriptor s[36:39], table byte 0
        0xbf8cc07f,                             # s_waitcnt lgkmcnt(0)
        scalar(0x26, 8, 8, 131),                # 3 * group.y
        scalar(0x26, 9, 9, 134),                # 6 * group.z
        scalar(0, 11, 7, 8), scalar(0, 11, 11, 9),
        scalar(0x1e, 11, 11, 133),              # 32 * flat group index
        vop2(0x25, 8, 11, 0),                   # v8 = element index
        0xe0002000 | (channels - 1) << 18, 0x80090108,  # buffer_load_format_<n> v[1:n], v8, s[36:39] idxen
        0xbf8c3f70,                             # s_waitcnt vmcnt(0)
        offset,
        0xe0001000 | STORES[channels] << 18, 0x80080109,  # buffer_store_dword<n> v[1:n], v9, s[32:35] offen
        0xbf810000,
    ]
    return struct.pack('<%dI' % len(words), *words)


def output_records(channels):
    return WORDS // channels * channels  # no store crosses the end of the output view


def expected(channels):
    return [INPUT[k] if k < RECORDS * channels else 0 if k < output_records(channels) else SENTINEL
            for k in range(WORDS)]


def check_words(got, want, label):
    first = next(((i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is not None:
        word, actual, expected_word = first
        raise AssertionError(f'{label}: word {word} expected {expected_word:08x}, read {actual:08x}')


def publish(compile_dxil, sdk, cache, key, channels):
    result = subprocess.run([compile_dxil, sdk, cache.parent / 'native', 'gfx1013-direct-cs@0',
                             '--requests', cache, cache], capture_output=True, text=True, timeout=60)
    manifest = cache / f'{key}.txt'
    assert result.returncode == 0 and manifest.is_file(), (result.stdout, result.stderr)
    text = manifest.read_text().splitlines()
    assert f'srv 0 typed table 1 0 bytes {4 * channels}' in text and 'uav 0 table 1 32' in text, text


def check_compiler(compile_dxil, sdk):
    with TemporaryDirectory() as directory:
        for channels in FLOAT_FORMATS:
            cache = Path(directory) / str(channels)
            cache.mkdir()
            key = write_request(cache, program(channels), STATE, root=root_blob(table_range=True))
            publish(compile_dxil, sdk, cache, f'{key:016x}', channels)
    print('compile_dxil publishes srv typed lines for 1 to 4 components')


def check(compile_dxil, consumer, sdk, work, adapter, channels):
    cache, env = prepare_case(work, adapter)
    env.pop('X2W_NATIVE_SHADERS', None)
    env.pop('X2W_NATIVE_SHADERS_ONLY', None)
    (work / 'program').write_bytes(program(channels))
    (work / 'state').write_bytes(STATE)
    fields = work / 'fields.bin'
    env['X2W_TEST_DISPATCH_FIELDS'] = str(fields)
    word3 = 0x01000fac | FLOAT_FORMATS[channels] << 12
    untouched = [SENTINEL] * WORDS

    def run(stride=4 * channels, records=RECORDS, descriptor_word=word3, mode='dispatch-records32', *, probe=False):
        fields.write_bytes(struct.pack('<6I', stride, records, descriptor_word, 4, output_records(channels), 0))
        return dispatch(work, consumer, env, mode, probe=probe)

    got, result = run(probe=True)
    check_words(got, untouched, 'before translation')
    assert lines(result, 'Dispatch(3, 2, 2) refused: no translation'), result.stderr[-3000:]
    [request] = cache.glob('req_*.c67')
    key = request.stem.removeprefix('req_')
    publish(compile_dxil, sdk, cache, key, channels)

    def translated(label, **options):
        got, result = run(**options)
        check_words(got, expected(channels), label)
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused:'), result.stderr[-3000:]
        assert not [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        return result

    result = translated(f'{channels} channels')
    layers = lines(result, 'x2w-d3d12 debug layer: ')
    layer = 'enabled' if layers and '0x00000000' in layers[0] else 'unavailable'
    print(f'{channels} channels: {RECORDS} typed elements, then zeros, on {adapter} (debug layer {layer})')

    refusals = [
        ({'descriptor_word': 0x01038fac}, 'a typed input FORMAT is not 32 bits per channel'),  # 8_8_8_8_UNORM
        ({'stride': 4 * channels + 4}, 'a typed input stride is not its element size'),
        ({'descriptor_word': word3 | 3 << 28}, 'a typed input descriptor is swizzled or not bounds-checked by record'),
        ({'descriptor_word': word3 & ~0x01000000}, 'a typed input descriptor is swizzled or not bounds-checked by record'),
    ]
    if channels > 1:
        refusals.append(({'descriptor_word': word3 & ~0xfff | 0xfa5}, 'a typed input DST_SEL is not the identity'))  # X, Y swapped
        narrow = 0x01000fac | FLOAT_FORMATS[channels - 1] << 12
        refusals.append(({'descriptor_word': narrow, 'stride': 4 * (channels - 1)},
                         'a typed input has fewer channels than the program reads'))
    if channels == 4:
        translated('view token', mode='dispatch-srv-token-typed', descriptor_word=DXGI_R32G32B32A32_FLOAT)
        refusals.append(({'mode': 'dispatch-srv-token-typed', 'descriptor_word': DXGI_R8G8B8A8_UNORM},
                         'a typed input token is not a typed buffer view with 32-bit channels'))
        print('4 channels: a slot 18 typed buffer view token reads the same elements')
    for options, reason in refusals:
        got, result = run(**options)
        check_words(got, untouched, options)
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: ' + reason)) == 1, (options, result.stderr[-3000:])
    print(f'{channels} channels: {len(refusals)} invalid inputs refused and the output remained untouched')


def main(compile_dxil, consumer, renderer, sdk, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as directory:
        for channels in FLOAT_FORMATS:
            check(compile_dxil, consumer, sdk, Path(directory) / str(channels), adapter, channels)


if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        check_compiler(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
