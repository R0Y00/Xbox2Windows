"""Publish a synthetic D16 slot 67 program and check its typed buffer output."""

from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_dpp_mirror_check import vop2
from native_gather_lod_check import render
from native_scalar_lds_check import scalar, sop1, vmove
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a, write_request
from startup_gfx2hlsl_compute_check import SENTINEL, TGID_X, TGID_Y, TGID_Z, USER7, WORDS, root_blob


FORMAT_WIDTHS = ((1, 13), (2, 29), (4, 71))
OFFSET = 8
RECORDS = 13
MODE = 'dispatch-records32-output-offset'
STATE = state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z)
D16_FORMAT = dict(FORMAT_WIDTHS)


def mtbuf_store(channels, *, indexed=True):
    opcode = 0x0c + channels - 1
    return (0xe8000000 | D16_FORMAT[channels] << 19 | (opcode & 7) << 16 | (0x2000 if indexed else 0),
            0x80280100 | (8 if indexed else 0))  # opcode high bit, s[32:35], v1 data, inline zero soffset


def program(channels, *, mtbuf=False, indexed=True, store=None):
    words = [
        sop1(3, 20, 6), sop1(3, 21, 128),
        0xf408080a, 0xfa000020,                  # output descriptor s[32:35], table byte 32
        0xbf8cc07f,
        scalar(0x26, 8, 8, 131),                # 3 * group.y
        scalar(0x26, 9, 9, 134),                # 6 * group.z
        scalar(0, 11, 7, 8), scalar(0, 11, 11, 9),
        scalar(0x1e, 11, 11, 133),              # 32 * flat group index
        vop2(0x25, 8, 11, 0),                  # v8 = element index within the view
        vop2(0x1d, 1, 255, 0), 0xc0003c00,     # v1 = packed [-2, 1 + lane/1024]
        0x7e0402ff, 0xbc003800,                 # v2 = packed [-1, 0.5]
    ]
    if not indexed:
        words += [vmove(0, 131),                # a mistaken vaddr read indexes element 3 instead of 0
                  0xbf06800b, 0xbf840003,        # s_cmp_eq_u32 s11, 0; s_cbranch_scc0 to s_endpgm
                  sop1(3, 126, 129)]           # one writer in group zero
    if store is None:
        if mtbuf:
            store = mtbuf_store(channels, indexed=indexed)
        else:
            store = (0xe2102000 | ((channels - 1) << 18), 0x80080108)
    words += [*store, 0xbf810000]
    return struct.pack('<%dI' % len(words), *words)


def expected(channels, *, indexed=True):
    data = bytearray(struct.pack('<%dI' % WORDS, *([SENTINEL] * WORDS)))
    for lane in range(RECORDS if indexed else 1):
        halves = (0x3c00 + lane, 0xc000, 0x3800, 0xbc00)
        struct.pack_into('<%dH' % channels, data, OFFSET + lane * channels * 2, *halves[:channels])
    return list(struct.unpack('<%dI' % WORDS, data))


def check_words(got, want, label):
    first = next(((i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
    if first is not None:
        word, actual, expected_word = first
        raise AssertionError(f'{label}: word {word} expected {expected_word:08x}, read {actual:08x}')


def publish(compile_dxil, sdk, cache, key, stride):
    result = subprocess.run([compile_dxil, sdk, cache.parent / 'native', 'gfx1013-direct-cs@0',
                             '--requests', cache, cache], capture_output=True, text=True, timeout=30)
    manifest = cache / f'{key}.txt'
    assert result.returncode == 0 and manifest.is_file(), (result.stdout, result.stderr)
    text = manifest.read_text()
    assert f'uav 0 bufhalf table 1 32 bytes {stride}' in text.splitlines(), text
    assert not (cache / f'{key}.cs.hlsl').exists(), 'expected native publication'


def check_compiler(compile_dxil, sdk):
    with TemporaryDirectory() as directory:
        work = Path(directory)
        for channels, _ in FORMAT_WIDTHS:
            for mtbuf, indexed in ((False, True), (True, True), (True, False)):
                cache = work / f'{channels}-{mtbuf}-{indexed}'
                cache.mkdir()
                key = write_request(cache, program(channels, mtbuf=mtbuf, indexed=indexed), STATE,
                                    root=root_blob(table_range=True))
                publish(compile_dxil, sdk, cache, f'{key:016x}', channels * 2)
        a, b = mtbuf_store(2)
        for index, (store, reason) in enumerate((
            ((a | 2, b), 'MTBUF D16 store requires index or the first element and zero byte offsets'),
            ((a | 0x1000, b), 'MTBUF D16 store requires index or the first element and zero byte offsets'),
            ((a, b | 0x400000), 'MTBUF D16 store requires index or the first element and zero byte offsets'),
            ((a, b | 0x01000000), 'MTBUF D16 store requires index or the first element and zero byte offsets'),
            (((a & ~(127 << 19)) | 23 << 19, b), 'MTBUF D16 store requires an exact-width FLOAT format'),
            (((a & ~(7 << 16)) | 6 << 16, b), 'MTBUF D16 store requires an exact-width FLOAT format'),
        )):
            cache = work / f'bad-{index}'
            cache.mkdir()
            write_request(cache, program(2, store=store), STATE, root=root_blob(table_range=True))
            result = subprocess.run([compile_dxil, sdk, work / 'native', 'gfx1013-direct-cs@0',
                                     '--requests', cache, cache], capture_output=True, text=True, timeout=30)
            assert result.returncode and reason in result.stdout + result.stderr, (store, result.stdout, result.stderr)
            assert not list(cache.glob('*.cs.cso')) and not list(cache.glob('*.txt')), list(cache.iterdir())


def check(compile_dxil, consumer, sdk, work, adapter, channels, format_code, *, mtbuf=False, indexed=True):
    cache, env = prepare_case(work, adapter)
    env.pop('X2W_NATIVE_SHADERS', None)
    env.pop('X2W_NATIVE_SHADERS_ONLY', None)
    (work / 'program').write_bytes(program(channels, mtbuf=mtbuf, indexed=indexed))
    (work / 'state').write_bytes(STATE)
    fields = work / 'fields.bin'
    env['X2W_TEST_DISPATCH_FIELDS'] = str(fields)
    stride = channels * 2
    word3 = 0x01000fac | (format_code << 12)
    untouched = [SENTINEL] * WORDS

    def run(view_stride=stride, records=RECORDS, descriptor_word=word3, *, probe=False):
        fields.write_bytes(struct.pack('<6I', 4, WORDS, 0, view_stride, records, descriptor_word))
        return dispatch(work, consumer, env, MODE, probe=probe)

    got, result = run(probe=True)
    check_words(got, untouched, 'before translation')
    assert lines(result, 'Dispatch(3, 2, 2) refused: no translation'), result.stderr[-3000:]
    [request] = cache.glob('req_*.c67')
    key = request.stem.removeprefix('req_')
    publish(compile_dxil, sdk, cache, key, stride)
    got, result = run()
    check_words(got, expected(channels, indexed=indexed), f'{channels} channels')
    assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused:'), result.stderr[-3000:]
    assert not [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
    layers = lines(result, 'x2w-d3d12 debug layer: ')
    layer = 'enabled' if layers and '0x00000000' in layers[0] else 'unavailable'
    profile = 'MUBUF' if not mtbuf else 'MTBUF indexed' if indexed else 'MTBUF first element'
    print(f'{profile}, {channels} channels: {RECORDS if indexed else 1} typed records at byte {OFFSET} '
          f'and all guards match on {adapter} '
          f'(debug layer {layer})')

    wrong_shape = 'a half buffer store requires matching format, stride and structured bounds'
    for options, reason in (
        ({'view_stride': stride + 2}, wrong_shape),
        ({'view_stride': stride | 0x8000}, wrong_shape),
        ({'descriptor_word': 0x01016fac}, wrong_shape),
        ({'descriptor_word': word3 ^ 1}, wrong_shape),
        ({'descriptor_word': word3 | 1 << 28}, wrong_shape),
        ({'records': WORDS * 4 // stride}, 'an output range crosses the end of a placed buffer'),
    ):
        got, result = run(**options)
        check_words(got, untouched, options)
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: ' + reason)) == 1, (options, result.stderr[-3000:])
    print(f'{channels} channels: invalid views refused and output remained untouched')


def main(compile_dxil, consumer, renderer, sdk, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as directory:
        for channels, format_code in FORMAT_WIDTHS:
            for mtbuf, indexed in ((False, True), (True, True), (True, False)):
                work = Path(directory) / f'{channels}-{mtbuf}-{indexed}'
                check(compile_dxil, consumer, sdk, work, adapter, channels, format_code, mtbuf=mtbuf, indexed=indexed)


if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        check_compiler(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
