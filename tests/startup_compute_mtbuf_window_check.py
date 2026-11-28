"""Dispatch a published MTBUF byte window with matching and swapped DST_SEL."""

from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_gather_lod_check import render
from native_mtbuf_check import BYTE_WINDOW
from native_scalar_lds_check import scalar, sop1, vmove
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a, write_request
from startup_gfx2hlsl_compute_check import SENTINEL, TGID_X, TGID_Y, TGID_Z, USER7, WORDS, root_blob


INPUT = [0x01234567 + 0x1020304 * i & 0xffffffff for i in range(WORDS)]
STATE = state_a((32, 1, 1), USER7 | TGID_X | TGID_Y | TGID_Z)
IDENTITY = 0x0103efac
SWAPPED_XY = 0x0103efa5
EXPECTED = INPUT[:10] + [0] * 14 + [SENTINEL] * (WORDS - 24)
GUARDS = [SENTINEL] * WORDS
VIEW = 'srv 0 buf table 1 0 identity_dst_sel'


def program(*, writable=False):
    words = [
        sop1(3, 20, 6), sop1(3, 21, 128),
        0xf408080a, 0xfa000000,                # SRV in s[32:35]
        0xf408090a, 0xfa000000 if writable else 0xfa000020,  # revisit the same descriptor for UAV in the negative
        *BYTE_WINDOW,
        scalar(0x26, 8, 8, 131),               # y * 3
        scalar(0x26, 9, 9, 134),               # z * 6
        scalar(0, 11, 7, 8), scalar(0, 11, 11, 9),
        scalar(0x1e, 12, 11, 131),             # 8 * flattened group
        sop1(3, 126, 129), vmove(8, 12),       # one writer per group
        0xe8001000 | 62 << 19 | 1 << 16, 0x80080a08,  # tbuffer_load_format_xy R32G32_UINT
        0xe0741000, 0x80090a08,                # buffer_store_dwordx2 v[10:11], v8
        0xbf810000,
    ]
    return struct.pack('<%dI' % len(words), *words)


def controls(cli, sdk):
    with TemporaryDirectory() as temp:
        work = Path(temp)
        key = write_request(work, program(writable=True), STATE, root=root_blob(table_range=True))
        result = subprocess.run([cli, sdk, work / 'native', 'gfx1013-direct-cs@0', '--requests', work, work],
                                capture_output=True, text=True, timeout=30)
        reason = 'a writable MTBUF byte window has no verified view contract'
        assert result.returncode != 0 and reason in result.stderr, result.stdout + result.stderr
        assert not (work / f'{key:016x}.txt').exists() and not (work / f'{key:016x}.cs.cso').exists()
        print('MTBUF writable byte-window request refused before DXIL publication')


def prepare(work, adapter):
    cache, env = prepare_case(work, adapter)
    env.pop('X2W_NATIVE_SHADERS', None)
    env.pop('X2W_NATIVE_SHADERS_ONLY', None)
    (work / 'program').write_bytes(program())
    (work / 'state').write_bytes(STATE)
    source = work / 'source.bin'
    source.write_bytes(struct.pack('<%dI' % WORDS, *INPUT))
    env['X2W_TEST_DISPATCH_IN'] = str(source)
    env['X2W_TEST_DISPATCH_FIELDS'] = str(work / 'fields.bin')
    return cache, env


def run(work, consumer, env, word3, *, probe=False, mapping=None):
    # The 44-byte input window holds five complete pairs and one dword.
    (work / 'fields.bin').write_bytes(struct.pack('<6I', 11, 4, word3, 4, WORDS, 0))
    if mapping:
        env = dict(env, X2W_TEST_DISPATCH_INPUT_VIEW=mapping)
    return dispatch(work, consumer, env, 'dispatch-records32', probe=probe)


def compare(got, expected, result):
    if got != expected:
        word = next(i for i, (actual, want) in enumerate(zip(got, expected)) if actual != want)
        raise AssertionError(f'word {word}: expected {expected[word]:08x}, read {got[word]:08x}\n{result.stderr[-3000:]}')


def main(cli, consumer, renderer, sdk, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as temp:
        work = Path(temp)
        cache, env = prepare(work, adapter)
        got, result = run(work, consumer, env, IDENTITY, probe=True)
        compare(got, GUARDS, result)
        assert lines(result, 'Dispatch(3, 2, 2) refused: no translation'), result.stderr
        [request] = cache.glob('req_*.c67')
        key = request.stem.removeprefix('req_')
        command = [cli, sdk, work / 'native', 'gfx1013-direct-cs@0', '--requests', cache]
        for index, dest in enumerate((cache, work / 'warm')):
            result = subprocess.run([*command, dest], capture_output=True, text=True, timeout=30)
            assert result.returncode == 0 and f'frontend_cache_hit={index}' in result.stdout, result.stdout + result.stderr
            manifest = dest / f'{key}.txt'
            assert VIEW in manifest.read_text().splitlines(), manifest.read_text()
        assert (cache / f'{key}.txt').read_bytes() == (work / 'warm' / f'{key}.txt').read_bytes()

        got, result = run(work, consumer, env, IDENTITY)
        compare(got, EXPECTED, result)
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
        assert not lines(result, 'refused: '), result.stderr[-3000:]
        assert not [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        got, result = run(work, consumer, env, SWAPPED_XY)
        compare(got, GUARDS, result)
        refusal = 'Dispatch(3, 2, 2) refused: an MTBUF byte window requires identity DST_SEL'
        assert len(lines(result, refusal)) == 1, result.stderr[-3000:]
        got, result = run(work, consumer, env, IDENTITY, mapping='identity')
        compare(got, EXPECTED, result)
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]
        assert not lines(result, 'refused: '), result.stderr[-3000:]
        got, result = run(work, consumer, env, IDENTITY, mapping='swapped')
        compare(got, GUARDS, result)
        refusal = 'Dispatch(3, 2, 2) refused: an MTBUF byte window requires identity component mapping'
        assert len(lines(result, refusal)) == 1, result.stderr[-3000:]
        got, result = run(work, consumer, env, IDENTITY, mapping='cbv')
        compare(got, GUARDS, result)
        refusal = 'Dispatch(3, 2, 2) refused: an MTBUF byte window requires a buffer SRV'
        assert len(lines(result, refusal)) == 1, result.stderr[-3000:]
        print(f'{adapter}: identity readback matched five pairs, seven out-of-range pairs and all guards. '
              'Both swaps and the CBV token refused. Cold and warm manifests matched.')


if __name__ == '__main__':
    if sys.argv[1] == 'controls':
        controls(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
