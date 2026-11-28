"""Check published D16 image stores on narrow typed UAVs."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_direct_container_check import words
from native_gather_lod_check import render
from startup_compute_dispatch_check import lines
from startup_compute_dispatch_texture_check import clean, run
from startup_compute_image_store_check import ARRAY_END, ARRAY_PROLOGUE, ARRAY_SLICE, RSRC2
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a, write_request
from startup_gfx2hlsl_compute_check import root_blob


# Each selected mip is 32 wide. The layered paths select a view starting at
# physical layer 1 and write its relative layer 1.
CASES = (
    ('r-line', 1, 0, 'tex1d', 1, 1),
    ('r', 1, 1, 'tex2d', 8, 1),
    ('rg', 2, 1, 'tex2d', 8, 1),
    ('r-typeless', 1, 1, 'tex2d', 8, 1),
    ('rg-array', 2, 5, 'tex2da', 8, 6),
    ('r-volume', 1, 2, 'tex3d', 8, 6),
)
ROOT = root_blob(3, table_range=True)


def program(channels, dimension):
    return (
        words(0xbefe03c1)                       # s_mov_b32 exec_lo, -1, before the vector prologue
        + ARRAY_PROLOGUE + ARRAY_SLICE[1]
        + words(0xbf8cc07f, 0x7e1c02ff, 0xc0003c00, 0xbefe03ff, 0x55)
        + words(0xf0201000 | (((1 << channels) - 1) << 8) | (dimension << 3), 0x80060e08)
        + ARRAY_END
    )


def state(height):
    return state_a((4, 1 if height == 1 else 2, 1), RSRC2)


def compile_request(cli, sdk, cache):
    return subprocess.run([cli, sdk, cache.parent / 'compiler-cache', 'gfx1013-direct-cs@0',
                           '--requests', cache, cache], capture_output=True, text=True, timeout=30)


def publish(cli, sdk, cache, key, channels, dimension):
    result = compile_request(cli, sdk, cache)
    manifest = cache / f'{key}.txt'
    assert result.returncode == 0 and manifest.is_file(), result.stdout + result.stderr
    expected = f'uav 0 {dimension} half table 1 64 channels {channels}'
    assert expected in manifest.read_text().splitlines(), manifest.read_text()


def controls(cli, sdk):
    with TemporaryDirectory() as directory:
        for name, channels, dimension, view, height, _ in CASES:
            cache = Path(directory) / name
            cache.mkdir()
            key = write_request(cache, program(channels, dimension), state(height), root=ROOT)
            publish(cli, sdk, cache, f'{key:016x}', channels, view)
        cache = Path(directory) / 'unsupported-1d-array'
        cache.mkdir()
        key = f'{write_request(cache, program(2, 4), state(8), root=ROOT):016x}'
        result = compile_request(cli, sdk, cache)
        reason = 'unsupported startup runtime manifest bindings: UAV kind half2_texture_uav 1d-array'
        assert result.returncode == 0 and reason in result.stdout, result.stdout + result.stderr
        assert (cache / f'{key}.cs.cso').read_bytes().startswith(b'DXBC')
        assert not (cache / f'{key}.txt').exists()
        print('1D-array D16 store compiled and was refused at manifest publication')


def compare(got, expected, label, result):
    if got != expected:
        word = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)
        raise AssertionError(f'{label}: component {word} expected {expected[word]:04x}, '
                             f'read {got[word]:04x}\n{result.stderr[-3000:]}')


def stored(initial, channels, height, layer, start):
    expected = initial.copy()
    values = [0x3c00, 0xc000][:channels]
    for y in range(min(height, 2)):
        for x in (0, 2, 4, 6):
            i = start + ((layer * height + y) * 32 + x) * channels
            expected[i:i + channels] = values
    return expected


def check(directory, cli, sdk, consumer, adapter, case):
    name, channels, dimension, view, height, layers = case
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1',
               X2W_TEST_COMPUTE_ROOT=str(directory / 'compute_root'))
    for flag in ('X2W_NATIVE_SHADERS', 'X2W_NATIVE_SHADERS_ONLY', 'X2W_TEST_DISPATCH_HARDWARE'):
        env.pop(flag, None)
    if adapter == 'hardware':
        env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
    (directory / 'compute_root').write_bytes(ROOT)
    (directory / 'program').write_bytes(program(channels, dimension))
    (directory / 'data').write_bytes(DATA)
    (directory / 'state').write_bytes(state(height))
    mode = 'texture-exact-half-' + name
    base_mip = [0x7000 + i for i in range(64 * 16)] if name == 'r' else []
    initial = base_mip + [0x5000 + i for i in range(layers * 32 * height * channels)]
    layout = f'<{len(initial)}H'

    got, result = run(directory, consumer, env, mode, layout)
    compare(got, initial, name + ' before translation', result)
    assert lines(result, 'Dispatch(2, 1, 1) refused: no translation'), result.stderr
    key = lines(result, 'x2w-draw compute request')[0].split()[3]
    publish(cli, sdk, cache, key, channels, view)

    expected = stored(initial, channels, height, 2 if layers > 1 else 0, len(base_mip))
    got, result = run(directory, consumer, env, mode, layout)
    compare(got, expected, name, result)
    assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr
    layer = clean(result)
    print(f'{adapter} {name}: all {len(expected)} halfwords match including mip 1 (debug layer {layer})')

    if name == 'r-volume':
        got, result = run(directory, consumer, env, mode + '-shifted-w', layout)
        compare(got, stored(initial, channels, height, 1, 0), 'shifted volume W range', result)
        assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr
        clean(result)
        changed = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
        assert len(changed) == 16, changed
        first = changed[0]
        print(f'{adapter} shifted W control: {len(changed)} differences, first component {first} '
              f'expected {expected[first]:04x}, read {got[first]:04x}')

    if name in ('r', 'rg'):
        # Same channel count, but an R32/R32G32 view is from a different format family.
        untouched = [0x30000000 + i for i in range(32 * height * channels)]
        got, result = run(directory, consumer, env, 'texture-exact-' + name, f'<{len(untouched)}I')
        compare(got, untouched, name + ' wrong format', result)
        assert len(lines(result, 'Dispatch(2, 1, 1) refused: a D16 image store requires a matching '
                                 'half-float UAV and resource')) == 1, result.stderr


def main(cli, consumer, renderer, sdk, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, wave.stdout + wave.stderr
    with TemporaryDirectory() as directory:
        for case in CASES:
            check(Path(directory) / case[0], cli, sdk, consumer, adapter, case)


if __name__ == '__main__':
    if sys.argv[1] == 'compile':
        controls(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
