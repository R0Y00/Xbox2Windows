"""Named reasons for a slot-19 output view whose byte interval the translated dispatch refuses.

usage: startup_compute_token_interval_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The program is case A of startup_gfx2hlsl_compute_check.py with its two buffer descriptors loaded from 32-byte table
records, as in startup_compute_dispatch_native_check.py. The consumer's dispatch-uav-token-raw mode writes, at table
byte 32, the token of a raw slot-19 view of the 1280-byte placed output buffer: first element 4, 312 elements. After
gfx2hlsl.py --pending with DXC that view is bound and words 4-315 match the case's expectation. Then the view is
replaced through X2W_TEST_DISPATCH_UAV, one condition at a time, and each dispatch is refused with its own reason in
the log and the census and writes no output word:

- an R16_UINT view of one element (2 bytes) at byte 16;
- a raw view of 4 Mi + 4 elements (16 MiB + 16 bytes) of a committed buffer 64 KiB over 16 MiB;
- an R32_UINT view at byte 4, inside the buffer.

The other three reasons (an empty interval, one that starts past the buffer and one that crosses its end) need a
view that D3D12 itself rejects when it is created; with the debug layer on, that removes the device. Exit 77 when
the adapter is not available.
"""
from pathlib import Path
import os
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_dispatch_native_check import LOAD_X8, LOADS_X4
from startup_stream_check import gfx2hlsl as G

RAW, R16_UINT, R32_UINT = 39, 57, 42  # DXGI_FORMAT values


def view(first, elements, fmt=RAW):
    """D3D12_UNORDERED_ACCESS_VIEW_DESC of a buffer: format, dimension, first, count, stride, counter, flags."""
    return struct.pack('<IIQIIQI4x', fmt, 1, first, elements, 0, 0, 1 if fmt == RAW else 0)


CONTROLS = (
    ('raw', view(8, 1, R16_UINT), 'an output UAV interval is not whole dwords'),
    ('large', view(0, (4 << 20) + 4), 'an output UAV interval is over 16 MiB'),
    ('raw', view(1, 300, R32_UINT), 'an output UAV interval is not 16-byte aligned in its buffer'),
)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        env.pop('X2W_NATIVE_SHADERS', None)  # Exercise published translations.
        program = directory / 'program'
        case_a = program.read_bytes()
        assert case_a.count(LOAD_X8) == 1
        program.write_bytes(case_a.replace(LOAD_X8, LOADS_X4))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        got, result = dispatch(directory, consumer, env, 'dispatch-uav-token-raw', probe=True)
        print(result.stdout, end='')
        assert got == untouched and len(lines(result, 'refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)
        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out

        got, result = dispatch(directory, consumer, env, 'dispatch-uav-token-raw')
        bound = untouched[:4] + expected[:-8] + untouched[-4:]
        assert got == bound and not lines(result, 'refused'), result.stderr[-3000:]
        assert lines(result, 'offset=16 bytes=1248 result=bound'), result.stderr[-3000:]

        for backing, desc, reason in CONTROLS:
            (directory / 'uav').write_bytes(desc)
            got, result = dispatch(directory, consumer, dict(env, X2W_TEST_DISPATCH_UAV=str(directory / 'uav')),
                                   f'dispatch-uav-token-{backing}')
            name = f'{backing} {struct.unpack("<IIQIIQI4x", desc)[:4]}'
            assert got == untouched, f'{name}: a refused Dispatch wrote the output'
            refused = lines(result, 'refused: ')
            census = lines(result, 'x2w-render compute census')
            assert len(refused) == 1 and refused[0].endswith('refused: ' + reason), f'{name}:\n{result.stderr[-3000:]}'
            assert len(census) == 3 and census[1].endswith(f'reason "{reason}": 1 keys, 1 dispatches'), (name, census)
        print(f'startup_compute_token_interval: PASS on {adapter} (a raw slot-19 view bound, words 4-{C.WORDS - 5} '
              f'match; {len(CONTROLS)} intervals refused, each with its own reason and an untouched output)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
