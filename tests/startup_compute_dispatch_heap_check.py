"""A translated compute program whose input is the resource heap record a root constant indexes (list slot 14).

usage: startup_compute_dispatch_heap_check.py <startup_stream_consumer> <dxc> <warp|hardware>

The consumer creates the program of startup_gfx2hlsl_heap_record_check.py as a device slot 67 pipeline, with a
root signature that indexes the resource heap directly, and records the descriptor heap, the pipeline, the root
signature, three root constants, one table and Dispatch(3, 2, 2). The output descriptor is at table byte 16 as in
startup_compute_dispatch_check.py. The input descriptor is only in heap record 2 (heap byte 64), which the third
root constant indexes; heap bytes 0 and 32 hold the output's descriptor, so a dispatch that ignores the index or
scales it by 16 is refused instead of passing. Three runs share one cache:

1. before translation: the request is published and the Dispatch is refused with "no translation";
2. after gfx2hlsl.py --pending with DXC: the manifest names the heap record and all 320 words equal the
   expectation of case A of startup_gfx2hlsl_compute_check.py (the program computes the same words);
3. dispatch-heap-outside (the constant is 0x07ffffff, a record past the heap): refused with one
   logged reason, every word keeps the sentinel and the input is unchanged.
Exit 77 when the adapter is not available.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
import startup_gfx2hlsl_heap_record_check as H
from startup_compute_dispatch_check import dispatch, lines
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

OUTSIDE = "a heap record is outside the list's view heap"


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare(directory)
        env.update(X2W_D3D12_LIST_NOOP='1', X2W_D3D12_DEBUG='1')  # guest lists; debug-layer messages go to the log
        if adapter == 'hardware':
            env['X2W_TEST_DISPATCH_HARDWARE'] = '1'
        (directory / 'compute_root').write_bytes(H.root_blob(table_range=True))
        env['X2W_TEST_COMPUTE_ROOT'] = str(directory / 'compute_root')
        assert H.GROUPS == (3, 2, 2) and H.INDEX_WORD == 2, 'the consumer hard-codes the dispatch and the index word'
        (directory / 'program').write_bytes(H.WORDS)
        (directory / 'data').write_bytes(DATA)
        (directory / 'state').write_bytes(state_a(H.GROUP, H.RSRC2))
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')
        assert expected != untouched

        got, result = dispatch(directory, consumer, env, 'dispatch-heap')
        print(result.stdout, end='')
        assert got == untouched, 'a Dispatch without a translation wrote the output'
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        published = lines(result, 'x2w-draw compute request')
        assert published and ' published ' in published[0], result.stderr[-3000:]
        key = int(published[0].split()[3], 16)

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=dxc)).stdout
        assert f'{key:016x}: ok' in out, out
        assert (cache / f'{key:016x}.txt').read_text() == H.MANIFEST

        got, result = dispatch(directory, consumer, env, 'dispatch-heap')
        wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
        assert not wrong, 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
        layer = lines(result, 'x2w-d3d12 debug layer: ')
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))

        got, result = dispatch(directory, consumer, env, 'dispatch-heap-outside')
        assert got == untouched, 'a refused Dispatch wrote the output'
        refused = lines(result, 'refused: ')
        census = lines(result, 'x2w-render compute census')
        assert len(refused) == 1 and OUTSIDE in refused[0], result.stderr[-3000:]
        assert len(census) == 3 and census[1].endswith(f'reason "{OUTSIDE}": 1 keys, 1 dispatches'), census
        print(f'startup_compute_dispatch_heap: PASS on {adapter} (all {C.WORDS} words match through heap record 2; '
              'untouched before translation and for a record outside the heap)')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
