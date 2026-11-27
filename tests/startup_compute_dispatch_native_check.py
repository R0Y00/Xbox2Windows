"""A slot 67 compute program translated in-process by the C++ compiler and dispatched through the stand-in (#2934).

usage: startup_compute_dispatch_native_check.py <startup_stream_consumer> <DXC directory> <warp|hardware>

X2W_NATIVE_SHADERS names the directory of dxcompiler.dll and dxil.dll. The cache starts empty and no offline
translator runs before step 3. The program is case A of startup_gfx2hlsl_compute_check.py with its two buffer
descriptors loaded from 32-byte table records (two s_load_dwordx4, at table bytes 0 and 32), the layout the C++
compiler reads a table in. The consumer's dispatch-records32 mode writes the table that way. The reference is
that case's CPU expectation.

1. Native: the first Dispatch translates the request in the stand-in and is recorded. All 320 words match, the log
   names the native translation, and the cache holds request files and compiler cache entries only (no <key>.txt,
   <key>.cs.cso or <key>.fail). The same program over the plain table layout, which has no descriptor at byte 32,
   is then refused when it is bound and every word keeps the sentinel: the record at byte 32 is what was bound.
2. Unsupported workload: case A itself reads its second descriptor from table byte 16. The C++ compiler refuses
   it, the reason is logged, the Dispatch is refused with "no translation" and every word keeps the sentinel.
3. Fallback: gfx2hlsl.py --pending then publishes both requests, and with native translation still on the next run
   dispatches case A through the published translation. All 320 words match.
4. Order: the first program runs again, now with a published translation in the cache. The native translation is
   still the one used, and all 320 words match.
5. Table views: a scalar buffer read and a 2D image load feed a 3D image store. Replacing the load with a sample
   adds a sampler and must leave the volume untouched. A zero-size UAV W range also leaves it untouched.
6. Texture outputs: 1D float, 2D-array float, and 2D packed D16 stores match CPU references.

A direct compute module declares wave size 32. On an adapter without 32-lane waves, steps 1, 5 and 6 check that the
native translation is refused by that name with the output untouched. The sampler control and steps 2 and 3 run as
written, step 4 uses the published translation, and the check exits 77. Exit 77 also when the adapter is unavailable.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_gfx2hlsl_compute_check as C
import startup_compute_dispatch_texture_check as T
import startup_compute_image_store_check as A
import startup_compute_image_store_d16_check as D
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

LOAD_X8 = bytes.fromhex('05030cf4' '000000fa')
LOADS_X4 = bytes.fromhex('050308f4' '000000fa' '050408f4' '200000fa')
RECORD_LOADS = ['s_load_dwordx4 s[12:15], s[10:11], 0x0', 's_load_dwordx4 s[16:19], s[10:11], 0x20']
NO_WAVES = 'not translated natively: the adapter has no 32-lane waves'
SAMPLER_REFUSAL = 'not translated natively: a sampler, or more entry values, inputs or outputs than the dispatch path binds'
TEXTURE_DISPATCH = 'Dispatch(2, 1, 1)'
ZERO_W_REFUSAL = 'an output 3D texture W range is outside the mip'


def mismatch(got, expected, result):
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    return '' if not wrong else 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])


def table_program(sample):
    # Keep the volume fixture's table pointer, UAV descriptor, coordinates, and final store.
    wait = bytes.fromhex('7fc08cbf')
    prologue, _, tail = T.store_program('volume').partition(wait)
    loads = bytes.fromhex(
        '060808f4000000fa'  # s_load_dwordx4 s[32:35], s[12:13], 0x0
        '06060cf4200000fa'  # s_load_dwordx8 s[24:31], s[12:13], 0x20
    )
    values = bytes.fromhex(
        '100920f4000000fa'  # s_buffer_load_dword s36, s[32:35], 0x0
    ) + wait + bytes.fromhex(
        '2402107e'  # v_mov_b32 v8, s36
        '80020c7e'  # v_mov_b32 v6, 0
        '80020e7e'  # v_mov_b32 v7, 0
    )
    if sample:
        image = bytes.fromhex(
            '8004a8be8004aabe'  # s_mov_b64 s[40:41], 0; s_mov_b64 s[42:43], 0
            '08019cf006094601'  # image_sample_lz v9, v[6:7], s[24:31], s[40:43], dmask:1, 2D
        )
    else:
        image = bytes.fromhex('080100f006090600')  # image_load v9, v[6:7], s[24:31], dmask:1, 2D
    # The new reads replace the first two literal moves (two dwords each).
    return prologue + loads + wait + values + image + bytes.fromhex('700f8cbf') + tail[16:]


def table_views(directory, consumer, dxc_dir, adapter, no_waves):
    cache, env = prepare_case(directory, adapter)
    env['X2W_NATIVE_SHADERS'] = dxc_dir
    root = Path(env['X2W_TEST_COMPUTE_ROOT'])
    # Two SRVs in records 0 and 1, followed by one UAV in record 2.
    blob = bytearray(C.root_blob(3, table_range=True)[:-24])
    blob.extend(struct.pack('<12I', 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2))
    payload = 44  # DXBC header, part offset, and RTS0 part header
    struct.pack_into('<I', blob, 24, len(blob))
    struct.pack_into('<I', blob, 40, len(blob) - payload)
    struct.pack_into('<I', blob, payload + 16, len(blob) - payload)  # static sampler offset
    struct.pack_into('<I', blob, payload + 60, 2)  # descriptor table range count
    root.write_bytes(blob)
    group = T.store_group('volume')
    (directory / 'state').write_bytes(state_a(group, T.STORE_RSRC2))
    plane = 32 * 8 * 4
    initial = [0xbf000000 | (z << 12) | i for z in range(6) for i in range(plane)]
    expected = initial.copy()
    for y in range(group[1]):
        start = 2 * plane + y * 32 * 4  # view FirstWSlice 1, shader Z 1
        expected[start:start + 32] = (0x3f200000, initial[0], *T.STORE_VALUES[2:]) * 8

    for mode, sample, zero_w in (
            ('texture-volume-native-inputs', False, False),
            ('texture-volume-native-inputs', True, False),
            ('texture-volume-native-zero-w', False, True)):
        (directory / 'program').write_bytes(table_program(sample))
        got, result = T.run(directory, consumer, env, mode, '<%dI' % len(initial))
        if sample or no_waves:
            reason = SAMPLER_REFUSAL if sample else NO_WAVES
            assert len(lines(result, reason)) == 1, result.stderr[-3000:]
            assert len(lines(result, 'Dispatch(2, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
            assert got == initial, 'a refused Dispatch wrote the volume'
        else:
            assert len(lines(result, 'translated natively: 3 entry values, 2 inputs, 1 outputs')) == 1, result.stderr[-3000:]
            if zero_w:
                refusal = lines(result, f'{TEXTURE_DISPATCH} refused:')
                assert len(refusal) == 1 and refusal[0].endswith(ZERO_W_REFUSAL), result.stderr[-3000:]
                assert not lines(result, f'{TEXTURE_DISPATCH} translated'), result.stderr[-3000:]
                assert got == initial, 'the zero WSize Dispatch changed the volume'
                continue
            assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]
            assert got == expected, 'native table views changed %d unexpected words\n%s' % (
                sum(a != b for a, b in zip(got, expected)), result.stderr[-3000:])
            T.clean(result)
    assert all(p.name.startswith('req_') or p.name == 'native' for p in cache.iterdir())
    if not no_waves:
        print('native table views: volume words match, and the sampler and zero WSize controls leave them untouched')


def half4_program(f32_words):
    moves = b''.join(struct.pack('<2I', 0x7e0002ff | reg << 17, bits)
                     for reg, bits in enumerate(f32_words, 10))
    # v_cvt_f16_f32 vN, vN, followed by v_pack_b32_f16 v[14:15] from pairs in v[10:13].
    converts = b''.join(struct.pack('<I', 0x7e001500 | reg << 17 | reg) for reg in range(10, 14))
    packs = b''.join(struct.pack('<2I', 0xd7110000 | dest, (256 + low) | (257 + low) << 9)
                     for dest, low in ((14, 10), (15, 12)))
    return A.ARRAY_PROLOGUE + bytes.fromhex('7fc08cbf') + moves + converts + packs + D.STORE + A.ARRAY_END


def texture_outputs(directory, consumer, dxc_dir, adapter, no_waves):
    cache, env = prepare_case(directory, adapter)
    env['X2W_NATIVE_SHADERS'] = dxc_dir
    # One UAV at guest record 2, with the same root layout as the volume store fixture.
    Path(env['X2W_TEST_COMPUTE_ROOT']).write_bytes(
        C.root_blob(3, table_range=True)[:-24] + struct.pack('<6I', 1, 1, 0, 0, 0, 2))
    # Include ties on both sides of the even-half rule and values just above/below a midpoint.
    f32_words = (0x3f801000, 0xbf803000, 0x3e801001, 0xc0000fff)
    floats = struct.unpack('<4f', struct.pack('<4I', *f32_words))
    half_values = struct.unpack('<4H', struct.pack('<4e', *floats))
    cases = (
        ('texture-line', T.store_program('line'), T.store_group('line'), 1, 1, 0, 'I', T.STORE_VALUES),
        ('texture-array', A.array_program(1), A.fixture.GROUP, 8, 6, 2, 'I', A.ARRAY_VALUES),
        ('texture-store', half4_program(f32_words), D.fixture.GROUP, 8, 1, 0, 'H', half_values),
    )
    for mode, program, group, height, layers, target_layer, unit, values in cases:
        (directory / 'program').write_bytes(program)
        (directory / 'state').write_bytes(state_a(group, T.STORE_RSRC2))
        row = 32 * 4
        plane = row * height
        initial = ([0x5000 + i for i in range(plane)] if unit == 'H' else
                   [0xbf000000 | (z << 12) | i for z in range(layers) for i in range(plane)])
        expected = initial.copy()
        for y in range(group[1]):
            start = target_layer * plane + y * row
            expected[start:start + 32] = values * 8  # 2 groups of 4 threads in X

        got, result = T.run(directory, consumer, env, mode, f'<{len(initial)}{unit}')
        if no_waves:
            assert len(lines(result, NO_WAVES)) == 1, result.stderr[-3000:]
            assert len(lines(result, f'{TEXTURE_DISPATCH} refused: no translation')) == 1, result.stderr[-3000:]
            assert got == initial, f'{mode}: a Dispatch without 32-lane waves wrote the output'
        else:
            assert len(lines(result, 'translated natively: 3 entry values, 0 inputs, 1 outputs')) == 1, result.stderr[-3000:]
            assert len(lines(result, f'{TEXTURE_DISPATCH} translated')) == 1, result.stderr[-3000:]
            assert got == expected, (mode, sum(a != b for a, b in zip(got, expected)), result.stderr[-3000:])
            T.clean(result)
            print(f'native {mode}: all {len(got)} output values match the CPU reference')
    assert all(p.name.startswith('req_') or p.name == 'native' for p in cache.iterdir())


def main(consumer, dxc_dir, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        cache, env = prepare_case(directory, adapter)
        env['X2W_NATIVE_SHADERS'] = dxc_dir
        case_a = (directory / 'program').read_bytes()
        records32 = case_a.replace(LOAD_X8, LOADS_X4)
        assert case_a.count(LOAD_X8) == 1 and [i[3] for i in G.decode(records32)[2:4]] == RECORD_LOADS
        untouched, expected = [C.SENTINEL] * C.WORDS, C.expected('A')

        (directory / 'program').write_bytes(records32)
        got, result = dispatch(directory, consumer, env, 'dispatch-records32')
        print(result.stdout, end='')
        key32 = int(lines(result, 'x2w-draw compute request')[0].split()[3], 16)
        no_waves = bool(lines(result, NO_WAVES))
        if no_waves:
            assert got == untouched, 'a Dispatch without 32-lane waves wrote the output'
            assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        else:
            assert not mismatch(got, expected, result), mismatch(got, expected, result)
            assert len(lines(result, 'translated natively: 2 entry values, 1 inputs, 1 outputs')) == 1, result.stderr[-3000:]
            assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1 and not lines(result, 'refused'), result.stderr[-3000:]
            layer = lines(result, 'x2w-d3d12 debug layer: ')
            errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
            assert not errors, errors[0]
            print('debug layer: ' + ('enabled, no error message' if layer and '0x00000000' in layer[0] else 'unavailable'))
            got, result = dispatch(directory, consumer, env, 'dispatch')
            refused = lines(result, 'refused: ')
            assert got == untouched, 'a Dispatch with no descriptor at table byte 32 wrote the output'
            assert len(refused) == 1 and 'no translation' not in refused[0], result.stderr[-3000:]
        published = sorted(p.name for p in cache.iterdir() if not p.name.startswith('req_') and p.name != 'native')
        assert not published, f'the native route left {published} in the cache'

        (directory / 'program').write_bytes(case_a)
        got, result = dispatch(directory, consumer, env, 'dispatch')
        reason = lines(result, 'not translated natively: ')
        assert got == untouched, 'a Dispatch of a program the C++ compiler refused wrote the output'
        assert len(reason) == 1 and 'unaligned descriptor table entry' in reason[0], result.stderr[-3000:]
        assert len(lines(result, 'Dispatch(3, 2, 2) refused: no translation')) == 1, result.stderr[-3000:]
        key = int(reason[0].split('compute pipeline ')[1].split()[0], 16)

        out = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True, capture_output=True,
                             text=True, env=dict(os.environ, X2W_DXC=str(Path(dxc_dir) / 'dxc.exe'))).stdout
        assert f'{key:016x}: ok' in out, out
        got, result = dispatch(directory, consumer, env, 'dispatch')
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert lines(result, 'not translated natively: ') and len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, \
            result.stderr[-3000:]

        assert f'{key32:016x}: ok' in out, out
        (directory / 'program').write_bytes(records32)
        got, result = dispatch(directory, consumer, env, 'dispatch-records32')
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert bool(lines(result, f'{key32:016x} translated natively')) != no_waves, result.stderr[-3000:]
        assert len(lines(result, 'Dispatch(3, 2, 2) translated')) == 1, result.stderr[-3000:]

        table_views(directory / 'table-views', consumer, dxc_dir, adapter, no_waves)
        texture_outputs(directory / 'texture-outputs', consumer, dxc_dir, adapter, no_waves)
        controls = ('the C++ compiler refuses case A with every word untouched, and a published translation then runs '
                    'it')
        if no_waves:
            print(f'startup_compute_dispatch_native: SKIP on {adapter}: the adapter has no 32-lane waves. Checked: the '
                  f'native translation is refused by that name with every word untouched; {controls}; the first '
                  f'program matches all {C.WORDS} words through its published translation')
            sys.exit(77)
        print(f'startup_compute_dispatch_native: PASS on {adapter} (all {C.WORDS} words match after the in-process '
              f'translation from an empty cache, and again with a published translation in the cache; the plain table '
              f'layout is refused when bound; {controls})')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
