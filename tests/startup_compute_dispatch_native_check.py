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
5. Constant samplers (#3149): four samples of a two-mip 2x1 R32_FLOAT texture feed a 3D float image store, each
   through its own constant sampler: image_sample_lz (wrap), image_sample_l with LOD 1.25 under a [0, 0] clamp
   (mirror), image_sample_l with LOD 0 under a [1, 1] clamp, and image_sample_c_lz (LESS). U and the reference
   double with the global X, so the eight threads of a row cover both texels, the three address modes and both
   comparison results. Every volume word is compared with a CPU model of the sampler words. Control: with the first
   sampler's words loaded from memory (the table) instead, the C++ compiler refuses the program and the volume
   keeps its words.

A direct compute module declares wave size 32. On an adapter without 32-lane waves, steps 1 and 5 instead check that
the native translation is refused by that name with the output untouched (the memory sampler control runs as
written), steps 2 and 3 run as written, step 4 runs the
first program through its published translation, and the check exits 77. Exit 77 also when the adapter is not
available.
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
import startup_gfx2hlsl_image_store_3d1d_check as V
from startup_compute_dispatch_check import dispatch, lines, prepare_case
from startup_compute_translate_check import state_a
from startup_stream_check import gfx2hlsl as G

LOAD_X8 = bytes.fromhex('05030cf4' '000000fa')
LOADS_X4 = bytes.fromhex('050308f4' '000000fa' '050408f4' '200000fa')
RECORD_LOADS = ['s_load_dwordx4 s[12:15], s[10:11], 0x0', 's_load_dwordx4 s[16:19], s[10:11], 0x20']
NO_WAVES = 'not translated natively: the adapter has no 32-lane waves'
# Step 5. Sampler words [address, LOD limits, filter, border] in register order: U address mode in bits 2:0 over
# clamp V, MIN_LOD and MAX_LOD as 4.8 fixed point in bits 11:0 and 23:12 of word 1, point filters.
WRAP, MIRROR, CLAMP, POINT_MIP = 0, 1, 2, 1 << 26
SAMPLERS = (
    (WRAP | CLAMP << 3, 0xfff << 12, POINT_MIP, 0),       # image_sample_lz
    (MIRROR | CLAMP << 3, 0, POINT_MIP, 0),               # image_sample_l, LOD 1.25: MAX_LOD 0 keeps mip 0
    (CLAMP | CLAMP << 3, 0x100 << 12 | 0x100, POINT_MIP, 0),  # image_sample_l, LOD 0: MIN_LOD 1 selects mip 1
    (CLAMP | CLAMP << 3 | 1 << 12 | 1 << 27, 0, 0, 0),    # image_sample_c_lz: LESS, point
)
U_BASE, REFERENCE_BASE = 0x3e99999a, 0xbe400000  # 0.3 and -0.1875: adding X << 23 doubles them X times
SAMPLED_PROGRAM = [
    's_load_dwordx8 s[24:31], s[12:13], 0x20', 's_waitcnt lgkmcnt(0)',
    *('s_mov_b32 s%d, 0x%x' % (40 + i, word) for i, word in enumerate(w for words in SAMPLERS for w in words)),
    's_mov_b32 s56, 0x3e99999a', 's_mov_b32 s57, 0xbe400000',
    'v_lshl_add_u32 v12, v2, 23, s56', 'v_mov_b32_e32 v13, 0x3e800000', 'v_mov_b32_e32 v14, 0x3fa00000',
    'image_sample_lz v8, v[12:13], s[24:31], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'image_sample_l v9, v[12:14], s[24:31], s[44:47] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'v_mov_b32_e32 v14, 0',
    'image_sample_l v10, v[12:14], s[24:31], s[48:51] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    'v_lshl_add_u32 v15, v2, 23, s57', 'v_mov_b32_e32 v16, v12', 'v_mov_b32_e32 v17, v13',
    'image_sample_c_lz v11, v[15:17], s[24:31], s[52:55] dmask:0x1 dim:SQ_RSRC_IMG_2D',
    's_waitcnt vmcnt(0) lgkmcnt(15)',
]
MEMORY_SAMPLER = 's_load_dwordx4 s[40:43], s[12:13], 0x60'
MEMORY_SAMPLER_REFUSAL = 'nonconstant sampler'


def mismatch(got, expected, result):
    wrong = [i for i in range(C.WORDS) if got[i] != expected[i]]
    return '' if not wrong else 'word=%d got=%08x expected=%08x (%d words differ)\n%s' % (
        wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])


def scalar_window_case(consumer, dxc_dir, adapter):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        _, env = prepare_case(directory, adapter)
        env.update(X2W_NATIVE_SHADERS=dxc_dir, X2W_NATIVE_SHADERS_ONLY='1')
        program = directory / 'program'
        words = program.read_bytes()
        op = {instruction: offset for offset, _, instruction, _ in G.decode(words)}
        # Keep case A's output address calculation. Read the scalar pair at +36
        # through the three-record input at table byte 64, then store s10.
        load_descriptors = bytes.fromhex('050308f4400000fa 050408f4200000fa')
        read_scalar = bytes.fromhex('860224f4240000fa 7fc08cbf 0a020a7e')
        wait = next(row[0] for row in G.decode(words) if row[2] == 's_waitcnt')
        program.write_bytes(words[:op['s_load_dwordx8']] + load_descriptors +
                            words[wait:op['buffer_load_dword']] + read_scalar + words[op['buffer_store_dword']:])
        source = directory / 'input'
        env['X2W_TEST_DISPATCH_IN'] = str(source)
        for value in (1024, 0x45678):
            data = [0x11000000 + i for i in range(C.WORDS)]
            data[9] = value
            source.write_bytes(struct.pack(f'<{C.WORDS}I', *data))
            expected = [value] * 288 + [C.SENTINEL] * 32
            got, result = dispatch(directory, consumer, env, 'dispatch-scalar-window')
            if lines(result, NO_WAVES):
                assert got == [C.SENTINEL] * C.WORDS
                return
            assert not mismatch(got, expected, result), mismatch(got, expected, result)
            assert lines(result, 'translated natively'), result.stderr[-3000:]
        got, result = dispatch(directory, consumer, env, 'dispatch-scalar-window-short')
        assert got == [C.SENTINEL] * C.WORDS, result.stderr[-3000:]
        assert lines(result, 'refused: an input is in title memory outside any placed buffer'), result.stderr[-3000:]


def sampled_program(memory_sampler):
    def s_mov(register, word):
        return struct.pack('<2I', 0xbe8003ff | register << 16, word)

    def scaled(register, base):  # v_lshl_add_u32 v<register>, v2, 23, s<base>
        return struct.pack('<2I', 0xd7460000 | register, 0x102 | 151 << 9 | base << 18)

    def image(op, data, address, sampler):
        return struct.pack('<2I', 0xf0000108 | op << 18, address | data << 8 | 6 << 16 | sampler // 4 << 21)

    # Keep the volume fixture's table pointer, UAV descriptor, coordinates and final store.
    wait = bytes.fromhex('7fc08cbf')
    prologue, _, tail = V.program('volume').partition(wait)
    words = [word for sampler in SAMPLERS for word in sampler]
    body = (bytes.fromhex('06060cf4200000fa') + wait + b''.join(s_mov(40 + i, word) for i, word in enumerate(words))
            + s_mov(56, U_BASE) + s_mov(57, REFERENCE_BASE) + scaled(12, 56)
            + struct.pack('<4I', 0x7e1a02ff, 0x3e800000, 0x7e1c02ff, 0x3fa00000)  # V 0.25, LOD 1.25
            + image(39, 8, 12, 40) + image(36, 9, 12, 44) + bytes.fromhex('80021c7e') + image(36, 10, 12, 48)
            + scaled(15, 57) + bytes.fromhex('0c03207e' '0d03227e') + image(47, 11, 15, 52) + bytes.fromhex('700f8cbf'))
    assert [i[3] for i in G.decode(body)] == SAMPLED_PROGRAM, G.decode(body)
    if memory_sampler:  # after the literal words, so the sample reads the loaded ones
        body = body.replace(scaled(12, 56), bytes.fromhex('060a08f4600000fa') + scaled(12, 56))
        assert MEMORY_SAMPLER in [i[3] for i in G.decode(body)]
    return prologue + body + tail[32:]  # the four samples replace the four literal moves


def sampled_reference(x, mips):
    """The four words thread x stores: what each sampler's address mode, LOD clamp and comparison select."""
    def f32(bits):
        return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]

    def texel(sampler, lod):
        address, limits = SAMPLERS[sampler][:2]
        low, high = (limits & 0xfff) / 256, (limits >> 12 & 0xfff) / 256
        mip = mips[min(int(min(max(lod, low), high) + 0.5), len(mips) - 1)]  # point mip filter
        u = f32(U_BASE + (x << 23))
        u = u % 1 if address & 7 == WRAP else 1 - abs(u % 2 - 1) if address & 7 == MIRROR else min(max(u, 0), 1)
        return mip[min(int(u * len(mip)), len(mip) - 1)]

    assert SAMPLERS[3][0] >> 12 & 7 == 1, 'the model compares with LESS'
    passed = f32(REFERENCE_BASE + (x << 23)) < f32(texel(3, 0))
    return texel(0, 0), texel(1, 1.25), texel(2, 0), 0x3f800000 if passed else 0


def sampled_views(directory, consumer, dxc_dir, adapter, no_waves):
    cache, env = prepare_case(directory, adapter)
    env['X2W_NATIVE_SHADERS'] = dxc_dir
    # The guest table: two SRV records (the texture is record 1), then one UAV in record 2.
    blob = bytearray(C.root_blob(3, table_range=True)[:-24])
    blob.extend(struct.pack('<12I', 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2))
    payload = 44  # DXBC header, part offset, and RTS0 part header
    struct.pack_into('<I', blob, 24, len(blob))
    struct.pack_into('<I', blob, 40, len(blob) - payload)
    struct.pack_into('<I', blob, payload + 16, len(blob) - payload)  # static sampler offset
    struct.pack_into('<I', blob, payload + 60, 2)  # descriptor table range count
    Path(env['X2W_TEST_COMPUTE_ROOT']).write_bytes(blob)
    group = V.group('volume')
    (directory / 'state').write_bytes(state_a(group, A.RSRC2))
    plane = 32 * 8 * 4
    initial = [0xbf000000 | (z << 12) | i for z in range(6) for i in range(plane)]
    # The input's mip 0 holds the first two sentinel words and its mip 1 the word at upload byte 512.
    mips = (initial[:2], initial[128:129])
    expected = initial.copy()
    for y in range(group[1]):
        start = 2 * plane + y * 32 * 4  # view FirstWSlice 1, shader Z 1
        expected[start:start + 32] = [word for x in range(8) for word in sampled_reference(x, mips)]
    # The row must tell the cases apart: both texels under wrap and mirror in different threads, both comparison results.
    rows = [expected[2 * plane + 4 * x:2 * plane + 4 * x + 4] for x in range(8)]
    assert all(len({row[c] for row in rows}) == 2 for c in (0, 1, 3)) and len({tuple(row[:2]) for row in rows}) > 2, rows

    for memory_sampler in (False, True):
        (directory / 'program').write_bytes(sampled_program(memory_sampler))
        got, result = T.run(directory, consumer, env, 'texture-volume-native-sampled', '<%dI' % len(initial))
        if memory_sampler or no_waves:
            reason = lines(result, 'not translated natively: ')
            want = MEMORY_SAMPLER_REFUSAL if memory_sampler else NO_WAVES.split(': ')[1]
            assert len(reason) == 1 and want in reason[0], result.stderr[-3000:]
            assert len(lines(result, 'Dispatch(2, 1, 1) refused: no translation')) == 1, result.stderr[-3000:]
            assert got == initial, 'a refused Dispatch wrote the volume'
            continue
        assert len(lines(result, 'translated natively: 3 entry values, 2 inputs, 1 outputs')) == 1, result.stderr[-3000:]
        assert len(lines(result, 'Dispatch(2, 1, 1) translated')) == 1, result.stderr[-3000:]
        wrong = [i for i in range(len(got)) if got[i] != expected[i]]
        assert not wrong, 'sampled: word %d got=%08x expected=%08x (%d words differ)\n%s' % (
            wrong[0], got[wrong[0]], expected[wrong[0]], len(wrong), result.stderr[-3000:])
        print('constant samplers: all %d volume words match the CPU model (wrap, mirror with MAX_LOD, clamp with '
              'MIN_LOD, LESS comparison); debug layer: %s' % (len(got), T.clean(result)))
    assert all(p.name.startswith('req_') or p.name in ('native', 'revision.txt') for p in cache.iterdir())


def main(consumer, dxc_dir, adapter):
    scalar_window_case(consumer, dxc_dir, adapter)
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
        # The first request writes revision.txt; it is cache metadata, not a publication.
        published = sorted(p.name for p in cache.iterdir()
                           if not p.name.startswith('req_') and p.name not in ('native', 'revision.txt'))
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

        sampled_views(directory / 'sampled', consumer, dxc_dir, adapter, no_waves)
        controls = ('the C++ compiler refuses case A with every word untouched, and a published translation then runs '
                    'it')
        if no_waves:
            print(f'startup_compute_dispatch_native: SKIP on {adapter}: the adapter has no 32-lane waves. Checked: the '
                  f'native translation is refused by that name with every word untouched; {controls}; the first '
                  f'program matches all {C.WORDS} words through its published translation')
            sys.exit(77)
        shared_loads = LOADS_X4[:4] + bytes.fromhex('200000fa') + LOADS_X4[8:]
        (directory / 'program').write_bytes(case_a.replace(LOAD_X8, shared_loads))
        got, result = dispatch(directory, consumer, dict(env, X2W_NATIVE_SHADERS_ONLY='1'), 'dispatch-read-write-uav')
        expected[288:] = [C.input_word(i) for i in range(288, C.WORDS)]
        assert not mismatch(got, expected, result), mismatch(got, expected, result)
        assert lines(result, '0 inputs, 1 outputs'), result.stderr[-3000:]
        errors = [line for line in lines(result, 'x2w-d3d12-debug ') if ' sev=0 ' in line or ' sev=1 ' in line]
        assert not errors, errors[0]
        print('shared guest UAV: in-place reads and writes matched with untouched tail')
        print(f'startup_compute_dispatch_native: PASS on {adapter} (all {C.WORDS} words match after the in-process '
              f'translation from an empty cache, and again with a published translation in the cache; the plain table '
              f'layout is refused when bound; {controls})')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit(__doc__)
    main(*sys.argv[1:])
