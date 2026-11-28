"""Check gfx2hlsl's pixel-stage image_store on a hand-assembled gfx1013 program (#1996 item F; M200-01 I).

The pixel program loads an image descriptor from descriptor table 0, sets the texel address (1, 2) and
four distinct binary32 values, stores them with image_store ... dmask:0xf dim:SQ_RSRC_IMG_2D, and exports
the same values to mrt0. The words were assembled by hand from the MIMG layout and are decoded back by
tools/disasm_rdna2.py here; --llvm-mc reassembles the source and compares the words.

A full-mask 2D-array program stores the same four values at layer 1. The address is x, y and the
layer, and the manifest line is tex2da. A partial mask on that form is refused.

A second program uses dmask:0x5 (x and z) so VDATA is two registers. The lowering writes only those
components. It does not read the texel. The UAV starts at -1 in every channel, so y and w stay -1.

A third program draws two pixels in one pass. Both address texel (1, 2). The pixel at x=0 stores channel
x (dmask:0x1, 1.0) and the pixel at x=1 stores channel z (dmask:0x4, 2.0). Both updates must survive.
Writing the second value into y instead of z is the negative control.

Without arguments: translation controls and refusals. With draw: also compile the translation with DXC and
draw it through offscreen_check --startup-image-store, which reads every texel of a 4x4 image back. Negative
controls: the translator refuses the program when the lowering is removed, and the translated shader with
its store deleted, or with x and y exchanged, leaves texel (1, 2) at its starting value. A binding that
asks for a typed UAV load is refused before the draw when the format cannot load.

The store runs under the lane's exec bit. Two variants put v_cmpx_eq_u32 before the store, so exec is not
known at translation time: with 1 == v0 the lane stays on and the texel is written, with 2 == v0 every lane
is off and the texel keeps its starting value. A control draws the exec-off variant with the guard removed
from the store and must see the texel written.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'tools'))
from startup_image_3d_check import FULLSCREEN_VS, G, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402
import disasm_rdna2 as D  # noqa: E402

VALUES = (1.0, 2.0, 3.0, 4.0)
BITS = ['%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in VALUES]

SOURCE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32 v0, 1
v_mov_b32 v1, 2
v_mov_b32 v4, 1.0
v_mov_b32 v5, 2.0
v_mov_b32 v6, 0x40400000
v_mov_b32 v7, 4.0
s_waitcnt lgkmcnt(0)
image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
STORE = bytes.fromhex('080f20f000040400')
# dmask:0x5 → VDATA v[4:5] written to channels x then z (same PREFIX: v4=1.0, v5=2.0).
STORE_XZ = bytes.fromhex('080520f000040400')
ARRAY_STORE = bytes.fromhex('280f20f000040400')
PREFIX = bytes.fromhex('04040cf4000000fa' '8102007e' '8202027e' 'f202087e' 'f4020a7e' 'ff020c7e00004040'
                       'f6020e7e' '7fc08cbf')
SUFFIX = bytes.fromhex('703f8cbf' '0f1800f804050607' '000081bf')
WORDS = PREFIX + STORE + SUFFIX
WORDS_XZ = PREFIX + STORE_XZ + SUFFIX
# Two pixels, one texel. Packed position is v0 (SPI input bit 15). The pixel whose x is 0 stores
# channel x; the pixel whose x is 1 stores channel z. Both write texel (1, 2).
SOURCE_LANES = """\
v_cmp_eq_u32_e32 0, v0
v_mov_b32_e32 v2, 1
v_cndmask_b32_e32 v3, 0, v2
v_mov_b32_e32 v0, 1
v_mov_b32_e32 v1, 2
v_mov_b32_e32 v4, 1.0
v_mov_b32_e32 v6, 2.0
s_load_dwordx8 s[16:23], s[8:9], 0x0
s_waitcnt lgkmcnt(0)
v_cmpx_eq_u32_e32 1, v3
image_store v4, v[0:1], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D
s_mov_b64 exec, -1
v_cmpx_eq_u32_e32 0, v3
image_store v6, v[0:1], s[16:23] dmask:0x4 dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v4, v6, v6 done vm
s_endpgm
"""
WORDS_LANES = bytes.fromhex(
    '8000847d' '8102047e' '80040602' '8102007e' '8202027e' 'f202087e' 'f4020c7e'
    '04040cf4000000fa' '7fc08cbf' '8106a47d' '080120f000040400' 'c104febe' '8006a47d'
    '080420f000060400' '703f8cbf' '0f1800f804040606' '000081bf')
LANE_X = 'if (exec & 1) w0[uint2(v0, v1)].x = asfloat(v4);'
LANE_Z = 'if (exec & 1) w0[uint2(v0, v1)].z = asfloat(v6);'
STORE_LINE = 'w0[uint2(v0, v1)] = asfloat(uint4(v4, v5, v6, v7));'
# Init -1: an xz component store leaves the texel at (1.0, -1.0, 2.0, -1.0). No texel read.
SPARSE_LINE = 'w0[uint2(v0, v1)].xz = float2(asfloat(v4), asfloat(v5));'
GUARDED_LINE = 'if (exec & 1) ' + STORE_LINE
SPARSE_BITS = ['3f800000', 'bf800000', '40000000', 'bf800000']  # 1.0, -1.0, 2.0, -1.0
ARRAY_SOURCE = SOURCE.replace(
    'image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D',
    'v_mov_b32 v2, 1\nimage_store v[4:7], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY',
)
ARRAY_STORE_LINE = 'w0[uint3(v0, v1, v2)] = asfloat(uint4(v4, v5, v6, v7));'
ARRAY_GUARDED_LINE = 'if (exec & 1) ' + ARRAY_STORE_LINE


def mov_v2_inline(constant):
    """v_mov_b32_e32 v2, <constant>: VOP1 opcode 0x01, vdst=2, inline integer 0..64 (0x80 + n)."""
    return struct.pack('<I', 0x7e040280 + constant)


def array_program(layer):
    return PREFIX + mov_v2_inline(layer) + ARRAY_STORE + SUFFIX


ARRAY_WORDS = array_program(1)
ARRAY_WORDS_LAST = array_program(3)


def cmpx_eq(constant):
    """v_cmpx_eq_u32_e32 <constant>, v0: VOPC (bits 31:25 = 0x3e), opcode 0xd2, VSRC1 v0, SRC0 an inline
    integer 1..64 (0x80 + n). In gfx10 the VOPC v_cmpx form writes only exec."""
    return struct.pack('<I', 0x7c000000 | (0xd2 << 17) | (0 << 9) | (0x80 + constant))


EXEC_ON = PREFIX + cmpx_eq(1) + STORE + SUFFIX   # v0 is 1: the lane stays active
EXEC_OFF = PREFIX + cmpx_eq(2) + STORE + SUFFIX  # v0 is 1: every lane is turned off before the store
ARRAY_EXEC_ON = PREFIX + mov_v2_inline(1) + cmpx_eq(1) + ARRAY_STORE + SUFFIX
ARRAY_EXEC_OFF = PREFIX + mov_v2_inline(1) + cmpx_eq(2) + ARRAY_STORE + SUFFIX


def translate(words):
    return G.Pipeline(VS_WORDS, words, root_blob())


def translate_lanes(words):
    # SPI_PS_INPUT bit 15 alone: the packed position occupies v0.
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=(0x8000, 0x8000))


def refused(words, *needles):
    try:
        translate(words)
    except G.Unsupported as error:
        text = str(error)
        assert all(n in text for n in needles), (needles, text)
        return text
    raise AssertionError('accepted: %s' % (needles,))


def with_store(store):
    return PREFIX + store + SUFFIX


def patched(byte, mask, value=None):
    store = bytearray(STORE)
    store[byte] = value if value is not None else store[byte] | mask
    return bytes(store)


def controls():
    text = D.decode_instruction(STORE, 0)[2]
    assert text == 'image_store v[4:7], v[0:1], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D', text
    pipe = translate(WORDS)
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    assert 'RWTexture2D<float4> w0 : register(u0);' in ps, ps
    assert '    ' + STORE_LINE in ps, ps
    assert 'register(t' not in ps, ps
    assert 'uav 0 tex2d table 0 0' in manifest and 'srv ' not in manifest, manifest
    assert manifest.startswith('x2w-manifest 2\n'), manifest
    print('store: %s manifest uav 0 tex2d table 0 0' % STORE_LINE)

    # Cache-policy bits and unorm (integer texel coordinates) do not change the written texel.
    for name, store in (('glc', patched(1, 0x20)), ('slc', patched(3, 0x02)),
                        ('dlc', patched(0, 0x80)), ('unorm', patched(1, 0x10))):
        assert D.decode_instruction(store, 0)[2].endswith(' ' + name), D.decode_instruction(store, 0)
        assert STORE_LINE in translate(with_store(store)).hlsl_ps(), name
    print('glc, slc, dlc and unorm accepted')

    # 2D-array pixel store (#2742): three address registers, RWTexture2DArray<float4>, manifest tex2da.
    array_text = D.decode_instruction(ARRAY_STORE, 0)[2]
    assert array_text == 'image_store v[4:7], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY', array_text
    array_pipe = translate(ARRAY_WORDS)
    array_ps, array_manifest = array_pipe.hlsl_ps(), array_pipe.manifest()
    assert 'RWTexture2DArray<float4> w0 : register(u0);' in array_ps, array_ps
    assert '    ' + ARRAY_STORE_LINE in array_ps, array_ps
    assert 'register(t' not in array_ps, array_ps
    assert 'uav 0 tex2da table 0 0' in array_manifest and 'srv ' not in array_manifest, array_manifest
    assert array_manifest.startswith('x2w-manifest 2\n'), array_manifest
    print('array store: %s manifest uav 0 tex2da table 0 0' % ARRAY_STORE_LINE)

    # Refusals, each by name: packed data or addresses, fault-status returns, a 128-bit descriptor,
    # and other image dimensions.
    cases = (('d16', patched(7, 0x80)), ('a16', patched(7, 0x40)), ('tfe', patched(2, 0x01)),
             ('lwe', patched(2, 0x02)), ('r128', patched(1, 0x80)), ('reserved', patched(1, 0x40)),
             ('SQ_RSRC_IMG_3D', bytes.fromhex('100f20f000040400')),
             ('SQ_RSRC_IMG_CUBE', bytes.fromhex('180f20f000040400')),
             ('SQ_RSRC_IMG_1D', bytes.fromhex('000f20f000040400')))
    for name, store in cases:
        decoded = D.decode_instruction(store, 0)[2]
        message = refused(with_store(store), 'image_store')
        print('refused %-20s %s' % (name + ':', message))
        assert decoded.startswith('image_store'), decoded
    # 2D-array requires three address registers, a full mask, and still refuses pixel d16.
    try:
        array_pipe.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY'])
    except G.Unsupported as error:
        assert 'x, y and layer expected' in str(error), error
    else:
        raise AssertionError('2D-array store with two address registers was accepted')
    try:
        array_pipe.ps.op_image_store(['v[4:5]', 'v[0:2]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D_ARRAY', 'd16'])
    except G.Unsupported as error:
        assert 'modifier d16' in str(error), error
    else:
        raise AssertionError('pixel 2D-array d16 store was accepted')
    try:
        array_pipe.ps.op_image_store(['v[4:5]', 'v[0:2]', 's[16:23]'], ['dmask:0x5', 'dim:SQ_RSRC_IMG_2D_ARRAY'])
    except G.Unsupported as error:
        assert 'writes all four channels' in str(error), error
    else:
        raise AssertionError('2D-array sparse dmask was accepted')
    print('refused 2D-array with two address registers, pixel d16 and sparse dmask 0x5')
    # Sparse DMASK on a pixel 2D store: selected components only, in VDATA order. The UAV starts at
    # -1; dmask:0x5 with v[4:5] = (1.0, 2.0) yields (1, -1, 2, -1) without reading the texel.
    decoded_xz = D.decode_instruction(STORE_XZ, 0)[2]
    assert decoded_xz == 'image_store v[4:5], v[0:1], s[16:23] dmask:0x5 dim:SQ_RSRC_IMG_2D', decoded_xz
    sparse_ps = translate(WORDS_XZ).hlsl_ps()
    assert '    ' + SPARSE_LINE in sparse_ps, sparse_ps
    assert 't = w0[' not in sparse_ps, sparse_ps
    assert 'typedload' not in translate(WORDS_XZ).manifest()
    print('sparse dmask 0x5: %s' % SPARSE_LINE)
    partial = {
        0x7: 'w0[uint2(v0, v1)].xyz = float3(asfloat(v4), asfloat(v5), asfloat(v6));',
        0x1: 'w0[uint2(v0, v1)].x = asfloat(v4);',
        0x3: 'w0[uint2(v0, v1)].xy = float2(asfloat(v4), asfloat(v5));',
    }
    for mask, data, count in ((0x7, 'v[4:6]', 3), (0x1, 'v4', 1), (0x3, 'v[4:5]', 2)):
        store = patched(1, 0, mask)
        decoded = D.decode_instruction(store, 0)[2]
        assert decoded.startswith('image_store %s, ' % data) and 'dmask:%#x' % mask in decoded, decoded
        text = translate(with_store(store)).hlsl_ps()
        assert '    ' + partial[mask] in text, (mask, text)
        assert 't = w0[' not in text, text
        print('sparse dmask %#x accepted (%d channel(s))' % (mask, count))
    # Wrong VGPR count for a sparse mask is refused for the data, not the mask.
    try:
        pipe.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0x5', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert '2 VGPRs expected' in str(error), error
    else:
        raise AssertionError('dmask 0x5 with four data registers was accepted')
    print('refused dmask 0x5 with four data registers (v[4:7])')
    # Empty and out-of-range masks still refuse like a full-channel typed store.
    for bad in (0x0, 0x10):
        try:
            pipe.ps.op_image_store(['v4', 'v[0:1]', 's[16:23]'], ['dmask:%#x' % bad, 'dim:SQ_RSRC_IMG_2D'])
        except G.Unsupported as error:
            assert 'writes all four channels' in str(error), error
        else:
            raise AssertionError('dmask %#x was accepted' % bad)
    print('refused dmask 0x0 and 0x10')
    # Four data registers are required for the full mask: three are refused for the data, not the mask.
    try:
        pipe.ps.op_image_store(['v[4:6]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'four VGPRs expected' in str(error), error
    else:
        raise AssertionError('dmask 0xf with three data registers was accepted')
    print('refused dmask 0xf with three data registers (v[4:6])')
    # Compute still requires the full mask (shared handler; pixel-only sparse).
    try:
        pipe.ps.kind = 'cs'
        pipe.ps.op_image_store(['v[4:5]', 'v[0:1]', 's[16:23]'], ['dmask:0x5', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'writes all four channels' in str(error), error
    else:
        raise AssertionError('compute sparse dmask was accepted')
    finally:
        pipe.ps.kind = 'ps'
    print('refused compute-stage sparse dmask 0x5')
    # Mask text that is not an integer is a named refusal, not a ValueError.
    try:
        pipe.ps.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xz', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert str(error) == 'image_store with dmask 0xz: not an integer', error
    else:
        raise AssertionError('a malformed dmask was accepted')
    print('refused dmask 0xz (not an integer)')
    refused(with_store(bytes.fromhex('080f24f000040400')), 'image_store_mip')
    print('refused image_store_mip (not lowered)')
    try:
        pipe.vs.op_image_store(['v[4:7]', 'v[0:1]', 's[16:23]'], ['dmask:0xf', 'dim:SQ_RSRC_IMG_2D'])
    except G.Unsupported as error:
        assert 'outside a pixel stage' in str(error), error
    else:
        raise AssertionError('vertex-stage image_store accepted')
    print('refused image_store outside a pixel stage')

    # Reading and storing the same image in one program would need a read-write view: refused.
    load = bytes.fromhex('080f00f000080400')  # image_load v[8:11], v[0:1], s[16:23] dmask:0xf dim:2D
    assert D.decode_instruction(load, 0)[2].startswith('image_load v[8:11]'), D.decode_instruction(load, 0)
    refused(PREFIX + load + STORE + SUFFIX, 'both read and stored')
    assert translate(PREFIX + load + SUFFIX).manifest().startswith('x2w-manifest 1\n')
    print('refused a load and a store of the same image')

    # Exec not known at translation time: the store, and only the store line, is guarded by the lane's bit.
    for constant, words in ((1, EXEC_ON), (2, EXEC_OFF)):
        decoded = D.decode_instruction(cmpx_eq(constant), 0)[2]
        assert decoded == 'v_cmpx_eq_u32_e32 %d, v0' % constant, decoded
        ps = translate(words).hlsl_ps()
        assert '    exec = ((0x%08xu == v0) ? 1u : 0u);' % constant in ps, ps
        assert '    ' + GUARDED_LINE in ps, ps
        assert ps.count(STORE_LINE) == 1, ps
    # With exec known (the entry mask) the store is not guarded.
    assert GUARDED_LINE not in translate(WORDS).hlsl_ps()
    for constant, words in ((1, ARRAY_EXEC_ON), (2, ARRAY_EXEC_OFF)):
        ps = translate(words).hlsl_ps()
        assert '    exec = ((0x%08xu == v0) ? 1u : 0u);' % constant in ps, ps
        assert '    ' + ARRAY_GUARDED_LINE in ps, ps
        assert ps.count(ARRAY_STORE_LINE) == 1, ps
    assert ARRAY_GUARDED_LINE not in array_ps
    print('exec after v_cmpx: %s' % GUARDED_LINE)

    # Same texel, disjoint masks: two invocations write x and z, and neither store reads the texel.
    lanes = translate_lanes(WORDS_LANES).hlsl_ps()
    assert 'v0 = (uint(i.pos.x) & 0xffffu) | ((uint(i.pos.y) & 0xffffu) << 16);' in lanes, lanes
    assert '    ' + LANE_X in lanes and '    ' + LANE_Z in lanes, lanes
    assert 't = w0[' not in lanes and 'typedload' not in translate_lanes(WORDS_LANES).manifest(), lanes
    swapped = lanes.replace(LANE_Z, LANE_Z.replace('.z = asfloat(v6)', '.y = asfloat(v6)'))
    assert swapped != lanes and '.z = asfloat(v6)' not in swapped
    print('same texel: %s and %s' % (LANE_X, LANE_Z))
    print('control: swapping the z store onto y changes the lowering')

    # A binding that asks for a typed load is refused in pixel_uav_view, before the view is filled.
    # The translator does not emit that request for a component store.
    draw_inl = (HERE.parent / 'src/startup_stubs/d3d12_draw.inl').read_text(encoding='utf-8')
    view = draw_inl.split('const char* pixel_uav_view', 1)[1].split('\nstruct Translation', 1)[0]
    assert 'needs_typed_load' in view and 'typed_uav_load_refusal' in view and 'bool array' in view, view
    assert view.find('typed_uav_load_refusal') < view.find('out.Format = format'), view
    parser = draw_inl.split('const char* read_pixel_uav_line', 1)[1].split('\nbool float_uav_format', 1)[0]
    assert 'typedload' in parser and 'needs_typed_load' in parser and 'tex2da' in parser, parser
    store_cpp = (HERE / 'startup_image_store.cpp').read_text(encoding='utf-8')
    gate = store_cpp.split('check_startup_image_store_typed_load', 1)[1]
    assert 'DrawInstanced' not in gate and 'typed_uav_load_refusal' in gate, gate
    print('typed UAV load is refused before the view is filled and before any draw')

    # Negative control: without the lowering the program is refused as before.
    lowering = G.Stage.op_image_store
    del G.Stage.op_image_store
    try:
        refused(WORDS, 'image_store')
    finally:
        G.Stage.op_image_store = lowering
    print('control: with op_image_store removed the program is refused')


def check_binary(llvm_mc):
    for name, source, words in (
            ('full mask', SOURCE, WORDS),
            ('2d array', ARRAY_SOURCE, ARRAY_WORDS),
            ('same texel', SOURCE_LANES, WORDS_LANES)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def mismatch_line(stderr):
    """The texel_mismatch report: on a machine with several adapters stderr starts with adapter notices."""
    return next(line for line in stderr.splitlines() if 'texel_mismatch ' in line)


def check_draw(dxc, renderer, adapter):
    source = translate(WORDS).hlsl_ps()
    expected = ','.join(BITS)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = compile_hlsl(dxc, 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)

        def draw(text, label, want=expected, coverage=None):
            ps = compile_hlsl(dxc, 'dxc', text, label, 'ps_6_0', directory)
            command = [renderer, adapter, '--startup-image-store', str(vs), str(ps), '1,2', want]
            if coverage is not None:
                command.append(str(coverage))
            return subprocess.run(command, capture_output=True, text=True)

        run = draw(source, 'store')
        sys.stdout.write(run.stdout)
        if run.returncode == 77:
            sys.exit(77)
        if run.returncode:
            raise AssertionError('store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw: texel (1, 2) = %s, all other texels untouched' % expected)
        untouched = 'actual=' + ','.join(['bf800000'] * 4)
        for label, text, needle in (
                ('store deleted', source.replace(STORE_LINE, ''), 'x=1 y=2 expected=%s %s' % (expected, untouched)),
                ('x and y exchanged', source.replace('w0[uint2(v0, v1)]', 'w0[uint2(v1, v0)]'), 'x=2 y=1')):
            assert text != source, label
            run = draw(text, label.replace(' ', '_'))
            if run.returncode == 0 or 'texel_mismatch ' + needle not in run.stderr:
                raise AssertionError('control %s was not detected\n%s\n%s' % (label, run.stdout, run.stderr))
            print('control %s: fails as expected (%s)' % (label, mismatch_line(run.stderr)))

        # Exec after v_cmpx: an active lane writes the texel, an inactive one leaves it at its starting value.
        untouched_bits = ','.join(['bf800000'] * 4)
        on, off = translate(EXEC_ON).hlsl_ps(), translate(EXEC_OFF).hlsl_ps()
        for label, text, want in (('exec on', on, expected), ('exec off', off, untouched_bits)):
            run = draw(text, label.replace(' ', '_'), want)
            if run.returncode:
                raise AssertionError('%s draw failed\n%s\n%s' % (label, run.stdout, run.stderr))
            print('draw %s: texel (1, 2) = %s, all other texels untouched' % (label, want))
        # Control: the exec-off shader with the guard removed from the store writes the texel.
        unguarded = off.replace(GUARDED_LINE, STORE_LINE)
        assert unguarded != off
        run = draw(unguarded, 'exec_off_unguarded', untouched_bits)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (untouched_bits, expected)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control exec guard removed was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control exec guard removed: fails as expected (%s)' % mismatch_line(run.stderr))

        # Sparse dmask 0x5: only x and z change; y and w stay at the UAV clear (-1.0).
        sparse_want = ','.join(SPARSE_BITS)
        sparse_source = translate(WORDS_XZ).hlsl_ps()
        assert SPARSE_LINE in sparse_source, sparse_source
        run = draw(sparse_source, 'store_xz', sparse_want)
        if run.returncode:
            raise AssertionError('sparse store draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw sparse dmask 0x5: texel (1, 2) = %s' % sparse_want)
        # Control: writing all four channels instead of the selected components must mismatch on y and w.
        full_instead = sparse_source.replace(SPARSE_LINE, STORE_LINE)
        assert full_instead != sparse_source
        run = draw(full_instead, 'sparse_as_full', sparse_want)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (sparse_want, expected)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control sparse-as-full was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control sparse-as-full: fails as expected (%s)' % mismatch_line(run.stderr))

        # Two invocations, one draw, one texel, disjoint masks. Both component writes must survive.
        lanes = translate_lanes(WORDS_LANES).hlsl_ps()
        assert LANE_X in lanes and LANE_Z in lanes, lanes
        assert 't = w0[' not in lanes, lanes
        run = draw(lanes, 'same_texel', sparse_want, coverage=2)
        if run.returncode:
            raise AssertionError('same-texel draw failed\n%s\n%s' % (run.stdout, run.stderr))
        print('draw same texel disjoint masks: texel (1, 2) = %s' % sparse_want)
        # Negative control: the z store written into y must mismatch.
        swapped = lanes.replace(LANE_Z, LANE_Z.replace('.z = asfloat(v6)', '.y = asfloat(v6)'))
        assert swapped != lanes and '.z = asfloat(v6)' not in swapped
        run = draw(swapped, 'same_texel_y', sparse_want, coverage=2)
        swapped_actual = '3f800000,40000000,bf800000,bf800000'  # (1, 2, -1, -1)
        needle = 'texel_mismatch x=1 y=2 expected=%s actual=%s' % (sparse_want, swapped_actual)
        if run.returncode == 0 or needle not in run.stderr:
            raise AssertionError('control same-texel channel swap was not detected\n%s\n%s' % (run.stdout, run.stderr))
        print('control same-texel channel swap: fails as expected (%s)' % mismatch_line(run.stderr))

        refused_load = subprocess.run([renderer, adapter, '--startup-image-store-reject-load'],
                                      capture_output=True, text=True)
        if refused_load.returncode == 77:
            sys.exit(77)
        if refused_load.returncode == 0 or 'typed_uav_load_refused' not in refused_load.stderr:
            raise AssertionError('typed UAV load was not refused before the draw\n%s\n%s' % (
                refused_load.stdout, refused_load.stderr))
        if 'image store:' in refused_load.stdout:
            raise AssertionError('typed UAV load refusal drew\n%s' % refused_load.stdout)
        print('typed UAV load refused before the draw')


def main():
    controls()
    if len(sys.argv) == 1:
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
