"""Level-zero D16 image_load lowering in tools/startup/gfx2hlsl.py (Part of #2023, item D).

  (no arguments)                       translation, decoder, rounding and refusal controls, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  decode <libLLVM>                     disassemble the MIMG words with LLVM's C API and compare with disasm_rdna2.py
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw two
                                       pixel fixtures against 8x2 R16_UNORM (16-bit depth SRV) and 8x2
                                       R32G32B32A32_FLOAT textures into an R32G32B32A32_UINT target, checking all
                                       64 output words per draw and rejecting six lowering mutations

On RDNA 2 (gfx1013), image_load with d16 narrows the enabled dmask channels from the loaded float4 texel to IEEE
binary16 (round to nearest even) and packs them pairwise into ceil(popcount(dmask) / 2) consecutive VGPRs (low half
first, high half second; an odd final channel zeros bits 31:16). HLSL f32tof16 rounding is adapter-dependent (WARP
truncates toward zero, changing 31,740 of the 65,536 R16_UNORM depth values by 1 ULP relative to RDNA hardware);
gfx2hlsl.py therefore emits an integer round-to-nearest-even helper (pack16_rne) when a stage uses image_load with
d16. Explicit-mip image_load_mip and sampler-based MIMG operations with d16 remain refused.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402

IMG_LOAD = 0x00
IMG_LOAD_MIP = 0x01
IMG_SAMPLE = 0x20
IMG_SAMPLE_LZ = 0x27
DIM_1D = 0
DIM_2D = 1
DIM_3D = 2
DIM_CUBE = 3
DIM_2D_ARRAY = 5

WIDTH = 8
HEIGHT = 2
PACKED_INPUT = (0x8000, 0x8000)
UINT_RTV = (42,)  # DXGI_FORMAT_R32G32B32A32_UINT

DEPTH_TEXELS = (
    0x0000, 0x0001, 0x0003, 0x1234, 0x2aad, 0x4008, 0x5555, 0x8000,
    0x8010, 0x8020, 0xa018, 0xc018, 0xe00c, 0xffd0, 0xfffe, 0xffff,
)

FLOAT_TEXELS = (
    (0x3f801000, 0x3f803000, 0x3f801001, 0x3f7fffff),
    (0x387fe000, 0x387fdfff, 0x33800000, 0x33000000),
    (0x33000001, 0xb5c00000, 0x80000000, 0x00000000),
    (0xbf803000, 0xbf801000, 0x477fefff, 0x477ff000),
    (0xc77ff000, 0x7f800000, 0xff800000, 0x7fc00001),
    (0xffc00123, 0x35555555, 0xbeaaaaab, 0x40490fdb),
    (0x3c000000, 0xbc800000, 0x3d000000, 0xbd800000),
    (0x42000000, 0xc2200000, 0x42400000, 0xc2600000),
    (0x3e200000, 0x3ea00000, 0x3ef00000, 0x3f100000),
    (0xbe200000, 0xbea00000, 0xbef00000, 0xbf100000),
    (0x39001000, 0x39003000, 0xb9003000, 0x39001001),
    (0x44002000, 0x44006000, 0xc4006000, 0x44002001),
    (0x34000000, 0x34800000, 0x35000000, 0x35800000),
    (0xb4000000, 0xb4800000, 0xb5000000, 0xb5800000),
    (0x3fc00000, 0xc0000000, 0x40200000, 0xc0400000),
    (0x477fe000, 0xc77fe000, 0x38800000, 0xb8800000),
)


def encode_mimg(op, *, dmask=0xf, dim=DIM_2D, vaddr=0, vdata=4, srsrc=16, ssamp=0, nsa=0, extra_addrs=(),
                dlc=0, unorm=0, glc=0, r128=0, tfe=0, lwe=0, slc=0, a16=0, d16=0, w0_reserved=0, w1_reserved=0):
    w0 = (0xf0000000 | ((op & 0x7f) << 18) | ((op >> 7) & 1) | ((nsa & 0x3) << 1)
          | ((dim & 0x7) << 3) | ((dlc & 1) << 7) | ((dmask & 0xf) << 8)
          | ((unorm & 1) << 12) | ((glc & 1) << 13) | ((r128 & 1) << 15)
          | ((tfe & 1) << 16) | ((lwe & 1) << 17) | ((slc & 1) << 25) | w0_reserved)
    w1 = ((vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16)
          | (((ssamp // 4) & 0x1f) << 21) | ((a16 & 1) << 30) | ((d16 & 1) << 31) | w1_reserved)
    ext = bytes(extra_addrs) + bytes(max(0, 4 * nsa - len(extra_addrs)))
    return struct.pack('<II', w0, w1) + ext


# Entry v0 holds x | (y << 16). v1 and v3 hold out-of-bounds sentinels (63) so dropping a16 on v0
# (reading v[0:1]) or ignoring NSA on [v2, v9] (reading v[2:3]) misses every pixel of the 8x2 texture.
PRE = """\
s_load_dwordx8 s[16:23], s[8:9], 0x0
v_mov_b32_e32 v1, 63
v_and_b32_e32 v2, 7, v0
v_mov_b32_e32 v3, 63
v_and_b32_e32 v8, 7, v0
v_lshrrev_b32_e32 v9, 16, v0
s_waitcnt lgkmcnt(0)
"""
PRE_WORDS = bytes.fromhex(
    '04040cf4000000fa'
    'bf02027e'
    '87000436'
    'bf02067e'
    '87001036'
    '9000122c'
    '7fc08cbf'
)

POST = """\
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
"""
POST_WORDS = bytes.fromhex('703f8cbf' '0f1800f804050607' '000081bf')

MIMG_LLVM_MC = {
    'd16_x': (
        'image_load v4, v[8:9], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0x1, vaddr=8, vdata=4, d16=1),
    ),
    'd16_xz_a16': (
        'image_load v5, v0, s[16:23] dmask:0x5 dim:SQ_RSRC_IMG_2D a16 d16',
        encode_mimg(IMG_LOAD, dmask=0x5, vaddr=0, vdata=5, a16=1, d16=1),
    ),
    'd16_xyz_nsa': (
        'image_load v[6:7], [v2, v9], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D glc slc dlc d16',
        encode_mimg(IMG_LOAD, dmask=0x7, vaddr=2, vdata=6, nsa=1, extra_addrs=(9,), glc=1, slc=1, dlc=1, d16=1),
    ),
    'd16_xyzw': (
        'image_load v[4:5], v[8:9], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16',
        encode_mimg(IMG_LOAD, dmask=0xf, vaddr=8, vdata=4, unorm=1, d16=1),
    ),
    'd16_y': (
        'image_load v6, v[8:9], s[16:23] dmask:0x2 dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0x2, vaddr=8, vdata=6, d16=1),
    ),
    'd16_zw': (
        'image_load v7, v[8:9], s[16:23] dmask:0xc dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD, dmask=0xc, vaddr=8, vdata=7, d16=1),
    ),
    'd16_3d': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_3D, vaddr=0, vdata=4, d16=1),
    ),
    'd16_2da': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_2D_ARRAY, vaddr=0, vdata=4, d16=1),
    ),
    'mip_d16_refused': (
        'image_load_mip v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D d16',
        encode_mimg(IMG_LOAD_MIP, dmask=0xf, vaddr=0, vdata=4, d16=1),
    ),
    '1d_d16_refused': (
        'image_load v4, v0, s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_1D d16',
        encode_mimg(IMG_LOAD, dmask=0x1, dim=DIM_1D, vaddr=0, vdata=4, d16=1),
    ),
    'cube_d16_refused': (
        'image_load v[4:5], v[0:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_CUBE d16',
        encode_mimg(IMG_LOAD, dmask=0xf, dim=DIM_CUBE, vaddr=0, vdata=4, d16=1),
    ),
}
DECODE_KEYS = tuple(MIMG_LLVM_MC)

# Fixture A exercises 1-channel depth d16 (dmask:0x1), 2-channel non-contiguous a16+d16 (dmask:0x5)
# and 3-channel NSA d16 (dmask:0x7, whose odd third half lands in v7[15:0] with v7[31:16] == 0).
PS_SOURCE_A = PRE + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('d16_x', 'd16_xz_a16', 'd16_xyz_nsa')) + '\n' + POST
PS_WORDS_A = PRE_WORDS + b''.join(MIMG_LLVM_MC[k][1] for k in ('d16_x', 'd16_xz_a16', 'd16_xyz_nsa')) + POST_WORDS

# Fixture B exercises full 4-channel d16 (dmask:0xf), single non-zero y channel (dmask:0x2) and upper pair (dmask:0xc).
PS_SOURCE_B = PRE + '\n'.join(MIMG_LLVM_MC[k][0] for k in ('d16_xyzw', 'd16_y', 'd16_zw')) + '\n' + POST
PS_WORDS_B = PRE_WORDS + b''.join(MIMG_LLVM_MC[k][1] for k in ('d16_xyzw', 'd16_y', 'd16_zw')) + POST_WORDS


def f32_bits_from_unorm16(u16):
    return struct.unpack('<I', struct.pack('<f', (u16 & 0xffff) / 65535.0))[0]


def pack16_rne_ref(bits):
    """Independent IEEE-754 binary32-to-binary16 round-to-nearest-even reference."""
    bits &= 0xffffffff
    sign = (bits >> 16) & 0x8000
    abs_bits = bits & 0x7fffffff
    if abs_bits > 0x7f800000:
        return 0x7e00
    if abs_bits >= 0x477ff000:
        return sign | 0x7c00
    val = struct.unpack('<f', struct.pack('<I', bits))[0]
    try:
        return struct.unpack('<H', struct.pack('<e', val))[0]
    except OverflowError:
        return sign | 0x7c00


def pack16_rtz_ref(bits):
    """HLSL f32tof16 (DXBC ftof16) round-toward-zero reference for the RTZ mutation control."""
    bits &= 0xffffffff
    sign = (bits >> 16) & 0x8000
    abs_bits = bits & 0x7fffffff
    if abs_bits > 0x7f800000:
        return 0x7e00  # quiet NaN bit 9 set; mask payload on comparison only when finite
    if abs_bits == 0x7f800000:
        return sign | 0x7c00
    if abs_bits >= 0x47800000:
        return sign | 0x7c00
    if abs_bits >= 0x477fe000:
        return sign | 0x7bff
    if abs_bits >= 0x38800000:
        return sign | ((abs_bits - 0x38000000) >> 13)
    if abs_bits < 0x33800000:
        return sign
    sig = 0x00800000 | (abs_bits & 0x007fffff)
    shift = 126 - (abs_bits >> 23)
    return sign | (sig >> shift)


def texel_f32_words(tex_mode, x, y):
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return (0, 0, 0, 0)
    idx = y * WIDTH + x
    if tex_mode == 'r16_unorm':
        # Default D3D12 SRV mapping for R16_UNORM supplies (r, 0.0f, 0.0f, 1.0f).
        return (f32_bits_from_unorm16(DEPTH_TEXELS[idx]), 0, 0, 0x3f800000)
    return FLOAT_TEXELS[idx]


def pack_pair(lo, hi):
    return (lo & 0xffff) | ((hi & 0xffff) << 16)


def expected_words(fixture, tex_mode, mutant=None):
    narrow = pack16_rtz_ref if mutant == 'rtz' else pack16_rne_ref
    out = []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            rgba = texel_f32_words(tex_mode, x, y)
            hx, hy, hz, hw = (narrow(c) for c in rgba)
            if fixture == 'A':
                # Load 1: v4 = dmask:0x1 at (x, y)
                w0 = rgba[0] if mutant == 'unpacked_x' else hx
                # Load 2: v5 = dmask:0x5 a16 at v0 = x | (y << 16)
                ax, ay = (x | (y << 16), 63) if mutant == 'drop_a16' else (x, y)
                rgba_a16 = texel_f32_words(tex_mode, ax, ay)
                hax, _, haz, _ = (narrow(c) for c in rgba_a16)
                if mutant == 'swap_halves':
                    w1 = pack_pair(haz, hax)
                elif mutant == 'drop_shift':
                    w1 = (hax & 0xffff) | (haz & 0xffff)
                else:
                    w1 = pack_pair(hax, haz)
                # Load 3: v[6:7] = dmask:0x7 NSA [v2, v9]
                nx, ny = (x, 63) if mutant == 'drop_nsa' else (x, y)
                rgba_nsa = texel_f32_words(tex_mode, nx, ny)
                hnx, hny, hnz, _ = (narrow(c) for c in rgba_nsa)
                w2 = pack_pair(hnx, hny)
                w3 = hnz
                out.extend((w0, w1, w2, w3))
            else:
                out.extend((pack_pair(hx, hy), pack_pair(hz, hw), hy, pack_pair(hz, hw)))
    return out


def expect_refused(fn, substring, label):
    try:
        fn()
    except G.Unsupported as exc:
        assert substring in str(exc), f'{label}: expected {substring!r} in {str(exc)!r}'
    else:
        raise AssertionError(f'{label}: unexpectedly accepted')


def check_rounding_formula():
    """Cross-check the HLSL integer RNE formula in Python against struct.pack('<e', ...) on boundary words."""
    def hlsl_pack16_rne(bits):
        bits &= 0xffffffff
        sign = (bits >> 16) & 0x8000
        abs_bits = bits & 0x7fffffff
        if abs_bits > 0x7f800000:
            return 0x7e00
        if abs_bits >= 0x477ff000:
            return sign | 0x7c00
        if abs_bits >= 0x38800000:
            shifted = abs_bits - 0x38000000
            return sign | ((shifted + 0x00000fff + ((shifted >> 13) & 1)) >> 13)
        if abs_bits < 0x33000000:
            return sign
        sig = 0x00800000 | (abs_bits & 0x007fffff)
        shift = 126 - (abs_bits >> 23)
        half_m = sig >> shift
        rem = sig & ((1 << shift) - 1)
        mid = 1 << (shift - 1)
        if rem > mid or (rem == mid and (half_m & 1) != 0):
            half_m += 1
        return sign | half_m

    anchors = (
        0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x00800000,
        0x32ffffff, 0x33000000, 0x33000001, 0x337fffff, 0x33800000, 0x33800001,
        0x387fdfff, 0x387fe000, 0x387fe001, 0x387fffff, 0x38800000, 0x38801000,
        0x3f7fffff, 0x3f800000, 0x3f801000, 0x3f801001, 0x3f803000,
        0x477fdfff, 0x477fe000, 0x477fefff, 0x477ff000, 0x47800000,
        0x7f800000, 0x7f800001, 0x7fc00000, 0x7fffffff,
    )
    tested = 0
    for base in anchors:
        for delta in (-2, -1, 0, 1, 2):
            word = (base + delta) & 0xffffffff
            for signed_word in (word, word ^ 0x80000000):
                got = hlsl_pack16_rne(signed_word)
                want = pack16_rne_ref(signed_word)
                assert got == want, f'0x{signed_word:08x}: got 0x{got:04x}, want 0x{want:04x}'
                tested += 1
    for u16 in range(65536):
        w = f32_bits_from_unorm16(u16)
        assert hlsl_pack16_rne(w) == pack16_rne_ref(w), f'u16=0x{u16:04x}'
    rtz_depth_diffs = sum(
        pack16_rne_ref(f32_bits_from_unorm16(u)) != pack16_rtz_ref(f32_bits_from_unorm16(u))
        for u in DEPTH_TEXELS
    )
    assert rtz_depth_diffs == 7, f'expected 7 RNE-vs-RTZ depth discriminators, got {rtz_depth_diffs}'
    print(f'pack16_rne: {tested} boundary floats and all 65,536 R16_UNORM values match IEEE binary16 RNE')


def check_decoder_and_masks():
    # Verify PRE_WORDS instruction decoding.
    off = 0
    pre_decoded = []
    while off < len(PRE_WORDS):
        n, _, asm, _ = D.decode_instruction(PRE_WORDS, off)
        pre_decoded.append(asm)
        off += n
    assert pre_decoded == PRE.strip().splitlines(), pre_decoded

    for key in DECODE_KEYS:
        text, raw = MIMG_LLVM_MC[key]
        length, _, asm, _ = D.decode_instruction(raw, 0)
        assert length == len(raw) and asm == text, (key, length, asm, text)

    # Check all 15 non-zero dmasks for destination register width and pairwise component order.
    for dmask in range(1, 16):
        comps = [c for bit, c in enumerate('xyzw') if dmask & (1 << bit)]
        expected_regs = (len(comps) + 1) // 2
        raw = encode_mimg(IMG_LOAD, dmask=dmask, vaddr=8, vdata=4, d16=1)
        _, _, asm, _ = D.decode_instruction(raw, 0)
        expected_dst = 'v4' if expected_regs == 1 else 'v[4:5]'
        assert asm.startswith(f'image_load {expected_dst}, '), (dmask, asm)
        pipe = G.Pipeline(VS_WORDS, PRE_WORDS + raw + POST_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
        ps = pipe.hlsl_ps()
        assert 'uint pack16_rne(float value)' in ps
        assert 'SamplerState' not in ps and 'smp ' not in pipe.manifest()
        for i in range(0, len(comps), 2):
            reg = 4 + i // 2
            if i + 1 == len(comps):
                want = f'v{reg} = pack16_rne(t.{comps[i]});'
            else:
                want = f'v{reg} = pack16_rne(t.{comps[i]}) | (pack16_rne(t.{comps[i + 1]}) << 16);'
            assert want in ps, (dmask, want, ps)

    # Verify 3D and 2D-array level-zero D16 loads, plus vertex-stage emission.
    for key, hlsl_type, srv_kind in (('d16_3d', 'Texture3D<float4>', 'tex3d'),
                                     ('d16_2da', 'Texture2DArray<float4>', 'tex2da')):
        ps_3 = PRE_WORDS + MIMG_LLVM_MC[key][1] + POST_WORDS
        pipe = G.Pipeline(VS_WORDS, ps_3, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV)
        hlsl = pipe.hlsl_ps()
        assert hlsl_type in hlsl and 'r0.Load(int4(v0, v1, v2, 0))' in hlsl, hlsl
        assert f'srv 0 {srv_kind} table 0 0' in pipe.manifest()

    # Vertex stage using image_load ... d16 emits pack16_rne and Load(int3(..., 0)), while a pipeline
    # without d16 omits pack16_rne entirely.
    vs_d16 = (
        bytes.fromhex('04040cf4000000fa' '8002027e' '7fc08cbf')
        + encode_mimg(IMG_LOAD, dmask=0x3, vaddr=0, vdata=4, d16=1)
        + bytes.fromhex('703f8cbf' 'cf0800f804040404' '000081bf')
    )
    ps_plain = bytes.fromhex('8002007e' '0f1800f800000000' '000081bf')
    vs_pipe = G.Pipeline(vs_d16, ps_plain, root_blob())
    assert 'uint pack16_rne(float value)' in vs_pipe.hlsl_vs()
    assert 'v4 = pack16_rne(t.x) | (pack16_rne(t.y) << 16);' in vs_pipe.hlsl_vs()
    plain_pipe = G.Pipeline(VS_WORDS, ps_plain, root_blob())
    assert 'pack16_rne' not in plain_pipe.hlsl_vs() and 'pack16_rne' not in plain_pipe.hlsl_ps()
    print('decoder & masks: all 15 non-zero dmasks, 2D/a16/NSA/3D/2D_ARRAY and VS/PS paths verified')


def check_refusals():
    def build_ps(mimg_bytes):
        return G.Pipeline(VS_WORDS, PRE_WORDS + mimg_bytes + POST_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT)

    # Explicit-mip loads, 1D/cube dimensions, 3D/2D-array with a16, and unsupported MIMG flags stay refused.
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['mip_d16_refused'][1]), 'image_load_mip', 'image_load_mip d16')
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['1d_d16_refused'][1]), '1D', '1D image_load d16')
    expect_refused(lambda: build_ps(MIMG_LLVM_MC['cube_d16_refused'][1]), 'SQ_RSRC_IMG_CUBE', 'cube image_load d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, dim=DIM_3D, a16=1, d16=1)), 'image_load', '3D a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, dim=DIM_2D_ARRAY, a16=1, d16=1)), 'image_load', '2D_ARRAY a16+d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, dmask=0, d16=1)), 'dmask:0x0', 'dmask 0 d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, tfe=1, d16=1)), 'tfe', 'tfe d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, lwe=1, d16=1)), 'lwe', 'lwe d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, r128=1, d16=1)), 'r128', 'r128 d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, w0_reserved=0x40, d16=1)), 'reserved', 'w0 reserved d16')
    expect_refused(lambda: build_ps(encode_mimg(IMG_LOAD, a16=1, d16=1, nsa=1, extra_addrs=(1,))), 'invalid_nsa', 'a16+d16 fake NSA')

    # Sampler MIMG ops with d16 remain refused.
    sample_pre = PRE_WORDS + bytes.fromhex('800398be800399be80039abe80039bbe')
    for name, op in (('image_sample', IMG_SAMPLE), ('image_sample_lz', IMG_SAMPLE_LZ)):
        expect_refused(
            lambda o=op: G.Pipeline(VS_WORDS, sample_pre + encode_mimg(o, ssamp=24, d16=1) + POST_WORDS,
                                    root_blob(), ps_hw_inputs=PACKED_INPUT),
            f'{name} modifier d16',
            f'{name} d16',
        )

    # Direct Stage.image checks for wrong destination register count and wrong address register count.
    pipe = build_ps(MIMG_LLVM_MC['d16_x'][1])
    expect_refused(
        lambda: pipe.ps.image(['v[4:7]', 'v[8:9]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0xf', 'd16'], 'load'),
        'image_load d16 destination v[4:7] for dmask 0xf',
        '4-VGPR dst on dmask 0xf d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v[4:5]', 'v[8:9]', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x3', 'd16'], 'load'),
        'image_load d16 destination v[4:5] for dmask 0x3',
        '2-VGPR dst on dmask 0x3 d16',
    )
    expect_refused(
        lambda: pipe.ps.image(['v4', 'v8', 's[16:23]'], ['dim:SQ_RSRC_IMG_2D', 'dmask:0x1', 'd16'], 'load'),
        'image_load address or modifiers',
        '1-VGPR addr without a16',
    )
    print('refusals: image_load_mip, sample/sample_lz, 1D/cube, 3D/array a16, tfe/lwe/r128/reserved and bad tuple widths refused')


def controls():
    check_rounding_formula()
    check_decoder_and_masks()
    check_refusals()


def check_binary(llvm_mc):
    assert assemble(llvm_mc, PS_SOURCE_A) == PS_WORDS_A, 'PS_SOURCE_A encoding mismatch'
    assert assemble(llvm_mc, PS_SOURCE_B) == PS_WORDS_B, 'PS_SOURCE_B encoding mismatch'
    for key, (inst, expected) in sorted(MIMG_LLVM_MC.items()):
        built = assemble(llvm_mc, inst + '\n')
        if built != expected:
            raise AssertionError(f'{key}: assembled {built.hex()}, recorded {expected.hex()}')
    print(f'llvm-mc: {len(MIMG_LLVM_MC)} MIMG encodings and both pixel fixtures match recorded words')


def check_decode(library):
    from pixel_aot import disassemble
    blob = b''.join(MIMG_LLVM_MC[k][1] for k in DECODE_KEYS)
    llvm_lines = disassemble(blob, library)
    ours = [D.decode_instruction(MIMG_LLVM_MC[k][1], 0)[2] for k in DECODE_KEYS]
    assert llvm_lines == ours, f'LLVM vs disasm_rdna2 mismatch: {list(zip(llvm_lines, ours))}'
    print(f'libLLVM: {len(DECODE_KEYS)} MIMG instructions match disasm_rdna2.py verbatim')


PACK16_RNE_HLSL = """\
uint pack16_rne(float value) {
    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return 0x7e00u;
    if (abs_bits >= 0x477ff000u) return sign | 0x7c00u;
    if (abs_bits >= 0x38800000u) {
        uint shifted = abs_bits - 0x38000000u;
        return sign | ((shifted + 0x00000fffu + ((shifted >> 13) & 1u)) >> 13);
    }
    if (abs_bits < 0x33000000u) return sign;
    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);
    uint half_m = sig >> shift, rem = sig & ((1u << shift) - 1u), mid = 1u << (shift - 1u);
    if (rem > mid || (rem == mid && (half_m & 1u) != 0u)) half_m += 1u;
    return sign | half_m;
}"""

PACK16_RTZ_HLSL = """\
uint pack16_rne(float value) {
    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return 0x7e00u;
    if (abs_bits >= 0x47800000u) return sign | 0x7c00u;
    if (abs_bits >= 0x477fe000u) return sign | 0x7bffu;
    if (abs_bits >= 0x38800000u) return sign | ((abs_bits - 0x38000000u) >> 13);
    if (abs_bits < 0x33800000u) return sign;
    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);
    return sign | (sig >> shift);
}"""


def mutate_hlsl_a(source, mutant):
    if mutant == 'rtz':
        assert source.count(PACK16_RNE_HLSL) == 1, source
        mutated = source.replace(PACK16_RNE_HLSL, PACK16_RTZ_HLSL, 1)
    elif mutant == 'unpacked_x':
        target = 'v4 = pack16_rne(t.x);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'v4 = U(t.x);', 1)
    elif mutant == 'swap_halves':
        target = 'v5 = pack16_rne(t.x) | (pack16_rne(t.z) << 16);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'v5 = pack16_rne(t.z) | (pack16_rne(t.x) << 16);', 1)
    elif mutant == 'drop_shift':
        target = 'v5 = pack16_rne(t.x) | (pack16_rne(t.z) << 16);'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 'v5 = pack16_rne(t.x) | pack16_rne(t.z);', 1)
    elif mutant == 'drop_a16':
        target = 't = r0.Load(int3(v0 & 0xffffu, v0 >> 16, 0));'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 't = r0.Load(int3(v0, v1, 0));', 1)
    elif mutant == 'drop_nsa':
        target = 't = r0.Load(int3(v2, v9, 0));'
        assert source.count(target) == 1, source
        mutated = source.replace(target, 't = r0.Load(int3(v2, v3, 0));', 1)
    else:
        raise ValueError(mutant)
    assert mutated != source, mutant
    return mutated


def run_d16_draw(renderer, adapter, tex_mode, vs, ps, words, directory, tag):
    expected_path = directory / f'{tag}.bin'
    expected_path.write_bytes(struct.pack('<64I', *words))
    run = subprocess.run(
        [renderer, adapter, '--startup-image-load-d16', tex_mode, str(vs), str(ps), str(expected_path)],
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='', flush=True)
        sys.exit(77)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    hlsl_a = G.Pipeline(VS_WORDS, PS_WORDS_A, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    hlsl_b = G.Pipeline(VS_WORDS, PS_WORDS_B, root_blob(), ps_hw_inputs=PACKED_INPUT, rtv_formats=UINT_RTV).hlsl_ps()
    mutants = ('rtz', 'unpacked_x', 'swap_halves', 'drop_shift', 'drop_a16', 'drop_nsa')

    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='image-load-d16-', dir=out_dir) as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, f'vs_{kind}', vs_profile, directory)
            ps_a = compile_hlsl(compiler, kind, hlsl_a, f'ps_a_{kind}', ps_profile, directory)
            ps_b = compile_hlsl(compiler, kind, hlsl_b, f'ps_b_{kind}', ps_profile, directory)

            for tex_mode in ('r16_unorm', 'rgba32f'):
                ref_a = expected_words('A', tex_mode)
                ref_b = expected_words('B', tex_mode)
                run_a = run_d16_draw(renderer, adapter, tex_mode, vs, ps_a, ref_a, directory, f'ref_a_{tex_mode}_{kind}')
                assert run_a.returncode == 0, f'{kind} A {tex_mode} failed:\n{run_a.stdout}{run_a.stderr}'
                run_b = run_d16_draw(renderer, adapter, tex_mode, vs, ps_b, ref_b, directory, f'ref_b_{tex_mode}_{kind}')
                assert run_b.returncode == 0, f'{kind} B {tex_mode} failed:\n{run_b.stdout}{run_b.stderr}'
                print(f'{kind} {tex_mode}: fixtures A and B matched all 128 words (32 pixels)', flush=True)

            # Every mutation must fail against the reference and match its mutated model across every
            # applicable texture mode (drop_shift uses rgba32f because t.z is 0.0 on single-channel R16_UNORM).
            for mut in mutants:
                ps_mut = compile_hlsl(compiler, kind, mutate_hlsl_a(hlsl_a, mut), f'mut_{mut}_{kind}', ps_profile, directory)
                modes = ('rgba32f',) if mut == 'drop_shift' else ('r16_unorm', 'rgba32f')
                for tex_mode in modes:
                    ref_words = expected_words('A', tex_mode)
                    fail_run = run_d16_draw(renderer, adapter, tex_mode, vs, ps_mut, ref_words,
                                            directory, f'fail_{mut}_{tex_mode}_{kind}')
                    assert fail_run.returncode == 1, (
                        f'{kind} mutant {mut} ({tex_mode}) unexpectedly passed:\n{fail_run.stdout}{fail_run.stderr}'
                    )
                    mut_words = expected_words('A', tex_mode, mutant=mut)
                    match_run = run_d16_draw(renderer, adapter, tex_mode, vs, ps_mut, mut_words,
                                             directory, f'model_{mut}_{tex_mode}_{kind}')
                    assert match_run.returncode == 0, (
                        f'{kind} mutant {mut} ({tex_mode}) did not match model:\n{match_run.stdout}{match_run.stderr}'
                    )
                print(f'{kind} negative control ({mut}): rejected against reference and matched mutated model', flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary' and len(args) == 1:
        check_binary(args[0])
    elif mode == 'decode' and len(args) == 1:
        check_decode(args[0])
    elif mode == 'draw' and len(args) in (3, 4):
        controls()
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
