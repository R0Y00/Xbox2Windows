"""Pixel-stage numeric use of an image descriptor word in tools/startup/gfx2hlsl.py.

The pixel stage loads an 8-dword image descriptor and an image instruction uses those registers. Exact GFX10
size fields lower to GetDimensions(0) on that texture. WIDTH minus one is split into word 1 bits 31:30 and
word 2 bits 11:0. HEIGHT minus one is word 2 bits 27:14. DEPTH minus one is word 4 bits 12:0 and only for a
3D image. s_bfe_u32, s_and_b32 and s_lshr_b32 qualify only with that field's control or mask.

Everything else stays refused: a whole word, any other word or field, a shift of word 2 that keeps the
resource-level bit, depth of a 1D, 2D or 2D-array image, height of a 1D image, a modified word, m0, a
4-dword buffer descriptor, a load with no image instruction, and the vertex stage. A size field is
the descriptor load the image instruction reads: an earlier load at table offset 32 into the same
registers does not borrow the sampled dimension, and a reload of those registers after the sample does
not either. A copy of the sampled word taken before the reload still does.

The pixel prologue is the source-built image_sample fixture. The scalar ops are encoded from the SOP1/SOP2
layouts and the decoder must print each one back. With image_field_expr removed, every accepted case is
refused.
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_image_sample_l_check import WORDS  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

END = bytes.fromhex('0f1800f804050607000081bf')  # exp mrt0 v4..v7; s_endpgm


def sop1(op, d, a):
    return struct.pack('<I', 0xbe800000 | d << 16 | op << 8 | a)


def sop2(op, d, a, b, lit=None):
    word = struct.pack('<I', 0x80000000 | op << 23 | d << 16 | b << 8 | a)
    return word if lit is None else word + struct.pack('<I', lit)


def vop1(op, d, a):
    return struct.pack('<I', 0x7e000000 | d << 17 | op << 9 | a)


def smem(op, sdata, sbase, offset=0):
    w0 = 0xf4000000 | (op << 18) | (sdata << 6) | (sbase // 2)
    w1 = (125 << 25) | (offset & 0x1fffff)
    return struct.pack('<II', w0, w1)


BFE, AND, LSHR, LSHL, OR, ADD, CSEL = 0x27, 0x0e, 0x20, 0x1e, 0x10, 0x00, 0x0a
LIT, ZERO, ONE, TWO = 255, 128, 129, 130
VMOV = vop1(0x01, 4, 10)

OPS = {
    's_bfe_u32 s10, s17, 0x2001e': sop2(BFE, 10, 17, LIT, 0x2001e),
    's_lshr_b32 s10, s17, 0x1e': sop2(LSHR, 10, 17, LIT, 30),
    's_and_b32 s10, s17, 0xc0000000': sop2(AND, 10, 17, LIT, 0xc0000000),
    's_and_b32 s10, s18, 0xfff': sop2(AND, 10, 18, LIT, 0xfff),
    's_and_b32 s11, s18, 0xfff': sop2(AND, 11, 18, LIT, 0xfff),
    's_bfe_u32 s10, s18, 0xc0000': sop2(BFE, 10, 18, LIT, 0xc0000),
    's_bfe_u32 s10, s18, 0xe000e': sop2(BFE, 10, 18, LIT, 0xe000e),
    's_and_b32 s10, s18, 0xfffc000': sop2(AND, 10, 18, LIT, 0x0fffc000),
    's_and_b32 s10, s20, 0x1fff': sop2(AND, 10, 20, LIT, 0x1fff),
    's_bfe_u32 s10, s20, 0xd0000': sop2(BFE, 10, 20, LIT, 0xd0000),
    's_lshr_b32 s10, s18, 0xe': sop2(LSHR, 10, 18, LIT, 14),
    's_bfe_u32 s10, s18, 0xe000f': sop2(BFE, 10, 18, LIT, 0xe000f),
    's_bfe_u32 s10, s19, 0x4000c': sop2(BFE, 10, 19, LIT, 0x4000c),
    's_bfe_u32 s10, s21, 0x40004': sop2(BFE, 10, 21, LIT, 0x40004),
    's_bfe_u32 s10, s17, 0x90014': sop2(BFE, 10, 17, LIT, 0x90014),
    's_and_b32 s10, s17, 0x80000000': sop2(AND, 10, 17, LIT, 0x80000000),
    's_add_u32 s10, s18, 1': sop2(ADD, 10, 18, ONE),
    's_add_u32 s17, s17, 1': sop2(ADD, 17, 17, ONE),
    's_mov_b32 m0, s17': sop1(0x03, 124, 17),
    's_mov_b32 s28, s18': sop1(0x03, 28, 18),
    's_and_b32 s10, s28, 0xfff': sop2(AND, 10, 28, LIT, 0xfff),
    's_cselect_b32 s10, 0, s17': sop2(CSEL, 10, ZERO, 17),
    's_lshl_b32 s11, s11, 2': sop2(LSHL, 11, 11, TWO),
    's_or_b32 s12, s10, s11': sop2(OR, 12, 10, 11),
    's_add_u32 s12, s12, 1': sop2(ADD, 12, 12, ONE),
    's_load_dwordx8 s[16:23], s[8:9], 0x20': smem(0x03, 16, 8, 0x20),
    's_waitcnt lgkmcnt(0)': bytes.fromhex('7fc08cbf'),
    'v_mov_b32_e32 v4, s10': VMOV,
    'v_mov_b32_e32 v4, s12': vop1(0x01, 4, 12),
    'v_mov_b32_e32 v4, s17': vop1(0x01, 4, 17),
    'v_mov_b32_e32 v4, m0': vop1(0x01, 4, 124),
}
OPS.update({'v_mov_b32_e32 v4, s%d' % n: vop1(0x01, 4, n) for n in range(16, 24)})
for text, code in OPS.items():
    (_off, _n, _name, decoded), = G.decode(code + bytes.fromhex('000081bf'))[:1]
    assert decoded == text, (text, decoded)


def splice(code, ops, at):
    ins = G.decode(code)
    if at == 'image':
        off = next(off for off, _n, name, _asm in ins if name.startswith('image_'))
    elif at == 'after-image':
        off, n, _name, _asm = next(item for item in ins if item[2].startswith('image_'))
        off += n
    else:
        off = next(off for off, _n, name, _asm in ins if name == 's_endpgm')
    extra = b''.join(OPS[op] for op in ops)
    return code[:off] + extra + code[off:]


def use(base, *ops, at='image'):
    return splice(WORDS[base], ops + ('v_mov_b32_e32 v4, s10',), at)


W_LO = '((img_width_tex2d(r0) - 1u) & 3u)'
W_HI = '((img_width_tex2d(r0) - 1u) >> 2)'
W_AND = '(((img_width_tex2d(r0) - 1u) & 3u) << 30)'
H = '((img_height_tex2d(r0) - 1u) & 0x3fffu)'
H_AND = '(((img_height_tex2d(r0) - 1u) & 0x3fffu) << 14)'
WIDTH_1D = '((img_width_tex1d(r0) - 1u) & 3u)'
H_ARRAY = '((img_height_tex2da(r0) - 1u) & 0x3fffu)'
DEPTH = '((img_depth_tex3d(r0) - 1u) & 0x1fffu)'

HELPERS = {
    'tex2d-w': 'uint img_width_tex2d(Texture2D<float4> tex) { uint w, h, m; tex.GetDimensions(0u, w, h, m); return w; }',
    'tex2d-h': 'uint img_height_tex2d(Texture2D<float4> tex) { uint w, h, m; tex.GetDimensions(0u, w, h, m); return h; }',
    'tex1d': 'uint img_width_tex1d(Texture1D<float4> tex) { uint w, m; tex.GetDimensions(0u, w, m); return w; }',
    'tex2da': 'uint img_height_tex2da(Texture2DArray<float4> tex) { uint w, h, e, m; tex.GetDimensions(0u, w, h, e, m); return h; }',
    'tex3d': 'uint img_depth_tex3d(Texture3D<float4> tex) { uint w, h, d, m; tex.GetDimensions(0u, w, h, d, m); return d; }',
}

ACCEPTED = (
    ('width lo bfe', use('lz', 's_bfe_u32 s10, s17, 0x2001e'), W_LO, 'tex2d-w'),
    ('width lo lshr', use('lz', 's_lshr_b32 s10, s17, 0x1e'), W_LO, 'tex2d-w'),
    ('width lo and', use('lz', 's_and_b32 s10, s17, 0xc0000000'), W_AND, 'tex2d-w'),
    ('width hi and', use('lz', 's_and_b32 s10, s18, 0xfff'), W_HI, 'tex2d-w'),
    ('width hi bfe', use('lz', 's_bfe_u32 s10, s18, 0xc0000'), W_HI, 'tex2d-w'),
    ('height bfe', use('lz', 's_bfe_u32 s10, s18, 0xe000e'), H, 'tex2d-h'),
    ('height and', use('lz', 's_and_b32 s10, s18, 0xfffc000'), H_AND, 'tex2d-h'),
    ('width 1d', use('dim1d', 's_bfe_u32 s10, s17, 0x2001e'), WIDTH_1D, 'tex1d'),
    ('height array', use('array', 's_bfe_u32 s10, s18, 0xe000e'), H_ARRAY, 'tex2da'),
    ('depth and', use('three', 's_and_b32 s10, s20, 0x1fff'), DEPTH, 'tex3d'),
    ('depth bfe', use('three', 's_bfe_u32 s10, s20, 0xd0000'), DEPTH, 'tex3d'),
    ('copied word', use('lz', 's_mov_b32 s28, s18', 's_and_b32 s10, s28, 0xfff'), W_HI, 'tex2d-w'),
)


def ps(code, vs=None):
    pipe = G.Pipeline(vs or VS_WORDS, code, root_blob(flags=0))
    return pipe.hlsl_ps(), pipe.manifest()


def refused(code, text, vs=None):
    try:
        ps(code, vs)
    except G.Unsupported as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('accepted, expected %r' % text)


def without_field(code, vs=None):
    saved = G.Stage.image_field_expr
    G.Stage.image_field_expr = lambda self, base, srcs, syms: None
    try:
        try:
            ps(code, vs)
        except G.Unsupported:
            return
        raise AssertionError('translated with image_field_expr removed')
    finally:
        G.Stage.image_field_expr = saved


def controls():
    for label, code, expr, helper in ACCEPTED:
        text, manifest = ps(code)
        assert expr in text, (label, text)
        assert HELPERS[helper] in text, (label, text)
        assert 'srv 0 ' in manifest, (label, manifest)
        without_field(code)
    composed = splice(WORDS['lz'], (
        's_bfe_u32 s10, s17, 0x2001e',
        's_and_b32 s11, s18, 0xfff',
        's_lshl_b32 s11, s11, 2',
        's_or_b32 s12, s10, s11',
        's_add_u32 s12, s12, 1',
        'v_mov_b32_e32 v4, s12',
    ), 'image')
    # The and writes s11, so the recorded v_mov of s10 is not part of this chain.
    text, _manifest = ps(composed)
    assert W_LO in text and W_HI in text and 's12 = s12 + 0x00000001u;' in text, text
    without_field(composed)

    for word in range(8):
        refused(splice(WORDS['lz'], ('v_mov_b32_e32 v4, s%d' % (16 + word),), 'image'),
                'numeric use of s%d holding image descriptor word %d' % (16 + word, word))
    refused(use('lz', 's_lshr_b32 s10, s18, 0xe'), 's_lshr_b32 of image descriptor word 2')
    refused(use('lz', 's_bfe_u32 s10, s18, 0xe000f'), 's_bfe_u32 of image descriptor word 2')
    refused(use('lz', 's_bfe_u32 s10, s19, 0x4000c'), 's_bfe_u32 of image descriptor word 3')
    refused(use('lz', 's_bfe_u32 s10, s21, 0x40004'), 's_bfe_u32 of image descriptor word 5')
    refused(use('lz', 's_bfe_u32 s10, s17, 0x90014'), 's_bfe_u32 of image descriptor word 1')
    refused(use('lz', 's_and_b32 s10, s17, 0x80000000'), 's_and_b32 of image descriptor word 1')
    refused(use('lz', 's_add_u32 s10, s18, 1'), 's_add_u32 of image descriptor word 2')
    refused(use('lz', 's_and_b32 s10, s20, 0x1fff'), 'image descriptor depth of a 2D image')
    refused(use('dim1d', 's_bfe_u32 s10, s18, 0xe000e'), 'image descriptor height of a 1D image')
    refused(use('array', 's_and_b32 s10, s20, 0x1fff'), 'image descriptor depth of a 2D-array image')
    refused(splice(WORDS['lz'], ('s_mov_b32 m0, s17', 'v_mov_b32_e32 v4, m0'), 'image'),
            'numeric use of m0 holding image descriptor word 1')
    refused(splice(WORDS['lz'], ('s_add_u32 s17, s17, 1', 'v_mov_b32_e32 v4, s17'), 'after-image'),
            'numeric use of s17 holding a modified image descriptor word 1')
    refused(splice(WORDS['lz'], ('s_cselect_b32 s10, 0, s17',), 'image'),
            'numeric use of s17 holding image descriptor word 1')
    image3d = next(WORDS['three'][off:off + n]
                   for off, n, name, _asm in G.decode(WORDS['three']) if name.startswith('image_'))
    both = splice(WORDS['lz'], ('s_bfe_u32 s10, s17, 0x2001e', 'v_mov_b32_e32 v4, s10'), 'image')
    end_at = next(off for off, _n, name, _asm in G.decode(both) if name == 's_endpgm')
    both = both[:end_at] + image3d + both[end_at:]
    refused(both, 'image descriptor word used as two dimensions')

    bare_image = smem(0x03, 16, 8) + OPS['s_bfe_u32 s10, s17, 0x2001e'] + VMOV + END
    refused(bare_image, 'numeric use of s10 holding descnum')
    bare_buf = smem(0x02, 16, 8) + OPS['s_and_b32 s10, s18, 0xfff'] + VMOV + END
    refused(bare_buf, 'numeric use of s10 holding descnum')
    vertex = smem(0x03, 16, 8) + OPS['s_bfe_u32 s10, s17, 0x2001e'] + VMOV + VS_WORDS
    refused(WORDS['lz'], 'numeric use of s10 holding descnum', vs=vertex)

    # The offset-32 load is read before the fixture's own offset-0 load. The sample uses offset 0, so
    # the earlier word must not borrow that dimension. The v_mov is after the image instruction.
    early = (OPS['s_load_dwordx8 s[16:23], s[8:9], 0x20'] + OPS['s_waitcnt lgkmcnt(0)']
             + OPS['s_and_b32 s10, s18, 0xfff'] + WORDS['lz'])
    early = splice(early, ('v_mov_b32_e32 v4, s10',), 'after-image')
    refused(early, 'numeric use of s10 holding descnum')

    # s18 is copied before the sample. After it, s[16:23] is loaded again from offset 32.
    # The copy is still the sampled descriptor. The reloaded s18 is not.
    kept = splice(WORDS['lz'], ('s_mov_b32 s28, s18',), 'image')
    kept = splice(kept, (
        's_load_dwordx8 s[16:23], s[8:9], 0x20',
        's_waitcnt lgkmcnt(0)',
        's_and_b32 s10, s28, 0xfff',
        'v_mov_b32_e32 v4, s10',
    ), 'after-image')
    text, manifest = ps(kept)
    assert W_HI in text, text
    assert manifest.count('\nsrv ') == 1 and 'srv 0 tex2d table 0 0' in manifest, manifest
    assert 'table 0 32' not in manifest, manifest
    without_field(kept)
    reloaded = splice(WORDS['lz'], ('s_mov_b32 s28, s18',), 'image')
    reloaded = splice(reloaded, (
        's_load_dwordx8 s[16:23], s[8:9], 0x20',
        's_waitcnt lgkmcnt(0)',
        's_and_b32 s10, s18, 0xfff',
        'v_mov_b32_e32 v4, s10',
    ), 'after-image')
    refused(reloaded, 'numeric use of s10 holding descnum')
    print('pixel image descriptor size fields: width, height and 3D depth lowered; other words refused')


if __name__ == '__main__':
    controls()
