"""buffer_load_format_* through a resource-heap descriptor in a draw stage: manifest `heapwords 1`.

Synthetic code (layout of startup_draw_heap_check): root flag 0x400 puts the resource-heap base in s8; the pixel
stage loads a 4-dword buffer descriptor with `s_load_dwordx4 s[16:19], s[8:9], s10` and then
`buffer_load_format_x v4, v0, s[16:19], 0 idxen`. The format and swizzle are in the descriptor's word 3, which
the title's heap slot does not hold as a number (it holds a view token). Contract (mesh_work/RUNTIME_NEEDS.md 6.3,
the same as mesh2hlsl's x2w_hword; runtime d3d12_vdesc.inl, bound for draws by d3d12_draw.inl): vdesc, 32 bytes per
view id, root SRV t2 space1, word i at id * 32 + 4 * i, 0 for an unknown slot or id >= 90000.

Checked: with X2W_DRAW_FEATURES heap,bufword3,heapwords the load is x2w_fmt_load(heapbuf[...], a,
x2w_hword(h0, 3u), 0u), the address is the index times the view's vinfo stride, vdesc and x2w_hword are declared
and the manifest has `heap 1` then `heapwords 1`. The helper is evaluated in Python on a token / id table.
Negative controls: without heapwords (heap,bufword3 only) the old refusal "through heapdyn" stays; a program with
no heap format load has no vdesc declaration and no `heapwords 1` line.

Words read as numbers (an image's base-level width and the null test `word1 == 0`, the waterfall dimension code):
`s_and_b32 s21, s18, 0x3fff; s_bfe_u32 s20, s17, 0x2001e; s_cmp_eq_i32 s17, 0; s_cselect_b32 s21, 0, s21` read
x2w_hword(h0, 2u) / (h0, 1u), with no heapmem slot kind in use (heap 1 and the helpers still declared); the emitted
expressions are evaluated on a vdesc word table. Refused: without heapwords, and a word of a descriptor the program
edited in place (s_and_b32 s19, s19, 0xcfffffff: no longer the synthesized word).
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import VS_WORDS  # noqa: E402
from startup_draw_heap_check import heap_root, words, S_MOV_S10_64, S_LOAD_X8, V_MOV_V0_1, V_MOV_V1_2, \
    IMAGE_LOAD, EXPORT  # noqa: E402

S_LOAD_X4 = [0xf4000000 | 2 << 18 | 16 << 6 | 8 >> 1, 10 << 25]       # s_load_dwordx4 s[16:19], s[8:9], s10
FORMAT_X = [0xe0000000 | 0 << 18 | 1 << 13, 0 | 4 << 8 | 4 << 16 | 128 << 24]  # buffer_load_format_x v4, v0, s[16:19], 0 idxen
MOVS = [0x7e000000 | n << 17 | 1 << 9 | 130 for n in (5, 6, 7)]       # v_mov_b32 v5..v7, 2


def sop2(op, dst, a, b):
    return 0x80000000 | op << 23 | dst << 16 | b << 8 | a


# s_and_b32 s21, s18, 0x3fff; s_bfe_u32 s20, s17, 0x2001e; s_cmp_eq_i32 s17, 0; s_cselect_b32 s21, 0, s21;
# v_mov_b32 v4, s21
NUMERIC = [sop2(14, 21, 18, 255), 0x3fff, sop2(39, 20, 17, 255), 0x2001e, 0xbf000000 | 128 << 8 | 17,
           sop2(10, 21, 128, 21), 0x7e000000 | 4 << 17 | 1 << 9 | 21]
EDITED = [sop2(14, 19, 19, 255), 0xcfffffff, 0xbf000000 | 128 << 8 | 19]  # s_and_b32 s19, s19, 0xcfffffff; s_cmp_eq_i32 s19, 0


def hword(heapmem, vdesc, h, i):
    """x2w_hword as emitted: token check at heapmem[h], id below 90000, else 0."""
    token, vid = heapmem.get(h, (0, 0))
    vid = vid if token == G.HEAP_TOKEN else 0xffffffff
    return vdesc.get((vid, i), 0) if vid < 90000 else 0


def main():
    ps = words(S_MOV_S10_64, S_LOAD_X4, V_MOV_V0_1, FORMAT_X, MOVS, EXPORT)
    text = [x[3] for x in G.decode(ps)]
    assert text[1] == 's_load_dwordx4 s[16:19], s[8:9], s10', text
    assert text[3] == 'buffer_load_format_x v4, v0, s[16:19], 0 idxen', text

    for features in (('heap', 'bufword3'), ('heap', 'bufword3', 'uav')):
        try:
            G.Pipeline(VS_WORDS, ps, heap_root(), features=features).hlsl_ps()
        except G.Unsupported as error:
            assert 'buffer_load_format_x through heapdyn' in str(error), str(error)
        else:
            raise AssertionError('heap format load accepted without heapwords (%s)' % (features,))

    pipe = G.Pipeline(VS_WORDS, ps, heap_root(), features=('heap', 'bufword3', 'heapwords'))
    hlsl, manifest = pipe.hlsl_ps(), pipe.manifest()
    for line in ('ByteAddressBuffer x2w_vdesc : register(t2, space1);',
                 'uint x2w_hword(uint h, uint i) { uint id = x2w_vid(h); return id < 90000u ? '
                 'x2w_vdesc.Load(id * 32u + 4u * i) : 0u; }',
                 'ByteAddressBuffer heapbuf[720008] : register(t0, space6);',
                 'a = v0 * x2w_vstride(h0) + 0;',
                 'v4 = x2w_fmt_load(heapbuf[NonUniformResourceIndex(x2w_hix(h0, 4u))], a, x2w_hword(h0, 3u), 0u);'):
        assert line in hlsl, (line, hlsl)
    assert 'uint x2w_fmt_load(' in hlsl, 'the format decoder is emitted'
    lines = manifest.split('\n')
    assert 'heap 1' in lines and 'heapwords 1' in lines and lines.index('heapwords 1') == lines.index('heap 1') + 1, manifest
    assert not any(l.startswith('udsrc') and 'bufword3' in l for l in lines), manifest

    # the helper, evaluated: slot 64 holds view 7, whose word 3 is 0x00026fac (an arbitrary value)
    heapmem = {64: (G.HEAP_TOKEN, 7), 96: (0x12345678, 7), 128: (G.HEAP_TOKEN, 90000)}
    vdesc = {(7, 3): 0x00026fac, (7, 1): 0x00100000}
    assert hword(heapmem, vdesc, 64, 3) == 0x00026fac and hword(heapmem, vdesc, 64, 1) == 0x00100000
    assert hword(heapmem, vdesc, 96, 3) == 0, 'a slot without the view token reads 0'
    assert hword(heapmem, vdesc, 128, 3) == 0, 'an id at the limit reads 0'
    assert hword(heapmem, vdesc, 32, 3) == 0, 'an empty slot reads 0'

    # Negative control: a heap program without a format load has neither vdesc nor `heapwords 1`.
    plain = G.Pipeline(VS_WORDS, words(S_MOV_S10_64, S_LOAD_X8, V_MOV_V0_1, V_MOV_V1_2, IMAGE_LOAD, EXPORT),
                       heap_root(), features=('heap', 'bufword3', 'heapwords'))
    assert 'x2w_vdesc' not in plain.hlsl_ps() and 'heapwords 1' not in plain.manifest().split('\n'), plain.manifest()
    numeric = numeric_words(heapmem)
    print('heapwords: heap typed load reads word 3 from vdesc (t2 space1); manifest heap 1 + heapwords 1; '
          'refused without heapwords; helper evaluated on 4 slots; image-only heap program unchanged; '
          'numeric word reads evaluated on %d views, refused without heapwords and after an in-place edit' % numeric)


def numeric_words(heapmem):
    ps = words(S_MOV_S10_64, S_LOAD_X4, NUMERIC, MOVS, EXPORT)
    text = [x[3] for x in G.decode(ps)]
    assert text[2:7] == ['s_and_b32 s21, s18, 0x3fff', 's_bfe_u32 s20, s17, 0x2001e', 's_cmp_eq_i32 s17, 0',
                         's_cselect_b32 s21, 0, s21', 'v_mov_b32_e32 v4, s21'], text
    pipe = G.Pipeline(VS_WORDS, ps, heap_root(), features=('heap', 'bufword3', 'heapwords'))
    hlsl, manifest = pipe.hlsl_ps(), pipe.manifest()
    body = [line.strip() for line in hlsl.splitlines()]
    expected = ['h0 = 0x00000040u;', 's21 = x2w_hword(h0, 2u) & 0x00003fffu;',
                's20 = (x2w_hword(h0, 1u) >> (0x0002001eu & 31)) & ((1u << ((0x0002001eu >> 16) & 127)) - 1);',
                'scc = int(x2w_hword(h0, 1u)) == int(0x00000000u);', 's21 = scc ? 0x00000000u : s21;', 'v4 = s21;']
    for line in expected:
        assert line in body, (line, hlsl)
    assert 'ByteAddressBuffer heapmem : register(t0, space1);' in body and 'ByteAddressBuffer x2w_vdesc : register(t2, space1);' in body, hlsl
    lines = manifest.splitlines()
    assert 'heap 1' in lines and 'heapwords 1' in lines, manifest
    # The emitted lines, evaluated: v4 = WIDTH_HI field of word 2, or 0 for a null view (word 1 == 0).
    views = 0
    for word1, word2 in ((0x00100000 | 2 << 30, 0x12345), (0, 0x12345), (1 << 8, 0x3fff), (0xc0000000, 0)):
        vdesc = {(7, 1): word1, (7, 2): word2}
        w1, w2 = hword(heapmem, vdesc, 64, 1), hword(heapmem, vdesc, 64, 2)
        s21 = w2 & 0x3fff
        s20 = (w1 >> (0x2001e & 31)) & ((1 << ((0x2001e >> 16) & 127)) - 1)
        assert s20 == word1 >> 30
        v4 = 0 if w1 == 0 else s21
        assert v4 == (0 if word1 == 0 else word2 & 0x3fff)
        views += 1
    for features, program, why in ((('heap', 'bufword3'), NUMERIC, 'numeric use of s17 holding desc'),
                                   (('heap', 'bufword3', 'heapwords'), EDITED, 'numeric use of s19 holding desc')):
        try:
            G.Pipeline(VS_WORDS, words(S_MOV_S10_64, S_LOAD_X4, program, MOVS, EXPORT), heap_root(),
                       features=features).hlsl_ps()
        except G.Unsupported as error:
            assert why in str(error), str(error)
        else:
            raise AssertionError('accepted: ' + why)
    return views


if __name__ == '__main__':
    main()
