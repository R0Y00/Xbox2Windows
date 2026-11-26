"""Resource-heap descriptors at a run-time offset in a VERTEX stage (manifest `heap 1`, X2W_DRAW_FEATURES=heap).

Synthetic code: root flag 0x400 puts the resource-heap base in s8. The vertex stage loads a 4-dword buffer
descriptor with `s_load_dwordx4 s[16:19], s[8:9], s10` (s10 = 64, a register offset) and reads one dword through it
with buffer_load_dword offen. startup_draw_heap_check covers the pixel stage; this is the vertex form f5802183
uses (seven such loads, waterfall-indexed).

Checks:
  lowered:  the VS HLSL declares heapmem / vinfo / heapbuf, assigns h0 = 64 and reads heapbuf[x2w_hix(h0, 4)].
  CPU:      the emitted x2w_vid / x2w_hix bodies are evaluated on a synthetic heap memory. A view token at byte 64
            names view id 7, so the buffer is index 8 + 8 * 7 + 4 = 68. A slot without the token (a raw title
            descriptor, see msgs 031623) resolves to the empty view, index 4.
  refused:  the same stage with the feature off, and a heap descriptor word read as a number.
"""
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from startup_draw_heap_check import S_MOV_S10_64, heap_root, words  # noqa: E402

S_LOAD_X4 = [0xf4000000 | 2 << 18 | 16 << 6 | 8 >> 1, 10 << 25]   # s_load_dwordx4 s[16:19], s[8:9], s10
BUFFER_LOAD = [0xe0301000, 0x80040400]                            # buffer_load_dword v4, v0, s[16:19], 0 offen
EXPORT_POS = [0xf80008cf, 0x04040404]                             # exp pos0 v4, v4, v4, v4 done
PS = words(*[0x7e000000 | v << 17 | 1 << 9 | 242 for v in (4, 5, 6, 7)], [0xf800180f, 0x07060504])


def helper(hlsl, name):
    m = re.search(r'uint %s\(uint h(?:, uint k)?\) \{ return (.*?); \}' % name, hlsl)
    assert m, (name, hlsl)
    return m.group(1)


def evaluate(hlsl, heapmem, h, k):
    """Evaluate the emitted x2w_vid / x2w_hix on a byte heap. Only their own forms are translated."""
    load = lambda a: struct.unpack_from('<I', heapmem, a)[0]  # noqa: E731
    vid_text = helper(hlsl, 'x2w_vid')
    m = re.fullmatch(r'heapmem\.Load\(h\) == 0x([0-9a-f]+)u \? heapmem\.Load\(h \+ 4\) : 0xffffffffu', vid_text)
    assert m, vid_text
    vid = load(h + 4) if load(h) == int(m.group(1), 16) else 0xffffffff
    full = re.search(r'uint x2w_hix\(uint h, uint k\) \{ uint i = x2w_vid\(h\); return i < (\d+)u \? (\d+)u \+ (\d+)u '
                     r'\* i \+ k : k; \}', hlsl)
    assert full, hlsl
    limit, first, step = map(int, full.groups())
    return first + step * vid + k if vid < limit else k


def main():
    vs = words(S_MOV_S10_64, S_LOAD_X4, BUFFER_LOAD, EXPORT_POS)
    text = [x[3] for x in G.decode(vs)]
    assert text[1:3] == ['s_load_dwordx4 s[16:19], s[8:9], s10', 'buffer_load_dword v4, v0, s[16:19], 0 offen'], text
    try:
        G.Pipeline(vs, PS, heap_root(), features=())
    except G.Unsupported as error:
        assert "s_load with a register offset through ('resheap',)" in str(error), str(error)
    else:
        raise AssertionError('vertex heap load accepted with the feature off')
    print('refused: the feature off')

    pipe = G.Pipeline(vs, PS, heap_root(), features=('heap',))
    hlsl, manifest = pipe.hlsl_vs(), pipe.manifest()
    for line in ('ByteAddressBuffer heapmem : register(t0, space1);', 'ByteAddressBuffer vinfo : register(t1, space1);',
                 'h0 = 0x00000040u;', 'v4 = heapbuf[NonUniformResourceIndex(x2w_hix(h0, 4u))].Load(a + 0);'):
        assert line in hlsl, (line, hlsl)
    assert re.search(r'ByteAddressBuffer heapbuf\[720008\] : register\(t0, space\d+\);', hlsl), hlsl
    assert 'heap 1' in manifest.split('\n'), manifest
    print('lowered: the vertex stage reads heapbuf[x2w_hix(h0, 4)] with h0 = 64; manifest heap 1')

    heapmem = bytearray(256)
    struct.pack_into('<II', heapmem, 64, G.HEAP_TOKEN, 7)          # a runtime view token: view id 7
    struct.pack_into('<II', heapmem, 96, 0x30016fac, 0x12345678)   # a raw title descriptor (no token)
    got = evaluate(hlsl, heapmem, 0x40, 4)
    assert got == 8 + 8 * 7 + 4 == 68, got
    assert evaluate(hlsl, heapmem, 96, 4) == 4, 'a raw slot is the empty view'
    print('CPU: token slot -> heapbuf index %d (view 7, kind 4); a raw slot -> the empty view 4' % got)

    numeric = words(S_MOV_S10_64, S_LOAD_X4, 0x7e000000 | 4 << 17 | 1 << 9 | 17, EXPORT_POS)  # v_mov_b32 v4, s17
    try:
        G.Pipeline(numeric, PS, heap_root(), features=('heap',)).hlsl_vs()
    except G.Unsupported as error:
        assert 'holding desc' in str(error), str(error)
    else:
        raise AssertionError('a vertex heap descriptor word was read as a number')
    print('refused: a heap descriptor word read as a number')


if __name__ == '__main__':
    main()
