"""A word of a root-table descriptor that a draw stage reads as a number is the runtime's `udsrc <n> descword <p> <byte>`.

The title's vertex stages test V# word 1 against 0 and convert NUM_RECORDS (word 2) to float straight from the
s_load_dwordx4 of a descriptor table entry (MEASURED at 542ffac61: 'numeric use of sN holding desc' in 103 draw
requests, 98 of them `s_cmp_eq_i32 <word 1>, 0`). With descword in X2W_DRAW_FEATURES the word is an entry constant
whose value the draw runtime reads from the table bound at root parameter <p> (d3d12_draw.inl draw_descword_value:
the title's dword, or the synthesized V# / T# word of a view token), as mesh2hlsl's descword does for mesh stages.

Synthetic code (root parameter 0 a table, so its pointer is s[8:9]):
    s_load_dwordx4 s[12:15], s[8:9], 0x20 ; s_waitcnt lgkmcnt(0) ; [edit] ; [reload]
    v_cvt_f32_u32_e32 v1, s14 ; v_cvt_f32_u32_e32 v2, v0 ; v_mov_b32 v4, 1.0 ; exp pos0 v1, v2, v4, v4 done
Checks:
  on:      manifest `udsrc <n> descword 0 40` (entry +0x20, word 2), the HLSL reads ud<n> for v1, FXC compiles;
  off:     without the feature the read is refused as before ('numeric use of s14 holding desc');
  edited:  `s_and_b32 s14, s14, 0xffff` (an in-place edit, the descriptor symbol is kept) before the read: the read
           takes the edit's run-time value (the descnum rule's shadow); after `s_mov_b64 s[16:17], s[14:15]` of the
           edited word (no shadow for a pair copy): refused;
  clamp:   `s_cselect_b32 s14, 0, s14` reads the word as a number and writes s14 at run time; the read takes that;
  copied:  the unedited word copied by s_mov_b64 is still the table's word;
  reload:  the same edit followed by a fresh s_load_dwordx4 of the entry: the read is the table's word again.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import PS_WORDS, root_blob  # noqa: E402

LOAD = [0xf4080304, 0xfa000020, 0xbf8cc07f]      # s_load_dwordx4 s[12:15], s[8:9], 0x20; s_waitcnt lgkmcnt(0)
EDIT = [0x870eff0e, 0x0000ffff]                  # s_and_b32 s14, s14, 0xffff
SELECT = [0x850e0e80]                            # s_cselect_b32 s14, 0, s14 (the title's NUM_RECORDS clamp)
MOV64 = [0xbe90040e]                             # s_mov_b64 s[16:17], s[14:15]
READ16 = [0x7e020c10]                            # v_cvt_f32_u32_e32 v1, s16
TAIL = [0x7e020c0e, 0x7e040d00, 0x7e0802f2, 0xf80008cf, 0x04040201, 0xbf810000]


def program(*parts):
    words = [w for p in parts for w in p]
    return struct.pack('<%dI' % len(words), *words)


def pipeline(code, features):
    return G.Pipeline(code, PS_WORDS, root_blob(), features=features)


def refused(code, features, text):
    try:
        p = pipeline(code, features)
        p.hlsl_vs()
        p.manifest()
    except G.Unsupported as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('accepted: expected %r' % text)


def main():
    code = program(LOAD, TAIL)
    asm = [x[3] for x in G.decode(code)]
    assert asm[0] == 's_load_dwordx4 s[12:15], s[8:9], 0x20' and asm[2] == 'v_cvt_f32_u32_e32 v1, s14', asm
    assert [x[3] for x in G.decode(program(EDIT))] == ['s_and_b32 s14, s14, 0xffff'], G.decode(program(EDIT))
    features = ('heap', 'bufword3', 'descword')
    p = pipeline(code, features)
    hlsl = p.hlsl_vs()
    manifest = p.manifest()
    lines = manifest.splitlines() if isinstance(manifest, str) else list(manifest)
    src = [ln for ln in lines if ' descword ' in ln]
    assert len(src) == 1 and src[0].endswith(' descword 0 40'), lines
    n = int(src[0].split()[1])
    assert ('v1 = U(float(ud%d));' % n) in hlsl or ('float(ud%d)' % n) in hlsl, hlsl
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get('X2W_DRAW_FEATURES')
        os.environ['X2W_DRAW_FEATURES'] = ','.join(features)
        try:
            result = G.translate_to(d, 1, code, PS_WORDS, root_blob())
        finally:
            if old is None:
                os.environ.pop('X2W_DRAW_FEATURES')
            else:
                os.environ['X2W_DRAW_FEATURES'] = old
        assert result == 'ok', result
        text = open(os.path.join(d, '%016x.txt' % 1)).read()
        assert 'descword 0 40' in text, text
        assert os.path.getsize(os.path.join(d, '%016x.vs.cso' % 1)) > 0
    refused(code, ('heap', 'bufword3'), 'numeric use of s14 holding desc')
    edited = pipeline(program(LOAD, EDIT, TAIL), features)
    text = edited.hlsl_vs()
    n = int([ln for ln in edited.manifest().splitlines() if ' descword 0 40' in ln][0].split()[1])
    # the in-place edit's run-time value (Stage.tw_scalar shadow, the descnum rule) is what the read takes
    assert ('s14 = ud%d & 0x0000ffffu;' % n) in text and 'v1 = U(float(s14));' in text, text
    assert [x[3] for x in G.decode(program(SELECT, MOV64, READ16))] == [
        's_cselect_b32 s14, 0, s14', 's_mov_b64 s[16:17], s[14:15]', 'v_cvt_f32_u32_e32 v1, s16']
    clamp = pipeline(program(LOAD, SELECT, TAIL), features)
    text = clamp.hlsl_vs()
    n = int([ln for ln in clamp.manifest().splitlines() if ' descword 0 40' in ln][0].split()[1])
    # s_cselect reads the word as a number and writes s14 at run time: the later read takes that run-time value
    assert ('s14 = scc ? 0x00000000u : ud%d;' % n) in text and 'v1 = U(float(s14));' in text, text
    refused(program(LOAD, EDIT, MOV64, READ16, TAIL[1:]), features,
            'numeric use of an edited table descriptor word (s16: table 0 +32 word 2)')
    copied = pipeline(program(LOAD, MOV64, READ16, TAIL[1:]), features)
    copied.hlsl_vs()
    assert 'descword 0 40' in copied.manifest(), copied.manifest()
    again = pipeline(program(LOAD, EDIT, LOAD, TAIL), features)
    again.hlsl_vs()
    assert 'descword 0 40' in again.manifest(), again.manifest()
    print('draw descword: table word 2 -> udsrc descword 0 40, FXC ok; off: refused; clamp and in-place edit: run-time values; edited then copied by s_mov_b64: refused; '
          'copied unedited: table word; reloaded after the edit: the table word again')


if __name__ == '__main__':
    main()
