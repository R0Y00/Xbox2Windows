"""Scalar arithmetic over table descriptor words in a draw stage is computed at run time (X2W_DRAW_FEATURES=descword).

The title's vertex stages derive numbers from T# / V# words with SALU (an image's width from words 1 and 2 plus one,
shifted by the base level of word 3, clamped to 1 and to 0 for a null descriptor) and convert them to float. The
result was ('descnum',), refused at its numeric use ('numeric use of sN holding descnum', 240 sites in the 103 draw
requests that read table words). With descword each op is a lazy run-time node over the entry values `udsrc <n>
descword <param> <byte>` (Stage.tw_scalar, the draw form of mesh2hlsl's dwnum): only nodes a numeric read reaches are
emitted, so arithmetic nobody reads leaves the translation unchanged.

Synthetic code (root parameter 0 a table, its pointer in s[8:9]; s_load_dwordx4 s[12:15], s[8:9], 0x20):
  chain:   s_bfe_u32 s16, s13, 0x10000e ; s_add_u32 s16, s16, 1 ; s_cmp_eq_i32 s13, 0 ; s_cselect_b32 s16, 0, s16 ;
           v_cvt_f32_u32 v1, s16 -> one entry value `descword 0 36` (word 1), the three lines (no line for the load:
           an unedited word reads its entry value); FXC compiles;
  dead:    the same plus s_bfe_u32 s17, s14, 0x10000e that nothing reads -> no `descword 0 40`, no s17 line;
  inplace: s_bfe_u32 s13, s13, 0x10000e ; v_cvt_f32_u32 v1, s13 (the descriptor handling keeps the symbol) -> the
           read takes the run-time node (the extract from ud<n>);
  join:    s_cmp_eq_u32 s2, 0 ; s_cbranch_scc1 +1 ; s_mov_b32 s13, 5 ; v_cvt_f32_u32 v1, s13 (s13 the loaded word on
           one path, a number on the other: 'holding conflict' before) -> s13 = ud<n> at the load (the word's shadow),
           a run-time number at the join; without the read nothing is added;
  loop:    s16 = 0, s17 = 0; L: v_cvt_f32_u32 v1, s16 ; s_add_u32 s16, s16, s13 ; s_add_u32 s17, s17, 1 ;
           s_cmp_lt_u32 s17, 4 ; s_cbranch_scc1 L -> the add is kept (the head reads it through the back edge, where
           no read marks the node);
  off:     without descword the chain is refused as before (its s_cmp reads word 1: 'holding desc'), the join
           as 'holding conflict'.
"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from startup_image_3d_check import PS_WORDS, root_blob  # noqa: E402


def sop2(op, dst, src1, src0):
    return 0x80000000 | op << 23 | dst << 16 | src1 << 8 | src0


LOAD = [0xf4080304, 0xfa000020, 0xbf8cc07f]      # s_load_dwordx4 s[12:15], s[8:9], 0x20; s_waitcnt lgkmcnt(0)
CHAIN = [sop2(0x27, 16, 0xff, 13), 0x0010000e,   # s_bfe_u32 s16, s13, 0x10000e
         sop2(0x00, 16, 0x81, 16),               # s_add_u32 s16, s16, 1
         0xbf00800d,                             # s_cmp_eq_i32 s13, 0
         sop2(0x0a, 16, 16, 0x80)]               # s_cselect_b32 s16, 0, s16
DEAD = [sop2(0x27, 17, 0xff, 14), 0x0010000e]    # s_bfe_u32 s17, s14, 0x10000e
INPLACE = [sop2(0x27, 13, 0xff, 13), 0x0010000e]  # s_bfe_u32 s13, s13, 0x10000e
READ16, READ13 = [0x7e020c10], [0x7e020c0d]      # v_cvt_f32_u32_e32 v1, s16 / s13
LOOP = [0xbe900380, 0xbe910380,                 # s_mov_b32 s16, 0; s_mov_b32 s17, 0
        0x7e020c10, sop2(0x00, 16, 13, 16),       # L: v_cvt_f32_u32 v1, s16; s_add_u32 s16, s16, s13
        sop2(0x00, 17, 0x81, 17), 0xbf0a8411,     # s_add_u32 s17, s17, 1; s_cmp_lt_u32 s17, 4
        0xbf85fffb]                               # s_cbranch_scc1 L
JOIN = [0xbf068002, 0xbf850001, 0xbe8d0385]       # s_cmp_eq_u32 s2, 0; s_cbranch_scc1 +1; s_mov_b32 s13, 5
TAIL = [0x7e040d00, 0x7e0802f2, 0xf80008cf, 0x04040201, 0xbf810000]
FEATURES = ('heap', 'bufword3', 'uav', 'heapwords', 'descword')


def program(*parts):
    words = [w for p in parts for w in p]
    return struct.pack('<%dI' % len(words), *words)


def translate(code, features=FEATURES):
    p = G.Pipeline(code, PS_WORDS, root_blob(), features=features)
    return p.hlsl_vs(), p.manifest()


def entry(manifest, source):
    lines = [ln for ln in manifest.splitlines() if ln.startswith('udsrc ') and ln.endswith(source)]
    assert len(lines) == 1, manifest
    return int(lines[0].split()[1])


def main():
    asm = [x[3] for x in G.decode(program(LOAD, CHAIN, DEAD, INPLACE))]
    assert asm[2:] == ['s_bfe_u32 s16, s13, 0x10000e', 's_add_u32 s16, s16, 1', 's_cmp_eq_i32 s13, 0',
                       's_cselect_b32 s16, 0, s16', 's_bfe_u32 s17, s14, 0x10000e', 's_bfe_u32 s13, s13, 0x10000e'], asm
    code = program(LOAD, CHAIN, READ16, TAIL)
    hlsl, manifest = translate(code)
    n = entry(manifest, ' descword 0 36')
    for line in ('s16 = (ud%d >> (0x0010000eu & 31)) & ((1u << ((0x0010000eu >> 16) & 127)) - 1);' % n,
                 'scc = int(ud%d) == int(0x00000000u);' % n,
                 's16 = s16 + 0x00000001u;', 's16 = scc ? 0x00000000u : s16;', 'v1 = U(float(s16));'):
        assert line in hlsl, (line, hlsl)
    assert 'X2W_' not in hlsl and 'descword 0 40' not in manifest and 's13 = ' not in hlsl, (hlsl, manifest)
    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get('X2W_DRAW_FEATURES')
        os.environ['X2W_DRAW_FEATURES'] = ','.join(FEATURES)
        try:
            result = G.translate_to(d, 1, code, PS_WORDS, root_blob())
        finally:
            if old is None:
                os.environ.pop('X2W_DRAW_FEATURES')
            else:
                os.environ['X2W_DRAW_FEATURES'] = old
        assert result == 'ok', result
        assert os.path.getsize(os.path.join(d, '%016x.vs.cso' % 1)) > 0
    dead_hlsl, dead_manifest = translate(program(LOAD, CHAIN, DEAD, READ16, TAIL))
    assert dead_manifest == manifest and dead_hlsl == hlsl, (dead_manifest, dead_hlsl)
    in_hlsl, in_manifest = translate(program(LOAD, INPLACE, READ13, TAIL))
    m = entry(in_manifest, ' descword 0 36')
    assert in_hlsl.index('s13 = (ud%d >> (0x0010000eu & 31))' % m) < in_hlsl.index('v1 = U(float(s13));'), in_hlsl
    join_hlsl, join_manifest = translate(program(LOAD, JOIN, READ13, TAIL))
    j = entry(join_manifest, ' descword 0 36')
    assert (join_hlsl.index('s13 = ud%d;' % j) < join_hlsl.index('s13 = 0x00000005u;')
            < join_hlsl.index('v1 = U(float(s13));')), join_hlsl
    unread_hlsl, unread_manifest = translate(program(LOAD, JOIN, [0x7e020d00], TAIL))  # v1 from v0
    assert 'descword' not in unread_manifest and 's13 = ud' not in unread_hlsl, (unread_manifest, unread_hlsl)
    loop_hlsl, loop_manifest = translate(program(LOAD, LOOP, TAIL))
    q = entry(loop_manifest, ' descword 0 36')
    assert loop_hlsl.index('v1 = U(float(s16));') < loop_hlsl.index('s16 = s16 + ud%d;' % q), loop_hlsl
    for case, reason in ((code, 'numeric use of s13 holding desc'),
                         (program(LOAD, JOIN, READ13, TAIL), 'numeric use of s13 holding conflict')):
        try:
            translate(case, ('heap', 'bufword3', 'uav', 'heapwords'))
        except G.Unsupported as error:
            assert reason in str(error), (reason, str(error))
        else:
            raise AssertionError('accepted without descword: ' + reason)
    print('draw descnum: T# word arithmetic at run time from udsrc descword 0 36, FXC ok; an unread op adds nothing; '
          'an in-place field extract is read at run time; a join of the loaded word with a number is a run-time '
          'number; a loop-carried node is kept; without descword refused')


if __name__ == '__main__':
    main()
