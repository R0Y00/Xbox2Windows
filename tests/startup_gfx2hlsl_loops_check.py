"""Structured natural loops in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare the recorded words
  compile <dxc>                        compile the generated vertex and pixel shaders with DXC

The simple fixture is a bottom-tested loop: s0 counts from 1 to 4 and the branch returns to the
header while s0 < 4. The nested fixture has an outer loop whose body contains a self-loop on s1.
The header-exit fixture is a top-tested loop with a two-block body, so its exit is taken from the
header. The fifth fixture jumps from the preheader into the loop body, so the header does not
dominate the body; that shape must keep the pc dispatcher.

The binary and compile modes are optional. The compile mode needs DXC only; it does not open a
GPU adapter. A missing DXC exits 77 with a named skip reason.
"""
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_WORDS, assemble, compile_hlsl, root_blob  # noqa: E402


SIMPLE_SOURCE = """\
s_mov_b32 s0, 0
.Lloop:
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 4
s_cbranch_scc1 -3  ; .Lloop
v_mov_b32_e32 v0, s0
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
SIMPLE_WORDS = bytes.fromhex(
    '800380be' '00810080' '008404bf' 'fdff85bf' '0002007e' '0f1800f8' '00000000' '000081bf')

NESTED_SOURCE = """\
s_mov_b32 s0, 0
.Louter:
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 3
s_cbranch_scc0 5  ; .Lexit
s_mov_b32 s1, 0
.Linner:
s_add_u32 s1, s1, 1
s_cmp_lt_i32 s1, 2
s_cbranch_scc1 -3  ; .Linner
s_branch -8  ; .Louter
.Lexit:
v_mov_b32_e32 v0, s0
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
NESTED_WORDS = bytes.fromhex(
    '800380be' '00810080' '008304bf' '050084bf' '800381be' '01810180' '018204bf'
    'fdff85bf' 'f8ff82bf' '0002007e' '0f1800f8' '00000000' '000081bf')

# The exit is taken from the loop header and the body has two blocks, separated by an unconditional
# forward jump to the next block. The preheader sets both counters, so this also pins the constant
# invalidation at the header and both body blocks.
HEADER_EXIT_MULTI_BODY_SOURCE = """\
s_mov_b32 s0, 0
s_mov_b32 s1, 0
.Lheader:
s_cmp_lt_i32 s0, 3
s_cbranch_scc0 4  ; .Lexit
s_add_u32 s0, s0, 1
s_branch 0  ; .Lbody2
.Lbody2:
s_add_u32 s1, s1, 1
s_branch -6  ; .Lheader
.Lexit:
v_mov_b32_e32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
HEADER_EXIT_MULTI_BODY_WORDS = bytes.fromhex(
    '800380be' '800381be' '008304bf' '040084bf' '00810080' '000082bf'
    '01810180' 'faff82bf' '0102007e' '0f1800f8' '00000000' '000081bf')

# The preheader jumps into .Lbody, and the header jumps to .Lexit on the other edge. The body's branch
# back to the header therefore is not a natural loop: the header does not dominate the body.
UNSUPPORTED_SOURCE = """\
s_mov_b32 s0, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 3  ; .Lbody
.Lheader:
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 4
s_cbranch_scc0 2  ; .Lexit
.Lbody:
s_add_u32 s1, s1, 1
s_branch -5  ; .Lheader
.Lexit:
v_mov_b32_e32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
UNSUPPORTED_WORDS = bytes.fromhex(
    '800380be' '008006bf' '030085bf' '00810080' '008404bf' '020084bf' '01810180'
    'fbff82bf' '0102007e' '0f1800f8' '00000000' '000081bf')

# The loop's exit target is .Lskip, while .Lfall is the block immediately after the loop. The
# structurer only accepts an exit to that adjacent block, so this shape keeps the dispatcher.
NONADJACENT_SOURCE = """\
s_mov_b32 s0, 0
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 3
s_cbranch_scc0 3  ; exit block
s_branch -4  ; loop
s_mov_b32 s1, 1
s_branch 1  ; after
s_mov_b32 s1, 2
v_mov_b32_e32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
NONADJACENT_WORDS = bytes.fromhex(
    '800380be' '00810080' '008304bf' '030084bf' 'fcff82bf' '810381be'
    '010082bf' '820381be' '0102007e' '0f1800f800000000' '000081bf')

# Two forward exits from the same loop. The structurer accepts exactly one, so the dispatcher stays.
TWO_EXITS_SOURCE = """\
s_mov_b32 s0, 0
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 3
s_cbranch_scc0 3  ; .Lexit1
s_cmp_eq_u32 s0, 2
s_cbranch_scc1 3  ; .Lexit2
s_branch -6  ; .Lloop
s_mov_b32 s1, 1
s_branch 1  ; .Lafter
s_mov_b32 s1, 2
v_mov_b32_e32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
TWO_EXITS_WORDS = bytes.fromhex(
    '800380be' '00810080' '008304bf' '030084bf' '008206bf' '030085bf'
    'faff82bf' '810381be' '010082bf' '820381be' '0102007e' '0f1800f800000000' '000081bf')


def pipeline(words):
    return G.Pipeline(VS_WORDS, words, root_blob())


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def controls():
    simple = pipeline(SIMPLE_WORDS)
    body = '\n'.join(simple.ps.body())
    check('simple: one structured loop', body.count('for (;;) {') == 1)
    check('simple: the pc dispatcher is gone', 'uint pc' not in body and 'if (pc ==' not in body)
    check('simple: the loop exits with break', 'if (!(scc)) break;' in body)
    check('simple: the counter accumulates in the loop', 's0 = s0 + 0x00000001u;' in body)
    check('simple: every block has the 65536 guard', body.count('guard >= 65536u') == len(simple.ps.blocks))
    check('simple: every block advances the guard', body.count('++guard;') == len(simple.ps.blocks))
    print('  body:\n' + '\n'.join('    ' + line for line in body.splitlines()))

    nested = pipeline(NESTED_WORDS)
    body = '\n'.join(nested.ps.body())
    check('nested: two structured loops', body.count('for (;;) {') == 2)
    check('nested: the inner loop is nested', '        for (;;) {' in body)
    check('nested: the pc dispatcher is gone', 'uint pc' not in body and 'if (pc ==' not in body)
    check('nested: both counters accumulate in their loops',
          's0 = s0 + 0x00000001u;' in body and 's1 = s1 + 0x00000001u;' in body)
    check('nested: every block has the 65536 guard', body.count('guard >= 65536u') == len(nested.ps.blocks))
    print('  body:\n' + '\n'.join('    ' + line for line in body.splitlines()))

    header_exit = pipeline(HEADER_EXIT_MULTI_BODY_WORDS)
    body = '\n'.join(header_exit.ps.body())
    check('header exit multi-block: one structured loop', body.count('for (;;) {') == 1)
    check('header exit multi-block: the pc dispatcher is gone', 'uint pc' not in body)
    check('header exit multi-block: the header exit becomes break', 'if (!scc) break;' in body)
    check('header exit multi-block: both body blocks accumulate',
          's0 = s0 + 0x00000001u;' in body and 's1 = s1 + 0x00000001u;' in body)
    check('header exit multi-block: five blocks', len(header_exit.ps.blocks) == 5)
    check('header exit multi-block: every block has the 65536 guard',
          body.count('guard >= 65536u') == len(header_exit.ps.blocks))
    print('  body:\n' + '\n'.join('    ' + line for line in body.splitlines()))

    unsupported = pipeline(UNSUPPORTED_WORDS)
    body = '\n'.join(unsupported.ps.body())
    check('conditional entry: the pc dispatcher remains',
          'uint pc = 0;' in body and '[loop] for (uint guard' in body and 'if (pc == 0)' in body)
    check('conditional entry: no structured loop', 'for (;;) {' not in body)
    check('conditional entry: the preheader enters the body block', 'pc = (scc) ? 2 : 1;' in body)
    print('  body:\n' + '\n'.join('    ' + line for line in body.splitlines()))

    nonadjacent = pipeline(NONADJACENT_WORDS)
    body = '\n'.join(nonadjacent.ps.body())
    check('non-adjacent exit: the pc dispatcher remains', 'uint pc = 0;' in body and 'for (;;) {' not in body)

    two_exits = pipeline(TWO_EXITS_WORDS)
    body = '\n'.join(two_exits.ps.body())
    check('two exits: the pc dispatcher remains', 'uint pc = 0;' in body and 'for (;;) {' not in body)


def check_binary(llvm_mc):
    for name, source, words in (('simple', SIMPLE_SOURCE, SIMPLE_WORDS),
                                ('nested', NESTED_SOURCE, NESTED_WORDS),
                                ('header exit multi-block', HEADER_EXIT_MULTI_BODY_SOURCE,
                                 HEADER_EXIT_MULTI_BODY_WORDS),
                                ('conditional entry', UNSUPPORTED_SOURCE, UNSUPPORTED_WORDS),
                                ('non-adjacent exit', NONADJACENT_SOURCE, NONADJACENT_WORDS),
                                ('two exits', TWO_EXITS_SOURCE, TWO_EXITS_WORDS)):
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('fixture words match llvm-mc')


def resolve_dxc(value):
    path = Path(value)
    if path.is_file():
        return str(path)
    return shutil.which(value)


def check_compile(dxc):
    compiler = resolve_dxc(dxc)
    if not compiler:
        print('skip: DXC not available (%s)' % dxc, file=sys.stderr)
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, words in (('simple', SIMPLE_WORDS), ('nested', NESTED_WORDS),
                            ('header_exit_multi_block', HEADER_EXIT_MULTI_BODY_WORDS),
                            ('conditional_entry', UNSUPPORTED_WORDS)):
            pipe = pipeline(words)
            compile_hlsl(compiler, 'dxc', pipe.hlsl_vs(), 'vs_%s' % name, 'vs_6_0', directory)
            compile_hlsl(compiler, 'dxc', pipe.hlsl_ps(), 'ps_%s' % name, 'ps_6_0', directory)
            print('%s: DXC compiled the generated vertex and pixel shaders' % name, flush=True)
    return 0


def main():
    if len(sys.argv) == 1:
        controls()
        return 0
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'compile':
        controls()
        return check_compile(*args)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main())
