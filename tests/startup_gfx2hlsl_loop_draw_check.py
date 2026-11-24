"""Dispatcher versus structured loops on the same source-built pixel program.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare the recorded words
  draw <dxc> <offscreen_check> <warp|hardware>
                                       compile both forms with DXC and compare their readbacks

The three pixel fixtures count a scalar in a loop and export the result as a float. The zero-trip
fixture is a top-tested loop whose first condition is false; the multi-trip fixture is a bottom-tested
loop that runs three times; the inverted fixture keeps the same shape but continues while the counter
is at least three, so it exits after one pass. Each fixture is translated twice, once with the natural
loop structurer and once with the uint pc dispatcher, and both forms must read back the same value.
"""
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import VS_WORDS, assemble, compile_hlsl, draw, root_blob  # noqa: E402


ZERO_SOURCE = """\
s_mov_b32 s0, 0
s_mov_b32 s1, 5
s_cmp_lt_i32 s0, 0
s_cbranch_scc0 3  ; .Lexit
s_add_u32 s1, s1, 1
s_add_u32 s0, s0, 1
s_branch -5  ; .Lheader
v_cvt_f32_u32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
ZERO_WORDS = bytes.fromhex(
    '800380be' '850381be' '008004bf' '030084bf' '01810180' '00810080'
    'fbff82bf' '010c007e' '0f1800f800000000' '000081bf')

MULTI_SOURCE = """\
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_add_u32 s1, s1, 1
s_add_u32 s0, s0, 1
s_cmp_lt_i32 s0, 3
s_cbranch_scc1 -4  ; .Lloop
v_cvt_f32_u32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
MULTI_WORDS = bytes.fromhex(
    '800380be' '800381be' '01810180' '00810080' '008304bf' 'fcff85bf'
    '010c007e' '0f1800f800000000' '000081bf')

INVERTED_SOURCE = """\
s_mov_b32 s0, 0
s_mov_b32 s1, 0
s_add_u32 s1, s1, 1
s_add_u32 s0, s0, 1
s_cmp_ge_i32 s0, 3
s_cbranch_scc1 -4  ; .Lloop
v_cvt_f32_u32 v0, s1
exp mrt0 v0, v0, v0, v0 done vm
s_endpgm
"""
INVERTED_WORDS = bytes.fromhex(
    '800380be' '800381be' '01810180' '00810080' '008303bf' 'fcff85bf'
    '010c007e' '0f1800f800000000' '000081bf')

CASES = (
    ('zero-trip', ZERO_SOURCE, ZERO_WORDS, 5.0),
    ('multi-trip', MULTI_SOURCE, MULTI_WORDS, 3.0),
    ('inverted-exit', INVERTED_SOURCE, INVERTED_WORDS, 1.0),
)


def pipeline(words, structured):
    return G.Pipeline(VS_WORDS, words, root_blob(), structured_loops=structured)


def expected_words(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return ','.join('%08x' % bits for _ in range(4))


def check(label, ok):
    print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
    if not ok:
        raise AssertionError(label)


def controls():
    for name, _, words, _ in CASES:
        structured = '\n'.join(pipeline(words, True).ps.body())
        dispatcher = '\n'.join(pipeline(words, False).ps.body())
        check('%s: structured loop emitted' % name, 'for (;;) {' in structured and 'uint pc' not in structured)
        check('%s: dispatcher retained' % name, 'uint pc = 0;' in dispatcher and 'for (;;) {' not in dispatcher)
        check('%s: the accumulator is not folded to a constant' % name, 's1 = s1 + 0x00000001u;' in structured)
        check('%s: the two forms carry the same comparison' % name,
              ('int(s0) < int(0x00000000u)' in structured and 'int(s0) < int(0x00000000u)' in dispatcher) or
              ('int(s0) < int(0x00000003u)' in structured and 'int(s0) < int(0x00000003u)' in dispatcher) or
              ('int(s0) >= int(0x00000003u)' in structured and 'int(s0) >= int(0x00000003u)' in dispatcher))


def check_binary(llvm_mc):
    for name, source, words, _ in CASES:
        built = assemble(llvm_mc, source)
        if built != words:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), words.hex()))
    print('loop fixture words match llvm-mc')


def resolve_dxc(value):
    path = Path(value)
    if path.is_file():
        return str(path)
    return shutil.which(value)


def readback(run):
    match = re.search(r'startup_flat_param words=([0-9a-f,]+)', run.stdout)
    if not match:
        raise AssertionError('no startup_flat_param readback line\n%s\n%s' % (run.stdout, run.stderr))
    return match.group(1)


def check_draw(dxc, renderer, adapter):
    compiler = resolve_dxc(dxc)
    if not compiler:
        print('skip: DXC not available (%s)' % dxc, file=sys.stderr)
        return 77
    renderer_path = Path(renderer)
    if not renderer_path.is_file():
        print('skip: renderer not available (%s)' % renderer, file=sys.stderr)
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, _, words, value in CASES:
            expected = expected_words(value)
            observed = {}
            for structured in (True, False):
                pipe = pipeline(words, structured)
                label = 'structured' if structured else 'dispatcher'
                vs = compile_hlsl(compiler, 'dxc', pipe.hlsl_vs(), '%s_%s_vs' % (name, label), 'vs_6_0', directory)
                ps = compile_hlsl(compiler, 'dxc', pipe.hlsl_ps(), '%s_%s_ps' % (name, label), 'ps_6_0', directory)
                run = draw(str(renderer_path), adapter, vs, ps, expected)
                if run.returncode:
                    raise AssertionError('%s %s draw failed\n%s\n%s' % (name, label, run.stdout, run.stderr))
                observed[label] = readback(run)
            if observed['structured'] != expected or observed['dispatcher'] != expected:
                raise AssertionError('%s: expected %s, structured %s, dispatcher %s' %
                                     (name, expected, observed['structured'], observed['dispatcher']))
            print('%s: structured and dispatcher read %s' % (name, expected), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return 0
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        controls()
        return check_draw(*args)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main())
