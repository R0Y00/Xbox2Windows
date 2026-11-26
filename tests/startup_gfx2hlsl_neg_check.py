"""Translate encoded binary32 VOP3 NEG fixtures and optionally dispatch them on D3D12."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_flat_param_check import root_blob
from startup_gfx2hlsl_bit_alu_check import G, POST_WORDS, PRE_WORDS, vs_words
from startup_gfx2hlsl_refusals_check import vop3

SIGN = 0x80000000
ONE = 0x3f800000
MIN_NORMAL = 0x00800000
# (encoded selector, optional literal dword)
IMMEDIATES = ((128, None), (129, None), (192, None), (255, 0x40a00000))
# (encoded opcode, arity, runtime src1, runtime src2, scalar reference)
OPS = {
    'v_fma_f32': (0x14b, 3, ONE, SIGN | ONE, lambda a, b, c: a * b + c),
    'v_mul_f32': (0x108, 2, ONE, 0, lambda a, b, c: a * b),
    'v_med3_f32': (0x157, 3, SIGN | ONE, 0xc0000000, lambda a, b, c: sorted((a, b, c))[1]),
}


def float_value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def float_word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def reference(instruction, b, c, evaluate):
    control, = struct.unpack_from('<I', instruction, 4)
    selector = control & 0x1ff
    word = struct.unpack_from('<I', instruction, 8)[0] if selector == 255 else selector - 128
    if control & (1 << 29):
        word ^= SIGN
    return float_word(evaluate(float_value(word), float_value(b), float_value(c)))


def fixture(op):
    opcode, arity, b, c, evaluate = OPS[op]
    instructions = [
        vop3(opcode, [selector, 260 + lane, 264 + lane if arity == 3 else 0],
             vdst=12 + lane, neg=1, literal=literal)
        for lane, (selector, literal) in enumerate(IMMEDIATES)
    ]
    inputs = [(0, b, c)] * 4
    hlsl = G.Pipeline(vs_words(inputs), PRE_WORDS + b''.join(instructions) + POST_WORDS, root_blob()).hlsl_ps()
    for lane, word in enumerate((0x80000000, 0x80000001, 0x80000040, 0xc0a00000)):
        statement = next(line for line in hlsl.splitlines() if line.strip().startswith(f'v{12 + lane} ='))
        assert f'F(0x{word:08x}u)' in statement, (op, statement)
    return hlsl, inputs, tuple(reference(code, b, c, evaluate) for code in instructions)


def dispatch(dxc, runner, adapter, hlsl, inputs, directory):
    # Keep the other sources opaque to the compiler and use strict floating-point
    # compilation so signed zero is observable.
    fill = [(i + 1) * 0x9e3779b1 & 0xffffffff | 1 for i in range(16)]
    words = [initial ^ value for initial, value in zip(fill, (row[arg] for arg in range(3) for row in inputs))]
    params = '\n'.join('    i.fparam%d = data.Load4(%d) ^ uint4(%s);' %
                       (p, p * 16, ', '.join(f'0x{word:08x}u' for word in words[p * 4:p * 4 + 4]))
                       for p in range(3))
    source = hlsl.replace('PSOut main(PSIn i)', 'PSOut translated(PSIn i)') + '''
RWByteAddressBuffer data : register(u0);
[numthreads(1, 1, 1)]
void main() {
    PSIn i = (PSIn)0;
''' + params + '''
    data.Store4(48, asuint(translated(i).mrt0));
}
'''
    src, cso, output = (directory / name for name in ('neg.hlsl', 'neg.cso', 'neg.bin'))
    src.write_text(source, encoding='ascii')
    subprocess.run([dxc, '-T', 'cs_6_0', '-Gis', '-E', 'main', '-Fo', str(cso), str(src)], check=True)
    run = subprocess.run([runner, adapter, '--raw-uav-dump', str(cso), '1', '64', str(output)],
                         capture_output=True, text=True)
    print(run.stdout, end='')
    if run.returncode == 77:
        sys.exit(77)
    if run.returncode or run.stderr:
        raise AssertionError(run.stderr or f'raw UAV dump exited with {run.returncode}')
    actual = struct.unpack('<16I', output.read_bytes())
    assert actual[:12] == tuple(fill[:12]), 'operand region changed'
    return actual[12:]


def check_dispatch(dxc, runner, adapter):
    runner = str(Path(runner).resolve())
    out = Path(__file__).resolve().parents[1] / 'out'
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        for op in OPS:
            hlsl, inputs, expected = fixture(op)
            actual = dispatch(dxc, runner, adapter, hlsl, inputs, Path(tmp))
            for lane, (got, want) in enumerate(zip(actual, expected)):
                # FMA and MED3 always produce normals here. MUL produces exact -0
                # for inline 0 and a normal for the literal. The two nonzero inline
                # MUL results can flush to signed zero under the adapter's FTZ mode.
                allowed = (want, want & SIGN) if 0 < (want & ~SIGN) < MIN_NORMAL else (want,)
                assert got in allowed, (op, lane, f'{got:08x}', [f'{word:08x}' for word in allowed])
            print(f'{op}: ' + ' '.join(f'{word:08x}' for word in actual))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        for op in OPS:
            fixture(op)
        print('VOP3 NEG fixtures contain the encoded sign-flipped words')
    elif len(sys.argv) == 5 and sys.argv[1] == 'dispatch':
        check_dispatch(*sys.argv[2:])
    else:
        sys.exit('usage: startup_gfx2hlsl_neg_check.py [dispatch <dxc> <offscreen_check> <warp|hardware>]')
