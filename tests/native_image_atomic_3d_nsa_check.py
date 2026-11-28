"""Check direct typed image atomics with volume and non-sequential addresses."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_image_atomic_binops_check import OPERAND_DIAGNOSTIC
from native_image_atomic_check import OPERAND, TEXELS, UINT_MASK, WIDTH, image_atomic, program_words, render
from native_scalar_lds_check import sop1


CASES = {
    'add': (2, (16, 17, 18), ()),       # v[16:18]
    'umax': (1, (0, 9), (9,)),          # [v0, v9]
    'cmpswap': (2, (16, 19, 21), (19, 21)),
}
XYZ = (2, 3, 1)
DATA = {'add': 2, 'umax': 1, 'cmpswap': 14}
ADDEND = 17
COMPARE = 32


def stores_literal(dump, pointer, literal):
    """A VGPR write is stored directly or through the EXEC select of the emitter."""
    defs = dict(re.findall(r'^\s*(%[-\w.]+) = (.+?)\s*(?:;.*)?$', dump, re.M))
    for value in re.findall(rf'store i32 (\S+), i32\* {re.escape(pointer)},', dump):
        if value == str(literal) or re.fullmatch(rf'select i1 \S+, i32 {literal}, i32 \S+', defs.get(value, '')):
            return True
    return False


def compile_shader(cli, sdk, work, op, *, dim=None, nsa_words=None):
    dimension, registers, nsa = CASES[op]
    if dim is None:
        dim = dimension
    if nsa_words is None:
        nsa_words = int(bool(nsa))
    code = program_words('add')[:4]
    code[1] = 0xF40C0600  # s_load_dwordx8 s[24:31], s[0:1], 0
    values = dict(zip(registers, XYZ))
    # For UMax, v1 also poisons the y coordinate if lowered as a contiguous pair.
    values[DATA[op]] = ADDEND if op == 'add' else OPERAND
    if op == 'cmpswap':
        values[DATA[op] + 1] = COMPARE
    for register, value in values.items():
        code.extend((0x7e0002ff | register << 17, value))  # v_mov_b32 vN, literal
    code.extend((sop1(3, 126, 255), 0x55555555))  # Enable exactly 16 lanes.
    flags = 0x1000 | (nsa_words << 1) | (0x2000 if op == 'cmpswap' else 0)
    code.extend(image_atomic(op, coord=registers[0], data=DATA[op], resource=24, dim=dim, flags=flags))
    if nsa_words:
        extra = sum(register << (8 * j) for j, register in enumerate(nsa))
        code.extend([extra] + [0] * (nsa_words - 1))
    code.extend((0xBBFD0000, 0xBF810000))
    source = work / f'{op}.cs'
    shader = source.with_suffix('.cso')
    shader.unlink(missing_ok=True)
    source.write_bytes(compute(words(*code), table=True))
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


def controls(cli, sdk, work):
    for op, (dim, registers, _) in CASES.items():
        shader, run = compile_shader(cli, sdk, work, op)
        assert run.returncode == 0, (op, run.stdout, run.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader],
                              capture_output=True, text=True, timeout=30, check=True).stdout
        calls = re.findall(r'(%[-\w.]+) = call i32 @dx\.op\.(atomic\w+)\.i32\(([^\n)]+)\)', dump)
        assert len(calls) == 1, dump
        result, operation, operands = calls[0]
        opcode, handle, *args = operands.split(', ')
        if op == 'cmpswap':
            assert (operation, opcode) == ('atomicCompareExchange', 'i32 79'), dump
        else:
            assert (operation, opcode, args.pop(0)) == (
                'atomicBinOp', 'i32 78', 'i32 0' if op == 'add' else 'i32 7',
            ), dump
        props = 4100 if dim == 2 else 4098  # UAV flag plus DXIL Texture3D or Texture2D kind.
        assert re.search(
            rf'^\s*{re.escape(handle.split()[-1])} = call .*@dx\.op\.annotateHandle'
            rf'\(i32 216, .*%dx\.types\.ResourceProperties \{{ i32 {props}, i32 261 \}}\)',
            dump, re.M,
        ), dump
        assert len(args) == (5 if op == 'cmpswap' else 4), dump
        if dim == 1:
            assert args[2] == 'i32 undef', dump
        # Trace each coordinate through its VGPR load to the distinct literal
        # assigned by the fixture. Dumpbin may assign numeric SSA names.
        loads = dict(re.findall(r'(%[-\w.]+) = load i32, i32\* (%[-\w.]+)', dump))
        for arg, coordinate in zip(args[:len(registers)], XYZ):
            pointer = loads[arg.split()[-1]]
            assert stores_literal(dump, pointer, coordinate), dump
        data = loads[args[-1].split()[-1]]
        value = ADDEND if op == 'add' else OPERAND - (UINT_MASK + 1)  # signed LLVM literal
        assert stores_literal(dump, data, value), dump
        if op == 'cmpswap':
            compare = loads[args[-2].split()[-1]]
            assert stores_literal(dump, compare, COMPARE), dump
            assert f'store i32 {result}, i32* {data},' in dump, dump

    for op, options in (
        ('add', {'nsa_words': 2}),
        ('umax', {'nsa_words': 2}),
        ('add', {'dim': 3}),  # Cube
        ('add', {'dim': 5}),  # 2D array
    ):
        _, run = compile_shader(cli, sdk, work, op, **options)
        assert run.returncode and OPERAND_DIAGNOSTIC in run.stderr, (op, options, run.stdout, run.stderr)
    print('Typed 3D and NSA image atomic DXIL shapes and refusal diagnostics matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    print(wave.stdout, end='')
    volume_flag = (1 << 31) | (1 << 30)
    volume_texels = sum((WIDTH >> level) * (8 >> level) * (4 >> level) for level in range(3))
    for op, selector, count in (
        ('umax', 1 << 27, TEXELS),
        ('add', volume_flag | (1 << 25), volume_texels),
    ):
        shader, run = compile_shader(cli, sdk, work, op)
        assert run.returncode == 0, (op, run.stdout, run.stderr)
        initial = [0x10f00000 + index for index in range(count)]
        expected = initial.copy()
        x, y, z = XYZ
        target = ((z * 8 if op == 'add' else 0) + y) * WIDTH + x
        expected[target] = ((initial[target] + 16 * ADDEND) & UINT_MASK
                            if op == 'add' else OPERAND)
        payload = work / 'texture.bin'
        payload.write_bytes(words(1 | selector, 0, *initial, *expected))
        run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
        assert run.returncode == 0, (op, run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
        print(f'{op}: {count} texture words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
