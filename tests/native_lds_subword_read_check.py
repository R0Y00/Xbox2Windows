"""Check native DXIL LDS subword reads using source-built instructions."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_dpp_mirror_check import vop2
from native_image_atomic_check import FORMATS, TEXELS, image_atomic, render
from native_scalar_lds_check import ds, sop1, vmove


DST = 0x6543210f
SSA = r'%[-\w.]+'
CASES = (
    (0xa6, 6, 16, False, 0),
    (0xa7, 4, 16, False, 16),
    (0x3c, 6, 16, False, None),
    (0x3b, 6, 16, True, None),
    (0x3a, 7, 8, False, None),
    (0x39, 7, 8, True, None),
    (0xa6, 5, 16, False, 0),
)


def program(opcode, offset, *, write_half=False, gds=False, data0=0, data1=0):
    # The quartet index is x and each quartet uses a different LDS word.
    # Only lane zero of each quartet seeds, reads, and exports the result.
    code = [0xbe810380, 0xf40c0200, 0xfa000000, 0xbf8cc07f,
            vop2(0x16, 8, 130, 0), vmove(9, 128),
            vop2(0x1a, 10, 130, 8), vmove(12, 255), 0xb2345678,
            vmove(13, 255), 0x91a2b3c0, vmove(14, 255), DST,
            sop1(3, 126, 255), 0x11111111, *ds(0x0d, 10, 12, offset=4),
            0xbf8cc07f]
    if write_half:
        code += [*ds(0x1f, 10, 13, offset=6), 0xbf8cc07f]
    first, second = ds(opcode, 10, data0, data1, dst=14, offset=offset)
    return code + [first | (int(gds) << 17), second, 0xbf8cc07f,
                   *image_atomic('add'), 0xbbfd0000, 0xbf810000]


def compile_shader(cli, sdk, work, opcode, offset, **options):
    source = work / 'subword_read.cs'
    shader = source.with_suffix('.cso')
    shader.unlink(missing_ok=True)
    source.write_bytes(compute(words(*program(opcode, offset, **options)), table=True))
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                            capture_output=True, text=True, timeout=30)
    return shader, result


def check_shape(dump, offset, width, signed, destination_bit):
    definitions = dict(re.findall(rf'^\s*({SSA}) = (.+)$', dump, re.M))

    def operands(value, pattern):
        match = re.fullmatch(pattern, definitions.get(value, ''))
        assert match, (value, pattern, definitions.get(value), dump)
        return match.groups()

    def block(name):
        match = re.search(rf'^{name}:.*\n((?:[ \t]+[^\n]+\n)+)', dump, re.M)
        assert match, (name, dump)
        return match[1]

    loads = re.findall(rf'^\s*({SSA}) = load i32, i32 addrspace\(3\)\* ({SSA}), align 4$', dump, re.M)
    assert len(loads) == 1, dump
    word, ptr = loads[0]
    index, = operands(ptr, rf'getelementptr(?: inbounds)? (?:\[128 x i32\], )?'
                           rf'\[128 x i32\] addrspace\(3\)\* @lds, i32 0, i32 ({SSA})')
    address, = operands(index, rf'lshr i32 ({SSA}), 2')
    operands(address, rf'add i32 {SSA}, {offset}')

    branches = re.findall(rf'br i1 ({SSA}), label %ds(\d+), label %dsDone\2\b', dump)
    assert len(branches) == 2, dump  # Initialization and the read.
    enabled, label = branches[-1]
    lane_mask, = operands(enabled, rf'icmp ne i32 ({SSA}), 0')
    operands(lane_mask, rf'and i32 {SSA}, {SSA}')
    read_block = block(f'ds{label}')
    if width == 16:
        alignment = re.search(rf'br i1 ({SSA}), label %dsReadAligned{label}, label %dsDone{label}\b',
                              read_block)
        assert alignment and f'{word} = load ' not in read_block, dump
        odd, = operands(alignment[1], rf'icmp eq i32 ({SSA}), 0')
        operands(odd, rf'and i32 {re.escape(address)}, 1')
        read_block = block(f'dsReadAligned{label}')
    assert f'{word} = load ' in read_block and f'br label %dsDone{label}' in read_block, dump

    writes = re.findall(rf'^\s*store i32 ({SSA}), i32\* ({SSA}), align 4$', read_block, re.M)
    assert len(writes) == 1, dump
    value, dst_ptr = writes[0]
    initializers = re.findall(rf'^\s*store i32 ({SSA}), i32\* {re.escape(dst_ptr)}, align 4$', dump, re.M)
    assert len(initializers) == 2 and initializers[-1] == value, dump
    initial_old, = operands(initializers[0], rf'select i1 {SSA}, i32 {DST}, i32 ({SSA})')
    operands(initial_old, rf'load i32, i32\* {re.escape(dst_ptr)}, align 4')
    if destination_bit is not None:
        preserved, value = operands(value, rf'or i32 ({SSA}), ({SSA})')
        preserve_mask = '65535' if destination_bit else r'(?:-65536|4294901760)'
        old, = operands(preserved, rf'and i32 ({SSA}), {preserve_mask}')
        operands(old, rf'load i32, i32\* {re.escape(dst_ptr)}, align 4')
        if destination_bit:
            value, = operands(value, rf'shl i32 ({SSA}), 16')
    if signed:
        left, = operands(value, rf'ashr i32 ({SSA}), {32 - width}')
        field, = operands(left, rf'shl i32 ({SSA}), {32 - width}')
    else:
        field, = operands(value, rf'and i32 ({SSA}), {(1 << width) - 1}')
    shift, = operands(field, rf'lshr i32 {re.escape(word)}, ({SSA})')
    low, = operands(shift, rf'shl i32 ({SSA}), 3')
    operands(low, rf'and i32 {re.escape(address)}, 3')


def controls(cli, sdk, work):
    for opcode, offset, width, signed, destination_bit in CASES:
        shader, run = compile_shader(cli, sdk, work, opcode, offset)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader], capture_output=True,
                              text=True, timeout=30, check=True).stdout
        check_shape(dump, offset, width, signed, destination_bit)
    for opcode, options, diagnostic in (
        (0xa2, {}, 'unsupported local DS opcode 162'),
        (0xa6, {'gds': True}, 'expected a local DS instruction with GDS clear'),
        (0xa6, {'data0': 1}, 'invalid local DS register tuple'),
        (0xa7, {'data1': 1}, 'invalid local DS register tuple'),
    ):
        _, run = compile_shader(cli, sdk, work, opcode, 6, **options)
        assert run.returncode and run.stderr.strip() in (diagnostic, f'job=1 input=subword_read.cs {diagnostic}'), (
            opcode, options, run.stdout, run.stderr)
    print('LDS subword read DXIL and refusal diagnostics matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    initial = [0] * TEXELS
    for opcode, offset, result in (
        (0xa6, 6, 0x6543b3c0),
        (0xa7, 6, 0xb3c0210f),
        (0xa6, 5, DST),
    ):
        shader, run = compile_shader(cli, sdk, work, opcode, offset, write_half=True)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        expected = [result] * 8 + [0] * (TEXELS - 8)
        payload = work / 'texture.bin'
        payload.write_bytes(words(1 | FORMATS['add'], 0, *initial, *expected))
        run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
        print(f'LDS opcode {opcode:#x} offset {offset}: {TEXELS} texture words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as directory:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(directory), *args)
