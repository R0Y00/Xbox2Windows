"""Check direct compute image_gather4_lz_o with a packed per-lane XY offset."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import compute_gather4_l_check as texture
from native_direct_container_check import compute, words
from native_gather_lod_check import float_bits, render, render_compute

REFUSAL = 'direct image opcode or descriptor modifiers'


def program(*, nsa=True, opcode=0x57, dim=1, mask=4, flags=0, address_flags=0):
    offset, u, v = (2, 0, 1) if nsa else (0, 1, 2)
    code = [
        0xBE810380, 0xF40C0200, 0xFA000000,  # texture descriptor s[8:15]
        0xF4080800, 0xFA000020,  # output descriptor s[32:35]
        0x34060084,  # v_lshlrev_b32 v3, 4, v0 (output byte offset)
        # v_mul_u32_u24 and v_add_nc_u32 form 0x3f3e + 257 * lane in the offset VGPR.
        11 << 25 | offset << 17 | 0xff, 257,
        37 << 25 | offset << 17 | offset << 9 | 0xff, 0x3f3e,
    ]
    for reg, value in enumerate(texture.SAMPLER, start=4):
        code += [0xBE8003FF | reg << 16, value]
    for reg, value in ((u, 0.546875), (v, 0.6875)):
        code += [0x7E0002FF | reg << 17, float_bits(value)]
    code += [
        0xBF8CC07F,
        0xF0000000 | opcode << 18 | dim << 3 | mask << 8 | (2 if nsa else 0) | flags,
        0x00220400 | offset | address_flags,
    ]
    if nsa:
        code.append(u | v << 8)
    code += [0xBF8C3F70, 0xE0781000, 0x80080403, 0xBF810000]
    return compute(words(*code), threads=4, table=True), (offset, u, v)


def compile_shader(cli, sdk, work, name, **encoding):
    source = work / (name + '.cs')
    shader = source.with_suffix('.cso')
    blob, registers = program(**encoding)
    source.write_bytes(blob)
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, registers, run


def check_gather(dump, registers):
    offset, u, v = registers
    calls = re.findall(r'(%[\w.]+) = call %dx\.types\.ResRet\.f32 @dx\.op\.textureGather\.f32\(([^)]*)\)', dump)
    assert len(calls) == 1, dump
    result, signature = calls[0]
    args = signature.split(', ')
    assert len(args) == 10 and args[0] == 'i32 73', args
    assert all(re.fullmatch(r'%dx\.types\.Handle %[\w.]+', arg) for arg in args[1:3]), args
    assert args[5:7] == ['float undef', 'float undef'] and args[9] == 'i32 2', args
    assert '@dx.op.textureGatherCmp' not in dump and '@dx.op.sample' not in dump, dump

    loads = dict(re.findall(r'(%[\w.]+) = load i32, i32\* %r(\d+),', dump))
    casts = dict(re.findall(r'(%[\w.]+) = bitcast i32 (%[\w.]+) to float', dump))
    assert [loads[casts[arg.removeprefix('float ')]] for arg in args[3:5]] == [str(256 + u), str(256 + v)], args
    shifts = {name: (op, operand, bits) for name, op, operand, bits in re.findall(
        r'(%[\w.]+) = (shl|ashr) i32 (%[\w.]+), (\d+)', dump)}
    for arg, left_bits in zip(args[7:9], ('26', '18')):
        op, shifted, bits = shifts[arg.removeprefix('i32 ')]
        assert (op, bits) == ('ashr', '26'), args
        op, packed, bits = shifts[shifted]
        assert (op, loads[packed], bits) == ('shl', str(256 + offset), left_bits), args
    extracts = re.findall(r'extractvalue %dx\.types\.ResRet\.f32 ' + re.escape(result) + r', (\d+)', dump)
    assert extracts == ['0', '1', '2', '3'], dump


def controls(cli, sdk, work):
    for nsa in (True, False):
        shader, registers, run = compile_shader(cli, sdk, work, f'gather-{nsa}', nsa=nsa)
        assert run.returncode == 0, (run.stdout, run.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader],
                              capture_output=True, text=True, timeout=30, check=True).stdout
        check_gather(dump, registers)
    for label, encoding in (
        ('3d', {'dim': 2}),
        ('array', {'dim': 5}),
        ('compare-offset', {'opcode': 0x5f}),
        ('multi-channel', {'mask': 3}),
        ('unnormalized', {'flags': 0x1000}),
        ('a16', {'address_flags': 0x40000000}),
    ):
        _, _, run = compile_shader(cli, sdk, work, label, **encoding)
        assert run.returncode and REFUSAL in run.stderr, (label, run.stdout, run.stderr)
    print('2D LOD-zero gather DXIL and six refusal controls matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    for nsa in (True, False):
        shader, _, run = compile_shader(cli, sdk, work, f'gather-{nsa}', nsa=nsa)
        assert run.returncode == 0, (run.stdout, run.stderr)
        # View mip 0 is resource mip 1 (16x4). The unshifted footprint starts at (8, 2).
        expected = [texture.texel(1, 8 + lane - 2 + dx, min(3, 2 + lane - 1 + dy), 2)
                    for lane in range(4) for dx, dy in texture.ORDER]
        expected += [texture.SENTINEL] * (320 - len(expected))
        result = render_compute(renderer, adapter, shader, work / 'expected.bin', 1, expected)
        assert result.returncode == 0 and 'compared_words=320 mismatches=0' in result.stdout, (
            result.stdout, result.stderr)
        if 'debug_layer=enabled' in result.stdout:
            assert 'debug_errors=0 debug_warnings=0' in result.stdout, result.stdout
    print('Four lanes matched all four blue taps with both address encodings')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
