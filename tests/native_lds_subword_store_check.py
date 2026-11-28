"""Check direct gfx1013 LDS byte and halfword stores using source-built words."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_dpp_mirror_check import vop2
from native_image_atomic_check import FORMATS, TEXELS, image_atomic, render
from native_scalar_lds_check import ds, sop1, vmove
from native_vopc_cmp64_check import vopc


SEED = 0x76543210
DATA = 0x91a2b3c0
CASES = ((0x1e, 4, 255), (0x1f, 4, 65535), (0x1f, 5, 65535))


def program(opcode, offset, *, gds=False, data1=0, dst=0, contend=False):
    # Four lanes share each word. Lane 0 of each quartet seeds and reads it.
    # Lanes 0 and 2 perform the subword store, at distinct byte addresses.
    code = [0xbe810380, 0xf40c0200, 0xfa000000, 0xbf8cc07f,
            vop2(0x16, 8, 130, 0), vmove(9, 128),
            vop2(0x1a, 10, 130, 8), vop2(0x1b, 11, 131 if opcode == 0x1e else 130, 0),
            vop2(0x1c, 11, 266, 11),
            vmove(12, 255), SEED, vmove(13, 255), DATA,
            vop2(0x1c, 13, 256, 13)]
    if contend:
        # Both enabled lanes write byte zero, using 0xf0 and 0x0f.
        code += [vmove(11, 266), vop2(0x1b, 13, 130, 0),
                 vop2(0x1a, 13, 129, 13), vmove(15, 255), 0xf0,
                 vop2(0x16, 13, 269, 15)]
    code += [sop1(3, 126, 255), 0x11111111, *ds(0x0d, 10, 12, offset=4),
            0xbf8cc07f, 0xbf8a0000,
            sop1(3, 126, 255), 0x55555555]
    first, second = ds(opcode, 11, 13, data1, dst, offset)
    code += [first | (int(gds) << 17), second, 0xbf8cc07f, 0xbf8a0000,
             sop1(3, 126, 255), 0x11111111, *ds(0x36, 10, dst=14, offset=4),
             0xbf8cc07f]
    if contend:
        # The texture runner compares exact words. Map one legal winner to the
        # other, leaving every invalid result (including the OR merge) intact.
        code += [vmove(15, 255), (SEED & ~255) | 0x0f,
                 vopc(0xc2, 255, 14), (SEED & ~255) | 0xf0,
                 vop2(0x01, 14, 270, 15)]
    code += [*image_atomic('add'), 0xbbfd0000, 0xbf810000]
    return code


def compile_shader(cli, sdk, work, opcode, offset, **encoding):
    source = work / 'subword.cs'
    shader = source.with_suffix('.cso')
    shader.unlink(missing_ok=True)
    source.write_bytes(compute(words(*program(opcode, offset, **encoding)), table=True))
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


SSA = r'%[-\w.]+'


def check_shape(dump, offset, mask):
    definitions = dict(re.findall(rf'^\s*({SSA}) = (.+)$', dump, re.M))
    operations = re.findall(rf'^\s*({SSA}) = cmpxchg i32 addrspace\(3\)\* ({SSA}), i32 ({SSA}), i32 ({SSA}) monotonic monotonic$', dump, re.M)
    assert len(operations) == 1 and 'atomicrmw' not in dump, dump
    exchanged, ptr, old, replacement = operations[0]

    def operands(value, pattern):
        match = re.fullmatch(pattern, definitions[value])
        assert match, (value, pattern, definitions[value])
        return match.groups()

    index, = operands(ptr, rf'getelementptr(?: inbounds)? (?:\[128 x i32\], )?\[128 x i32\] addrspace\(3\)\* @lds, i32 0, i32 ({SSA})')
    address, = operands(index, rf'lshr i32 ({SSA}), 2')
    operands(address, rf'add i32 {SSA}, {offset}')
    preserved, insert = operands(replacement, rf'or i32 ({SSA}), ({SSA})')
    clear, = operands(preserved, rf'and i32 {re.escape(old)}, ({SSA})')
    field, = operands(clear, rf'xor i32 ({SSA}), -1')
    shift, = operands(field, rf'shl i32 {mask}, ({SSA})')
    low, = operands(shift, rf'shl i32 ({SSA}), 3')
    operands(low, rf'and i32 {re.escape(address)}, 3')
    data, = operands(insert, rf'shl i32 ({SSA}), {re.escape(shift)}')
    operands(data, rf'and i32 {SSA}, {mask}')

    # Trace both loop inputs and verify the retry and predicate edges.
    initial, entry, observed, label = operands(
        old, rf'phi i32 \[ ({SSA}), %(ds(?:Aligned)?\d+) \], \[ ({SSA}), %dsRetry(\d+) \]')
    operands(initial, rf'load i32, i32 addrspace\(3\)\* {re.escape(ptr)}, align 4')
    operands(observed, rf'extractvalue \{{ i32, i1 \}} {re.escape(exchanged)}, 0')

    def block(name):
        match = re.search(rf'^{name}:.*\n((?:[ \t][^\n]+\n)+)', dump, re.M)
        assert match, dump
        return match[1]

    loop = block(f'dsRetry{label}')
    assert f'{exchanged} = cmpxchg ' in loop, dump
    retry = re.search(rf'br i1 ({SSA}), label %dsDone{label}, label %dsRetry{label}\b', loop)
    assert retry, dump
    operands(retry[1], rf'icmp eq i32 {re.escape(observed)}, {re.escape(old)}')
    assert f'br label %dsRetry{label}\n' in block(entry), dump
    exec_branch = re.search(rf'br i1 ({SSA}), label %ds{label}, label %dsDone{label}\b', dump)
    assert exec_branch, dump
    lane_mask, = operands(exec_branch[1], rf'icmp ne i32 ({SSA}), 0')
    operands(lane_mask, rf'and i32 {SSA}, {SSA}')
    if mask == 65535:
        assert entry == f'dsAligned{label}', dump
        alignment = re.search(rf'br i1 ({SSA}), label %dsAligned{label}, label %dsDone{label}\b',
                              block(f'ds{label}'))
        assert alignment, dump
        odd, = operands(alignment[1], rf'icmp eq i32 ({SSA}), 0')
        operands(odd, rf'and i32 {re.escape(address)}, 1')
    else:
        assert entry == f'ds{label}', dump


def controls(cli, sdk, work):
    for opcode, offset, mask in CASES:
        # The odd halfword fixture documents the alignment branch and skip.
        shader, run = compile_shader(cli, sdk, work, opcode, offset)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader], capture_output=True,
                              text=True, timeout=30, check=True).stdout
        check_shape(dump, offset, mask)
    for opcode, options, diagnostic in (
        (0xa1, {}, 'unsupported local DS opcode 161'),
        (0x1e, {'gds': True}, 'expected a local DS instruction with GDS clear'),
        (0x1f, {'data1': 1}, 'invalid local DS register tuple'),
        (0x1e, {'dst': 1}, 'invalid local DS register tuple'),
    ):
        _, run = compile_shader(cli, sdk, work, opcode, 4, **options)
        assert run.returncode and run.stderr.strip() in (diagnostic, f'job=1 input=subword.cs {diagnostic}'), (
            opcode, options, run.stdout, run.stderr)
    print('LDS subword store DXIL and refusal diagnostics matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    initial = [0] * TEXELS
    for opcode, offset, mask, contend in (
        *(case + (False,) for case in CASES), (0x1e, 4, 255, True),
    ):
        shader, run = compile_shader(cli, sdk, work, opcode, offset, contend=contend)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        expected = initial.copy()
        for x in range(8):
            if contend:
                expected[x] = (SEED & ~255) | 0x0f
                continue
            result = bytearray(words(SEED))
            for lane in (4 * x, 4 * x + 2):
                start = (lane & 2) + offset - 4
                if mask == 255:
                    result[start] = (DATA | lane) & 255
                elif start % 2 == 0:
                    result[start:start + 2] = words(DATA | lane)[:2]
            expected[x] = int.from_bytes(result, 'little')
        payload = work / 'texture.bin'
        payload.write_bytes(words(1 | FORMATS['add'], 0, *initial, *expected))
        run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
        assert run.returncode == 0, (opcode, offset, run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
        print(f'LDS opcode {opcode:#x} offset {offset} contended={contend}: {TEXELS} texture words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as directory:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(directory), *args)
