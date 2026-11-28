"""Check allocation-free direct DXIL lowering of guest wave32 DS gathers."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_dpp_mirror_check import vop2
from native_image_atomic_check import FORMATS, TEXELS, WIDTH, image_atomic, render
from native_scalar_lds_check import ds, sop1


OFFSET = 134
ACTIVE = 0xFFFFFEFD  # Disable source and destination lanes 1 and 8.
GATHER = ds(0xb3, addr=2, data0=3, dst=14, offset=OFFSET)
PREFIX = [
    # Load the image descriptor, then assign unique output texels.
    0xBE810380, 0xF40C0200, 0xFA000000, 0xBF8CC07F,
    vop2(0x1b, 8, 135, 0), vop2(0x16, 9, 131, 0),
    vop2(0x25, 8, 130, 8), vop2(0x25, 9, 129, 9),
    # vaddr = 4*lane + 3, source = 20 + lane, old destination = 17 + lane.
    vop2(0x1a, 2, 130, 0), vop2(0x25, 2, 131, 2),
    vop2(0x25, 3, 148, 0), vop2(0x25, 14, 145, 0),
    sop1(3, 126, 255), ACTIVE,
]


def compile_shader(cli, sdk, work, name, instruction, *, threads=32, wave32=True):
    source = work / f'{name}.cdx'
    shader = source.with_suffix('.cso')
    # Counter wait, restore EXEC, then observe every lane through the UAV.
    tail = [0xBD7D0000, sop1(3, 126, 193), *image_atomic('add'), 0xBBFD0000, 0xBF810000]
    source.write_bytes(compute(words(*PREFIX, *instruction, *tail), threads,
                               table=True, wave32=wave32, lds_blocks=0))
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


def controls(cli, sdk, work):
    shader, run = compile_shader(cli, sdk, work, 'gather', GATHER)
    assert run.returncode == 0, (run.stdout, run.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader],
                          capture_output=True, text=True, timeout=30, check=True).stdout
    assert 'addrspace(3)' not in dump and '@dx.op.barrier(i32 80, i32 8)' not in dump, dump
    assert dump.count('@dx.op.waveReadLaneAt.i32(i32 117,') == 32, dump
    # Check the SSA chain, including the separate source and destination EXEC
    # predicates and the old destination value.
    value = r'%[-\w.]+'
    shape = r'''
        (?P<addr>V) = load i32, i32\* %r258, align 4
        (?P<sum>V) = add i32 (?P=addr), 134
        (?P<word>V) = lshr i32 (?P=sum), 2
        (?P<lane>V) = and i32 (?P=word), 31
        (?P<bit>V) = shl i32 1, (?P=lane)
        (?P<exec>V) = load i32, i32\* %r126, align 4
        (?P<src_mask>V) = and i32 (?P=exec), (?P=bit)
        (?P<src_active>V) = icmp ne i32 (?P=src_mask), 0
        (?P<data>V) = load i32, i32\* %r259, align 4
    '''
    lines = [line.strip().replace('V', value) for line in shape.strip().splitlines()]
    match = re.search(r'\n +'.join(lines), dump)
    assert match, dump
    # Every lane is read with a constant index (a per-lane index read the wrong
    # lane on WARP), and the lane named by the address is selected.
    reads = re.findall(rf'({value}) = call i32 @dx\.op\.waveReadLaneAt\.i32\(i32 117, i32 '
                       rf'{re.escape(match["data"])}, i32 (\d+)\)', dump)
    assert sorted(int(k) for _, k in reads) == list(range(32)), dump
    for k in range(32):
        assert re.search(rf'= icmp eq i32 {re.escape(match["lane"])}, {k}\n', dump), (k, dump)
    gathered = re.search(rf'({value}) = select i1 {re.escape(match["src_active"])}, i32 {value}, i32 0\n', dump)
    assert gathered, dump
    # The independent destination predicate uses the current physical lane bit.
    lane = re.search(rf'({value}) = call i32 @dx\.op\.waveGetLaneIndex\(i32 111\)', dump)
    assert lane, dump
    lane_bit = re.search(rf'({value}) = shl i32 1, {re.escape(lane[1])}\n', dump)
    assert lane_bit, dump
    destination = [
        rf'(?P<exec>{value}) = load i32, i32\* %r126, align 4',
        rf'(?P<mask>{value}) = and i32 (?P=exec), {re.escape(lane_bit[1])}',
        rf'(?P<active>{value}) = icmp ne i32 (?P=mask), 0',
        rf'(?P<out>{value}) = select i1 (?P=active), i32 {re.escape(gathered[1])}, i32 (?P=old)',
        r'store i32 (?P=out), i32\* %r270, align 4',
    ]
    # The old-value load can be emitted before or after the mask operations,
    # depending on the compiler's C++ argument evaluation order.
    tail = dump[match.end():].split('store i32 ', 1)
    assert len(tail) == 2, dump
    old = re.search(rf'({value}) = load i32, i32\* %r270, align 4\n', tail[0])
    assert old, dump
    normalized = tail[0].replace(old[0], '')
    destination = [line.replace('(?P=old)', re.escape(old[1])) for line in destination]
    assert re.search(r'\n +'.join(destination), normalized + 'store i32 ' + tail[1]), dump

    for name, instruction, threads, wave32, message in (
        ('lds-read', ds(0x36, addr=2, dst=14), 32, True, 'LDS instruction without allocation'),
        ('scatter', ds(0xb2, addr=2, data0=3), 32, True, 'LDS instruction without allocation'),
        ('wave64', GATHER, 64, False, 'ds_bpermute_b32 requires guest wave32'),
        ('short-wave64', GATHER, 32, False, 'ds_bpermute_b32 requires guest wave32'),
    ):
        shader, run = compile_shader(cli, sdk, work, name, instruction, threads=threads, wave32=wave32)
        diagnostic = f'native compute +0x{len(PREFIX) * 4:x}: {message}'
        assert run.returncode and any(line.endswith(diagnostic) for line in run.stderr.splitlines()), (
            name, run.stdout, run.stderr,
        )
        assert not shader.exists(), name
    print('DS gather DXIL dataflow and refusal diagnostics matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    shader, run = compile_shader(cli, sdk, work, 'gather', GATHER)
    assert run.returncode == 0, (run.stdout, run.stderr)
    initial = [0] * TEXELS
    expected = initial.copy()
    for lane in range(32):
        source = ((4 * lane + 3 + OFFSET) >> 2) % 32
        result = 17 + lane
        if ACTIVE & (1 << lane):
            result = 20 + source if ACTIVE & (1 << source) else 0
        x, y = 2 + lane % 8, 1 + lane // 8
        expected[y * WIDTH + x] = result
    payload = work / 'texture.bin'
    payload.write_bytes(words(1 | FORMATS['add'], 0, *initial, *expected))
    run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
    assert run.returncode == 0, (run.stdout, run.stderr)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'DS gather and destination preservation matched across {TEXELS} texture words')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
