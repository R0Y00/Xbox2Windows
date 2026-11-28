"""Check source-built descriptor moves through VCC in direct compute."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_image_atomic_check import render


def mov(dst, src, *, pair=False):
    return 0xbe800000 | dst << 16 | (4 if pair else 3) << 8 | src


VCC_LO, VCC_HI, EXEC, M0, ZERO = 106, 107, 126, 124, 128
END = 0xbf810000
VCCZ = 0xbf860001  # Target skips the fallthrough s_endpgm.
WAIT = 0xbf8cc07f
SETUP = (mov(1, ZERO), mov(4, 0, pair=True))
LOAD = (0xf4080000 | 16 << 6 | 2, 125 << 25, WAIT)  # s[16:19], s[4:5], 0
PARK = (mov(VCC_LO, 18, pair=True),)
CLEAR = mov(18, ZERO, pair=True)
STORE = (
    0x1a << 25 | 1 << 17 | 130,   # v_lshlrev_b32 v1, 2, v0 (local x)
    0x7e000200 | 2 << 17 | 135,    # v_mov_b32 v2, 7
    0xe0701000, 128 << 24 | 4 << 16 | 2 << 8 | 1,  # buffer_store_dword, offen
)
VOPC = (0x7c000000 | 0xc2 << 17 | ZERO,)  # v_cmp_eq_u32 vcc, 0, v0
CARRY = (0xd70f0000 | VCC_LO << 8 | 3, ZERO | 256 << 9)  # v_add_co_u32 v3, vcc, 0, v0
CNDMASK = 1 << 25 | 3 << 17 | ZERO  # v_cndmask_b32 v3, 0, v0, vcc

BASELINE = (*SETUP, CLEAR, *LOAD, *STORE)
PAIR_COPY = (*SETUP, *LOAD, *PARK, CLEAR, mov(18, VCC_LO, pair=True), *STORE)
WORD_COPY = (*SETUP, *LOAD, mov(VCC_LO, 18), mov(VCC_HI, 19), CLEAR,
             mov(18, VCC_LO), mov(19, VCC_HI), *STORE)


def compile_shader(cli, sdk, work, name, code, *, wave64):
    source = work / f'{name}.cdx'
    shader = source.with_suffix('.cso')
    source.write_bytes(compute(words(*code, END), threads=64 if wave64 else 32,
                               table=True, wave32=not wave64))
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                            capture_output=True, text=True, timeout=30)
    return shader, result


def accept(cli, sdk, work, name, code, *, wave64):
    shader, result = compile_shader(cli, sdk, work, name, code, wave64=wave64)
    assert result.returncode == 0 and shader.is_file(), (name, result.stdout, result.stderr)
    return shader


def reject(cli, sdk, work, name, prefix, consumer, message, *, wave64, tail=()):
    shader, result = compile_shader(cli, sdk, work, name, (*prefix, *consumer, *tail), wave64=wave64)
    diagnostic = f'native compute +0x{4 * len(prefix):x}: {message}'
    assert result.returncode and not shader.exists(), (name, result.stdout, result.stderr)
    assert diagnostic in result.stderr.splitlines(), (name, diagnostic, result.stderr)


def store_call(sdk, shader):
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader],
                          capture_output=True, text=True, timeout=30, check=True).stdout
    stores = [line.strip() for line in dump.splitlines()
              if 'call void @dx.op.' in line and 'bufferstore' in line.lower()]
    assert len(stores) == 1 and re.fullmatch(
        r'call void @dx\.op\.rawBufferStore\.i32\(i32 140, %dx\.types\.Handle %\w+, '
        r'i32 %\w+, i32 undef, i32 %\w+, i32 undef, i32 undef, i32 undef, i8 1, i32 4\)'
        r'(?: #[0-9]+)?(?:\s*;.*)?', stores[0]
    ), dump
    return stores[0]


def controls(cli, sdk, work):
    for wave64 in (False, True):
        width = 64 if wave64 else 32
        baseline = store_call(sdk, accept(cli, sdk, work, f'base{width}', BASELINE, wave64=wave64))
        for name, code in (('pair', PAIR_COPY), ('word', WORD_COPY)):
            shader = accept(cli, sdk, work, f'{name}{width}', code, wave64=wave64)
            assert store_call(sdk, shader) == baseline, name

        parked = (*SETUP, *LOAD, *PARK)
        reject(cli, sdk, work, f'vccz{width}', parked, (VCCZ,), 'unbound numeric SGPR s106',
               wave64=wave64, tail=(END,))
        # A single symbolic half keeps the diagnostic independent of operand evaluation order.
        cndmask_input = (*parked, mov(VCC_HI, ZERO)) if wave64 else parked
        reject(cli, sdk, work, f'cndmask{width}', cndmask_input, (CNDMASK,), 'unbound numeric SGPR s106',
               wave64=wave64)
        reject(cli, sdk, work, f'exec{width}', parked, (mov(EXEC, VCC_LO, pair=True),),
               'descriptor used as mask', wave64=wave64)
        reject(cli, sdk, work, f'm0{width}', parked, (mov(M0, VCC_HI),),
               'descriptor used as mask', wave64=wave64)
        reject(cli, sdk, work, f'no-restore{width}', (*parked, CLEAR, *STORE[:2]), STORE[2:],
               'non-contiguous resource descriptor', wave64=wave64)

        for name, clobber in (('compare', VOPC), ('carry', CARRY)):
            overwritten = (*parked, *clobber)
            accept(cli, sdk, work, f'{name}-vccz{width}', (*overwritten, VCCZ, END), wave64=wave64)
            if not wave64:
                reject(cli, sdk, work, f'{name}-hi{width}', overwritten, (mov(M0, VCC_HI),),
                       'descriptor used as mask', wave64=False)

    # Move a mixed pair (numeric low, descriptor high), and restore the
    # high descriptor through another mixed pair without touching s18.
    mixed = (*SETUP, *LOAD, mov(22, ZERO), mov(23, 19), mov(VCC_LO, 22, pair=True),
             mov(19, ZERO), mov(22, VCC_LO, pair=True), mov(19, 23), *STORE)
    store_call(sdk, accept(cli, sdk, work, 'mixed', mixed, wave64=True))
    high_only = (*SETUP, *LOAD, mov(VCC_LO, ZERO), mov(VCC_HI, 19))
    reject(cli, sdk, work, 'vccz-high64', high_only, (VCCZ,), 'unbound numeric SGPR s107',
           wave64=True, tail=(END,))
    reject(cli, sdk, work, 'cndmask-high64', high_only, (CNDMASK,), 'unbound numeric SGPR s107',
           wave64=True)
    print('VCC descriptor copies, raw UAV store and numeric-use refusals matched')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    case = work / 'store.bin'
    for width in (32, 64):
        case.write_bytes(words(1, *([7] * width), *([0xc5c5c5c5] * (320 - width))))
        for name, code in (('pair', PAIR_COPY), ('word', WORD_COPY)):
            shader = accept(cli, sdk, work, f'{name}{width}', code, wave64=width == 64)
            result = render(renderer, adapter, '--compute-store-sized-case', shader, case)
            assert result.returncode == 0, (name, width, result.stdout, result.stderr)
            assert 'groups=1 compared_words=320 mismatches=0' in result.stdout, result.stdout
            print(f'{name} wave{width}: 320 output and guard words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
