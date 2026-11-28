"""Check direct DXIL lowering of a source-built A16 RGBA32 image store."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from native_direct_container_check import compute, words
from native_dpp_mirror_check import vop2
from native_image_atomic_check import TEXELS, WIDTH, render
from native_scalar_lds_check import sop1


CHANNEL_BASES = (0x3f800000, 0x40000000, 0x40800000, 0x41000000)
REFUSAL = 'packed image addressing'
SSA = r'%[\w.$-]+'


def program(op, dim):
    # Load s[8:15] from the descriptor table and wait for the scalar load.
    code = [0xBE810380, 0xF40C0200, 0xFA000000, 0xBF8CC07F]
    # v0 is local_id.x (0..31). v8 packs (x, 1) into one VGPR.
    code += [vop2(0x25, 8, 255, 0), 1 << 16]
    for reg, base in enumerate(CHANNEL_BASES, start=14):
        code += [vop2(0x25, reg, 255, 0), base]
    code += [sop1(3, 126, 255), 0x55555555]  # EXEC explicitly selects even lanes.
    code += [0xF0001F00 | (op << 18) | (dim << 3), 0x40020E08]
    return compute(words(*code, 0xBBFD0000, 0xBF810000), table=True)


def compile_shader(cli, sdk, work, name, *, op=8, dim=1):
    source = work / f'{name}.cs'
    shader = source.with_suffix('.cso')
    source.write_bytes(program(op, dim))
    run = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-cs', source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


def valid_shader(cli, sdk, work):
    shader, run = compile_shader(cli, sdk, work, 'store-2d')
    assert run.returncode == 0 and shader.is_file(), (run.stdout, run.stderr)
    return shader


def controls(cli, sdk, work):
    shader = valid_shader(cli, sdk, work)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader], capture_output=True,
                          text=True, timeout=30, check=True).stdout
    stores = re.findall(
        rf'call void @dx\.op\.textureStore\.f32\(i32 67, %dx\.types\.Handle {SSA}, '
        rf'i32 ({SSA}), i32 ({SSA}), i32 undef, (?:float {SSA}, ){{4}}i8 15\)', dump,
    )
    assert len(stores) == 1 and '2d' in dump, dump
    x, y = stores[0]
    low = re.search(rf'^\s*{re.escape(x)} = and i32 ({SSA}), 65535$', dump, re.MULTILINE)
    assert low, dump
    assert re.search(rf'^\s*{re.escape(y)} = lshr i32 {re.escape(low[1])}, 16$', dump,
                     re.MULTILINE), dump
    assert 'legacyF16ToF32' not in dump, dump

    # Opcode 1 is image_load_mip, which reaches the shared A16 LOD guard.
    # image_store_mip is rejected by the earlier opcode validation.
    for name, encoding in (('load-mip-2d', {'op': 1}), ('store-3d', {'dim': 2})):
        output, run = compile_shader(cli, sdk, work, name, **encoding)
        assert run.returncode and not output.exists(), (name, run.stdout, run.stderr)
        assert re.search(rf'native compute \+0x[0-9a-f]+: {REFUSAL}$', run.stderr,
                         re.MULTILINE), (name, run.stdout, run.stderr)
    print('A16 RGBA32 textureStore coordinates and both refusal diagnostics matched')


def hardware(cli, sdk, work, renderer, adapter):
    shader = valid_shader(cli, sdk, work)
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    print(wave.stdout, end='')
    initial = [0xbf800000 + i for i in range(TEXELS * 4)]
    expected = initial.copy()
    for lane in range(0, WIDTH, 2):
        index = (WIDTH + lane) * 4  # mip 0, y=1, x=lane
        expected[index:index + 4] = [base + lane for base in CHANNEL_BASES]
    payload = work / 'texture.bin'
    # The RGBA32 FLOAT runner takes initial/output arrays and a separate SRV upload.
    payload.write_bytes(words(1 | (1 << 30), 0, *initial, *expected, *initial))
    run = render(renderer, adapter, '--compute-texture-store-case', shader, payload)
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert f'compared_words={len(expected)} mismatches=0' in run.stdout, run.stdout
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'{adapter}: {len(expected)} RGBA32 texture words matched')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
