"""Read the final SRV of a translated draw table on WARP and check its upper bound."""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_formatted_view_draw_check import CASES, CLEAR, PIXEL, TARGET, spec
from startup_stream_check import gfx2hlsl
from startup_stream_draw_check import vertex


def check(work, consumer, fxc):
    cache = work / 'cache'
    cache.mkdir()
    vs, ps, root, output = (work / name for name in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(vertex())
    ps.write_bytes(PIXEL)
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache), X2W_FXC=fxc,
               X2W_DRAW_FEATURES='bufword3', X2W_TEST_FORMATTED_VIEW=spec(CASES[0]))

    def draw():
        result = subprocess.run([consumer, vs, ps, root, 'warp', output, str(TARGET)], env=env,
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0, result.stderr[-2000:]
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr[-2000:]
        return struct.unpack('<4I', output.read_bytes()[:16]), result.stderr

    for _ in range(2):
        draw()
        subprocess.run([sys.executable, '-B', gfx2hlsl.__file__, '--pending', cache], env=env, check=True)
    variant = next(p for p in cache.glob('*.txt') if 'rtv1' in p.read_text())
    hlsl = variant.with_suffix('.ps.hlsl')
    source = hlsl.read_text()
    original_manifest = variant.read_text()
    assert 'srv 0 buf table 0 0' in original_manifest and 'register(t0)' in source
    # Only the final slot names the typed view. Reading any filler slot returns a different value.
    for slot in (31, 32, 63, 64):
        entries = ['srv %d buf vb 7' % i for i in range(slot)] + ['srv %d buf table 0 0' % slot]
        manifest = original_manifest.replace('srv 0 buf table 0 0', '\n'.join(entries))
        for kind in ('stride', 'bufword3'):
            manifest = re.sub(r'(udsrc \d+ %s )0$' % kind, lambda m: m[1] + str(slot), manifest, flags=re.M)
        variant.write_text(manifest.replace(' bufword3 ', ' descword3 '))
        hlsl.write_text(source.replace('register(t0)', 'register(t%d)' % slot))
        subprocess.run([fxc, '/nologo', '/T', 'ps_5_1', '/E', 'main', '/Fo', variant.with_suffix('.ps.cso'), hlsl],
                       check=True, capture_output=True)
        words, log = draw()
        accepted = slot < 64
        assert ('-> drawn;' in log) == accepted, (slot, log[-2000:])
        assert words == (CASES[0][5] if accepted else CLEAR), (slot, words, log[-2000:])
        print('t%d: %s' % (slot, 'bit-exact' if accepted else 'over-limit translation refused'))


if __name__ == '__main__':
    if not Path(sys.argv[2]).is_file():
        raise SystemExit(77)
    with tempfile.TemporaryDirectory() as temp:
        check(Path(temp), *sys.argv[1:])
