"""Draw through a null-target export after a colour export (#1723).

usage:
  startup_gfx2hlsl_export_null_draw_check.py draw <dxc> <offscreen_check> <warp|hardware>

The pixel stage exports `mrt0`, then `exp null, off, off, off, off done`. The null target
writes no channel and keeps the four constants in target 0. The negative control makes the
null target write `mrt0 = 0`, which must produce a pixel mismatch.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import root_blob  # noqa: E402
import startup_image_3d_check as I3  # noqa: E402

S_ENDPGM = struct.pack('<I', 0xbf810000)


def exp(target, en, vsrc, *, done=True, vm=True):
    w0 = 0xf8000000 | (en & 0xf) | (target & 0x3f) << 4 | int(done) << 11 | int(vm) << 12
    return struct.pack('<2I', w0, sum((v & 0xff) << (8 * k) for k, v in enumerate(vsrc)))


def mov(dst, value):
    return struct.pack('<II', 0x7e0002ff | (dst << 17), value)


CONSTANTS = mov(0, 0x3f800000) + mov(1, 0x40000000) + mov(2, 0x40400000) + mov(3, 0x40800000)
NULL = exp(9, 0, (0, 0, 0, 0))
MRT0 = exp(0, 0xf, (0, 1, 2, 3))
VS_WORDS = CONSTANTS + exp(12, 0xf, (0, 1, 2, 3)) + S_ENDPGM
PS_WORDS = CONSTANTS + exp(0, 0xf, (0, 1, 2, 3), done=False) + NULL + S_ENDPGM


def main(dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        pipe = G.Pipeline(VS_WORDS, PS_WORDS, root_blob())
        hlsl = pipe.hlsl_ps()
        if 'o.mrt0 = ' not in hlsl or 'mrt9' in hlsl:
            print(hlsl)
            raise SystemExit('translated HLSL does not keep exactly mrt0')
        vs = I3.compile_hlsl(dxc, 'dxc', I3.FULLSCREEN_VS, 'null_export_vs', 'vs_6_0', directory)
        ps = I3.compile_hlsl(dxc, 'dxc', hlsl, 'null_export_ps', 'ps_6_0', directory)
        expected = directory / 'null_export_expected.bin'
        expected.write_bytes(struct.pack('<64I', *([0x3f800000, 0x40000000, 0x40400000, 0x40800000] * 16)))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(ps), '4', '4',
                              str(expected)], capture_output=True, text=True)
        if run.returncode == 77:
            print(run.stdout + run.stderr)
            return 77
        print('  ' + ' '.join(line for line in run.stdout.splitlines()
                              if line.startswith(('backend=', 'adapter=', 'driver='))))
        if run.returncode or 'all 16 pixels matched' not in run.stdout:
            print(run.stdout + run.stderr)
            raise SystemExit('null-after-mrt0 draw did not keep the colour export')

        bad_hlsl = hlsl.replace('    return o;', '    o.mrt0 = float4(0.0, 0.0, 0.0, 0.0);\n    return o;')
        bad = I3.compile_hlsl(dxc, 'dxc', bad_hlsl, 'null_writes_mrt0_ps', 'ps_6_0', directory)
        bad_run = subprocess.run([renderer, adapter, '--startup-pixel-words', str(vs), str(bad), '4', '4',
                                  str(expected)], capture_output=True, text=True)
        if bad_run.returncode == 0 or 'pixel mismatch' not in bad_run.stderr:
            print(bad_run.stdout + bad_run.stderr)
            raise SystemExit('negative control did not detect a null export that writes mrt0')
        print('null export draw: 16 pixels match; the writing-null control differs')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != 'draw':
        raise SystemExit(__doc__)
    sys.exit(main(*sys.argv[2:]))
