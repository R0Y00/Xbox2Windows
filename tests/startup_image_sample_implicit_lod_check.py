"""Plain image_sample keeps its implicit LOD past a uniform join in tools/startup/gfx2hlsl.py.

A plain image_sample takes its LOD from the coordinate derivatives of the pixel quad. After a block boundary the
translation lowered it to SampleLevel(.., 0), so the guest's mip was lost. Where the rule image_sample_b and
image_get_lod already use holds (pixel stage, entry EXEC mask, wave-uniform control flow, no divergent back edge over
the call) it is now the implicit-LOD Sample; everywhere else it stays level 0 (no new refusal).

Fixture (synthetic): the recorded c_lz stage of startup_image_sample_c_check with the MIMG opcode set to 0x20
(image_sample, a 2-register address), the same edits as startup_gfx2hlsl_image_get_lod_check.py:
  straight    nothing before the sample                        -> Sample (as before)
  branch      s_cbranch_scc0 to the next instruction            -> Sample
  join        a uniform forward join (scc), entry EXEC          -> Sample
  divergent   s_mov_b32 exec_lo, 0 before the sample            -> SampleLevel 0 (as before)
  cf_<op>     s_cbranch_vccz / vccnz / execz / execnz before it -> SampleLevel 0 (as before)
  back_<op>   a divergent back edge over the sample             -> SampleLevel 0 (as before)
  exit_<op>   a divergent forward exit, loop closed by s_branch -> SampleLevel 0 (as before)

  (no arguments)   translation controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_image_sample_c_check as C  # noqa: E402
from startup_gfx2hlsl_image_get_lod_check import AT, MIP_POINT, W2_AT, sopp  # noqa: E402
from startup_sampler_heap_check import VS_WORDS, root_blob  # noqa: E402

IMAGE_SAMPLE = 0x20


def stage(branch=False, divergent=False, join=False, cf_op=None, back_op=None, exit_op=None):
    w = bytearray(C.WORDS['lz'])
    w[W2_AT:W2_AT + 4] = struct.pack('<II', 0xbe9a03ff, MIP_POINT)  # s_mov_b32 s26, literal (mip filter point)
    w0, w1 = struct.unpack_from('<II', w, AT)
    w0 = w0 & ~(0x7f << 18) & ~(0xf << 8) & ~(7 << 3) | IMAGE_SAMPLE << 18 | 1 << 8 | 1 << 3
    struct.pack_into('<II', w, AT, w0, w1)
    code = bytes(w)
    if back_op is not None:  # a backward branch right after the sample, to it
        code = code[:AT + 8] + sopp(back_op, -3) + code[AT + 8:]
    if exit_op is not None:  # a forward exit right after the sample, then an unconditional branch back to it
        code = code[:AT + 8] + sopp(exit_op, 1) + sopp(2, -4) + code[AT + 8:]
    insert = b''
    if branch:
        insert += sopp(4, 0)
    if join:
        insert += bytes.fromhex('800380be') + bytes.fromhex('008006bf') + sopp(5, 1) + bytes.fromhex('810381be')
    if divergent:
        insert += bytes.fromhex('8003febe')
    if cf_op is not None:
        insert += bytes.fromhex('800380be') + sopp(cf_op, 1) + bytes.fromhex('810381be')
    return code[:AT] + insert + code[AT:] if insert else code


def calls(**kwargs):
    text = G.Pipeline(VS_WORDS, stage(**kwargs), root_blob(flags=0)).hlsl_ps()
    return [l.strip() for l in text.splitlines() if '.Sample' in l and 'SamplerState' not in l]


def main():
    for name, kwargs in (('straight', {}), ('branch', {'branch': True}), ('join', {'join': True})):
        got = calls(**kwargs)
        assert len(got) == 1 and '.Sample(' in got[0] and 'SampleLevel' not in got[0], (name, got)
    cases = [('divergent', {'divergent': True})]
    cases += [('cf_%d' % op, {'cf_op': op}) for op in (6, 7, 8, 9)]   # vccz, vccnz, execz, execnz
    cases += [('back_%d' % op, {'back_op': op}) for op in (7, 9)]      # vccnz, execnz back edge over the sample
    cases += [('exit_%d' % op, {'exit_op': op}) for op in (7, 9)]      # a divergent forward exit of a loop closed by s_branch
    for name, kwargs in cases:
        got = calls(**kwargs)
        assert len(got) == 1 and 'SampleLevel(' in got[0] and got[0].rstrip(';').endswith(', 0)'), (name, got)
    print('plain image_sample: implicit LOD past a uniform branch/join, level 0 with divergent EXEC or control flow')


def draw(dxc, renderer, adapter, fxc=None):
    from startup_image_get_lod_view_check import FULLSCREEN_VS, UNPACK_XY, compile_hlsl, read_pixel

    # Use screen coordinates so one pixel step is four texels at mip 0.
    # The 4x4 fixture must select physical mip 2 through either SRV window.
    code = stage(join=True)
    constant_coords = bytes.fromhex('f002007eff02027e0000803ef202047ef402067e')
    assert code.count(constant_coords) == 1
    code = code.replace(constant_coords, UNPACK_XY)
    lod_limits = bytes.fromhex('800399be')
    assert code.count(lod_limits) == 1
    code = code.replace(lod_limits, struct.pack('<II', 0xbe9903ff, 2 << 20))
    pipe = G.Pipeline(VS_WORDS, code, root_blob(flags=0), ps_hw_inputs=(0x8000, 0x8000))
    source = pipe.hlsl_ps()
    sampler, = (line for line in pipe.manifest().splitlines() if line.startswith('smpl '))
    implicit = '.Sample(q0, float2(F(v0), F(v1)))'
    assert source.count(implicit) == 1
    forced_level_zero = source.replace(implicit, '.SampleLevel(q0, float2(F(v0), F(v1)), 0)')

    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    if fxc:
        compilers.append(('fxc', fxc, 'vs_5_1', 'ps_5_1'))
    work_root = HERE.parent / 'out'
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        work = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_' + kind, vs_profile, work)
            guest = compile_hlsl(compiler, kind, source, 'implicit_' + kind, ps_profile, work)
            forced = compile_hlsl(compiler, kind, forced_level_zero, 'forced_' + kind, ps_profile, work)
            # Every channel exports the texel's red field. At (0,0), physical mip m stores (1+32m)/128.
            wanted = struct.pack('>4f', *((65.0 / 128,) * 4))
            for window, first_mip in (('0/3', 0), ('1/2', 1)):
                actual = read_pixel(renderer, adapter, str(vs), str(guest), sampler, window)
                control = read_pixel(renderer, adapter, str(vs), str(forced), sampler, window)
                wrong = struct.pack('>4f', *(((1 + 32 * first_mip) / 128,) * 4))
                assert actual == wanted, (kind, window, actual.hex(), wanted.hex())
                assert control == wrong and control != actual, (kind, window, control.hex(), wrong.hex())
                print(f'{kind} {adapter} {window}: implicit physical mip 2 {actual.hex()}, '
                      f'forced physical mip {first_mip} {control.hex()}')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == 'draw' and len(sys.argv) in (5, 6):
        draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)
