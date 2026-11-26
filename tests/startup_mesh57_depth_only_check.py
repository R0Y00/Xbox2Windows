"""Slot 57 mesh requests with a depth-only pixel stage translate in tools/startup/gfx2hlsl.py (#2413, T5).

usage: startup_mesh57_depth_only_check.py

A pixel stage without a colour output ends in `exp null off, off, off, off done vm` (LLVM SILateBranchLowering,
Mesa ACO create_fs_null_export). Before an alpha test it narrows EXEC, so the `vm` export kills the lanes it left.
The mesh stage is the accepted fixture of startup_mesh57_skeleton_check.py. The pixel words are built here:
v_interp_mov_f32 v0, p0, attr0.x; v_cmp_gt_f32 vcc, 0.5, v0; s_andn2_b64 exec, exec, vcc; exp null done vm;
s_endpgm. The pair must translate to a pixel stage with no SV_Target that keeps the discard, and a manifest with
an empty `targets` line. Controls: without the EXEC narrowing there is no discard, a null export followed by
a position export is still refused by the role check, and a null export before or after an mrt0, mrtz or param0
export is refused with no source or manifest written.
"""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import startup_mesh57_skeleton_check as S  # noqa: E402

INTERP, COMPARE, NARROW = 'c8020002', '7c0800f0', '8afe6a7e'
NULL_EXPORT, ENDPGM = ['f8001890', '00000000'], 'bf810000'
ALPHA_TEST = [INTERP, COMPARE, NARROW] + NULL_EXPORT + [ENDPGM]


def translate(pixel):
    with tempfile.TemporaryDirectory() as tmp:
        key, result = S.request(tmp, S.MESH, pixel)
        files = [Path(tmp) / ('%016x.%s' % (key, ext)) for ext in ('ps57.hlsl', 'ms57.txt')]
        return (result,) + tuple(f.read_text() if f.exists() else '' for f in files)


def main():
    os.environ['X2W_FXC'] = ''  # source and manifest only; no compiler needed
    S.expect('alpha-test words decode', [x[3] for x in S.G.decode(S.code(ALPHA_TEST))],
             ['v_interp_mov_f32_e32 v0, p0, attr0.x', 'v_cmp_gt_f32_e32 vcc_lo, 0.5, v0', 's_andn2_b64 exec, exec, vcc',
              'exp null off, off, off, off done vm', 's_endpgm'])
    result, hlsl, manifest = translate(ALPHA_TEST)
    S.expect('depth-only pair translates', result, 'ok')
    S.expect('pixel stage has no colour output', ('SV_Target' in hlsl, 'void main(PSIn i)' in hlsl), (False, True))
    S.expect('pixel stage keeps the vm discard', 'if (!(exec & 1)) discard;' in hlsl, True)
    S.expect('manifest names no target', manifest.splitlines()[-1:], ['targets '])
    S.expect('manifest is not the mesh-only form', 'ps none' in manifest.splitlines(), False)
    # Control: the same stage with all lanes alive exports under the entry EXEC, so nothing is discarded.
    result, hlsl, _ = translate([INTERP] + NULL_EXPORT + [ENDPGM])
    S.expect('null export alone translates', result, 'ok')
    S.expect('no narrowing, no discard', 'discard' in hlsl, False)
    # Control: a null export followed by a position export is not a depth-only pixel stage.
    position = ['f80008cf', '03020100']
    S.expect('null plus position is refused', translate(NULL_EXPORT + position + [ENDPGM])[0],
             'unsupported: slot 57 pixel stage has an incompatible export or termination')
    # Control: the slot 57 profile is the null export alone. A colour, depth or parameter export before or after
    # it is refused before any source or manifest is written.
    null_first = ['f8000090', '00000000']  # exp null off, off, off, off
    for target, before, after in (('mrt0', 'f800000f', 'f800180f'), ('mrtz', 'f8000081', 'f8001881'),
                                  ('param0', 'f800020f', 'f8001a0f')):
        for label, words, order in (
                ('%s + null' % target, [INTERP, before, '00000000'] + NULL_EXPORT + [ENDPGM], [target, 'null']),
                ('null + %s' % target, [INTERP] + null_first + [after, '00000000', ENDPGM], ['null', target])):
            decoded = [x[3] for x in S.G.decode(S.code(words))]
            S.expect(label + ' decodes', ([a.split()[1] for a in decoded[1:3]], decoded[2].endswith('done vm')),
                     (order, True))
            with tempfile.TemporaryDirectory() as tmp:
                key, result = S.request(tmp, S.MESH, words)
                S.expect(label + ' is refused', result,
                         'unsupported: slot 57 pixel stage mixes the null export with another export')
                S.expect(label + ' leaves only the failure', sorted(p.name for p in Path(tmp).iterdir()
                                                                    if not p.name.startswith('req_')),
                         ['%016x.fail' % key])
    if S.FAILED:
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
