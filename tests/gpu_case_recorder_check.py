"""tools/record_gpu_cases.py writes what the GPU check package replays: python gpu_case_recorder_check.py [dxc]."""
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tests')]
import record_gpu_cases as recorder  # noqa: E402
import startup_gfx2hlsl_compute_check as check  # noqa: E402


def main(dxc):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rows = recorder.record(dxc, out)
        assert [r[0] for r in rows if r[2] == 'pass'] == ['gfx2hlsl_compute/' + n for n in check.CASES], rows
        controls = {r[0]: r for r in rows if r[2] == 'fail'}
        assert sorted(controls) == sorted('gfx2hlsl_compute/control-' + m for m in check.CONTROLS), sorted(controls)
        manifest = [line.split('\t') for line in (out / 'manifest.tsv').read_text(encoding='ascii').splitlines()]
        assert manifest == [list(r) for r in rows], 'manifest rows differ from the recorded cases'
        for ident, compiler, expect, mismatch in manifest:
            assert compiler == 'dxc' and expect in ('pass', 'fail'), ident
            cso, case = (out / ('%s.%s' % (ident, ext)) for ext in ('cso', 'case'))
            assert cso.read_bytes()[:4] == b'DXBC', ident
            # group counts, WORDS expected words, 16 root constants, then any formatted input
            assert len(case.read_bytes()) >= 4 + 4 * check.WORDS + 64, ident
        for mutation, (name, _, _) in check.CONTROLS.items():
            base = out / ('gfx2hlsl_compute/%s.cso' % name)
            ctrl = out / ('gfx2hlsl_compute/control-%s.cso' % mutation)
            assert base.read_bytes() != ctrl.read_bytes(), 'control %s compiled to the unmutated shader' % mutation
            pinned = controls['gfx2hlsl_compute/control-' + mutation][3]
            assert pinned == check.CONTROL_MISMATCH.get(mutation, ''), mutation
    print('recorded %d gfx2hlsl compute cases and %d controls' % (len(check.CASES), len(check.CONTROLS)))
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    found = sys.argv[1] if len(sys.argv) == 2 else shutil.which('dxc')
    if not found:
        print('skip: no dxc on PATH and none given')
        sys.exit(77)
    sys.exit(main(found))
