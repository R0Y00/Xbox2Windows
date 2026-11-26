"""Record the compiled gfx2hlsl compute cases for the GPU check package.

    python -B tools/record_gpu_cases.py <dxc> <output-dir>

For every case and every mutation control of tests/startup_gfx2hlsl_compute_check.py this writes the DXC cs_6_0
object and the case file that `offscreen_check <warp|hardware> --compute-store-case <cso> <case>` reads, plus a
manifest the package's Run-GpuCheck.ps1 replays on the user's own adapter. The cases are built by the same
run_case() the dispatch tests use, so the bytes are the ones those tests dispatch. Nothing here talks to a GPU:
run_case() is given a runner that does nothing.

manifest.tsv, one row per case, tab separated, no header:
    id <TAB> compiler <TAB> expect <TAB> mismatch
    id         gfx2hlsl_compute/<case> or gfx2hlsl_compute/control-<mutation>; the files are <id>.cso and <id>.case
    compiler   dxc (cs_6_0)
    expect     pass: exit 0 and no diagnostics; fail: exit 1 and a `word=... expected=...` diagnostic (a mutation
               control that PASSES on a card is the failure)
    mismatch   for fail rows with a pinned first bad word, that diagnostic line; otherwise empty
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT / 'tools'), str(ROOT / 'tools' / 'startup')]
import startup_gfx2hlsl_compute_check as check  # noqa: E402

COMPILER = 'dxc'
GROUP = 'gfx2hlsl_compute'


def record(dxc, output, cases=None, controls=None):
    """Write the cases and the manifest into `output`; return the manifest rows."""
    output = Path(output)
    target = output / GROUP
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = [(name, None) for name in (check.CASES if cases is None else cases)]
    jobs += [(check.CONTROLS[m][0], m) for m in (check.CONTROLS if controls is None else controls)]
    for name, mutation in jobs:
        ident = '%s/%s' % (GROUP, 'control-' + mutation if mutation else name)
        with tempfile.TemporaryDirectory() as tmp:
            # The runner is only started to be refused: python rejects the adapter word as a script name.
            check.run_case(name, dxc, sys.executable, 'warp', Path(tmp), mutation)
            for ext in ('cso', 'case'):
                shutil.copyfile(Path(tmp) / ('%s.%s' % (name, ext)), output / ('%s.%s' % (ident, ext)))
        rows.append((ident, COMPILER, 'fail' if mutation else 'pass', check.CONTROL_MISMATCH.get(mutation, '') if mutation else ''))
    with open(output / 'manifest.tsv', 'w', newline='\n', encoding='ascii') as manifest:
        for row in rows:
            manifest.write('\t'.join(row) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dxc')
    parser.add_argument('output')
    args = parser.parse_args()
    rows = record(args.dxc, args.output)
    print('recorded %d cases (%d controls) into %s' % (len(rows), sum(r[2] == 'fail' for r in rows), args.output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
