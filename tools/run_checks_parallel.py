"""Run independent Python check scripts side by side and fail if any of them fails.

Usage:
    python tools/run_checks_parallel.py [-j N] tests/a_check.py tests/b_check.py ...

Each script runs as `python -B <script>` from the current directory. Output is printed one
script at a time, in the order given, so a failure log stays readable. The exit status is 1
when any script exits non-zero. Scripts that exit 77 (skip) are listed under `SKIPPED:`,
apart from the `FAILED:` line; a skip still makes the exit status 1.

`run_checks(checks, jobs=None)` runs the same runner directly over check pairs. Each pair is
`(command, extra_env)`: `command` is the argument list placed after `python -B` and
`extra_env` (empty or None to inherit) is merged over the current environment for that
check only.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time

SKIP_EXIT_CODE = 77


def _validate_script_path(raw_path: str) -> str:
    if not raw_path or not raw_path.strip():
        raise ValueError("script path must not be empty")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"script path does not exist: {raw_path}")
    if path.is_dir():
        raise IsADirectoryError(f"script path is a directory: {raw_path}")
    if not path.is_file():
        raise ValueError(f"script path is not a regular file: {raw_path}")
    return raw_path


def run_one(check):
    command, extra_env = check
    if not command:
        raise ValueError("check command must not be empty")
    env = {**os.environ, **extra_env} if extra_env else None
    start = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, "-B", *command], env=env,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        msg = f"{type(exc).__name__}: {detail}\n" if detail else f"{type(exc).__name__}\n"
        return command[0], 1, msg, time.monotonic() - start
    return command[0], proc.returncode, proc.stdout, time.monotonic() - start


def run_checks(checks, jobs=None):
    checks = list(checks)
    failed = []
    skipped = []
    max_workers = max(1, os.cpu_count() or 2) if jobs is None else max(1, jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for script, code, output, secs in pool.map(run_one, checks):
            print(f"::group::{script} ({'ok' if code == 0 else f'exit {code}'}, {secs:.1f}s)", flush=True)
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                print()
            print("::endgroup::", flush=True)
            if code == SKIP_EXIT_CODE:
                skipped.append(script)
            elif code != 0:
                failed.append(script)
    if failed:
        print("FAILED: " + " ".join(failed))
    if skipped:
        print("SKIPPED: " + " ".join(skipped))
    if failed or skipped:
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    parser.add_argument("scripts", nargs="+")
    args = parser.parse_args(argv)
    try:
        scripts = [_validate_script_path(script) for script in args.scripts]
        return run_checks([([script], None) for script in scripts], jobs=args.jobs)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        msg = f"{detail}\n" if detail else f"{type(exc).__name__}\n"
        parser.exit(1, msg)


if __name__ == "__main__":
    sys.exit(main())
