"""Malformed option values of tools/startup/debug_launch.py end in a one-line message and exit code 2.

The tool must return before it creates a job object or a process, so the executable need not exist.
The tool imports Windows-only ctypes entry points, so other hosts skip (exit code 77).

usage: startup_debug_launch_args_check.py [path/to/debug_launch.py]
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tools" / "startup" / "debug_launch.py"

BAD = [
    ["--timeout", "abc"],
    ["--timeout", "-1"],
    ["--timeout", "inf"],
    ["--timeout", "nan"],
    ["--memory-gib", "abc"],
    ["--memory-gib", "inf"],
    ["--memory-gib", "-inf"],
    ["--memory-gib", "nan"],
    ["--memory-gib", "-1"],
    ["--memory-gib", "0"],
    ["--memory-gib", "1e-12"],
    ["--memory-gib", "1e30"],
    ["--continue-asserts", "x"],
    ["--continue-asserts", "1.5"],
    ["--continue-asserts", "-2"],
]


def main():
    if os.name != "nt":
        print("debug_launch.py needs Windows; skipped")
        return 77
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "missing.exe")
        for option in BAD:
            logdir = os.path.join(tmp, "log")
            run = subprocess.run([sys.executable, "-B", str(TOOL), exe, tmp, logdir] + option,
                                 capture_output=True, text=True, timeout=60)
            text = run.stdout + run.stderr
            problems = []
            if run.returncode != 2:
                problems.append(f"exit code {run.returncode}")
            if "Traceback" in text:
                problems.append("traceback")
            if not text.startswith("expected "):
                problems.append("no 'expected ...' message")
            if len(text.strip().splitlines()) != 1:
                problems.append("not a single line")
            if os.path.exists(logdir):
                problems.append("log directory created before validation")
            if problems:
                failures.append(f"{' '.join(option)}: {', '.join(problems)}: {text.strip()[-200:]!r}")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        return 1
    print(f"{len(BAD)} malformed option values rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
