"""tests/pixel_aot_main.py behaves like running tests/pixel_aot_check.py by path.

A draw mode with a compiler path that does not exist fails fast, without DXC or a GPU. The entry script
must forward the mode and arguments, return the same failing exit status, and still write the
form-coverage record at exit.
"""
import os
import subprocess
import sys
import tempfile

TESTS = os.path.dirname(os.path.abspath(__file__))
MISSING = os.path.join(tempfile.gettempdir(), "pixel_aot_main_check_no_such_dxc")


def run(script, record):
    env = dict(os.environ, FORM_COVERAGE_RECORD=record)
    return subprocess.run([sys.executable, "-B", os.path.join(TESTS, script), "draw-sad", MISSING, MISSING, "warp"],
                          capture_output=True, text=True, env=env, timeout=120)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        direct_record = os.path.join(tmp, "direct.txt")
        main_record = os.path.join(tmp, "main.txt")
        direct = run("pixel_aot_check.py", direct_record)
        entry = run("pixel_aot_main.py", main_record)
        assert direct.returncode != 0, "control run unexpectedly passed"
        assert entry.returncode == direct.returncode, (entry.returncode, direct.returncode, entry.stderr[-400:])
        assert os.path.basename(MISSING) in entry.stderr, entry.stderr[-400:]
        assert os.path.exists(main_record), "exit record was not written"
        with open(direct_record, encoding="utf-8") as a, open(main_record, encoding="utf-8") as b:
            assert a.read() == b.read()
    print("pixel_aot_main forwards arguments, exit status and the exit record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
