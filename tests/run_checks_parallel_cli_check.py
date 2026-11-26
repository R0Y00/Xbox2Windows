#!/usr/bin/env python3
"""Pure-Python CLI, path-validation, subprocess-error, and mutation suite for tools/run_checks_parallel.py."""

import concurrent.futures
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "run_checks_parallel.py"
sys.path.insert(0, str(TOOLS_DIR))
import run_checks_parallel as R  # noqa: E402


def _run_cli(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = R.main(argv)
        except SystemExit as exc:
            code = exc.code
            rc = 0 if code is None else (int(code) if isinstance(code, int) else 1)
            if not isinstance(code, (type(None), int)):
                stderr.write(f"{code}\n")
    return rc, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        timeout=30,
    )


class RunChecksParallelCliPositiveTests(unittest.TestCase):
    def test_signature_and_main_positive_modes(self):
        sig = inspect.signature(R.run_checks)
        params = list(sig.parameters.values())
        self.assertEqual([p.name for p in params], ["checks", "jobs"])
        self.assertIs(params[1].default, None)
        self.assertEqual(R.SKIP_EXIT_CODE, 77)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ok1 = base / "ok1.py"
            ok2 = base / "ok2.py"
            ok1.write_text("import sys\nassert sys.flags.dont_write_bytecode\nprint('one')\n", encoding="utf-8")
            ok2.write_text("print('two')\n", encoding="utf-8")

            rc, stdout, stderr = _run_cli([str(ok1), str(ok2)])
            self.assertIs(type(rc), int)
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn(f"::group::{ok1} (ok, ", stdout)
            self.assertIn(f"::group::{ok2} (ok, ", stdout)
            self.assertLess(stdout.index("one\n"), stdout.index("two\n"))
            self.assertEqual(stdout.splitlines().count("::endgroup::"), 2)
            self.assertIn("all 2 checks passed", stdout)

            # main(None) reads sys.argv.
            with mock.patch.object(sys, "argv", [str(TOOL), str(ok1)]):
                rc_argv, out_argv, err_argv = _run_cli(None)
            self.assertIs(type(rc_argv), int)
            self.assertEqual(rc_argv, 0, err_argv)
            self.assertEqual(err_argv, "")
            self.assertIn("all 1 checks passed", out_argv)

            # __main__ entrypoint raises SystemExit(0).
            out_main, err_main = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(ok2)]),
                redirect_stdout(out_main),
                redirect_stderr(err_main),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(err_main.getvalue(), "")
            self.assertIn("all 1 checks passed", out_main.getvalue())

            # Subprocess positive smoke test.
            proc = _run_subprocess(str(ok1), str(ok2))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("all 2 checks passed", proc.stdout)

    def test_jobs_clamping_ordering_skip_and_env_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            slow = base / "slow.py"
            fast = base / "fast.py"
            skip = base / "skip.py"
            partial = base / "partial.py"
            env_echo = base / "env_echo.py"

            slow.write_text("import time\ntime.sleep(0.08)\nprint('slow done')\n", encoding="utf-8")
            fast.write_text("print('fast done')\n", encoding="utf-8")
            skip.write_text("import sys\nprint('skip reason')\nsys.exit(77)\n", encoding="utf-8")
            partial.write_text("import sys\nsys.stderr.write('partial line')\nsys.exit(3)\n", encoding="utf-8")
            env_echo.write_text(
                "import os, sys\nprint(sys.argv[1], os.environ.get('RCP_CLI_VAR', 'unset'))\n",
                encoding="utf-8",
            )

            seen_workers: list[int] = []
            real_executor = concurrent.futures.ThreadPoolExecutor

            def tracking_executor(*args, **kwargs):
                seen_workers.append(kwargs.get("max_workers"))
                return real_executor(*args, **kwargs)

            with mock.patch.object(R.concurrent.futures, "ThreadPoolExecutor", side_effect=tracking_executor):
                rc_zero, out_zero, err_zero = _run_cli(["-j", "0", str(fast)])
                self.assertEqual(rc_zero, 0, err_zero)
                self.assertIn("all 1 checks passed", out_zero)

                stdout_buf = io.StringIO()
                with redirect_stdout(stdout_buf):
                    rc_neg = R.run_checks([([str(fast)], None)], jobs=-4)
                self.assertEqual(rc_neg, 0)

            self.assertEqual(seen_workers, [1, 1])

            # Ordered groups, partial newline normalization, SKIP_EXIT_CODE=77, and FAILED ordering.
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_mix = R.run_checks(
                    [([str(slow)], None), ([str(skip)], None), ([str(partial)], None), ([str(fast)], None)],
                    jobs=4,
                )
            text = buf.getvalue()
            self.assertEqual(rc_mix, 1)
            self.assertLess(text.index("slow done\n"), text.index("skip reason\n"))
            self.assertLess(text.index("skip reason\n"), text.index("partial line\n::endgroup::"))
            self.assertLess(text.index("partial line\n"), text.index("fast done\n"))
            self.assertIn(f"::group::{skip} (exit 77, ", text)
            self.assertIn(f"::group::{partial} (exit 3, ", text)
            self.assertIn(f"FAILED: {partial}\n", text)
            self.assertIn(f"SKIPPED: {skip}\n", text)

            # Per-check extra_env must not leak across siblings.
            env_buf = io.StringIO()
            with mock.patch.dict(os.environ, {"RCP_CLI_VAR": "base"}):
                with redirect_stdout(env_buf):
                    rc_env = R.run_checks(
                        [([str(env_echo), "a"], {"RCP_CLI_VAR": "custom"}), ([str(env_echo), "b"], None)],
                        jobs=2,
                    )
            self.assertEqual(rc_env, 0)
            env_out = env_buf.getvalue()
            self.assertIn("a custom\n", env_out)
            self.assertIn("b base\n", env_out)

    def test_invalid_utf8_child_output_uses_replacement_without_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "emit_bad_bytes.py"
            script.write_text("import sys\nsys.stdout.buffer.write(b'ok \\xff\\xfe\\n')\n", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            proc = _run_subprocess(str(script), env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("all 1 checks passed", proc.stdout)


class RunChecksParallelCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        expected_substrings: tuple[str, ...] = (),
        check_subprocess: bool = False,
    ) -> str:
        rc, stdout, stderr = _run_cli(argv)
        if expected_code is None:
            self.assertNotEqual(rc, 0)
        else:
            self.assertEqual(rc, expected_code, stderr)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr.strip())
        self.assertNotIn("Traceback", stderr)
        for fragment in expected_substrings:
            self.assertIn(fragment, stderr)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code)
            self.assertEqual(proc.stdout, "")
            self.assertTrue(proc.stderr.strip())
            self.assertNotIn("Traceback", proc.stderr)
            for fragment in expected_substrings:
                self.assertIn(fragment, proc.stderr)
        return stderr

    def test_argparse_rejections_are_py314_compatible(self):
        # Missing positional scripts argument.
        err_missing = self._assert_clean_failure([], expected_code=None, check_subprocess=True)
        self.assertIn("required", err_missing.lower())

        # Invalid -j integer argument.
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "ok.py"
            ok.write_text("print('ok')\n", encoding="utf-8")
            err_jobs = self._assert_clean_failure(["-j", "abc", str(ok)], expected_code=None, check_subprocess=True)
            self.assertIn("invalid", err_jobs.lower())

    def test_upfront_script_path_validation_guards_kill_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid = base / "valid.py"
            valid.write_text("print('should not run')\n", encoding="utf-8")
            missing = base / "missing.py"
            script_dir = base / "dir_as_script"
            script_dir.mkdir()

            invalid_cases = (
                ("missing_single", [str(missing)], (str(missing), "does not exist"), True),
                ("dir_single", [str(script_dir)], (str(script_dir), "directory"), True),
                ("empty_str", [""], ("empty",), False),
                ("whitespace_str", ["   "], ("empty",), False),
                # Second script invalid: kills missing upfront loop / running first check anyway.
                ("second_missing", [str(valid), str(missing)], (str(missing), "does not exist"), True),
                ("second_dir", [str(valid), str(script_dir)], (str(script_dir), "directory"), True),
            )
            for label, argv, fragments, subproc in invalid_cases:
                with self.subTest(case=label):
                    with mock.patch.object(
                        R,
                        "run_checks",
                        side_effect=AssertionError(f"run_checks called for invalid case {label}"),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            expected_substrings=fragments,
                            check_subprocess=subproc,
                        )

            # Existing non-regular path (e.g. FIFO/special file): kills removal of `not path.is_file()`.
            with (
                mock.patch.object(R.Path, "exists", return_value=True),
                mock.patch.object(R.Path, "is_dir", return_value=False),
                mock.patch.object(R.Path, "is_file", return_value=False),
                mock.patch.object(R, "run_checks", side_effect=AssertionError("run_checks must not be called")),
            ):
                self._assert_clean_failure(
                    [str(valid)],
                    expected_code=1,
                    expected_substrings=(str(valid), "regular file"),
                )

    def test_run_one_catches_oserror_and_subprocess_error_without_crashing_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            broken = base / "broken.py"
            healthy = base / "healthy.py"
            broken.write_text("print('broken')\n", encoding="utf-8")
            healthy.write_text("print('healthy ok')\n", encoding="utf-8")

            error_cases = (
                ("filenotfound", FileNotFoundError("python missing"), "FileNotFoundError: python missing"),
                ("permission", PermissionError("access denied"), "PermissionError: access denied"),
                ("subproc_err", subprocess.SubprocessError("spawn failure"), "SubprocessError: spawn failure"),
                ("timeout_err", subprocess.TimeoutExpired(["python"], 5), "TimeoutExpired:"),
                ("empty_oserror", OSError(), "OSError"),
            )
            for label, exc, expected_msg in error_cases:
                with self.subTest(case=label):
                    with mock.patch.object(R.subprocess, "run", side_effect=exc):
                        script, code, output, elapsed = R.run_one(([str(broken)], None))
                    self.assertEqual(script, str(broken))
                    self.assertNotEqual(code, 0)
                    self.assertEqual(code, 1)
                    self.assertGreaterEqual(elapsed, 0.0)
                    self.assertIn(expected_msg, output)
                    self.assertTrue(output.endswith("\n"))
                    self.assertNotIn("Traceback", output)

            # Empty command raises ValueError cleanly (guarded by main()).
            with self.assertRaises(ValueError):
                R.run_one(([], None))

            # Pool isolation: one failing spawn must not crash ThreadPoolExecutor or lose sibling output.
            real_run = subprocess.run

            def selective_run(cmd, *args, **kwargs):
                if str(broken) in cmd:
                    raise OSError("cannot execute python")
                return real_run(cmd, *args, **kwargs)

            buf = io.StringIO()
            with mock.patch.object(R.subprocess, "run", side_effect=selective_run):
                with redirect_stdout(buf):
                    rc = R.run_checks([([str(broken)], None), ([str(healthy)], None)], jobs=2)
            out = buf.getvalue()
            self.assertEqual(rc, 1)
            self.assertIn(f"::group::{broken} (exit 1, ", out)
            self.assertIn("OSError: cannot execute python\n", out)
            self.assertIn(f"::group::{healthy} (ok, ", out)
            self.assertIn("healthy ok\n", out)
            self.assertEqual(out.splitlines().count("::endgroup::"), 2)
            self.assertIn(f"FAILED: {broken}\n", out)
            self.assertNotIn("Traceback", out)

    def test_main_guards_value_os_and_subprocess_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "ok.py"
            ok.write_text("print('ok')\n", encoding="utf-8")

            for label, exc, expected in (
                ("value_error", ValueError("invalid command"), ("invalid command",)),
                ("os_error", OSError("executor init failed"), ("executor init failed",)),
                ("subproc_error", subprocess.SubprocessError("subproc failure"), ("subproc failure",)),
                ("empty_value_error", ValueError(), ("ValueError",)),
            ):
                with self.subTest(case=label):
                    with mock.patch.object(R, "run_checks", side_effect=exc):
                        self._assert_clean_failure([str(ok)], expected_code=1, expected_substrings=expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
