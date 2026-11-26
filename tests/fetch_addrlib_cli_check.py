#!/usr/bin/env python3
"""Pure-Python offline CLI, path-guard, subprocess-error, and mutation suite for tools/fetch_addrlib.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "fetch_addrlib.py"
sys.path.insert(0, str(TOOLS_DIR))
import fetch_addrlib as F  # noqa: E402


def _run_cli(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = F.main(argv)
        except SystemExit as exc:
            code = exc.code
            rc = 0 if code is None else (int(code) if isinstance(code, int) else 1)
            if not isinstance(code, (type(None), int)):
                stderr.write(f"{code}\n")
    return rc, stdout.getvalue(), stderr.getvalue()


_REAL_SUBPROCESS_RUN = subprocess.run


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return _REAL_SUBPROCESS_RUN(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def _make_fake_git(
    dest: Path,
    calls: list[tuple[list[str], Path | None]],
    *,
    origin_url: str = F.URL,
    dirty_calls: set[int] | None = None,
    head_commit: str = F.COMMIT,
    create_header: bool = True,
):
    status_count = 0
    dirty_indices = dirty_calls or set()

    def fake_run(cmd, cwd=None, check=False, capture_output=False, text=False):
        nonlocal status_count
        calls.append((list(cmd), Path(cwd) if cwd is not None else None))
        sub = cmd[1:]
        if sub[:2] == ["init", "-q"]:
            (dest / ".git").mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if sub[:3] == ["remote", "get-url", "origin"]:
            return subprocess.CompletedProcess(cmd, 0, f"{origin_url}\n", "")
        if sub[:2] == ["status", "--porcelain"]:
            status_count += 1
            out = " M src/amd/addrlib/file.cpp\n" if status_count in dirty_indices else ""
            return subprocess.CompletedProcess(cmd, 0, out, "")
        if sub[:2] == ["checkout", "-q"]:
            if create_header:
                header = dest / F.SUBDIR / "inc" / "addrinterface.h"
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text("// stub header\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if sub[:2] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, f"{head_commit}\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


class FetchAddrlibCliPositiveTests(unittest.TestCase):
    def test_fresh_checkout_and_entrypoints_succeed_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "mesa_fresh"
            calls: list[tuple[list[str], Path | None]] = []
            with mock.patch.object(F.subprocess, "run", side_effect=_make_fake_git(dest, calls)):
                rc, stdout, stderr = _run_cli(["--dest", str(dest)])

            self.assertIs(type(rc), int)
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            addrlib = dest.resolve() / F.SUBDIR
            self.assertIn(f"addrlib {F.COMMIT} at {addrlib}", stdout)
            self.assertIn(f"configure with -DXBOX2WINDOWS_ADDRLIB_DIR={addrlib.as_posix()}", stdout)
            self.assertEqual(
                [cmd[1] for cmd, _ in calls],
                ["init", "remote", "sparse-checkout", "fetch", "checkout", "status", "rev-parse"],
            )
            self.assertTrue(all(cwd == dest.resolve() for _, cwd in calls))

            # main(None) reads sys.argv.
            dest_argv = Path(tmp) / "mesa_argv"
            calls_argv: list[tuple[list[str], Path | None]] = []
            with (
                mock.patch.object(sys, "argv", [str(TOOL), "--dest", str(dest_argv)]),
                mock.patch.object(F.subprocess, "run", side_effect=_make_fake_git(dest_argv, calls_argv)),
            ):
                rc_argv, out_argv, err_argv = _run_cli(None)
            self.assertIs(type(rc_argv), int)
            self.assertEqual(rc_argv, 0, err_argv)
            self.assertEqual(err_argv, "")
            self.assertIn(f"addrlib {F.COMMIT}", out_argv)

            # __main__ entrypoint raises SystemExit(0).
            dest_main = Path(tmp) / "mesa_main"
            calls_main: list[tuple[list[str], Path | None]] = []
            out_main, err_main = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), "--dest", str(dest_main)]),
                mock.patch.object(F.subprocess, "run", side_effect=_make_fake_git(dest_main, calls_main)),
                redirect_stdout(out_main),
                redirect_stderr(err_main),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(err_main.getvalue(), "")
            self.assertIn(f"addrlib {F.COMMIT}", out_main.getvalue())

    def test_existing_clean_checkout_and_help_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "mesa_existing"
            (dest / ".git").mkdir(parents=True)
            calls: list[tuple[list[str], Path | None]] = []
            with mock.patch.object(F.subprocess, "run", side_effect=_make_fake_git(dest, calls)):
                rc, stdout, stderr = _run_cli(["--dest", str(dest)])

            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn(f"addrlib {F.COMMIT}", stdout)
            self.assertEqual(
                [cmd[1] for cmd, _ in calls],
                ["remote", "status", "sparse-checkout", "fetch", "checkout", "status", "rev-parse"],
            )

        proc = _run_subprocess("--help")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertIn("--dest", proc.stdout)


class FetchAddrlibCliNegativeAndMutationTests(unittest.TestCase):
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

    def test_argparse_rejection_is_py314_compatible(self):
        err = self._assert_clean_failure(["--bogus-flag"], expected_code=None, check_subprocess=True)
        self.assertIn("unrecognized", err.lower())

    def test_dest_path_and_non_empty_guards_kill_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            regular_file = base / "not_a_dir.txt"
            regular_file.write_text("file contents", encoding="utf-8")

            non_empty = base / "non_empty"
            non_empty.mkdir()
            (non_empty / ".keep").write_text("x", encoding="utf-8")

            # Existing regular file passed as --dest (and as parent of --dest) -> clean exit 1 before any git call.
            for label, target, substrs, subproc in (
                ("regular_file_dest", regular_file, ("fetch_addrlib:", "not a directory"), True),
                ("file_as_parent", regular_file / "child", ("fetch_addrlib:",), True),
                ("non_empty_dir", non_empty, ("is not empty",), True),
            ):
                with self.subTest(case=label):
                    with mock.patch.object(
                        F.subprocess,
                        "run",
                        side_effect=AssertionError(f"subprocess.run called for {label}"),
                    ):
                        self._assert_clean_failure(
                            ["--dest", str(target)],
                            expected_code=1,
                            expected_substrings=substrs,
                            check_subprocess=subproc,
                        )

    def test_checkout_refusals_preserved_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # Wrong origin remote.
            wrong_origin = base / "wrong_origin"
            (wrong_origin / ".git").mkdir(parents=True)
            calls_origin: list[tuple[list[str], Path | None]] = []
            with mock.patch.object(
                F.subprocess,
                "run",
                side_effect=_make_fake_git(wrong_origin, calls_origin, origin_url="https://example.invalid/repo.git"),
            ):
                self._assert_clean_failure(
                    ["--dest", str(wrong_origin)],
                    expected_code=1,
                    expected_substrings=("not the Mesa dependency checkout",),
                )
            self.assertEqual([cmd[1] for cmd, _ in calls_origin], ["remote"])

            # Dirty checkout before fetch (status call #1).
            dirty_pre = base / "dirty_pre"
            (dirty_pre / ".git").mkdir(parents=True)
            with mock.patch.object(
                F.subprocess,
                "run",
                side_effect=_make_fake_git(dirty_pre, [], dirty_calls={1}),
            ):
                self._assert_clean_failure(
                    ["--dest", str(dirty_pre)],
                    expected_code=1,
                    expected_substrings=("has local changes", "src/amd/addrlib/file.cpp"),
                )

            # Dirty checkout after checkout (status call #2).
            dirty_post = base / "dirty_post"
            (dirty_post / ".git").mkdir(parents=True)
            with mock.patch.object(
                F.subprocess,
                "run",
                side_effect=_make_fake_git(dirty_post, [], dirty_calls={2}),
            ):
                self._assert_clean_failure(
                    ["--dest", str(dirty_post)],
                    expected_code=1,
                    expected_substrings=("has local changes",),
                )

            # Commit mismatch.
            bad_commit = base / "bad_commit"
            with mock.patch.object(
                F.subprocess,
                "run",
                side_effect=_make_fake_git(bad_commit, [], head_commit="0" * 40),
            ):
                self._assert_clean_failure(
                    ["--dest", str(bad_commit)],
                    expected_code=1,
                    expected_substrings=("checked out", f"expected {F.COMMIT}"),
                )

            # Missing addrinterface.h header.
            no_header = base / "no_header"
            with mock.patch.object(
                F.subprocess,
                "run",
                side_effect=_make_fake_git(no_header, [], create_header=False),
            ):
                self._assert_clean_failure(
                    ["--dest", str(no_header)],
                    expected_code=1,
                    expected_substrings=("has no inc/addrinterface.h",),
                )

    def test_oserror_valueerror_and_subprocess_error_guards_kill_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "err_dest"

            for label, exc, expected in (
                ("called_process_error", subprocess.CalledProcessError(128, ["git", "fetch"]), ("fetch_addrlib:", "128")),
                ("timeout_expired", subprocess.TimeoutExpired(["git", "fetch"], 10), ("fetch_addrlib:", "timed out")),
                ("missing_git_oserror", FileNotFoundError("git executable not found"), ("fetch_addrlib:", "git executable not found")),
            ):
                with self.subTest(case=label):
                    target = dest / label
                    with mock.patch.object(F.subprocess, "run", side_effect=exc):
                        self._assert_clean_failure(
                            ["--dest", str(target)],
                            expected_code=1,
                            expected_substrings=expected,
                        )

            # ValueError inside guarded block -> clean exit 1 via parser.exit.
            empty_dest = dest / "value_err_dest"
            empty_dest.mkdir(parents=True)
            with mock.patch.object(F.pathlib.Path, "iterdir", side_effect=ValueError("invalid path value")):
                self._assert_clean_failure(
                    ["--dest", str(empty_dest)],
                    expected_code=1,
                    expected_substrings=("fetch_addrlib:", "invalid path value"),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
