#!/usr/bin/env python3
"""Pure-Python CLI validation, atomic-report, and mutation suite for tools/intake_exe.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_DIR / "intake_exe.py"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

import intake_exe  # noqa: E402
from intake_exe_check import _build_pe, _fake_scan_for, _synthetic_package  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = intake_exe.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class IntakeExeCliPositiveTests(unittest.TestCase):
    def test_inspected_awaiting_error_json_and_require_routing_with_zero_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # 1) awaiting_executable -> exit 3, 0 .tmp files.
            pkg_wait, base_w, int_w = _synthetic_package(base / "wait", write_candidate=False)
            rep_wait = base / "wait" / "nested" / "reports"
            code_w, out_w, err_w = _run_main([
                "--build", str(pkg_wait),
                "--baseline", str(base_w),
                "--integrity", str(int_w),
                "--reports", str(rep_wait),
            ])
            self.assertIs(type(code_w), int)
            self.assertEqual(code_w, 3, (out_w, err_w))
            self.assertEqual(err_w, "")
            self.assertIn("awaiting_executable |", out_w)
            wait_files = list(rep_wait.iterdir())
            self.assertEqual(len(wait_files), 1)
            self.assertTrue(wait_files[0].name.endswith(".awaiting_executable.json"))
            self.assertEqual(json.loads(wait_files[0].read_text(encoding="utf-8"))["status"], "awaiting_executable")

            # 2) inspected with explicit PE candidate -> exit 0, 0 .tmp files.
            pe_ok = _build_pe(imports=[("normal", "xmem.dll", "XMemAlloc")])
            pkg_ok, base_ok, int_ok = _synthetic_package(base / "ok", write_candidate=False)
            cand_ok = base / "ok" / "candidate.exe"
            cand_ok.write_bytes(pe_ok)
            rep_ok = base / "ok" / "reports"
            with mock.patch.object(intake_exe, "scan", _fake_scan_for(pkg_ok, pe_ok)):
                code_ok, out_ok, err_ok = _run_main([
                    str(cand_ok),
                    "--build", str(pkg_ok),
                    "--baseline", str(base_ok),
                    "--integrity", str(int_ok),
                    "--reports", str(rep_ok),
                    "--require-routing",
                ])
            self.assertEqual(code_ok, 0, (out_ok, err_ok))
            self.assertEqual(err_ok, "")
            self.assertIn("inspected |", out_ok)
            ok_files = list(rep_ok.iterdir())
            self.assertEqual(len(ok_files), 1)
            self.assertTrue(ok_files[0].name.endswith(".inspected.json"))
            ok_payload = json.loads(ok_files[0].read_text(encoding="utf-8"))
            self.assertEqual(ok_payload["status"], "inspected")
            self.assertEqual(ok_payload["pe"]["machine"], "AMD64")

            # 3) --require-routing failure -> exit 1, stderr names unsupported import, JSON written w/ 0 .tmp.
            pe_bad_route = _build_pe(
                imports=[("normal", "xmem.dll", "XMemAlloc"), ("normal", "mystery.dll", "MissingExport")]
            )
            cand_bad = base / "ok" / "bad_route.exe"
            cand_bad.write_bytes(pe_bad_route)
            rep_route = base / "ok" / "reports_route"
            with mock.patch.object(intake_exe, "scan", _fake_scan_for(pkg_ok, pe_bad_route)):
                code_rt, out_rt, err_rt = _run_main([
                    str(cand_bad),
                    "--build", str(pkg_ok),
                    "--baseline", str(base_ok),
                    "--integrity", str(int_ok),
                    "--reports", str(rep_route),
                    "--require-routing",
                ])
            self.assertEqual(code_rt, 1, (out_rt, err_rt))
            self.assertIn("unsupported: main.exe -> mystery.dll!MissingExport", err_rt)
            self.assertIn("inspected |", out_rt)
            rt_files = list(rep_route.iterdir())
            self.assertEqual(len(rt_files), 1)
            self.assertTrue(rt_files[0].name.endswith(".inspected.json"))

            # 4) error JSON (malformed MicrosoftGame.Config) -> exit 2, 0 .tmp files.
            (pkg_ok / "MicrosoftGame.Config").write_text("<broken", encoding="utf-8")
            rep_err = base / "ok" / "reports_err"
            code_err, out_err, err_err = _run_main([
                "--build", str(pkg_ok),
                "--baseline", str(base_ok),
                "--integrity", str(int_ok),
                "--reports", str(rep_err),
            ])
            self.assertEqual(code_err, 2, (out_err, err_err))
            self.assertNotIn("Traceback", err_err)
            self.assertIn("error |", out_err)
            err_files = list(rep_err.iterdir())
            self.assertEqual(len(err_files), 1)
            self.assertTrue(err_files[0].name.endswith(".error.json"))
            self.assertEqual(json.loads(err_files[0].read_text(encoding="utf-8"))["status"], "error")

    def test_sys_argv_and_script_entrypoint_and_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg, baseline, integrity = _synthetic_package(base / "entry", write_candidate=False)
            rep = base / "entry" / "reports"
            cli_argv = [
                str(TOOL),
                "--build", str(pkg),
                "--baseline", str(baseline),
                "--integrity", str(integrity),
                "--reports", str(rep),
            ]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertEqual(code, 3, (out, err))
            self.assertEqual(err, "")
            self.assertIn("awaiting_executable |", out)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 3)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("awaiting_executable |", stdout.getvalue())

            proc = _run_subprocess(*cli_argv[1:])
            self.assertEqual(proc.returncode, 3, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("awaiting_executable |", proc.stdout)


class IntakeExeCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr for {argv!r}")
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr for {argv!r}")
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_upfront_build_candidate_baseline_integrity_guards_before_collect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg, baseline, integrity = _synthetic_package(base / "pkg", write_candidate=False)
            rep_absent = base / "never_created_reports"
            regular_file = base / "regular.txt"
            regular_file.write_text("keep", encoding="utf-8")
            a_dir = base / "a_dir"
            a_dir.mkdir()
            missing = base / "missing.json"

            with mock.patch.object(intake_exe, "collect", side_effect=AssertionError("collect called upfront")):
                for label, argv, frag in (
                    ("missing_build", ["--build", str(base / "no_pkg"), "--reports", str(rep_absent)], "directory not found"),
                    ("file_build", ["--build", str(regular_file), "--reports", str(rep_absent)], "not a directory"),
                    (
                        "dir_candidate",
                        [str(a_dir), "--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(rep_absent)],
                        "candidate path is not a regular file",
                    ),
                    (
                        "missing_baseline",
                        ["--build", str(pkg), "--baseline", str(missing), "--integrity", str(integrity), "--reports", str(rep_absent)],
                        "--baseline file not found",
                    ),
                    (
                        "dir_baseline",
                        ["--build", str(pkg), "--baseline", str(a_dir), "--integrity", str(integrity), "--reports", str(rep_absent)],
                        "--baseline path is a directory",
                    ),
                    (
                        "missing_integrity",
                        ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(missing), "--reports", str(rep_absent)],
                        "--integrity file not found",
                    ),
                    (
                        "dir_integrity",
                        ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(a_dir), "--reports", str(rep_absent)],
                        "--integrity path is a directory",
                    ),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=1, fragments=(frag,))

            self.assertFalse(rep_absent.exists())
            self.assertEqual(regular_file.read_text(encoding="utf-8"), "keep")

    def test_reports_file_symlink_parent_file_and_inside_build_rejected_before_collect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg, baseline, integrity = _synthetic_package(base / "pkg", write_candidate=False)
            rep_file = base / "rep_file.txt"
            rep_file.write_text("keep-rep", encoding="utf-8")
            real_dir = base / "real_dir"
            real_dir.mkdir()
            symlink_rep = base / "symlink_rep"

            try:
                symlink_rep.symlink_to(real_dir, target_is_directory=True)
                has_symlinks = True
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                has_symlinks = False

            with mock.patch.object(intake_exe, "collect", side_effect=AssertionError("collect called for bad --reports")):
                self._assert_clean_failure(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(rep_file)],
                    expected_code=1,
                    fragments=("failed to write report", "regular file"),
                )
                self._assert_clean_failure(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(rep_file / "sub" / "out")],
                    expected_code=1,
                    fragments=("failed to write report", "not a directory"),
                )
                self._assert_clean_failure(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(pkg / "inside")],
                    expected_code=2,
                    fragments=("outside game files",),
                )
                if has_symlinks:
                    self._assert_clean_failure(
                        ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(symlink_rep)],
                        expected_code=1,
                        fragments=("failed to write report", "symlink"),
                    )
                else:
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == real_dir):
                        self._assert_clean_failure(
                            ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(real_dir)],
                            expected_code=1,
                            fragments=("failed to write report", "symlink"),
                            check_subprocess=False,
                        )
            self.assertEqual(rep_file.read_text(encoding="utf-8"), "keep-rep")
            self.assertEqual(list(real_dir.iterdir()), [])

    def test_report_replace_failure_leaves_zero_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pkg, baseline, integrity = _synthetic_package(base / "pkg", write_candidate=False)
            rep_dir = base / "rep_dir"
            rep_dir.mkdir()
            (rep_dir / "existing.txt").write_text("keep", encoding="utf-8")

            with mock.patch.object(intake_exe.os, "replace", side_effect=OSError("simulated replace fail")):
                self._assert_clean_failure(
                    ["--build", str(pkg), "--baseline", str(baseline), "--integrity", str(integrity), "--reports", str(rep_dir)],
                    expected_code=1,
                    fragments=("failed to write report", "simulated replace fail"),
                    check_subprocess=False,
                )
            self.assertEqual([p.name for p in rep_dir.iterdir()], ["existing.txt"])

    def test_argparse_errors_py314_safe(self) -> None:
        for argv, frags in (
            (["--bogus-flag"], ("unrecognized",)),
            (["--build"], ("expected one argument",)),
            (["--baseline"], ("expected one argument",)),
            (["--integrity"], ("expected one argument",)),
            (["--reports"], ("expected one argument",)),
        ):
            with self.subTest(argv=argv):
                self._assert_clean_failure(argv, expected_code=None, fragments=frags)


if __name__ == "__main__":
    unittest.main()
