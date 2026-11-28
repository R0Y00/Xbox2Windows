#!/usr/bin/env python3
"""Offline CLI, atomic-write, path-guard, error-handling, and mutation suite for tools/startup/ui_regression_matrix.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_STARTUP = ROOT / "tools" / "startup"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_STARTUP / "ui_regression_matrix.py"
sys.path.insert(0, str(TOOLS_STARTUP))
sys.path.insert(0, str(TESTS_DIR))

import startup_ui_regression_matrix_check as matrix_fixture  # noqa: E402
import ui_regression_matrix as matrix  # noqa: E402

_REAL_SUBPROCESS_RUN = subprocess.run


def _run_main(module, argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return _REAL_SUBPROCESS_RUN(
        [sys.executable, "-B", str(TOOL), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def _try_symlink(link: Path, target: Path) -> bool:
    try:
        link.symlink_to(target)
        return True
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        return False


class _MatrixWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.build = root / "build"
        self.build.mkdir()
        self.fake = root / "fake_ctest.py"
        self.fake.write_text(matrix_fixture.FAKE_CTEST, encoding="utf-8")
        self.serial = 0

    def env_for(self, scenario: dict[str, object]) -> tuple[dict[str, str], Path]:
        self.serial += 1
        stem = f"case-{self.serial}"
        scenario_path = self.root / f"{stem}.scenario.json"
        invocations = self.root / f"{stem}.invocations.jsonl"
        scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
        env = dict(
            os.environ,
            UI_MATRIX_SCENARIO=str(scenario_path),
            UI_MATRIX_INVOCATIONS=str(invocations),
        )
        return env, invocations

    def base_argv(self, *extra: str, build_dir: Path | None = None) -> list[str]:
        return [
            "--build-dir",
            str(build_dir if build_dir is not None else self.build),
            "--ctest",
            str(self.fake),
            *extra,
        ]


class UiRegressionMatrixCliPositiveTests(unittest.TestCase):
    def test_atomic_nested_markdown_and_json_writes_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _MatrixWorkspace(Path(tmp))
            env, invocations = ws.env_for(matrix_fixture.FULL_SCENARIO)
            md_out = ws.root / "nested" / "md" / "matrix.md"
            json_out = ws.root / "nested" / "json" / "matrix.json"
            self.assertFalse(md_out.parent.exists())
            self.assertFalse(json_out.parent.exists())

            replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
                src_path, dst_path = Path(src), Path(dst)
                replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.resolve(), dst_path.parent.resolve())
                self.assertTrue(src_path.name.endswith(".tmp"))
                self.assertTrue(src_path.is_file())
                real_replace(src, dst)

            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(matrix.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main(
                    matrix,
                    ws.base_argv("--output", str(md_out), "--json", str(json_out)),
                )

            self.assertIs(type(code), int)
            self.assertEqual(code, 1, err)
            self.assertEqual(err, "")
            self.assertIn(f"wrote {md_out}", out)
            self.assertIn(f"wrote {json_out}", out)
            self.assertEqual(len(replaces), 2)
            self.assertEqual([p.name for p in md_out.parent.iterdir()], ["matrix.md"])
            self.assertEqual([p.name for p in json_out.parent.iterdir()], ["matrix.json"])
            self.assertEqual(len(invocations.read_text(encoding="utf-8").splitlines()), 2)

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "xbox2windows-startup-ui-regression-matrix-v1")
            self.assertEqual(payload["summary"], {"selected": 8, "passed": 3, "skipped": 4, "failed": 1})
            md_text = md_out.read_text(encoding="utf-8")
            self.assertIn("# Startup UI regression matrix", md_text)
            self.assertIn("| PASS |", md_text)
            self.assertIn("| SKIP |", md_text)
            self.assertIn("| FAIL |", md_text)

            # Atomic overwrite of existing reports leaves zero .tmp files and updates outputs.
            pass_env, _ = ws.env_for({"tests": [matrix_fixture.test_case("startup_rect_gap_warp", "pass")]})
            with (
                mock.patch.dict(os.environ, pass_env, clear=True),
                mock.patch.object(matrix.os, "replace", side_effect=tracking_replace),
            ):
                code_ok, _, err_ok = _run_main(
                    matrix,
                    ws.base_argv("--output", str(md_out), "--json", str(json_out), "--filter", "^startup_rect_gap_warp$"),
                )
            self.assertEqual(code_ok, 0, err_ok)
            self.assertEqual(len(replaces), 4)
            self.assertEqual([p.name for p in md_out.parent.iterdir()], ["matrix.md"])
            self.assertEqual([p.name for p in json_out.parent.iterdir()], ["matrix.json"])
            self.assertEqual(
                json.loads(json_out.read_text(encoding="utf-8"))["summary"],
                {"selected": 1, "passed": 1, "skipped": 0, "failed": 0},
            )

    def test_stdout_mode_sys_argv_entrypoint_and_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _MatrixWorkspace(Path(tmp))
            skip_scenario = {
                "tests": [
                    matrix_fixture.test_case("startup_rect_gap_warp", "pass"),
                    matrix_fixture.test_case(
                        "startup_rect_gap_hardware",
                        "skip",
                        "SKIP_REGULAR_EXPRESSION: adapter missing",
                    ),
                ]
            }
            env, _ = ws.env_for(skip_scenario)
            json_only = ws.root / "only" / "report.json"

            # Omitting --output prints Markdown to stdout while atomically writing --json.
            with mock.patch.dict(os.environ, env, clear=True):
                code, out, err = _run_main(matrix, ws.base_argv("--json", str(json_only)))
            self.assertEqual(code, 3, err)
            self.assertEqual(err, "")
            self.assertIn("# Startup UI regression matrix", out)
            self.assertIn(f"wrote {json_only}", out)
            self.assertEqual([p.name for p in json_only.parent.iterdir()], ["report.json"])
            self.assertEqual(
                json.loads(json_only.read_text(encoding="utf-8"))["summary"],
                {"selected": 2, "passed": 1, "skipped": 1, "failed": 0},
            )

            # main(None) and __main__ entrypoint via subprocess & runpy.
            pass_env, _ = ws.env_for({"tests": [matrix_fixture.test_case("startup_rect_gap_warp", "pass")]})
            sub_md = ws.root / "sub" / "out.md"
            sub_json = ws.root / "sub" / "out.json"
            cli_argv = ws.base_argv("--output", str(sub_md), "--json", str(sub_json))
            with (
                mock.patch.dict(os.environ, pass_env, clear=True),
                mock.patch.object(sys, "argv", [str(TOOL), *cli_argv]),
            ):
                code_none, _, err_none = _run_main(matrix, None)
            self.assertEqual(code_none, 0, err_none)
            self.assertTrue(sub_md.is_file() and sub_json.is_file())

            sub_md.unlink()
            sub_json.unlink()
            out_buf, err_buf = io.StringIO(), io.StringIO()
            with (
                mock.patch.dict(os.environ, pass_env, clear=True),
                mock.patch.object(sys, "argv", [str(TOOL), *cli_argv]),
                redirect_stdout(out_buf),
                redirect_stderr(err_buf),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(err_buf.getvalue(), "")
            self.assertEqual(sorted(p.name for p in sub_md.parent.iterdir()), ["out.json", "out.md"])

            proc = _run_subprocess(*cli_argv, env=pass_env)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("1 passed, 0 skipped, 0 failed", proc.stdout)


class UiRegressionMatrixCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(
        self,
        argv: list[str],
        *,
        expected_fragments: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        check_subprocess: bool = False,
    ) -> str:
        code, out, err = _run_main(matrix, argv)
        self.assertEqual(code, 2, (code, out, err))
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        for fragment in expected_fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv, env=env)
            self.assertEqual(proc.returncode, 2, (proc.returncode, proc.stdout, proc.stderr))
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())
            for fragment in expected_fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_hostile_output_json_and_build_dir_rejected_upfront_before_ctest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _MatrixWorkspace(Path(tmp))
            existing_dir = ws.root / "existing_dir"
            existing_dir.mkdir()
            regular_file = ws.root / "regular_file"
            regular_file.write_text("KEEP_ME", encoding="utf-8")
            other_file = ws.root / "other_file"
            other_file.write_text("KEEP_OTHER", encoding="utf-8")
            hardlink_json = ws.root / "hardlink.json"
            os.link(regular_file, hardlink_json)
            self.assertTrue(regular_file.samefile(hardlink_json))
            self.assertNotEqual(regular_file.resolve(), hardlink_json.resolve())

            with (
                mock.patch.object(
                    matrix,
                    "selected_test_names",
                    side_effect=AssertionError("selected_test_names invoked despite invalid upfront path"),
                ),
                mock.patch.object(
                    matrix,
                    "run_tests",
                    side_effect=AssertionError("run_tests invoked despite invalid upfront path"),
                ),
            ):
                cases = (
                    ("output_dir", ws.base_argv("--output", str(existing_dir)), ("--output", "directory"), True),
                    ("json_dir", ws.base_argv("--json", str(existing_dir)), ("--json", "directory"), True),
                    ("output_empty", ws.base_argv("--output", ""), ("--output", "directory"), False),
                    ("json_empty", ws.base_argv("--json", ""), ("--json", "directory"), False),
                    (
                        "output_parent_file",
                        ws.base_argv("--output", str(regular_file / "sub" / "report.md")),
                        ("--output", "not a directory"),
                        True,
                    ),
                    (
                        "json_parent_file",
                        ws.base_argv("--json", str(regular_file / "report.json")),
                        ("--json", "not a directory"),
                        True,
                    ),
                    (
                        "same_resolve",
                        ws.base_argv(
                            "--output",
                            str(ws.root / "no_such" / ".." / "shared.out"),
                            "--json",
                            str(ws.root / "shared.out"),
                        ),
                        ("--output", "--json", "same path"),
                        True,
                    ),
                    (
                        "same_hardlink",
                        ws.base_argv("--output", str(regular_file), "--json", str(hardlink_json)),
                        ("--output", "--json", "same path"),
                        True,
                    ),
                    (
                        "missing_build_dir",
                        ws.base_argv(build_dir=ws.root / "missing_build"),
                        ("build directory does not exist",),
                        True,
                    ),
                    (
                        "file_build_dir",
                        ws.base_argv(build_dir=regular_file),
                        ("build directory does not exist",),
                        True,
                    ),
                )
                for label, argv, frags, subproc in cases:
                    with self.subTest(case=label):
                        self._assert_exit_2_clean(argv, expected_fragments=frags, check_subprocess=subproc)

                # Symlinked --output, --json, dangling parent, and --build-dir (real symlinks or mocked fallback).
                live_file_link = ws.root / "live_file_link.md"
                dangling_link = ws.root / "dangling_link.json"
                build_link = ws.root / "build_link"
                dangling_build = ws.root / "dangling_build"
                if _try_symlink(live_file_link, other_file):
                    _try_symlink(dangling_link, ws.root / "missing_target")
                    _try_symlink(build_link, ws.build)
                    _try_symlink(dangling_build, ws.root / "missing_build_target")
                    symlink_cases = (
                        ("output_live_symlink", ws.base_argv("--output", str(live_file_link)), ("--output", "symlink")),
                        ("json_dangling_symlink", ws.base_argv("--json", str(dangling_link)), ("--json", "symlink")),
                        (
                            "output_dangling_parent",
                            ws.base_argv("--output", str(dangling_link / "nested.md")),
                            ("--output", "not a directory"),
                        ),
                        ("build_dir_live_symlink", ws.base_argv(build_dir=build_link), ("build directory", "symlink")),
                        (
                            "build_dir_dangling_symlink",
                            ws.base_argv(build_dir=dangling_build),
                            ("build directory", "symlink"),
                        ),
                    )
                    for label, argv, frags in symlink_cases:
                        with self.subTest(case=label):
                            self._assert_exit_2_clean(argv, expected_fragments=frags, check_subprocess=True)
                else:
                    target_out = ws.root / "mock_out.md"
                    target_json = ws.root / "mock_out.json"
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == target_out):
                        self._assert_exit_2_clean(
                            ws.base_argv("--output", str(target_out)),
                            expected_fragments=("--output", "symlink"),
                        )
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == target_json):
                        self._assert_exit_2_clean(
                            ws.base_argv("--json", str(target_json)),
                            expected_fragments=("--json", "symlink"),
                        )
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == ws.build):
                        self._assert_exit_2_clean(
                            ws.base_argv(),
                            expected_fragments=("build directory", "symlink"),
                        )

                # Non-regular existing file (FIFO if available, plus unconditional Path.is_file mock).
                if hasattr(os, "mkfifo"):
                    fifo = ws.root / "report.fifo"
                    os.mkfifo(fifo)
                    self._assert_exit_2_clean(
                        ws.base_argv("--output", str(fifo)),
                        expected_fragments=("--output", "not a regular file"),
                        check_subprocess=True,
                    )
                with mock.patch.object(
                    Path,
                    "is_file",
                    autospec=True,
                    side_effect=lambda p: p != regular_file,
                ):
                    self._assert_exit_2_clean(
                        ws.base_argv("--json", str(regular_file)),
                        expected_fragments=("--json", "not a regular file"),
                    )

            self.assertEqual(regular_file.read_text(encoding="utf-8"), "KEEP_ME")
            self.assertEqual(other_file.read_text(encoding="utf-8"), "KEEP_OTHER")

    def test_os_replace_and_staging_failures_preserve_prior_json_and_leave_zero_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = _MatrixWorkspace(Path(tmp))
            env, _ = ws.env_for({"tests": [matrix_fixture.test_case("startup_rect_gap_warp", "pass")]})
            out_dir = ws.root / "reports"
            out_dir.mkdir()
            json_out = out_dir / "matrix.json"
            prior_json = '{"prior": true}\n'
            json_out.write_text(prior_json, encoding="utf-8")

            def failing_replace(src: str | os.PathLike[str], _dst: str | os.PathLike[str]) -> None:
                self.assertTrue(Path(src).is_file())
                raise OSError("simulated replace failure")

            # 1) os.replace fails on pre-existing --json -> prior intact, 0 temps, exit 2 cleanly.
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(matrix.os, "replace", side_effect=failing_replace),
            ):
                self._assert_exit_2_clean(
                    ws.base_argv("--json", str(json_out)),
                    expected_fragments=("simulated replace failure",),
                )
            self.assertEqual(json_out.read_text(encoding="utf-8"), prior_json)
            self.assertEqual([p.name for p in out_dir.iterdir()], ["matrix.json"])

            # 2) os.replace fails on fresh nested --output / --json -> 0 outputs & 0 temps, exit 2.
            fresh_md = ws.root / "fresh" / "matrix.md"
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(matrix.os, "replace", side_effect=failing_replace),
            ):
                self._assert_exit_2_clean(
                    ws.base_argv("--output", str(fresh_md), "--json", str(json_out)),
                    expected_fragments=("simulated replace failure",),
                )
            self.assertEqual(list(fresh_md.parent.iterdir()), [])
            self.assertEqual(json_out.read_text(encoding="utf-8"), prior_json)

            # 3) Staging write failure -> os.replace never called, prior JSON intact, 0 temps, exit 2.
            real_ntf = matrix.tempfile.NamedTemporaryFile

            class FailingWriteHandle:
                def __init__(self, inner) -> None:
                    self._inner = inner
                    self.name = inner.name

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return self._inner.__exit__(exc_type, exc, tb)

                def write(self, _data: str) -> int:
                    raise OSError("simulated staging write failure")

            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(matrix.os, "replace") as replace_mock,
                mock.patch.object(
                    matrix.tempfile,
                    "NamedTemporaryFile",
                    side_effect=lambda *a, **kw: FailingWriteHandle(real_ntf(*a, **kw)),
                ),
            ):
                self._assert_exit_2_clean(
                    ws.base_argv("--json", str(json_out)),
                    expected_fragments=("simulated staging write failure",),
                )
            replace_mock.assert_not_called()
            self.assertEqual(json_out.read_text(encoding="utf-8"), prior_json)
            self.assertEqual([p.name for p in out_dir.iterdir()], ["matrix.json"])

            # 4) mkdir OSError during report write -> caught cleanly as EXIT_USAGE (2).
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(matrix.Path, "mkdir", side_effect=PermissionError("mkdir denied")),
            ):
                self._assert_exit_2_clean(
                    ws.base_argv("--json", str(ws.root / "blocked" / "matrix.json")),
                    expected_fragments=("mkdir denied",),
                )

    def test_fast_self_mutation_controls(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            ws = _MatrixWorkspace(Path(tmp))
            env, _ = ws.env_for({"tests": [matrix_fixture.test_case("startup_rect_gap_warp", "pass")]})
            regular_file = ws.root / "regular_file"
            regular_file.write_text("KEEP", encoding="utf-8")
            existing_dir = ws.root / "existing_dir"
            existing_dir.mkdir()
            hardlink_json = ws.root / "hardlink.json"
            os.link(regular_file, hardlink_json)

            def check_drop_build_symlink(mod) -> None:
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == ws.build),
                ):
                    code, _, err = _run_main(mod, ws.base_argv())
                self.assertEqual(code, 0, err)

            def check_drop_report_symlink(mod) -> None:
                target = ws.root / "mut_sym.md"
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == target),
                ):
                    code, _, err = _run_main(mod, ws.base_argv("--output", str(target)))
                self.assertEqual(code, 0, err)

            def check_drop_report_dir(mod) -> None:
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(mod, "selected_test_names", return_value=["startup_rect_gap_warp"]),
                    mock.patch.object(
                        mod,
                        "run_tests",
                        side_effect=AssertionError("sentinel: reached CTest past dir guard"),
                    ),
                ):
                    code, _, err = _run_main(mod, ws.base_argv("--output", str(existing_dir)))
                self.assertEqual(code, 2)
                self.assertNotIn("is a directory", err.lower())
                self.assertIn("not a regular file", err.lower())

            def check_drop_report_parent(mod) -> None:
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(mod, "selected_test_names", return_value=["startup_rect_gap_warp"]),
                    mock.patch.object(
                        mod,
                        "run_tests",
                        side_effect=AssertionError("sentinel: reached CTest past parent guard"),
                    ),
                ):
                    with self.assertRaisesRegex(AssertionError, "reached CTest past parent guard"):
                        _run_main(mod, ws.base_argv("--output", str(regular_file / "sub" / "out.md")))

            def check_drop_samefile(mod) -> None:
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(mod, "selected_test_names", return_value=["startup_rect_gap_warp"]),
                    mock.patch.object(
                        mod,
                        "run_tests",
                        side_effect=AssertionError("sentinel: reached CTest past samefile guard"),
                    ),
                ):
                    with self.assertRaisesRegex(AssertionError, "reached CTest past samefile guard"):
                        _run_main(mod, ws.base_argv("--output", str(regular_file), "--json", str(hardlink_json)))

            def check_drop_temp_unlink(mod) -> None:
                leak_dir = ws.root / "leak_dir"
                leak_dir.mkdir()
                leak_json = leak_dir / "matrix.json"
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(mod.os, "replace", side_effect=OSError("boom")),
                ):
                    code, _, _ = _run_main(mod, ws.base_argv("--json", str(leak_json)))
                self.assertEqual(code, 2)
                leftovers = [p.name for p in leak_dir.iterdir() if p.name.endswith(".tmp")]
                self.assertTrue(leftovers, "expected leaked .tmp file when unlink guard is removed")

            def check_drop_write_oserror_catch(mod) -> None:
                with (
                    mock.patch.dict(os.environ, env, clear=True),
                    mock.patch.object(mod.os, "replace", side_effect=OSError("uncaught replace boom")),
                ):
                    with self.assertRaisesRegex(OSError, "uncaught replace boom"):
                        _run_main(mod, ws.base_argv("--json", str(ws.root / "uncaught.json")))

            mutations = (
                (
                    "drop_build_dir_symlink_guard",
                    r"        if raw_build_dir\.is_symlink\(\):\n"
                    r"            parser\.error\([^\n]*\)\n",
                    "        pass\n",
                    check_drop_build_symlink,
                ),
                (
                    "drop_report_symlink_guard",
                    r"    if path\.is_symlink\(\):\n"
                    r"        raise MatrixError\([^\n]*\)\n",
                    "    pass\n",
                    check_drop_report_symlink,
                ),
                (
                    "drop_report_dir_guard",
                    r"    if path\.is_dir\(\):\n"
                    r"        raise MatrixError\([^\n]*\)\n",
                    "    pass\n",
                    check_drop_report_dir,
                ),
                (
                    "drop_report_parent_guard",
                    r"    for parent in path\.parents:\n"
                    r"        if parent\.is_file\(\) or \(\(parent\.exists\(\) or parent\.is_symlink\(\)\) and not parent\.is_dir\(\)\):\n"
                    r"            raise MatrixError\([^\n]*\)\n",
                    "    pass\n",
                    check_drop_report_parent,
                ),
                (
                    "drop_same_path_samefile",
                    r"    return left\.exists\(\) and right\.exists\(\) and left\.samefile\(right\)\n",
                    "    return False\n",
                    check_drop_samefile,
                ),
                (
                    "drop_temp_unlink",
                    r"        if tmp_path is not None:\n"
                    r"            tmp_path\.unlink\(missing_ok=True\)\n",
                    "        pass\n",
                    check_drop_temp_unlink,
                ),
                (
                    "drop_write_oserror_catch",
                    r"    try:\n"
                    r"        if args\.output is not None:\n"
                    r"            _write\(args\.output, markdown\)\n"
                    r"        if args\.json is not None:\n"
                    r"            _write\(args\.json, json\.dumps\(payload, indent=2\) \+ \"\\n\"\)\n"
                    r"    except OSError as error:\n"
                    r"        print\(f\"ui_regression_matrix: error: \{error\}\", file=sys\.stderr\)\n"
                    r"        return EXIT_USAGE\n",
                    "    if args.output is not None:\n"
                    "        _write(args.output, markdown)\n"
                    "    if args.json is not None:\n"
                    '        _write(args.json, json.dumps(payload, indent=2) + "\\n")\n',
                    check_drop_write_oserror_catch,
                ),
            )
            for label, pattern, replacement, check_fn in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(len(re.findall(pattern, source)), 1, f"target not unique for {label}")
                    mod = types.ModuleType(f"ui_regression_matrix_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(
                            compile(re.sub(pattern, lambda _m, rep=replacement: rep, source, count=1), str(TOOL), "exec"),
                            mod.__dict__,
                        )
                        check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
