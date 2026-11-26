#!/usr/bin/env python3
"""Pure-Python synthetic log/cache CLI, validation, and mutation suite for tools/startup/triage_run.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_STARTUP = ROOT / "tools" / "startup"
TOOL = TOOLS_STARTUP / "triage_run.py"
sys.path.insert(0, str(TOOLS_STARTUP))
import triage_run  # noqa: E402


SYNTHETIC_LOG = "\n".join([
    "x2w-d3d12 Device::slot70 unsupported memory SRV arguments: reason=unsupported_description "
    "res{dim=1 fmt=0 size=4096x1x1 mips=1 layout=0x1 flags=0x0 align=0 samples=1/0} "
    "view{dim=11 fmt=0 map=0x1688} dest=0x1 dest_in_view_heap=1 address=0x2 address_readable=0",
    "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed",
    "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x100 size=0x100 -> read failed",
    "x2w-runtime 12345678::slot5(0x1) -> 0x80004005 from +0x100",
    "x2w-runtime 12345678::slot6(0x1) -> 0x80070005 from +0x200",
    "x2w-runtime 12345678::slot7(0x1) -> 0x80070057 from +0x300",
]) + "\n"


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


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class StartupTriageRunCliPositiveTests(unittest.TestCase):
    def test_directory_logdir_with_shader_cache_and_max_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            (logdir / "nested").mkdir(parents=True)
            (logdir / "stderr.txt").write_text(SYNTHETIC_LOG, encoding="utf-8")
            # Mirrored trap.log line should be deduplicated against sibling stderr.txt.
            (logdir / "trap.log").write_text(
                "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x0 size=0x100 -> zlib failed\n",
                encoding="utf-8",
            )
            (logdir / "nested" / "extra.out").write_text(
                "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x200 size=0x100 -> bcpack failed\n",
                encoding="utf-8",
            )
            # Non-log extension must be ignored.
            (logdir / "ignored.json").write_text(
                "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x1000 source=0x1 offset=0x300 size=0x100 -> read failed\n",
                encoding="utf-8",
            )

            cache_dir = root / "shader_cache"
            cache_dir.mkdir()
            (cache_dir / "0000000000000001.fail").write_text("image_load_mip tfe\n", encoding="utf-8")
            (cache_dir / "0000000000000002.fail").write_text("image_load_mip tfe\n", encoding="utf-8")
            (cache_dir / "0000000000000003.fail").write_text("image_load_mip a16\n", encoding="utf-8")
            (cache_dir / "req_0000000000000003.c67").write_bytes(b"\0" * 4)

            argv = [str(logdir), "--shader-cache", str(cache_dir), "--max-items", "2"]
            code, out, err = _run_main(triage_run, argv)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("## Startup Run Triage Summary", out)
            self.assertIn("- Log files scanned: `3`", out)
            self.assertIn("- `X2W_SHADER_CACHE` `.fail` shaders: `3` (2 unique reasons)", out)
            self.assertIn("### `X2W_SHADER_CACHE` `.fail` Reasons", out)
            self.assertIn("| 2 | `image_load_mip tfe` |", out)
            self.assertIn("| 1 | `compute: image_load_mip a16` |", out)
            self.assertIn("_Showing top 2 of 3 unique entries (3 total events)._", out)

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(proc.stdout, out)

    def test_single_file_logdir_argv_none_and_main_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single.log"
            single.write_text(
                "x2w-dstorage   options=0x1 dest=0x1 dest_size=0x10 source=0x1 offset=0x0 size=0x10 -> read failed\n",
                encoding="utf-8",
            )

            with mock.patch.object(sys, "argv", [str(TOOL), str(single), "--max-items", "1"]):
                code, out, err = _run_main(triage_run, None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("- Log files scanned: `1`", out)
            self.assertIn("| 1 | read failed |", out)
            self.assertNotIn("X2W_SHADER_CACHE", out)

            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(single)]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr_m.getvalue(), "")
            self.assertIn("| 1 | read failed |", stdout_m.getvalue())


class StartupTriageRunCliNegativeTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(triage_run, argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
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
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_missing_and_non_regular_logdir_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_log = root / "missing_logs"
            non_regular = root / "special_entry"

            with mock.patch.object(
                triage_run,
                "triage_run",
                side_effect=AssertionError("triage_run called for invalid logdir"),
            ):
                self._assert_clean_failure(
                    [str(missing_log)],
                    expected_code=1,
                    fragments=("log path not found",),
                )

                orig_exists = Path.exists
                orig_is_file = Path.is_file
                orig_is_dir = Path.is_dir

                def fake_exists(p: Path) -> bool:
                    return True if p == non_regular else orig_exists(p)

                def fake_is_file(p: Path) -> bool:
                    return False if p == non_regular else orig_is_file(p)

                def fake_is_dir(p: Path) -> bool:
                    return False if p == non_regular else orig_is_dir(p)

                with (
                    mock.patch.object(Path, "exists", fake_exists),
                    mock.patch.object(Path, "is_file", fake_is_file),
                    mock.patch.object(Path, "is_dir", fake_is_dir),
                ):
                    self._assert_clean_failure(
                        [str(non_regular)],
                        expected_code=1,
                        fragments=("regular file or directory",),
                        check_subprocess=False,
                    )

    def test_missing_and_file_shader_cache_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()
            (logdir / "stderr.txt").write_text("", encoding="utf-8")
            missing_cache = root / "missing_cache"
            file_cache = root / "not_a_dir.txt"
            file_cache.write_text("file contents", encoding="utf-8")

            with mock.patch.object(
                triage_run,
                "triage_run",
                side_effect=AssertionError("triage_run called for invalid shader_cache"),
            ):
                self._assert_clean_failure(
                    [str(logdir), "--shader-cache", str(missing_cache)],
                    expected_code=1,
                    fragments=("shader cache directory not found",),
                )
                self._assert_clean_failure(
                    [str(logdir), "--shader-cache", str(file_cache)],
                    expected_code=1,
                    fragments=("shader cache path is not a directory",),
                )

    def test_non_positive_max_items_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()

            with mock.patch.object(
                triage_run,
                "triage_run",
                side_effect=AssertionError("triage_run called for invalid --max-items"),
            ):
                for bad_value in ("0", "-1", "-100"):
                    with self.subTest(max_items=bad_value):
                        self._assert_clean_failure(
                            [str(logdir), "--max-items", bad_value],
                            expected_code=1,
                            fragments=("--max-items must be at least 1", bad_value),
                        )

    def test_runtime_oserror_valueerror_and_main_exit_code_1_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()

            for exc in (OSError("simulated I/O failure"), ValueError("simulated parse failure")):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(triage_run, "triage_run", side_effect=exc):
                        self._assert_clean_failure(
                            [str(logdir)],
                            expected_code=1,
                            fragments=(str(exc),),
                            check_subprocess=False,
                        )

            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(root / "absent")]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout_m.getvalue(), "")
            self.assertNotIn("Traceback", stderr_m.getvalue())
            self.assertIn("log path not found", stderr_m.getvalue().lower())

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logdir = root / "logs"
            logdir.mkdir()

            for argv, frags in (
                ([], ("required",)),
                ([str(logdir), "--shader-cache"], ("expected one argument",)),
                ([str(logdir), "--max-items"], ("expected one argument",)),
                ([str(logdir), "--max-items", "abc"], ("invalid",)),
                ([str(logdir), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_logdir = root / "logs"
            valid_logdir.mkdir()
            (valid_logdir / "stderr.txt").write_text("", encoding="utf-8")
            missing_log = root / "missing_logs"
            non_regular = root / "special_entry"
            missing_cache = root / "missing_cache"
            file_cache = root / "file_cache.txt"
            file_cache.write_text("not a dir", encoding="utf-8")

            mutations = [
                (
                    "drop_logdir_exists_guard",
                    "    if not args.logdir.exists():\n",
                    "    if False:\n",
                    [str(missing_log)],
                    False,
                    "log path not found",
                ),
                (
                    "drop_logdir_regular_or_dir_guard",
                    "    if not (args.logdir.is_file() or args.logdir.is_dir()):\n",
                    "    if False:\n",
                    [str(non_regular)],
                    True,
                    None,
                ),
                (
                    "drop_shader_cache_exists_guard",
                    "        if not args.shader_cache.exists():\n",
                    "        if False:\n",
                    [str(valid_logdir), "--shader-cache", str(missing_cache)],
                    False,
                    "shader cache directory not found",
                ),
                (
                    "drop_shader_cache_is_dir_guard",
                    "        if not args.shader_cache.is_dir():\n",
                    "        if False:\n",
                    [str(valid_logdir), "--shader-cache", str(file_cache)],
                    False,
                    None,
                ),
                (
                    "allow_zero_max_items",
                    "    if args.max_items < 1:\n",
                    "    if args.max_items < 0:\n",
                    [str(valid_logdir), "--max-items", "0"],
                    False,
                    None,
                ),
            ]

            for label, target, replacement, argv, mock_non_regular, broken_fragment in mutations:
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mutated = types.ModuleType(f"triage_run_mut_{label}")
                    mutated.__file__ = str(TOOL)
                    sys.modules[mutated.__name__] = mutated
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mutated.__dict__)

                        if mock_non_regular:
                            orig_exists = Path.exists
                            orig_is_file = Path.is_file
                            orig_is_dir = Path.is_dir
                            with (
                                mock.patch.object(Path, "exists", lambda p: True if p == non_regular else orig_exists(p)),
                                mock.patch.object(Path, "is_file", lambda p: False if p == non_regular else orig_is_file(p)),
                                mock.patch.object(Path, "is_dir", lambda p: False if p == non_regular else orig_is_dir(p)),
                            ):
                                code, _, err = _run_main(mutated, argv)
                        else:
                            code, _, err = _run_main(mutated, argv)
                    finally:
                        sys.modules.pop(mutated.__name__, None)

                    if broken_fragment is None:
                        self.assertEqual(code, 0, f"mutation {label} unexpectedly failed: {err}")
                    else:
                        self.assertNotIn(broken_fragment, err.lower(), f"mutation {label} still emitted {broken_fragment!r}")


if __name__ == "__main__":
    unittest.main()
