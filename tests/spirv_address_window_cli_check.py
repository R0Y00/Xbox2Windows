#!/usr/bin/env python3
"""Pure-Python mocked-subprocess CLI, atomic-write, path-guard, and cleanup suite for tools/spirv_address_window.py."""

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "spirv_address_window.py"
FIXTURES = ROOT / "tests" / "spirv_window"
sys.path.insert(0, str(TOOLS_DIR))
import spirv_address_window as saw  # noqa: E402


SUBTRACT_ASM = (FIXTURES / "subtract.spvasm").read_text(encoding="utf-8")
_REAL_SUBPROCESS_RUN = subprocess.run


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    import io

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = saw.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return _REAL_SUBPROCESS_RUN(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class SpirvAddressWindowCliPositiveTests(unittest.TestCase):
    def test_atomic_spv_in_nested_dir_overwrite_and_zero_temp_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            nested_out = work / "nested" / "deep" / "window.spv"

            calls: list[list[str]] = []
            seen_asm: dict[str, str] = {}
            created_tempdirs: list[Path] = []
            replaced_pairs: list[tuple[Path, Path]] = []
            real_tempdir = tempfile.TemporaryDirectory
            real_replace = Path.replace
            payload_counter = [0]

            class TrackingTempDir(real_tempdir):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    created_tempdirs.append(Path(self.name))

            def tracking_replace(self_path: Path, target: Path):
                replaced_pairs.append((self_path, Path(target)))
                self.assertEqual(self_path.parent.parent.resolve(), Path(target).parent.resolve())
                return real_replace(self_path, target)

            def fake_run(cmd, check=False, **_kwargs):
                self.assertTrue(check)
                calls.append([str(part) for part in cmd])
                if cmd[0] == "mock-as":
                    asm_path = Path(cmd[3])
                    seen_asm[asm_path.name] = asm_path.read_text(encoding="utf-8")
                    out_idx = cmd.index("-o") + 1
                    payload_counter[0] += 1
                    Path(cmd[out_idx]).write_bytes(f"SPV_V{payload_counter[0]}".encode("ascii"))
                return subprocess.CompletedProcess(cmd, 0)

            with (
                mock.patch.object(saw.subprocess, "run", side_effect=fake_run),
                mock.patch.object(saw.tempfile, "TemporaryDirectory", TrackingTempDir),
                mock.patch.object(Path, "replace", tracking_replace),
            ):
                # 1) Nested missing parent directories, atomic replace, 0 leftover temp dirs.
                code, out, err = _run_main([
                    str(src),
                    str(nested_out),
                    "--assembler",
                    "  mock-as  ",
                    "--validator",
                    "  mock-val  ",
                ])

                self.assertIs(type(code), int)
                self.assertEqual(code, 0, err)
                self.assertEqual(out, "")
                self.assertEqual(err, "")
                self.assertEqual(nested_out.read_bytes(), b"SPV_V2")
                self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["window.spv"])
                self.assertEqual(len(created_tempdirs), 1)
                self.assertFalse(created_tempdirs[0].exists())
                self.assertEqual(len(replaced_pairs), 1)
                self.assertEqual(replaced_pairs[0][1].resolve(), nested_out.resolve())

                self.assertEqual(len(calls), 4)
                self.assertEqual(calls[0][:3], ["mock-as", "--target-env", "vulkan1.2"])
                self.assertEqual(Path(calls[0][3]).name, "original.spvasm")
                self.assertEqual(calls[1][:3], ["mock-val", "--target-env", "vulkan1.2"])
                self.assertEqual(calls[2][:3], ["mock-as", "--target-env", "vulkan1.2"])
                self.assertEqual(Path(calls[2][3]).name, "logical.spvasm")
                self.assertEqual(calls[3][:3], ["mock-val", "--target-env", "vulkan1.2"])
                self.assertEqual(seen_asm["original.spvasm"], SUBTRACT_ASM)
                self.assertIn("OpMemoryModel Logical GLSL450", seen_asm["logical.spvasm"])
                self.assertNotIn("PhysicalStorageBuffer", seen_asm["logical.spvasm"])

                # 2) Atomic overwrite on existing file leaves 0 temp dirs.
                calls.clear()
                code_ow, out_ow, err_ow = _run_main([
                    str(src),
                    str(nested_out),
                    "--assembler",
                    "mock-as",
                    "--validator",
                    "mock-val",
                ])
                self.assertEqual(code_ow, 0, err_ow)
                self.assertEqual(out_ow, "")
                self.assertEqual(err_ow, "")
                self.assertEqual(nested_out.read_bytes(), b"SPV_V4")
                self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["window.spv"])
                self.assertEqual(len(created_tempdirs), 2)
                self.assertFalse(created_tempdirs[1].exists())

                # 3) argv=None forwards sys.argv cleanly.
                with mock.patch.object(
                    sys,
                    "argv",
                    [str(TOOL), str(src), str(nested_out), "--assembler", "mock-as", "--validator", "mock-val"],
                ):
                    code_none, out_none, err_none = _run_main(None)
                self.assertEqual(code_none, 0, err_none)
                self.assertEqual(out_none, "")
                self.assertEqual(err_none, "")
                self.assertEqual(nested_out.read_bytes(), b"SPV_V6")
                self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["window.spv"])
                self.assertFalse(created_tempdirs[2].exists())

            # 4) __main__ entrypoint exits cleanly on invalid source without raw traceback.
            import io

            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(work / "absent.spvasm"), str(work / "out.spv")]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout_m.getvalue(), "")
            self.assertNotIn("Traceback", stderr_m.getvalue())
            self.assertIn("source file not found", stderr_m.getvalue().lower())


class SpirvAddressWindowCliNegativeTests(unittest.TestCase):
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

    def test_missing_dir_and_bad_utf8_source_rejected_without_creating_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            dir_src = work / "dir_source"
            dir_src.mkdir()
            bad_utf8 = work / "bad_utf8.spvasm"
            bad_utf8.write_bytes(b"\xff\xfe\x00\x80")
            absent_out = work / "never_created" / "window.spv"

            with mock.patch.object(
                saw.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run called for invalid source"),
            ):
                for label, src, frags in (
                    ("missing_source", work / "missing.spvasm", ("source file not found",)),
                    ("directory_source", dir_src, ("directory",)),
                    ("bad_utf8_source", bad_utf8, ("decode",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(src), str(absent_out), "--assembler", "mock-as", "--validator", "mock-val"],
                            expected_code=1,
                            fragments=frags,
                        )
            self.assertFalse(absent_out.parent.exists())

    def test_output_guards_dir_file_parent_and_samefile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            out_dir = work / "existing_dir.spv"
            out_dir.mkdir()
            file_parent = work / "file_parent"
            file_parent.write_bytes(b"keep-parent-bytes")
            hardlink = work / "subtract_hardlink.spv"
            hardlink.hardlink_to(src)
            missing_src = work / "missing.spvasm"

            with mock.patch.object(
                saw.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run called for invalid output"),
            ):
                for label, argv, frags in (
                    ("output_is_dir", [str(src), str(out_dir)], ("directory",)),
                    ("parent_is_file", [str(src), str(file_parent / "window.spv")], ("not a directory",)),
                    ("ancestor_is_file", [str(src), str(file_parent / "sub" / "window.spv")], ("not a directory",)),
                    ("same_exact_path", [str(src), str(src)], ("separate input assembly and output binary",)),
                    (
                        "same_unnormalized_path",
                        [str(src), str(work / "nested" / ".." / "subtract.spvasm")],
                        ("separate input assembly and output binary",),
                    ),
                    (
                        "same_missing_alias",
                        [str(missing_src), str(work / "nested" / ".." / "missing.spvasm")],
                        ("separate input assembly and output binary",),
                    ),
                    ("hardlink_samefile", [str(src), str(hardlink)], ("separate input assembly and output binary",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=1, fragments=frags)

            self.assertEqual(src.read_text(encoding="utf-8"), SUBTRACT_ASM)
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertEqual(file_parent.read_bytes(), b"keep-parent-bytes")

    def test_output_symlink_rejected_before_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            out_dir = work / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.spv"
            real_target.write_bytes(b"UNTOUCHED")
            live_link = out_dir / "live_link.spv"
            dangling_link = out_dir / "dangling_link.spv"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.spv")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with mock.patch.object(
                saw.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run called for symlink output"),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(src), str(link), "--assembler", "mock-as", "--validator", "mock-val"],
                            expected_code=1,
                            fragments=("symlink",),
                        )

            self.assertEqual(real_target.read_bytes(), b"UNTOUCHED")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.spv", "live_link.spv", "real.spv"],
            )

    def test_empty_and_whitespace_assembler_and_validator_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            absent_out = work / "absent_dir" / "window.spv"

            with mock.patch.object(
                saw.subprocess,
                "run",
                side_effect=AssertionError("subprocess.run called for empty tool flag"),
            ):
                for label, flag, val in (
                    ("empty_assembler", "--assembler", ""),
                    ("blank_assembler", "--assembler", "   \t  "),
                    ("empty_validator", "--validator", ""),
                    ("blank_validator", "--validator", "   \n  "),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(src), str(absent_out), flag, val],
                            expected_code=1,
                            fragments=(flag, "non-empty"),
                        )
            self.assertFalse(absent_out.parent.exists())

    def test_subprocess_and_replace_failures_clean_up_temp_dirs_and_preserve_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            target_dir = work / "target_dir"
            target_dir.mkdir()
            protected = target_dir / "kept.spv"
            sentinel = b"SENTINEL_KEEP_BYTES"
            protected.write_bytes(sentinel)

            created_tempdirs: list[Path] = []
            real_tempdir = tempfile.TemporaryDirectory
            calls: list[list[str]] = []
            fail_on_call: dict[int, BaseException] = {}

            class TrackingTempDir(real_tempdir):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    created_tempdirs.append(Path(self.name))

            def fake_run(cmd, check=False, **_kwargs):
                calls.append([str(part) for part in cmd])
                idx = len(calls)
                if idx in fail_on_call:
                    raise fail_on_call[idx]
                if cmd[0] == "mock-as":
                    out_idx = cmd.index("-o") + 1
                    Path(cmd[out_idx]).write_bytes(b"MOCK_SPV")
                return subprocess.CompletedProcess(cmd, 0)

            with (
                mock.patch.object(saw.subprocess, "run", side_effect=fake_run),
                mock.patch.object(saw.tempfile, "TemporaryDirectory", TrackingTempDir),
            ):
                failure_cases = (
                    ("orig_as_called_process_error", 1, subprocess.CalledProcessError(1, ["mock-as"]), ("returned non-zero",)),
                    ("orig_val_called_process_error", 2, subprocess.CalledProcessError(1, ["mock-val"]), ("returned non-zero",)),
                    ("lower_as_called_process_error", 3, subprocess.CalledProcessError(1, ["mock-as"]), ("returned non-zero",)),
                    ("lower_val_called_process_error", 4, subprocess.CalledProcessError(1, ["mock-val"]), ("returned non-zero",)),
                    ("missing_tool_file_not_found", 1, FileNotFoundError("no mock-as binary"), ("no mock-as binary",)),
                    ("validator_timeout", 2, subprocess.TimeoutExpired(["mock-val"], 5), ("timed out",)),
                )
                for label, call_idx, exc, frags in failure_cases:
                    with self.subTest(case=label):
                        calls.clear()
                        fail_on_call.clear()
                        fail_on_call[call_idx] = exc
                        self._assert_clean_failure(
                            [str(src), str(protected), "--assembler", "mock-as", "--validator", "mock-val"],
                            expected_code=1,
                            fragments=frags,
                            check_subprocess=False,
                        )
                        self.assertEqual(protected.read_bytes(), sentinel)
                        self.assertEqual([p.name for p in target_dir.iterdir()], ["kept.spv"])
                        self.assertFalse(created_tempdirs[-1].exists())

                fail_on_call.clear()
                with mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")):
                    self._assert_clean_failure(
                        [str(src), str(protected), "--assembler", "mock-as", "--validator", "mock-val"],
                        expected_code=1,
                        fragments=("simulated replace failure",),
                        check_subprocess=False,
                    )
                self.assertEqual(protected.read_bytes(), sentinel)
                self.assertEqual([p.name for p in target_dir.iterdir()], ["kept.spv"])
                self.assertFalse(created_tempdirs[-1].exists())

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            src = work / "subtract.spvasm"
            src.write_text(SUBTRACT_ASM, encoding="utf-8")
            out = work / "out.spv"

            for argv, frags in (
                ([], ("required",)),
                ([str(src)], ("required",)),
                ([str(src), str(out), "--assembler"], ("expected one argument",)),
                ([str(src), str(out), "--validator"], ("expected one argument",)),
                ([str(src), str(out), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )


if __name__ == "__main__":
    unittest.main()
