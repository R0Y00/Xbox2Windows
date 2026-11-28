#!/usr/bin/env python3
"""Pure-Python synthetic PE32+ CLI, atomic-write, path-guard, and mutation suite for tools/bink_descriptors_aot.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_DIR / "bink_descriptors_aot.py"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

import bink_descriptors_aot  # noqa: E402
from bink_descriptors_aot_check import build_image  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = bink_descriptors_aot.main(argv)
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


class BinkDescriptorsAotCliPositiveTests(unittest.TestCase):
    def test_default_and_frame_planes_atomic_nested_outputs(self) -> None:
        raw = build_image()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "synthetic_bink_desc.dll"
            image.write_bytes(raw)

            # 1) Default mode (no --frame-planes) in nested directory, tracking atomic os.replace.
            default_out = base / "nested" / "default" / "descriptors.cpp"
            seen_replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.resolve(), dst_path.parent.resolve())
                self.assertTrue(src_path.name.endswith(".tmp"))
                return real_replace(src, dst)

            with mock.patch.object(bink_descriptors_aot.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([str(image), str(default_out)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(len(seen_replaces), 1)
            self.assertEqual([p.name for p in default_out.parent.iterdir()], ["descriptors.cpp"])
            default_text = default_out.read_text(encoding="ascii")
            self.assertEqual(default_text, bink_descriptors_aot.generate(image, False, False))
            self.assertIn("group_init", default_text)
            self.assertIn("group_append", default_text)
            self.assertNotIn("frame_planes", default_text)

            # 2) --frame-planes mode atomically overwrites existing .cpp with 0 .tmp files left.
            default_out.write_text("stale content", encoding="ascii")
            code_fp, out_fp, err_fp = _run_main([str(image), str(default_out), "--frame-planes"])
            self.assertEqual(code_fp, 0, err_fp)
            self.assertEqual(out_fp, "")
            self.assertEqual(err_fp, "")
            self.assertEqual([p.name for p in default_out.parent.iterdir()], ["descriptors.cpp"])
            fp_text = default_out.read_text(encoding="ascii")
            self.assertEqual(fp_text, bink_descriptors_aot.generate(image, False, True))
            self.assertIn("frame_planes", fp_text)
            self.assertIn("append_frame", fp_text)

            # 3) Subprocess end-to-end positive check.
            sub_out = base / "sub_dir" / "cli_out.cpp"
            proc = _run_subprocess(str(image), str(sub_out), "--frame-planes")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertEqual([p.name for p in sub_out.parent.iterdir()], ["cli_out.cpp"])
            self.assertEqual(sub_out.read_text(encoding="ascii"), fp_text)

    def test_sys_argv_and_main_entrypoint(self) -> None:
        raw = build_image()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "entry.dll"
            image.write_bytes(raw)
            out_cpp = base / "entry.cpp"

            with mock.patch.object(sys, "argv", [str(TOOL), str(image), str(out_cpp)]):
                code, out, err = _run_main(None)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(out_cpp.read_text(encoding="ascii"), bink_descriptors_aot.generate(image, False, False))

            out_cpp.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(image), str(out_cpp), "--frame-planes"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(out_cpp.read_text(encoding="ascii"), bink_descriptors_aot.generate(image, False, True))


class BinkDescriptorsAotCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None,
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

    def test_missing_and_directory_image_rejected_before_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dir_image = base / "dir_image"
            dir_image.mkdir()
            absent_out = base / "absent_dir" / "descriptors.cpp"

            with (
                mock.patch.object(
                    bink_descriptors_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid image"),
                ),
                mock.patch.object(
                    bink_descriptors_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid image"),
                ),
            ):
                self._assert_clean_failure(
                    [str(base / "missing.dll"), str(absent_out)],
                    expected_code=1,
                    fragments=("image file not found",),
                )
                self._assert_clean_failure(
                    [str(dir_image), str(absent_out)],
                    expected_code=1,
                    fragments=("image path is a directory",),
                )
            self.assertFalse(absent_out.parent.exists())

    def test_corrupt_pe_rejected_cleanly_without_creating_output_or_temps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            absent_out = base / "absent_dir" / "descriptors.cpp"
            for name, payload in (
                ("empty.dll", b""),
                ("not_pe.dll", b"not a valid PE32+"),
                ("trunc_pe.dll", b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x80)),
            ):
                with self.subTest(name=name):
                    bad_pe = base / name
                    bad_pe.write_bytes(payload)
                    self._assert_clean_failure([str(bad_pe), str(absent_out)], expected_code=1)
                    self.assertFalse(absent_out.parent.exists())

    def test_output_guards_rejected_before_generate_and_write(self) -> None:
        raw = build_image()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(raw)
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(raw)
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)
            out_dir = base / "existing_dir.cpp"
            out_dir.mkdir()
            file_parent = base / "regular_parent.txt"
            file_parent.write_text("keep-me", encoding="ascii")

            with (
                mock.patch.object(
                    bink_descriptors_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid output"),
                ),
                mock.patch.object(
                    bink_descriptors_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid output"),
                ),
            ):
                for label, argv, code, frags in (
                    ("non_cpp_suffix", [str(image), str(base / "out.txt")], 2, ("distinct .cpp output",)),
                    ("empty_output", [str(image), ""], 2, ("distinct .cpp output",)),
                    ("same_path", [str(same_cpp), str(same_cpp)], 2, ("distinct .cpp output",)),
                    ("hardlink_samefile", [str(image), str(hardlink_cpp)], 2, ("distinct .cpp output",)),
                    ("output_is_dir", [str(image), str(out_dir)], 1, ("directory",)),
                    ("parent_is_file", [str(image), str(file_parent / "out.cpp")], 1, ("not a directory",)),
                    ("ancestor_is_file", [str(image), str(file_parent / "sub" / "out.cpp")], 1, ("not a directory",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=code, fragments=frags)

            self.assertEqual(same_cpp.read_bytes(), raw)
            self.assertEqual(image.read_bytes(), raw)
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertEqual(file_parent.read_text(encoding="ascii"), "keep-me")
            self.assertFalse((base / "out.txt").exists())

    def test_output_symlink_rejected_before_generate_and_write(self) -> None:
        raw = build_image()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(raw)
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="ascii")
            live_link = out_dir / "live_link.cpp"
            dangling_link = out_dir / "dangling_link.cpp"

            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.cpp")
                has_symlinks = True
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                has_symlinks = False

            with (
                mock.patch.object(
                    bink_descriptors_aot,
                    "generate",
                    side_effect=AssertionError("generate called for symlink output"),
                ),
                mock.patch.object(
                    bink_descriptors_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                if has_symlinks:
                    for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                        with self.subTest(case=label):
                            self._assert_clean_failure(
                                [str(image), str(link)],
                                expected_code=1,
                                fragments=("symlink",),
                            )
                else:
                    # Fallback when Windows account lacks SeCreateSymbolicLinkPrivilege (WinError 1314).
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == real_target):
                        self._assert_clean_failure(
                            [str(image), str(real_target)],
                            expected_code=1,
                            fragments=("symlink",),
                            check_subprocess=False,
                        )
            self.assertEqual(real_target.read_text(encoding="ascii"), "untouched")

    def test_os_replace_and_staging_failures_preserve_target_and_leave_zero_tmp_files(self) -> None:
        raw = build_image()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(raw)
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "descriptors.cpp"
            out_cpp.write_text("original-target-bytes", encoding="ascii")

            # 1) os.replace fails on existing target -> target preserved, 0 .tmp files.
            replace_calls: list[tuple[Path, Path]] = []

            def fail_replace(src, dst):
                src_path = Path(src)
                replace_calls.append((src_path, Path(dst)))
                self.assertTrue(src_path.is_file())
                raise OSError("simulated replace failure")

            with mock.patch.object(bink_descriptors_aot.os, "replace", side_effect=fail_replace):
                self._assert_clean_failure(
                    [str(image), str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(len(replace_calls), 1)
            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original-target-bytes")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["descriptors.cpp"])

            # 2) os.replace fails on fresh nested path -> 0 output or .tmp files.
            fresh_dir = base / "fresh_dir"
            fresh_cpp = fresh_dir / "fresh.cpp"
            with mock.patch.object(bink_descriptors_aot.os, "replace", side_effect=fail_replace):
                self._assert_clean_failure(
                    [str(image), str(fresh_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(list(fresh_dir.iterdir()), [])

            # 3) Staging failure after .tmp creation -> os.replace not called, .tmp unlinked, target preserved.
            real_ntf = bink_descriptors_aot.tempfile.NamedTemporaryFile

            class FailingWriteHandle:
                def __init__(self, inner):
                    self._inner = inner
                    self.name = inner.name

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return self._inner.__exit__(exc_type, exc, tb)

                def write(self, _data):
                    raise OSError("simulated write failure")

            with (
                mock.patch.object(bink_descriptors_aot.os, "replace") as replace_mock,
                mock.patch.object(
                    bink_descriptors_aot.tempfile,
                    "NamedTemporaryFile",
                    side_effect=lambda *a, **kw: FailingWriteHandle(real_ntf(*a, **kw)),
                ),
            ):
                self._assert_clean_failure(
                    [str(image), str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated write failure",),
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original-target-bytes")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["descriptors.cpp"])

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic")
            out_cpp = base / "out.cpp"

            for argv, frags in (
                ([], ("required",)),
                ([str(image)], ("required",)),
                ([str(image), str(out_cpp), "--bogus-flag"], ("unrecognized",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(argv, expected_code=None, fragments=frags)


if __name__ == "__main__":
    unittest.main()
