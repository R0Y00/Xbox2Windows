#!/usr/bin/env python3
"""Pure-Python CLI, atomic-write, path/count guard, and mutation suite for tools/fuzz_cpu.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "fuzz_cpu.py"
sys.path.insert(0, str(TOOLS_DIR))

import fuzz_cpu  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = fuzz_cpu.main(argv)
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


class FuzzCpuCliPositiveTests(unittest.TestCase):
    def test_emit_multi_area_atomic_nested_outputs_and_zero_tmp_files(self) -> None:
        areas_to_check = ("leaf", "frames", "calls", "switches", "scalar_fp", "x87", "bits")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_replace = os.replace

            for idx, area in enumerate(areas_to_check):
                with self.subTest(area=area):
                    out_dir = base / f"nested_{idx}" / area
                    asm_path = out_dir / f"{area}.asm"
                    def_path = out_dir / f"{area}.def"
                    seen_replaces: list[tuple[Path, Path]] = []

                    def tracking_replace(src, dst):
                        src_path, dst_path = Path(src), Path(dst)
                        seen_replaces.append((src_path, dst_path))
                        self.assertEqual(src_path.parent.resolve(), dst_path.parent.resolve())
                        self.assertTrue(src_path.name.endswith(".tmp"))
                        return real_replace(src, dst)

                    with mock.patch.object(fuzz_cpu.os, "replace", side_effect=tracking_replace):
                        code, out, err = _run_main([
                            "emit",
                            "0xA902",
                            "4",
                            str(asm_path),
                            str(def_path),
                            "--area",
                            area,
                        ])

                    self.assertIs(type(code), int)
                    self.assertEqual(code, 0, err)
                    self.assertEqual(out, "")
                    self.assertEqual(err, "")
                    self.assertEqual(len(seen_replaces), 2)
                    self.assertEqual(sorted(p.name for p in out_dir.iterdir()), sorted([asm_path.name, def_path.name]))
                    asm_text = asm_path.read_text(encoding="utf-8")
                    def_text = def_path.read_text(encoding="utf-8")
                    self.assertIn(".code", asm_text)
                    self.assertTrue(asm_text.endswith("END\n"))
                    expected_lib = fuzz_cpu.AREAS[area][1]
                    self.assertTrue(def_text.startswith(f"LIBRARY {expected_lib}\nEXPORTS\n"))

            # End-to-end subprocess check with overwrite on existing files.
            sub_dir = base / "sub_emit"
            sub_asm = sub_dir / "out.asm"
            sub_def = sub_dir / "out.def"
            sub_dir.mkdir()
            sub_asm.write_text("stale asm", encoding="utf-8")
            sub_def.write_text("stale def", encoding="utf-8")
            proc = _run_subprocess("emit", "0x1234", "2", str(sub_asm), str(sub_def), "--area", "loops")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertEqual(proc.stderr, "")
            self.assertEqual(sorted(p.name for p in sub_dir.iterdir()), ["out.asm", "out.def"])
            self.assertIn("LIBRARY cpu_fuzz_loops", sub_def.read_text(encoding="utf-8"))

    def test_translate_atomic_write_and_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "synthetic.dll"
            image.write_bytes(b"MZ-synthetic-placeholder")
            out_cpp = base / "nested" / "lifted.cpp"
            translated_calls: list[tuple[Path, list[str]]] = []

            def fake_translate(img: Path, names: list[str]) -> tuple[str, dict]:
                translated_calls.append((img, list(names)))
                return "// lifted\nint ok = 1;\n", {}

            fake_mod = types.SimpleNamespace(translate=fake_translate)
            with mock.patch.object(fuzz_cpu, "translator_module", return_value=fake_mod):
                code, out, err = _run_main(["translate", "0xA902", "2", str(image), str(out_cpp)])

            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(len(translated_calls), 1)
            self.assertEqual(translated_calls[0][0], image)
            self.assertEqual(translated_calls[0][1], ["fuzz_0", "fuzz_1"])
            self.assertEqual(out_cpp.read_text(encoding="utf-8"), "// lifted\nint ok = 1;\n")
            self.assertEqual([p.name for p in out_cpp.parent.iterdir()], ["lifted.cpp"])

            # sys.argv forwarding (argv=None) and __main__ entrypoint via runpy.
            asm_entry = base / "entry.asm"
            def_entry = base / "entry.def"
            cli_argv = [str(TOOL), "emit", "0x55", "1", str(asm_entry), str(def_entry)]
            with mock.patch.object(sys, "argv", cli_argv):
                code_none, out_none, err_none = _run_main(None)
            self.assertEqual(code_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertTrue(asm_entry.is_file() and def_entry.is_file())

            asm_entry.unlink()
            def_entry.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(asm_entry.is_file() and def_entry.is_file())


class FuzzCpuCliNegativeAndMutationTests(unittest.TestCase):
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

    def test_non_positive_count_rejected_upfront_in_emit_translate_and_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "image.dll"
            lifted = base / "lifted.dll"
            o2 = base / "o2.dll"
            for dll in (image, lifted, o2):
                dll.write_bytes(b"MZ-placeholder")
            absent_asm = base / "absent_emit" / "out.asm"
            absent_def = base / "absent_emit" / "out.def"
            absent_cpp = base / "absent_trans" / "out.cpp"

            for bad_count in ("0", "-3"):
                with self.subTest(bad_count=bad_count):
                    with (
                        mock.patch.object(fuzz_cpu, "_write_atomic", side_effect=AssertionError("should not write")),
                        mock.patch.object(fuzz_cpu, "translator_module", side_effect=AssertionError("should not lift")),
                        mock.patch.object(fuzz_cpu.ctypes, "CDLL", side_effect=AssertionError("should not load DLL")),
                    ):
                        self._assert_clean_failure(
                            ["emit", "1", bad_count, str(absent_asm), str(absent_def)],
                            fragments=("count must be >= 1",),
                        )
                        self._assert_clean_failure(
                            ["translate", "1", bad_count, str(image), str(absent_cpp)],
                            fragments=("count must be >= 1",),
                        )
                        self._assert_clean_failure(
                            ["check", "1", bad_count, str(image), str(lifted), str(o2)],
                            fragments=("count must be >= 1",),
                        )

            self.assertFalse(absent_asm.parent.exists())
            self.assertFalse(absent_cpp.parent.exists())

    def test_missing_and_directory_pe_rejected_upfront_in_translate_and_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid_dll = base / "valid.dll"
            valid_dll.write_bytes(b"MZ-placeholder")
            dir_dll = base / "dir_dll"
            dir_dll.mkdir()
            missing_dll = base / "missing.dll"
            absent_cpp = base / "never_created" / "out.cpp"

            with (
                mock.patch.object(fuzz_cpu, "translator_module", side_effect=AssertionError("should not lift")),
                mock.patch.object(fuzz_cpu, "_write_atomic", side_effect=AssertionError("should not write")),
            ):
                self._assert_clean_failure(
                    ["translate", "1", "2", str(missing_dll), str(absent_cpp)],
                    fragments=("image file not found",),
                )
                self._assert_clean_failure(
                    ["translate", "1", "2", str(dir_dll), str(absent_cpp)],
                    fragments=("image path is a directory",),
                )
            self.assertFalse(absent_cpp.parent.exists())

            with mock.patch.object(fuzz_cpu.ctypes, "CDLL", side_effect=AssertionError("should not load DLL")):
                for label, argv, frag in (
                    ("missing_native", ["check", "1", "2", str(missing_dll), str(valid_dll), str(valid_dll)], "native file not found"),
                    ("dir_native", ["check", "1", "2", str(dir_dll), str(valid_dll), str(valid_dll)], "native path is a directory"),
                    ("missing_lifted", ["check", "1", "2", str(valid_dll), str(missing_dll), str(valid_dll)], "lifted file not found"),
                    ("dir_lifted", ["check", "1", "2", str(valid_dll), str(dir_dll), str(valid_dll)], "lifted path is a directory"),
                    ("missing_o2", ["check", "1", "2", str(valid_dll), str(valid_dll), str(missing_dll)], "o2 file not found"),
                    ("dir_o2", ["check", "1", "2", str(valid_dll), str(valid_dll), str(dir_dll)], "o2 path is a directory"),
                    (
                        "missing_mutant",
                        ["check", "1", "2", str(valid_dll), str(valid_dll), str(valid_dll), "--area", "frames", "--mutant", str(missing_dll)],
                        "mutant file not found",
                    ),
                    (
                        "dir_mutant",
                        ["check", "1", "2", str(valid_dll), str(valid_dll), str(valid_dll), "--area", "frames", "--mutant", str(dir_dll)],
                        "mutant path is a directory",
                    ),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, fragments=(frag,))

    def test_corrupt_pe_translate_exits_cleanly_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            corrupt = base / "corrupt.dll"
            corrupt.write_bytes(b"not a PE image")
            out_cpp = base / "absent" / "lifted.cpp"
            self._assert_clean_failure(["translate", "0xA902", "1", str(corrupt), str(out_cpp)])
            self.assertFalse(out_cpp.parent.exists())

    def test_output_dir_file_parent_and_collision_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"MZ-input")
            existing_dir = base / "out_dir"
            existing_dir.mkdir()
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep-parent", encoding="utf-8")
            existing_asm = base / "existing.asm"
            existing_asm.write_text("keep-asm", encoding="utf-8")
            hardlink_def = base / "hardlink.def"
            hardlink_def.hardlink_to(existing_asm)
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)

            with (
                mock.patch.object(fuzz_cpu, "_write_atomic", side_effect=AssertionError("should not write")),
                mock.patch.object(fuzz_cpu, "translator_module", side_effect=AssertionError("should not lift")),
            ):
                for label, argv, frag in (
                    ("emit_asm_dir", ["emit", "1", "1", str(existing_dir), str(base / "ok.def")], "directory"),
                    ("emit_def_dir", ["emit", "1", "1", str(base / "ok.asm"), str(existing_dir)], "directory"),
                    ("emit_asm_file_parent", ["emit", "1", "1", str(file_parent / "a.asm"), str(base / "ok.def")], "not a directory"),
                    ("emit_def_ancestor_file", ["emit", "1", "1", str(base / "ok.asm"), str(file_parent / "sub" / "a.def")], "not a directory"),
                    ("emit_same_path", ["emit", "1", "1", str(base / "same.asm"), str(base / "same.asm")], "distinct"),
                    ("emit_alias_path", ["emit", "1", "1", str(base / "same.asm"), str(existing_dir / ".." / "same.asm")], "distinct"),
                    ("emit_hardlink_asm_def", ["emit", "1", "1", str(existing_asm), str(hardlink_def)], "distinct"),
                    ("translate_out_dir", ["translate", "1", "1", str(image), str(existing_dir)], "directory"),
                    ("translate_file_parent", ["translate", "1", "1", str(image), str(file_parent / "out.cpp")], "not a directory"),
                    ("translate_same_image", ["translate", "1", "1", str(image), str(image)], "distinct"),
                    ("translate_hardlink_image", ["translate", "1", "1", str(image), str(hardlink_cpp)], "distinct"),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, fragments=(frag,))

            self.assertEqual(file_parent.read_text(encoding="utf-8"), "keep-parent")
            self.assertEqual(existing_asm.read_text(encoding="utf-8"), "keep-asm")
            self.assertEqual(image.read_bytes(), b"MZ-input")
            self.assertFalse((base / "ok.asm").exists())
            self.assertFalse((base / "ok.def").exists())

    def test_output_symlink_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"MZ-input")
            out_dir = base / "out"
            out_dir.mkdir()
            real_target = out_dir / "real.txt"
            real_target.write_text("untouched", encoding="utf-8")
            live_link = out_dir / "live_link.asm"
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
                mock.patch.object(fuzz_cpu, "_write_atomic", side_effect=AssertionError("should not write")),
                mock.patch.object(fuzz_cpu, "translator_module", side_effect=AssertionError("should not lift")),
            ):
                if has_symlinks:
                    self._assert_clean_failure(
                        ["emit", "1", "1", str(live_link), str(out_dir / "ok.def")],
                        fragments=("symlink",),
                    )
                    self._assert_clean_failure(
                        ["translate", "1", "1", str(image), str(dangling_link)],
                        fragments=("symlink",),
                    )
                else:
                    with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=lambda p: p == real_target):
                        self._assert_clean_failure(
                            ["emit", "1", "1", str(real_target), str(out_dir / "ok.def")],
                            fragments=("symlink",),
                            check_subprocess=False,
                        )
                        self._assert_clean_failure(
                            ["translate", "1", "1", str(image), str(real_target)],
                            fragments=("symlink",),
                            check_subprocess=False,
                        )
            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched")

    def test_os_replace_and_staging_failures_preserve_target_and_leave_zero_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out_dir = base / "out_dir"
            out_dir.mkdir()
            asm_path = out_dir / "fuzz.asm"
            def_path = out_dir / "fuzz.def"
            asm_path.write_text("prior-asm", encoding="utf-8")
            def_path.write_text("prior-def", encoding="utf-8")

            # 1) os.replace fails on existing target -> target preserved, 0 .tmp files.
            def fail_replace(src, _dst):
                self.assertTrue(Path(src).is_file())
                raise OSError("simulated replace failure")

            with mock.patch.object(fuzz_cpu.os, "replace", side_effect=fail_replace):
                self._assert_clean_failure(
                    ["emit", "0xA902", "2", str(asm_path), str(def_path)],
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(asm_path.read_text(encoding="utf-8"), "prior-asm")
            self.assertEqual(def_path.read_text(encoding="utf-8"), "prior-def")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["fuzz.asm", "fuzz.def"])

            # 2) os.replace fails on translate -> prior .cpp preserved, 0 .tmp files.
            image = base / "input.dll"
            image.write_bytes(b"MZ-input")
            cpp_path = out_dir / "lifted.cpp"
            cpp_path.write_text("prior-cpp", encoding="utf-8")
            fake_mod = types.SimpleNamespace(translate=lambda _img, _names: ("// new\n", {}))
            with (
                mock.patch.object(fuzz_cpu, "translator_module", return_value=fake_mod),
                mock.patch.object(fuzz_cpu.os, "replace", side_effect=fail_replace),
            ):
                self._assert_clean_failure(
                    ["translate", "0xA902", "2", str(image), str(cpp_path)],
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )
            self.assertEqual(cpp_path.read_text(encoding="utf-8"), "prior-cpp")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["fuzz.asm", "fuzz.def", "lifted.cpp"])

            # 3) Staging failure after .tmp creation -> .tmp unlinked, os.replace not called, target preserved.
            real_ntf = fuzz_cpu.tempfile.NamedTemporaryFile

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
                    raise OSError("simulated staging write failure")

            with (
                mock.patch.object(fuzz_cpu.os, "replace") as replace_mock,
                mock.patch.object(
                    fuzz_cpu.tempfile,
                    "NamedTemporaryFile",
                    side_effect=lambda *a, **kw: FailingWriteHandle(real_ntf(*a, **kw)),
                ),
            ):
                self._assert_clean_failure(
                    ["emit", "0xA902", "2", str(asm_path), str(def_path)],
                    fragments=("simulated staging write failure",),
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(asm_path.read_text(encoding="utf-8"), "prior-asm")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["fuzz.asm", "fuzz.def", "lifted.cpp"])

    def test_argparse_errors_py314_safe(self) -> None:
        for argv, frags in (
            ([], ("required",)),
            (["emit"], ("required",)),
            (["emit", "0x1", "2", "a.asm", "a.def", "--area", "bogus"], ("invalid choice",)),
            (["translate", "bad_seed", "1", "a.dll", "a.cpp"], ("invalid",)),
            (["check", "1", "1", "n.dll", "l.dll", "o.dll", "--bogus-flag"], ("unrecognized",)),
        ):
            with self.subTest(argv=argv):
                self._assert_clean_failure(argv, expected_code=None, fragments=frags)


if __name__ == "__main__":
    unittest.main()
