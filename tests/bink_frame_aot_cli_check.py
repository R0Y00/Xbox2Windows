#!/usr/bin/env python3
"""Pure-Python CLI, atomic-write, path-guard, and mutation suite for tools/bink_frame_aot.py."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "bink_frame_aot.py"
sys.path.insert(0, str(TOOLS_DIR))
import bink_frame_aot  # noqa: E402


SYNTHETIC_CPP = (
    '#include <cstdint>\n'
    'using word = uint64_t;\n'
    'extern "C" __declspec(dllexport) uint32_t frame_allocate(word, word, uint32_t) { return 0; }\n'
)


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


class BinkFrameAotCliPositiveTests(unittest.TestCase):
    def test_atomic_cpp_output_in_nested_dirs_and_overwrite_with_zero_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "synthetic.dll"
            image.write_bytes(b"synthetic-image-placeholder")
            nested_out = base / "nested" / "sub" / "frame_aot.cpp"

            real_replace = os.replace
            replaced_pairs: list[tuple[Path, Path]] = []

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                replaced_pairs.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                return real_replace(src, dst)

            with (
                mock.patch.object(bink_frame_aot, "generate", return_value=SYNTHETIC_CPP) as gen,
                mock.patch.object(os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main(bink_frame_aot, [str(image), str(nested_out)])
            gen.assert_called_once_with(image)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), SYNTHETIC_CPP)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["frame_aot.cpp"])
            self.assertEqual(len(replaced_pairs), 1)
            self.assertEqual(replaced_pairs[0][1].resolve(), nested_out.resolve())

            # Atomic overwrite preserves zero .tmp files.
            updated_cpp = SYNTHETIC_CPP + "// updated\n"
            with mock.patch.object(bink_frame_aot, "generate", return_value=updated_cpp):
                code_ow, out_ow, err_ow = _run_main(bink_frame_aot, [str(image), str(nested_out)])
            self.assertEqual(code_ow, 0, err_ow)
            self.assertEqual(out_ow, "")
            self.assertEqual(err_ow, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), updated_cpp)
            self.assertEqual([p.name for p in nested_out.parent.iterdir()], ["frame_aot.cpp"])

            # argv=None forwards sys.argv cleanly.
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(image), str(nested_out)]),
                mock.patch.object(bink_frame_aot, "generate", return_value=SYNTHETIC_CPP),
            ):
                code_none, out_none, err_none = _run_main(bink_frame_aot, None)
            self.assertEqual(code_none, 0, err_none)
            self.assertEqual(out_none, "")
            self.assertEqual(err_none, "")
            self.assertEqual(nested_out.read_text(encoding="ascii"), SYNTHETIC_CPP)

            # __main__ entrypoint propagates SystemExit(1) cleanly on unmatched image.
            stdout_m, stderr_m = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(image), str(base / "unused.cpp")]),
                redirect_stdout(stdout_m),
                redirect_stderr(stderr_m),
                self.assertRaises(SystemExit) as exited,
            ):
                runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 1)
            self.assertEqual(stdout_m.getvalue(), "")
            self.assertNotIn("Traceback", stderr_m.getvalue())
            self.assertTrue(stderr_m.getvalue().strip())


class BinkFrameAotCliNegativeTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(bink_frame_aot, argv)
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

    def test_missing_dir_corrupt_and_unmatched_hash_image_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            dir_image = base / "dir_image"
            dir_image.mkdir()
            absent_out = base / "absent_dir" / "frame_aot.cpp"

            with (
                mock.patch.object(
                    bink_frame_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid image path"),
                ),
                mock.patch.object(
                    bink_frame_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid image path"),
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

            # Unmatched SHA-256 hash image fails cleanly without creating output parent.
            unmatched = base / "unmatched.dll"
            unmatched.write_bytes(b"not-the-pinned-bink-sha256")
            self._assert_clean_failure(
                [str(unmatched), str(absent_out)],
                expected_code=1,
                fragments=("version-query profile",),
            )
            self.assertFalse(absent_out.parent.exists())

            # Corrupt PE bytes (with patched SHA-256) fail cleanly without creating output parent.
            import hashlib

            digest = hashlib.sha256(b"not a PE32+").hexdigest()
            corrupt = base / "corrupt.dll"
            corrupt.write_bytes(b"not a PE32+")
            with mock.patch.object(bink_frame_aot, "BINK_SHA256", digest):
                self._assert_clean_failure(
                    [str(corrupt), str(absent_out)],
                    expected_code=1,
                    check_subprocess=False,
                )
            self.assertFalse(absent_out.parent.exists())

    def test_output_guards_dir_file_parent_samefile_and_non_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            same_cpp = base / "same.cpp"
            same_cpp.write_bytes(b"synthetic-cpp-bytes")
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)
            out_dir = base / "existing_dir.cpp"
            out_dir.mkdir()
            file_parent = base / "regular_parent.txt"
            file_parent.write_text("keep-me", encoding="ascii")

            with (
                mock.patch.object(
                    bink_frame_aot,
                    "generate",
                    side_effect=AssertionError("generate called for invalid output path"),
                ),
                mock.patch.object(
                    bink_frame_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for invalid output path"),
                ),
            ):
                for label, argv, code, frags in (
                    ("non_cpp_suffix", [str(image), str(base / "out.txt")], 2, ("distinct .cpp output",)),
                    ("empty_output", [str(image), ""], 2, ("distinct .cpp output",)),
                    ("same_path", [str(same_cpp), str(same_cpp)], 2, ("distinct .cpp output",)),
                    ("hardlink_samefile", [str(image), str(hardlink_cpp)], 2, ("distinct .cpp output",)),
                    ("output_is_dir", [str(image), str(out_dir)], 1, ("directory",)),
                    ("parent_is_file", [str(image), str(file_parent / "frame_aot.cpp")], 1, ("not a directory",)),
                    ("ancestor_is_file", [str(image), str(file_parent / "sub" / "frame_aot.cpp")], 1, ("not a directory",)),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=code, fragments=frags)

            self.assertEqual(same_cpp.read_bytes(), b"synthetic-cpp-bytes")
            self.assertEqual(image.read_bytes(), b"synthetic-image-bytes")
            self.assertEqual(list(out_dir.iterdir()), [])
            self.assertEqual(file_parent.read_text(encoding="ascii"), "keep-me")
            self.assertFalse((base / "out.txt").exists())

    def test_output_symlink_rejected_before_generate_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            real_target = out_dir / "real.cpp"
            real_target.write_text("untouched", encoding="ascii")
            live_link = out_dir / "live_link.cpp"
            dangling_link = out_dir / "dangling_link.cpp"
            try:
                live_link.symlink_to(real_target)
                dangling_link.symlink_to(out_dir / "missing.cpp")
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(
                    bink_frame_aot,
                    "generate",
                    side_effect=AssertionError("generate called for symlink output"),
                ),
                mock.patch.object(
                    bink_frame_aot,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called for symlink output"),
                ),
            ):
                for label, link in (("live_symlink", live_link), ("dangling_symlink", dangling_link)):
                    with self.subTest(case=label):
                        self._assert_clean_failure(
                            [str(image), str(link)],
                            expected_code=1,
                            fragments=("symlink",),
                        )

            self.assertEqual(real_target.read_text(encoding="ascii"), "untouched")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["dangling_link.cpp", "live_link.cpp", "real.cpp"],
            )

    def test_os_replace_failure_preserves_target_bytes_and_leaves_zero_tmp_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "frame_aot.cpp"
            out_cpp.write_text("original", encoding="ascii")

            with (
                mock.patch.object(bink_frame_aot, "generate", return_value=SYNTHETIC_CPP),
                mock.patch.object(os, "replace", side_effect=OSError("simulated replace failure")),
            ):
                self._assert_clean_failure(
                    [str(image), str(out_cpp)],
                    expected_code=1,
                    fragments=("simulated replace failure",),
                    check_subprocess=False,
                )

            self.assertEqual(out_cpp.read_text(encoding="ascii"), "original")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["frame_aot.cpp"])

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic")

            for argv, frags in (
                ([], ("required",)),
                ([str(image)], ("required",)),
                ([str(image), str(base / "frame_aot.cpp"), "--bogus-flag"], ("unrecognized",)),
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
            base = Path(tmp)
            image = base / "input.dll"
            image.write_bytes(b"synthetic-image-bytes")
            dir_image = base / "dir_image"
            dir_image.mkdir()
            hardlink_cpp = base / "hardlink.cpp"
            hardlink_cpp.hardlink_to(image)
            file_parent = base / "file_parent.txt"
            file_parent.write_text("keep", encoding="ascii")
            out_dir = base / "out_dir"
            out_dir.mkdir()
            out_cpp = out_dir / "frame_aot.cpp"
            out_cpp.write_text("original", encoding="ascii")

            # 1) Removing directory image check misses "image path is a directory".
            # 2) Removing parent.is_file check drops "output parent is not a directory".
            # 3) Removing samefile check allows hardlink collision past _validate_args.
            # 4) Removing temp unlink leaves a .tmp file behind on os.replace failure.
            for label, target, replacement, check_fn in (
                (
                    "drop_image_is_dir_guard",
                    '    if args.image.is_dir():\n        raise IsADirectoryError(f"image path is a directory: {args.image}")\n',
                    "    pass\n",
                    lambda mod: self.assertNotIn(
                        "image path is a directory",
                        _run_main(mod, [str(dir_image), str(out_dir / "a.cpp")])[2].lower(),
                    ),
                ),
                (
                    "drop_parent_is_file_guard",
                    '        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):\n'
                    '            raise NotADirectoryError(f"output parent is not a directory: {parent}")\n',
                    "        pass\n",
                    lambda mod: self.assertNotIn(
                        "output parent is not a directory",
                        _run_main(mod, [str(image), str(file_parent / "frame_aot.cpp")])[2].lower(),
                    ),
                ),
                (
                    "drop_samefile_guard",
                    "    if args.output.resolve() == args.image.resolve() or (args.output.exists() and args.image.samefile(args.output)):\n",
                    "    if args.output.resolve() == args.image.resolve():\n",
                    lambda mod: self.assertNotIn(
                        "choose a distinct .cpp output",
                        _run_main(mod, [str(image), str(hardlink_cpp)])[2].lower(),
                    ),
                ),
                (
                    "drop_tmp_unlink",
                    "        if tmp_path is not None:\n            tmp_path.unlink(missing_ok=True)\n",
                    "        pass\n",
                    lambda mod: self._verify_tmp_leaked_on_replace_fail(mod, image, out_cpp, out_dir),
                ),
            ):
                with self.subTest(mutation=label):
                    self.assertEqual(source.count(target), 1, target)
                    mod = types.ModuleType(f"bink_frame_aot_mut_{label}")
                    mod.__file__ = str(TOOL)
                    sys.modules[mod.__name__] = mod
                    try:
                        exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                        with mock.patch.object(mod, "generate", return_value=SYNTHETIC_CPP):
                            check_fn(mod)
                    finally:
                        sys.modules.pop(mod.__name__, None)

    def _verify_tmp_leaked_on_replace_fail(self, mod, image: Path, out_cpp: Path, out_dir: Path) -> None:
        with mock.patch.object(os, "replace", side_effect=OSError("boom")):
            code, _, _ = _run_main(mod, [str(image), str(out_cpp)])
        self.assertEqual(code, 1)
        leaked = [p for p in out_dir.iterdir() if p.suffix == ".tmp"]
        self.assertTrue(leaked, "expected unlinked .tmp file under mutation")
        for p in leaked:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
