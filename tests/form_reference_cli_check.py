"""Pure-Python CLI, atomic-write, path-guard, and mutation suite for tools/form_reference.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "form_reference.py"
sys.path.insert(0, str(TOOLS_DIR))

import form_reference  # noqa: E402


SYNTHETIC_RST = {
    "AMDGPUAsmGFX10.rst": (
        "SOPP\n----\n\n.. parsed-literal::\n\n    s_endpgm\n\n"
        "EXP\n---\n\n    exp\n"
    ),
    "AMDGPUAsmGFX1011.rst": (
        "SOPK\n----\n\n    s_addk_i32     sdst, simm16\n"
    ),
    "AMDGPUAsmGFX1013.rst": (
        "SMEM\n----\n\n    s_atomic_add   sdata, sbase, offset\n\n"
        "VOP1\n----\n\n    v_mov_b32_e32  vdst, src0\n"
    ),
    "AMDGPUAsmGFX1030.rst": (
        "SOPP\n----\n\n    s_endpgm\n\n"
        "VOP3\n----\n\n    v_fmac_f32_e64 vdst, src0, src1\n\n"
        ".. toctree::\n    :hidden:\n\n    gfx1030_ignored_entry\n"
    ),
}


def _populate_docs_dir(docs_dir: Path, *, skip: str | None = None) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, text in SYNTHETIC_RST.items():
        if name == skip:
            continue
        (docs_dir / name).write_text(text, encoding="utf-8")


def _run_main(argv: list[str] | None, *, module=form_reference) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = module.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str, script: Path = TOOL) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(script), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class FormReferenceCliPositiveTests(unittest.TestCase):
    def test_synthetic_rst_and_nested_out_dir_atomic_generation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "llvm_docs"
            _populate_docs_dir(docs)
            out_dir = root / "nested" / "coverage"
            self.assertFalse(out_dir.exists())

            replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_p, dst_p = Path(src), Path(dst)
                replaces.append((src_p, dst_p))
                self.assertEqual(src_p.parent.resolve(), dst_p.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(form_reference.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([str(docs), "--out-dir", str(out_dir)])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(replaces), 3)
            self.assertIn("gfx1013.tsv 5", out)
            self.assertIn("rdna2.tsv 2", out)
            self.assertIn("x64_zen2.tsv", out)
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["gfx1013.tsv", "rdna2.tsv", "x64_zen2.tsv"],
            )

            gfx_lines = (out_dir / "gfx1013.tsv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(gfx_lines[0].startswith("# 5 forms. Source: "), gfx_lines[0])
            self.assertEqual(
                gfx_lines[1:],
                [
                    "EXP\texp",
                    "SMEM\ts_atomic_add",
                    "SOPK\ts_addk_i32",
                    "SOPP\ts_endpgm",
                    "VOP1\tv_mov_b32_e32",
                ],
            )
            rdna_lines = (out_dir / "rdna2.tsv").read_text(encoding="utf-8").splitlines()
            self.assertTrue(rdna_lines[0].startswith("# 2 forms. Source: "), rdna_lines[0])
            self.assertEqual(rdna_lines[1:], ["SOPP\ts_endpgm", "VOP3\tv_fmac_f32_e64"])
            x64_lines = (out_dir / "x64_zen2.tsv").read_text(encoding="utf-8").splitlines()
            expected_x64 = form_reference.x64_forms()
            self.assertTrue(x64_lines[0].startswith(f"# {len(expected_x64)} forms. Source: "))
            self.assertEqual(len(x64_lines) - 1, len(expected_x64))

    def test_atomic_overwrite_sys_argv_and_main_entrypoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            _populate_docs_dir(docs)
            out_dir = root / "out"
            out_dir.mkdir()
            for name in form_reference.OUTPUT_TSV_NAMES:
                (out_dir / name).write_text("stale\n", encoding="utf-8")

            proc = _run_subprocess(str(docs), "--out-dir", str(out_dir))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["gfx1013.tsv", "rdna2.tsv", "x64_zen2.tsv"],
            )
            self.assertTrue(
                (out_dir / "rdna2.tsv").read_text(encoding="utf-8").startswith("# 2 forms.")
            )

            # Verify argv=None forwarding and __main__ entrypoint.
            out_entry = root / "entry_out"
            cli_argv = [str(TOOL), str(docs), "--out-dir", str(out_entry)]
            with mock.patch.object(sys, "argv", cli_argv):
                code, _, err = _run_main(None)
            self.assertEqual(code, 0, err)
            self.assertTrue((out_entry / "gfx1013.tsv").is_file())

            out_runpy = root / "runpy_out"
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(docs), "--out-dir", str(out_runpy)]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(
                sorted(p.name for p in out_runpy.iterdir()),
                ["gfx1013.tsv", "rdna2.tsv", "x64_zen2.tsv"],
            )


class FormReferenceCliNegativeTests(unittest.TestCase):
    def _assert_exit_1_clean(self, argv: list[str], *, fragment: str, check_subproc: bool = True) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, 1, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn("form_reference:", err)
        self.assertIn(fragment, err)

        if check_subproc:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("form_reference:", proc.stderr)
            self.assertIn(fragment, proc.stderr)

    def test_missing_and_non_directory_docs_dir_rejected_upfront(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "never_created"
            reg_file = root / "not_a_dir.txt"
            reg_file.write_text("file", encoding="utf-8")

            with (
                mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")),
                mock.patch.object(form_reference, "x64_forms", side_effect=AssertionError("x64")),
            ):
                for bad_docs, frag in (
                    (str(root / "missing_docs"), "not a directory"),
                    (str(reg_file), "not a directory"),
                    ("", "must not be empty"),
                    ("   ", "must not be empty"),
                ):
                    with self.subTest(bad_docs=bad_docs):
                        self._assert_exit_1_clean(
                            [bad_docs, "--out-dir", str(out_dir)],
                            fragment=frag,
                            check_subproc=False,
                        )
            self.assertFalse(out_dir.exists())

            proc = _run_subprocess(str(root / "missing_docs"), "--out-dir", str(out_dir))
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("not a directory", proc.stderr)
            self.assertFalse(out_dir.exists())

    def test_missing_or_directory_rst_rejected_upfront(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "never_created"

            for missing_name in SYNTHETIC_RST:
                docs = root / f"missing_{missing_name}"
                _populate_docs_dir(docs, skip=missing_name)
                with (
                    mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")),
                    mock.patch.object(form_reference, "x64_forms", side_effect=AssertionError("x64")),
                ):
                    self._assert_exit_1_clean(
                        [str(docs), "--out-dir", str(out_dir)],
                        fragment=missing_name,
                        check_subproc=False,
                    )
                self.assertFalse(out_dir.exists())

            dir_rst_docs = root / "dir_rst_docs"
            _populate_docs_dir(dir_rst_docs, skip="AMDGPUAsmGFX1030.rst")
            (dir_rst_docs / "AMDGPUAsmGFX1030.rst").mkdir()
            with mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")):
                self._assert_exit_1_clean(
                    [str(dir_rst_docs), "--out-dir", str(out_dir)],
                    fragment="AMDGPUAsmGFX1030.rst",
                    check_subproc=True,
                )
            self.assertFalse(out_dir.exists())

    def test_output_dir_file_parent_and_empty_out_dir_rejected_upfront(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            _populate_docs_dir(docs)

            # 1) Output file path is an existing directory.
            out_with_dir = root / "out_with_dir"
            out_with_dir.mkdir()
            (out_with_dir / "gfx1013.tsv").mkdir()
            with mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")):
                self._assert_exit_1_clean(
                    [str(docs), "--out-dir", str(out_with_dir)],
                    fragment="must not be a directory",
                )
            self.assertEqual([p.name for p in out_with_dir.iterdir()], ["gfx1013.tsv"])

            # 2) --out-dir itself is an existing regular file.
            out_is_file = root / "file_out"
            out_is_file.write_text("keep-me", encoding="utf-8")
            with mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")):
                self._assert_exit_1_clean(
                    [str(docs), "--out-dir", str(out_is_file)],
                    fragment="not a directory",
                )
            self.assertEqual(out_is_file.read_text(encoding="utf-8"), "keep-me")

            # 3) Parent ancestor of --out-dir is a regular file.
            nested_under_file = out_is_file / "sub"
            with mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")):
                self._assert_exit_1_clean(
                    [str(docs), "--out-dir", str(nested_under_file)],
                    fragment="output parent is not a directory",
                )
            self.assertEqual(out_is_file.read_text(encoding="utf-8"), "keep-me")

            # 4) Empty --out-dir.
            self._assert_exit_1_clean([str(docs), "--out-dir", "  "], fragment="--out-dir must not be empty")

    def test_output_symlink_rejected_upfront(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            _populate_docs_dir(docs)
            out_dir = root / "out"
            out_dir.mkdir()
            real_target = root / "real_target.tsv"
            real_target.write_text("untouched\n", encoding="utf-8")
            link_target = out_dir / "x64_zen2.tsv"
            try:
                link_target.symlink_to(real_target)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with mock.patch.object(form_reference, "llvm_forms", side_effect=AssertionError("parsed")):
                self._assert_exit_1_clean(
                    [str(docs), "--out-dir", str(out_dir)],
                    fragment="symlink",
                )
            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched\n")
            self.assertEqual([p.name for p in out_dir.iterdir()], ["x64_zen2.tsv"])

    def test_os_replace_failure_leaves_zero_tmp_and_preserves_existing_tsv(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            _populate_docs_dir(docs)
            out_dir = root / "out"
            out_dir.mkdir()
            for name in form_reference.OUTPUT_TSV_NAMES:
                (out_dir / name).write_text(f"prior-{name}\n", encoding="utf-8")

            with mock.patch.object(form_reference.os, "replace", side_effect=OSError("simulated replace fail")):
                self._assert_exit_1_clean(
                    [str(docs), "--out-dir", str(out_dir)],
                    fragment="simulated replace fail",
                    check_subproc=False,
                )

            self.assertEqual(
                sorted(p.name for p in out_dir.iterdir()),
                ["gfx1013.tsv", "rdna2.tsv", "x64_zen2.tsv"],
            )
            for name in form_reference.OUTPUT_TSV_NAMES:
                self.assertEqual((out_dir / name).read_text(encoding="utf-8"), f"prior-{name}\n")

    def test_argparse_errors_py314_safe(self) -> None:
        with TemporaryDirectory() as tmp:
            docs = Path(tmp) / "docs"
            _populate_docs_dir(docs)
            for argv, frag in (
                ([], "required"),
                ([str(docs), "--bogus-flag"], "unrecognized"),
                ([str(docs), "--out-dir"], "expected one argument"),
            ):
                with self.subTest(argv=argv):
                    proc = _run_subprocess(*argv)
                    self.assertEqual(proc.returncode, 2)
                    self.assertEqual(proc.stdout, "")
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn("usage:", proc.stderr.lower())
                    self.assertIn(frag, proc.stderr)

    def test_fast_temp_mutations_killed(self) -> None:
        """Verify mutated copies of form_reference.py fail targeted guard assertions."""
        original_src = TOOL.read_text(encoding="utf-8")
        mutations = (
            (
                "drop_docs_is_dir",
                "    if not docs.is_dir():",
                "    if False:",
                lambda m_docs, m_out: [str(m_docs / "not_a_dir"), "--out-dir", str(m_out)],
                "not a directory",
            ),
            (
                "drop_rst_is_file",
                "        if not rst.is_file():",
                "        if False:",
                lambda m_docs, m_out: [str(m_docs / "missing_rst"), "--out-dir", str(m_out)],
                "missing or non-regular",
            ),
            (
                "drop_out_dir_check",
                "    if path.exists() and path.is_dir():",
                "    if False:",
                lambda m_docs, m_out: [str(m_docs / "valid"), "--out-dir", str(m_out / "dir_collision")],
                "output path must not be a directory",
            ),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_docs = root / "valid"
            _populate_docs_dir(valid_docs)
            (root / "not_a_dir").write_text("file", encoding="utf-8")
            _populate_docs_dir(root / "missing_rst", skip="AMDGPUAsmGFX1030.rst")
            out_base = root / "out"
            out_base.mkdir()
            (out_base / "dir_collision").mkdir()
            (out_base / "dir_collision" / "gfx1013.tsv").mkdir()

            for label, old, new, argv_fn, expected_msg in mutations:
                mutated_src = original_src.replace(old, new, 1)
                self.assertNotEqual(mutated_src, original_src, label)
                mutated_tool = root / f"mut_{label}.py"
                mutated_tool.write_text(mutated_src, encoding="utf-8")
                argv = argv_fn(root, out_base)
                proc = _run_subprocess(*argv, script=mutated_tool)
                self.assertNotIn(expected_msg, proc.stderr, label)

            # Mutating temp-file cleanup must leak a .tmp file on replace failure.
            no_cleanup_src = original_src.replace("tmp_path.unlink(missing_ok=True)", "pass", 1)
            self.assertNotEqual(no_cleanup_src, original_src)
            mut_ns: dict[str, object] = {"__file__": str(TOOL)}
            exec(compile(no_cleanup_src, str(root / "mut_cleanup.py"), "exec"), mut_ns)
            leak_dir = root / "leak_out"
            leak_dir.mkdir()
            with mock.patch.object(mut_ns["os"], "replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    mut_ns["_write_atomic"](leak_dir / "gfx1013.tsv", "content")
            self.assertTrue(any(p.suffix == ".tmp" for p in leak_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
