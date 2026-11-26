"""CLI, path/collision guard, error handling, and mutation suite for tools/emit_variants.py."""

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
TOOL = ROOT / "tools" / "emit_variants.py"
sys.path.insert(0, str(ROOT / "tools"))

import emit_variants  # noqa: E402


VALID_FIXTURE = '''\
from pathlib import Path

CALLS = []


def emit(image, output, control="lifted"):
    CALLS.append((image.name, control))
    output.write_text(f"{image.name}:{control}:{len(CALLS)}\\n", encoding="ascii")
'''

RAISING_VALUE_ERROR_FIXTURE = '''\
def emit(image, output, control):
    raise ValueError(f"unsupported control {control}")
'''

NO_EMIT_FIXTURE = '''\
VALUE = 42
'''

NON_CALLABLE_EMIT_FIXTURE = '''\
emit = "not_callable"
'''

SYNTAX_ERROR_FIXTURE = '''\
def emit(
'''

IMPORT_ERROR_FIXTURE = '''\
import nonexistent_xbox2windows_module_xyz

def emit(image, output, control):
    pass
'''


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = emit_variants.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args):
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class EmitVariantsCliPositiveTests(unittest.TestCase):
    def test_nested_output_parents_created_and_variants_emitted_in_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "sample_emit.py"
            image = root / "sample.dll"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")

            out1 = root / "nested" / "a" / "first.cpp"
            out2 = root / "nested" / "b" / "second.cpp"
            self.assertFalse(out1.parent.exists())
            self.assertFalse(out2.parent.exists())

            code, out, err = _run_main(
                [str(script), str(image), f"lifted={out1}", f"pc64={out2}"]
            )
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(out1.read_text(encoding="ascii"), "sample.dll:lifted:1\n")
            self.assertEqual(out2.read_text(encoding="ascii"), "sample.dll:pc64:2\n")

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "entry_emit.py"
            image = root / "entry.dll"
            out_file = root / "out" / "entry.cpp"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")
            cli_argv = [str(TOOL), str(script), str(image), f"lifted={out_file}"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(out_file.read_text(encoding="ascii"), "entry.dll:lifted:1\n")

            out_file.unlink()
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
            self.assertTrue(out_file.is_file())


class EmitVariantsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(self, argv, *, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, 1, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn("emit_variants:", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("emit_variants:", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_script_and_image_missing_or_directory_rejected_before_loading(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "ok_script.py"
            image = root / "ok_image.dll"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")
            missing_script = root / "missing_script.py"
            missing_image = root / "missing_image.dll"
            dir_path = root / "some_dir"
            dir_path.mkdir()
            nested_out = root / "never_created" / "out.cpp"

            for argv, frag in (
                ([str(missing_script), str(image), f"v={nested_out}"], "script not found"),
                ([str(dir_path), str(image), f"v={nested_out}"], "regular file"),
                ([str(script), str(missing_image), f"v={nested_out}"], "image not found"),
                ([str(script), str(dir_path), f"v={nested_out}"], "regular file"),
            ):
                with self.subTest(argv=argv):
                    with mock.patch.object(emit_variants, "load") as load_mock:
                        self._assert_clean_failure(argv, expected_fragment=frag)
                    load_mock.assert_not_called()
                    self.assertFalse(nested_out.parent.exists())

    def test_malformed_and_duplicate_variants_rejected_before_loading(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "ok_script.py"
            image = root / "ok_image.dll"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")
            out1 = root / "never" / "a.cpp"
            out2 = root / "never" / "b.cpp"

            for bad_pair in ("no-equals", f"={out1}", "lifted="):
                with self.subTest(bad_pair=bad_pair):
                    with mock.patch.object(emit_variants, "load") as load_mock:
                        self._assert_clean_failure(
                            [str(script), str(image), bad_pair],
                            expected_fragment="expected <variant>=<output>",
                        )
                    load_mock.assert_not_called()

            with mock.patch.object(emit_variants, "load") as load_mock:
                self._assert_clean_failure(
                    [str(script), str(image), f"lifted={out1}", f"lifted={out2}"],
                    expected_fragment="duplicate variant",
                )
            load_mock.assert_not_called()
            self.assertFalse(out1.parent.exists())

    def test_output_directory_duplicate_containment_and_input_collisions_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            script = root / "script.py"
            image = root / "image.dll"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            other_out = root / "never_created" / "other.cpp"

            # 1) Existing directory output rejected upfront.
            with mock.patch.object(emit_variants, "load") as load_mock:
                self._assert_clean_failure(
                    [str(script), str(image), f"v1={existing_dir}", f"v2={other_out}"],
                    expected_fragment="existing directory",
                )
            load_mock.assert_not_called()
            self.assertFalse(other_out.parent.exists())

            # 2) Duplicate output via resolve() (kills resolve() in _same_path).
            ghost_a = root / "out" / "shared.cpp"
            alias_b = root / "out" / ".." / "out" / "shared.cpp"
            with mock.patch.object(emit_variants, "load") as load_mock:
                self._assert_clean_failure(
                    [str(script), str(image), f"a={ghost_a}", f"b={alias_b}"],
                    expected_fragment="collides with 'a'",
                )
            load_mock.assert_not_called()
            self.assertFalse(ghost_a.parent.exists())

            # 3) Duplicate output via hardlink samefile() (kills samefile() in _same_path).
            real_out = root / "real_out.cpp"
            link_out = sub / "link_out.cpp"
            real_out.write_text("keep-out\n", encoding="ascii")
            os.link(real_out, link_out)
            self.assertNotEqual(real_out.resolve(), link_out.resolve())
            self.assertTrue(real_out.samefile(link_out))
            with mock.patch.object(emit_variants, "load") as load_mock:
                self._assert_clean_failure(
                    [str(script), str(image), f"a={real_out}", f"b={link_out}"],
                    expected_fragment="collides with 'a'",
                )
            load_mock.assert_not_called()
            self.assertEqual(real_out.read_text(encoding="ascii"), "keep-out\n")

            # 4) Output parent containment rejected upfront.
            parent_out = root / "nest" / "item"
            child_out = parent_out / "child.cpp"
            for pairs in ([f"a={parent_out}", f"b={child_out}"], [f"a={child_out}", f"b={parent_out}"]):
                with mock.patch.object(emit_variants, "load") as load_mock:
                    self._assert_clean_failure(
                        [str(script), str(image), *pairs],
                        expected_fragment="cannot contain one another",
                        check_subprocess=False,
                    )
                load_mock.assert_not_called()
                self.assertFalse(parent_out.parent.exists())

            # 5) Output colliding with script or image via resolve() and hardlink samefile().
            script_alias = root / "sub" / ".." / "script.py"
            image_alias = root / "sub" / ".." / "image.dll"
            script_link = sub / "script_link.cpp"
            image_link = sub / "image_link.cpp"
            os.link(script, script_link)
            os.link(image, image_link)

            for out_path, frag in (
                (script_alias, "collides with script"),
                (script_link, "collides with script"),
                (image_alias, "collides with image"),
                (image_link, "collides with image"),
            ):
                with self.subTest(out_path=out_path):
                    with mock.patch.object(emit_variants, "load") as load_mock:
                        self._assert_clean_failure(
                            [str(script), str(image), f"v={out_path}"],
                            expected_fragment=frag,
                        )
                    load_mock.assert_not_called()

    def test_missing_non_callable_emit_syntax_import_and_value_errors_caught_cleanly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.dll"
            image.write_bytes(b"MZ")
            nested_out = root / "out_dir" / "out.cpp"

            cases = [
                ("no_emit.py", NO_EMIT_FIXTURE, "does not define callable emit"),
                ("non_callable_emit.py", NON_CALLABLE_EMIT_FIXTURE, "does not define callable emit"),
                ("syntax_err.py", SYNTAX_ERROR_FIXTURE, "syntax_err.py"),
                ("import_err.py", IMPORT_ERROR_FIXTURE, "nonexistent_xbox2windows_module_xyz"),
            ]
            for name, content, frag in cases:
                with self.subTest(script=name):
                    script = root / name
                    script.write_text(content, encoding="ascii")
                    self._assert_clean_failure(
                        [str(script), str(image), f"v={nested_out}"],
                        expected_fragment=frag,
                    )
                    self.assertFalse(nested_out.parent.exists())

            val_script = root / "val_err.py"
            val_script.write_text(RAISING_VALUE_ERROR_FIXTURE, encoding="ascii")
            self._assert_clean_failure(
                [str(val_script), str(image), f"bad={nested_out}"],
                expected_fragment="unsupported control bad",
            )

    def test_argparse_missing_arguments_exit_2_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.py"
            image = root / "image.dll"
            script.write_text(VALID_FIXTURE, encoding="ascii")
            image.write_bytes(b"MZ")
            for bad_argv in ([], [str(script)], [str(script), str(image)]):
                proc = _run_subprocess(*bad_argv)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, "")
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("usage:", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
