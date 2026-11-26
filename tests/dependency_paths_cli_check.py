"""CLI, schema-validation, encoding, and mutation control suite for tools/dependency_paths.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dependency_paths.py"
sys.path.insert(0, str(ROOT / "tools"))

import dependency_paths  # noqa: E402


def _synthetic_inventory() -> dict:
    return {
        "images": {
            "app/game.exe": {"sha256": "a" * 64, "exports": []},
            "app/helper.dll": {
                "sha256": "b" * 64,
                "exports": [{"name": "HelperInit"}],
            },
        },
        "imports": [
            {
                "image": "app/game.exe",
                "kind": "normal",
                "dll": "helper.dll",
                "symbol": "HelperInit",
                "bundle_exports_in": ["app/helper.dll"],
                "desktop_export": "export_not_found",
            }
        ],
    }


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = dependency_paths.main(argv)
        except SystemExit as exc:
            return exc.code, stdout.getvalue(), stderr.getvalue()
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        env=env,
        timeout=30,
    )


class DependencyPathsCliPositiveTests(unittest.TestCase):
    def _assert_document_schema(self, doc: dict, expected_sha256: str, expected_entries: list[str]) -> None:
        self.assertEqual(set(doc), {"inventory_sha256", "entries"})
        self.assertEqual(doc["inventory_sha256"], expected_sha256)
        self.assertEqual(sorted(doc["entries"]), sorted(expected_entries))
        for name in expected_entries:
            entry = doc["entries"][name]
            self.assertEqual(
                set(entry),
                {"entry", "reachable", "findings", "dynamic_lookup_imports", "unlinked_in_this_graph", "scope"},
            )
            self.assertEqual(entry["entry"], name)

    def test_single_entry_and_all_entries_return_zero_with_exact_json_schema(self) -> None:
        payload = json.dumps(_synthetic_inventory()).encode("utf-8")
        expected_sha = hashlib.sha256(payload).hexdigest()

        with TemporaryDirectory() as tmp:
            inv_path = Path(tmp) / "inventory.json"
            inv_path.write_bytes(payload)

            # 1) All entries mode (omitted entry arg)
            code, out, err = _run_main([str(inv_path)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            all_doc = json.loads(out)
            self._assert_document_schema(all_doc, expected_sha, ["app/game.exe", "app/helper.dll"])
            self.assertEqual(
                sorted(all_doc["entries"]["app/game.exe"]["reachable"]),
                ["app/game.exe", "app/helper.dll"],
            )
            self.assertEqual(all_doc["entries"]["app/game.exe"]["findings"], [])

            # 2) Single entry mode
            code, out, err = _run_main([str(inv_path), "app/game.exe"])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            one_doc = json.loads(out)
            self._assert_document_schema(one_doc, expected_sha, ["app/game.exe"])

            proc = _run_subprocess(str(inv_path), "app/helper.dll")
            self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
            self.assertEqual(proc.stderr, b"")
            sub_doc = json.loads(proc.stdout)
            self._assert_document_schema(sub_doc, expected_sha, ["app/helper.dll"])
            self.assertEqual(sub_doc["entries"]["app/helper.dll"]["unlinked_in_this_graph"], ["app/game.exe"])

    def test_non_ascii_survives_cp1252_stdout_and_sys_argv_entrypoint(self) -> None:
        non_ascii_name = "pkg/\u03a9mega_\u2603.dll"
        inv = {
            "images": {non_ascii_name: {"sha256": "0" * 64, "exports": []}},
            "imports": [],
        }
        payload = json.dumps(inv, ensure_ascii=False).encode("utf-8")
        expected_sha = hashlib.sha256(payload).hexdigest()

        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        env.pop("PYTHONUTF8", None)

        with TemporaryDirectory() as tmp:
            inv_path = Path(tmp) / "non_ascii.json"
            inv_path.write_bytes(payload)

            proc = _run_subprocess(str(inv_path), non_ascii_name, env=env)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
            self.assertEqual(proc.stderr, b"")
            doc = json.loads(proc.stdout)
            self._assert_document_schema(doc, expected_sha, [non_ascii_name])

            # Forward sys.argv when argv=None and verify __main__ raises SystemExit(0).
            cli_argv = [str(TOOL), str(inv_path)]
            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self._assert_document_schema(json.loads(out), expected_sha, [non_ascii_name])

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self._assert_document_schema(json.loads(stdout.getvalue()), expected_sha, [non_ascii_name])


class DependencyPathsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_exit(
        self,
        argv: list[str],
        *,
        expected_code: int,
        expected_fragments: tuple[str, ...],
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code)
            self.assertEqual(proc.stdout, b"")
            stderr = proc.stderr.decode(errors="replace")
            self.assertNotIn("Traceback", stderr)
            for fragment in expected_fragments:
                self.assertIn(fragment, stderr)

    def test_missing_and_directory_inventory_rejected_cleanly(self) -> None:
        # Kills removal of OSError guard and removal of args.inventory.is_dir() guard.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            self._assert_clean_exit(
                [str(missing)],
                expected_code=1,
                expected_fragments=(str(missing),),
            )

            inv_dir = root / "inv_dir"
            inv_dir.mkdir()
            with mock.patch.object(dependency_paths, "trace") as trace_mock:
                self._assert_clean_exit(
                    [str(inv_dir)],
                    expected_code=1,
                    expected_fragments=(str(inv_dir), "Inventory path is a directory"),
                )
            trace_mock.assert_not_called()

    def test_invalid_utf8_and_invalid_json_rejected_cleanly(self) -> None:
        # Kills removal of UnicodeDecodeError / ValueError guards.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_utf8 = root / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe\x80\x81")
            self._assert_clean_exit(
                [str(bad_utf8)],
                expected_code=1,
                expected_fragments=(str(bad_utf8),),
            )

            bad_json = root / "bad_json.json"
            bad_json.write_bytes(b"{not valid json")
            self._assert_clean_exit(
                [str(bad_json)],
                expected_code=1,
                expected_fragments=(str(bad_json),),
            )

    def test_schema_guards_kill_mutations_on_root_images_and_imports(self) -> None:
        # Each case uses values that would succeed or raise AttributeError if the corresponding guard were removed.
        schema_cases = (
            ("root_list", json.dumps(["images", "imports"]), "root must be an object"),
            ("root_int", json.dumps(42), "root must be an object"),
            ("missing_images", json.dumps({"imports": []}), "'images' object"),
            ("images_empty_list", json.dumps({"images": [], "imports": []}), "'images' object"),
            ("missing_imports", json.dumps({"images": {}}), "'imports' list"),
            ("imports_empty_dict", json.dumps({"images": {"a.exe": {"sha256": "0" * 64, "exports": []}},
                                               "imports": {}}), "'imports' list"),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, content, fragment in schema_cases:
                with self.subTest(label=label):
                    inv_path = root / f"{label}.json"
                    inv_path.write_text(content, encoding="utf-8")
                    self._assert_clean_exit(
                        [str(inv_path)],
                        expected_code=1,
                        expected_fragments=(str(inv_path), fragment),
                    )

    def test_trace_keyerror_and_typeerror_and_missing_entry_arg(self) -> None:
        # Kills removal of KeyError/TypeError from except tuple and entry presence/is-not-None guards.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            missing_sha = root / "missing_sha.json"
            missing_sha.write_text(json.dumps({"images": {"a.exe": {"exports": []}}, "imports": []}), encoding="utf-8")
            self._assert_clean_exit(
                [str(missing_sha)],
                expected_code=1,
                expected_fragments=(str(missing_sha), "sha256"),
            )

            bad_image_type = root / "bad_image_type.json"
            bad_image_type.write_text(json.dumps({"images": {"a.exe": None}, "imports": []}), encoding="utf-8")
            self._assert_clean_exit(
                [str(bad_image_type)],
                expected_code=1,
                expected_fragments=(str(bad_image_type),),
            )

            valid = root / "valid.json"
            valid.write_text(json.dumps(_synthetic_inventory()), encoding="utf-8")
            for missing_entry in ("missing.exe", ""):
                with self.subTest(entry=missing_entry):
                    self._assert_clean_exit(
                        [str(valid), missing_entry],
                        expected_code=2,
                        expected_fragments=("Entry does not appear in this inventory",),
                    )


if __name__ == "__main__":
    unittest.main()
