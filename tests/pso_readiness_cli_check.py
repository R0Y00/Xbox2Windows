"""CLI, schema, path guard, error handling, and mutation suite for tools/pso_readiness.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "pso_readiness.py"
sys.path.insert(0, str(ROOT / "tools"))

import pso_readiness  # noqa: E402

EXPECTED_TOP_KEYS = {
    "archive_records",
    "records_with_native_sha256",
    "unique_native_stage_hashes",
    "pipeline_count",
    "pipeline_status_counts",
    "pipelines",
}


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = pso_readiness.main(argv)
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


def _write_fixture(root: Path):
    code = b"synthetic shader bytes"
    digest = hashlib.sha256(code).hexdigest()
    pixel = {"stage": "pixel", "sha256": digest}
    manifest = {"shaders": [{**pixel, "native_file": "pixel.bin"}]}
    pipelines = {
        "pipelines": [
            {"id": "p0", "effective_root_signature": "present", "shaders": [pixel]},
        ]
    }
    (root / "pixel.bin").write_bytes(code)
    manifest_path = root / "manifest.json"
    pipelines_path = root / "pipelines.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pipelines_path.write_text(json.dumps(pipelines), encoding="utf-8")
    return manifest, pipelines, manifest_path, pipelines_path, code, digest


class PsoReadinessCliPositiveTests(unittest.TestCase):
    def test_cli_json_schema_without_llvm_and_with_llvm_file(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path, pipelines_path, code_bytes, digest = _write_fixture(root)

            code, out, err = _run_main([str(manifest_path), str(pipelines_path)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            doc = json.loads(out)
            self.assertEqual(set(doc), EXPECTED_TOP_KEYS)
            self.assertEqual(doc["archive_records"], 1)
            self.assertEqual(doc["records_with_native_sha256"], 1)
            self.assertEqual(doc["unique_native_stage_hashes"], 1)
            self.assertEqual(doc["pipeline_count"], 1)
            self.assertEqual(doc["pipeline_status_counts"], {"needs_evidence": 1})
            self.assertEqual(doc["pipelines"][0]["status"], "needs_evidence")
            self.assertEqual(doc["pipelines"][0]["shaders"]["pixel"]["status"], "pixel_scan_untested")

            proc = _run_subprocess(str(manifest_path), str(pipelines_path))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), doc)

            # With --llvm pointing to an existing regular file, CLI accepts it and passes path string.
            llvm_file = root / "fake_llvm.dll"
            llvm_file.write_bytes(b"MZ")
            seen_calls = []

            def fake_disassemble(blob, llvm_arg, **kwargs):
                seen_calls.append((blob, llvm_arg, kwargs))
                return ["v_mov_b32_e32 v0, v1", "s_endpgm"]

            fake_pixel_aot = types.SimpleNamespace(disassemble=fake_disassemble)
            with (
                mock.patch.dict(sys.modules, {"pixel_aot": fake_pixel_aot}),
                mock.patch.object(pso_readiness, "assess_pixel", return_value=("pixel_lowering_passed", None)),
            ):
                code_l, out_l, err_l = _run_main([
                    str(manifest_path), str(pipelines_path), "--llvm", str(llvm_file),
                ])

            self.assertIs(type(code_l), int)
            self.assertEqual(code_l, 0, err_l)
            self.assertEqual(err_l, "")
            self.assertEqual(seen_calls, [(code_bytes, str(llvm_file), {})])
            doc_l = json.loads(out_l)
            self.assertEqual(doc_l["pipelines"][0]["status"], "needs_pso_validation")
            self.assertEqual(doc_l["pipelines"][0]["shaders"]["pixel"]["sha256"], digest)
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["fake_llvm.dll", "manifest.json", "pipelines.json", "pixel.bin"],
            )

    def test_report_preserves_string_llvm_for_direct_callers(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, pipelines, _, _, code_bytes, _ = _write_fixture(root)
            seen = []

            def fake_disassemble(blob, llvm_arg, **kwargs):
                seen.append((blob, llvm_arg))
                return ["s_endpgm"]

            fake_pixel_aot = types.SimpleNamespace(disassemble=fake_disassemble)
            with (
                mock.patch.dict(sys.modules, {"pixel_aot": fake_pixel_aot}),
                mock.patch.object(pso_readiness, "assess_pixel", return_value=("pixel_lowering_passed", None)),
            ):
                doc = pso_readiness.report(manifest, pipelines, root, llvm="test-llvm")

            self.assertEqual(seen, [(code_bytes, "test-llvm")])
            self.assertEqual(doc["pipelines"][0]["status"], "needs_pso_validation")

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path, pipelines_path, _, _ = _write_fixture(root)
            cli_argv = [str(TOOL), str(manifest_path), str(pipelines_path)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(json.loads(out)["pipeline_count"], 1)

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue())["pipeline_count"], 1)


class PsoReadinessCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(self, argv, *, expected_code=1, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)
        if expected_code == 1:
            self.assertTrue(err.startswith("PSO report: "), err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)
            if expected_code == 1:
                self.assertTrue(proc.stderr.startswith("PSO report: "), proc.stderr)

    def test_argparse_errors_exit_nonzero_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path, pipelines_path, _, _ = _write_fixture(root)

            for argv, frag in (
                ([], "usage:"),
                ([str(manifest_path)], "usage:"),
                ([str(manifest_path), str(pipelines_path), "--unknown-flag"], "unrecognized"),
            ):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            pso_readiness.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self._assert_clean_failure(argv, expected_code=2, expected_fragment=frag)

    def test_manifest_and_pipelines_schema_guards_kill_mutations(self):
        valid_manifest = {"shaders": []}
        valid_pipelines = {"pipelines": []}

        bad_manifests = [
            ([], "manifest"),
            ("text", "manifest"),
            (42, "manifest"),
            ({}, "shaders"),
            ({"shaders": {}}, "shaders"),
            ({"shaders": "text"}, "shaders"),
            ({"shaders": 5}, "shaders"),
            ({"shaders": None}, "shaders"),
            ({"shaders": [123]}, "manifest shader entries"),
        ]
        for bad_manifest, frag in bad_manifests:
            with self.subTest(bad_manifest=bad_manifest):
                with self.assertRaises(ValueError) as raised:
                    pso_readiness.report(bad_manifest, valid_pipelines, Path("."))
                self.assertIn(frag, str(raised.exception))

        bad_pipelines = [
            ([], "pipelines"),
            ("text", "pipelines"),
            (42, "pipelines"),
            ({}, "pipelines"),
            ({"pipelines": {}}, "pipelines"),
            ({"pipelines": "text"}, "pipelines"),
            ({"pipelines": 5}, "pipelines"),
            ({"pipelines": None}, "pipelines"),
        ]
        for bad_pipeline, frag in bad_pipelines:
            with self.subTest(bad_pipeline=bad_pipeline):
                with self.assertRaises(ValueError) as raised:
                    pso_readiness.report(valid_manifest, bad_pipeline, Path("."))
                self.assertIn(frag, str(raised.exception))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            pipelines_path = root / "pipelines.json"
            manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")
            pipelines_path.write_text(json.dumps(valid_pipelines), encoding="utf-8")

            for bad_manifest, frag in bad_manifests:
                manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
                self._assert_clean_failure(
                    [str(manifest_path), str(pipelines_path)],
                    expected_fragment=frag,
                    check_subprocess=False,
                )
            manifest_path.write_text(json.dumps(valid_manifest), encoding="utf-8")

            for bad_pipeline, frag in bad_pipelines:
                pipelines_path.write_text(json.dumps(bad_pipeline), encoding="utf-8")
                self._assert_clean_failure(
                    [str(manifest_path), str(pipelines_path)],
                    expected_fragment=frag,
                    check_subprocess=False,
                )

            # Corrupt JSON also exits 1 with exact prefix and zero traceback.
            pipelines_path.write_text("{corrupt", encoding="utf-8")
            self._assert_clean_failure(
                [str(manifest_path), str(pipelines_path)],
                expected_fragment="Expecting",
            )

    def test_manifest_pipelines_and_llvm_path_guards_kill_mutations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path, pipelines_path, _, _ = _write_fixture(root)
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            missing_file = root / "missing.bin"

            # 1) manifest is an existing directory (including "").
            for bad_manifest in (str(existing_dir), ""):
                with mock.patch.object(pso_readiness, "report") as report_mock:
                    self._assert_clean_failure(
                        [bad_manifest, str(pipelines_path)],
                        expected_fragment="existing directory",
                        check_subprocess=bool(bad_manifest),
                    )
                report_mock.assert_not_called()

            # 2) pipelines is an existing directory (including "").
            for bad_pipelines in (str(existing_dir), ""):
                with mock.patch.object(pso_readiness, "report") as report_mock:
                    self._assert_clean_failure(
                        [str(manifest_path), bad_pipelines],
                        expected_fragment="existing directory",
                        check_subprocess=bool(bad_pipelines),
                    )
                report_mock.assert_not_called()

            # 3) --llvm missing or directory rejected upfront before report().
            for bad_llvm, frag in (
                (str(missing_file), "not found"),
                (str(existing_dir), "regular file"),
                ("", "regular file"),
            ):
                with mock.patch.object(pso_readiness, "report") as report_mock:
                    self._assert_clean_failure(
                        [str(manifest_path), str(pipelines_path), "--llvm", bad_llvm],
                        expected_fragment=frag,
                        check_subprocess=bool(bad_llvm),
                    )
                report_mock.assert_not_called()

            # 4) Missing manifest / pipelines files exit 1 cleanly.
            self._assert_clean_failure(
                [str(missing_file), str(pipelines_path)],
                expected_fragment=missing_file.name,
            )
            self._assert_clean_failure(
                [str(manifest_path), str(missing_file)],
                expected_fragment=missing_file.name,
            )

    def test_os_value_type_and_key_errors_caught_cleanly_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, manifest_path, pipelines_path, _, _ = _write_fixture(root)
            for exc in (
                OSError("simulated io error"),
                ValueError("simulated value error"),
                TypeError("simulated type error"),
                KeyError("simulated_key"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(pso_readiness, "report", side_effect=exc):
                        self._assert_clean_failure(
                            [str(manifest_path), str(pipelines_path)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["manifest.json", "pipelines.json", "pixel.bin"],
            )


if __name__ == "__main__":
    unittest.main()
