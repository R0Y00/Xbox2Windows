"""CLI, atomic manifest, path/JSON/bounds guard, and mutation suite for tools/ttpf_batch_export.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ttpf_batch_export.py"
sys.path.insert(0, str(ROOT / "tools"))

import ttpf_batch_export as batch  # noqa: E402


SYNTHETIC_TTPF_MIPS = '''\
from pathlib import Path
import struct

BC_FORMATS = {71, 98}


class Pool:
    def __init__(self, path):
        raw = Path(path).read_bytes()
        if raw.startswith(b"BADSTRUCT"):
            struct.unpack("<I", raw[:2])
        if not raw.startswith(b"RDS5"):
            raise ValueError("invalid RDS5 header")
        self.textures = 3

    def close(self):
        pass


def analyze(pool, index):
    return {
        "layout": {"size": 4096},
        "available": [True],
        "full_chain": True,
        "descriptor": {"format": 71, "width": 4, "height": 4, "mips": 1},
        "shape": "2d",
    }


def export(pool, index, out_path, *, allow_provisional=False, history=None):
    Path(out_path).write_bytes(f"DDS_{index}".encode("ascii"))
    return 0
'''


def _write_fake_tools(tools_dir: Path) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "ttpf_mips.py").write_text(SYNTHETIC_TTPF_MIPS, encoding="ascii")
    return tools_dir


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = batch.main(argv)
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


class TtpfBatchExportCliPositiveTests(unittest.TestCase):
    def test_atomic_nested_manifest_summary_schema_and_count_zero(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools_dir = _write_fake_tools(root / "fake_tools")
            pool = root / "sample.ttpf"
            pool.write_bytes(b"RDS5_SYNTHETIC")
            out_dir = root / "out" / "dds"
            manifest_path = root / "manifests" / "nested" / "manifest.json"
            self.assertFalse(out_dir.exists())
            self.assertFalse(manifest_path.parent.exists())

            replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                replaces.append((src_path, dst_path))
                if dst_path == manifest_path:
                    self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(batch.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([
                    str(pool),
                    "-o", str(out_dir),
                    "--start", "1",
                    "--count", "2",
                    "--threads", "1",
                    "--manifest", str(manifest_path),
                    "--tools-path", str(tools_dir),
                ])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            summary = json.loads(out)
            self.assertEqual(
                set(summary),
                {"pool", "total_textures", "processed", "manifest", "entries", "elapsed_seconds", "stats"},
            )
            self.assertEqual(summary["total_textures"], 3)
            self.assertEqual(summary["processed"], 2)
            self.assertEqual(summary["entries"], 2)
            self.assertEqual(summary["stats"], {"strict": 2, "provisional": 0, "failed": 0, "skipped": 0})
            self.assertTrue(any(dst == manifest_path for _, dst in replaces))
            self.assertEqual([p.name for p in manifest_path.parent.iterdir()], ["manifest.json"])
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([e["index"] for e in entries], [1, 2])

            # --count 0 is valid and produces an empty manifest cleanly.
            proc_zero = _run_subprocess(
                str(pool),
                "-o", str(out_dir),
                "--count", "0",
                "--manifest", str(manifest_path),
                "--tools-path", str(tools_dir),
            )
            self.assertEqual(proc_zero.returncode, 0, proc_zero.stderr)
            summary_zero = json.loads(proc_zero.stdout)
            self.assertEqual(summary_zero["processed"], 0)
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), [])

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools_dir = _write_fake_tools(root / "fake_tools")
            pool = root / "entry.ttpf"
            pool.write_bytes(b"RDS5_ENTRY")
            out_dir = root / "entry_out"
            cli_argv = [
                str(TOOL),
                str(pool),
                "-o", str(out_dir),
                "--count", "1",
                "--tools-path", str(tools_dir),
            ]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["processed"], 1)
            self.assertTrue((out_dir / "manifest.json").is_file())

            (out_dir / "manifest.json").unlink()
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
            self.assertEqual(json.loads(stdout.getvalue())["processed"], 1)
            self.assertTrue((out_dir / "manifest.json").is_file())


class TtpfBatchExportCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(self, argv, *, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_pool_missing_or_directory_and_out_file_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            pool.write_bytes(b"RDS5")
            pool_dir = root / "pool_dir"
            pool_dir.mkdir()
            existing_file = root / "not_a_dir.txt"
            existing_file.write_text("keep", encoding="utf-8")

            with mock.patch.object(batch, "load_tools") as load_mock:
                self._assert_exit_2_clean([str(root / "missing.ttpf")], expected_fragment="Pool not found")
                self._assert_exit_2_clean([str(pool_dir)], expected_fragment="regular file")
                self._assert_exit_2_clean([str(pool), "-o", str(existing_file)], expected_fragment="not a directory")
            load_mock.assert_not_called()
            self.assertEqual(existing_file.read_text(encoding="utf-8"), "keep")

    def test_bounds_threads_start_count_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            pool.write_bytes(b"RDS5")
            out_dir = root / "never_created"

            for flag, bad_val, frag in (
                ("--threads", "0", "--threads must be >= 1"),
                ("--threads", "-2", "--threads must be >= 1"),
                ("--start", "-1", "--start must be >= 0"),
                ("--count", "-1", "--count must be >= 0"),
            ):
                with self.subTest(flag=flag, val=bad_val):
                    with mock.patch.object(batch, "load_tools") as load_mock:
                        self._assert_exit_2_clean(
                            [str(pool), "-o", str(out_dir), flag, bad_val],
                            expected_fragment=frag,
                        )
                    load_mock.assert_not_called()
                    self.assertFalse(out_dir.exists())

    def test_manifest_dir_and_pool_or_out_collisions_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            pool = root / "pool.ttpf"
            pool_bytes = b"RDS5_POOL"
            pool.write_bytes(pool_bytes)
            existing_dir = root / "manifest_dir"
            existing_dir.mkdir()
            out_dir = root / "never_out"

            # 1) --manifest is an existing directory.
            with mock.patch.object(batch, "load_tools") as load_mock:
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--manifest", str(existing_dir)],
                    expected_fragment="existing directory",
                )
            load_mock.assert_not_called()
            self.assertFalse(out_dir.exists())

            # 2) --manifest collides with pool via resolve() (kills resolve() check).
            alias_manifest = root / "sub" / ".." / "pool.ttpf"
            with mock.patch.object(batch, "load_tools") as load_mock:
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--manifest", str(alias_manifest)],
                    expected_fragment="collides with pool",
                )
            load_mock.assert_not_called()

            # 3) --manifest collides with pool via hardlink samefile() (kills samefile() check).
            hardlink_manifest = sub / "pool_link.json"
            os.link(pool, hardlink_manifest)
            self.assertNotEqual(pool.resolve(), hardlink_manifest.resolve())
            self.assertTrue(pool.samefile(hardlink_manifest))
            with mock.patch.object(batch, "load_tools") as load_mock:
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--manifest", str(hardlink_manifest)],
                    expected_fragment="collides with pool",
                )
            load_mock.assert_not_called()
            self.assertEqual(pool.read_bytes(), pool_bytes)

            # 4) --manifest collides with --out or is an ancestor of --out.
            with mock.patch.object(batch, "load_tools") as load_mock:
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--manifest", str(out_dir)],
                    expected_fragment="collides with --out",
                )
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir / "child"), "--manifest", str(out_dir)],
                    expected_fragment="ancestor of --out",
                )
            load_mock.assert_not_called()
            self.assertFalse(out_dir.exists())

    def test_history_calibration_and_tools_path_guards(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool.ttpf"
            pool.write_bytes(b"RDS5")
            out_dir = root / "never_out"
            a_dir = root / "a_dir"
            a_dir.mkdir()
            bad_json = root / "bad.json"
            bad_json.write_text("{broken", encoding="utf-8")
            list_json = root / "list.json"
            list_json.write_text("[]", encoding="utf-8")
            bad_ver = root / "bad_ver.json"
            bad_ver.write_text('{"version": 2}', encoding="utf-8")
            regular_file = root / "not_tools.txt"
            regular_file.write_text("x", encoding="utf-8")

            # --history: directory, bad JSON, non-dict JSON, missing file (even without --allow-provisional)
            for h_path, frag in (
                (a_dir, "directory"),
                (bad_json, "--history"),
                (list_json, "JSON object"),
                (root / "missing_hist.json", "--history"),
            ):
                with self.subTest(history=h_path), mock.patch.object(batch, "load_tools") as load_mock:
                    self._assert_exit_2_clean(
                        [str(pool), "-o", str(out_dir), "--allow-provisional", "--history", str(h_path)],
                        expected_fragment=frag,
                    )
                    load_mock.assert_not_called()

            # --calibration: directory, bad JSON, wrong version, missing file
            for c_path, frag in (
                (a_dir, "directory"),
                (bad_json, "--calibration"),
                (bad_ver, "Unsupported calibration version"),
                (root / "missing_cal.json", "--calibration"),
            ):
                with self.subTest(calibration=c_path), mock.patch.object(batch, "load_tools") as load_mock:
                    self._assert_exit_2_clean(
                        [str(pool), "-o", str(out_dir), "--calibration", str(c_path)],
                        expected_fragment=frag,
                    )
                    load_mock.assert_not_called()

            # --tools-path: missing or regular file
            for t_path, frag in (
                (root / "missing_tools", "--tools-path not found"),
                (regular_file, "--tools-path must be a directory"),
            ):
                with self.subTest(tools_path=t_path), mock.patch.object(batch, "load_tools") as load_mock:
                    self._assert_exit_2_clean(
                        [str(pool), "-o", str(out_dir), "--tools-path", str(t_path)],
                        expected_fragment=frag,
                    )
                    load_mock.assert_not_called()
            self.assertFalse(out_dir.exists())

    def test_struct_error_from_pool_caught_cleanly_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools_dir = _write_fake_tools(root / "fake_tools")
            bad_pool = root / "bad_struct.ttpf"
            bad_pool.write_bytes(b"BADSTRUCT")
            out_dir = root / "out_bad_struct"

            self._assert_exit_2_clean(
                [str(bad_pool), "-o", str(out_dir), "--tools-path", str(tools_dir)],
                expected_fragment="unpack requires",
            )
            self.assertFalse(out_dir.exists())

    def test_manifest_os_replace_failure_leaves_zero_temp_dirs_and_preserves_prior(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools_dir = _write_fake_tools(root / "fake_tools")
            pool = root / "pool.ttpf"
            pool.write_bytes(b"RDS5")
            out_dir = root / "out"
            manifest_dir = root / "manifest_out"
            manifest_path = manifest_dir / "manifest.json"
            real_replace = os.replace

            def fail_manifest_replace(src, dst):
                if Path(dst) == manifest_path:
                    self.assertTrue(Path(src).is_file())
                    raise OSError("simulated manifest replace failure")
                return real_replace(src, dst)

            # 1) Fresh manifest: os.replace fails -> exit 2, zero temp dirs, zero manifest files.
            with mock.patch.object(batch.os, "replace", side_effect=fail_manifest_replace):
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--count", "1", "--threads", "1",
                     "--manifest", str(manifest_path), "--tools-path", str(tools_dir)],
                    expected_fragment="simulated manifest replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(list(manifest_dir.iterdir()), [])

            # 2) Pre-existing manifest: os.replace fails -> prior file intact, zero temp dirs.
            manifest_path.write_text("prior-manifest\n", encoding="utf-8")
            with mock.patch.object(batch.os, "replace", side_effect=fail_manifest_replace):
                self._assert_exit_2_clean(
                    [str(pool), "-o", str(out_dir), "--count", "1", "--threads", "1",
                     "--manifest", str(manifest_path), "--tools-path", str(tools_dir)],
                    expected_fragment="simulated manifest replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "prior-manifest\n")
            self.assertEqual([p.name for p in manifest_dir.iterdir()], ["manifest.json"])

    def test_argparse_missing_required_pool_exit_2_without_traceback(self):
        proc = _run_subprocess()
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("usage:", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
