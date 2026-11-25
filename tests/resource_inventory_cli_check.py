#!/usr/bin/env python3
"""Pure-Python CLI, atomic-write, and error-guard suite for tools/resource_inventory.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import resource_inventory  # noqa: E402


VALID_LAYOUT = """<?xml version="1.0" encoding="utf-8"?>
<Package>
  <Chunk Id="1000" Marker="Launch">
    <FileGroup SourcePath=".\\" DestinationPath="\\" Include="sample.bin" />
    <FileGroup SourcePath=".\\" DestinationPath="\\" Include="shaders.rpf" />
  </Chunk>
</Package>
"""


def _dxbc_dxil_container() -> bytes:
    """Build a minimal DXBC container with a framed DXIL program header."""
    program = struct.pack("<II4sIII", 0, 8, b"DXIL", 0, 16, 4) + b"BC\xc0\xde" + bytes(4)
    dxil_part = b"DXIL" + struct.pack("<I", len(program)) + program
    table_end = 32 + 4
    total_size = table_end + len(dxil_part)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<IIII", 1, total_size, 1, table_end)
        + dxil_part
    )


def _minimal_pe_bin() -> bytes:
    """Build a minimal PE header recognized by resource_inventory.signature."""
    buf = bytearray(256)
    buf[0:2] = b"MZ"
    pe_offset = 64
    struct.pack_into("<I", buf, 60, pe_offset)
    buf[pe_offset:pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<HH", buf, pe_offset + 4, 0x8664, 3)
    struct.pack_into("<I", buf, pe_offset + 40, 0x1234)
    return bytes(buf)


def _sample_rpf_bytes(container: bytes) -> bytes:
    """Build synthetic RPF9-like bytes with one aligned zlib stream and one raw-DEFLATE scene record."""
    z_payload = zlib.compress(container)
    pad_to_16 = (-len(z_payload)) % 16
    encoder = zlib.compressobj(wbits=-15)
    raw_deflate = encoder.compress(container) + encoder.flush()
    record_hdr = b"rec0\0\0" + struct.pack("<4I", len(raw_deflate), 0, len(raw_deflate), len(container))
    return b"9FPR" + bytes(12) + z_payload + bytes(pad_to_16) + record_hdr + raw_deflate


def _copy_isolated_tools(workspace: Path) -> Path:
    """Copy only the tools modules needed to run resource_inventory in isolation."""
    tools_copy = workspace / "tools"
    tools_copy.mkdir(parents=True, exist_ok=True)
    for name in ("resource_inventory.py", "verify_build.py"):
        shutil.copy2(TOOLS_DIR / name, tools_copy / name)
    return tools_copy / "resource_inventory.py"


def _make_validator_command(directory: Path, name: str, script_body: str) -> Path:
    """Create a directly invocable validator script for the current platform."""
    py_script = directory / f"{name}.py"
    py_script.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / f"{name}.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" -B "{py_script}" %*\r\n', encoding="utf-8")
        return wrapper
    wrapper = directory / f"{name}.sh"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -B "{py_script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def _run_in_process(argv: list[str], tool_path: Path | None = None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    patcher = mock.patch.object(resource_inventory, "__file__", str(tool_path)) if tool_path else None
    try:
        if patcher is not None:
            patcher.start()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                rc = resource_inventory.main(argv)
            except SystemExit as exc:
                code = exc.code
                rc = 0 if code is None else (int(code) if isinstance(code, int) else 1)
                if not isinstance(code, (type(None), int)):
                    stderr.write(f"{code}\n")
    finally:
        if patcher is not None:
            patcher.stop()
    return rc, stdout.getvalue(), stderr.getvalue()


class ResourceInventoryCliTests(unittest.TestCase):
    def _create_valid_build(self, workspace: Path, *, with_rpf: bool = True) -> tuple[Path, bytes]:
        build_dir = workspace / "sample_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "layout_test.xml").write_text(VALID_LAYOUT, encoding="utf-8")
        (build_dir / "sample.bin").write_bytes(_minimal_pe_bin())
        (build_dir / "ignored.torrent").write_bytes(b"ignored")
        (build_dir / "extra.txt").write_bytes(b"note")
        container = _dxbc_dxil_container()
        if with_rpf:
            (build_dir / "shaders.rpf").write_bytes(_sample_rpf_bytes(container))
        return build_dir, container

    def test_positive_cli_and_subprocess_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, container = self._create_valid_build(workspace)
            control = workspace / "control.bin"
            control.write_bytes(container)
            validator = _make_validator_command(
                workspace,
                "ok_validator",
                'import json, sys\n'
                'sys.stdin.buffer.read()\n'
                'print(json.dumps({"call_hr": "0x00000000", "status": "0x00000000"}))\n',
            )

            rc, stdout, stderr = _run_in_process(
                [
                    str(build_dir),
                    "--shader-control",
                    str(control),
                    "--scene-records",
                    "--scene-validator",
                    str(validator),
                ],
                tool_path=tool_copy,
            )
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn("1 chunks | 4 files |", stdout)
            self.assertIn("shaders.rpf", stdout)

            out_dir = workspace / "out" / "resources"
            report_path = out_dir / f"{build_dir.name}.json"
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [report_path.name])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["build"], build_dir.name)
            self.assertEqual(report["layout"], "layout_test.xml")
            self.assertEqual(report["outside_layout"], ["extra.txt", "layout_test.xml"])
            self.assertEqual(report["positive_control"]["sha256"], hashlib.sha256(container).hexdigest())
            self.assertEqual(report["positive_control"]["markers"]["framed_dxil_programs"], 1)
            self.assertEqual(report["archive_probes"]["shaders.rpf"]["raw_markers"]["framed_dxil_programs"], 0)
            self.assertEqual(len(report["archive_probes"]["shaders.rpf"]["sample_streams"]), 1)
            self.assertEqual(
                report["archive_probes"]["shaders.rpf"]["sample_streams"][0]["decoded_markers"]["framed_dxil_programs"],
                1,
            )
            self.assertEqual(report["scene_records"]["records"], 1)
            self.assertEqual(report["scene_records"]["validator_samples"], 1)
            pe_row = next(row for row in report["files"] if row["path"] == "sample.bin")
            self.assertEqual(pe_row["pe_machine"], "0x8664")
            self.assertEqual(pe_row["pe_sections"], 3)
            self.assertEqual(pe_row["pe_entry_rva"], 0x1234)

            proc = subprocess.run(
                [sys.executable, "-B", str(tool_copy), str(build_dir), "--scene-records"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [report_path.name])

    def test_atomic_write_preserves_existing_report_and_leaves_zero_temp_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, _ = self._create_valid_build(workspace, with_rpf=False)
            out_dir = workspace / "out" / "resources"
            out_dir.mkdir(parents=True)
            report_path = out_dir / f"{build_dir.name}.json"
            sentinel = '{"preserved": true}\n'
            report_path.write_text(sentinel, encoding="utf-8")

            for patch_target, exc in (
                ((resource_inventory.os, "replace"), OSError("disk full")),
                ((resource_inventory.json, "dump"), OSError("write error")),
                ((resource_inventory.json, "dump"), TypeError("unserializable")),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(patch_target[0], patch_target[1], side_effect=exc):
                        rc, stdout, stderr = _run_in_process([str(build_dir)], tool_path=tool_copy)
                    self.assertEqual(rc, 1)
                    self.assertEqual(stdout, "")
                    self.assertTrue(stderr.startswith("resource_inventory: "), stderr)
                    self.assertIn(str(exc), stderr)
                    self.assertNotIn("Traceback", stderr)
                    self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [report_path.name])
                    self.assertEqual(report_path.read_text(encoding="utf-8"), sentinel)

    def test_output_directory_and_output_inside_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, _ = self._create_valid_build(workspace, with_rpf=False)
            out_dir = workspace / "out" / "resources"
            bad_output_dir = out_dir / f"{build_dir.name}.json"
            bad_output_dir.mkdir(parents=True)

            with mock.patch.object(resource_inventory.json, "dump", side_effect=AssertionError("unexpected dump")):
                rc, _, stderr = _run_in_process([str(build_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr.startswith("resource_inventory: "), stderr)
            self.assertIn("directory", stderr)
            self.assertNotIn("Traceback", stderr)
            self.assertTrue(bad_output_dir.is_dir())
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [bad_output_dir.name])

            # Input directory is the workspace root itself, so out/resources/<name>.json sits inside root.
            (workspace / "layout_root.xml").write_text("<Package/>", encoding="utf-8")
            rc, _, stderr = _run_in_process([str(workspace)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr.startswith("resource_inventory: "), stderr)
            self.assertIn("outside", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_directory_and_layout_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)

            # Missing CLI args (Python 3.14-compatible argparse assertion).
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as cm:
                    resource_inventory.main([])
            self.assertNotEqual(cm.exception.code, 0)
            self.assertIn("directory", stderr.getvalue().lower())
            self.assertNotIn("Traceback", stderr.getvalue())

            # Missing input directory -> exit 1.
            rc, _, stderr_text = _run_in_process([str(workspace / "missing_dir")], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertNotIn("Traceback", stderr_text)

            # File passed instead of directory -> exit 1.
            not_a_dir = workspace / "file.txt"
            not_a_dir.write_text("hello", encoding="utf-8")
            rc, _, stderr_text = _run_in_process([str(not_a_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertNotIn("Traceback", stderr_text)

            # layout*.xml count != 1 preserves parser.error exit 2.
            empty_dir = workspace / "no_layout"
            empty_dir.mkdir()
            rc, _, stderr_text = _run_in_process([str(empty_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 2)
            self.assertIn("layout*.xml", stderr_text)
            self.assertNotIn("Traceback", stderr_text)

            # Malformed XML and missing attributes -> exit 1.
            for label, content in (
                ("malformed_xml", "<Package>"),
                ("missing_id", "<Package><Chunk/></Package>"),
                ("missing_include", '<Package><Chunk Id="1"><FileGroup DestinationPath="\\"/></Chunk></Package>'),
            ):
                with self.subTest(case=label):
                    bad_dir = workspace / label
                    bad_dir.mkdir()
                    (bad_dir / "layout.xml").write_text(content, encoding="utf-8")
                    rc, _, stderr_text = _run_in_process([str(bad_dir)], tool_path=tool_copy)
                    self.assertEqual(rc, 1)
                    self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
                    self.assertNotIn("Traceback", stderr_text)

            # Unreadable layout (OSError) -> exit 1.
            valid_dir, _ = self._create_valid_build(workspace, with_rpf=False)
            with mock.patch.object(resource_inventory, "read_layout", side_effect=OSError("layout read denied")):
                rc, _, stderr_text = _run_in_process([str(valid_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertIn("layout read denied", stderr_text)
            self.assertNotIn("Traceback", stderr_text)

    def test_shader_control_and_probe_archive_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, _ = self._create_valid_build(workspace, with_rpf=False)

            bad_control = workspace / "no_dxil.bin"
            bad_control.write_bytes(b"not a dxil container")
            control_dir = workspace / "control_dir"
            control_dir.mkdir()

            for label, arg_path in (
                ("missing", workspace / "missing_control.bin"),
                ("directory", control_dir),
                ("no_dxil", bad_control),
            ):
                with self.subTest(shader_control=label):
                    rc, _, stderr_text = _run_in_process(
                        [str(build_dir), "--shader-control", str(arg_path)],
                        tool_path=tool_copy,
                    )
                    self.assertEqual(rc, 1)
                    self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
                    self.assertNotIn("Traceback", stderr_text)

            # 0-byte RPF9 archive triggers ValueError in probe_archive -> exit 1.
            (build_dir / "shaders.rpf").write_bytes(b"")
            rc, _, stderr_text = _run_in_process([str(build_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertIn("Empty archive", stderr_text)
            self.assertNotIn("Traceback", stderr_text)

            # OSError from probe_archive -> exit 1.
            (build_dir / "shaders.rpf").write_bytes(b"9FPR" + bytes(16))
            with mock.patch.object(resource_inventory, "probe_archive", side_effect=OSError("archive io error")):
                rc, _, stderr_text = _run_in_process([str(build_dir)], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertIn("archive io error", stderr_text)
            self.assertNotIn("Traceback", stderr_text)

    def test_scene_records_and_validator_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, _ = self._create_valid_build(workspace, with_rpf=False)

            # Missing shaders.rpf with --scene-records -> exit 1.
            rc, _, stderr_text = _run_in_process([str(build_dir), "--scene-records"], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertNotIn("Traceback", stderr_text)

            # shaders.rpf is a directory -> exit 1.
            rpf_dir = build_dir / "shaders.rpf"
            rpf_dir.mkdir()
            rc, _, stderr_text = _run_in_process([str(build_dir), "--scene-records"], tool_path=tool_copy)
            self.assertEqual(rc, 1)
            self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
            self.assertNotIn("Traceback", stderr_text)
            rpf_dir.rmdir()

            # Restore valid shaders.rpf for validator guard tests.
            build_dir, _ = self._create_valid_build(workspace, with_rpf=True)
            val_dir = workspace / "val_dir"
            val_dir.mkdir()
            fail_val = _make_validator_command(workspace, "fail_val", "import sys\nsys.exit(3)\n")
            bad_json_val = _make_validator_command(workspace, "bad_json", 'print("{not json")\n')
            missing_key_val = _make_validator_command(workspace, "missing_key", 'print("{}")\n')

            for label, val_path in (
                ("missing_validator", workspace / "missing_validator.exe"),
                ("dir_validator", val_dir),
                ("failing_validator", fail_val),
                ("bad_json_validator", bad_json_val),
                ("missing_key_validator", missing_key_val),
            ):
                with self.subTest(validator=label):
                    rc, _, stderr_text = _run_in_process(
                        [str(build_dir), "--scene-validator", str(val_path)],
                        tool_path=tool_copy,
                    )
                    self.assertEqual(rc, 1)
                    self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
                    self.assertNotIn("Traceback", stderr_text)

            # Guarded exception types in scene_records -> exit 1 without traceback.
            dummy_val = _make_validator_command(workspace, "dummy_val", "import sys\nsys.exit(0)\n")
            for exc in (
                OSError("mmap failure"),
                ValueError("bad value"),
                struct.error("short unpack"),
                subprocess.TimeoutExpired(cmd="val", timeout=15),
                json.JSONDecodeError("bad", "doc", 0),
            ):
                with self.subTest(scene_exc=type(exc).__name__):
                    with mock.patch.object(resource_inventory, "scene_records", side_effect=exc):
                        rc, _, stderr_text = _run_in_process(
                            [str(build_dir), "--scene-records", "--scene-validator", str(dummy_val)],
                            tool_path=tool_copy,
                        )
                    self.assertEqual(rc, 1)
                    self.assertTrue(stderr_text.startswith("resource_inventory: "), stderr_text)
                    self.assertNotIn("Traceback", stderr_text)

    def test_subprocess_negative_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            tool_copy = _copy_isolated_tools(workspace)
            build_dir, _ = self._create_valid_build(workspace, with_rpf=False)
            (build_dir / "shaders.rpf").write_bytes(b"")

            proc = subprocess.run(
                [sys.executable, "-B", str(tool_copy), str(build_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(proc.stderr.startswith("resource_inventory: "), proc.stderr)
            self.assertIn("Empty archive", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
