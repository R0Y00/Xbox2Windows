"""CLI, atomic HLSL write, path-collision, error-guard, and mutation suite for tools/shader_profile_audit.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
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
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "shader_profile_audit.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import shader_profile_audit  # noqa: E402
import shader_profile_audit_check as fixture  # noqa: E402


def _synthetic_archive() -> tuple[bytes, dict[str, bytes]]:
    ps_ok = fixture.make_shader_blob(0, fixture.PS_GENERAL_TAG, marker=1)
    ps_cb = fixture.make_shader_blob(0, fixture.PS_CB_TAG, marker=2)
    ps_bad = fixture.make_shader_blob(0, fixture.PS_BAD_TAG, marker=3)
    cs_ok = fixture.make_shader_blob(5, fixture.CS_STORE_TAG, psv_group=(4, 1, 8), marker=4)
    ms_ok = fixture.make_shader_blob(13, fixture.MS_POINT_TAG, code_bytes=832, psv_group=(128, 1, 1), marker=5)
    vs_skip = fixture.make_shader_blob(1, fixture.PS_GENERAL_TAG, marker=6)
    archive = (
        fixture.rpf_record(ps_ok, "ps_ok")
        + fixture.rpf_record(ps_cb, "ps_cb")
        + fixture.rpf_record(ps_bad, "ps_bad")
        + fixture.rpf_record(cs_ok, "cs_ok")
        + fixture.rpf_record(ms_ok, "ms_ok")
        + fixture.rpf_record(vs_skip, "vs_skip")
    )
    blobs = {"ps_ok": ps_ok, "cs_ok": cs_ok, "ms_ok": ms_ok}
    return archive, blobs


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = shader_profile_audit.main(argv)
        except SystemExit as exc:
            return exc.code, stdout.getvalue(), stderr.getvalue()
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class ShaderProfileAuditCliPositiveTests(unittest.TestCase):
    def test_main_writes_hlsl_and_probe_atomically_and_returns_zero_json(self) -> None:
        archive_bytes, blobs = _synthetic_archive()
        expected_hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            source_dir = root / "nested" / "hlsl_out"
            self.assertFalse(source_dir.exists())

            seen_replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with (
                mock.patch.object(shader_profile_audit.pixel_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.mesh_entry_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main([str(archive), "--llvm", "stub", "--source-dir", str(source_dir), "--cb-reads"])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            payload = json.loads(out)
            self.assertEqual(set(payload), {"records", "hlsl_records", "profiles", "first_stops", "cb_reads"})
            self.assertEqual(payload["records"], {"PS": 3, "CS": 1, "MS": 1, "VS": 1})
            self.assertEqual(payload["hlsl_records"], {"PS": 1, "CS": 1, "MS": 1})
            self.assertEqual(payload["profiles"], {"PS": {"general": 1}, "CS": {"store": 1}, "MS": {"point mesh": 1}})
            self.assertEqual(payload["cb_reads"]["first_descriptor"], [["entry s9 + 0x0", 1]])

            expected_files = {
                source_dir / "general" / f"{expected_hashes['ps_ok']}.hlsl",
                source_dir / "store" / f"{expected_hashes['cs_ok']}.hlsl",
                source_dir / "point-mesh" / f"{expected_hashes['ms_ok']}.hlsl",
                source_dir / "s9-slot0-read-probe.hlsl",
            }
            self.assertEqual(len(seen_replaces), len(expected_files))
            self.assertEqual({dst.resolve() for _, dst in seen_replaces}, {p.resolve() for p in expected_files})
            for path in expected_files:
                self.assertTrue(path.is_file(), f"missing emitted file: {path}")
                self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [path.name] if path.parent != source_dir else sorted(["general", "store", "point-mesh", "s9-slot0-read-probe.hlsl"]))

            # Second run atomically overwrites stale contents with zero leftover temp dirs.
            for path in expected_files:
                path.write_text("STALE_CONTENT", encoding="utf-8")
            with (
                mock.patch.object(shader_profile_audit.pixel_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.mesh_entry_aot, "disassemble", side_effect=fixture.stub_disassemble),
            ):
                code, _, err = _run_main([str(archive), "--llvm", "stub", "--source-dir", str(source_dir), "--cb-reads"])
            self.assertEqual(code, 0, err)
            for path in expected_files:
                self.assertNotEqual(path.read_text(encoding="utf-8"), "STALE_CONTENT")
                if path.parent != source_dir:
                    self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])
            self.assertEqual(sorted(p.name for p in source_dir.iterdir()), ["general", "point-mesh", "s9-slot0-read-probe.hlsl", "store"])

    def test_main_forwards_default_sys_argv_and_script_entrypoint_returns_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "empty.rpf"
            archive.write_bytes(b"")
            cli_argv = [str(TOOL), str(archive), "--llvm", "stub"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(json.loads(out), {"records": {}, "hlsl_records": {}, "profiles": {}, "first_stops": {}})

            stdout = io.StringIO()
            stderr = io.StringIO()
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
            self.assertEqual(json.loads(stdout.getvalue()), {"records": {}, "hlsl_records": {}, "profiles": {}, "first_stops": {}})

            proc = _run_subprocess(str(archive), "--llvm", "stub", "--cb-reads")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("cb_reads", json.loads(proc.stdout))


class ShaderProfileAuditCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int = 1,
        expected_fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr output for {argv!r}")
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr output for {argv!r}")
            for fragment in expected_fragments:
                self.assertIn(fragment, proc.stderr)

    def test_missing_and_directory_archive_rejected_cleanly(self) -> None:
        # Kills removal of OSError/ValueError handling and args.archive.is_dir() guard.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.rpf"
            self._assert_clean_failure(
                [str(missing), "--llvm", "stub"],
                expected_fragments=("Shader profile audit:", "Missing archive file"),
            )

            dir_archive = root / "archive_dir"
            dir_archive.mkdir()
            argv = [str(dir_archive), "--llvm", "stub"]
            with mock.patch.object(shader_profile_audit, "scan") as scan_mock:
                self._assert_clean_failure(
                    argv,
                    expected_fragments=("Shader profile audit:", "Archive path is a directory"),
                    check_subprocess=False,
                )
            scan_mock.assert_not_called()

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Shader profile audit: Archive path is a directory", proc.stderr)

    def test_source_dir_regular_file_and_archive_collisions_rejected_upfront(self) -> None:
        # Kills removal of upfront --source-dir regular-file and resolve()/samefile() collision guards.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "empty.rpf"
            archive.write_bytes(b"")
            sub = root / "sub"
            sub.mkdir()

            existing_file = root / "not_a_dir.txt"
            existing_file.write_bytes(b"preserve-me")
            with mock.patch.object(shader_profile_audit, "scan") as scan_mock:
                self._assert_clean_failure(
                    [str(archive), "--llvm", "stub", "--source-dir", str(existing_file)],
                    expected_fragments=("Shader profile audit:", "Source directory is not a directory"),
                )
            scan_mock.assert_not_called()
            self.assertEqual(existing_file.read_bytes(), b"preserve-me")

            # 1) Relative alias to existing archive.
            alias = sub / ".." / "empty.rpf"
            with mock.patch.object(shader_profile_audit, "scan") as scan_mock:
                self._assert_clean_failure(
                    [str(archive), "--llvm", "stub", "--source-dir", str(alias)],
                    expected_fragments=("Shader profile audit:", "collides with archive"),
                )
            scan_mock.assert_not_called()

            # 2) Non-existent archive alias (kills mutating resolve() == resolve()).
            ghost = root / "ghost.rpf"
            ghost_alias = sub / ".." / "ghost.rpf"
            with mock.patch.object(shader_profile_audit, "scan") as scan_mock:
                self._assert_clean_failure(
                    [str(ghost), "--llvm", "stub", "--source-dir", str(ghost_alias)],
                    expected_fragments=("Shader profile audit:", "collides with archive"),
                )
            scan_mock.assert_not_called()

            # 3) Hardlink alias where resolve() differs (kills mutating samefile()).
            hardlink = root / "empty_hardlink.rpf"
            os.link(archive, hardlink)
            self.assertNotEqual(archive.resolve(), hardlink.resolve())
            self.assertTrue(archive.samefile(hardlink))
            with mock.patch.object(shader_profile_audit, "scan") as scan_mock:
                self._assert_clean_failure(
                    [str(archive), "--llvm", "stub", "--source-dir", str(hardlink)],
                    expected_fragments=("Shader profile audit:", "collides with archive"),
                )
            scan_mock.assert_not_called()

    def test_output_file_colliding_with_archive_rejected_without_overwrite(self) -> None:
        # Kills removal of per-destination archive collision guard.
        archive_bytes, _ = _synthetic_archive()
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp)
            archive = source_dir / "s9-slot0-read-probe.hlsl"
            archive.write_bytes(archive_bytes)
            with (
                mock.patch.object(shader_profile_audit.pixel_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.mesh_entry_aot, "disassemble", side_effect=fixture.stub_disassemble),
            ):
                self._assert_clean_failure(
                    [str(archive), "--llvm", "stub", "--source-dir", str(source_dir), "--cb-reads"],
                    expected_fragments=("Shader profile audit:", "Output file collides with archive"),
                    check_subprocess=False,
                )
            self.assertEqual(archive.read_bytes(), archive_bytes)

    def test_os_replace_failure_leaves_zero_partial_files_or_temp_dirs(self) -> None:
        archive_bytes, _ = _synthetic_archive()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            source_dir = root / "atomic_fail"
            replace_calls = []

            def failing_replace(src, dst):
                src_path = Path(src)
                replace_calls.append((src_path, Path(dst)))
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                raise OSError("simulated replace failure")

            with (
                mock.patch.object(shader_profile_audit.pixel_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.mesh_entry_aot, "disassemble", side_effect=fixture.stub_disassemble),
                mock.patch.object(shader_profile_audit.os, "replace", side_effect=failing_replace),
            ):
                self._assert_clean_failure(
                    [str(archive), "--llvm", "stub", "--source-dir", str(source_dir), "--cb-reads"],
                    expected_fragments=("Shader profile audit:", "simulated replace failure"),
                    check_subprocess=False,
                )

            self.assertEqual(len(replace_calls), 1)
            # Only the parent directory of the first target exists; 0 .hlsl files and 0 temp dirs remain.
            self.assertEqual(list(source_dir.rglob("*.hlsl")), [])
            for directory in [source_dir, *(p for p in source_dir.rglob("*") if p.is_dir())]:
                self.assertFalse(any(p.name.startswith("tmp") for p in directory.iterdir()), list(directory.iterdir()))

    def test_mid_scan_struct_and_zlib_errors_caught_without_partial_hlsl_or_traceback(self) -> None:
        # Kills removal of struct.error and zlib.error from except (OSError, ValueError, struct.error, zlib.error).
        archive_bytes, _ = _synthetic_archive()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)

            for exc_obj, fragment in (
                (struct.error("unpack_from requires a buffer of at least 4 bytes"), "unpack_from requires a buffer"),
                (zlib.error("Error -3 while decompressing data: incorrect header check"), "Error -3 while decompressing data"),
            ):
                source_dir = root / f"fail_{type(exc_obj).__name__}"
                with self.subTest(exc=type(exc_obj).__name__):
                    with (
                        mock.patch.object(shader_profile_audit.pixel_aot, "disassemble", side_effect=fixture.stub_disassemble),
                        mock.patch.object(shader_profile_audit.mesh_entry_aot, "disassemble", side_effect=fixture.stub_disassemble),
                        mock.patch.object(shader_profile_audit, "cb_diagnostic", side_effect=exc_obj),
                    ):
                        self._assert_clean_failure(
                            [str(archive), "--llvm", "stub", "--source-dir", str(source_dir), "--cb-reads"],
                            expected_fragments=("Shader profile audit:", fragment),
                            check_subprocess=False,
                        )
                    self.assertFalse(source_dir.exists())

            # Subprocess OSError check (zero traceback).
            missing = root / "missing_sub.rpf"
            proc = _run_subprocess(str(missing), "--llvm", "stub")
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("Shader profile audit:", proc.stderr)

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "empty.rpf"
            archive.write_bytes(b"")
            for bad_argv in ([], [str(archive)], [str(archive), "--llvm", "stub", "--unknown-flag"]):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(bad_argv, expected_code=2)


if __name__ == "__main__":
    unittest.main()
