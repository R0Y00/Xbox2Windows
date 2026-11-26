"""CLI, atomic report write, path/IO error guard, determinism, and mutation suite for tools/verify_build.py."""

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
TOOL = ROOT / "tools" / "verify_build.py"
sys.path.insert(0, str(ROOT / "tools"))

import verify_build  # noqa: E402

EXPECTED_REPORT_KEYS = {
    "started_utc",
    "directory",
    "torrent",
    "torrent_sha256",
    "payload_bytes",
    "payload_count",
    "disk_file_count",
    "missing",
    "size_mismatches",
    "extra_disk_files",
    "layout",
    "torrent_order",
    "pieces",
    "changed_during_verification",
    "torrent_payload_passed",
    "finished_utc",
}


def _bencode(value):
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b"e"
    if isinstance(value, bytes):
        return str(len(value)).encode("ascii") + b":" + value
    if isinstance(value, list):
        return b"l" + b"".join(_bencode(item) for item in value) + b"e"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        return b"d" + b"".join(_bencode(k) + _bencode(v) for k, v in items) + b"e"
    raise TypeError(type(value))


def _create_package(pkg: Path, *, corrupt_second: bool = False):
    pkg.mkdir(parents=True, exist_ok=True)
    sub = pkg / "sub"
    sub.mkdir(exist_ok=True)
    data_a = b"alpha-payload-1234"
    data_b = b"bravo-payload-5678"
    (pkg / "b.bin").write_bytes(data_b)
    (sub / "a.bin").write_bytes(data_a)
    layout_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<Package>\n"
        '  <Chunk Id="1000" Marker="Launch">\n'
        '    <FileGroup SourcePath="." DestinationPath="\\sub\\" Include="a.bin" />\n'
        '    <FileGroup SourcePath="." DestinationPath="." Include="b.bin" />\n'
        "  </Chunk>\n"
        "</Package>\n"
    )
    (pkg / "layout.xml").write_text(layout_xml, encoding="utf-8")

    piece_len = 16
    stream = data_a + data_b
    pieces = b"".join(
        hashlib.sha1(stream[i:i + piece_len]).digest()
        for i in range(0, len(stream), piece_len)
    )
    torrent = {
        b"info": {
            b"files": [
                {b"length": len(data_a), b"path": [b"sub", b"a.bin"]},
                {b"length": len(data_b), b"path": [b"b.bin"]},
            ],
            b"name": b"pkg",
            b"piece length": piece_len,
            b"pieces": pieces,
        }
    }
    (pkg / "pkg.torrent").write_bytes(_bencode(torrent))
    if corrupt_second:
        (pkg / "b.bin").write_bytes(b"X" + data_b[1:])


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = verify_build.main(argv)
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


class VerifyBuildCliPositiveTests(unittest.TestCase):
    def test_atomic_report_write_schema_and_fail_status_1(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "package"
            _create_package(pkg)
            out_file = root / "nested" / "reports" / "integrity.json"
            self.assertFalse(out_file.parent.exists())

            seen_replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                self.assertEqual(src_path.name, dst_path.name)
                return real_replace(src, dst)

            with mock.patch.object(verify_build.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([str(pkg), "--output", str(out_file)])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("Payload passed: True", out)
            self.assertEqual(len(seen_replaces), 1)
            self.assertEqual(sorted(p.name for p in out_file.parent.iterdir()), ["integrity.json"])

            doc = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(set(doc), EXPECTED_REPORT_KEYS)
            self.assertTrue(doc["torrent_payload_passed"])
            self.assertEqual(doc["payload_count"], 2)
            self.assertEqual(doc["missing"], [])
            self.assertEqual(doc["size_mismatches"], [])
            self.assertEqual(doc["pieces"]["issues"], [])
            self.assertEqual(doc["changed_during_verification"], [])

            # Subprocess run overwrites existing report atomically; corrupt payload exits 1 cleanly.
            (pkg / "b.bin").write_bytes(b"Z" * len(b"bravo-payload-5678"))
            proc = _run_subprocess(str(pkg), "--output", str(out_file))
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("Payload passed: False", proc.stdout)
            self.assertEqual(sorted(p.name for p in out_file.parent.iterdir()), ["integrity.json"])
            bad_doc = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertFalse(bad_doc["torrent_payload_passed"])
            self.assertTrue(bad_doc["pieces"]["issues"])

    def test_inventory_and_manifest_glob_are_sorted(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _create_package(pkg)
            (pkg / "m").mkdir()
            for rel in ("z_extra.bin", "m/z.bin", "a_extra.bin", "m/a.bin"):
                (pkg / rel).write_bytes(b"extra")

            expected_extra = ["a_extra.bin", "layout.xml", "m/a.bin", "m/z.bin", "pkg.torrent", "z_extra.bin"]
            keys = list(verify_build.inventory(pkg).keys())
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(
                keys,
                ["a_extra.bin", "b.bin", "layout.xml", "m/a.bin", "m/z.bin", "pkg.torrent", "sub/a.bin", "z_extra.bin"],
            )

            real_rglob = Path.rglob

            def reversed_rglob(path_obj, *args, **kwargs):
                return reversed(list(real_rglob(path_obj, *args, **kwargs)))

            with mock.patch.object(Path, "rglob", reversed_rglob):
                reversed_keys = list(verify_build.inventory(pkg).keys())
            self.assertEqual(reversed_keys, keys)

            real_glob = Path.glob

            def reversed_glob(path_obj, *args, **kwargs):
                return reversed(list(real_glob(path_obj, *args, **kwargs)))

            out_file = root / "out.json"
            with (
                mock.patch.object(Path, "rglob", reversed_rglob),
                mock.patch.object(Path, "glob", reversed_glob),
            ):
                code, _, err = _run_main([str(pkg), "--output", str(out_file)])
            self.assertEqual(code, 0, err)
            doc = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(doc["extra_disk_files"], expected_extra)

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _create_package(pkg)
            out_file = root / "entry.json"
            cli_argv = [str(TOOL), str(pkg), "--output", str(out_file)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("Payload passed: True", out)
            self.assertTrue(out_file.is_file())

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
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("Payload passed: True", stdout.getvalue())
            self.assertTrue(out_file.is_file())


class VerifyBuildCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(self, argv, *, expected_fragment, check_subprocess=True):
        code, _, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_argparse_errors_exit_nonzero_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _create_package(pkg)

            for argv, frag in (
                ([], "usage:"),
                ([str(pkg), "--no-such-flag"], "unrecognized"),
            ):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            verify_build.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self._assert_exit_2_clean(argv, expected_fragment=frag)

    def test_directory_regular_file_and_output_guards_kill_mutations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _create_package(pkg)
            regular_file = root / "not_a_dir.txt"
            regular_file.write_text("file-content\n", encoding="utf-8")
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            missing_dir = root / "missing_dir"

            # 1) Regular file passed as directory rejected upfront before verify_pieces.
            with mock.patch.object(verify_build, "verify_pieces") as vp_mock:
                self._assert_exit_2_clean(
                    [str(regular_file), "--output", str(root / "never.json")],
                    expected_fragment="not a directory",
                )
            vp_mock.assert_not_called()

            # 2) Missing directory rejected cleanly.
            with mock.patch.object(verify_build, "verify_pieces") as vp_mock:
                self._assert_exit_2_clean(
                    [str(missing_dir)],
                    expected_fragment="cannot read",
                )
            vp_mock.assert_not_called()

            # 3) Output inside input directory rejected.
            with mock.patch.object(verify_build, "verify_pieces") as vp_mock:
                self._assert_exit_2_clean(
                    [str(pkg), "--output", str(pkg / "inside.json")],
                    expected_fragment="outside the input directory",
                )
            vp_mock.assert_not_called()

            # 4) Output is an existing directory (including "") rejected before verify_pieces.
            for bad_out in (str(existing_dir), ""):
                with mock.patch.object(verify_build, "verify_pieces") as vp_mock:
                    self._assert_exit_2_clean(
                        [str(pkg), "--output", bad_out],
                        expected_fragment="existing directory",
                        check_subprocess=bool(bad_out),
                    )
                vp_mock.assert_not_called()

    def test_verify_pieces_and_atomic_write_os_value_errors_preserve_prior_and_leave_zero_temp_dirs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _create_package(pkg)
            out_dir = root / "reports"
            out_dir.mkdir()
            out_file = out_dir / "integrity.json"
            out_file.write_text("prior-report\n", encoding="utf-8")

            # 1) verify_pieces OSError & ValueError both exit 2 cleanly and preserve prior report.
            for exc in (OSError("simulated read error"), ValueError("Piece count disagrees")):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(verify_build, "verify_pieces", side_effect=exc):
                        self._assert_exit_2_clean(
                            [str(pkg), "--output", str(out_file)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )
                    self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")

            # 2) os.replace OSError on pre-existing output -> prior intact, zero temp dirs.
            def failing_replace(src, dst):
                src_path = Path(src)
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                raise OSError("simulated replace failure")

            with mock.patch.object(verify_build.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    [str(pkg), "--output", str(out_file)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["integrity.json"])

            # 3) os.replace OSError on fresh output -> zero files & zero temp dirs.
            out_file.unlink()
            with mock.patch.object(verify_build.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    [str(pkg), "--output", str(out_file)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [])

            # 4) Staging write_text OSError -> os.replace never called, prior intact, zero temp dirs.
            out_file.write_text("prior-report\n", encoding="utf-8")
            with (
                mock.patch.object(verify_build.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated stage write failure")),
            ):
                self._assert_exit_2_clean(
                    [str(pkg), "--output", str(out_file)],
                    expected_fragment="simulated stage write failure",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["integrity.json"])

            # 5) Serialization ValueError -> os.replace never called, prior intact, zero temp dirs.
            real_dumps = json.dumps
            dump_calls = 0

            def fail_second_dumps(*args, **kwargs):
                nonlocal dump_calls
                dump_calls += 1
                if dump_calls >= 2:
                    raise ValueError("simulated json value error")
                return real_dumps(*args, **kwargs)

            with (
                mock.patch.object(verify_build.os, "replace") as replace_mock,
                mock.patch.object(verify_build.json, "dumps", side_effect=fail_second_dumps),
            ):
                self._assert_exit_2_clean(
                    [str(pkg), "--output", str(out_file)],
                    expected_fragment="simulated json value error",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["integrity.json"])


if __name__ == "__main__":
    unittest.main()
