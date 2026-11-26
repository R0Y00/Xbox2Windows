"""CLI, empty-archive, path/collision, error-guard, and mutation suite for tools/compute_container_index.py."""

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
TOOL = ROOT / "tools" / "compute_container_index.py"
sys.path.insert(0, str(ROOT / "tools"))

import compute_container_index  # noqa: E402


def _build_compute_container(hash_part: bytes | None, *, stage: int = 5) -> bytes:
    program = struct.pack("<II4sIII", stage << 16, 7, b"DXIL", 0, 16, 4) + b"stub"
    parts = [(b"DXIL", program)]
    if hash_part is not None:
        parts.append((b"HASH", hash_part))
    pos = 32 + 4 * len(parts)
    offsets = []
    payload = bytearray()
    for name, part in parts:
        offsets.append(pos + len(payload))
        payload.extend(struct.pack("<4sI", name, len(part)) + part)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<III", 1, pos + len(payload), len(parts))
        + struct.pack(f"<{len(offsets)}I", *offsets)
        + payload
    )


def _archive_record(index: int, blob: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(blob) + compressor.flush()
    size = len(compressed)
    return f"fixture-{index}".encode("ascii") + b"\x00\x00" + struct.pack("<4I", size, 0, size, len(blob)) + compressed


def _slot_data(digest: bytes) -> bytes:
    return struct.pack("<III", 0x403, 24, 0) + struct.pack("<I", 0x43) + digest + bytes(12)


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = compute_container_index.main(argv)
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


class ComputeContainerIndexCliPositiveTests(unittest.TestCase):
    def _assert_schema(self, doc: dict, *, selection: str) -> None:
        self.assertEqual(set(doc), {"selection", "summary", "requests"})
        self.assertEqual(doc["selection"], selection)
        self.assertEqual(
            set(doc["summary"]),
            {
                "cache_requests",
                "selected_keys",
                "request_statuses",
                "archive_records",
                "archive_framing_rejections",
            },
        )

    def test_empty_archive_and_nonempty_cache_modes_return_zero_json(self) -> None:
        digest = b"\x11" * 16
        blob = _build_compute_container(bytes(4) + digest)
        expected_sha = hashlib.sha256(blob).hexdigest()
        key_hex = f"{1:016x}"

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / f"req_{key_hex}.c67").write_bytes(b"slot67")
            (cache / f"req_{key_hex}.data").write_bytes(_slot_data(digest))

            # 1) 0-byte archive must not crash mmap.mmap(..., 0); passes b"" to scene_records.
            empty_archive = root / "empty.bin"
            empty_archive.write_bytes(b"")
            code, out, err = _run_main([str(empty_archive), str(cache)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            empty_doc = json.loads(out)
            self._assert_schema(empty_doc, selection="cache")
            self.assertEqual(empty_doc["summary"]["cache_requests"], 1)
            self.assertEqual(empty_doc["summary"]["selected_keys"], 1)
            self.assertEqual(empty_doc["summary"]["archive_records"], {})
            self.assertEqual(empty_doc["requests"][key_hex], {"status": "unmatched", "candidates": []})

            proc = _run_subprocess(str(empty_archive), str(cache))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), empty_doc)

            # 2) Non-empty archive with --trace returns matched unique candidate and dispatch_rows.
            archive = root / "shaders.bin"
            archive.write_bytes(_archive_record(0, blob))
            trace = root / "dispatch.log"
            trace.write_text(f"x2w-mesh-compute mode=direct key={key_hex} frame=1\n", encoding="utf-8")

            code, out, err = _run_main([str(archive), str(cache), "--trace", str(trace)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            traced_doc = json.loads(out)
            self._assert_schema(traced_doc, selection="dispatch-trace")
            self.assertEqual(
                traced_doc["requests"][key_hex],
                {
                    "status": "unique",
                    "dispatch_rows": {"direct": 1},
                    "candidates": [{"sha256": expected_sha, "hash_flags": 0, "records": 1}],
                },
            )
            self.assertEqual(sorted(p.name for p in cache.iterdir()), [f"req_{key_hex}.c67", f"req_{key_hex}.data"])

    def test_main_forwards_sys_argv_and_script_entrypoint_returns_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "empty.bin"
            archive.write_bytes(b"")
            cache = root / "cache"
            cache.mkdir()
            cli_argv = [str(TOOL), str(archive), str(cache)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self._assert_schema(json.loads(out), selection="cache")

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
            self._assert_schema(json.loads(stdout.getvalue()), selection="cache")


class ComputeContainerIndexCliNegativeAndMutationTests(unittest.TestCase):
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
        self.assertTrue(err.strip(), f"expected non-empty stderr for {argv!r}")
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected non-empty subprocess stderr for {argv!r}")
            for fragment in expected_fragments:
                self.assertIn(fragment, proc.stderr)

    def test_invalid_archive_cache_and_trace_paths_rejected_upfront(self) -> None:
        # Kills removal of _validate_inputs() archive/cache/trace and _same_path() guards.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            archive = root / "archive.bin"
            archive.write_bytes(b"")
            cache = root / "cache"
            cache.mkdir()
            regular_file = root / "regular.txt"
            regular_file.write_bytes(b"not-a-dir")
            missing = root / "missing"
            dir_path = root / "some_dir"
            dir_path.mkdir()

            cases = (
                ("dir_archive", [str(dir_path), str(cache)], "Archive path is a directory"),
                ("missing_archive", [str(missing), str(cache)], "Archive path is not a regular file"),
                ("missing_cache", [str(archive), str(missing)], "Cache directory does not exist"),
                ("file_cache", [str(archive), str(regular_file)], "Cache path is not a directory"),
                ("dir_trace", [str(archive), str(cache), "--trace", str(dir_path)], "Trace path is a directory"),
                ("missing_trace", [str(archive), str(cache), "--trace", str(missing)], "Trace path is not a regular file"),
                (
                    "trace_relative_alias_archive",
                    [str(archive), str(cache), "--trace", str(sub / ".." / "archive.bin")],
                    "Trace path collides with archive",
                ),
                (
                    "trace_nonexistent_alias_archive",
                    [str(root / "ghost.bin"), str(cache), "--trace", str(sub / ".." / "ghost.bin")],
                    "Trace path collides with archive",
                ),
            )
            for label, argv, fragment in cases:
                with self.subTest(label=label):
                    with mock.patch.object(compute_container_index, "build_index") as build_mock:
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=("ValueError:", fragment),
                            check_subprocess=False,
                        )
                    build_mock.assert_not_called()
                    self._assert_clean_failure(argv, expected_fragments=("ValueError:", fragment))

            # Hardlink --trace colliding with archive where resolve() differs (kills samefile()).
            hardlink_trace = root / "archive_hardlink.log"
            os.link(archive, hardlink_trace)
            self.assertNotEqual(archive.resolve(), hardlink_trace.resolve())
            self.assertTrue(archive.samefile(hardlink_trace))
            with mock.patch.object(compute_container_index, "build_index") as build_mock:
                self._assert_clean_failure(
                    [str(archive), str(cache), "--trace", str(hardlink_trace)],
                    expected_fragments=("ValueError:", "Trace path collides with archive"),
                    check_subprocess=False,
                )
            build_mock.assert_not_called()
            self._assert_clean_failure(
                [str(archive), str(cache), "--trace", str(hardlink_trace)],
                expected_fragments=("ValueError:", "Trace path collides with archive"),
            )

            # Direct build_index() call validates inputs too.
            with self.assertRaisesRegex(ValueError, "Trace path collides with archive"):
                compute_container_index.build_index(archive, cache, hardlink_trace)

    def test_oserror_struct_error_and_zlib_error_guards_in_main(self) -> None:
        # Kills removal of OSError, ValueError, struct.error, and zlib.error from except tuple in main().
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "empty.bin"
            archive.write_bytes(b"")
            cache = root / "cache"
            cache.mkdir()
            argv = [str(archive), str(cache)]

            for exc_obj, expected_type, fragment in (
                (OSError("simulated I/O failure"), "OSError", "simulated I/O failure"),
                (ValueError("simulated validation failure"), "ValueError", "simulated validation failure"),
                (struct.error("unpack_from requires a buffer of at least 4 bytes"), "error", "unpack_from requires a buffer"),
                (zlib.error("Error -3 while decompressing data: incorrect header check"), "error", "Error -3 while decompressing data"),
            ):
                with self.subTest(exc=expected_type):
                    with mock.patch.object(compute_container_index, "build_index", side_effect=exc_obj):
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=(f"{expected_type}:", fragment),
                            check_subprocess=False,
                        )

            # Real subprocess OSError when req_<key>.data is a directory.
            key_hex = f"{2:016x}"
            (cache / f"req_{key_hex}.c67").write_bytes(b"slot67")
            (cache / f"req_{key_hex}.data").mkdir()
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "empty.bin"
            archive.write_bytes(b"")
            for bad_argv in ([], [str(archive)], [str(archive), str(tmp), "--unknown-flag"]):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(bad_argv, expected_code=2)


if __name__ == "__main__":
    unittest.main()
