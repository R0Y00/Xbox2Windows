"""CLI, path-collision, error-guard, and mutation control suite for tools/startup/archive_requests.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
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
TOOL = ROOT / "tools" / "startup" / "archive_requests.py"
sys.path.insert(0, str(ROOT / "tools" / "startup"))

import archive_requests  # noqa: E402
import gfx2hlsl  # noqa: E402

PREFETCH = bytes.fromhex("0300a0bf")
CODE_END = bytes.fromhex("00009fbf")
ENDPGM = struct.pack("<I", 0xBF810000)


def _program(stage: str, serial: int, *, blocks: int = 1) -> bytes:
    target = {"vs": 12, "ps": 0}[stage]
    header = b"\x00\x00\x80\xd5" + struct.pack("<I", serial) + bytes(28)
    body = header + PREFETCH + struct.pack("<2I", 0xF800080F | (target << 4), 0) + ENDPGM
    size = 0x40 * blocks
    return body + CODE_END * ((size - len(body)) // 4)


def _stream(layout: list[tuple[int, bytes]], *, length: int = 0x400) -> bytes:
    buf = bytearray(length)
    for at, blob in layout:
        buf[at:at + len(blob)] = blob
    return bytes(buf)


def _synthetic_pair_archive() -> tuple[bytes, bytes, bytes]:
    ps_blob = _program("ps", 1)
    vs_blob = _program("vs", 2, blocks=2)
    payload = _stream([(0x2C, ps_blob), (0x12C, vs_blob)])
    return zlib.compress(payload, 9), ps_blob, vs_blob


def _parse_stats(line: str) -> dict[str, int]:
    tokens = line.strip().split()
    return {key: int(value) for key, value in (tok.split("=", 1) for tok in tokens)}


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = archive_requests.main(argv)
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


class ArchiveRequestsCliPositiveTests(unittest.TestCase):
    def test_main_writes_requests_returns_zero_and_preserves_stdout_stats(self) -> None:
        archive_bytes, ps_blob, vs_blob = _synthetic_pair_archive()
        root_bytes = b"DXBC-synthetic-root-one"
        expected_key = gfx2hlsl.pipeline_key(ps_blob, vs_blob, root_bytes)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            roots_dir = root / "roots"
            roots_dir.mkdir()
            (roots_dir / "req_0000000000000001.root").write_bytes(root_bytes)
            cache = root / "cache"

            argv = [str(archive), str(cache), "--roots", str(roots_dir)]
            code, out, err = _run_main(argv)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")

            stats = _parse_stats(out)
            self.assertEqual(
                list(stats),
                [
                    "streams", "oversized", "programs", "incomplete", "ps", "vs",
                    "other", "undecodable", "pairs", "written", "existing", "failed", "roots",
                ],
            )
            self.assertEqual(
                (stats["streams"], stats["pairs"], stats["written"], stats["existing"], stats["failed"], stats["roots"]),
                (1, 1, 1, 0, 0, 1),
            )
            stem = cache / f"req_{expected_key:016x}"
            self.assertEqual(stem.with_suffix(".a").read_bytes(), ps_blob)
            self.assertEqual(stem.with_suffix(".b").read_bytes(), vs_blob)
            self.assertEqual(stem.with_suffix(".root").read_bytes(), root_bytes)
            self.assertEqual(list(cache.glob("*.tmp")), [])

            # Second run via subprocess reports existing=1, written=0, exit 0.
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            again_stats = _parse_stats(proc.stdout)
            self.assertEqual((again_stats["written"], again_stats["existing"], again_stats["roots"]), (0, 1, 1))

    def test_main_forwards_sys_argv_default_cache_roots_and_script_entrypoint(self) -> None:
        archive_bytes, ps_blob, vs_blob = _synthetic_pair_archive()
        root_bytes = b"DXBC-default-cache-root"
        expected_key = gfx2hlsl.pipeline_key(ps_blob, vs_blob, root_bytes)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            cache = root / "cache_with_root"
            cache.mkdir()
            (cache / "req_0000000000000000.root").write_bytes(root_bytes)

            cli_argv = [str(TOOL), str(archive), str(cache)]
            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(_parse_stats(out)["written"], 1)
            self.assertTrue((cache / f"req_{expected_key:016x}.a").is_file())

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
            self.assertEqual(_parse_stats(stdout.getvalue())["existing"], 1)


class ArchiveRequestsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int = 1,
        expected_fragments: tuple[str, ...] = (),
        expect_stdout_stats: bool = False,
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        if expect_stdout_stats:
            self.assertTrue(out.strip())
        else:
            self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected non-empty stderr for {argv!r}")
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            if expect_stdout_stats:
                self.assertTrue(proc.stdout.strip())
            else:
                self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected non-empty subprocess stderr for {argv!r}")
            for fragment in expected_fragments:
                self.assertIn(fragment, proc.stderr)

    def test_missing_and_directory_archive_rejected_upfront_without_reading_roots(self) -> None:
        # Kills removal of upfront archive.is_dir() and not archive.is_file() guards.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.rpf"
            cache = root / "cache_missing"
            with mock.patch.object(archive_requests, "read_roots") as read_roots_mock:
                self._assert_clean_failure(
                    [str(missing), str(cache)],
                    expected_fragments=("Archive path is not a regular file",),
                    check_subprocess=False,
                )
            read_roots_mock.assert_not_called()
            self.assertFalse(cache.exists())
            self._assert_clean_failure(
                [str(missing), str(cache)],
                expected_fragments=("Archive path is not a regular file",),
            )

            dir_archive = root / "dir_archive"
            dir_archive.mkdir()
            with mock.patch.object(archive_requests, "read_roots") as read_roots_mock:
                self._assert_clean_failure(
                    [str(dir_archive), str(cache)],
                    expected_fragments=("Archive path is a directory",),
                    check_subprocess=False,
                )
            read_roots_mock.assert_not_called()
            self._assert_clean_failure(
                [str(dir_archive), str(cache)],
                expected_fragments=("Archive path is a directory",),
            )

            # Direct streams() API checks.
            with self.assertRaisesRegex(ValueError, "directory"):
                list(archive_requests.streams(dir_archive, {}))
            with self.assertRaisesRegex(ValueError, "Missing archive"):
                list(archive_requests.streams(missing, {}))

    def test_cache_regular_file_and_collisions_rejected_upfront(self) -> None:
        # Kills removal of cache.is_dir() and _same_path() resolve()/samefile() checks.
        archive_bytes, _, _ = _synthetic_pair_archive()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            sub = root / "sub"
            sub.mkdir()
            roots_dir = root / "roots"
            roots_dir.mkdir()
            root_file = roots_dir / "req_0000000000000001.root"
            root_file.write_bytes(b"DXBC-root")

            file_cache = root / "regular_file.bin"
            file_cache.write_bytes(b"keep-me")
            with mock.patch.object(archive_requests, "read_roots") as read_roots_mock:
                self._assert_clean_failure(
                    [str(archive), str(file_cache), "--roots", str(roots_dir)],
                    expected_fragments=("Cache path is not a directory",),
                    check_subprocess=False,
                )
            read_roots_mock.assert_not_called()
            self.assertEqual(file_cache.read_bytes(), b"keep-me")
            self._assert_clean_failure(
                [str(archive), str(file_cache), "--roots", str(roots_dir)],
                expected_fragments=("Cache path is not a directory",),
            )

            # 1) Cache colliding with archive via relative alias (resolve()) and hardlink (samefile()).
            archive_alias = sub / ".." / "shaders.rpf"
            hardlink_archive = root / "shaders_hardlink.rpf"
            os.link(archive, hardlink_archive)
            self.assertNotEqual(archive.resolve(), hardlink_archive.resolve())
            self.assertTrue(archive.samefile(hardlink_archive))

            for bad_cache in (archive_alias, hardlink_archive):
                with self.subTest(bad_cache=str(bad_cache)):
                    with mock.patch.object(archive_requests, "read_roots") as read_roots_mock:
                        self._assert_clean_failure(
                            [str(archive), str(bad_cache), "--roots", str(roots_dir)],
                            expected_fragments=("Cache directory collides with archive",),
                            check_subprocess=False,
                        )
                    read_roots_mock.assert_not_called()
                    self._assert_clean_failure(
                        [str(archive), str(bad_cache), "--roots", str(roots_dir)],
                        expected_fragments=("Cache directory collides with archive",),
                    )

            # 2) Cache colliding with --roots directory or file (including non-existent resolve() alias & hardlink).
            roots_alias = sub / ".." / "roots"
            ghost_roots = root / "ghost_roots"
            ghost_cache = sub / ".." / "ghost_roots"
            hardlink_root = root / "root_hardlink.root"
            os.link(root_file, hardlink_root)
            for cache_arg, roots_arg in (
                (roots_alias, roots_dir),
                (ghost_cache, ghost_roots),
                (hardlink_root, root_file),
            ):
                with self.subTest(cache=str(cache_arg), roots=str(roots_arg)):
                    with mock.patch.object(archive_requests, "read_roots") as read_roots_mock:
                        self._assert_clean_failure(
                            [str(archive), str(cache_arg), "--roots", str(roots_arg)],
                            expected_fragments=("Cache directory collides with --roots",),
                            check_subprocess=False,
                        )
                    read_roots_mock.assert_not_called()
                    self._assert_clean_failure(
                        [str(archive), str(cache_arg), "--roots", str(roots_arg)],
                        expected_fragments=("Cache directory collides with --roots",),
                    )

    def test_roots_validation_and_write_failure_atomic_cleanup(self) -> None:
        archive_bytes, ps_blob, vs_blob = _synthetic_pair_archive()
        root_bytes = b"DXBC-root"
        key = gfx2hlsl.pipeline_key(ps_blob, vs_blob, root_bytes)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            absent_roots = root / "absent_roots"
            cache = root / "cache"

            with self.assertRaisesRegex(ValueError, "root signature path not found"):
                archive_requests.read_roots([str(absent_roots)])

            self._assert_clean_failure(
                [str(archive), str(cache), "--roots", str(absent_roots)],
                expected_fragments=("root signature path not found",),
            )
            self.assertFalse(cache.exists())

            # Empty roots directory -> "no root signature found", cache untouched.
            empty_roots = root / "empty_roots"
            empty_roots.mkdir()
            self._assert_clean_failure(
                [str(archive), str(cache), "--roots", str(empty_roots)],
                expected_fragments=("no root signature found",),
            )
            self.assertFalse(cache.exists())

            # Directory matching req_*.root inside --roots -> clean ValueError, exit 1.
            bad_roots = root / "bad_roots"
            (bad_roots / "req_0000000000000000.root").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "directory"):
                archive_requests.read_roots([str(bad_roots)])
            self._assert_clean_failure(
                [str(archive), str(cache), "--roots", str(bad_roots)],
                expected_fragments=("root signature path is a directory",),
            )

            # Blocked .root destination leaves 0 *.tmp and 0 .a, prints stats, exits 1.
            valid_roots = root / "valid_roots"
            valid_roots.mkdir()
            (valid_roots / "req_0000000000000001.root").write_bytes(root_bytes)
            blocked_cache = root / "blocked_cache"
            blocked_cache.mkdir()
            (blocked_cache / f"req_{key:016x}.root").mkdir()
            self._assert_clean_failure(
                [str(archive), str(blocked_cache), "--roots", str(valid_roots)],
                expected_fragments=("could not be written",),
                expect_stdout_stats=True,
            )
            self.assertFalse((blocked_cache / f"req_{key:016x}.a").exists())
            self.assertEqual(list(blocked_cache.glob("*.tmp")), [])

            # Archive without any pair -> prints stats, exits 1.
            nopair = root / "nopair.rpf"
            nopair.write_bytes(zlib.compress(_stream([(0x2C, ps_blob)]), 9))
            self._assert_clean_failure(
                [str(nopair), str(root / "nopair_cache"), "--roots", str(valid_roots)],
                expected_fragments=("no program pair found",),
                expect_stdout_stats=True,
            )

    def test_oserror_struct_error_and_zlib_error_guards_in_main(self) -> None:
        # Kills removal of OSError, struct.error, and zlib.error from except tuple in main().
        archive_bytes, _, _ = _synthetic_pair_archive()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "shaders.rpf"
            archive.write_bytes(archive_bytes)
            roots_dir = root / "roots"
            roots_dir.mkdir()
            (roots_dir / "req_0000000000000001.root").write_bytes(b"DXBC-root")
            cache = root / "cache"
            argv = [str(archive), str(cache), "--roots", str(roots_dir)]

            with mock.patch.object(archive_requests.os, "makedirs", side_effect=OSError("simulated mkdir failure")):
                self._assert_clean_failure(
                    argv,
                    expected_fragments=("simulated mkdir failure",),
                    check_subprocess=False,
                )

            for exc_obj, fragment in (
                (struct.error("unpack_from requires a buffer of at least 4 bytes"), "unpack_from requires a buffer"),
                (zlib.error("Error -3 while decompressing data: incorrect header check"), "Error -3 while decompressing data"),
            ):
                with self.subTest(exc=type(exc_obj).__name__):
                    with mock.patch.object(archive_requests, "streams", side_effect=exc_obj):
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=(fragment,),
                            check_subprocess=False,
                        )

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "empty.rpf"
            archive.write_bytes(b"")
            for bad_argv in ([], [str(archive)], [str(archive), str(Path(tmp) / "cache"), "--unknown-flag"]):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(bad_argv, expected_code=2)


if __name__ == "__main__":
    unittest.main()
