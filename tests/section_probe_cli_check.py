#!/usr/bin/env python3
"""CLI, error-guard, window-dedup, and mutation suite for tools/section_probe.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
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
TOOL = ROOT / "tools" / "section_probe.py"
sys.path.insert(0, str(ROOT / "tools"))

import section_probe  # noqa: E402

EXPECTED_KEYS = {"file", "bytes_sampled", "valid_zlib", "section_records", "layouts"}


def _make_doc(*items: tuple[bytes, bytes]) -> bytes:
    """Pack big-endian (4-byte tag, uint32 length, payload) section records."""
    parts = []
    for tag, payload in items:
        assert len(tag) == 4
        parts.append(struct.pack(">4sI", tag, 8 + len(payload)) + payload)
    return b"".join(parts)


def _pad16(blob: bytes) -> bytes:
    return blob + bytes(-len(blob) % 16)


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = section_probe.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


class SectionProbeCliPositiveTests(unittest.TestCase):
    def test_empty_single_byte_and_multi_archive_json_schema(self) -> None:
        doc = _make_doc((b"HEAD", b"abcd"), (b"FOOT", b""))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_path = root / "empty.bin"
            one_byte_path = root / "one.bin"
            valid_path = root / "valid.bin"
            empty_path.write_bytes(b"")
            one_byte_path.write_bytes(b"\x78")
            valid_path.write_bytes(zlib.compress(doc))

            code, out, err = _run_main([
                str(empty_path),
                str(one_byte_path),
                str(valid_path),
                "--first",
                "HEAD",
                "--last",
                "FOOT",
            ])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            reports = json.loads(out)
            self.assertEqual(len(reports), 3)
            for item in reports:
                self.assertEqual(set(item), EXPECTED_KEYS)
            self.assertEqual(
                reports[0],
                {"file": "empty.bin", "bytes_sampled": 0, "valid_zlib": 0, "section_records": 0, "layouts": {}},
            )
            self.assertEqual(
                reports[1],
                {"file": "one.bin", "bytes_sampled": 1, "valid_zlib": 0, "section_records": 0, "layouts": {}},
            )
            self.assertEqual(
                reports[2],
                {
                    "file": "valid.bin",
                    "bytes_sampled": valid_path.stat().st_size,
                    "valid_zlib": 1,
                    "section_records": 1,
                    "layouts": {"HEAD FOOT": 1},
                },
            )

            proc = _run_subprocess(empty_path, valid_path, "--first", "HEAD", "--last", "FOOT")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), [reports[0], reports[2]])
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["empty.bin", "one.bin", "valid.bin"])

    def test_valid_corrupt_preset_dict_and_unaligned_zlib_streams(self) -> None:
        matching_doc = _make_doc((b"HEAD", b"1234"), (b"BODY", b"5678"), (b"FOOT", b""))
        other_end_doc = _make_doc((b"HEAD", b"1234"), (b"TAIL", b""))
        bad_section_payload = b"HEAD\x00\x00\x00\x04"

        valid_match = zlib.compress(matching_doc)
        valid_other_end = zlib.compress(other_end_doc)
        valid_bad_sections = zlib.compress(bad_section_payload)
        corrupt_checksum = valid_match[:-1] + bytes([valid_match[-1] ^ 0xFF])
        truncated_stream = valid_match[:6]

        dict_comp = zlib.compressobj(wbits=15, zdict=b"HEAD BODY FOOT")
        preset_dict = dict_comp.compress(matching_doc) + dict_comp.flush()
        with mock.patch.object(section_probe.zlib, "decompressobj", side_effect=AssertionError("FDICT skipped")) as decomp_mock:
            self.assertEqual(section_probe.scan_window(preset_dict, 0, "HEAD", "FOOT"), (0, {}))
        decomp_mock.assert_not_called()

        # Place an otherwise valid stream at +1 misalignment inside a 16-byte chunk.
        unaligned_block = b"\x00" + valid_match

        archive = b"".join([
            _pad16(valid_match),
            _pad16(valid_other_end),
            _pad16(valid_bad_sections),
            _pad16(corrupt_checksum),
            _pad16(truncated_stream),
            _pad16(preset_dict),
            _pad16(unaligned_block),
        ])

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.bin"
            path.write_bytes(archive)

            direct = section_probe.inspect(path, "HEAD", "FOOT")
            self.assertEqual(direct["valid_zlib"], 3)
            self.assertEqual(direct["section_records"], 1)
            self.assertEqual(dict(direct["layouts"]), {"HEAD BODY FOOT": 1})

            proc = _run_subprocess(path, "--first", "HEAD", "--last", "FOOT")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(
                json.loads(proc.stdout),
                [{
                    "file": "mixed.bin",
                    "bytes_sampled": len(archive),
                    "valid_zlib": 3,
                    "section_records": 1,
                    "layouts": {"HEAD BODY FOOT": 1},
                }],
            )

    def test_multi_window_overlap_dedup_and_three_window_sampling(self) -> None:
        packet = zlib.compress(_make_doc((b"HEAD", b"w"), (b"FOOT", b"")))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 9 MiB file: all three 8 MiB windows overlap and merge into one [0, 9 MiB] span.
            overlap_size = 9 << 20
            overlap_buf = bytearray(overlap_size)
            for offset in (0, 2 << 20, 7 << 20, (8 << 20) + 1024):
                overlap_buf[offset:offset + len(packet)] = packet
            overlap_path = root / "overlap.bin"
            overlap_path.write_bytes(overlap_buf)

            overlap_report = section_probe.inspect(overlap_path, "HEAD", "FOOT")
            self.assertEqual(overlap_report["bytes_sampled"], overlap_size)
            self.assertEqual(overlap_report["valid_zlib"], 4)
            self.assertEqual(overlap_report["section_records"], 4)
            self.assertEqual(dict(overlap_report["layouts"]), {"HEAD FOOT": 4})

            # 25 MiB file: three disjoint 8 MiB windows at [0, 8M], [8.5M, 16.5M], [17M, 25M].
            large_size = 25 << 20
            large_buf = bytearray(large_size)
            mid_start = (large_size - (8 << 20)) // 2  # 8.5 MiB, 16-byte aligned
            sampled_offsets = (0, mid_start + 1024, 24 << 20)
            unsampled_gap_offset = (8 << 20) + 64
            for offset in (*sampled_offsets, unsampled_gap_offset):
                large_buf[offset:offset + len(packet)] = packet
            large_path = root / "large.bin"
            large_path.write_bytes(large_buf)

            large_report = section_probe.inspect(large_path, "HEAD", "FOOT")
            self.assertEqual(large_report["bytes_sampled"], 3 * (8 << 20))
            self.assertEqual(large_report["valid_zlib"], 3)
            self.assertEqual(large_report["section_records"], 3)
            self.assertEqual(dict(large_report["layouts"]), {"HEAD FOOT": 3})

    def test_main_sys_argv_forwarding_and_script_entrypoint(self) -> None:
        packet = zlib.compress(_make_doc((b"HEAD", b"ok"), (b"FOOT", b"")))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "entry.bin"
            path.write_bytes(packet)
            cli_argv = [str(TOOL), str(path), "--first", "HEAD", "--last", "FOOT"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(json.loads(out)[0]["section_records"], 1)

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
            self.assertEqual(json.loads(stdout.getvalue())[0]["section_records"], 1)


class SectionProbeCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv,
        *,
        expected_code=1,
        expected_fragment,
        require_prefix=True,
        check_subprocess=True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)
        if require_prefix:
            self.assertTrue(err.startswith("section_probe: "), err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)
            if require_prefix:
                self.assertTrue(proc.stderr.startswith("section_probe: "), proc.stderr)

    def test_missing_directory_and_non_regular_archive_refusals(self) -> None:
        packet = zlib.compress(_make_doc((b"HEAD", b"ok"), (b"FOOT", b"")))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path = root / "valid.bin"
            valid_path.write_bytes(packet)
            existing_dir = root / "dir_archive"
            existing_dir.mkdir()
            missing_path = root / "missing.bin"

            # Directories and missing files must be rejected by is_file() before open() or size stat runs.
            real_open = Path.open
            open_calls = []

            def guarded_open(self_path, *args, **kwargs):
                open_calls.append(self_path)
                return real_open(self_path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                with self.assertRaisesRegex(ValueError, "directory"):
                    section_probe.inspect(existing_dir, "HEAD", "FOOT")
                with self.assertRaises(FileNotFoundError):
                    section_probe.inspect(missing_path, "HEAD", "FOOT")
            self.assertEqual(open_calls, [])

            dir_mock = mock.MagicMock(spec=Path)
            dir_mock.is_file.return_value = False
            dir_mock.is_dir.return_value = True
            dir_mock.__str__.return_value = "mock_dir"
            with self.assertRaisesRegex(ValueError, "directory"):
                section_probe.inspect(dir_mock, "HEAD", "FOOT")
            dir_mock.stat.assert_not_called()
            dir_mock.open.assert_not_called()

            # Non-regular file (exists=True, is_dir=False, is_file=False) raises ValueError.
            special_path = mock.MagicMock(spec=Path)
            special_path.is_file.return_value = False
            special_path.is_dir.return_value = False
            special_path.exists.return_value = True
            special_path.__str__.return_value = "special_fifo"
            with self.assertRaisesRegex(ValueError, "regular file"):
                section_probe.inspect(special_path, "HEAD", "FOOT")
            special_path.stat.assert_not_called()
            special_path.open.assert_not_called()

            # CLI refuses directories (including ""), missing paths, and mixed lists with exit 1 and no stdout.
            for bad_arg, frag, subproc in (
                (str(existing_dir), "directory", True),
                ("", "directory", False),
                (str(missing_path), missing_path.name, True),
            ):
                with self.subTest(bad_arg=bad_arg):
                    self._assert_clean_failure(
                        [bad_arg, "--first", "HEAD", "--last", "FOOT"],
                        expected_fragment=frag,
                        check_subprocess=subproc,
                    )
                    self._assert_clean_failure(
                        [str(valid_path), bad_arg, "--first", "HEAD", "--last", "FOOT"],
                        expected_fragment=frag,
                        check_subprocess=subproc,
                    )

    def test_caught_exceptions_and_py314_argparse_errors_without_traceback(self) -> None:
        packet = zlib.compress(_make_doc((b"HEAD", b"ok"), (b"FOOT", b"")))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_path = root / "valid.bin"
            valid_path.write_bytes(packet)

            for exc in (
                OSError("simulated io failure"),
                ValueError("simulated value failure"),
                struct.error("simulated struct failure"),
                zlib.error("simulated zlib failure"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(section_probe, "inspect", side_effect=exc):
                        self._assert_clean_failure(
                            [str(valid_path), "--first", "HEAD", "--last", "FOOT"],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )

            # Invalid 4-char tags and missing/unknown argparse flags exit non-zero without tracebacks.
            for argv, frag in (
                ([str(valid_path), "--first", "head", "--last", "FOOT"], "section tag"),
                ([str(valid_path), "--first", "HEAD", "--last", "foot"], "section tag"),
                ([str(valid_path), "--first", "HEA", "--last", "FOOT"], "section tag"),
                ([str(valid_path), "--first", "HEADS", "--last", "FOOT"], "section tag"),
                ([str(valid_path), "--first", "HÉAD", "--last", "FOOT"], "section tag"),
                ([str(valid_path), "--first", "A-CD", "--last", "FOOT"], "section tag"),
                ([], "usage:"),
                ([str(valid_path)], "--first"),
                ([str(valid_path), "--first", "HEAD"], "--last"),
                ([str(valid_path), "--first", "HEAD", "--last", "FOOT", "--bogus"], "--bogus"),
            ):
                with self.subTest(argv=argv):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(stdout), redirect_stderr(stderr):
                            section_probe.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertIn(frag, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

            proc = _run_subprocess(valid_path, "--first", "bad", "--last", "FOOT")
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stdout, "")
            self.assertIn("section tag", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

            self.assertEqual(sorted(p.name for p in root.iterdir()), ["valid.bin"])


if __name__ == "__main__":
    unittest.main()
