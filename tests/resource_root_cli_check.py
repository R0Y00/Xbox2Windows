"""CLI, bounds-validation, error-guard, and mutation control suite for tools/resource_root.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "resource_root.py"
sys.path.insert(0, str(ROOT / "tools"))

import resource_root  # noqa: E402

BASE = 0x50000000
ROOT_TYPE = 0x1234ABCD
ALT_TYPE = 0x00008888
TWO_MIB = 2 << 20


def _build_prd0(
    total_size: int = 400,
    *,
    root_index: int = 2,
    root_type: int = ROOT_TYPE,
    root_bytes: int = 32,
    alt_type: int = ALT_TYPE,
    alt_bytes: int = 16,
) -> bytearray:
    """Build a synthetic relocated PRD0 v0x7a resource buffer."""
    data = bytearray(total_size)
    struct.pack_into("<Q4sH", data, 8, BASE + 112, b"0DRP", 0x7A)
    struct.pack_into("<I", data, 32, root_index)
    struct.pack_into("<Q", data, 40, BASE + 160)
    struct.pack_into("<Q", data, 56, BASE + 256)
    struct.pack_into("<3H", data, 72, 2, 0, 2)
    for address, kind, size in ((160, root_type, root_bytes), (192, alt_type, alt_bytes)):
        struct.pack_into("<I", data, address, kind)
        struct.pack_into("<I", data, address + 24, size)
    struct.pack_into("<IIQIIQ", data, 256, alt_type, alt_bytes, BASE + 304, root_type, root_bytes, BASE + 352)
    return data


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = resource_root.main(argv)
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


class ResourceRootCliPositiveTests(unittest.TestCase):
    def _assert_json_schema(self, stdout_text: str, expected_hash: str, expected_bytes: int) -> None:
        payload = json.loads(stdout_text)
        self.assertEqual(set(payload), {"root_type_hash", "root_bytes"})
        self.assertRegex(payload["root_type_hash"], r"^0x[0-9a-f]{8}$")
        self.assertIs(type(payload["root_bytes"]), int)
        self.assertEqual(payload, {"root_type_hash": expected_hash, "root_bytes": expected_bytes})

    def test_zero_and_nonzero_decimal_and_hex_arguments_return_zero_and_exact_json(self) -> None:
        primary = bytes(_build_prd0(root_index=2))
        primary_packed = zlib.compress(primary)
        alternate = bytes(_build_prd0(root_index=1))
        alternate_packed = zlib.compress(alternate)

        cases = (
            (
                "zero_offset_decimal",
                b"",
                primary_packed,
                b"",
                "0",
                str(len(primary_packed)),
                str(len(primary)),
                "0x1234abcd",
                32,
            ),
            (
                "nonzero_decimal_offset_hex_sizes",
                b"PREFIX_13BYTE",
                alternate_packed,
                b"SUFFIX",
                "13",
                hex(len(alternate_packed)),
                hex(len(alternate)),
                "0x00008888",
                16,
            ),
            (
                "hex_offset_decimal_stored_hex_decoded",
                b"A" * 0x20,
                primary_packed,
                b"TAIL",
                "0x20",
                str(len(primary_packed)),
                hex(len(primary)),
                "0x1234abcd",
                32,
            ),
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, prefix, packed, suffix, offset_arg, stored_arg, decoded_arg, exp_hash, exp_bytes in cases:
                with self.subTest(label=label):
                    archive = root / f"{label}.bin"
                    archive.write_bytes(prefix + packed + suffix)
                    argv = [
                        str(archive),
                        "--offset",
                        offset_arg,
                        "--stored",
                        stored_arg,
                        "--decoded",
                        decoded_arg,
                    ]

                    code, out, err = _run_main(argv)
                    self.assertIs(type(code), int)
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    self._assert_json_schema(out, exp_hash, exp_bytes)

                    proc = _run_subprocess(*argv)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(proc.stderr, "")
                    self._assert_json_schema(proc.stdout, exp_hash, exp_bytes)

    def test_exact_two_mib_decoded_boundary_succeeds(self) -> None:
        decoded = bytes(_build_prd0(total_size=TWO_MIB))
        packed = zlib.compress(decoded)
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "exact_2mib.bin"
            archive.write_bytes(b"PAD" + packed)
            for decoded_arg in (str(TWO_MIB), hex(TWO_MIB)):
                with self.subTest(decoded_arg=decoded_arg):
                    argv = [
                        str(archive),
                        "--offset",
                        "3",
                        "--stored",
                        str(len(packed)),
                        "--decoded",
                        decoded_arg,
                    ]
                    code, out, err = _run_main(argv)
                    self.assertIs(type(code), int)
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    self._assert_json_schema(out, "0x1234abcd", 32)

                    proc = _run_subprocess(*argv)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    self.assertEqual(proc.stderr, "")
                    self._assert_json_schema(proc.stdout, "0x1234abcd", 32)

    def test_main_forwards_default_sys_argv_and_script_entrypoint_returns_zero(self) -> None:
        decoded = bytes(_build_prd0(root_index=2))
        packed = zlib.compress(decoded)
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "entry.bin"
            archive.write_bytes(b"HEAD" + packed)
            cli_argv = [
                str(TOOL),
                str(archive),
                "--offset",
                "0x4",
                "--stored",
                str(len(packed)),
                "--decoded",
                str(len(decoded)),
            ]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self._assert_json_schema(out, "0x1234abcd", 32)

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
            self._assert_json_schema(stdout.getvalue(), "0x1234abcd", 32)


class ResourceRootCliNegativeAndMutationTests(unittest.TestCase):
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

    def test_missing_and_directory_archives_rejected_cleanly(self) -> None:
        # Kills removal of OSError from except and removal of args.archive.is_dir() guard.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.bin"
            self._assert_clean_failure(
                [str(missing), "--offset", "0", "--stored", "16", "--decoded", "400"],
                expected_fragments=(f"{missing}: ",),
            )

            dir_archive = root / "archive_dir"
            dir_archive.mkdir()
            argv = [str(dir_archive), "--offset", "0", "--stored", "16", "--decoded", "400"]
            with mock.patch.object(resource_root, "read_member") as read_mock:
                self._assert_clean_failure(
                    argv,
                    expected_fragments=(f"{dir_archive}: ", "Archive path is a directory"),
                    check_subprocess=False,
                )
            read_mock.assert_not_called()

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(f"{dir_archive}: ", proc.stderr)
            self.assertIn("Archive path is a directory", proc.stderr)

    def test_negative_offset_rejected_before_opening_archive(self) -> None:
        # Testing existing + missing archives kills removal/reordering of args.offset < 0.
        decoded = bytes(_build_prd0())
        packed = zlib.compress(decoded)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.bin"
            existing.write_bytes(packed)
            missing = root / "missing.bin"
            for archive in (existing, missing):
                for offset_args in (["--offset", "-1"], ["--offset=-0x10"]):
                    with self.subTest(archive=archive.name, offset_args=offset_args):
                        argv = [
                            str(archive),
                            *offset_args,
                            "--stored",
                            str(len(packed)),
                            "--decoded",
                            str(len(decoded)),
                        ]
                        with mock.patch.object(resource_root, "read_member") as read_mock:
                            self._assert_clean_failure(
                                argv,
                                expected_fragments=(f"{archive}: ", "--offset", "non-negative"),
                                check_subprocess=False,
                            )
                        read_mock.assert_not_called()

            proc = _run_subprocess(
                str(missing),
                "--offset",
                "-1",
                "--stored",
                str(len(packed)),
                "--decoded",
                str(len(decoded)),
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(f"{missing}: ", proc.stderr)
            self.assertIn("--offset", proc.stderr)

    def test_non_positive_stored_and_decoded_rejected_before_opening_archive(self) -> None:
        # Testing 0 and negative values on existing + missing archives kills <= 0 -> < 0 and removal/reordering.
        decoded = bytes(_build_prd0())
        packed = zlib.compress(decoded)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.bin"
            existing.write_bytes(packed)
            missing = root / "missing.bin"
            for archive in (existing, missing):
                for stored_args in (["--stored", "0"], ["--stored", "-1"], ["--stored=-0x10"]):
                    with self.subTest(archive=archive.name, stored_args=stored_args):
                        argv = [
                            str(archive),
                            "--offset",
                            "0",
                            *stored_args,
                            "--decoded",
                            str(len(decoded)),
                        ]
                        with mock.patch.object(resource_root, "read_member") as read_mock:
                            self._assert_clean_failure(
                                argv,
                                expected_fragments=(f"{archive}: ", "--stored", "positive"),
                                check_subprocess=False,
                            )
                        read_mock.assert_not_called()

                for decoded_args in (["--decoded", "0"], ["--decoded", "-1"], ["--decoded=-0x10"]):
                    with self.subTest(archive=archive.name, decoded_args=decoded_args):
                        argv = [
                            str(archive),
                            "--offset",
                            "0",
                            "--stored",
                            str(len(packed)),
                            *decoded_args,
                        ]
                        with mock.patch.object(resource_root, "read_member") as read_mock:
                            self._assert_clean_failure(
                                argv,
                                expected_fragments=(f"{archive}: ", "--decoded", "positive"),
                                check_subprocess=False,
                            )
                        read_mock.assert_not_called()

            for flag in ("--stored", "--decoded"):
                proc = _run_subprocess(
                    str(missing),
                    "--offset",
                    "0",
                    "--stored",
                    "0" if flag == "--stored" else str(len(packed)),
                    "--decoded",
                    "0" if flag == "--decoded" else str(len(decoded)),
                )
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertEqual(proc.stdout, "")
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn(f"{missing}: ", proc.stderr)
                self.assertIn(flag, proc.stderr)

    def test_over_budget_stored_and_decoded_rejected(self) -> None:
        decoded = bytes(_build_prd0())
        packed = zlib.compress(decoded)
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "budget.bin"
            archive.write_bytes(packed)
            for stored_val, decoded_val in ((TWO_MIB + 1, len(decoded)), (len(packed), TWO_MIB + 1)):
                with self.subTest(stored=stored_val, decoded=decoded_val):
                    self._assert_clean_failure(
                        [
                            str(archive),
                            "--offset",
                            "0",
                            "--stored",
                            str(stored_val),
                            "--decoded",
                            str(decoded_val),
                        ],
                        expected_fragments=(f"{archive}: ", "2 MiB", "budget"),
                    )

    def test_corrupt_zlib_and_zlib_error_guard(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corrupt = root / "corrupt_zlib.bin"
            payload = b"not-a-valid-zlib-stream-at-all!"
            corrupt.write_bytes(payload)
            self._assert_clean_failure(
                [str(corrupt), "--offset", "0", "--stored", str(len(payload)), "--decoded", "400"],
                expected_fragments=(f"{corrupt}: ", "Invalid compressed segment"),
            )

            # Kills removal of zlib.error from except (OSError, ValueError, struct.error, zlib.error).
            valid_decoded = bytes(_build_prd0())
            valid_packed = zlib.compress(valid_decoded)
            valid = root / "valid.bin"
            valid.write_bytes(valid_packed)
            with mock.patch.object(
                resource_root,
                "read_member",
                side_effect=zlib.error("Error -3 while decompressing data: incorrect header check"),
            ):
                self._assert_clean_failure(
                    [
                        str(valid),
                        "--offset",
                        "0",
                        "--stored",
                        str(len(valid_packed)),
                        "--decoded",
                        str(len(valid_decoded)),
                    ],
                    expected_fragments=(f"{valid}: ", "Error -3 while decompressing data"),
                    check_subprocess=False,
                )

    def test_truncated_and_malformed_prd0_and_struct_error_guard(self) -> None:
        def patch_prd0(offset: int, fmt: str, value: int | bytes) -> bytes:
            buf = _build_prd0()
            struct.pack_into(fmt, buf, offset, value)
            return bytes(buf)

        bad_prd0_cases = (
            ("truncated_header", b"\x00" * 64, "PRD0 v0x7a"),
            ("bad_magic", patch_prd0(16, "<4s", b"BAD!"), "PRD0 v0x7a"),
            ("bad_version", patch_prd0(20, "<H", 0x79), "PRD0 v0x7a"),
            ("zero_root_index", patch_prd0(32, "<I", 0), "root index"),
            ("overlapping_extents", patch_prd0(8, "<Q", BASE + 160), "Overlapping"),
            ("schema_mismatch", patch_prd0(184, "<I", 64), "matching the root type"),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, raw_prd0, fragment in bad_prd0_cases:
                with self.subTest(label=label):
                    packed = zlib.compress(raw_prd0)
                    archive = root / f"{label}.bin"
                    archive.write_bytes(packed)
                    self._assert_clean_failure(
                        [
                            str(archive),
                            "--offset",
                            "0",
                            "--stored",
                            str(len(packed)),
                            "--decoded",
                            str(len(raw_prd0)),
                        ],
                        expected_fragments=(f"{archive}: ", fragment),
                    )

            # Kills removal of struct.error from except (OSError, ValueError, struct.error, zlib.error).
            valid_decoded = bytes(_build_prd0())
            valid_packed = zlib.compress(valid_decoded)
            valid = root / "valid_struct.bin"
            valid.write_bytes(valid_packed)
            with mock.patch.object(
                resource_root.struct,
                "unpack_from",
                side_effect=struct.error("unpack_from requires a buffer of at least 8 bytes"),
            ):
                self._assert_clean_failure(
                    [
                        str(valid),
                        "--offset",
                        "0",
                        "--stored",
                        str(len(valid_packed)),
                        "--decoded",
                        str(len(valid_decoded)),
                    ],
                    expected_fragments=(f"{valid}: ", "unpack_from requires a buffer of at least 8 bytes"),
                    check_subprocess=False,
                )

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "dummy.bin"
            archive.write_bytes(b"x")

            for missing_argv in (
                [str(archive)],
                [str(archive), "--offset", "0", "--stored", "1"],
                [str(archive), "--offset", "0", "--decoded", "1"],
                [str(archive), "--stored", "1", "--decoded", "1"],
            ):
                with self.subTest(missing_argv=missing_argv):
                    self._assert_clean_failure(missing_argv, expected_code=2)

            for flag in ("--offset", "--stored", "--decoded"):
                with self.subTest(invalid_flag=flag):
                    argv = [
                        str(archive),
                        "--offset",
                        "not_an_int" if flag == "--offset" else "0",
                        "--stored",
                        "1.5" if flag == "--stored" else "1",
                        "--decoded",
                        "xyz" if flag == "--decoded" else "1",
                    ]
                    self._assert_clean_failure(
                        argv,
                        expected_code=2,
                        expected_fragments=(flag,),
                    )


if __name__ == "__main__":
    unittest.main()
