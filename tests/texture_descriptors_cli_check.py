#!/usr/bin/env python3
"""CLI validation, multi-archive lookup, and error-exit suite for tools/texture_descriptors.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "texture_descriptors.py"
sys.path.insert(0, str(ROOT / "tools"))

from paged_resource import PagedResource  # noqa: E402
import texture_descriptors as td  # noqa: E402


def _pack_boundary(stored: int, logical: int, codec: int = 0, trailing_bytes: int = 0) -> bytes:
    assert stored % 256 == 0 and logical % 4096 == 0
    stored_units = stored // 256
    logical_pages = logical // 4096
    low = (stored_units & 0x03ffffff) | ((codec & 15) << 26) | ((logical_pages & 3) << 30)
    high = ((logical_pages >> 2) & 0xfffff) | ((trailing_bytes & 0xfff) << 20)
    return struct.pack("<II", low, high)


def _shape(width: int, height: int) -> int:
    return ((height - 1) << 14) | (width - 1)


def _flags(dxgi_format: int, mip_marker: int) -> int:
    return (mip_marker << 24) | (dxgi_format << 3)


def _build_synthetic_archive(
    *,
    keys: tuple[int, ...] = (0x0123456789abcdef, (1 << 64) - 1),
    raw_padding: int = 512,
) -> tuple[bytearray, bytes, bytes]:
    """Build a synthetic RDS5/ETPF archive with one padded raw range and one zlib range."""
    raw_payload = bytes((i * 17 + 5) % 256 for i in range(4096))
    zlib_plain = b"ZPAGE0_" + b"X" * 4089 + b"ZPAGE1_" + b"Y" * 4089
    zlib_packed = zlib.compress(zlib_plain)

    raw_start = 4352
    raw_end = raw_start + 4096 + raw_padding
    zlib_slot = ((len(zlib_packed) + 255) // 256) * 256
    total_size = raw_end + zlib_slot

    archive = bytearray(total_size)
    struct.pack_into("<4sIQ", archive, 0, b"RDS5", 4, 16384)
    archive[16:24] = _pack_boundary(256, 0, codec=0, trailing_bytes=0)
    archive[24:32] = _pack_boundary(raw_start, 4096, codec=0, trailing_bytes=raw_padding)
    archive[32:40] = _pack_boundary(raw_end, 8192, codec=8, trailing_bytes=0)
    archive[40:48] = _pack_boundary(total_size, 16384, codec=0, trailing_bytes=0)

    records = len(keys)
    struct.pack_into("<4sIII", archive, 256, b"ETPF", 0x27a, records, 0)
    struct.pack_into("<4I", archive, 272, _shape(64, 32), _flags(71, 3), 1, 0)
    if records > 1:
        struct.pack_into("<4I", archive, 288, _shape(128, 16), _flags(98, 1), 3, 0)
    struct.pack_into(f"<{records}Q", archive, 272 + records * 16, *keys)

    archive[raw_start:raw_start + 4096] = raw_payload
    archive[raw_end:raw_end + len(zlib_packed)] = zlib_packed
    return archive, raw_payload, zlib_plain


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = td.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TextureDescriptorsCliPositiveTests(unittest.TestCase):
    def test_direct_main_all_modes_and_boundary_keys(self) -> None:
        max_u64 = (1 << 64) - 1
        key_a = 0x0123456789ABCDEF
        arch_bytes_1, raw_payload, zlib_plain = _build_synthetic_archive(keys=(0, key_a))
        arch_bytes_2, _, _ = _build_synthetic_archive(keys=(key_a, max_u64))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arch1 = root / "first.ttpf"
            arch2 = root / "nested" / "second.ttpf"
            arch2.parent.mkdir()
            arch1.write_bytes(arch_bytes_1)
            arch2.write_bytes(arch_bytes_2)

            # 1) Inventory mode.
            code, out, err = _run_main([str(arch1)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(
                json.loads(out),
                {
                    "descriptor_count": 2,
                    "groups": [
                        {"width": 64, "height": 32, "dxgi_format": 71, "mip_marker": 3, "count": 1},
                        {"width": 128, "height": 16, "dxgi_format": 98, "mip_marker": 1, "count": 1},
                    ],
                },
            )

            # 2) --descriptor mode (0 and 1).
            code, out, err = _run_main([str(arch1), "--descriptor", "0"])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(
                json.loads(out),
                {
                    "descriptor_index": 0,
                    "width": 64,
                    "height": 32,
                    "dxgi_format": 71,
                    "mip_marker": 3,
                    "logical_offset": 4096,
                },
            )
            code, out, err = _run_main([str(arch1), "--descriptor", "1"])
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["logical_offset"], 12288)

            # 3) --raw-page-map mode.
            code, out, err = _run_main([str(arch1), "--raw-page-map", "0"])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            rpm = json.loads(out)
            self.assertEqual(rpm, {"logical_offset": 4096, "file_offset": 4352})
            self.assertEqual(arch_bytes_1[rpm["file_offset"]:rpm["file_offset"] + 4096], raw_payload)

            # 4) --zlib-segment mode.
            code, out, err = _run_main([str(arch1), "--zlib-segment", "1"])
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            zseg = json.loads(out)
            self.assertEqual(zseg["segment_logical_offset"], 8192)
            self.assertEqual(zseg["page_offset_in_segment"], 4096)
            self.assertEqual(zseg["segment"]["codec"], "zlib")
            self.assertEqual(zseg["segment"]["size"], 8192)
            with arch1.open("rb") as stream:
                seg_obj = td.zlib_segment(stream, 1)
                decoded = PagedResource(stream, [seg_obj.segment]).read(seg_obj.page_offset_in_segment, 4096)
            self.assertEqual(decoded, zlib_plain[4096:])

            # 5) --lookup-key decimal, 0x/0X hex, boundary 0 and 2^64-1, and absent key.
            for raw_key, expected in (
                (str(key_a), [(arch1, 1), (arch2, 0)]),
                (f"0x{key_a:x}", [(arch1, 1), (arch2, 0)]),
                (f"0X{key_a:X}", [(arch1, 1), (arch2, 0)]),
                ("0", [(arch1, 0)]),
                ("0x0", [(arch1, 0)]),
                (str(max_u64), [(arch2, 1)]),
                (hex(max_u64), [(arch2, 1)]),
                ("0xdeadbeefcafebabe", []),
            ):
                with self.subTest(raw_key=raw_key):
                    code, out, err = _run_main([str(arch1), str(arch2), "--lookup-key", raw_key])
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    self.assertEqual(
                        json.loads(out),
                        {
                            "matches": [
                                {"archive": str(path), "descriptor_index": index}
                                for path, index in expected
                            ]
                        },
                    )

    def test_sys_argv_and_script_entrypoint_and_subprocess(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            arch = Path(tmp) / "entry.ttpf"
            arch.write_bytes(arch_bytes)
            cli_argv = [str(TOOL), str(arch), "--descriptor", "0"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(json.loads(out)["descriptor_index"], 0)

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
            self.assertEqual(json.loads(stdout.getvalue())["width"], 64)

            proc = _run_subprocess(str(arch), "--lookup-key", "0x0123456789abcdef")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(
                json.loads(proc.stdout),
                {"matches": [{"archive": str(arch), "descriptor_index": 0}]},
            )


class TextureDescriptorsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(argv)
        if expected_code is None:
            self.assertNotEqual(code, 0, (code, err))
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertTrue(err.strip(), f"expected stderr for {argv!r}")
        self.assertNotIn("Traceback", err)
        err_lower = err.lower()
        for fragment in fragments:
            self.assertIn(fragment.lower(), err_lower)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0, proc.stderr)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr for {argv!r}")
            self.assertNotIn("Traceback", proc.stderr)
            sub_lower = proc.stderr.lower()
            for fragment in fragments:
                self.assertIn(fragment.lower(), sub_lower)
        return err

    def test_upfront_archive_existence_and_is_file_before_opening(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.ttpf"
            valid.write_bytes(arch_bytes)
            missing = root / "missing.ttpf"
            dir_arch = root / "dir_arch"
            dir_arch.mkdir()

            # Single-archive missing & directory checks never open files.
            for bad_path, frag in (
                (missing, f"{missing}: archive file not found"),
                (dir_arch, f"{dir_arch}: archive path is not a regular file"),
            ):
                with self.subTest(bad_path=bad_path):
                    with mock.patch.object(Path, "open") as open_mock:
                        self._assert_clean_failure(
                            [str(bad_path)],
                            expected_code=1,
                            fragments=(frag,),
                            check_subprocess=False,
                        )
                    open_mock.assert_not_called()
                    self._assert_clean_failure([str(bad_path)], expected_code=1, fragments=(frag,))

            # Multi-archive --lookup-key validates ALL archives before opening the first!
            for argv, frag in (
                ([str(valid), str(missing), "--lookup-key", "0"], f"{missing}: archive file not found"),
                ([str(valid), str(dir_arch), "--lookup-key", "0"], f"{dir_arch}: archive path is not a regular file"),
            ):
                with self.subTest(argv=argv):
                    with mock.patch.object(Path, "open") as open_mock:
                        self._assert_clean_failure(
                            argv,
                            expected_code=1,
                            fragments=(frag,),
                            check_subprocess=False,
                        )
                    open_mock.assert_not_called()
                    self._assert_clean_failure(argv, expected_code=1, fragments=(frag,))

    def test_lookup_key_range_validated_upfront_before_opening(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.ttpf"
            valid.write_bytes(arch_bytes)

            for argv_tail, frag in (
                (["--lookup-key", "-1"], "unsigned 64-bit integer"),
                (["--lookup-key=-0x1"], "invalid literal"),
                (["--lookup-key", str(1 << 64)], "unsigned 64-bit integer"),
                (["--lookup-key", hex(1 << 64)], "unsigned 64-bit integer"),
                (["--lookup-key", "not_a_key"], "invalid literal"),
                (["--lookup-key", ""], "invalid literal"),
            ):
                with self.subTest(argv_tail=argv_tail):
                    with mock.patch.object(Path, "open") as open_mock:
                        self._assert_clean_failure(
                            [str(valid), *argv_tail],
                            expected_code=1,
                            fragments=(f"{valid}:", frag),
                            check_subprocess=False,
                        )
                    open_mock.assert_not_called()
                    self._assert_clean_failure(
                        [str(valid), *argv_tail],
                        expected_code=1,
                        fragments=(f"{valid}:", frag),
                    )

    def test_negative_descriptor_index_validated_before_file_checks(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.ttpf"
            valid.write_bytes(arch_bytes)
            missing = root / "missing.ttpf"

            for target in (valid, missing):
                for flag in ("--descriptor", "--raw-page-map", "--zlib-segment"):
                    with self.subTest(target=target.name, flag=flag):
                        with mock.patch.object(Path, "open") as open_mock:
                            self._assert_clean_failure(
                                [str(target), flag, "-1"],
                                expected_code=1,
                                fragments=(f"{target}:", "nonnegative"),
                                check_subprocess=False,
                            )
                        open_mock.assert_not_called()

    def test_truncated_corrupt_zlib_and_duplicate_archive_failures(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.ttpf"
            valid.write_bytes(arch_bytes)

            empty = root / "empty.ttpf"
            empty.write_bytes(b"")
            self._assert_clean_failure([str(empty)], expected_code=1, fragments=(f"{empty}:", "Truncated RDS5 header"))

            short = root / "short.ttpf"
            short.write_bytes(b"RDS5" + bytes(12))
            self._assert_clean_failure([str(short)], expected_code=1, fragments=(f"{short}:", "Truncated RDS5 header"))

            corrupt_zlib = root / "corrupt_zlib.ttpf"
            corrupt = bytearray(arch_bytes)
            corrupt[4352 + 4096 + 512] ^= 0xFF
            corrupt_zlib.write_bytes(corrupt)
            self._assert_clean_failure(
                [str(corrupt_zlib), "--zlib-segment", "1"],
                expected_code=1,
                fragments=(f"{corrupt_zlib}:", "Invalid zlib stream"),
            )

            # Duplicate archive input: identical path and hardlink alias.
            alias = root / "alias.ttpf"
            os.link(valid, alias)
            for dup_pair in ([str(valid), str(valid)], [str(valid), str(alias)]):
                with self.subTest(dup_pair=dup_pair):
                    self._assert_clean_failure(
                        [*dup_pair, "--lookup-key", "0x0123456789abcdef"],
                        expected_code=1,
                        fragments=(f"{dup_pair[1]}:", "Duplicate archive input"),
                    )

    def test_argparse_errors_py314_safe(self) -> None:
        arch_bytes, _, _ = _build_synthetic_archive()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arch1 = root / "one.ttpf"
            arch2 = root / "two.ttpf"
            arch1.write_bytes(arch_bytes)
            arch2.write_bytes(arch_bytes)

            for argv, frags in (
                ([], ("required",)),
                ([str(arch1), "--bogus-flag"], ("unrecognized",)),
                ([str(arch1), "--descriptor", "bad"], ("invalid",)),
                ([str(arch1), "--descriptor"], ("expected one argument",)),
                ([str(arch1), "--descriptor", "0", "--lookup-key", "0"], ("not allowed with argument",)),
                ([str(arch1), str(arch2)], ("--lookup-key",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(
                        argv,
                        expected_code=None,
                        fragments=frags,
                    )


if __name__ == "__main__":
    unittest.main()
