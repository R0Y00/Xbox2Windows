"""CLI, read-only, mmap fallback, directory/struct.error guard, and mutation suite for tools/pe_authentihash.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import runpy
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "pe_authentihash.py"
sys.path.insert(0, str(ROOT / "tools"))

import pe_authentihash  # noqa: E402

HEADERS = 0x200
PE_AT = 0x40
OPTIONAL = PE_AT + 24
CHECKSUM = OPTIONAL + 64
CERT_CONTENT = bytes((7 * i + 3) % 256 for i in range(48))


def _build_pe(*, plus: bool, cert: bytes | None = None) -> bytes:
    optional_size = 240 if plus else 224
    dirs_at = OPTIONAL + (112 if plus else 96)
    bodies = [bytes((i * 13 + k) % 251 for i in range(0x200 * (k + 1))) for k in range(2)]
    image = bytearray(bytes((i * 5 + 1) % 253 for i in range(HEADERS)))
    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, PE_AT)
    image[PE_AT:PE_AT + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", image, PE_AT + 4, 0x8664, 2, 0, 0, 0, optional_size, 0x2022)
    struct.pack_into("<H", image, OPTIONAL, 0x20B if plus else 0x10B)
    struct.pack_into("<II", image, OPTIONAL + 32, 0x1000, 0x200)
    struct.pack_into("<I", image, OPTIONAL + 56, 0x3000)
    struct.pack_into("<I", image, CHECKSUM, 0x11223344)
    struct.pack_into("<H", image, OPTIONAL + 68, 3)
    struct.pack_into("<I", image, OPTIONAL + 60, HEADERS)
    struct.pack_into("<I", image, dirs_at - 4, 16)
    struct.pack_into("<II", image, dirs_at + 8 * 4, 0, 0)
    section_table = OPTIONAL + optional_size
    pointer = HEADERS
    for k, body in enumerate(bodies):
        row = bytearray(40)
        row[0:8] = f".s{k}".encode().ljust(8, b"\0")
        struct.pack_into("<IIII", row, 8, len(body), 0x1000 * (k + 1), len(body), pointer)
        image[section_table + 40 * k:section_table + 40 * (k + 1)] = row
        pointer += len(body)
    image += b"".join(bodies)
    if cert is not None:
        struct.pack_into("<II", image, dirs_at + 32, len(image), len(cert))
        image += cert
    return bytes(image)


def _reference_digest(image: bytes, *, plus: bool, cert_len: int = 0) -> str:
    cert_entry = OPTIONAL + (112 if plus else 96) + 32
    end = len(image) - cert_len
    parts = [
        image[:CHECKSUM],
        image[CHECKSUM + 4:cert_entry],
        image[cert_entry + 8:HEADERS],
        image[HEADERS:end],
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = pe_authentihash.main(argv)
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


class PeAuthentihashCliPositiveTests(unittest.TestCase):
    def test_pe32_pe32plus_and_signed_cert_table_readonly_digests(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("pe32.exe", False, None),
                ("pe32plus.exe", True, None),
                ("pe32_signed.exe", False, CERT_CONTENT),
                ("pe32plus_signed.exe", True, CERT_CONTENT),
            ]
            digests = {}
            for name, plus, cert in cases:
                raw = _build_pe(plus=plus, cert=cert)
                expected = _reference_digest(raw, plus=plus, cert_len=len(cert) if cert else 0)
                img = root / name
                img.write_bytes(raw)

                code, out, err = _run_main([str(img)])
                self.assertIs(type(code), int)
                self.assertEqual(code, 0, err)
                self.assertEqual(err, "")
                self.assertEqual(out.strip(), expected)
                self.assertEqual(len(out.strip()), 64)
                self.assertEqual(img.read_bytes(), raw)

                proc = _run_subprocess(str(img))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stderr, "")
                self.assertEqual(proc.stdout.strip(), expected)
                self.assertEqual(img.read_bytes(), raw)
                digests[name] = out.strip()

            self.assertEqual(digests["pe32.exe"], digests["pe32_signed.exe"])
            self.assertEqual(digests["pe32plus.exe"], digests["pe32plus_signed.exe"])
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["pe32.exe", "pe32_signed.exe", "pe32plus.exe", "pe32plus_signed.exe"],
            )

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = _build_pe(plus=True, cert=CERT_CONTENT)
            expected = _reference_digest(raw, plus=True, cert_len=len(CERT_CONTENT))
            img = root / "entry.exe"
            img.write_bytes(raw)
            cli_argv = [str(TOOL), str(img)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(out.strip(), expected)

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
            self.assertEqual(stdout.getvalue().strip(), expected)


class PeAuthentihashCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(self, argv, *, expected_code=1, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_argparse_errors_exit_nonzero_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "pe.exe"
            img.write_bytes(_build_pe(plus=True))

            for argv, frag in (
                ([], "usage:"),
                ([str(img), "--unknown-flag"], "unrecognized"),
            ):
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            pe_authentihash.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self._assert_clean_failure(argv, expected_code=2, expected_fragment=frag)

    def test_directory_missing_empty_and_truncated_inputs_exit_1_cleanly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub_dir = root / "sub_dir"
            sub_dir.mkdir()
            missing = root / "missing.exe"
            empty = root / "empty.exe"
            empty.write_bytes(b"")
            truncated = root / "truncated.exe"
            truncated.write_bytes(_build_pe(plus=True)[:0x150])

            # 1) Directory (including "") rejected upfront before authentihash().
            for bad_dir in (str(sub_dir), ""):
                with mock.patch.object(pe_authentihash, "authentihash") as hash_mock:
                    self._assert_clean_failure(
                        [bad_dir],
                        expected_fragment="directory",
                        check_subprocess=bool(bad_dir),
                    )
                hash_mock.assert_not_called()

            # 2) Missing file rejected upfront before authentihash().
            with mock.patch.object(pe_authentihash, "authentihash") as hash_mock:
                self._assert_clean_failure(
                    [str(missing)],
                    expected_fragment=f"{missing}:",
                )
            hash_mock.assert_not_called()

            # 3) Non-regular file (is_file() False) rejected upfront before authentihash().
            with (
                mock.patch.object(Path, "is_dir", return_value=False),
                mock.patch.object(Path, "is_file", return_value=False),
                mock.patch.object(pe_authentihash, "authentihash") as hash_mock,
            ):
                self._assert_clean_failure(
                    [str(empty)],
                    expected_fragment="regular file",
                    check_subprocess=False,
                )
            hash_mock.assert_not_called()

            # 4) 0-byte file exercises mmap ValueError fallback cleanly -> empty memoryview -> exit 1.
            self._assert_clean_failure(
                [str(empty)],
                expected_fragment=f"{empty}: Not a PE file: missing MZ header",
            )
            seen_views = []

            def check_empty_view(view):
                seen_views.append(bytes(view))
                return "0" * 64

            with mock.patch.object(pe_authentihash, "authentihash", side_effect=check_empty_view):
                code_0, out_0, err_0 = _run_main([str(empty)])
            self.assertEqual(code_0, 0, err_0)
            self.assertEqual(out_0.strip(), "0" * 64)
            self.assertEqual(seen_views, [b""])

            # 5) Truncated PE rejected cleanly with exit 1 and zero traceback.
            self._assert_clean_failure(
                [str(truncated)],
                expected_fragment=f"{truncated}:",
            )

            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["empty.exe", "sub_dir", "truncated.exe"],
            )

    def test_os_value_and_struct_errors_caught_cleanly_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "sample.exe"
            img.write_bytes(_build_pe(plus=False))

            for exc in (
                OSError("simulated io failure"),
                ValueError("simulated value failure"),
                struct.error("unpack_from requires a buffer of at least 4 bytes"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(pe_authentihash, "authentihash", side_effect=exc):
                        self._assert_clean_failure(
                            [str(img)],
                            expected_fragment=f"{img}: {exc}",
                            check_subprocess=False,
                        )

            with mock.patch.object(Path, "open", side_effect=OSError("simulated open failure")):
                self._assert_clean_failure(
                    [str(img)],
                    expected_fragment=f"{img}: simulated open failure",
                    check_subprocess=False,
                )


if __name__ == "__main__":
    unittest.main()
