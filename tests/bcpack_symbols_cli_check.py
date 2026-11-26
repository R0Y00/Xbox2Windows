"""Pure-Python CLI, file-guard, and error-handling suite for tools/bcpack_symbols.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bcpack_symbols.py"
VECTORS = ROOT / "tools" / "_bcpack_rans" / "vectors"
sys.path.insert(0, str(ROOT / "tools"))

import bcpack_symbols as symbols_mod  # noqa: E402
from _bcpack_rans import LEGAL, model  # noqa: E402

BC1_WIDTHS = (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8)


def _encode_single_lane(bits: int, q: int, values: list[int], *, narrow: int = 0, flush: bool = True) -> bytes:
    """Encode one synthetic single-lane rANS fragment using the published model."""
    m = model(bits - 1 - narrow, q)
    state = 1 << 24
    emitted = bytearray()
    for value in reversed(values):
        freq, start = m.forward(value)
        if not freq:
            raise ValueError("Zero-frequency symbol")
        while state >= freq << 16:
            emitted.append(state & 0xFF)
            state >>= 8
        state = (state // freq << 16) + (state % freq) + start
    payload = state.to_bytes(4, "big") + bytes(reversed(emitted))
    header = (narrow << 6) | (0x10 if flush else 0) | q
    if len(payload) <= 256:
        return bytes((header, len(payload) - 1)) + payload
    return bytes((header | 0x20,)) + (len(payload) - 257).to_bytes(2, "little") + payload


class BcpackSymbolsCliTests(unittest.TestCase):
    def _assert_main_exit(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 1,
        fragment: str = "",
        prefix: str = "BCPack symbols: ",
    ) -> str:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exc:
                symbols_mod.main(argv)
        if expected_code is None:
            self.assertNotEqual(exc.exception.code, 0)
        else:
            self.assertEqual(exc.exception.code, expected_code)
        self.assertEqual(stdout.getvalue(), "")
        err = stderr.getvalue()
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr message for {argv!r}")
        if prefix:
            self.assertIn(prefix, err)
        if fragment:
            self.assertIn(fragment, err)
        return err

    def test_main_argv_valid_single_and_multi_lane_packets(self):
        values = [0, 1, 2, 3, 1, 0, 3, 2] * 3
        single_packet = _encode_single_lane(2, 0, values)
        vector = json.loads((VECTORS / "synthetic_11.json").read_text(encoding="utf-8"))
        multi_packet = bytes.fromhex(vector["chunk_hex"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single_path = root / "single.bin"
            single_path.write_bytes(single_packet)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = symbols_mod.main([
                    str(single_path),
                    "--alphabet-bits", "2",
                    "--symbols", str(len(values)),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["syms"], [values])
            self.assertEqual(payload["consumed"], len(single_packet))
            self.assertNotIn("order", payload)
            self.assertTrue(all(lane["final"] in LEGAL for lane in payload["lanes"]))

            # Zero-padded packet up to the exact MAX_INPUT boundary succeeds.
            exact_max_path = root / "exact_max.bin"
            exact_max_path.write_bytes(single_packet.ljust(symbols_mod.MAX_INPUT, b"\x00"))
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = symbols_mod.main([
                    str(exact_max_path),
                    "--alphabet-bits", "2",
                    "--symbols", str(len(values)),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(json.loads(stdout.getvalue())["syms"], [values])

            # Multi-lane synthetic vector decodes cleanly via main(argv).
            multi_path = root / "multi.bin"
            multi_path.write_bytes(multi_packet)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = symbols_mod.main([
                    str(multi_path),
                    "--alphabet-bits", *[str(w) for w in BC1_WIDTHS],
                    "--symbols", str(vector["blocks"]),
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(stderr.getvalue(), "")
            multi_payload = json.loads(stdout.getvalue())
            self.assertEqual(multi_payload["syms"], vector["symbols"])
            self.assertEqual(multi_payload["consumed"], len(multi_packet))

            # Bounded real subprocess check.
            proc = subprocess.run(
                [
                    sys.executable, "-B", str(TOOL),
                    str(single_path), "--alphabet-bits", "2", "--symbols", str(len(values)),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), payload)

    def test_upfront_file_and_size_guards_before_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.bin"
            folder = root / "packet_dir"
            folder.mkdir()
            empty = root / "empty.bin"
            empty.write_bytes(b"")
            oversize = root / "oversize.bin"
            oversize.write_bytes(bytes(symbols_mod.MAX_INPUT + 1))

            with mock.patch.object(Path, "open", side_effect=AssertionError("open must not run")):
                self._assert_main_exit(
                    [str(missing), "--alphabet-bits", "1", "--symbols", "1"],
                    fragment=missing.name,
                )
                self._assert_main_exit(
                    [str(folder), "--alphabet-bits", "1", "--symbols", "1"],
                    fragment=folder.name,
                )
                self._assert_main_exit(
                    [str(empty), "--alphabet-bits", "1", "--symbols", "1"],
                    fragment="Expected 1..2097152 bytes",
                )
                self._assert_main_exit(
                    [str(oversize), "--alphabet-bits", "1", "--symbols", "1"],
                    fragment="Expected 1..2097152 bytes",
                )

            for target, expected in (
                (empty, "Expected 1..2097152 bytes"),
                (folder, folder.name),
                (missing, missing.name),
            ):
                with self.subTest(target=target.name):
                    proc = subprocess.run(
                        [
                            sys.executable, "-B", str(TOOL),
                            str(target), "--alphabet-bits", "1", "--symbols", "1",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(proc.returncode, 1, proc.stderr)
                    self.assertEqual(proc.stdout, "")
                    self.assertIn("BCPack symbols: ", proc.stderr)
                    self.assertIn(expected, proc.stderr)
                    self.assertNotIn("Traceback", proc.stderr)

    def test_out_of_range_widths_and_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "valid.bin"
            packet.write_bytes(_encode_single_lane(1, 0, [0, 1] * 4))

            bad_cases = (
                (["--alphabet-bits", "0", "--symbols", "1"], "Expected integer symbol widths in 1..9"),
                (["--alphabet-bits", "10", "--symbols", "1"], "Expected integer symbol widths in 1..9"),
                (["--alphabet-bits", *["8"] * (symbols_mod.MAX_LANES + 1), "--symbols", "1"], "Expected 1..16 lane widths"),
                (["--alphabet-bits", "8", "--symbols", "0"], "Expected a positive count"),
                (["--alphabet-bits", "8", "--symbols", "-3"], "Expected a positive count"),
                (
                    ["--alphabet-bits", "8", "8", "--symbols", str(symbols_mod.MAX_SYMBOLS // 2 + 1)],
                    "at most 1048576 symbols",
                ),
            )
            for extra_args, fragment in bad_cases:
                with self.subTest(args=extra_args):
                    self._assert_main_exit([str(packet), *extra_args], fragment=fragment)

            # Invalid --alphabet-bits must still report cleanly even when file exceeds MAX_INPUT.
            oversize = root / "oversize.bin"
            oversize.write_bytes(bytes(symbols_mod.MAX_INPUT + 1))
            self._assert_main_exit(
                [str(oversize), "--alphabet-bits", "10", "--symbols", "1"],
                fragment="Expected integer symbol widths in 1..9",
            )

            # Direct decode() guards for non-bytes and boolean inputs.
            for bad_data in (b"", bytes(symbols_mod.MAX_INPUT + 1), [0], "text"):
                with self.assertRaisesRegex(ValueError, "bytes"):
                    symbols_mod.decode(bad_data, [1], 1)
            with self.assertRaisesRegex(ValueError, "1..9"):
                symbols_mod.decode(b"\x10", [True], 1)
            with self.assertRaisesRegex(ValueError, "positive count"):
                symbols_mod.decode(b"\x10", [1], True)

    def test_corrupt_packet_and_nonzero_trailing_refusals(self):
        values = [0, 1] * 7
        valid_bytes = _encode_single_lane(1, 0, values)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            trailing = root / "trailing.bin"
            trailing.write_bytes(valid_bytes + b"\x00\x01")
            self._assert_main_exit(
                [str(trailing), "--alphabet-bits", "1", "--symbols", str(len(values))],
                fragment="Non-zero trailing data",
            )

            truncated = root / "truncated.bin"
            truncated.write_bytes(valid_bytes[:-2])
            self._assert_main_exit(
                [str(truncated), "--alphabet-bits", "1", "--symbols", str(len(values))],
            )

            illegal = root / "illegal.bin"
            illegal.write_bytes(bytes.fromhex("10 03 01 00 00 02"))
            self._assert_main_exit(
                [str(illegal), "--alphabet-bits", "1", "--symbols", "1"],
                fragment="illegal",
            )

            bad_header = root / "bad_header.bin"
            bad_header.write_bytes(b"\xc0")
            self._assert_main_exit(
                [str(bad_header), "--alphabet-bits", "1", "--symbols", "1"],
            )

            # Decoded symbol shape mismatch from backend raises BCPackError cleanly.
            with mock.patch.object(
                symbols_mod,
                "decode_lanes",
                return_value={"syms": [[2]], "consumed": 1, "cycles": 1, "lanes": [], "order": []},
            ):
                with self.assertRaisesRegex(symbols_mod.BCPackError, "lane shape"):
                    symbols_mod.decode(b"\x10", [1], 1)
                self._assert_main_exit(
                    [str(bad_header), "--alphabet-bits", "1", "--symbols", "1"],
                    fragment="lane shape",
                )

            proc = subprocess.run(
                [
                    sys.executable, "-B", str(TOOL),
                    str(trailing), "--alphabet-bits", "1", "--symbols", str(len(values)),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertIn("BCPack symbols: Non-zero trailing data", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_argparse_errors_py314_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp) / "pkt.bin"
            packet.write_bytes(b"\x10")
            cases = (
                ([], "required"),
                ([str(packet), "--symbols", "1"], "--alphabet-bits"),
                ([str(packet), "--alphabet-bits", "1"], "--symbols"),
                ([str(packet), "--alphabet-bits", "1", "--symbols", "abc"], "invalid int"),
                ([str(packet), "--alphabet-bits", "bad", "--symbols", "1"], "invalid int"),
            )
            for argv, fragment in cases:
                with self.subTest(argv=argv):
                    self._assert_main_exit(
                        argv,
                        expected_code=None,
                        fragment=fragment,
                        prefix="",
                    )


if __name__ == "__main__":
    unittest.main()
