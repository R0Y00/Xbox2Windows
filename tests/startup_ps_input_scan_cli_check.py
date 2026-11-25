"""CLI, atomic JSON output, path/collision guard, error handling, and mutation suite for tools/startup/ps_input_scan.py."""

from contextlib import redirect_stderr, redirect_stdout
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
TOOL = ROOT / "tools" / "startup" / "ps_input_scan.py"
sys.path.insert(0, str(ROOT / "tools" / "startup"))

import gfx2hlsl as G  # noqa: E402
import ps_input_scan as scanner  # noqa: E402


END = 0xBF810000


def _w(*words):
    return struct.pack("<%dI" % len(words), *words)


def _ps_blob(mask: int) -> bytes:
    # Minimal synthetic gfx1013 PS: v_mov_b32 v4, v1; exp mrt0 v4, v4, v4, v4 done vm; s_endpgm
    code = _w(0x7E000000 | (4 << 17) | (1 << 9) | (256 + 1), 0xF800180F, 0x04040404, END)
    header = _w(2, 1, mask, mask)
    return header + G.PREFETCH + code


def _populate_pixel_dir(pixel_dir: Path) -> None:
    pixel_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate((0x0002, 0x0003)):
        (pixel_dir / f"shader_{idx:02d}.bin").write_bytes(_ps_blob(mask))


def _populate_request_dir(req_dir: Path) -> None:
    req_dir.mkdir(parents=True, exist_ok=True)
    # Minimal VS blob (prefetch + s_endpgm) + PS blob
    vs_blob = G.PREFETCH + _w(END)
    ps_blob = _ps_blob(0x0002)
    key = "0000000000001000"
    (req_dir / f"req_{key}.a").write_bytes(vs_blob)
    (req_dir / f"req_{key}.b").write_bytes(ps_blob)


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = scanner.main(argv)
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


class PsInputScanCliPositiveTests(unittest.TestCase):
    def test_nested_atomic_json_write_list_and_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixel_dir = root / "pixels"
            _populate_pixel_dir(pixel_dir)
            json_out = root / "nested" / "reports" / "scan.json"
            self.assertFalse(json_out.parent.exists())

            replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(scanner.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([
                    "--pixel-dir", str(pixel_dir),
                    "--json", str(json_out),
                    "--list",
                ])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("pipelines: 2 scanned, 0 skipped", out)
            self.assertIn("p000001", out)
            self.assertEqual(len(replaces), 1)
            self.assertEqual(sorted(p.name for p in json_out.parent.iterdir()), ["scan.json"])

            payload = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "pipelines", "skipped", "informative", "distinct_contracts",
                    "excluded_unsatisfiable", "ambiguous_pipelines", "contracts",
                    "candidates", "detail_truncated", "verdict",
                    "top_candidate_pipelines", "footer",
                },
            )
            self.assertEqual(payload["pipelines"], 2)
            self.assertEqual([c["id"] for c in payload["contracts"]], ["p000001", "p000002"])

    def test_main_sys_argv_forwarding_and_script_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixel_dir = root / "pixels"
            _populate_pixel_dir(pixel_dir)
            json_out = root / "out.json"
            cli_argv = [str(TOOL), "--pixel-dir", str(pixel_dir), "--json", str(json_out)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertIn("pipelines: 2 scanned", out)
            self.assertTrue(json_out.is_file())

            json_out.unlink()
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
            self.assertIn("pipelines: 2 scanned", stdout.getvalue())
            self.assertTrue(json_out.is_file())


class PsInputScanCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(self, argv, *, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_json_directory_and_input_collisions_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            req_dir = root / "requests"
            pixel_dir = root / "pixels"
            _populate_request_dir(req_dir)
            _populate_pixel_dir(pixel_dir)
            cap_file = root / "capture.bin"
            cap_bytes = b"CAP_PLACEHOLDER"
            cap_file.write_bytes(cap_bytes)
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()

            # 1) --json is an existing directory (including "").
            for bad_json in (str(existing_dir), ""):
                with mock.patch.object(scanner, "run") as run_mock:
                    self._assert_exit_2_clean(
                        ["--pixel-dir", str(pixel_dir), "--json", bad_json],
                        expected_fragment="existing directory",
                        check_subprocess=bool(bad_json),
                    )
                run_mock.assert_not_called()

            # 2) --json collides with --capture via resolve() and hardlink samefile().
            self.assertTrue(
                scanner._same_path(root / "ghost.json", root / "no_such_dir" / ".." / "ghost.json")
            )
            cap_alias = root / "no_such_dir" / ".." / "capture.bin"
            cap_link = sub / "cap_link.json"
            os.link(cap_file, cap_link)
            self.assertNotEqual(cap_file.resolve(), cap_link.resolve())
            self.assertTrue(cap_file.samefile(cap_link))
            for bad_json in (cap_alias, cap_link):
                with mock.patch.object(scanner, "run") as run_mock:
                    self._assert_exit_2_clean(
                        ["--capture", str(cap_file), "--json", str(bad_json)],
                        expected_fragment="collides with --capture",
                    )
                run_mock.assert_not_called()
            self.assertEqual(cap_file.read_bytes(), cap_bytes)

            # 3) --json collides with request_dir or --pixel-dir, inside them, or hardlink to file inside them.
            req_alias = root / "sub" / ".." / "requests"
            pix_alias = root / "sub" / ".." / "pixels"
            req_inside = req_dir / "nested" / "out.json"
            pix_inside = pixel_dir / "out.json"
            req_link = sub / "req_file_link.json"
            pix_link = sub / "pix_file_link.json"
            os.link(req_dir / "req_0000000000001000.a", req_link)
            os.link(pixel_dir / "shader_00.bin", pix_link)

            cases = [
                ([str(req_dir), "--json", str(req_alias)], "collides with request directory"),
                ([str(req_dir), "--json", str(req_inside)], "inside request directory"),
                ([str(req_dir), "--json", str(req_link)], "collides with file in request directory"),
                (["--pixel-dir", str(pixel_dir), "--json", str(pix_alias)], "collides with --pixel-dir"),
                (["--pixel-dir", str(pixel_dir), "--json", str(pix_inside)], "inside --pixel-dir"),
                (["--pixel-dir", str(pixel_dir), "--json", str(pix_link)], "collides with file in --pixel-dir"),
            ]
            for argv, frag in cases:
                with self.subTest(argv=argv), mock.patch.object(scanner, "run") as run_mock:
                    self._assert_exit_2_clean(argv, expected_fragment=frag)
                run_mock.assert_not_called()
            self.assertFalse((req_dir / "nested").exists())

    def test_os_value_struct_and_zlib_errors_caught_cleanly_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixel_dir = root / "pixels"
            _populate_pixel_dir(pixel_dir)
            json_out = root / "never" / "out.json"

            # Corrupt capture file hits ValueError cleanly.
            bad_cap = root / "corrupt_capture.bin"
            bad_cap.write_bytes(b"TRUNCATED")
            self._assert_exit_2_clean(
                ["--capture", str(bad_cap), "--json", str(json_out)],
                expected_fragment="truncated chunk header",
            )
            self.assertFalse(json_out.parent.exists())

            # Explicit OSError, ValueError, struct.error, zlib.error during run() all exit 2 cleanly.
            for exc in (
                OSError("simulated read error"),
                ValueError("simulated value error"),
                struct.error("simulated struct error"),
                zlib.error("simulated zlib error"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(scanner, "run", side_effect=exc):
                        self._assert_exit_2_clean(
                            ["--pixel-dir", str(pixel_dir), "--json", str(json_out)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )
                    self.assertFalse(json_out.parent.exists())

    def test_os_replace_and_staging_failures_leave_zero_temp_dirs_and_preserve_prior(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pixel_dir = root / "pixels"
            _populate_pixel_dir(pixel_dir)
            out_dir = root / "out"
            json_out = out_dir / "scan.json"

            def failing_replace(src, dst):
                src_path = Path(src)
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                raise OSError("simulated replace failure")

            # 1) Fresh --json: os.replace OSError -> exit 2, zero files & zero temp dirs.
            with mock.patch.object(scanner.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    ["--pixel-dir", str(pixel_dir), "--json", str(json_out)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [])

            # 2) Pre-existing --json: os.replace OSError -> prior file intact, zero temp dirs.
            json_out.write_text("prior-scan\n", encoding="utf-8")
            with mock.patch.object(scanner.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    ["--pixel-dir", str(pixel_dir), "--json", str(json_out)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(json_out.read_text(encoding="utf-8"), "prior-scan\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["scan.json"])

            # 3) Staging write failure -> os.replace never called, prior file intact, zero temp dirs.
            with (
                mock.patch.object(scanner.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated stage failure")),
            ):
                self._assert_exit_2_clean(
                    ["--pixel-dir", str(pixel_dir), "--json", str(json_out)],
                    expected_fragment="simulated stage failure",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(json_out.read_text(encoding="utf-8"), "prior-scan\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["scan.json"])

    def test_argparse_and_input_validation_exit_2_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_file = root / "not_a_dir.bin"
            a_file.write_bytes(b"x")
            a_dir = root / "not_a_file"
            a_dir.mkdir()

            for bad_argv in (
                [],
                [str(a_file)],
                ["--pixel-dir", str(a_file)],
                ["--capture", str(a_dir)],
                ["--pixel-dir", str(a_dir), "--top", "0"],
                ["--pixel-dir", str(a_dir), "--region-limit", "4"],
            ):
                proc = _run_subprocess(*bad_argv)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, "")
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("usage:", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
