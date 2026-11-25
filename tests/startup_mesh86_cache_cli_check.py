#!/usr/bin/env python3
"""Pure-Python CLI, path/key/timeout-guard, and atomic-publish suite for tools/startup/mesh86_cache.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOLS_STARTUP = ROOT / "tools" / "startup"
TOOL = TOOLS_STARTUP / "mesh86_cache.py"
sys.path.insert(0, str(TOOLS_STARTUP))
import mesh86_cache  # noqa: E402


VALID_MS_DUMP = """; Resource Bindings:
;
; Name Type Format Dim ID HLSL Bind Count
; ---------------------------------------
; source texture byte r/o T0 t0 1

%1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)
%2 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %1, i32 0, i32 undef, i8 1, i32 4)
call void @dx.op.emitIndices(i32 169, i32 0, i32 0, i32 1, i32 2)
"""

VALID_PS_DUMP = """%1 = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 undef)
call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 0, float %1)
"""


def _source_envelope(stage: int) -> bytes:
    """Build a synthetic Xbox shader envelope compatible with mesh86_cache.load_request."""
    image = b"FSSM" + bytes(192)
    compressed = zlib.compress(image)
    native = b"XBCH" + struct.pack("<IQQ", 3, len(compressed), len(image)) + compressed
    program = (
        struct.pack("<I", (stage << 16) | 0x65)
        + bytes(20)
        + b"XDAT"
        + struct.pack("<IIIII", 1, 36, 13, 48, len(native))
        + native
    )
    part = b"DXIL" + struct.pack("<I", len(program)) + program
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, 36 + len(part), 1, 36) + part


def _desktop_dxbc(stage: int, marker_byte: int = 0) -> bytes:
    """Build a valid desktop DXBC/DXIL container accepted by mesh86_cache.desktop_stage."""
    program = struct.pack("<II4sIII", (stage << 16) | 0x65, 7, b"DXIL", 0, 16, 4) + b"BC\xc0\xde"
    if marker_byte:
        program += struct.pack("<I", marker_byte)
        program = program[:4] + struct.pack("<I", len(program) // 4) + program[8:]
    part = b"DXIL" + struct.pack("<I", len(program)) + program
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, 36 + len(part), 1, 36) + part


def _write_request(cache_dir: Path, *, with_pixel: bool = True) -> int:
    ms = _source_envelope(13)
    ps = _source_envelope(0) if with_pixel else b""
    root = b"DXBC" + bytes(28)
    stream = bytes(32)
    key = mesh86_cache.request_key(ms, ps, root, stream)
    stem = cache_dir / f"req_{key:016x}"
    for suffix, data in zip((".m86", ".ps86", ".root", ".stream86"), (ms, ps, root, stream)):
        stem.with_suffix(suffix).write_bytes(data)
    return key


def _make_fake_dxc(directory: Path, name: str, script_body: str) -> Path:
    py_script = directory / f"{name}.py"
    py_script.write_text(script_body, encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / f"{name}.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" -B "{py_script}" %*\r\n', encoding="utf-8")
        return wrapper
    wrapper = directory / f"{name}.sh"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -B "{py_script}" "$@"\n', encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = mesh86_cache.main(argv)
        except SystemExit as exc:
            code = exc.code
            rc = 0 if code is None else (int(code) if isinstance(code, int) else 1)
            if not isinstance(code, (type(None), int)):
                stderr.write(f"{code}\n")
    return rc, stdout.getvalue(), stderr.getvalue()


class Mesh86CacheCliTests(unittest.TestCase):
    def test_positive_inspect_publish_and_prove_root_srv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            key = _write_request(cache, with_pixel=True)
            ms_cso = root / "desktop.ms.cso"
            ps_cso = root / "desktop.ps.cso"
            ms_cso.write_bytes(_desktop_dxbc(13))
            ps_cso.write_bytes(_desktop_dxbc(0))

            # Inspect mode via main(argv) -> 0.
            rc, stdout, stderr = _run_cli([str(cache), f"{key:016x}"])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn(f"{key:016x}: MS", stdout)

            # Publish mode without --prove-root-srv.
            rc, stdout, stderr = _run_cli([str(cache), f"{key:016x}", "--mesh", str(ms_cso), "--pixel", str(ps_cso)])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertIn("published desktop stages", stdout)
            ready = cache / f"{key:016x}.mesh86.txt"
            self.assertTrue(ready.read_text(encoding="ascii").startswith("x2w-mesh86 1 "))
            self.assertEqual(
                sorted(p.name for p in cache.iterdir()),
                sorted([
                    f"{key:016x}.mesh86.txt",
                    f"{key:016x}.ms.cso",
                    f"{key:016x}.ps.cso",
                    f"req_{key:016x}.m86",
                    f"req_{key:016x}.ps86",
                    f"req_{key:016x}.root",
                    f"req_{key:016x}.stream86",
                ]),
            )

            # Publish mode with --prove-root-srv and timeout=30 verification.
            fake_dxc = _make_fake_dxc(
                root,
                "fake_dxc",
                "import sys\n"
                "target = sys.argv[-1]\n"
                f"print({VALID_MS_DUMP!r} if target.endswith('ms') else {VALID_PS_DUMP!r})\n",
            )
            real_run = subprocess.run
            observed_timeouts = []

            def spy_run(*args, **kwargs):
                observed_timeouts.append(kwargs.get("timeout"))
                return real_run(*args, **kwargs)

            with mock.patch.object(mesh86_cache.subprocess, "run", side_effect=spy_run):
                rc, stdout, stderr = _run_cli([
                    str(cache),
                    f"{key:016x}",
                    "--mesh",
                    str(ms_cso),
                    "--pixel",
                    str(ps_cso),
                    "--prove-root-srv",
                    str(fake_dxc),
                ])
            self.assertEqual(rc, 0, stderr)
            self.assertEqual(stderr, "")
            self.assertEqual(observed_timeouts, [30, 30])
            marker_text = ready.read_text(encoding="ascii")
            self.assertTrue(marker_text.startswith("x2w-mesh86 2 "))
            self.assertTrue(marker_text.endswith(" srv 0 0 4\n"))

            # Subprocess positive smoke test.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(cache), f"{key:016x}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn(f"{key:016x}: MS", proc.stdout)

    def test_key_range_and_argparse_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache"
            cache.mkdir()

            # Missing args / invalid hex -> non-zero SystemExit without traceback.
            for bad_argv, expected_substr in (
                ([], "required"),
                ([str(cache), "not_hex"], "invalid"),
            ):
                with self.subTest(bad_argv=bad_argv):
                    rc, _, stderr = _run_cli(bad_argv)
                    self.assertNotEqual(rc, 0)
                    self.assertIn(expected_substr, stderr.lower())
                    self.assertNotIn("Traceback", stderr)

            # Out-of-range 64-bit keys -> exit 1 without touching cache files.
            for bad_key in ("-1", "10000000000000000", "1ffffffffffffffff"):
                with self.subTest(bad_key=bad_key):
                    rc, _, stderr = _run_cli([str(cache), bad_key])
                    self.assertEqual(rc, 1)
                    self.assertIn("64-bit", stderr)
                    self.assertNotIn("Traceback", stderr)

            # Boundary 64-bit keys (0 and 2^64-1) pass the range check and only fail on missing req files.
            for boundary_key in ("0", "ffffffffffffffff"):
                with self.subTest(boundary_key=boundary_key):
                    rc, _, stderr = _run_cli([str(cache), boundary_key])
                    self.assertEqual(rc, 1)
                    self.assertNotIn("64-bit", stderr)
                    self.assertNotIn("Traceback", stderr)

    def test_cache_and_input_path_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            key = _write_request(cache, with_pixel=True)
            valid_ms = root / "ms.cso"
            valid_ps = root / "ps.cso"
            valid_ms.write_bytes(_desktop_dxbc(13))
            valid_ps.write_bytes(_desktop_dxbc(0))
            regular_file = root / "not_a_dir.bin"
            regular_file.write_bytes(b"data")
            dir_path = root / "a_dir"
            dir_path.mkdir()

            # Cache missing or regular file -> exit 1.
            for bad_cache in (root / "missing_cache", regular_file):
                with self.subTest(bad_cache=bad_cache):
                    rc, _, stderr = _run_cli([str(bad_cache), f"{key:016x}"])
                    self.assertEqual(rc, 1)
                    self.assertIn("cache directory", stderr)
                    self.assertNotIn("Traceback", stderr)

            # --pixel or --prove-root-srv without --mesh -> exit 1.
            for extra_args in (["--pixel", str(valid_ps)], ["--prove-root-srv", str(valid_ms)]):
                with self.subTest(extra_args=extra_args):
                    rc, _, stderr = _run_cli([str(cache), f"{key:016x}", *extra_args])
                    self.assertEqual(rc, 1)
                    self.assertIn("--mesh", stderr)
                    self.assertNotIn("Traceback", stderr)

            # Missing or directory paths for --mesh, --pixel, --prove-root-srv -> exit 1.
            cases = (
                (["--mesh", str(root / "missing.cso")], "--mesh"),
                (["--mesh", str(dir_path)], "--mesh"),
                (["--mesh", str(valid_ms), "--pixel", str(root / "missing.cso")], "--pixel"),
                (["--mesh", str(valid_ms), "--pixel", str(dir_path)], "--pixel"),
                (["--mesh", str(valid_ms), "--pixel", str(valid_ps), "--prove-root-srv", str(root / "missing_dxc")], "--prove-root-srv"),
                (["--mesh", str(valid_ms), "--pixel", str(valid_ps), "--prove-root-srv", str(dir_path)], "--prove-root-srv"),
            )
            for extra_args, opt in cases:
                with self.subTest(opt=opt, extra_args=extra_args):
                    rc, _, stderr = _run_cli([str(cache), f"{key:016x}", *extra_args])
                    self.assertEqual(rc, 1)
                    self.assertIn(opt, stderr)
                    self.assertNotIn("Traceback", stderr)

    def test_timeout_subprocess_struct_and_atomic_preservation_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            key = _write_request(cache, with_pixel=True)
            ms_cso = root / "ms.cso"
            ps_cso = root / "ps.cso"
            ms_cso.write_bytes(_desktop_dxbc(13, marker_byte=1))
            ps_cso.write_bytes(_desktop_dxbc(0, marker_byte=1))

            # Publish initial valid artifacts so we can verify preservation on subsequent failures.
            rc, _, stderr = _run_cli([str(cache), f"{key:016x}", "--mesh", str(ms_cso), "--pixel", str(ps_cso)])
            self.assertEqual(rc, 0, stderr)
            ready = cache / f"{key:016x}.mesh86.txt"
            ms_out = cache / f"{key:016x}.ms.cso"
            ps_out = cache / f"{key:016x}.ps.cso"
            prev_ready = ready.read_bytes()
            prev_ms = ms_out.read_bytes()
            prev_ps = ps_out.read_bytes()
            expected_entries = sorted(p.name for p in cache.iterdir())

            # Replace desktop inputs with distinct valid bytes to detect any partial write on failure.
            ms_cso.write_bytes(_desktop_dxbc(13, marker_byte=2))
            ps_cso.write_bytes(_desktop_dxbc(0, marker_byte=2))

            failing_dxc = _make_fake_dxc(root, "fail_dxc", "import sys\nsys.exit(7)\n")
            with mock.patch.object(
                mesh86_cache.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd="dxc", timeout=30),
            ):
                rc, _, stderr = _run_cli([
                    str(cache),
                    f"{key:016x}",
                    "--mesh",
                    str(ms_cso),
                    "--pixel",
                    str(ps_cso),
                    "--prove-root-srv",
                    str(failing_dxc),
                ])
            self.assertEqual(rc, 1)
            self.assertNotIn("Traceback", stderr)
            self.assertEqual(ready.read_bytes(), prev_ready)
            self.assertEqual(ms_out.read_bytes(), prev_ms)
            self.assertEqual(ps_out.read_bytes(), prev_ps)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), expected_entries)

            # Real failing subprocess (--prove-root-srv non-zero exit) preserves existing artifacts.
            rc, _, stderr = _run_cli([
                str(cache),
                f"{key:016x}",
                "--mesh",
                str(ms_cso),
                "--pixel",
                str(ps_cso),
                "--prove-root-srv",
                str(failing_dxc),
            ])
            self.assertEqual(rc, 1)
            self.assertNotIn("Traceback", stderr)
            self.assertEqual(ready.read_bytes(), prev_ready)
            self.assertEqual(ms_out.read_bytes(), prev_ms)
            self.assertEqual(ps_out.read_bytes(), prev_ps)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), expected_entries)

            # struct.error and ValueError in load_request exit 1 without traceback.
            with mock.patch.object(mesh86_cache, "load_request", side_effect=struct.error("short unpack")):
                rc, _, stderr = _run_cli([str(cache), f"{key:016x}"])
            self.assertEqual(rc, 1)
            self.assertIn("short unpack", stderr)
            self.assertNotIn("Traceback", stderr)

            # Subprocess negative smoke test.
            proc = subprocess.run(
                [sys.executable, "-B", str(TOOL), str(cache), "10000000000000000"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("64-bit", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
