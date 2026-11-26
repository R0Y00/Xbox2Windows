#!/usr/bin/env python3
"""Pure-Python synthetic PE32+ CLI, atomic-write, and path-guard suite for tools/call_evidence.py."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "call_evidence.py"
sys.path.insert(0, str(TOOLS_DIR))
import call_evidence  # noqa: E402


IMAGE_BASE = 0x180000000
SECTION_RVA = 0x1000
RAW_OFFSET = 0x200
RAW_SIZE = 0x1000


def _make_dxbc_container() -> bytes:
    """Build a minimal synthetic DXBC container with DXIL and RTS0 parts."""
    program = struct.pack("<II4sIII", 0, 8, b"DXIL", 0, 16, 4) + b"BC\xc0\xde" + bytes(4)
    rts0 = struct.pack("<6I", 2, 3, 24, 1, 0, 0x10)
    parts = [
        b"DXIL" + struct.pack("<I", len(program)) + program,
        b"RTS0" + struct.pack("<I", len(rts0)) + rts0,
    ]
    cursor = 32 + 4 * len(parts)
    offsets = []
    for part in parts:
        offsets.append(cursor)
        cursor += len(part)
    body = b"".join(parts)
    total = 32 + 4 * len(parts) + len(body)
    return (
        b"DXBC"
        + bytes(16)
        + struct.pack("<III", 1, total, len(parts))
        + struct.pack(f"<{len(parts)}I", *offsets)
        + body
    )


def _build_pe32_plus(
    imports: list[tuple[bytes, bytes]] = (),
    *,
    containers: list[bytes] = (),
) -> tuple[bytes, dict[tuple[bytes, bytes], int], list[int]]:
    """Build a minimal AMD64 PE32+ DLL with optional imports and mapped DXBC containers."""
    payload = bytearray(RAW_SIZE)
    cursor = 0

    def alloc(size: int, align: int) -> int:
        nonlocal cursor
        cursor = (cursor + align - 1) & ~(align - 1)
        rva = SECTION_RVA + cursor
        if cursor + size > len(payload):
            raise AssertionError("synthetic section overflow")
        cursor += size
        return rva

    def put(rva: int, blob: bytes) -> None:
        offset = rva - SECTION_RVA
        payload[offset:offset + len(blob)] = blob

    grouped: dict[bytes, list[bytes]] = {}
    for dll_bytes, sym in imports:
        grouped.setdefault(dll_bytes, []).append(sym)

    dll_rvas = {}
    for dll_bytes in grouped:
        raw = dll_bytes + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        dll_rvas[dll_bytes] = rva

    hint_rvas = {}
    for dll_bytes, syms in grouped.items():
        for sym in syms:
            raw = struct.pack("<H", 0) + sym + b"\0"
            rva = alloc(len(raw), 2)
            put(rva, raw)
            hint_rvas[(dll_bytes, sym)] = rva

    iat_rvas: dict[tuple[bytes, bytes], int] = {}
    descriptors = []
    for dll_bytes, syms in grouped.items():
        entries = [hint_rvas[(dll_bytes, sym)] for sym in syms]
        table_blob = b"".join(struct.pack("<Q", value) for value in [*entries, 0])
        lookup_rva = alloc(len(table_blob), 8)
        put(lookup_rva, table_blob)
        iat_rva = alloc(len(table_blob), 8)
        put(iat_rva, table_blob)
        for idx, sym in enumerate(syms):
            iat_rvas[(dll_bytes, sym)] = iat_rva + idx * 8
        descriptors.append((lookup_rva, dll_rvas[dll_bytes], iat_rva))

    import_dir_rva = import_dir_size = 0
    if descriptors:
        import_dir_size = 20 * (len(descriptors) + 1)
        import_dir_rva = alloc(import_dir_size, 4)
        for idx, (lookup_rva, dll_rva, iat_rva) in enumerate(descriptors):
            put(import_dir_rva + idx * 20, struct.pack("<5I", lookup_rva, 0, 0, dll_rva, iat_rva))

    container_rvas = []
    for blob in containers:
        c_rva = alloc(len(blob), 16)
        put(c_rva, blob)
        container_rvas.append(c_rva)

    vsize = max(1, cursor)
    size_of_image = SECTION_RVA + ((vsize + 0xFFF) // 0x1000) * 0x1000
    pe_off = 0x80
    buf = bytearray(RAW_OFFSET + RAW_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B, 0, 0, 0, RAW_SIZE, 0, 0, SECTION_RVA, IMAGE_BASE,
        0x1000, 0x200, 6, 0, 0, 0, 6, 0, 0, size_of_image, RAW_OFFSET,
        0, 3, 0, 0, 0, 0, 0, 0, 16,
    )
    sec = opt + 240
    struct.pack_into(
        "<8sIIIIIIHHI",
        buf,
        sec,
        b".rdata\0\0", vsize, SECTION_RVA, RAW_SIZE, RAW_OFFSET, 0, 0, 0, 0, 0x40000040,
    )
    if import_dir_rva:
        struct.pack_into("<II", buf, opt + 112 + 8, import_dir_rva, import_dir_size)
    buf[RAW_OFFSET:RAW_OFFSET + RAW_SIZE] = payload
    return bytes(buf), iat_rvas, container_rvas


def _format_disasm(lines: list[tuple[int, str]]) -> bytes:
    body = "\n".join(f"  {IMAGE_BASE + rva:016X}: {text}" for rva, text in lines)
    return f"Dump of file fixture.dll\n\nFile Type: DLL\n\n{body}\n".encode("ascii")


def _populate_valid_build_dir(build_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    pe0, iat0, _ = _build_pe32_plus(imports=[(b"xmem.dll", b"XMemAlloc")])
    pe1, iat1, c_rvas1 = _build_pe32_plus(
        imports=[(b"d3d12_xs.dll", b"D3D12SerializeVersionedRootSignature")],
        containers=[_make_dxbc_container()],
    )
    pe2, _, _ = _build_pe32_plus()
    images = {
        call_evidence.NAMES[0]: pe0,
        call_evidence.NAMES[1]: pe1,
        call_evidence.NAMES[2]: pe2,
    }
    disasm = {
        call_evidence.NAMES[0]: _format_disasm([
            (0x2100, f"call        qword ptr [{IMAGE_BASE + iat0[(b'xmem.dll', b'XMemAlloc')]:016X}h]"),
        ]),
        call_evidence.NAMES[1]: _format_disasm([
            (0x2200, f"jmp         qword ptr [{IMAGE_BASE + iat1[(b'd3d12_xs.dll', b'D3D12SerializeVersionedRootSignature')]:016X}h]"),
            (0x2210, f"lea         rcx, [{IMAGE_BASE + c_rvas1[0]:016X}h]"),
        ]),
        call_evidence.NAMES[2]: b"",
    }
    build_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in images.items():
        (build_dir / name).write_bytes(blob)
    return images, disasm


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = call_evidence.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


class CallEvidenceCliPositiveTests(unittest.TestCase):
    def test_atomic_write_creates_nested_dirs_replaces_stale_and_leaves_zero_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            images, disasm = _populate_valid_build_dir(build_dir)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            expected_out = (base / "repo" / "out" / "calls" / "six-apis.json").resolve()

            def fake_dumpbin(cmd: list[str]) -> bytes:
                return disasm[Path(cmd[-1]).name]

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin),
            ):
                code, out, err = _run_main([str(build_dir)])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(Path(out.strip()), expected_out)
            self.assertEqual(sorted(p.name for p in expected_out.parent.iterdir()), ["six-apis.json"])

            parsed = json.loads(expected_out.read_text(encoding="utf-8"))
            self.assertEqual(list(parsed.keys()), list(call_evidence.NAMES))
            for name, blob in images.items():
                self.assertEqual(parsed[name]["sha256"], hashlib.sha256(blob).hexdigest())
            self.assertEqual(parsed[call_evidence.NAMES[0]]["calls"]["XMemAlloc"]["sites"][0]["rva"], "0x2100")
            self.assertEqual(len(parsed[call_evidence.NAMES[1]]["containers"]), 1)

            # Seed stale bytes and verify atomic overwrite leaves 0 .tmp files.
            expected_out.write_text("{stale", encoding="utf-8")
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin),
            ):
                code2, out2, err2 = _run_main([str(build_dir)])
            self.assertEqual(code2, 0, err2)
            self.assertEqual(err2, "")
            self.assertEqual(Path(out2.strip()), expected_out)
            self.assertEqual(sorted(p.name for p in expected_out.parent.iterdir()), ["six-apis.json"])
            self.assertEqual(json.loads(expected_out.read_text(encoding="utf-8")), parsed)

    def test_sys_argv_none_and_main_entrypoint_system_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _, disasm = _populate_valid_build_dir(build_dir)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            fake_tool.write_text(TOOL.read_text(encoding="utf-8"), encoding="utf-8")
            expected_out = (base / "repo" / "out" / "calls" / "six-apis.json").resolve()

            def fake_dumpbin(cmd: list[str]) -> bytes:
                return disasm[Path(cmd[-1]).name]

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(sys, "argv", [str(fake_tool), str(build_dir)]),
                mock.patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin),
            ):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(Path(out.strip()), expected_out)
            self.assertTrue(expected_out.is_file())

            # Verify __main__ raises SystemExit(0) on success and SystemExit(1) on PE error.
            expected_out.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(fake_tool), str(build_dir)]),
                mock.patch("subprocess.check_output", side_effect=fake_dumpbin),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(fake_tool), run_name="__main__")
            self.assertIs(type(exited.exception.code), int)
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(Path(stdout.getvalue().strip()), expected_out)
            self.assertTrue(expected_out.is_file())

            (build_dir / call_evidence.NAMES[0]).write_bytes(b"corrupt")
            stdout2, stderr2 = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(fake_tool), str(build_dir)]),
                redirect_stdout(stdout2),
                redirect_stderr(stderr2),
            ):
                with self.assertRaises(SystemExit) as exited_err:
                    runpy.run_path(str(fake_tool), run_name="__main__")
            self.assertEqual(exited_err.exception.code, 1)
            self.assertEqual(stdout2.getvalue(), "")
            self.assertNotIn("Traceback", stderr2.getvalue())
            self.assertIn("call_evidence:", stderr2.getvalue())


class CallEvidenceCliNegativeTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip())
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip())
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_missing_dir_non_dir_and_missing_dll_rejected_before_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            missing_dir = base / "missing_dir"
            file_input = base / "regular.bin"
            file_input.write_bytes(b"data")
            empty_dir = base / "empty"
            empty_dir.mkdir()
            partial_dir = base / "partial"
            partial_dir.mkdir()
            pe, _, _ = _build_pe32_plus()
            (partial_dir / call_evidence.NAMES[0]).write_bytes(pe)
            (partial_dir / call_evidence.NAMES[1]).write_bytes(pe)

            with (
                mock.patch.object(
                    call_evidence,
                    "inspect",
                    side_effect=AssertionError("inspect called on invalid directory/inputs"),
                ),
                mock.patch.object(
                    call_evidence,
                    "_write_atomic",
                    side_effect=AssertionError("_write_atomic called on invalid directory/inputs"),
                ),
            ):
                self._assert_clean_failure([str(missing_dir)], expected_code=2)
                self._assert_clean_failure([str(file_input)], expected_code=2, fragments=("not a directory",))
                self._assert_clean_failure([str(empty_dir)], expected_code=2, fragments=("missing expected image",))
                self._assert_clean_failure(
                    [str(partial_dir)],
                    expected_code=2,
                    fragments=("missing expected image", call_evidence.NAMES[2]),
                )

    def test_dll_guards_reject_directory_unreadable_and_symlink_before_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _populate_valid_build_dir(build_dir)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            out_dir = base / "repo" / "out"

            # 1) Expected DLL path is a directory.
            dll_dir = build_dir / call_evidence.NAMES[1]
            dll_dir.unlink()
            dll_dir.mkdir()
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "directory"),
                    check_subprocess=False,
                )
            self.assertFalse(out_dir.exists())

            # 2) Expected DLL path is unreadable (os.access false or open PermissionError).
            dll_dir.rmdir()
            pe_ok, _, _ = _build_pe32_plus()
            dll_dir.write_bytes(pe_ok)
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                with mock.patch("call_evidence.os.access", return_value=False):
                    self._assert_clean_failure(
                        [str(build_dir)],
                        expected_code=1,
                        fragments=("call_evidence:", "not readable"),
                        check_subprocess=False,
                    )
                with mock.patch.object(Path, "open", side_effect=PermissionError("denied")):
                    self._assert_clean_failure(
                        [str(build_dir)],
                        expected_code=1,
                        fragments=("call_evidence:", "denied"),
                        check_subprocess=False,
                    )
            self.assertFalse(out_dir.exists())

            # 3) Expected DLL path is a live or dangling symlink.
            target_dll = base / "real_target.dll"
            target_dll.write_bytes(pe_ok)
            dll_link = build_dir / call_evidence.NAMES[2]
            dll_link.unlink()
            try:
                dll_link.symlink_to(target_dll)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "symlink"),
                    check_subprocess=False,
                )
                dll_link.unlink()
                dll_link.symlink_to(base / "dangling_target.dll")
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "symlink"),
                    check_subprocess=False,
                )
            self.assertFalse(out_dir.exists())

    def test_output_guards_reject_inside_root_dir_and_file_parent_before_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _populate_valid_build_dir(build_dir)
            repo_dir = base / "repo"
            fake_tool = repo_dir / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            out_file = repo_dir / "out" / "calls" / "six-apis.json"

            # 1) Output inside root (repo_dir has valid DLLs, and out/calls has valid DLLs).
            _populate_valid_build_dir(repo_dir)
            nested_calls = repo_dir / "out" / "calls"
            _populate_valid_build_dir(nested_calls)
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                for overlapping in (repo_dir, nested_calls):
                    with self.subTest(overlapping=overlapping.name):
                        self._assert_clean_failure(
                            [str(overlapping)],
                            expected_code=2,
                            fragments=("separate from the report directory",),
                            check_subprocess=False,
                        )
            self.assertFalse(out_file.exists())
            shutil.rmtree(repo_dir / "out")

            # 2) Output path is an existing directory.
            out_file.mkdir(parents=True)
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "directory"),
                    check_subprocess=False,
                )
            out_file.rmdir()

            # 3) Output parent ('calls') is a regular file.
            out_file.parent.rmdir()
            out_file.parent.write_text("keep-calls", encoding="utf-8")
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "not a directory"),
                    check_subprocess=False,
                )
            self.assertEqual(out_file.parent.read_text(encoding="utf-8"), "keep-calls")

            # 4) Output ancestor ('out') is a regular file.
            out_file.parent.unlink()
            out_file.parent.parent.rmdir()
            out_file.parent.parent.write_text("keep-out", encoding="utf-8")
            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "not a directory"),
                    check_subprocess=False,
                )
            self.assertEqual(out_file.parent.parent.read_text(encoding="utf-8"), "keep-out")

    def test_output_symlink_and_symlink_parent_rejected_before_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _populate_valid_build_dir(build_dir)
            repo_dir = base / "repo"
            fake_tool = repo_dir / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            out_file = repo_dir / "out" / "calls" / "six-apis.json"
            out_file.parent.mkdir(parents=True)
            real_target = base / "real_report.json"
            real_target.write_text("untouched", encoding="utf-8")

            try:
                out_file.symlink_to(real_target)
            except OSError as exc:
                if getattr(exc, "winerror", None) != 1314:
                    raise
                self.skipTest("this Windows account cannot create symlinks (WinError 1314)")

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch.object(call_evidence, "inspect", side_effect=AssertionError("inspect called")),
                mock.patch.object(call_evidence, "_write_atomic", side_effect=AssertionError("write called")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "symlink"),
                    check_subprocess=False,
                )
                out_file.unlink()
                out_file.symlink_to(base / "missing_report.json")
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "symlink"),
                    check_subprocess=False,
                )
                out_file.unlink()
                out_file.parent.rmdir()
                out_file.parent.symlink_to(base / "dangling_calls_dir")
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "not a directory"),
                    check_subprocess=False,
                )
            self.assertEqual(real_target.read_text(encoding="utf-8"), "untouched")

    def test_corrupt_pe_and_dumpbin_failures_exit_1_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _populate_valid_build_dir(build_dir)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            out_dir = base / "repo" / "out"

            # 1) dumpbin SubprocessError and FileNotFoundError on valid synthetic PEs.
            with mock.patch.object(call_evidence, "__file__", str(fake_tool)):
                with mock.patch(
                    "call_evidence.subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(1, ["dumpbin"]),
                ):
                    self._assert_clean_failure(
                        [str(build_dir)],
                        expected_code=1,
                        fragments=("call_evidence:",),
                        check_subprocess=False,
                    )
                with mock.patch(
                    "call_evidence.subprocess.check_output",
                    side_effect=FileNotFoundError("dumpbin missing"),
                ):
                    self._assert_clean_failure(
                        [str(build_dir)],
                        expected_code=1,
                        fragments=("call_evidence:", "dumpbin missing"),
                        check_subprocess=False,
                    )
            self.assertFalse(out_dir.exists())

            # 2) Corrupt PE bytes (tested both in-process and via real subprocess).
            (build_dir / call_evidence.NAMES[0]).write_bytes(b"not a valid PE32+ image")
            with mock.patch.object(call_evidence, "__file__", str(fake_tool)):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:",),
                )
            self.assertFalse(out_dir.exists())

    def test_atomic_write_failures_leave_zero_tmp_files_and_preserve_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            build_dir = base / "build"
            _, disasm = _populate_valid_build_dir(build_dir)
            fake_tool = base / "repo" / "tools" / "call_evidence.py"
            fake_tool.parent.mkdir(parents=True)
            out_file = base / "repo" / "out" / "calls" / "six-apis.json"
            out_file.parent.mkdir(parents=True)
            out_file.write_text("prior-report", encoding="utf-8")

            def fake_dumpbin(cmd: list[str]) -> bytes:
                return disasm[Path(cmd[-1]).name]

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin),
                mock.patch.object(Path, "replace", side_effect=OSError("simulated replace failure")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "simulated replace failure"),
                    check_subprocess=False,
                )
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report")
            self.assertEqual(sorted(p.name for p in out_file.parent.iterdir()), ["six-apis.json"])

            with (
                mock.patch.object(call_evidence, "__file__", str(fake_tool)),
                mock.patch("call_evidence.subprocess.check_output", side_effect=fake_dumpbin),
                mock.patch("call_evidence.tempfile.NamedTemporaryFile", side_effect=OSError("disk full")),
            ):
                self._assert_clean_failure(
                    [str(build_dir)],
                    expected_code=1,
                    fragments=("call_evidence:", "disk full"),
                    check_subprocess=False,
                )
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report")
            self.assertEqual(sorted(p.name for p in out_file.parent.iterdir()), ["six-apis.json"])

    def test_argparse_negative_controls_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "build"
            _populate_valid_build_dir(build_dir)
            for bad_argv in (
                [],
                [str(build_dir), str(build_dir)],
                [str(build_dir), "--unknown-flag"],
            ):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(bad_argv, expected_code=2)


if __name__ == "__main__":
    unittest.main()
