"""CLI, atomic write, path validation, subprocess error, and mutation suite for tools/exe_aot.py."""

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

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "exe_aot.py"
sys.path.insert(0, str(ROOT / "tools"))

import cpu_stdio_exe_aot  # noqa: E402
import exe_aot as tool  # noqa: E402


IMAGE_BASE = 0x140000000
RAW_OFFSET = 0x200
SECTION_ALIGN = 0x1000
TEXT_RVA = 0x1000
RDATA_RVA = 0x2000
SECTION_SIZE = 0x1000


def _rel32(next_ip: int, target: int) -> bytes:
    return struct.pack("<i", target - next_ip)


def build_console_exe(
    *,
    machine: int = 0x8664,
    magic: int = 0x20B,
    subsystem: int = 3,
    characteristics: int = 0x0022,
    include_finish_exe: bool = False,
    extra_imports: list[tuple[bytes, list[bytes | int]]] | None = None,
    entry_mode: str = "stdio",
    export_native_entry: bool = True,
    register_entry_pdata: bool = True,
):
    """Build a minimal synthetic AMD64 PE32+ console EXE for exe_aot CLI tests."""
    text = bytearray(SECTION_SIZE)
    rdata = bytearray(SECTION_SIZE)
    text_cursor = 0
    rdata_cursor = 0

    def emit_text(blob: bytes, align: int = 16) -> int:
        nonlocal text_cursor
        text_cursor = (text_cursor + align - 1) & ~(align - 1)
        rva = TEXT_RVA + text_cursor
        text[text_cursor:text_cursor + len(blob)] = blob
        text_cursor += len(blob)
        return rva

    def alloc_rdata(size: int, align: int = 8) -> int:
        nonlocal rdata_cursor
        rdata_cursor = (rdata_cursor + align - 1) & ~(align - 1)
        rva = RDATA_RVA + rdata_cursor
        rdata_cursor += size
        return rva

    def put_rdata(rva: int, blob: bytes) -> None:
        offset = rva - RDATA_RVA
        rdata[offset:offset + len(blob)] = blob

    stdio_symbols: list[bytes | int] = [b"getchar", b"putchar", b"fflush"]
    modules: list[tuple[bytes, list[bytes | int]]] = [
        (cpu_stdio_exe_aot.STDIO.encode("ascii"), stdio_symbols),
    ]
    if include_finish_exe:
        modules.append((b"cpu_exe_io.dll", [b"finish_exe"]))
    if extra_imports:
        modules.extend(extra_imports)

    import_size = 20 * (len(modules) + 1)
    import_rva = alloc_rdata(import_size, 4)
    iat_slots: dict[tuple[bytes, bytes | int], int] = {}

    for idx, (dll_name, symbols) in enumerate(modules):
        dll_rva = alloc_rdata(len(dll_name) + 1, 2)
        put_rdata(dll_rva, dll_name + b"\0")
        entries: list[int] = []
        for sym in symbols:
            if isinstance(sym, int):
                entries.append(0x8000000000000000 | sym)
            else:
                hint_rva = alloc_rdata(2 + len(sym) + 1, 2)
                put_rdata(hint_rva, b"\0\0" + sym + b"\0")
                entries.append(hint_rva)
        table_bytes = struct.pack(f"<{len(entries) + 1}Q", *entries, 0)
        ilt_rva = alloc_rdata(len(table_bytes), 8)
        put_rdata(ilt_rva, table_bytes)
        iat_rva = alloc_rdata(len(table_bytes), 8)
        put_rdata(iat_rva, table_bytes)
        for sym_idx, sym in enumerate(symbols):
            iat_slots[(dll_name.lower(), sym)] = iat_rva + sym_idx * 8
        put_rdata(import_rva + idx * 20, struct.pack("<5I", ilt_rva, 0, 0, dll_rva, iat_rva))

    getchar_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"getchar")]
    putchar_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"putchar")]
    fflush_slot = iat_slots[(cpu_stdio_exe_aot.STDIO.encode("ascii"), b"fflush")]

    code = bytearray()
    entry_rva = TEXT_RVA + ((text_cursor + 15) & ~15)
    if entry_mode == "stdio":
        code += b"\x48\x83\xEC\x28"
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, getchar_slot)
        code += b"\x89\xC1"
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, putchar_slot)
        code += b"\x31\xC9"
        code += b"\xFF\x15" + _rel32(entry_rva + len(code) + 6, fflush_slot)
        code += b"\x48\x83\xC4\x28\xC3"
    elif entry_mode == "cpuid":
        code += b"\x0F\xA2\x31\xC0\xC3"
    else:
        raise ValueError(f"unknown entry_mode {entry_mode!r}")

    actual_entry = emit_text(bytes(code))
    assert actual_entry == entry_rva
    entry_end = entry_rva + len(code)

    export_rva = 0
    export_size = 0
    if export_native_entry:
        exp_dll_rva = alloc_rdata(len(b"synthetic_console.exe\0"), 2)
        put_rdata(exp_dll_rva, b"synthetic_console.exe\0")
        exp_name_rva = alloc_rdata(len(b"native_entry\0"), 2)
        put_rdata(exp_name_rva, b"native_entry\0")
        eat_rva = alloc_rdata(4, 4)
        put_rdata(eat_rva, struct.pack("<I", entry_rva))
        ent_rva = alloc_rdata(4, 4)
        put_rdata(ent_rva, struct.pack("<I", exp_name_rva))
        eot_rva = alloc_rdata(2, 2)
        put_rdata(eot_rva, struct.pack("<H", 0))
        export_size = 40
        export_rva = alloc_rdata(export_size, 4)
        put_rdata(
            export_rva,
            struct.pack(
                "<IIHHIIIIIII",
                0, 0, 0, 0, exp_dll_rva, 1, 1, 1, eat_rva, ent_rva, eot_rva,
            ),
        )

    pdata_rva = 0
    pdata_size = 0
    if register_entry_pdata:
        unwind_rva = alloc_rdata(4, 4)
        put_rdata(unwind_rva, b"\x01\x00\x00\x00")
        pdata_size = 12
        pdata_rva = alloc_rdata(pdata_size, 4)
        put_rdata(pdata_rva, struct.pack("<III", entry_rva, entry_end, unwind_rva))

    buf = bytearray(RAW_OFFSET + SECTION_SIZE * 2)
    buf[0:2] = b"MZ"
    pe_off = 0x80
    struct.pack_into("<I", buf, 60, pe_off)
    buf[pe_off:pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_off + 4, machine, 2, 0, 0, 0, 0xF0, characteristics)
    opt = pe_off + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        magic, 14, 0, SECTION_SIZE, SECTION_SIZE, 0, entry_rva, TEXT_RVA, IMAGE_BASE,
        SECTION_ALIGN, 0x200, 6, 0, 0, 0, 6, 0, 0, 0x3000, RAW_OFFSET, 0, subsystem,
        0x160, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16,
    )
    dd = opt + 112
    if export_size:
        struct.pack_into("<II", buf, dd + 0 * 8, export_rva, export_size)
    struct.pack_into("<II", buf, dd + 1 * 8, import_rva, import_size)
    if pdata_size:
        struct.pack_into("<II", buf, dd + 3 * 8, pdata_rva, pdata_size)

    sections_off = opt + 0xF0
    sections = [
        (b".text\0\0\0", SECTION_SIZE, TEXT_RVA, SECTION_SIZE, RAW_OFFSET, 0x60000020),
        (b".rdata\0\0", SECTION_SIZE, RDATA_RVA, SECTION_SIZE, RAW_OFFSET + SECTION_SIZE, 0x40000040),
    ]
    for idx, (name, vsize, va, rsize, roff, chars) in enumerate(sections):
        struct.pack_into("<8sIIIIIIHHI", buf, sections_off + idx * 40, name, vsize, va, rsize, roff, 0, 0, 0, 0, chars)

    buf[RAW_OFFSET:RAW_OFFSET + SECTION_SIZE] = text
    buf[RAW_OFFSET + SECTION_SIZE:RAW_OFFSET + SECTION_SIZE * 2] = rdata
    return bytes(buf), {"entry_rva": entry_rva, "import_rva": import_rva}


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = tool.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args):
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=15,
    )


class ExeAotCliPositiveTests(unittest.TestCase):
    def test_no_build_atomic_writes_provenance_and_entrypoints(self):
        raw, _ = build_console_exe(include_finish_exe=True)
        digest = hashlib.sha256(raw).hexdigest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            image = root / "sample.exe"
            image.write_bytes(raw)
            out_dir = root / "out"
            out_dir.mkdir()
            # Pre-seed stale build artifacts that must be cleared once --no-build succeeds.
            for ext in (".exe", ".obj", ".lib", ".exp"):
                (out_dir / f"sample_aot{ext}").write_bytes(b"stale")

            seen_replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                self.assertEqual(src_path.name, dst_path.name)
                return real_replace(src, dst)

            with mock.patch.object(tool.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main([str(image), str(out_dir), "--entry-export", "native_entry", "--no-build"])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("translated with profile console-stdio:", out)
            self.assertEqual(len(seen_replaces), 2)
            self.assertEqual([dst.name for _, dst in seen_replaces], ["sample_aot.cpp", "report.json"])
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json", "sample_aot.cpp"])

            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "translated")
            self.assertEqual(report["profile"], "console-stdio")
            self.assertEqual(report["sha256"], digest)
            self.assertEqual(report["entry_export"], "native_entry")
            self.assertNotIn("exe", report)
            self.assertEqual(
                (out_dir / "sample_aot.cpp").read_text(encoding="ascii"),
                cpu_stdio_exe_aot.translate(image, entry_export="native_entry"),
            )

            # Verify sys.argv main(None), runpy __main__, and subprocess invocation on plain EXE.
            raw_plain, _ = build_console_exe(include_finish_exe=False)
            plain_image = root / "plain.exe"
            plain_image.write_bytes(raw_plain)
            out_sub = root / "nested" / "plain_out"
            cli_argv = [str(TOOL), str(plain_image), str(out_sub), "--no-build"]
            with mock.patch.object(sys, "argv", cli_argv):
                code_none, out_none, err_none = _run_main(None)
            self.assertEqual(code_none, 0, err_none)
            self.assertIn("translated with profile console-stdio:", out_none)

            out_runpy = root / "runpy_out"
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", [str(TOOL), str(plain_image), str(out_runpy), "--no-build"]),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertIn("translated with profile console-stdio:", stdout.getvalue())

            out_proc = root / "proc_out"
            proc = _run_subprocess(str(plain_image), str(out_proc), "--no-build")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("translated with profile console-stdio:", proc.stdout)
            self.assertEqual(sorted(p.name for p in out_proc.iterdir()), ["plain_aot.cpp", "report.json"])

    def test_build_success_with_mocked_compiler_passes_bounded_timeout(self):
        raw, _ = build_console_exe(include_finish_exe=False)
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            image = root / "app.exe"
            image.write_bytes(raw)
            out_dir = root / "build_out"
            captured = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                (Path(kwargs["cwd"]) / "app_aot.exe").write_bytes(b"MZ-fake-exe")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                mock.patch.object(tool.shutil, "which", return_value=r"C:\fake\cl.exe"),
                mock.patch.object(tool.subprocess, "run", side_effect=fake_run),
            ):
                code, out, err = _run_main([str(image), str(out_dir), "--cxx", "custom-cl"])

            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("built with profile console-stdio:", out)
            self.assertEqual(captured["cmd"][0], r"C:\fake\cl.exe")
            self.assertIn("/NOIMPLIB", captured["cmd"])
            self.assertIn("/NOEXP", captured["cmd"])
            self.assertEqual(captured["kwargs"]["timeout"], tool.BUILD_TIMEOUT_S)
            self.assertGreater(captured["kwargs"]["timeout"], 0)
            self.assertEqual(captured["kwargs"]["cwd"], out_dir.resolve())
            self.assertEqual(captured["kwargs"]["stdin"], subprocess.DEVNULL)
            report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "built")
            self.assertEqual(report["compiler"], "cl.exe")
            self.assertEqual(Path(report["exe"]).name, "app_aot.exe")


class ExeAotCliRefusalTests(unittest.TestCase):
    def test_header_imports_and_translate_refusals_clear_stale_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # 1) Corrupt non-PE and truncated PE header -> header refusal.
            non_pe = root / "bad.exe"
            non_pe.write_bytes(b"not a PE image\n")

            raw_pe, _ = build_console_exe()
            trunc_pe = root / "trunc_hdr.exe"
            trunc_pe.write_bytes(raw_pe[:0x90])

            for bad_img in (non_pe, trunc_pe):
                out_dir = root / f"out_{bad_img.stem}"
                out_dir.mkdir()
                (out_dir / f"{bad_img.stem}_aot.cpp").write_text("stale\n", encoding="ascii")
                (out_dir / f"{bad_img.stem}_aot.exe").write_bytes(b"stale")
                code, _, err = _run_main([str(bad_img), str(out_dir), "--no-build"])
                self.assertEqual(code, 1)
                self.assertIn("refused at header:", err)
                self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json"])
                rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
                self.assertEqual((rep["status"], rep["refusal"]["stage"]), ("refused", "header"))

            # PEFormatError during parse_data_directories -> header refusal without traceback.
            valid_img = root / "valid_pe.exe"
            valid_img.write_bytes(raw_pe)
            with mock.patch.object(tool.pefile.PE, "parse_data_directories",
                                   side_effect=tool.pefile.PEFormatError("corrupt import table")):
                code, _, err = _run_main([str(valid_img), str(root / "out_corrupt_dd"), "--no-build"])
            self.assertEqual(code, 1)
            self.assertIn("refused at header: not a PE image:", err)
            self.assertIn("corrupt import table", err)

            # 2) Header field refusals: machine, magic, DLL, unlinked.
            for label, kwargs in (
                ("machine", dict(machine=0x014C)),
                ("magic", dict(magic=0x10B)),
                ("dll", dict(characteristics=0x2022)),
                ("unlinked", dict(characteristics=0x0000)),
            ):
                with self.subTest(header_case=label):
                    raw, _ = build_console_exe(**kwargs)
                    img = root / f"{label}.exe"
                    img.write_bytes(raw)
                    code, _, err = _run_main([str(img), str(root / f"out_{label}"), "--no-build"])
                    self.assertEqual(code, 1)
                    self.assertIn("refused at header:", err)

            # 3) Imports refusal with multiple unbound imports prints "also unbound:" lines.
            raw_unbound, _ = build_console_exe(extra_imports=[(b"kernel32.dll", [b"ExitProcess", 42])])
            unbound_img = root / "unbound.exe"
            unbound_img.write_bytes(raw_unbound)
            code, _, err = _run_main([str(unbound_img), str(root / "out_unbound"), "--no-build"])
            self.assertEqual(code, 1)
            self.assertIn("refused at imports:", err)
            self.assertIn("kernel32.dll!ExitProcess", err)
            self.assertIn("also unbound: kernel32.dll!#42", err)

            # 4) Translate refusal (cpuid) removes stale cpp/exe and records rva.
            raw_cpuid, meta_cpuid = build_console_exe(entry_mode="cpuid")
            cpuid_img = root / "cpuid.exe"
            cpuid_img.write_bytes(raw_cpuid)
            out_cpuid = root / "out_cpuid"
            out_cpuid.mkdir()
            (out_cpuid / "cpuid_aot.cpp").write_text("stale\n", encoding="ascii")
            (out_cpuid / "cpuid_aot.exe").write_bytes(b"stale")
            code, _, err = _run_main([str(cpuid_img), str(out_cpuid), "--no-build"])
            self.assertEqual(code, 1)
            self.assertIn("refused at translate:", err)
            self.assertIn("cpuid", err)
            self.assertEqual(sorted(p.name for p in out_cpuid.iterdir()), ["report.json"])
            rep_cpuid = json.loads((out_cpuid / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(rep_cpuid["refusal"]["rva"], f"{meta_cpuid['entry_rva']:#x}")

    def test_build_refusals_handle_missing_timeout_oserror_and_subprocess_error(self):
        raw, _ = build_console_exe()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            image = root / "app.exe"
            image.write_bytes(raw)

            # 1) Missing compiler -> refused at build, C++ kept, no EXE.
            out_missing = root / "out_missing"
            with mock.patch.object(tool.shutil, "which", return_value=None):
                code, _, err = _run_main([str(image), str(out_missing), "--cxx", "no-such-cl"])
            self.assertEqual(code, 1)
            self.assertIn("refused at build:", err)
            self.assertIn("not found", err)
            self.assertEqual(sorted(p.name for p in out_missing.iterdir()), ["app_aot.cpp", "report.json"])

            # 2) Non-zero exit code + partial EXE removed -> refused at build, tail lines reported.
            out_fail = root / "out_fail"

            def fail_compile(cmd, **kwargs):
                (Path(kwargs["cwd"]) / "app_aot.exe").write_bytes(b"partial")
                return subprocess.CompletedProcess(cmd, 2, stdout="fatal error C1001\n", stderr="detail line\n")

            with (
                mock.patch.object(tool.shutil, "which", return_value=r"C:\fake\cl.exe"),
                mock.patch.object(tool.subprocess, "run", side_effect=fail_compile),
            ):
                code, _, err = _run_main([str(image), str(out_fail)])
            self.assertEqual(code, 1)
            self.assertIn("refused at build: cl.exe exited 2", err)
            self.assertIn("fatal error C1001", err)
            self.assertEqual(sorted(p.name for p in out_fail.iterdir()), ["app_aot.cpp", "report.json"])

            # 3) TimeoutExpired (bytes stdout/stderr) -> Refusal('build', ... timed out ...).
            out_timeout = root / "out_timeout"
            timeout_exc = subprocess.TimeoutExpired(
                ["cl.exe"], tool.BUILD_TIMEOUT_S, output=b"compiling...\n", stderr=b"hung\n"
            )
            with (
                mock.patch.object(tool.shutil, "which", return_value=r"C:\fake\cl.exe"),
                mock.patch.object(tool.subprocess, "run", side_effect=timeout_exc),
            ):
                code, _, err = _run_main([str(image), str(out_timeout)])
            self.assertEqual(code, 1)
            self.assertNotIn("Traceback", err)
            self.assertIn("refused at build: cl.exe timed out", err)
            self.assertIn("compiling...", err)
            rep_to = json.loads((out_timeout / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((rep_to["status"], rep_to["refusal"]["stage"]), ("refused", "build"))

            # 4) OSError & SubprocessError in subprocess.run -> Refusal('build', ...).
            for exc, label in (
                (OSError("exec format failure"), "oserror"),
                (subprocess.SubprocessError("generic subprocess failure"), "subproc"),
            ):
                with self.subTest(build_exc=label):
                    out_err = root / f"out_{label}"
                    with (
                        mock.patch.object(tool.shutil, "which", return_value=r"C:\fake\cl.exe"),
                        mock.patch.object(tool.subprocess, "run", side_effect=exc),
                    ):
                        code, _, err = _run_main([str(image), str(out_err)])
                    self.assertEqual(code, 1)
                    self.assertNotIn("Traceback", err)
                    self.assertIn("refused at build: failed to run cl.exe:", err)
                    self.assertIn(str(exc), err)
                    self.assertEqual(sorted(p.name for p in out_err.iterdir()), ["app_aot.cpp", "report.json"])


class ExeAotCliGuardAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(self, argv, expected_fragment, *, check_subprocess=True):
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

    def test_argparse_errors_exit_nonzero_cleanly(self):
        for bad_argv, fragment in (
            ([], "required"),
            (["only_one_arg"], "required"),
            (["img.exe", "out_dir", "--no-such-flag"], "unrecognized"),
        ):
            with self.subTest(bad_argv=bad_argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as exited:
                        tool.main(bad_argv)
                self.assertNotEqual(exited.exception.code, 0)
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertIn(fragment, stderr.getvalue())

    def test_upfront_image_and_out_dir_guards_kill_mutations(self):
        raw, _ = build_console_exe()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            valid_img = root / "valid.exe"
            valid_img.write_bytes(raw)
            img_dir = root / "img_dir"
            img_dir.mkdir()
            missing_img = root / "missing.exe"
            ghost_out = root / "ghost_out"

            # 1) Missing image -> exit 2, out_dir not even created.
            self._assert_exit_2_clean([str(missing_img), str(ghost_out), "--no-build"], "not found")
            self.assertFalse(ghost_out.exists())

            # 2) Image is a directory (and empty string via Path("") -> ".") -> exit 2, out_dir not created.
            self._assert_exit_2_clean([str(img_dir), str(ghost_out), "--no-build"], "directory")
            self.assertFalse(ghost_out.exists())
            self._assert_exit_2_clean(["", str(ghost_out), "--no-build"], "directory", check_subprocess=False)
            self.assertFalse(ghost_out.exists())

            # 3) Direct convert() with empty strings or non-regular image.
            with self.assertRaises(ValueError):
                tool.convert("", ghost_out, compile_exe=False)
            with self.assertRaises(ValueError):
                tool.convert(valid_img, "", compile_exe=False)
            with mock.patch.object(Path, "exists", return_value=True), \
                 mock.patch.object(Path, "is_dir", return_value=False), \
                 mock.patch.object(Path, "is_file", return_value=False):
                with self.assertRaisesRegex(ValueError, "not a regular file"):
                    tool.convert(valid_img, ghost_out, compile_exe=False)

            # 4) out_dir is an existing regular file -> exit 2, regular file bytes preserved.
            out_file = root / "existing_file"
            out_file.write_bytes(b"keep-file-intact")
            self._assert_exit_2_clean([str(valid_img), str(out_file), "--no-build"], "not a directory")
            self.assertEqual(out_file.read_bytes(), b"keep-file-intact")

            # 5) out_dir whose parent is a file -> mkdir raises OSError -> exit 2 cleanly.
            self._assert_exit_2_clean([str(valid_img), str(out_file / "child"), "--no-build"], "")

            # 6) out_dir is a broken/non-dir symlink (real if allowed + mocked).
            with mock.patch.object(Path, "is_symlink", autospec=True,
                                   side_effect=lambda p: p.resolve() == ghost_out.resolve()):
                self._assert_exit_2_clean(
                    [str(valid_img), str(ghost_out), "--no-build"],
                    "not a directory",
                    check_subprocess=False,
                )

    def test_collision_directory_and_symlink_output_guards_kill_mutations(self):
        raw, _ = build_console_exe()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # 1) Collision with report.json directly and via hardlink alias.
            col_dir = root / "col"
            col_dir.mkdir()
            report_input = col_dir / "report.json"
            report_input.write_bytes(b'{"sentinel":1}\n')
            self._assert_exit_2_clean(
                [str(report_input), str(col_dir), "--no-build"],
                "output would overwrite the input",
            )
            self.assertEqual(report_input.read_bytes(), b'{"sentinel":1}\n')

            hardlink_img = root / "alias_input.exe"
            hardlink_img.hardlink_to(report_input)
            self._assert_exit_2_clean(
                [str(hardlink_img), str(col_dir), "--no-build"],
                "output would overwrite the input",
            )
            self.assertEqual(hardlink_img.read_bytes(), b'{"sentinel":1}\n')

            # 2) Collision with source (_aot.cpp) via hardlink alias.
            src_alias_dir = root / "src_alias"
            src_alias_dir.mkdir()
            src_target = src_alias_dir / "probe_aot.cpp"
            src_target.write_bytes(b"sentinel-cpp")
            probe_img = root / "probe.exe"
            probe_img.hardlink_to(src_target)
            self._assert_exit_2_clean(
                [str(probe_img), str(src_alias_dir), "--no-build"],
                "output would overwrite the input",
            )
            self.assertEqual(src_target.read_bytes(), b"sentinel-cpp")

            # 3) report.json or source or exe is an existing directory -> exit 2, prior files intact.
            dir_guard = root / "dir_guard"
            dir_guard.mkdir()
            img = root / "target.exe"
            img.write_bytes(raw)
            prior_src = dir_guard / "target_aot.cpp"
            prior_src.write_text("prior-cpp\n", encoding="ascii")
            (dir_guard / "report.json").mkdir()
            self._assert_exit_2_clean([str(img), str(dir_guard), "--no-build"], "report path is a directory")
            self.assertEqual(prior_src.read_text(encoding="ascii"), "prior-cpp\n")
            (dir_guard / "report.json").rmdir()

            prior_src.unlink()
            (dir_guard / "target_aot.cpp").mkdir()
            prior_rep = dir_guard / "report.json"
            prior_rep.write_text("prior-report\n", encoding="utf-8")
            self._assert_exit_2_clean([str(img), str(dir_guard), "--no-build"], "source path is a directory")
            self.assertEqual(prior_rep.read_text(encoding="utf-8"), "prior-report\n")
            (dir_guard / "target_aot.cpp").rmdir()

            (dir_guard / "target_aot.exe").mkdir()
            self._assert_exit_2_clean([str(img), str(dir_guard), "--no-build"], "artifact path is a directory")
            self.assertEqual(prior_rep.read_text(encoding="utf-8"), "prior-report\n")
            (dir_guard / "target_aot.exe").rmdir()

            # 4) Symlink guards on report.json and source (real symlink if permitted, plus mock fallback).
            sym_dir = root / "sym_guard"
            sym_dir.mkdir()
            real_target = root / "real_target.txt"
            real_target.write_text("do-not-touch\n", encoding="utf-8")
            for label, target_name in (("report", "report.json"), ("source", "target_aot.cpp")):
                candidate = sym_dir / target_name
                try:
                    candidate.symlink_to(real_target)
                    created_real = True
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 1314:
                        raise
                    created_real = False

                if created_real:
                    self._assert_exit_2_clean(
                        [str(img), str(sym_dir), "--no-build"],
                        f"{label} path is a symbolic link",
                    )
                    candidate.unlink()
                else:
                    with mock.patch.object(Path, "is_symlink", autospec=True,
                                           side_effect=lambda p, expected=candidate: p.resolve() == expected.resolve()):
                        self._assert_exit_2_clean(
                            [str(img), str(sym_dir), "--no-build"],
                            f"{label} path is a symbolic link",
                            check_subprocess=False,
                        )
            self.assertEqual(real_target.read_text(encoding="utf-8"), "do-not-touch\n")

    def test_atomic_write_failures_preserve_prior_files_and_leave_zero_temp_dirs(self):
        raw, _ = build_console_exe()
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            image = root / "app.exe"
            image.write_bytes(raw)
            out_dir = root / "atomic_out"
            out_dir.mkdir()
            src_path = out_dir / "app_aot.cpp"
            exe_path = out_dir / "app_aot.exe"
            rep_path = out_dir / "report.json"

            # 1) os.replace fails on source write -> preserves prior source, exe, report; zero temp dirs.
            src_path.write_text("prior-source\n", encoding="ascii")
            exe_path.write_bytes(b"prior-exe")
            rep_path.write_text("prior-report\n", encoding="utf-8")

            def fail_first_replace(src, dst):
                self.assertEqual(Path(dst).resolve(), src_path.resolve())
                self.assertTrue(Path(src).is_file())
                self.assertEqual(Path(src).parent.parent.resolve(), out_dir.resolve())
                raise OSError("simulated source replace failure")

            with mock.patch.object(tool.os, "replace", side_effect=fail_first_replace):
                self._assert_exit_2_clean(
                    [str(image), str(out_dir), "--no-build"],
                    "simulated source replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(src_path.read_text(encoding="ascii"), "prior-source\n")
            self.assertEqual(exe_path.read_bytes(), b"prior-exe")
            self.assertEqual(rep_path.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["app_aot.cpp", "app_aot.exe", "report.json"])

            # 2) Staging write_text fails on source -> os.replace never called, prior files intact, zero temp dirs.
            with (
                mock.patch.object(tool.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated stage source failure")),
            ):
                self._assert_exit_2_clean(
                    [str(image), str(out_dir), "--no-build"],
                    "simulated stage source failure",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(src_path.read_text(encoding="ascii"), "prior-source\n")
            self.assertEqual(rep_path.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["app_aot.cpp", "app_aot.exe", "report.json"])

            # 3) os.replace fails on report.json write (succeeds on source) -> preserves prior report.json, zero temp dirs.
            real_replace = os.replace

            def fail_report_replace(src, dst):
                if Path(dst).resolve() == rep_path.resolve():
                    self.assertTrue(Path(src).is_file())
                    self.assertEqual(Path(src).parent.parent.resolve(), out_dir.resolve())
                    raise OSError("simulated report replace failure")
                return real_replace(src, dst)

            with mock.patch.object(tool.os, "replace", side_effect=fail_report_replace):
                self._assert_exit_2_clean(
                    [str(image), str(out_dir), "--no-build"],
                    "simulated report replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(rep_path.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["app_aot.cpp", "report.json"])

            # 4) Staging write_text fails on report.json -> preserves prior report.json, zero temp dirs.
            real_write_text = Path.write_text

            def fail_report_stage(path_self, data, *args, **kwargs):
                if path_self.name == "report.json":
                    raise OSError("simulated stage report failure")
                return real_write_text(path_self, data, *args, **kwargs)

            with mock.patch.object(Path, "write_text", autospec=True, side_effect=fail_report_stage):
                self._assert_exit_2_clean(
                    [str(image), str(out_dir), "--no-build"],
                    "simulated stage report failure",
                    check_subprocess=False,
                )
            self.assertEqual(rep_path.read_text(encoding="utf-8"), "prior-report\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["app_aot.cpp", "report.json"])

    def test_main_catches_oserror_valueerror_and_subproc_error_without_traceback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            out_dir = root / "out"
            for exc in (
                OSError("simulated io error"),
                ValueError("simulated value error"),
                subprocess.SubprocessError("simulated subprocess error"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with mock.patch.object(tool, "convert", side_effect=exc):
                        self._assert_exit_2_clean(
                            ["any.exe", str(out_dir)],
                            str(exc),
                            check_subprocess=False,
                        )


if __name__ == "__main__":
    unittest.main()
