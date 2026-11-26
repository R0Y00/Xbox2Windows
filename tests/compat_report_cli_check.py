"""CLI, atomic output, path/error guard, and mutation suite for tools/compat_report.py."""

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

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compat_report.py"
sys.path.insert(0, str(ROOT / "tools"))

import compat_report  # noqa: E402


SECTION_RVA = 0x1000
RAW_OFFSET = 0x200
RAW_SIZE = 0x1000
IMAGE_BASE = 0x180000000

CLEAN_IMPORTS = [
    ("normal", "kernel32.dll", "ExitProcess"),
    ("normal", "xmem.dll", "XMemAlloc"),
    ("delay", "xgameruntime.dll", "XUserGetId"),
]
UNKNOWN_IMPORTS = CLEAN_IMPORTS + [("normal", "mystery.dll", "MissingExport")]


def _build_synthetic_pe32_plus(imports, *, machine=0x8664):
    """Build a minimal synthetic PE32+ image with normal and delay imports."""
    data = bytearray(RAW_SIZE)
    cursor = 0

    def alloc(size, align):
        nonlocal cursor
        cursor = (cursor + align - 1) & ~(align - 1)
        rva = SECTION_RVA + cursor
        if cursor + size > len(data):
            raise AssertionError("synthetic section overflow")
        cursor += size
        return rva

    def put(rva, blob):
        offset = rva - SECTION_RVA
        data[offset:offset + len(blob)] = blob

    module_rvas = {}
    for module in sorted({mod for _, mod, _ in imports}):
        raw = module.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        module_rvas[module] = rva

    hint_rvas = {}
    for kind, module, symbol in imports:
        raw = struct.pack("<H", 0) + symbol.encode("ascii") + b"\0"
        rva = alloc(len(raw), 2)
        put(rva, raw)
        hint_rvas[(kind, module, symbol)] = rva

    groups = {}
    for kind, module, symbol in imports:
        groups.setdefault((kind, module), []).append(symbol)

    tables = {}
    for key in sorted(groups):
        values = [hint_rvas[(key[0], key[1], sym)] for sym in groups[key]]
        blob = b"".join(struct.pack("<Q", value) for value in [*values, 0])
        lookup = alloc(len(blob), 8)
        put(lookup, blob)
        iat = alloc(len(blob), 8)
        put(iat, blob)
        tables[key] = (lookup, iat)

    normal_groups = sorted(k for k in groups if k[0] == "normal")
    delay_groups = sorted(k for k in groups if k[0] == "delay")
    normal_rva = normal_size = delay_rva = delay_size = 0

    if normal_groups:
        normal_size = 20 * (len(normal_groups) + 1)
        normal_rva = alloc(normal_size, 4)
        for idx, key in enumerate(normal_groups):
            lookup, iat = tables[key]
            put(normal_rva + idx * 20, struct.pack("<5I", lookup, 0, 0, module_rvas[key[1]], iat))

    if delay_groups:
        delay_size = 32 * (len(delay_groups) + 1)
        delay_rva = alloc(delay_size, 4)
        for idx, key in enumerate(delay_groups):
            lookup, iat = tables[key]
            put(delay_rva + idx * 32, struct.pack("<8I", 1, module_rvas[key[1]], 0, iat, lookup, 0, 0, 0))

    pe_offset = 0x80
    buf = bytearray(RAW_OFFSET + RAW_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 60, pe_offset)
    buf[pe_offset:pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, pe_offset + 4, machine, 1, 0, 0, 0, 240, 0x2022)
    opt = pe_offset + 24
    struct.pack_into(
        "<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
        buf,
        opt,
        0x20B,
        0,
        0,
        0,
        RAW_SIZE,
        0,
        0,
        SECTION_RVA,
        IMAGE_BASE,
        0x1000,
        0x200,
        6,
        0,
        0,
        0,
        6,
        0,
        0,
        0x2000,
        RAW_OFFSET,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        16,
    )
    sec = opt + 240
    struct.pack_into(
        "<8sIIIIIIHHI",
        buf,
        sec,
        b".rdata\0\0",
        RAW_SIZE,
        SECTION_RVA,
        RAW_SIZE,
        RAW_OFFSET,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    directories = opt + 112
    if normal_groups:
        struct.pack_into("<II", buf, directories + 1 * 8, normal_rva, normal_size)
    if delay_groups:
        struct.pack_into("<II", buf, directories + 13 * 8, delay_rva, delay_size)
    buf[RAW_OFFSET:RAW_OFFSET + RAW_SIZE] = data
    return bytes(buf)


def _run_main(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = compat_report.main(argv)
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


class CompatReportCliPositiveTests(unittest.TestCase):
    def test_nested_dirs_atomic_replace_and_exit_0_clean(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "clean.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            json_out = root / "nested" / "json" / "report.json"
            md_out = root / "nested" / "md" / "report.md"
            self.assertFalse(json_out.parent.exists())
            self.assertFalse(md_out.parent.exists())

            replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with mock.patch.object(compat_report.os, "replace", side_effect=tracking_replace):
                code, out, err = _run_main(
                    [str(image), "--json", str(json_out), "--markdown", str(md_out)]
                )
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertEqual(len(replaces), 2)
            self.assertEqual([p.name for p in json_out.parent.iterdir()], ["report.json"])
            self.assertEqual([p.name for p in md_out.parent.iterdir()], ["report.md"])

            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], compat_report.SCHEMA)
            self.assertEqual(report["inputs"]["image"], "clean.exe")
            routing = report["sections"]["import_routing"]
            self.assertEqual(routing["summary"]["imports"], 3)
            self.assertEqual(routing["summary"]["unknown"], 0)
            md_text = md_out.read_text(encoding="utf-8")
            self.assertIn("## Summary", md_text)
            self.assertIn("## Unknown imports", md_text)
            self.assertIn("None.", md_text)

    def test_exit_1_unknown_import_and_atomic_overwrite_of_existing_outputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_pe = root / "clean.exe"
            unknown_pe = root / "unknown.exe"
            clean_pe.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            unknown_pe.write_bytes(_build_synthetic_pe32_plus(UNKNOWN_IMPORTS))

            out_dir = root / "reports"
            out_dir.mkdir()
            json_out = out_dir / "report.json"
            md_out = out_dir / "report.md"
            json_out.write_text("stale-json\n", encoding="utf-8")
            md_out.write_text("stale-md\n", encoding="utf-8")

            proc = _run_subprocess(
                str(unknown_pe), "--json", str(json_out), "--markdown", str(md_out)
            )
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("unsupported: unknown.exe -> mystery.dll!MissingExport", proc.stdout)
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json", "report.md"])

            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(report["sections"]["import_routing"]["summary"]["unknown"], 1)
            self.assertIn("mystery.dll", md_out.read_text(encoding="utf-8"))

            proc_clean = _run_subprocess(
                str(clean_pe), "--json", str(json_out), "--markdown", str(md_out)
            )
            self.assertEqual(proc_clean.returncode, 0, proc_clean.stderr)
            self.assertEqual(proc_clean.stdout, "")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json", "report.md"])
            self.assertEqual(
                json.loads(json_out.read_text(encoding="utf-8"))["sections"]["import_routing"]["summary"]["unknown"],
                0,
            )

    def test_main_sys_argv_forwarding_and_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "entry.exe"
            json_out = root / "entry.json"
            md_out = root / "entry.md"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            cli_argv = [str(TOOL), str(image), "--json", str(json_out), "--markdown", str(md_out)]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertTrue(json_out.is_file() and md_out.is_file())

            json_out.unlink()
            md_out.unlink()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(json_out.is_file() and md_out.is_file())


class CompatReportCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_exit_2_clean(self, argv, *, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, 2, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertIn("compat_report:", err)
        self.assertIn(expected_fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("compat_report:", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)

    def test_missing_directory_corrupt_and_wrong_machine_pe_exit_2(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.exe"
            json_out = root / "nested" / "report.json"
            md_out = root / "nested" / "report.md"
            self._assert_exit_2_clean(
                [str(missing), "--json", str(json_out), "--markdown", str(md_out)],
                expected_fragment="missing.exe",
            )
            self.assertFalse(json_out.parent.exists())

            image_dir = root / "dir.exe"
            image_dir.mkdir()
            with mock.patch.object(compat_report, "build_report") as build_mock:
                self._assert_exit_2_clean(
                    [str(image_dir), "--json", str(json_out), "--markdown", str(md_out)],
                    expected_fragment="directory",
                    check_subprocess=False,
                )
            build_mock.assert_not_called()
            self.assertFalse(json_out.parent.exists())
            proc = _run_subprocess(str(image_dir), "--json", str(json_out), "--markdown", str(md_out))
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("compat_report:", proc.stderr)
            self.assertIn("directory", proc.stderr)

            corrupt = root / "corrupt.exe"
            corrupt.write_bytes(b"NOT_A_PE_IMAGE")
            self._assert_exit_2_clean(
                [str(corrupt), "--json", str(json_out), "--markdown", str(md_out)],
                expected_fragment="corrupt.exe",
            )
            self.assertFalse(json_out.parent.exists())

            x86_pe = root / "x86.exe"
            x86_pe.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS, machine=0x014C))
            self._assert_exit_2_clean(
                [str(x86_pe), "--json", str(json_out), "--markdown", str(md_out)],
                expected_fragment="IMAGE_FILE_MACHINE_I386",
            )

    def test_struct_error_in_pe_parse_returns_exit_2_without_traceback(self):
        # Kills removal of struct.error from the PE parse exception guard.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "sample.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            json_out = root / "out" / "report.json"
            md_out = root / "out" / "report.md"

            with mock.patch.object(
                compat_report, "import_tables", side_effect=struct.error("truncated import table")
            ):
                self._assert_exit_2_clean(
                    [str(image), "--json", str(json_out), "--markdown", str(md_out)],
                    expected_fragment="truncated import table",
                    check_subprocess=False,
                )
            self.assertFalse(json_out.parent.exists())

            with mock.patch.object(
                compat_report.pefile, "PE", side_effect=struct.error("truncated PE header")
            ):
                self._assert_exit_2_clean(
                    [str(image), "--json", str(json_out), "--markdown", str(md_out)],
                    expected_fragment="truncated PE header",
                    check_subprocess=False,
                )
            self.assertFalse(json_out.parent.exists())

    def test_output_directory_rejected_upfront_before_building_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            json_file = root / "json_parent" / "report.json"
            md_file = root / "md_parent" / "report.md"

            for bad_argv, missing_parent in (
                ([str(image), "--json", str(existing_dir), "--markdown", str(md_file)], md_file.parent),
                ([str(image), "--json", str(json_file), "--markdown", str(existing_dir)], json_file.parent),
            ):
                with self.subTest(argv=bad_argv):
                    with mock.patch.object(compat_report, "build_report") as build_mock:
                        self._assert_exit_2_clean(
                            bad_argv,
                            expected_fragment="existing directory",
                            check_subprocess=False,
                        )
                    build_mock.assert_not_called()
                    self.assertFalse(missing_parent.exists())
                    proc = _run_subprocess(*bad_argv)
                    self.assertEqual(proc.returncode, 2)
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn("existing directory", proc.stderr)
                    self.assertFalse(missing_parent.exists())

    def test_output_paths_colliding_or_containing_each_other_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent --json and --markdown colliding via resolve() (kills resolve() check).
            ghost_json = root / "out" / "report.out"
            alias_md = root / "out" / ".." / "out" / "report.out"
            with mock.patch.object(compat_report, "build_report") as build_mock:
                self._assert_exit_2_clean(
                    [str(image), "--json", str(ghost_json), "--markdown", str(alias_md)],
                    expected_fragment="collide",
                )
            build_mock.assert_not_called()
            self.assertFalse(ghost_json.parent.exists())

            # 2) Pre-existing hardlink alias where resolve() differs (kills samefile() check).
            real_out = root / "report_a.out"
            link_out = sub / "report_b.out"
            real_out.write_text("keep-a\n", encoding="utf-8")
            os.link(real_out, link_out)
            self.assertNotEqual(real_out.resolve(), link_out.resolve())
            self.assertTrue(real_out.samefile(link_out))

            with mock.patch.object(compat_report, "build_report") as build_mock:
                self._assert_exit_2_clean(
                    [str(image), "--json", str(real_out), "--markdown", str(link_out)],
                    expected_fragment="collide",
                )
            build_mock.assert_not_called()
            self.assertEqual(real_out.read_text(encoding="utf-8"), "keep-a\n")

            # 3) --json and --markdown containing one another (kills parent containment guard).
            parent_out = root / "nested_pair" / "item"
            child_out = parent_out / "child.md"
            for j_path, m_path in ((parent_out, child_out), (child_out, parent_out)):
                with mock.patch.object(compat_report, "build_report") as build_mock:
                    self._assert_exit_2_clean(
                        [str(image), "--json", str(j_path), "--markdown", str(m_path)],
                        expected_fragment="contain one another",
                        check_subprocess=False,
                    )
                build_mock.assert_not_called()
                self.assertFalse(parent_out.parent.exists())

    def test_output_collision_with_image_via_resolve_and_samefile_rejected_upfront(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            other_out = root / "other_out" / "side.txt"

            # 1) Non-existent image + alias colliding via resolve() (kills resolve() vs image).
            ghost_img = root / "missing.exe"
            alias_path = root / "sub" / ".." / "missing.exe"
            for argv in (
                [str(ghost_img), "--json", str(alias_path), "--markdown", str(other_out)],
                [str(ghost_img), "--json", str(other_out), "--markdown", str(alias_path)],
            ):
                with mock.patch.object(compat_report, "build_report") as build_mock:
                    self._assert_exit_2_clean(argv, expected_fragment="collides with image")
                build_mock.assert_not_called()
                self.assertFalse(other_out.parent.exists())

            # 2) Existing image + hardlink alias where resolve() differs (kills samefile() vs image).
            real_img = root / "real.exe"
            hardlink = sub / "hardlink.out"
            pe_bytes = _build_synthetic_pe32_plus(CLEAN_IMPORTS)
            real_img.write_bytes(pe_bytes)
            os.link(real_img, hardlink)
            self.assertNotEqual(real_img.resolve(), hardlink.resolve())
            self.assertTrue(real_img.samefile(hardlink))

            for argv in (
                [str(real_img), "--json", str(hardlink), "--markdown", str(other_out)],
                [str(real_img), "--json", str(other_out), "--markdown", str(hardlink)],
            ):
                with mock.patch.object(compat_report, "build_report") as build_mock:
                    self._assert_exit_2_clean(argv, expected_fragment="collides with image")
                build_mock.assert_not_called()
                self.assertEqual(real_img.read_bytes(), pe_bytes)
                self.assertFalse(other_out.parent.exists())

    def test_os_replace_and_staging_failures_leave_zero_temp_dirs_and_preserve_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            out_dir = root / "out"
            json_out = out_dir / "report.json"
            md_out = out_dir / "report.md"

            def failing_replace(src, dst):
                src_path = Path(src)
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                raise OSError("simulated replace failure")

            # 1) Fresh outputs: os.replace OSError -> exit 2, zero output files and zero temp dirs.
            with mock.patch.object(compat_report.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    [str(image), "--json", str(json_out), "--markdown", str(md_out)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(list(out_dir.iterdir()), [])

            # 2) Pre-existing outputs: os.replace OSError -> prior files preserved, zero temp dirs.
            json_out.write_text("prior-json\n", encoding="utf-8")
            md_out.write_text("prior-md\n", encoding="utf-8")
            with mock.patch.object(compat_report.os, "replace", side_effect=failing_replace):
                self._assert_exit_2_clean(
                    [str(image), "--json", str(json_out), "--markdown", str(md_out)],
                    expected_fragment="simulated replace failure",
                    check_subprocess=False,
                )
            self.assertEqual(json_out.read_text(encoding="utf-8"), "prior-json\n")
            self.assertEqual(md_out.read_text(encoding="utf-8"), "prior-md\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json", "report.md"])

            # 3) Direct _write_text staging write failure -> os.replace never called, zero temp dirs.
            with (
                mock.patch.object(compat_report.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated stage write failure")),
            ):
                with self.assertRaises(OSError):
                    compat_report._write_text(json_out, "new-json\n")
            replace_mock.assert_not_called()
            self.assertEqual(json_out.read_text(encoding="utf-8"), "prior-json\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json", "report.md"])

    def test_argparse_missing_required_flags_exit_2_without_traceback(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.exe"
            image.write_bytes(_build_synthetic_pe32_plus(CLEAN_IMPORTS))
            for bad_argv in ([str(image)], [str(image), "--json", str(Path(tmp) / "r.json")]):
                proc = _run_subprocess(*bad_argv)
                self.assertEqual(proc.returncode, 2)
                self.assertEqual(proc.stdout, "")
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("usage:", proc.stderr.lower())
                self.assertIn("--markdown", proc.stderr)


if __name__ == "__main__":
    unittest.main()
