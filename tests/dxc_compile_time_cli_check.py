"""CLI, atomic JSON/CSV report, path/collision guard, error handling, and mutation suite for tools/dxc_compile_time.py."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dxc_compile_time.py"
sys.path.insert(0, str(ROOT / "tools"))

import dxc_compile_time as tool  # noqa: E402

SAMPLE_FTIME = (
    ";    0.1250 ( 75.0%)   0.0250 (100.0%)   0.1500 ( 80.0%)   0.1510 ( 80.0%)  "
    "Global Value Numbering\n"
    ";    0.1550 (100.0%)   0.0250 (100.0%)   0.1800 (100.0%)   0.1888 (100.0%)  Total\n"
)

EXPECTED_JSON_TOP_KEYS = {
    "recorded_utc",
    "dxc",
    "dxc_version",
    "target",
    "entry",
    "flags",
    "repeat",
    "timeout_s",
    "ftime_report",
    "variants",
}

EXPECTED_VARIANT_KEYS = {
    "variant",
    "family",
    "size",
    "hlsl_bytes",
    "dxil_bytes",
    "reps",
    "wall_median_s",
    "wall_min_s",
    "wall_max_s",
    "peak_median_mib",
    "peak_max_mib",
    "outcome",
    "exit_codes",
    "first_diagnostic",
    "pass_total_s",
    "passes",
    "command",
    "runs",
}


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


def _fake_run_once(dxc, hlsl, cso, flags, timeout_s, sample_interval=0.005):
    name = hlsl.stem
    cso.write_bytes(b"DXBC" + b"\x00" * 12)
    return {
        "args": [str(dxc), *flags, "-Fo", str(cso), str(hlsl)],
        "wall_s": 0.25 if name == "control" else 0.5,
        "peak_bytes": 16 * 1024 * 1024,
        "exit_code": 0,
        "outcome": "dxil",
        "cso_bytes": 16,
        "first_diagnostic": "",
        "output": SAMPLE_FTIME if "-ftime-report" in flags else "",
    }


def _fake_subprocess_run(args, **kwargs):
    return subprocess.CompletedProcess(args, 0, stdout="dxcompiler.dll: 1.8.2407\n", stderr="")


class DxcCompileTimeCliPositiveTests(unittest.TestCase):
    def test_list_variants_and_entrypoint(self):
        code, out, err = _run_main(["--list-variants"])
        self.assertIs(type(code), int)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertIn("variant", out)
        self.assertIn("control", out)
        self.assertIn("condstore_1024", out)
        self.assertIn("stores_16384", out)

        proc = _run_subprocess("--list-variants")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, out)

        cli_argv = [str(TOOL), "--list-variants"]
        with mock.patch.object(sys, "argv", cli_argv):
            code2, out2, err2 = _run_main(None)
        self.assertIs(type(code2), int)
        self.assertEqual(code2, 0, err2)
        self.assertEqual(out2, out)

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
        self.assertEqual(stdout.getvalue(), out)

    def test_atomic_json_and_csv_reports_with_mocked_dxc(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dxc = root / "dxc.exe"
            fake_dxc.write_bytes(b"MZ")
            work_dir = root / "work"
            json_out = root / "nested" / "json" / "report.json"
            csv_out = root / "nested" / "csv" / "report.csv"

            seen_replaces = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                self.assertEqual(src_path.name, dst_path.name)
                return real_replace(src, dst)

            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                mock.patch.object(tool.os, "replace", side_effect=tracking_replace),
            ):
                code, out, err = _run_main([
                    "--dxc", str(fake_dxc),
                    "--variants", "control,condstore_64",
                    "--timeout", "90",
                    "--ftime-report",
                    "--work-dir", str(work_dir),
                    "--out", str(json_out),
                ])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn(f"wrote {json_out}", out)
            self.assertEqual(len(seen_replaces), 1)
            self.assertEqual(sorted(p.name for p in json_out.parent.iterdir()), ["report.json"])
            self.assertEqual(
                sorted(p.name for p in work_dir.iterdir()),
                ["condstore_64.cso", "condstore_64.hlsl", "control.cso", "control.hlsl"],
            )

            doc = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(set(doc), EXPECTED_JSON_TOP_KEYS)
            self.assertEqual(doc["dxc"], str(fake_dxc))
            self.assertEqual(doc["dxc_version"], "dxcompiler.dll: 1.8.2407")
            self.assertEqual(doc["target"], tool.DEFAULT_TARGET)
            self.assertEqual(doc["entry"], tool.DEFAULT_ENTRY)
            self.assertEqual(doc["repeat"], 1)
            self.assertEqual(doc["timeout_s"], 90.0)
            self.assertTrue(doc["ftime_report"])
            self.assertEqual([v["variant"] for v in doc["variants"]], ["control", "condstore_64"])
            for variant_rec in doc["variants"]:
                self.assertEqual(set(variant_rec), EXPECTED_VARIANT_KEYS)
                self.assertEqual(variant_rec["outcome"], "dxil")
                self.assertEqual(variant_rec["pass_total_s"], 0.18)
                self.assertEqual(variant_rec["passes"][0]["name"], "Global Value Numbering")

            # Overwrite pre-existing JSON and write CSV report via atomic staging.
            json_out.write_text("stale-json\n", encoding="utf-8")
            seen_replaces.clear()
            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                mock.patch.object(tool.os, "replace", side_effect=tracking_replace),
            ):
                code_j2, _, err_j2 = _run_main([
                    "--dxc", str(fake_dxc),
                    "--variants", "control",
                    "--timeout", "90",
                    "--out", str(json_out),
                ])
                code_c, out_c, err_c = _run_main([
                    "--dxc", str(fake_dxc),
                    "--variants", "control,condstore_64",
                    "--timeout", "90",
                    "--ftime-report",
                    "--out", str(csv_out),
                ])

            self.assertEqual(code_j2, 0, err_j2)
            self.assertEqual(code_c, 0, err_c)
            self.assertIn(f"wrote {csv_out}", out_c)
            self.assertEqual(len(seen_replaces), 2)
            self.assertEqual(sorted(p.name for p in json_out.parent.iterdir()), ["report.json"])
            self.assertEqual(sorted(p.name for p in csv_out.parent.iterdir()), ["report.csv"])
            self.assertEqual(
                [v["variant"] for v in json.loads(json_out.read_text(encoding="utf-8"))["variants"]],
                ["control"],
            )

            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(tool.CSV_FIELDS))
                rows = list(reader)
            self.assertEqual([row["variant"] for row in rows], ["control", "condstore_64"])
            self.assertEqual(rows[0]["outcome"], "dxil")
            self.assertIn("Global Value Numbering=0.150s", rows[0]["top_passes"])

            # No --out writes nothing and still returns 0.
            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
            ):
                code_no_out, out_no_out, err_no_out = _run_main([
                    "--dxc", str(fake_dxc),
                    "--variants", "control",
                    "--timeout", "90",
                ])
            self.assertEqual(code_no_out, 0, err_no_out)
            self.assertIn("No --out given; nothing written", out_no_out)

    def test_dxc_version_handles_subprocess_outcomes(self):
        fake_dxc = Path("fake-dxc.exe")
        with mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run):
            self.assertEqual(tool.dxc_version(fake_dxc), "dxcompiler.dll: 1.8.2407")
        for exc in (OSError("missing"), subprocess.TimeoutExpired(["dxc", "--version"], 30)):
            with mock.patch.object(tool.subprocess, "run", side_effect=exc):
                self.assertEqual(tool.dxc_version(fake_dxc), "")


class DxcCompileTimeCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(self, argv, *, expected_code=1, expected_fragment, check_subprocess=True):
        code, out, err = _run_main(argv)
        self.assertEqual(code, expected_code, err)
        self.assertNotIn("Traceback", err)
        self.assertIn(expected_fragment, err)
        if expected_code == 1:
            self.assertIn("dxc_compile_time:", err)
        else:
            self.assertEqual(out, "")

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn(expected_fragment, proc.stderr)
            if expected_code == 1:
                self.assertIn("dxc_compile_time:", proc.stderr)
            else:
                self.assertEqual(proc.stdout, "")

    def test_argparse_and_cli_parameter_errors_exit_nonzero_cleanly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dxc = root / "dxc.exe"
            fake_dxc.write_bytes(b"MZ")
            a_dir = root / "a_dir"
            a_dir.mkdir()
            missing_dxc = root / "missing_dxc.exe"

            cases = [
                (["--dxc", str(fake_dxc)], "--timeout"),
                (["--dxc", str(fake_dxc), "--timeout", "0"], "positive"),
                (["--dxc", str(fake_dxc), "--timeout", "10", "--repeat", "0"], "--repeat"),
                (["--timeout", "10"], "--dxc"),
                (["--dxc", str(missing_dxc), "--timeout", "10"], "DXC executable not found"),
                (["--dxc", str(a_dir), "--timeout", "10"], "DXC executable not found"),
                (["--dxc", str(fake_dxc), "--timeout", "10", "--variants", " , "], "selects nothing"),
                (["--dxc", str(fake_dxc), "--timeout", "10", "--variants", "bogus_1"], "unknown variant"),
                (["--dxc", str(fake_dxc), "--timeout", "10", "--variants", "condstore_65537"], "exceeds"),
                (["--no-such-flag"], "unrecognized"),
            ]
            for argv, frag in cases:
                with self.subTest(argv=argv):
                    with self.assertRaises(SystemExit) as exited:
                        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            tool.main(argv)
                    self.assertNotEqual(exited.exception.code, 0)
                    self._assert_clean_failure(
                        argv,
                        expected_code=2,
                        expected_fragment=frag,
                    )

    def test_upfront_out_and_work_dir_guards_kill_mutations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            fake_dxc = root / "dxc.exe"
            dxc_bytes = b"MZ-DXC-SENTINEL"
            fake_dxc.write_bytes(dxc_bytes)

            existing_dir = root / "existing_dir"
            existing_dir.mkdir()
            work_file = root / "work_file.txt"
            work_file.write_text("keep-work-file\n", encoding="utf-8")

            dxc_alias = root / "sub" / ".." / "dxc.exe"
            dxc_link = sub / "dxc_link.json"
            os.link(fake_dxc, dxc_link)
            self.assertNotEqual(fake_dxc.resolve(), dxc_link.resolve())
            self.assertTrue(fake_dxc.samefile(dxc_link))

            # 1) --out is an existing directory (including "").
            for bad_out in (str(existing_dir), ""):
                with mock.patch.object(tool, "measure_variant") as mv_mock:
                    self._assert_clean_failure(
                        ["--dxc", str(fake_dxc), "--timeout", "10", "--out", bad_out],
                        expected_fragment="existing directory",
                        check_subprocess=bool(bad_out),
                    )
                mv_mock.assert_not_called()

            # 2) --out collides with --dxc via resolve() and hardlink samefile().
            for bad_out in (dxc_alias, dxc_link):
                with mock.patch.object(tool, "measure_variant") as mv_mock:
                    self._assert_clean_failure(
                        ["--dxc", str(fake_dxc), "--timeout", "10", "--out", str(bad_out)],
                        expected_fragment="collides with --dxc",
                    )
                mv_mock.assert_not_called()
            self.assertEqual(fake_dxc.read_bytes(), dxc_bytes)

            # 3) --out collides with --work-dir via resolve().
            ghost_work = root / "new_work"
            with mock.patch.object(tool, "measure_variant") as mv_mock:
                self._assert_clean_failure(
                    [
                        "--dxc", str(fake_dxc), "--timeout", "10",
                        "--work-dir", str(ghost_work),
                        "--out", str(root / "sub" / ".." / "new_work"),
                    ],
                    expected_fragment="collides with --work-dir",
                )
            mv_mock.assert_not_called()
            self.assertFalse(ghost_work.exists())

            # 4) --work-dir is an existing regular file.
            with mock.patch.object(tool, "measure_variant") as mv_mock:
                self._assert_clean_failure(
                    [
                        "--dxc", str(fake_dxc), "--timeout", "10",
                        "--work-dir", str(work_file),
                    ],
                    expected_fragment="existing regular file",
                )
            mv_mock.assert_not_called()
            self.assertEqual(work_file.read_text(encoding="utf-8"), "keep-work-file\n")

    def test_runtime_os_value_and_subprocess_errors_exit_1_cleanly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dxc = root / "dxc.exe"
            fake_dxc.write_bytes(b"NOT_A_REAL_PE_EXECUTABLE")
            out_file = root / "reports" / "out.json"
            out_file.parent.mkdir()
            out_file.write_text("prior-report\n", encoding="utf-8")

            # Simulated Popen launch OSError through real run_once -> measure_variant -> main exits 1 cleanly.
            with mock.patch.object(tool.subprocess, "Popen", side_effect=OSError("simulated exec format error")):
                self._assert_clean_failure(
                    [
                        "--dxc", str(fake_dxc),
                        "--variants", "control",
                        "--timeout", "10",
                        "--out", str(out_file),
                    ],
                    expected_fragment="simulated exec format error",
                    check_subprocess=False,
                )
            self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")

            # Bounded real subprocess control with a deterministic failing child (sys.executable rejects -T).
            real_out = root / "reports" / "real_child.json"
            proc_real = _run_subprocess(
                "--dxc", sys.executable,
                "--variants", "control",
                "--timeout", "5",
                "--work-dir", str(root / "real_work"),
                "--out", str(real_out),
            )
            self.assertEqual(proc_real.returncode, 0, proc_real.stderr)
            self.assertEqual(proc_real.stderr, "")
            real_doc = json.loads(real_out.read_text(encoding="utf-8"))
            self.assertEqual(real_doc["variants"][0]["outcome"], "diagnostic")
            self.assertEqual(real_doc["variants"][0]["exit_codes"], [2])

            # work_dir.mkdir OSError (ancestor is a regular file) exits 1 cleanly.
            blocker = root / "blocker_file"
            blocker.write_bytes(b"x")
            with mock.patch.object(tool, "measure_variant") as mv_mock:
                self._assert_clean_failure(
                    [
                        "--dxc", str(fake_dxc),
                        "--variants", "control",
                        "--timeout", "10",
                        "--work-dir", str(blocker / "nested"),
                    ],
                    expected_fragment="dxc_compile_time:",
                )
            mv_mock.assert_not_called()

            # measure_variant / dxc_version / write_report failures across OSError, ValueError, SubprocessError.
            for exc in (
                OSError("simulated io error"),
                ValueError("simulated value error"),
                subprocess.TimeoutExpired(["dxc"], 1),
            ):
                with self.subTest(stage="measure_variant", exc=type(exc).__name__):
                    with (
                        mock.patch.object(tool, "measure_variant", side_effect=exc),
                        mock.patch.object(tool, "write_report") as write_mock,
                    ):
                        self._assert_clean_failure(
                            ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(out_file)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )
                    write_mock.assert_not_called()
                    self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")

                with self.subTest(stage="dxc_version", exc=type(exc).__name__):
                    with (
                        mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                        mock.patch.object(tool, "dxc_version", side_effect=exc),
                        mock.patch.object(tool, "write_report") as write_mock,
                    ):
                        self._assert_clean_failure(
                            ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(out_file)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )
                    write_mock.assert_not_called()
                    self.assertEqual(out_file.read_text(encoding="utf-8"), "prior-report\n")

                with self.subTest(stage="write_report", exc=type(exc).__name__):
                    with (
                        mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                        mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                        mock.patch.object(tool, "write_report", side_effect=exc),
                    ):
                        self._assert_clean_failure(
                            ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(out_file)],
                            expected_fragment=str(exc),
                            check_subprocess=False,
                        )

    def test_atomic_write_report_failures_preserve_prior_and_leave_zero_temp_dirs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_dxc = root / "dxc.exe"
            fake_dxc.write_bytes(b"MZ")
            out_dir = root / "out"

            def failing_replace(src, dst):
                src_path = Path(src)
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                raise OSError("simulated replace failure")

            for ext, prior_text in ((".json", "prior-json\n"), (".csv", "prior-csv\n")):
                out_path = out_dir / f"report{ext}"

                # 1) Fresh target + os.replace OSError -> exit 1, zero temp artifacts.
                with (
                    mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                    mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                    mock.patch.object(tool.os, "replace", side_effect=failing_replace),
                ):
                    self._assert_clean_failure(
                        ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(out_path)],
                        expected_fragment="simulated replace failure",
                        check_subprocess=False,
                    )
                self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [])

                # 2) Pre-existing target + os.replace OSError -> prior file intact, zero temp artifacts.
                out_path.write_text(prior_text, encoding="utf-8")
                with (
                    mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                    mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                    mock.patch.object(tool.os, "replace", side_effect=failing_replace),
                ):
                    self._assert_clean_failure(
                        ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(out_path)],
                        expected_fragment="simulated replace failure",
                        check_subprocess=False,
                    )
                self.assertEqual(out_path.read_text(encoding="utf-8"), prior_text)
                self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [out_path.name])
                out_path.unlink()

            # 3) Staging write failures preserve pre-existing JSON and CSV and leave zero temp artifacts.
            json_out = out_dir / "report.json"
            json_out.write_text("keep-json\n", encoding="utf-8")
            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                mock.patch.object(tool.os, "replace") as replace_mock,
                mock.patch.object(Path, "write_text", side_effect=OSError("simulated stage write failure")),
            ):
                self._assert_clean_failure(
                    ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(json_out)],
                    expected_fragment="simulated stage write failure",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(json_out.read_text(encoding="utf-8"), "keep-json\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.json"])

            csv_out = out_dir / "report.csv"
            csv_out.write_text("keep-csv\n", encoding="utf-8")
            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool.subprocess, "run", side_effect=_fake_subprocess_run),
                mock.patch.object(tool.os, "replace") as replace_mock,
                mock.patch.object(Path, "open", side_effect=OSError("simulated csv stage failure")),
            ):
                self._assert_clean_failure(
                    ["--dxc", str(fake_dxc), "--variants", "control", "--timeout", "10", "--out", str(csv_out)],
                    expected_fragment="simulated csv stage failure",
                    check_subprocess=False,
                )
            replace_mock.assert_not_called()
            self.assertEqual(csv_out.read_text(encoding="utf-8"), "keep-csv\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.csv", "report.json"])

            # 4) Serialization failures (json.dumps ValueError, csv_row KeyError) preserve prior files.
            with mock.patch.object(tool.os, "replace") as replace_mock:
                circular = {}
                circular["self"] = circular
                with self.assertRaises(ValueError):
                    tool.write_report(json_out, circular, [])
                with self.assertRaises(KeyError):
                    tool.write_report(csv_out, {}, [{"variant": "broken"}])
            replace_mock.assert_not_called()
            self.assertEqual(json_out.read_text(encoding="utf-8"), "keep-json\n")
            self.assertEqual(csv_out.read_text(encoding="utf-8"), "keep-csv\n")
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["report.csv", "report.json"])


if __name__ == "__main__":
    unittest.main()
