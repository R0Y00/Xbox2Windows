"""CLI, atomic report write, path/collision guard, and mutation suite for tools/mesh_od_pipeline.py."""

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
TOOL = ROOT / "tools" / "mesh_od_pipeline.py"
sys.path.insert(0, str(ROOT / "tools"))

import mesh_od_pipeline as tool  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = tool.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _fake_run_once(_dxc, _hlsl, cso, _flags, _timeout):
    cso.write_bytes(b"DXBC" + b"\x00" * 32)
    return {
        "args": ["dxc", "-Fo", str(cso)],
        "wall_s": 0.01,
        "peak_bytes": 1024 * 1024,
        "exit_code": 0,
        "outcome": "dxil",
        "cso_bytes": 36,
        "first_diagnostic": "",
        "output": "",
    }


def _fake_run_draw(_renderer, _adapter, _mesh, _pixel, data_path, _timeout):
    is_ref = data_path.name == "reference.bin"
    return {
        "outcome": "pass" if is_ref else "mismatch",
        "detail": "" if is_ref else "1024 pixels",
        "exit_code": 0 if is_ref else 1,
        "wall_s": 0.02,
        "peak_mib": 8.0,
        "pipeline_ms": 0.75 if is_ref else None,
        "output": "adapter=mock\nmesh_pipeline_ms=0.75\n",
    }


def _fake_unsupported(*_args, **_kwargs):
    return {
        "outcome": "unsupported",
        "detail": "mesh shader tier",
        "exit_code": 77,
        "wall_s": 0.01,
        "peak_mib": 4.0,
        "pipeline_ms": None,
        "output": "mesh_shader_tier=unsupported\n",
    }


def _make_fake_tools(root: Path) -> tuple[Path, Path]:
    dxc = root / "dxc.exe"
    renderer = root / "offscreen_check.exe"
    dxc.write_bytes(b"DXC_SENTINEL")
    renderer.write_bytes(b"RENDERER_SENTINEL")
    return dxc, renderer


def _base_argv(dxc: Path, renderer: Path, *extra: str) -> list[str]:
    return [
        "--dxc", str(dxc),
        "--renderer", str(renderer),
        "--adapter", "warp",
        "--variants", "control",
        "--compile-timeout", "10",
        "--pipeline-timeout", "10",
        *extra,
    ]


class MeshOdPipelineCliPositiveTests(unittest.TestCase):
    def test_main_writes_nested_json_and_csv_atomically_and_overwrites_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            json_out = root / "nested" / "json" / "report.json"
            csv_out = root / "nested" / "csv" / "report.csv"
            self.assertFalse(json_out.parent.exists())
            self.assertFalse(csv_out.parent.exists())

            seen_replaces: list[tuple[Path, Path]] = []
            real_replace = os.replace

            def tracking_replace(src, dst):
                src_path, dst_path = Path(src), Path(dst)
                seen_replaces.append((src_path, dst_path))
                self.assertTrue(src_path.is_file())
                self.assertEqual(src_path.parent.parent.resolve(), dst_path.parent.resolve())
                return real_replace(src, dst)

            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool, "run_draw", side_effect=_fake_run_draw),
                mock.patch.object(tool, "dxc_version", return_value="dxcompiler 1.8"),
                mock.patch.object(tool.os, "replace", side_effect=tracking_replace),
            ):
                code_json, out_json, err_json = _run_main(
                    _base_argv(dxc, renderer, "--check", "--out", str(json_out))
                )
                code_csv, out_csv, err_csv = _run_main(
                    _base_argv(dxc, renderer, "--check", "--out", str(csv_out))
                )

            self.assertIs(type(code_json), int)
            self.assertEqual(code_json, 0, err_json)
            self.assertEqual(err_json, "")
            self.assertIn(f"wrote {json_out}", out_json)

            self.assertIs(type(code_csv), int)
            self.assertEqual(code_csv, 0, err_csv)
            self.assertEqual(err_csv, "")
            self.assertIn(f"wrote {csv_out}", out_csv)

            self.assertEqual(len(seen_replaces), 2)
            self.assertEqual(seen_replaces[0][1].resolve(), json_out.resolve())
            self.assertEqual(seen_replaces[1][1].resolve(), csv_out.resolve())
            self.assertEqual(sorted(p.name for p in json_out.parent.iterdir()), ["report.json"])
            self.assertEqual(sorted(p.name for p in csv_out.parent.iterdir()), ["report.csv"])

            loaded_json = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertEqual(loaded_json["dxc_version"], "dxcompiler 1.8")
            self.assertEqual(loaded_json["adapter"], "warp")
            self.assertEqual(len(loaded_json["variants"]), 2)

            with csv_out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), list(tool.CSV_FIELDS))
            self.assertEqual(rows[0]["variant"], "control")
            self.assertEqual(rows[0]["draw_outcome"], "pass")

            # Second run atomically overwrites stale contents with zero leftover temp dirs.
            json_out.write_text("STALE_JSON", encoding="utf-8")
            csv_out.write_text("STALE_CSV", encoding="utf-8")
            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool, "run_draw", side_effect=_fake_run_draw),
                mock.patch.object(tool, "dxc_version", return_value="dxcompiler 1.8"),
            ):
                self.assertEqual(_run_main(_base_argv(dxc, renderer, "--out", str(json_out)))[0], 0)
                self.assertEqual(_run_main(_base_argv(dxc, renderer, "--out", str(csv_out)))[0], 0)

            self.assertNotEqual(json_out.read_text(encoding="utf-8"), "STALE_JSON")
            self.assertNotEqual(csv_out.read_text(encoding="utf-8"), "STALE_CSV")
            self.assertEqual(sorted(p.name for p in json_out.parent.iterdir()), ["report.json"])
            self.assertEqual(sorted(p.name for p in csv_out.parent.iterdir()), ["report.csv"])

    def test_work_dir_creation_and_capability_skip_exit_77(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            work_dir = root / "work" / "nested"
            self.assertFalse(work_dir.exists())

            with (
                mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                mock.patch.object(tool, "run_draw", side_effect=_fake_unsupported),
            ):
                code, out, err = _run_main(
                    _base_argv(dxc, renderer, "--check", "--work-dir", str(work_dir))
                )

            self.assertIs(type(code), int)
            self.assertEqual(code, 77, err)
            self.assertEqual(err, "")
            self.assertIn("capability skip", out)
            self.assertTrue(work_dir.is_dir())
            self.assertIn("pixel.hlsl", sorted(p.name for p in work_dir.iterdir()))

    def test_main_forwards_sys_argv_and_script_entrypoint_exits_zero(self) -> None:
        cli_argv = [str(TOOL), "--list-variants", "--variants", "control,chain_1", "--seed", "0x5eed0001"]

        with mock.patch.object(sys, "argv", cli_argv):
            code, out, err = _run_main(None)
        self.assertIs(type(code), int)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("control", lines[0])
        self.assertIn("chain_1", lines[1])

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
        self.assertEqual(stdout.getvalue(), out)

        proc = _run_subprocess(*cli_argv[1:])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stderr, "")
        self.assertEqual(proc.stdout, out)


class MeshOdPipelineCliNegativeAndMutationTests(unittest.TestCase):
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
        if expected_code == 1 and check_subprocess:
            self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr output for {argv!r}")
        for fragment in expected_fragments:
            self.assertIn(fragment, err)

        if check_subprocess:
            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr output for {argv!r}")
            for fragment in expected_fragments:
                self.assertIn(fragment, proc.stderr)

    def test_os_replace_failure_preserves_existing_reports_and_leaves_zero_temp_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)

            for suffix, sentinel in ((".json", '{"preserved": true}\n'), (".csv", "preserved,csv\n")):
                report_path = root / f"dir_{suffix[1:]}" / f"report{suffix}"
                report_path.parent.mkdir()
                report_path.write_text(sentinel, encoding="utf-8")
                replace_calls = []

                def failing_replace(src, dst):
                    src_path = Path(src)
                    replace_calls.append((src_path, Path(dst)))
                    self.assertTrue(src_path.is_file())
                    self.assertEqual(src_path.parent.parent.resolve(), Path(dst).parent.resolve())
                    raise OSError("simulated replace failure")

                with self.subTest(suffix=suffix):
                    with (
                        mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                        mock.patch.object(tool, "run_draw", side_effect=_fake_run_draw),
                        mock.patch.object(tool, "dxc_version", return_value="dxcompiler 1.8"),
                        mock.patch.object(tool.os, "replace", side_effect=failing_replace),
                    ):
                        self._assert_clean_failure(
                            _base_argv(dxc, renderer, "--out", str(report_path)),
                            expected_fragments=("error: failed to write", "simulated replace failure"),
                            check_subprocess=False,
                        )
                    self.assertEqual(len(replace_calls), 1)
                    self.assertEqual(report_path.read_text(encoding="utf-8"), sentinel)
                    self.assertEqual(
                        sorted(p.name for p in report_path.parent.iterdir()),
                        [report_path.name],
                    )

            # Non-existent nested target also leaves zero temp artifacts when os.replace fails.
            fresh_target = root / "fresh" / "nested" / "out.json"
            with mock.patch.object(tool.os, "replace", side_effect=OSError("replace boom")):
                with self.assertRaises(OSError):
                    tool.write_report(fresh_target, {"seed": "0x1"}, [])
            self.assertFalse(fresh_target.exists())
            self.assertEqual(sorted(p.name for p in fresh_target.parent.iterdir()), [])

    def test_out_existing_directory_rejected_upfront_in_main_and_write_report(self) -> None:
        # Kills removal of --out directory guard in _validate_paths() and write_report().
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            out_dir = root / "report_dir"
            out_dir.mkdir()
            sentinel = out_dir / "keep.txt"
            sentinel.write_text("KEEP", encoding="utf-8")

            argv = _base_argv(dxc, renderer, "--out", str(out_dir))
            with mock.patch.object(tool, "measure") as measure_mock:
                self._assert_clean_failure(
                    argv,
                    expected_fragments=("error:", "directory"),
                    check_subprocess=False,
                )
            measure_mock.assert_not_called()

            proc = _run_subprocess(*argv)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertIn("error:", proc.stderr)
            self.assertIn("directory", proc.stderr)
            self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["keep.txt"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "KEEP")

            with self.assertRaises(ValueError):
                tool.write_report(out_dir, {}, [])

    def test_out_collisions_with_dxc_and_renderer_rejected_via_resolve_and_samefile(self) -> None:
        # Kills removal of resolve() and samefile() collision guards against --dxc and --renderer.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            sub = root / "sub"
            sub.mkdir()

            # 1) Non-existent relative alias check directly on _same_path (kills mutating resolve()).
            ghost = root / "ghost_report.json"
            ghost_alias = sub / ".." / "ghost_report.json"
            self.assertTrue(tool._same_path(ghost, ghost_alias))

            # 2) Relative alias collisions with --dxc and --renderer.
            dxc_alias = sub / ".." / "dxc.exe"
            renderer_alias = sub / ".." / "offscreen_check.exe"
            for flag_name, alias in (("--dxc", dxc_alias), ("--renderer", renderer_alias)):
                argv = _base_argv(dxc, renderer, "--out", str(alias))
                with self.subTest(kind="alias", target=flag_name):
                    with mock.patch.object(tool, "measure") as measure_mock:
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=("error:", "collides", flag_name),
                        )
                    measure_mock.assert_not_called()

            # 3) Hardlink aliases where resolve() differs (kills mutating samefile()).
            dxc_hardlink = root / "dxc_hardlink.exe"
            renderer_hardlink = root / "renderer_hardlink.exe"
            os.link(dxc, dxc_hardlink)
            os.link(renderer, renderer_hardlink)
            self.assertNotEqual(dxc.resolve(), dxc_hardlink.resolve())
            self.assertNotEqual(renderer.resolve(), renderer_hardlink.resolve())
            self.assertTrue(dxc.samefile(dxc_hardlink))
            self.assertTrue(renderer.samefile(renderer_hardlink))

            for flag_name, link_path in (("--dxc", dxc_hardlink), ("--renderer", renderer_hardlink)):
                argv = _base_argv(dxc, renderer, "--out", str(link_path))
                with self.subTest(kind="hardlink", target=flag_name):
                    with mock.patch.object(tool, "measure") as measure_mock:
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=("error:", "collides", flag_name),
                        )
                    measure_mock.assert_not_called()

            self.assertEqual(dxc.read_bytes(), b"DXC_SENTINEL")
            self.assertEqual(renderer.read_bytes(), b"RENDERER_SENTINEL")

    def test_work_dir_regular_file_and_out_file_parent_rejected_cleanly(self) -> None:
        # Kills removal of --work-dir existing regular file guard and --out parent file guard.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            regular_file = root / "existing_file.txt"
            regular_file.write_bytes(b"PRESERVE_FILE")

            with mock.patch.object(tool, "measure") as measure_mock:
                self._assert_clean_failure(
                    _base_argv(dxc, renderer, "--work-dir", str(regular_file)),
                    expected_fragments=("error:", "--work-dir", "not a directory"),
                )
            measure_mock.assert_not_called()

            with mock.patch.object(tool, "measure") as measure_mock:
                self._assert_clean_failure(
                    _base_argv(dxc, renderer, "--out", str(regular_file / "report.json")),
                    expected_fragments=("error:", "not a directory"),
                )
            measure_mock.assert_not_called()
            self.assertEqual(regular_file.read_bytes(), b"PRESERVE_FILE")

    def test_dxc_version_and_write_report_errors_return_exit_1_without_traceback(self) -> None:
        # Kills removal of (OSError, ValueError) handlers around dxc_version() and write_report().
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            out_json = root / "out.json"
            argv = _base_argv(dxc, renderer, "--out", str(out_json))

            for exc_obj in (OSError("dxc version io failure"), ValueError("dxc version parse error")):
                with self.subTest(stage="dxc_version", exc=type(exc_obj).__name__):
                    with (
                        mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                        mock.patch.object(tool, "run_draw", side_effect=_fake_run_draw),
                        mock.patch.object(tool, "dxc_version", side_effect=exc_obj),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=("error:", str(exc_obj)),
                            check_subprocess=False,
                        )
                    self.assertFalse(out_json.exists())

            for exc_obj in (OSError("disk full"), ValueError("invalid payload")):
                with self.subTest(stage="write_report", exc=type(exc_obj).__name__):
                    with (
                        mock.patch.object(tool, "run_once", side_effect=_fake_run_once),
                        mock.patch.object(tool, "run_draw", side_effect=_fake_run_draw),
                        mock.patch.object(tool, "dxc_version", return_value="dxcompiler 1.8"),
                        mock.patch.object(tool, "write_report", side_effect=exc_obj),
                    ):
                        self._assert_clean_failure(
                            argv,
                            expected_fragments=("error: failed to write", str(exc_obj)),
                            check_subprocess=False,
                        )
                    self.assertFalse(out_json.exists())

    def test_argparse_errors_py314_compatible_non_zero_exit_and_semantic_stderr(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dxc, renderer = _make_fake_tools(root)
            missing_dxc = root / "missing_dxc.exe"

            cases = (
                ([], "required"),
                (["--unknown-flag"], "unrecognized"),
                (_base_argv(dxc, renderer, "--variants", ""), "selects nothing"),
                (_base_argv(dxc, renderer, "--repeat", "0"), "--repeat"),
                (_base_argv(missing_dxc, renderer), "--dxc not found"),
            )
            for bad_argv, expected_substr in cases:
                with self.subTest(argv=bad_argv):
                    stdout = io.StringIO()
                    stderr = io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as exited:
                            tool.main(bad_argv)
                    self.assertIsInstance(exited.exception.code, int)
                    self.assertNotEqual(exited.exception.code, 0)
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertIn(expected_substr, stderr.getvalue())

                    proc = _run_subprocess(*bad_argv)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertEqual(proc.stdout, "")
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn(expected_substr, proc.stderr)


if __name__ == "__main__":
    unittest.main()
