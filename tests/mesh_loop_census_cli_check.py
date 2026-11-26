"""Pure-Python CLI, path-validation, error-guard, and mutation suite for tools/mesh_loop_census.py."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "mesh_loop_census.py"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import mesh_loop_census  # noqa: E402
from mesh_two_exits_check import mesh_container  # noqa: E402

LIMITS = (4, 2)
LOOP_FREE_PROGRAM = [
    (0, "s_mov_b32 s10, 0"),
    (4, "s_endpgm"),
]
COUNTED_LOOP_PROGRAM = [
    (0, "s_mov_b32 s10, 0"),
    (4, "s_add_u32 s10, s10, 1"),
    (8, "s_cmp_lt_u32 s10, 4"),
    (12, "s_cbranch_scc1 65533"),
    (16, "s_endpgm"),
]


def _synthetic_dxbc(tag: bytes) -> bytes:
    return b"DXBC" + tag.ljust(28, b"\x00")


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = mesh_loop_census.main(argv)
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


class MeshLoopCensusCliPositiveTests(unittest.TestCase):
    def test_main_classifies_loop_free_counted_duplicate_non_dxbc_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()

            loop_free_blob = _synthetic_dxbc(b"loop-free")
            counted_blob = _synthetic_dxbc(b"counted-a")
            duplicate_blob = _synthetic_dxbc(b"counted-b")
            bad_blob = _synthetic_dxbc(b"bad-stage")

            (root / "a_loop_free.dxbc").write_bytes(loop_free_blob)
            (nested / "b_counted.dxbc").write_bytes(counted_blob)
            (nested / "c_duplicate.dxbc").write_bytes(duplicate_blob)
            (root / "d_bad.dxbc").write_bytes(bad_blob)
            (root / "e_notes.txt").write_bytes(b"not a dxbc container")

            seen_calls: list[tuple[bytes, object]] = []

            def fake_decode_mesh(blob: bytes, library: object):
                seen_calls.append((blob, library))
                if blob == loop_free_blob:
                    return list(LOOP_FREE_PROGRAM), LIMITS
                if blob in (counted_blob, duplicate_blob):
                    return list(COUNTED_LOOP_PROGRAM), LIMITS
                if blob == bad_blob:
                    raise ValueError("Unsupported shader stage 5")
                raise AssertionError(f"unexpected blob: {blob!r}")

            with mock.patch.object(mesh_loop_census.mesh_entry_aot, "decode_mesh", side_effect=fake_decode_mesh):
                code, out, err = _run_main([str(root), "--llvm", "mock-llvm.dll"])

            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(seen_calls), 4)
            self.assertTrue(all(lib == "mock-llvm.dll" for _, lib in seen_calls))
            self.assertEqual(
                sorted(p.name for p in root.iterdir()),
                ["a_loop_free.dxbc", "d_bad.dxbc", "e_notes.txt", "nested"],
            )
            self.assertEqual(
                sorted(p.name for p in nested.iterdir()),
                ["b_counted.dxbc", "c_duplicate.dxbc"],
            )

            expected_fragments = (
                "| Population | Programs |",
                "| containers not decoded: Unsupported shader stage N | 1 |",
                "| duplicate mesh programs | 1 |",
                "| files without a DXBC container | 1 |",
                "| loop-free | 1 |",
                "| with loops | 1 |",
                "| back-edges | 1 |",
                "| outermost back-edges | 1 |",
                "| programs with 1 loop | 1 |",
                "| s_cbranch_scc1 | 1 |",
                "| no | no | no | no | yes | constant pattern | 1 | 3 / 3 / 3 |",
                "| First refusal (entry lowerer, as decoded) | With loops | Loop-free |",
                "| at a latch | 1 | 0 |",
                "| Expected a forward mesh branch | 1 | 0 |",
            )
            for fragment in expected_fragments:
                self.assertIn(fragment, out, out)
            for leak in ("a_loop_free", "b_counted", "c_duplicate", "d_bad", "e_notes", "s10"):
                self.assertNotIn(leak, out, leak)

    def test_main_llvm_mc_backend_wires_disassembler_and_forwards_sys_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = _synthetic_dxbc(b"mc-counted")
            (root / "mesh.dxbc").write_bytes(blob)

            saved_pixel_disasm = mesh_loop_census.pixel_aot.disassemble
            saved_mesh_disasm = mesh_loop_census.mesh_entry_aot.disassemble
            seen_mc_cmds: list[list[str]] = []

            def fake_decode_mesh(container: bytes, library: object):
                self.assertEqual(container, blob)
                self.assertEqual(library, "mock-llvm-mc")
                decoded = mesh_loop_census.mesh_entry_aot.disassemble(
                    b"\x00" * 20,
                    library,
                    offsets=True,
                )
                return decoded, LIMITS

            def fake_subprocess_run(cmd, *, input, capture_output, text, check):
                seen_mc_cmds.append(list(cmd))
                stdout = "\n".join(
                    [
                        "s_mov_b32 s10, 0 ; encoding: [0x80,0x03,0x8a,0xbe]",
                        "s_add_u32 s10, s10, 1 ; encoding: [0x0a,0x81,0x0a,0x80]",
                        "s_cmp_lt_u32 s10, 4 ; encoding: [0x0a,0x84,0x0a,0xbf]",
                        "s_cbranch_scc1 65533 ; encoding: [0xfd,0xff,0x85,0xbf]",
                        "s_endpgm ; encoding: [0x00,0x00,0x81,0xbf]",
                    ]
                )
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

            try:
                with (
                    mock.patch.object(mesh_loop_census.mesh_entry_aot, "decode_mesh", side_effect=fake_decode_mesh),
                    mock.patch.object(mesh_loop_census.subprocess, "run", side_effect=fake_subprocess_run),
                ):
                    code, out, err = _run_main([str(root), "--llvm-mc", "mock-llvm-mc"])
            finally:
                mesh_loop_census.pixel_aot.disassemble = saved_pixel_disasm
                mesh_loop_census.mesh_entry_aot.disassemble = saved_mesh_disasm

            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual(len(seen_mc_cmds), 1)
            self.assertEqual(seen_mc_cmds[0][0], "mock-llvm-mc")
            self.assertIn("--disassemble", seen_mc_cmds[0])
            self.assertIn("| with loops | 1 |", out)
            self.assertIn("| s_cbranch_scc1 | 1 |", out)

        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp)
            cli_argv = [str(TOOL), str(empty_dir), "--llvm", "stub"]
            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("| Population | Programs |", out)

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
            self.assertIn("| Population | Programs |", stdout.getvalue())

            proc = _run_subprocess(str(empty_dir), "--llvm", "stub")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertIn("| Population | Programs |", proc.stdout)


class MeshLoopCensusCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int = 1,
        expected_fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> None:
        code, out, err = _run_main(argv)
        self.assertIs(type(code), int)
        self.assertEqual(code, expected_code, err)
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

    def test_missing_directory_and_regular_file_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing_dir"
            regular_file = root / "shader.dxbc"
            regular_file.write_bytes(_synthetic_dxbc(b"single"))

            for target, fragment in (
                (missing, "does not exist"),
                (regular_file, "not a directory"),
            ):
                with self.subTest(target=target.name):
                    with self.assertRaises(ValueError) as direct_err:
                        mesh_loop_census.read_programs(target, Counter(), "stub")
                    self.assertIn(fragment, str(direct_err.exception))

                    with mock.patch.object(mesh_loop_census, "use_llvm_mc") as mc_mock:
                        self._assert_clean_failure(
                            [str(target), "--llvm-mc", "mock-llvm-mc"],
                            expected_fragments=("mesh_loop_census:", fragment),
                            check_subprocess=False,
                        )
                    mc_mock.assert_not_called()

                    self._assert_clean_failure(
                        [str(target), "--llvm", "stub"],
                        expected_fragments=("mesh_loop_census:", fragment),
                    )

    def test_llvm_and_llvm_mc_directory_and_empty_paths_rejected_upfront(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "inputs"
            input_dir.mkdir()
            backend_dir = root / "backend_dir"
            backend_dir.mkdir()

            for flag in ("--llvm", "--llvm-mc"):
                for bad_value, fragment in (
                    (str(backend_dir), "existing directory"),
                    ("", "non-empty"),
                    ("   ", "non-empty"),
                ):
                    with self.subTest(flag=flag, bad_value=bad_value):
                        with (
                            mock.patch.object(mesh_loop_census, "use_llvm_mc") as mc_mock,
                            mock.patch.object(mesh_loop_census, "read_programs") as read_mock,
                        ):
                            self._assert_clean_failure(
                                [str(input_dir), flag, bad_value],
                                expected_fragments=("mesh_loop_census:", fragment),
                            )
                        mc_mock.assert_not_called()
                        read_mock.assert_not_called()

    def test_oserror_valueerror_struct_error_guards_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_mesh_blob = mesh_container(bytes.fromhex("80038abe000081bf"))
            (root / "sample.dxbc").write_bytes(valid_mesh_blob)

            # Subprocess OSError via nonexistent --llvm-mc executable and nonexistent --llvm library (no mocking).
            missing_mc = root / "nonexistent_llvm_mc"
            saved_pixel_disasm = mesh_loop_census.pixel_aot.disassemble
            saved_mesh_disasm = mesh_loop_census.mesh_entry_aot.disassemble
            try:
                self._assert_clean_failure(
                    [str(root), "--llvm-mc", str(missing_mc)],
                    expected_fragments=("mesh_loop_census:",),
                )
            finally:
                mesh_loop_census.pixel_aot.disassemble = saved_pixel_disasm
                mesh_loop_census.mesh_entry_aot.disassemble = saved_mesh_disasm
            missing_llvm = root / "nonexistent_llvm.dll"
            self._assert_clean_failure(
                [str(root), "--llvm", str(missing_llvm)],
                expected_fragments=("mesh_loop_census:",),
            )

            # In-process OSError from decode_mesh propagates cleanly to parser.exit(1, ...).
            with mock.patch.object(
                mesh_loop_census.mesh_entry_aot,
                "decode_mesh",
                side_effect=OSError("cannot open library"),
            ):
                self._assert_clean_failure(
                    [str(root), "--llvm", "stub"],
                    expected_fragments=("mesh_loop_census:", "cannot open library"),
                    check_subprocess=False,
                )

            # In-process ValueError and struct.error from census/read_programs caught cleanly.
            for exc_obj, fragment in (
                (ValueError("synthetic census failure"), "synthetic census failure"),
                (struct.error("unpack requires a buffer of 4 bytes"), "unpack requires a buffer of 4 bytes"),
            ):
                with self.subTest(exc=type(exc_obj).__name__):
                    with mock.patch.object(mesh_loop_census, "census", side_effect=exc_obj):
                        self._assert_clean_failure(
                            [str(root), "--llvm", "stub"],
                            expected_fragments=("mesh_loop_census:", fragment),
                            check_subprocess=False,
                        )

            # Negative control: per-container ValueError / KeyError / struct.error in decode_mesh
            # stays tallied in Population and exits 0.
            for exc_obj, expected_row in (
                (KeyError("DXIL"), "| containers not decoded: 'DXIL' | 1 |"),
                (struct.error("unpack_from requires 8 bytes"), "| containers not decoded: unpack_from requires N bytes | 1 |"),
            ):
                with self.subTest(container_exc=type(exc_obj).__name__):
                    with mock.patch.object(mesh_loop_census.mesh_entry_aot, "decode_mesh", side_effect=exc_obj):
                        code, out, err = _run_main([str(root), "--llvm", "stub"])
                    self.assertEqual(code, 0, err)
                    self.assertEqual(err, "")
                    self.assertIn(expected_row, out)

    def test_argparse_usage_errors_compatible_with_py314(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bad_argv, expected_fragment in (
                ([], "required"),
                ([str(root)], "required"),
                ([str(root), "--llvm", "a", "--llvm-mc", "b"], "not allowed"),
                ([str(root), "--llvm", "a", "--unknown-flag"], "unrecognized"),
            ):
                with self.subTest(argv=bad_argv):
                    self._assert_clean_failure(
                        bad_argv,
                        expected_code=2,
                        expected_fragments=(expected_fragment,),
                    )


if __name__ == "__main__":
    unittest.main()
