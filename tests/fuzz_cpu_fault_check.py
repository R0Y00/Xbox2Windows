#!/usr/bin/env python3
"""Unit and integration checks for the fault_reads fuzz area and bad-pointer equivalence."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "fuzz_cpu.py"
sys.path.insert(0, str(TOOLS_DIR))

import fuzz_cpu  # noqa: E402


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = fuzz_cpu.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


class FaultReadsGeneratorTests(unittest.TestCase):
    def test_area_and_mutant_registered(self) -> None:
        self.assertIn("fault_reads", fuzz_cpu.AREAS)
        self.assertEqual(fuzz_cpu.AREAS["fault_reads"][1], "cpu_fuzz_fault_reads")
        self.assertIs(fuzz_cpu.AREAS["fault_reads"][0], fuzz_cpu.generate_fault_reads)
        self.assertIn("fault_reads", fuzz_cpu.MUTANTS)
        self.assertEqual(fuzz_cpu.MUTANTS["fault_reads"][0], "drop_dead_read")
        self.assertIs(fuzz_cpu.MUTANTS["fault_reads"][1], fuzz_cpu.plant_drop_dead_read)

    def test_generator_deterministic_and_covers_widths_and_address_forms(self) -> None:
        a = fuzz_cpu.generate_fault_reads(0xFA17, 24)
        b = fuzz_cpu.generate_fault_reads(0xFA17, 24)
        c = fuzz_cpu.generate_fault_reads(0xFA18, 24)
        self.assertEqual([fn.asm() for fn in a], [fn.asm() for fn in b])
        self.assertNotEqual([fn.asm() for fn in a], [fn.asm() for fn in c])

        joined = "\n".join(fn.asm() for fn in a)
        for ptr_tok in ("BYTE PTR", "WORD PTR", "DWORD PTR", "QWORD PTR"):
            self.assertIn(ptr_tok, joined)
        self.assertIn("[rcx]", joined)
        self.assertIn("[rdx]", joined)
        self.assertRegex(joined, r"\[rcx\+\d+\]")
        for ctrl in (
            "fuzz_fault_control_live_b",
            "fuzz_fault_control_live_w",
            "fuzz_fault_control_live_d",
            "fuzz_fault_control_live_q",
            "fuzz_fault_control_skip_zero",
        ):
            self.assertTrue(any(fn.name == ctrl for fn in a), f"missing control {ctrl}")

    def test_emit_fault_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asm = Path(tmp) / "faults.asm"
            defn = Path(tmp) / "faults.def"
            code, out, err = _run_main(["emit", "0xFA17", "12", str(asm), str(defn), "--area", "fault_reads"])
            self.assertEqual(code, 0, err)
            self.assertEqual(out, "")
            self.assertEqual(err, "")
            self.assertTrue(defn.read_text(encoding="utf-8").startswith("LIBRARY cpu_fuzz_fault_reads\nEXPORTS\n"))
            self.assertIn("fuzz_fault_0 PROC FRAME", asm.read_text(encoding="utf-8"))

    def test_plant_drop_dead_read_mutates_and_guards_missing_needle(self) -> None:
        class FakeMod:
            @staticmethod
            def translate(_img, _names):
                return (
                    "template <class T> static T load_always(word address) {\n"
                    "    return *reinterpret_cast<const volatile T __unaligned*>(address);\n"
                    "}\n",
                    [],
                )

        fuzz_cpu.plant_drop_dead_read(FakeMod, Path("dummy.dll"), [])
        mutated, _ = FakeMod.translate(Path("dummy.dll"), [])
        self.assertNotIn("const volatile T __unaligned*", mutated)
        self.assertIn("const T __unaligned*", mutated)

        class EmptyMod:
            @staticmethod
            def translate(_img, _names):
                return "// nothing here\n", []

        fuzz_cpu.plant_drop_dead_read(EmptyMod, Path("dummy.dll"), [])
        with self.assertRaises(SystemExit):
            EmptyMod.translate(Path("dummy.dll"), [])


class FaultReadsCheckModeTests(unittest.TestCase):
    def _make_fake_dlls(self, tmp: Path) -> tuple[Path, Path, Path, Path]:
        paths = [tmp / name for name in ("native.dll", "lifted.dll", "o2.dll", "mutant.dll")]
        for path in paths:
            path.write_bytes(b"MZ-placeholder")
        return tuple(paths)  # type: ignore[return-value]

    def _baseline_results(self, fn_names: list[str], probe_name: str) -> dict[str, dict]:
        valid_buf = (b"\x01" + b"\x00" * 7 + b"\x02" + b"\x00" * 7 + b"\x5a" * 272).hex()
        fault_buf = (b"\x01" + b"\x00" * 7 + b"\x5a" * 280).hex()
        out = {}
        for name in fn_names:
            if probe_name in ("valid", "valid_alt"):
                out[name] = {"name": name, "status": 0, "return": 0x42, "buffer": valid_buf}
            elif probe_name == "null_both" and name == "fuzz_fault_control_skip_zero":
                out[name] = {"name": name, "status": 0, "return": 66, "buffer": valid_buf}
            elif probe_name in ("noaccess_rcx", "noaccess_rdx"):
                out[name] = {"name": name, "status": 0xC0000005, "return": None, "buffer": fault_buf}
            else:
                out[name] = {"name": name, "status": 0xC0000005, "return": None, "buffer": fault_buf}
        return out

    def test_check_mode_equivalence_and_mutation_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            native, lifted, o2, mutant = self._make_fake_dlls(Path(tmp))

            def fake_runner(dll_path: Path, _prefix: str, probe_name: str, fn_names: list[str]) -> dict[str, dict]:
                res = self._baseline_results(fn_names, probe_name)
                if dll_path == mutant and probe_name == "noaccess_both":
                    valid_buf = (b"\x01" + b"\x00" * 7 + b"\x02" + b"\x00" * 7 + b"\x5a" * 272).hex()
                    res["fuzz_fault_0"] = {"name": "fuzz_fault_0", "status": 0, "return": 0x42, "buffer": valid_buf}
                return res

            with mock.patch.object(fuzz_cpu, "_run_fault_probe_subprocess", side_effect=fake_runner):
                code, out, err = _run_main([
                    "check", "0xFA17", "2", str(native), str(lifted), str(o2),
                    "--area", "fault_reads", "--mutant", str(mutant),
                ])
            self.assertEqual(code, 0, err)
            self.assertIn("default and /O2 matched", out)
            self.assertIn("drop_dead_read", out)

            # Guard 1: /O2 drops fault (even with identical buffer) -> check MUST fail.
            def o2_drops_fault(dll_path: Path, _prefix: str, probe_name: str, fn_names: list[str]) -> dict[str, dict]:
                res = self._baseline_results(fn_names, probe_name)
                if dll_path == o2 and probe_name == "low_unmapped":
                    res["fuzz_fault_0"] = {
                        "name": "fuzz_fault_0",
                        "status": 0,
                        "return": 0x42,
                        "buffer": res["fuzz_fault_0"]["buffer"],
                    }
                return res

            with mock.patch.object(fuzz_cpu, "_run_fault_probe_subprocess", side_effect=o2_drops_fault):
                code, _, err = _run_main([
                    "check", "0xFA17", "2", str(native), str(lifted), str(o2), "--area", "fault_reads",
                ])
            self.assertEqual(code, 1)
            self.assertIn("exception=0xc0000005", err)

            # Guard 2: valid-pointer return mismatch -> check MUST fail.
            def valid_mismatch(dll_path: Path, _prefix: str, probe_name: str, fn_names: list[str]) -> dict[str, dict]:
                res = self._baseline_results(fn_names, probe_name)
                if dll_path == lifted and probe_name == "valid":
                    res["fuzz_fault_0"]["return"] = 0x999
                return res

            with mock.patch.object(fuzz_cpu, "_run_fault_probe_subprocess", side_effect=valid_mismatch):
                code, _, err = _run_main([
                    "check", "0xFA17", "2", str(native), str(lifted), str(o2), "--area", "fault_reads",
                ])
            self.assertEqual(code, 1)
            self.assertIn("return=0x999", err)

            # Guard 3: mutant that never drops a fault -> check MUST fail.
            def inert_mutant(dll_path: Path, _prefix: str, probe_name: str, fn_names: list[str]) -> dict[str, dict]:
                return self._baseline_results(fn_names, probe_name)

            with mock.patch.object(fuzz_cpu, "_run_fault_probe_subprocess", side_effect=inert_mutant):
                code, _, err = _run_main([
                    "check", "0xFA17", "2", str(native), str(lifted), str(o2),
                    "--area", "fault_reads", "--mutant", str(mutant),
                ])
            self.assertEqual(code, 1)
            self.assertIn("did not drop any faulting read", err)


def verify_compiled_dlls(native: Path, lifted: Path, o2: Path, mutant: Path, seed: str, count: int) -> None:
    import cpu_switch_aot

    for label, path in (("native", native), ("lifted", lifted), ("o2", o2), ("mutant", mutant)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} DLL: {path}")

    # Verify the translated C++ contains volatile load_always and mutant removes volatile.
    functions = fuzz_cpu.generate_fault_reads(int(seed, 0), count)
    names = [fn.name for fn in functions]
    clean_src, _ = cpu_switch_aot.translate(native, names)
    if "load_always<" not in clean_src or "const volatile T __unaligned*" not in clean_src:
        raise AssertionError("clean fault_reads translation is missing volatile load_always")

    # End-to-end equivalence check via isolated subprocesses (positive + mutant).
    proc = subprocess.run(
        [sys.executable, "-B", str(TOOL), "check", seed, str(count),
         str(native), str(lifted), str(o2), "--area", "fault_reads", "--mutant", str(mutant)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(f"fault_reads equivalence check failed:\n{proc.stdout}\n{proc.stderr}")
    print(proc.stdout.strip())

    # Negative control: passing the /O2 mutant DLL as the o2 translation MUST fail equivalence.
    neg = subprocess.run(
        [sys.executable, "-B", str(TOOL), "check", seed, str(count),
         str(native), str(lifted), str(mutant), "--area", "fault_reads"],
        capture_output=True, text=True, timeout=60,
    )
    if neg.returncode == 0:
        raise AssertionError("passing dead-read-dropping /O2 mutant as o2 unexpectedly passed!")
    if "0xc0000005" not in neg.stderr.lower():
        raise AssertionError(f"negative control failed without reporting 0xc0000005:\n{neg.stderr}")
    print("negative control: passing /O2 mutant as o2 failed equivalence as expected")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--lifted", type=Path)
    parser.add_argument("--o2", type=Path)
    parser.add_argument("--mutant", type=Path)
    parser.add_argument("--seed", default="0xFA17")
    parser.add_argument("--count", type=int, default=24)
    args, remaining = parser.parse_known_args(argv)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FaultReadsGeneratorTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(FaultReadsCheckModeTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    if args.native and args.lifted and args.o2 and args.mutant:
        verify_compiled_dlls(args.native, args.lifted, args.o2, args.mutant, args.seed, args.count)
    else:
        build_dir = ROOT / "out" / "build" / "dev"
        candidates = (
            build_dir / "cpu_fuzz_fault_reads.dll",
            build_dir / "cpu_fuzz_fault_reads_lifted.dll",
            build_dir / "cpu_fuzz_fault_reads_lifted_o2.dll",
            build_dir / "cpu_fuzz_fault_reads_mutant_o2.dll",
        )
        if all(path.is_file() for path in candidates):
            verify_compiled_dlls(*candidates, args.seed, args.count)
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
