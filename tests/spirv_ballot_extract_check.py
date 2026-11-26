#!/usr/bin/env python3
"""Pure-Python unit and mutation suite for tools/spirv_ballot_extract.py."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "spirv_ballot_extract.py"
BALLOT_SPVASM = ROOT / "tests" / "spirv_window" / "ballot.spvasm"
sys.path.insert(0, str(ROOT / "tools"))
import spirv_ballot_extract as sbe  # noqa: E402


EXTRACT_MODULE = BALLOT_SPVASM.read_text(encoding="utf-8")

FIND_MODULE = """OpCapability Shader
OpCapability GroupNonUniformBallot
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%scope = OpConstant %uint 0x3
%workgroup = OpConstant %uint 2
%main = OpFunction %void None %fn
%entry = OpLabel
%lsb = OpGroupNonUniformBallotFindLSB %uint %scope %mask
OpReturn
OpFunctionEnd
"""

COMBINED_MODULE = """OpCapability Shader
OpCapability GroupNonUniformBallot
OpMemoryModel Logical GLSL450
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%void = OpTypeVoid
%fn = OpTypeFunction %void
%scope = OpConstant %uint 3
%main = OpFunction %void None %fn
%entry = OpLabel
%bit = OpGroupNonUniformBallotBitExtract %bool %scope %mask %lane  ; trailing comment
%lsb = OpGroupNonUniformBallotFindLSB %uint %scope %mask
OpReturn
OpFunctionEnd
"""


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpirvBallotExtractTest(unittest.TestCase):
    def test_passthrough_preserves_input_verbatim(self) -> None:
        plain = "OpCapability Shader\nOpMemoryModel Logical GLSL450\n%main = OpFunction %void None %fn\n"
        for sample in ("", plain, "%x2w_ballot_unused = OpConstant %uint 0\n"):
            self.assertEqual(sbe.lower_ballot_extract(sample), sample)

        # Unmatched FindLSB signatures (e.g. wrong result type) pass through untouched.
        wrong_find = FIND_MODULE.replace(
            "%lsb = OpGroupNonUniformBallotFindLSB %uint %scope %mask",
            "%lsb = OpGroupNonUniformBallotFindLSB %bool %scope %mask",
        )
        self.assertEqual(sbe.lower_ballot_extract(wrong_find), wrong_find)

    def test_ballot_bit_extract_lowering(self) -> None:
        lowered = sbe.lower_ballot_extract(EXTRACT_MODULE)
        lines = lowered.splitlines()
        self.assertNotIn("OpGroupNonUniformBallotBitExtract", lowered)
        self.assertNotIn("OpExtInstImport", lowered)
        for c in (0, 1, 5, 31):
            self.assertIn(f"%x2w_ballot_c{c} = OpConstant %uint {c}", lines)
        for c in (32, 64, 96):
            self.assertNotIn(f"%x2w_ballot_c{c}", lowered)
        fn_idx = lines.index("%main = OpFunction %void None %fn")
        self.assertEqual(
            lines[fn_idx - 4:fn_idx],
            [f"%x2w_ballot_c{c} = OpConstant %uint {c}" for c in (0, 1, 5, 31)],
        )
        self.assertIn("%x2w_ballot_bit_word_index = OpShiftRightLogical %uint %i %x2w_ballot_c5", lines)
        self.assertIn("%x2w_ballot_bit_word = OpVectorExtractDynamic %uint %mask %x2w_ballot_bit_word_index", lines)
        self.assertIn("%x2w_ballot_bit_bit = OpBitwiseAnd %uint %i %x2w_ballot_c31", lines)
        self.assertIn("%x2w_ballot_bit_shifted = OpShiftRightLogical %uint %x2w_ballot_bit_word %x2w_ballot_bit_bit", lines)
        self.assertIn("%x2w_ballot_bit_low = OpBitwiseAnd %uint %x2w_ballot_bit_shifted %x2w_ballot_c1", lines)
        self.assertIn("%bit = OpINotEqual %bool %x2w_ballot_bit_low %x2w_ballot_c0", lines)

    def test_ballot_find_lsb_injects_glsl_import(self) -> None:
        lowered = sbe.lower_ballot_extract(FIND_MODULE)
        lines = lowered.splitlines()
        self.assertNotIn("OpGroupNonUniformBallotFindLSB", lowered)
        self.assertEqual(lines.count('%x2w_ballot_glsl = OpExtInstImport "GLSL.std.450"'), 1)
        mm_idx = lines.index("OpMemoryModel Logical GLSL450")
        self.assertEqual(lines[mm_idx - 1], '%x2w_ballot_glsl = OpExtInstImport "GLSL.std.450"')
        for c in (31, 32, 64, 96):
            self.assertIn(f"%x2w_ballot_c{c} = OpConstant %uint {c}", lines)
        for c in (0, 1, 5):
            self.assertNotIn(f"%x2w_ballot_c{c} = OpConstant", lowered)
        self.assertIn("%x2w_ballot_lsb_w0 = OpCompositeExtract %uint %mask 0", lines)
        self.assertIn("%x2w_ballot_lsb_c0 = OpExtInst %uint %x2w_ballot_glsl FindILsb %x2w_ballot_lsb_w0", lines)
        self.assertIn("%x2w_ballot_lsb_m3 = OpShiftRightArithmetic %uint %x2w_ballot_lsb_l3 %x2w_ballot_c31", lines)
        self.assertIn("%x2w_ballot_lsb_a3 = OpIAdd %uint %x2w_ballot_lsb_l3 %x2w_ballot_c96", lines)
        self.assertIn("%lsb = OpExtInst %uint %x2w_ballot_glsl UMin %x2w_ballot_lsb_c0 %x2w_ballot_lsb_u1", lines)

    def test_ballot_find_lsb_reuses_existing_glsl_import(self) -> None:
        with_import = FIND_MODULE.replace(
            "OpMemoryModel Logical GLSL450\n",
            '%existing_glsl = OpExtInstImport "GLSL.std.450"\n',
        )
        lowered = sbe.lower_ballot_extract(with_import)
        lines = lowered.splitlines()
        self.assertEqual(lowered.count("OpExtInstImport"), 1)
        self.assertNotIn("%x2w_ballot_glsl", lowered)
        self.assertIn("%x2w_ballot_lsb_c0 = OpExtInst %uint %existing_glsl FindILsb %x2w_ballot_lsb_w0", lines)
        self.assertIn("%lsb = OpExtInst %uint %existing_glsl UMin %x2w_ballot_lsb_c0 %x2w_ballot_lsb_u1", lines)

    def test_combined_extract_and_find_lsb(self) -> None:
        lowered = sbe.lower_ballot_extract(COMBINED_MODULE)
        lines = lowered.splitlines()
        self.assertNotIn("OpGroupNonUniformBallotBitExtract", lowered)
        self.assertNotIn("OpGroupNonUniformBallotFindLSB", lowered)
        fn_idx = lines.index("%main = OpFunction %void None %fn")
        self.assertEqual(
            lines[fn_idx - 7:fn_idx],
            [f"%x2w_ballot_c{c} = OpConstant %uint {c}" for c in (0, 1, 5, 31, 32, 64, 96)],
        )
        self.assertIn("%bit = OpINotEqual %bool %x2w_ballot_bit_low %x2w_ballot_c0", lines)
        self.assertIn("%lsb = OpExtInst %uint %x2w_ballot_glsl UMin %x2w_ballot_lsb_c0 %x2w_ballot_lsb_u1", lines)

    def test_negative_controls(self) -> None:
        for bad in (None, b"OpCapability Shader", 42, ["OpMemoryModel Logical GLSL450"]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(TypeError, "assembly must be a str"):
                    sbe.lower_ballot_extract(bad)

        reserved = EXTRACT_MODULE.replace("%one", "%x2w_ballot_taken")
        with self.assertRaisesRegex(ValueError, "Reserved ballot lowering ID already in use"):
            sbe.lower_ballot_extract(reserved)

        for bad_scope in ("%workgroup", "%missing", "%main"):
            with self.subTest(bad_scope=bad_scope):
                probe = FIND_MODULE.replace("%scope %mask", f"{bad_scope} %mask")
                with self.assertRaisesRegex(ValueError, "Subgroup scope constant"):
                    sbe.lower_ballot_extract(probe)

        no_fn_extract = EXTRACT_MODULE.replace("%main = OpFunction %void None %fn", "%main_alt = OpLabel")
        with self.assertRaisesRegex(ValueError, "OpFunction"):
            sbe.lower_ballot_extract(no_fn_extract)

        no_fn_find = FIND_MODULE.replace("%main = OpFunction %void None %fn", "%main_alt = OpLabel")
        with self.assertRaisesRegex(ValueError, "OpFunction"):
            sbe.lower_ballot_extract(no_fn_find)

        no_mm_find = FIND_MODULE.replace("OpMemoryModel Logical GLSL450\n", "")
        with self.assertRaisesRegex(ValueError, "OpMemoryModel"):
            sbe.lower_ballot_extract(no_mm_find)

    def test_fast_temp_mutations_killed(self) -> None:
        original_src = TOOL.read_text(encoding="utf-8")
        mutations = (
            (
                "drop_str_check",
                "    if not isinstance(assembly, str):",
                "    if False:",
                lambda mod: mod.lower_ballot_extract(["OpCapability Shader"]),
                TypeError,
                "assembly must be a str",
            ),
            (
                "drop_reserved_id_check",
                '    if "%x2w_ballot_" in assembly:',
                "    if False:",
                lambda mod: mod.lower_ballot_extract(EXTRACT_MODULE.replace("%one", "%x2w_ballot_taken")),
                ValueError,
                "Reserved ballot lowering ID already in use",
            ),
            (
                "wrong_subgroup_scope",
                "SUBGROUP_SCOPE = 3",
                "SUBGROUP_SCOPE = 2",
                lambda mod: mod.lower_ballot_extract(FIND_MODULE.replace("%scope %mask", "%workgroup %mask")),
                ValueError,
                "Subgroup scope constant",
            ),
            (
                "drop_opfunction_check",
                "    if not declared:",
                "    if False:",
                lambda mod: mod.lower_ballot_extract(
                    EXTRACT_MODULE.replace("%main = OpFunction %void None %fn", "%main_alt = OpLabel")
                ),
                ValueError,
                "OpFunction",
            ),
            (
                "drop_opmemorymodel_check",
                "    if import_needed and not imported:",
                "    if False:",
                lambda mod: mod.lower_ballot_extract(FIND_MODULE.replace("OpMemoryModel Logical GLSL450\n", "")),
                ValueError,
                "OpMemoryModel",
            ),
        )
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for label, old, new, trigger, expected_exc, expected_msg in mutations:
                mutated_src = original_src.replace(old, new, 1)
                self.assertNotEqual(mutated_src, original_src, label)
                mut_path = root / f"mut_{label}.py"
                mut_path.write_text(mutated_src, encoding="utf-8")
                mod = _load_module(mut_path, f"mut_{label}")
                try:
                    trigger(mod)
                except expected_exc as exc:
                    if expected_msg in str(exc):
                        raise AssertionError(f"mutation {label} was not killed: {exc}")
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
