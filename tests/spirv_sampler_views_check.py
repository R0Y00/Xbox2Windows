"""Unit, comment-stripping, table-splitting, refusal, and temp-copy mutation suite for tools/spirv_sampler_views.py."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "spirv_sampler_views.py"
sys.path.insert(0, str(ROOT / "tools"))

import spirv_sampler_views  # noqa: E402


BASE_MODULE = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %texture %samplers %out %tid
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %tid BuiltIn WorkgroupId
OpDecorate %texture DescriptorSet 0
OpDecorate %texture Binding 0
OpDecorate %samplers DescriptorSet 0
OpDecorate %samplers Binding 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%vec2 = OpTypeVector %float 2
%vec4 = OpTypeVector %float 4
%xyz = OpTypeVector %uint 3
%input = OpTypePointer Input %xyz
%tid = OpVariable %input Input
%zero = OpConstant %uint 0
%two = OpConstant %uint 2
%lod = OpConstant %float 0
%half = OpConstant %float 0.5
%uv = OpConstantComposite %vec2 %half %half
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%pair = OpTypeSampledImage %img
%smp = OpTypeSampler
%ptr_img = OpTypePointer UniformConstant %img
%texture = OpVariable %ptr_img UniformConstant
%samparr = OpTypeArray %smp %two
%_ptr_UC_rt_smp = OpTypePointer UniformConstant %samparr
%_ptr_UC_smp = OpTypePointer UniformConstant %smp
%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant
%out_ptr = OpTypePointer StorageBuffer %uint
%out = OpVariable %out_ptr StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
%xyzval = OpLoad %xyz %tid
%i = OpCompositeExtract %uint %xyzval 0
%p = OpAccessChain %_ptr_UC_smp %samplers %i
%s = OpLoad %smp %p
%im = OpLoad %img %texture
%si = OpSampledImage %pair %im %s
%raw = OpImageSampleExplicitLod %vec4 %si %uv Lod %lod
%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod
OpReturn
OpFunctionEnd
"""


MUTATIONS = (
    (
        "drop_non_str_guard",
        "    if not isinstance(assembly, str):\n"
        '        raise ValueError(f"assembly must be a str, got {type(assembly).__name__}")\n',
        "    if False:\n"
        '        raise ValueError(f"assembly must be a str, got {type(assembly).__name__}")\n',
    ),
    (
        "drop_isascii_guard",
        "    if not isinstance(token, str) or not token.isascii():\n",
        "    if not isinstance(token, str):\n",
    ),
    (
        "drop_hex_literal_parse",
        '    if len(token) > 2 and token[:2] in ("0x", "0X") and all(ch in "0123456789abcdefABCDEF" for ch in token[2:]):\n',
        "    if False:\n",
    ),
    (
        "drop_hex_af_class",
        '    if len(token) > 2 and token[:2] in ("0x", "0X") and all(ch in "0123456789abcdefABCDEF" for ch in token[2:]):\n',
        '    if len(token) > 2 and token[:2] in ("0x", "0X") and all(ch in "0123456789" for ch in token[2:]):\n',
    ),
    (
        "drop_hex_whitelist",
        '    if len(token) > 2 and token[:2] in ("0x", "0X") and all(ch in "0123456789abcdefABCDEF" for ch in token[2:]):\n',
        '    if len(token) > 2 and token[:2] in ("0x", "0X"):\n',
    ),
    (
        "decimal_as_hex",
        "    if token.isdigit():\n"
        '        if len(token) > 1 and token[0] == "0":\n'
        "            raise ValueError(\n"
        '                f"leading-zero decimal literal {token!r} is ambiguous: spirv-as reads it as octal; "\n'
        '                "use a plain decimal or 0x-prefixed hex literal"\n'
        "            )\n"
        "        return int(token, 10)\n",
        "    if token.isdigit():\n"
        '        if len(token) > 1 and token[0] == "0":\n'
        "            raise ValueError(\n"
        '                f"leading-zero decimal literal {token!r} is ambiguous: spirv-as reads it as octal; "\n'
        '                "use a plain decimal or 0x-prefixed hex literal"\n'
        "            )\n"
        "        return int(token, 16)\n",
    ),
    (
        "register_end_decimal_only",
        '    length_num = _parse_uint_literal(length[2]) if len(length) == 3 and length[0] == "OpConstant" else None\n',
        '    length_num = int(length[2]) if len(length) == 3 and length[0] == "OpConstant" and length[2].isdigit() else None\n',
    ),
    (
        "drop_register_end_safe_lookup",
        "    decl = defs.get(table, [])\n"
        "    pointer = defs.get(decl[1], []) if len(decl) >= 2 else []\n"
        "    shape = defs.get(pointer[2], []) if len(pointer) >= 3 else []\n"
        '    length = defs.get(shape[2], []) if len(shape) >= 3 and shape[0] == "OpTypeArray" else []\n',
        "    shape = defs[defs[defs[table][1]][2]]\n"
        '    length = defs.get(shape[2], []) if shape[0] == "OpTypeArray" else []\n',
    ),
    (
        "drop_query_lod_ordinary",
        '        elif words[2] == "OpImageQueryLod" or words[2].startswith(\n',
        "        elif words[2].startswith(\n",
    ),
    (
        "drop_unresolved_rule",
        '            unresolved = True  # an ordinary sample with an unknown sampler may use any table\n',
        "            pass\n",
    ),
    (
        "drop_failures_dispatch",
        '            raise ValueError(FAILURES[source[0]])\n',
        "            pass\n",
    ),
    (
        "drop_early_exit_no_tables",
        '    if not tables or not module["comparisons"] or not module["ordinary"]:\n',
        "    if False:\n",
    ),
    (
        "drop_indirect_decoration_guard",
        '    if module["indirect"]:\n',
        "    if False:\n",
    ),
    (
        "drop_bridge_set3_guard",
        "    if BRIDGE in assembly or set_three:\n",
        "    if False:\n",
    ),
    (
        "drop_overlap_check",
        "            if (stop is None or binding_num < stop) and (end is None or start < end):\n",
        "            if False:\n",
    ),
    (
        "overlap_le_stop",
        "            if (stop is None or binding_num < stop) and (end is None or start < end):\n",
        "            if (stop is None or binding_num <= stop) and (end is None or start < end):\n",
    ),
    (
        "overlap_le_start",
        "            if (stop is None or binding_num < stop) and (end is None or start < end):\n",
        "            if (stop is None or binding_num < stop) and (end is None or start <= end):\n",
    ),
    (
        "drop_decorated_sampler_path_guard",
        '        if module["decorated"].intersection((chain[2], words[4], sampled_load, pointer)):\n',
        "        if False:\n",
    ),
)


def _sub(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"expected single occurrence of {old!r}")
    return text.replace(old, new)


def _load_temp_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not create module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


class SpirvSamplerViewsTests(unittest.TestCase):
    def _assert_register_end_none(self, mod, defs, table, binding, what) -> None:
        """register_end must return None for missing/non-ASCII data, not crash.

        A KeyError/ValueError here would be a raw crash, not the specified
        safe-lookup behaviour, so convert it to an assertion failure.
        """
        try:
            result = mod.register_end(defs, table, binding)
        except Exception as exc:  # noqa: BLE001 - any crash is a contract failure
            self.fail(f"register_end({what}) raised {type(exc).__name__} instead of returning None: {exc}")
        self.assertIsNone(result, msg=what)

    def _verify_module_contract(self, mod) -> None:
        self.assertEqual(mod.BRIDGE, "%bridge_cmp")
        self.assertEqual(mod.BRIDGE_SET, "3")

        # strip_comments with quoted ';', escaped quotes, and multi-line strings.
        comment_input = (
            'OpEntryPoint GLCompute %main "main;x" %samplers ; entry comment\n'
            '%s = OpString "quote \\" ; keep" ; trailing\n'
            '%multi = OpString "first line\n'
            'second;still_quoted" ; after multi\n'
            "%c = OpConstant %uint 1 ; plain comment"
        )
        self.assertEqual(
            mod.strip_comments(comment_input),
            [
                'OpEntryPoint GLCompute %main "main;x" %samplers',
                '%s = OpString "quote \\" ; keep"',
                '%multi = OpString "first line',
                'second;still_quoted"',
                "%c = OpConstant %uint 1",
            ],
        )

        # Non-str rejection in strip_comments and split_sampler_views.
        # The expected behaviour is a ValueError naming the type; any other
        # exception (e.g. AttributeError from calling a method on None) is a
        # contract failure, not a kill, so convert it to an assertion here.
        for bad in (None, 123, b"Dref", ["OpCapability Shader"]):
            for fn_name in ("strip_comments", "split_sampler_views"):
                fn = getattr(mod, fn_name)
                try:
                    fn(bad)
                except ValueError as exc:
                    self.assertIn("assembly must be a str", str(exc), msg=f"{fn_name}({bad!r})")
                except Exception as exc:  # noqa: BLE001 - any other type is a contract failure
                    self.fail(f"{fn_name}({bad!r}) raised {type(exc).__name__} instead of ValueError: {exc}")
                else:
                    self.fail(f"{fn_name}({bad!r}) did not raise ValueError")

        # Single-table splitting preserves BRIDGE, BRIDGE_SET, OpEntryPoint, and Dref formatting.
        base_lines = BASE_MODULE.splitlines()
        cmp_index = base_lines.index("%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod")
        split_lines = mod.split_sampler_views(BASE_MODULE).splitlines()
        self.assertIn(
            'OpEntryPoint GLCompute %main "main" %texture %samplers %out %tid %bridge_cmp',
            split_lines,
        )
        self.assertIn("OpDecorate %bridge_cmp DescriptorSet 3", split_lines)
        self.assertIn("OpDecorate %bridge_cmp Binding 1", split_lines)
        self.assertIn("%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant", split_lines)
        self.assertIn("%raw = OpImageSampleExplicitLod %vec4 %si %uv Lod %lod", split_lines)
        prefix = f"%bridge_cmp_{cmp_index}"
        self.assertIn(f"{prefix}_ptr = OpAccessChain %_ptr_UC_smp %bridge_cmp %i", split_lines)
        self.assertIn(f"{prefix}_value = OpLoad %smp {prefix}_ptr", split_lines)
        self.assertIn(f"{prefix}_image = OpSampledImage %pair %im {prefix}_value", split_lines)
        self.assertIn(f"%cmp = OpImageSampleDrefExplicitLod %float {prefix}_image %uv %half Lod %lod", split_lines)

        # OpImageDrefGather rewrite preserves OpSampledImage + OpImageDrefGather formatting.
        dref_gather = _sub(
            BASE_MODULE,
            "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod",
            "%cmp = OpImageDrefGather %vec4 %si %uv %half",
        )
        gather_lines = mod.split_sampler_views(dref_gather).splitlines()
        self.assertIn(f"%cmp = OpImageDrefGather %vec4 {prefix}_image %uv %half", gather_lines)

        # OpImageQueryLod is classified as ordinary use and triggers split.
        query_lod = _sub(
            BASE_MODULE,
            "%raw = OpImageSampleExplicitLod %vec4 %si %uv Lod %lod",
            "%raw = OpImageQueryLod %vec2 %si %uv",
        )
        query_lines = mod.split_sampler_views(query_lod).splitlines()
        self.assertIn("%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant", query_lines)
        self.assertIn("%raw = OpImageQueryLod %vec2 %si %uv", query_lines)

        # Multi-table splitting: two mixed tables + one comparison-only table.
        multi = _sub(
            BASE_MODULE,
            'OpEntryPoint GLCompute %main "main" %texture %samplers %out %tid',
            'OpEntryPoint GLCompute %main "main" %texture %samplers %second %cmp_only %out %tid',
        )
        multi = _sub(
            multi,
            "OpDecorate %samplers Binding 1",
            "OpDecorate %samplers Binding 1\n"
            "OpDecorate %second DescriptorSet 1\n"
            "OpDecorate %second Binding 4\n"
            "OpDecorate %cmp_only DescriptorSet 2\n"
            "OpDecorate %cmp_only Binding 9",
        )
        multi = _sub(
            multi,
            "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant",
            "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant\n"
            "%rt_smp = OpTypeRuntimeArray %smp\n"
            "%_ptr_UC_rt2 = OpTypePointer UniformConstant %rt_smp\n"
            "%second = OpVariable %_ptr_UC_rt2 UniformConstant\n"
            "%cmp_only = OpVariable %_ptr_UC_rt_smp UniformConstant",
        )
        multi = _sub(
            multi,
            "OpReturn",
            "%p2 = OpAccessChain %_ptr_UC_smp %second %i\n"
            "%s2 = OpLoad %smp %p2\n"
            "%si2 = OpSampledImage %pair %im %s2\n"
            "%raw2 = OpImageSampleExplicitLod %vec4 %si2 %uv Lod %lod\n"
            "%cmp2 = OpImageSampleDrefExplicitLod %float %si2 %uv %half Lod %lod\n"
            "%p3 = OpAccessChain %_ptr_UC_smp %cmp_only %i\n"
            "%s3 = OpLoad %smp %p3\n"
            "%si3 = OpSampledImage %pair %im %s3\n"
            "%cmp3 = OpImageSampleDrefExplicitLod %float %si3 %uv %half Lod %lod\n"
            "OpReturn",
        )
        multi_lines = mod.split_sampler_views(multi).splitlines()
        self.assertIn("%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant", multi_lines)
        self.assertIn("%bridge_cmp_table_1 = OpVariable %_ptr_UC_rt2 UniformConstant", multi_lines)
        self.assertIn("OpDecorate %bridge_cmp Binding 1", multi_lines)
        self.assertIn("OpDecorate %bridge_cmp_table_1 Binding 4", multi_lines)
        self.assertFalse(any("Binding 9" in line and "bridge" in line for line in multi_lines))

        # Pass-through when no comparison, no mixed table, or bindless OpTypeSampledImage array.
        no_dref = _sub(BASE_MODULE, "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod\n", "")
        self.assertEqual(mod.split_sampler_views(no_dref), no_dref)

        bindless_sampled_image = _sub(
            BASE_MODULE,
            "%samparr = OpTypeArray %smp %two\n"
            "%_ptr_UC_rt_smp = OpTypePointer UniformConstant %samparr\n"
            "%_ptr_UC_smp = OpTypePointer UniformConstant %smp\n"
            "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant",
            "%pair_arr = OpTypeArray %pair %two\n"
            "%_ptr_UC_pair_arr = OpTypePointer UniformConstant %pair_arr\n"
            "%_ptr_UC_pair = OpTypePointer UniformConstant %pair\n"
            "%samplers = OpVariable %_ptr_UC_pair_arr UniformConstant",
        )
        bindless_sampled_image = _sub(
            bindless_sampled_image,
            "%p = OpAccessChain %_ptr_UC_smp %samplers %i\n"
            "%s = OpLoad %smp %p\n"
            "%im = OpLoad %img %texture\n"
            "%si = OpSampledImage %pair %im %s",
            "%p = OpAccessChain %_ptr_UC_pair %samplers %i\n"
            "%si = OpLoad %pair %p",
        )
        self.assertEqual(mod.split_sampler_views(bindless_sampled_image), bindless_sampled_image)

        # Conservative unresolved-ordinary rule: unknown ordinary sampler forces split of Dref table.
        unresolved_ordinary = _sub(
            BASE_MODULE,
            "%raw = OpImageSampleExplicitLod %vec4 %si %uv Lod %lod",
            "%weird_smp = OpUndef %smp\n"
            "%weird_si = OpSampledImage %pair %im %weird_smp\n"
            "%raw = OpImageSampleExplicitLod %vec4 %weird_si %uv Lod %lod",
        )
        unresolved_lines = mod.split_sampler_views(unresolved_ordinary).splitlines()
        self.assertIn("%bridge_cmp = OpVariable %_ptr_UC_rt_smp UniformConstant", unresolved_lines)
        self.assertIn("OpDecorate %bridge_cmp DescriptorSet 3", unresolved_lines)

        # Valid hex-versus-decimal controls: 0x2 does not collide with set 3; hex table binding 0x1 works & checks overlap.
        hex_ok = _sub(BASE_MODULE, "OpDecorate %out DescriptorSet 0", "OpDecorate %out DescriptorSet 0x2")
        hex_ok_lines = mod.split_sampler_views(hex_ok).splitlines()
        self.assertIn("OpDecorate %bridge_cmp DescriptorSet 3", hex_ok_lines)

        hex_binding_ok = _sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 0x1F")
        hex_binding_lines = mod.split_sampler_views(hex_binding_ok).splitlines()
        self.assertIn("OpDecorate %bridge_cmp Binding 0x1F", hex_binding_lines)

        hex_overlap = _sub(multi, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 0x1")
        hex_overlap = _sub(hex_overlap, "OpDecorate %second Binding 4", "OpDecorate %second Binding 0x2")
        with self.assertRaisesRegex(ValueError, "overlap"):
            mod.split_sampler_views(hex_overlap)

        # Decimal >= 10 overlap control: binding 9 (size 2 -> 9..10) overlaps binding 10 (fails if 10 parsed as 0x10=16).
        dec10_overlap = _sub(multi, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 9")
        dec10_overlap = _sub(dec10_overlap, "OpDecorate %second Binding 4", "OpDecorate %second Binding 10")
        with self.assertRaisesRegex(ValueError, "overlap"):
            mod.split_sampler_views(dec10_overlap)

        # _parse_uint_literal: 0, plain decimal and 0x/0X hex work; multi-digit
        # decimals starting with 0 are refused (spirv-as reads them as octal).
        self.assertEqual(mod._parse_uint_literal("0"), 0)
        self.assertEqual(mod._parse_uint_literal("10"), 10)
        self.assertEqual(mod._parse_uint_literal("0xa"), 10)
        self.assertEqual(mod._parse_uint_literal("0XA"), 10)
        self.assertIsNone(mod._parse_uint_literal("0x"))
        self.assertIsNone(mod._parse_uint_literal("%bad"))
        for leading_zero in ("010", "08", "00"):
            with self.assertRaisesRegex(ValueError, "leading-zero"):
                mod._parse_uint_literal(leading_zero)

        # Leading-zero witness (pigalot): Binding 010 is octal 8, so a table
        # at Binding 010 (8..10) overlaps a fixed array of 2 at Binding 7
        # (7..9) at register 8. Reading 010 as decimal 10 would miss it.
        leading_zero_overlap = _sub(multi, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 010")
        leading_zero_overlap = _sub(leading_zero_overlap, "OpDecorate %second Binding 4", "OpDecorate %second Binding 7")
        leading_zero_overlap = _sub(
            leading_zero_overlap,
            "%second = OpVariable %_ptr_UC_rt2 UniformConstant",
            "%second = OpVariable %_ptr_UC_rt_smp UniformConstant",
        )
        with self.assertRaisesRegex(ValueError, "leading-zero"):
            mod.split_sampler_views(leading_zero_overlap)
        for bad_binding in ("010", "08"):
            with self.assertRaisesRegex(ValueError, "leading-zero"):
                mod.split_sampler_views(
                    _sub(BASE_MODULE, "OpDecorate %samplers Binding 1", f"OpDecorate %samplers Binding {bad_binding}")
                )

        # DescriptorSet +3 is set 3 to spirv-as, so it must refuse like 3/03/0x3.
        with self.assertRaisesRegex(ValueError, "free descriptor set 3"):
            mod.split_sampler_views(_sub(BASE_MODULE, "OpDecorate %out DescriptorSet 0", "OpDecorate %out DescriptorSet +3"))

        # Boundary mutant killers (each pins one surviving boundary):
        # (a) length_num > 0 -> > 1: two adjacent length-1 tables (1..2, 2..3)
        # must split, not refuse as overlap.
        adj_len1 = _sub(multi, "%two = OpConstant %uint 2", "%two = OpConstant %uint 1")
        adj_len1 = _sub(adj_len1, "OpDecorate %second Binding 4", "OpDecorate %second Binding 2")
        adj_len1_lines = mod.split_sampler_views(adj_len1).splitlines()
        self.assertIn("OpDecorate %bridge_cmp Binding 1", adj_len1_lines)
        self.assertIn("OpDecorate %bridge_cmp_table_1 Binding 2", adj_len1_lines)

        # (b) start < end -> start < end - 1: reverse-order one-register overlap.
        # %samplers at Binding 2 (2..4) is processed first, %second at Binding 1
        # (1..3) shares register 2 and must be refused.
        rev_overlap = _sub(multi, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 2")
        rev_overlap = _sub(rev_overlap, "OpDecorate %second Binding 4", "OpDecorate %second Binding 1")
        rev_overlap = _sub(
            rev_overlap,
            "%second = OpVariable %_ptr_UC_rt2 UniformConstant",
            "%second = OpVariable %_ptr_UC_rt_smp UniformConstant",
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            mod.split_sampler_views(rev_overlap)

        # (c) uppercase-only hex class: a lowercase hex binding must split.
        hex_lower = _sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 0xa")
        hex_lower_lines = mod.split_sampler_views(hex_lower).splitlines()
        self.assertIn("OpDecorate %bridge_cmp Binding 0xa", hex_lower_lines)

        # (d) len(token) > 2 -> >= 2: bare 0x/0X must refuse with the pinned
        # message, not a raw ValueError from int('', 16).
        for bare in ("0x", "0X"):
            with self.assertRaisesRegex(ValueError, "literal DescriptorSet and Binding"):
                mod.split_sampler_views(
                    _sub(BASE_MODULE, "OpDecorate %samplers Binding 1", f"OpDecorate %samplers Binding {bare}")
                )

        # Adjacent non-overlapping register ranges in both orders (1..2 vs 3, and 3..4 vs 1) + hex array length 0x2.
        adj_asc = _sub(multi, "OpDecorate %second Binding 4", "OpDecorate %second Binding 3")
        adj_asc = _sub(adj_asc, "%two = OpConstant %uint 2", "%two = OpConstant %uint 0x2")
        adj_asc_lines = mod.split_sampler_views(adj_asc).splitlines()
        self.assertIn("OpDecorate %bridge_cmp Binding 1", adj_asc_lines)
        self.assertIn("OpDecorate %bridge_cmp_table_1 Binding 3", adj_asc_lines)

        adj_desc = _sub(multi, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 3")
        adj_desc = _sub(adj_desc, "OpDecorate %second Binding 4", "OpDecorate %second Binding 1")
        adj_desc = _sub(adj_desc, "%second = OpVariable %_ptr_UC_rt2 UniformConstant", "%second = OpVariable %_ptr_UC_rt_smp UniformConstant")
        adj_desc_lines = mod.split_sampler_views(adj_desc).splitlines()
        self.assertIn("OpDecorate %bridge_cmp Binding 3", adj_desc_lines)
        self.assertIn("OpDecorate %bridge_cmp_table_1 Binding 1", adj_desc_lines)

        # register_end handles missing OpTypePointer / OpTypeArray keys and non-ASCII constant digits without KeyError.
        self._assert_register_end_none(mod, {}, "%samplers", "1", "empty defs")
        self._assert_register_end_none(
            mod, {"%samplers": ["OpVariable", "%missing_ptr", "UniformConstant"]}, "%samplers", "1", "missing pointer"
        )
        self._assert_register_end_none(
            mod,
            {
                "%samplers": ["OpVariable", "%ptr", "UniformConstant"],
                "%ptr": ["OpTypePointer", "UniformConstant", "%missing_arr"],
            },
            "%samplers",
            "1",
            "missing array",
        )
        self._assert_register_end_none(
            mod,
            {
                "%samplers": ["OpVariable", "%ptr", "UniformConstant"],
                "%ptr": ["OpTypePointer", "UniformConstant", "%arr"],
                "%arr": ["OpTypeArray", "%smp"],
            },
            "%samplers",
            "1",
            "array without length",
        )
        self._assert_register_end_none(
            mod,
            {
                "%samplers": ["OpVariable", "%ptr", "UniformConstant"],
                "%ptr": ["OpTypePointer", "UniformConstant", "%arr"],
                "%arr": ["OpTypeArray", "%smp", "%two"],
                "%two": ["OpConstant", "%uint", "\u00b3"],
            },
            "%samplers",
            "1",
            "non-ASCII length",
        )
        self.assertEqual(
            mod.register_end(
                {
                    "%samplers": ["OpVariable", "%ptr", "UniformConstant"],
                    "%ptr": ["OpTypePointer", "UniformConstant", "%arr"],
                    "%arr": ["OpTypeArray", "%smp", "%two"],
                    "%two": ["OpConstant", "%uint", "2"],
                },
                "%samplers",
                "0x1",
            ),
            3,
        )

        # table_slot refuses missing table, missing/non-literal/duplicate/extra decorations.
        parsed = mod.read_module(mod.strip_comments(BASE_MODULE))
        self.assertEqual(mod.table_slot(parsed, "%samplers"), ("0", "1"))
        with self.assertRaisesRegex(ValueError, "one DescriptorSet and one Binding"):
            mod.table_slot(parsed, "%missing")

        # Refusals: indirect decorations, set 3 (decimal and hex), %bridge_cmp, overlap, FAILURES, decorated sampler, OpDecorate.
        indirect = _sub(
            BASE_MODULE,
            "OpDecorate %samplers Binding 1",
            "OpDecorate %samplers Binding 1\n%grp = OpDecorationGroup\nOpGroupDecorate %grp %raw",
        )
        with self.assertRaisesRegex(ValueError, "indirect"):
            mod.split_sampler_views(indirect)

        indirect_id = _sub(
            BASE_MODULE,
            "OpDecorate %samplers Binding 1",
            "OpDecorate %samplers Binding 1\nOpDecorateId %raw AlignmentId %two",
        )
        with self.assertRaisesRegex(ValueError, "indirect"):
            mod.split_sampler_views(indirect_id)

        for set3_line in (
            "OpDecorate %samplers DescriptorSet 3",
            "\tOpDecorate  %samplers  DescriptorSet 03",
            "OpDecorate %out DescriptorSet 0x3",
            "OpDecorate %out DescriptorSet 0X03",
        ):
            target_line = "OpDecorate %out DescriptorSet 0" if "%out" in set3_line else "OpDecorate %samplers DescriptorSet 0"
            with self.assertRaisesRegex(ValueError, "free descriptor set 3"):
                mod.split_sampler_views(_sub(BASE_MODULE, target_line, set3_line))

        with self.assertRaisesRegex(ValueError, "bridge_cmp"):
            mod.split_sampler_views(
                _sub(BASE_MODULE, "%two = OpConstant %uint 2", "%two = OpConstant %uint 2\n%bridge_cmp = OpConstant %uint 9")
            )

        # Overlapping set 3 register range (binding 1 size 2 covers 1..2; second at binding 2 overlaps).
        overlap = _sub(multi, "OpDecorate %second Binding 4", "OpDecorate %second Binding 2")
        with self.assertRaisesRegex(ValueError, "overlap"):
            mod.split_sampler_views(overlap)

        # FAILURES dispatch refusals (image / load / provenance) and decorated sampler path.
        bad_dref_image = _sub(
            BASE_MODULE,
            "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod",
            "%cmp = OpImageSampleDrefExplicitLod %float %im %uv %half Lod %lod",
        )
        with self.assertRaisesRegex(ValueError, "unsupported depth-reference sampled image"):
            mod.split_sampler_views(bad_dref_image)

        bad_dref_load = _sub(
            BASE_MODULE,
            "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod",
            "%bad_si = OpSampledImage %pair %im %im\n"
            "%cmp = OpImageSampleDrefExplicitLod %float %bad_si %uv %half Lod %lod",
        )
        with self.assertRaisesRegex(ValueError, "unsupported depth-reference sampler load"):
            mod.split_sampler_views(bad_dref_load)

        bad_dref_prov = _sub(
            BASE_MODULE,
            "%cmp = OpImageSampleDrefExplicitLod %float %si %uv %half Lod %lod",
            "%bad_ptr = OpUndef %_ptr_UC_smp\n"
            "%bad_s = OpLoad %smp %bad_ptr\n"
            "%bad_si = OpSampledImage %pair %im %bad_s\n"
            "%cmp = OpImageSampleDrefExplicitLod %float %bad_si %uv %half Lod %lod",
        )
        with self.assertRaisesRegex(ValueError, "unsupported depth-reference sampler provenance"):
            mod.split_sampler_views(bad_dref_prov)

        decorated_si = _sub(
            BASE_MODULE,
            "OpDecorate %samplers Binding 1",
            "OpDecorate %samplers Binding 1\nOpDecorate %si NonUniformEXT",
        )
        with self.assertRaisesRegex(ValueError, "unsupported decorated sampler path"):
            mod.split_sampler_views(decorated_si)

        # Malformed / non-ASCII OpDecorate refusals with pinned messages.
        for bad_dec, expected_msg in (
            (_sub(BASE_MODULE, "OpDecorate %samplers DescriptorSet 0\n", ""), "one DescriptorSet and one Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers Binding 1\n", ""), "one DescriptorSet and one Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers DescriptorSet"), "one DescriptorSet and one Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers DescriptorSet %bad"), "literal DescriptorSet and Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers DescriptorSet \u0663"), "literal DescriptorSet and Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding -1"), "literal DescriptorSet and Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 0_1"), "literal DescriptorSet and Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 0x1_0"), "literal DescriptorSet and Binding"),
            (_sub(BASE_MODULE, "OpDecorate %samplers Binding 1", "OpDecorate %samplers Binding 1 2"), "one DescriptorSet and one Binding"),
            (
                _sub(
                    BASE_MODULE,
                    "OpDecorate %samplers Binding 1",
                    "OpDecorate %samplers Binding 1\nOpDecorate %samplers Binding 2",
                ),
                "one DescriptorSet and one Binding",
            ),
            (
                _sub(
                    BASE_MODULE,
                    "OpDecorate %samplers Binding 1",
                    "OpDecorate %samplers Binding 1\nOpDecorate %samplers NonUniformEXT",
                ),
                "unsupported sampler table declaration",
            ),
        ):
            with self.assertRaisesRegex(ValueError, expected_msg):
                mod.split_sampler_views(bad_dec)

    def test_contract_and_refusals(self) -> None:
        self._verify_module_contract(spirv_sampler_views)

    def test_fast_temp_copy_mutations(self) -> None:
        # These three mutations used to die on raw crashes (AttributeError on
        # None, int('³'), KeyError '%samplers') that the broad exception tuple
        # accepted without checking the behaviour. The contract now converts
        # those crashes into explicit assertion failures (see
        # _verify_module_contract), so for these three only an AssertionError
        # counts as a kill; a raw crash means the contract is not pinning the
        # behaviour it claims to pin.
        explicit_assertion_mutations = {
            "drop_non_str_guard",
            "drop_isascii_guard",
            "drop_register_end_safe_lookup",
        }
        source = MODULE_PATH.read_text(encoding="utf-8")
        for label, target, replacement in MUTATIONS:
            with self.subTest(mutation=label), TemporaryDirectory() as tmp:
                self.assertEqual(source.count(target), 1, target)
                mut_path = Path(tmp) / f"spirv_sampler_views_{label}.py"
                mut_path.write_text(source.replace(target, replacement, 1), encoding="utf-8")
                mod_name = f"spirv_sampler_views_mut_{label}"
                mod = _load_temp_module(mut_path, mod_name)
                expected = (
                    (AssertionError,)
                    if label in explicit_assertion_mutations
                    else (AssertionError, ValueError, KeyError, IndexError, AttributeError, TypeError)
                )
                try:
                    with self.assertRaises(
                        expected,
                        msg=f"mutation {label} escaped detection",
                    ):
                        self._verify_module_contract(mod)
                finally:
                    sys.modules.pop(mod_name, None)


if __name__ == "__main__":
    unittest.main()
