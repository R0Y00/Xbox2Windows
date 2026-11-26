"""Check s_clause and s_bitcmp{0,1}_b{32,64} lowering on synthetic gfx1013 machine code."""

import re
from pathlib import Path
import struct
import sys

from mesh_dynamic_geometry_check import CODE, MESH_LIMITS, SECOND

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import disasm_rdna2 as D  # noqa: E402
from mesh_dynamic_aot import enumerate_variants, lower as lower_dynamic  # noqa: E402
from mesh_entry_aot import lower as lower_static  # noqa: E402

U32 = 0xFFFFFFFF
STATIC_GEOMETRY = [0x1003, SECOND]
HALF_FLOAT_WORD = 0x3F000000


def decode(code):
    """Decode gfx1013 machine-code bytes via disasm_rdna2.decode_instruction in wave64 mode."""
    instructions = []
    pc = 0
    while pc < len(code):
        size, _, text, _ = D.decode_instruction(code, pc)
        assert size and pc + size <= len(code), (pc, size)
        op, _, operands = text.partition(" ")
        if op == "s_branch" or op.startswith("s_cbranch_"):
            target = re.fullmatch(r"label_(0x[0-9a-f]+)", operands)
            assert target, (pc, text)
            displacement = int(target[1], 16) - pc - size
            assert displacement % 4 == 0, (pc, text)
            text = f"{op} {displacement // 4}"
        elif text == "s_sendmsg 0x9":
            text = "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)"
        elif text == "s_barrier 0x0":
            text = "s_barrier"
        elif text.startswith("exp target_20 "):
            text = "exp prim " + operands.removeprefix("target_20 ")
        elif op in ("v_cmp_eq_u32_e64", "v_cmp_gt_u32_e64", "v_cndmask_b32_e64"):
            args = operands.split(", ")
            mask = 0 if op.startswith("v_cmp_") else -1
            if args[mask] == "vcc_lo":
                args[mask] = "vcc"
            else:
                match = re.fullmatch(r"s(\d+)", args[mask])
                assert match and int(match[1]) % 2 == 0, args[mask]
                low = int(match[1])
                args[mask] = f"s[{low}:{low + 1}]"
            text = op + " " + ", ".join(args)
        elif op == "s_buffer_load_dword" and operands.endswith(", 0x0"):
            text = text.removesuffix("0x0") + "null"
        instructions.append((pc, text))
        pc += size
    return instructions


def opcode(table, name):
    return next(op for op, info in table.items() if info[0] == name)


def ssrc(operand):
    if isinstance(operand, str):
        if operand.startswith("v") and operand[1:].isdigit():
            return 256 + int(operand[1:]), None
        if operand.startswith("s[") and operand.endswith("]"):
            return int(operand[2:-1].split(":")[0]), None
        if operand.startswith("s") and operand[1:].isdigit():
            return int(operand[1:]), None
        specials = {
            "vcc": 106,
            "m0": 124,
            "null": 125,
            "exec": 126,
            "0.5": 240,
            "1.0": 242,
        }
        return specials[operand], None
    if 0 <= operand <= 64:
        return 128 + operand, None
    if -16 <= operand < 0:
        return 192 - operand, None
    return 255, operand & U32


def sopp(name, simm=0):
    return [0xBF800000 | (opcode(D.SOPP_OPS, name) << 16) | (simm & 0xFFFF)]


def sop1(name, dst, src):
    src_code, lit = ssrc(src)
    dst_code, _ = ssrc(dst)
    out = [0xBE800000 | (dst_code << 16) | (opcode(D.SOP1_OPS, name) << 8) | src_code]
    if lit is not None:
        out.append(lit)
    return out


def sopc(name, a, b):
    sa, la = ssrc(a)
    sb, lb = ssrc(b)
    assert la is None or lb is None, (a, b)
    out = [0xBF000000 | (opcode(D.SOPC_OPS, name) << 16) | (sb << 8) | sa]
    lit = la if la is not None else lb
    if lit is not None:
        out.append(lit)
    return out


def vop1(name, dst, src):
    src_code, lit = ssrc(src)
    assert dst.startswith("v") and dst[1:].isdigit()
    out = [0x7E000000 | (int(dst[1:]) << 17) | (opcode(D.VOP1_OPS, name) << 9) | src_code]
    if lit is not None:
        out.append(lit)
    return out


def vop3_cmp(name, dst, a, b):
    sa, la = ssrc(a)
    sb, lb = ssrc(b)
    assert la is None and lb is None, (a, b)
    dst_code, _ = ssrc(dst)
    op = opcode(D.VOPC_OPS, name.replace("_e64", "_e32"))
    w0 = 0xD4000000 | (op << 16) | dst_code
    w1 = (sb << 9) | sa
    return [w0, w1]


def smem_load(name, dst, base, imm_offset):
    op = next(code for code, op_name in D.SMEM_OPS.items() if op_name == name)
    sdata, _ = ssrc(dst)
    sbase, _ = ssrc(base)
    assert sbase % 2 == 0
    w0 = 0xF4000000 | (op << 18) | (sdata << 6) | (sbase // 2)
    w1 = (125 << 25) | (imm_offset & 0x1FFFFF)
    return [w0, w1]


def find_pc(code, prefix):
    matches = [pc for pc, text in decode(code) if text.startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    return matches[0]


def insert_words(code, at_pc, inserted_words):
    """Insert machine-code dwords at `at_pc` and shift any SOPP branch that spans `at_pc`."""
    shift_dwords = len(inserted_words)
    out = bytearray()
    pc = 0
    while pc < len(code):
        if pc == at_pc:
            out.extend(struct.pack(f"<{shift_dwords}I", *inserted_words))
        size, op_name, _, _ = D.decode_instruction(code, pc)
        chunk = bytearray(code[pc : pc + size])
        if op_name == "s_branch" or op_name.startswith("s_cbranch"):
            w0, = struct.unpack_from("<I", chunk, 0)
            simm = w0 & 0xFFFF
            signed = simm - 0x10000 if simm & 0x8000 else simm
            target = pc + 4 + signed * 4
            if pc < at_pc <= target:
                signed += shift_dwords
                w0 = (w0 & 0xFFFF0000) | (signed & 0xFFFF)
                struct.pack_into("<I", chunk, 0, w0)
            elif target <= at_pc < pc:
                signed -= shift_dwords
                w0 = (w0 & 0xFFFF0000) | (signed & 0xFFFF)
                struct.pack_into("<I", chunk, 0, w0)
        out.extend(chunk)
        pc += size
    assert pc == len(code)
    return bytes(out)


def expect_refused(decoded, text, *, static_ok=False):
    if not static_ok:
        try:
            lower_static(decoded, STATIC_GEOMETRY, MESH_LIMITS)
        except ValueError as error:
            assert text in str(error), error
        else:
            raise AssertionError(f"lower_static accepted invalid case: {text}")
    try:
        enumerate_variants(decoded, 2, MESH_LIMITS)
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"enumerate_variants accepted invalid case: {text}")


def select(variants, geometry):
    matches = [
        variant
        for variant in variants
        if all((geometry[index] == value) == equal for index, value, equal in variant.conditions)
    ]
    assert len(matches) == 1, (geometry, matches)
    return matches[0]


def check_s_clause():
    pos0_pc = find_pc(CODE, "exp pos0 ")
    base_static = lower_static(decode(CODE), STATIC_GEOMETRY, MESH_LIMITS)
    base_dynamic = lower_dynamic(decode(CODE), 2, MESH_LIMITS)

    for simm in (0x0, 0x3F):
        patched = insert_words(CODE, pos0_pc, sopp("s_clause", simm))
        decoded = decode(patched)
        assert (pos0_pc, f"s_clause 0x{simm:x}") in decoded, decoded
        assert lower_static(decoded, STATIC_GEOMETRY, MESH_LIMITS) == base_static
        assert lower_dynamic(decoded, 2, MESH_LIMITS) == base_dynamic

    # Also verify s_clause inside a uniform Dyn join arm, and s_clause 0x40 refused inside one.
    point_load = [
        *smem_load("s_load_dwordx4", "s[36:39]", "s[10:11]", 0x60),
        *sopp("s_waitcnt", 0xC07F),
        *smem_load("s_buffer_load_dword", "s4", "s[36:39]", 0x4),
        *sopp("s_waitcnt", 0xC07F),
        *sopc("s_cmp_eq_u32", "s4", 0),
        *sopp("s_cbranch_scc0", 4),
        *sopp("s_clause", 0x1),
        *sop1("s_mov_b32", "s5", 1),
        *sopc("s_bitcmp1_b32", "s5", 0),
        *sopp("s_branch", 2),
        *sopp("s_clause", 0x2),
        *sop1("s_mov_b32", "s5", 1),
    ]
    join_code = insert_words(CODE, pos0_pc, point_load)
    join_decoded = decode(join_code)
    assert lower_static(join_decoded, STATIC_GEOMETRY, MESH_LIMITS) == base_static
    assert lower_dynamic(join_decoded, 2, MESH_LIMITS) == base_dynamic

    # Reading the merged Select predicate from s_bitcmp0_b32 after a divergent join must refuse cleanly.
    select_code = insert_words(
        CODE,
        pos0_pc,
        [
            *smem_load("s_load_dwordx4", "s[36:39]", "s[10:11]", 0x60),
            *sopp("s_waitcnt", 0xC07F),
            *smem_load("s_buffer_load_dword", "s4", "s[36:39]", 0x4),
            *sopp("s_waitcnt", 0xC07F),
            *sopc("s_cmp_eq_u32", "s4", 0),
            *sopp("s_cbranch_scc0", 2),
            *sop1("s_mov_b32", "s5", 1),
            *sopp("s_branch", 1),
            *sop1("s_mov_b32", "s5", 0),
            *sopc("s_bitcmp0_b32", "s5", 0),
        ],
    )
    expect_refused(decode(select_code), "Dynamic scalar predicate")

    for bad_simm in (0x40, 0xFFFF):
        bad_code = insert_words(CODE, pos0_pc, sopp("s_clause", bad_simm))
        bad_decoded = decode(bad_code)
        assert (pos0_pc, f"s_clause 0x{bad_simm:x}") in bad_decoded
        expect_refused(bad_decoded, "Invalid s_clause operand")

        bad_join = insert_words(
            CODE,
            pos0_pc,
            [
                *smem_load("s_load_dwordx4", "s[36:39]", "s[10:11]", 0x60),
                *sopp("s_waitcnt", 0xC07F),
                *smem_load("s_buffer_load_dword", "s4", "s[36:39]", 0x4),
                *sopp("s_waitcnt", 0xC07F),
                *sopc("s_cmp_eq_u32", "s4", 0),
                *sopp("s_cbranch_scc0", 2),
                *sopp("s_clause", bad_simm),
                *sopp("s_branch", 1),
                *sopp("s_nop", 0),
            ],
        )
        expect_refused(decode(bad_join), "Invalid s_clause operand")


def run_bitcmp_program(setup_words, cmp_op, src0, src1, branch_op):
    """Build a synthetic gfx1013 mesh program where the bitcmp branch toggles v3 = 0.5."""
    pos0_pc = find_pc(CODE, "exp pos0 ")
    skip_words = vop1("v_mov_b32_e32", "v3", "0.5")
    assert len(skip_words) == 1
    patch = [
        *setup_words,
        *sopc(cmp_op, src0, src1),
        *sopp(branch_op, len(skip_words)),
        *skip_words,
    ]
    code = insert_words(CODE, pos0_pc, patch)
    decoded = decode(code)
    assert any(op_text.startswith(f"{cmp_op} ") for _, op_text in decoded), decoded
    hlsl = lower_static(decoded, STATIC_GEOMETRY, MESH_LIMITS)
    variants = enumerate_variants(decoded, 2, MESH_LIMITS)
    tri_z = select(variants, (0x1003, SECOND)).positions[0][2]
    quad_z = select(variants, (0x2004, SECOND)).positions[0][2]
    assert tri_z in (0, HALF_FLOAT_WORD) and quad_z in (0, HALF_FLOAT_WORD)
    return hlsl, tri_z, quad_z


def check_bitcmp_32():
    # Each case pairs (cmp_op, value, bit, branch_op) with whether v3 becomes 0.5 (branch NOT taken).
    cases = [
        # Bit 0 (LSB) with SGPR value and inline bit index
        ("s_bitcmp1_b32", 0x00000001, 0, "s_cbranch_scc0", True),
        ("s_bitcmp1_b32", 0xFFFFFFFE, 0, "s_cbranch_scc0", False),
        ("s_bitcmp0_b32", 0xFFFFFFFE, 0, "s_cbranch_scc1", False),
        ("s_bitcmp0_b32", 0x00000001, 0, "s_cbranch_scc1", True),
        # Bit 31 (MSB) with SGPR value and SGPR bit index
        ("s_bitcmp1_b32", 0x80000000, 31, "s_cbranch_scc0", True),
        ("s_bitcmp1_b32", 0x7FFFFFFF, 31, "s_cbranch_scc0", False),
        ("s_bitcmp0_b32", 0x80000000, 31, "s_cbranch_scc1", True),
        ("s_bitcmp0_b32", 0x7FFFFFFF, 31, "s_cbranch_scc1", False),
        # Bit index wraps modulo 32: 33 & 31 == 1
        ("s_bitcmp1_b32", 0x00000002, 33, "s_cbranch_scc0", True),
        ("s_bitcmp1_b32", 0xFFFFFFFD, 33, "s_cbranch_scc0", False),
        ("s_bitcmp0_b32", 0x00000002, 33, "s_cbranch_scc0", False),
        ("s_bitcmp0_b32", 0xFFFFFFFD, 33, "s_cbranch_scc0", True),
    ]
    for cmp_op, value, bit, branch_op, expect_half in cases:
        # Test both SGPR bit index (s5) and inline/literal bit index.
        for bit_operand, extra_setup in (("s5", sop1("s_mov_b32", "s5", bit)), (bit, [])):
            setup = [*sop1("s_mov_b32", "s4", value), *extra_setup]
            hlsl, tri_z, quad_z = run_bitcmp_program(setup, cmp_op, "s4", bit_operand, branch_op)
            expected_z = HALF_FLOAT_WORD if expect_half else 0
            assert tri_z == expected_z and quad_z == expected_z, (
                cmp_op, hex(value), bit_operand, branch_op, tri_z, quad_z,
            )
            marker = "asfloat(0x3f000000u)" if expect_half else "asfloat(0x00000000u)"
            assert marker in hlsl, (cmp_op, hex(value), bit_operand, branch_op, hlsl)

    # Inline constant sources (-1, 0, 1) directly in src0 of s_bitcmp*_b32.
    inline_cases = [
        ("s_bitcmp1_b32", -1, 31, "s_cbranch_scc0", True),
        ("s_bitcmp0_b32", -1, 31, "s_cbranch_scc0", False),
        ("s_bitcmp0_b32", 0, 0, "s_cbranch_scc0", True),
        ("s_bitcmp1_b32", 1, 0, "s_cbranch_scc0", True),
    ]
    for cmp_op, src0, bit, branch_op, expect_half in inline_cases:
        _, tri_z, _ = run_bitcmp_program([], cmp_op, src0, bit, branch_op)
        assert tri_z == (HALF_FLOAT_WORD if expect_half else 0)


def check_bitcmp_64():
    # Low-32 bits (bit 5) and high-32 bits (bit 35 = bit 3 of s5, bit 63 = bit 31 of s5, wrap 65 & 63 == 1).
    cases = [
        ("s_bitcmp1_b64", 1 << 5, 0, 5, "s_cbranch_scc0", True),
        ("s_bitcmp1_b64", 0, 1 << 5, 5, "s_cbranch_scc0", False),
        ("s_bitcmp1_b64", 0, 1 << 3, 35, "s_cbranch_scc0", True),
        ("s_bitcmp1_b64", 1 << 3, 0, 35, "s_cbranch_scc0", False),
        ("s_bitcmp0_b64", 0, 1 << 3, 35, "s_cbranch_scc1", True),
        ("s_bitcmp0_b64", 1 << 3, 0, 35, "s_cbranch_scc1", False),
        ("s_bitcmp1_b64", 0, 0x80000000, 63, "s_cbranch_scc0", True),
        ("s_bitcmp1_b64", 0x80000000, 0, 63, "s_cbranch_scc0", False),
        ("s_bitcmp0_b64", 0, 0x80000000, 63, "s_cbranch_scc0", False),
        # Wrap modulo 64: 65 & 63 == 1 (tests literal src1 = 0x41 too).
        ("s_bitcmp1_b64", 0x00000002, 0, 65, "s_cbranch_scc0", True),
        ("s_bitcmp1_b64", 0xFFFFFFFD, 0xFFFFFFFF, 65, "s_cbranch_scc0", False),
    ]
    for cmp_op, lo, hi, bit, branch_op, expect_half in cases:
        for bit_operand, extra_setup in (("s6", sop1("s_mov_b32", "s6", bit)), (bit, [])):
            setup = [
                *sop1("s_mov_b32", "s4", lo),
                *sop1("s_mov_b32", "s5", hi),
                *extra_setup,
            ]
            hlsl, tri_z, quad_z = run_bitcmp_program(setup, cmp_op, "s[4:5]", bit_operand, branch_op)
            expected_z = HALF_FLOAT_WORD if expect_half else 0
            assert tri_z == expected_z and quad_z == expected_z, (
                cmp_op, hex(lo), hex(hi), bit_operand, branch_op, tri_z, quad_z,
            )
            marker = "asfloat(0x3f000000u)" if expect_half else "asfloat(0x00000000u)"
            assert marker in hlsl

    # Inline -1 sign-extends to 64 bits (0xffffffffffffffff): bit 63 is 1.
    _, tri_z, _ = run_bitcmp_program([], "s_bitcmp1_b64", -1, 63, "s_cbranch_scc0")
    assert tri_z == HALF_FLOAT_WORD
    _, tri_z, _ = run_bitcmp_program([], "s_bitcmp0_b64", -1, 63, "s_cbranch_scc0")
    assert tri_z == 0

    # EXEC at pos0 export has bits 0..2 set for triangle (0x1003) and bits 0..3 set for quad (0x2004).
    _, tri_z, quad_z = run_bitcmp_program([], "s_bitcmp1_b64", "exec", 3, "s_cbranch_scc0")
    assert tri_z == 0 and quad_z == HALF_FLOAT_WORD, (tri_z, quad_z)
    _, tri_z, quad_z = run_bitcmp_program([], "s_bitcmp0_b64", "exec", 3, "s_cbranch_scc0")
    assert tri_z == HALF_FLOAT_WORD and quad_z == 0, (tri_z, quad_z)

    # EXEC before v_cmpx_eq_u32_e32 has all 64 bits set, including bit 35.
    pre_exec_pc = find_pc(CODE, "v_cmpx_eq_u32_e32 0, v0")
    full_exec_code = insert_words(
        CODE,
        pre_exec_pc,
        [
            *sopc("s_bitcmp1_b64", "exec", 35),
            *sopp("s_cbranch_scc0", 1),
            *sop1("s_mov_b32", "s4", 1),
        ],
    )
    # Branch is NOT taken when bit 35 of EXEC is 1, so s4 is written; reading s4 before pos0 must succeed.
    pos0_after = find_pc(full_exec_code, "exp pos0 ")
    full_exec_code = insert_words(
        full_exec_code,
        pos0_after,
        [
            *sopc("s_bitcmp1_b32", "s4", 0),
            *sopp("s_cbranch_scc0", 1),
            *vop1("v_mov_b32_e32", "v3", "0.5"),
        ],
    )
    assert "asfloat(0x3f000000u)" in lower_static(decode(full_exec_code), STATIC_GEOMETRY, MESH_LIMITS)

    # VCC written by v_cmp_eq_u32_e64: lane 1 has v0 == 1, so bit 1 of vcc is 1 and bit 2 is 0.
    pos0_pc = find_pc(CODE, "exp pos0 ")
    vcc_setup = vop3_cmp("v_cmp_eq_u32_e64", "vcc", 1, "v0")
    vcc_code = insert_words(
        CODE,
        pos0_pc,
        [
            *vcc_setup,
            *sopc("s_bitcmp1_b64", "vcc", 1),
            *sopp("s_cbranch_scc0", 1),
            *vop1("v_mov_b32_e32", "v3", "0.5"),
        ],
    )
    vcc_decoded = decode(vcc_code)
    assert (pos0_pc + 8, "s_bitcmp1_b64 vcc, 1") in vcc_decoded, vcc_decoded
    assert "asfloat(0x3f000000u)" in lower_static(vcc_decoded, STATIC_GEOMETRY, MESH_LIMITS)
    assert select(enumerate_variants(vcc_decoded, 2, MESH_LIMITS), (0x1003, SECOND)).positions[0][2] == HALF_FLOAT_WORD

    # Negative control for VCC: bit 2 is 0, so s_bitcmp1_b64 vcc, 2 takes the branch and keeps v3 == 0.
    vcc_zero_code = insert_words(
        CODE,
        pos0_pc,
        [
            *vcc_setup,
            *sopc("s_bitcmp1_b64", "vcc", 2),
            *sopp("s_cbranch_scc0", 1),
            *vop1("v_mov_b32_e32", "v3", "0.5"),
        ],
    )
    vcc_zero_decoded = decode(vcc_zero_code)
    assert "asfloat(0x00000000u)" in lower_static(vcc_zero_decoded, STATIC_GEOMETRY, MESH_LIMITS)
    assert select(enumerate_variants(vcc_zero_decoded, 2, MESH_LIMITS), (0x1003, SECOND)).positions[0][2] == 0

    # High-32 bits of VCC: before v_cmpx_eq_u32_e32, all 64 lanes are live, so v_cmp_gt_u32_e64 vcc, v0, 34
    # sets bits 35..63 on wave 0 (and all 64 bits on wave 1). Bit 35 is 1 while bit 34 is 0 on wave 0.
    vcc_hi_code = insert_words(
        CODE,
        pre_exec_pc,
        [
            *vop3_cmp("v_cmp_gt_u32_e64", "vcc", "v0", 34),
            *sopc("s_bitcmp1_b64", "vcc", 35),
            *sopp("s_cbranch_scc0", 1),
            *sop1("s_mov_b32", "s4", 1),
        ],
    )
    vcc_hi_pos0 = find_pc(vcc_hi_code, "exp pos0 ")
    vcc_hi_code = insert_words(
        vcc_hi_code,
        vcc_hi_pos0,
        [
            *sopc("s_bitcmp0_b64", "vcc", 34),
            *sopp("s_cbranch_scc0", 2),
            *sopc("s_bitcmp1_b32", "s4", 0),
            *sopp("s_cbranch_scc0", 1),
            *vop1("v_mov_b32_e32", "v3", "0.5"),
        ],
    )
    vcc_hi_decoded = decode(vcc_hi_code)
    assert "asfloat(0x3f000000u)" in lower_static(vcc_hi_decoded, STATIC_GEOMETRY, MESH_LIMITS)
    assert select(enumerate_variants(vcc_hi_decoded, 2, MESH_LIMITS), (0x1003, SECOND)).positions[0][2] == HALF_FLOAT_WORD


def check_refusals():
    # 1. UNDEF operand (written only inside wave-0-only s_cbranch_execz range) -> scc = None.
    # Dead bitcmp leaves scc = None without refusing; branching on scc is refused.
    alloc_pc = find_pc(CODE, "s_mov_b32 m0, s15")
    pos0_pc = find_pc(CODE, "exp pos0 ")
    skipped_code = insert_words(CODE, alloc_pc, [*sop1("s_mov_b32", "s32", 1), *sop1("s_mov_b32", "s33", 2)])
    skipped_pos0 = find_pc(skipped_code, "exp pos0 ")

    dead_32 = insert_words(skipped_code, skipped_pos0, sopc("s_bitcmp0_b32", "s32", 0))
    assert lower_static(decode(dead_32), STATIC_GEOMETRY, MESH_LIMITS)
    assert enumerate_variants(decode(dead_32), 2, MESH_LIMITS)

    dead_64 = insert_words(skipped_code, skipped_pos0, sopc("s_bitcmp1_b64", "s[32:33]", 35))
    assert lower_static(decode(dead_64), STATIC_GEOMETRY, MESH_LIMITS)

    # Make only s32 defined on both waves while s33 stays UNDEF on wave 1 -> pair is still UNDEF.
    partial_pair = insert_words(
        insert_words(CODE, alloc_pc, sop1("s_mov_b32", "s33", 2)),
        pos0_pc + 4,
        [*sop1("s_mov_b32", "s32", 1), *sopc("s_bitcmp1_b64", "s[32:33]", 0), *sopp("s_cbranch_scc0", 0)],
    )
    expect_refused(decode(partial_pair), "Undefined branch predicate")

    for cmp_op, src0, bit, branch_op in (
        ("s_bitcmp0_b32", "s32", 0, "s_cbranch_scc0"),
        ("s_bitcmp1_b32", 1, "s32", "s_cbranch_scc1"),
        ("s_bitcmp0_b64", "s[32:33]", 35, "s_cbranch_scc0"),
        ("s_bitcmp1_b64", "s[32:33]", "s32", "s_cbranch_scc1"),
    ):
        branch_undef = insert_words(
            skipped_code,
            skipped_pos0,
            [*sopc(cmp_op, src0, bit), *sopp(branch_op, 0)],
        )
        expect_refused(decode(branch_undef), "Undefined branch predicate")

    # 2. RuntimeWord operands in mesh_dynamic_aot.enumerate_variants must be refused cleanly.
    cmp_entry_pc = find_pc(CODE, "s_cmp_eq_u32 s15, 0x1003")
    for words_patch in (
        sopc("s_bitcmp0_b32", "s15", 0),
        sopc("s_bitcmp1_b32", "s14", 31),
        sopc("s_bitcmp0_b64", "s[14:15]", 35),
        [*sop1("s_mov_b32", "s4", 1), *sopc("s_bitcmp1_b32", "s4", "s14")],
    ):
        runtime_code = insert_words(CODE, cmp_entry_pc, words_patch)
        expect_refused(decode(runtime_code), "Unproven runtime geometry word", static_ok=True)

    # 3. Dynamic constant-buffer scalar string operand -> ValueError("Dynamic scalar predicate" / pair concrete).
    for patch_tail, expected_msg in (
        ([*sopc("s_bitcmp0_b32", "s4", 0)], "Dynamic scalar predicate"),
        ([*sopc("s_bitcmp1_b32", 1, "s4")], "Dynamic scalar predicate"),
        ([*sop1("s_mov_b32", "s5", 0), *sopc("s_bitcmp0_b64", "s[4:5]", 35)], "Scalar pair requires concrete words"),
        ([*sop1("s_mov_b32", "s6", 0), *sop1("s_mov_b32", "s7", 0), *sopc("s_bitcmp1_b64", "s[6:7]", "s4")], "Dynamic scalar predicate"),
    ):
        dyn_code = insert_words(
            CODE,
            pos0_pc,
            [
                *smem_load("s_load_dwordx4", "s[36:39]", "s[10:11]", 0x60),
                *sopp("s_waitcnt", 0xC07F),
                *smem_load("s_buffer_load_dword", "s4", "s[36:39]", 0x4),
                *sopp("s_waitcnt", 0xC07F),
                *patch_tail,
            ],
        )
        expect_refused(decode(dyn_code), expected_msg)

    # 4. Undefined VCC -> ValueError("Undefined wave predicate").
    undef_vcc = insert_words(CODE, pos0_pc, sopc("s_bitcmp0_b64", "vcc", 0))
    expect_refused(decode(undef_vcc), "Undefined wave predicate")


def check():
    check_s_clause()
    check_bitcmp_32()
    check_bitcmp_64()
    check_refusals()
    print("s_clause and s_bitcmp{0,1}_b{32,64} machine-code checks passed")


if __name__ == "__main__":
    check()
