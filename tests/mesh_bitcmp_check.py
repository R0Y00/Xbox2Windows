"""Check s_bitcmp0/1_b32 and s_bitcmp0/1_b64 in the bounded mesh lowerers (#2413)."""

from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))

from disasm_rdna2 import decode_instruction
from mesh_dynamic_aot import Geometry, RuntimeWave, enumerate_variants
from mesh_dynamic_aot import lower as lower_dynamic
from mesh_dynamic_aot_check import insert_scalar
from mesh_dynamic_geometry_check import CODE, MESH_LIMITS, SECOND, decode
from mesh_entry_aot import UNDEF, U32, WAVE, Wave
from mesh_entry_aot import lower as lower_static
from mesh_entry_check import PROGRAM, TRIANGLE, draw, enumerate_pcs


SOPC_CASES = (
    ("s_bitcmp0_b32 s15, 0", bytes.fromhex("0f800cbf")),
    ("s_bitcmp1_b32 s15, 12", bytes.fromhex("0f8c0dbf")),
    ("s_bitcmp0_b64 exec, 37", bytes.fromhex("7ea50ebf")),
    ("s_bitcmp1_b64 s[4:5], s6", bytes.fromhex("04060fbf")),
    ("s_bitcmp0_b32 s15, 0xffffffe5", bytes.fromhex("0fff0cbfe5ffffff")),
    ("s_bitcmp1_b64 s[4:5], 0x45", bytes.fromhex("04ff0fbf45000000")),
)

WORDS_32 = (0x00000000, 0xFFFFFFFF, 0x80000001, 0x00010020, 0x55555555, 0xAAAAAAAA, 0x0000001F, 0x00000020)
INDEXES_32 = (0, 1, 5, 16, 31, 32, 33, 37, 0xFFFFFFE5)
WORDS_64 = (
    0x00000000_00000000,
    0xFFFFFFFF_FFFFFFFF,
    0x00000001_00000000,
    0x80000020_00010001,
    0xAAAAAAAA_55555555,
    0x00000000_00000001,
)
INDEXES_64 = (0, 1, 31, 32, 37, 63, 64, 65, 101, 0xFFFFFFC5)

POINT, MATRIX = "0x60", "0x80"


def expect_error(fn, message):
    try:
        fn()
    except ValueError as error:
        assert message in str(error), f"Expected {message!r}, got {error!r}"
        return
    raise AssertionError(f"Expected refusal: {message}")


def check_wave_model():
    for op, want in (("s_bitcmp0_b32", 0), ("s_bitcmp1_b32", 1)):
        for word in WORDS_32:
            for idx in INDEXES_32:
                expected = (((word & U32) >> (idx & 31)) & 1) == want
                wave = Wave(0, [])
                wave.s[4] = word
                wave.s[6] = idx
                wave.written.update((4, 6))
                wave.scalar_op(op, ["s4", str(idx) if 0 <= idx <= 64 else hex(idx)])
                assert wave.scc is expected and wave.s[4] == word
                wave.scc = None
                wave.scalar_op(op, ["s4", "s6"])
                assert wave.scc is expected and wave.s[4] == word and wave.s[6] == idx

    for op, want in (("s_bitcmp0_b64", 0), ("s_bitcmp1_b64", 1)):
        for word in WORDS_64:
            for idx in INDEXES_64:
                expected = (((word & WAVE) >> (idx & 63)) & 1) == want
                wave = Wave(0, [])
                wave.s[4] = word & U32
                wave.s[5] = (word >> 32) & U32
                wave.s[6] = idx
                wave.exec = word
                wave.vcc = word ^ WAVE
                wave.written.update((4, 5, 6))
                wave.scalar_op(op, ["s[4:5]", "s6"])
                assert wave.scc is expected
                wave.scalar_op(op, ["exec", "s6"])
                assert wave.scc is expected
                wave.scalar_op(op, ["vcc", "s6"])
                assert wave.scc is (not expected)
                if 0 <= idx <= 64:
                    wave.scalar_op(op, ["s[4:5]", str(idx)])
                    assert wave.scc is expected

        # Inline -1 sign-extends to all 64 bits.
        wave = Wave(0, [])
        wave.scalar_op(op, ["-1", "37"])
        assert wave.scc is (want == 1)


def check_decode_and_specialization():
    for expected_asm, raw in SOPC_CASES:
        size, op, asm, _ = decode_instruction(raw, 0)
        assert (size, asm) == (len(raw), expected_asm), (size, op, asm)

    # Guard wave 0 in static PROGRAM using s_bitcmp0_b32 s3, 24 and 64-bit high-bit s_bitcmp1_b64 exec, 37.
    base_decoded = enumerate_pcs(PROGRAM)
    assert lower_static(concrete_wave_program(), [0x1003], (3, 1)) == lower_static(base_decoded, [0x1003], (3, 1))

    # Flipping s_bitcmp0_b32 -> s_bitcmp1_b32 makes wave 1 run alloc instead of wave 0 and fails Unverified allocation.
    wrong = [
        (pc, line.replace("s_bitcmp0_b32 s3, 24", "s_bitcmp1_b32 s3, 24"))
        for pc, line in concrete_wave_program()
    ]
    expect_error(lambda: lower_static(wrong, [0x1003], (3, 1)), "Unverified allocation")

    # Also prove 64-bit exec high bit test on entry EXEC = 0xFFFFFFFFFFFFFFFF.
    exec_guarded = enumerate_pcs(
        PROGRAM[:2] + ["s_bitcmp1_b64 exec, 37", "s_cbranch_scc0 0"] + PROGRAM[2:]
    )
    assert lower_static(exec_guarded, [0x1003], (3, 1)) == lower_static(base_decoded, [0x1003], (3, 1))

    # Splice a concrete bit-compare sequence into CODE after s_sendmsg (outside the skip branch).
    body = decode(CODE)
    after_alloc = next(pc for pc, op in body if op.startswith("s_sendmsg "))
    extra = decode(bytes.fromhex("0f8c0dbf") + bytes.fromhex("7ea50ebf"))
    assert [line for _, line in extra] == ["s_bitcmp1_b32 s15, 12", "s_bitcmp0_b64 exec, 37"]
    spliced = insert_scalar(body, after_alloc, ["s_bitcmp1_b32 s15, 12", "s_bitcmp0_b64 exec, 37"])
    # Adding a pure compare before an instruction that doesn't read SCC preserves dynamic variants and HLSL.
    assert enumerate_variants(spliced, 2, MESH_LIMITS) == enumerate_variants(body, 2, MESH_LIMITS)
    assert lower_dynamic(spliced, 2, MESH_LIMITS) == lower_dynamic(body, 2, MESH_LIMITS)


def concrete_wave_program():
    # Replace the inner allocation guard (`s_cbranch_execz 1` at PROGRAM[19]) with s_bitcmp0_b32 s3, 24
    # plus a 64-bit high-word bit test s_bitcmp1_b64 s[4:5], 37.
    idx = PROGRAM.index("s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)")
    assert PROGRAM[idx - 1] == "s_cbranch_execz 1"
    # Update outer s_cbranch_execz 2 (at idx - 4) to jump over our 4 instructions + s_sendmsg? No: outer branch
    # is before s_mov s7/m0, so keep its skip target at idx - 1 and guard s_sendmsg cleanly.
    lines = (
        PROGRAM[: idx - 1]
        + [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0x20",
            "s_bitcmp1_b64 s[4:5], 37",
            "s_cbranch_scc0 3",
            "s_bitcmp0_b32 s3, 24",
            "s_cbranch_scc0 1",
        ]
        + PROGRAM[idx:]
    )
    # Fix outer `s_cbranch_execz 2` (right before `s_mov_b32 s7, s15`) so wave 1 skips s_mov s7/m0 -> hits s_mov s4.
    assert lines[idx - 4] == "s_cbranch_execz 2"
    return enumerate_pcs(lines)


def join_program(bit_compare, branch_op, dynamic_bit=False):
    """Splice a dynamic bit-compare + uniform if/else choosing POINT vs MATRIX into PROGRAM."""
    table_arm = lambda off: [f"s_load_dwordx4 s[36:39], s[10:11], {off}", "s_waitcnt lgkmcnt(0)"]
    arm_true = table_arm(POINT)
    arm_false = table_arm(MATRIX)
    first = arm_true + [f"s_branch {len(arm_false)}"]
    loads = [
        "s_buffer_load_dword s14, s[16:19], null",
        *(["s_buffer_load_dword s13, s[16:19], 0x8"] if dynamic_bit else []),
        "s_waitcnt lgkmcnt(0)",
        bit_compare,
        f"{branch_op} {len(first)}",
    ]
    lines = (
        PROGRAM[:7]
        + loads
        + first
        + arm_false
        + [
            "s_buffer_load_dword s30, s[36:39], 0x4",
            "s_buffer_load_dword s31, s[20:23], 0xc",
            "s_waitcnt lgkmcnt(0)",
        ]
        + PROGRAM[10:]
    )
    return enumerate_pcs(lines)


def check_dynamic_joins():
    hlsl0 = lower_static(join_program("s_bitcmp0_b32 s14, 5", "s_cbranch_scc0"), [0x1003], (3, 1))
    assert "((asuint(pointData[0].x) & (1u << 5u)) == 0u)" in hlsl0, hlsl0

    hlsl1 = lower_static(join_program("s_bitcmp1_b32 s14, 5", "s_cbranch_scc1"), [0x1003], (3, 1))
    assert "((asuint(pointData[0].x) & (1u << 5u)) != 0u) ? matrixData[0].y : pointData[0].y" in hlsl1, hlsl1

    hlsl_dyn = lower_static(join_program("s_bitcmp1_b32 s14, s13", "s_cbranch_scc1", dynamic_bit=True),
                            [0x1003], (3, 1))
    assert "((asuint(pointData[0].x) & (1u << (asuint(pointData[0].z) & 31u))) != 0u) ? matrixData[0].y : pointData[0].y" in hlsl_dyn, hlsl_dyn


def check_refusals():
    # Unwritten SGPR must not be hidden by steps()'s written scan.
    expect_error(
        lambda: lower_static(enumerate_pcs(["s_bitcmp0_b32 s99, 0", "s_endpgm"]), [0x1003], (3, 1)),
        "Scalar register s99 used before it is written",
    )

    # Skipped-range UNDEF scalar leaves SCC None; a dead compare passes, a branch consuming it refuses.
    body = decode(CODE)
    skipped = insert_scalar(body, 160, ["s_mov_b32 s32, 1"], skip_pc=156)
    dead = insert_scalar(skipped, 172, ["s_bitcmp0_b32 s32, 5"])
    lower_dynamic(dead, 2, MESH_LIMITS)
    consumed = insert_scalar(skipped, 172, ["s_bitcmp0_b32 s32, 5", "s_cbranch_scc0 0"])
    expect_error(lambda: lower_dynamic(consumed, 2, MESH_LIMITS), "Undefined branch predicate")

    # Pending scalar load before s_waitcnt lgkmcnt(0).
    expect_error(
        lambda: lower_static(
            enumerate_pcs(PROGRAM[:7] + ["s_bitcmp0_b32 s15, 0", "s_endpgm"]),
            [0x1003],
            (3, 1),
        ),
        "Scalar load used before wait",
    )

    # Width and operand-shape refusals.
    for line, msg in (
        ("s_bitcmp0_b64 s4, 37", "Expected a 64-bit scalar operand"),
        ("s_bitcmp0_b64 exec_lo, 5", "Expected a 64-bit scalar operand"),
        ("s_bitcmp0_b32 s[4:5], 5", "Expected a 32-bit scalar operand"),
        ("s_bitcmp0_b32 exec, 5", "Expected a 32-bit scalar operand"),
        ("s_bitcmp0_b32 s4, s[6:7]", "Expected a 32-bit bit-index operand"),
        ("s_bitcmp0_b64 s[4:5], exec", "Expected a 32-bit bit-index operand"),
        ("s_bitcmp0_b32 s4", "Expected two operands"),
    ):
        wave = Wave(0, [])
        wave.s.update({4: 1, 5: 2, 6: 3, 7: 4})
        op, _, tail = line.partition(" ")
        expect_error(lambda w=wave, o=op, a=tail.split(", "): w.scalar_op(o, a), msg)

    # 64-bit dynamic string operand refused.
    wave = Wave(0, [])
    wave.s.update({4: "pointData[0].x", 5: 0})
    expect_error(lambda: wave.scalar_op("s_bitcmp0_b64", ["s[4:5]", "5"]), "Scalar pair requires concrete words")

    # Unproven RuntimeWord in RuntimeWave refused.
    rt = RuntimeWave(0, Geometry(2, ()))
    rt.s[15] = rt.geometry[0]
    expect_error(lambda: rt.scalar_op("s_bitcmp0_b32", ["s15", "0"]), "Unproven runtime geometry word")

    # Sampler-heap base/word in either operand refused in RuntimeWave.
    rt_heap = RuntimeWave(0, Geometry(2, (), sampler_heap_sgpr=28))
    rt_heap.s[29] = 0
    rt_heap.descriptor_scalar_op("s_load_dwordx4", ["s[36:39]", "s[28:29]", "0x10"])
    rt_heap.pending.clear()
    expect_error(lambda: rt_heap.scalar_op("s_bitcmp0_b32", ["s28", "0"]), "Sampler heap")
    expect_error(lambda: rt_heap.scalar_op("s_bitcmp1_b32", ["s36", "0"]), "Sampler heap")
    rt_heap.s[40] = 1
    expect_error(lambda: rt_heap.scalar_op("s_bitcmp1_b32", ["s40", "s36"]), "Sampler heap")


def check_controls():
    check_wave_model()
    check_decode_and_specialization()
    check_dynamic_joins()
    check_refusals()
    print("mesh s_bitcmp controls passed")


def check_binary(llvm_mc):
    source = "\n".join(asm for asm, _ in SOPC_CASES) + "\n"
    run = subprocess.run(
        [llvm_mc, "-triple=amdgcn--", "-mcpu=gfx1013", "-mattr=-wavefrontsize32,+wavefrontsize64", "-show-encoding"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    )
    encoded = bytearray()
    for line in run.stdout.splitlines():
        m = re.search(r"encoding: \[([^\]]*)\]", line)
        if m:
            encoded += bytes(int(b, 16) for b in m.group(1).split(","))
    expected = b"".join(raw for _, raw in SOPC_CASES)
    assert bytes(encoded) == expected, f"encoded {encoded.hex()} != {expected.hex()}"
    pc = 0
    for expected_asm, raw in SOPC_CASES:
        size, _, asm, _ = decode_instruction(encoded, pc)
        assert (size, asm) == (len(raw), expected_asm), (size, asm, expected_asm)
        pc += size
    print("mesh s_bitcmp binary encodings match llvm-mc")


def constants(selector_bits, depth_point, depth_matrix, matrix_w, bit_index=0):
    data = bytearray(1024)
    struct.pack_into("<IfI", data, 0, selector_bits, depth_point, bit_index)
    struct.pack_into("<f", data, 512 + 4, depth_matrix)
    struct.pack_into("<f", data, 512 + 12, matrix_w)
    return data


def join_cases():
    def triangle(z):
        return [(x, y, z, w) for x, y, _, w in TRIANGLE]

    # Bit 5 == 0 with non-zero low bits (0x1f) selects POINT (-0.625 + 0.375 = -0.25 -> clipped);
    # Bit 5 == 1 (0x20) selects MATRIX (0.25 + 0.375 = 0.625 -> visible).
    # Dynamic bit index 37 tests (37 & 31) == 5.
    return [
        (constants(0x0000001F, -0.625, 0.25, 0.375, bit_index=37), triangle(-0.25), False),
        (constants(0x00000020, -0.625, 0.25, 0.375, bit_index=37), triangle(0.625), True),
    ]


def check_draw(dxc, renderer, adapter):
    configs = (
        lower_static(concrete_wave_program(), [0x1003], (3, 1)),
        lower_static(join_program("s_bitcmp0_b32 s14, 5", "s_cbranch_scc0"), [0x1003], (3, 1)),
        lower_static(join_program("s_bitcmp1_b32 s14, 5", "s_cbranch_scc1"), [0x1003], (3, 1)),
        lower_static(join_program("s_bitcmp1_b32 s14, s13", "s_cbranch_scc1", dynamic_bit=True), [0x1003], (3, 1)),
    )
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        rc = draw(dxc, renderer, adapter, tmp, configs[0], [(constants(0, 0.125, 0.25, 0.375),
                  [(x, y, 0.5, w) for x, y, _, w in TRIANGLE], True)])
        if rc == 77:
            return 77
        cases = join_cases()
        for source in configs[1:]:
            rc = draw(dxc, renderer, adapter, tmp, source, cases)
            if rc == 77:
                return 77

        # Negative control: invert s_bitcmp0_b32 -> s_bitcmp1_b32 so arms flip and draw mismatches.
        inverted = lower_static(join_program("s_bitcmp1_b32 s14, 5", "s_cbranch_scc0"), [0x1003], (3, 1))
        bad_dir = tmp / "neg"
        bad_dir.mkdir()
        try:
            neg_rc = draw(dxc, renderer, adapter, bad_dir, inverted, cases)
        except subprocess.CalledProcessError:
            neg_rc = 1
        assert neg_rc != 0, "Inverted s_bitcmp passed GPU draw"
    print("mesh s_bitcmp draw: ok")
    return 0


def main():
    if len(sys.argv) == 1:
        check_controls()
        return 0
    if sys.argv[1] == "binary" and len(sys.argv) == 3:
        check_controls()
        check_binary(sys.argv[2])
        return 0
    if sys.argv[1] == "draw" and len(sys.argv) == 5:
        check_controls()
        return check_draw(*sys.argv[2:])
    raise SystemExit("Usage: mesh_bitcmp_check.py [binary <llvm-mc> | draw <dxc> <offscreen_check> <adapter>]")


if __name__ == "__main__":
    sys.exit(main())
