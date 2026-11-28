"""Remaining lowered VOP1/VOP2 SDWA profiles in direct compute."""
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import native_f16_alu_check as fixture
from native_direct_container_check import patched_pixel
from native_dpp_quad_check import request

DWORD, WORD_0, WORD_1 = 6, 4, 5
PAD, PRESERVE = 0, 2
MASK = 0xFFFFFFFF
OLD = 0xA55A1357
SDWA_REFUSAL = "unimplemented direct SDWA operand profile"
COMPUTE_PROLOGUE = [
    0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
    0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F,
]
OUTPUT_WORDS = 2064
ROW_WORDS = 32


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fvalue(bits):
    return struct.unpack("<f", struct.pack("<I", bits & MASK))[0]


def profile(name, opcode, unary, a, b=0, src0=DWORD, src1=None, dst=DWORD, unused=PAD):
    if src1 is None:
        src1 = 0 if unary else DWORD
    return {
        "name": name, "opcode": opcode, "unary": unary, "a": a, "b": b,
        "src0": src0, "src1": src1, "dst": dst, "unused": unused,
    }


PROFILES = (
    profile("v_sub_co_ci_u32", 0x29, False, 0xFFFE0003, 0x0004FFFD, WORD_0, WORD_1),
    profile("v_subrev_co_ci_u32", 0x2A, False, 0xFFFE0003, 0x0004FFFD, WORD_0, WORD_1),
    profile("v_mul_i32_i24", 0x09, False, 0xFFFE0003, 0x0004FFFD,
            WORD_1 | 8, WORD_0 | 8, WORD_1, PRESERVE),
    profile("v_mul_hi_i32_i24", 0x0A, False, 0xFF800001, 0x007FFFFE),
    profile("v_mul_hi_u32_u24", 0x0C, False, 0xFF800001, 0x007FFFFE),
    profile("v_ffbh_u32", 0x39, True, 0x1234FEDC, src0=WORD_1),
    profile("v_ffbl_b32", 0x3A, True, 0x1234FEDC, src0=WORD_0),
    profile("v_ffbh_i32", 0x3B, True, 0x00008001, src0=WORD_0 | 8),
    profile("v_sat_pk_u8_i16", 0x62, True, 0x0100FF00),
    profile("v_sin_f32", 0x35, True, fbits(0.25)),
    profile("v_cos_f32", 0x36, True, fbits(-0.25), src0=DWORD | 0x20),
    profile("v_mul_legacy_f32", 0x07, False, fbits(-1.5), fbits(2.0),
            src0=DWORD | 0x20, src1=DWORD | 0x10),
)


def sdwa_instruction(item, *, src0=None, src1=None, dst=None, unused=None):
    src0 = item["src0"] if src0 is None else src0
    src1 = item["src1"] if src1 is None else src1
    dst = item["dst"] if dst is None else dst
    unused = item["unused"] if unused is None else unused
    ext = 1 | (dst << 8) | (unused << 11) | (src0 << 16) | (src1 << 24)
    if item["unary"]:
        return fixture.vop1(item["opcode"], 4, 249) + [ext]
    return fixture.vop2(item["opcode"], 4, 249, 2) + [ext]


def selected(value, control):
    selector = control & 7
    shift = {0: 0, 1: 8, 2: 16, 3: 24, WORD_0: 0, WORD_1: 16, DWORD: 0}[selector]
    width = 8 if selector < 4 else 16 if selector in (WORD_0, WORD_1) else 32
    result = (value >> shift) & ((1 << width) - 1)
    if control & 8 and width < 32 and result & (1 << (width - 1)):
        result |= MASK ^ ((1 << width) - 1)
    if control & 0x20:
        result &= 0x7FFFFFFF
    if control & 0x10:
        result ^= 0x80000000
    return result & MASK


def signed24(value):
    value &= 0xFFFFFF
    return value - 0x1000000 if value & 0x800000 else value


def signed16(value):
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def first_high(value):
    return 32 - value.bit_length() if value else MASK


def expected(item):
    a = selected(item["a"], item["src0"])
    b = selected(item["b"], item["src1"])
    name = item["name"]
    if name == "v_sub_co_ci_u32":
        value = a - b
    elif name == "v_subrev_co_ci_u32":
        value = b - a
    elif name == "v_mul_i32_i24":
        value = signed24(a) * signed24(b)
    elif name == "v_mul_hi_i32_i24":
        value = (signed24(a) * signed24(b)) >> 32
    elif name == "v_mul_hi_u32_u24":
        value = ((a & 0xFFFFFF) * (b & 0xFFFFFF)) >> 32
    elif name == "v_ffbh_u32":
        value = first_high(a)
    elif name == "v_ffbl_b32":
        value = (a & -a).bit_length() - 1 if a else MASK
    elif name == "v_ffbh_i32":
        value = first_high(a ^ (MASK if a & 0x80000000 else 0))
    elif name == "v_sat_pk_u8_i16":
        lo = max(0, min(255, signed16(a)))
        hi = max(0, min(255, signed16(a >> 16)))
        value = lo | (hi << 8)
    elif name == "v_sin_f32":
        value = fbits(math.sin(2.0 * math.pi * fvalue(a)))
    elif name == "v_cos_f32":
        turns = fvalue(a)
        value = fbits(0.0) if turns == 0.25 else fbits(math.cos(2.0 * math.pi * turns))
    elif name == "v_mul_legacy_f32":
        value = fbits(fvalue(a) * fvalue(b))
    else:
        raise AssertionError(name)
    value &= MASK
    if item["dst"] == DWORD:
        return value
    shift = 16 if item["dst"] == WORD_1 else 0
    mask = 0xFFFF << shift
    other = OLD & ~mask if item["unused"] == PRESERVE else 0
    return other | ((value & 0xFFFF) << shift)


def emit_profiles(code):
    for slot, item in enumerate(PROFILES):
        code += fixture.mov(1, item["a"]) + fixture.mov(2, item["b"])
        code += fixture.mov(4, OLD)
        if item["name"] in ("v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            code += [fixture.sop1(3, 106, 128)]  # VCC input = zero
        code += sdwa_instruction(item)
        code += fixture.store(4 * slot, 4)
    return code + [0xBF810000]


def program():
    return emit_profiles(list(COMPUTE_PROLOGUE))


def ir_program():
    return emit_profiles(fixture.preamble()[:7])


def refused_program(instruction):
    return fixture.preamble()[:7] + instruction + [0xBF810000]


def refusal_cases():
    first = PROFILES[0]
    bad_source = sdwa_instruction(first, src0=7)
    bad_destination = sdwa_instruction(PROFILES[2], dst=7)
    unsupported = sdwa_instruction(profile("v_sqrt_f64", 0x44, True, 0))
    return (
        (bad_source, "invalid SDWA source modifier"),
        (bad_destination, "unsupported SDWA destination modifier"),
        (unsupported, SDWA_REFUSAL),
    )


def run_ir(compute_dumper, pixel_dumper):
    code = ir_program()
    with tempfile.TemporaryDirectory(prefix="x2w_sdwa_profiles_ir_") as tmp:
        work = Path(tmp)
        result = fixture.dump(compute_dumper, work, code, "profiles")
        assert result.returncode == 0, result.stderr
        ir = result.stdout
        for needle in ("sub i32", "mul i32", "unaryBits.i32", "@dx.op.unary.f32(i32 13,",
                       "@dx.op.unary.f32(i32 12,", "fmul float"):
            assert needle in ir, (needle, ir)
        assert ir.count("rawBufferStore.i32") >= len(PROFILES), ir.count("rawBufferStore.i32")

        for index, (instruction, why) in enumerate(refusal_cases()):
            result = fixture.dump(compute_dumper, work, refused_program(instruction), f"refused-{index}")
            assert result.returncode != 0 and why in result.stderr, (why, result.returncode, result.stderr)

        for item in PROFILES:
            graphics = work / (item["name"] + ".dxbc")
            graphics.write_bytes(patched_pixel(prefix=sdwa_instruction(item)))
            result = subprocess.run([pixel_dumper, graphics], capture_output=True, text=True)
            assert result.returncode != 0 and SDWA_REFUSAL in result.stderr, (
                item["name"], result.returncode, result.stderr)
    print(f"direct-compute SDWA IR: {len(PROFILES)} profiles admitted; "
          f"{len(refusal_cases())} malformed/unsupported controls and all graphics profiles refused")


def compile_request(compiler, sdk, work, root, code, label):
    marker = request(work, root, code)
    output = work / (label + ".cso")
    result = subprocess.run([compiler, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                            capture_output=True, text=True)
    return result, output


def make_root_signature(work, sdk):
    source, root = work / "root.hlsl", work / "root.bin"
    source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, source],
                   check=True)
    return root.read_bytes()


def run_dxil(compiler, sdk):
    with tempfile.TemporaryDirectory(prefix="x2w_sdwa_profiles_dxil_") as tmp:
        work = Path(tmp)
        root = make_root_signature(work, sdk)
        result, output = compile_request(compiler, sdk, work, root, program(), "profiles")
        assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
        dump = subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", output],
                              capture_output=True, text=True, check=True).stdout
        for needle in ("dx.op.unaryBits.i32", "dx.op.unary.f32", "fmul"):
            assert needle in dump, (needle, dump)
        for index, (instruction, why) in enumerate(refusal_cases()):
            result, output = compile_request(
                compiler, sdk, work, root, refused_program(instruction), f"refused-{index}")
            assert result.returncode != 0 and why in result.stderr, (why, result.stdout, result.stderr)
            assert not output.exists(), output
    print(f"direct-compute SDWA DXIL: {len(PROFILES)} profiles compiled; "
          f"{len(refusal_cases())} invalid/unsupported controls rejected")
    return 0


def same_result(got, want, item):
    if item["name"] in ("v_sin_f32", "v_cos_f32", "v_mul_legacy_f32"):
        return math.isclose(fvalue(got), fvalue(want), rel_tol=0.0, abs_tol=2e-6)
    return got == want


def run_hardware(replay, sdk):
    code = program()
    with tempfile.TemporaryDirectory(prefix="x2w_sdwa_profiles_gpu_") as tmp:
        work = Path(tmp)
        root = make_root_signature(work, sdk)
        marker = request(work, root, code)
        output = work / "readback.bin"
        result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
        if result.returncode == 77:
            return 77
        result.check_returncode()
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        wanted = [fixture.SENTINEL] * OUTPUT_WORDS
        expected_by_index = {}
        for lane in range(64):
            base = 8 + lane * ROW_WORDS
            for slot, item in enumerate(PROFILES):
                wanted[base + slot] = expected(item)
                expected_by_index[base + slot] = (wanted[base + slot], item)
        mismatches = []
        for index, (got, want) in enumerate(zip(actual, wanted)):
            if got != want and not (
                    index in expected_by_index and same_result(got, *expected_by_index[index])):
                mismatches.append((index, got, want))
        assert not mismatches, [(i, hex(g), hex(w)) for i, g, w in mismatches[:8]]
    print(f"direct-compute SDWA hardware: {len(PROFILES)} results and all guards matched")
    return 0


def main(argv):
    mode = argv[0]
    if mode == "ir" and len(argv) == 3:
        run_ir(argv[1], argv[2])
        return 0
    if mode == "dxil" and len(argv) == 3:
        return run_dxil(argv[1], argv[2])
    if mode == "hardware" and len(argv) == 3:
        return run_hardware(argv[1], argv[2])
    raise SystemExit("usage: native_sdwa_profiles_check.py ir COMPUTE_DUMPER PIXEL_DUMPER | "
                     "dxil COMPILE_DXIL SDK_BIN | hardware STORE_REPLAY SDK_BIN")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
