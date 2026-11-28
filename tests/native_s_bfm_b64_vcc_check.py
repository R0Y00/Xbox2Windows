"""Compile and replay s_bfm_b64 with VCC as the destination in both wave models."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_mirror_check import vop2
from native_scalar_lds_check import scalar, sop1, vmove
from native_wave64_exec_salu_check import (
    END, EXEC, GROUP_STRIDE_WORDS, GUARD_WORDS, NEG_ONE, OUTPUT_WORDS,
    SENTINEL, VCC, ZERO, compile_request, request,
)


BFM_B64 = 0x25
MOV_B32, MOV_B64, LITERAL = 3, 4, 255
M0, SCC_OUT = 124, 26
MASK32, MASK64 = (1 << 32) - 1, (1 << 64) - 1
PLANE_WORDS = 2 * GROUP_STRIDE_WORDS
# (width, offset, SCC before the bitfield insert). Width 64 wraps to an
# empty mask through the low-six-bits masking, as in native_salu64.
CASES = ((0, 0, 1), (16, 0, 0), (32, 32, 1), (20, 24, 0), (64, 0, 1))


def set_scc(value):
    # s_cmp_eq_u32 0, 0 (SCC = 1) or s_cmp_eq_u32 0, 1 (SCC = 0).
    return 0xBF060000 | (ZERO + (1 - value)) << 8 | ZERO


def store(vgpr, plane):
    return [0xE0701000 | (4 * PLANE_WORDS * plane), 0x80080008 | vgpr << 8]


def program(wave_size, width, offset, scc):
    # Prepare every lane's address before narrowing EXEC. The shader uses
    # the same two-group UAV layout as the wave64 EXEC SALU fixture.
    if wave_size == 64:
        words = [sop1(MOV_B64, EXEC, NEG_ONE)]
    else:
        # Explicitly enable all 32 lanes and seed the tracked upper word.
        words = [sop1(3, EXEC, NEG_ONE), sop1(3, EXEC + 1, ZERO + 1)]
    words += [
        sop1(3, 20, 4), sop1(3, 21, ZERO),
        0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
        scalar(0x1E, 7, 5, ZERO + 9),  # s_lshl_b32 s7, s5, 9
        0x34100082,  # v_lshlrev_b32 v8, 2, v0
        0x4A101007,  # v_add_nc_u32 v8, s7, v8
    ]
    # EXEC is never narrowed here (unlike the s_not_b64 template, which
    # narrows it to a mask), so there is nothing to restore before readback:
    # in wave64 it is already all ones, and in wave32 widening EXEC_HI from
    # the seeded 1 to all ones would only enable lanes that do not exist.
    words += [
        set_scc(scc),
        scalar(BFM_B64, VCC, ZERO + width, ZERO + offset),
        vmove(1, 129),  # v_mov_b32 v1, 1
        vop2(1, 9, 128, 1),  # v_cndmask_b32 v9, 0, v1, vcc
        scalar(0x0A, SCC_OUT, ZERO + 1, ZERO),  # s_cselect_b32 s26, 1, 0
    ]
    # Plane 0 holds the v_cndmask_b32 result: a fourth plane would need a
    # 4096-byte immediate offset, past the 12-bit field of the store word.
    # Store v9 directly: vmove(1, 9) would encode v_mov_b32 v1, s9 (an
    # unbound SGPR), not VGPR v9. Only SGPR sources go through vmove.
    words += [*store(9, 0)]
    for sgpr, plane in ((VCC, 1), (VCC + 1, 2), (SCC_OUT, 3)):
        words += [vmove(1, sgpr), *store(1, plane)]
    return words + [END]


def expected_words(wave_size, width, offset, scc):
    bits = ((1 << (width & 63)) - 1) << (offset & 63)
    expected = [SENTINEL] * OUTPUT_WORDS
    for group in range(2):
        for lane in range(wave_size):
            index = GUARD_WORDS + group * GROUP_STRIDE_WORDS + lane
            expected[index] = (bits >> lane) & 1
            expected[index + PLANE_WORDS] = bits & MASK32
            expected[index + 2 * PLANE_WORDS] = (bits >> 32) & MASK32
            expected[index + 3 * PLANE_WORDS] = scc
    return expected


def controls(cli, sdk, work, root):
    cache = work / "cache"
    failures = []
    for wave_size in (32, 64):
        for width, offset, scc in CASES:
            marker = request(work, root, wave_size, wave_size,
                             program(wave_size, width, offset, scc))
            output, result = compile_request(cli, sdk, cache, marker)
            if result.returncode or not output.is_file():
                failures.append((wave_size, width, offset, scc,
                                 result.stdout, result.stderr))
    assert not failures, failures

    prefix = [sop1(MOV_B32, 23, ZERO), sop1(MOV_B32, M0, ZERO), set_scc(1)]
    refusals = [
        (scalar(BFM_B64, M0, ZERO + 16, ZERO), "invalid s_bfm_b64 destination"),
        (scalar(BFM_B64, 23, ZERO + 16, ZERO), "invalid s_bfm_b64 destination"),
        (scalar(BFM_B64, 24, VCC, ZERO), "invalid s_bfm_b64 source"),
        (scalar(BFM_B64, 24, ZERO, M0), "invalid s_bfm_b64 source"),
    ]
    for instruction, error in refusals:
        marker = request(work, root, 64, 64, prefix + [instruction, END])
        output, result = compile_request(cli, sdk, cache, marker)
        assert result.returncode and error in result.stderr, (error, result.stdout, result.stderr)
        assert not output.exists(), output
    print("Wave32 and wave64 compile controls and invalid operand refusals passed")
    return 0


def hardware(replay, sdk, work, root):
    for wave_size in (32, 64):
        for width, offset, scc in CASES:
            marker = request(work, root, wave_size, wave_size,
                             program(wave_size, width, offset, scc))
            output = work / f"wave{wave_size}_{width}_{offset}.bin"
            result = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"])
            if result.returncode == 77:
                return 77
            result.check_returncode()
            actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
            for index, (got, want) in enumerate(
                    zip(actual, expected_words(wave_size, width, offset, scc))):
                assert got == want, (wave_size, width, offset, scc, index, hex(got), hex(want))
            print(f"wave{wave_size} bfm vcc {width},{offset} SCC={scc}: "
                  "both groups and guards matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader, root_file = work / "root.hlsl", work / "root.bin"
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root_file, shader], check=True)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root_file.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
