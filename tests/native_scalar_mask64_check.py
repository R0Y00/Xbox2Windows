"""Check s_andn2_wrexec_b64 in a wave64 request of at most 32 threads."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_scalar_lds_check import sop1
from native_wrexec64_check import (
    ANDN2_WREXEC_B64, EXEC_HI, LITERAL, MOV_B32, OLD_EXEC, OUTPUT_WORDS,
    expected_output, program, request,
)


BCNT1_I32_B64, VCC_LO, END = 0x10, 106, 0xBF810000
SOURCE = (0x96, 0x40000001)
REGISTER = r"%[-\w.]+"


def wrexec_program(dst=28, *, high="initial"):
    code = program(SOURCE, dst, 0)
    code[code.index(sop1(ANDN2_WREXEC_B64, dst, 22))] = sop1(ANDN2_WREXEC_B64, dst, 28)
    for src, dst_reg in ((22, 28), (23, 29)):
        code[code.index(sop1(MOV_B32, src, LITERAL))] = sop1(MOV_B32, dst_reg, LITERAL)
    if high == "initial":
        index = code.index(sop1(MOV_B32, EXEC_HI, LITERAL))
        del code[index:index + 2]
    elif high == "cmpx":
        index = code.index(sop1(ANDN2_WREXEC_B64, dst, 28))
        code.insert(index, 0x7DA40080)  # v_cmpx_eq_u32 0, v0 (lane 0 disabled)
    return code


def count_program(src=VCC_LO):
    return [sop1(MOV_B32, VCC_LO, LITERAL), 0x80000003,
            sop1(MOV_B32, VCC_LO + 1, LITERAL), 0x40000050,
            sop1(BCNT1_I32_B64, 16, src), END]


def compile_case(cli, sdk, work, marker, diagnostic=None):
    output = marker.with_suffix(".cso")
    run = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                         capture_output=True, text=True, timeout=30)
    if diagnostic:
        assert run.returncode and not output.exists(), (run.stdout, run.stderr)
        assert any(line.endswith(": " + diagnostic) for line in run.stderr.splitlines()), run.stderr
        return ""
    assert run.returncode == 0 and output.is_file(), (run.stdout, run.stderr)
    return subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", output],
                          capture_output=True, text=True, timeout=30, check=True).stdout


def definitions(dump):
    return {name: rhs.strip() for name, rhs in
            re.findall(rf"^\s*({REGISTER}) = ([^\n;]+)", dump, re.MULTILINE)}


def stored(dump, reg):
    return re.findall(rf"^\s*store i32 (\S+), i32\* %r{reg}, align 4", dump, re.MULTILINE)


def check_scc(defs, value, result):
    check = re.fullmatch(rf"zext i1 ({REGISTER}) to i32", defs[value])
    assert check and defs[check[1]] == f"icmp ne i32 {result}, 0", defs


def check_wrexec(dump, old_high):
    defs = definitions(dump)
    low, high = (stored(dump, reg)[-1] for reg in (28, 29))
    for result, src, exec_reg in ((low, 28, 126), (high, 29, 127)):
        operands = re.fullmatch(rf"and i32 ({REGISTER}), ({REGISTER})", defs[result])
        assert operands and defs[operands[1]] == f"load i32, i32* %r{src}, align 4", dump
        inv = re.fullmatch(rf"xor i32 ({REGISTER}), -1", defs[operands[2]])
        assert inv and defs[inv[1]] == f"load i32, i32* %r{exec_reg}, align 4", dump
        assert stored(dump, exec_reg)[-2] == result, dump  # followed by fixture restore
    assert int(stored(dump, EXEC_HI)[-3]) == old_high, dump
    scc = stored(dump, 128)[-1]
    merged, = [name for name, rhs in defs.items() if rhs == f"or i32 {low}, {high}"]
    check_scc(defs, scc, merged)


def controls(cli, sdk, work, root):
    for high in ("initial", "scalar", "cmpx"):
        marker = request(work, root, 32, wrexec_program(high=high))
        check_wrexec(compile_case(cli, sdk, work, marker), OLD_EXEC[1] if high == "scalar" else 0)
    for code, threads, diagnostic in (
        (wrexec_program(dst=29), 32, "unsupported wave64 scalar operation"),
        (count_program(src=107), 64, "invalid s_bcnt1_i32_b64 source"),
        # VCC stays refused as a 64-bit count source.
        (count_program(), 64, "invalid s_bcnt1_i32_b64 source"),
        (count_program(), 32, "invalid s_bcnt1_i32_b64 source"),
    ):
        compile_case(cli, sdk, work, request(work, root, threads, code), diagnostic)
    print("64-bit scalar mask DXIL and refusal diagnostics matched")
    return 0


def hardware(replay, sdk, work, root):
    for high in ("initial", "scalar", "cmpx"):
        marker = request(work, root, 32, wrexec_program(high=high))
        output = marker.with_suffix(".bin")
        run = subprocess.run([replay, sdk, marker, work / "cache", "0", output, "--direct"], timeout=30)
        if run.returncode == 77:
            return 77
        run.check_returncode()
        old_low = 0 if high == "cmpx" else OLD_EXEC[0]
        old_high = OLD_EXEC[1] if high == "scalar" else 0
        mask = (SOURCE[0] & ~old_low, SOURCE[1] & ~old_high)
        actual = struct.unpack(f"<{OUTPUT_WORDS}I", output.read_bytes())
        expected = expected_output(32, mask, int(any(mask)))
        assert list(actual) == expected, (high, actual, expected)
    print("32-thread wave64 WREXEC lane stores, destination words and SCC matched")
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        source, root = work / "root.hlsl", work / "root.bin"
        source.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, source], check=True, timeout=30)
        return {"controls": controls, "hardware": hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
