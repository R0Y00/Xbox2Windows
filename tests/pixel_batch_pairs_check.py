"""Compare a sequence of two pixel shaders on one device with isolated draws."""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory


def verify(dxc, renderer, adapter, directory):
    shaders = []
    cases = []
    for label, expression, output in (("rg", "color", (0.25, 0.75, 0.5, 1.0)),
                                      ("gr", "color.grba", (0.75, 0.25, 0.5, 1.0))):
        source = directory / f"{label}.hlsl"
        source.write_text("cbuffer Params : register(b0) { float4 color; };\n"
                          f"float4 main() : SV_Target {{ return {expression}; }}\n", encoding="ascii")
        shader = directory / f"{label}.dxil"
        subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        shaders.append(shader)
        case = directory / f"{label}.case"
        constants = struct.pack("<4f", 0.25, 0.75, 0.5, 1.0).ljust(256, b"\0")
        case.write_bytes(constants + struct.pack("<4f", *output) * 64)
        cases.append(case)

    def run(*args):
        return subprocess.run([renderer, adapter, *args], capture_output=True, text=True)

    a, b = shaders
    rg, gr = cases
    sequence = [(a, rg), (b, rg), (b, gr), (a, rg)]
    baseline = []
    for shader, case in sequence:
        single = run("--check-pixel-case", shader, case)
        if single.returncode == 77 and adapter == "hardware":
            print(single.stdout + single.stderr)
            return 77
        if single.returncode == 0:
            baseline.append("pass")
        elif single.returncode == 1 and "pixel_mismatch" in single.stderr:
            baseline.append("mismatch")
        else:
            raise AssertionError(single.stdout + single.stderr)
    assert baseline == ["pass", "mismatch", "pass", "pass"], baseline

    def check(pairs, statuses, code):
        batch = run("--check-pixel-pairs", *(path for pair in pairs for path in pair))
        lines = [line for line in batch.stdout.splitlines() if line.startswith("case=")]
        expected = [f"case={i} status={status}" for i, status in enumerate(statuses)]
        if batch.returncode != code or lines != expected:
            raise AssertionError(batch.stdout + batch.stderr)
        return batch

    mixed = check(sequence, baseline, 1)
    diagnostics = mixed.stderr.split("case=")
    assert len(diagnostics) == len(sequence) + 1, mixed.stderr
    for i, ((shader, case), section) in enumerate(zip(sequence, diagnostics[1:])):
        assert section.startswith(f"{i} path={case}\nshader={shader}\n"), section
        assert ("pixel_mismatch" in section) == (baseline[i] == "mismatch"), section
    check([(a, rg), (b, gr)], ["pass", "pass"], 0)
    check([(a, rg), (directory / "absent.dxil", rg), (b, gr)], ["pass", "error", "pass"], 1)
    print("Two shaders matched the four isolated results. The missing shader failed only its own case.")
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(verify(*sys.argv[1:], Path(folder)))
