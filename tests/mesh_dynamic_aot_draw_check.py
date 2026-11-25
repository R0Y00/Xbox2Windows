"""Run the generated runtime mesh shader through the existing D3D12 oracle."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_dynamic_geometry_check import CASES, CODE, MESH_LIMITS, RETIRED_PREFIX, SECOND, decode
from mesh_dynamic_aot import lower
from mesh_entry_aot import lower as lower_static


SIDE = 32
COLOR = b"\xff\x00\xff\xff"
CLEAR = b"\x00\x00\x00\xff"


def compile_shader(dxc, directory, stem, text, profile):
    source = directory / (stem + ".hlsl")
    output = directory / (stem + ".dxil")
    source.write_text(text, encoding="ascii")
    subprocess.run([dxc, "-WX", "-T", profile, "-E", "main", "-Fo", output, source], check=True)
    return output


def fixture(counts, geometry, indices):
    data = bytearray(1536)
    struct.pack_into("<II", data, 1024, counts, geometry)
    data.extend(b"\xcd" * 80)  # The unmodified shader has no trace UAV.
    full = (2, 1, 3) in indices
    triangle = (0, 1, 2) in indices
    pixels = (COLOR if full or (triangle and y > x) else CLEAR
              for y in range(SIDE) for x in range(SIDE))
    data.extend(b"".join(pixels))
    return data


def check(dxc, runner, backend, target):
    pixel = compile_shader(dxc, target, "solid", "float4 main() : SV_Target { return float4(1,0,1,1); }", "ps_6_0")
    mesh = compile_shader(dxc, target, "dynamic", lower(decode(CODE), 2, MESH_LIMITS), "ms_6_5")
    old_key = struct.pack("<I", 0x300402)
    assert CODE.count(old_key) == 1
    alternate_code = CODE.replace(old_key, struct.pack("<I", 0x300401))
    alternate = compile_shader(dxc, target, "changed", lower(decode(alternate_code), 2, MESH_LIMITS), "ms_6_5")
    cases = [(mesh, *case) for case in CASES]
    retired_program = decode(RETIRED_PREFIX + CODE)
    retired = compile_shader(dxc, target, "retired", lower(retired_program, 2, MESH_LIMITS), "ms_6_5")
    # One triangle and two triangles take distinct runtime geometry paths after the early exit.
    cases.extend((retired, "retired wave: " + name, *values) for name, *values in CASES[:2])
    static = compile_shader(dxc, target, "retired-static",
                            lower_static(retired_program, [0x2004, SECOND], MESH_LIMITS), "ms_6_5")
    cases.append((static, "retired wave: specialized quad", *CASES[1][1:]))
    cases.extend([
        (alternate, "changed source rejects old input", 0x2004, 0x300402, 0, ()),
        (alternate, "changed source accepts new input", 0x2004, 0x300401, 4, ((0, 1, 2), (1, 1, 3))),
    ])
    for index, (shader, name, counts, geometry, _, indices) in enumerate(cases):
        inputs = target / f"case-{index}.bin"
        inputs.write_bytes(fixture(counts, geometry, indices))
        print(name, flush=True)
        result = subprocess.run([runner, backend, "--mesh-indexed", shader, pixel, inputs])
        if result.returncode:
            return result.returncode
    wrong_case = target / "quad-with-triangle-reference.bin"
    wrong_case.write_bytes(fixture(0x2004, SECOND, ((0, 1, 2),)))
    wrong = subprocess.run([runner, backend, "--mesh-indexed", retired, pixel, wrong_case], capture_output=True, text=True)
    if (wrong.returncode != 1 or "Draw reference mismatch" not in wrong.stderr
            or "pixels=1024 mismatches=528" not in wrong.stdout.splitlines()):
        raise AssertionError(wrong.stdout + wrong.stderr)
    print("Retired-wave quad rejected the triangle reference at 528 pixels")
    return 0


if __name__ == "__main__":
    output = Path("out").resolve()
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-dynamic-native-", dir=output) as target:
        sys.exit(check(*sys.argv[1:], Path(target)))
