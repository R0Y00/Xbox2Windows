"""A translated direct indexed draw uses the aliased view's pre-submit contents."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from startup_address_srv_check import prepare_draw_cache

VERTEX = """
float4 main(uint vertex : SV_VertexID) : SV_Position {
    uint corner = vertex % 3;
    float2 xy = float2(corner == 1 ? 3.0 : -1.0, corner == 2 ? -3.0 : 1.0);
    return float4(xy.x + (vertex >= 3 ? 8.0 : 0.0), xy.y, 0.5, 1.0);
}
"""
PIXEL = "float4 main() : SV_Target { return float4(1, 0, 0, 1); }"
VISIBLE = bytes((255, 0, 0, 255))
CLIPPED = bytes((0, 0, 255, 255))
CASES = (
    ("visible", VISIBLE),
    ("clipped", CLIPPED),
    ("late-visible", VISIBLE),
    ("late-clipped", CLIPPED),
    ("late-visible-r32", VISIBLE),
)


def check(directory, consumer, backend):
    env, args, output = prepare_draw_cache(directory, consumer, PIXEL, {}, "", vertex_hlsl=VERTEX)
    env["X2W_D3D12_DEBUG"] = "1"
    for mode, color in CASES:
        output.unlink(missing_ok=True)
        result = subprocess.run([*args, backend, output], env={**env, "X2W_TEST_INDEX_PROBE": mode},
                                capture_output=True, text=True, errors="replace", timeout=20)
        if result.returncode == 77:
            print(result.stdout, result.stderr)
            return 77
        if result.returncode or re.search(r"x2w-d3d12-debug [^\n]*sev=[01]\b", result.stderr):
            raise RuntimeError(f"{mode} failed ({result.returncode})\n{result.stdout}\n{result.stderr[-4000:]}")
        if not re.search(r"slot133[^\n]*-> 0x00000000 aliased gpuva=", result.stderr):
            raise AssertionError(f"{mode}: index buffer was not aliased\n{result.stderr[-4000:]}")
        if not re.search(r"indexed\(3, 1, start [01], base 0\)[^\n]*-> drawn;", result.stderr):
            raise AssertionError(f"{mode}: translated indexed draw was not recorded\n{result.stderr[-4000:]}")
        pixels = output.read_bytes()
        if pixels != color * 32:
            raise AssertionError(f"{mode}: got {pixels.hex()}, expected 32 pixels of {color.hex()}")
        adapter = next(line for line in result.stdout.splitlines() if line.startswith("adapter="))
        debug_layer = "enabled" if "debug layer: 0x00000000 enabled" in result.stderr else "unavailable"
        print(f"{mode}: 32 exact pixels, {adapter}, debug layer {debug_layer}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
