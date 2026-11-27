"""A graphics root SRV (list slot 44) at a title address without an upload mapping, read by a translated draw.

tests/startup_root_srv_draw.cpp (X2W_TEST_ROOT_SRV_DRAW=write-after): slot 44 sees the buffer holding red, the title
writes green, then draws. Run with X2W_D3D12_ROOT_SRV_AT_DRAW unset and set:
  - both: the translated draw reads the SRV from its title address at the draw and draws green;
  - unset: slot 44 also copies a window for the desktop binding (logged "x2w-render root SRV 0 at title");
  - set: no draw reads that binding, so no copy is made and nothing is logged.
The left half of the 8x4 target is drawn, the right half keeps the clear colour.
"""
import os
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture

HLSL = (
    "ByteAddressBuffer colour : register(t0);\n"
    "float4 main(float4 p : SV_Position) : SV_Target {\n"
    "  return asfloat(colour.Load4(0));\n"
    "}\n"
)
CLEAR = bytes((0, 0, 255, 255))
GREEN = bytes((0, 255, 0, 255))
SWITCH = "X2W_D3D12_ROOT_SRV_AT_DRAW"
COPY_LINE = "x2w-render root SRV 0 at title"


def run(directory, consumer, backend, at_draw):
    folder = directory / f"write-after-{int(at_draw)}"
    folder.mkdir()
    log = folder / "trap.log"
    os.environ.pop(SWITCH, None)  # an inherited setting must not decide the unset case
    modes = {"X2W_TEST_ROOT_SRV_DRAW": "write-after", "X2W_TRAP_LOG": str(log)}
    if at_draw:
        modes[SWITCH] = "1"
    pixels = draw_texture(folder, consumer, backend, HLSL, modes, binding="srv 0 buf root 0")
    copies = [line for line in log.read_text(errors="replace").splitlines() if COPY_LINE in line] if log.exists() else []
    return pixels, copies


def check(directory, consumer, backend):
    expected = (GREEN * 4 + CLEAR * 4) * 4
    for at_draw in (False, True):
        pixels, copies = run(directory, consumer, backend, at_draw)
        if pixels is None:
            return 77
        label = f"{SWITCH}={int(at_draw)}"
        if pixels != expected:
            raise AssertionError(f"{label}: actual {pixels.hex()} expected {expected.hex()}")
        if bool(copies) == at_draw:
            raise AssertionError(f"{label}: {len(copies)} slot 44 copy lines, expected {'none' if at_draw else 'one'}")
        print(f"{backend} {label}: 32 pixels exact (green, read at the draw); slot 44 copy lines {len(copies)}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
