"""A table buffer view over a placed buffer: per-draw copies by default, a live alias with X2W_D3D12_TABLE_ALIAS=1."""
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture

VERTEX = """
ByteAddressBuffer records : register(t0);
struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint2 values : TEXCOORD0;
};
Vertex main(uint id : SV_VertexID) {
    const float2 corners[3] = {float2(-1, 1), float2(3, 1), float2(-1, -3)};
    Vertex output;
    output.position = float4(corners[id], 0.5, 1);
    output.values = uint2(records.Load(16), records.Load(20));
    return output;
}
"""

PIXEL = """
float4 main(float4 position : SV_Position, nointerpolation uint2 values : TEXCOORD0) : SV_Target {
    return float4(float2(values) / 255.0, 0, 1);
}
"""


def texel(value):
    return bytes((value, value + 1, 0, 255))


def check(directory, consumer, backend):
    # The fixture writes 40, draws the left half, writes 80, draws the right half, then writes 120 before the list runs.
    cases = (
        # The copy case clears the switch, so it keeps testing the default when the switch is set in the environment.
        ("copy", {"X2W_D3D12_TABLE_ALIAS": "0"}, texel(40) * 4 + texel(80) * 4),
        ("alias", {"X2W_D3D12_TABLE_ALIAS": "1"}, texel(120) * 8),
    )
    for name, switches, row in cases:
        folder = directory / name
        folder.mkdir()
        pixels = draw_texture(folder, consumer, backend, PIXEL, {"X2W_TEST_TABLE_ALIAS_WRITE": "1", **switches},
                              binding="srv 0 buf table 0 0", vertex_hlsl=VERTEX)
        if pixels is None:
            return 77
        expected = row * 4
        if pixels != expected:
            raise AssertionError(f"{backend}: {name} table buffer got {pixels.hex()}, expected {expected.hex()}")
        print(f"{backend}: 32 pixels matched the {name} reading of a rewritten table buffer")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
