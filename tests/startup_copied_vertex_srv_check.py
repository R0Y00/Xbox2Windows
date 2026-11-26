"""Read both sides of a 256-byte boundary through a copied vertex-buffer SRV."""
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
    output.values = uint2(records.Load(224), records.Load(288));
    return output;
}
"""

PIXEL = """
float4 main(float4 position : SV_Position, nointerpolation uint2 values : TEXCOORD0) : SV_Target {
    return float4(float2(values) / 255.0, 0, 1);
}
"""


def check(directory, consumer, backend):
    pixels = draw_texture(directory, consumer, backend, PIXEL,
                          {"X2W_TEST_COPIED_VERTEX_SRV": "1"},
                          binding="srv 0 buf table 0 0", vertex_hlsl=VERTEX)
    if pixels is None:
        return 77
    full = bytes((57, 181, 0, 255))
    short = bytes((57, 0, 0, 255))
    expected = (full * 2 + short * 4 + full * 2) * 4
    if pixels != expected:
        raise AssertionError(f"{backend}: copied vertex SRV got {pixels.hex()}, expected {expected.hex()}")
    print(f"{backend}: 32 pixels matched full, short, and restored copied descriptors")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
