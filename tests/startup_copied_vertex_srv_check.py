"""Read both sides of a 256-byte boundary through a copied vertex-buffer SRV."""
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture

VERTEX_SHADER = """
%s
struct Vertex {
    float4 position : SV_Position;
    nointerpolation uint2 values : TEXCOORD0;
};
Vertex main(uint id : SV_VertexID) {
    const float2 corners[3] = {float2(-1, 1), float2(3, 1), float2(-1, -3)};
    Vertex output;
    output.position = float4(corners[id], 0.5, 1);
    output.values = uint2(%s);
    return output;
}
"""

VERTEX = VERTEX_SHADER % (
    "ByteAddressBuffer records : register(t0);",
    "records.Load(224), records.Load(288)",
)
PAIR_VERTEX = VERTEX_SHADER % (
    "ByteAddressBuffer first : register(t0);\nByteAddressBuffer second : register(t1);",
    "first.Load(288), second.Load(288)",
)

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


def check_pair(directory, consumer, backend):
    for name, offset in (("distinct", 32), ("collapsed", 0)):
        folder = directory / name
        folder.mkdir()
        pixels = draw_texture(folder, consumer, backend, PIXEL,
                              {"X2W_TEST_COPIED_VERTEX_SRV_PAIR": "1"},
                              binding=f"srv 0 buf table 0 0\nsrv 1 buf table 0 {offset}", vertex_hlsl=PAIR_VERTEX)
        if pixels is None:
            return 77
        first = bytes((181, 181 if offset == 0 else 211, 0, 255))
        swapped = bytes((211, 211 if offset == 0 else 181, 0, 255))
        expected = (first * 2 + swapped * 4 + first * 2) * 4
        if pixels != expected:
            raise AssertionError(f"{backend}: {name} vertex SRVs got {pixels.hex()}, expected {expected.hex()}")
        print(f"{backend}: 32 pixels matched {name} vertex table offsets after swap and restore")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        result = check(Path(temporary), *sys.argv[1:])
        if result:
            sys.exit(result)
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check_pair(Path(temporary), *sys.argv[1:]))
