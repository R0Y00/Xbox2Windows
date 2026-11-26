"""Check full-mask gfx1013 RGBA32 FLOAT stores to a 2D array and a 3D texture."""
import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant
import compute_mimg_float_store_check as flat

HERE = Path(__file__).resolve().parent
ARRAY_LINES = (HERE / "compute_mimg_float_store_layers.s").read_text(encoding="ascii").splitlines()
ARRAY_STORE = next(line for line in ARRAY_LINES if line.startswith("image_store"))
VOLUME_STORE = ARRAY_STORE.replace("IMG_2D_ARRAY", "IMG_3D")
VOLUME_LINES = variant(ARRAY_LINES, (ARRAY_STORE, VOLUME_STORE))
ARRAY_WORDS = bytes.fromhex(
    "070394be 800395be 0a060cf4 400000fa 080046d7 0a040104 090046d7 0b040504"
    " 0a0046d7 0c020904 ff101c38 0000813f ff121e38 00001140 ff142038 00004140"
    " ff02227e 010081bf 7fc08cbf 280f20f0 080e0600 000081bf"
)
VOLUME_WORDS = ARRAY_WORDS.replace(bytes.fromhex("280f20f0"), bytes.fromhex("100f20f0"))
GROUP = (4, 4, 2)
GROUPS = 2
BINDING = dict(table_reg=7, group_reg=10, group_reg_y=11, group_reg_z=12, image_descriptor=0x40,
               image_format="R32G32B32A32_FLOAT", image_uav_mip=0)
# Selectors in tests/compute_check.cpp. Both views cover four slices at mip zero.
CASES = (
    ("2d-array", ARRAY_LINES, ARRAY_STORE, ARRAY_WORDS, "RWTexture2DArray", 1 << 31, 5376),
    ("3d", VOLUME_LINES, VOLUME_STORE, VOLUME_WORDS, "RWTexture3D", (1 << 31) | (1 << 30), 4672),
)
ADDRESS = "outputData[uint3(tid.x, tid.y, tid.z)]"


def controls(dimension, lines, store, texel_type):
    binding = BINDING | {"image_uav_dimension": dimension}
    source = aot.lower(lines, GROUP, **binding)
    assert f"{texel_type}<float4> outputData : register(u0);" in source, source
    assert ADDRESS in source, source
    wrong_dimension = "3d" if dimension == "2d-array" else "2d-array"
    for view in (wrong_dimension, "2d"):
        rejects(f"image_store dimension {dimension} does not match the {view} UAV view",
                aot.lower, lines, GROUP, **(binding | {"image_uav_dimension": view}))
    sparse = store.replace("v[14:17]", "v[14:15]").replace("dmask:0xf", "dmask:0x5")
    for changed, message in (
        (sparse, "image_store requires full DMASK 0xf"),
        (store.replace("v[8:10]", "v[8:9]"), "Unsupported image store target"),
        (store + " d16", "Unsupported compute instruction"),
        (store + " a16", "Unsupported compute instruction"),
        (store.replace("image_store", "image_store_mip"), "Unsupported compute instruction"),
        (store.replace("image_store", "image_load_mip"), "Unsupported compute instruction"),
    ):
        rejects(message, aot.lower, variant(lines, (store, changed)), GROUP, **binding)
    rejects("Image store mip does not match the UAV view", aot.lower, lines, GROUP,
            **(binding | {"image_uav_mip": 1}))
    print(f"{dimension}: translation controls passed", flush=True)
    return source


def combined_controls():
    # A 2D mip load from the separate SRV followed by a 2D array store binds both views.
    store = flat.STORE.replace("v[8:9]", "v[8:10]").replace("IMG_2D", "IMG_2D_ARRAY")
    source = aot.lower(variant(flat.LINES, (flat.STORE, store)), flat.GROUP,
                       **(flat.BINDING | {"image_uav_dimension": "2d-array"}))
    assert "Texture2D<float4> imageData : register(t0);" in source, source
    assert "RWTexture2DArray<float4> outputData : register(u0);" in source, source
    assert "outputData[uint3(tid.x, tid.y, 0x00000001u)] = " in source, source
    # Array and volume views are gfx1013-only.
    for dimension in ("2d-array", "3d"):
        rejects("Array and 3D image UAVs require a gfx1013 RGBA32 FLOAT store binding", aot.lower,
                ARRAY_LINES, GROUP, **(BINDING | {"image_uav_dimension": dimension, "cpu": "gfx1030"}))
    print("combined SRV load and array store, and gfx1030 refusal: controls passed", flush=True)


def binary(directory, assembler, objcopy, library, dimension, lines, store, words, source):
    code = assemble(directory, assembler, objcopy, lines)
    assert code == words, f"{dimension}: assembled {code.hex()} != recorded {words.hex()}"
    print(f"{dimension}: gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    binding = BINDING | {"image_uav_dimension": dimension}
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert [line for line in decoded if line.startswith("image_")] == [store]
    assert aot.lower(decoded, GROUP, **binding) == source
    container, output = directory / "layers.dxbc", directory / "layers.hlsl"
    container.write_bytes(store_container(code, GROUP))
    options = [arg for key, value in binding.items() for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *options], check=True)
    assert output.read_text() == source


def reference(dimension):
    initial, expected = [], []
    # D3D12 array subresources are ordered by layer, then mip. A volume has one
    # subresource per mip and stores all depth slices with a shrinking depth.
    array = dimension == "2d-array"
    for layer in range(4 if array else 1):
        for mip in range(3):
            for depth in range(1 if array else 4 >> mip):
                z = layer if array else depth
                for y in range(8 >> mip):
                    for x in range(32 >> mip):
                        tag = mip << 16 | z << 12 | y << 7 | x << 2
                        sentinels = [0xBF000000 | tag | c for c in range(4)]
                        initial.extend(sentinels)
                        if mip == 0 and x < 8 and y < 4 and z < 2:
                            expected.extend((0x3F810000 + x, 0x40110000 + y, 0x40410000 + z, 0xBF810001))
                        else:
                            expected.extend(sentinels)
    return initial, expected


def draw(directory, dxc, renderer, adapter, dimension, selector, count, source):
    hlsl, cso, case = (directory / f"layers.{ext}" for ext in ("hlsl", "cso", "case"))
    initial, expected = reference(dimension)
    assert len(initial) == len(expected) == count
    assert sum(before != after for before, after in zip(initial, expected, strict=True)) == 256
    case.write_bytes(struct.pack(f"<2I{2 * count}I", GROUPS | selector, 0, *initial, *expected))
    swapped = source.replace(ADDRESS, "outputData[uint3(tid.x, tid.z, tid.y)]")
    assert swapped != source
    for label, code, mismatches in (("translated", source, 0), ("swap y and slice/depth", swapped, 288)):
        hlsl.write_text(code, encoding="ascii")
        subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, "--compute-texture-store-case", cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f"{dimension} {label}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, label
        assert f"compared_words={count} mismatches={mismatches}" in result.stdout, label
        if mismatches:
            assert result.returncode == 1 and "mip=0 word=129 expected=40110001 actual=40110000" in result.stderr
        else:
            assert result.returncode == 0, label
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--binary", nargs=2, metavar=("LLVM_MC", "LLVM_OBJCOPY"))
    modes.add_argument("--draw", nargs=3, metavar=("DXC", "RENDERER", "ADAPTER"))
    parser.add_argument("--llvm", metavar="LIBLLVM", help="Also check disassembly and the container CLI")
    args = parser.parse_args()
    combined_controls()
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    status = 0
    with tempfile.TemporaryDirectory(prefix="compute_mimg_layers_", dir=out) as tmp:
        directory = Path(tmp)
        for dimension, lines, store, words, texel_type, selector, count in CASES:
            source = controls(dimension, lines, store, texel_type)
            if args.binary:
                binary(directory, *args.binary, args.llvm, dimension, lines, store, words, source)
            elif args.draw and draw(directory, *args.draw, dimension, selector, count, source) == 77:
                status = 77
    return status


if __name__ == "__main__":
    raise SystemExit(main())
