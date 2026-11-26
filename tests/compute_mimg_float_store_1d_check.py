"""Check gfx1013 1D RGBA32 FLOAT image_store and explicit-mip SRV reads."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
LINES = (HERE / "compute_mimg_float_store_1d.s").read_text(encoding="ascii").splitlines()
GROUP = (4, 1, 1)
GROUPS = 2
WIDTH = 32
MIPS = 3
BINDING = dict(table_reg=7, group_reg=10, image_descriptor=0x40, image_srv_descriptor=0x60,
               image_format="R32G32B32A32_FLOAT", image_uav_mip=0,
               image_uav_dimension="1d", image_srv_dimension="1d")
LOAD = next(line for line in LINES if line.startswith("image_load_mip"))
STORE = next(line for line in LINES if line.startswith("image_store"))
# Little-endian instruction words assembled by LLVM 15.0.7.
WORDS = bytes.fromhex(
    "07038cbe 80038dbe 06060cf4 400000fa 06080cf4 600000fa 080046d7 0a040104"
    " 81101236 8112124a 7fc08cbf 000f04f0 080e0800 703f8cbf 000f20f0 080e0600 000081bf"
)
TEXTURE_1D_FLAG = 1 << 23  # tests/compute_check.cpp texture_1d_flag
MIP_ADDRESS = "asint((0x00000001u + (0x00000001u & tid.x)))"
STORE_INDEX = "outputData[tid.x]"


def controls():
    source = aot.lower(LINES, GROUP, **BINDING)
    assert "Texture1D<float4> imageData : register(t0);" in source, source
    assert "RWTexture1D<float4> outputData : register(u0);" in source, source
    assert f"asuint(imageData.Load(int2(asint(tid.x), {MIP_ADDRESS})))" in source, source
    assert (f"{STORE_INDEX} = "
            "asfloat(uint4(image_load_0.x, image_load_0.y, image_load_0.z, image_load_0.w));") in source, source
    for instruction, kind in ((STORE, "store"), (LOAD, "load")):
        op = instruction.split()[0]
        sparse = instruction.replace("v[14:17]", "v[14:15]").replace("dmask:0xf", "dmask:0x5")
        d16 = instruction.replace("v[14:17]", "v[14:15]") + " d16"
        a16 = instruction.replace(", v[8:9],", ", v8,") + " a16"
        address = (", v8,", ", v[8:9],") if kind == "store" else (", v[8:9],", ", v[8:10],")
        array = instruction.replace("IMG_1D", "IMG_1D_ARRAY").replace(*address)
        for changed, diagnostic in (
            (sparse, f"{op} requires full DMASK 0xf"),
            (d16, "Unsupported compute instruction"),
            (a16, f"Unsupported image {kind} target: gfx1013 requires full-width"),
            (array, "Unsupported compute instruction"),
        ):
            rejects(diagnostic, aot.lower, variant(LINES, (instruction, changed)), GROUP, **BINDING)
    for original, changed, diagnostic in (
        (STORE, STORE.replace(", v8,", ", v[8:9],"), "Unsupported image store target"),
        (LOAD, LOAD.replace(", v[8:9],", ", v8,"), "Unsupported image load target"),
        (LOAD, LOAD.replace("image_load_mip", "image_load").replace(", v[8:9],", ", v8,"),
         "Unsupported image load target: gfx1013 requires full-width"),
        (STORE, STORE.replace("image_store", "image_store_mip").replace(", v8,", ", v[8:9],"),
         "Unsupported image store target: gfx1013 requires full-width"),
        (LOAD, LOAD.replace("s[32:39]", "s[24:31]"), "Unverified image descriptor"),
        (STORE, STORE.replace("image_store v[14:17]", "image_atomic_add v14").replace("dmask:0xf", "dmask:0x1"),
         "Unsupported compute instruction: image_atomic_add"),
        ("s_waitcnt vmcnt(0)", "", "before s_waitcnt vmcnt(0)"),
    ):
        rejects(diagnostic, aot.lower, variant(LINES, (original, changed)), GROUP, **BINDING)
    for binding, diagnostic in (
        ({"image_uav_dimension": "2d"}, "image_store dimension 1d does not match the 2d UAV view"),
        ({"image_srv_dimension": "2d"}, "image_load_mip dimension 1d does not match the 2d SRV view"),
        ({"image_srv_descriptor": 0x40}, "overlapping image SRV descriptor"),
        ({"image_srv_descriptor": None}, "1D image loads require a separate gfx1013 RGBA32 FLOAT SRV binding"),
        ({"cpu": "gfx1030"}, "1D image UAVs require a gfx1013 RGBA32 FLOAT store binding"),
        ({"cpu": "gfx1030", "image_uav_dimension": "2d"},
         "1D image loads require a separate gfx1013 RGBA32 FLOAT SRV binding"),
        ({"image_format": "R32G32B32A32_UINT"}, "1D image UAVs require a gfx1013 RGBA32 FLOAT store binding"),
    ):
        rejects(diagnostic, aot.lower, LINES, GROUP, **(BINDING | binding))
    print("1D translation controls passed", flush=True)
    return source


def binary(directory, assembler, objcopy, library, source):
    code = assemble(directory, assembler, objcopy, LINES)
    assert code == WORDS, f"assembled {code.hex()} != recorded {WORDS.hex()}"
    print("1D gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert [line for line in decoded if line.startswith("image_")] == [LOAD, STORE]
    assert aot.lower(decoded, GROUP, **BINDING) == source
    container, output = directory / "image-1d.dxbc", directory / "cli.hlsl"
    container.write_bytes(store_container(code, GROUP))
    options = [arg for key, value in BINDING.items() for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *options], check=True)
    assert output.read_text() == source


def input_word(mip, x, channel):
    return 0x3f000000 + mip * 0x100000 + x * 0x20 + channel * 2


def reference():
    initial, expected, texels = [], [], []
    for mip in range(MIPS):
        for x in range(WIDTH >> mip):
            words = [input_word(mip, x, channel) for channel in range(4)]
            sentinels = [word | 0x80000000 for word in words]
            texels.extend(words)
            initial.extend(sentinels)
            # Each invocation reads from one of two source mips and writes a unique destination.
            if mip == 0 and x < GROUP[0] * GROUPS:
                expected.extend(input_word(1 + x % 2, x, channel) for channel in range(4))
            else:
                expected.extend(sentinels)
    assert sum(a != b for a, b in zip(initial, expected, strict=True)) == 32
    return initial + expected + texels


def draw(directory, compiler, renderer, adapter, source):
    hlsl, cso, case = (directory / f"image-1d.{ext}" for ext in ("hlsl", "cso", "case"))
    payload = reference()
    case.write_bytes(struct.pack(f"<2I{len(payload)}I", GROUPS | TEXTURE_1D_FLAG, 0, *payload))
    cases = (
        ("translated", source, 0, None),
        ("force source mip zero", source.replace(MIP_ADDRESS, "0"),
         32, "mip=0 word=0 expected=3f100000 actual=3f000000"),
        ("shift destination X", source.replace(STORE_INDEX, "outputData[tid.x + 1u]"),
         36, "mip=0 word=0 expected=3f100000 actual=bf000000"),
    )
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_load_mip\nimage_store\n", encoding="ascii")
    for name, code, mismatches, first_mismatch in cases:
        assert (code == source) == (mismatches == 0), name
        hlsl.write_text(code, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, "--compute-texture-store-case", cso, case],
                                capture_output=True, text=True, timeout=15)
        print(f"{name}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, name
        assert f"compared_words=224 mismatches={mismatches}" in result.stdout, name
        assert result.returncode == int(mismatches != 0), name
        if first_mismatch:
            assert first_mismatch in result.stderr, name
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--binary", nargs=2, metavar=("LLVM_MC", "LLVM_OBJCOPY"))
    modes.add_argument("--draw", nargs=3, metavar=("DXC", "RENDERER", "ADAPTER"))
    parser.add_argument("--llvm", metavar="LIBLLVM", help="Also check disassembly and the container CLI")
    args = parser.parse_args()
    source = controls()
    if not (args.binary or args.draw):
        return 0
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compute_mimg_1d_", dir=out) as tmp:
        directory = Path(tmp)
        if args.binary:
            binary(directory, *args.binary, args.llvm, source)
            return 0
        return draw(directory, *args.draw, source)


if __name__ == "__main__":
    raise SystemExit(main())
