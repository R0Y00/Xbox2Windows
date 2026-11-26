"""Startup FP32 array UAV binding through the production slot-67 / slot-14 path (#2742)."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_compute_image_store_check as A
import startup_compute_dispatch_texture_check as T
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a

WIDTH, HEIGHT, LAYERS, FIRST = 32, 8, 6, 1
WORDS = WIDTH * HEIGHT * 4


def reference(layer=None, rows=2, layers=LAYERS):
    result = [0xbf000000 | (z << 12) | i for z in range(layers) for i in range(WORDS)]
    if layer is not None:
        for y in range(rows):
            for x in range(4 * A.fixture.GROUPS):
                at = (FIRST + layer) * WORDS + (y * WIDTH + x) * 4
                result[at:at + 4] = A.ARRAY_VALUES
    return result


def check(directory, consumer, dxc, adapter, layer, masked):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1")
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    (directory / "compute_root").write_bytes(T.C.root_blob(3, table_range=True))
    env["X2W_TEST_COMPUTE_ROOT"] = str(directory / "compute_root")
    program = A.array_program(layer)
    if masked:
        assert program.count(A.ARRAY_STORE) == 1
        program = program.replace(A.ARRAY_STORE, A.MASK + A.ARRAY_STORE)
    (directory / "program").write_bytes(program)
    (directory / "data").write_bytes(DATA)
    (directory / "state").write_bytes(state_a(A.fixture.GROUP, A.RSRC2))
    layout = "<%dI" % (WORDS * LAYERS)
    mode = "texture-array"
    got, result = T.run(directory, consumer, env, mode, layout)
    print(result.stdout, end="")
    assert got == reference(), "untranslated dispatch wrote the texture"
    assert len(T.lines(result, "Dispatch(2, 1, 1) refused: no translation")) == 1, result.stderr[-3000:]
    key = int(T.lines(result, "x2w-draw compute request")[0].split()[3], 16)
    T.translate(cache, dxc, key)
    manifest = (cache / f"{key:016x}.txt").read_text()
    assert "uav 0 tex2da float table 1 64 channels 4" in manifest.splitlines(), manifest
    hlsl = cache / f"{key:016x}.cs.hlsl"
    cso = cache / f"{key:016x}.cs.cso"
    source = hlsl.read_text()
    assert "RWTexture2DArray<float4> w0 : register(u0);" in source
    expected = reference(layer, 1 if masked else 2)
    got, result = T.run(directory, consumer, env, mode, layout)
    assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
    debug = T.clean(result)
    wrong = sum(a != b for a, b in zip(got, expected))
    assert wrong == 0, (layer, masked, wrong)
    print(f"{adapter}: relative layer {layer}, physical layer {FIRST + layer}, partial EXEC={masked}: "
          f"{len(got)} words, mismatches={wrong}; debug layer: {debug}", flush=True)

    # The actual runtime must refuse a plain 2D view of the same format before dispatch.
    got, result = T.run(directory, consumer, env, "texture-array-2d", "<%dI" % WORDS)
    assert got == reference(layers=1)
    refusal = T.lines(result, "Dispatch(2, 1, 1) refused:")
    assert len(refusal) == 1 and refusal[0].endswith("an output texture view is not a 2D array"), result.stderr[-3000:]
    assert not T.lines(result, "Dispatch(2, 1, 1) translated")
    print(f"{adapter}: incompatible 2D UAV refused; {len(got)} words unchanged", flush=True)

    # Negative controls run through the same production binding, not the offscreen shortcut.
    guard = "if (exec & 1) " + A.ARRAY_LINE
    if masked:
        assert source.count(guard) == 1
        mutant = source.replace(guard, A.ARRAY_LINE)
        label, count = "dropped EXEC", 32
    else:
        index = "uint3(v8, v9, v10)"
        assert source.count(index) == 1
        delta = "+ 1u" if layer == 1 else "- 1u"
        mutant = source.replace(index, f"uint3(v8, v9, v10 {delta})")
        label, count = "wrong layer", 128
    hlsl.write_text(mutant, encoding="ascii")
    subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
    got, result = T.run(directory, consumer, env, mode, layout)
    assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
    T.clean(result)
    wrong = sum(a != b for a, b in zip(got, expected))
    assert wrong == count, (layer, label, wrong, count)
    print(f"{adapter}: {label} control: {len(got)} words, mismatches={wrong} (expected {count})", flush=True)


def main(consumer, dxc, adapter):
    with tempfile.TemporaryDirectory() as temp:
        for layer in (1, 3):
            for masked in (False, True):
                check(Path(temp) / f"layer-{layer}-{masked}", consumer, dxc, adapter, layer, masked)
    print(f"startup_compute_dispatch_array: PASS on {adapter}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in ("warp", "hardware"):
        sys.exit(__doc__)
    main(*sys.argv[1:])
