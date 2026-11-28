"""Startup FP32 and packed D16 array UAV binding through the production slot-67 / slot-14 path (#2742)."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_compute_image_store_check as A
import startup_compute_image_store_d16_check as D
import startup_compute_dispatch_texture_check as T
from startup_compute_request_check import DATA, prepare
from startup_compute_translate_check import state_a

WIDTH, HEIGHT, LAYERS, FIRST = 32, 8, 6, 1
WORDS = WIDTH * HEIGHT * 4


def reference(layer=None, rows=2, layers=LAYERS, d16=False, values=None):
    result = ([0x5000 + i for i in range(layers * WORDS)] if d16 else
              [0xbf000000 | (z << 12) | i for z in range(layers) for i in range(WORDS)])
    if values is None:
        values = D.ARRAY_VALUES if d16 else A.ARRAY_VALUES
    if layer is not None:
        for y in range(rows):
            for x in range(4 * A.fixture.GROUPS):
                at = (FIRST + layer) * WORDS + (y * WIDTH + x) * 4
                result[at:at + 4] = values
    return result


def check(directory, consumer, dxc, adapter, layer, masked, d16=False):
    cache, env = prepare(directory)
    env.update(X2W_D3D12_LIST_NOOP="1", X2W_D3D12_DEBUG="1")
    if adapter == "hardware":
        env["X2W_TEST_DISPATCH_HARDWARE"] = "1"
    (directory / "compute_root").write_bytes(T.C.root_blob(3, table_range=True))
    env["X2W_TEST_COMPUTE_ROOT"] = str(directory / "compute_root")
    program = D.array_program(layer, masked) if d16 else A.array_program(layer)
    if masked and not d16:
        assert program.count(A.ARRAY_STORE) == 1
        program = program.replace(A.ARRAY_STORE, A.MASK + A.ARRAY_STORE)
    (directory / "program").write_bytes(program)
    (directory / "data").write_bytes(DATA)
    (directory / "state").write_bytes(state_a(A.fixture.GROUP, A.RSRC2))
    layout = ("<%dH" if d16 else "<%dI") % (WORDS * LAYERS)
    mode = "texture-array-d16" if d16 else "texture-array"
    unit = "halves" if d16 else "words"
    store = D.ARRAY_LINE if d16 else A.ARRAY_LINE
    got, result = T.run(directory, consumer, env, mode, layout)
    print(result.stdout, end="")
    assert got == reference(d16=d16), "untranslated dispatch wrote the texture"
    assert len(T.lines(result, "Dispatch(2, 1, 1) refused: no translation")) == 1, result.stderr[-3000:]
    key = int(T.lines(result, "x2w-draw compute request")[0].split()[3], 16)
    T.translate(cache, dxc, key)
    manifest = (cache / f"{key:016x}.txt").read_text()
    assert "uav 0 tex2da float table 1 64 channels 4" in manifest.splitlines(), manifest
    hlsl = cache / f"{key:016x}.cs.hlsl"
    cso = cache / f"{key:016x}.cs.cso"
    source = hlsl.read_text()
    assert "RWTexture2DArray<float4> w0 : register(u0);" in source
    expected = reference(layer, 1 if masked else 2, d16=d16)
    assert store in source
    assert sum(a != b for a, b in zip(reference(d16=d16), expected, strict=True)) == (32 if masked else 64)
    got, result = T.run(directory, consumer, env, mode, layout)
    assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
    debug = T.clean(result)
    wrong = sum(a != b for a, b in zip(got, expected))
    assert wrong == 0, (layer, masked, wrong)
    print(f"{adapter}: relative layer {layer}, physical layer {FIRST + layer}, partial EXEC={masked}: "
          f"{len(got)} {unit}, mismatches={wrong}; debug layer: {debug}", flush=True)

    # The actual runtime must refuse a plain 2D view of the same format before dispatch.
    got, result = T.run(directory, consumer, env, "texture-store" if d16 else "texture-array-2d",
                        ("<%dH" if d16 else "<%dI") % WORDS)
    assert got == reference(layers=1, d16=d16)
    refusal = T.lines(result, "Dispatch(2, 1, 1) refused:")
    assert len(refusal) == 1 and refusal[0].endswith("an output texture view is not a 2D array"), result.stderr[-3000:]
    assert not T.lines(result, "Dispatch(2, 1, 1) translated")
    print(f"{adapter}: incompatible 2D UAV refused; {len(got)} {unit} unchanged", flush=True)

    # Negative controls run through the same production binding, not the offscreen shortcut.
    guard = "if (exec & 1) " + store
    if masked:
        assert source.count(guard) == 1
        cases = [("dropped EXEC", source.replace(guard, store), reference(layer, d16=d16), 32)]
    else:
        index = "uint3(v8, v9, v10)"
        assert source.count(index) == 1
        delta = "+ 1u" if layer == 1 else "- 1u"
        cases = [("wrong layer", source.replace(index, f"uint3(v8, v9, v10 {delta})"),
                  reference(2, d16=d16), 128)]
        if d16 and layer == 1:
            parts = ", ".join(D.PARTS)
            assert source.count(parts) == 1 and source.count(store) == 1
            for label, order, count in (("swap halves", (1, 0, 2, 3), 32),
                                        ("swap VGPRs", (2, 3, 0, 1), 64)):
                cases.append((label, source.replace(parts, ", ".join(D.PARTS[i] for i in order)),
                              reference(layer, d16=True, values=tuple(D.ARRAY_VALUES[i] for i in order)), count))
            cases.append(("omit store", source.replace(store, ""), reference(d16=True), 64))
    for label, mutant, control_reference, count in cases:
        assert mutant != source
        predicted = sum(a != b for a, b in zip(control_reference, expected, strict=True))
        assert predicted == count, (layer, label, predicted, count)
        hlsl.write_text(mutant, encoding="ascii")
        subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        got, result = T.run(directory, consumer, env, mode, layout)
        assert len(T.lines(result, "Dispatch(2, 1, 1) translated")) == 1, result.stderr[-3000:]
        T.clean(result)
        wrong = sum(a != b for a, b in zip(got, expected, strict=True))
        assert wrong == predicted and got == control_reference, (layer, label, wrong, predicted)
        print(f"{adapter}: {label} control: {len(got)} {unit}, mismatches={wrong} (expected {predicted})", flush=True)


def main(consumer, dxc, adapter, d16=False):
    with tempfile.TemporaryDirectory() as temp:
        for layer in (1, 3):
            for masked in (False, True):
                check(Path(temp) / f"layer-{layer}-{masked}", consumer, dxc, adapter, layer, masked, d16)
    print(f"startup_compute_dispatch_array{'_d16' if d16 else ''}: PASS on {adapter}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[3] not in ("warp", "hardware") or (len(sys.argv) == 5 and sys.argv[4] != "--d16"):
        sys.exit(__doc__)
    main(*sys.argv[1:4], d16=len(sys.argv) == 5)
