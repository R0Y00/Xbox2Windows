"""A stream pixel shader that reads an SPI input VGPR takes its layout from the FSSM header's SPI_PS_INPUT_ENA.

Source-built gfx1013 code only. The vertex shader is startup_stream_interface_check's full-screen triangle. The
pixel shader reads entry v2 and writes red = v2 / 8. Its header holds ENA 0xf02 at +0x248: PERSP_CENTER (v0, v1),
then POS_X..POS_W (v2..v5), so v2 is SV_Position.x. A slot-86 request carries no .psin, so before this the read was
refused.

Without arguments only the translator cases run:
  lowered:   v2 reads the pixel position x.
  same:      a pixel stage that reads no SPI input translates byte for byte the same with and without the header
             word (the layout is only used when a read would otherwise be refused).
  refused:   no header word; a word whose position bits miss the SV_Position read mask; FRONT_FACE without a
             declared SV_IsFrontFace.
With a consumer and a backend the draw runs on an 8x4 target (left half drawn):
  draw:      red follows (x + 0.5) / 8 per column (16, 48, 80, 112, each within 1). This is the CPU reference:
             SV_Position.x is the pixel centre.
  shifted:   the same shader with ENA 0xf00 (PERSP_CENTER dropped, so v2 becomes SV_Position.z; a mutation the
             header check accepts) fails the gradient check.
"""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_stream_draw_check import vop1, vop2
from startup_stream_interface_check import (COLOR, EMPTY_ROOT, POSITION, PS_BASE, PS_RECORDS, TEXCOORD, check_pixels,
                                            container, expected_red, gfx2hlsl, pixel, pixel_words, signature, vertex)

FRONT_FACE = ("SV_ISFRONTFACE", 0, 1, 9, 0x1)


def reading_pixel(ena=0xf02, inputs=(POSITION,)):
    words = [vop2(8, 2, 255, 2), 0x3e000000,           # v_mul_f32 v2, 0.125, v2
             vop1(1, 6, 128), vop1(1, 7, 242),          # v6 = 0, v7 = 1.0
             0xf800180f, 0x07060602, 0xbf810000]         # exp mrt0 v2, v6, v6, v7 done vm
    table = {0x238: struct.pack("<I", 0)}
    if ena is not None:
        table[0x248] = struct.pack("<I", ena)
    return container(0, words, PS_BASE, [signature(b"ISG1", inputs)], table)


def translate(directory, vs, ps):
    directory.mkdir(parents=True)
    key = gfx2hlsl.pipeline_key(vs, ps, EMPTY_ROOT)
    for ext, data in ((".a", vs), (".b", ps), (".root", EMPTY_ROOT)):
        (directory / ("req_%016x%s" % (key, ext))).write_bytes(data)
    status = gfx2hlsl.translate_request(str(directory), key)
    hlsl = directory / ("%016x.ps.hlsl" % key)
    return status, hlsl.read_text() if hlsl.exists() else ""


def pixel_with_ena(ena):
    """startup_stream_interface_check.pixel() with the header word at +0x248 set."""
    table = {0x138: b"".join(bytes(r) for r in PS_RECORDS), 0x238: struct.pack("<I", len(PS_RECORDS)),
             0x248: struct.pack("<I", ena)}
    return container(0, pixel_words(0), PS_BASE, [signature(b"ISG1", (POSITION, COLOR, TEXCOORD))], table)


def translate_cases(directory):
    status, ps = translate(directory / "position", vertex(), reading_pixel())
    assert status == "ok", status
    assert "v2 = U(i.pos.x);  // SPI input pos_x_float" in ps, ps
    print("translate: v2 reads SV_Position.x from header ENA 0xf02")
    _, plain = translate(directory / "plain", vertex(), pixel())
    _, worded = translate(directory / "plain_word", vertex(), pixel_with_ena(0xf02))
    assert plain and plain == worded, "a stage without SPI input reads changed with the header word"
    print("translate: a stage that reads no SPI input is unchanged by the header word")
    refused = "unsupported: pixel stage reads entry VGPR v2 (an SPI input; layout not in the request) at 0x0"
    for label, ps in (("no word", reading_pixel(ena=None)), ("position bits miss the read mask", reading_pixel(0x302)),
                      ("front face undeclared", reading_pixel(0x1f02))):
        status, _ = translate(directory / label.replace(" ", "_"), vertex(), ps)
        assert status == refused, (label, status)
        print("translate: %s refused" % label)
    status, _ = translate(directory / "front_face_declared", vertex(), reading_pixel(0x1f02, (POSITION, FRONT_FACE)))
    assert status == "ok", status
    print("translate: FRONT_FACE with a declared SV_IsFrontFace accepted")
    return 0


def translate_checks(directory):
    saved = os.environ.get("X2W_FXC")
    os.environ["X2W_FXC"] = ""
    try:
        return translate_cases(directory)
    finally:
        if saved is None:
            os.environ.pop("X2W_FXC", None)
        else:
            os.environ["X2W_FXC"] = saved


def draw(directory, consumer, backend, ena):
    cache = directory / "cache"
    cache.mkdir(parents=True)
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(reading_pixel(ena))
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    subprocess.run(args, env=env, check=True)
    key = int(sorted(cache.glob("req_*.a"))[0].name[4:20], 16)
    status = gfx2hlsl.translate_request(str(cache), key)
    assert status == "ok", status
    result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if backend == "hardware" and result.returncode == 77:
        return None
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")
    return output.read_bytes()


def check(directory, consumer, backend):
    pixels = draw(directory / "position", consumer, backend, 0xf02)
    if pixels is None:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    difference = check_pixels(pixels)
    if difference:
        raise AssertionError("position: " + difference)
    print(f"{backend} draw: red follows SV_Position.x, {[expected_red(x) for x in range(4)]} within 1")
    pixels = draw(directory / "shifted", consumer, backend, 0xf00)
    if not check_pixels(pixels):
        raise AssertionError("ENA without PERSP_CENTER passed the gradient check")
    print(f"{backend} shifted layout: the gradient check fails (red {pixels[0]} in column 0)")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        result = translate_checks(Path(temp) / "translate")
        if len(sys.argv) > 1:
            result = check(Path(temp), *sys.argv[1:])
        sys.exit(result)
