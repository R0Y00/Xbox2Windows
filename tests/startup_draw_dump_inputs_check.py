"""X2W_DRAW_DUMP_INPUTS=<key>[:occurrence]: the inputs and the render target of one translated draw go to files.

Source-built gfx1013 code only: startup_stream_ps_inputs_check's draw (full-screen triangle, the pixel shader writes
red = SV_Position.x / 8 to the left half of an 8x4 RGBA8 target that is cleared to blue). The consumer never
presents, so it writes the dump through the shim's test export after its draw (x2w_draw_dump_inputs_test(1)).
  off:        without the switch: no dddump files, and the pixels are the reference pixels.
  dump:       X2W_DRAW_DUMP_INPUTS=<key>: the index names the plain draw (count 3), the entry values and the root
              constants; rtv0_pre is the cleared target (all blue); rtv0_post equals the pixels the consumer read back
              after the draw, byte for byte; the pixels are the same as with the switch off.
  occurrence: X2W_DRAW_DUMP_INPUTS=<key>:2 with one draw: nothing is selected and nothing is written (the consumer
              returns 63, its "no dump written" code).
  buffers:    startup_stream_fetch_check's two draws of one key (plain from vertex 3, then indexed: indices 4-9, base
              vertex 5) with X2W_DRAW_DUMP_INPUTS=<key>:1,<key>:2 (two entries of one key count that key's draws once):
              o1 is the plain draw, o2 the indexed one. Each vertex stream srv file holds exactly the six vertices the
              draw reads (vertices 3-8, then 9-14) of the fixture's two float3 streams, the indexed dump's indices file
              is the six u16 indices, and the pixels are the same as without the switch.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import struct

from startup_stream_interface_check import HEIGHT, WIDTH, check_pixels, gfx2hlsl, vertex
from startup_stream_ps_inputs_check import reading_pixel
import startup_stream_fetch_check as fetch

BLUE = bytes((0, 0, 255, 255))


def run(directory, consumer, backend, dump=None):
    cache = directory / "cache"
    out = directory / "dump"
    cache.mkdir(parents=True)
    out.mkdir()
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(reading_pixel(0xf02))
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache), X2W_DRAW_DUMP_INPUTS_DIR=str(out))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    env.pop("X2W_DRAW_DUMP_INPUTS", None)
    subprocess.run(args, env=env, check=True)
    key = int(sorted(cache.glob("req_*.a"))[0].name[4:20], 16)
    status = gfx2hlsl.translate_request(str(cache), key)
    assert status == "ok", status
    if dump is not None:
        env["X2W_DRAW_DUMP_INPUTS"] = dump % key
    result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if backend == "hardware" and result.returncode == 77:
        return None
    pixels = output.read_bytes() if output.exists() else None
    return result, key, pixels, sorted(out.iterdir())


def fixture_streams():
    """tests/startup_stream_draw.cpp X2W_TEST_VERTEX_FETCH: the TEXCOORD (slot 0) and POSITION (slot 1) float3 arrays."""
    texcoords, positions = [[0.0] * 3 for _ in range(15)], [[0.0] * 3 for _ in range(15)]

    def corners(out, first, x0, y0, x1, y1):
        for v, (x, y) in enumerate(((x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1))):
            out[first + v] = [x, y, 0.5]
    corners(texcoords, 3, -1.0, 1.0, 0.0, 0.5)
    corners(positions, 3, -0.75, 0.5, -0.25, -0.5)
    corners(texcoords, 9, -0.25, -0.5, 0.0, -1.0)
    corners(positions, 9, -1.0, 1.0, -0.75, -1.0)
    pack = lambda rows: b"".join(struct.pack("<3f", *r) for r in rows)
    return pack(texcoords), pack(positions)


def check_buffers(directory, consumer, backend):
    os.environ["X2W_UPSTREAM_STREAM_LAYOUT"] = "1"  # as startup_stream_fetch_<backend>: the request carries the layout
    run = fetch.Run(directory / "fetch_off", consumer, backend, fetch.vertex(), True)
    assert gfx2hlsl.translate_request(str(run.cache), run.key) == "ok"
    result = run.draw()
    assert not result.returncode and result.stderr.count("-> drawn;") >= 2, result.stderr[-3000:]
    reference = run.output.read_bytes()
    run = fetch.Run(directory / "fetch_dump", consumer, backend, fetch.vertex(), True)
    assert gfx2hlsl.translate_request(str(run.cache), run.key) == "ok"
    out = directory / "fetch_files"
    out.mkdir()
    run.env.update(X2W_DRAW_DUMP_INPUTS="%016x:1,%016x:2" % (run.key, run.key), X2W_DRAW_DUMP_INPUTS_DIR=str(out))
    result = run.draw()
    assert not result.returncode, (result.returncode, result.stdout, result.stderr[-3000:])
    assert run.output.read_bytes() == reference, "the dump changed the fetch draws' pixels"
    names = {f.name: f for f in out.iterdir()}
    index = sorted(n for n in names if n.endswith(".txt"))
    assert len(index) == 2 and "_o1_" in index[0] and "_o2_" in index[1], index
    texcoords, positions = fixture_streams()
    for name, first, draw_line in ((index[0], 3, "draw plain count 6 instances 1 start 3 "),
                                   (index[1], 9, "draw indexed count 6 instances 1 start 0 base_vertex 5 ")):
        stem = name[:-4]
        lines = names[name].read_text().splitlines()
        assert any(l.startswith(draw_line) for l in lines), (draw_line, lines)
        assert "vertices first %d count 6 topology 0x4" % first in lines, lines
        streams = [l for l in lines if l.startswith("file ") and " srv " in l and " buffer source 1(" in l]
        assert len(streams) == 2, lines
        for l in streams:
            slot = int(l.split(" buffer source 1(")[1].split(",")[0])
            data = names[l.split()[1]].read_bytes()
            want = (texcoords, positions)[slot][12 * first:12 * (first + 6)]
            assert data == want, (name, slot, data.hex(), want.hex())
            assert "bytes 72 of 72" in l and ": copied at record time, 72 bytes bound" in l, l
        if first == 9:
            assert names[stem + "_indices.bin"].read_bytes() == struct.pack("<6H", 4, 5, 6, 7, 8, 9)
    print(f"{backend} buffers: o1 = plain draw, o2 = indexed draw of one key; both vertex streams hold exactly the "
          "vertices each draw reads; indices exact; pixels unchanged")


def check(directory, consumer, backend):
    off = run(directory / "off", consumer, backend)
    if off is None:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    result, key, reference, files = off
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")
    assert not check_pixels(reference), check_pixels(reference)
    assert not files, files
    print(f"{backend} off: no dump files; reference pixels pass the gradient check")

    result, key, pixels, files = run(directory / "dump", consumer, backend, "%016x")
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"dump draw failed on {backend}: {result.returncode}\n{result.stdout}\n{result.stderr}")
    assert pixels == reference, "the dump changed the drawn pixels"
    names = {f.name: f for f in files}
    index = [n for n in names if n.endswith(".txt")]
    assert len(index) == 1, list(names)
    stem = index[0][:-4]
    assert stem.startswith("dddump_%016x_o1_f" % key), stem
    lines = names[index[0]].read_text().splitlines()
    assert lines[0] == "x2w-dddump 1 " + stem, lines[0]
    assert any(l.startswith("draw plain count 3 instances 1 ") for l in lines), lines
    assert any(l.startswith("entry ") for l in lines) and stem + "_root_constants.bin" in names, lines
    assert (names[stem + "_root_constants.bin"].stat().st_size == 16 * 8 * 4)
    tex = [l for l in lines if l.startswith("tex rtv0 ")]
    assert len(tex) == 2 and " pre " in tex[0] and " post " in tex[1], tex
    assert all(" %ux%ux1 " % (WIDTH, HEIGHT) in l and " fmt 28 " in l for l in tex), tex
    pre = names[stem + "_rtv0_pre.bin"].read_bytes()
    post = names[stem + "_rtv0_post.bin"].read_bytes()
    assert pre == BLUE * (WIDTH * HEIGHT), pre.hex()
    assert post == reference, (post.hex(), reference.hex())
    print(f"{backend} dump: index, entry and root constants written; rtv0 before the draw is the cleared target, "
          "after it equals the read-back pixels byte for byte; pixels unchanged")

    result, key, pixels, files = run(directory / "second", consumer, backend, "%016x:2")
    assert result.returncode == 63, (result.returncode, result.stdout, result.stderr)
    assert not files, files
    assert pixels == reference, "a non-selected dump changed the drawn pixels"
    print(f"{backend} occurrence 2 of one draw: nothing selected, nothing written")
    check_buffers(directory, consumer, backend)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: startup_draw_dump_inputs_check.py <consumer> <backend>")
        sys.exit(2)
    with tempfile.TemporaryDirectory() as temp:
        sys.exit(check(Path(temp), *sys.argv[1:]))
