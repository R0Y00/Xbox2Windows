"""A stream vertex shader with data after its exit is bounded at its single reachable S_ENDPGM.

Source-built gfx1013 code only. The fixture is the full-screen vertex shader of startup_stream_interface_check
followed by data words (an alignment S_ENDPGM, a count and a table with an S_ENDPGM-valued entry) and the final
S_ENDPGM, the layout a vertex program with a PC-relative constant table has. Translator cases only:
  bounded:     pixel_aot.container_code still refuses the section without vertex_data; stream_stage_code returns
               the code up to the first S_ENDPGM, with the whole section as .image, and the pipeline translates to
               the same HLSL as the shader without the data words (the reference).
  refused:     a branch around the first exit to a second reachable one, a reachable loop after the exit, and a
               branch out of the section. A pixel stage with two exits stays refused.
"""
import os
from pathlib import Path
import struct
import sys
import tempfile

from startup_stream_interface_check import (COLOR, POSITION, PS_BASE, TEXCOORD, VS_BASE, VS_RECORDS, gfx2hlsl,
                                            container, pixel, pixel_words, signature, translate, vertex_words)

ENDPGM = 0xbf810000
DATA = [ENDPGM, 3, 0x3f800000, ENDPGM, 0x12345678, ENDPGM]


def vertex_container(words):
    table = {0x1dc: b"".join(bytes(r) for r in VS_RECORDS), 0x29c: struct.pack("<I", len(VS_RECORDS))}
    return container(1, words, VS_BASE, [signature(b"OSG1", (POSITION, COLOR, TEXCOORD))], table)


def body():
    words = vertex_words()
    assert words[-1] == ENDPGM and ENDPGM not in words[:-1]
    return words[:-1]


def sopp(op, simm):
    return 0xbf800000 | op << 16 | (simm & 0xffff)


def hlsl_of(directory, vs):
    status, _ = translate(directory, vs, pixel())
    assert status == "ok", (directory.name, status)
    return {p.suffixes[-2]: p.read_text().replace(p.name.split(".")[0], "KEY")
            for p in directory.glob("*.hlsl")}


def cases(directory):
    from pixel_aot import container_code
    from resource_inventory import parts
    plain = vertex_container(body() + [ENDPGM])
    data = vertex_container(body() + [ENDPGM] + DATA)
    program = parts(data)[b"DXIL"]
    try:
        container_code(program, stage=1)
    except ValueError as error:
        assert "multi-exit" in str(error), error
    else:
        raise AssertionError("container_code accepted a vertex section with three S_ENDPGM words")
    code = gfx2hlsl.stream_stage_code(data, 1)
    whole = container_code(program, stage=1, vertex_data=True)
    assert isinstance(code, gfx2hlsl.VertexCode) and bytes(code) == whole[:4 * len(body()) + 4], len(code)
    assert code.image == whole and whole[len(code):] == struct.pack("<%dI" % len(DATA), *DATA)
    assert struct.unpack_from("<I", whole, len(whole) - 4)[0] == ENDPGM
    print("bounded: the code ends at the first S_ENDPGM (+0x%x); the image keeps %d data words"
          % (len(code) - 4, (len(whole) - len(code)) // 4))
    reference, bounded = hlsl_of(directory / "plain", plain), hlsl_of(directory / "data", data)
    assert reference and reference == bounded, sorted(reference)
    print("bounded: the pipeline translates to the HLSL of the shader without the data (%s)" % ", ".join(sorted(bounded)))

    pre, tail = body()[:-6], body()[-6:]   # tail: the three exports
    for label, words, want in (
            ("second exit", pre + [sopp(5, 1), ENDPGM] + tail + [ENDPGM] + DATA,
             "vertex code: 2 reachable S_ENDPGM"),
            ("loop after the exit", pre + [sopp(5, len(tail) + 1)] + tail + [ENDPGM, sopp(2, -1)] + DATA,
             "vertex code: reachable instruction after the exit"),
            ("branch out of the section", pre + [sopp(5, 0x4000)] + tail + [ENDPGM] + DATA,
             "vertex code: transfer to +0x%x outside the section" % (4 * len(pre) + 4 + 4 * 0x4000))):
        status, _ = translate(directory / label.replace(" ", "_"), vertex_container(words), pixel())
        assert status == "unsupported: " + want, (label, status)
        print("refused: %s" % label)
    two = container(0, pixel_words()[:-1] + [ENDPGM, ENDPGM] + DATA, PS_BASE, [], {})
    try:
        container_code(parts(two)[b"DXIL"], stage=0, vertex_data=True)
    except ValueError as error:
        assert "multi-exit" in str(error), error
        print("refused: a pixel stage with data after its exit (unchanged)")
    else:
        raise AssertionError("container_code accepted a pixel section with data after its exit")
    return 0


if __name__ == "__main__":
    saved = os.environ.get("X2W_FXC")
    os.environ["X2W_FXC"] = ""
    try:
        with tempfile.TemporaryDirectory() as temp:
            sys.exit(cases(Path(temp)))
    finally:
        if saved is not None:
            os.environ["X2W_FXC"] = saved
