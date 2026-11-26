"""A stream vertex shader reads a constant table after its exit through S_GETPC_B64 and GLOBAL_LOAD.

Source-built gfx1013 code only. The vertex shader is the full-screen triangle of startup_stream_interface_check.
It computes the signed word index i = 3 * vertex_id - 3, sign-extends it to 64 bits and shifts it left by 2, adds
the code address s_getpc_b64 + s_add_u32 / s_addc_u32 point at (four words into the table), and loads two words
with global_load_dwordx2 into param0 (TEXCOORD.xy). The table follows the exit: an alignment S_ENDPGM, a count,
then 12 words in which words 1, 4 and 7 are 0.75 and words 2, 5 and 8 are 0.25. Every other word differs (word 0 is
the S_ENDPGM word), so reading one word off, missing the carry of vertex 0's negative index, or taking the address
from the wrong instruction changes the colour. The pixel shader writes TEXCOORD.xy to red and green.

Without arguments only the translator cases run (no consumer, device or compiler):
  lowered:  the address pair is the constant virtual address (the image at address 0) and the image is the HLSL
            table; the expected words come from the fixture's bytes (the CPU reference below).
  refused:  the address read as data, a load whose high word comes from another addition, an EXEC write while a
            VGPR holds an address word, s_getpc_b64 without the displacement pair, a global_load through an
            address that is not the code address, a load offset modifier, and a stage given the code without the
            data after its exit.
With a consumer and a backend the draw runs first on an 8x4 target (left half drawn):
  draw:     red 191 (0.75) and green 64 (0.25), each within 1.
  shifted:  the same pipeline with the address one word lower (a mutation) fails the colour check.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from startup_stream_draw_check import vop1, vop2
from startup_stream_interface_check import (EMPTY_ROOT, HEIGHT, POSITION, PS_BASE, PS_RECORDS, VS_BASE, WIDTH,
                                            container, gfx2hlsl, signature, vertex_words)

ENDPGM = 0xbf810000
TEXCOORD = ("TEXCOORD", 0, 2, 0, 0x3)
VS_RECORDS = [(6, 2, 0, 0, 0, 0)]       # TEXCOORD -> param0
F = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
TABLE = [ENDPGM if j == 0 else F(0.75) if j % 3 == 1 and j < 9 else F(0.25) if j % 3 == 2 and j < 9
         else F(0.5 + j / 64) for j in range(12)]   # words 9-11 differ too: three words too high is caught
BIAS = 4                                 # the address the pair forms is table word 4


def vop3(op, vdst, src0, src1, sdst=0):
    return [0xd4000000 | op << 16 | sdst << 8 | vdst, src0 | src1 << 9]


GETPC = 0xbe901f00                       # s_getpc_b64 s[16:17]
ADDC = 0x82118011                        # s_addc_u32 s17, s17, 0
LOAD_LO = [0xd70f6a07, 0x107 | 0x10 << 9]            # v_add_co_u32 v7, vcc_lo, v7, s16
LOAD_HI = [vop2(0x28, 8, 0x11, 8)]                    # v_add_co_ci_u32 v8, vcc_lo, s17, v8, vcc_lo
LOAD = [0xdc348000, 0x057d0007]                       # global_load_dwordx2 v[5:6], v[7:8], off


def vertex_code(variant="table"):
    """-> (instruction words with the displacement filled in, offset of s_getpc_b64)."""
    words = vertex_words()[:11]                       # v1..v4 = the clip position
    words += [vop2(0x0b, 7, 0x83, 0),                 # v_mul_u32_u24 v7, 3, v0
              vop2(0x25, 7, 0xc3, 7),                 # v_add_nc_u32 v7, -3, v7
              vop2(0x18, 8, 0x9f, 7)]                 # v_ashrrev_i32 v8, 31, v7
    words += vop3(0x2ff, 7, 0x82, 256 + 7)            # v_lshlrev_b64 v[7:8], 2, v[7:8]
    getpc = 4 * len(words)
    pair = [GETPC, 0x8010ff10, None, ADDC]            # s_add_u32 s16, s16, <displacement>
    address = LOAD_LO + LOAD_HI
    if variant == "escape":
        pair += [vop1(1, 9, 16)]                      # v_mov_b32 v9, s16
    elif variant == "no pair":
        pair = [GETPC, 0xbf800000, 0xbf800000, 0xbf800000]   # s_nop
    elif variant == "other carry":
        address = LOAD_LO + [0xd70f6a09, 0x109 | 0x10 << 9] + LOAD_HI   # v_add_co_u32 v9, vcc_lo, v9, s16
    elif variant == "exec":
        address += [0xbefe03c1]                       # s_mov_b32 exec_lo, -1
    elif variant == "plain load":
        pair, address = [], []
    load = [LOAD[0] | 4, LOAD[1]] if variant == "offset" else LOAD   # offset:4
    words += pair + address + load + [0xbf8c3f70,     # s_waitcnt vmcnt(0)
                                      0xf80000cf, 0x04030201,     # exp pos0 v1, v2, v3, v4
                                      0xf8000a0f, 0x06060605,     # exp param0 v5, v6, v6, v6 done
                                      ENDPGM]
    table = 4 * (len(words) + 2)                      # after the exit: alignment S_ENDPGM, count
    if None in words:
        words[words.index(None)] = (table + 4 * BIAS - (getpc + 4)) & 0xffffffff
    return words, getpc


def vertex(variant="table"):
    words, _ = vertex_code(variant)
    image = {0x1dc: b"".join(bytes(r) for r in VS_RECORDS), 0x29c: struct.pack("<I", len(VS_RECORDS))}
    return container(1, words + [ENDPGM, len(TABLE)] + TABLE + [ENDPGM], VS_BASE,
                     [signature(b"OSG1", (POSITION, TEXCOORD))], image)


def pixel():
    words = [0xc8080000, 0xc8090001, 0xc80c0100, 0xc80d0101,   # v2, v3 = attr0.x, attr0.y
             vop1(1, 4, 128), vop1(1, 5, 242),                 # v4 = 0, v5 = 1.0
             0xf800180f, 0x05040302, ENDPGM]                   # exp mrt0 v2, v3, v4, v5 done vm
    image = {0x138: b"".join(bytes(r) for r in PS_RECORDS), 0x238: struct.pack("<I", len(PS_RECORDS))}
    return container(0, words, PS_BASE, [signature(b"ISG1", (POSITION, TEXCOORD))], image)


def reference():
    """The CPU reference: each vertex's 64-bit address, wrapped as the hardware adds it, read from the fixture."""
    words, getpc = vertex_code()
    image = struct.pack("<%dI" % (len(words) + 2 + len(TABLE) + 1), *words, ENDPGM, len(TABLE), *TABLE, ENDPGM)
    base = (getpc + 4 + struct.unpack("<i", struct.pack("<I", words[getpc // 4 + 2]))[0]) % 2 ** 64
    values = []
    for vertex_id in range(3):
        index = 3 * vertex_id - 3
        address = (base + (index % 2 ** 64) * 4) % 2 ** 64
        values.append(tuple(struct.unpack_from("<f", image, address + 4 * k)[0] for k in range(2)))
    return base, image, values


def translate(directory, vs, ps):
    directory.mkdir(parents=True)
    key = gfx2hlsl.pipeline_key(vs, ps, EMPTY_ROOT)
    for ext, data in ((".a", vs), (".b", ps), (".root", EMPTY_ROOT)):
        (directory / ("req_%016x%s" % (key, ext))).write_bytes(data)
    status = gfx2hlsl.translate_request(str(directory), key)
    hlsl = directory / ("%016x.vs.hlsl" % key)
    return status, hlsl.read_text() if hlsl.exists() else ""


def translate_cases(directory):
    base, image, values = reference()
    assert values == [(0.75, 0.25)] * 3, values
    status, vs = translate(directory / "table", vertex(), pixel())
    assert status == "ok", status
    assert re.search(r"\bs16 = 0x%08xu;" % base, vs) and re.search(r"\bs17 = 0x00000000u;", vs), vs
    table = re.search(r"static const uint x2w_pcimage\[(\d+)\] = \{([^}]*)\};", vs)
    assert table and int(table.group(1)) == len(image) // 4, vs
    assert [int(w, 16) for w in re.findall(r"0x([0-9a-f]+)u", table.group(2))] == list(
        struct.unpack("<%dI" % (len(image) // 4), image)), "the HLSL table is not the image"
    assert vs.count("x2w_pcimage[min(") == 2, vs
    print("translate: the pair is virtual address %#x (table word %d); the HLSL holds the %d-word image" % (
        base, BIAS, len(image) // 4))
    words, getpc = vertex_code()
    for label, want in (
            ("escape", "s_getpc_b64 data pointer: s16 (lo) is read by v_mov_b32_e32 at +0x%x" % (getpc + 16)),
            ("other carry", "s_getpc_b64 data pointer: load global_load_dwordx2 v[5:6], v[7:8], off is not through "
                            "one proven image address at +0x%x" % (getpc + 36)),
            ("exec", "s_getpc_b64 data pointer: s_mov_b32 writes EXEC while a VGPR holds an address word at +0x%x"
             % (getpc + 28)),
            ("no pair", "s_getpc_b64: the shader load address is only elided for a program that also calculates a "
                        "branch target"),
            ("plain load", "global_load without a proven s_getpc_b64 image address"),
            ("offset", "s_getpc_b64 data pointer: load global_load_dwordx2 v[5:6], v[7:8], off offset:4 is not "
                       "through one proven image address at +0x%x" % (getpc + 28))):
        status, _ = translate(directory / label.replace(" ", "_"), vertex(label), pixel())
        assert status == "unsupported: " + want, (label, status)
        print("translate: %s refused" % label)
    # The table is only in the bounded section's .image: a loader that kept the code alone must refuse, not read 0.
    original = gfx2hlsl.stream_stage_code
    gfx2hlsl.stream_stage_code = lambda blob, stage: bytes(original(blob, stage))
    try:
        status, _ = translate(directory / "lost image", vertex(), pixel())
    finally:
        gfx2hlsl.stream_stage_code = original
    assert status == "unsupported: global_load: the s_getpc_b64 address %#x is outside the %d-byte shader image" % (
        base, 4 * len(words)), status
    print("translate: a stage without the image refused")
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


WANT = (191, 64, 0, 255)


def check_pixels(pixels):
    for y in range(HEIGHT):
        for x in range(WIDTH):
            got = tuple(pixels[(y * WIDTH + x) * 4:(y * WIDTH + x) * 4 + 4])
            want = WANT if x < WIDTH // 2 else (0, 0, 255, 255)
            if any(abs(g - w) > 1 for g, w in zip(got, want)):
                return "pixel (%d, %d) is %s, expected %s" % (x, y, got, want)
    return None


def draw(directory, consumer, backend, shifted):
    cache = directory / "cache"
    cache.mkdir(parents=True)
    vs, ps, root, output = (directory / name for name in ("vs", "ps", "root", "pixels"))
    vs.write_bytes(vertex())
    ps.write_bytes(pixel())
    args = [consumer, vs, ps, root]
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1", X2W_D3D12_RENDER="1",
               X2W_SHADER_CACHE=str(cache))
    env.setdefault("X2W_D3D12_DEBUG", "0")
    subprocess.run(args, env=env, check=True)
    key = int(sorted(cache.glob("req_*.a"))[0].name[4:20], 16)
    original = gfx2hlsl.pc_data_proof
    if shifted:
        def lower(ins):
            return {off: (r[0], r[1], r[2] - 4) if r[0] == 'lo' else r for off, r in original(ins).items()}
        gfx2hlsl.pc_data_proof = lower
    try:
        status = gfx2hlsl.translate_request(str(cache), key)
    finally:
        gfx2hlsl.pc_data_proof = original
    assert status == "ok", status
    result = subprocess.run(args + [backend, output], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    if backend == "hardware" and result.returncode == 77:
        return None
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"draw failed on {backend}: {result.returncode}\n{result.stderr}")
    return output.read_bytes()


def check(directory, consumer, backend):
    pixels = draw(directory / "table", consumer, backend, False)
    if pixels is None:
        print("No D3D12 feature-level 12 hardware adapter")
        return 77
    difference = check_pixels(pixels)
    if difference:
        raise AssertionError("table: " + difference)
    print(f"{backend} draw: TEXCOORD.xy from the table, {WANT} within 1")
    pixels = draw(directory / "shifted", consumer, backend, True)
    if not check_pixels(pixels):
        raise AssertionError("an address one word lower passed the colour check")
    print(f"{backend} shifted address: the colour check fails ({tuple(pixels[:4])} at (0, 0))")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp:
        result = translate_checks(Path(temp) / "translate")
        if len(sys.argv) > 1:
            result = check(Path(temp), *sys.argv[1:])
        sys.exit(result)
