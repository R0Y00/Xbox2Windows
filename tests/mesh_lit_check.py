"""Light an indexed triangle with an interpolated PARAM0 normal and a translated N.L pixel program."""

from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_attribute_check import program
from mesh_entry_check import binary, enumerate_pcs, fixture
from mesh_fallback_check import compile_fallback, fallback_fixture
from mesh_indexed_check import HEADER, SMALL, WORDS, inputs, lower
from pixel_aot import disassemble, lower as lower_pixel, native_program
from pixel_aot_check import compile_shader, envelope

POSITIONS = [SMALL[i] for i in (1, 2, 0)]
NORMAL_Z = Fraction(1, 4)
LIGHT_OFFSET = 0x80  # params[8] of the pixel stage's b0, unused by the indexed mesh program
# Light direction and ambient term. Every light component is a power of two and N.L is a multiple
# of 1/128 at every pixel center, so with an ambient of exactly 1/4 each result is itself a binary16
# value. An interpolated normal one or two binary32 ulps low (up to 2^-23 observed on hardware, 0 on
# WARP) then loses a whole binary16 step in the round-toward-zero export. The ambient 2^-17 above
# 1/4 keeps every positive result strictly between two binary16 values, at least 2^-17 from either,
# and exports the same binary16 value as the exact sum without the offset. export16() asserts it.
AMBIENT = Fraction(1, 4) + Fraction(1, 2 ** 17)
LIGHT = (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 16), AMBIENT)
SIDE_LIGHT = (Fraction(-1, 4), Fraction(-1, 2), Fraction(1, 16), AMBIENT)
INVERTED = (-LIGHT[0], -LIGHT[1], -LIGHT[2], LIGHT[3])
NORMAL = ["v_mul_f32_e32 v12, s4, v8", "v_fmac_f32_e32 v12, s5, v9", "v_fmac_f32_e32 v12, s6, v10"]
CLAMP = "v_max_f32_e32 v12, 0, v12"


def pixel_program(normal=NORMAL, clamp=True):
    """Interpolate the normal, load the light from b0 and export max(N.L, 0) + ambient as gray."""
    code = ["s_mov_b32 s30, s14", "s_mov_b32 m0, s30"]
    for phase in (1, 2):
        for i, channel in enumerate("wzxy"):
            code.append(f"v_interp_p{phase}_f32_e32 v{8 + i}, v{phase - 1}, attr0.{channel}")
    code += ["s_mov_b32 s40, s8", "s_mov_b32 s41, 0", "s_load_dwordx4 s[12:15], s[40:41], null",
             "s_waitcnt lgkmcnt(0)", f"s_buffer_load_dwordx4 s[4:7], s[12:15], {LIGHT_OFFSET:#x}",
             "s_waitcnt lgkmcnt(0)", *normal, *([CLAMP] if clamp else []), "v_add_f32_e32 v12, s7, v12",
             "v_cvt_pkrtz_f16_f32_e32 v13, v12, v12", "v_cvt_pkrtz_f16_f32_e32 v14, v12, v11",
             "exp mrt0 v13, v13, v14, v14 done compr vm", "s_endpgm"]
    return code


def translate(code):
    return lower_pixel(code, 8, 14, None, 0)


def unorm8(value):
    scaled = value * 255
    assert abs(scaled - round(scaled)) != Fraction(1, 2)
    return min(max(round(scaled), 0), 255)


def export16(value):
    """v_cvt_pkrtz_f16_f32: round toward zero. Positive inputs must not be binary16 values."""
    if value <= 0:
        return Fraction(0)  # The 8-bit target clamps every negative result to zero.
    bits, = struct.unpack("<H", struct.pack("<e", float(value)))
    if Fraction(struct.unpack("<e", struct.pack("<H", bits))[0]) > value:
        bits -= 1
    lower = Fraction(struct.unpack("<e", struct.pack("<H", bits))[0])
    upper = Fraction(struct.unpack("<e", struct.pack("<H", bits + 1))[0])
    assert lower < value < upper
    return lower


def covered(x, y):
    # Screen vertices (4,28), (28,28), (4,5), evaluated at twice the pixel centers.
    return x >= 4 and y < 28 and 23 * (2 * x + 1 - 8) < 24 * (2 * y + 1 - 10)


def shade(light, *, use_normal=True, clamp=True):
    """Independent reference: exact N.L on the interpolated normal, then the 8-bit target."""
    pixels = bytearray()
    for y in range(32):
        for x in range(32):
            if not covered(x, y):
                pixels.extend((0, 0, 0, 255))
                continue
            # PARAM0 carries the vertex NDC x/y as the normal's x/y, so at a pixel center the
            # interpolated normal is the center's NDC coordinate and a constant z.
            normal = ((Fraction(2 * x - 31, 32), Fraction(31 - 2 * y, 32), NORMAL_Z)
                      if use_normal else (0, 0, 1))
            dot = sum(n * l for n, l in zip(normal, light))
            gray = unorm8(export16((max(dot, 0) if clamp else dot) + light[3]))
            pixels.extend((gray, gray, gray, 255))
    return bytes(pixels)


def scene(light):
    data = inputs(SMALL)
    for i, (x, y, _, _) in enumerate(SMALL):
        struct.pack_into("<3f", data, 1028 + 64 + i * 12, x, y, NORMAL_Z)
    struct.pack_into("<4f", data, LIGHT_OFFSET, *map(float, light))
    return data


def check_binary(tmp, assembler, objcopy, library):
    binary(tmp, assembler, objcopy, library, program=program(), words=WORDS, header=HEADER)
    source, obj, raw = (tmp / name for name in ("lit.s", "lit.o", "lit.bin"))
    source.write_text("\n".join(pixel_program()) + "\n", encoding="ascii")
    subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", source, "-o", obj],
                   check=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
    archive = tmp / "lit.rpf"
    archive.write_bytes(envelope(raw.read_bytes()))
    _, native = native_program(archive, "fixture")
    assert native == raw.read_bytes()
    assert translate(disassemble(native, library)) == translate(pixel_program())


def compile_pixel(tmp, compiler, code, name, *, fallback=False):
    source, shader = tmp / f"{name}.hlsl", tmp / f"{name}.dxil"
    source.write_text(translate(code), encoding="ascii")
    if fallback:
        # The fixture uses normal-range arithmetic and does not need -denorm (SM 6.2+).
        subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
    else:
        compile_shader(compiler, source, shader)
    return shader


def draw(renderer, adapter, shaders, path, data, reference, *, fallback, mismatches=0, allow_skip=False):
    if fallback:
        # Raw PARAM0 is (z, 1, y, x) in the indexed order already used by POSITIONS.
        attributes = [(NORMAL_Z, 1, y, x) for x, y, _, _ in POSITIONS]
        content = fallback_fixture(data, POSITIONS, reference, attributes)
        operation = "--mesh-indexed-fallback"
    else:
        content = fixture(data, POSITIONS, reference, True)
        operation = "--mesh-indexed"
    path.write_bytes(content)
    print(path.stem, flush=True)
    result = subprocess.run([renderer, adapter, operation, *shaders, path],
                            capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    if allow_skip and result.returncode == 77:
        return 77
    lines = result.stdout.splitlines()
    clean = (f"backend={adapter} debug_layer=unavailable" in lines
             or "debug_errors=0 debug_warnings=0" in lines)
    if (result.returncode != int(mismatches != 0) or not clean
            or f"pixels=1024 mismatches={mismatches}" not in lines
            or (fallback and "mesh_exports bytes=128 matched=1" not in lines)
            or (mismatches and "Draw reference mismatch" not in result.stderr.splitlines())):
        raise AssertionError(f"Expected {mismatches} pixel mismatches for {path.stem}\n"
                             f"{result.stdout}\n{result.stderr}")
    return 0


def differences(first, second):
    return sum(first[i:i + 4] != second[i:i + 4] for i in range(0, len(first), 4))


def run(tmp, compiler, renderer, adapter, *, fallback=False):
    if fallback:
        *geometry, lit = compile_fallback(compiler, tmp, program(), translate(pixel_program()))
    else:
        mesh_source, mesh = tmp / "mesh.hlsl", tmp / "mesh.dxil"
        mesh_source.write_text(lower(enumerate_pcs(program()), WORDS, (3, 1), HEADER), encoding="ascii")
        subprocess.run([compiler, "-T", "ms_6_5", "-E", "main", "-DTRACE_EXPORTS", "-Fo", mesh, mesh_source],
                       check=True)
        geometry = [mesh]
        lit = compile_pixel(tmp, compiler, pixel_program(), "lit")

    def check(label, light, pixel, reference, *, mismatches=0, allow_skip=False):
        return draw(renderer, adapter, [*geometry, pixel], tmp / f"{label}.case", scene(light),
                    reference, fallback=fallback, mismatches=mismatches, allow_skip=allow_skip)

    base = shade(LIGHT)
    if check("baseline", LIGHT, lit, base, allow_skip=True) == 77:
        return 77
    check("side-light", SIDE_LIGHT, lit, shade(SIDE_LIGHT))
    inverted = shade(INVERTED)
    check("inverted-light", INVERTED, lit, inverted)

    # Inverted light: the same shaders and normals with the light negated in b0.
    expected = differences(base, inverted)
    # Every covered pixel has a nonzero N.L, so all 276 change and the background still matches.
    assert expected == 276
    check("stale-light", INVERTED, lit, base, mismatches=expected)
    print(f"Expected failure: inverted light, mismatches={expected}")

    # Ignored normal: a pixel program that lights a fixed +Z normal, the constant PARAM0.y of one.
    flat = compile_pixel(tmp, compiler, pixel_program(["v_mul_f32_e32 v12, s6, v11"]), "flat", fallback=fallback)
    ignored = shade(LIGHT, use_normal=False)
    expected = differences(base, ignored)
    assert expected == 276
    check("flat-normal", LIGHT, flat, ignored)
    check("stale-normal", LIGHT, flat, base, mismatches=expected)
    print(f"Expected failure: ignored normal, mismatches={expected}")

    # Without the clamp, only the pixels facing away from the light change.
    unclamped = compile_pixel(tmp, compiler, pixel_program(clamp=False), "unclamped", fallback=fallback)
    negative = shade(LIGHT, clamp=False)
    expected = differences(base, negative)
    assert expected == 132
    check("unclamped", LIGHT, unclamped, negative)
    check("stale-clamp", LIGHT, unclamped, base, mismatches=expected)
    print(f"Expected failure: missing clamp, mismatches={expected}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        if sys.argv[1] == "binary":
            check_binary(Path(directory), *sys.argv[2:])
            sys.exit(0)
        if sys.argv[1] == "fallback":
            sys.exit(run(Path(directory), *sys.argv[2:], fallback=True))
        sys.exit(run(Path(directory), *sys.argv[1:]))
