"""Verify entry-mask interpolation of all four attr0 components with no sampler, in binary32 and binary16."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import (attribute_color, attribute_half, half_rtz, half_value, lower, must_reject,
                             narrow_reference, translate_attribute)


def run(directory, compiler, renderer, backend):
    for rg in (False, True):
        code = attribute_color(rg)
        must_reject(lambda: lower(code, 14, 8, None))
        must_reject(translate_attribute, [s.replace("v8, v1, attr0.w", "v8, v1, attr0.z") for s in code])
        must_reject(translate_attribute, [s.replace("v9, v0, attr0.z", "v9, v1, attr0.z") for s in code])
        must_reject(translate_attribute, [s.replace("attr0.z", "attr1.z") for s in code])
        src, cso, case = [directory / name for name in ("color.hlsl", "color.cso", "color.case")]
        src.write_text(translate_attribute(code), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, src], check=True)
        cmd = [renderer, backend, "--check-pixel-attribute-rg" if rg else "--check-pixel-attribute", cso, case]
        data = bytearray(256)
        for y in range(32):
            for x in range(32):
                u, v = (x + 0.5) / 32, (y + 0.5) / 32
                data += struct.pack("<4f", 1-v, 0.25 + 0.5*u, -1 if rg else u, -1 if rg else v)
        case.write_bytes(data)
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
        # A wrong channel at an interior pixel must be detected.
        struct.pack_into("<f", data, 256 + (9 * 32 + 17) * 16, 0)
        case.write_bytes(data)
        wrong = subprocess.run(cmd, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch x=17 y=9" not in wrong.stderr:
            raise AssertionError("Wrong interpolated channel escaped detection")


def run_half(directory, compiler, renderer, backend):
    """Draw the 16-bit interpolation fixture. Each result is the attribute narrowed toward zero."""
    src, cso, case = [directory / name for name in ("half.hlsl", "half.cso", "half.case")]
    hlsl = translate_attribute(attribute_half())
    src.write_text(hlsl, encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, src], check=True)

    def references(fine):
        # The same channel order as attribute_color: w, z, x, y. The fine vertices carry 1 + value / 64.
        exact = []
        for y in range(32):
            for x in range(32):
                u, v = (x + 0.5) / 32, (y + 0.5) / 32
                values = (1 - v, 0.25 + 0.5 * u, u, v)
                exact.append([1 + value / 64 for value in values] if fine else list(values))
        return exact

    def draw(operation, pixels):
        case.write_bytes(bytes(256) + b"".join(struct.pack("<4f", *pixel) for pixel in pixels))
        return subprocess.run([renderer, backend, operation, cso, case], capture_output=True, text=True)

    # Every coarse attribute value is an exact binary16, so the narrowing leaves it unchanged.
    coarse = references(False)
    assert all(half_rtz(value) == value for pixel in coarse for value in pixel)
    result = draw("--check-pixel-attribute", coarse)
    sys.stdout.write(result.stdout)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    # No fine value is an exact binary16. Truncation and nearest even differ in every pixel and in half the
    # pixels of each channel. At least 2^-13 separates each value from the binary16 below and above it, so
    # an interpolation error of a few binary32 ulps cannot change either result.
    fine = references(True)
    truncated = [[half_rtz(value) for value in pixel] for pixel in fine]
    nearest = [[half_value(narrow_reference(value)) for value in pixel] for pixel in fine]
    for exact, down, near in zip(fine, truncated, nearest):
        assert all(2.0 ** -13 <= value - low <= 7 * 2.0 ** -13 for value, low in zip(exact, down))
        assert any(low != other for low, other in zip(down, near))
    for channel in range(4):
        assert sum(down[channel] != near[channel] for down, near in zip(truncated, nearest)) == 512
    result = draw("--check-pixel-attribute-fine", truncated)
    sys.stdout.write(result.stdout)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    # Negative controls on the references: nearest even, no narrowing, and one wrong interior channel.
    wrong = [list(pixel) for pixel in truncated]
    wrong[9 * 32 + 17][2] += 2.0 ** -10
    for name, pixels in (("nearest-even", nearest), ("unnarrowed", fine), ("one-step", wrong)):
        result = draw("--check-pixel-attribute-fine", pixels)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"The {name} reference escaped detection")
    # Mutants of the translation: a nearest-even narrowing, and a shader drawn with the coarse vertices.
    assert hlsl.count("asfloat(narrow16_rtz(uv.") == 4
    src.write_text(hlsl.replace("asfloat(narrow16_rtz(uv.", "asfloat(narrow16(uv."), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, src], check=True)
    result = draw("--check-pixel-attribute-fine", truncated)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("A nearest-even narrowing escaped detection")
    if draw("--check-pixel-attribute-fine", nearest).returncode:
        raise AssertionError("The nearest-even mutant does not match its own reference")
    result = draw("--check-pixel-attribute", truncated)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("The coarse vertices matched the fine reference")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(run(Path(tmp), *sys.argv[1:]) or run_half(Path(tmp), *sys.argv[1:]))
