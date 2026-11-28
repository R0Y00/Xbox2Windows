"""Check that a strict BCPack decode feeds two ordered mode-6 blocks to D3D12."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bc7_blocks import decode, pack, unpack
from pixel_aot_check import assembly, translate

# Synthetic zero-history stream from stan495's 5e1af169 fixture for issue #769.
PACKED = bytes.fromhex("""
101010101010101005050505050506060101010101010110
000000000000042002027546281800000303df5f4b7c0521
030350782337043110101010101010100605060505050505
1010010101010110200801000000000400000008e4db27a4
21090294fe6b020431330404c4fb37250202bd19500f0341
41050934f4cb070502414111
""")


def mode6_block(endpoint0, endpoint1, selectors):
    fields = {"mode": 6, "p0": endpoint0[0] & 1, "p1": endpoint1[0] & 1}
    for endpoint, channels in enumerate((endpoint0, endpoint1)):
        for component, channel in zip("rgba", channels):
            fields[f"{component}{endpoint}"] = channel >> 1
    fields.update({f"index{i}": selector for i, selector in enumerate(selectors)})
    return pack(fields)


def reference_rgba(left, right):
    left_rgba, right_rgba = decode(left), decode(right)
    # Each raster row contains four texels from each block.
    return b"".join(left_rgba[i:i + 16] + right_rgba[i:i + 16] for i in range(0, 64, 16))


def main():
    compiler, renderer, backend = sys.argv[1:]
    reference_tools = os.environ.get("BCPACK_REFERENCE_TOOLS")
    if not reference_tools:
        print("Set BCPACK_REFERENCE_TOOLS to the pinned Leonida tools directory", flush=True)
        return 77
    tools = Path(reference_tools).resolve()
    if not (tools / "bcpack.py").is_file():
        raise ValueError(f"No bcpack.py in BCPACK_REFERENCE_TOOLS directory {tools}")
    sys.path.insert(0, str(tools))
    from bcpack import BcpackError, decode_with_info

    left = mode6_block((197, 71, 41, 25), (144, 160, 180, 124),
                       (1, 0, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 0, 15))
    right = mode6_block((40, 120, 220, 200), (161, 231, 81, 241), range(16))
    expected = reference_rgba(left, right)
    decoded, provisional = decode_with_info(PACKED, 7, len(left + right), allow_provisional=False)
    if decoded != left + right or provisional:
        raise AssertionError(f"Strict decode expected {(left + right).hex()}, got {decoded.hex()} ({provisional})")
    print(f"Strict decode produced {len(decoded)} ordered BC7 bytes with no provisional entries", flush=True)
    for length in (len(PACKED) - 1, len(PACKED) // 2, 32):
        try:
            decode_with_info(PACKED[:length], 7, len(decoded), allow_provisional=False)
        except BcpackError as error:
            print(f"Truncated stream ({length} bytes) rejected before upload: {error}", flush=True)
        else:
            raise AssertionError(f"Strict decode accepted a stream truncated to {length} bytes")

    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bc7-mode6-", dir=out) as directory:
        work = Path(directory)
        shader, source = work / "copy.cso", work / "copy.hlsl"
        source.write_text(translate(assembly(False)), encoding="ascii")
        subprocess.run([compiler, "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        case, capture = work / "surface.case", work / "actual.rgba"
        header = struct.pack("<4sII", b"BC7C", 8, 4)
        draw = [renderer, backend, "--check-bc7", shader, case]
        case.write_bytes(header + decoded + expected)
        result = subprocess.run([*draw, capture])
        if result.returncode:
            return result.returncode
        if capture.read_bytes() != expected:
            raise AssertionError("GPU capture differs from the 8x4 RGBA reference")

        decoded_left, decoded_right = decoded[:16], decoded[16:]
        changed_fields = unpack(decoded_right)
        changed_fields["index15"] = 0
        changed_right = pack(changed_fields)
        if reference_rgba(decoded_left, changed_right) != expected[:-4] + bytes((40, 120, 220, 200)):
            raise AssertionError("Changing the last selector must change only the texel at (7, 3)")
        for label, blocks, coordinate in (
            ("swapped blocks", decoded_right + decoded_left, "x=0 y=0"),
            ("changed selector", decoded_left + changed_right, "x=7 y=3"),
        ):
            case.write_bytes(header + blocks + expected)
            wrong = subprocess.run(draw, capture_output=True, text=True)
            mismatch = f"pixel_mismatch {coordinate}"
            # A skip after the positive draw is a failure of the negative control.
            if wrong.returncode != 1 or mismatch not in wrong.stderr.splitlines():
                raise AssertionError(f"{label}: expected {mismatch}, got {wrong.returncode}\n"
                                     f"{wrong.stdout}{wrong.stderr}")
            print(f"{label}: {mismatch}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
