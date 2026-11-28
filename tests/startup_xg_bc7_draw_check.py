"""Draw tagged, source-built BC7 through the wrapped uploader."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from bc7_draw_check import make_block
from startup_stream_check import container, gfx2hlsl


def write_fixture(upload, width, height):
    blocks = bytearray()
    pixels = bytearray(width * height * 4)
    for y in range((height + 3) // 4):
        for x in range((width + 3) // 4):
            block, colors = make_block(x % 21, y % 19)
            blocks.extend(block)
            for row in range(min(4, height - y * 4)):
                destination = ((y * 4 + row) * width + x * 4) * 4
                count = min(4, width - x * 4) * 4
                pixels[destination:destination + count] = colors[row * 16:row * 16 + count]
    upload.write_bytes(blocks)
    return pixels


def display64k_blocks(linear, width, height):
    """BC7 block order within one display tile is x0 y0 x1 y1 y2 x2 y3 x3 y4 x4 y5 x5."""
    tiled = bytearray(len(linear))
    for y in range(height // 4):
        for x in range(width // 4):
            tile = (y // 64) * (width // 256) + x // 64
            block = sum(((x >> bit) & 1) << shift for bit, shift in enumerate((0, 2, 5, 7, 9, 11)))
            block |= sum(((y >> bit) & 1) << shift for bit, shift in enumerate((1, 3, 4, 6, 8, 10)))
            dest = (tile * 4096 + block) * 16
            src = (y * (width // 4) + x) * 16
            tiled[dest:dest + 16] = linear[src:src + 16]
    return tiled

def check(work, consumer, dxc, backend):
    cache = work / "cache"
    cache.mkdir()
    vs, ps, root, image, upload = (work / name for name in ("vs", "ps", "root", "pixels", "blocks"))
    vs.write_bytes(container(1))
    ps.write_bytes(container(0))
    env = dict(os.environ, X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1",
               X2W_D3D12_RENDER="1", X2W_XG_BC7_4K_S_LAYOUT="1", X2W_SHADER_CACHE=str(cache))
    base = [consumer, vs, ps, root, backend, image, upload]
    bootstrap = subprocess.run(base + ["64", "64", "linear"], env=dict(env, X2W_TEST_TRUNCATED_ONLY="1"))
    if bootstrap.returncode == 77:
        print(f"{backend}: D3D12 feature level 12 unavailable")
        return 77
    bootstrap.check_returncode()
    key = f"{gfx2hlsl.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes()):016x}"
    (cache / f"{key}.txt").write_text("udsrc 0 rect\nsrv 0 tex2d table 0 0\n", encoding="ascii")

    sources = {
        "vs": """float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
        }""",
        "ps": """Texture2D<float4> tex : register(t0);
        float4 main(float4 position : SV_Position) : SV_Target {
            return tex.Load(int3(int2(position.xy), 0));
        }""",
    }
    for stage, source in sources.items():
        shader = work / f"{stage}.hlsl"
        shader.write_text(source, encoding="ascii")
        subprocess.run([dxc, "-T", f"{stage}_6_0", "-E", "main", "-Fo",
                        cache / f"{key}.{stage}.cso", shader], check=True)

    mismatches = []
    for width, height in ((64, 64), (81, 69), (512, 256)):
        reference = write_fixture(upload, width, height)
        layouts = ("linear", "display64k") if width == 512 else ("linear", "tiled")
        for layout in layouts:
            if layout == "display64k":
                linear = upload.read_bytes()
                tiled = display64k_blocks(linear, width, height)
                assert tiled != linear
                upload.write_bytes(tiled)
            if image.exists():
                image.unlink()
            result = subprocess.run(base + [str(width), str(height), layout], env=env, capture_output=True, text=True)
            print(result.stdout, end="")
            for line in result.stderr.splitlines():
                if "debug layer:" in line or "x2w-d3d12-debug " in line:
                    print(line)
            if result.returncode or "-> drawn;" not in result.stderr:
                raise RuntimeError(f"{backend} {width}x{height} {layout} draw failed ({result.returncode})\n{result.stderr}")
            actual = image.read_bytes()
            assert len(actual) == len(reference), f"{layout}: read {len(actual)} bytes, expected {len(reference)}"
            if actual != reference:
                first = next(i for i in range(0, len(reference), 4) if actual[i:i + 4] != reference[i:i + 4])
                x, y = first // 4 % width, first // 4 // width
                mismatches.append(f"{backend} {width}x{height} {layout}: pixel ({x}, {y}) "
                                  f"read {actual[first:first + 4].hex()}, expected {reference[first:first + 4].hex()}")
            else:
                print(f"{backend}: {width}x{height} {layout} draw matched all {width * height} reference pixels")

    # Non-BC upload test (RGBA8 layout 0x105, 4KB_S) and negative swizzle control.
    rgba8_reference = bytearray(64 * 64 * 4)
    for y in range(64):
        for x in range(64):
            idx = (y * 64 + x) * 4
            rgba8_reference[idx:idx + 4] = bytes(((x * 4 + 7) & 255, (y * 4 + 13) & 255, (x ^ y ^ 0xA5) & 255, 255))
    upload.write_bytes(rgba8_reference)

    # 1. Tiled upload via XG slot 8 into 0x105, detiled by upload_texture via is_gfx10_tiled.
    if image.exists():
        image.unlink()
    result = subprocess.run(base + ["64", "64", "rgba8_4k_s"], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    for line in result.stderr.splitlines():
        if "debug layer:" in line or "x2w-d3d12-debug " in line:
            print(line)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend} 64x64 rgba8_4k_s draw failed ({result.returncode})\n{result.stderr}")
    actual = image.read_bytes()
    assert len(actual) == len(rgba8_reference), f"rgba8_4k_s: read {len(actual)} bytes, expected {len(rgba8_reference)}"
    if actual != rgba8_reference:
        first = next(i for i in range(0, len(rgba8_reference), 4) if actual[i:i + 4] != rgba8_reference[i:i + 4])
        x, y = first // 4 % 64, first // 4 // 64
        mismatches.append(f"{backend} 64x64 rgba8_4k_s: pixel ({x}, {y}) "
                          f"read {actual[first:first + 4].hex()}, expected {rgba8_reference[first:first + 4].hex()}")
    else:
        print(f"{backend}: 64x64 rgba8_4k_s draw matched all {64 * 64} reference pixels")

    # 2. Negative control: linear / untiled data placed in 0x105 buffer must fail to match reference.
    if image.exists():
        image.unlink()
    result = subprocess.run(base + ["64", "64", "rgba8_4k_s_untiled"], env=env, capture_output=True, text=True)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend} 64x64 rgba8_4k_s_untiled draw failed ({result.returncode})\n{result.stderr}")
    actual_untiled = image.read_bytes()
    assert actual_untiled != rgba8_reference, f"{backend}: negative control failed, untiled draw unexpectedly matched reference"
    diff_count = sum(actual_untiled[i:i + 4] != rgba8_reference[i:i + 4] for i in range(0, len(rgba8_reference), 4))
    assert diff_count >= 3000, f"{backend}: negative control only differed on {diff_count}/4096 pixels"
    print(f"{backend}: 64x64 rgba8_4k_s negative swizzle control confirmed ({diff_count}/4096 pixels differ)")

    # 3. Multi-mip linear upload (RGBA8 64x64 3-mip 0x100) placed smallest-first by XG slot 8,
    # detiled/uploaded by upload_texture via is_gfx10_tiled.
    if image.exists():
        image.unlink()
    result = subprocess.run(base + ["64", "64", "rgba8_linear_3m"], env=env, capture_output=True, text=True)
    print(result.stdout, end="")
    for line in result.stderr.splitlines():
        if "debug layer:" in line or "x2w-d3d12-debug " in line:
            print(line)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend} 64x64 rgba8_linear_3m draw failed ({result.returncode})\n{result.stderr}")
    actual_linear_3m = image.read_bytes()
    assert len(actual_linear_3m) == len(rgba8_reference), f"rgba8_linear_3m: read {len(actual_linear_3m)} bytes, expected {len(rgba8_reference)}"
    if actual_linear_3m != rgba8_reference:
        first = next(i for i in range(0, len(rgba8_reference), 4) if actual_linear_3m[i:i + 4] != rgba8_reference[i:i + 4])
        x, y = first // 4 % 64, first // 4 // 64
        mismatches.append(f"{backend} 64x64 rgba8_linear_3m: pixel ({x}, {y}) "
                          f"read {actual_linear_3m[first:first + 4].hex()}, expected {rgba8_reference[first:first + 4].hex()}")
    else:
        print(f"{backend}: 64x64 rgba8_linear_3m draw matched all {64 * 64} reference pixels")

    # 4. Negative control: linear 3-mip texture placed in standard desktop footprint order (offset 0 for mip 0)
    # must fail to match reference because upload_texture reads mip 0 from XG's smallest-first offset 12288.
    if image.exists():
        image.unlink()
    result = subprocess.run(base + ["64", "64", "rgba8_linear_3m_desktop"], env=env, capture_output=True, text=True)
    if result.returncode or "-> drawn;" not in result.stderr:
        raise RuntimeError(f"{backend} 64x64 rgba8_linear_3m_desktop draw failed ({result.returncode})\n{result.stderr}")
    actual_linear_desktop = image.read_bytes()
    assert actual_linear_desktop != rgba8_reference, f"{backend}: negative control failed, desktop layout unexpectedly matched reference"
    diff_count = sum(actual_linear_desktop[i:i + 4] != rgba8_reference[i:i + 4] for i in range(0, len(rgba8_reference), 4))
    assert diff_count >= 1000, f"{backend}: negative control only differed on {diff_count}/4096 pixels"
    print(f"{backend}: 64x64 rgba8_linear_3m negative layout control confirmed ({diff_count}/4096 pixels differ)")

    if mismatches:
        raise AssertionError("\n".join(mismatches))
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
