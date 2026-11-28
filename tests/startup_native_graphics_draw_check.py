"""Exercise first-use native graphics translation and a fresh-process cache hit."""
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_direct_container_check import patched_pixel, vertex
from startup_stream_draw_check import vop1, vop2


def assert_debug_clean(stderr):
    for line in stderr.splitlines():
        if "x2w-d3d12-debug " in line:
            assert " sev=0 " not in line and " sev=1 " not in line, line


def triangle():
    code = [vop2(0x1a, 1, 129, 0), vop2(0x1b, 1, 130, 1), vop2(0x1b, 2, 130, 0)]
    for v in (1, 2):
        code.extend([vop1(6, v, 256 + v), vop2(8, v, 244, v), vop2(3, v, 243, v)])
    code.extend([vop1(1, 3, 128), vop1(1, 4, 242), 0xf80008cf, 0x04030201, 0xbf810000])
    return vertex(program=code)


def check(folder, consumer, sdk, adapter, compiler):
    cache = folder / "cache"
    cache.mkdir()
    vs, ps, root, pixels = [folder / name for name in ("vs", "ps", "root", "pixels")]
    vs.write_bytes(triangle())
    ps.write_bytes(patched_pixel(2, [vop1(1, 0, 128), vop1(1, 1, 242)], export=(0xf800180f, 0x01000001)))
    env = dict(os.environ, X2W_SHADER_CACHE=str(cache), X2W_NATIVE_SHADERS=sdk, X2W_NATIVE_SHADERS_ONLY="0",
               X2W_D3D12_DEBUG="1", X2W_D3D12_RENDER="1", X2W_D3D12_PERMISSIVE="1", X2W_D3D12_LIST_NOOP="1")
    clear = bytes((0, 0, 255, 255))
    red = bytes((255, 0, 0, 255))
    args = [consumer, vs, ps, root, adapter, pixels]
    for name, hits in (("cold", "0,0,0"), ("hot", "1,1,1")):
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
        if run.returncode == 77:
            return 77
        if run.returncode or f"ready=1 cache(frontend,vs,ps)={hits}" not in run.stderr or "-> drawn;" not in run.stderr:
            raise RuntimeError(run.stderr[-8000:])
        for line in run.stderr.splitlines():
            if "x2w-d3d12-debug " in line:
                assert " sev=0 " not in line and " sev=1 " not in line, line
            if "debug layer:" in line:
                print(line)
        print(run.stdout.strip())
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        print(f"{adapter} {name}: cache={hits}, 16 red and 16 clear pixels")
    # Publish a verified disk pair, then hold back its ready marker until after the first draw.
    subprocess.run([compiler, "--watch", sdk, cache, cache, "--once"], check=True, timeout=45)
    marker, = cache.glob("req_*.a")
    manifest = cache / (marker.stem.removeprefix("req_") + ".txt")
    published = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in
                 (manifest, manifest.with_suffix(".vs.cso"), manifest.with_suffix(".ps.cso"))}
    published_draw = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
    assert published_draw.returncode == 0 and "-> drawn;" in published_draw.stderr, published_draw.stderr
    assert "native graphics " not in published_draw.stderr, published_draw.stderr
    assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
    changed = [(path.name, path.read_bytes(), expected[0]) for path, expected in published.items()
               if path.read_bytes() != expected[0]]
    assert not changed, [(name, len(got), len(expected), got[:48].hex(), expected[:48].hex())
                         for name, got, expected in changed]
    print(f"{adapter} existing disk pair takes precedence with native enabled")
    for fault, diagnostic in (("MANIFEST", "unsupported manifest version"), ("ROOT", "failed 0x80004005")):
        manifest.rename(str(manifest) + ".pending")
        first = Path(str(pixels) + ".first")
        first.unlink(missing_ok=True)
        pixels.unlink()
        fault_env = dict(env, X2W_TEST_PUBLISH_MANIFEST=str(manifest), X2W_NATIVE_SHADERS_ONLY="0")
        fault_env[f"X2W_TEST_NATIVE_GRAPHICS_{fault}_FAILURE"] = "1"
        run = subprocess.run(args, env=fault_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0 and diagnostic in run.stderr and "-> drawn;" in run.stderr, run.stderr
        assert first.read_bytes() == clear * 32
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
                   for path, expected in published.items())
        print(f"{adapter} {fault}: same-process fallback drew 16 red pixels after publication")

    # The published pair alone draws the target when native preparation is disabled.
    pixels.unlink()
    disk_env = {name: value for name, value in env.items() if name != "X2W_NATIVE_SHADERS"}
    run = subprocess.run(args, env=disk_env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0, run.stderr
    assert "native graphics" not in run.stderr and "-> drawn;" in run.stderr, run.stderr[-8000:]
    assert_debug_clean(run.stderr)
    assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
    print(f"{adapter} disk pair: native disabled, 16 red and 16 clear pixels")

    # No ready disk marker at the first draw, then the valid pair is moved into place just
    # before the failed native candidate asks for its fallback. This proves the native PSO was
    # attempted and that the disk pair won only after that failure, not before it.
    pso_env = dict(env, X2W_NATIVE_SHADERS_ONLY="0", X2W_TEST_NATIVE_GRAPHICS_PSO_FAILURE="1")
    manifest_pending = Path(str(manifest) + ".pending")
    manifest.rename(manifest_pending)
    try:
        pixels.unlink(missing_ok=True)
        fallback_env = dict(pso_env, X2W_TEST_MANIFEST_BEFORE_FALLBACK=str(manifest))
        run = subprocess.run(args, env=fallback_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "candidate pipeline failed 0x80004005 (injected)" in run.stderr, run.stderr[-8000:]
        assert "manifest published before fallback" in run.stderr, run.stderr[-8000:]
        assert "published disk pair recovered" in run.stderr and "-> drawn;" in run.stderr, run.stderr[-8000:]
        assert_debug_clean(run.stderr)
        assert pixels.read_bytes() == (red * 4 + clear * 4) * 4
        assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
                   for path, expected in published.items())
        print(f"{adapter} PSO: native attempt failed, then the pair made ready before fallback drew 16 red pixels")
    finally:
        if manifest_pending.exists():
            manifest_pending.rename(manifest)

    # Native-only mode refuses without touching the pending pair.
    manifest.rename(manifest_pending)
    try:
        pixels.unlink(missing_ok=True)
        only_env = dict(env, X2W_NATIVE_SHADERS_ONLY="1", X2W_TEST_NATIVE_GRAPHICS_PSO_FAILURE="1")
        run = subprocess.run(args, env=only_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "no published disk pair in native-only mode" in run.stderr, run.stderr[-8000:]
        assert "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == clear * 32
        assert manifest_pending.exists() and not manifest.exists()
        print(f"{adapter} native-only: refused with the pair still pending, 32 clear pixels")
    finally:
        if manifest_pending.exists():
            manifest_pending.rename(manifest)
    assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
               for path, expected in published.items())

    # An RTV variant is a separate published pair. Draw it natively first so the runtime
    # publishes req_<key>.rtv, compile that request, then make its pair ready before the fallback.
    variant_args = args + ["2"]  # DXGI_FORMAT_R32G32B32A32_FLOAT
    variant_red = bytes.fromhex("0000803f00000000000000000000803f")
    variant_clear = bytes.fromhex("00000000000000000000803f0000803f")
    pixels.unlink(missing_ok=True)
    run = subprocess.run(variant_args, env=env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0 and "-> drawn;" in run.stderr, run.stderr[-8000:]
    assert "no RTV translation" not in run.stderr, run.stderr[-8000:]
    assert pixels.read_bytes() == variant_red, pixels.read_bytes().hex()
    variant_request, = cache.glob("req_*.rtv")
    variant_key = variant_request.stem.removeprefix("req_")
    subprocess.run([compiler, "--watch", sdk, cache, cache, "--once"], check=True, timeout=45)
    variant_manifest = cache / (variant_key + ".txt")
    variant_published = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in
                         (variant_manifest, variant_manifest.with_suffix(".vs.cso"),
                          variant_manifest.with_suffix(".ps.cso"))}
    variant_pending = Path(str(variant_manifest) + ".pending")
    variant_manifest.rename(variant_pending)
    try:
        pixels.unlink(missing_ok=True)
        variant_fallback_env = dict(pso_env, X2W_TEST_MANIFEST_BEFORE_FALLBACK=str(variant_manifest))
        run = subprocess.run(variant_args, env=variant_fallback_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "candidate pipeline failed 0x80004005 (injected)" in run.stderr, run.stderr[-8000:]
        assert "manifest published before fallback" in run.stderr, run.stderr[-8000:]
        assert "published disk pair recovered" in run.stderr and "-> drawn;" in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == variant_red, pixels.read_bytes().hex()
        assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
                   for path, expected in variant_published.items())
        print(f"{adapter} RTV variant: native attempt failed, then the variant pair made ready before fallback")
    finally:
        if variant_pending.exists():
            variant_pending.rename(variant_manifest)

    held_variant = Path(str(variant_manifest) + ".hold")
    variant_manifest.rename(held_variant)
    try:
        pixels.unlink(missing_ok=True)
        run = subprocess.run(variant_args, env=pso_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "no valid published disk pair" in run.stderr and "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == variant_clear, pixels.read_bytes().hex()
        assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
                   for path, expected in variant_published.items() if path != variant_manifest)
        print(f"{adapter} RTV variant: a missing manifest refused, 32 clear pixels")
    finally:
        held_variant.rename(variant_manifest)

    # A missing or invalid pair refuses the draw under the disk-first precedence.
    ps_cso = manifest.with_suffix(".ps.cso")
    held_ps = Path(str(ps_cso) + ".hold")
    ps_cso.rename(held_ps)
    try:
        pixels.unlink()
        run = subprocess.run(args, env=pso_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == clear * 32
        print(f"{adapter} missing pair: refused, 32 clear pixels")
    finally:
        held_ps.rename(ps_cso)
    held_manifest = Path(str(manifest) + ".hold")
    manifest.rename(held_manifest)
    manifest.write_text("x2w-manifest 999\n")
    try:
        pixels.unlink()
        run = subprocess.run(args, env=pso_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "unsupported manifest version" in run.stderr, run.stderr[-8000:]
        assert "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == clear * 32
        print(f"{adapter} invalid pair: refused, 32 clear pixels")
    finally:
        manifest.unlink()
        held_manifest.rename(manifest)
    assert all(path.read_bytes() == expected[0] and path.stat().st_mtime_ns == expected[1]
               for path, expected in published.items())

    # The manifest and vertex blob stay valid, but the pixel blob no longer creates a PSO.
    held_ps = Path(str(ps_cso) + ".hold")
    ps_cso.rename(held_ps)
    invalid_ps = b"not a DXIL container"
    ps_cso.write_bytes(invalid_ps)
    invalid_mtime = ps_cso.stat().st_mtime_ns
    try:
        pixels.unlink()
        run = subprocess.run(args, env=pso_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == clear * 32, pixels.read_bytes().hex()
        assert ps_cso.read_bytes() == invalid_ps and ps_cso.stat().st_mtime_ns == invalid_mtime
        print(f"{adapter} invalid PS blob: disk PSO refused, 32 clear pixels, file unchanged")
    finally:
        ps_cso.unlink()
        held_ps.rename(ps_cso)
    # Native candidate fails, then a valid manifest with an uncreatable PS is made ready:
    # the fallback must run and name the failed disk PSO.
    held_manifest = Path(str(manifest) + ".pso-hold")
    held_ps = Path(str(ps_cso) + ".pso-hold")
    manifest.rename(held_manifest)
    ps_cso.rename(held_ps)
    Path(str(manifest) + ".pending").write_bytes(held_manifest.read_bytes())
    ps_cso.write_bytes(b"not a DXIL container")
    try:
        pixels.unlink(missing_ok=True)
        fallback_env = dict(pso_env, X2W_TEST_MANIFEST_BEFORE_FALLBACK=str(manifest))
        run = subprocess.run(args, env=fallback_env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        assert "candidate pipeline failed 0x80004005 (injected)" in run.stderr, run.stderr[-8000:]
        assert "published disk pair pipeline failed" in run.stderr, run.stderr[-8000:]
        assert "-> drawn;" not in run.stderr, run.stderr[-8000:]
        assert pixels.read_bytes() == clear * 32, pixels.read_bytes().hex()
        print(f"{adapter} PSO: fallback loaded the pair, then named its failed PSO")
    finally:
        for path in (manifest, Path(str(manifest) + ".pending")):
            if path.exists():
                path.unlink()
        ps_cso.unlink(missing_ok=True)
        held_manifest.rename(manifest)
        held_ps.rename(ps_cso)
    ps.write_bytes(patched_pixel(1, [0xffffffff], export=(0xf800180f, 0x01000001)))
    run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
    assert run.returncode == 0 and "refused:" in run.stderr and "-> drawn;" not in run.stderr, run.stderr
    assert pixels.read_bytes() == clear * 32
    print("unsupported stage: all 32 pixels retain the clear color")
    # The first DWORD is 0.25 in record zero, 1.0 in record one. Its byte-1 window is 0x003e8000.
    records = bytearray(64)
    struct.pack_into("<f", records, 0, 0.25)
    struct.pack_into("<f", records, 16, 1.0)
    env["X2W_TEST_FORMATTED_VIEW"] = f"0:1688:0:{records.hex()}"
    for index, stride in ((1, 16), (1, 8), (0xffffffff, 16)):
        env["X2W_TEST_INDEXED_STRIDE"] = str(stride)
        program = [0xbe810380, 0xf4080200, 0xfa000000, vop1(1, 0, 255), index,
                   0xe0302000, 0x80020000, vop1(1, 1, 128), vop1(1, 2, 242)]
        ps.write_bytes(patched_pixel(len(program), program, table=True, uses=[(0x300, 0, 0, 16, 4)],
                                     export=(0xf800180f, 0x02010100)))
        run = subprocess.run(args, env=env, capture_output=True, text=True, timeout=45)
        assert run.returncode == 0, run.stderr
        for line in run.stderr.splitlines():
            if "x2w-d3d12-debug " in line:
                assert " sev=0 " not in line and " sev=1 " not in line, line
        image = pixels.read_bytes()
        if stride == 8:
            assert "needs stride 16, got 8" in run.stderr and "-> drawn;" not in run.stderr, run.stderr[-7000:]
            assert image == clear * 32
        else:
            assert "-> drawn;" in run.stderr, run.stderr[-7000:]
            color = red if index == 1 else bytes((0, 0, 0, 255))
            assert image == (color * 4 + clear * 4) * 4, (image.hex(), run.stderr[-6000:])
        print(f"{adapter} idxbuf index={index:#x} actual stride={stride}: first RGBA {image[:4].hex()}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(check(Path(directory), *sys.argv[1:]))
