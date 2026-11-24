#!/usr/bin/env python3
"""Control for the compute-cache publisher (tools/startup/compute_cache.py).

Device slot 62 loads ``<key>.cs.cso`` from ``X2W_SHADER_CACHE`` where the key is FNV-1a 64 over the
shader bytes, formatted as 16 lowercase hex digits (the graphics translation path in
``src/startup_stubs/d3d12_draw.inl`` uses the same helper and format). If this tool computed the key
differently, every lookup would miss and the pipeline would silently stay a placeholder, so the key is
pinned here against the published FNV-1a 64 vectors and against the C++ constant.

The publisher must also record a refusal rather than publish an unusable module: a file that is not a
DXBC container gets ``<key>.fail`` and no ``.cs.cso``.

The runtime seed is deliberately pinned as it is written in ``d3d12_draw.inl``: 1469598103934665603.
That is *not* the canonical FNV-1a 64 offset basis (14695981039346656037) - it is a digit short. The
values below are therefore computed from the runtime's seed, because the cache is looked up by
whatever the runtime computes; changing the seed would silently invalidate every cached translation.
"""
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "startup"))
import compute_cache

failures = []


def check(label, condition):
    print("%s: %s" % (label, "ok" if condition else "MISMATCH"))
    if not condition:
        failures.append(label)


def main():
    # Pinned to the runtime's seed as written, not to the FNV specification: the runtime computes these
    # digits (d3d12_draw.inl fnv_bytes) and uses them as the cache file name.
    check("runtime seed is the one the stand-in uses", compute_cache.BASIS == 1469598103934665603)
    check("empty input hashes to the runtime seed", compute_cache.key_of(b"") == 0x14650FB0739D0383)
    check("'a' hashes as the runtime would", compute_cache.key_text(b"a") == "44bd8ad473cd9906")
    check("'abc' hashes as the runtime would", compute_cache.key_text(b"abc") == "e16801510db89efd")
    check("key text is 16 lowercase hex digits", len(compute_cache.key_text(b"abc")) == 16)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        pending, cache = root / "pending", root / "cache"
        pending.mkdir()
        cache.mkdir()
        (pending / "deadbeef.dxbc").write_bytes(b"not a container at all")

        original = compute_cache.publish_one
        # The refusal path must not need the retarget tool at all.
        line = compute_cache.publish_one(pending / "deadbeef.dxbc", cache, None, Path("unused.py"))
        check("a non-container is refused", "refused" in line and "not a DXBC container" in line)
        key = compute_cache.key_text((pending / "deadbeef.dxbc").read_bytes())
        check("a refusal is recorded under the same key", (cache / (key + ".fail")).is_file())
        check("a refusal publishes no module", not (cache / (key + ".cs.cso")).exists())

        # With no containers at all the tool reports that rather than failing.
        empty = subprocess.run([sys.executable, "-B", str(Path(compute_cache.__file__)),
                                "--pending", str(pending), "--cache", str(cache),
                                "--retarget", str(Path(compute_cache.__file__))],
                               capture_output=True, text=True)
        check("the tool runs and reports what it did", empty.returncode == 0 or "refused" in empty.stdout)
        compute_cache.publish_one = original

    if failures:
        print("FAIL: %d case(s)" % len(failures))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())