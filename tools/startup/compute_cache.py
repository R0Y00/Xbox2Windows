#!/usr/bin/env python3
"""Publish validated desktop compute modules under the key the runtime looks them up by.

Device slot 62 in the D3D12 stand-in does not inspect bytecode: it hashes the container the way the
graphics translation path does (FNV-1a 64 over the shader bytes, formatted ``%016llx``) and loads
``<key>.cs.cso`` from ``X2W_SHADER_CACHE``. This tool produces those files.

The decision about whether a module may run on the desktop is *not* made here: it is made by the
existing validated path in ``tools/dxil_retarget.py`` (retarget the target triple, gate ``xbox.*``
symbols, validate container and module, re-sign). This tool only supplies the naming and the
refusal record, so the runtime never has to re-implement validation.

usage: compute_cache.py --pending <dump-dir> --cache <shader-cache> --probe <root_probe.exe>
                        [--retarget <dxil_retarget.py>]

``--pending`` is the directory the stand-in dumps containers into (``X2W_SHADER_DUMP_DIR``); each
``*.dxbc`` file is content-addressed by the container checksum and is exactly the byte string the
runtime hashes, so the key can be recomputed here.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASIS = 1469598103934665603
PRIME = 1099511628211


def key_of(data: bytes) -> int:
    """The runtime's FNV-1a 64: instantiate with BASIS, multiply by PRIME after every xor."""
    value = BASIS
    for byte in data:
        value = ((value ^ byte) * PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def key_text(data: bytes) -> str:
    return "%016x" % key_of(data)


def publish_one(container: Path, cache: Path, probe: Path | None, retarget: Path) -> str:
    """Write <key>.cs.cso for one container, or <key>.fail with the reason. Returns a status line."""
    raw = container.read_bytes()
    key = key_text(raw)
    if len(raw) < 32 or raw[:4] != b"DXBC":
        (cache / (key + ".fail")).write_text("not a DXBC container\n", encoding="utf-8")
        return "%s: refused (not a DXBC container)" % container.name

    command = [sys.executable, "-B", str(retarget), str(container), str(cache / (key + ".cs.cso"))]
    if probe is not None:
        command += ["--probe", str(probe), "--sign"]
    result = subprocess.run(command, capture_output=True, text=True)
    destination = cache / (key + ".cs.cso")
    if result.returncode != 0 or not destination.exists():
        reason = (result.stderr or result.stdout or "retarget failed").strip().splitlines()
        (cache / (key + ".fail")).write_text("\n".join(reason[-4:]) + "\n", encoding="utf-8")
        return "%s: refused (%s)" % (container.name, reason[-1] if reason else "retarget failed")

    if destination.read_bytes()[:4] != b"DXBC":
        destination.unlink()
        (cache / (key + ".fail")).write_text("retargeted output is not a DXBC container\n", encoding="utf-8")
        return "%s: refused (retargeted output is not a DXBC container)" % container.name
    return "%s: published %s.cs.cso" % (container.name, key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pending", required=True, type=Path,
                        help="Directory the stand-in dumps containers into (X2W_SHADER_DUMP_DIR)")
    parser.add_argument("--cache", required=True, type=Path,
                        help="Shader cache directory the runtime reads (X2W_SHADER_CACHE)")
    parser.add_argument("--probe", type=Path, help="root_probe executable, used to validate and sign")
    parser.add_argument("--retarget", type=Path,
                        default=Path(__file__).resolve().parents[1] / "dxil_retarget.py",
                        help="dxil_retarget.py (default: the one beside this tool's parent)")
    args = parser.parse_args()

    if not args.pending.is_dir():
        parser.error("--pending is not a directory: %s" % args.pending)
    args.cache.mkdir(parents=True, exist_ok=True)
    if not args.retarget.is_file():
        parser.error("--retarget is not a file: %s" % args.retarget)

    containers = sorted(args.pending.glob("*.dxbc"))
    if not containers:
        print("no containers in %s" % args.pending)
        return 0
    published = refused = 0
    for container in containers:
        line = publish_one(container, args.cache, args.probe, args.retarget)
        print(line)
        if ".cs.cso" in line:
            published += 1
        else:
            refused += 1
    print("published %d, refused %d" % (published, refused))
    return 0


if __name__ == "__main__":
    sys.exit(main())