"""Read back slots in the C++ raw graphics sampler-table lowering."""
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from native_direct_container_check import publish
from startup_sampler_heap_check import CONST_WORDS, FULLSCREEN_VS, MIXED_WORDS, PS_WORDS, VS_WORDS


def move(dst, src):
    return struct.pack("<I", 0xbe800300 | (dst << 16) | src)


def program(body, dropped=False):
    # Root table at s8, constants at s9:s10, sampler heap at s11. The
    # existing draw harness puts the varying byte offset in constant word 1.
    prefix = (move(10, 9) if dropped else b"") + move(12, 11) + move(13, 128) + move(9, 128)
    body = body.replace(bytes.fromhex("050608f400000012"), bytes.fromhex("060608f400000014"))
    return prefix + body.replace(bytes.fromhex("050608f4100000fa"), bytes.fromhex("060608f4100000fa"))


def main(frontend, renderer, sdk, adapter):
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        root = folder / "root.bin"
        source = folder / "fixture.hlsl"
        source.write_text('#define RS "RootFlags(SAMPLER_HEAP_DIRECTLY_INDEXED),'
                          'DescriptorTable(SRV(t0)),RootConstants(num32BitConstants=2,b0)"', encoding="ascii")
        dxc = Path(sdk) / "dxc.exe"
        subprocess.run([dxc, "-T", "rootsig_1_1", "-E", "RS", "-Fo", root, source], check=True)
        vs = folder / "fullscreen.cso"
        source.write_text(FULLSCREEN_VS, encoding="ascii")
        subprocess.run([dxc, "-T", "vs_6_0", "-Fo", vs, source], check=True)

        for case, words, reason in (("dynamic", PS_WORDS, ""), ("dropped", PS_WORDS, ""),
                                    ("constant", CONST_WORDS, "constant sampler heap offset"),
                                    ("mixed", MIXED_WORDS, "mixed sampler heap descriptor words")):
            directory, output = folder / case, folder / (case + "-out")
            key = publish(directory, VS_WORDS, program(words, case == "dropped"), root.read_bytes())
            cmd = [frontend, sdk, folder / "cache", "gfx1013-container-pair", "--requests", directory, output]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if reason:
                assert result.returncode and reason in result.stderr, result.stderr
                continue
            assert result.returncode == 0, result.stderr
            warm = subprocess.run(cmd, capture_output=True, text=True, check=True)
            assert "frontend_cache_hit=1" in warm.stdout, warm.stdout
            manifest = output / (key + ".txt")
            assert "smpheap 2048 1 fallback 2047" in manifest.read_text()
            cases = ((0, "3e800000"), (16, "3f400000"), (32, "00000000"), (8, "3f000000"),
                     (2047 * 16, "3f000000"), (0xfffffff0, "3f000000"))
            for offset, expected in (cases if case == "dynamic" else ((16, "3f400000"),)):
                draw = subprocess.run([renderer, adapter, "--startup-sampler-heap", vs, output / (key + ".ps.cso"),
                                       str(offset), expected, manifest], capture_output=True, text=True)
                print(draw.stdout, end="")
                if draw.returncode == 77:
                    return 77
                if case == "dropped":
                    assert draw.returncode == 1 and "expected=3f400000 actual=3e800000" in draw.stderr, draw.stderr
                else:
                    assert draw.returncode == 0, draw.stderr
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
