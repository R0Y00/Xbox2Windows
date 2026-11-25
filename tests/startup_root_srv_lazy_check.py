"""Startup root srv lazy check."""
from pathlib import Path
import sys
import tempfile

from startup_address_srv_check import draw_texture
from startup_root_cbv_check import CLEAR, HLSL


def check(directory, consumer, backend):
    yellow = bytes((255, 255, 0, 255))
    expected = (yellow * 4 + CLEAR * 4) * 4
    for lazy in (False, True):
        folder = directory / ("lazy" if lazy else "copy")
        folder.mkdir()
        log_path = folder / "trap.log"
        env = {"X2W_TEST_ROOT_CBV": "root_srv", "X2W_TRAP_LOG": str(log_path)}
        if lazy:
            env["X2W_ROOT_SRV_LAZY"] = "1"
        pixels = draw_texture(folder, consumer, backend, HLSL, env, binding="srv 0 buf root 0")
        if pixels is None:
            return 77
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        copied = "x2w-render root SRV 0 at title" in log
        deferred = "X2W_ROOT_SRV_LAZY=1: root SRVs are copied at a real-pipeline draw only" in log
        print(f"{backend} lazy={int(lazy)}: pixels {'exact' if pixels == expected else 'DIFFER'}, record-time copy line {int(copied)}, "
              f"deferred line {int(deferred)}")
        if pixels != expected:
            raise AssertionError(f"lazy={lazy}: actual {pixels.hex()} expected {expected.hex()}")
        if copied == lazy or deferred != lazy:
            raise AssertionError(f"lazy={lazy}: copy line {copied}, deferred line {deferred}")
    print("PASS")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
