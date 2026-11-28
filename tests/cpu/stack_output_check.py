"""Compare imported stack outputs and verify failures before generating code."""
import ctypes
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import Binding, translate


def compare(caller, lifted, name, expected):
    native_fn = getattr(ctypes.CDLL(str(caller)), name)
    lifted_fn = getattr(ctypes.CDLL(str(lifted)), f"lifted_{name}")
    for fn in (native_fn, lifted_fn):
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = ctypes.c_uint64
    for value in (0, 1, 0xffffffffffffffff, 0x123489abcdef):
        actual = (native_fn(0, value, 0), lifted_fn(0, value, 0))
        want = expected(value)
        if actual != (want, want):
            raise AssertionError(f"{name} input {value:#x}: {actual}, expected {want:#x} on both sides")
    print(f"{name}: four original and translated outputs agree")


def main(caller, provider, with_status, with_void):
    compare(caller, with_status, "from_frame",
            lambda value: (((value ^ 0x711) ^ 0x1981ee11d960cef1) + 0x98231417) & 0xffffffffffffffff)
    compare(caller, with_void, "from_rsp", lambda _: 0x867594414c474131)
    output = Binding(provider, "write_constant", None, 1, 0, (1, 8))
    for name, binding, reason in (
        ("overlaps_home", output, "overlap with callee home space"),
        ("bypasses_lea", output, "needs a proven frame address"),
        ("misaligned", output, "misaligned stack output"),
        ("from_rsp", output._replace(output=(1, 4)), "undefined stack byte"),
        ("from_rsp", output._replace(output=(2, 8)), "invalid stack output argument"),
        ("from_rsp", output._replace(output=None), "frame address escapes"),
    ):
        try:
            translate(caller, [name], binding=binding)
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{name}: expected {reason!r}, got {exc}") from exc
        else:
            raise AssertionError(f"{name}: translation should have refused {reason}")
        print(f"{name}: {reason}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
