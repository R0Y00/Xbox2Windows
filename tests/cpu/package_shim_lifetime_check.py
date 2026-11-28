"""Run the same source-built EXE with and without a process-lifetime provider."""
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from pe_package_preflight import inspect
from pe_run_package import prototype, run


def main(mapper, fixture, shim):
    module_handle = prototype(C.WinDLL("kernel32"), "GetModuleHandleW", [W.LPCWSTR], W.HMODULE)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        shutil.copyfile(fixture, root / fixture.name)
        plan = inspect(root, fixture.name)
        providers = {shim.name.casefold(): shim}
        for should_pin in (False, True):
            result = run(root, plan, mapper, providers, set(), pin_shims=providers if should_pin else ())
            if result["exit_code"] != 42 or not result["entry_executed"]:
                raise AssertionError(f"package did not return 42: {result}")
            if bool(module_handle(shim.name)) != should_pin:
                raise AssertionError(f"{shim.name} lifetime differs with pin={should_pin}")
    print("unselected provider unloads after entry returns; selected provider stays loaded")


if __name__ == "__main__":
    main(*(Path(arg).resolve(strict=True) for arg in sys.argv[1:]))
