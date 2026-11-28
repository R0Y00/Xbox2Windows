"""Compare the source-built ACP address and memory mapping caller under Windows and pe_map.

The caller imports acphal.dll exports by name:
  - ApuAlloc
  - ApuFree
  - ApuMapVirtualAddress
  - ApuMapApuAddress
  - ApuIsVirtualAddressValid

Call shapes follow public MIT DirectXTK12 call sites and APU mapping semantics:
- observable result: bidirectional identity ApuMapApuAddress(ApuMapVirtualAddress(ptr)) == ptr
- linear offset preservation: ptr + offset <-> apu_addr + offset
- validity lifecycle: ApuIsVirtualAddressValid is TRUE while allocated, FALSE after ApuFree
- refusal controls: null, stack, and unmapped APU addresses are rejected.

Controls:
- A faulty provider where ApuMapApuAddress fails causes the caller to abort at Step 4 (0xF04).
- pe_map stops before entry when any export is withheld.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS

PASS = 0x8A1
FAIL_FAULTY = 0xF04
QUERIES = [
    "get_apu_map_virtual_calls",
    "get_apu_map_apu_calls",
    "get_apu_is_valid_calls",
    "get_apu_unmapped_refusals",
    "get_apu_alloc_count",
    "get_apu_free_count",
]


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        result = {}
        for module in pe.DIRECTORY_ENTRY_IMPORT:
            if any(not imp.name for imp in module.imports):
                raise AssertionError(f"{module.dll.decode()} has an ordinal import")
            result[module.dll.decode().casefold()] = sorted(imp.name.decode() for imp in module.imports)
        return result


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def run_native(exe, provider_dll, io_dll, *, live_worker=False):
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        temp_exe = temp_dir / exe.name
        shutil.copyfile(exe, temp_exe)
        shutil.copyfile(provider_dll, temp_dir / "acphal.dll")
        shutil.copyfile(io_dll, temp_dir / io_dll.name)
        env = dict(os.environ)
        env.pop("CPU_CRC_TEST_LIVE_WORKER", None)
        if live_worker:
            env["CPU_CRC_TEST_LIVE_WORKER"] = "1"
        return subprocess.run([str(temp_exe)], cwd=str(temp_dir), env=env, timeout=20).returncode


def load_provider(path):
    lib = ctypes.WinDLL(str(path))
    for name in QUERIES:
        getattr(lib, name).restype = ctypes.c_uint32
    lib.reset_apu_address_records.restype = None
    return lib


def compare(mapper_path, exe_path, provider_path, faulty_path, io_path):
    mapper_path = Path(mapper_path)
    exe = Path(exe_path)
    provider = Path(provider_path)
    faulty = Path(faulty_path)
    io_dll = Path(io_path)
    io_name = io_dll.name.casefold()

    imports = imports_by_dll(exe)
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "acphal.dll": sorted([
            "ApuAlloc",
            "ApuFree",
            "ApuIsVirtualAddressValid",
            "ApuMapApuAddress",
            "ApuMapVirtualAddress",
            "get_apu_unmapped_refusals",
        ]),
    })
    print(f"caller imports: {imports}")

    # Windows loader, with and without a live worker at exit.
    native_code = run_native(exe, provider, io_dll)
    check("native exit", native_code, PASS)
    check("native exit with live worker", run_native(exe, provider, io_dll, live_worker=True), PASS)

    # pe_map, with the provider's records read back in this process.
    prov = load_provider(provider)
    io_lib = ctypes.WinDLL(str(io_dll))
    prov.reset_apu_address_records()
    mapped_code = run_mapped(mapper_path, exe, {"acphal.dll": prov, io_name: io_lib})
    check("mapped exit", mapped_code, PASS)
    check("mapped matches native", mapped_code, native_code)
    check("mapped ApuAlloc calls", prov.get_apu_alloc_count(), 1)
    check("mapped ApuFree calls", prov.get_apu_free_count(), 1)
    if prov.get_apu_map_virtual_calls() < 4:
        raise AssertionError(f"Expected at least 4 map_virtual calls, got {prov.get_apu_map_virtual_calls()}")
    if prov.get_apu_map_apu_calls() < 4:
        raise AssertionError(f"Expected at least 4 map_apu calls, got {prov.get_apu_map_apu_calls()}")
    if prov.get_apu_is_valid_calls() < 4:
        raise AssertionError(f"Expected at least 4 is_valid calls, got {prov.get_apu_is_valid_calls()}")
    if prov.get_apu_unmapped_refusals() < 6:
        raise AssertionError(f"Expected at least 6 unmapped refusals, got {prov.get_apu_unmapped_refusals()}")
    print(f"native and mapped exit {PASS:#x}; all address translation assertions passed")

    # Control: a faulty provider is caught at Step 4.
    check("native faulty exit", run_native(exe, faulty, io_dll), FAIL_FAULTY)
    faulty_lib = load_provider(faulty)
    faulty_lib.reset_apu_address_records()
    faulty_code = run_mapped(mapper_path, exe, {"acphal.dll": faulty_lib, io_name: io_lib})
    check("mapped faulty exit", faulty_code, FAIL_FAULTY)
    print(f"faulty provider: native and mapped exit {FAIL_FAULTY:#x} at Step 4")

    # Control: withholding any export stops pe_map before entry, with no calls.
    mapper = Mapper(mapper_path)
    exports = {
        name: ctypes.cast(getattr(prov, name), ctypes.c_void_p).value
        for name in (
            "ApuAlloc",
            "ApuFree",
            "ApuMapVirtualAddress",
            "ApuMapApuAddress",
            "ApuIsVirtualAddressValid",
            "get_apu_unmapped_refusals",
        )
    }
    for withheld in ("ApuMapVirtualAddress", "ApuMapApuAddress", "ApuIsVirtualAddressValid"):
        prov.reset_apu_address_records()
        table = {name.encode(): address for name, address in exports.items() if name != withheld}
        base, status, detail, _ = mapper(exe, {io_name: io_lib, "acphal.dll": table})
        if base:
            mapper.release(base)
            raise AssertionError(f"caller mapped without {withheld}")
        check(f"withheld {withheld} status", status, STATUS["unresolved"])
        check(f"withheld {withheld} diagnostic", detail, f"unresolved import acphal.dll!{withheld}")
        check(f"withheld {withheld} calls", prov.get_apu_alloc_count() + prov.get_apu_map_virtual_calls(), 0)
        print(f"before entry: {detail}")


if __name__ == "__main__":
    compare(*sys.argv[1:])
