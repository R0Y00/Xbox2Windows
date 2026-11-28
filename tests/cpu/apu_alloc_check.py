"""Compare the source-built ApuAlloc/ApuFree caller under Windows and pe_map.

The caller imports acphal.dll!ApuAlloc and acphal.dll!ApuFree by name with the call
shape from public DirectXTK12 call sites. A test-only recording provider serves them
from a 64 KiB arena (proposed desktop policy, not console evidence):

- observable result: a 4 KiB, 2 KiB-aligned block is returned, written and read back;
- refusal: a 128 KiB request fails with E_OUTOFMEMORY and leaves the output untouched;
- a 64 KiB request is refused while the first block is live and accepted after ApuFree.

Controls: a lenient provider that never refuses for size makes the caller stop at the
refusal step under both loaders, and pe_map stops before entry when either import is
withheld.
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

PASS = 0x8A0
FAIL_REFUSAL = 0xF02
S_OK = 0
E_OUTOFMEMORY = 0x8007000E
EXPECTED_SIZES = [0x1000, 0x20000, 0x10000, 0x10000]
EXPECTED_RESULTS = [S_OK, E_OUTOFMEMORY, E_OUTOFMEMORY, S_OK]
QUERIES = [
    "get_apu_alloc_call_count",
    "get_apu_free_call_count",
    "get_apu_live_block_count",
    "get_apu_rejected_free_count",
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
    for name in ("get_apu_recorded_size", "get_apu_recorded_alignment", "get_apu_recorded_result"):
        getattr(lib, name).restype = ctypes.c_uint32
        getattr(lib, name).argtypes = [ctypes.c_uint32]
    lib.reset_apu_records.restype = None
    return lib


def compare(mapper_path, exe_path, provider_path, lenient_path, io_path):
    mapper_path = Path(mapper_path)
    exe = Path(exe_path)
    provider = Path(provider_path)
    lenient = Path(lenient_path)
    io_dll = Path(io_path)
    io_name = io_dll.name.casefold()

    imports = imports_by_dll(exe)
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "acphal.dll": sorted(["ApuAlloc", "ApuFree", *QUERIES]),
    })
    print(f"caller imports: {imports}")

    # Windows loader, with and without a live worker at exit.
    native_code = run_native(exe, provider, io_dll)
    check("native exit", native_code, PASS)
    check("native exit with live worker", run_native(exe, provider, io_dll, live_worker=True), PASS)

    # pe_map, with the provider's records read back in this process.
    prov = load_provider(provider)
    io_lib = ctypes.WinDLL(str(io_dll))
    prov.reset_apu_records()
    mapped_code = run_mapped(mapper_path, exe, {"acphal.dll": prov, io_name: io_lib})
    check("mapped exit", mapped_code, PASS)
    check("mapped matches native", mapped_code, native_code)
    check("mapped ApuAlloc calls", prov.get_apu_alloc_call_count(), len(EXPECTED_SIZES))
    check("mapped ApuFree calls", prov.get_apu_free_call_count(), 2)
    check("mapped live blocks", prov.get_apu_live_block_count(), 0)
    check("mapped rejected frees", prov.get_apu_rejected_free_count(), 0)
    check("recorded sizes", [prov.get_apu_recorded_size(i) for i in range(4)], EXPECTED_SIZES)
    check("recorded alignments", [prov.get_apu_recorded_alignment(i) for i in range(4)], [0x800] * 4)
    check("recorded results", [prov.get_apu_recorded_result(i) for i in range(4)], EXPECTED_RESULTS)
    print(f"native and mapped exit {PASS:#x}; results {[hex(r) for r in EXPECTED_RESULTS]}")

    # Control: a provider that never refuses for size is caught at the refusal step.
    check("native lenient exit", run_native(exe, lenient, io_dll), FAIL_REFUSAL)
    lenient_lib = load_provider(lenient)
    lenient_lib.reset_apu_records()
    lenient_code = run_mapped(mapper_path, exe, {"acphal.dll": lenient_lib, io_name: io_lib})
    check("mapped lenient exit", lenient_code, FAIL_REFUSAL)
    check("lenient second result", lenient_lib.get_apu_recorded_result(1), S_OK)
    print(f"lenient provider: native and mapped exit {FAIL_REFUSAL:#x} at the refusal step")

    # Control: withholding either import stops pe_map before entry, with no calls.
    mapper = Mapper(mapper_path)
    exports = {name: ctypes.cast(getattr(prov, name), ctypes.c_void_p).value
               for name in ("ApuAlloc", "ApuFree", *QUERIES)}
    for withheld in ("ApuAlloc", "ApuFree"):
        prov.reset_apu_records()
        table = {name.encode(): address for name, address in exports.items() if name != withheld}
        base, status, detail, _ = mapper(exe, {io_name: io_lib, "acphal.dll": table})
        if base:
            mapper.release(base)
            raise AssertionError(f"caller mapped without {withheld}")
        check(f"withheld {withheld} status", status, STATUS["unresolved"])
        check(f"withheld {withheld} diagnostic", detail, f"unresolved import acphal.dll!{withheld}")
        check(f"withheld {withheld} calls", prov.get_apu_alloc_call_count() + prov.get_apu_free_call_count(), 0)
        print(f"before entry: {detail}")


if __name__ == "__main__":
    compare(*sys.argv[1:])
