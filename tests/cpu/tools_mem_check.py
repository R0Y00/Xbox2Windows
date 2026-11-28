"""Compare the source-built XMemIsToolsMemAvailable caller under Windows and pe_map.

The recording providers return FALSE (fallback, exit 42) or TRUE (tools path, exit 84).
The caller checks that both invocations match and the provider counted two calls.
A missing provider must stop pe_map before entry.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): sorted(imp.name.decode() if imp.name else f"#{imp.ordinal}" for imp in module.imports)
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def run_native(exe, provider_dll, io_path):
    """Run the native caller EXE in a temporary directory where provider_dll is served as xmem.dll."""
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        temp_exe = temp_dir / exe.name
        temp_xmem = temp_dir / "xmem.dll"
        shutil.copyfile(exe, temp_exe)
        shutil.copyfile(provider_dll, temp_xmem)
        shutil.copyfile(io_path, temp_dir / io_path.name)
        env = dict(os.environ, CPU_CRC_TEST_LIVE_WORKER="1")
        proc = subprocess.run([str(temp_exe)], cwd=str(temp_dir), env=env, timeout=20)
        return proc.returncode


def compare(mapper_path, exe_path, provider_path, provider_devkit_path, io_path):
    mapper_path, exe, io_path = map(Path, (mapper_path, exe_path, io_path))
    io_name = io_path.name.casefold()
    imports = imports_by_dll(exe)
    check("caller imports", imports, {io_name: ["finish_exe"],
                                      "xmem.dll": ["XMemIsToolsMemAvailable", "get_tools_mem_call_count"]})

    io = ctypes.WinDLL(str(io_path))
    mapper = Mapper(mapper_path)
    cases = (("FALSE", Path(provider_path), 42), ("TRUE", Path(provider_devkit_path), 84))
    false_provider = ctypes.WinDLL(str(cases[0][1]))
    count = ctypes.cast(false_provider.get_tools_mem_call_count, ctypes.c_void_p).value
    base, status, detail, _ = mapper(exe, {io_name: io, "xmem.dll": {b"get_tools_mem_call_count": count}})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without the tools-memory export")
    check("missing export status", status, STATUS["unresolved"])
    check("missing export diagnostic", detail, "unresolved import xmem.dll!XMemIsToolsMemAvailable")
    print(f"negative control before entry: {detail}")

    with patch.dict(os.environ, {"CPU_CRC_TEST_LIVE_WORKER": "0"}):
        for label, provider, expected in cases:
            native = run_native(exe, provider, io_path)
            check(f"{label} native exit", native, expected)

            module = ctypes.WinDLL(str(provider))
            module.reset_tools_mem_call_count.restype = None
            module.get_tools_mem_call_count.restype = ctypes.c_uint32
            module.reset_tools_mem_call_count()
            mapped = run_mapped(mapper_path, exe, {io_name: io, "xmem.dll": module})
            check(f"{label} mapped exit", mapped, expected)
            check(f"{label} mapped provider count", module.get_tools_mem_call_count(), 2)
            print(f"controlled {label}: native and mapped exit {expected}, two matching queries each")


if __name__ == "__main__":
    compare(*sys.argv[1:])
