"""Record the D3D12ConfigureVirtualMemoryX call shape under Windows and pe_map (#874).

The EXEs mirror the main executable's call: a 32-byte block with one DWORD sample at +4, then two
XMemVirtualAlloc reservations. Recording stand-ins named d3d12_xs.dll and xmem.dll log each call,
so both loaders must produce the same log and exit code. Nothing here implements the contract.
"""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pefile

from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS

SAMPLES = {3: 0x400000000, 4: 0x600000000}  # DWORD at +4 -> reservation size in the observed caller
FAILURE = 0x8007000E  # any failed HRESULT; the EXE must return it unchanged
EXIT_BLOCK, EXIT_CANARY = 1, 2


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): sorted(imp.name.decode() if imp.name else f"#{imp.ordinal}"
                                                       for imp in module.imports)
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def expected_log(sample):
    size = SAMPLES[sample]
    block = bytes(4) + sample.to_bytes(4, "little") + bytes(24)
    return [f"D3D12ConfigureVirtualMemoryX {block.hex()}",
            f"XMemVirtualAlloc 0 {size:x} 20402000 21 4",
            f"XMemVirtualAlloc 0 {size:x} 20402000 3 801"]


def providers(directory):
    return {name: ctypes.WinDLL(str(directory / name))
            for name in ("d3d12_xs.dll", "xmem.dll", "cpu_exe_io.dll")}


def run(mapper_path, exe, mapped, hresult=0, write=0):
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "calls.txt"
        env = dict(os.environ, CONFIGURE_VM_LOG=str(log), CONFIGURE_VM_HRESULT=hex(hresult),
                   CONFIGURE_VM_WRITE=str(write), CPU_CRC_TEST_LIVE_WORKER="0" if mapped else "1")
        if mapped:
            child = subprocess.run([sys.executable, "-B", __file__, "--child", str(mapper_path), str(exe)],
                                   env=env, capture_output=True, text=True, timeout=20)
            if child.returncode:
                raise AssertionError(f"mapped child failed: {child.stderr.strip()}")
            code = int(child.stdout.split()[-1])
        else:
            code = subprocess.run([str(exe)], cwd=str(exe.parent), env=env, timeout=20).returncode
        lines = log.read_text().splitlines() if log.exists() else []
    return code, lines


def compare(mapper_path, *exes):
    mapper_path = Path(mapper_path)
    for exe, sample in zip(map(Path, exes), (3, 4)):
        check(f"{exe.name} imports", imports_by_dll(exe),
              {"d3d12_xs.dll": ["D3D12ConfigureVirtualMemoryX"], "xmem.dll": ["XMemVirtualAlloc"],
               "cpu_exe_io.dll": ["finish_exe"]})

        # Each missing provider stops mapping before the entry point.
        for missing in ("d3d12_xs.dll", "xmem.dll"):
            bound = {name: module for name, module in providers(exe.parent).items() if name != missing}
            base, status, detail, _ = Mapper(mapper_path)(exe, bound)
            if base:
                Mapper(mapper_path).release(base)
                raise AssertionError(f"{exe.name} mapped without {missing}")
            check("missing provider status", status, STATUS["unresolved"])
            if not detail.startswith(f"unresolved import {missing}!"):
                raise AssertionError(f"missing {missing} diagnostic: {detail}")

        cases = (("success", {}, 0), ("failure HRESULT", {"hresult": FAILURE}, FAILURE),
                 ("write inside block", {"write": 1}, EXIT_BLOCK),
                 ("write after block", {"write": 2}, EXIT_CANARY))
        for label, options, exit_code in cases:
            for mapped in (False, True):
                runner = "pe_map" if mapped else "native"
                code, lines = run(mapper_path, exe, mapped, **options)
                check(f"sample {sample} {label} {runner} exit", code, exit_code)
                check(f"sample {sample} {label} {runner} calls", lines, expected_log(sample))
            print(f"sample {sample} {label}: exit {exit_code:#x} under both loaders")
        print(f"sample {sample}: " + " | ".join(expected_log(sample)))


def child(mapper_path, exe):
    exe = Path(exe)
    print(run_mapped(Path(mapper_path), exe, providers(exe.parent)))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:])
    else:
        compare(*sys.argv[1:])
