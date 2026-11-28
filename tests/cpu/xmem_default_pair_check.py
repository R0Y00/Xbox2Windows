"""Start the XMem default-pair EXE under the Windows loader and pe_map; compare the exit codes.

tests/cpu/xmem_default_pair.asm imports XMemAllocDefault and XMemFreeDefault by name from xmem.dll,
VirtualQuery, SetLastError and GetLastError from kernel32.dll, heap_block_busy from the heap probe
and finish_exe (exe_startup.asm) from cpu_exe_io.dll. Its body returns 0x875 when all 21 steps pass,
or 0xF00 + the first failing step. Each assertion below prints a named line.
"""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from exe_process import launch
from native_map_check import Mapper, STATUS, kernel

PASS = 0x875
PAIR = ("XMemAllocDefault", "XMemFreeDefault")
XMEM, IO, PROBE = "xmem.dll", "cpu_exe_io.dll", "cpu_heap_probe.dll"
REQUESTS = sorted([[XMEM, PAIR[0]], [XMEM, PAIR[1]], ["kernel32.dll", "VirtualQuery"],
                   ["kernel32.dll", "SetLastError"], ["kernel32.dll", "GetLastError"],
                   [PROBE, "heap_block_busy"], [IO, "finish_exe"]])
# Step groups of the body, for naming a failing step.
GROUPS = [(range(1, 6), "heap_block"), (range(6, 10), "graphics_block"), (range(10, 15), "rejected_inputs"),
          (range(15, 19), "induced_failure"), (range(19, 21), "graphics_release_virtualquery"),
          (range(21, 22), "heap_release_ownership")]
EXPECTED = {"original": PASS, "skip_gpu_free": 0xF14, "skip_heap_free": 0xF15, "uninduced": 0xF0F}


def describe(code):
    if code == PASS:
        return "all steps"
    step = code - 0xF00
    return next((f"step {step} ({name})" for steps, name in GROUPS if step in steps), f"code {code:#x}")


def stage(folder, image, xmem, io, probe):
    folder = Path(folder)
    for source, name in ((image, Path(image).name), (xmem, XMEM), (io, IO), (probe, PROBE)):
        shutil.copyfile(source, folder / name)
    return folder / Path(image).name


def import_names(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {entry.dll.decode().casefold(): sorted(i.name.decode() if i.name else f"#{i.ordinal}"
                                                      for i in entry.imports)
                for entry in pe.DIRECTORY_ENTRY_IMPORT}


def native(image):
    row = launch([image], b"", timeout=60)
    if not row["process_signaled"]:
        raise AssertionError(f"{image.name}: process did not end: {row}")
    if row["stdout"]["bytes"] or row["stderr"]["bytes"]:
        raise AssertionError(f"{image.name}: unexpected output {row['stdout']['bytes']!r} {row['stderr']['bytes']!r}")
    return row["code"]


def child(mapper_path, image, folder, omit=""):
    """Map and start the EXE in this (child) process; print the observation as JSON."""
    folder = Path(folder)
    mapper = Mapper(mapper_path)
    run_entry = ctypes.CDLL(str(mapper_path)).pe_map_run
    run_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    run_entry.restype = ctypes.c_int
    module = ctypes.CDLL(str(folder / XMEM))
    bindings = {XMEM: module, IO: ctypes.CDLL(str(folder / IO)), PROBE: ctypes.CDLL(str(folder / PROBE)),
                "kernel32.dll": kernel}
    if omit:
        # A provider that lacks one of the pair: bind the other export by address only.
        bindings[XMEM] = {name.encode(): kernel.GetProcAddress(module._handle, name.encode())
                          for name in PAIR if name != omit}
    base, status, detail, requests = mapper(Path(image), bindings)
    if not base:
        print(json.dumps({"status": status, "detail": detail, "requests": requests}))
        return
    code = ctypes.c_uint32(0xdeadbeef)
    started = run_entry(base, ctypes.byref(code))
    mapper.release(base)
    print(json.dumps({"status": status, "started": started, "exit": code.value, "requests": requests}))


def mapped(mapper_path, image, folder, omit=""):
    result = subprocess.run([sys.executable, "-B", __file__, "--child", str(mapper_path), str(image),
                             str(folder), omit], capture_output=True, text=True, timeout=120)
    if result.returncode or result.stderr:
        raise AssertionError(f"mapped {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def main(mapper_path, original, skip_gpu_free, skip_heap_free, uninduced, xmem, io, probe):
    images = {"original": original, "skip_gpu_free": skip_gpu_free, "skip_heap_free": skip_heap_free,
              "uninduced": uninduced}
    for name, image in images.items():
        imports = import_names(image)
        if imports.get(XMEM) != sorted(PAIR):
            raise AssertionError(f"pe_import_names {name}: xmem.dll imports {imports.get(XMEM)}")
    print(f"pe_import_names: xmem.dll imports {', '.join(sorted(PAIR))} by name in all {len(images)} EXEs")

    with TemporaryDirectory() as temp:
        for name, image in images.items():
            staged = stage(temp, image, xmem, io, probe)
            expected = EXPECTED[name]
            launched = native(staged)
            result = mapped(mapper_path, staged, temp)
            if result.get("status") != STATUS["ok"] or result.get("started") != 1:
                raise AssertionError(f"mapped {name}: {result}")
            requests = sorted([dll.casefold(), symbol] for dll, symbol in result["requests"])
            if requests != REQUESTS:
                raise AssertionError(f"mapped_resolver_names {name}: requests {result['requests']}")
            if launched != expected or result["exit"] != expected:
                raise AssertionError(f"{name}: Windows launch exited {launched:#x} ({describe(launched)}), "
                                     f"pe_map {result['exit']:#x} ({describe(result['exit'])}), "
                                     f"expected {expected:#x} ({describe(expected)}) for both")
            if name == "original":
                print(f"mapped_resolver_names: the resolver was asked for {', '.join(PAIR)} "
                      f"and {len(REQUESTS) - 2} other imports, nothing else")
                for _, group in GROUPS:
                    print(f"{group}: passed under the Windows loader and pe_map")
            print(f"{name}: Windows launch {launched:#x}, pe_map {result['exit']:#x} ({describe(expected)})")

        staged = Path(temp) / Path(original).name
        for omit in PAIR:
            result = mapped(mapper_path, staged, temp, omit)
            if result.get("status") != STATUS["unresolved"] or "started" in result or \
                    result.get("detail") != f"unresolved import {XMEM}!{omit}":
                raise AssertionError(f"provider without {omit}: {result}")
            print(f"missing_provider_{omit}: {result['detail']}, before the entry point")
    print("XMem default pair native and mapped: PASS")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:5], *(sys.argv[5:6] or [""]))
    else:
        main(*map(Path, sys.argv[1:]))
