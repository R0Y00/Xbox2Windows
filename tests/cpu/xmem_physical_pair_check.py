"""Start the physical-page EXE under the Windows loader and pe_map; compare the exit codes.

tests/cpu/xmem_physical_pair.asm imports XMemAllocatePhysicalPages and XMemFreePhysicalPages by name
from xmem.dll and finish_exe (exe_startup.asm) from cpu_exe_io.dll. Its body returns 0x880 when every
step passes, or 0xF00 + the first failing step. The FULL_COUNT and SKIP_RELEASE variants are negative
controls with known failing steps. The comparison follows triggered02's independent fixture for #880.
"""
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from exe_process import launch
from native_map_check import Mapper, STATUS, kernel

PASS = 0x880
EXPECTED = {"original": PASS, "full_count": 0xF04, "skip_release": 0xF06}
PAIR = ("XMemAllocatePhysicalPages", "XMemFreePhysicalPages")
XMEM, IO = "xmem.dll", "cpu_exe_io.dll"
REQUESTS = sorted([[XMEM, PAIR[0]], [XMEM, PAIR[1]], [IO, "finish_exe"]])


def stage(folder, image, xmem, io):
    folder = Path(folder)
    for source, name in ((image, Path(image).name), (xmem, XMEM), (io, IO)):
        shutil.copyfile(source, folder / name)
    return folder / Path(image).name


def native(image):
    row = launch([image], b"", timeout=30)
    if not row["process_signaled"]:
        raise AssertionError(f"{image.name}: process did not end: {row}")
    if row["stdout"]["bytes"] or row["stderr"]["bytes"]:
        raise AssertionError(f"{image.name}: unexpected output {row['stdout']['bytes']!r} {row['stderr']['bytes']!r}")
    return row["code"]


def child(mapper_path, image, xmem, io, omit=""):
    """Map and start the EXE in this (child) process; print the observation as JSON."""
    mapper = Mapper(mapper_path)
    run_entry = ctypes.CDLL(str(mapper_path)).pe_map_run
    run_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    run_entry.restype = ctypes.c_int
    module = ctypes.CDLL(str(xmem))
    bindings = {XMEM: module, IO.casefold(): ctypes.CDLL(str(io))}
    if omit:
        # A provider that lacks one of the pair: bind the other export by address only.
        kept = {name.encode(): kernel.GetProcAddress(module._handle, name.encode())
                for name in PAIR if name != omit}
        bindings[XMEM] = kept
    base, status, detail, requests = mapper(Path(image), bindings)
    if not base:
        print(json.dumps({"status": status, "detail": detail, "requests": requests}))
        return
    code = ctypes.c_uint32(0xdeadbeef)
    started = run_entry(base, ctypes.byref(code))
    mapper.release(base)
    print(json.dumps({"status": status, "started": started, "exit": code.value, "requests": requests}))


def mapped(mapper_path, image, xmem, io, omit=""):
    result = subprocess.run([sys.executable, "-B", __file__, "--child", str(mapper_path), str(image),
                             str(xmem), str(io), omit], capture_output=True, text=True, timeout=60)
    if result.returncode or result.stderr:
        raise AssertionError(f"mapped {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def main(mapper_path, original, full_count, skip_release, xmem, io):
    images = {"original": original, "full_count": full_count, "skip_release": skip_release}
    with TemporaryDirectory() as temp:
        for name, image in images.items():
            staged = stage(temp, image, xmem, io)
            expected = EXPECTED[name]
            launched = native(staged)
            result = mapped(mapper_path, staged, Path(temp) / XMEM, Path(temp) / IO)
            if result.get("status") != STATUS["ok"] or result.get("started") != 1:
                raise AssertionError(f"mapped {name}: {result}")
            requests = sorted([dll.casefold(), symbol] for dll, symbol in result["requests"])
            if requests != REQUESTS:
                raise AssertionError(f"mapped {name}: import requests {result['requests']}")
            if launched != expected or result["exit"] != expected:
                raise AssertionError(f"{name}: Windows launch exited {launched:#x}, pe_map {result['exit']:#x}, "
                                     f"expected {expected:#x} for both")
            print(f"{name}: Windows launch {launched:#x}, pe_map {result['exit']:#x}")

        # A provider without one of the pair stops the mapping before the entry point runs.
        staged = Path(temp) / Path(original).name
        for omit in PAIR:
            result = mapped(mapper_path, staged, Path(temp) / XMEM, Path(temp) / IO, omit)
            if result.get("status") != STATUS["unresolved"] or "started" in result or \
                    result.get("detail") != f"unresolved import {XMEM}!{omit}":
                raise AssertionError(f"provider without {omit}: {result}")
            print(f"provider without {omit}: {result['detail']}")
    print("XMem physical-page pair: PASS")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:6], *(sys.argv[6:7] or [""]))
    else:
        main(*map(Path, sys.argv[1:]))
