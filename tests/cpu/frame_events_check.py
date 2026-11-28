"""Compare the source-built Xbox frame events caller under Windows and pe_map (#874).

Exercises the frame events scheduling, origin wait, and token advancement controls:
- Public sample call shape: SetFrameIntervalX -> ScheduleFrameEventX(ORIGIN) once -> WaitFrameEventX(ORIGIN)
  with bounded/infinite wait, carrying the returned token into PresentX (follow-up).
- Controlled tick timing check:
  - Bounded timeout before tick leaves output sentinel unchanged (WAIT_TIMEOUT).
  - One Schedule registration starts tick delivery.
  - Waiters block until ticks arrive, verified by an explicit blocked-waiter witness.
  - Second wait blocks and delivers advancing token (token > last_token) without re-registering.
- Separated fault controls:
  - Sentinel clobber fault on refusal is caught specifically at step 2 (exit 0xE03).
  - Early release fault before tick is caught specifically at step 3 (exit 0xE04).
- Multi-loader verification:
  - Runs natively under Windows loader (exit 0x874).
  - Runs with live worker thread under CPU_CRC_TEST_LIVE_WORKER.
  - Runs mapped under pe_map (exit 0x874).
  - Withholding d3d12_xs.dll stops pe_map at import resolution with STATUS["unresolved"].
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


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): sorted(imp.name.decode() for imp in module.imports if imp.name)
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def run_native(exe, provider_dll, io_dll, *, live_worker=False, extra_env=None):
    """Run native caller EXE in temporary directory with d3d12_xs.dll and cpu_exe_io.dll."""
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        temp_exe = temp_dir / exe.name
        temp_d3d12 = temp_dir / "d3d12_xs.dll"
        temp_io = temp_dir / "cpu_exe_io.dll"
        shutil.copyfile(exe, temp_exe)
        shutil.copyfile(provider_dll, temp_d3d12)
        shutil.copyfile(io_dll, temp_io)
        env = dict(os.environ)
        if live_worker:
            env["CPU_CRC_TEST_LIVE_WORKER"] = "1"
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run([str(temp_exe)], cwd=str(temp_dir), env=env, timeout=20)
        return proc.returncode


def compare(mapper_path, exe_path, provider_path, clobber_provider_path, early_provider_path, io_path):
    mapper_path = Path(mapper_path)
    exe = Path(exe_path)
    provider = Path(provider_path)
    clobber_provider = Path(clobber_provider_path)
    early_provider = Path(early_provider_path)
    io_dll = Path(io_path)
    io_name = io_dll.name.casefold()

    # 1. Verify import descriptors in caller PE image
    imports = imports_by_dll(exe)
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "d3d12_xs.dll": ["ScheduleFrameEventX", "SetFrameIntervalX", "WaitFrameEventX"],
    })
    print(f"frame events caller imports: {imports}")

    # Set up provider functions for verification
    prov_lib = ctypes.WinDLL(str(provider))
    prov_lib.get_frame_events_origin_waits.restype = ctypes.c_uint64
    prov_lib.get_frame_events_last_token.restype = ctypes.c_uint64
    prov_lib.get_frame_events_blocked_waits.restype = ctypes.c_uint64
    prov_lib.get_frame_events_timeouts.restype = ctypes.c_uint64
    prov_lib.get_frame_events_rejections.restype = ctypes.c_uint64
    prov_lib.get_frame_events_schedule_calls.restype = ctypes.c_uint64
    prov_lib.get_frame_events_mode.restype = ctypes.c_uint32

    # Case 1: Native Windows execution with valid recording provider (exit 0x874)
    code = run_native(exe, provider, io_dll)
    check("native execution exit", code, 0x874)
    print("native execution: exit 0x874 PASS")

    # Case 2: Clean exit with live worker thread
    code_worker = run_native(exe, provider, io_dll, live_worker=True)
    check("native execution with live worker exit", code_worker, 0x874)
    print("native execution (live worker): exit 0x874 PASS")

    # Case 3A: Sentinel clobber fault control (must exit specifically 0xE03 at step 2)
    code_clobber = run_native(exe, clobber_provider, io_dll)
    check("sentinel clobber fault exit", code_clobber, 0xE03)
    print(f"mutation 1 (sentinel clobber): caught at step 2 with exit {code_clobber:#x} PASS")

    # Case 3B: Early release fault control (must exit specifically 0xE04 at step 3)
    code_early = run_native(exe, early_provider, io_dll)
    check("early release fault exit", code_early, 0xE04)
    print(f"mutation 2 (early release): caught at step 3 with exit {code_early:#x} PASS")

    # Case 4: Mapped execution under pe_map
    io_loaded = ctypes.WinDLL(str(io_dll))
    providers_dict = {
        "d3d12_xs.dll": prov_lib,
        io_name: io_loaded,
    }
    mapped_code = run_mapped(mapper_path, exe, providers_dict)
    check("mapped execution exit", mapped_code, 0x874)
    print("mapped execution under pe_map: exit 0x874 PASS")
    observed = (prov_lib.get_frame_events_schedule_calls(), prov_lib.get_frame_events_blocked_waits(),
                prov_lib.get_frame_events_origin_waits(), prov_lib.get_frame_events_last_token(),
                prov_lib.get_frame_events_timeouts(), prov_lib.get_frame_events_rejections(),
                prov_lib.get_frame_events_mode())
    check("mapped recorder (registrations, waits signaled, waits completed, token, timeouts, refusals, mode)",
          observed, (1, 2, 2, 2, 1, 2, 0))
    print("mapped recorder: one registration, two pending waits delivered, one timeout, two refusals PASS")

    # Case 5: Withholding d3d12_xs.dll stops pe_map at import resolution
    withheld_providers = {io_name: io_loaded}
    mapper = Mapper(mapper_path)
    base, status, detail, _ = mapper(exe, withheld_providers)
    if base:
        mapper.release(base)
        raise AssertionError("image mapped successfully despite withheld d3d12_xs.dll")
    check("withheld provider status", status, STATUS["unresolved"])
    if not detail.startswith("unresolved import d3d12_xs.dll!"):
        raise AssertionError(f"unexpected diagnostic on withholding d3d12_xs.dll: {detail}")
    print(f"withheld d3d12_xs.dll control: caught unresolved import ({detail}) PASS")


def main():
    if len(sys.argv) != 7:
        print("Usage: frame_events_check.py <pe_map> <caller.exe> <provider.dll> <clobber.dll> <early.dll> <io.dll>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


if __name__ == "__main__":
    main()
