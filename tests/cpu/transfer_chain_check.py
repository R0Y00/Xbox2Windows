"""Compare the source-built memory transfer prerequisite and failure chain caller under Windows and pe_map.

Exercises the prerequisite, query, and transfer failure isolation contracts under a controlled provider:
- Success path: tools-memory available, working-set statistics provide budget, reaches 16 MiB transfer (exit 0x870).
- Prerequisite blocking: tools-memory unavailable (FALSE) blocks sequence; transfer is never called (exit 0x101).
- Working-set failure: failed statistics query aborts sequence; transfer is never called (exit 0x102).
- Zero budget: when working set reports zero available budget, transfer is skipped (exit 0x103).
- Transfer failure: the provider records the attempted request, and the caller distinguishes failure (exit 0x104).
- Changed-selector patch control: modifying selector 0x2000000a to 0x2000000b fails statistics query and aborts (exit 0x102).
- Clean exit: verified with live worker thread under CPU_CRC_TEST_LIVE_WORKER.
- Mapped return: verifies mapped return to host under pe_map.
- Withholding controls: withholding any prerequisite export stops pe_map before entry with STATUS["unresolved"].
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


FIXTURE_ENV = ("CPU_CRC_TEST_LIVE_WORKER", "XMEM_TEST_TOOLS_MEM_AVAILABLE", "XMEM_TEST_WORKING_SET_SUCCESS",
               "XMEM_TEST_ZERO_BUDGET", "XMEM_TEST_TRANSFER_SUCCESS")


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): sorted(imp.name.decode() for imp in module.imports if imp.name)
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def run_native(exe, provider_dll, io_dll, *, live_worker=False, extra_env=None):
    """Run native caller EXE in temporary directory with xmem.dll and cpu_exe_io.dll."""
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        temp_exe = temp_dir / exe.name
        temp_xmem = temp_dir / "xmem.dll"
        temp_io = temp_dir / "cpu_exe_io.dll"
        shutil.copyfile(exe, temp_exe)
        shutil.copyfile(provider_dll, temp_xmem)
        shutil.copyfile(io_dll, temp_io)
        env = dict(os.environ)
        for name in FIXTURE_ENV:
            env.pop(name, None)
        if live_worker:
            env["CPU_CRC_TEST_LIVE_WORKER"] = "1"
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run([str(temp_exe)], cwd=str(temp_dir), env=env, timeout=20)
        return proc.returncode


def patch_binary(original_path, target_pattern, replacement, patched_path):
    data = Path(original_path).read_bytes()
    idx = data.find(target_pattern)
    if idx == -1:
        raise AssertionError(f"patch pattern {target_pattern.hex()} not found in {original_path}")
    if data.find(target_pattern, idx + 1) != -1:
        raise AssertionError(f"multiple occurrences of pattern {target_pattern.hex()} in {original_path}")
    patched = data[:idx] + replacement + data[idx + len(target_pattern):]
    Path(patched_path).write_bytes(patched)


def compare(mapper_path, exe_path, provider_path, io_path):
    mapper_path = Path(mapper_path)
    exe = Path(exe_path)
    provider = Path(provider_path)
    io_dll = Path(io_path)
    io_name = io_dll.name.casefold()

    # 1. Verify import descriptors in caller PE image
    imports = imports_by_dll(exe)
    expected_xmem_imports = sorted([
        "XMemIsToolsMemAvailable",
        "XMemGetWorkingSetStatistics",
        "XMemTransferMemory",
    ])
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "xmem.dll": expected_xmem_imports,
    })
    print(f"chain caller imports: {imports}")

    # Set up provider functions
    prov_lib = ctypes.WinDLL(str(provider))
    prov_lib.set_chain_tools_mem_available.argtypes = [ctypes.c_int]
    prov_lib.set_chain_tools_mem_available.restype = None
    prov_lib.set_chain_working_set_success.argtypes = [ctypes.c_int]
    prov_lib.set_chain_working_set_success.restype = None
    prov_lib.set_chain_working_set_budget.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
    prov_lib.set_chain_working_set_budget.restype = None
    prov_lib.set_chain_transfer_success.argtypes = [ctypes.c_int]
    prov_lib.set_chain_transfer_success.restype = None

    prov_lib.get_chain_tools_mem_calls.restype = ctypes.c_uint32
    prov_lib.get_chain_working_set_calls.restype = ctypes.c_uint32
    prov_lib.get_chain_transfer_calls.restype = ctypes.c_uint32
    prov_lib.get_chain_transferred_bytes.restype = ctypes.c_uint64
    prov_lib.reset_chain_records.restype = None

    providers = {
        "xmem.dll": prov_lib,
        io_name: ctypes.WinDLL(str(io_dll)),
    }

    # 2. Case 1: Full sequence reaches transfer (exit 0x870)
    # Native execution
    prov_lib.reset_chain_records()
    native_code = run_native(exe, provider, io_dll)
    check("native chain success exit code", native_code, 0x870)
    native_worker_code = run_native(exe, provider, io_dll, live_worker=True)
    check("native chain success exit code with live worker", native_worker_code, 0x870)

    # Mapped execution
    prov_lib.reset_chain_records()
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("mapped chain success exit code", mapped_code, 0x870)
    check("mapped chain tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
    check("mapped chain working-set calls", prov_lib.get_chain_working_set_calls(), 1)
    check("mapped chain transfer calls", prov_lib.get_chain_transfer_calls(), 1)
    check("mapped chain requested bytes", prov_lib.get_chain_transferred_bytes(), 0x1000000)
    print("case 1 (success reaches transfer): exit 0x870, 16 MiB request accepted by the recorder")

    # 3. Case 2: Tools-memory query blocks sequence (exit 0x101)
    # Native execution
    native_blocked = run_native(exe, provider, io_dll, extra_env={"XMEM_TEST_TOOLS_MEM_AVAILABLE": "0"})
    check("native blocked by tools-mem exit code", native_blocked, 0x101)

    # Mapped execution
    prov_lib.reset_chain_records()
    prov_lib.set_chain_tools_mem_available(0)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("blocked by tools-mem exit code", mapped_code, 0x101)
    check("blocked tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
    check("blocked working-set calls", prov_lib.get_chain_working_set_calls(), 0)
    check("blocked transfer calls", prov_lib.get_chain_transfer_calls(), 0)
    print("case 2 (tools-mem blocks): exit 0x101, transfer uncalled")

    # 4. Case 3: Failed working-set query leaves transfer uncalled (exit 0x102)
    # Native execution
    native_ws_fail = run_native(exe, provider, io_dll, extra_env={"XMEM_TEST_WORKING_SET_SUCCESS": "0"})
    check("native failed working-set exit code", native_ws_fail, 0x102)

    # Mapped execution
    prov_lib.reset_chain_records()
    prov_lib.set_chain_tools_mem_available(1)
    prov_lib.set_chain_working_set_success(0)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("failed working-set exit code", mapped_code, 0x102)
    check("failed ws tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
    check("failed ws working-set calls", prov_lib.get_chain_working_set_calls(), 1)
    check("failed ws transfer calls", prov_lib.get_chain_transfer_calls(), 0)
    print("case 3 (working-set failure): exit 0x102, transfer uncalled")

    # 5. Case 4: Zero budget skips transfer (exit 0x103)
    # Native execution
    native_zero = run_native(exe, provider, io_dll, extra_env={"XMEM_TEST_ZERO_BUDGET": "1"})
    check("native zero budget exit code", native_zero, 0x103)

    # Mapped execution
    prov_lib.reset_chain_records()
    prov_lib.set_chain_tools_mem_available(1)
    prov_lib.set_chain_working_set_success(1)
    prov_lib.set_chain_working_set_budget(0x1000000, 0x1000000) # total == used
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("zero budget exit code", mapped_code, 0x103)
    check("zero budget tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
    check("zero budget working-set calls", prov_lib.get_chain_working_set_calls(), 1)
    check("zero budget transfer calls", prov_lib.get_chain_transfer_calls(), 0)
    print("case 4 (zero budget skips transfer): exit 0x103, transfer uncalled")

    # 6. Case 5: Transfer failure handled cleanly (exit 0x104)
    # Native execution
    native_xfer_fail = run_native(exe, provider, io_dll, extra_env={"XMEM_TEST_TRANSFER_SUCCESS": "0"})
    check("native transfer failure exit code", native_xfer_fail, 0x104)

    # Mapped execution
    prov_lib.reset_chain_records()
    prov_lib.set_chain_tools_mem_available(1)
    prov_lib.set_chain_working_set_success(1)
    prov_lib.set_chain_working_set_budget(0x2000000, 0x1000000)
    prov_lib.set_chain_transfer_success(0)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("transfer failure exit code", mapped_code, 0x104)
    check("transfer failure tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
    check("transfer failure working-set calls", prov_lib.get_chain_working_set_calls(), 1)
    check("transfer failure transfer calls", prov_lib.get_chain_transfer_calls(), 1)
    check("transfer failure requested bytes", prov_lib.get_chain_transferred_bytes(), 0x1000000)
    print("case 5 (transfer failure isolated): exit 0x104 after one rejected 16 MiB request")

    # 7. Changed-selector patch control: patch 0x2000000a to 0x2000000b
    # mov ecx, 2000000Ah is encoded as b9 0a 00 00 20
    with TemporaryDirectory() as temp_patch_dir:
        patched_exe = Path(temp_patch_dir) / "patched_chain_caller.exe"
        patch_binary(exe, b"\xb9\x0a\x00\x00\x20", b"\xb9\x0b\x00\x00\x20", patched_exe)

        # Native run of patched caller
        prov_lib.reset_chain_records()
        native_patched = run_native(patched_exe, provider, io_dll)
        check("native patched selector exit code", native_patched, 0x102)

        # Mapped run of patched caller
        prov_lib.reset_chain_records()
        mapped_patched = run_mapped(mapper_path, patched_exe, providers)
        check("mapped patched selector exit code", mapped_patched, 0x102)
        check("patched selector tools-mem calls", prov_lib.get_chain_tools_mem_calls(), 1)
        check("patched selector working-set calls", prov_lib.get_chain_working_set_calls(), 1)
        check("patched selector transfer calls", prov_lib.get_chain_transfer_calls(), 0)
        print("changed-selector patch control: exit 0x102, transfer uncalled")

    # 8. Withholding negative controls
    prov_lib.reset_chain_records()
    mapper = Mapper(mapper_path)
    io_lib = ctypes.WinDLL(str(io_dll))

    tools_fn = ctypes.cast(prov_lib.XMemIsToolsMemAvailable, ctypes.c_void_p).value
    ws_fn = ctypes.cast(prov_lib.XMemGetWorkingSetStatistics, ctypes.c_void_p).value
    xfer_fn = ctypes.cast(prov_lib.XMemTransferMemory, ctypes.c_void_p).value

    # Control A: withhold XMemIsToolsMemAvailable
    prov_without_tools = {
        b"XMemGetWorkingSetStatistics": ws_fn,
        b"XMemTransferMemory": xfer_fn,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": prov_without_tools})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemIsToolsMemAvailable")
    check("withheld XMemIsToolsMemAvailable status", status, STATUS["unresolved"])
    check("withheld XMemIsToolsMemAvailable detail", detail, "unresolved import xmem.dll!XMemIsToolsMemAvailable")
    print(f"withheld tools-mem control: {detail}")

    # Control B: withhold XMemGetWorkingSetStatistics
    prov_without_ws = {
        b"XMemIsToolsMemAvailable": tools_fn,
        b"XMemTransferMemory": xfer_fn,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": prov_without_ws})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemGetWorkingSetStatistics")
    check("withheld XMemGetWorkingSetStatistics status", status, STATUS["unresolved"])
    check("withheld XMemGetWorkingSetStatistics detail", detail, "unresolved import xmem.dll!XMemGetWorkingSetStatistics")
    print(f"withheld working-set control: {detail}")

    # Control C: withhold XMemTransferMemory
    prov_without_xfer = {
        b"XMemIsToolsMemAvailable": tools_fn,
        b"XMemGetWorkingSetStatistics": ws_fn,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": prov_without_xfer})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemTransferMemory")
    check("withheld XMemTransferMemory status", status, STATUS["unresolved"])
    check("withheld XMemTransferMemory detail", detail, "unresolved import xmem.dll!XMemTransferMemory")
    print(f"withheld transfer control: {detail}")
    check("withheld providers leave an empty trace",
          (prov_lib.get_chain_tools_mem_calls(), prov_lib.get_chain_working_set_calls(),
           prov_lib.get_chain_transfer_calls()), (0, 0, 0))


if __name__ == "__main__":
    compare(*sys.argv[1:])
