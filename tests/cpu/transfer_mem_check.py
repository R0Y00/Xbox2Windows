"""Compare the source-built memory transfer caller under Windows and pe_map.

Exercises XMemTransferMemory contract under controlled recording provider:
- XMemTransferMemory: 2 MiB boundary alignment, valid partition 1, and 16 MiB maximum boundary (test-provider policies).
- Refusal controls: invalid partition, zero bytes, unaligned byte count, and oversized byte count.
- Trace verification: verifies all six ordered argument pairs (partition, bytes, result) in the recorded trace.
- Changed-first-call control: changing the first request from 2 MiB to 4 MiB is caught by internal trace checks (exit 0xF0B) and alters recorded trace.
- Live-worker control: verifies clean finish_exe process exit under CPU_CRC_TEST_LIVE_WORKER.
- Mapped return: verifies mapped return to host under pe_map.
- Independent negative control: pe_map stops before entry when XMemTransferMemory is withheld.
Native console transfer semantics and partition effects remain open.
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


def run_native(exe, provider_dll, io_dll, *, live_worker=False):
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
        if live_worker:
            env["CPU_CRC_TEST_LIVE_WORKER"] = "1"
        proc = subprocess.run([str(temp_exe)], cwd=str(temp_dir), env=env, timeout=20)
        return proc.returncode


def compare(mapper_path, exe_path, provider_path, io_path):
    mapper_path = Path(mapper_path)
    exe = Path(exe_path)
    provider = Path(provider_path)
    io_dll = Path(io_path)
    io_name = io_dll.name.casefold()

    # 1. Verify import descriptors in caller PE image
    imports = imports_by_dll(exe)
    expected_xmem_imports = sorted([
        "XMemTransferMemory",
        "get_argument_rejection_count",
        "get_recorded_call_bytes",
        "get_recorded_call_partition",
        "get_recorded_call_result",
        "get_total_call_count",
        "get_transfer_call_count",
    ])
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "xmem.dll": expected_xmem_imports,
    })
    print(f"caller imports: {imports}")

    # 2. Native execution under Windows loader (standard and with live worker)
    native_code = run_native(exe, provider, io_dll)
    check("native transfer exit code", native_code, 0x870)
    native_worker_code = run_native(exe, provider, io_dll, live_worker=True)
    check("native transfer exit code with live worker", native_worker_code, 0x870)

    prov_lib = ctypes.WinDLL(str(provider))
    prov_lib.get_total_call_count.restype = ctypes.c_uint32
    prov_lib.get_transfer_call_count.restype = ctypes.c_uint32
    prov_lib.get_argument_rejection_count.restype = ctypes.c_uint32
    prov_lib.get_recorded_call_partition.restype = ctypes.c_uint32
    prov_lib.get_recorded_call_partition.argtypes = [ctypes.c_uint32]
    prov_lib.get_recorded_call_bytes.restype = ctypes.c_uint64
    prov_lib.get_recorded_call_bytes.argtypes = [ctypes.c_uint32]
    prov_lib.get_recorded_call_result.restype = ctypes.c_uint32
    prov_lib.get_recorded_call_result.argtypes = [ctypes.c_uint32]
    prov_lib.reset_transfer_mem_records.restype = None

    # Reset records before mapped run
    prov_lib.reset_transfer_mem_records()

    # 3. Mapped execution under pe_map
    providers = {
        "xmem.dll": prov_lib,
        io_name: ctypes.WinDLL(str(io_dll)),
    }
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("mapped transfer exit code", mapped_code, 0x870)
    check("mapped matches native", mapped_code, native_code)

    # Verify provider recorded calls: 2 accepted calls, 4 rejections, 6 total
    check("mapped total call count", prov_lib.get_total_call_count(), 6)
    check("mapped transfer call count", prov_lib.get_transfer_call_count(), 2)
    check("mapped rejection count", prov_lib.get_argument_rejection_count(), 4)

    # Verify all six ordered argument pairs and results
    expected_trace = [
        (1, 0x200000, 1),    # Step 1: Valid 2 MiB transfer
        (1, 0x1000000, 1),   # Step 2: Valid 16 MiB transfer
        (0, 0x200000, 0),    # Step 3: Rejected invalid partition
        (1, 0, 0),           # Step 4: Rejected zero byte count
        (1, 0x200100, 0),    # Step 5: Rejected unaligned byte count
        (1, 0x2000000, 0),   # Step 6: Rejected oversized byte count
    ]
    for i, (expected_part, expected_bytes, expected_res) in enumerate(expected_trace):
        check(f"call {i} partition", prov_lib.get_recorded_call_partition(i), expected_part)
        check(f"call {i} bytes", prov_lib.get_recorded_call_bytes(i), expected_bytes)
        check(f"call {i} result", prov_lib.get_recorded_call_result(i), expected_res)

    print("memory transfer: native and mapped exit 0x870, all six ordered argument pairs verified")

    # 4. Changed-first-call control: patching the first 2 MiB request to 4 MiB
    # must alter recorded trace and fail the caller's internal trace check (exit 0xF0B).
    raw = exe.read_bytes()
    # In transfer_mem_caller.asm: "mov rdx, 0200000h" encodes as 48 c7 c2 00 00 20 00
    target_imm = b"\x48\xc7\xc2\x00\x00\x20\x00"
    replacement_imm = b"\x48\xc7\xc2\x00\x00\x40\x00"
    if target_imm not in raw:
        raise AssertionError("target 2 MiB immediate not found in caller binary")
    changed_raw = raw.replace(target_imm, replacement_imm, 1)

    with TemporaryDirectory() as temp:
        changed_exe = Path(temp) / "changed_transfer_caller.exe"
        changed_exe.write_bytes(changed_raw)

        # Under native loader: changed first call triggers internal trace check failure (0xF0B)
        prov_lib.reset_transfer_mem_records()
        changed_native_code = run_native(changed_exe, provider, io_dll)
        check("changed-first-call native exit code", changed_native_code, 0xF0B)

        # Under pe_map: changed first call also returns 0xF0B and records 4 MiB in call 0
        prov_lib.reset_transfer_mem_records()
        changed_mapped_code = run_mapped(mapper_path, changed_exe, providers)
        check("changed-first-call mapped exit code", changed_mapped_code, 0xF0B)
        check("changed call 0 recorded bytes", prov_lib.get_recorded_call_bytes(0), 0x400000)
        print("changed-first-call control: exit 0xF0B under both loaders, recorded 4 MiB in call 0")

    # 5. Independent negative control: withhold XMemTransferMemory
    mapper = Mapper(mapper_path)
    io_lib = ctypes.WinDLL(str(io_dll))

    cnt0 = ctypes.cast(prov_lib.get_total_call_count, ctypes.c_void_p).value
    cnt1 = ctypes.cast(prov_lib.get_transfer_call_count, ctypes.c_void_p).value
    cnt2 = ctypes.cast(prov_lib.get_argument_rejection_count, ctypes.c_void_p).value
    q_part = ctypes.cast(prov_lib.get_recorded_call_partition, ctypes.c_void_p).value
    q_bytes = ctypes.cast(prov_lib.get_recorded_call_bytes, ctypes.c_void_p).value
    q_res = ctypes.cast(prov_lib.get_recorded_call_result, ctypes.c_void_p).value

    provider_without_transfer = {
        b"get_total_call_count": cnt0,
        b"get_transfer_call_count": cnt1,
        b"get_argument_rejection_count": cnt2,
        b"get_recorded_call_partition": q_part,
        b"get_recorded_call_bytes": q_bytes,
        b"get_recorded_call_result": q_res,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": provider_without_transfer})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemTransferMemory")
    check("withheld XMemTransferMemory status", status, STATUS["unresolved"])
    check("withheld XMemTransferMemory diagnostic", detail, "unresolved import xmem.dll!XMemTransferMemory")
    print(f"negative control without transfer: {detail}")


if __name__ == "__main__":
    compare(*sys.argv[1:])
