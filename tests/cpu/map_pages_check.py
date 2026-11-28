"""Compare the source-built physical-page mapping caller under Windows and pe_map.

Exercises contiguous mapping and scatter mapping contracts under controlled recording provider:
- XMemMapPhysicalPages: 64 KiB alignment check, valid map, valid unmap (NULL page array), and unaligned/null refusals.
- XMemMapPhysicalPagesScatter: valid scatter map, valid scatter unmap (NULL page array), and null array refusal.
- Changed-argument controls: wrong page IDs and swapped scatter destinations are refused.
- Trace verification: verifies expected count, page IDs, and ordered addresses in recorded trace.
- Live-worker control: verifies clean finish_exe process exit under CPU_CRC_TEST_LIVE_WORKER.
- Mapped return: verifies mapped return to host under pe_map.
- Independent negative controls: pe_map stops before entry when each named export is withheld independently.
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

# Synthetic fixture inputs chosen for this fixture
FIXTURE_ALIGNED_ADDR = 0x10000000
FIXTURE_SCATTER_ADDR0 = 0x10000000
FIXTURE_SCATTER_ADDR1 = 0x10010000
FIXTURE_PAGE_0 = 1
FIXTURE_PAGE_1 = 2
FIXTURE_PAGE_COUNT = 2


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
        "XMemMapPhysicalPages",
        "XMemMapPhysicalPagesScatter",
        "get_map_pages_call_count",
        "get_map_pages_unmap_count",
        "get_scatter_map_call_count",
        "get_scatter_unmap_count",
        "get_argument_mismatch_count",
    ])
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "xmem.dll": expected_xmem_imports,
    })
    print(f"caller imports: {imports}")

    # 2. Native execution under Windows loader (standard and with live worker)
    native_code = run_native(exe, provider, io_dll)
    check("native mapping exit code", native_code, 0x870)
    native_worker_code = run_native(exe, provider, io_dll, live_worker=True)
    check("native mapping exit code with live worker", native_worker_code, 0x870)

    prov_lib = ctypes.WinDLL(str(provider))
    prov_lib.get_map_pages_call_count.restype = ctypes.c_uint32
    prov_lib.get_map_pages_unmap_count.restype = ctypes.c_uint32
    prov_lib.get_scatter_map_call_count.restype = ctypes.c_uint32
    prov_lib.get_scatter_unmap_count.restype = ctypes.c_uint32
    prov_lib.get_argument_mismatch_count.restype = ctypes.c_uint32
    prov_lib.get_recorded_map_address.restype = ctypes.c_uint64
    prov_lib.get_recorded_map_count.restype = ctypes.c_uint64
    prov_lib.get_recorded_map_page.restype = ctypes.c_uint64
    prov_lib.get_recorded_map_page.argtypes = [ctypes.c_uint32]
    prov_lib.get_recorded_scatter_address.restype = ctypes.c_uint64
    prov_lib.get_recorded_scatter_address.argtypes = [ctypes.c_uint32]
    prov_lib.get_recorded_scatter_count.restype = ctypes.c_uint64
    prov_lib.get_recorded_scatter_page.restype = ctypes.c_uint64
    prov_lib.get_recorded_scatter_page.argtypes = [ctypes.c_uint32]
    prov_lib.reset_map_pages_records.restype = None

    # Reset records before mapped run
    prov_lib.reset_map_pages_records()

    # 3. Mapped execution under pe_map
    providers = {
        "xmem.dll": prov_lib,
        io_name: ctypes.WinDLL(str(io_dll)),
    }
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("mapped mapping exit code", mapped_code, 0x870)
    check("mapped matches native", mapped_code, native_code)

    # Verify provider recorded calls: 1 map + 1 unmap for contiguous, 1 map + 1 unmap for scatter
    check("mapped contiguous map calls", prov_lib.get_map_pages_call_count(), 1)
    check("mapped contiguous unmap calls", prov_lib.get_map_pages_unmap_count(), 1)
    check("mapped scatter map calls", prov_lib.get_scatter_map_call_count(), 1)
    check("mapped scatter unmap calls", prov_lib.get_scatter_unmap_count(), 1)
    check("mapped argument mismatches", prov_lib.get_argument_mismatch_count(), 2)

    # Verify recorded trace distinguishes expected count, page IDs, and ordered addresses
    check("trace contiguous address", prov_lib.get_recorded_map_address(), FIXTURE_ALIGNED_ADDR)
    check("trace contiguous page count", prov_lib.get_recorded_map_count(), FIXTURE_PAGE_COUNT)
    check("trace contiguous page 0", prov_lib.get_recorded_map_page(0), FIXTURE_PAGE_0)
    check("trace contiguous page 1", prov_lib.get_recorded_map_page(1), FIXTURE_PAGE_1)

    check("trace scatter address 0", prov_lib.get_recorded_scatter_address(0), FIXTURE_SCATTER_ADDR0)
    check("trace scatter address 1", prov_lib.get_recorded_scatter_address(1), FIXTURE_SCATTER_ADDR1)
    check("trace scatter page count", prov_lib.get_recorded_scatter_count(), FIXTURE_PAGE_COUNT)
    check("trace scatter page 0", prov_lib.get_recorded_scatter_page(0), FIXTURE_PAGE_0)
    check("trace scatter page 1", prov_lib.get_recorded_scatter_page(1), FIXTURE_PAGE_1)

    print("contiguous and scatter mapping: native and mapped exit 0x870, trace and ordered addresses verified")

    # 4. Independent negative controls: withhold each named mapping export independently
    mapper = Mapper(mapper_path)
    io_lib = ctypes.WinDLL(str(io_dll))

    # Helper function pointers for partial provider table
    map_fn = ctypes.cast(prov_lib.XMemMapPhysicalPages, ctypes.c_void_p).value
    scatter_fn = ctypes.cast(prov_lib.XMemMapPhysicalPagesScatter, ctypes.c_void_p).value
    cnt1 = ctypes.cast(prov_lib.get_map_pages_call_count, ctypes.c_void_p).value
    cnt2 = ctypes.cast(prov_lib.get_map_pages_unmap_count, ctypes.c_void_p).value
    cnt3 = ctypes.cast(prov_lib.get_scatter_map_call_count, ctypes.c_void_p).value
    cnt4 = ctypes.cast(prov_lib.get_scatter_unmap_count, ctypes.c_void_p).value
    cnt5 = ctypes.cast(prov_lib.get_argument_mismatch_count, ctypes.c_void_p).value

    # Control A: Withhold XMemMapPhysicalPages (provide Scatter + query funcs)
    provider_without_map = {
        b"XMemMapPhysicalPagesScatter": scatter_fn,
        b"get_map_pages_call_count": cnt1,
        b"get_map_pages_unmap_count": cnt2,
        b"get_scatter_map_call_count": cnt3,
        b"get_scatter_unmap_count": cnt4,
        b"get_argument_mismatch_count": cnt5,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": provider_without_map})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemMapPhysicalPages")
    check("withheld XMemMapPhysicalPages status", status, STATUS["unresolved"])
    check("withheld XMemMapPhysicalPages diagnostic", detail, "unresolved import xmem.dll!XMemMapPhysicalPages")
    print(f"negative control without contiguous map: {detail}")

    # Control B: Withhold XMemMapPhysicalPagesScatter (provide Map + query funcs)
    provider_without_scatter = {
        b"XMemMapPhysicalPages": map_fn,
        b"get_map_pages_call_count": cnt1,
        b"get_map_pages_unmap_count": cnt2,
        b"get_scatter_map_call_count": cnt3,
        b"get_scatter_unmap_count": cnt4,
        b"get_argument_mismatch_count": cnt5,
    }
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xmem.dll": provider_without_scatter})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without XMemMapPhysicalPagesScatter")
    check("withheld XMemMapPhysicalPagesScatter status", status, STATUS["unresolved"])
    check("withheld XMemMapPhysicalPagesScatter diagnostic", detail, "unresolved import xmem.dll!XMemMapPhysicalPagesScatter")
    print(f"negative control without scatter map: {detail}")


if __name__ == "__main__":
    compare(*sys.argv[1:])
