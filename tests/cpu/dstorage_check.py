"""Compare the source-built DirectStorage factory and queue caller under Windows and pe_map.

Exercises the typed factory adapter, queue creation, and delayed-submit negative controls:
- Case 1 (Success path): DStorageGetFactory acquires IID_FixtureFactory ({ca891497-6f8f-4d03-b1a2-f239b4ce41af}),
  configures staging buffer size (32 MiB), creates IID_FixtureQueue ({82af1362-6445-4b96-9fbf-5d2d416b605e}),
  enqueues request with destination buffer, verifies completion is withheld before Submit and destination
  buffer remains unmodified (0xAA), calls Submit, verifies exact readback transfer, and releases objects (exit 0x870).
- Case 2 (PC interface refusal): PC IDStorageFactory ({6924ea0c-c3cd-4826-b10a-f64f4ed927c1}) is refused with
  E_NOINTERFACE and leaves *ppv untouched.
- Case 3 (Factory failure refusal): Failed factory creation aborts sequence (exit 0x301).
- Case 4 (Queue failure refusal): Failed queue creation aborts sequence (exit 0x303).
- Case 5 (Eager-provider control): Faulty provider completing before Submit is caught by delayed-submit check (exit 0x304).
- Changed-IID patch control: Altering factory IID fails QueryInterface with E_NOINTERFACE (exit 0x301).
- Clean exit: Verified with live worker thread under CPU_CRC_TEST_LIVE_WORKER.
- Mapped return: Verified under pe_map.
- Withholding control: Withholding DStorageGetFactory stops pe_map at import resolution with STATUS["unresolved"].
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
    """Run native caller EXE in temporary directory with xcrdapi.dll and cpu_exe_io.dll."""
    with TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        temp_exe = temp_dir / exe.name
        temp_xcrd = temp_dir / "xcrdapi.dll"
        temp_io = temp_dir / "cpu_exe_io.dll"
        shutil.copyfile(exe, temp_exe)
        shutil.copyfile(provider_dll, temp_xcrd)
        shutil.copyfile(io_dll, temp_io)
        env = dict(os.environ)
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
    check("caller imports", imports, {
        io_name: ["finish_exe"],
        "xcrdapi.dll": ["DStorageGetFactory"],
    })
    print(f"dstorage caller imports: {imports}")

    # Set up provider functions
    prov_lib = ctypes.WinDLL(str(provider))
    prov_lib.get_dstorage_factory_calls.restype = ctypes.c_uint32
    prov_lib.get_dstorage_queue_calls.restype = ctypes.c_uint32
    prov_lib.get_dstorage_staging_buffer_size.restype = ctypes.c_uint32
    prov_lib.get_dstorage_staged_requests_count.restype = ctypes.c_uint32
    prov_lib.get_dstorage_submitted_count.restype = ctypes.c_uint32
    prov_lib.set_dstorage_factory_success.argtypes = [ctypes.c_int]
    prov_lib.set_dstorage_factory_success.restype = None
    prov_lib.set_dstorage_queue_success.argtypes = [ctypes.c_int]
    prov_lib.set_dstorage_queue_success.restype = None
    prov_lib.set_dstorage_eager_completion.argtypes = [ctypes.c_int]
    prov_lib.set_dstorage_eager_completion.restype = None
    prov_lib.reset_dstorage_records.restype = None

    providers = {
        "xcrdapi.dll": prov_lib,
        io_name: ctypes.WinDLL(str(io_dll)),
    }

    # 2. Case 1: Full sequence reaches completion after submit (exit 0x870)
    # Native execution
    prov_lib.reset_dstorage_records()
    native_code = run_native(exe, provider, io_dll)
    check("native dstorage success exit code", native_code, 0x870)
    native_worker_code = run_native(exe, provider, io_dll, live_worker=True)
    check("native dstorage success exit code with live worker", native_worker_code, 0x870)

    # Mapped execution
    prov_lib.reset_dstorage_records()
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("mapped dstorage success exit code", mapped_code, 0x870)
    check("mapped factory calls", prov_lib.get_dstorage_factory_calls(), 2)  # step 0 (refusal) + step 1 (success)
    check("mapped staging buffer size", prov_lib.get_dstorage_staging_buffer_size(), 0x2000000)
    check("mapped queue calls", prov_lib.get_dstorage_queue_calls(), 1)
    check("mapped staged requests", prov_lib.get_dstorage_staged_requests_count(), 1)
    check("mapped submitted batches", prov_lib.get_dstorage_submitted_count(), 1)
    print("case 1 (success reaches submit): exit 0x870, 32 MiB buffer, delayed submit & readback verified")

    # 3. Case 2: Factory creation refusal (exit 0x301)
    prov_lib.reset_dstorage_records()
    prov_lib.set_dstorage_factory_success(0)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("failed factory exit code", mapped_code, 0x301)
    check("failed factory queue calls", prov_lib.get_dstorage_queue_calls(), 0)
    print("case 2 (factory failure refused): exit 0x301, queue uncreated")

    # 4. Case 3: Queue creation refusal (exit 0x303)
    prov_lib.reset_dstorage_records()
    prov_lib.set_dstorage_factory_success(1)
    prov_lib.set_dstorage_queue_success(0)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("failed queue exit code", mapped_code, 0x303)
    check("failed queue buffer size", prov_lib.get_dstorage_staging_buffer_size(), 0x2000000)
    check("failed queue queue calls", prov_lib.get_dstorage_queue_calls(), 0)
    print("case 3 (queue creation refused): exit 0x303, staging intact")

    # 5. Case 4: Eager-provider negative control (exit 0x304)
    # A faulty provider completing prematurely before Submit is caught and rejected
    prov_lib.reset_dstorage_records()
    prov_lib.set_dstorage_factory_success(1)
    prov_lib.set_dstorage_queue_success(1)
    prov_lib.set_dstorage_eager_completion(1)
    mapped_code = run_mapped(mapper_path, exe, providers)
    check("eager provider rejected exit code", mapped_code, 0x304)
    print("case 4 (eager-provider negative control): premature completion caught with exit 0x304")

    # 6. Changed-IID patch control: patch IID_FixtureFactory
    target_iid_bytes = bytes([
        0x97, 0x14, 0x89, 0xca, 0x8f, 0x6f, 0x03, 0x4d, 0xb1, 0xa2, 0xf2, 0x39, 0xb4, 0xce, 0x41, 0xaf
    ])
    patched_iid_bytes = bytes([
        0x98, 0x14, 0x89, 0xca, 0x8f, 0x6f, 0x03, 0x4d, 0xb1, 0xa2, 0xf2, 0x39, 0xb4, 0xce, 0x41, 0xaf
    ])
    with TemporaryDirectory() as temp_patch_dir:
        patched_exe = Path(temp_patch_dir) / "patched_dstorage_caller.exe"
        patch_binary(exe, target_iid_bytes, patched_iid_bytes, patched_exe)

        prov_lib.reset_dstorage_records()
        native_patched = run_native(patched_exe, provider, io_dll)
        check("native patched iid exit code", native_patched, 0x301)

        prov_lib.reset_dstorage_records()
        mapped_patched = run_mapped(mapper_path, patched_exe, providers)
        check("mapped patched iid exit code", mapped_patched, 0x301)
        check("patched iid queue calls", prov_lib.get_dstorage_queue_calls(), 0)
        print("changed-IID patch control: exit 0x301, E_NOINTERFACE rejected unrecognized interface")

    # 7. Negative withholding control: withhold DStorageGetFactory
    mapper = Mapper(mapper_path)
    io_lib = ctypes.WinDLL(str(io_dll))
    base, status, detail, _ = mapper(exe, {io_name: io_lib, "xcrdapi.dll": {}})
    if base:
        mapper.release(base)
        raise AssertionError("caller mapped without DStorageGetFactory")
    check("withheld DStorageGetFactory status", status, STATUS["unresolved"])
    check("withheld DStorageGetFactory detail", detail, "unresolved import xcrdapi.dll!DStorageGetFactory")
    print(f"withheld DStorageGetFactory control: {detail}")


if __name__ == "__main__":
    compare(*sys.argv[1:])
