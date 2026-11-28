"""Observe loader-time cookie initialization before a source-built image's CRT could run."""
import ctypes
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS, kernel

DEFAULT_COOKIE = 0x2b992ddfa232
PRESEEDED_COOKIE = 0x531813da5590


def mapped_cookie(mapper, image):
    base, status, detail, requests = mapper(image, {})
    if not base or status or requests:
        raise AssertionError(f"{image.name}: unexpected mapping result {status}: {detail}, {requests}")
    try:
        read_cookie = ctypes.WINFUNCTYPE(ctypes.c_uint64)(mapper.export(base, b"read_cookie"))
        return read_cookie()
    finally:
        if not mapper.release(base):
            raise AssertionError(f"{image.name}: failed to release mapping")


def compare(mapper, image, expected):
    native = ctypes.WinDLL(str(image), winmode=0x900)
    try:
        native.read_cookie.restype = ctypes.c_uint64
        values = (native.read_cookie(), mapped_cookie(mapper, image))
    finally:
        if not kernel.FreeLibrary(native._handle):
            raise AssertionError(f"{image.name}: failed to unload native fixture")
    if expected is None:
        if any(not 0 < val < 1 << 48 or val == DEFAULT_COOKIE for val in values):
            raise AssertionError(f"{image.name}: both loaders must seed the default cookie: {values}")
    elif values != (expected, expected):
        raise AssertionError(f"{image.name}: both loaders must preserve {expected:#x}: {values}")
    print(f"{image.name}: native and mapped agree on {'seeding' if expected is None else 'preservation'}")


def control(source, output, *, size=None, directory_size=None, flags=None, initial=None, bad_slot=False):
    with pefile.PE(str(source)) as pe:
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[10]
        config = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct
        if size is not None:
            config.Size = size
        if directory_size is not None:
            directory.Size = directory_size
        if flags is not None:
            config.GuardFlags = flags
        if initial is not None:
            pe.set_qword_at_rva(config.SecurityCookie - pe.OPTIONAL_HEADER.ImageBase, initial)
        if bad_slot:
            config.SecurityCookie = pe.OPTIONAL_HEADER.ImageBase + directory.VirtualAddress
        pe.write(str(output))


def main(mapper_path, dll, exe, io):
    mapper = Mapper(mapper_path)
    with pefile.PE(str(dll)) as pe:
        config = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct
        if pe.get_qword_at_rva(config.SecurityCookie - pe.OPTIONAL_HEADER.ImageBase) != DEFAULT_COOKIE:
            raise AssertionError("load-config fixture does not point to the default cookie")
    compare(mapper, dll, None)
    native_code = subprocess.run([str(exe)], capture_output=True, timeout=10).returncode
    mapped_code = run_mapped(mapper_path, exe, {io.name: ctypes.CDLL(str(io), winmode=0x900)})
    if (native_code, mapped_code) != (42, 42):
        raise AssertionError(f"EXE entry must see a seeded cookie: native={native_code} mapped={mapped_code}")
    print("EXE startup: both loaders return 42 with a seeded cookie")
    with TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for name, options, expected in (
            ("prefix-at-bound", {"size": 0x70, "directory_size": 0x70}, None),
            ("short-structure", {"size": 0x6f}, DEFAULT_COOKIE),
            ("short-directory", {"directory_size": 0x6f}, DEFAULT_COOKIE),
            ("unused", {"flags": 0x800}, DEFAULT_COOKIE),
            ("preseeded", {"initial": PRESEEDED_COOKIE}, PRESEEDED_COOKIE),
        ):
            copy = folder / (name + ".dll")
            control(dll, copy, **options)
            compare(mapper, copy, expected)
        bad = folder / "readonly.dll"
        control(dll, bad, bad_slot=True)
        base, status, detail, _ = mapper(bad, {})
        if base:
            mapper.release(base)
        if base or status != STATUS["format"] or "writable image section" not in detail:
            raise AssertionError(f"read-only cookie target should fail before entry: {status}: {detail}")
        print("read-only cookie destination rejected by pe_map")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
