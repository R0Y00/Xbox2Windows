"""Compare the same reserve-only XMem fixture under Windows and pe_map."""
import ctypes
from pathlib import Path
import subprocess
import sys

from exe_entry_check import check
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS


def compare(mapper_path, exe, xmem):
    kernel = {"kernel32.dll": ctypes.WinDLL("kernel32.dll")}
    mapper = Mapper(mapper_path)
    base, status, detail, requests = mapper(exe, kernel)
    if base:
        mapper.release(base)
        raise AssertionError("mapped the EXE without xmem.dll")
    check("missing xmem provider", (status, detail, requests[-1]),
          (STATUS["unresolved"], "unresolved import xmem.dll!XMemVirtualAlloc",
           ("xmem.dll", "XMemVirtualAlloc")))
    mapped = [sys.executable, "-B", __file__, "--mapped", mapper_path, exe, xmem]
    for runner, command in (("Windows", [exe]), ("pe_map", mapped)):
        child = subprocess.run(command, capture_output=True, timeout=20)
        check(runner, (child.returncode, child.stdout, child.stderr), (0, b"", b""))
        print(f"{runner}: both 16 GiB and 24 GiB pairs stayed reserved, refused bad requests, and released")


if __name__ == "__main__":
    if sys.argv[1] == "--mapped":
        mapper_path, exe, xmem = map(Path, sys.argv[2:])
        sys.exit(run_mapped(mapper_path, exe, {"kernel32.dll": ctypes.WinDLL("kernel32.dll"),
                                              "xmem.dll": ctypes.WinDLL(str(xmem))}))
    compare(*map(Path, sys.argv[1:]))
