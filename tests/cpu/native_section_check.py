"""Compare every mapped section of the register probe with the Windows loader, on copies with patched section file fields.

Windows reads raw section data in whole 0x200-byte file sectors, and leaves a section with PointerToRawData 0 zero-filled.
Each section is compared up to the end of its last page: raw data past VirtualSize stays visible up to that page end.
Raw data past the end of the file reads as zero, except that the last section's raw data must end inside the file and
no raw range may reach 4 GiB; copies that break either rule must be refused by both loaders.
"""

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, STATUS

FIELDS = {"VirtualSize": 8, "SizeOfRawData": 16, "PointerToRawData": 20}  # offsets in IMAGE_SECTION_HEADER
SECTOR = 0x200
PAGE = 0x1000  # x64 Windows page size
ERROR_BAD_EXE_FORMAT = 193
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
kernel.LoadLibraryExW.restype = ctypes.c_void_p
kernel.FreeLibrary.argtypes = [ctypes.c_void_p]

# (case, section, header edits from the original section and file size, bytes cut from the end of the file,
# must differ from an exact read). .rsrc edits keep its resource directory intact, which LoadLibraryExW validates.
# .rsrc is the last section of the probe; .rdata and .pdata are not.
CASES = [
    ("no file offset", ".rsrc", lambda s, n: {"PointerToRawData": 0}, 0, True),
    ("start inside a sector", ".rdata", lambda s, n: {"PointerToRawData": s.PointerToRawData + 0x10}, 0, True),
    ("start in the first sector", ".rdata", lambda s, n: {"PointerToRawData": SECTOR // 2}, 0, True),
    ("raw size below a sector", ".rdata", lambda s, n: {"SizeOfRawData": 0x10}, 0, True),
    ("raw data past VirtualSize", ".rdata", lambda s, n: {"VirtualSize": 0x40}, 0, True),
    ("raw data crossing a sector", ".rdata",
     lambda s, n: {"PointerToRawData": s.PointerToRawData + SECTOR - 0x10, "SizeOfRawData": 0x20}, 0, True),
    ("rounded end past the file", ".rsrc",
     lambda s, n: {"VirtualSize": SECTOR, "SizeOfRawData": SECTOR - 0x10}, 0x10, False),
    ("raw data running past the file", ".rdata", lambda s, n: {"PointerToRawData": n - SECTOR}, 0, False),
    ("raw data after the file", ".pdata", lambda s, n: {"PointerToRawData": n + 8 * SECTOR}, 0, False),
    ("no raw data, offset after the file", ".pdata",
     lambda s, n: {"PointerToRawData": n + 8 * SECTOR, "SizeOfRawData": 0}, 0, False),
]

# (case, section, header edits, bytes cut from the end of the file): LoadLibraryExW refuses these copies.
REJECTIONS = [
    ("raw range reaching 4 GiB", ".pdata", lambda s, n: {"PointerToRawData": 0x10, "SizeOfRawData": 2**32 - 0x10}, 0),
    ("last section one byte past the file", ".rsrc", lambda s, n: {}, 1),
]


def layout(image):
    """Return (name, rva, extent) for each section, the extent rounded up to whole pages; refuse images whose load
    would run code."""
    with pefile.PE(str(image), fast_load=True) as pe:
        if pe.OPTIONAL_HEADER.AddressOfEntryPoint or pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].Size:
            raise AssertionError(f"{image.name}: needs no entry point and no imports to be loaded safely")
        return [(s.Name.rstrip(b"\0").decode(), s.VirtualAddress, pages(s.Misc_VirtualSize or s.SizeOfRawData))
                for s in pe.sections]


def pages(size):
    return -(-size // PAGE) * PAGE


def patch(source, target, name, edits, cut):
    """Copy source with one section header edited and the file shortened; return the exact-read bytes."""
    data = bytearray(source.read_bytes())
    with pefile.PE(data=bytes(data), fast_load=True) as pe:
        section = next(s for s in pe.sections if s.Name.rstrip(b"\0") == name.encode())
        fields = edits(section, len(data))
        if cut and (section.PointerToRawData + section.SizeOfRawData != len(data) or section.SizeOfRawData != SECTOR):
            raise AssertionError(f"{name}: needs to be the last {SECTOR:#x} bytes of the file to shorten it")
        for field, value in fields.items():
            struct.pack_into("<I", data, section.get_file_offset() + FIELDS[field], value)
    if cut:
        del data[-cut:]
    target.write_bytes(data)
    span = fields.get("VirtualSize", section.Misc_VirtualSize)
    offset = fields.get("PointerToRawData", section.PointerToRawData)
    size = fields.get("SizeOfRawData", section.SizeOfRawData)
    exact = bytes(data[offset:offset + min(size, span)])
    return (exact + bytes(pages(span)))[:pages(span)]


def read(base, sections):
    return {name: bytes((ctypes.c_ubyte * span).from_address(base + rva)) for name, rva, span in sections}


def windows_sections(image, sections):
    module = kernel.LoadLibraryExW(str(image), None, 0)
    if not module:
        raise AssertionError(f"{image.name}: LoadLibraryExW failed with {ctypes.get_last_error()}")
    try:
        return read(module, sections)
    finally:
        kernel.FreeLibrary(module)


def mapped_sections(mapper, image, sections):
    base, status, detail, requests = mapper(image, {})
    if status != STATUS["ok"] or not base or requests:
        raise AssertionError(f"{image.name}: map returned {status} ({detail}), requests {requests}")
    try:
        return read(base, sections)
    finally:
        mapper.release(base)


def refused(mapper, case, image):
    """Both loaders must refuse the copy: Windows with ERROR_BAD_EXE_FORMAT, pe_map with a format status."""
    module = kernel.LoadLibraryExW(str(image), None, 0)
    if module:
        kernel.FreeLibrary(module)
        raise AssertionError(f"{case}: LoadLibraryExW now loads this copy")
    if ctypes.get_last_error() != ERROR_BAD_EXE_FORMAT:
        raise AssertionError(f"{case}: LoadLibraryExW failed with {ctypes.get_last_error()}")
    base, status, detail, _ = mapper(image, {})
    if base:
        mapper.release(base)
    if status != STATUS["format"]:
        raise AssertionError(f"{case}: Windows refuses this copy, but pe_map returned {status} ({detail})")
    print(f"{case}: refused by LoadLibraryExW and by pe_map ({detail})")


def compare(mapper, case, image, exact=None, target=None):
    sections = layout(image)
    native = windows_sections(image, sections)
    mapped = mapped_sections(mapper, image, sections)
    for name, _, _ in sections:
        if mapped[name] != native[name]:
            first = next(i for i, (a, b) in enumerate(zip(mapped[name], native[name])) if a != b)
            hint = ", as an exact read of the header fields gives" if name == target and mapped[name] == exact else ""
            raise AssertionError(f"{case}: pe_map {name} differs from Windows at byte {first:#x}{hint}")
    print(f"{case}: {len(sections)} sections match the Windows loader byte for byte")
    return native


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD), ("RegionSize", ctypes.c_size_t),
                ("State", wintypes.DWORD), ("Protect", wintypes.DWORD), ("Type", wintypes.DWORD)]


def protections(base, sections):
    result = {}
    for name, rva, _ in sections:
        info = MEMORY_BASIC_INFORMATION()
        kernel.VirtualQuery(ctypes.c_void_p(base + rva), ctypes.byref(info), ctypes.sizeof(info))
        result[name] = info.Protect
    return result


def edit_header(source, target, edit):
    with pefile.PE(str(source)) as pe:
        edit(pe)
        pe.write(str(target))


def check_protections(mapper, folder, probe):
    """Section page protections must match Windows, also when .pdata loses IMAGE_SCN_MEM_READ.

    RtlAddFunctionTable reads the table, so pe_map must register it before applying a protection that
    leaves it unreadable. Windows maps such an image and keeps the page PAGE_NOACCESS."""
    def unreadable_pdata(pe):
        section = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".pdata")
        section.Characteristics &= ~(0x40000000 | 0x80000000 | 0x20000000)

    unpatched = folder / "protect_original.dll"
    unpatched.write_bytes(probe.read_bytes())
    patched = folder / "protect_pdata.dll"
    edit_header(probe, patched, unreadable_pdata)
    for case, image in (("unpatched", unpatched), (".pdata without read access", patched)):
        if image is patched:
            base, status, detail, _ = mapper(image, {})
            if base:
                mapper.release(base)
            if base or status != STATUS["unsupported"] or "exception table is not readable" not in detail:
                raise AssertionError(f"{case}: expected unsupported, got {status} ({detail})")
            continue
        sections = layout(image)
        module = kernel.LoadLibraryExW(str(image), None, 0)
        if not module:
            raise AssertionError(f"{case}: LoadLibraryExW failed with {ctypes.get_last_error()}")
        try:
            native = protections(module, sections)
        finally:
            kernel.FreeLibrary(module)
        base, status, detail, _ = mapper(image, {})
        if status != STATUS["ok"] or not base:
            raise AssertionError(f"{case}: map returned {status} ({detail})")
        try:
            mapped = protections(base, sections)
        finally:
            mapper.release(base)
        if mapped != native:
            raise AssertionError(f"{case}: pe_map protections {mapped} differ from Windows {native}")
        print(f"{case}: {len(sections)} section protections match the Windows loader")


# (SectionAlignment, pe_map status, reason). Windows refuses each with ERROR_BAD_EXE_FORMAT (193).
ALIGNMENTS = [
    (0x2000, "format", "not aligned to 0x2000"),
    (0x1001, "format", "is not a power of two"),
    (0x800, "unsupported", "below the page size"),
]


def check_alignments(mapper, folder, probe):
    """A section that does not start on SectionAlignment could share a page with the headers or with
    another section, whose protection it would then change. Windows refuses these alignments."""
    for alignment, status_name, reason in ALIGNMENTS:
        image = folder / f"align_{alignment:x}.dll"
        edit_header(probe, image, lambda pe: setattr(pe.OPTIONAL_HEADER, "SectionAlignment", alignment))
        module = kernel.LoadLibraryExW(str(image), None, 0)
        if module:
            kernel.FreeLibrary(module)
            raise AssertionError(f"SectionAlignment {alignment:#x}: LoadLibraryExW accepted the image")
        error = ctypes.get_last_error()
        base, status, detail, _ = mapper(image, {})
        if base:
            mapper.release(base)
        if status != STATUS[status_name] or reason not in detail:
            raise AssertionError(f"SectionAlignment {alignment:#x}: map returned {status} ({detail}), "
                                 f"expected {status_name} with {reason!r}")
        print(f"SectionAlignment {alignment:#x}: Windows refuses with {error}, pe_map with {status_name}: {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("probe", type=Path, help="source-built image with no entry point and no imports")
    args = parser.parse_args()
    # Each case prints "start" before it loads anything. With line buffering, the output CTest shows
    # after its TIMEOUT names a case that started but never returned.
    sys.stdout.reconfigure(line_buffering=True)
    # Refused images must return loader errors instead of a Bad Image dialog on interactive runners.
    kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001)  # SEM_FAILCRITICALERRORS
    mapper = Mapper(args.mapper)

    with TemporaryDirectory() as folder:
        original = Path(folder) / "section_original.dll"
        original.write_bytes(args.probe.read_bytes())
        before = compare(mapper, "unpatched", original)

        failures = []
        for index, (case, name, edits, cut, discriminating) in enumerate(CASES):
            image = Path(folder) / f"section_case{index}.dll"
            exact = patch(args.probe, image, name, edits, cut)
            print(f"start {name} {case}")
            try:
                after = compare(mapper, f"{name} {case}", image, exact, name)
            except AssertionError as error:
                print(f"FAIL {error}")
                failures.append(case)
                continue
            if discriminating and after[name] == exact:
                raise AssertionError(f"{name} {case}: Windows matches an exact read, so the case checks nothing")
            changed = [other for other in before if other != name and after[other] != before[other]]
            if changed:
                raise AssertionError(f"{name} {case}: patching {name} changed other sections: {changed}")
            if case == "no file offset" and any(after[name]):
                raise AssertionError(f"Windows no longer zero-fills {name} without a file offset")
        for index, (case, name, edits, cut) in enumerate(REJECTIONS):
            image = Path(folder) / f"section_refused{index}.dll"
            patch(args.probe, image, name, edits, cut)
            print(f"start {name} {case}")
            try:
                refused(mapper, f"{name} {case}", image)
            except AssertionError as error:
                print(f"FAIL {error}")
                failures.append(case)
        if failures:
            total = len(CASES) + len(REJECTIONS)
            raise AssertionError(f"{len(failures)} of {total} cases differ from the Windows loader: {failures}")
        check_protections(mapper, Path(folder), args.probe)
        check_alignments(mapper, Path(folder), args.probe)


if __name__ == "__main__":
    main()
