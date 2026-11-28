"""Compare a callback pointer stored in the fixture's load-config field under Windows and pe_map."""

import ctypes
from pathlib import Path
import subprocess
import sys

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from mapped_exe_chain_check import run_mapped

SLOT_OFFSET = 0x78
SLOT_SIZE = 0x138
EXPECTED = {
    "present": 0xB111,
    "absent": 0xB211,
    "alternate": 0xB151,
}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def inspect(image):
    with pefile.PE(str(image)) as pe:
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[10]
        config = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct
        slot_rva = directory.VirtualAddress + SLOT_OFFSET
        section = next(section for section in pe.sections
                       if section.VirtualAddress <= slot_rva <
                       section.VirtualAddress + max(section.Misc_VirtualSize, section.SizeOfRawData))
        relocations = [(entry.rva, entry.type) for block in getattr(pe, "DIRECTORY_ENTRY_BASERELOC", [])
                       for entry in block.entries]
        value = pe.get_qword_at_rva(slot_rva)
        return {
            "size": config.Size,
            "directory_size": directory.Size,
            "slot_rva": slot_rva,
            "slot_value": value,
            "read_only": not (section.Characteristics & 0x20000000),
            "relocated": any(rva == slot_rva and kind == 10 for rva, kind in relocations),
            "guard_flags": config.GuardFlags,
        }


def native_code(image):
    done = subprocess.run([str(image)], cwd=str(image.parent), capture_output=True, timeout=20)
    return done.returncode & 0xFFFFFFFF


def mapped_code(mapper_path, image, io):
    provider = ctypes.CDLL(str(io), winmode=0x900)
    return run_mapped(mapper_path, image, {io.name.casefold(): provider})


def describe(fields):
    return (f"size={fields['size']:#x} dir={fields['directory_size']:#x} slot={fields['slot_rva']:#x} "
            f"value={fields['slot_value']:#x} read_only={fields['read_only']} "
            f"relocated={fields['relocated']} guard_flags={fields['guard_flags']:#x}")


def main(mapper_path, present, absent, alternate, io):
    mapper_path, present, absent, alternate, io = map(
        Path, (mapper_path, present, absent, alternate, io))
    images = {"present": present, "absent": absent, "alternate": alternate}
    fields = {}
    for name, image in images.items():
        fields[name] = inspect(image)
        check(f"{name} structure size", fields[name]["size"], SLOT_SIZE)
        check(f"{name} directory size", fields[name]["directory_size"], SLOT_SIZE)
        check(f"{name} read-only slot", fields[name]["read_only"], True)
        print(f"{name}: {describe(fields[name])}")

    check("present slot", fields["present"]["slot_value"] != 0, True)
    check("present relocation", fields["present"]["relocated"], True)
    check("absent slot", fields["absent"]["slot_value"], 0)
    check("absent relocation", fields["absent"]["relocated"], False)
    check("alternate slot", fields["alternate"]["slot_value"] != 0, True)
    check("alternate relocation", fields["alternate"]["relocated"], True)

    for name in ("present", "absent", "alternate"):
        native = native_code(images[name])
        mapped = mapped_code(mapper_path, images[name], io)
        check(f"{name} native", native, EXPECTED[name])
        check(f"{name} mapped", mapped, EXPECTED[name])
        print(f"{name}: native and mapped {native:#x}; callback count/argument/return matched")


if __name__ == "__main__":
    main(*sys.argv[1:])
