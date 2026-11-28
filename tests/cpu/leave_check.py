"""Differential checks and rejection controls for LEAVE on a proven RBP anchor in the jump-table translator."""

import collections
import ctypes
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import translate
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

# The export list of cpu_translation(cpu_leave ...) in CMakeLists.txt.
ENTRIES = ("leave_frame", "leave_lea", "leave_join")


def reference(name, x, y):
    """RAX, the three trace words and the path taken by name(x, y, trace); an untouched word keeps the fill."""
    trace = [MASK] * 3
    if name == "leave_frame":
        rax = (x + y) & MASK
        trace[0] = rax
        trace[1] = (MASK & ~0xff) | int(x < y)  # SETB after LEAVE reads the flags of CMP RCX, RDX
        path = "CF set" if x < y else "CF clear"
    elif name == "leave_lea":
        rax = (x - (x ^ y)) & MASK
        trace[0] = rax
        path = "anchored by LEA"
    else:
        rax = (x + y) & MASK if y & 1 else x ^ y
        trace[0] = rax
        path = "odd path" if y & 1 else "even path"
    return rax, (GUARDS[0], *trace, GUARDS[1]), path


def check(probe_path, original_path, lifted_path, lifted_source):
    probe = register_probe(probe_path)
    source, _ = translate(original_path, list(ENTRIES))
    if Path(lifted_source).read_text(encoding="utf-8") != source:
        raise AssertionError(f"{lifted_source} is not the translation of {','.join(ENTRIES)}")
    print(f"the built translation is the translation of {','.join(ENTRIES)}")
    for name in ENTRIES:
        body = source.split(f"lifted_{name}(")[1].split('\nextern "C"')[0]
        saves = re.findall(r"store<std::uint64_t>\(reinterpret_cast<word>\(frame \+ (\d+)\), rbp\);", body)
        loads = re.findall(r"rbp = static_cast<std::uint64_t>\(load<std::uint64_t>\("
                           r"reinterpret_cast<word>\(frame \+ (\d+)\)\)\);", body)
        if len(saves) != 1 or loads != saves:
            raise AssertionError(f"{name}: PUSH RBP stores to {saves}, LEAVE reloads RBP from {loads}")
        print(f"{name}: LEAVE reloads RBP from frame + {loads[0]}, the slot PUSH RBP wrote")
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    inputs = input_pairs()
    total = 0
    for name in ENTRIES:
        paths = collections.Counter()
        for x, y in inputs:
            rax, trace, path = reference(name, x, y)
            functions = (("native", getattr(native, name)), ("translated", getattr(lifted, "lifted_" + name)))
            for label, function in functions:
                got = observe(probe, function, x, y)
                if got != (rax, trace, SENTINELS):
                    raise AssertionError(f"{label} {name}({x:#x}, {y:#x}) = {got}, expected {rax:#x} {trace}")
            paths[path] += 1
            total += 1
        print(f"{name}: {len(inputs)} input pairs match native and translated; paths: "
              + ", ".join(f"{path} {count}" for path, count in sorted(paths.items())))
    print(f"{total} calls per side match the reference on return, guarded trace, RBX and RSI")


def find(pe, name, text, occurrence=0):
    """The occurrence-th instruction of the export whose text is text."""
    (begin, end), = function_ranges(pe, export_rva(pe, name))
    found = [ins for ins in Decoder(64, pe.get_data(begin, end - begin), ip=begin) if str(ins) == text]
    if len(found) <= occurrence:
        raise AssertionError(f"{name}: no instruction {text!r} number {occurrence}")
    return found[occurrence]


# (export, [(instruction text, occurrence, replacement bytes)], expected diagnostic). Each replacement has the
# length of the instruction it replaces.
PATCHES = [
    # Without MOV RBP, RSP no anchor is proven when LEAVE runs.
    ("leave_frame", [("mov rbp,rsp", 0, "0f 1f 00")], "LEAVE without a proven frame anchor"),
]
# Exports that only exist to be rejected.
EXPORTS = [("leave_split", "frame pointer differs at join"),
           ("leave_unsaved", "undefined stack byte at entry RSP-0x8"),
           ("leave_overwritten", "nonvolatile register rbp is not restored"),
           ("leave_released", "undefined stack byte at entry RSP-0x10"),
           ("leave_lowers", "LEAVE with the frame anchor below RSP"),
           ("leave_above", "LEAVE would raise RSP above entry RSP"),
           ("mov_rsp_frame", "dynamic frame"),
           ("enter_frame", "dynamic frame"),
           ("leave_word", "unsupported LEAVE operand size")]


def check_rejections(original_path):
    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        controls = [(name, [(find(pe, name, text, occurrence), patch) for text, occurrence, patch in edits], reason)
                    for name, edits, reason in PATCHES]
        offset = pe.get_offset_from_rva
        with TemporaryDirectory() as folder:
            path = Path(folder) / "control.dll"
            for name, edits, reason in controls:
                damaged, changes = bytearray(data), []
                for ins, patch in edits:
                    patch = bytes.fromhex(patch)
                    replacement = list(Decoder(64, patch, ip=ins.ip))
                    if len(patch) != ins.len or sum(i.len for i in replacement) != ins.len:
                        raise AssertionError(f"{name}: {ins} and {replacement} differ in length")
                    damaged[offset(ins.ip):offset(ins.ip) + ins.len] = patch
                    changes.append(f"{ins} -> {'; '.join(map(str, replacement))}")
                path.write_bytes(damaged)
                run_control(name, ", ".join(changes), path, reason)
    for name, reason in EXPORTS:
        run_control(name, "source export", original_path, reason)
    print(f"{len(controls) + len(EXPORTS)} rejection controls passed")


def run_control(name, change, image, reason):
    try:
        translate(image, [name])
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(f"{name} ({change}): expected {reason!r}, got {exc}") from exc
        print(f"rejected {name} ({change}): {exc}")
    else:
        raise AssertionError(f"accepted control {name} ({change})")


if __name__ == "__main__":
    check(*sys.argv[1:])
    check_rejections(sys.argv[2])
