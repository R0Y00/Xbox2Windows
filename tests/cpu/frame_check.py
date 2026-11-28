"""Differential checks and rejection controls for stack frames in functions that make no call."""

import collections
import ctypes
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import MAX_FRAME, translate
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

# The export list of cpu_translation(cpu_frame ...) in CMakeLists.txt. frame_bound is translated and never called.
ENTRIES = ("push_loop", "spill_widths", "home_save", "home_args", "two_paths", "home_spill", "loop_counter",
           "realloc_slot", "anchored_frame", "frame_bound")
# Bytes of each frame array: from the lowest RSP, rounded down to 8 mod 16, to entry RSP + 40.
SIZES = {"push_loop": 64, "spill_widths": 64, "home_save": 80, "home_args": 64, "two_paths": 64, "home_spill": 48,
         "loop_counter": 64, "realloc_slot": 64, "anchored_frame": 80, "frame_bound": MAX_FRAME}


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def reference(name, x, y):
    """RAX, the three trace words and the path taken by name(x, y, trace); an untouched word keeps the fill."""
    trace = [MASK] * 3
    if name == "push_loop":
        rax = x
        for _ in range(y & 31):
            rax = (rax * 3 + y) & MASK
        trace[0] = rax
        path = "loop runs" if y & 31 else "loop skipped"
    elif name == "spill_widths":
        stack = bytearray(16)  # the bytes from RSP after SUB RSP, 18h, kept as the processor stores them
        stack[0:8] = x.to_bytes(8, "little")
        stack[8:12] = (y & 0xffffffff).to_bytes(4, "little")
        stack[12:14] = (y & 0xffff).to_bytes(2, "little")
        stack[14], stack[15] = x & 0xff, 0x5a
        rax = int.from_bytes(stack[4:8], "little")
        r9 = int.from_bytes(stack[7:9], "little")
        r10 = int.from_bytes(stack[8:16], "little")
        stack[0:8] = ((x + y) & MASK).to_bytes(8, "little")
        stack[8] = int(x < y)
        moved = signed(x, 32) < signed(y, 32)
        if moved:
            rax = int.from_bytes(stack[8:16], "little")
        trace = [r10, r9, int.from_bytes(stack[0:8], "little")]
        path = "CMOVL moves" if moved else "CMOVL keeps"
    elif name == "home_save":
        if y:
            rax = trace[0] = ((x + y) * (x ^ y)) & MASK
        else:
            trace[1], rax = x, x & 0xffffffff
        path = "first RET" if y else "second RET"
    elif name == "home_args":
        largest = y if x < y else x
        trace[0], rax = largest, (largest + x) & MASK
        path = "stores y" if x < y else "stores x"
    elif name == "two_paths":
        if x & 1:
            trace[1] = (x - y) & MASK
        else:
            trace[0] = (x + y) & MASK
        rax = y
        if x >= y:
            rax = trace[2] = x ^ y
        path = ("pushes rdi" if x & 1 else "pushes rsi") + (", early RET" if x < y else ", last RET")
    elif name == "home_spill":
        rax = trace[0] = (x + y) & MASK
        trace[1], path = x, "straight line"
    elif name == "loop_counter":
        count, rax = y & 15, 0
        for _ in range(count):
            rax = (rax * 3 + x) & MASK
        trace[0] = trace[1] = count
        path = "loop runs" if count else "loop skipped"
    elif name == "realloc_slot":
        rax = trace[0] = (x + y) & MASK
        path = "straight line"
    elif name == "anchored_frame":
        rax = trace[0] = (x + (y if y & 1 else 0)) & MASK
        path = "reanchored" if y & 1 else "original anchor"
    return rax, (GUARDS[0], *trace, GUARDS[1]), path


def check_layout(source, functions):
    """Each frame array has the expected size, and RSP never appears in the generated C++."""
    bodies = source.split('extern "C" __declspec(dllexport) word lifted_')[1:]
    for name, function, body in zip(ENTRIES, functions, bodies):
        arrays = re.findall(r"alignas\(16\) std::uint8_t frame\[(\d+)\] = \{\};  // (.*)", body)
        if [int(size) for size, _ in arrays] != [SIZES[name]] or function.return_width != 8:
            raise AssertionError(f"{name}: frame arrays {arrays}, return width {function.return_width}")
        if re.search(r"\brsp\b", body):
            raise AssertionError(f"{name}: RSP appears in the generated C++")
        print(f"control {name}: one frame array of {SIZES[name]} bytes ({arrays[0][1]}); no C++ variable "
              "or expression holds RSP")


def check(probe_path, original_path, lifted_path, lifted_source):
    probe = register_probe(probe_path)
    source, functions = translate(original_path, list(ENTRIES))
    if Path(lifted_source).read_text(encoding="utf-8") != source:
        raise AssertionError(f"{lifted_source} is not the translation of {','.join(ENTRIES)}")
    print(f"the built translation is the translation of {','.join(ENTRIES)}")
    check_layout(source, functions)
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    inputs = input_pairs()
    total = 0
    for name in ENTRIES[:-1]:
        paths = collections.Counter()
        for x, y in inputs:
            rax, trace, path = reference(name, x, y)
            for label, function in (("native", getattr(native, name)), ("translated", getattr(lifted, "lifted_" + name))):
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
    ("push_loop", [("pop rsi", 0, "90")], "stack not balanced at RET"),
    ("two_paths", [("pop rsi", 0, "90")], "stack depth differs at join"),
    # PUSH RBX and NOP in place of the loop's DEC ESI: each iteration would push again.
    ("push_loop", [("dec esi", 0, "53 90")], "stack depth differs at join"),
    ("home_save", [("mov rbx,[rsp+30h]", 0, "0f 1f 44 00 00")], "nonvolatile register rbx is not restored"),
    ("home_save", [("mov rsi,[rsp+38h]", 0, "48 8b 74 24 30")], "hold the saved rbx"),
    ("push_loop", [("pop rsi", 0, "5b")], "hold the saved rsi"),
    ("home_save", [("mov rbx,[rsp+30h]", 0, "48 8b 44 24 30")], "hold the saved rbx"),
    ("home_args", [("mov [rsp],rax", 0, "0f 1f 40 00")], "undefined stack byte at entry RSP-0x18"),
    # Bytes 0 to 3 of the 8 that MOV R10, [RSP+8] reads are then never written; bytes 4 to 7 are.
    ("spill_widths", [("mov [rsp+8],edx", 0, "0f 1f 40 00")], "undefined stack byte at entry RSP-0x10"),
    ("home_args", [("mov [rsp+10h],rdx", 0, "0f 1f 44 00 00")], "undefined stack byte at entry RSP+0x10"),
    # Without its initial store the counter is written only on the loop's back edge.
    ("loop_counter", [("mov dword ptr [rsp],0", 0, "0f 1f 80 00 00 00 00")], "undefined stack byte at entry RSP-0x18"),
    ("home_args", [("mov rcx,[rsp+30h]", 0, "48 8b 4c 24 40")], "unsupported stack argument"),
    ("home_args", [("mov rax,[rsp+20h]", 0, "48 8b 44 24 18")], "return address slot"),
    ("home_args", [("mov [rsp+8],rcx", 0, "48 89 4c 24 00")], "return address slot"),
    ("home_args", [("mov rax,[rsp+28h]", 0, "48 8d 44 24 08")], "frame address escapes"),
    ("home_args", [("mov rax,rcx", 0, "48 8b c4")], "frame address escapes"),
    ("anchored_frame", [("lea rbp,[rsp+20h]", 0, "48 8d 6c 24 18")], "frame pointer differs at join"),
    ("anchored_frame", [("mov rax,[rbp-8]", 0, "48 8d 45 f8")], "unsupported stack operand"),
    ("anchored_frame", [("mov rax,[rbp-8]", 0, "48 8b 45 f0")], "undefined stack byte"),
    ("anchored_frame", [("pop rbp", 0, "5b")], "hold the saved rbp"),
    ("home_args", [("sub rsp,18h", 0, "48 2b e1 90")], "dynamic frame"),
    ("home_args", [("sub rsp,18h", 0, "48 83 ec 14"), ("add rsp,18h", 0, "48 83 c4 14")],
     "unsupported stack adjustment"),
    ("home_args", [("sub rsp,18h", 0, "48 83 c4 18")], "unsupported stack adjustment"),
    # Without the CMP, JAE would read the flags of SUB RSP, 18h.
    ("home_args", [("cmp [rsp+20h],rax", 0, "0f 1f 44 00 00")], "flags of a stack adjustment"),
    ("spill_widths", [("mov eax,[rsp+4]", 0, "8b 44 24 fc")], "stack access below RSP"),
    ("frame_bound", [("sub rsp,0FD8h", 0, "48 81 ec e8 0f 00 00"), ("add rsp,0FD8h", 0, "48 81 c4 e8 0f 00 00")],
     f"exceeds the {MAX_FRAME}-byte bound"),
    # Without the store into the second allocation, ADD RAX, [RSP] would read what the first one left there.
    ("realloc_slot", [("mov [rsp],rdx", 0, "0f 1f 40 00")], "undefined stack byte at entry RSP-0x18"),
    ("push_loop", [("mov rbx,rcx", 0, "48 03 d9")], "nonvolatile register rbx may hold the caller's value"),
    ("push_loop", [("push rbx", 0, "50")], "unsupported push or pop"),
    ("push_loop", [("and esi,1Fh", 0, "6a 05 90")], "unsupported push or pop"),
]
# Exports that only exist to be rejected. guest_spill is refused at its store: a register byte that may hold a
# guest address is never defined, and a store reads its source register.
EXPORTS = [("stale_save", "undefined stack byte at entry RSP-0x8"),
           ("stale_pop", "undefined stack byte at entry RSP-0x10"),
           ("guest_spill", "undefined register rax"),
           ("split_restore", "hold the saved rbx"),
           ("partial_push", "undefined register rbx"),
           ("reuse_restored", "nonvolatile register rbx may hold the caller's value")]


def check_rejections(original_path):
    data = Path(original_path).read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        controls = [(name, [(find(pe, name, text, occurrence), patch) for text, occurrence, patch in edits], reason)
                    for name, edits, reason in PATCHES]
        offset = pe.get_offset_from_rva
        source, _ = translate(original_path, ["frame_bound"])
        if f"std::uint8_t frame[{MAX_FRAME}]" not in source:
            raise AssertionError("frame_bound does not translate with a frame array at the bound")
        print(f"accepted frame_bound: its frame array is {MAX_FRAME} bytes, the bound")
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
