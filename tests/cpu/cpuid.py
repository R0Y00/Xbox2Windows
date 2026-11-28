"""Compare native and translated CPUID and XGETBV in the opt-in host-CPU profile.

CPUID is in cpuid.asm and XGETBV in xgetbv.asm, each with its own test. Both report host values, so every
native, translated and harness read runs pinned to one logical processor, and the previous thread affinity is
restored afterwards. XGETBV runs only after the OSXSAVE check, reads XCR0 only, and an invalid index runs in a
child process.
"""
import contextlib
import ctypes
from pathlib import Path
import re
import subprocess
import sys

import pefile
from iced_x86 import Decoder, FlowControl, Mnemonic as M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
MNEMONIC = {"cpuid": M.CPUID, "xgetbv": M.XGETBV}
PRESETS = (0, 1, 1 << 63, 3 << 62)  # add r8, r8: (ZF, CF) = (1, 0), (0, 0), (1, 1), (0, 1)
# Leaf and subleaf pairs, kept when the leaf is at most the host's maximum basic or extended leaf. Leaves 4, 7,
# 0BH and 0DH take a subleaf.
CASES = ((0, 0), (1, 0), (4, 0), (4, 1), (7, 0), (7, 1), (0xB, 0), (0xB, 1), (0xD, 0), (0xD, 1),
         (0x80000000, 0), (0x80000001, 0))
INVALID_INDEX = 0x7FFFFFFF  # a reserved XCR: XGETBV raises #GP


def exports(part):
    return [f"{part}_op", f"{part}_unused"]


CPUID_TEXT = ("__cpuidex(cpu_regs, static_cast<int>(rax), static_cast<int>(rcx)); "
              "rax = static_cast<uint32_t>(cpu_regs[0]); rbx = static_cast<uint32_t>(cpu_regs[1]); "
              "rcx = static_cast<uint32_t>(cpu_regs[2]); rdx = static_cast<uint32_t>(cpu_regs[3]); }")
XGETBV_TEXT = "const volatile uint64_t xcr = _xgetbv(static_cast<uint32_t>(rcx)); rax = static_cast<uint32_t>(xcr); rdx = xcr >> 32; }"
# part: {variant: [(literal text, replacement)]}. Every replacement is anchored on the cpu_regs or xcr statement.
CONTROLS = {
    "cpuid": {
        # RAX keeps the upper half that was set before CPUID.
        "noext": [("rax = static_cast<uint32_t>(cpu_regs[0]);",
                   "rax = (rax & ~0xffffffffull) | static_cast<uint32_t>(cpu_regs[0]);")],
        # EBX and EDX trade places.
        "swap": [("rbx = static_cast<uint32_t>(cpu_regs[1]); rcx = static_cast<uint32_t>(cpu_regs[2]); "
                  "rdx = static_cast<uint32_t>(cpu_regs[3]);",
                  "rbx = static_cast<uint32_t>(cpu_regs[3]); rcx = static_cast<uint32_t>(cpu_regs[2]); "
                  "rdx = static_cast<uint32_t>(cpu_regs[1]);")],
        # The subleaf in ECX is dropped.
        "nosubleaf": [("__cpuidex(cpu_regs, static_cast<int>(rax), static_cast<int>(rcx));",
                       "__cpuidex(cpu_regs, static_cast<int>(rax), 0);")],
        # A fixed result instead of the host's.
        "constant": [("__cpuidex(cpu_regs, static_cast<int>(rax), static_cast<int>(rcx));",
                      "cpu_regs[0] = 1; cpu_regs[1] = 2; cpu_regs[2] = 3; cpu_regs[3] = 4;")],
        # ZF and CF are changed.
        "flags": [("rdx = static_cast<uint32_t>(cpu_regs[3]); }",
                   "rdx = static_cast<uint32_t>(cpu_regs[3]); zero = !zero; carry = !carry; }")],
    },
    "xgetbv": {
        "noext": [("rax = static_cast<uint32_t>(xcr);", "rax = (rax & ~0xffffffffull) | static_cast<uint32_t>(xcr);")],
        "swap": [("rax = static_cast<uint32_t>(xcr); rdx = xcr >> 32;", "rax = xcr >> 32; rdx = static_cast<uint32_t>(xcr);")],
        # Zero is never XCR0: its bit 0 (x87 state) is always set.
        "constant": [("_xgetbv(static_cast<uint32_t>(rcx))", "0x0ull")],
        "flags": [("rdx = xcr >> 32; }", "rdx = xcr >> 32; zero = !zero; carry = !carry; }")],
        # Built only at /O2: without volatile the optimizer removes the read whose result the guest discards.
        "plain": [("const volatile uint64_t xcr", "const uint64_t xcr")],
    },
}


def emit(part, image, output, variant="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in exports(part):
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            found = [i.mnemonic for i in body if i.mnemonic in MNEMONIC.values()]
            if found != [MNEMONIC[part]]:
                raise AssertionError(f"{name} contains {found}, expected one {part.upper()}")
            code = lower(body, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {}, host_cpu=True)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word state) {{\n{code}\n}}\n')
    text = "".join(parts)
    anchor = CPUID_TEXT if part == "cpuid" else XGETBV_TEXT
    if text.count(anchor) != len(exports(part)):
        raise AssertionError(f"expected the {part} statement once per export")
    for literal, replacement in CONTROLS[part].get(variant, []) if variant != "lifted" else []:
        count = text.count(literal)
        if not count:
            raise AssertionError(f"variant {variant} changed nothing for {literal!r}")
        text = text.replace(literal, replacement)
        print(f"variant {variant}: {count} replacements")
    output.write_text(text, encoding="ascii")


def rejected_by_default(part, image):
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in exports(part):
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            try:
                lower(region(pe, *bounds), ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {})
            except ValueError as exc:
                if "unsupported query operation" not in str(exc):
                    raise
            else:
                raise AssertionError(f"{name}: the default profile lowered {part.upper()}")


def load_lib(path, part):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in exports(part):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.POINTER(U64)], U64
    return lib


def load_harness(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.cpuid_raw.argtypes, lib.cpuid_raw.restype = [ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(U64)], None
    lib.has_osxsave.argtypes, lib.has_osxsave.restype = [], ctypes.c_uint32
    lib.xgetbv_raw.argtypes, lib.xgetbv_raw.restype = [ctypes.c_uint32], U64
    return lib


class ProcessorNumber(ctypes.Structure):
    _fields_ = [("group", ctypes.c_uint16), ("number", ctypes.c_uint8), ("reserved", ctypes.c_uint8)]


class GroupAffinity(ctypes.Structure):
    _fields_ = [("mask", ctypes.c_size_t), ("group", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3)]


@contextlib.contextmanager
def pinned():
    """Run the body on the logical processor the thread is on, in its processor group, then restore the previous
    group affinity."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.GetCurrentProcessorNumberEx.argtypes = [ctypes.POINTER(ProcessorNumber)]
    kernel32.SetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GroupAffinity),
                                                ctypes.POINTER(GroupAffinity)]
    thread = kernel32.GetCurrentThread()

    def current():
        number = ProcessorNumber()
        kernel32.GetCurrentProcessorNumberEx(ctypes.byref(number))
        return number.group, number.number

    group, cpu = current()
    previous = GroupAffinity()
    if not kernel32.SetThreadGroupAffinity(thread, ctypes.byref(GroupAffinity(1 << cpu, group)),
                                           ctypes.byref(previous)):
        raise OSError(ctypes.get_last_error(), "SetThreadGroupAffinity")
    try:
        for _ in range(200):
            kernel32.Sleep(1)
            if current() == (group, cpu):
                break
        else:
            raise AssertionError(f"pinned to logical processor {cpu} of group {group}, still running elsewhere")
        yield cpu if group == 0 else f"{group}:{cpu}"
    finally:
        if not kernel32.SetThreadGroupAffinity(thread, ctypes.byref(previous), None):
            raise OSError(ctypes.get_last_error(), "restoring the thread group affinity")


def call(lib, name, a, b, preset=0):
    state = (U64 * 6)(*([0xdeadbeefdeadbeef] * 6))
    result = getattr(lib, name)(a, b, preset, ctypes.cast(state, ctypes.POINTER(U64)))
    return list(state), result


def flags_of(preset):
    total = preset + preset
    return int(total > MASK64) << 1 | int(total & MASK64 == 0)


def raw_cpuid(harness, leaf, subleaf):
    out = (U64 * 4)()
    harness.cpuid_raw(leaf, subleaf, out)
    return list(out)


def cases(harness):
    basic, extended = raw_cpuid(harness, 0, 0)[0], raw_cpuid(harness, 0x80000000, 0)[0]
    return [(leaf, sub) for leaf, sub in CASES if leaf <= (extended if leaf >= 0x80000000 else basic)]


def cpuid_violation(harness, lib, selected):
    for leaf, sub in selected:
        expected = raw_cpuid(harness, leaf, sub)
        for preset in PRESETS:
            state, _ = call(lib, "cpuid_op", leaf, sub, preset)
            if any(v > MASK32 for v in state[:4]):
                return f"CPUID({leaf:#x}, {sub}) left upper bits: {[hex(v) for v in state[:4]]}"
            if state[:4] != expected:
                return (f"CPUID({leaf:#x}, {sub}) gave {[hex(v) for v in state[:4]]}, "
                        f"host {[hex(v) for v in expected]}")
            if state[4] != flags_of(preset):
                return f"CPUID({leaf:#x}, {sub}) changed ZF/CF: preset {preset:#x} gave {state[4]}"
        _, unused = call(lib, "cpuid_unused", leaf, sub)
        if unused != 7:
            return f"cpuid_unused returned {unused:#x}"
    return None


def xgetbv_violation(harness, lib):
    expected = harness.xgetbv_raw(0)
    for preset in PRESETS:
        state, _ = call(lib, "xgetbv_op", 0, 0, preset)
        if state[0] > MASK32 or state[1] > MASK32:
            return f"XGETBV(0) left upper bits: rax {state[0]:#x} rdx {state[1]:#x}"
        if state[1] << 32 | state[0] != expected:
            return f"XGETBV(0) gave {state[1] << 32 | state[0]:#x}, host XCR0 {expected:#x}"
        if state[4] != flags_of(preset):
            return f"XGETBV(0) changed ZF/CF: preset {preset:#x} gave {state[4]}"
    _, unused = call(lib, "xgetbv_unused", 0, 0)
    if unused != 7:
        return f"xgetbv_unused returned {unused:#x}"
    return None


CHILD = """
import ctypes, sys
ctypes.windll.kernel32.SetErrorMode(0x8003)
lib = ctypes.CDLL(sys.argv[1])
fn = lib.xgetbv_op
fn.argtypes, fn.restype = [ctypes.c_uint64] * 3 + [ctypes.POINTER(ctypes.c_uint64)], ctypes.c_uint64
state = (ctypes.c_uint64 * 6)()
print("calling", flush=True)
try:
    fn(int(sys.argv[2]), 0, 0, state)
except OSError as exc:
    print("fault:", exc, flush=True)
else:
    print("returned", flush=True)
"""


def invalid_index(path):
    """The fault a child reports when it calls xgetbv_op with a reserved index. ctypes turns the structured
    exception into OSError, so the child reports it and exits normally; a return is an error."""
    child = subprocess.run([sys.executable, "-I", "-c", CHILD, str(Path(path).resolve()), str(INVALID_INDEX)],
                           capture_output=True, text=True, timeout=60)
    lines = child.stdout.splitlines()
    if child.returncode or child.stderr or len(lines) != 2 or lines[0] != "calling" or not lines[1].startswith("fault: "):
        raise AssertionError(f"{Path(path).name}: child exit {child.returncode} stdout {child.stdout!r} "
                             f"stderr {child.stderr!r}")
    return lines[1][len("fault: "):]


def reads_in(path, part):
    """The CPUID or XGETBV instructions reachable from each export of a DLL, following every branch."""
    with pefile.PE(str(path), fast_load=True) as pe:
        base, image = pe.OPTIONAL_HEADER.ImageBase, pe.get_memory_mapped_image()
        reads = {}
        for name in exports(part):
            pending, seen, found = [base + export_rva(pe, name)], set(), 0
            while pending:
                ip = pending.pop()
                while ip not in seen:
                    seen.add(ip)
                    if not base <= ip < base + len(image):
                        raise AssertionError(f"{Path(path).name} {name}: branch leaves the image at {ip:#x}")
                    instruction = Decoder(64, image[ip - base:ip - base + 15], ip=ip).decode()
                    found += instruction.mnemonic == MNEMONIC[part]
                    flow = instruction.flow_control
                    if flow == FlowControl.RETURN:
                        break
                    if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
                        pending.append(instruction.near_branch_target)
                        if flow == FlowControl.UNCONDITIONAL_BRANCH:
                            break
                    elif flow not in (FlowControl.NEXT, FlowControl.CALL):
                        raise AssertionError(f"{Path(path).name} {name}: cannot follow {instruction} at {ip:#x}")
                    ip = instruction.next_ip
            reads[name] = found
    return reads


def check(part, harness_path, native_path, lifted_path, *extra_paths):
    harness = load_harness(harness_path)
    if part == "xgetbv" and not harness.has_osxsave():
        print("skip: this host does not report OSXSAVE (CPUID.01H:ECX bit 27), so XGETBV is unavailable")
        raise SystemExit(77)
    rejected_by_default(part, native_path)
    print(f"the default profile rejects every {part.upper()}")
    native, lifted = load_lib(native_path, part), load_lib(lifted_path, part)
    skipped = []
    with pinned() as cpu:
        if part == "cpuid":
            selected = cases(harness)
            pairs = {}
            for leaf, sub in selected:
                pairs.setdefault(leaf, {})[sub] = raw_cpuid(harness, leaf, sub)
            sensitive = [leaf for leaf, subs in pairs.items() if len(subs) == 2 and subs[0] != subs[1]]
            violation = lambda lib: cpuid_violation(harness, lib, selected)
            scope = f"{len(selected)} leaf/subleaf pairs, subleaf-dependent leaves {[hex(l) for l in sensitive]}"
        else:
            sensitive = None
            violation = lambda lib: xgetbv_violation(harness, lib)
            scope = f"XCR0 {harness.xgetbv_raw(0):#x}"
        for label, lib in (("native", native), ("lifted", lifted)):
            problem = violation(lib)
            if problem:
                raise AssertionError(f"{label}: {problem}")
        print(f"logical processor {cpu}: native and translated {part.upper()} match the host for {scope}, "
              "zero-extend every result, keep ZF/CF and execute the discarded read")
        for path in extra_paths:
            stem = Path(path).stem
            if stem.endswith("_o2"):
                variant = stem[len(f"cpu_{part}_"):-len("_o2")]
                reads = reads_in(path, part)
                if variant == "plain":
                    if reads[f"{part}_unused"]:
                        raise AssertionError(f"{stem}: the discarded {part.upper()} survived without volatile")
                    print(f"{stem}: without volatile, /O2 drops the discarded read and the walk reports it: {reads}")
                    continue
                if any(count != 1 for count in reads.values()):
                    raise AssertionError(f"{stem}: reads {reads}, expected one per export")
                problem = violation(load_lib(path, part))
                if problem:
                    raise AssertionError(f"{stem}: {problem}")
                print(f"{stem}: every {part.upper()} is still in the /O2 code, including the discarded one, "
                      "and the rules hold")
                continue
            variant = stem.rsplit("_", 1)[-1]
            if variant == "nosubleaf" and not sensitive:
                skipped.append("control nosubleaf is not measurable: no covered leaf differs between subleaf 0 and 1")
                continue
            problem = violation(load_lib(path, part))
            print(f"control {variant}: {problem}")
            if problem is None:
                raise AssertionError(f"control {variant} kept every property")
    if part == "xgetbv":
        faults = {label: invalid_index(path) for label, path in (("native", native_path), ("lifted", lifted_path))}
        if faults["native"] != faults["lifted"]:
            raise AssertionError(f"XGETBV({INVALID_INDEX:#x}) faults differ: {faults}")
        print(f"XGETBV({INVALID_INDEX:#x}) faults in a child process, natively and translated alike: "
              f"{faults['native']}")
    if skipped:
        print("skip: " + "; ".join(skipped))
        raise SystemExit(77)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "emit":
        part, image, output, *variant = args
        emit(part, Path(image), Path(output), *variant)
    elif mode == "check":
        check(*args)
    else:
        raise ValueError(mode)
