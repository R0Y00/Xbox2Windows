"""Differential Windows x64 ABI checks for a caller with two different direct callees."""

import ctypes
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_call_aot import translate
from cpu_leaf_aot import bounded_code, export_rva
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe


def mix_args(a, b, altered=False):
    stored = (a - b + 11) & MASK
    return (stored + int(altered)) & MASK, (a, b, stored)


def mix_other(a, b, altered=False):
    stored = ((a ^ b) + 29) & MASK
    return (stored + int(altered)) & MASK, (a, b, stored)


def callee_reference(callee, x, y):
    returned, trace = callee(x, y)
    return returned, (GUARDS[0], *trace, GUARDS[1]), SENTINELS


def reference(x, y, *, swapped=False, first_altered=False, second_altered=False):
    first_fn, second_fn = (mix_other, mix_args) if swapped else (mix_args, mix_other)
    # ALTER_FIRST and ALTER_SECOND change mix_args and mix_other, whatever their order.
    altered = {mix_args: first_altered, mix_other: second_altered}
    first, _ = first_fn((x + 7) & MASK, (y - 3) & MASK, altered[first_fn])
    # The second call overwrites the trace, so it records the second arguments.
    second, trace = second_fn((first + x) & MASK, x, altered[second_fn])
    return (second + x) & MASK, (GUARDS[0], *trace, GUARDS[1]), SENTINELS


def check(probe_path, original_path, swapped_path, lifted_path, swapped_lifted_path,
          first_path, second_path):
    probe = register_probe(probe_path)
    if observe(probe, ctypes.CDLL(probe_path).clobber_control, 0, 0)[2] != (0, 0):
        raise AssertionError("physical register probe missed the clobber control")
    native, native_swapped, lifted, lifted_swapped, first, second = map(ctypes.CDLL, (
        original_path, swapped_path, lifted_path, swapped_lifted_path, first_path, second_path))
    inputs = input_pairs()
    swap_changes = 0
    for x, y in inputs:
        expected = reference(x, y)
        original = observe(probe, native.call_pair, x, y)
        if original != expected or observe(probe, lifted.lifted, x, y) != original:
            raise AssertionError(f"call_pair({x:#x}, {y:#x}): {original=} {expected=}")
        swapped = observe(probe, native_swapped.call_pair, x, y)
        if (swapped != reference(x, y, swapped=True)
                or observe(probe, lifted_swapped.lifted, x, y) != swapped):
            raise AssertionError(f"swapped call_pair({x:#x}, {y:#x}): {swapped=}")
        swap_changes += swapped[:2] != original[:2]
        # Each altered callee changes the caller's return on every pair.
        for compiled, flags in ((first.lifted, {"first_altered": True}),
                                (second.lifted, {"second_altered": True})):
            changed = observe(probe, compiled, x, y)
            if changed[0] == original[0] or changed != reference(x, y, **flags):
                raise AssertionError(f"altered {flags} call_pair({x:#x}, {y:#x}): {changed=}")
        # Callees are named in call order, so the swapped build lifts mix_other first.
        for source, compiled, callee in (
                (native.mix_args, lifted.lifted_callee, mix_args),
                (native.mix_other, lifted.lifted_callee2, mix_other),
                (native.mix_other, lifted_swapped.lifted_callee, mix_other),
                (native.mix_args, lifted_swapped.lifted_callee2, mix_args)):
            actual = observe(probe, source, x, y)
            if actual != callee_reference(callee, x, y) or observe(probe, compiled, x, y) != actual:
                raise AssertionError(f"{source.__name__}({x:#x}, {y:#x}): {actual=}")
    if not swap_changes:
        raise AssertionError("swapping the two calls changed no result")
    print(f"{len(inputs)} input pairs: caller, swapped caller and all four lifted callees match "
          f"arguments, guarded writes, RBX, RSI and returns; swapping the calls changes "
          f"{swap_changes} results; each altered callee changes every caller return")


def call_sites(image, caller_name, callees):
    # Cross-check each linked rel32 against the independently exported RVAs.
    with pefile.PE(image, fast_load=True) as pe:
        caller = export_rva(pe, caller_name)
        body = bounded_code(pe, caller)
        found = {}
        for callee_name in callees:
            callee = export_rva(pe, callee_name)
            sites = [i for i in range(len(body) - 4) if body[i] == 0xe8
                     and caller + i + 5 + struct.unpack_from("<i", body, i + 1)[0] == callee]
            if len(sites) != 1:
                raise AssertionError(f"{caller_name} to {callee_name}: expected one rel32, found {sites}")
            found[callee_name] = (callee, sites[0], pe.get_offset_from_rva(caller + sites[0] + 1))
    return caller, found


def expect_rejected(path, reason):
    try:
        translate(path, "call_pair")
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(str(exc)) from exc
    else:
        raise AssertionError(f"accepted {path.name}, expected {reason!r}")


def check_translation(image):
    caller, sites = call_sites(image, "call_pair", ("mix_args", "mix_other"))
    first_callee, first_site, first_offset = sites["mix_args"]
    second_callee, second_site, second_offset = sites["mix_other"]
    original = Path(image).read_bytes()
    cases = ((second_site, second_offset, first_callee + 1, "no unique function interval"),
             (first_site, first_offset, second_callee + 1, "no unique function interval"),
             (second_site, second_offset, caller, "overlapping call target"))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "patched.dll"
        for site, offset, target, reason in cases:
            damaged = bytearray(original)
            struct.pack_into("<i", damaged, offset, target - (caller + site + 5))
            path.write_bytes(damaged)
            expect_rejected(path, reason)

        # Keep the second call but replace its R8 reload with mov rsi, rsi.
        with pefile.PE(image, fast_load=True) as pe:
            reload_offset = pe.get_offset_from_rva(caller + second_site - 3)
        if original[reload_offset:reload_offset + 3] not in (b"\x49\x89\xf0", b"\x4c\x8b\xc6"):
            raise AssertionError("expected mov r8, rsi immediately before the second call")
        damaged = bytearray(original)
        damaged[reload_offset:reload_offset + 3] = b"\x48\x89\xf6"
        path.write_bytes(damaged)
        expect_rejected(path, "undefined register")

        # Overlapping callee intervals: the runtime-function table itself is refused.
        with pefile.PE(image, fast_load=True) as pe:
            table = pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
            table_offset = pe.get_offset_from_rva(table.VirtualAddress)
        records = [struct.unpack_from("<III", original, table_offset + 12 * i)
                   for i in range(table.Size // 12)]
        index = [begin for begin, _, _ in records].index(first_callee)
        overlapping = bytearray(original)
        struct.pack_into("<I", overlapping, table_offset + 12 * index + 4, second_callee + 4)
        path.write_bytes(overlapping)
        expect_rejected(path, "invalid runtime-function interval")
    print(f"one rel32 to each callee; {len(cases)} redirected calls and overlapping callee "
          "intervals are rejected")


if __name__ == "__main__":
    probe_path, original_path, swapped_path, lifted_path, swapped_lifted_path, *altered = sys.argv[1:]
    check(probe_path, original_path, swapped_path, lifted_path, swapped_lifted_path, *altered)
    check_translation(original_path)
    for path in (lifted_path, swapped_lifted_path, *altered):
        call_sites(path, "lifted", ("lifted_callee", "lifted_callee2"))
