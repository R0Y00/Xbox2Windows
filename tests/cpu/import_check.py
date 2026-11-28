"""Check named IAT calls in the original and translated PE with isolated Windows loaders."""

import ctypes
import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pefile

from direct_check import input_pairs, observe, reference, register_probe
from cpu_call_aot import resolve_import
from cpu_leaf_aot import bounded_code, export_rva


def import_site(image, entry, provider, ordinal):
    with pefile.PE(str(image), fast_load=True) as pe:
        start = export_rva(pe, entry)
        code = bounded_code(pe, start)
        pe.parse_data_directories(directories=[1])
        slots = []
        for desc in pe.DIRECTORY_ENTRY_IMPORT:
            if desc.dll.decode("ascii").casefold() != provider.name.casefold():
                continue
            for i, imp in enumerate(desc.imports):
                if ordinal:
                    if imp.name is not None or imp.ordinal != ordinal:
                        continue
                    lookup = desc.struct.OriginalFirstThunk or desc.struct.FirstThunk
                    assert pe.get_qword_at_rva(lookup + 8 * i) == (1 << 63) | ordinal
                elif imp.name != b"mix_args":
                    continue
                slots.append(desc.struct.FirstThunk + 8 * i)
        slot, = slots
        sites = [i for i in range(len(code) - 5) if code[i:i + 2] == b"\xff\x15"
                 and start + i + 6 + struct.unpack_from("<i", code, i + 2)[0] == slot]
        site, = sites
        return pe.get_offset_from_rva(start + site + 2), slot


def load_case(case, native, lifted, probe_path):
    # A fresh process per provider configuration avoids Windows' loaded-module cache.
    expected_error = {"missing": 126, "wrong-export": 127, "wrong-ordinal": 182}.get(case)
    if expected_error:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        # This disposable child must return loader errors even on interactive runners.
        kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001)  # SEM_FAILCRITICALERRORS
        load = kernel.LoadLibraryExW
        load.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
        load.restype = ctypes.c_void_p
        for path in (native, lifted):
            handle = load(str(path), None, 0x1100)  # DLL_LOAD_DIR | DEFAULT_DIRS
            error = ctypes.get_last_error()
            if handle or error != expected_error:
                raise AssertionError(f"{path.name}: expected loader error {expected_error}, got {handle=}, {error=}")
        print(f"{case}: both callers rejected with Windows error {expected_error}")
        return
    probe = register_probe(str(probe_path))
    original, compiled = (ctypes.CDLL(str(path)) for path in (native, lifted))
    inputs = input_pairs()
    for x, y in inputs:
        actual = observe(probe, compiled.lifted, x, y)
        source = observe(probe, original.call_mix, x, y)
        baseline = reference(x, y, caller=True)
        if actual != source or source != reference(x, y, caller=True, altered=case == "altered"):
            raise AssertionError(f"{case} provider({x:#x}, {y:#x}): {source=} {actual=}")
        if case == "altered" and (actual[0] == baseline[0] or actual[1:] != baseline[1:]):
            raise AssertionError("altered provider must change only the return")
    print(f"{case}: {len(inputs)} argument pairs, guarded trace, register and return checks passed")


def reject(image, provider, reason, binding):
    tool = Path(__file__).resolve().parents[2] / "tools/cpu_call_aot.py"
    output = image.with_suffix(".cpp")
    run = subprocess.run([sys.executable, "-B", str(tool), str(image), "call_mix", str(output),
                          "--bind", str(provider), *binding], capture_output=True, text=True)
    if run.returncode != 1 or reason not in run.stderr or output.exists():
        raise AssertionError(f"expected {reason!r}, got {run.returncode}: {run.stderr}")


def check_named_binding_rejections(image, provider, slot):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[1])
        base = pe.OPTIONAL_HEADER.ImageBase
        (module, imp), = [(desc.dll, imp) for desc in pe.DIRECTORY_ENTRY_IMPORT
                          for imp in desc.imports if imp.address == base + slot]
        assert not imp.import_by_ordinal and imp.name == b"mix_args"

    # Supply the already parsed import entries directly to the resolver.
    desc = SimpleNamespace(dll=module, imports=[imp])
    caller = SimpleNamespace(
        OPTIONAL_HEADER=SimpleNamespace(ImageBase=base),
        DIRECTORY_ENTRY_IMPORT=[desc],
        parse_data_directories=lambda *, directories: None,
    )
    assert resolve_import(caller, slot, provider, "mix_args") == "mix_args"
    # Model a PE import descriptor spelling the same DLL with different case.
    desc.dll = module.upper() if provider.name == provider.name.lower() else module.lower()
    assert desc.dll.decode("ascii") != provider.name
    assert resolve_import(caller, slot, provider, "mix_args") == "mix_args"
    desc.dll = module

    def expect_rejection(bound, symbol, reason, *, link_name=None):
        try:
            resolve_import(caller, slot, bound, symbol, link_name=link_name)
        except ValueError as exc:
            if str(exc) != reason:
                raise AssertionError(f"expected {reason!r}, got {str(exc)!r}") from exc
        else:
            raise AssertionError(f"expected {reason!r}, but the binding succeeded")

    dll = module.decode("ascii")
    identity = f"{dll}!mix_args"
    expect_rejection(provider, "mix_args", f"unexpected link name for named import {identity}",
                     link_name="mix_args")
    expect_rejection(provider, "MIX_ARGS", f"unbound import {identity} at IAT slot RVA {slot:#x}")
    desc.dll = module + b"."
    expect_rejection(provider, "mix_args", f"unbound import {dll}.!mix_args at IAT slot RVA {slot:#x}")
    desc.dll = module

    caller.DIRECTORY_ENTRY_IMPORT.append(SimpleNamespace(dll=module, imports=[imp]))
    expect_rejection(provider, "mix_args", f"no unique ordinary import at IAT slot RVA {slot:#x}")
    caller.DIRECTORY_ENTRY_IMPORT.pop()

    folder = image.parent / "non-executable-provider"
    folder.mkdir()
    bound = folder / provider.name
    with pefile.PE(str(provider), fast_load=True) as library:
        rva = export_rva(library, "mix_args")
        section = library.get_section_by_rva(rva)
        assert section is not None and section.Characteristics & 0x20000000
        section.Characteristics &= ~0x20000000
        library.write(str(bound))
    expect_rejection(bound, "mix_args", f"non-executable export for {identity}")


def system_ordinal(image, provider):
    # pefile names ordinal imports from ws2_32.dll with its built-in table (#77 becomes
    # WSAInstallServiceClassA). The binding must still use the ordinal the image stores.
    folder = image.parent / "system-ordinal"
    folder.mkdir()
    renamed, bound, output = folder / image.name, folder / "ws2_32.dll", folder / "lifted.cpp"
    shutil.copy2(provider, bound)
    with pefile.PE(str(image)) as pe:
        desc, = pe.DIRECTORY_ENTRY_IMPORT
        assert [(i.name, i.ordinal) for i in desc.imports] == [(None, 77)]
        pe.set_bytes_at_rva(desc.struct.Name, b"ws2_32.dll\0")
        pe.write(str(renamed))
    tool = Path(__file__).resolve().parents[2] / "tools/cpu_call_aot.py"
    run = subprocess.run([sys.executable, "-B", str(tool), str(renamed), "call_mix", str(output),
                          "--bind", str(bound), "#77", "--link-name", "mix_args"], capture_output=True, text=True)
    if run.returncode or "dllimport) word mix_args(" not in output.read_text(encoding="utf-8"):
        raise AssertionError(f"system ordinal binding failed with {run.returncode}: {run.stderr}")
    reject(renamed, bound, "unbound import ws2_32.dll!#77", ["WSAInstallServiceClassA"])


def check(native, lifted, provider, altered, probe, ordinal_control=None):
    ordinal = 77 if ordinal_control else None
    binding = ["#77", "--link-name", "mix_args"] if ordinal else ["mix_args"]
    if ordinal:
        for image in (provider, altered):
            with pefile.PE(str(image)) as pe:
                assert [(e.ordinal, e.name) for e in pe.DIRECTORY_ENTRY_EXPORT.symbols] == [(ordinal, None)]
    disp, slot = import_site(native, "call_mix", provider, ordinal)
    import_site(lifted, "lifted", provider, ordinal)
    wrong_export = ("wrong-ordinal", ordinal_control) if ordinal else ("wrong-export", probe)
    with TemporaryDirectory() as temp:
        for case, implementation in (("normal", provider), ("altered", altered), ("missing", None), wrong_export):
            folder = Path(temp) / case
            folder.mkdir()
            source = Path(shutil.copy2(native, folder))
            compiled = Path(shutil.copy2(lifted, folder))
            bound = folder / provider.name
            if implementation:
                shutil.copy2(implementation, bound)
            subprocess.run([sys.executable, "-B", __file__, "--load", case, str(source), str(compiled),
                            str(probe)], cwd=folder, check=True, timeout=45)
            identity = f"{provider.name}!{binding[0]}"
            if case == "missing":
                reject(source, bound, f"missing provider for {identity}", binding)
            elif case == wrong_export[0]:
                reject(source, bound, f"unresolved import {identity}", binding)
            elif case == "normal":
                if ordinal:
                    reject(source, bound, f"unbound import {identity}", ["mix_args"])
                    reject(source, bound, f"import-library link name required for {identity}", ["#77"])
                    reject(source, bound, f"unsupported C import name {identity}",
                           ["#77", "--link-name", "m\u00e9"])
                    system_ordinal(source, bound)
                else:
                    check_named_binding_rejections(source, bound, slot)
                # Move the RIP-relative call one byte into the same IAT slot.
                data = bytearray(source.read_bytes())
                delta, = struct.unpack_from("<i", data, disp)
                struct.pack_into("<i", data, disp, delta + 1)
                source.write_bytes(data)
                reject(source, bound, f"no unique ordinary import at IAT slot RVA {slot + 1:#x}", binding)


if __name__ == "__main__":
    if sys.argv[1] == "--load":
        load_case(sys.argv[2], *map(Path, sys.argv[3:]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("native", "lifted", "provider", "altered", "probe"):
            parser.add_argument(arg, type=Path)
        parser.add_argument("--ordinal-control", type=Path)
        check(**vars(parser.parse_args()))
