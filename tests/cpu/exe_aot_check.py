"""Convert source-built console EXEs with tools/exe_aot.py and compare each converted EXE with its input.

Usage: exe_aot_check.py CXX CRC_EXE CASTAGNOLI_EXE TRACE_EXE IO_DLL
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, Mnemonic

from exe_entry_check import check, observation
from exe_process import launch

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from cpu_leaf_aot import export_rva  # noqa: E402
from cpu_stdio_exe_aot import translate  # noqa: E402

CASES = (
    ("line", b"123456789\n"),
    ("final line without newline", b"123456789"),
    ("empty line", b"\n"),
    ("EOF before input", b""),
)


def convert(image, out_dir, *options):
    result = subprocess.run([sys.executable, "-B", str(TOOLS / "exe_aot.py"), str(image), str(out_dir), *options],
                            capture_output=True, text=True, timeout=300)
    report = json.loads((Path(out_dir) / "report.json").read_text(encoding="utf-8"))
    return result, report


def check_built(cxx, image, out_dir, expected_line):
    result, report = convert(image, out_dir, "--entry-export", "native_entry", "--cxx", cxx)
    check(f"{image.name} converter exit ({result.stderr.strip()})", result.returncode, 0)
    check(f"{image.name} status", (report["status"], report["profile"]), ("built", "console-stdio"))
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    check(f"{image.name} report SHA-256", report["sha256"], digest)
    # The driver must route to the shared console translator and not alter its output.
    source = Path(report["source"]).read_text(encoding="ascii")
    check(f"{image.name} generated C++", source, translate(image, entry_export="native_entry"))
    converted = Path(report["exe"])
    with pefile.PE(str(converted), fast_load=True) as pe:
        embedded = pe.get_string_at_rva(export_rva(pe, "aot_source_sha256"))
    check(f"{converted.name} embedded input SHA-256", embedded, digest.encode("ascii"))
    with pefile.PE(str(converted), fast_load=True) as pe:
        pe.parse_data_directories(directories=[1])
        dlls = sorted(m.dll.decode("ascii").lower() for m in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
    check(f"{converted.name} does not load the input's companion DLL", "cpu_exe_io.dll" in dlls, False)

    expectations = {"line": (expected_line, 0), "final line without newline": (expected_line, 0),
                    "empty line": (b"00000000\r\n", 0), "EOF before input": (b"", 1)}
    for label, data in CASES:
        native = observation(launch([image], data))
        aot = observation(launch([converted], data))
        check(f"{image.name} {label} native", (native.output, native.exit_code), expectations[label])
        check(f"{image.name} {label} converted vs native", aot, native)
        print(f"{image.name} {label}: stdout {aot.output!r}, exit {aot.exit_code} on native and converted EXEs")
    return converted


def check_refused(image, out_dir, stage, expected, *options):
    result, report = convert(image, out_dir, *options)
    check(f"{image.name} {stage} refusal exit", result.returncode, 1)
    refusal = report.get("refusal", {})
    check(f"{image.name} refusal stage", (report["status"], refusal.get("stage")), ("refused", stage))
    for key, value in expected.items():
        check(f"{image.name} refusal {key}", refusal.get(key), value)
    left = sorted(path.name for path in Path(out_dir).glob(f"{image.stem}_aot.*"))
    check(f"{image.name} outputs after {stage} refusal", left, [])
    print(f"{image.name}: refused at {stage}: {refusal['message']}")
    return refusal


def patch_first(image, target, mnemonic, replacement):
    """Copy image to target with the first matching two-byte instruction in native_entry replaced."""
    data = bytearray(image.read_bytes())
    with pefile.PE(data=bytes(data), fast_load=True) as pe:
        rva = export_rva(pe, "native_entry")
        offset = pe.get_offset_from_rva(rva)
        for ins in Decoder(64, bytes(data[offset:offset + 64]), ip=rva):
            if ins.mnemonic == mnemonic and ins.len == len(replacement):
                at = pe.get_offset_from_rva(ins.ip)
                data[at:at + ins.len] = replacement
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return ins.ip
    raise AssertionError(f"no two-byte {mnemonic} in {image.name} native_entry")


def main(cxx, crc, castagnoli, trace, io_dll):
    crc, castagnoli, trace, io_dll = map(Path, (crc, castagnoli, trace, io_dll))
    with TemporaryDirectory() as folder:
        root = Path(folder)
        converted = check_built(cxx, crc, root / "crc", b"CBF43926\r\n")
        # A different polynomial gives a different line, so the converted output follows the input program.
        check_built(cxx, castagnoli, root / "castagnoli", b"E3069283\r\n")

        # Negative controls. Each must exit 1, name its stage and leave no generated files.
        check_refused(trace, root / "trace", "imports",
                      {"unbound": ["cpu_exe_io.dll!emit_trace", "cpu_exe_io.dll!read_selector",
                                   "cpu_import_provider.dll!mix_args"]},
                      "--entry-export", "native_entry", "--no-build")
        # Without --entry-export the startup wrapper's import is part of the lifted program.
        check_refused(crc, root / "startup", "imports", {"unbound": ["cpu_exe_io.dll!finish_exe"]}, "--no-build")
        check_refused(io_dll, root / "dll", "header", {})
        text = root / "text" / "notes.exe"
        text.parent.mkdir()
        text.write_bytes(b"not a PE image\n")
        check_refused(text, root / "text", "header", {})

        # Reject an input that is also an output, before touching its bytes.
        report_input = root / "collision" / "report.json"
        report_input.parent.mkdir()
        sentinel = b'{"sentinel":"caller input"}\n'
        report_input.write_bytes(sentinel)
        result, _ = convert(report_input, report_input.parent, "--no-build")
        check("report collision exit", result.returncode, 2)
        check("report collision diagnostic", "output would overwrite the input" in result.stderr, True)
        check("report collision preserves input", report_input.read_bytes(), sentinel)

        aliased_input = root / "input.json"
        aliased_input.hardlink_to(report_input)
        result, _ = convert(aliased_input, report_input.parent, "--no-build")
        check("report hard link exit", result.returncode, 2)
        check("report hard link preserves input", aliased_input.read_bytes(), sentinel)

        # An unsupported instruction in the entry body. Converting into the directory that holds the earlier
        # successful conversion must also remove that stale EXE.
        patched = root / "patched" / crc.name
        rva = patch_first(crc, patched, Mnemonic.XOR, b"\x0f\xa2")
        check(f"stale {converted.name} present before refusal", converted.is_file(), True)
        refusal = check_refused(patched, root / "crc", "translate", {"rva": f"{rva:#x}"},
                                "--entry-export", "native_entry")
        check("translate refusal names cpuid", refusal["message"].endswith("cpuid"), True)

        # A missing compiler stops at build after the C++ is written; no EXE is left behind.
        result, report = convert(crc, root / "nocxx", "--entry-export", "native_entry",
                                 "--cxx", str(root / "missing-cl.exe"))
        check("missing compiler exit", (result.returncode, report["refusal"]["stage"]), (1, "build"))
        check("missing compiler outputs", sorted(p.name for p in (root / "nocxx").glob("*_aot.*")),
              [f"{crc.stem}_aot.cpp"])
        print("missing compiler: refused at build, C++ kept, no EXE")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])
