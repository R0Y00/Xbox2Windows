#!/usr/bin/env python3
"""Collect GDK export names and ordinals from dumpbin text or a PE DLL.

The GDK libraries stay outside the repository. This tool writes the inventory
JSON that tools/gdk_shim_coverage.py compares with the shim .def files.

Commands:
  parse-dumpbin <dumpbin.txt>... --json <out> [--kind lib|dll]
  scan-pe <dll>... --json <out>

parse-dumpbin accepts a file or a directory of text captured from:
    dumpbin /nologo /exports <library>
Its sha256 identifies that capture, not the LIB or DLL it was taken from.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


DLL_ROW = re.compile(r"^\s*(\d+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+(\S+)\s*$")
LIB_ROW = re.compile(r"^\s*(\d+)?\s+([A-Za-z_?$@][^\s]*)\s*$")
SKIP = {"Exports", "Summary", "ordinal", "name", "hint", "RVA"}


def sha256(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest().upper()


def parse_dumpbin_report(text):
    exports = []
    started = False
    unparsed = 0
    for line in text.splitlines():
        if re.match(r"^\s*ordinal\s+name\b", line) or re.match(
            r"^\s*ordinal\s+hint\s+RVA\s+name\b", line
        ):
            started = True
            continue
        if not started:
            if line.strip() == "Exports":
                started = True
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Summary") or stripped.startswith("Dump of file"):
            break
        match = DLL_ROW.match(line)
        if match:
            exports.append(
                {
                    "ordinal": int(match.group(1)),
                    "name": match.group(4),
                    "rva": "0x%X" % int(match.group(3), 16),
                }
            )
            continue
        match = LIB_ROW.match(line)
        if match and match.group(2) not in SKIP:
            exports.append(
                {
                    "ordinal": int(match.group(1)) if match.group(1) else None,
                    "name": match.group(2),
                    "rva": None,
                }
            )
            continue
        unparsed += 1
    return exports, unparsed


def parse_dumpbin_text(text):
    return parse_dumpbin_report(text)[0]


def module_name(path, kind):
    name = Path(path).stem
    for suffix in (".lib", ".dll"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def collect_dumpbin(path, kind, sdk="261000"):
    path = Path(path)
    if path.is_dir():
        entries = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".txt"
        )
    else:
        entries = [path]
    modules = []
    for entry in entries:
        text = entry.read_text(encoding="utf-8", errors="replace")
        parsed, unparsed = parse_dumpbin_report(text)
        module = {
            "sdk": sdk,
            "module": module_name(entry, kind),
            "file": str(entry),
            "kind": kind,
            "sha256": sha256(entry),
            "exports": parsed,
        }
        if unparsed:
            module["unparsed"] = unparsed
        if not parsed:
            module["error"] = "no exports parsed"
        elif unparsed:
            module["error"] = f"{unparsed} unparsed export line(s)"
        modules.append(module)
    return modules


def scan_pe(path, sdk="261000"):
    import pefile  # Imported here so parse-dumpbin needs only the standard library.

    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]]
    )
    exports = []
    directory = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    if directory:
        for symbol in directory.symbols:
            exports.append(
                {
                    "ordinal": symbol.ordinal,
                    "name": symbol.name.decode("ascii", "replace") if symbol.name else "",
                    "rva": "0x%X" % symbol.address,
                }
            )
    module = {
        "sdk": sdk,
        "module": module_name(path, "dll"),
        "file": str(path),
        "kind": "dll",
        "sha256": sha256(path),
        "exports": exports,
    }
    if not exports:
        module["error"] = "no exports parsed"
    return module


def write_document(path, modules, sdk="261000"):
    document = {
        "sdk": sdk,
        "count": sum(len(module["exports"]) for module in modules),
        "modules": modules,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"modules": len(modules), "exports": document["count"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    dumpbin = sub.add_parser("parse-dumpbin", help="parse captured dumpbin /exports text")
    dumpbin.add_argument("inputs", nargs="+")
    dumpbin.add_argument("--json", required=True)
    dumpbin.add_argument("--kind", choices=("lib", "dll"), default="lib")
    dumpbin.add_argument("--sdk", default="261000")
    dumpbin.set_defaults(
        func=lambda args: write_document(
            args.json,
            [module for path in args.inputs for module in collect_dumpbin(path, args.kind, args.sdk)],
            args.sdk,
        )
    )

    pe = sub.add_parser("scan-pe", help="read exports from a PE DLL")
    pe.add_argument("dlls", nargs="+")
    pe.add_argument("--json", required=True)
    pe.add_argument("--sdk", default="261000")
    pe.set_defaults(
        func=lambda args: write_document(
            args.json, [scan_pe(path, args.sdk) for path in args.dlls], args.sdk
        )
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
