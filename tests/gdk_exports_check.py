#!/usr/bin/env python3
"""Checks for tools/gdk_exports.py and tools/gdk_shim_coverage.py.

Uses synthetic dumpbin text and synthetic .def files. It needs neither Windows
nor the GDK; the PE scanner is a thin wrapper over pefile and is exercised by
the real inventory outside the repository.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gdk_exports as exports  # noqa: E402
import gdk_shim_coverage as coverage  # noqa: E402


DLL_TEXT = """Microsoft (R) COFF/PE Dumper Version 14.44.35207.0

Dump of file xg_xs.dll

File Type: DLL

  Exports

    ordinal hint RVA      name

          1    0 00001000 XGComputeBufferLayout
          2    1 00001200 XGComputeTexture2DLayout
          3    2 00001400 IID_IXGThing
"""

LIB_TEXT = """Dump of file xmem.lib

File Type: LIBRARY

Exports

       ordinal    name

              XMemAlloc
              XMemFree
              ?XMemInternal@@YAXXZ
"""


def check(condition, what):
    if not condition:
        raise AssertionError(what)


def check_parse():
    dll = exports.parse_dumpbin_text(DLL_TEXT)
    check(dll == [
        {"ordinal": 1, "name": "XGComputeBufferLayout", "rva": "0x1000"},
        {"ordinal": 2, "name": "XGComputeTexture2DLayout", "rva": "0x1200"},
        {"ordinal": 3, "name": "IID_IXGThing", "rva": "0x1400"},
    ], f"DLL parse {dll}")
    lib = exports.parse_dumpbin_text(LIB_TEXT)
    check(lib == [
        {"ordinal": None, "name": "XMemAlloc", "rva": None},
        {"ordinal": None, "name": "XMemFree", "rva": None},
        {"ordinal": None, "name": "?XMemInternal@@YAXXZ", "rva": None},
    ], f"LIB parse {lib}")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "xg_xs.txt"
        source.write_text(DLL_TEXT, encoding="utf-8")
        output = directory / "exports.json"
        code = exports.main([
            "parse-dumpbin", str(source), "--json", str(output), "--kind", "dll"
        ])
        check(code == 0, f"parse-dumpbin exit {code}")
        document = json.loads(output.read_text(encoding="utf-8"))
        check(document["count"] == 3, f"count {document['count']}")
        check(document["modules"][0]["module"] == "xg_xs", f"module {document['modules'][0]['module']}")
        check(len(document["modules"][0]["sha256"]) == 64, "sha256")
    print("exports: DLL, LIB, ordinal and CLI parsing passed")


def check_coverage():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        inventory = {
            "sdk": "261000",
            "modules": [
                {
                    "file": "xg_xs.dll",
                    "exports": [
                        {"ordinal": 1, "name": "XGComputeBufferLayout", "rva": "0x1000"},
                        {"ordinal": 2, "name": "XGComputeTexture2DLayout", "rva": "0x1200"},
                        {"ordinal": 3, "name": "IID_IXGThing", "rva": "0x1400"},
                    ],
                },
                {
                    "file": "xmem.lib",
                    "exports": [
                        {"ordinal": None, "name": "XMemAlloc", "rva": None},
                        {"ordinal": None, "name": "XMemFree", "rva": None},
                    ],
                },
            ],
        }
        inventory_path = directory / "exports.json"
        inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        xg_def = directory / "xg_xs.def"
        xg_def.write_text(
            "LIBRARY xg_xs\nEXPORTS\nXGComputeBufferLayout\nIID_IXGThing DATA\nx2w_local\n",
            encoding="utf-8",
        )
        xmem_def = directory / "xmem.def"
        xmem_def.write_text("LIBRARY xmem\nEXPORTS\nXMemAlloc\nXMemFree\n", encoding="utf-8")
        output_json = directory / "coverage.json"
        output_md = directory / "coverage.md"
        code = coverage.main([
            "--exports", str(inventory_path),
            "--def", "xg_xs=" + str(xg_def),
            "--def", "xmem=" + str(xmem_def),
            "--json", str(output_json),
            "--markdown", str(output_md),
        ])
        check(code == 0, f"coverage exit {code}")
        rows = {row["module"]: row for row in json.loads(output_json.read_text(encoding="utf-8"))}
        xg = rows["xg_xs"]
        check(xg["gdk_functions"] == 2, f"xg functions {xg['gdk_functions']}")
        check(xg["shim_functions"] == 2, f"xg shim functions {xg['shim_functions']}")
        check(xg["covered_functions"] == 1, f"xg covered {xg['covered_functions']}")
        check(xg["missing_functions"] == ["XGComputeTexture2DLayout"], f"xg missing {xg['missing_functions']}")
        check(xg["covered_data"] == 1, f"xg data {xg['covered_data']}")
        check(xg["extra"] == ["x2w_local"], f"xg extra {xg['extra']}")
        check(xg["ordinals"] == {
            "IID_IXGThing": 3, "XGComputeBufferLayout": 1, "XGComputeTexture2DLayout": 2
        }, f"xg ordinals {xg['ordinals']}")
        check("| xg_xs | 2 | 2 | 1 | 1 |" in output_md.read_text(encoding="utf-8"), "markdown table")
        check(coverage.main([
            "--exports", str(inventory_path),
            "--def", "xg_xs=" + str(xg_def),
            "--fail-on-missing",
        ]) == 1, "fail-on-missing exit")
    print("coverage: functions, data, extras, ordinals and exit codes passed")


def main():
    check_parse()
    check_coverage()
    print("gdk_exports_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
