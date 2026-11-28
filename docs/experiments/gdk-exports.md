# GDK exports and shim coverage

tools/gdk_exports.py reads captured dumpbin /exports text or a PE DLL and writes
an inventory with module, file, SHA-256, export name, ordinal and RVA.
tools/gdk_shim_coverage.py compares that inventory with the Xbox2Windows .def
files and writes a JSON matrix plus a Markdown table.

The SHA-256 identifies the captured .txt when parse-dumpbin is used, and the
DLL when scan-pe is used. A module with no inventory entry, a capture with zero
exports, or a partially parsed capture is reported as unmeasured;
--fail-on-missing fails on those rows as well as on missing names.

The GDK libraries and the full inventory stay outside the repository. The
inventory used for the 261000 edition has 89 modules and 8,035 export names.
When a module has both a .lib and a .dll, the coverage tool merges their exports
and deduplicates the names, so d3d12_xs reports the union of the two files.

    python -B tests/gdk_exports_check.py
    python -B tools/gdk_exports.py parse-dumpbin D:\gdk-oracle\dumpbin\xg_xs.txt --json out\xg_xs.json --kind dll
    python -B tools/gdk_shim_coverage.py --exports D:\gdk-oracle\knowledge\exports-all.json --repo . --json out\coverage.json --markdown out\coverage.md

Measured on 28 November 2026 with the 261000 LIB/DLL set:

| Module | GDK functions | Shim functions | Covered | Missing | Status |
|---|---:|---:|---:|---:|---|
| xmem | 36 | 27 | 13 | 23 | ok |
| d3d12_xs | 28 | 5 | 4 | 24 | ok |
| acphal | 10 | 10 | 9 | 1 | ok |
| xapu | 6 | 6 | 5 | 1 | ok |
| chad | 7 | 5 | 5 | 2 | ok |
| xg_xs | 27 | 2 | 2 | 25 | ok |
| xcrdapi | 1 | 3 | 1 | 0 | ok |
| gameinput | 1 | 1 | 1 | 0 | ok |

`xgameruntime` is intentionally outside `DEFAULT_DEFS`: its `.def` is
fixture-only and its exported thunk names are generated from
`XGAMERUNTIME_THUNK_TABLE`, so a name comparison would report every GDK runtime
export as missing. An inventory entry that is absent or unreadable is reported
as unmeasured rather than covered.

The xg_xs gap remains the largest: layout computation, DCC, tile shapes,
hardware-version setup and descriptor validation are not exported by the shim.
xmem lacks the statistics, tags, Win32 heap and configuration families. d3d12_xs
lacks the desktop D3D12 entry points and several DXGIX helpers; the IID list is
counted as data, not functions.

The counts are names, not signatures and not tests. A name can exist with a
different calling convention, and a covered name can still be untested at
runtime. The ordinals are recorded where dumpbin prints them; a title that
imports by ordinal still needs the matching import library and a runtime check.
Xbox One gen8 libraries have their own names and are outside this table.
