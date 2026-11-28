#!/usr/bin/env python3
"""Compare a GDK export inventory with the Xbox2Windows shim .def files.

The inventory comes from tools/gdk_exports.py. An export name is not a
signature and not a test: this matrix prioritizes work, it does not claim full
coverage.

Commands:
  --exports <json> --repo <Xbox2Windows> [--json <out>] [--markdown <out>]
  --exports <json> --def xmem=src/xmem_shim.def ... [--json <out>]
"""

import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_DEFS = {
    "xmem": "src/xmem_shim.def",
    "d3d12_xs": "src/d3d12_xs_shim.def",
    "acphal": "src/startup_stubs/acphal.def",
    "xapu": "src/startup_stubs/xapu.def",
    "chad": "src/startup_stubs/chad.def",
    "xg_xs": "src/startup_stubs/xg_xs.def",
    "xcrdapi": "src/startup_stubs/xcrdapi.def",
    "gameinput": "src/startup_stubs/gameinput.def",
}
DATA_PREFIXES = ("IID_", "WKPDID_")


def is_data(name):
    return name.startswith(DATA_PREFIXES)


def read_def(path):
    names = set()
    ordinals = {}
    data = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith(";") or text.upper().startswith("LIBRARY"):
            continue
        if text.upper() == "EXPORTS":
            continue
        fields = text.split("=", 1)[0].split()
        if not fields:
            continue
        name = fields[0]
        ordinal = None
        match = re.fullmatch(r"(.+)@(\d+)", name)
        if match:
            name, ordinal_text = match.groups()
            ordinal = int(ordinal_text)
        else:
            for token in text.split():
                match = re.fullmatch(r"@(\d+)", token)
                if match:
                    ordinal = int(match.group(1))
                    break
        if not name or name == "PRIVATE":
            continue
        if "DATA" in text.split():
            data.add(name)
        names.add(name)
        if ordinal is not None:
            ordinals[name] = ordinal
    return {"names": names, "ordinals": ordinals, "data": data}


def normalize_module(name):
    name = Path(name).stem.lower()
    for suffix in (".lib", ".dll"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_exports(document):
    if isinstance(document, list):
        modules = document
    else:
        modules = document.get("modules") or [document]
    result = {}
    for module in modules:
        stem = normalize_module(module.get("module") or Path(module["file"]).stem)
        entry = result.get(stem)
        if entry is None:
            entry = {"file": module["file"], "exports": [], "errors": []}
            result[stem] = entry
        if module.get("error"):
            entry["errors"].append(module["error"])
        entry["exports"].extend(module.get("exports", []))
    return result


def find_module(modules, name):
    return modules.get(normalize_module(name))


def rows_for(args):
    document = json.loads(Path(args.exports).read_text(encoding="utf-8"))
    modules = load_exports(document)
    definitions = {}
    if args.repo:
        for module, relative in DEFAULT_DEFS.items():
            path = Path(args.repo) / relative
            if path.exists():
                definitions[module] = path
    for item in args.defs:
        module, separator, path = item.partition("=")
        if not separator:
            raise SystemExit("--def expects module=path: " + item)
        definitions[module] = Path(path)
    rows = []
    for module, def_path in sorted(definitions.items()):
        shim = read_def(def_path)
        gdk = find_module(modules, module)
        if gdk is None:
            rows.append(
                {
                    "module": module,
                    "gdk_lib": None,
                    "def": str(def_path),
                    "gdk_exports": 0,
                    "gdk_functions": 0,
                    "gdk_data": 0,
                    "shim_exports": len(shim["names"]),
                    "shim_functions": len(shim["names"]),
                    "shim_data": len(shim["data"]),
                    "covered_functions": 0,
                    "covered_data": 0,
                    "missing_functions": [],
                    "missing_data": [],
                    "extra": [],
                    "ordinals": {},
                    "def_ordinals": {
                        name: shim["ordinals"][name] for name in sorted(shim["ordinals"])
                    },
                    "unmeasured": "no inventory entry",
                }
            )
            continue
        gdk_names = {item["name"] for item in gdk["exports"] if item.get("name")}
        gdk_functions = {name for name in gdk_names if not is_data(name)}
        gdk_data = {name for name in gdk_names if is_data(name)}
        shim_functions = {name for name in shim["names"] if name not in shim["data"] and not is_data(name)}
        shim_data = {name for name in shim["data"] if is_data(name)}
        missing_functions = sorted(gdk_functions - shim_functions)
        missing_data = sorted(gdk_data - shim_data)
        ordinals = {
            item["name"]: item["ordinal"]
            for item in gdk["exports"]
            if item.get("name") and item.get("ordinal") is not None
        }
        row = {
            "module": module,
            "gdk_lib": gdk["file"],
            "def": str(def_path),
            "gdk_exports": len(gdk_names),
            "gdk_functions": len(gdk_functions),
            "gdk_data": len(gdk_data),
            "shim_exports": len(shim["names"]),
            "shim_functions": len(shim_functions),
            "shim_data": len(shim_data),
            "covered_functions": len(gdk_functions & shim_functions),
            "covered_data": len(gdk_data & shim_data),
            "missing_functions": missing_functions,
            "missing_data": missing_data,
            "extra": sorted(shim["names"] - gdk_names),
            "ordinals": {name: ordinals[name] for name in sorted(ordinals)},
            "def_ordinals": {
                name: shim["ordinals"][name] for name in sorted(shim["ordinals"])
            },
        }
        if gdk["errors"]:
            row["unmeasured"] = "; ".join(gdk["errors"])
        rows.append(row)
    return rows


def render_markdown(rows):
    lines = [
        "# GDK export coverage of the Xbox2Windows shims",
        "",
        "| Module | GDK functions | Shim functions | Covered | Missing | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        unmeasured = row.get("unmeasured")
        data = dict(row)
        data.update(
            missing="?" if unmeasured else len(row["missing_functions"]),
            gdk_functions="?" if unmeasured else row["gdk_functions"],
            covered_functions="?" if unmeasured else row["covered_functions"],
            status=unmeasured or "ok",
        )
        lines.append(
            "| {module} | {gdk_functions} | {shim_functions} | {covered_functions} | {missing} | {status} |".format(**data)
        )
    lines += [
        "",
        "Names only: an export is not a signature and not a test. Data exports",
        "(IID_ and WKPDID_) are counted separately, and x2w_* locals are listed",
        "as extras.",
        "",
    ]
    for row in rows:
        lines.append("## " + row["module"])
        lines.append("")
        if row.get("unmeasured"):
            lines.append("Unmeasured: " + row["unmeasured"])
            lines.append("")
            continue
        lines.append(
            "Missing functions: "
            + (", ".join(row["missing_functions"]) if row["missing_functions"] else "none")
        )
        if row["missing_data"]:
            lines.append("")
            lines.append("Missing data: " + ", ".join(row["missing_data"]))
        if row["extra"]:
            lines.append("")
            lines.append("Local extras: " + ", ".join(row["extra"]))
        if row["ordinals"]:
            lines.append("")
            lines.append("GDK ordinals: " + ", ".join(
                f"{name}@{ordinal}" for name, ordinal in row["ordinals"].items()
            ))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True)
    parser.add_argument("--repo")
    parser.add_argument("--def", dest="defs", action="append", default=[])
    parser.add_argument("--json")
    parser.add_argument("--markdown")
    parser.add_argument("--fail-on-missing", action="store_true")
    args = parser.parse_args(argv)
    rows = rows_for(args)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(rows)
    if args.markdown:
        Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    missing = sum(len(row["missing_functions"]) for row in rows)
    unmeasured = sum(1 for row in rows if row.get("unmeasured"))
    return 1 if args.fail_on_missing and (missing or unmeasured) else 0


if __name__ == "__main__":
    sys.exit(main())
