"""Trace package import paths from a chosen PE image, including verified conditional targets."""

import argparse
from collections import defaultdict
import hashlib
import heapq
import json
from pathlib import Path

FFX_LOADER = "431bdd2e26c0c8214fd7601140eb9fe6d8a249f5000539453a9a9b6a0b7c0ee8"
FFX_LOADER_NAME = "amd_fidelityfx_loader_xbox.dll"
FFX_EXPORTS = ("ffxConfigure", "ffxCreateContext", "ffxDestroyContext", "ffxDispatch", "ffxQuery")


def trace(report, entry):
    images = report["images"]
    by_name = defaultdict(list)
    for path in sorted(images):
        by_name[Path(path).name.casefold()].append(path)
    imports = defaultdict(list)
    for row in report["imports"]:
        imports[row["image"]].append(row)

    # Prefer an all-normal-import path, then the shortest explanation within that class.
    queue = [(False, 0, (entry,), ())]
    reached, findings, lookups = {}, [], []
    while queue:
        conditional, depth, path, reasons = heapq.heappop(queue)
        image = path[-1]
        if image in reached:
            continue
        reached[image] = dict(conditional=conditional, path=path, reasons=reasons)
        targets = set()
        forwarded = defaultdict(set)
        known_hops = set()
        for row in imports[image]:
            dll, symbol, kind = row["dll"], row["symbol"], row["kind"]
            providers = by_name[dll.casefold()]
            if symbol in {"GetProcAddress", "LoadPackagedLibrary", "LoadLibraryA", "LoadLibraryW",
                          "LoadLibraryExA", "LoadLibraryExW"}:
                lookups.append(dict(image=image, symbol=symbol, kind=kind))
            package = row.get("package_resolution")
            if package and package["chain"][0].get("known_dll"):
                # The loader maps a known DLL from System32, so a package copy with that name is not loaded.
                providers = []
                # A forwarder in the System32 module can still name packaged modules; those are loaded,
                # and the package chain's terminal status decides the edge.
                hops = tuple(dict.fromkeys(s["provider"] for s in package["chain"][1:]
                                           if s["package"] and s["provider"]))
                if hops:
                    known_hops.add((kind, hops))
                if any(s["package"] for s in package["chain"][1:]):
                    if package["status"] != "export_present":
                        findings.append(dict(path=path, reasons=reasons + (kind,),
                                             conditional=conditional or kind == "delay", dll=dll, symbol=symbol,
                                             problem=package["status"], candidates=[],
                                             package_chain=package["chain"]))
                    continue
            if len(providers) == 1:
                targets.add((kind, providers[0]))
                if providers[0] in row["bundle_exports_in"]:
                    if not package:
                        continue
                    # Snapping a forwarded export also loads each packaged module named in its chain.
                    hops = tuple(dict.fromkeys(s["provider"] for s in package["chain"][1:]
                                               if s["package"] and s["provider"] and s["provider"] != providers[0]))
                    if hops:
                        forwarded[(kind, providers[0])].add(hops)
                    if package["status"] == "export_present":
                        continue
                    problem = package["status"]
                else:
                    problem = "bundled_export_missing"
            elif len(providers) > 1:
                problem = "ambiguous_package_basename"
            else:
                problem = row.get("desktop_resolution", {}).get("status", row["desktop_export"])
                if problem == "export_present":
                    continue
            finding = dict(path=path, reasons=reasons + (kind,), conditional=conditional or kind == "delay",
                           dll=dll, symbol=symbol, problem=problem, candidates=providers)
            if not providers and "desktop_resolution" in row:
                finding["desktop_chain"] = row["desktop_resolution"]["chain"]
            elif package and len(providers) == 1 and problem != "bundled_export_missing":
                finding["package_chain"] = package["chain"]
            findings.append(finding)

        if images[image]["sha256"] == FFX_LOADER:
            for dll, condition in (
                    ("amd_fidelityfx_upscaler_xbox.dll", "ffx:0x10000"),
                    ("amd_fidelityfx_framegeneration_xbox.dll", "ffx:0x20000|0x30000")):
                providers = by_name[dll]
                if len(providers) == 1:
                    targets.add((condition, providers[0]))
                    names = {e["name"] for e in images[providers[0]]["exports"]}
                    missing = sorted(set(FFX_EXPORTS) - names)
                    if not missing:
                        continue
                    problem = "conditional_exports_missing"
                else:
                    missing = []
                    problem = "conditional_module_absent_from_package" if not providers else "ambiguous_package_basename"
                findings.append(dict(path=path, reasons=reasons + (condition,), conditional=True, dll=dll,
                                     missing_exports=missing, problem=problem, candidates=providers))
        elif Path(image).name.casefold() == FFX_LOADER_NAME:
            # The conditional routes are pinned to one loader hash. Report a different build instead
            # of returning no conditional edge and exiting successfully.
            findings.append(dict(path=path, reasons=reasons, dll=Path(image).name,
                                 problem="unverified_loader_hash",
                                 sha256=images[image]["sha256"], expected=FFX_LOADER))
        for edge, target in sorted(targets):
            heapq.heappush(queue, (conditional or edge != "normal", depth + 1, path + (target,), reasons + (edge,)))
            for hops in sorted(forwarded[(edge, target)]):
                for n in range(1, len(hops) + 1):
                    heapq.heappush(queue, (conditional or edge != "normal", depth + 1 + n,
                                           path + (target, *hops[:n]), reasons + (edge,) + ("forwarder",) * n))
        # Packaged modules reached through a forwarder of a known System32 module: the System32
        # module itself is not part of the package graph, so the first packaged hop follows the image.
        for edge, hops in sorted(known_hops):
            for n in range(1, len(hops) + 1):
                heapq.heappush(queue, (conditional or edge != "normal", depth + n,
                                       path + hops[:n], reasons + ("forwarder",) * n))
    return dict(entry=entry, reachable=reached, findings=findings, dynamic_lookup_imports=lookups,
                unlinked_in_this_graph=sorted(set(images) - set(reached)),
                scope="Package imports, packaged export forwarders and identified FFX routes. Desktop findings "
                      "use terminal export status where available and first-hop status for older inventories. "
                      "Dynamic targets and actual execution order require independent evidence.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("entry", nargs="?", help="Relative PE path, or omit to assess every entry independently")
    args = parser.parse_args(argv)
    try:
        if args.inventory.is_dir():
            raise ValueError(f"Inventory path is a directory: {args.inventory}")
        data = args.inventory.read_bytes()
        inventory = json.loads(data)
        if not isinstance(inventory, dict):
            raise ValueError("Inventory JSON root must be an object")
        images = inventory.get("images")
        if not isinstance(images, dict):
            raise ValueError("Inventory JSON must contain an 'images' object")
        imports = inventory.get("imports")
        if not isinstance(imports, list):
            raise ValueError("Inventory JSON must contain an 'imports' list")
        if args.entry is not None and args.entry not in images:
            parser.error("Entry does not appear in this inventory")
        entries = [args.entry] if args.entry is not None else sorted(images)
        document = dict(inventory_sha256=hashlib.sha256(data).hexdigest(),
                        entries={name: trace(inventory, name) for name in entries})
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{args.inventory}: {exc}\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
