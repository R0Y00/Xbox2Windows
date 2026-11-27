#!/usr/bin/env python3
"""Rank the first shader refusals of a run by the draws and dispatches they block.

Usage:
  refusal_census.py --cache <X2W_SHADER_CACHE dir> --log <stand-in stderr> [--log ...]
                    [--out-md FILE] [--out-tsv FILE] [--top N] [--shareable]

The shader cache holds one request set per key (req_<key>.<ext>) and, for every key the translator
refused, <key>.fail whose first line is the first refusal. The stage comes from the request files:
.c67 compute, .m57 / .m86 mesh, .a / .b graphics (VS+PS pair), .rtv render-target variant.

The weight of a refused key is the largest count the stand-in log gives it (each refused key blocks
every draw or dispatch that needed it):
  x2w-render compute census key K: N dispatches, T translated[, first refusal: ...]   -> N
  x2w-draw no RTV translation for K                                                 -> lines
  x2w-render translated ExecuteIndirect Draw refused, shader K                      -> lines
  x2w-render ExecuteIndirect refused, key K                                         -> lines
  x2w-render key use graphics|compute K ... uses N   (per-key use report, if the build prints it) -> N
  x2w-render not drawn key K: N                       (not-drawn report, if the build prints it)  -> N
A refused key no log line names has weight 0: it was requested but never needed in that run.

Classes: the fixed category labels of triage_run.py when one matches, otherwise the first refusal with
registers, numbers and addresses normalized. --shareable keeps only the fixed labels (unknown text becomes
"other (text omitted)"), like triage_run.py reports. Standard library only, deterministic output.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_run import OTHER, classify_shader_reason  # noqa: E402

STAGES = (("c67", "compute"), ("m57", "mesh57"), ("m86", "mesh86"), ("a", "graphics"), ("b", "graphics"),
          ("rtv", "rtv"))
KEY = r"([0-9a-f]{16})"
# (pattern, group holding the count or None to count lines)
WEIGHT_LINES = (
    (re.compile(r"compute census key " + KEY + r": (\d+) dispatches"), 2),
    (re.compile(r"no RTV translation for " + KEY), None),
    (re.compile(r"translated ExecuteIndirect Draw refused, shader " + KEY), None),
    (re.compile(r"ExecuteIndirect refused, key " + KEY), None),
    (re.compile(r"key use (?:graphics|compute) " + KEY + r" .* uses (\d+)"), 2),
    (re.compile(r"not drawn key " + KEY + r": (\d+)"), 2),
)
NORMALIZE = (
    (re.compile(r"\b(?:s|v|ttmp)(?:\d+|\[\d+:\d+\])"), "R"),
    (re.compile(r"\b(?:vcc_hi|vcc_lo|vcc|exec_lo|exec_hi|exec|m0)\b"), "R"),
    (re.compile(r"0x[0-9a-fA-F]+"), "H"),
    (re.compile(r"\d+"), "N"),
)


def first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0].strip() if stripped else "(empty .fail)"


def normalized(line: str) -> str:
    for pattern, replacement in NORMALIZE:
        line = pattern.sub(replacement, line)
    return line[:140]


def refusal_class(line: str, shareable: bool) -> str:
    label = classify_shader_reason(line)
    if label != OTHER or shareable:
        return label
    return normalized(line)


def stage_of(extensions: set[str]) -> str:
    return next((stage for ext, stage in STAGES if ext in extensions), "unknown")


def load_cache(cache: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    extensions: dict[str, set[str]] = collections.defaultdict(set)
    refusals: dict[str, str] = {}
    for path in sorted(cache.iterdir()):
        name = path.name
        if name.startswith("req_"):
            key, _, ext = name[4:].partition(".")
            extensions[key].add(ext)
        elif name.endswith(".fail"):
            refusals[name[:-5]] = first_line(path.read_text(errors="replace"))
    return extensions, refusals


def load_weights(logs: list[Path]) -> collections.Counter[str]:
    weights: collections.Counter[str] = collections.Counter()
    for log in logs:
        per_line: collections.Counter[tuple[int, str]] = collections.Counter()
        with log.open(errors="replace") as lines:
            for line in lines:
                if "x2w-" not in line:
                    continue
                for index, (pattern, group) in enumerate(WEIGHT_LINES):
                    match = pattern.search(line)
                    if not match:
                        continue
                    count = int(match[group]) if group else 1
                    slot = (index, match[1])
                    # Count lines add up; reported totals are cumulative, so the last (largest) one wins.
                    per_line[slot] = per_line[slot] + count if group is None else max(per_line[slot], count)
                    break
        for (_, key), count in per_line.items():
            weights[key] = max(weights[key], count)
    return weights


def census_rows(cache: Path, logs: list[Path], shareable: bool) -> list[tuple[str, str, str, int]]:
    """(key, stage, class, weight) of every refused key, heaviest first."""
    extensions, refusals = load_cache(cache)
    weights = load_weights(logs)
    rows = [(key, stage_of(extensions.get(key, set())), refusal_class(line, shareable), weights[key])
            for key, line in refusals.items()]
    return sorted(rows, key=lambda row: (-row[3], row[0]))


def ranked_classes(rows: list[tuple[str, str, str, int]]) -> list[tuple[str, int, int, str]]:
    """(class, keys, weight, stages) heaviest first."""
    totals: dict[str, list] = collections.defaultdict(lambda: [0, 0, set()])
    for _, stage, label, weight in rows:
        entry = totals[label]
        entry[0] += 1
        entry[1] += weight
        entry[2].add(stage)
    ranked = sorted(totals.items(), key=lambda item: (-item[1][1], -item[1][0], item[0]))
    return [(label, keys, weight, ",".join(sorted(stages))) for label, (keys, weight, stages) in ranked]


def markdown(rows: list[tuple[str, str, str, int]], top: int) -> str:
    out = ["# Shader refusal census", "",
           "Weight = draws/dispatches the stand-in log names for the refused key.", "",
           "| stage | refused keys | used in the run | blocked weight |", "|---|---|---|---|"]
    by_stage: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for _, stage, _, weight in rows:
        by_stage[stage][0] += 1
        by_stage[stage][1] += weight > 0
        by_stage[stage][2] += weight
    out += ["| %s | %d | %d | %d |" % (stage, *by_stage[stage]) for stage in sorted(by_stage)]
    out += ["", "| # | first refusal class | keys | blocked weight | stages |", "|---|---|---|---|---|"]
    for rank, (label, keys, weight, stages) in enumerate(ranked_classes(rows)[:top], 1):
        out.append("| %d | %s | %d | %d | %s |" % (rank, label.replace("|", "/"), keys, weight, stages))
    return "\n".join(out) + "\n"


def tsv(rows: list[tuple[str, str, str, int]]) -> str:
    lines = ["key\tstage\tclass\tweight"] + ["%s\t%s\t%s\t%d" % row for row in rows]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, required=True, help="X2W_SHADER_CACHE directory")
    parser.add_argument("--log", type=Path, action="append", required=True, help="stand-in stderr (repeatable)")
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-tsv", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--shareable", action="store_true", help="fixed category labels only, no refusal text")
    args = parser.parse_args(argv)
    if not args.cache.is_dir():
        parser.error("--cache %s is not a directory" % args.cache)
    missing = [str(log) for log in args.log if not log.is_file()]
    if missing:
        parser.error("--log not found: %s" % ", ".join(missing))
    rows = census_rows(args.cache, args.log, args.shareable)
    report = markdown(rows, args.top)
    if args.out_md:
        args.out_md.write_text(report)
    else:
        sys.stdout.write(report)
    if args.out_tsv:
        args.out_tsv.write_text(tsv(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
