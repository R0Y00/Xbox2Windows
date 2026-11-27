#!/usr/bin/env python3
"""tools/startup/refusal_census.py on a synthetic shader cache and stand-in log, through its command line."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "startup" / "refusal_census.py"

COMPUTE = "00000000000000c1"
GRAPHICS = "00000000000000a1"
GRAPHICS_SAME_CLASS = "00000000000000a2"
TRANSLATED = "00000000000000d1"
UNUSED = "00000000000000e1"

LOG = "\n".join([
    f"x2w-render compute census key {COMPUTE}: 5 dispatches, 0 translated, first refusal: x [00 00 00 00]",
    f"x2w-render compute census key {COMPUTE}: 7 dispatches, 0 translated, first refusal: x [00 00 00 00]",
    f"x2w-draw no RTV translation for {GRAPHICS} (request)",
    f"x2w-draw no RTV translation for {GRAPHICS} (request)",
    f"x2w-render translated ExecuteIndirect Draw refused, shader {GRAPHICS}: no translation",
    f"x2w-render compute census key {TRANSLATED}: 9 dispatches, 9 translated",
    "unrelated line naming 00000000000000e1 without the stand-in prefix",
]) + "\n"


def write_cache(cache: Path) -> None:
    files = {
        f"req_{COMPUTE}.c67": "",
        f"{COMPUTE}.fail": "native compute +0x40: unsupported instruction ds_swizzle_b32\nmore detail\n",
        f"req_{GRAPHICS}.a": "", f"req_{GRAPHICS}.b": "",
        f"{GRAPHICS}.fail": "lane walk of v12 through s[4:5] at 0x30\n",
        f"req_{GRAPHICS_SAME_CLASS}.a": "", f"req_{GRAPHICS_SAME_CLASS}.b": "",
        f"{GRAPHICS_SAME_CLASS}.fail": "lane walk of v3 through s[8:9] at 0x1c\n",
        f"req_{TRANSLATED}.c67": "",
        f"req_{UNUSED}.c67": "", f"{UNUSED}.fail": "native compute +0x8: unsupported opcode 77\n",
    }
    for name, text in files.items():
        (cache / name).write_text(text)


class RefusalCensusCli(unittest.TestCase):
    def run_tool(self, *extra: str, logs: tuple[str, ...] = (LOG,),
                 cached_failures: tuple[str, ...] = ()) -> tuple[str, list[list[str]]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            write_cache(cache)
            for key in cached_failures:
                (cache / f"{key}.fail").write_text("unsupported opcode\n")
            log_args = []
            for index, content in enumerate(logs):
                log = root / f"stderr-{index}.txt"
                log.write_text(content)
                log_args.extend(["--log", str(log)])
            out_tsv = root / "census.tsv"
            result = subprocess.run([sys.executable, "-I", str(TOOL), "--cache", str(cache), *log_args,
                                     "--out-tsv", str(out_tsv), *extra],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [line.split("\t") for line in out_tsv.read_text().splitlines()]
        return result.stdout, rows

    def test_rows_weight_each_refused_key_by_the_work_it_blocks(self):
        _, rows = self.run_tool()
        self.assertEqual(rows, [
            ["key", "stage", "class", "weight"],
            [COMPUTE, "compute", "ds_* instruction", "7"],
            [GRAPHICS, "graphics", "lane walk of R through R at H", "2"],
            [GRAPHICS_SAME_CLASS, "graphics", "lane walk of R through R at H", "0"],
            [UNUSED, "compute", "unsupported opcode", "0"],
        ])

    def test_report_ranks_classes_by_blocked_weight(self):
        report, _ = self.run_tool()
        ranking = [line.split(" | ")[1] for line in report.splitlines() if line.startswith("| 1 ") or
                   line.startswith("| 2 ") or line.startswith("| 3 ")]
        self.assertEqual(ranking, ["ds_* instruction", "lane walk of R through R at H", "unsupported opcode"])
        self.assertIn("| lane walk of R through R at H | 2 | 2 | graphics |", report)
        self.assertIn("| compute | 2 | 1 | 7 |", report)

    def test_shareable_report_omits_unknown_refusal_text(self):
        report, rows = self.run_tool("--shareable")
        self.assertNotIn("lane walk", report)
        self.assertEqual(rows[2][2], "other (text omitted)")

    def test_missing_log_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-I", str(TOOL), "--cache", tmp, "--log",
                                     str(Path(tmp) / "absent.txt")], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--log not found", result.stderr)

    def test_compute_weights_count_blocked_dispatches(self):
        log = "\n".join([
            f"x2w-render compute census key {COMPUTE}: 12 dispatches, 10 translated, first refusal: example",
            f"x2w-render compute census key {TRANSLATED}: 12 dispatches, 12 translated",
            f"x2w-render compute census key {UNUSED}: 8 dispatches, first refusal: old format",
            f"x2w-render compute census key {GRAPHICS_SAME_CLASS}: 4 dispatches, 5 translated",
            f"x2w-render key use graphics {GRAPHICS} first frame 1 uses 6",
        ])
        estimates = "\n".join(f"x2w-render key use compute {key} first frame 1 uses 99"
                              for key in (COMPUTE, TRANSLATED, UNUSED))
        _, rows = self.run_tool(logs=(estimates, log), cached_failures=(TRANSLATED,))
        self.assertEqual({row[0]: int(row[3]) for row in rows[1:]}, {
            COMPUTE: 2, TRANSLATED: 0, UNUSED: 8, GRAPHICS: 6, GRAPHICS_SAME_CLASS: 0,
        })

    def test_runtime_refusals_are_cumulative_and_separate(self):
        def line(key: str, reason: str, calls: int) -> str:
            return f"x2w-d3d12 indirect-draw refusal 1: key={key} reason={reason} calls={calls}"

        logs = (
            LOG + "\n".join([
                *(line(GRAPHICS, "missing_guest_root", calls) for calls in (2, 5, 5)),
                *(line(GRAPHICS, "missing_pipeline", calls) for calls in (1, 3, 3)),
                *(line(TRANSLATED, "missing_guest_root", calls) for calls in (1, 4, 4)),
                *(line("0" * 16, "missing_guest_root", calls) for calls in (2, 8, 8)),
            ]),
            "\n".join([line(GRAPHICS, "missing_guest_root", 4),
                       line(TRANSLATED, "missing_guest_root", 6)]),
        )
        report, rows = self.run_tool(logs=logs)
        _, shader_rows = self.run_tool()
        self.assertEqual(rows, shader_rows + [
            [''],
            ["key", "indirect-draw reason", "calls"],
            [TRANSLATED, "missing_guest_root", "6"],
            [GRAPHICS, "missing_guest_root", "5"],
            [GRAPHICS, "missing_pipeline", "3"],
        ])
        self.assertIn("## Runtime indirect-draw refusals", report)
        self.assertIn(f"| {TRANSLATED} | missing_guest_root | 6 |", report)
        self.assertIn(f"| {GRAPHICS} | missing_guest_root | 5 |", report)
        self.assertNotIn("0000000000000000", report)


if __name__ == "__main__":
    unittest.main()
