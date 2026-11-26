#!/usr/bin/env python3
"""Pure-Python CLI validation, comment-ID guard, and mutation suite for tools/hardware_reports.py."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"
TOOL = TOOLS_DIR / "hardware_reports.py"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TESTS_DIR))

import hardware_reports as hr  # noqa: E402
from hardware_reports_check import README, comment, report  # noqa: E402


def _run_main(argv: list[str] | None) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = hr.main(argv)
        except SystemExit as exc:
            code = exc.code
            if not isinstance(code, int):
                if code is not None:
                    stderr.write(f"{code}\n")
                code = 0 if code is None else 1
    return code, stdout.getvalue(), stderr.getvalue()


def _run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(TOOL), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
    )


class HardwareReportsCliPositiveTests(unittest.TestCase):
    def test_markdown_json_after_filtering_and_boundary_id_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            readme = base / "README.md"
            readme.write_text(README, encoding="utf-8")

            c0 = comment(0, report(adapter="Example GPU 000", device="0001", driver="0.9"))
            c101 = comment(101, report())
            c202 = comment(202, report(adapter="Example GPU 300", device="3333", driver="3.0"))
            c205 = comment(205, report(adapter="Example GPU 400", device="4444", driver="4.0"))

            page1 = base / "page1.json"
            page2 = base / "page2.json"
            page1.write_text(json.dumps([c0, c101]), encoding="utf-8")
            page2.write_text(json.dumps([c202]) + "\n" + json.dumps([c205]), encoding="utf-16")

            # 1) Markdown output across multiple comment files, including id=0.
            code, out, err = _run_main([str(page1), str(page2), "--readme", str(readme)])
            self.assertIs(type(code), int)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertIn("Read 4 comments. 4 reports use the generated layout; README.md links 1 of them.", out)
            self.assertIn("## 1. Example GPU 000 (new row)", out)
            self.assertIn("## 2. Example GPU 300 (new row)", out)
            self.assertIn("## 3. Example GPU 400 (new row)", out)

            # 2) --json with --after 0 includes positive IDs (and excludes id=0).
            code_a0, out_a0, err_a0 = _run_main(
                [str(page1), str(page2), "--readme", str(readme), "--after", "0", "--json"]
            )
            self.assertEqual(code_a0, 0, err_a0)
            self.assertEqual(err_a0, "")
            data_a0 = json.loads(out_a0)
            self.assertEqual(data_a0["after"], 0)
            self.assertEqual([item["comment_id"] for item in data_a0["unrecorded"]], [202, 205])

            # 3) --json with --after 202 filters to > 202.
            code_a202, out_a202, err_a202 = _run_main(
                [str(page1), str(page2), "--readme", str(readme), "--after", "202", "--json"]
            )
            self.assertEqual(code_a202, 0, err_a202)
            data_a202 = json.loads(out_a202)
            self.assertEqual(data_a202["after"], 202)
            self.assertEqual([item["comment_id"] for item in data_a202["unrecorded"]], [205])

            # 4) Subprocess end-to-end positive check.
            proc = _run_subprocess(str(page1), str(page2), "--readme", str(readme), "--after", "202", "--json")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stderr, "")
            self.assertEqual(json.loads(proc.stdout), data_a202)

    def test_sys_argv_and_script_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            readme = base / "README.md"
            readme.write_text(README, encoding="utf-8")
            comments_file = base / "comments.json"
            comments_file.write_text(
                json.dumps([comment(202, report(adapter="Example GPU 300", device="3333", driver="3.0"))]),
                encoding="utf-8",
            )
            cli_argv = [str(TOOL), str(comments_file), "--readme", str(readme), "--json"]

            with mock.patch.object(sys, "argv", cli_argv):
                code, out, err = _run_main(None)
            self.assertEqual(code, 0, err)
            self.assertEqual(err, "")
            self.assertEqual([item["comment_id"] for item in json.loads(out)["unrecorded"]], [202])

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(sys, "argv", cli_argv),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(TOOL), run_name="__main__")
            self.assertEqual(exited.exception.code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual([item["comment_id"] for item in json.loads(stdout.getvalue())["unrecorded"]], [202])


class HardwareReportsCliNegativeAndMutationTests(unittest.TestCase):
    def _assert_clean_failure(
        self,
        argv: list[str],
        *,
        expected_code: int | None = 2,
        fragments: tuple[str, ...] = (),
        check_subprocess: bool = True,
    ) -> str:
        code, out, err = _run_main(argv)
        if expected_code is None:
            self.assertNotEqual(code, 0)
        else:
            self.assertEqual(code, expected_code, err)
        self.assertEqual(out, "")
        self.assertNotIn("Traceback", err)
        self.assertTrue(err.strip(), f"expected stderr for {argv!r}")
        for fragment in fragments:
            self.assertIn(fragment.lower(), err.lower())

        if check_subprocess:
            proc = _run_subprocess(*argv)
            if expected_code is None:
                self.assertNotEqual(proc.returncode, 0)
            else:
                self.assertEqual(proc.returncode, expected_code, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertNotIn("Traceback", proc.stderr)
            self.assertTrue(proc.stderr.strip(), f"expected subprocess stderr for {argv!r}")
            for fragment in fragments:
                self.assertIn(fragment.lower(), proc.stderr.lower())
        return err

    def test_upfront_paths_and_negative_after_rejected_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            readme = base / "README.md"
            readme.write_text(README, encoding="utf-8")
            valid_comments = base / "valid.json"
            valid_comments.write_text("[]", encoding="utf-8")
            missing_comments = base / "missing.json"
            missing_readme = base / "missing_readme.md"
            a_dir = base / "a_dir"
            a_dir.mkdir()

            with (
                mock.patch.object(hr, "load_comments", side_effect=AssertionError("load_comments called upfront")),
                mock.patch.object(hr, "summarize", side_effect=AssertionError("summarize called upfront")),
            ):
                for label, argv, frag in (
                    ("missing_comments", [str(missing_comments), "--readme", str(readme)], "comments file not found"),
                    (
                        "second_missing_comments",
                        [str(valid_comments), str(missing_comments), "--readme", str(readme)],
                        "comments file not found",
                    ),
                    ("dir_comments", [str(a_dir), "--readme", str(readme)], "comments path is a directory"),
                    (
                        "second_dir_comments",
                        [str(valid_comments), str(a_dir), "--readme", str(readme)],
                        "comments path is a directory",
                    ),
                    ("missing_readme", [str(valid_comments), "--readme", str(missing_readme)], "--readme file not found"),
                    ("dir_readme", [str(valid_comments), "--readme", str(a_dir)], "--readme path is a directory"),
                    ("neg_after_1", [str(valid_comments), "--readme", str(readme), "--after", "-1"], "non-negative"),
                    ("neg_after_99", [str(valid_comments), "--readme", str(readme), "--after", "-99"], "non-negative"),
                ):
                    with self.subTest(case=label):
                        self._assert_clean_failure(argv, expected_code=2, fragments=(frag,))

    def test_load_comments_rejects_bool_negative_non_int_ids_and_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            readme = base / "README.md"
            readme.write_text(README, encoding="utf-8")

            for label, bad_id in (
                ("bool_true", True),
                ("bool_false", False),
                ("neg_one", -1),
                ("neg_large", -100),
                ("str_id", "101"),
                ("float_id", 10.5),
                ("none_id", None),
            ):
                with self.subTest(case=label, bad_id=bad_id):
                    bad_file = base / f"{label}.json"
                    bad_file.write_text(
                        json.dumps([{"id": bad_id, "body": "x"}]),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError) as ctx:
                        hr.load_comments([bad_file])
                    self.assertIn("non-negative integer id", str(ctx.exception))
                    self._assert_clean_failure(
                        [str(bad_file), "--readme", str(readme)],
                        expected_code=2,
                        fragments=("non-negative integer id",),
                    )

            # Missing body / non-dict item / malformed JSON / bad UTF-8 / bad README table.
            missing_body = base / "missing_body.json"
            missing_body.write_text(json.dumps([{"id": 1}]), encoding="utf-8")
            self._assert_clean_failure(
                [str(missing_body), "--readme", str(readme)],
                expected_code=2,
                fragments=("expected github issue comment objects",),
            )

            non_dict = base / "non_dict.json"
            non_dict.write_text(json.dumps([123]), encoding="utf-8")
            self._assert_clean_failure(
                [str(non_dict), "--readme", str(readme)],
                expected_code=2,
                fragments=("expected github issue comment objects",),
            )

            malformed = base / "malformed.json"
            malformed.write_text('[{"id": 1, "body": ', encoding="utf-8")
            self._assert_clean_failure([str(malformed), "--readme", str(readme)], expected_code=2)

            bad_utf8 = base / "bad_utf8.json"
            bad_utf8.write_bytes(b"\xff\xfe\x00")
            self._assert_clean_failure([str(bad_utf8), "--readme", str(readme)], expected_code=2)

            valid_comments = base / "valid.json"
            valid_comments.write_text("[]", encoding="utf-8")
            bad_readme = base / "bad_readme.md"
            bad_readme.write_text("# No coverage table here\n", encoding="utf-8")
            self._assert_clean_failure(
                [str(valid_comments), "--readme", str(bad_readme)],
                expected_code=2,
                fragments=("no hardware coverage table header",),
            )

    def test_argparse_errors_py314_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            comments = base / "comments.json"
            comments.write_text("[]", encoding="utf-8")

            for argv, frags in (
                ([], ("required",)),
                ([str(comments), "--bogus-flag"], ("unrecognized",)),
                ([str(comments), "--after", "not_int"], ("invalid",)),
                ([str(comments), "--readme"], ("expected one argument",)),
            ):
                with self.subTest(argv=argv):
                    self._assert_clean_failure(argv, expected_code=None, fragments=frags)


if __name__ == "__main__":
    unittest.main()
