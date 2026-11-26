# CI check lists

The inline lists in `.github/workflows/windows.yml` stay as they are. A new Python check can
instead go in a small text file here, so two pull requests that each add a check do not edit
the same lines.

| Directory | Read by | One line holds |
|---|---|---|
| `python-linux/*.txt` | the `Run Python checks` step of the `python-linux` job | a script path, for example `tests/my_check.py` |
| `pixel-binary/*.txt` | the machine-code step of the `pixel-binary` job | a script path and its arguments, for example `tests/my_check.py binary llvm-mc-15` |

Rules, enforced by `tools/run_checks_parallel.py`:

- Use one file per topic or pull request, for example `python-linux/my_topic.txt`.
- Blank lines and lines starting with `#` are ignored. Files are read in name order.
- A script that does not exist, or a command listed twice across the files, fails the job.
- A command that is also in the inline list runs once.

Windows jobs register their tests with CTest instead: see [tests/cmake](../cmake/README.md).
