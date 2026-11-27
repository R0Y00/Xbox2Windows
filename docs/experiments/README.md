# Experiment notes, one file per topic

Write a new experiment note in `docs/experiments/<topic>.md` instead of appending to
[docs/experiments.md](../experiments.md), so two pull requests that each add a note do not edit
the same lines. The older notes stay in `docs/experiments.md`.

A note follows the same rules as before: what was measured, the exact command, the result, the
environment, and what remains a hypothesis. Start the file with a `#` heading. Links in these
files are checked by `tests/docs_links_check.py`, like every tracked Markdown file.
