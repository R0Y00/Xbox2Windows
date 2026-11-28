# Labels

Labels are repository settings. They are not created by a pull request. A
maintainer with write access runs `tools/setup-labels.ps1` after this taxonomy is
accepted. The script creates or updates labels and never deletes existing ones.

Keep the label set small. Use one type label and one area label on a task when
possible. Use a read-only readiness label only when it changes what a contributor
can do next.

## Type

| Label | Use |
|---|---|
| `bug` | A reproducible build, runtime or fixture failure. |
| `documentation` | A documentation, glossary or diagram change. |
| `enhancement` | A proposed change to code, tooling or documentation. |
| `research` | A compatibility question with a reproducible test. |
| `build` | CMake, toolchain, packaging or build-script work. |
| `ci` | GitHub Actions and automation. |
| `tests` | Fixtures, coverage and test infrastructure. |
| `question` | A request for information that is not yet a task. |

Keep the GitHub defaults `duplicate`, `invalid`, `wontfix` and `accessibility`.

## Area

| Label | Use |
|---|---|
| `area/cpu` | x64 lifting, ABI, unwind and PE execution. |
| `area/gpu` | D3D12, BVH, DXR and GPU coverage. |
| `area/shader` | gfx1013 decoding, HLSL lowering and DXIL. |
| `area/ffx` | FidelityFX loader, upscaler and FSR4. |
| `area/textures` | BC7, BCPack, mip tails and texture views. |
| `area/script` | RPF, PSIN, state trees and script replay. |
| `area/pe` | PE mapping, imports, intake and dependency paths. |
| `area/runtime` | Heap, memory queries and Windows shims. |
| `area/tooling` | Python tools and fixtures. |
| `area/meta` | Contribution workflow, templates, labels and repository process. |

## Readiness

| Label | Use |
|---|---|
| `needs-repro` | The report is missing a reproducible input or command. |
| `needs-hardware` | A physical GPU or adapter-specific result is required. |
| `needs-windows` | The task requires the Windows build or D3D12 runtime. |
| `needs-game-files` | The task needs a local package; public fixtures are not enough. |
| `blocked` | Work cannot proceed until another task or decision is resolved. |

Keep `help wanted` for tasks that can proceed independently and
`good first issue` for tasks with a small first step.

## Rules

- Do not use `priority/*` labels. Use a GitHub Project or milestone for priority.
- Do not use `[PR]` or `[ISSUE]` title prefixes. GitHub already distinguishes
  the object types.
- Do not add `area/*` labels that have no matching code or documentation area.
- Hardware reports belong in the shared coverage thread (#1), not in a new issue.
- Remove `needs-repro` after a contributor supplies a reproducible command.

## Applying the labels

Preview the plan:

```powershell
.\tools\setup-labels.ps1 -DryRun
```

The dry run prints one row per proposed label with `CREATE`, `UPDATE` or
`UNCHANGED`. For an update it shows the old and new color or description. It also
lists existing labels that are not in the proposal and would be left untouched.

Create new labels and update only the labels whose color or description differs:

```powershell
.\tools\setup-labels.ps1
```

The script uses `gh label create` for new names and `gh label edit` for existing
names. It does not delete labels. It requires an authenticated `gh` CLI with
write access to the repository.
