# All-or-none gate for the BVH builder's compute stages

The console ray-tracing structure builder is a set of compute programs from `XboxBvhBuilder.dll`.
Its stages feed each other through title memory; a stage that runs on data a missing stage never
wrote computes store offsets from garbage. In the observed failure, 14 of 35 translated stages ran
and the title died within a second on pointers that had become bounding-box bit patterns.

`X2W_COMPUTE_BUILDER_GATE=1` makes the set run all or none. It is off by default and changes no
existing path when unset.

## Identification

At device slot 62, the gate resolves the caller module of the return address with
`GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS)` and compares the basename with
`XboxBvhBuilder.dll`, case-insensitively. A compute pipeline is tracked only when that resolution
succeeds; an unresolved caller is left untracked rather than guessed. The header's pure classifier
and state machine are tested without a device or a title module.

## Decision

A builder Dispatch is allowed when every tracked builder pipeline has a real desktop PSO. On the
first builder Dispatch the gate makes the missing pipelines on the spot. If one cannot be made, the
builder is turned off for the rest of the run and the Dispatch is refused with a named reason. A
non-builder Dispatch is unchanged. The gate is keyed on the slot 62 wrapper's builder tag, not on
`compute_request_key`: slot 67 never receives that tag, so its request publication path is
unchanged. The gate is counted in the compute totals line as `builder gate refused N`.

## Scope and limits

This is a safety gate, not a translation. The slot 62 programs are console GPU code; the gate's
on-the-spot creation first requires a structurally valid desktop compute container and a desktop
root signature. A console DXIL program fails by name and latches the builder off, which is the
intended conservative result until the translation path publishes real compute PSOs.

The runtime test loads a source-built `XboxBvhBuilder.dll` helper and calls the public slot 62
path through it. Two fake console pipelines are tracked, a direct `Dispatch` and an indirect
`ExecuteIndirect` are both refused with the gate enabled, and the same direct call through a
non-builder slot 62 wrapper is left untouched. A separate run of the existing slot 67 request check
with `X2W_COMPUTE_BUILDER_GATE=1` passes, so request publication keeps its normal path.

The helper test proves the caller-module tag and the wiring into both public Dispatch paths. It
does not prove that a translated builder set can be completed; the fake console container fails
desktop PSO creation by name and latches the builder off, which is the intended conservative result.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate$' -V
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate_runtime$' -V
ctest --preset dev --no-tests=error -R '^startup_compute_builder_gate_request_warp$' -V
```
