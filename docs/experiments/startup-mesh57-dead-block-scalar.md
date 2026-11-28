# Slot 57 mesh: dead scalar ALU in any EXEC-zero block

Part of #2413. Stacked on the pre-exchange wave block (#2783).

#2643 accepted scalar ALU with no reader past the count exchange, but only in the block that
`s_cbranch_execz` skips into the count exchange. Every other `s_cbranch_execz` block refused any
scalar instruction with `slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes`.

The model skips per lane, while the guest skips per wave. A scalar result that no path reads past
the join is invisible either way:
- A lane that skips the block never reads it.
- A lane that runs the block computes the wave-uniform value the guest wave computes.

This change applies that rule to every `s_cbranch_execz` block. The instruction must be scalar ALU
that `mesh_wave_block_scalar` (#2783) accepts: SGPR, SCC and VCC operands only, with no EXEC, M0,
`src_*`, MODE, memory, branch or message. Each register it writes, including the SCC every `s_` op
may write, must not be live at the join. Scalar loads keep their own rules.

## Tests

`tests/startup_mesh57_skeleton_check.py`:
- **New draw case:** the lane-masked if/else with `s_add_u32 s50, s13, 1` in the then arm. It is
  drawn on WARP and hardware with the if/else image of the existing case.
- **Former #2643 refusal controls, now accepted translations:**
  - `dead scalar ALU in a different EXEC-zero block`;
  - `an EXEC-zero branch that skips a dead scalar write`.
- **New refusal:** the same if/else with `v_mov_b32 v8, s50` after the EXEC restore, so the scalar is
  live at the join.
- **Fail-first:** without `dead_block_scalar`, all three accepted cases go red.

## Local capture census (counts only)

- On a local capture at main `96a7b4fb`, together with #2783: 91 of 407 slot 57 requests that stop
  at the execz refusal now pass it.
- They stop at `v_readfirstlane_b32 outside the group-index read is not covered`. That is a
  first-lane read of a formatted buffer load in the pre-exchange code.
- The 91 are all post-count primitive-export blocks with an `s_add_u32` that is dead past the jump
  to `s_endpgm`.
- No complete translation is added yet.
