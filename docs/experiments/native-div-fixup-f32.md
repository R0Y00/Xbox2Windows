# Direct DXIL `v_div_fixup_f32`

Lowers `v_div_fixup_f32` (VOP3 `0x15f`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) matching the RDNA 2 ISA reference guide
(section 12.12, opcode 351) using exact 32-bit integer IEEE-754 binary32
predicates.

## Scope

- Covers `S2`/`S1` NaN quieting with `Quiet(S2)` taking priority over
  `Quiet(S1)`, `0/0` and `Inf/Inf` returning `0xffc00000`, `x/0` and `Inf/x`
  (`±Inf`), `0/x` and `x/Inf` (`±0.0`), exponent underflow
  `exponent(S2) - exponent(S1) < -150` returning signed zero, default
  `sign_out | abs(S0)` (including when `S0` is NaN), and VOP3 `abs`/`neg`/`clamp`.
- Refuses unsupported modifiers (`omod`, `op_sel`).
- Negative controls: omitting `sign_out` XOR, omitting `0/0`/`Inf/Inf` invalid
  check, swapping `S1`/`S2` NaN priority, omitting exponent underflow, and
  omitting `clamp` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_div_fixup_f32_' -V
```
