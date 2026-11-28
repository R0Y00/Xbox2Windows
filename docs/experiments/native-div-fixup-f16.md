# Direct DXIL `v_div_fixup_f16`

Lowers `v_div_fixup_f16` (VOP3 `0x35f`) in the native direct-DXIL compute path
(`src/gfx1013_direct_compute.inl`) matching the RDNA 2 ISA reference guide
(section 12.12, opcode 863) using exact 32-bit integer IEEE-754 binary16
predicates while preserving `D[31:16]`.

## Scope

- Covers `S2`/`S1` NaN quieting (`| 0x0200`) with `Quiet(S2)` taking priority
  over `Quiet(S1)`, `0/0` and `Inf/Inf` returning `0xfe00`, `x/0` and `Inf/x`
  (`±Inf`), `0/x` and `x/Inf` (`±0.0h`), default `sign_out | abs(S0)` (including
  when `S0` is NaN), inline binary16 float constant `0.5` (`src0=240`), VOP3
  `abs`/`neg`/`clamp`, and `D[31:16]` preservation.
- Refuses unsupported modifiers (`omod`, `op_sel`) and graphics-input mode.
- Negative controls: omitting `sign_out` XOR, omitting `0/0`/`Inf/Inf` invalid
  check, swapping `S1`/`S2` NaN priority, omitting `clamp`, reading inline `0.5`
  as `0x0000`, and zeroing `D[31:16]` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_div_fixup_f16_' -V
```
