# Direct DXIL `v_frexp_mant_f16` and `v_frexp_exp_i16_f16`

Lowers `v_frexp_mant_f16` (VOP1 `0x59` / VOP3A `0x1d9`) and
`v_frexp_exp_i16_f16` (VOP1 `0x5a` / VOP3A `0x1da`) in the native direct-DXIL
compute path (`src/gfx1013_direct_compute.inl`) via exact integer IEEE-754
binary16 manipulation while preserving `D[31:16]`.

## Scope

- Covers ±0.0h, normals, smallest normal, largest and smallest subnormals, and
  +Inf across VOP1 and VOP3A (`neg`, `abs`, and `clamp` on `v_frexp_mant_f16`),
  verifying `D[31:16]` preservation on every output.
- Refuses unsupported modifiers (`clamp` on `v_frexp_exp_i16_f16`, `omod` on
  `v_frexp_mant_f16`).
- Negative controls: flushing denormals, omitting `clamp`, and zeroing
  `D[31:16]` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_frexp_f16_' -V
```
