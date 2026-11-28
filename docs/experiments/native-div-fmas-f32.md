# Native direct-DXIL compute: `v_div_fmas_f32`

Refs #2935, #3327.

## Scope

Lowers VOP3A `v_div_fmas_f32` (opcode `0x16f`) in the native direct-DXIL compute path
([gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl)):

- Reads the per-lane `VCC` bit (`vcc_lo` in wave32, `vcc_lo`/`vcc_hi` selected by `upper_half` in wave64).
- Computes fused binary32 multiply-add via `dx.op.tertiary.f32(46)` (`FMad`), scaling by `2^32` (`4294967296.0f`)
  when the lane's `VCC` bit is set and leaving the result unscaled when `VCC` is clear.
- Pre-scales inputs by `2^32` when `|s2| < 2^95` and either `|s0| < 2^95` or `|s1| < 2^95` so subnormal intermediate
  sums (such as `2^-126 * 0.5 = 2^-127`) are not flushed before scaling, avoiding any dependency on
  `DoublePrecisionFloatShaderOps`; falls back to post-scaling `fma(s0, s1, s2) * 2^32` when inputs exceed `2^95` or
  are non-finite.
- Supports VOP3 `abs`, `neg`, and `clamp` (`dx.op.unary.f32(7)`); refuses `omod`, `op_sel`, `subword`, `dpp`, and
  non-VOP3 encodings. Graphics-input mode remains refused.

## Verification

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_div_fmas_f32_' -V
```

- `native_div_fmas_f32_controls`: compiles plain, `-|s0|`/`-s2`, and `clamp` forms, asserts DXIL
  `dx.op.tertiary.f32(i32 46, ...)` and `dx.op.unary.f32(i32 7, ...)`, verifies `omod` and `op_sel` refusals, and
  checks negative-control reference mutants (`ignore_vcc`, `invert_vcc`, `unfused`, `no_clamp`).
- `native_div_fmas_f32_hardware`: dispatches two wave32 thread groups with `vcc_lo = 0xAAAAAAAA` (even lanes `VCC=0`,
  odd lanes `VCC=1`) across 8 cases including subnormal intermediate `2^-126 * 0.5`, large-`s0` pre-scale,
  large-`s2` post-scale overflow, exact FMA cancellation `(1 + 2^-23)*(1 - 2^-23) - 1`, `clamp`, `+Inf`, and `qNaN`,
  and verifies all 2064 readback words (including sentinel guards). Returns capability skip (77) if SM 6.6 / wave32
  is unavailable.

## Limits

This validates synthetic compute shaders compiled through the direct-DXIL path on desktop D3D12 hardware; it does not
establish full-game startup or console equivalence.
