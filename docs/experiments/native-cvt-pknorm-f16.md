# Direct DXIL `v_cvt_pknorm_i16_f16` and `v_cvt_pknorm_u16_f16`

Lowers `v_cvt_pknorm_i16_f16` (VOP3 `0x312`) and `v_cvt_pknorm_u16_f16` (VOP3
`0x313`) in the native direct-DXIL compute path (`src/gfx1013_direct_compute.inl`)
using exact 32-bit integer RNE normalization of two binary16 inputs into a
packed 2×16-bit integer destination.

## Scope

- Covers ±0.0h, ±min_sub, ±0.25h, ±0.5h ties, double-rounding boundary values
  (`0x3801`–`0x3804`), ±1.0h, ±2.0h/±4.0h saturation, ±Inf, and qNaN, plus
  VOP3 `op_sel[1:0]`, `abs`/`neg`, and inline float constant `0.5` (`src0=240`).
- Refuses unsupported modifiers (`clamp`, `omod`, `op_sel[3:2]`, non-VGPR
  high-half `op_sel`, unused `src2`) and non-default binary16 `MODE`.
- Negative controls: truncating instead of RNE and reading inline `0.5` as
  `0x0000` each differ from the reference.
- Does not establish full-game execution.

## Reproduction

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_cvt_pknorm_f16_' -V
```
