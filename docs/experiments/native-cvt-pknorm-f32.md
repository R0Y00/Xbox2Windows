# Packed normalized integer conversion from binary32 (`v_cvt_pknorm_i16_f32` / `v_cvt_pknorm_u16_f32`) in direct DXIL

The native frontend lowers gfx1013 VOP3 `v_cvt_pknorm_i16_f32` (`0x368`) and
`v_cvt_pknorm_u16_f32` (`0x369`) in `DirectCompute::float_alu()` for both Compute
and Pixel shaders, packing `cvt_norm(S0.f32)` into `D[15:0]` and
`cvt_norm(S1.f32)` into `D[31:16]`.

## Exact 32-bit integer RNE lowering and semantics

Both lanes compute exact `round_rne(clamp(x, -1, 1) * 32767)` (`i16`) and
`round_rne(clamp(x, 0, 1) * 65535)` (`u16`) in 32-bit unsigned integer
arithmetic without floating-point intermediaries or 64-bit integers:

- For normal binary32 inputs with `mag in [2^-17, 1.0)` (`exp in [110, 126]`),
  the 24-bit significand `m` is multiplied by `2^k - 1` (`k = 15` for `i16`,
  `k = 16` for `u16`) as `(m << k) - m` split cleanly at bit `k` into `(p_hi,
  p_lo)`, and rounded to nearest even using both `rem_hi` against `mid_hi` and
  the nonzero carry-over `p_lo != 0` bit.
- Magnitudes `< 2^-17` (zeros and all binary32 subnormals included) round to `0`.
- Magnitudes `>= 1.0` (`±Inf` included) saturate to `±32767` (`i16`) or `65535`
  / `0` (`u16`).
- Any negative value (`sign != 0`) maps to `0` on `v_cvt_pknorm_u16_f32`.
- Non-finite NaN operands (`mag > 0x7f800000`) map to `0`.
- Supported VOP3 input modifiers are `abs[1:0]` and `neg[1:0]` on `S0` and
  `S1`, with `EXEC`-masked destination writes and VGPR, SGPR, 32-bit literal,
  and inline `f32`/integer sources via `operand32()`.

## Refused encodings

The translator refuses `v_cvt_pknorm_i16_f32` and `v_cvt_pknorm_u16_f32` when
any of the following hold:

- VOP3 `clamp != 0` or `omod != 0`.
- Any `op_sel` bits (`op_sel[3:0]`), third-operand `abs`/`neg` bits
  (`abs[2]`/`neg[2]`), or non-zero `src2`.
- DPP16, SDWA, permuted/subword encodings, or unsupported operand codes (such as
  `254`).

## Fixture coverage and reproduction

[native_cvt_pknorm_f32_check.py](../../tests/native_cvt_pknorm_f32_check.py)
checks 26 Compute cases (`<= 32` words/lane in `native_store_replay`), 8
refusal controls, and lossless 16-bit Pixel readback across 13 draws (25
full-`EXEC` cases padded to 26) on WARP and D3D12 hardware:

```powershell
cmake --build --preset dev --target compile_dxil native_compute_ir_dump native_store_replay offscreen_check
ctest --test-dir out/build/dev --no-tests=error --output-on-failure -j 1 -R '^native_cvt_pknorm_f32_' -V
```

Compute replay requires shader model 6.6 with wave32; adapters without wave32
report a skip (`77`) on `native_cvt_pknorm_f32_compute_hardware`.
