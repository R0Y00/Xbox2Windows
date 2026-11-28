# Binary16 VOP1/VOP3/SDWA `v_sin_f16` and `v_cos_f16` in direct Compute DXIL

Issue #3298 (Refs #2935) lowers `v_sin_f16` (`0x60` / `0x1e0`) and `v_cos_f16`
(`0x61` / `0x1e1`) in the direct Compute DXIL translator via
`binary16_unary()` (`src/gfx1013_compute_native.cpp`) and `DirectCompute::trig_turns()`
/ `float_alu()` (`src/gfx1013_direct_compute.inl`).

## Lowering & Semantics

- Both instructions share `trig_turns()` with `v_sin_f32` and `v_cos_f32`,
  reducing the widened binary32 angle in revolutions (`part = a - trunc(a)` with
  signed zero preserved) before multiplying by `2 * pi`, and selecting exact
  `+0.0` at exact quarter/half turns (`|part| == 0.5` for `v_sin_f16`;
  `|part| == 0.25` or `0.75` for `v_cos_f16`) prior to `packed_half_rne()`.
- Special values on the modified binary16 source:
  - `±Inf` and `NaN` (`|source| >= 0x7c00`) map to canonical quiet NaN (`0x7e00`).
  - `v_sin_f16(-0.0)` (`0x8000`) preserves `-0.0` (`0x8000`); `v_cos_f16(±0.0)`
    returns `+1.0` (`0x3c00`).
- Preserves destination high bits `D[31:16]` on VOP1/VOP3 and supports SDWA
  subword selection/padding/preservation, `abs`/`neg`, SGPR/VGPR/literal, and
  inline float/integer constants under the default binary16 RNE + keep-denormals
  `MODE` contract.
- Refuses `clamp`, `omod`, high-half VOP3 `op_sel`, DPP16/DPP8 encodings,
  non-default binary16 `MODE`, and SDWA high-half selection on inline constants.

## Verification

- [native_sin_cos_f16_check.py](../../tests/native_sin_cos_f16_check.py) checks
  34 cases (`VOP1`, `VOP3`, `SDWA`, `abs`/`neg`, inline/literal/SGPR/overlap,
  `±0`, `±0.25`, `±0.5`, `±0.75`, `-1.75`, `2048.0`, `65504.0`, `±Inf`, `qNaN`,
  `sNaN`, subnormals, general angles) and 8 negative refusal cases across `ir`,
  `controls`, and `hardware` (`tests/cmake/native_sin_cos_f16.cmake`).
