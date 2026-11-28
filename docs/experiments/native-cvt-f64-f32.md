# Native direct-DXIL compute: `v_cvt_f64_f32`

Refs #2935, #3368.

## Scope

Lowers VOP1/VOP3A `v_cvt_f64_f32` (`0x10` / `0x190`) in the native direct-DXIL
compute path ([gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl))
and registers its 64-bit VGPR pair destination in `clear_vector_spills`
([gfx1013_compute_native.cpp](../../src/gfx1013_compute_native.cpp)):

- Constructs the exact IEEE-754 binary64 word pair `v[d:d+1]` from a binary32
  source using pure 32-bit integer arithmetic and `dx.op.unaryBits.i32(33)`
  (`FirstbitHi`). Requires neither `DoublePrecisionFloatShaderOps` nor
  `Int64ShaderOps`.
- Covers normals, binary32 subnormals (normalized to exact binary64 normals),
  signed zero (`±0.0`), `±Inf`, and `sNaN`/`qNaN` quieting (`| 0x00400000` with
  payload preserved in `hi[19:0]` and `lo[31:29]`).
- Supports VOP1 and VOP3A encodings with VGPR, SGPR, inline-float (`240..248`),
  and 32-bit literal sources, VOP3A `abs`/`neg`, and overlapping `D[0] == S0`.
- Refuses `pixel_stage`, `graphics_input`, `clamp`, `omod`, `op_sel`,
  `subword`, `dpp`, unused VOP3 sources/modifiers, and `dst == v255`.

## Verification

```powershell
cmake --build --preset dev --target native_compute_ir_dump compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_cvt_f64_f32_' -V
```

- `native_cvt_f64_f32_ir`: evaluates all 32 lane words on CPU against exact
  IEEE-754 widening, asserts `dx.op.unaryBits.i32(33)` and no `double`/`i64`,
  and verifies 7 refusals.
- `native_cvt_f64_f32_controls`: compiles to DXIL, asserts no `double`/`i64`
  or `Double`/`Int64` capability flags, and verifies 7 refusals and 4
  negative-control reference mutants.
- `native_cvt_f64_f32_hardware`: dispatches two wave32 thread groups and
  verifies all 2064 readback words (returns 77 if SM 6.6 / wave32 is
  unavailable).

## Limits

Validates synthetic compute shaders through the direct-DXIL path on desktop
D3D12 hardware; does not establish full-game startup or console equivalence.
