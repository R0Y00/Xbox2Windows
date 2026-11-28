# Native direct-DXIL compute: `v_cvt_f64_i32` and `v_cvt_f64_u32`

Refs #2935, #3363.

## Scope

Lowers VOP1/VOP3A `v_cvt_f64_i32` (`0x04` / `0x184`) and `v_cvt_f64_u32`
(`0x16` / `0x196`) in the native direct-DXIL compute path
([gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl)) and
registers both 64-bit VGPR pair destinations in `clear_vector_spills`
([gfx1013_compute_native.cpp](../../src/gfx1013_compute_native.cpp)):

- Constructs the exact IEEE-754 binary64 word pair `v[d:d+1]` from a 32-bit
  signed or unsigned integer using pure 32-bit integer arithmetic and
  `dx.op.unaryBits.i32(33)` (`FirstbitHi`). Requires neither
  `DoublePrecisionFloatShaderOps` nor `Int64ShaderOps`.
- Supports VOP1 and VOP3A encodings with VGPR, SGPR, inline-integer, and
  32-bit literal sources, including overlapping `D[0] == S0`.
- Refuses `pixel_stage`, `graphics_input`, `abs`, `neg`, `clamp`, `omod`,
  `op_sel`, `subword`, `dpp`, unused VOP3 `src1`/`src2`, `dst == v255`, and
  inline-float sources.

## Verification

```powershell
cmake --build --preset dev --target native_compute_ir_dump compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_cvt_f64_int_' -V
```

- `native_cvt_f64_int_ir`: evaluates all 32 lane words on CPU against
  `struct.pack('<d', float(x))`, asserts `dx.op.unaryBits.i32(33)` and no
  `double`/`i64` ops, and verifies 8 refusals.
- `native_cvt_f64_int_controls`: compiles to DXIL, asserts no `double`/`i64`
  or `Double`/`Int64` capability flags, and verifies 8 refusals and 3
  negative-control reference mutants.
- `native_cvt_f64_int_hardware`: dispatches two wave32 thread groups and
  verifies all 2064 readback words (returns 77 if SM 6.6 / wave32 is
  unavailable).

## Limits

Validates synthetic compute shaders through the direct-DXIL path on desktop
D3D12 hardware; does not establish full-game startup or console equivalence.
