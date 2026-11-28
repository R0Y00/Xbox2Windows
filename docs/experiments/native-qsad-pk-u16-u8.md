# Native direct-DXIL compute: `v_qsad_pk_u16_u8` and `v_mqsad_pk_u16_u8`

Refs #2935, #3359.

## Scope

Lowers VOP3A `v_qsad_pk_u16_u8` (`0x172`) and `v_mqsad_pk_u16_u8` (`0x173`) in
the native direct-DXIL compute path ([gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl))
and registers both 64-bit VGPR pair destinations in `clear_vector_spills`
([gfx1013_compute_native.cpp](../../src/gfx1013_compute_native.cpp)):

- Reads 64-bit `S0` (8 unsigned bytes `b[0..7]` from a VGPR pair or even SGPR
  pair), 32-bit `S1` (4 unsigned reference bytes `r[0..3]`), and 64-bit `S2`
  (4 packed `uint16` accumulators from a VGPR pair or even SGPR pair),
  snapshotting all inputs before writing `v[d:d+1]`.
- Computes 4 packed `uint16` lanes `i = 0..3`:
  `acc[i] + sum_{k=0..3} term(b[i+k], r[k])`, where `v_mqsad_pk_u16_u8` skips
  terms with `r[k] == 0`.
- Pure 32-bit integer arithmetic (no `Int64ShaderOps` required). Supports VOP3
  `clamp` (saturating each `uint16` lane to `0xFFFF`) and modulo-`2^16` wrap
  when unclamped.
- Refuses `pixel_stage`, `graphics_input`, `abs`, `neg`, `omod`, `op_sel`,
  `subword`, `dpp`, non-pair `S0`/`S2`, `dst == v255`, and inline-float `S1`.

## Verification

```powershell
cmake --build --preset dev --target native_compute_ir_dump compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_qsad_pk_u16_u8_' -V
```

- `native_qsad_pk_u16_u8_ir`: evaluates the emitted IR on CPU against the
  Python model for all 32 lane words (VGPR/SGPR pair `S0`, overlapping
  `D == S2`, zero reference bytes, wrap vs clamp) and verifies 8 refusals.
- `native_qsad_pk_u16_u8_controls`: compiles to DXIL, asserts no `i64` ops,
  verifies 8 refusals and 4 negative-control reference mutants.
- `native_qsad_pk_u16_u8_hardware`: dispatches two wave32 thread groups and
  verifies all 2064 readback words (returns 77 if SM 6.6 / wave32 is
  unavailable).

## Limits

Validates synthetic compute shaders through the direct-DXIL path on desktop
D3D12 hardware; does not establish full-game startup or console equivalence.
