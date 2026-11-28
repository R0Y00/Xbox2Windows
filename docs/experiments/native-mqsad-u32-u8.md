# Native direct-DXIL compute: `v_mqsad_u32_u8`

Refs #2935, #3361.

## Scope

Lowers VOP3A `v_mqsad_u32_u8` (`0x175`) in the native direct-DXIL compute path
([gfx1013_direct_compute.inl](../../src/gfx1013_direct_compute.inl)) and
registers its 128-bit VGPR quad destination (`v[d:d+3]`) in `clear_vector_spills`
([gfx1013_compute_native.cpp](../../src/gfx1013_compute_native.cpp)):

- Snapshots 64-bit `S0` (8 unsigned bytes from a VGPR pair or even SGPR pair),
  32-bit `S1` (4 reference bytes), and 128-bit VGPR quad `S2` (4 `uint32`
  accumulators) before writing `v[d:d+3]` (supporting overlapping `D == S2`).
- Computes 4 `uint32` lanes `i = 0..3`:
  `acc[i] + sum_{k=0..3} ((r[k] == 0) ? 0 : |b[i+k] - r[k]|)` using pure 32-bit
  integer arithmetic (no `Int64ShaderOps` required).
- Supports VOP3 `clamp` (exact `uint32` saturation to `0xFFFFFFFF` via
  `sum < acc`) and modulo-`2^32` wrap when unclamped.
- Refuses `pixel_stage`, `graphics_input`, `abs`, `neg`, `omod`, `op_sel`,
  `subword`, `dpp`, non-pair `S0`, non-VGPR-quad `S2`/`D`, and inline-float `S1`.

## Verification

```powershell
cmake --build --preset dev --target native_compute_ir_dump compile_dxil native_store_replay
ctest --preset dev --no-tests=error -R '^native_mqsad_u32_u8_' -V
```

- `native_mqsad_u32_u8_ir`: evaluates all 32 lane words on CPU against the
  Python model (VGPR/SGPR pair `S0`, overlapping `D == S2`, zero reference
  bytes, wrap vs clamp) and verifies 9 refusals and no `i64` ops.
- `native_mqsad_u32_u8_controls`: compiles to DXIL, asserts no `i64`/`Int64`,
  and verifies 9 refusals and 4 negative-control reference mutants.
- `native_mqsad_u32_u8_hardware`: dispatches two wave32 thread groups and
  verifies all 2064 readback words (returns 77 if SM 6.6 / wave32 is
  unavailable).

## Limits

Validates synthetic compute shaders through the direct-DXIL path on desktop
D3D12 hardware; does not establish full-game startup or console equivalence.
