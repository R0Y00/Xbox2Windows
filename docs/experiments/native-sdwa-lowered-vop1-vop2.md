# Remaining lowered VOP1/VOP2 SDWA profiles in direct compute

Issue #3303 admits exactly these already-lowered direct-compute SDWA profiles:

| Class | Profiles |
|---|---|
| Integer/bit | `v_sub_co_ci_u32`, `v_subrev_co_ci_u32`, `v_mul_i32_i24`, `v_mul_hi_i32_i24`, `v_mul_hi_u32_u24`, `v_ffbh_u32`, `v_ffbl_b32`, `v_ffbh_i32`, `v_sat_pk_u8_i16` |
| Floating point | `v_sin_f32`, `v_cos_f32`, `v_mul_legacy_f32` |

The change is limited to the direct compute SDWA profile gate. Graphics SDWA decode still rejects all twelve
profiles. Existing source selection, source modifiers, carry handling, arithmetic, and destination placement are
reused; the tests check generated compute IR and DXIL, malformed and unsupported forms, graphics refusals, and
GPU readback with untouched-word guards. The hardware test reports capability skip 77 when the adapter cannot
run the fixture.

From a configured Windows `dev` build:

```powershell
cmake --preset dev
cmake --build --preset dev --target compile_dxil native_compute_ir_dump native_pixel_ir_dump native_store_replay
ctest --preset dev --no-tests=error -R '^native_sdwa_profiles_(ir|dxil|hardware)$' -V -j 1
```

IR and DXIL checks do not require a physical adapter. Record hardware results only when the gated GPU test
actually runs; an unavailable adapter is a skip, not a pass.
