# Startup packed binary16 min/max and arithmetic sources

The startup translator lowers `v_pk_min_f16`, `v_pk_max_f16`,
`v_pk_add_f16`, `v_pk_mul_f16` and `v_pk_fma_f16` on gfx1013.
Each source is a single VGPR, numeric SGPR, 32-bit hex literal, or inline
integer zero. The inputs are packed words. `op_sel` selects each source's
half for the result's low half and `op_sel_hi` selects each source's half for
the result's high half. The defaults select the low and high halves
respectively. A short hex literal has zero in its high half. SGPR sources
must hold numeric data. Inline floats, nonzero inline integers and modifiers
other than the two selectors are refused.

The min/max helper strips the binary16 sign and compares the remaining encoding
against infinity to identify NaNs. One NaN yields the other operand's
encoding. Two NaNs yield the chosen canonical quiet NaN `0x7e00`. The
two-NaN payload choice has no hardware confirmation. Numeric comparison uses
ordered integer keys and returns the original bits. In either operand order,
min chooses negative zero and max chooses positive zero when the signs
differ. Equal zeros retain their sign.

Add, multiply and fused multiply-add convert the selected halves with
`f16tof32` and round each result to binary16 with `f16rn`.

The Python controls check emitted HLSL and operand refusals. The WARP and
hardware draw modes use a source-built gfx1013 fixture and compare both halves
with an independent CPU reference, including signed zeros, NaNs, infinity,
subnormals and literal and SGPR selector flips. Each arithmetic draw checks
literal and SGPR inputs with default and flipped selectors, including the
literal's zero high half and results that round up in binary16. The reference
uses Python's `struct` binary16 format for round-to-nearest-even.

```sh
python -B tests/startup_gfx2hlsl_pk_minmax_check.py
ctest --preset dev --no-tests=error -R '^startup_gfx2hlsl_pk_minmax'
```
