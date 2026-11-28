# Published typed buffer inputs (`srv <i> typed`)

Part of #3224. The direct compiler reads `buffer_load_format_x` to `_xyzw` (MUBUF, element index only) as a
`Buffer<uint4>` (`typed_buffer_srv`). `format_compute_runtime_manifest` now publishes it as
`srv <register> typed table <parameter> <byte> bytes <n>`, where `n` is 4 bytes per component of the widest load
of that binding.

The stand-in (`src/startup_stubs/d3d12_compute_dispatch.inl`) resolves the table entry with the existing input
range rules and binds a UINT buffer view of the title's 32-bit components: `R32_UINT` to `R32G32B32A32_UINT`. A
UINT view returns the same bits as the guest format unit only for 32-bit UINT, SINT and FLOAT components, so only
those are bound:

- **Hardware descriptor:** FORMAT 20-22, 62-64, 72-74 or 75-77 (gfx10 `BUF_FMT_32*`), stride equal to the
  element size, identity DST_SEL over the components read, unified FORMAT (RESOURCE_LEVEL 1), structured bounds
  (OOB_SELECT 0), and no swizzle, index stride or ADD_TID.
- **Slot 18 view token:** a typed buffer view with 32-bit channels and the default component mapping, read through
  the UINT format of the same width (`typed_dword_channels` / `typed_dword_uint_format` in
  `src/startup_stubs/native_buffer_stride.h`).
- **All four words zero:** a null buffer that reads zeros, as for raw inputs.

Each other shape is refused before a dispatch is recorded, with one of these reasons:

- `a typed input FORMAT is not 32 bits per channel`
- `a typed input stride is not its element size`
- `a typed input DST_SEL is not the identity`
- `a typed input descriptor is swizzled or not bounds-checked by record`
- `a typed input has fewer channels than the program reads`
- `a typed input token is not a typed buffer view with 32-bit channels`
- `a typed input range does not start at a whole element`

The range checks of raw inputs also apply. A typed input on an indirect Dispatch is refused, and the in-process
`X2W_NATIVE_SHADERS` route is unchanged. 8- and 16-bit formats, UNORM, SNORM and 16-bit FLOAT need a conversion
that a UINT view does not do.

## Test

`tests/startup_compute_typed_buffer_srv_check.py` publishes a source-built program for 1 to 4 components with
`compile_dxil --requests`. It then dispatches it through slot 14 with a 32-bit FLOAT descriptor of 50 elements.
The 384 lanes store what they read, so the lanes past element 49 must read zeros. The 4-component case also reads
through a slot 18 `R32G32B32A32_FLOAT` view token (consumer mode `dispatch-srv-token-typed`). Each invalid shape
above must be refused with the output untouched. Before publication the dispatch reports `no translation`.

```powershell
ctest --preset dev --no-tests=error -R '^startup_compute_typed_buffer_srv_' -V
```

Measured at `d418f70e` plus this change, on Windows 11 build 26300 with the Windows SDK 10.0.26100 DXC, Debug,
with the D3D12 debug layer enabled:

- **NVIDIA GeForce RTX 4090, driver 32.0.16.1088:** passed for 1, 2, 3 and 4 components, the token, and 4, 6, 6
  and 7 refusal controls.
- **WARP 10.0.26100.9818:** capability skip (77). WARP reports 4-lane waves and the direct module needs 32.
- **Negative control:** with the runtime's parse of the `typed` line disabled, the published manifest is not
  bound and the hardware check reads the sentinel at word 0.

Not measured: how many of the cohort's typed inputs have a 32-bit FORMAT. The census re-measure answers that.
