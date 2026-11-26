# Startup compute FP32 array UAV binding

Refs [#2742](https://github.com/R0Y00/Xbox2Windows/issues/2742).
Depends on the full-mask FP32 compute array image store lowering in
[#2744](https://github.com/R0Y00/Xbox2Windows/pull/2744). Pixel and packed-half
array stores remain refused.

The production startup compute manifest consumer accepts
`uav N tex2da float table P O channels 4`. It requires an explicitly described
2D array UAV with a valid single-plane mip and slice range, retains the caller's
FirstArraySlice and ArraySize and refuses a plain 2D UAV before Dispatch.
The existing 2D and raw buffer paths are unchanged. Shader layer indices are
view-relative and are not clamped.

The fixture publishes the encoded program through slot 67, translates it with
DXC, and binds a 32x8 RGBA32_FLOAT resource through slot 19. Its six physical
slices have distinct sentinels; the UAV selects four slices starting at
FirstArraySlice=1. Relative layers 1 and 3 write physical slices 2 and 4.
The actual translated dispatch goes through slot 14. After the fence, compare
all 6,144 words, including every adjacent and out-of-view slice. Full and
partial EXEC are checked for both layers. The wrong-layer and dropped-EXEC
controls must differ by 128 and 32 words respectively. A mismatched 2D view
must be refused with all 1,024 words untouched.

Reproduce on Windows (source-built fixtures only):

```powershell
cmake --build --preset dev --target startup_stream_consumer offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^startup_compute_dispatch_array_(warp|hardware)$' -V
```

No game input, retail program coverage, title execution or guest
out-of-bounds behavior is established.
