# Startup compute FP32 and packed D16 array UAV binding

Refs [#2742](https://github.com/R0Y00/Xbox2Windows/issues/2742).
Depends on the full-mask FP32 compute array image store lowering in
[#2744](https://github.com/R0Y00/Xbox2Windows/pull/2744) and the production
binding in [#2777](https://github.com/R0Y00/Xbox2Windows/pull/2777). Packed
D16 compute stores reuse that binding and the existing 2D two-VGPR lowering;
no new manifest or runtime path is needed. Pixel array stores remain refused.

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

The packed D16 case uses the same 32x8 six-slice allocation with
RGBA16_FLOAT and a 256-byte row pitch. Packed VGPRs 0xc0003c00 and
0x7bff3800 produce half encodings 0x3c00, 0xc000, 0x3800 and 0x7bff
(R/G/B/A). The reference compares all 6,144 raw halfwords after the fence.
For each relative layer (1 and 3), full and partial EXEC must have no
mismatches. The wrong-layer, dropped-EXEC, swapped first two halves, swapped
VGPRs and omitted-store controls must differ by 128, 32, 32, 64 and 64
halfwords respectively; each mutated output is also compared with its own
CPU reference. An incompatible RGBA16_FLOAT 2D view must be refused with
1,024 halfwords unchanged. Before translation, all six slices must remain
unchanged. The source-built instruction sequences for both layers and EXEC
variants are checked against LLVM 15 with the D16 translation controls.

Reproduce on Windows (source-built fixtures only):

```powershell
cmake --build --preset dev --target startup_stream_consumer offscreen_check
ctest --preset dev --no-tests=error -j 1 -R '^startup_compute_dispatch_array(_d16)?_(warp|hardware)$' -V
```

No game input, retail program coverage, title execution or guest
out-of-bounds behavior is established.
