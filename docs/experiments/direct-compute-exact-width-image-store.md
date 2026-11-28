# Exact-width image stores in direct compute

The direct gfx1013 compute translator lowers non-D16 `image_store` with
`dmask:0x1`, `0x3`, or `0x7` on 2D, 2D array, and 3D images. It consumes
one, two, or three consecutive VDATA VGPRs and declares a typed float,
float2, or float3 UAV. The DXIL store uses the full typed-store mask.
The narrower resource type determines which components reach the texture.
The `unorm` instruction modifier continues to leave the integer texel
coordinates unchanged.

Manifest version 2 describes these outputs with
`uav <index> <tex2d|tex2da|tex3d> float table <parameter> <byte> channels <1|2|3>`.
The in-process path gets the same width from the translated binding.
The runtime requires the view and resource formats to have the specified
component count and checks adapter support for typed stores on the resource
format. It creates the UAV with that format and retains the bound subresource
range. This makes each accepted store a whole texel write. Sparse masks still
receive the translator's descriptor modifier refusal.

The readback fixture dispatches two X groups of 4x2x1 threads into a 32x8 mip.
It explicitly sets EXEC to `0x55`, writing columns 0, 2, 4, and 6 in rows
0 and 1. The array and volume cases use relative slice 1 of a view that
starts at physical slice 1. Every other texel retains its sentinel.
The format cases use R32_FLOAT, R32G32_FLOAT, and R11G11B10_FLOAT. The
RGBA8 control leaves the complete image untouched and checks the runtime
channel-count refusal.

## Published one- and two-channel D16 stores

The direct compiler already emits scalar and two-channel D16 image stores.
The published manifest uses
`uav <index> <tex1d|tex2d|tex2da|tex3d> half table <parameter> <byte> channels <1|2>`.
The runtime requires an `R16_FLOAT` or `R16G16_FLOAT` UAV of the corresponding
typed or typeless resource and checks desktop typed-store support on the
view format. Dispatch retains the UAV's mip, first array slice or W slice,
and range.

The source-built fixture writes 1.0 and, for two channels, -2.0 to even
columns of mip 1. It checks 1D, 2D, 2D array and volume readbacks, including
an R16 typeless backing resource. The layered tests bind a view that starts
at physical slice 1 and write relative slice 1. All other pixels and slices
keep their original halfwords. The R32_FLOAT and R32G32_FLOAT controls
exercise the format refusal with the same channel counts.

The translated descriptor follows the [D3D12 UAV contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_unordered_access_view_desc),
including the [W range of 3D views](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_tex3d_uav).

```powershell
ctest --preset release --no-tests=error -R '^startup_compute_half_image_uav_' -V
```
