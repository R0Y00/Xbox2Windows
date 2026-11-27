# First `image_bvh_intersect_ray` slice: FP32 box nodes

`tools/startup/gfx2hlsl.py` now lowers the 32-bit `image_bvh_intersect_ray` (RDNA 2 MIMG opcode 230) for
one node form: the 128-byte FP32 box node, kind 5. The result is four child pointers, with `0xffffffff` for a
disabled child or a missed box. The helper performs the four slab tests, clips against `ray_extent`, and sorts
hits by entry distance only when bit 31 of descriptor DWORD 1 is set. With sorting disabled each
result remains in its original child slot, including holes. Sorted misses use a positive-infinity
key so that a valid hit at `FLT_MAX` still precedes them.

The node layout is the one in Mesa 25.1.0 `src/amd/vulkan/bvh/bvh.h` (MIT): `children[4]` at bytes 0..15,
`coords[4][2][3]` at 16..111, and four reserved words at 112..127. The slab test follows Mesa
`radv_nir_rt_common.c` (MIT): inverse direction clamped to finite extremes, `tmin`/`tmax` from the three axes,
an inactive box whose `min.x` is NaN, `tmax >= max(0, tmin)`, `tmin < ray_extent`, and the five compare-exchange
stages that sort the four children. A child word of `0xffffffff` is a miss even when its stored box would hit.
The BVH T# fields are RDNA 2 ISA Table 49: base 39:0 (encoded in 256-byte units), reserved 54:40,
box growing 62:55, box sorting 63, size 105:64, triangle return mode 120, big page 123 and type 8 at 127:124.

## Accepted slice

- compute stage only;
- `image_bvh_intersect_ray v[4], v[11], s[4]` with the packed address form (the disassembler prints `v[16:31]`
  for the eleven consumed dwords);
- a constant inline T# built by four SGPR constants, type 8, nonzero encoded base, box growing zero,
  size at least one 64-byte unit and below 4 GiB, and the node inside that size;
- a constant node pointer whose low three bits are 5, 128-byte aligned and inside the 4 GiB relative
  address window; the entire node must fit within the 48-bit absolute address space;
- either value of the sort bit.

The manifest names the node allocation with `srv <i> bvh base 0x<base> size 0x<size>`, using the decoded
byte address for `base` and the unchanged descriptor size field for `size`. All four original descriptor
DWORDs are passed to the helper. For example, base field `0x1000` means byte address `0x100000`, and
`0x1008` means `0x100800`; both are aligned by construction. The host must bind a raw
`ByteAddressBuffer` at the decoded base covering `(size + 1) * 64` bytes. That host binding is not part
of this change.

## Refused by name

Triangle nodes (kind 0), 16-bit box nodes (kind 4), every other node kind, `image_bvh64_intersect_ray`, A16,
non-sequential addresses, a pixel or vertex stage, a node pointer that is not a straight-line constant, a T#
loaded at run time, a type other than 8, nonzero box growing, an absolute (zero) base, a size that
does not cover the node, size bits above 4 GiB, a node address that wraps the 48-bit address space,
and reserved T# fields. The checks assert that a refusal emits no helper call and no destination
VGPR assignment.

## Evidence

```sh
python3 -B tests/gfx1030_bvh_box/check.py --model-only
python3 -B tests/startup_gfx2hlsl_bvh_check.py
```

The translator check compares synthetic nodes and rays against the independently existing binary32
reference in `tests/gfx1030_bvh_box/bvh_model.py`. It checks the sort-bit negative control (word 0
changes from `0x1d` to `0x25`), the three-hit sort-disabled result with its second slot still invalid
(`0x25, 0xffffffff, 0x2d, 0x35`), and the finite `FLT_MAX` hit with infinite extent. It separately checks
the translation, full descriptor literals, and manifest for `0x1000`, `0x1008`, and a base using
the high byte of DWORD 1. The top-of-address-space witness checks a valid node ending at the
48-bit boundary and refuses a node whose start wraps to zero in the independent model, before
binding a resource or emitting a helper. The check also covers the named refusals and the actual
emitted helper's mutation sites. It compiles the translated compute shader with DXC when available;
without DXC it prints `dxc compile: skip` and does not count that compilation as a pass.

```text
python -B tests/startup_gfx2hlsl_bvh_check.py dispatch <dxc> <offscreen_check> warp
```

The dispatch mode compiles the same emitted helper with a `RWByteAddressBuffer` in place of the
bound `ByteAddressBuffer`, writes synthetic nodes and rays, and compares the four result DWORDs
for every case. Its negative controls remove miss filling, compact the unsorted slots, substitute
`FLT_MAX` for the infinite miss key, reintroduce the erroneous low-byte alignment check, and ignore
the high byte of the encoded base in the zero-base check. Each mutant must change its designated
witness result. This is a reproduction procedure, not a new GPU execution result.

This is a translator slice. It does not execute the native gfx1030 probe, does not establish the
host binding of the node allocation on a D3D12 device, and does not establish any title behavior.
