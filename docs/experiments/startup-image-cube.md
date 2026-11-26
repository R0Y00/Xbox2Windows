# Cube image sampling in gfx2hlsl (#2943)

`tools/startup/gfx2hlsl.py` lowers `image_sample`, `image_sample_lz`, `image_sample_l` and `image_sample_b` with
`dim:SQ_RSRC_IMG_CUBE` to `TextureCube<float4>` `Sample`, `SampleLevel` and `SampleBias` (outside a pixel stage the
implicit-LOD form samples level 0 and the bias form is refused, as for 2D textures), and
writes the manifest kind `texcube` for the binding. Before, the stage was refused with
`image dimension SQ_RSRC_IMG_CUBE`.

## The address

On gfx10 a cube sample takes three address dwords after any bias: the face coordinates `s` and `t` and the face
id. Mesa's `prepare_cube_coords` (`src/amd/common/nir/ac_nir_lower_tex.c`, read at Mesa 25.0.7 through the
`openbsd/xenocara` mirror, commit `d31967950e30`) builds `s = sc / |ma| + 1.5` and `t = tc / |ma| + 1.5` from
`v_cubesc`, `v_cubetc` and `v_cubema`, and a cube-array id as `8 * layer + face`. LLVM 15.0.7
`ConstantFoldAMDGCNCubeIntrinsic` (`llvm/lib/Analysis/ConstantFolding.cpp`) defines `v_cubema` as twice the signed
major coordinate, so `s` and `t` lie in [1, 2] and `s - 1`, `t - 1` are the position on the face.

The translator emits a helper `x2w_cube_dir(s, t, face)` that rebuilds a direction whose major component is +1 or
-1 for the face and whose other components are `u = 2(s - 1.5)` and `v = 2(t - 1.5)` with the signs of the per-face
`sc` and `tc` that the existing `v_cube*` lowering uses: +X (1, -v, -u), -X (-1, -v, u), +Y (u, 1, v), -Y (u, -1, -v),
+Z (u, -v, 1), -Z (-u, -v, -1). Projecting that direction again gives back `u` and `v`.

## Refused

- `image_load` and `image_load_mip` on a cube (`Load is not defined on a TextureCube`), `image_sample_d`
  (cube derivatives are not converted), `image_get_lod`, `a16` and `d16`.
- A cube in a compute program: the compute dispatch has no `texcube` view binding.
- A constant sampler with `DISABLE_CUBE_WRAP` (`SQ_IMG_SAMP_WORD0` bit 28, Mesa
  `src/amd/registers/gfx10-rsrc.json`): the guest then filters without crossing face edges, while a D3D12 cube
  always filters across them. With the bit clear both cross.
- Compare, gather and store forms were already limited to 2D and 2D arrays and are unchanged.

## Limits

- A cube-array view is not recognisable from the instruction (gfx10 MIMG has no array bit; the descriptor sets
  the dimension). The helper keeps the face bits of the id (`id & 7`) and samples layer 0 of a `TextureCube`,
  which is wrong for a later layer of a cube array.
- A face id of 6 or 7 maps to face 5, a negative, NaN or huge id to face 0, and `s` or `t` outside [1, 2] are
  clamped to the face edge. Mesa does not produce these for a finite non-zero direction; none of them is measured.
- The implicit-LOD and bias forms take the desktop derivatives of the rebuilt direction. Inside one face these
  follow the guest's face coordinates; a quad that straddles a face edge can choose another level.
- Heap and register samplers are not read at translation, so their `DISABLE_CUBE_WRAP` bit is not validated (a set
  bit would filter differently at face edges), and their LOD range stays unvalidated as for 2D textures.
- Only pixel-stage draws are measured; the vertex-stage form is translated but not drawn.
- The runtime binding of a `texcube` view is PR #2938. Without it the main runtime parses `texcube` as an unknown
  kind and would bind the slot as a buffer, so this lowering must not be merged before #2938.

## Fixture

`tests/startup_image_cube_check.py` (controls, `binary <llvm-mc>`, `draw <dxc> <offscreen_check> <warp|hardware>`).
`offscreen_check --startup-image-cube` binds six 2x2 R32G32B32A32_FLOAT faces as a TextureCube whose texel (x, y)
of face f holds ((1 + x + 2y + 4f) / 32, x, y, f). The sampler is the all-zero inline sampler (point, wrap), and
every `s` and `t` is a texel centre.

- Seven programs set `(s, t, face)` as constants, without `v_cube*`, so the reference is only the address meaning
  (texel x = 0 below `s = 1.5`, else 1, the same for `t`). The desktop driver picks the face of the rebuilt
  direction itself, so a wrong face, sign or axis in the helper reads another texel. Every face has a case whose
  texel has x != y.
- `image_sample`, `image_sample_l` and `image_sample_b` each draw one case.
- Controls: the face one higher and `s`, `t` exchanged must read those texels; `image_sample_l` and
  `image_sample_b` with the face in the LOD or bias dword must read face 0.
- Thirteen translator mutants edit the emitted helper (for each face the sign of the `u` term flipped, `u` and `v`
  exchanged, and faces 2 and 3 exchanged). Each must read the texel a D3D cube lookup of its direction gives.
- Six programs run the `v_cube` chain (`v_cubema`, `v_cubesc`, `v_cubetc`, `v_cubeid`, `v_rcp_f32` of `|ma|` and
  `v_fma_f32` with 1.5) on one direction per face. Each must match the face texel LLVM's table gives and a desktop
  pixel shader that samples the same direction from the TextureCube.

Measured on 27.11.2026, commit 6ee878d8 plus this change, Windows 11 24H2 (10.0.26100.9711), debug layer
unavailable on both adapters: the 22 reference draws (7 literal, 3 forms, 6 chain, 6 desktop TextureCube) matched
on NVIDIA Quadro M1000M (driver 32.0.15.7371) and on WARP (Microsoft Basic Render Driver 10.0.26100.9711), and all
17 controls and mutants read the texel they predict. With the translator change reverted, the controls fail on
`image dimension SQ_RSRC_IMG_CUBE`.
The 16 fixture programs reassemble to the recorded words with LLVM 15.0.7.

```powershell
ctest --preset dev --no-tests=error -R '^startup_image_cube_' -V
```

Not shown: that the console's face order and face orientation equal D3D's. That rests on LLVM's cube table being
the D3D cube face table. LOD and bias values on a cube are not tested, because only one mip is bound; the
`image_sample_l` and `image_sample_b` draws only show that the extra address dword is not read as a coordinate.
