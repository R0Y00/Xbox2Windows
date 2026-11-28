# Constant static samplers on the in-process compute route (#3149)

With `X2W_NATIVE_SHADERS` set, the stand-in translates a slot 67 compute request with the C++ compiler at its
first Dispatch ([route](startup-compute-native-dispatch.md)). Until this change it refused every program that
samples. The compiler already resolves a sampler whose four words are constants into a `ComputeSampler`, and
`translate_compute_natively` (`src/startup_stubs/d3d12_compute_dispatch.inl`) now turns those into the static
samplers of the translated root signature, in register order.

## What is bound

- A comparison sampler (`comparison`) is bound as a comparison static sampler, as an `smpc` manifest line is.
- Any other sampler with `explicit_lod` carries the descriptor's MIN_LOD and MAX_LOD, as an `smpl` line does.
  The direct compute path sets `explicit_lod` on every sampler it emits (`sample || implicit` in
  `src/gfx1013_direct_compute.inl`, and the gathers through `gather_base_level`), so this route produces no
  plain `smp` sampler. The conversion keeps that case, but no compute program reaches it.
- The texture of a comparison sample (`comparison_texture_srv`, 2D) is bound like a plain 2D texture input.
- A natively bound 2D texture input sets the slot's input dimension. Without it the dispatch path takes the
  input for a described 1D, 2D-array or 3D view and refuses the Dispatch with "an input texture view is not 3D",
  so no natively translated program could read a 2D texture.
- A failed native translation clears its samplers, so a published `<key>.txt` read next numbers its sampler
  lines from 0.

The route has no refusal that names a sampler. Sampler words that are neither constants nor one entry of the
sampler heap are refused by the compiler ("nonconstant sampler") before the stand-in binds anything. A
fixed-index sampler of the directly indexed sampler heap (#3140) reaches the route as a `heap_sampler` view,
which is not bound and is refused by the view-kind message ("view kind N is not bound for a natively
translated program"). That is read from the code, not run here; binding that heap is the follow-up of #3076.
The capacity refusal now reads "more entry values, inputs or outputs than the dispatch path binds".

## Fixture

Step 5 of [`startup_compute_dispatch_native_check.py`](../../tests/startup_compute_dispatch_native_check.py),
source-built. The consumer mode `texture-volume-native-sampled` is the 3D float store of `texture-volume` plus a
2x1 R32_FLOAT texture with a 1x1 second mip, whose three texels hold different sentinel words. One program samples it four times and
stores the four results as the channels of a 3D float texture:

| Channel | Instruction | Sampler words | What it selects |
|---|---|---|---|
| 0 | `image_sample_lz` | wrap U, MAX_LOD 15.996 | mip 0, wrapped texel |
| 1 | `image_sample_l`, LOD 1.25 | mirror U, MIN_LOD 0, MAX_LOD 0 | mip 0 (the clamp), mirrored texel |
| 2 | `image_sample_l`, LOD 0 | clamp U, MIN_LOD 1, MAX_LOD 1 | mip 1 (the clamp) |
| 3 | `image_sample_c_lz` | clamp, LESS, point | 1.0 where the reference is below the texel, else 0.0 |

U is 0.3 and the reference -0.1875, each doubled once per global X. The eight threads of a row therefore read
both texels under wrap and mirror in different threads, and the comparison passes in six threads and fails in
two. All filters are point filters, so the result is exact. Every one of the 6,144 volume words is compared
with a CPU model that decodes the address mode, the LOD limits and the comparison from the sampler words.

Control: the same program with the first sampler's words loaded from the descriptor table
(`s_load_dwordx4 s[40:43], s[12:13], 0x60`) is refused with "nonconstant sampler", the Dispatch is refused with
"no translation", and every volume word keeps its sentinel.

## Command and result

```powershell
.\tools\dev.ps1
cmake --build --preset dev --target d3d12_xs_shim startup_stream_consumer compile_dxil
ctest --preset dev --no-tests=error -V -R '^(startup_compute_dispatch_native_(hardware|warp)|native_image_store_exact_width_hardware_(r|rg|rgb|array|volume|rgba|sparse)|startup_compute_dispatch_texture_hardware|startup_compute_descriptor_scc_hardware)$'
```

Commit `af6045df` on `main` at `f5a6f6ba`, Windows 11 build 26300.9974, Debug build, MSVC 14.44.35207, Windows
SDK 10.0.26100 DXC, D3D12 debug layer enabled. 11 tests selected: 10 passed, 1 capability skip, 0 failed.

| Test | Adapter, driver | Result |
|---|---|---|
| `startup_compute_dispatch_native_hardware` | NVIDIA GeForce RTX 4090, 32.0.16.1088 | PASS. The four sampled channels match the CPU model in all 6,144 volume words, the memory sampler control leaves the volume untouched, and the other steps of the check pass. No debug-layer error message. |
| `native_image_store_exact_width_hardware_{r,rg,rgb,array,volume,rgba,sparse}` | the same | PASS (7 tests). These dispatch exact-width outputs through the same binding loop. |
| `startup_compute_dispatch_texture_hardware` | the same | PASS. It shares the consumer's texture modes. |
| `startup_compute_descriptor_scc_hardware` | the same | PASS. |

Capability skip, not a pass: `startup_compute_dispatch_native_warp` exits 77 on the Microsoft Basic Render
Driver 10.0.26100.9818, which reports no 32-lane waves. Before it exits, the check confirms that the sampled
program is refused by that capability reason with the volume untouched, and that the memory sampler control is
refused with "nonconstant sampler".

Earlier revisions of this change, stacked on #3049, gave the same result for the sampled step on `main` at
`429f11d5`, `7fcde81f` and `ee3347d3`.

## Mutations

Each mutation was applied to `src/startup_stubs/d3d12_compute_dispatch.inl` at `af6045df` on its own, the
stand-in rebuilt, `startup_compute_dispatch_native_hardware` run, and the file restored.

| Mutation | Result |
|---|---|
| The file as on `main` (`f5a6f6ba`) | FAIL at the sampled program: `not translated natively: view kind 11 is not bound for a natively translated program`. The comparison texture view is refused before the sampler check is reached. |
| Only the sampler check restored; the comparison view stays accepted | FAIL at the sampled program: `not translated natively: a sampler, or more entry values, inputs or outputs than the dispatch path binds`. |
| The 2D input dimension left unset | FAIL: the program translates, then `Dispatch(2, 1, 1) refused: an input texture view is not 3D`. |
| `lod_limits` always false | FAIL: `sampled: word 2049 got=bf000080 expected=bf000000 (32 words differ)`. Channels 1 and 2 of all 16 threads read the other mip. |
| `comparison` always false | FAIL: `sampled: word 2051 got=bf000080 expected=00000000 (16 words differ)`. Channel 3 of all 16 threads holds a texel, not a comparison result. |
| Samplers bound in reverse register order | FAIL: `sampled: word 2049 got=bf000080 expected=bf000000 (56 words differ)`. |
| Restored | PASS. |

The comparison and order mutations were caught by the readback, not by a debug-layer message.

## Limits

- Constant sampler words only. A sampler read from memory, including the sampler heap, stays refused (#3076).
- One 2D texture format was sampled (R32_FLOAT), with point filters. Linear and anisotropic filters, border
  colors and 3D or array textures were not run on this route.
- A plain sampler without LOD limits was not run: the direct compute path does not emit one.
- A comparison sampler is bound without the LOD limits, as on the published route. The compute path compares at
  level zero and the compiler requires MIN_LOD 0, so the limits cannot change the result.
- Clearing the samplers after a failed native translation follows from the code (the manifest reader requires
  sampler line `i` to be the next entry). No run exercises it.
- One adapter, in a Debug build. No start-up run used the switch.
