# GPU-written Mesh arguments and descriptor history

Refs [#3310](https://github.com/R0Y00/Xbox2Windows/issues/3310).
This adds a source-built reproducer and an opt-in descriptor diagnostic. It
does not establish the issue's matched integration acceptance or repair the
zero launch count observed there.

## What the fixture measures

`startup_mesh_argument_provenance_hardware` uses the existing
`startup_indirect_draw_indexed` executable in `mesh-provenance` mode. All shader
and buffer inputs come from repository source. No external capture, shader
cache, game content or fitted parameters are required.

The fixture follows this chain:

1. Create a raw buffer SRV in a CPU-only guest descriptor heap. Copy it to a
   shader-visible heap, overwrite the source SRV, then copy the updated view.
   The guest token and desktop shadow record have different identities.
2. A source-built Compute dispatch writes the scalar input: `2` at byte 16
   and `0` at byte 32. Copy those bytes to readback before the updater. After
   fence 1 completes, verify that this GPU write supplied the nonzero input.
3. A second Compute dispatch reads `launch.Load(0)` through the copied SRV
   and writes `(X,1,1)` into the argument buffer at byte 4. Copy the argument
   buffer to a separate readback allocation and complete fence 1.
4. In a separate submission on the same DIRECT queue, copy that buffer again
   immediately before the guest command-list slot 61 `ExecuteIndirect` call.
   Complete fence 2 before mapping the second snapshot. Report the exact
   consumed 12-byte interval, backing resource and byte offset.
5. Read back the scene-color target after fence 3. The positive case checks
   every pixel of the 4 by 2 RGBA8 target: left red, right green. No compositor
   participates. Check the argument buffer's leading, trailing and neighboring
   record guards, device-removal status and debug-layer errors too.

The positive oracle requires the same resource **and** offset, identical
12-byte records with X greater than zero, intact guards and all eight expected
pixels. Each negative control must fail that same oracle while satisfying its
own expected record and pixel results:

| Case | Post-write record at byte 4 | Pre-consume record | Changed pixels |
|---|---|---|---|
| `positive` | `(2,1,1)` | byte 4: `(2,1,1)` | 8/8 |
| `missing-copy` | `(0,1,1)` | byte 4: `(0,1,1)` | 0/8 |
| `stale-copy` | `(0,1,1)` | byte 4: `(0,1,1)` | 0/8 |
| `wrong-srv-offset` | `(0,1,1)` | byte 4: `(0,1,1)` | 0/8 |
| `wrong-argument-offset` | `(2,1,1)` | byte 20: `(0,1,1)` | 0/8 |

The missing-copy destination is explicitly initialized with a null SRV. The
stale-copy case copies the original view at byte 32, then overwrites the source
to byte 16 without copying again. The wrong-SRV case copies a view still
pointing at byte 32. The wrong-argument case consumes a valid zero record in
the same resource, so a same-resource-only comparison would miss the error.
The Python driver also repeats the positive case with history tracing disabled
and requires the same argument and pixel results without history lines.

## Runtime diagnostic

Set `X2W_D3D12_DESCRIPTOR_HISTORY=1` **before process startup**. The setting is
cached. Output follows the usual trap logger, stderr unless `X2W_TRAP_LOG` is
set. It emits `x2w-descriptor-history` lines for intercepted view creation into
tracked shadow slots and intercepted descriptor copies:

| Field | Meaning |
|---|---|
| `seq` | CPU shadow-publication order, serialized by `note_lock`; not GPU execution order |
| `op` | `create` or `copy`; a create with a different `old`/`new` ID is an overwrite |
| `guest`, `desktop`, `source` | Destination guest address, destination desktop handle, and source guest address for a copy |
| `old`, `new` | Shadow IDs before and after publication; 0 means unknown, otherwise the view-record index plus 1 |
| `slot` | Recorded device view-creation slot, for example 18 for SRV |
| `resource`, `backing` | Recorded guest resource identity and its unwrapped desktop backing resource |
| `first`, `stride` | Buffer SRV `FirstElement` and `StructureByteStride`; raw SRV byte offset is `first * 4` |
| `materialized` | Whether the recorded view was materialized on the desktop |
| `guest_bytes`, `guest_read`, `guest_before` | Requested token sample size, successful complete read flag, and up to eight hexadecimal DWORDs **before this operation writes the guest token** |

SRV tokens occupy 32 bytes. Sampler tokens use the shorter sampler stride; the
remaining printed words are zero padding. Only a sample with `guest_read=1`
can be interpreted. Guest token word 1 is a zero-based view-record index,
whereas `old`/`new` use one-based IDs. Native desktop descriptor bytes are
opaque and are not read as guest tokens.

The diagnostic records at most 8,192 events per process, then emits one
`truncated limit=8192` line. A truncated run has incomplete history. Arbitrary
guest CPU stores or `memcpy` operations are not intercepted. View creations
without a tracked desktop shadow slot and view-record allocation failures do
not produce creation events. The sequence describes shadow updates; guest
token memory writes occur afterward and concurrently reused slots still
require the caller's synchronization.

This switch adds no GPU commands, barriers, readbacks or waits. The source-built
fixture supplies its own fenced GPU snapshots. When the switch is unset, it
does not sample guest memory or add the per-descriptor history loop. The
diagnostic changes neither descriptor-copy policy nor the rendering path.

## Reproduce

Use the ordinary Windows build prerequisites in
[CONTRIBUTING.md](../../CONTRIBUTING.md#build-and-test), including DXC and Python.
Mesh requires a capable adapter/runtime. On an older Windows installation,
configure the existing `OFFSCREEN_AGILITY_620_DIR` option with a local directory
containing the public Agility 620 `D3D12Core.dll` and `d3d12SDKLayers.dll`.
Neither DLL is part of this contribution.

```powershell
. .\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev --target startup_indirect_draw_indexed startup_compute_binding_trace startup_view_alloc startup_view_census
ctest --preset dev --no-tests=error -j 1 -V -R '^startup_(mesh_argument_provenance_(warp|hardware)|indirect_mesh_(warp|hardware)|view_alloc_.*|view_census_.*|compute_binding_trace_warp)$'

cmake --preset release
cmake --build --preset release --target startup_indirect_draw_indexed startup_compute_binding_trace startup_view_alloc startup_view_census
ctest --preset release --no-tests=error -j 1 -V -R '^startup_(mesh_argument_provenance_(warp|hardware)|indirect_mesh_(warp|hardware)|view_alloc_.*|view_census_.*|compute_binding_trace_warp)$'
```

To inspect the unfiltered diagnostic for one case:

```powershell
$env:X2W_D3D12_DESCRIPTOR_HISTORY = '1'
$env:X2W_MESH_PROVENANCE_CASE = 'positive'
.\out\build\dev\startup_indirect_draw_indexed.exe hardware mesh-provenance
```

`startup_mesh_argument_provenance_warp` is included in the PR smoke-test preset.
Capability skips return 77 and must be reported separately from passes.

## Local result and remaining integration work

Tested on 28 November 2026, Windows 10 Pro build 19045, NVIDIA GeForce RTX
4070 Ti SUPER, driver `32.0.15.7270`, MSVC `19.43.34810`, DXC `1.9.2609`,
Agility SDK 620, D3D12 debug layer enabled. Build: this change on base
`32dc3d7c3cdc4be79afdad4cc04709e76d99a795`.

Debug and Release each ran the 23 selected tests: 21 passed, zero failed,
and two capability skips (`startup_indirect_mesh_warp` and
`startup_mesh_argument_provenance_warp`, Mesh shaders unavailable on local
WARP). The hardware provenance test runs six processes: the positive case,
four negative controls and the positive case without tracing. All had the
expected records, pixel readback and descriptor history, with no debug-layer
ERROR/CORRUPTION messages or device removal.

The producers use desktop source-built HLSL on a native command list. View
creation/copy and the indirect Mesh consumer use the guest wrappers; queue
execution and fences are native. This does not exercise a translated gfx1013
producer, guest queue preprocessing, XMem argument refresh, or the real
integration's root/attachment state.

Issue #3310 remains open until a matched integration run identifies the actual
scalar-input write and descriptor history, reads identical nonzero records at
the same resource and offset after production and before consumption, and
observes a real Mesh submission changing scene color before composition. The
issue's reported 48 zero records were not reproduced by this fixture. No
matched integration configuration or capture is included in this change.
The native integration work in
[#3091](https://github.com/R0Y00/Xbox2Windows/pull/3091) and
[#3094](https://github.com/R0Y00/Xbox2Windows/pull/3094) was still open when this
result was recorded; it has not been substituted for a test of `main`.
