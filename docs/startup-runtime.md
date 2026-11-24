# Startup runtime stand-ins (experimental)

This page describes the experimental desktop stand-ins under `src/startup_stubs/` and the tools
under `tools/startup/`. They let a console executable start on desktop Windows under a debugger,
so that its startup path can be studied one missing piece at a time.

This is not a port and not an emulator. No executable is playable. Every stand-in is opt-in, and
most answers are shortcuts that are logged as such. The open problems are tracked in the issues
whose titles start with `startup-runtime:`.

## Derived pipeline blend state

Device slot 55 accepts up to 64 fixed-size derived pipeline patches. The translated graphics
path reads the blend descriptor from part 2 of a 332-byte patch. Within one derivation the last
blend patch wins. A further derivation inherits that state until it supplies another blend patch.
Other patch parts are outside this blend implementation. The record layout can also be compared
with [XCL's implementation at 64b3f35](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/d3d12x/device_x.rs#L501-L515).

Run `ctest --preset dev --no-tests=error -R "^startup_stream_draw_(hardware|warp)$" -V` after
building `startup_stream_consumer`. The source-built shader writes to half of an 8×4 target.
On A8 the parent preserves the zero clear and the child writes alpha 255. R8 records red 0 or
255 according to the input shader. The untouched half stays zero in all four cases.

## Output window and fallback diagnostics

The desktop swap chain has its own top-level window, class `X2WOutputWindow`.
When a present cannot show its surface, the GDI fallback selects a visible,
unowned guest window in the same process. If only the presenter is visible,
the fallback leaves it alone. To exercise both z-orders with source-built
windows, build `startup_overlay_window` and run
`ctest --preset release --no-tests=error -R '^startup_overlay_window$' -V`.

## Null Texture2D descriptors

When the adapter lacks typed UAV support for B5G5R5A1_UNORM or render-target support for
B4G4R4A4_UNORM, the wrapper uses an R8G8B8A8_UNORM null descriptor if that format supports the
requested view. This applies only to Texture2D views with no resource or counter. Run
`ctest --test-dir out/bin --no-tests=error -R "^startup_null_views_" -V` against your build
directory to exercise both slots on hardware and WARP.

## Null bindings in translated draws

The `x2w-render totals` line reports `null bindings N`, then the same count split by cause as
`(entry, buffer, token, wrapper, dimension)`. Each null binding falls in exactly one cause, so
the five parts add up to N:

- entry: the descriptor table entry is null, empty or unreadable.
- buffer: the buffer address or size is missing or unreadable, or the frame copy failed.
- token: the descriptor memory names a view id the wrapper never issued.
- wrapper: the texture's resource is missing or is not a resource wrapper.
- dimension: the view dimension does not match the shader, so a null texture is bound.

The classification lives in `src/startup_stubs/null_binding_cause.h`. The source-built
`startup_null_views_check` fixture checks every cause, that the parts sum to the total, and that
valid bindings add nothing. It does not draw, so the counters in the totals line itself are checked
by reading a render log.

## What is measured

With the switches below, an x64 console executable on an AMD RDNA4 GPU (Windows 11, release
build of this branch):

- presents frames at about 29 fps through the D3D12 stand-in, and its translated UI draws reach the
  output window;
- reads the virtual gamepad and answers the first-run user prompt for one local user;
- streams data through the DirectStorage stand-in (about 20 000 requests in the first two minutes,
  zlib-inflated on the CPU);
- plays its intro video through the GPU-view mirror (colours are still wrong, see the issues);
- then stops in its texture-streaming path: either at a soft assert (an `int3` the title steps
  over when no debugger is attached), or, past that, on a black screen.

## Build

The stand-ins are part of the normal build. From a developer prompt:

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release --no-tests=error -R '^startup_stubs$' -V
```

The build produces `xmem.dll`, `d3d12_xs.dll`, `AcpHal.dll`, `XCRDAPI.dll`, `chad.dll`, `xapu.dll`,
`xg_xs.dll` and `GameInput.dll`. Without the switches, every export that has no implementation
traps: it logs `x2w-trap <dll>!<export>` with its argument registers and ends the process with a
code that names the export (see `src/startup_stubs/trap.h`).

## Run

Prepare a folder with the executable and its package DLLs yourself. Never commit or upload it.
Then:

```powershell
.\tools\startup\run.ps1 -Exe <folder>\<title>.exe -DataDir <package root> -LogDir out\startup\run1 `
    -Seconds 180 -Render -FakeUser -GpuMirror -ContinueAsserts 1000 -InputScript "4@70"
```

`run.ps1` copies the stand-ins next to the executable and starts `tools/startup/debug_launch.py`.
The debugger runs the process in a kill-on-close job with a memory cap, **offline**: file opens of
socket, NetBIOS and DNS devices are refused (`--offline 0` turns this off; do not). It writes:

| File | Content |
|---|---|
| `<log>.out` | why the run stopped: exception, registers, stack, likely return addresses, raw stack |
| `<log>/events.txt` | module loads and every exception with its time |
| `<log>/stderr.txt` | the stand-ins' logs (`x2w-d3d12`, `x2w-dstorage`, `x2w-xmem`, `x2w-runtime`, ...) |
| `<log>/files.txt` | every file the process opened |

`tools/startup/live_view.py` follows a run in a second console. `X2W_TRAP_DUMP="exe+RVA:len,reg+off:len"`
adds memory dumps to the stop report.

## Launcher check

`tests/startup_launch_check.py` (ctest `startup_launch`) drives `tools/startup/launch.py` on the
source-built `tests/startup_launch_probe.cpp`. A finished run reports the child's exit code and saves
its stdout, stderr and trap log; a child that outlives `--timeout` is terminated; `--memory-gib` and
`--max-processes` stop an over-limit allocation and a helper process, and the same requests succeed
under larger limits; a missing executable is reported instead of looking like a finished run. No GPU
and no game files are needed, and the `pr` preset builds the probe and runs the test:

```powershell
ctest --preset release --no-tests=error -R '^startup_launch$' -V
```

## Switches

| Variable | Effect |
|---|---|
| `X2W_XMEM_PERMISSIVE=1` | accepts the reservation, mapping and allocation shapes a startup run uses |
| `X2W_XMEM_GPU_MIRROR=1` | pairs a no-access graphics reservation with the read-write reservation of the same size made just before it, and maps every physical page into both (same section) |
| `X2W_D3D12_PERMISSIVE=1` | wraps the device: recording command lists, console present, view tracking |
| `X2W_D3D12_RENDER=1` | draws through translated pipelines from `X2W_SHADER_CACHE` |
| `X2W_SHADER_CACHE=<dir>` | translation cache: the runtime writes source requests `req_<key>.{a,b,root}` and, when needed, RTV variants `req_<key>.rtv`. Run `tools/startup/gfx2hlsl.py --pending <dir>`, then replay the draw. |
| `X2W_DRAW_NULL_IDENTITY=1` | null descriptors read as identity values instead of zero |
| `X2W_D3D12_DUMP_SURFACE=<bmp>` | writes the last presented surface |
| `X2W_DRAW_DUMP_FRAME=<n>`, `X2W_DRAW_DUMP_DIR` | Full-resolution RGB snapshots of single-sample RGBA8/BGRA8 UNORM or sRGB targets after translated draws. A later present writes completed queue-fenced copies. |
| `X2W_AUDIO_PERMISSIVE=1` | audio HAL / APU stand-ins accept their calls (no output) |
| `X2W_RUNTIME_PROXY=1` | proxies the gaming runtime and logs every call |
| `X2W_FAKE_USER=1` | answers the user, gamertag and save APIs for one local user |
| `X2W_STORAGE_PATH`, `X2W_STORAGE_MB`, `X2W_TITLE_ID` | persistent local storage and the manifest values the runtime would return |
| `X2W_INPUT_SCRIPT=mask@sec,...` | timed virtual gamepad presses (0x4 = A); keyboard and XInput pad 0 also work |
| `X2W_SAMPLER_HEAP=1` | `gfx2hlsl.py` translates a sampler-heap load with a run-time offset into `SamplerState qh[2048] : register(s0, space1)` and writes `smpheap 2048 1 fallback 2047`. The stand-in uses entries 0 to 2046 for the bound heap and reserves entry 2047 for unaligned or out-of-range offsets. It also accepts the original `smpheap 2048 1` manifest with all 2,048 entries available. [#1143](https://github.com/R0Y00/Xbox2Windows/issues/1143) |
| `X2W_XG_VERBOSE=1` | logs every texture layout query with its mip sizes |
| `X2W_XG_BC7_4K_S_LAYOUT=1` | enables the BC7 tiled layout, copy and draw upload when `X2W_D3D12_PERMISSIVE=1` |
| `X2W_TRAP_CONTINUE=<exports>` | lets listed trap exports return 0 instead of ending the process |

The startup shader cache key includes the translator revision. After updating from
an earlier translation revision, run the recording build once to emit
requests, run `python tools/startup/gfx2hlsl.py --pending <cache-directory>`,
then start the build again. The new key keeps the former bytecode and manifest
together under their original name. A 16-dword table load can contain two
independent image descriptors at offsets 0 and 32 from the load address.

## Game saves

With `X2W_FAKE_USER=1`, the game-save stand-in keeps containers and blobs below
`X2W_STORAGE_PATH`, in `x2w-gamesave`: one directory per container, one file per blob
([#1200](https://github.com/R0Y00/Xbox2Windows/issues/1200)). Opening a container whose directory
exists succeeds. Names must be one plain path component of at most 128 bytes (no separators, `:`,
`..`, control characters, device names or the reserved suffixes `.x2w-part` and `.x2w-old`, in any
case) and a blob is at most 64 MiB. The storage root itself is trusted; a container directory or blob
that is a symbolic link or, on Windows, a junction is refused by every operation.

Blob writes and deletes are kept with the update and reach the disk only when the update is
submitted (the public API makes the whole update atomic at submit). An update closed without a
submit changes nothing. At submit every write goes to a staging file and every delete is checked
first; then each old file is moved aside, the new one put in place and, on any failure, everything
is put back, so a failing action publishes none of the update. This holds for failures inside this
process; a crash in the middle of the replacement can still leave some blobs replaced and `.x2w-old`
files behind, and a reader can see the blobs of a submit one after the other.

The local provider supports create container (16), delete container (10, 11), create and close update (23, 24), staging blob writes and deletes (25, 26), and submit (27, 28 with its result in 29).

For provider `704c3f58-e629-4cc2-b197-30511b996fe2` and interface `1bfff3af-f14a-40a3-8e35-9ada906593f9`, exact container lookup (13), container enumeration (14) and blob enumeration (18) call back with committed metadata. The local store uses the container name for its display name. Synchronous read (20) returns an array of records followed by names, NUL terminators and payloads in the caller's buffer. The count is in records, the capacity is in bytes, and each record reports the byte count of its own payload. Slots 15, 19, 21 and 22 remain unimplemented. [PC GDK thunk mapping](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L261-L270) · [Microsoft enumeration and sizing sample](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesaveenumerateblobinfo) · [Microsoft synchronous read contract](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xgamesave/functions/xgamesavereadblobdata)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_read$' -V` runs two source-built processes through the provider vtable. The writer commits two preferences under a generated name. The system variant, `startup_gamesave_read_async_system`, first rejects a null block and confirms that synchronous read still reports a missing blob. It retries through slots 28 and 29 on an immediate Windows gaming runtime queue. The restarted reader discovers the nonempty container, obtains the blob names and sizes, and reads six payload bytes from a 74-byte allocation. It also checks early enumeration stop, an unknown interface identity, a short buffer and a missing blob.

`ctest --preset dev --no-tests=error -R '^startup_gamesave_persist$' -V` checks the storage code the
slots call, not the slots: it writes in one process and reads in a second, including a close without
submit, a failing second action, a second process that sees only committed state, and a redirected
container directory.

Slot 28 first asks the Windows gaming runtime to accept the caller's async block. On rejection it returns before taking staged actions. Once accepted it commits the files and calls `XAsyncComplete` with the status for that block. Slot 29 retrieves the status with the matching operation identity. File publication is synchronous within slot 28. The caller's completion routine can run inline or through its selected completion queue. [XAsyncBegin](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasyncbegin) · [XAsyncComplete](https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasynccomplete) · [PC GDK async interface slots](https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L57-L64)

`ctest --preset dev --no-tests=error -R '^startup_gamesave_submit_'` checks denied starts and retries, two overlapping blocks and completion that reads the file and slot 29 before slot 28 returns. The system variant uses the installed Windows gaming runtime with an immediate work/completion queue. It reports a capability skip when the runtime or queue is unavailable.

## Findings

These are measured on one executable; treat them as hypotheses for others.

- **User API, second interface.** The user provider hands out a second interface for the
  gamertag (method 3: handle, component, size, buffer, used). It must be answered separately from
  the main user object, whose method 3 is the handle duplicate. Without a gamertag, the first-run
  prompt offers no player and never advances.
- **GPU-view reservations.** The title reserves the same large range twice in a row: read-write,
  then no-access with the graphics flag. It maps physical pages only into the first, yet places GPU
  resources (for example the video frame planes) at the same offset in the second. On the console
  the second is a GPU view of the first. Mapping each page's section into both views makes those
  resources readable; without it, a YUV video shows a solid green screen (all-zero planes).
- **DirectStorage ordering.** Requests must run after `Submit`, in order with status and fence
  entries. Titles fill destination page-array entries after `EnqueueRequest`; an entry that is 0
  at enqueue is read again when the request runs.
- **Sub-dword addressing (SDWA).** The translator lowers the byte and word selects of the 32-bit
  integer VOP1/VOP2 SDWA forms: a source selection is zero-extended, or sign-extended under `sext()`,
  and the low bits of the result are placed in the destination field, with `UNUSED_PAD` zeroing the
  others and `UNUSED_PRESERVE` keeping them. The selection helpers match `tools/pixel_aot.py`, which
  already lowers the same forms. Float SDWA forms and a `dst_unused:UNUSED_SEXT` destination still
  stop translation rather than translating to something inexact. [#1141](https://github.com/R0Y00/Xbox2Windows/issues/1141)
- **Soft asserts.** Retail code keeps `int3` instructions followed by a continuation. Without a
  debugger the title's handler steps over them; under the startup debugger use
  `-ContinueAsserts N`. One of them fires about ten times a second in the texture-streaming path
  (a chunked read whose length ends inside a compressed chunk); its cause is open.
- **Struct-returning methods.** When both register arguments of a wrapped method look like objects,
  the one on the calling thread's stack is the return buffer.
- **Inline samplers.** Pixel shaders build sampler descriptors from constants (`s_movk_i32`,
  `s_bfm_b32`, 64-bit moves); the translator folds those into static samplers. A source-built
  pixel stage checks all four manifest words, rejects a mask width from a root constant, and
  decodes the folded words with the stand-in's static sampler path. The clamp sampler reads
  column 1 at x=1.25, while the wrap control reads column 0. Run
  `ctest --preset release --no-tests=error -R "^startup_inline_sampler_"`.
- **3D images.** `image_sample` with `dim:SQ_RSRC_IMG_3D` becomes a `Texture3D<float4>` sampled with
  a `float3` of the three address registers in order; 1D images are refused. A source-built pixel
  stage draws one texel of a 2x2x2 texture on WARP and hardware with DXC and FXC
  (`ctest --preset release --no-tests=error -R "^startup_image_3d_"`).
- **Flat parameters.** `v_interp_mov_f32` reads a `nointerpolation uint4` copy written from the
  exported registers, so integer words reach the pixel stage unchanged. FXC folds `asfloat()` of a
  constant and flushes binary32 denormal patterns such as 1 or 2 to 0, which a float copy lost
  ([#1259](https://github.com/R0Y00/Xbox2Windows/issues/1259)). Colour exports of constant integer
  words still go through `float4` and have the same loss. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_flat_param_"`.
- **Stream-cache identity.** A cached zlib stream is served only to the exact queued
  continuation entries in the same DirectStorage queue. The lookup checks the enqueued request
  id, the source, the compressed size and the destination chunk size. A later independent
  request at a matching offset still decodes its own bytes or fails without touching its
  destination.
- **DirectStorage file ranges.** A nonempty file request checks its byte offset and length against
  the current file size before allocating the staging buffer. A range ending at EOF is valid.
  An overrun or short read fails the queue request and preserves the destination.
  The `startup_storage_directory_batch` fixture submits ten requests across two synthetic RDS5/ETPF
  files on one queue. In each file, moving the first absolute source offset by 896 KiB selects a
  different valid 28-byte zlib stream. Both return success and produce 4096 bytes. Comparing the
  entire page detects the wrong stream. Moving the second request by the same distance reaches
  EOF and leaves the destination untouched. A subsequent request using its directory offset succeeds.
  Reproduce with `ctest --preset release --no-tests=error -R '^startup_storage_directory_batch$' -V`.

## Shortcuts that are known to be wrong

- A shader-visible CBV/SRV/UAV heap larger than the adapter accepts is created at the limit the
  adapter does take (measured once, then bisected) and the title's full-size memory stays the heap's
  CPU side. Desktop descriptor writes and copies use a CPU-only heap for the remainder. Translated
  graphics draws resolve their root tables against the bound logical heap and publish the requested
  records into the draw's private shader-visible heap. Changing the bound heap invalidates its
  tables, and a record extending beyond the logical end supplies a null view. This covers translated
  graphics draws; other consumers need their own binding path. Drivers refuse the request
  with different codes for the same condition (`E_INVALIDARG` on a GTX 1660 SUPER, `E_OUTOFMEMORY` on
  an RTX 3060), so any failure of an oversized shader-visible view heap takes this path. The measured
  limits so far are 1,000,000 descriptors on both adapters. Reproduce with
  `ctest --preset release --no-tests=error -R '^startup_(view_heap|logical_view)_' -V`, which skips (77) on an
  adapter that accepts the full request.
- BCPack requests are completed with zero-filled destinations (blank textures).
- The XG stand-in normally uses desktop copyable footprints. Its opt-in BC7 diagnostic
  reports standard GFX10 4 KiB tiled geometry for a single-slice, single-mip, non-MSAA
  `BC7_UNORM` 2D texture with an unknown layout, default alignment, and no resource flags.
  The diagnostic selects the profile explicitly. Native XG selector values still need
  independent evidence. `CopyIntoSubresource` writes the visible BC7 blocks from pitched source rows
  and preserves tile padding. In render mode, a successful copy marks the backing so the D3D12
  stand-in can detile BC7 blocks into the desktop upload footprint. The companion D3D12
  stand-in must be loaded for render-mode copies. The reported pitch describes the padded
  block-row extent of the surface. Synthetic copy and draw checks run with
  `ctest --preset dev --no-tests=error -R '^startup_xg_bc7_' -V`.
- The default translator uses a static default sampler for a dynamically indexed sampler-heap load.
  With the sampler-heap opt-in, translated draws snapshot the bound heap's 16-byte sampler entries and
  bind a private desktop table beside the static samplers. Missing, out-of-range, or unreadable entries
  use a transparent-black border sampler and have separate counters. The runtime keeps recorded tables
  alive through queue completion. Constant sampler-heap offsets still report an unsupported operation.
  Wrapped sampler heaps keep the 16-byte title stride separate from the desktop descriptor increment
  for CPU handles and descriptor copies.
  The source-built readback uses `ctest --preset dev --no-tests=error -R '^startup_sampler_heap_' -V`.
- The startup shader translator lowers `image_sample_l` (explicit level, 2D and 3D, any `dmask`) to
  `SampleLevel(sampler, uv, level)`, taking the level from the last address register. Offset, compare,
  gather, clamp, cube, array, 1D and `unorm` forms are refused. Only the translation is checked
  (`python -B tests/startup_image_sample_l_check.py`); a WARP readback of distinct mip levels is still open.
- Slot 86 graphics streams can submit VS/PS requests to the offline cache. The fixture sends
  complete shader containers and a wrapped root through WARP, runs `gfx2hlsl.py --pending`, and
  reloads the compiled stages. A second fixture draws a clipped triangle through the translated
  pipeline, submits the wrapped list, waits for its fence and checks all 32 RGBA pixels on WARP and
  hardware. Swapping the red/green shader changes the 16 covered pixels. Reproduce with
  `ctest --preset release --no-tests=error -R "^startup_stream_(request|draw)_"`.
  Other stage combinations and full stream rasterizer/depth state remain open.
- Translated draws use the bound typed RTV formats to request colour-export variants. The
  manifest's `targets` line lists exported slots. A separate `rtv1` line identifies the base
  shader and the ordered RTV formats, and the draw rejects a variant with a mismatched line.
  Uncompressed `R32_UINT` and `R32_SINT` outputs use integer shader signatures.
  Uncompressed binary32 FLOAT outputs bitcast the export words after XOR with a private
  zero-valued entry constant so optimized FXC keeps denormal bit patterns. UNORM outputs still
  undergo float-to-UNORM conversion. Packed integer exports and other integer formats remain
  unsupported by this path.
- Translating the captured console colour-blend register state blocks remains open (#1216).
  A graphics stream can supply a desktop `D3D12_BLEND_DESC` directly. `src/startup_stubs/cb_blend.h`
  decodes `CB_BLENDn_CONTROL` and a `CB_TARGET_MASK` nibble into `D3D12_RENDER_TARGET_BLEND_DESC`. It
  refuses forms D3D12 can't express (`BOTH_SRC_ALPHA`, constant alpha without alpha blend factor
  support, MIN/MAX with factors other than ONE). Its draws match a CPU model of the register fields
  (`ctest --preset release --no-tests=error -R "^startup_cb_blend"`). Where those registers sit in the
  state block is not measured yet, so the draw path doesn't call it.
- Audio produces no output from the console audio path. A separate desktop render sink exists
  (see "Audio output"); nothing connects the console path to it yet.

## Audio output

`src/startup_stubs/audio_sink.{h,cpp}` is a desktop WASAPI shared-mode render sink. It opens the
default render endpoint, takes the mix format from `GetMixFormat` (32-bit float or 16-bit PCM only),
converts each submitted block (format, frame count, bytes) to it, and writes it through
`IAudioRenderClient`. Consumed frames are the submitted frames minus `GetCurrentPadding`; the
`IAudioClock` position is reported next to them. It records the first accepted block, the
submitted format, per-block frame counts and peak amplitude, the drained interval (queue empty
while the producer withholds data) and the recovery (resume until consumption advances). The
render API reports no discontinuity flag, so the glitch count is "not available" and is kept
apart from both intervals.

```powershell
cmake --build --preset release --target startup_audio_output
ctest --preset release --no-tests=error -R '^startup_audio_output$' -V
```

The fixture runs three cases: no submission (nothing accepted or consumed); a stream of eight
generated 40 ms tone blocks with distinct amplitudes, one of them all zeros (accepted, counted,
peak 0); and starvation (drain to padding 0, hold 300 ms, resume, consumption recovers). Exit
code 77 means no usable render endpoint (or an unsupported mix format). It prints the endpoint,
format and Windows build. A skip is neither a pass nor a failure. Every wait is bounded.

Not established: the layout of the console audio command queue (`xapu.cpp` is not decoded
here), what is audible (no capture-path check), and any hardware or driver glitch count.

## Uniform first-lane reads

The startup shader translator accepts `v_readfirstlane_b32` in straight-line code with the
entry mask when its active VGPR holds an integer conversion of a fixed-offset scalar buffer
value. Run `python -B tests/startup_uniform_lane_check.py` to check the producer and the
varying-source, register-clobber, mask and branch controls.

## Frame capture reader

`tools/startup/capture.py` reads the `capture.bin` the frame capture writes. A truncated file, a chunk
with a short header, or an unknown tag raises a `ValueError` that names the tag and offset.
`ctest --preset dev --no-tests=error -R '^startup_capture$' -V` checks every chunk layout against a
synthetic file.

## NV12 plane upload

For a single-mip, single-sample, even-sized 2D NV12 image the desktop stand-in
reports two linear footprints. XG copies luma and interleaved chroma separately,
and the draw uploader addresses both D3D12 subresources. NV12 arrays remain
unsupported. The source-built draw compares a synthetic image and its U/V-swapped
control against the same integer colour conversion on the CPU. This verifies
the stand-in path; the texture format used in the reported video remains to be confirmed.

```powershell
cmake --build --preset release --target startup_nv12_draw d3d12_xs_shim
ctest --preset release --no-tests=error -R '^startup_nv12_draw_' -V
```

## Startup memory accounting

The DirectStorage heartbeat (`x2w-dstorage heartbeat <n> s:` in `stderr.txt`) reports this
process's peak and current commit charge in MiB and the number of live 64 KiB physical-page
sections, so runs that stop at different points can be compared by memory spent rather than by
elapsed time. A physical-page section exists from a page's first use (a mapping or a DirectStorage
write) until its grant is released; `xmem.dll` publishes that count through the experimental
`x2w_xmem_page_stats` export. The heartbeat also prints while the request counters are unchanged
when either memory value changed, so the cost of a stalled load is visible.

```powershell
cmake --build --preset release --target xmem_shim startup_stub_xcrdapi
ctest --preset release --no-tests=error -R '^(xmem_page_stats|startup_heartbeat_commit)$' -V
```

The `startup_heartbeat_commit` check starts the heartbeat before loading the page provider, then
maps two synthetic sections and compares the reported values with `GetProcessMemoryInfo` in the
same process. It releases a separate commit allocation to check that a current-only change
produces another report.
