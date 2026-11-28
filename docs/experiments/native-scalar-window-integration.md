# Native scalar input windows in the integration stack

The public integration stack in [#3248](https://github.com/R0Y00/Xbox2Windows/pull/3248),
at `7f97cb5ba467d16019dfd3ecdc86cf09c6dbcf09`, retains the native compiler's
`Slot::scalar_window` metadata but does not consume it when resolving compute inputs.
The bounded unplaced-window fallback exists in the public runtime branch of
[#3094](https://github.com/R0Y00/Xbox2Windows/pull/3094) at `97cde4d5`.

## Reproduction and change

The existing source-built
[native compute dispatch check](../../tests/startup_compute_dispatch_native_check.py)
loads its scalar descriptor from table 1 byte 64 and reads a scalar at buffer byte 36.
The descriptor has three records of stride 16 in readable CPU memory outside the
fixture's placed buffers. A valid decoy at table byte 0 contains a different value.

On the unmodified integration head, the first positive case is refused with
`an input is in title memory outside any placed buffer`. All 288 result words stay
at `0xc5c5c5c5` instead of becoming `0x400`; the 32 guard words remain untouched.
Native translation itself succeeds.

Restore the bounded fallback after normal GPU-backed range resolution. It applies
only to native scalar views, non-token descriptors, and that specific unplaced-range
refusal. The descriptor must cover the compiler's required bytes, contain whole
dwords, be at most 64 KiB, and be completely readable. The existing frame upload
copy retains the data for the recording. Successful normal resolution, invalid
tokens, partial placements and short windows do not enter this fallback. Existing
compute input source counters identify the fallback as a CPU snapshot.

This restores integration behavior; it does not enable the separate opt-in snapshot
of non-aliased placed buffers and does not change that policy.

## Validation

Windows 10 Pro build 19045, NVIDIA GeForce RTX 4070 Ti SUPER, driver
`32.0.15.7270`, Release `/O2`, MSVC `19.43.34810`, DXC `1.9.2609`,
Agility SDK 620, D3D12 debug layer enabled. The local WARP driver is
`10.0.19041.5794`.

```powershell
. .\tools\dev.ps1
cmake --build --preset runtime
# Use Windows' locale encoding for localized native error messages in this fixture.
$env:PYTHONUTF8 = '0'
ctest --preset release --no-tests=error -j 1 -V -R '^(native_fixed_mesh57_(varying|buffers)?_?hardware|startup_native_graphics_draw_hardware|startup_compute_dispatch_(native|input)_hardware|startup_slot67_indirect_dispatch_hardware)$'
ctest --preset release --no-tests=error -j 1 -V -R '^(startup_compute_dispatch_(native|input)|startup_slot67_indirect_dispatch)_warp$'
```

- Hardware: 7 selected, 7 passed, 0 failed, 0 skipped, 24.09 seconds.
- The scalar positives match all 288 result words for both `0x400` and `0x45678`,
  with all 32 guards unchanged. The two-record (32-byte) control is refused and
  leaves the complete output unchanged. The ordinary native, published fallback,
  wrong table and shared read/write UAV cases also pass.
- Existing input rejection, opt-in snapshot, memory-view token, indirect dispatch
  and fixed Mesh pixel checks pass without changing their assertions.
- WARP: 3 selected, 2 passed, 0 failed, 1 capability skip, 7.24 seconds. Native
  compute skips because this WARP adapter has no 32-lane waves; its published
  fallback and refusal controls run before the skip.

With Python 3.14's UTF-8 default, the graphics check separately fails decoding a
localized missing-DXC error. `PYTHONUTF8=0` permits its existing negative control
to run; no graphics or test source change was needed for that local encoding issue.

## Scope

This fixture uses only synthetic inputs. Its descriptor shape is relevant to
[#3310](https://github.com/R0Y00/Xbox2Windows/issues/3310), but it does not establish
the cause of that issue's 48 zero argument records. A matched integration capture
must still identify the actual SRV producer, demonstrate nonzero matching
post-write and pre-consume records at the same resource and offset, and show a real
Mesh submission changing scene color before composition.
