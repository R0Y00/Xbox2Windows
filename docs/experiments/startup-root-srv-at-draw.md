# Graphics root SRV: copy only for a draw that reads the desktop binding (opt-in)

Refs [#2948](https://github.com/R0Y00/Xbox2Windows/issues/2948),
[#2949](https://github.com/R0Y00/Xbox2Windows/pull/2949).

List slot 44 (`SetGraphicsRootShaderResourceView`) at a title address without an upload mapping copies a 64 KiB
window of title memory into the shared upload ring and binds it on the desktop list (`list_root_srv`). Two kinds of
draw read that desktop binding: a real-pipeline draw (`list_draw`, gated on `srv_bound` bit 0) and an indirect
graphics draw on the title's root signature. Translated draws set their own root signature and copy the SRV from its
title address (`pass.root_address`, `source == 2` in `draw_translated`) at the draw. In the current code a real stream
pipeline is built only over a root signature without bindings (`build_stream_pipeline`, `root_without_bindings`), so
no shader reads the slot 44 copy.

## What the switch changes

- `X2W_D3D12_ROOT_SRV_AT_DRAW=1`: slot 44 keeps the address pending (`pass.srv_pending`) instead of copying.
  `bind_pending_root_srvs` makes the copy and the desktop binding at the two draws above, through the same
  `bind_root_srv_copy` as before; the bytes are then read at that draw instead of at slot 44.
- Pending bits are cleared wherever `srv_bound` is cleared: a new slot 44 call for the parameter, slot 34, and the
  translated indirect draw that restores the title's root signature.
- With the switch set, the render totals are followed every 300 frames by
  `x2w-render root SRV copies (X2W_D3D12_ROOT_SRV_AT_DRAW): deferred N, made at a draw M`.
- Unset, slot 44 copies at every call, as before.

## Fixture

```powershell
ctest --preset release --no-tests=error -j 1 -R '^startup_root_srv_draw_(warp|hardware)$' -V
```

`startup_root_srv_draw_*` (`X2W_TEST_ROOT_SRV_DRAW=write-after`, `tests/startup_root_srv_draw.cpp`): the stream
root's parameter 0 is a root SRV. Slot 44 binds a `VirtualAlloc` buffer holding red, the title writes green, then a
translated draw reads the SRV through its root pointer. Both modes draw green over the left half. After the draw is
recorded the fixture reads the stand-in's counts (`x2w_d3d12_root_srv_copies`): unset, `deferred 0, made at a draw 0,
copied 1`; set, `deferred 1, made at a draw 0, copied 0`. Mutant: ignoring the switch fails
`X2W_D3D12_ROOT_SRV_AT_DRAW=1: copies (deferred, made at a draw, copied) (0, 0, 1), expected (1, 0, 0)`.

Not covered by a pixel check: a real-pipeline draw that reads the binding, because none can be built (see above).
The clearing of pending bits follows `srv_bound` by construction, and `list_close` resets the whole pass, so a
pending address never reaches the next recording.

## Startup run of the main executable

Local diagnostic runs, Windows 10 22H2 (19045), RTX 3080 Ti, NVIDIA driver 32.0.16.1047, i7-4790K, 16 GB, release
build of this change on `main` `98e4a2ee`, warm shader cache (revision 29), scripted input, 250 s. One build; only
the switches differ. Aggregates only; logs stay local.

Each switch alone, against no switch:

| | none | `ROOT_SRV_AT_DRAW` | `RING_RELEASE_AT_SUBMIT` |
|---|---|---|---|
| null bindings with cause `buffer` | 164,727 | 88 | 0 |
| `upload ring reservation refused` logged | yes | yes | no |
| visible | single black frames in the post-load animation | none seen | not watched |

With `X2W_D3D12_RING_RELEASE_AT_SUBMIT=1` and `X2W_D3D12_TABLE_ALIAS=1` set in every run, two pairs, the second in
reverse order:

| | switch unset | switch set |
|---|---|---|
| mean fps 150–240 s, pair 1 / pair 2 | 6.3 / 7.1 | 14.8 / 14.0 |
| frames presented, pair 1 / pair 2 | 3,663 / 3,541 | 4,689 / 4,804 |
| executes, pair 1 / pair 2 | 33,408 / 36,029 | 72,177 / 70,916 |
| slot 44 time 150–240 s (threads; µs per call) | 1.14–1.19; 43–49 | 0.28–0.29; 5 |
| copies deferred / made at a draw | n/a | about 7 million / 0 |
| null bindings `token` per 1,000 translated draws | 33.8 / 32.3 | 36.7 / 26.9 |
| null bindings `buffer`, device removal | 0, none | 0, none |
| process commit peak | 19.1 / 19.0 GiB | 19.8 / 19.4 GiB |

- The slot 44 copies were most of the upload ring traffic that #2948 describes: the switch alone removes all but
  88 of the buffer null bindings, but some reservations are still refused, so it complements #2949 rather than
  replacing it.
- Without the table alias the frame rate barely changes (5.9 → 6.1 fps): the draw path is then bound by `frame_copy`
  under `draw_lock`.
- Every run reads `deferred N, made at a draw 0`: no draw in these runs read the desktop binding.

Not established: titles that draw with real pipelines over a root SRV (none can be built today), and other adapters.
