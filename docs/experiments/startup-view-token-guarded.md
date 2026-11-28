# View token written under an exception guard (opt-in)

A view call (device slots 17–22) ends in `note_view` (`d3d12_capture.inl`), which writes a 32-byte token (16 bytes
for a sampler) into the title's descriptor memory, so that descriptor copies the title makes itself can be followed.
The write is preceded by `x2w_local::writable`: `commit_lazy` (with `X2W_XMEM_LAZY_COMMIT`, `x2w_xmem_commit_range`,
which takes xmem's commit mutex) and then a `VirtualQuery`. The destination is always inside a registered descriptor
range (`title_backed`). Late in the main executable's load the title creates about 10^5 views per second, from five
recording threads at once.

## What the switch changes

- `X2W_D3D12_VIEW_TOKEN_GUARDED=1`: the token is written by `write_view_token` under `__try`, without the commit call
  and the query. A page that refuses the write (no access, read-only, a guard page, an in-page error) leaves the
  descriptor without a token, as `writable` returning false did; a guard page has its guard bit set again, on that
  page only, as `probe_readable` does.
- Lazy commit: with `X2W_XMEM_LAZY_COMMIT`, xmem's vectored handler commits a page on the first touch, before the
  `__except` filter sees the fault. Only the startup runs below exercise this; the fixture commits its memory up
  front and loads no xmem.dll.
- Concurrency: between the fault that clears a guard bit and the re-arm, another thread's write to that page goes
  through, where `writable` would have refused it. The runs below refused no write, so it did not occur there.
- Only a token that lies inside one 4 KiB page takes the guarded write, so a refusal comes on its first byte and
  changes nothing. A descriptor heap's title backing has no alignment check, so a token can cross a page boundary;
  such a token keeps the full-span `writable` check, with or without the switch.
- With the switch set, the render totals are followed every 300 frames by
  `x2w-render view tokens (X2W_D3D12_VIEW_TOKEN_GUARDED): written N, refused M`.
- Unset, the check and the write are as before.
- Only the token write changes; the view record, the shadow slot and the view references are untouched.

## Fixture

```powershell
ctest --preset dev --no-tests=error -j 1 -R '^startup_view_token_' -V
```

`startup_view_token_{unset,set}_{warp,hardware}` (`tests/startup_view_token.cpp`) creates three descriptor heaps over
title memory the fixture owns and, after creation, makes the second read-only and gives the third a guard page. A
texture SRV goes into each heap, and a second one into the writable heap. In both modes the writable heap holds both
tokens, the read-only heap stays zero and the process keeps running, and the guard page keeps its guard bit and stays
zero. Two more heaps start 16 bytes before a page boundary, so their first token crosses into a read-only page or a
guard page: in both modes the 16-byte writable prefix keeps its bytes, the left page stays `PAGE_READWRITE` without
a guard bit, and the right page keeps its protection and stays zero. The fixture-only export
`x2w_d3d12_view_token_writes` reports the guarded writes: unset `0, 0`, set `2, 2` (crossing tokens are not
guarded writes). Mutants: ignoring the switch fails `the guarded path did not write 2 tokens and refuse 2`; not
setting the guard bit again fails `the guard page lost its guard bit`; setting it again over the whole region
(`info.BaseAddress`, `info.RegionSize`) fails `the page after the guard page changed its protection`; guarding
crossing tokens too fails
`a crossing heap's writable prefix changed` (read-only right page) and leaves the left page `0x104`
(`PAGE_READWRITE | PAGE_GUARD`, guard right page).

## Startup run of the main executable

Local diagnostic runs, Windows 10 22H2 (19045), i7-4790K (4 cores, 8 threads), 16 GB, RTX 3080 Ti, NVIDIA driver
32.0.16.1047. Release build of this change on `main` `36d2cbeb`, warm shader cache (revision 29), scripted input,
250 s. `X2W_D3D12_RING_RELEASE_AT_SUBMIT=1`, `X2W_D3D12_TABLE_ALIAS=1` and `X2W_D3D12_ROOT_SRV_AT_DRAW=1` in every
run; only this switch differs. Two pairs, the second in reverse order. Aggregates only; logs stay local.

| 150–240 s | switch unset | switch set |
|---|---|---|
| mean fps, pair 1 / pair 2 | 15.5 / 15.1 | 16.6 / 15.6 |
| time in view calls (threads); µs per call | 1.19 / 1.20; 10.4 / 11.2 | 0.88 / 0.92; 7.0 / 6.7 |
| view calls per second | 114k / 107k | 126k / 136k |
| null bindings `buffer`; device removal | 0; none | 0; none |
| null bindings `token` per 1,000 translated draws | 25.7 / 23.5 | 33.6 / 30.1 |
| process commit peak | 19.3 / 18.9 GiB | 19.6 / 19.4 GiB |

- A view call costs about 3.6 µs less. In a breakdown of a view call on the same machine, the `writable` check was
  about a third of its time.
- Why it shows in the frame rate: five recording threads draw the late load. A thread-state sample of them
  (`NtQuerySystemInformation`, every 20 ms) found them runnable but without a free core in 14–21% of the samples,
  so CPU time they save is CPU time they use for the next lists.
- A confirmation run with the switch set (the report line above) logged `written 16447410, refused 0`: the guard
  refused no write. That shows no token was lost to a refusal, not that every token matches the default path.
- The `token` null bindings were higher with the switch (33.6 / 30.1 against 25.7 / 23.5 per 1,000 translated
  draws). Runs without this change gave 23–34 per 1,000 on the same machine, so the difference lies within that
  spread, but it is not explained.

Not established: other adapters and drivers, hosts with more cores, and Wine (where #2911 changes `writable`).
