# `--watch` mode in `compile_dxil`

[The test fixture](../../tests/native_compile_dxil_watch_check.py) verifies continuous and one-shot
`--watch` operation in `tools/startup/compile_dxil.cpp` on `cpp-migration` (part of #2935):

- `compile_dxil --watch SDK_DIRECTORY CACHE_DIRECTORY REQUEST_DIRECTORY [--jobs N] [--interval MS] [--once]`
  loads and fingerprints the DXC SDK once and translates pending startup graphics (`req_<key>.a`) and slot-67
  compute (`req_<key>.c67`) requests into `<key>.{vs,ps,cs}.cso`, optional `<key>.psdep`, and `<key>.txt`, or
  writes `<key>.fail` on refusal.
- Pending and resume checks match `tools/startup/gfx2hlsl.py`: completed `.txt` entries with valid DXBC `.cso`
  containers are skipped, truncated or missing stage `.cso` files are resumed, `.psin` additions/updates/removals
  trigger rechecks, pre-existing `.fail` entries are retried once on the first pass, and unchanged `.fail` entries
  are skipped on subsequent passes until the marker or `.psin` is rewritten.
- The published `<key>.psinh` digest is taken from the base `.psin`/`.ps57in` pair. An `.rtv` variant resolves the
  same base sidecars as its mtime check, so a deliberate or preserved-mtime base sidecar change withdraws the
  variant too. The digest is computed before and after the graphic translation; a change during compilation
  refuses the publish instead of pairing new bytes with DXIL made from the old ones.

## Verification

```powershell
.\tools\dev.ps1
cmake --preset release
cmake --build --preset release --target compile_dxil
ctest --preset release --no-tests=error -V -R "^native_compile_dxil_watch$"
ctest --preset release --no-tests=error -j 1 -R "^(native_startup_request_cache_controls|native_rtv_variant_controls)$"
```

## Limits

- Watches startup graphics (`req_<key>.a`), RTV variant (`req_<key>.rtv`), and slot-67 compute
  (`req_<key>.c67`) requests; slot-57 mesh requests remain outside the native path.
- `run.ps1 -Translate` still invokes `gfx2hlsl.py --watch` until the remaining graphics/runtime migration items
  in #2935 land.
