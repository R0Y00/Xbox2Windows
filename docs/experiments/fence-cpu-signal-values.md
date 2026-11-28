# CPU fence signal values

The source-built `startup_fence_values` fixture compares a shim fence's method
result and cached completion word with its underlying desktop `ID3D12Fence`.
It signals `9`, `2`, `0`, then `7` on the CPU. Each successful CPU signal must
publish the observed desktop value synchronously, including a lower value and
zero. The cached word is read before the shim method, so a method read cannot
hide a stale word.

The public [ID3D12Fence::Signal contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12fence-signal)
allows a CPU signal to set the completion value. The
[fence synchronization guidance](https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization)
also explicitly permits rewinding a fence. The shim previously applied an
atomic maximum and published the requested value even if the desktop call
failed. It now updates the cached word from the desktop observation after a
successful CPU signal; a failed call leaves the word unchanged.
Each fence serializes the native CPU signal, desktop observation and cached
publication with a fence-local SRW lock. A later or equal callback retains the
existing value through the same publication path.

## Reproduction

From an x64 Windows developer PowerShell at the repository root:

```powershell
. .\tools\dev.ps1
cmake --preset dev -DINTAKE_PYTHON=C:/path/to/python.exe
cmake --build --preset dev --target startup_fence_values startup_queue_wait --parallel 4
ctest --preset dev --no-tests=error -V -R '^startup_(fence_values|queue_wait)_warp$'
ctest --preset dev --no-tests=error -V -R '^startup_fence_values_hardware$'
```

The test registrations enable `X2W_D3D12_PERMISSIVE`, `X2W_D3D12_LIST_NOOP`,
`X2W_D3D12_RENDER`, and `X2W_D3D12_DEBUG`. The CPU-only fixture creates no
command queue and submits no GPU commands. Hardware enumeration returns skip
code 77 if no suitable adapter is available; a skip is not a pass.

## Measured result

Measured on 2026-11-28 against base commit
`240e0fb9c460f24d51683abe2d4cce2800b136c1`, with the focused CPU fix applied,
using MSVC 19.43.34810, x64 Debug, and Windows 10 Pro 10.0.19045. The D3D12
debug layer reported enabled in every run.

| Adapter | DXGI driver version integer | CPU signals | Result |
| --- | --- | --- | --- |
| Microsoft Basic Render Driver (WARP) | 2814751014983330 | 9, 2, 0, 7 | Native, method, and cached word agree for all four |
| NVIDIA GeForce RTX 4070 Ti SUPER | 9007199255731302 | 9, 2, 0, 7 | Native, method, and cached word agree for all four |

On the unmodified base, WARP reported native values `9, 2, 0, 7`, but the shim
method and cached word remained `9, 9, 9, 9`. With the fix, both registered fence
tests passed. The existing `startup_queue_wait_warp` test also passed. These
results cover the CPU signal contract and a two-caller barrier regression, not
full startup or queued signals. A failed desktop signal was not induced in this run.

## Separate queued rewind limitation

The same fixture provides an optional diagnostic, deliberately excluded from
the passing CTest registrations:

```powershell
$env:X2W_D3D12_PERMISSIVE = '1'
$env:X2W_D3D12_LIST_NOOP = '1'
$env:X2W_D3D12_RENDER = '1'
$env:X2W_D3D12_DEBUG = '1'
.\out\build\dev\startup_fence_values.exe warp --queue-rewind
```

After the CPU sequence ends at `7`, the diagnostic blocks a queue, enqueues
`Signal(3)` and an independent marker, then releases the blocker. The marker
proves that the queue reached its signal. The diagnostic polls only the cached
word before reading the shim method. On WARP with this CPU-only fix, the desktop
fence became `3` while the method and cached word remained `7`, and the diagnostic
returned failure as expected for this unresolved behavior.

Replacing the queued callback's atomic maximum with an assignment is insufficient:
an event registered for target `3` is already satisfied while the fence is `7`,
before the queue executes the lower signal. Sampling the desktop fence in that
early callback is also insufficient. Queued completion publication needs an
independent queue-order completion mechanism, such as a private monotonic marker.
That behavior is outside this CPU signal fix.

Mixed CPU/queue signals have another unresolved interleaving: a queue callback
can retain an earlier larger target, run after a successful CPU rewind, then
publish that old target over the CPU value through `fence_reached`. This
interleaving was identified from the callback path and was not induced in this
run. The passing CPU tests create no queue and make no claim about pending
callbacks or concurrent CPU and queue producers. The shim completion method
reads the lock-protected cached word; direct address reads remain atomically
published by the same fence-local lock on success.
