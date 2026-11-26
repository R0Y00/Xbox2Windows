# Windows hardware CTest on a GeForce GTX 1080 Ti

The source revision was `96a7b4fb265840c3bbe594be922f3e66c6c2f182`.
The host ran Windows 10 22H2 x64 (build 19045.6456) with an NVIDIA GeForce
GTX 1080 Ti and driver 32.0.15.8278 (582.78). The Debug build used MSVC
19.31.31105.0, Python 3.12.14 and DXC 1.8.2502.11 from Windows SDK 10.0.26100.0.

From a dedicated worktree at that revision, set up the Python environment and
run the suite serially from PowerShell.

```powershell
python -m venv out/pe-env
.\out\pe-env\Scripts\python.exe -m pip install -r requirements-dev.txt
. .\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev -j 8
$env:XBOX2WINDOWS_ADAPTER = 'NVIDIA GeForce GTX 1080 Ti'
ctest --preset dev --no-tests=error -j 1 -R '_hardware' --output-junit (Join-Path $PWD 'out/hardware-junit.xml')
```

The filter selects every test whose name contains `_hardware`, including the
device-independent `tooling_hardware_reports_cli_check`. Supported fixtures
are expected to pass. A fixture that reports an unsupported capability is
counted separately as a skip.

## Adapter capabilities

The capability probes reported Shader Model 6.5, DXR Tier 1.0, a wave lane
range of 32–32 and mesh shader tier 0. The D3D12 debug layer was available.
The logs show it enabled for the offscreen fixtures and off for fixtures
configured to exercise the no-debug path.

## Results

CTest completed all 685 selected entries in 4194.11 seconds and returned exit
code 8. It recorded 577 passes, 19 failures and 89 skips. The failures
included seven segmentation faults and one timeout.

The first failure was `startup_view_dimension_hardware`. CTest reported
`SegFault` after 0.35 seconds with the debug layer off. The last fixture
diagnostic was that the runtime skipped a view whose dimension did not suit the
resource. The two sampler variants also reported the same readback
difference at table 2, probe 3. Both expected 34 and read back 34.25.
`startup_image_load_d16_hardware` hit its 120-second CTest timeout.

Skip diagnostics include mesh shader, Shader Model 6.6, wave-width and DXR
Tier 1.1 requirements. Fourteen SPIR-V bridge tests skipped because
`spirv-as`, `spirv-val` and `spirv-cross` were absent at configure time.
The two-exit mesh draw also reported a missing LLVM 15 AMDGPU prerequisite.
The run covers the repository's source-built fixtures on this configuration.
