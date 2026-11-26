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
