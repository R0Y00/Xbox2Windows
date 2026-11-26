[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    & "$PSScriptRoot\dev.ps1"
    $revision = git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve the source revision.' }
    $build = 'out/build/gpu-check-release'
    cmake --preset dev -B $build -DCMAKE_BUILD_TYPE=Release -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
    if ($LASTEXITCODE -ne 0) { throw 'Release configuration failed.' }
    cmake --build $build --target offscreen_check
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }

    $output = New-Item -ItemType Directory -Force -Path 'out/gpu-check'
    $stage = New-Item -ItemType Directory -Force -Path "$build/package"
    Copy-Item -LiteralPath "$build/offscreen_check.exe", "$PSScriptRoot/gpu-check/Run-GpuCheck.ps1", "$PSScriptRoot/gpu-check/README.md", 'LICENSE' -Destination $stage
    Set-Content -LiteralPath "$stage/SOURCE_COMMIT.txt" -Value $revision -Encoding ascii

    # Compile the gfx2hlsl compute cases here so the package needs no Python or DXC on the user's machine.
    $dxcLine = Select-String -LiteralPath "$build/CMakeCache.txt" -Pattern '^DXC_EXECUTABLE:[A-Z]+=(.+)$' | Select-Object -First 1
    if (-not $dxcLine) { throw 'DXC_EXECUTABLE is not in the CMake cache.' }
    $dxc = $dxcLine.Matches[0].Groups[1].Value
    $python = Join-Path $root 'out/pe-env/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw "Missing $python. Create out/pe-env and install requirements-dev.txt before packaging." }
    $cases = Join-Path $stage 'cases'
    if (Test-Path -LiteralPath $cases) { Remove-Item -LiteralPath $cases -Recurse -Force }
    & $python -B tools/record_gpu_cases.py $dxc $cases
    if ($LASTEXITCODE -ne 0) { throw 'Recording the shader cases failed.' }

    $files = 'offscreen_check.exe', 'Run-GpuCheck.ps1', 'README.md', 'LICENSE', 'SOURCE_COMMIT.txt', 'cases' |
        ForEach-Object { Join-Path $stage $_ }
    $name = 'Xbox2Windows-GpuCheck-win-x64.zip'
    $zip = Join-Path $output $name
    Compress-Archive -LiteralPath $files -DestinationPath $zip -Force
    $hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$zip.sha256" -Encoding ascii -Value "$hash  $name"

    $smoke = Join-Path $build ('smoke-' + [guid]::NewGuid().ToString('N'))
    Expand-Archive -LiteralPath $zip -DestinationPath $smoke
    & powershell.exe -NoProfile -File "$smoke/Run-GpuCheck.ps1" -Backend warp
    if ($LASTEXITCODE -ne 0) { throw 'The extracted package failed its WARP tests.' }
    $report = Get-Content -Raw -LiteralPath "$smoke/gpu-report-warp.md"
    # The Windows 10 in-box WARP (SM 6.2) skips the SM 6.5 fixtures by name; anything else must pass.
    $sm65Missing = 'bvh_waves_warp', 'bvh_atomics_warp', 'bvh_packed_warp' | Where-Object {
        $report -notmatch "$_ - PASS" -and
        $report -notmatch "(?s)### $_ - SKIP[^#]*skip: fixture requires Shader Model 6\.5"
    }
    if ($report -notmatch 'gfx2hlsl_shaders_warp - PASS' -or $report -notmatch 'cases: ([1-9][0-9]*) pass, 0 fail, 0 skip of \1\b') {
        throw 'The extracted package did not replay every recorded shader case on WARP.'
    }
    if ($report -notmatch 'dual_uav_warp - PASS' -or $report -notmatch 'texture_mips_warp - PASS' -or
        $sm65Missing -or $report -notmatch 'bvh_root_roundtrip_warp - PASS' -or
        $report -notmatch [regex]::Escape($revision) -or
        $report -notmatch '### Summary' -or
        $report -notmatch '### Submit this report' -or
        $report -notmatch 'https://github.com/R0Y00/Xbox2Windows/issues/1') {
        throw 'The extracted package is missing a required WARP pass, source revision or submit path.'
    }
} finally {
    Pop-Location
}
