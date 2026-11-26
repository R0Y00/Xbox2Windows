<#
.SYNOPSIS
Runs a console executable on the desktop stand-ins under the startup debugger (docs/startup-runtime.md).

.DESCRIPTION
Copies the stand-in DLLs from the build into -RunDir (a folder that holds the executable and the
package's other DLLs; prepare it yourself, never commit it), sets the permissive switches and starts
tools/startup/debug_launch.py. The run is offline: the debugger refuses sockets, NetBIOS and DNS.
Logs go to -LogDir; -LogDir.out holds the debugger's summary.

.EXAMPLE
.\tools\startup\run.ps1 -Exe D:\run\title.exe -DataDir G:\ -LogDir out\startup\run1 -Seconds 180 -Render -FakeUser
#>
param(
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$DataDir,   # the package root the title opens its files from (its working directory)
    [Parameter(Mandatory)] [string]$LogDir,
    [int]$Seconds = 120,
    [string]$BuildDir = "out/build/release",
    [string]$NativeLauncher = "",
    [int]$MemoryGiB = 14,
    # Host safety (runtime_work/NOTES.md): pagefile page-slab cap in MiB (X2W_XMEM_SLAB_BUDGET_MIB) and a
    # directory for file-backed slabs, which carry no commit charge (X2W_XMEM_SLAB_DIR, absolute, off G:/S:).
    [int]$SlabBudgetMiB = 0,
    [string]$SlabDir = "",
    [int]$ContinueAsserts = 0,                 # main-image int3s to log and step over (title soft asserts)
    [string]$InputScript = "",                 # X2W_INPUT_SCRIPT, e.g. "4@70" = A at 70 s
    [string]$StoragePath = "",                 # X2W_STORAGE_PATH: persistent local storage root (a trailing \ is added when missing)
    [string]$ShaderCache = "out/startup/shader_cache",
    [switch]$Render,                           # draw through translated pipelines
    [switch]$FakeUser,                         # answer the user API for one local user
    [switch]$GpuMirror,                        # map CPU-view pages into the paired GPU-view reservation
    [switch]$HostCpu,                          # rewrite instructions the host processor lacks (X2W_HOST_CPU)
    [switch]$FastExit,                         # at a timeout or Ctrl+C: rip-only threads.txt, no window capture or movie frame
    [int]$LogRepeats = 200,                    # log a repeating exception or runtime/movie hook this often, then only count it (0 = all)
    [switch]$SharedViews,                      # X2W_XMEM_SHARED_VIEWS=1: adjacent mapped pages share one view (faster map and exit)
    [switch]$Translate,                        # with -Render: translate new shader requests during the run (gfx2hlsl.py --watch)
    [switch]$StartupBelow40,                   # place the 0x283 large-page startup blocks below 2^40 (X2W_XMEM_STARTUP_BELOW_40)
    [switch]$TraceCalls,                       # also break on runtime and movie-player calls (--trace-calls 1; slows the process)
    [switch]$TraceApi                          # log runtime provider method calls (X2W_TRACE_API)
)
$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$build = if ([IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $repo $BuildDir }
if ($NativeLauncher) {
    $NativeLauncher = (Resolve-Path -LiteralPath $NativeLauncher).Path
    if ($Translate) { throw "The native launcher takes precompiled shaders. Omit -Translate." }
}
$runDir = Split-Path -Parent (Resolve-Path $Exe)
foreach ($dll in "xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput") {
    Copy-Item (Join-Path $build "$dll.dll") $runDir -Force
}
$env:X2W_AUDIO_PERMISSIVE = "1"; $env:X2W_D3D12_DEVICE_NOOP = "1"; $env:X2W_D3D12_LIST_NOOP = "1"
$env:X2W_D3D12_PERMISSIVE = "1"; $env:X2W_XMEM_PERMISSIVE = "1"; $env:X2W_RUNTIME_PROXY = "1"
$config = Join-Path $DataDir "MicrosoftGame.config"
if (Test-Path $config) {
    $text = Get-Content $config -Raw
    if ($text -match "<TitleId>([^<]*)") { $env:X2W_TITLE_ID = $Matches[1] }
    if ($text -match "<SizeMB>([^<]*)") { $env:X2W_STORAGE_MB = $Matches[1] }
}
if ($StoragePath) { $env:X2W_STORAGE_PATH = $StoragePath }
if ($InputScript) { $env:X2W_INPUT_SCRIPT = $InputScript }
if ($FakeUser) { $env:X2W_FAKE_USER = "1" }
if (-not $FakeUser -and $env:X2W_FAKE_USER -ne "1") {
    Write-Warning "Without -FakeUser, user API calls go to the host gaming runtime (docs/startup-runtime.md)."
}
if ($GpuMirror) { $env:X2W_XMEM_GPU_MIRROR = "1" }
if ($SlabBudgetMiB -gt 0) { $env:X2W_XMEM_SLAB_BUDGET_MIB = "$SlabBudgetMiB" }
if ($SlabDir) { $env:X2W_XMEM_SLAB_DIR = $SlabDir }
if ($HostCpu) { $env:X2W_HOST_CPU = "1" }
if ($SharedViews) { $env:X2W_XMEM_SHARED_VIEWS = "1" }
if ($TraceApi) { $env:X2W_TRACE_API = "1" }
if ($StartupBelow40) { $env:X2W_XMEM_STARTUP_BELOW_40 = "1" }
if ($Render) {
    New-Item -ItemType Directory -Force $ShaderCache | Out-Null
    if (-not (Test-Path Env:X2W_PRESENT_IN_TITLE_WINDOW)) { $env:X2W_PRESENT_IN_TITLE_WINDOW = "1" }
    $env:X2W_D3D12_RENDER = "1"; $env:X2W_DRAW_NULL_IDENTITY = "1"
    $env:X2W_SHADER_CACHE = (Resolve-Path $ShaderCache).Path
}
New-Item -ItemType Directory -Force $LogDir | Out-Null
if ($NativeLauncher) {
    & $NativeLauncher $Exe $DataDir $LogDir --timeout $Seconds --memory-gib $MemoryGiB --offline 1 `
        --trace-files 1 --continue-asserts $ContinueAsserts --fast-exit ([int]$FastExit.IsPresent) `
        --trace-calls ([int]$TraceCalls.IsPresent) --log-repeats $LogRepeats *> "$LogDir.out"
    $debuggerExit = $LASTEXITCODE
    Get-Content "$LogDir.out" -TotalCount 5
    exit $debuggerExit
}
$python = Join-Path $repo "out\pe-env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$translator = $null
if ($Translate -and -not $Render) { Write-Warning "-Translate needs -Render (no shader cache without it); ignored" }
if ($Translate -and $Render) {
    $translator = Start-Process -FilePath $python -PassThru -NoNewWindow `
        -ArgumentList "-B", "`"$(Join-Path $PSScriptRoot 'gfx2hlsl.py')`"", "--watch", "`"$env:X2W_SHADER_CACHE`"" `
        -RedirectStandardOutput (Join-Path $LogDir "translate.txt") -RedirectStandardError (Join-Path $LogDir "translate_err.txt")
}
$traceArgs = @()
if ($TraceCalls) { $traceArgs = @("--trace-calls", "1") }
try {
    & $python -B (Join-Path $PSScriptRoot "debug_launch.py") $Exe $DataDir $LogDir --timeout $Seconds --memory-gib $MemoryGiB `
        --trace-files 1 --continue-asserts $ContinueAsserts --fast-exit $(if ($FastExit) { 1 } else { 0 }) `
        --log-repeats $LogRepeats @traceArgs *> "$LogDir.out"
    $debuggerExit = $LASTEXITCODE
} finally {
    # taskkill /T needs a live parent PID to reach the watcher's workers.
    if ($translator -and -not $translator.HasExited) {
        # Stop the workers as well. The watcher or a worker can exit during taskkill;
        # in Windows PowerShell 5.1 its redirected stderr would then abort the launcher.
        & {
            $ErrorActionPreference = "Continue"
            & taskkill.exe /T /F /PID $translator.Id 2>&1 | Out-Null
        }
    }
}
Get-Content "$LogDir.out" -TotalCount 5
exit $debuggerExit
