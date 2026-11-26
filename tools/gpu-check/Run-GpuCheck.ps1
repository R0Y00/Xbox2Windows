#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('hardware', 'warp')]
    [string]$Backend = 'hardware',
    [string]$ReportPath,
    [Alias('Adapter')][string]$AdapterFilter,
    [switch]$NoShaders
)

$ErrorActionPreference = 'Stop'
if (-not $ReportPath) { $ReportPath = Join-Path $PSScriptRoot "gpu-report-$Backend.md" }
$binary = (Get-Item -LiteralPath (Join-Path $PSScriptRoot 'offscreen_check.exe')).FullName
$revision = (Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'SOURCE_COMMIT.txt')).Trim()
if ($revision -notmatch '^[0-9a-f]{40}$') { throw 'The package is missing a valid source revision.' }
$manifest = Join-Path $PSScriptRoot 'cases/manifest.tsv'
if (-not $NoShaders) {
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'The package is missing cases/manifest.tsv.' }
    $rows = @(Get-Content -LiteralPath $manifest | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if (-not $rows.Count) { throw 'The shader case manifest is empty.' }
}
$windows = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$threadUrl = 'https://github.com/R0Y00/Xbox2Windows/issues/1'

function Get-GpuField {
    param(
        [string[]]$Blocks,
        [Parameter(Mandatory = $true)][string]$Pattern
    )
    foreach ($block in $Blocks) {
        if ($block -match $Pattern) { return $Matches[1].Trim() }
    }
    return $null
}

function Format-GpuValue {
    param([string]$Value)
    if ($Value) { return $Value }
    return 'not reported'
}

$cases = [System.Collections.Generic.List[object]]::new()
$failed = $false
foreach ($test in 'dual-uav', 'texture-mips', 'bvh-waves', 'bvh-wave64', 'bvh-differential', 'bvh-packed', 'bvh-atomics', 'bvh-numeric', 'bvh-root-roundtrip', 'dxr-pipeline', 'dxr-update') {
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $binary
    $process.StartInfo.Arguments = "$Backend --$test"
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try {
        if ($AdapterFilter -and $Backend -eq 'hardware') { $process.StartInfo.EnvironmentVariables['XBOX2WINDOWS_ADAPTER'] = $AdapterFilter }
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $timedOut = -not $process.WaitForExit(60000)
        if ($timedOut) {
            $process.Kill()
            $process.WaitForExit()
        }
        $details = ($stdout.Result.Trim() + [Environment]::NewLine + $stderr.Result.Trim()).Trim()
        if ($timedOut) { $status = 'FAIL'; $details += "`nTimed out after 60 seconds." }
        elseif ($process.ExitCode -eq 0) { $status = 'PASS' }
        elseif ($process.ExitCode -eq 77) { $status = 'SKIP' }
        else { $status = 'FAIL'; $details += "`nProcess exit code: $($process.ExitCode)" }
    } catch {
        $status = 'FAIL'
        $details = $_.Exception.Message
    } finally {
        $process.Dispose()
    }
    if ($status -eq 'FAIL') { $failed = $true }
    $caseName = $test.Replace('-', '_')
    $cases.Add([pscustomobject]@{
        Name = "${caseName}_${Backend}"
        Status = $status
        Details = $details
    })
}

function Invoke-ShaderCase {
    param([string]$Id, [string]$Expect, [string]$Mismatch)
    $files = foreach ($extension in 'cso', 'case') { Join-Path $PSScriptRoot "cases/$Id.$extension" }
    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath $file)) { return [pscustomobject]@{ Status = 'FAIL'; Note = "missing $file" } }
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo.FileName = $binary
    $process.StartInfo.Arguments = "$Backend --compute-store-case `"$($files[0])`" `"$($files[1])`""
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.CreateNoWindow = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.RedirectStandardError = $true
    try {
        if ($AdapterFilter -and $Backend -eq 'hardware') { $process.StartInfo.EnvironmentVariables['XBOX2WINDOWS_ADAPTER'] = $AdapterFilter }
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(60000)) {
            $process.Kill()
            $process.WaitForExit()
            return [pscustomobject]@{ Status = 'FAIL'; Note = 'timed out after 60 seconds' }
        }
        $code = $process.ExitCode
        # Adapter enumeration notices are not diagnostics of the case.
        $lines = @(($stderr.Result -split "`r?`n") | Where-Object { $_ -and -not $_.StartsWith('XBOX2WINDOWS_ADAPTER: skipping ') })
    } catch {
        return [pscustomobject]@{ Status = 'FAIL'; Note = $_.Exception.Message }
    } finally {
        $process.Dispose()
    }
    if ($code -eq 77) { return [pscustomobject]@{ Status = 'SKIP'; Note = 'capability skip (exit 77)' } }
    $first = if ($lines.Count) { $lines[0] } else { '' }
    if ($Expect -eq 'pass') {
        if ($code -eq 0 -and -not $lines.Count) { return [pscustomobject]@{ Status = 'PASS'; Note = '' } }
        return [pscustomobject]@{ Status = 'FAIL'; Note = "exit $code $first" }
    }
    # A mutation control must be rejected: a card that reproduces the unmutated result would hide a real difference.
    $rejected = $code -eq 1 -and ($lines | Where-Object { $_ -like '*expected=*' }) -and (-not $Mismatch -or $first -eq $Mismatch)
    if ($rejected) { return [pscustomobject]@{ Status = 'PASS'; Note = 'control rejected' } }
    return [pscustomobject]@{ Status = 'FAIL'; Note = "control not rejected: exit $code $first" }
}

# Compiled gfx2hlsl compute cases recorded by tools/record_gpu_cases.py: each one is dispatched on this adapter and
# its output words are compared inside offscreen_check. They need no compiler here. One report section sums them up.
if (-not $NoShaders) {
    $shaderLines = [System.Collections.Generic.List[string]]::new()
    $counts = @{ PASS = 0; FAIL = 0; SKIP = 0 }
    $shaderLines.Add('translated compute shaders replayed from the package (compiler, case, result)')
    foreach ($row in $rows) {
        $field = $row -split "`t"
        $result = Invoke-ShaderCase -Id $field[0] -Expect $field[2] -Mismatch $field[3]
        $counts[$result.Status]++
        $shaderLines.Add(("{0} {1} {2}{3}" -f $field[1], $field[0], $result.Status, $(if ($result.Note) { ": $($result.Note)" } else { '' })))
    }
    $shaderLines.Insert(1, "cases: $($counts.PASS) pass, $($counts.FAIL) fail, $($counts.SKIP) skip of $($rows.Count)")
    $shaderStatus = if ($counts.FAIL) { 'FAIL' } elseif ($counts.PASS) { 'PASS' } else { 'SKIP' }
    if ($shaderStatus -eq 'FAIL') { $failed = $true }
    $cases.Add([pscustomobject]@{
        Name = "gfx2hlsl_shaders_${Backend}"
        Status = $shaderStatus
        Details = ($shaderLines -join [Environment]::NewLine)
    })
}

$blocks = @($cases | ForEach-Object { $_.Details })
$adapter = Get-GpuField $blocks '(?m)^adapter=(.+?)\s+vendor='
$vendor = Get-GpuField $blocks '(?m)^adapter=.+?\s+vendor=([0-9A-Fa-f]+)\s+device='
$device = Get-GpuField $blocks '(?m)^adapter=.+?\s+vendor=[0-9A-Fa-f]+\s+device=([0-9A-Fa-f]+)'
$driver = Get-GpuField $blocks '(?m)^driver=(.+)$'
$debugLayer = Get-GpuField $blocks '(?m)^backend=\S+\s+debug_layer=(\S+)'
$laneRange = Get-GpuField $blocks '(?m)^native_lane_range=(\S+)'
$viewHeapLimit = Get-GpuField $blocks '(?m)^max_view_descriptor_heap_size=([0-9]+)\r?$'

$report = [System.Collections.Generic.List[string]]::new()
$report.Add('## Xbox2Windows GPU report')
$report.Add('')
$report.Add("Source commit: $revision")
$report.Add("Windows build: $($windows.CurrentBuildNumber).$($windows.UBR)")
$report.Add("Backend: $Backend")
if ($AdapterFilter -and $Backend -eq 'hardware') { $report.Add("Adapter filter: $AdapterFilter") }
$report.Add('')
$report.Add('### Summary')
$report.Add('')
if ($adapter -and $vendor -and $device) {
    $report.Add("- Adapter: $adapter (vendor=$vendor device=$device)")
} elseif ($adapter) {
    $report.Add("- Adapter: $adapter")
} else {
    $report.Add('- Adapter: not reported')
}
$report.Add("- Driver: $(Format-GpuValue $driver)")
$report.Add("- Debug layer: $(Format-GpuValue $debugLayer)")
$report.Add("- Native lane range: $(Format-GpuValue $laneRange)")
$report.Add("- Shader-visible view heap limit: $(Format-GpuValue $viewHeapLimit)")
$report.Add('')
$report.Add('| Test | Result |')
$report.Add('|---|---|')
foreach ($case in $cases) {
    $report.Add("| $($case.Name) | $($case.Status) |")
}
$report.Add('')

foreach ($case in $cases) {
    $report.Add("### $($case.Name) - $($case.Status)")
    $report.Add('')
    $report.Add('```text')
    $report.Add($case.Details)
    $report.Add('```')
    $report.Add('')
}

$report.Add('### Submit this report')
$report.Add('')
$report.Add('Paste the full contents of this file as a comment on the hardware coverage thread:')
$report.Add($threadUrl)
$report.Add('')
$report.Add('Do not open a new GitHub issue. Say in the comment if you would like your name or handle in the coverage table.')
$report.Add('')

$report | Write-Output
[IO.File]::WriteAllLines($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ReportPath), $report)
Write-Output "Saved report: $ReportPath"
Write-Output "Paste this report as a comment on $threadUrl - do not open a new issue."
if ($failed) { exit 1 }
exit 0
