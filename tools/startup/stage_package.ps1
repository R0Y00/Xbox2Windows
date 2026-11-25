<#
.SYNOPSIS
Stages a small, disposable executable directory for an offline startup run.

.DESCRIPTION
Copies one executable and the DLLs beside it into a separate run directory. Package assets
remain in PackageDir and are supplied as run.ps1's -DataDir. No recursive copy is performed.
#>
param(
    [Parameter(Mandatory)] [string]$PackageDir,
    [Parameter(Mandatory)] [string]$Exe,
    [Parameter(Mandatory)] [string]$RunDir
)
$ErrorActionPreference = "Stop"

$package = (Resolve-Path -LiteralPath $PackageDir).Path.TrimEnd('\')
$source = if ([IO.Path]::IsPathRooted($Exe)) {
    (Resolve-Path -LiteralPath $Exe).Path
} else {
    (Resolve-Path -LiteralPath (Join-Path $package $Exe)).Path
}
$sourceDir = Split-Path -Parent $source
$provider = $null
$drive = $null
$run = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath(
    $RunDir, [ref]$provider, [ref]$drive).TrimEnd('\')
if ($provider.Name -ne 'FileSystem') { throw "RunDir must be a filesystem path." }
if (-not (Test-Path -LiteralPath $source -PathType Leaf) -or [IO.Path]::GetExtension($source) -ine '.exe') {
    throw "Exe must name an executable file."
}

function Test-PathInside([string]$Path, [string]$Parent) {
    $prefix = $Parent.TrimEnd('\') + '\'
    return $Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
}

if (-not (Test-PathInside $source $package)) {
    throw "Exe must be inside PackageDir."
}
if ($run.Equals($package, [StringComparison]::OrdinalIgnoreCase) -or
    (Test-PathInside $run $package) -or (Test-PathInside $package $run)) {
    throw "RunDir must be separate from PackageDir and its parent directories."
}

# Do not write through an existing junction/symlink or overwrite a previous package.
for ($ancestor = $run; $ancestor; $ancestor = Split-Path -Parent $ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and
        ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "RunDir must not pass through a reparse point."
    }
}
if ((Test-Path -LiteralPath $run) -and
    (-not (Test-Path -LiteralPath $run -PathType Container) -or
     @(Get-ChildItem -LiteralPath $run -Force).Count)) {
    throw "RunDir must be absent or empty. Use a fresh directory for staging."
}

New-Item -ItemType Directory -Force -Path $run | Out-Null
Copy-Item -LiteralPath $source -Destination $run -Force
$dlls = @(Get-ChildItem -LiteralPath $sourceDir -File -Filter '*.dll' |
    Where-Object { $_.Extension -ieq '.dll' })
foreach ($dll in $dlls) {
    Copy-Item -LiteralPath $dll.FullName -Destination $run -Force
}
Write-Output "Staged executable and $($dlls.Count) adjacent DLLs in $run; package assets remain in $package."
