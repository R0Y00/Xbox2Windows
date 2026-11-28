param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is required to create labels.'
}

$labels = @(
    @{ Name = 'research';         Color = '5319e7'; Description = 'A compatibility question with a reproducible test.' },
    @{ Name = 'build';            Color = 'bfd4f2'; Description = 'CMake, toolchain, packaging or build-script work.' },
    @{ Name = 'ci';               Color = 'c5def5'; Description = 'GitHub Actions and automation.' },
    @{ Name = 'tests';            Color = 'd4c5f9'; Description = 'Fixtures, coverage and test infrastructure.' },
    @{ Name = 'area/cpu';         Color = '1d76db'; Description = 'x64 lifting, ABI, unwind and PE execution.' },
    @{ Name = 'area/gpu';         Color = '0e8a16'; Description = 'D3D12, BVH, DXR and GPU coverage.' },
    @{ Name = 'area/shader';      Color = 'b60205'; Description = 'gfx1013 decoding, HLSL lowering and DXIL.' },
    @{ Name = 'area/ffx';         Color = 'd93f0b'; Description = 'FidelityFX loader, upscaler and FSR4.' },
    @{ Name = 'area/textures';    Color = 'fbca04'; Description = 'BC7, BCPack, mip tails and texture views.' },
    @{ Name = 'area/script';      Color = '006b75'; Description = 'RPF, PSIN, state trees and script replay.' },
    @{ Name = 'area/pe';          Color = '5319e7'; Description = 'PE mapping, imports, intake and dependency paths.' },
    @{ Name = 'area/runtime';     Color = '0052cc'; Description = 'Heap, memory queries and Windows shims.' },
    @{ Name = 'area/tooling';     Color = 'bfdadc'; Description = 'Python tools and fixtures.' },
    @{ Name = 'area/meta';        Color = '6f42c1'; Description = 'Contribution workflow, templates, labels and repository process.' },
    @{ Name = 'needs-repro';      Color = 'e4e669'; Description = 'A reproducible input or command is missing.' },
    @{ Name = 'needs-hardware';   Color = 'd4c5f9'; Description = 'A physical GPU or adapter-specific result is required.' },
    @{ Name = 'needs-windows';    Color = 'c2e0c6'; Description = 'The task requires the Windows build or D3D12 runtime.' },
    @{ Name = 'needs-game-files'; Color = 'f9d0c4'; Description = 'The task needs a local package; public fixtures are not enough.' },
    @{ Name = 'blocked';          Color = '000000'; Description = 'Work cannot proceed until another task or decision is resolved.' }
)

$response = gh label list --limit 100 --json name,color,description
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to read the repository labels with gh.'
}

$existing = @{}
foreach ($item in ($response | ConvertFrom-Json)) {
    $existing[$item.name] = $item
}

$proposedNames = @($labels | ForEach-Object { $_.Name })

if ($DryRun) {
    Write-Output '| Action | Label | Color | Description | Change |'
    Write-Output '|---|---|---|---|---|'
}

foreach ($label in $labels) {
    $action = 'CREATE'
    $change = 'new label'
    if ($existing.ContainsKey($label.Name)) {
        $current = $existing[$label.Name]
        $changes = @()
        if ($current.color -ne $label.Color) {
            $changes += "color #$($current.color) -> #$($label.Color)"
        }
        if ($current.description -ne $label.Description) {
            $changes += "description '$($current.description)' -> '$($label.Description)'"
        }
        if ($changes.Count -eq 0) {
            $action = 'UNCHANGED'
            $change = 'no change'
        } else {
            $action = 'UPDATE'
            $change = $changes -join '; '
        }
    }

    if ($DryRun) {
        Write-Output ("| {0} | {1} | #{2} | {3} | {4} |" -f $action, $label.Name, $label.Color, $label.Description, $change)
        continue
    }

    if ($action -eq 'UNCHANGED') {
        continue
    }
    if ($action -eq 'CREATE') {
        gh label create $label.Name --color $label.Color --description $label.Description
    } else {
        gh label edit $label.Name --color $label.Color --description $label.Description
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply label $($label.Name)."
    }
}

if ($DryRun) {
    $untouched = @($existing.Keys | Where-Object { $proposedNames -notcontains $_ } | Sort-Object)
    Write-Output ''
    if ($untouched.Count -eq 0) {
        Write-Output 'Existing labels not in the proposal: none.'
    } else {
        Write-Output ('Existing labels not in the proposal (unchanged): ' + ($untouched -join ', '))
    }
}
