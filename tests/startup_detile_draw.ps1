function Invoke-DetileDraw($Dir, $Consumer, $Backend, $Fxc, $PixelShader, $Kind) {
    $env:X2W_GPU_ABI = '0'
    $env:X2W_D3D12_PERMISSIVE = '1'
    $env:X2W_D3D12_DEVICE_NOOP = '1'
    $env:X2W_D3D12_LIST_NOOP = '1'
    $env:X2W_D3D12_RENDER = '1'
    $env:X2W_D3D12_DEBUG = '0'
    $env:X2W_TRAP_LOG = "$dir/trap.log"
    $env:X2W_SHADER_CACHE = $dir
    # Slot 86 recognizes the console container signature and hashes the opaque stage bytes.
    # The desktop stage binaries under that key are compiled from the fixture's HLSL below.
    $container = New-Object byte[] 32
    [Text.Encoding]::ASCII.GetBytes('DXBC').CopyTo($container, 0)
    foreach ($stage in 'vs', 'ps') {
        [Text.Encoding]::ASCII.GetBytes($stage).CopyTo($container, 4)
        [IO.File]::WriteAllBytes("$dir/$stage", $container)
    }
    & $Consumer "$dir/vs" "$dir/ps" "$dir/root" | Out-Host
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $keys = @(Get-ChildItem $dir -Filter 'req_*' | ForEach-Object {
        if ($_.Name -match '^req_([0-9a-f]{16})\.(rtv|a)$') { $Matches[1] }
    } | Sort-Object -Unique)
    if ($keys.Count -ne 1) { throw "Expected one captured pipeline, got $($keys.Count)" }
    $prefix = Join-Path $dir $keys[0]
    $source = @"
float4 vs(uint id : SV_VertexID) : SV_Position {
    return float4(2.0 * float2((id << 1) & 2, id & 2) - 1.0, 0, 1);
}
$PixelShader
"@
    [IO.File]::WriteAllText("$dir/test.hlsl", $source)
    foreach ($stage in 'vs', 'ps') {
        & $Fxc /nologo /T "${stage}_5_0" /E $stage /Fo "$prefix.$stage.cso" "$dir/test.hlsl" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Failed to compile $stage" }
    }
    [IO.File]::WriteAllLines("$prefix.txt", @('x2w-manifest 1', 'ud 1', 'udsrc 0 rect', "srv 0 $Kind table 0 0", 'params 0', 'targets 0'))
    & $Consumer "$dir/vs" "$dir/ps" "$dir/root" $Backend "$dir/pixels" | Out-Host
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    return ,([IO.File]::ReadAllBytes("$dir/pixels"))
}
