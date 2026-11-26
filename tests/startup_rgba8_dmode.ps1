param(
    [Parameter(Mandatory)] [string] $Consumer,
    [Parameter(Mandatory)] [string] $Backend,
    [Parameter(Mandatory)] [string] $Fxc
)
. "$PSScriptRoot/startup_detile_draw.ps1"
$ErrorActionPreference = 'Stop'
$dir = Join-Path ([IO.Path]::GetTempPath()) ('x2w-rgba8-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($dir) | Out-Null
try {
    # Fixed RGBA8 512x512 offsets from the RDNA2 RB+ 64 KiB display equation.
    # Each row gives mip, x, y and byte offset. Mips 3 through 7 share the tail block.
    $points = @(
        @(0, 4, 1, 0x60050), @(0, 510, 511, 0x15fff8),
        @(1, 4, 1, 0x20050), @(1, 254, 255, 0x5fff8),
        @(2, 4, 1, 0x10050), @(2, 126, 127, 0x1fff8),
        @(3, 4, 1, 0x8050),  @(3, 62, 63, 0xbff8),
        @(4, 4, 1, 0x4050),  @(4, 30, 31, 0x4ff8),
        @(5, 4, 1, 0x2050),  @(5, 14, 15, 0x23f8),
        @(6, 4, 1, 0x1050),  @(6, 6, 7, 0x10f8),
        @(7, 3, 1, 0x081c),  @(7, 2, 3, 0x0838)
    )
    $surface = New-Object byte[] 0x160000
    $colors = @()
    $loads = for ($i = 0; $i -lt $points.Count; $i++) {
        $mip, $x, $y, $offset = $points[$i]
        $rgba = [byte[]]@((63 + 9 * $i), (253 - 11 * $i), (21 + 5 * $i), (240 - 7 * $i))
        $rgba.CopyTo($surface, $offset)
        $colors += ,$rgba
        "case ${i}u: return tex.Load(int3($x,$y,$mip));"
    }
    $env:X2W_TEST_DETILE_SRV = "$dir/surface"
    [IO.File]::WriteAllBytes($env:X2W_TEST_DETILE_SRV, $surface)
    $env:X2W_TEST_DETILE_DESC = '10a 3 28 512 512 1 8'
    $env:X2W_MEMORY_SRV_DETILE = '1'
    $shader = @"
Texture2D<float4> tex : register(t0);
float4 ps(float4 pos : SV_Position) : SV_Target {
    switch (uint(pos.x) % 16) {
        $($loads -join [Environment]::NewLine)
    }
    return 0;
}
"@
    $pixels = Invoke-DetileDraw $dir $Consumer $Backend $Fxc $shader 'tex2d'
    if ($pixels.Length -ne 64 * 64 * 4) { throw "Unexpected readback size $($pixels.Length)" }
    for ($i = 0; $i -lt 4096; $i++) {
        $probe = $i % 16
        for ($channel = 0; $channel -lt 4; $channel++) {
            $value = $pixels[4 * $i + $channel]
            $expected = $colors[$probe][$channel]
            if ($value -ne $expected) {
                Get-Content "$dir/trap.log" | Select-String -Pattern 'memsrv|slot70|x2w-draw' | Select-Object -Last 18 | Out-Host
                throw "pixel $i, mip/x/y/offset $($points[$probe]), channel $channel sampled $value, expected $expected"
            }
        }
    }
    Write-Output "$Backend sampled 16 RB+ addresses across all eight RGBA8 mips"
} finally {
    [IO.Directory]::Delete($dir, $true)
}
