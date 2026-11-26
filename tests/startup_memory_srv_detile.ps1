param([string] $Consumer, [string] $Backend, [string] $Fxc, [string] $LayoutCheck)
. "$PSScriptRoot/startup_detile_draw.ps1"
$ErrorActionPreference = 'Stop'
$env:X2W_TEXTURE_CENSUS = '1'
$env:X2W_TEXTURE_CENSUS_EVERY = '1'

# name, layout, dimension, DXGI format, width, height, depth/array size, mip count
$cases = @(
    @('bc1_256_9mips_4k_s', 0x105, 3, 71, 256, 256, 1, 9),
    @('bc1_256x128x2_8mips_64k_d', 0x10a, 3, 71, 256, 128, 2, 8),
    @('bc3_256_9mips_4k_s', 0x105, 3, 77, 256, 256, 1, 9),
    @('rgba16f_17cubed_3d_5mips', 0x105, 4, 10, 17, 17, 17, 5),
    @('rgba16f_100x60_3mips_linear', 0x100, 3, 10, 100, 60, 1, 3),
    @('bc1_132_8mips_4k_s', 0x105, 3, 71, 132, 132, 1, 8),
    @('bc3_132x68x2_8mips_64k_d', 0x10a, 3, 77, 132, 68, 2, 8)
)

function Test-Surface($spec, $mode, $override = 0) {
    $name, $layout, $dim, $format, $width, $height, $depth, $mips = $spec
    $dir = Join-Path $root "$name-$mode"
    [IO.Directory]::CreateDirectory($dir) | Out-Null
    $bc = $format -in 71, 77
    if ($dim -eq 4) { $kind = 'tex3d'; $type = 'Texture3D' }
    elseif ($depth -gt 1) { $kind = 'tex2da'; $type = 'Texture2DArray' }
    else { $kind = 'tex2d'; $type = 'Texture2D' }
    $body = [Text.StringBuilder]::new()
    $expected = 0
    for ($mip = 0; $mip -lt $mips; $mip++) {
        $w = [Math]::Max(1, $width -shr $mip)
        $h = [Math]::Max(1, $height -shr $mip)
        if ($bc) { $w = ($w + 3) -shr 2; $h = ($h + 3) -shr 2 }
        $slices = $depth; $d = 1
        if ($dim -eq 4) { $slices = 1; $d = [Math]::Max(1, $depth -shr $mip) }
        $count = $w * $h * $d
        for ($s = 0; $s -lt $slices; $s++) {
            $expected += $count
            $xy = if ($bc) { '4*x,4*y' } else { 'x,y' }
            $coord = switch ($kind) {
                'tex3d'  { "int4(x,y,z,$mip)" }
                'tex2da' { "int4($xy,$s,$mip)" }
                'tex2d'  { "int3($xy,$mip)" }
            }
            $tag = $mip + 8 * $s
            $test = if ($bc) {
                $alpha = if ($format -eq 77) { " || uint(round(c.a*255)) != $(16*$tag+5)u" } else { '' }
                "uint id=(uint(round(c.r*31))<<11)|(uint(round(c.g*63))<<5)|uint(round(c.b*31)); if (id!=(x|(y<<6)|(${tag}u<<12))$alpha) bad++;"
            } else { "if(any(c!=float4(x,y,z,$($mip+1)))) bad++;" }
            [void]$body.AppendLine("[loop] for(uint i=p; i<${count}u; i+=4096u){uint x=i%${w}u,y=(i/${w}u)%${h}u,z=i/$($w*$h)u;float4 c=t.Load($coord); $test n++;}")
        }
    }
    $shader = @"
${type}<float4> t : register(t0);
float4 ps(float4 pos : SV_Position) : SV_Target {
    uint p=uint(pos.y)*64u+uint(pos.x), bad=0, n=0;
    $body
    return float4(bad ? 0 : 1,n/255.0,0,1);
}
"@
    $env:X2W_MEMORY_SRV_DETILE = if ($mode -eq 'off') { $null } else { '1' }
    if ($override) { $layout = $override }
    $env:X2W_TEST_DETILE_DESC = '{0:x} {1} {2} {3} {4} {5} {6}' -f $layout,$dim,$format,$width,$height,$depth,$mips
    $env:X2W_TEST_DETILE_SRV = "$root/$name.bin"
    $pixels = Invoke-DetileDraw $dir $Consumer $Backend $Fxc $shader $kind
    if ($pixels.Length -ne 16384) { throw "$name read back $($pixels.Length) bytes" }
    $bad = 0; $checked = 0
    for ($i=0; $i -lt $pixels.Length; $i+=4) {
        if ($pixels[$i] -ne 255) { $bad++ }
        $checked += $pixels[$i+1]
    }
    $log = Get-Content "$dir/trap.log"
    $detiled = @($log -like '*x2w-memsrv detiled memory SRV*')
    $refused = @($log -like '*Device::slot70*reason=*')
    if ($checked -ne $expected) { throw "$name ($mode) shader checked $checked of $expected elements" }
    switch ($mode) {
        'on' {
            $lastSlot = @($log -like 'x2w-texcensus slot70 calls*')[-1]
            $lastDraw = @($log -like 'x2w-texcensus draw textures*')[-1]
            $layoutTag = 'layout 0x{0:x}' -f $layout
            if ($bad -or !$detiled.Count -or !($log -like "*$layoutTag*") -or
                $lastSlot -notlike '*stand-in buffer 0, detiled 2, refused 0*' -or $lastDraw -notlike '*detiled 1)*') {
                throw "$name mismatch pixels=$bad, slot=$lastSlot, draw=$lastDraw"
            }
        }
        'off' { if (!$bad -or $refused.Count -ne 2 -or $detiled.Count) { throw "$name switch-off control failed" } }
        'wrong' { if (!$bad -or !$detiled.Count) { throw "$name wrong-layout control failed" } }
        'declined' {
            $classes = @($log -like 'x2w-texcensus   2 x *')
            $counters = @($log -like 'x2w-texcensus detile declines of refused texture calls*')
            $first = @($log -like '*new refusal class*')
            if (!$bad -or $detiled.Count -or $classes[-1] -notlike '*detile: layout or element size*' -or
                $counters[-1] -notlike '*layout or element size 2*' -or $first.Count -ne 1 -or
                $first[0] -notlike '*detile: layout or element size;*') { throw "$name decline accounting failed" }
        }
    }
    Write-Host "$Backend $name ($mode) checked $checked elements, mismatch pixels=$bad"
}

$root = Join-Path ([IO.Path]::GetTempPath()) ('x2w-detile-' + [Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($root) | Out-Null
try {
    foreach ($case in $cases) {
        $name = $case[0]
        $surface = "$root/$name.bin"
        & $LayoutCheck --write-tiled $name $surface --addrlib
        if ($LASTEXITCODE -eq 77) {
            Write-Host "$name uses native layout output (plumbing check only)"
            & $LayoutCheck --write-tiled $name $surface
        }
        if ($LASTEXITCODE -ne 0) { throw "$name surface generation failed" }
        Test-Surface $case 'on'
    }
    Test-Surface $cases[0] 'off'
    foreach ($index in 0, 5, 6) { Test-Surface $cases[$index] 'wrong' 0x109 }
    # XOR display tiling is outside the single-sample address equations.
    Test-Surface @('bc1_132_8mips_4k_s', 0x11a, 3, 28, 64, 64, 1, 1) 'declined'
} finally {
    [IO.Directory]::Delete($root, $true)
}
