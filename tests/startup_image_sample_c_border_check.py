"""Comparison sampling with CLAMP_BORDER and fixed border colors in tools/startup/gfx2hlsl.py (Part of #2023, item F).

  (no arguments)                       translation, manifest and refusal controls, pure Python
  binary <llvm-mc>                     reassemble every fixture source and compare it with the recorded words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate, compile (DXC ps_6_0, and FXC ps_5_1 when given) and draw
                                       against the 2x2 R32_FLOAT and R32G32B32A32_FLOAT fixtures: checks
                                       in-bounds, negative/positive out-of-bounds, mixed X/Y clamp+border,
                                       bilinear border-edge PCF, all three fixed border colors, and mutants

SQ_IMG_SAMP_WORD0 bits 2:0 (CLAMP_X) and 5:3 (CLAMP_Y) select CLAMP_LAST_TEXEL (2) or CLAMP_BORDER (6).
When either axis uses CLAMP_BORDER, SQ_IMG_SAMP_WORD3 bits 31:30 (BORDER_COLOR_TYPE, Mesa gfx10-rsrc.json)
select TRANS_BLACK (0, depth 0.0), OPAQUE_BLACK (1, depth 0.0, alpha 1.0) or OPAQUE_WHITE (2, depth 1.0).
Value 3 (REGISTER, indexed by BORDER_COLOR_PTR in bits 11:0) has no D3D12 static-sampler equivalent and is
refused whenever X or Y uses CLAMP_BORDER, while remaining ignored when both X and Y use CLAMP_LAST_TEXEL.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import disasm_rdna2 as D  # noqa: E402
import gfx2hlsl as G  # noqa: E402
import pixel_aot as P  # noqa: E402
from startup_image_3d_check import (  # noqa: E402
    FULLSCREEN_VS,
    VS_WORDS,
    CompileQueue,
    assemble,
    compile_hlsl,
    root_blob,
)

CLAMP_LAST_TEXEL = 2
CLAMP_BORDER = 6
LESS = 1 << 12
TRUNC_COORD = 1 << 27
BILINEAR_W2 = (1 << 20) | (1 << 22)

TRANS_BLACK = 0 << 30
OPAQUE_BLACK = 1 << 30
OPAQUE_WHITE = 2 << 30
REGISTER_BORDER = 3 << 30

IMG_SAMPLE_LZ = 0x27
IMG_SAMPLE_C = 0x28
IMG_SAMPLE_C_LZ = 0x2F
DIM_2D = 1
DIM_2D_ARRAY = 5


INLINE_FLOATS = {
    0x3f000000: 0xf0,
    0xbf000000: 0xf1,
    0x3f800000: 0xf2,
    0xbf800000: 0xf3,
    0x40000000: 0xf4,
    0xc0000000: 0xf5,
    0x40800000: 0xf6,
    0xc0800000: 0xf7,
    0x3e22f983: 0xf8,
}


def s_mov_b32(sdst, imm):
    """s_mov_b32 with the operand coding llvm-mc picks: an inline constant when one exists."""
    imm &= 0xffffffff
    code = INLINE_FLOATS.get(
        imm,
        0x80 + imm if imm <= 64 else 0xc0 + (1 << 32) - imm if imm >= 0xfffffff0 else None,
    )
    if code is not None:
        return struct.pack('<I', 0xbe800300 | (sdst << 16) | code)
    return struct.pack('<II', 0xbe8003ff | (sdst << 16), imm)


def v_mov_sgpr(vdst, ssrc):
    return struct.pack('<I', 0x7e000200 | (vdst << 17) | (ssrc & 0xff))


def v_mov_inline(vdst, src_code):
    return struct.pack('<I', 0x7e000200 | (vdst << 17) | (src_code & 0x1ff))


def v_mov_lit(vdst, bits):
    return struct.pack('<II', 0x7e0002ff | (vdst << 17), bits & 0xffffffff)


def mimg_nsa(op, vdst, addrs, *, dmask=0x1, dim=DIM_2D, srsrc=16, ssamp=24):
    assert len(addrs) in (3, 4)
    nsa = 1
    w0 = 0xf0000000 | ((op & 0x7f) << 18) | (nsa << 1) | ((dim & 0x7) << 3) | ((dmask & 0xf) << 8)
    w1 = (addrs[0] & 0xff) | ((vdst & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16) | (((ssamp // 4) & 0x1f) << 21)
    ext = bytes(addrs[1:]) + bytes(4 - len(addrs[1:]))
    return struct.pack('<II', w0, w1) + ext


def mimg_seq(op, vdst, vaddr, *, dmask=0xf, dim=DIM_2D, srsrc=16, ssamp=24):
    w0 = 0xf0000000 | ((op & 0x7f) << 18) | ((dim & 0x7) << 3) | ((dmask & 0xf) << 8)
    w1 = (vaddr & 0xff) | ((vdst & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16) | (((ssamp // 4) & 0x1f) << 21)
    return struct.pack('<II', w0, w1)


S_LOAD_DWORDX8 = bytes.fromhex('04040cf4000000fa')
S_WAIT_LGKM = bytes.fromhex('7fc08cbf')
S_WAIT_VM = bytes.fromhex('703f8cbf')
EXP_MRT0_V4_V7 = bytes.fromhex('0f1800f804050607')
S_ENDPGM = bytes.fromhex('000081bf')


def fmt_imm(val):
    return '0' if val == 0 else '%#x' % (val & 0xffffffff)


def build_point_program(w0, w1=0, w2=0, w3=0, op=IMG_SAMPLE_C_LZ, array=False):
    """Four probes in one pixel stage: (0.75, 0.75), (-0.25, 0.25), (1.25, 0.25), (0.25, 1.25); v0 = s9 (ref)."""
    mnemonic = 'image_sample_c' if op == IMG_SAMPLE_C else 'image_sample_c_lz'
    dim_name = 'SQ_RSRC_IMG_2D_ARRAY' if array else 'SQ_RSRC_IMG_2D'
    dim_code = DIM_2D_ARRAY if array else DIM_2D
    lines = [
        's_load_dwordx8 s[16:23], s[8:9], 0x0',
        's_mov_b32 s24, %s' % fmt_imm(w0),
        's_mov_b32 s25, %s' % fmt_imm(w1),
        's_mov_b32 s26, %s' % fmt_imm(w2),
        's_mov_b32 s27, %s' % fmt_imm(w3),
        'v_mov_b32 v0, s9',
        'v_mov_b32 v1, 0x3f400000',
        'v_mov_b32 v2, 0x3e800000',
        'v_mov_b32 v3, 0xbe800000',
        'v_mov_b32 v8, 0x3fa00000',
    ]
    code = bytearray(S_LOAD_DWORDX8)
    for reg, val in ((24, w0), (25, w1), (26, w2), (27, w3)):
        code += s_mov_b32(reg, val)
    code += v_mov_sgpr(0, 9)
    code += v_mov_lit(1, 0x3f400000)
    code += v_mov_lit(2, 0x3e800000)
    code += v_mov_lit(3, 0xbe800000)
    code += v_mov_lit(8, 0x3fa00000)
    if array:
        lines.append('v_mov_b32 v9, 0')
        code += v_mov_inline(9, 128)
    lines.append('s_waitcnt lgkmcnt(0)')
    code += S_WAIT_LGKM
    probes = ((4, (0, 1, 1)), (5, (0, 3, 2)), (6, (0, 8, 2)), (7, (0, 2, 8)))
    for vdst, uv in probes:
        addrs = uv + ((9,) if array else ())
        reg_list = ', '.join('v%d' % r for r in addrs)
        lines.append('%s v%d, [%s], s[16:23], s[24:27] dmask:0x1 dim:%s' % (mnemonic, vdst, reg_list, dim_name))
        code += mimg_nsa(op, vdst, addrs, dmask=0x1, dim=dim_code)
    lines += ['s_waitcnt vmcnt(0)', 'exp mrt0 v4, v5, v6, v7 done vm', 's_endpgm']
    code += S_WAIT_VM + EXP_MRT0_V4_V7 + S_ENDPGM
    return '\n'.join(lines) + '\n', bytes(code)


def build_bilinear_program(w0, w2=BILINEAR_W2, w3=OPAQUE_WHITE):
    """Four bilinear probes on 2x2 R32_FLOAT: (0.5, 0.5), (0.0, 0.25), (0.25, 0.0), (0.0, 0.0); v0 = s9 (ref)."""
    lines = [
        's_load_dwordx8 s[16:23], s[8:9], 0x0',
        's_mov_b32 s24, %s' % fmt_imm(w0),
        's_mov_b32 s25, 0',
        's_mov_b32 s26, %s' % fmt_imm(w2),
        's_mov_b32 s27, %s' % fmt_imm(w3),
        'v_mov_b32 v0, s9',
        'v_mov_b32 v1, 0.5',
        'v_mov_b32 v2, 0',
        'v_mov_b32 v3, 0x3e800000',
        's_waitcnt lgkmcnt(0)',
        'image_sample_c_lz v4, [v0, v1, v1], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
        'image_sample_c_lz v5, [v0, v2, v3], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
        'image_sample_c_lz v6, [v0, v3, v2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
        'image_sample_c_lz v7, [v0, v2, v2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D',
        's_waitcnt vmcnt(0)',
        'exp mrt0 v4, v5, v6, v7 done vm',
        's_endpgm',
    ]
    code = bytearray(S_LOAD_DWORDX8)
    for reg, val in ((24, w0), (25, 0), (26, w2), (27, w3)):
        code += s_mov_b32(reg, val)
    code += v_mov_sgpr(0, 9)
    code += v_mov_inline(1, 240)  # 0.5
    code += v_mov_inline(2, 128)  # 0
    code += v_mov_lit(3, 0x3e800000)
    code += S_WAIT_LGKM
    for vdst, addrs in ((4, (0, 1, 1)), (5, (0, 2, 3)), (6, (0, 3, 2)), (7, (0, 2, 2))):
        code += mimg_nsa(IMG_SAMPLE_C_LZ, vdst, addrs, dmask=0x1, dim=DIM_2D)
    code += S_WAIT_VM + EXP_MRT0_V4_V7 + S_ENDPGM
    return '\n'.join(lines) + '\n', bytes(code)


def build_rgba_border_program(w3):
    """Ordinary image_sample_lz at out-of-bounds (1.25, 0.25) with CLAMP_BORDER to verify RGBA static border color."""
    w0 = CLAMP_BORDER | (CLAMP_BORDER << 3)
    lines = [
        's_load_dwordx8 s[16:23], s[8:9], 0x0',
        's_mov_b32 s24, %s' % fmt_imm(w0),
        's_mov_b32 s25, 0',
        's_mov_b32 s26, 0',
        's_mov_b32 s27, %s' % fmt_imm(w3),
        'v_mov_b32 v1, 0x3fa00000',
        'v_mov_b32 v2, 0x3e800000',
        's_waitcnt lgkmcnt(0)',
        'image_sample_lz v[4:7], v[1:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D',
        's_waitcnt vmcnt(0)',
        'exp mrt0 v4, v5, v6, v7 done vm',
        's_endpgm',
    ]
    code = bytearray(S_LOAD_DWORDX8)
    for reg, val in ((24, w0), (25, 0), (26, 0), (27, w3)):
        code += s_mov_b32(reg, val)
    code += v_mov_lit(1, 0x3fa00000)
    code += v_mov_lit(2, 0x3e800000)
    code += S_WAIT_LGKM
    code += mimg_seq(IMG_SAMPLE_LZ, 4, 1, dmask=0xf, dim=DIM_2D)
    code += S_WAIT_VM + EXP_MRT0_V4_V7 + S_ENDPGM
    return '\n'.join(lines) + '\n', bytes(code)


POINT_BORDER_W0 = CLAMP_BORDER | (CLAMP_BORDER << 3) | LESS | TRUNC_COORD
POINT_BORDER_X_CLAMP_Y_W0 = CLAMP_BORDER | (CLAMP_LAST_TEXEL << 3) | LESS | TRUNC_COORD
POINT_CLAMP_X_BORDER_Y_W0 = CLAMP_LAST_TEXEL | (CLAMP_BORDER << 3) | LESS | TRUNC_COORD
POINT_CLAMP_W0 = CLAMP_LAST_TEXEL | (CLAMP_LAST_TEXEL << 3) | LESS | TRUNC_COORD
BILINEAR_BORDER_W0 = CLAMP_BORDER | (CLAMP_BORDER << 3) | LESS

FIXTURES = {
    'border_white': build_point_program(POINT_BORDER_W0, w3=OPAQUE_WHITE),
    'border_trans_black': build_point_program(POINT_BORDER_W0, w3=TRANS_BLACK),
    'border_opaque_black': build_point_program(POINT_BORDER_W0, w3=OPAQUE_BLACK | 0xfff),
    'border_opaque_black_inline': build_point_program(POINT_BORDER_W0, w3=OPAQUE_BLACK),
    'border_x_clamp_y': build_point_program(POINT_BORDER_X_CLAMP_Y_W0, w3=OPAQUE_WHITE),
    'clamp_x_border_y': build_point_program(POINT_CLAMP_X_BORDER_Y_W0, w3=OPAQUE_WHITE),
    'border_c_max0': build_point_program(POINT_BORDER_W0, w1=0, w3=OPAQUE_WHITE, op=IMG_SAMPLE_C),
    'border_array': build_point_program(POINT_BORDER_W0, w3=OPAQUE_WHITE, array=True),
    'clamp_ignored_reg_border': build_point_program(POINT_CLAMP_W0, w3=REGISTER_BORDER | 0xfff),
    'bilinear_border_white': build_bilinear_program(BILINEAR_BORDER_W0, w2=BILINEAR_W2, w3=OPAQUE_WHITE),
    'rgba_trans_black': build_rgba_border_program(TRANS_BLACK),
    'rgba_opaque_black': build_rgba_border_program(OPAQUE_BLACK),
    'rgba_opaque_white': build_rgba_border_program(OPAQUE_WHITE),
    # Refused variants.
    'refuse_reg_border_xy': build_point_program(POINT_BORDER_W0, w3=REGISTER_BORDER),
    'refuse_reg_border_x': build_point_program(POINT_BORDER_X_CLAMP_Y_W0, w3=REGISTER_BORDER),
    'refuse_reg_border_y': build_point_program(POINT_CLAMP_X_BORDER_Y_W0, w3=REGISTER_BORDER),
    'refuse_half_border_x': build_point_program(4 | (CLAMP_LAST_TEXEL << 3) | LESS | TRUNC_COORD, w3=OPAQUE_WHITE),
    'refuse_mirror_once_border_y': build_point_program(CLAMP_LAST_TEXEL | (7 << 3) | LESS | TRUNC_COORD, w3=OPAQUE_WHITE),
    'refuse_mirror_once_x': build_point_program(3 | (CLAMP_BORDER << 3) | LESS | TRUNC_COORD, w3=OPAQUE_WHITE),
    'refuse_mirror_y': build_point_program(CLAMP_BORDER | (1 << 3) | LESS | TRUNC_COORD, w3=OPAQUE_WHITE),
    'refuse_w3_reserved': build_point_program(POINT_BORDER_W0, w3=OPAQUE_WHITE | (1 << 12)),
}


def pipeline(name):
    return G.Pipeline(VS_WORDS, FIXTURES[name][1], root_blob())


def manifest_sampler(pipe):
    lines = [line for line in pipe.manifest().splitlines() if line.startswith(('smp ', 'smpc ', 'smpl '))]
    assert len(lines) == 1, pipe.manifest()
    return lines[0]


def f32_hex(*vals):
    return ','.join('%08x' % struct.unpack('<I', struct.pack('<f', v))[0] for v in vals)


def check_disassembly():
    """Every constructed fixture must decode cleanly through disasm_rdna2 without unknown words."""
    for name, (_src, raw) in FIXTURES.items():
        off = 0
        decoded = []
        while off < len(raw):
            size, mnemonic, asm, _ = D.decode_instruction(raw, off)
            assert size > 0 and not mnemonic.startswith(('v_unknown', 's_unknown', 'unknown')), (name, off, mnemonic, asm)
            decoded.append(asm)
            off += size
        assert off == len(raw) and decoded[-1] == 's_endpgm', (name, off, len(raw), decoded[-1])


def controls():
    check_disassembly()

    # Accepted comparison-sampler combinations and their exact smpc manifest lines.
    expected_smpc = {
        'border_white': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_W0, OPAQUE_WHITE),
        'border_trans_black': 'smpc 0 0x%x 0x0 0x0 0x0' % POINT_BORDER_W0,
        'border_opaque_black': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_W0, OPAQUE_BLACK | 0xfff),
        'border_opaque_black_inline': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_W0, OPAQUE_BLACK),
        'border_x_clamp_y': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_X_CLAMP_Y_W0, OPAQUE_WHITE),
        'clamp_x_border_y': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_CLAMP_X_BORDER_Y_W0, OPAQUE_WHITE),
        'border_c_max0': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_W0, OPAQUE_WHITE),
        'border_array': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_BORDER_W0, OPAQUE_WHITE),
        'clamp_ignored_reg_border': 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_CLAMP_W0, REGISTER_BORDER | 0xfff),
        'bilinear_border_white': 'smpc 0 0x%x 0x0 0x%x 0x%x' % (BILINEAR_BORDER_W0, BILINEAR_W2, OPAQUE_WHITE),
    }
    for name, want_line in expected_smpc.items():
        pipe = pipeline(name)
        ps = pipe.hlsl_ps()
        assert 'SamplerComparisonState q0 : register(s0);' in ps, (name, ps)
        assert ps.count('.SampleCmpLevelZero(q0,') == 4, (name, ps)
        got_line = manifest_sampler(pipe)
        assert got_line == want_line, (name, got_line, want_line)

    # Standalone pixel_aot.comparison_sampler_state default (allow_border=False) keeps refusing CLAMP_BORDER.
    for w0 in (POINT_BORDER_W0, POINT_BORDER_X_CLAMP_Y_W0, POINT_CLAMP_X_BORDER_Y_W0):
        try:
            P.comparison_sampler_state((w0, 0, 0, OPAQUE_WHITE))
        except ValueError as error:
            assert 'CLAMP_LAST_TEXEL in X and Y' in str(error), error
        else:
            raise AssertionError('standalone pixel_aot accepted CLAMP_BORDER: %#x' % w0)

    # Refused sampler configurations in gfx2hlsl.
    refusals = (
        ('refuse_reg_border_xy', 'register border color'),
        ('refuse_reg_border_x', 'register border color'),
        ('refuse_reg_border_y', 'register border color'),
        ('refuse_half_border_x', 'CLAMP_LAST_TEXEL or CLAMP_BORDER'),
        ('refuse_mirror_once_border_y', 'CLAMP_LAST_TEXEL or CLAMP_BORDER'),
        ('refuse_mirror_once_x', 'CLAMP_LAST_TEXEL or CLAMP_BORDER'),
        ('refuse_mirror_y', 'CLAMP_LAST_TEXEL or CLAMP_BORDER'),
        ('refuse_w3_reserved', 'word 3 bits 12-29'),
    )
    for name, msg in refusals:
        try:
            pipeline(name).hlsl_ps()
        except G.Unsupported as error:
            assert msg in str(error), (name, str(error))
        else:
            raise AssertionError('%s was accepted' % name)
    print('controls: CLAMP_BORDER + fixed border colors accepted; register border, half-border and wrap/mirror refused')


def check_binary(llvm_mc):
    for name, (src, recorded) in FIXTURES.items():
        built = assemble(llvm_mc, src)
        if built != recorded:
            raise AssertionError('%s fixture: assembled %s, recorded %s' % (name, built.hex(), recorded.hex()))
    print('fixture words match llvm-mc (%d programs)' % len(FIXTURES))


def check_draw(dxc, renderer, adapter, fxc=None):
    # 2x2 R32_FLOAT depths in startup_image_3d.cpp: (0,0)=0.125, (1,0)=0.250, (0,1)=0.375, (1,1)=0.500.
    draw_cases = (
        # Point border_white (border=1.0): ref=0.75 -> in-bounds (1,1)=0.500 fails (0), -X/+X/+Y border=1.0 pass (1).
        ('border_white', '0.75', f32_hex(0.0, 1.0, 1.0, 1.0)),
        # Same shader at ref=0.25 -> in-bounds (1,1)=0.500 passes (1), and all three border probes also pass (1).
        ('border_white', '0.25', f32_hex(1.0, 1.0, 1.0, 1.0)),
        # Point border_trans_black & border_opaque_black (border=0.0): ref=0.0625 -> in-bounds passes (1), borders fail (0).
        ('border_trans_black', '0.0625', f32_hex(1.0, 0.0, 0.0, 0.0)),
        ('border_opaque_black', '0.0625', f32_hex(1.0, 0.0, 0.0, 0.0)),
        ('border_opaque_black_inline', '0.0625', f32_hex(1.0, 0.0, 0.0, 0.0)),
        # Mixed X/Y addressing with OPAQUE_WHITE at ref=0.75: only the border axis returns 1.0 out of bounds.
        ('border_x_clamp_y', '0.75', f32_hex(0.0, 1.0, 1.0, 0.0)),
        ('clamp_x_border_y', '0.75', f32_hex(0.0, 0.0, 0.0, 1.0)),
        # Implicit image_sample_c with MAX_LOD 0 and OPAQUE_WHITE border.
        ('border_c_max0', '0.75', f32_hex(0.0, 1.0, 1.0, 1.0)),
        # Bilinear PCF on 2x2 with OPAQUE_WHITE: (0.5,0.5)=0, (0,0.25)=0.5, (0.25,0)=0.5, (0,0)=0.75 at ref=0.75;
        # at ref=0.3125, two of four interior texels (0.375, 0.500) pass -> (0.5, 0.5, 0.5, 0.75).
        ('bilinear_border_white', '0.75', f32_hex(0.0, 0.5, 0.5, 0.75)),
        ('bilinear_border_white', '0.3125', f32_hex(0.5, 0.5, 0.5, 0.75)),
        # Ordinary RGBA sample at (1.25, 0.25) separates OPAQUE_BLACK (0,0,0,1) from TRANS_BLACK (0,0,0,0) and OPAQUE_WHITE.
        ('rgba_trans_black', '0', f32_hex(0.0, 0.0, 0.0, 0.0)),
        ('rgba_opaque_black', '0', f32_hex(0.0, 0.0, 0.0, 1.0)),
        ('rgba_opaque_white', '0', f32_hex(1.0, 1.0, 1.0, 1.0)),
    )
    shader_names = sorted({name for name, _, _ in draw_cases})
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])

    pipes = {name: pipeline(name) for name in shader_names}
    samplers = {name: manifest_sampler(pipes[name]) for name in shader_names}

    with tempfile.TemporaryDirectory() as tmp, CompileQueue() as queue:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            queue.submit(compile_hlsl, compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            for name in shader_names:
                queue.submit(compile_hlsl, compiler, kind, pipes[name].hlsl_ps(),
                             '%s_%s' % (name, kind), ps_profile, directory)

        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = queue.call(compile_hlsl, compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            cso = {
                name: queue.call(compile_hlsl, compiler, kind, pipes[name].hlsl_ps(),
                                 '%s_%s' % (name, kind), ps_profile, directory)
                for name in shader_names
            }
            for name, ref, expected in draw_cases:
                cmd = [renderer, adapter, '--startup-image-2d', str(vs), str(cso[name]), ref, expected, samplers[name]]
                run = subprocess.run(cmd, capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    sys.exit(77)
                if run.returncode != 0:
                    raise AssertionError('%s %s ref=%s failed:\n%s\n%s'
                                         % (kind, name, ref, run.stdout, run.stderr))
                print('%s %s ref=%s: %s' % (kind, name, ref, expected), flush=True)

            # Negative controls on the GPU: dropping w3 border color (the pre-fix TRANSPARENT_BLACK behavior),
            # forcing CLAMP_LAST_TEXEL, swapping X/Y addressing, and dropping OPAQUE_BLACK alpha must all fail.
            mutants = (
                ('dropped_w3_border_color', 'border_white', '0.75', f32_hex(0.0, 1.0, 1.0, 1.0),
                 samplers['border_trans_black'], f32_hex(0.0, 0.0, 0.0, 0.0)),
                ('forced_clamp_last_texel', 'border_white', '0.75', f32_hex(0.0, 1.0, 1.0, 1.0),
                 'smpc 0 0x%x 0x0 0x0 0x%x' % (POINT_CLAMP_W0, OPAQUE_WHITE), f32_hex(0.0, 0.0, 0.0, 0.0)),
                ('swapped_xy_addressing', 'border_x_clamp_y', '0.75', f32_hex(0.0, 1.0, 1.0, 0.0),
                 samplers['clamp_x_border_y'], f32_hex(0.0, 0.0, 0.0, 1.0)),
                ('opaque_black_as_trans_black', 'rgba_opaque_black', '0', f32_hex(0.0, 0.0, 0.0, 1.0),
                 samplers['rgba_trans_black'], f32_hex(0.0, 0.0, 0.0, 0.0)),
            )
            for label, name, ref, expected, bad_sampler, want_actual in mutants:
                cmd = [renderer, adapter, '--startup-image-2d', str(vs), str(cso[name]), ref, expected, bad_sampler]
                failed = subprocess.run(cmd, capture_output=True, text=True)
                if failed.returncode == 0 or ('actual=' + want_actual) not in failed.stderr:
                    raise AssertionError('%s mutant %s did not produce %s:\n%s\n%s'
                                         % (kind, label, want_actual, failed.stdout, failed.stderr))
                print('%s mutant %s caught (%s)' % (kind, label, want_actual), flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
