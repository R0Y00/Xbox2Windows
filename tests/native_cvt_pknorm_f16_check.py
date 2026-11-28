"""Direct compute VOP3 v_cvt_pknorm_i16_f16 and v_cvt_pknorm_u16_f16."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

# Packed (s0, s1) pairs with distinct low and high binary16 lanes.
CASES = (
    (0x38000000, 0xBC008000),  # lo: +0.0h, -0.0h; hi: +0.5h, -1.0h
    (0x3C003801, 0xB801BC00),  # lo: 0x3801 (double-rounding edge), -1.0h; hi: +1.0h, 0xB801
    (0x40003802, 0xC000B802),  # lo: 0x3802, 0xB802; hi: +2.0h, -2.0h (clamp to +-1)
    (0x7C000001, 0xFC008001),  # lo: +min_sub, -min_sub; hi: +Inf, -Inf
    (0x7E013400, 0x7E02B400),  # lo: +0.25h, -0.25h; hi: qNaN, qNaN
    (0x3BFF3800, 0xBBFFB800),  # lo: +0.5h, -0.5h (exact RNE tie); hi: max<1, min>-1
    (0x38043803, 0xB804B803),  # lo/hi: 0x3803/0x3804/0xB803/0xB804 RNE boundaries
    (0x3C004400, 0xBC00C400),  # lo: +4.0h, -4.0h; hi: +1.0h, -1.0h
)


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def vop3_binary(op, vdst, src0, src1, *, abs_m=0, neg_m=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_m & 7) << 8) | (vdst & 0xFF))
    w1 = ((neg_m & 7) << 29) | ((omod & 3) << 27) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF)
    return [w0, w1]


def program():
    code = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
            0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F]
    for ci, (s0, s1) in enumerate(CASES):
        code += v_mov_lit(2, s0) + v_mov_lit(3, s1)
        # 0: v_cvt_pknorm_i16_f16 v4, v2, v3 (low halves)
        code += vop3_binary(0x312, 4, 258, 259) + [0xE0701000 | (4 * (ci * 4 + 0)), 0x80080408]
        # 1: v_cvt_pknorm_u16_f16 v4, v2, v3 (low halves)
        code += vop3_binary(0x313, 4, 258, 259) + [0xE0701000 | (4 * (ci * 4 + 1)), 0x80080408]
        # 2: v_cvt_pknorm_i16_f16 v4, -v2.hi, |v3.hi| (op_sel[1:0]=3, neg0=1, abs1=1)
        code += vop3_binary(0x312, 4, 258, 259, op_sel=3, neg_m=1, abs_m=2) + [
            0xE0701000 | (4 * (ci * 4 + 2)), 0x80080408]
        # 3: v_cvt_pknorm_u16_f16 v4, 0.5, v3 (inline float constant src0=240 -> 0x3800)
        code += vop3_binary(0x313, 4, 240, 259) + [0xE0701000 | (4 * (ci * 4 + 3)), 0x80080408]
    return code + [0xBF810000]


def cvt_norm_f16(bits, signed_out, *, truncate=False):
    bits &= 0xFFFF
    sign = (bits & 0x8000) != 0
    mag = bits & 0x7FFF
    if mag > 0x7C00 or mag == 0:
        return 0
    scale = 32767 if signed_out else 65535
    if (not signed_out) and sign:
        return 0
    if mag >= 0x3C00:
        q_mag = scale
    else:
        exp = mag >> 10
        if exp == 0:
            mant, shift = mag, 24
        else:
            mant, shift = 1024 | (mag & 1023), 25 - exp
        prod = mant * scale
        q0 = prod >> shift
        if truncate:
            q_mag = q0
        else:
            mid = 1 << (shift - 1)
            rem = prod & ((1 << shift) - 1)
            up = rem > mid or (rem == mid and (q0 & 1) != 0)
            q_mag = q0 + int(up)
    if signed_out and sign:
        return (-q_mag) & 0xFFFF
    return q_mag & 0xFFFF


def reference(*, truncate=False, inline_as_zero=False):
    words = [SENTINEL] * 8
    for _lane in range(64):
        for s0, s1 in CASES:
            p_i = cvt_norm_f16(s0, True, truncate=truncate) | (cvt_norm_f16(s1, True, truncate=truncate) << 16)
            p_u = cvt_norm_f16(s0, False, truncate=truncate) | (cvt_norm_f16(s1, False, truncate=truncate) << 16)
            hi0 = ((s0 >> 16) & 0xFFFF) ^ 0x8000
            hi1 = (s1 >> 16) & 0x7FFF
            p_mod = cvt_norm_f16(hi0, True, truncate=truncate) | (cvt_norm_f16(hi1, True, truncate=truncate) << 16)
            inl = 0x0000 if inline_as_zero else 0x3800
            p_inl = cvt_norm_f16(inl, False, truncate=truncate) | (cvt_norm_f16(s1, False, truncate=truncate) << 16)
            words.extend((p_i, p_u, p_mod, p_inl))
    return words + [SENTINEL] * 8


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    assert 'mul i32 ' in dump and '32767' in dump and '65535' in dump, dump
    for tag, bad, msg in (
        ('clamp', vop3_binary(0x312, 4, 258, 259, clamp=1) + [0xBF810000], 'unsupported v_cvt_pknorm_i16_f16 clamp or omod'),
        ('omod', vop3_binary(0x313, 4, 258, 259, omod=1) + [0xBF810000], 'unsupported v_cvt_pknorm_u16_f16 clamp or omod'),
        ('opsel_hi', vop3_binary(0x312, 4, 258, 259, op_sel=4) + [0xBF810000], 'unsupported v_cvt_pknorm_i16_f16 operand modifiers'),
        ('bad_mode', [0xBFA4000C] + vop3_binary(0x313, 4, 258, 259) + [0xBF810000], 'unsupported binary16 rounding or denormal mode'),
    ):
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (tag, bad_res.stdout, bad_res.stderr)
    assert reference() != reference(truncate=True)
    assert reference() != reference(inline_as_zero=True)
    print('v_cvt_pknorm_i16/u16_f16 DXIL verified; invalid modifiers/mode refused; RNE and inline controls differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    expected = reference()
    actual = struct.unpack(f'<{len(expected)}I', output.read_bytes())
    mismatch = next((i for i, (a, e) in enumerate(zip(actual, expected)) if a != e), None)
    if mismatch is not None:
        raise AssertionError(f'word {mismatch}: got {actual[mismatch]:08x}, expected {expected[mismatch]:08x}')
    print('v_cvt_pknorm_i16/u16_f16: 2048 result words and 16 guards matched')
    return 0


def main(mode, executable, sdk):
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_pknorm_f16_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
