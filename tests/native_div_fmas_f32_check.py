"""v_div_fmas_f32 in the direct compute path (#3327, Refs #2935).

  native_div_fmas_f32_check.py controls <compile_dxil> <sdk bin dir>
  native_div_fmas_f32_check.py hardware <native_store_replay> <sdk bin dir>

Lowers VOP3A v_div_fmas_f32 (opcode 0x16f): fused binary32 multiply-add of S0, S1, S2
scaled by 2^32 when the lane's VCC bit is set, otherwise unscaled fma(S0, S1, S2).
The prologue sets vcc_lo = 0xAAAAAAAA so even lanes test VCC=0 and odd lanes test VCC=1
in the same wave32 dispatch. Inputs are pre-scaled by 2^32 when |S2| < 2^95 and |S0| or
|S1| < 2^95 so subnormal intermediates (such as 2^-126 * 0.5 = 2^-127) are preserved
before scaling without requiring double-precision shader capability.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

CASES = [
    # (s0, s1, s2) as IEEE-754 binary32 bit patterns
    (0x3FC00000, 0x40000000, 0x3E800000),  # 1.5 * 2.0 + 0.25 = 3.25
    (0x00800000, 0x3F000000, 0x00000000),  # 2^-126 * 0.5 + 0.0 = 2^-127 (subnormal intermediate -> 2^-95 when VCC=1)
    (0x71800000, 0x0C800000, 0x3E000000),  # 2^100 * 2^-102 + 0.125 = 0.375 (|s0| >= 2^95, pre-scales s1 & s2)
    (0x3F800000, 0x3F800000, 0x6F800000),  # 1.0 * 1.0 + 2^96 (|s2| >= 2^95, post-scale overflows to +Inf when VCC=1)
    (0xBE800000, 0x40000000, 0x3F400000),  # -0.25 * 2.0 + 0.75 = 0.25 (clamp keeps 0.25 when VCC=0, saturates when VCC=1)
    (0x3F800001, 0x3F7FFFFE, 0xBF800000),  # (1+2^-23)*(1-2^-23) - 1.0 = -2^-46 (exact FMA vs unfused 0.0)
    (0x7F800000, 0x3F800000, 0x3F800000),  # +Inf * 1.0 + 1.0 = +Inf
    (0x7FC00001, 0x3F800000, 0x3F800000),  # qNaN * 1.0 + 1.0 = qNaN (clamp -> +0.0)
]

# Standard compute prologue + s_mov_b32 vcc_lo, 0xAAAAAAAA (even lanes VCC=0, odd lanes VCC=1).
PROLOGUE = [
    0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
    0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F,
    0xBEEA03FF, 0xAAAAAAAA,
]

# VOP3A v_div_fmas_f32 (opcode 0x16f -> 0xD56F0000), src0=v1(257), src1=v2(258), src2=v3(259) -> 0x040E0501:
#   v_div_fmas_f32 v4, v1, v2, v3
#   v_div_fmas_f32 v5, -|v1|, v2, -v3  (abs=0b001, neg=0b101)
#   v_div_fmas_f32 v6, v1, v2, v3 clamp
OPS = [
    (0xD56F0004, 0x040E0501, 4),
    (0xD56F0105, 0xA40E0501, 5),
    (0xD56F8006, 0x040E0501, 6),
]

# Refused encodings: omod mul:2 (word 1 bits 28:27 = 1) and op_sel (word 0 bit 11 = 1).
REFUSED = [
    ('omod', (0xD56F0004, 0x0C0E0501)),
    ('op_sel', (0xD56F0804, 0x040E0501)),
]

VCC_MASK = 0xAAAAAAAA
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def program(ops=OPS):
    code = list(PROLOGUE)
    for index, (a, b, c) in enumerate(CASES):
        code += [0x7E0202FF, a, 0x7E0402FF, b, 0x7E0602FF, c]
        for w0, w1, _ in ops:
            code += [w0, w1]
        for slot, (_, _, vgpr) in enumerate(ops):
            code += [0xE0701000 | (16 * index + 4 * slot), 0x80080008 | (vgpr << 8)]
    return code + [0xBF810000]


def f32_to_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bits_to_f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def is_nan_bits(bits):
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_inf_bits(bits):
    return (bits & 0x7FFFFFFF) == 0x7F800000


def is_subnormal_bits(bits):
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


def exact_fma_scaled(a_bits, b_bits, c_bits, vcc_set, *, clamp=False, unfused=False, ignore_vcc=False,
                     invert_vcc=False, no_clamp=False):
    cond = False if ignore_vcc else (not vcc_set if invert_vcc else vcc_set)
    scale = Decimal(2) ** 32 if cond else Decimal(1)

    if is_nan_bits(a_bits) or is_nan_bits(b_bits) or is_nan_bits(c_bits):
        return 0x00000000 if (clamp and not no_clamp) else 0x7FC00000

    a_f, b_f, c_f = bits_to_f32(a_bits), bits_to_f32(b_bits), bits_to_f32(c_bits)
    if is_inf_bits(a_bits) or is_inf_bits(b_bits) or is_inf_bits(c_bits):
        try:
            prod = a_f * b_f
            res = prod + c_f
        except OverflowError:
            res = float('inf')
        if res != res:
            return 0x00000000 if (clamp and not no_clamp) else 0x7FC00000
        if clamp and not no_clamp:
            res = min(max(res, 0.0), 1.0)
        return f32_to_bits(res)

    if unfused:
        prod_f = bits_to_f32(f32_to_bits(a_f * b_f))
        res_f = prod_f + c_f
        if cond:
            res_f = res_f * 4294967296.0
        if clamp and not no_clamp:
            res_f = min(max(res_f, 0.0), 1.0)
        try:
            return f32_to_bits(res_f)
        except OverflowError:
            return 0xFF800000 if res_f < 0 else 0x7F800000

    with localcontext() as ctx:
        ctx.prec = 100
        exact = (Decimal.from_float(a_f) * Decimal.from_float(b_f) + Decimal.from_float(c_f)) * scale
        if exact == 0:
            sign = 0x80000000 if ((a_bits ^ b_bits) & 0x80000000) and (c_bits & 0x80000000) else 0
            return 0x00000000 if (clamp and not no_clamp) else sign
        if clamp and not no_clamp:
            if exact <= 0:
                return 0x00000000
            if exact >= 1:
                return 0x3F800000
        try:
            return f32_to_bits(float(exact))
        except OverflowError:
            return 0xFF800000 if exact < 0 else 0x7F800000


def results_for_lane(a, b, c, lane, **mutation):
    vcc_set = bool((VCC_MASK >> (lane & 31)) & 1)
    r0 = exact_fma_scaled(a, b, c, vcc_set, clamp=False, **mutation)
    r1 = exact_fma_scaled(a | 0x80000000, b, c ^ 0x80000000, vcc_set, clamp=False, **mutation)
    r2 = exact_fma_scaled(a, b, c, vcc_set, clamp=True, **mutation)
    return [r0, r1, r2]


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 32
        for index, (a, b, c) in enumerate(CASES):
            expected[base + 4 * index:base + 4 * index + 3] = results_for_lane(a, b, c, lane, **mutation)
    return expected


def words_match(actual, expected):
    if actual == expected:
        return True
    if is_nan_bits(expected) and is_nan_bits(actual):
        return True
    # D3D12 float FMA may flush subnormal binary32 outputs to signed zero when VCC=0.
    if is_subnormal_bits(expected) and (actual == (expected & 0x80000000)):
        return True
    return False


def compile_program(cli, sdk, work, root, code, suffix):
    marker = request(work, root, code)
    output = marker.with_suffix(suffix)
    return subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                          capture_output=True, text=True), output


def controls(cli, sdk, work, root):
    result, output = compile_program(cli, sdk, work, root, program(), '.cso')
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output],
                          capture_output=True, text=True, check=True).stdout
    assert '@dx.op.tertiary.f32(i32 46,' in dump, dump
    assert '@dx.op.unary.f32(i32 7,' in dump, dump
    for n, (label, words) in enumerate(REFUSED):
        result, _ = compile_program(cli, sdk, work, root, program([words + (4,)]), f'.refused{n}.cso')
        assert result.returncode != 0 and 'unsupported v_div_fmas_f32 encoding' in result.stdout + result.stderr, (
            label, result.stdout, result.stderr)
    expected = reference()
    for label, mutation in (('ignore VCC', {'ignore_vcc': True}),
                            ('invert VCC', {'invert_vcc': True}),
                            ('unfused mul+add', {'unfused': True}),
                            ('no clamp', {'no_clamp': True})):
        assert reference(**mutation) != expected, label
    print('compiled three v_div_fmas_f32 forms with DXIL tertiary.f32(46) and unary.f32(7); '
          'two modifier forms are refused; ignore-VCC, invert-VCC, unfused and no-clamp references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    expected = reference()
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, expected)) if not words_match(x, y)]
    assert not diff, diff[:10]
    print(f'v_div_fmas_f32: all {OUTPUT_WORDS} words matched')
    return 0


def main(mode, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
