"""v_cvt_i32_f64 and v_cvt_u32_f64 in the direct compute path (#3370, Refs #2935).

  native_cvt_int_f64_check.py ir <native_compute_ir_dump>
  native_cvt_int_f64_check.py controls <compile_dxil> <sdk bin dir>
  native_cvt_int_f64_check.py hardware <native_store_replay> <sdk bin dir>

Lowers VOP1/VOP3A v_cvt_i32_f64 (0x03 / 0x183) and v_cvt_u32_f64 (0x15 / 0x195)
using pure 32-bit integer arithmetic (no DoublePrecisionFloatShaderOps, no Int64ShaderOps).
"""
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_f16_alu_check import dump as dump_ir, interpret as interpret_ir

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969


def f64_words(val):
    return struct.unpack('<II', struct.pack('<d', val))


# 8 cases x 4 ops (i32, u32, i32 -|x|, u32 |x|) = 32 words/lane (2048 result + 16 guards = 2064).
CASES = (
    f64_words(0.0),                # +0.0 -> 0
    f64_words(-3.75),              # negative fractional -> -3 (i32), 0 (u32); |x| -> +3
    f64_words(0.9999999),          # |x| < 1 -> truncates to 0
    f64_words(float(0x01000001)),  # 2^24 + 1 (exercises e > 20 low-word merge)
    f64_words(2147483647.999),     # INT32_MAX + frac -> 0x7FFFFFFF (both)
    f64_words(-2147483648.0),      # INT32_MIN -> 0x80000000 (i32), 0 (u32); |x| -> 0x80000000 (u32)
    f64_words(4294967295.75),      # UINT32_MAX + frac -> INT32_MAX (i32), UINT32_MAX (u32)
    (0x00000001, 0x7FF80000),      # qNaN -> 0 (both, even with -|x| / |x|)
)

PROLOGUE = [
    0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
    0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F,
]


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def s_mov_lit(sdst, lit):
    return [0xBE8003FF | ((sdst & 0x7F) << 16), lit & MASK]


def vop1(op, vdst, src0):
    return [0x7E000000 | ((vdst & 0xFF) << 17) | ((op & 0xFF) << 9) | (src0 & 0x1FF)]


def vop3_unary(op, vdst, src0, *, src1=0, src2=0, abs_m=0, neg_m=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_m & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_m & 7) << 29) | ((omod & 3) << 27) |
          ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def store_word(byte_off, vgpr):
    return [0xE0701000 | (byte_off & 0xFFF), 0x80080008 | ((vgpr & 0xFF) << 8)]


def program():
    """Tests VOP1 VGPR pair, VOP1 SGPR pair (s[8:9]), and VOP3A abs/neg."""
    code = list(PROLOGUE)
    slot = 0
    for idx, (lo, hi) in enumerate(CASES):
        code += v_mov_lit(2, lo) + v_mov_lit(3, hi)
        code += s_mov_lit(8, lo) + s_mov_lit(9, hi)
        # 0: v_cvt_i32_f64 v4, v[2:3] (or s[8:9] on odd cases)
        code += vop1(0x03, 4, 8 if (idx & 1) else 258)
        code += store_word(4 * slot, 4)
        slot += 1
        # 1: v_cvt_u32_f64 v4, v[2:3] (or s[8:9] on even cases)
        code += vop1(0x15, 4, 258 if (idx & 1) else 8)
        code += store_word(4 * slot, 4)
        slot += 1
        # 2: VOP3A v_cvt_i32_f64 v4, -|v[2:3]|
        code += vop3_unary(0x183, 4, 258, abs_m=1, neg_m=1)
        code += store_word(4 * slot, 4)
        slot += 1
        # 3: VOP3A v_cvt_u32_f64 v4, |v[2:3]|
        code += vop3_unary(0x195, 4, 258, abs_m=1)
        code += store_word(4 * slot, 4)
        slot += 1
    assert slot == 32
    return code + [0xBF810000]


REFUSALS = (
    ('clamp', vop3_unary(0x183, 4, 258, clamp=1) + [0xBF810000], 'unsupported v_cvt_i32_f64 stage or operand modifiers'),
    ('omod', vop3_unary(0x195, 4, 258, omod=1) + [0xBF810000], 'unsupported v_cvt_u32_f64 stage or operand modifiers'),
    ('op_sel', vop3_unary(0x183, 4, 258, op_sel=1) + [0xBF810000], 'unsupported v_cvt_i32_f64 stage or operand modifiers'),
    ('abs_s1', vop3_unary(0x195, 4, 258, abs_m=2) + [0xBF810000], 'unsupported v_cvt_u32_f64 stage or operand modifiers'),
    ('unused_src1', vop3_unary(0x183, 4, 258, src1=1) + [0xBF810000], 'unsupported v_cvt_i32_f64 stage or operand modifiers'),
    ('odd_sgpr', vop1(0x03, 4, 9) + [0xBF810000], 'v_cvt_i32_f64 requires a 64-bit register pair source'),
    ('inline_src', vop1(0x15, 4, 128) + [0xBF810000], 'v_cvt_u32_f64 requires a 64-bit register pair source'),
    ('v255_src', vop1(0x03, 4, 511) + [0xBF810000], 'v_cvt_i32_f64 requires a 64-bit register pair source'),
)


def cvt_int_f64(lo, hi, is_signed, *, drop_lo=False, wrong_signedness=False, round_nearest=False):
    if drop_lo:
        lo = 0
    use_signed = (not is_signed) if wrong_signedness else is_signed
    val = struct.unpack('<d', struct.pack('<II', lo, hi))[0]
    if math.isnan(val):
        return 0
    if math.isinf(val):
        if use_signed:
            return 0x80000000 if val < 0 else 0x7FFFFFFF
        return 0 if val < 0 else 0xFFFFFFFF
    iv = round(val) if round_nearest else math.trunc(val)
    if use_signed:
        iv = max(-0x80000000, min(0x7FFFFFFF, iv))
        return iv & MASK
    iv = max(0, min(0xFFFFFFFF, iv))
    return iv & MASK


def lane_results(**kwargs):
    words = []
    for lo, hi in CASES:
        words.append(cvt_int_f64(lo, hi, True, **kwargs))
        words.append(cvt_int_f64(lo, hi, False, **kwargs))
        words.append(cvt_int_f64(lo, (hi & 0x7FFFFFFF) | 0x80000000, True, **kwargs))
        words.append(cvt_int_f64(lo, hi & 0x7FFFFFFF, False, **kwargs))
    return words


def reference(**kwargs):
    lane = lane_results(**kwargs)
    return [SENTINEL] * 8 + lane * 64 + [SENTINEL] * 8


def run_ir(dumper):
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_int_f64_ir_') as temp:
        work = Path(temp)
        run = dump_ir(dumper, work, program(), 'cvt_int_f64')
        assert run.returncode == 0, run.stderr
        body = run.stdout.split('define void @main()', 1)[1].split('\n}', 1)[0]
        assert 'double' not in body and ' i64 ' not in body, 'v_cvt_i32/u32_f64 must stay 32-bit integer'
        stores = sorted(interpret_ir(run.stdout))
        expected = lane_results()
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(expected))], stores[:4]
        mismatch = next(((i, got, want) for i, ((_, got), want) in enumerate(zip(stores, expected)) if got != want),
                        None)
        if mismatch is not None:
            i, got, want = mismatch
            raise AssertionError(f'slot {i}: got {got:08x}, expected {want:08x}')
        for tag, bad, msg in REFUSALS:
            bad_run = dump_ir(dumper, work, bad, f'refused_{tag}')
            assert bad_run.returncode != 0 and msg in (bad_run.stdout + bad_run.stderr), (
                tag, bad_run.stdout, bad_run.stderr)
    print(f'native_cvt_int_f64 IR: {len(expected)} CPU-evaluated words match model; {len(REFUSALS)} refusals verified')
    return 0


def controls(cli, sdk, work, root):
    marker = request(work, root, program())
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True,
                          text=True, check=True).stdout
    body = dump.split('define void @main()', 1)[1].split('\n}', 1)[0]
    assert 'double' not in body and ' i64 ' not in body and 'Double' not in dump and 'Int64' not in dump, (
        'v_cvt_i32/u32_f64 DXIL must not require double or i64 capability')
    for tag, bad, msg in REFUSALS:
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (
            tag, bad_res.stdout, bad_res.stderr)
    expected = reference()
    for label, kwargs in (('drop low word', {'drop_lo': True}),
                          ('wrong signedness', {'wrong_signedness': True}),
                          ('round nearest instead of trunc', {'round_nearest': True})):
        assert reference(**kwargs) != expected, label
    print(f'v_cvt_i32_f64/v_cvt_u32_f64 DXIL verified (no double/i64); {len(REFUSALS)} refusals and 3 negative controls passed')
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
    print(f'v_cvt_i32_f64/v_cvt_u32_f64: all {len(expected)} words matched')
    return 0


def main(args):
    mode = args[0]
    if mode == 'ir':
        return run_ir(args[1])
    executable, sdk = args[1], args[2]
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_int_f64_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
