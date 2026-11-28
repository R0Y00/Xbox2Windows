"""v_cvt_f64_f32 in the direct compute path (#3368, Refs #2935).

  native_cvt_f64_f32_check.py ir <native_compute_ir_dump>
  native_cvt_f64_f32_check.py controls <compile_dxil> <sdk bin dir>
  native_cvt_f64_f32_check.py hardware <native_store_replay> <sdk bin dir>

Lowers VOP1/VOP3A v_cvt_f64_f32 (0x10 / 0x190) using pure 32-bit integer arithmetic
and FirstbitHi (dx.op.unaryBits.i32 33), requiring neither DoublePrecisionFloatShaderOps
nor Int64ShaderOps.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request
from native_f16_alu_check import dump as dump_ir, interpret as interpret_ir

MASK = 0xFFFFFFFF
SENTINEL = 0x69696969

# 8 cases x 2 variants (plain + modified/inline/overlap) x 2 output dwords = 32 words/lane (2064 total).
CASES = (
    0x00000000,  # +0.0
    0x80000000,  # -0.0 (sign preserved)
    0x3F800000,  # +1.0
    0xBF800001,  # -(1.0 + 2^-23) -> exercises low word bit 29
    0x00000001,  # smallest positive binary32 subnormal (2^-149 -> exact normal binary64)
    0x80400001,  # negative binary32 subnormal -> exercises both hi and lo words
    0xFF800000,  # -Inf
    0x7F800005,  # sNaN with payload 5 -> quieted to 0x7FF80000_A0000000
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
    """Tests VOP1 VGPR, VOP1 SGPR, VOP1 literal, VOP1 inline float (240=0.5), VOP3A abs/neg, and overlap D[0]==S0."""
    code = list(PROLOGUE)
    slot = 0
    for idx, x in enumerate(CASES):
        code += v_mov_lit(2, x) + s_mov_lit(8, x)
        # Variant 0: plain conversion of x across VOP1 VGPR, VOP1 SGPR, VOP1 literal, and overlapping D[0]==S0
        if idx % 4 == 0:
            code += vop1(0x10, 4, 258)
        elif idx % 4 == 1:
            code += vop1(0x10, 4, 8)
        elif idx % 4 == 2:
            code += vop1(0x10, 4, 255) + [x]
        else:
            code += v_mov_lit(4, x) + vop1(0x10, 4, 260)  # overlapping v[4:5] from v4
        code += store_word(4 * slot, 4) + store_word(4 * (slot + 1), 5)
        slot += 2

        # Variant 1: VOP3A -|x| (for idx != 0) or VOP1 inline float 0.5 (240, for idx == 0)
        if idx == 0:
            code += vop1(0x10, 6, 240)  # inline 0.5f -> 0.5d
        else:
            code += vop3_unary(0x190, 6, 258, abs_m=1, neg_m=1)  # -|x|
        code += store_word(4 * slot, 6) + store_word(4 * (slot + 1), 7)
        slot += 2
    assert slot == 32
    return code + [0xBF810000]


REFUSALS = (
    ('clamp', vop3_unary(0x190, 4, 258, clamp=1) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('omod', vop3_unary(0x190, 4, 258, omod=1) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('op_sel', vop3_unary(0x190, 4, 258, op_sel=1) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('abs_s1', vop3_unary(0x190, 4, 258, abs_m=2) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('neg_s2', vop3_unary(0x190, 4, 258, neg_m=4) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('unused_src1', vop3_unary(0x190, 4, 258, src1=1) + [0xBF810000], 'unsupported v_cvt_f64_f32 stage or operand modifiers'),
    ('dst_v255', vop1(0x10, 255, 258) + [0xBF810000], 'v_cvt_f64_f32 requires a 64-bit VGPR destination pair'),
)


def cvt_f64_f32(bits, *, flush_subnormal=False, drop_quiet=False, swap_words=False, ignore_neg_zero=False):
    sign = bits & 0x80000000
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0 and mant == 0:
        lo, hi = 0, (0 if ignore_neg_zero else sign)
    elif exp == 0xFF and mant == 0:
        lo, hi = 0, sign | 0x7FF00000
    elif exp == 0xFF:
        mq = mant if drop_quiet else (mant | 0x00400000)
        lo, hi = (mq << 29) & MASK, sign | 0x7FF00000 | (mq >> 3)
    elif exp == 0:
        if flush_subnormal:
            lo, hi = 0, sign
        else:
            f = struct.unpack('<f', struct.pack('<I', bits))[0]
            lo, hi = struct.unpack('<II', struct.pack('<d', f))
    else:
        f = struct.unpack('<f', struct.pack('<I', bits))[0]
        lo, hi = struct.unpack('<II', struct.pack('<d', f))
    return (hi, lo) if swap_words else (lo, hi)


def lane_results(**kwargs):
    words = []
    for idx, x in enumerate(CASES):
        words.extend(cvt_f64_f32(x, **kwargs))
        if idx == 0:
            words.extend(cvt_f64_f32(0x3F000000, **kwargs))
        else:
            words.extend(cvt_f64_f32((x & 0x7FFFFFFF) ^ 0x80000000, **kwargs))
    return words


def reference(**kwargs):
    lane = lane_results(**kwargs)
    return [SENTINEL] * 8 + lane * 64 + [SENTINEL] * 8


def run_ir(dumper):
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_f64_f32_ir_') as temp:
        work = Path(temp)
        run = dump_ir(dumper, work, program(), 'cvt_f64_f32')
        assert run.returncode == 0, run.stderr
        body = run.stdout.split('define void @main()', 1)[1].split('\n}', 1)[0]
        assert 'double' not in body and ' i64 ' not in body, 'v_cvt_f64_f32 must stay 32-bit integer'
        assert '@dx.op.unaryBits.i32(i32 33,' in body, body
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
    print(f'native_cvt_f64_f32 IR: {len(expected)} CPU-evaluated words match exact IEEE-754 widening; {len(REFUSALS)} refusals verified')
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
        'v_cvt_f64_f32 DXIL must not require double or i64 capability')
    for tag, bad, msg in REFUSALS:
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (
            tag, bad_res.stdout, bad_res.stderr)
    expected = reference()
    for label, kwargs in (('flush subnormal', {'flush_subnormal': True}),
                          ('drop quiet bit', {'drop_quiet': True}),
                          ('swap words', {'swap_words': True}),
                          ('ignore -0.0', {'ignore_neg_zero': True})):
        assert reference(**kwargs) != expected, label
    print(f'v_cvt_f64_f32 DXIL verified (no double/i64); {len(REFUSALS)} refusals and 4 negative controls passed')
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
    print(f'v_cvt_f64_f32: all {len(expected)} words matched')
    return 0


def main(args):
    mode = args[0]
    if mode == 'ir':
        return run_ir(args[1])
    executable, sdk = args[1], args[2]
    with tempfile.TemporaryDirectory(prefix='x2w_native_cvt_f64_f32_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
