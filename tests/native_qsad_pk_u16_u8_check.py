"""v_qsad_pk_u16_u8 and v_mqsad_pk_u16_u8 in the direct compute path (#3359, Refs #2935).

  native_qsad_pk_u16_u8_check.py ir <native_compute_ir_dump>
  native_qsad_pk_u16_u8_check.py controls <compile_dxil> <sdk bin dir>
  native_qsad_pk_u16_u8_check.py hardware <native_store_replay> <sdk bin dir>

Lowers VOP3A v_qsad_pk_u16_u8 (0x172) and v_mqsad_pk_u16_u8 (0x173) using pure 32-bit
integer arithmetic (no Int64ShaderOps). Tests wrap and clamp (uint16 saturation), zero
reference bytes (skipped by mqsad, included by qsad), SGPR pair S0, overlapping D == S2,
and modifier / operand-width refusals.
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
OUTPUT_WORDS = 2064

# 4 cases x 4 ops x 2 output dwords = 32 words per lane (2048 result words + 16 guards = 2064).
# Each tuple: (s0_lo, s0_hi, s1_ref, s2_lo, s2_hi)
CASES = (
    # Basic distinct bytes, zero accumulators, no zero ref bytes
    (0x04030201, 0x08070605, 0x05040302, 0x00000000, 0x00000000),
    # Zero reference bytes (bytes 1 and 3 are 0x00): qsad adds |b - 0|, mqsad skips them
    (0xF0E0D0C0, 0x40302010, 0x00A00050, 0x00200010, 0x00400030),
    # Overflow / clamp boundary: acc near 0xFFFF, max byte diffs 0xFF vs 0x01
    (0xFFFFFFFF, 0x00000000, 0x01FE01FE, 0xFFF0FF00, 0xFFE0FFFF),
    # Mixed high/low differences with asymmetric accumulators
    (0x109030B0, 0x7050D020, 0x8000FF40, 0xFE000100, 0xFC8003FF),
)

PROLOGUE = [
    0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020,
    0x8F078C05, 0x34100087, 0x4A101007, 0xBF8CC07F,
]


def v_mov_lit(vdst, lit):
    return [0x7E0002FF | ((vdst & 0xFF) << 17), lit & MASK]


def s_mov_lit(sdst, lit):
    return [0xBE8003FF | ((sdst & 0x7F) << 16), lit & MASK]


def vop3(op, vdst, src0, src1, src2, *, abs_m=0, neg_m=0, clamp=0, omod=0, op_sel=0):
    w0 = (0xD4000000 | ((op & 0x3FF) << 16) | ((clamp & 1) << 15) |
          ((op_sel & 0xF) << 11) | ((abs_m & 7) << 8) | (vdst & 0xFF))
    w1 = (((neg_m & 7) << 29) | ((omod & 3) << 27) |
          ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF))
    return [w0, w1]


def store_word(byte_off, vgpr):
    return [0xE0701000 | (byte_off & 0xFFF), 0x80080008 | ((vgpr & 0xFF) << 8)]


def program():
    """For each case, runs 4 variants (8 stored dwords):
    0: v_qsad_pk_u16_u8  v[6:7], v[0:1], v2, v[4:5]       (wrap)
    1: v_qsad_pk_u16_u8  v[6:7], s[8:9], v2, v[4:5] clamp (SGPR pair S0 + saturate)
    2: v_mqsad_pk_u16_u8 v[6:7], v[0:1], v2, v[4:5]       (masked wrap)
    3: v_mqsad_pk_u16_u8 v[4:5], v[0:1], v2, v[4:5] clamp (overlapping D == S2 + masked saturate)
    """
    code = list(PROLOGUE)
    slot = 0
    for s0_lo, s0_hi, s1, s2_lo, s2_hi in CASES:
        code += v_mov_lit(0, s0_lo) + v_mov_lit(1, s0_hi)
        code += s_mov_lit(8, s0_lo) + s_mov_lit(9, s0_hi)
        code += v_mov_lit(2, s1)
        code += v_mov_lit(4, s2_lo) + v_mov_lit(5, s2_hi)
        # 0: qsad wrap (VGPR pair S0)
        code += vop3(0x172, 6, 256, 258, 260, clamp=0)
        code += store_word(4 * slot, 6) + store_word(4 * (slot + 1), 7)
        slot += 2
        # 1: qsad clamp (SGPR pair s[8:9] as S0)
        code += vop3(0x172, 6, 8, 258, 260, clamp=1)
        code += store_word(4 * slot, 6) + store_word(4 * (slot + 1), 7)
        slot += 2
        # 2: mqsad wrap
        code += vop3(0x173, 6, 256, 258, 260, clamp=0)
        code += store_word(4 * slot, 6) + store_word(4 * (slot + 1), 7)
        slot += 2
        # 3: mqsad clamp with overlapping D == S2 (v[4:5])
        code += vop3(0x173, 4, 256, 258, 260, clamp=1)
        code += store_word(4 * slot, 4) + store_word(4 * (slot + 1), 5)
        slot += 2
    assert slot == 32
    return code + [0xBF810000]


REFUSALS = (
    ('omod', vop3(0x172, 6, 256, 258, 260, omod=1) + [0xBF810000], 'unsupported v_qsad_pk_u16_u8 stage or operand modifiers'),
    ('op_sel', vop3(0x173, 6, 256, 258, 260, op_sel=1) + [0xBF810000], 'unsupported v_mqsad_pk_u16_u8 stage or operand modifiers'),
    ('abs', vop3(0x172, 6, 256, 258, 260, abs_m=1) + [0xBF810000], 'unsupported v_qsad_pk_u16_u8 stage or operand modifiers'),
    ('neg', vop3(0x173, 6, 256, 258, 260, neg_m=2) + [0xBF810000], 'unsupported v_mqsad_pk_u16_u8 stage or operand modifiers'),
    ('odd_sgpr_s0', vop3(0x172, 6, 9, 258, 260) + [0xBF810000], 'v_qsad_pk_u16_u8 requires 64-bit register pair operands'),
    ('non_pair_s2', vop3(0x173, 6, 256, 258, 128) + [0xBF810000], 'v_mqsad_pk_u16_u8 requires 64-bit register pair operands'),
    ('dst_v255', vop3(0x172, 255, 256, 258, 260) + [0xBF810000], 'v_qsad_pk_u16_u8 requires 64-bit register pair operands'),
    ('inline_f32_s1', vop3(0x173, 6, 256, 242, 260) + [0xBF810000], 'v_mqsad_pk_u16_u8 with an inline float reference'),
)


def qsad_pk_u16_u8(s0_lo, s0_hi, s1, s2_lo, s2_hi, *, masked=False, clamp=False,
                   swap_halves=False, wrong_shift=False):
    b = [(s0_lo >> (8 * k)) & 0xFF for k in range(4)] + [(s0_hi >> (8 * k)) & 0xFF for k in range(4)]
    r = [(s1 >> (8 * k)) & 0xFF for k in range(4)]
    acc = [s2_lo & 0xFFFF, (s2_lo >> 16) & 0xFFFF, s2_hi & 0xFFFF, (s2_hi >> 16) & 0xFFFF]
    out = []
    for i in range(4):
        s = acc[i]
        for k in range(4):
            idx = (i + k + (1 if wrong_shift else 0)) & 7
            if not (masked and r[k] == 0):
                s += abs(b[idx] - r[k])
        out.append(min(s, 0xFFFF) if clamp else (s & 0xFFFF))
    d_lo = out[0] | (out[1] << 16)
    d_hi = out[2] | (out[3] << 16)
    return (d_hi, d_lo) if swap_halves else (d_lo, d_hi)


def lane_results(*, swap_masked=False, no_clamp=False, swap_halves=False, wrong_shift=False):
    words = []
    for s0_lo, s0_hi, s1, s2_lo, s2_hi in CASES:
        for masked, clamp in ((False, False), (False, True), (True, False), (True, True)):
            d_lo, d_hi = qsad_pk_u16_u8(
                s0_lo, s0_hi, s1, s2_lo, s2_hi,
                masked=(not masked if swap_masked else masked),
                clamp=(False if no_clamp else clamp),
                swap_halves=swap_halves,
                wrong_shift=wrong_shift,
            )
            words.extend((d_lo, d_hi))
    return words


def reference(**kwargs):
    lane = lane_results(**kwargs)
    return [SENTINEL] * 8 + lane * 64 + [SENTINEL] * 8


def run_ir(dumper):
    with tempfile.TemporaryDirectory(prefix='x2w_native_qsad_ir_') as temp:
        work = Path(temp)
        run = dump_ir(dumper, work, program(), 'qsad')
        assert run.returncode == 0, run.stderr
        body = run.stdout.split('define void @main()', 1)[1].split('\n}', 1)[0]
        assert ' i64 ' not in body, 'v_qsad/v_mqsad must not require i64 ops'
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
    print(f'native_qsad_pk_u16_u8 IR: {len(expected)} CPU-evaluated words match model; {len(REFUSALS)} refusals verified')
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
    assert ' i64 ' not in body and 'Int64' not in dump, 'v_qsad/v_mqsad DXIL must stay 32-bit'
    for tag, bad, msg in REFUSALS:
        sub = work / tag
        sub.mkdir(exist_ok=True)
        bad_marker = request(sub, root, bad)
        bad_res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', bad_marker,
                                  bad_marker.with_suffix('.cso')], capture_output=True, text=True)
        assert bad_res.returncode != 0 and msg in (bad_res.stdout + bad_res.stderr), (
            tag, bad_res.stdout, bad_res.stderr)
    expected = reference()
    for label, kwargs in (('swap qsad/mqsad', {'swap_masked': True}),
                          ('no clamp', {'no_clamp': True}),
                          ('swap halves', {'swap_halves': True}),
                          ('wrong byte shift', {'wrong_shift': True})):
        assert reference(**kwargs) != expected, label
    print(f'v_qsad_pk_u16_u8/v_mqsad_pk_u16_u8 DXIL verified (no i64); {len(REFUSALS)} refusals and 4 negative controls passed')
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
    print(f'v_qsad_pk_u16_u8/v_mqsad_pk_u16_u8: all {len(expected)} words matched')
    return 0


def main(args):
    mode = args[0]
    if mode == 'ir':
        return run_ir(args[1])
    executable, sdk = args[1], args[2]
    with tempfile.TemporaryDirectory(prefix='x2w_native_qsad_') as temp:
        work = Path(temp)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return (controls if mode == 'controls' else hardware)(executable, sdk, work, root.read_bytes())


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
