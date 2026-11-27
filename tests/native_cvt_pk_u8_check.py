"""v_cvt_pk_u8_f32 in the direct compute path (#2997).

  native_cvt_pk_u8_check.py controls <compile_dxil> <sdk bin dir>
  native_cvt_pk_u8_check.py hardware <native_store_replay> <sdk bin dir>

D = (S2 & ~(0xff << 8 * S1[1:0])) | (flt32_to_uint8(S0) << 8 * S1[1:0]). The conversion is the one gfx2hlsl.py
measured on RDNA 2 hardware (see docs/experiments.md): round to nearest even, saturated to [0, 255], with NaN, both
zeros and every negative value giving 0 and +inf giving 255. The VOP3 word (opcode 0x15e) was assembled with LLVM
15.0.7 (clang -cc1as, gfx1013); the other words were checked with tools/disasm_rdna2.py --hex.

Each case below sets S0 and S1 with v_mov literals over S2 = 0xa5a5a5a5 and stores the result; every lane of two
groups of 32 stores all of them. References that round half up, truncate, or let NaN or negative inputs through
must each differ from the expected words.
"""
from pathlib import Path
import math
import struct
import subprocess
import sys
import tempfile

from native_dpp_quad_check import request

REST = 0xA5A5A5A5
F = lambda value: struct.unpack('<I', struct.pack('<f', value))[0]  # noqa: E731
CASES = [(F(0.5), 0), (F(1.5), 1), (F(2.5), 2), (F(254.5), 3), (F(255.5), 0), (F(1e30), 1), (0x7F800000, 2),
         (0x80000000, 3), (F(-1.0), 0), (0x7FC00000, 1), (0x00000001, 2), (F(127.4999), 3), (F(3.0), 6),
         (0xFF800000, 0), (F(0.0), 1), (F(255.0), 2)]
PROLOGUE = [0xBE940304, 0xBE950380, 0xF408080A, 0xFA000020, 0x8F078B05, 0x34100086, 0x4A101007, 0xBF8CC07F]
CVT = (0xD55E0004, 0x040E0501)  # v_cvt_pk_u8_f32 v4, v1, v2, v3
SENTINEL = 0x69696969
GUARD_WORDS = 8
OUTPUT_WORDS = 2064


def program(modifier=0):
    code = PROLOGUE + [0x7E0602FF, REST]  # v_mov_b32 v3, 0xa5a5a5a5
    for index, (value, select) in enumerate(CASES):
        code += [0x7E0202FF, value, 0x7E0402FF, select, CVT[0], CVT[1] | modifier,
                 0xE0701000 | 4 * index, 0x80080408]  # buffer_store_dword v4, v8, s[32:35], 0 offen
    return code + [0xBF810000]


def to_u8(bits, half_up=False, truncate=False, lax=False):
    value = struct.unpack('<f', struct.pack('<I', bits))[0]
    if math.isnan(value) or value <= 0:
        if not lax or math.isnan(value):
            return 0
        return int(max(-128, value)) & 0xFF  # a lowering that forgets the zero test
    value = min(value, 255.0)
    if truncate:
        return int(value)
    if half_up:
        return int(math.floor(value + 0.5))
    return round(value)  # Python rounds half to even


def reference(**mutation):
    expected = [SENTINEL] * OUTPUT_WORDS
    for lane in range(64):
        base = GUARD_WORDS + lane * 16
        for index, (value, select) in enumerate(CASES):
            shift = 8 * (select & 3)
            expected[base + index] = (REST & ~(0xFF << shift) & 0xFFFFFFFF) | (to_u8(value, **mutation) << shift)
    return expected


def compile_program(cli, sdk, work, root, code):
    marker = request(work, root, code)
    return marker, subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker,
                                   marker.with_suffix('.cso')], capture_output=True, text=True)


def controls(cli, sdk, work, root):
    marker, result = compile_program(cli, sdk, work, root, program())
    assert result.returncode == 0 and marker.with_suffix('.cso').is_file(), (result.stdout, result.stderr)
    # The module, before any GPU run: one FMin (DXIL binary opcode 36, not FMax 35) with 255.0 and one Round_ne
    # (unary opcode 26) per case, and no FMax.
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', marker.with_suffix('.cso')], capture_output=True,
                          text=True, check=True).stdout
    assert dump.count('@dx.op.binary.f32(i32 36,') == len(CASES), dump
    assert dump.count('@dx.op.unary.f32(i32 26,') == len(CASES), dump
    assert '@dx.op.binary.f32(i32 35,' not in dump, dump
    _, result = compile_program(cli, sdk, work, root, program(modifier=1 << 29))  # neg on S0
    assert result.returncode and 'direct ALU operand modifiers' in result.stderr, (result.stdout, result.stderr)
    expected = reference()
    for label, mutation in (('half up', {'half_up': True}), ('truncated', {'truncate': True}),
                            ('negatives let through', {'lax': True})):
        assert reference(**mutation) != expected, label
    print('compiled with one FMin and one Round_ne per case; a negated source refused; the half-up, truncating and '
          'negative-passing references differ')
    return 0


def hardware(replay, sdk, work, root):
    marker = request(work, root, program())
    output = work / 'out.bin'
    result = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if result.returncode == 77:
        return 77
    result.check_returncode()
    actual = list(struct.unpack(f'<{OUTPUT_WORDS}I', output.read_bytes()))
    diff = [(i, hex(x), hex(y)) for i, (x, y) in enumerate(zip(actual, reference())) if x != y]
    assert not diff, diff[:10]
    print('v_cvt_pk_u8_f32: all 2064 words matched (1024 results, guards and sentinels)')
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
