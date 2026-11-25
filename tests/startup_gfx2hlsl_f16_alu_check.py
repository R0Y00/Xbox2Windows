"""16-bit vector ALU lowering in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_f16_alu_check.py

v_add/sub/subrev/mul/min/max/fmac/fmamk/fmaak_f16 and v_cvt_f16_i16/u16, v_cvt_i16/u16_f16 read the low 16 bits of
their sources and write the low 16 bits of the destination (RDNA 2 zeroes the upper half). The check lowers each
instruction through Stage.one() and runs the emitted statement with a small Python model of the HLSL helpers, so the
result is compared as bits, not as text.
"""
import math
import os
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def r32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def fma16o(a, b, c):
    """Python model of the HLSL fma16o helper: every operation rounded to binary32, round to odd on the sum."""
    p = r32(a * b)
    s = r32(p + c)
    t = r32(s - p)
    e = r32(r32(p - r32(s - t)) + r32(c - t))
    if not math.isfinite(s) or e == 0.0:
        return s
    u = struct.unpack('<I', struct.pack('<f', s))[0]
    if u & 1 == 0:
        u = u + 1 if (e > 0.0) == (s > 0.0) else u - 1
    return struct.unpack('<f', struct.pack('<I', u))[0]


def stage():
    s = G.Stage.__new__(G.Stage)
    s.lines, s.sym, s.used_s, s.used_v, s.masks = [], {}, set(), set(), set()
    s.exec_known = True
    return s


def run(asm, regs):
    """Lower one instruction and evaluate its single statement over the register values in regs (v0, v1, ...)."""
    s = stage()
    name = asm.split()[0]
    s.one(name, asm)
    assert len(s.lines) == 1, s.lines
    dst, expr = re.fullmatch(r'(v\d+) = (.*);', s.lines[0]).groups()
    env = {f'v{i}': v for i, v in enumerate(regs)}
    env.update(
        f16tof32=lambda x: h2f(x), fma16o=fma16o, f16rn=lambda x: f2h(x) if abs(x) < 65520.0 else (0xfc00 if x < 0 else 0x7c00), saturate=lambda x: min(max(x, 0.0), 1.0),
        abs=abs, min=min, max=max, clamp=lambda x, lo, hi: min(max(x, lo), hi),
        trunc=lambda x: float(int(x)), isnan=math.isnan,
        asint=lambda x: x - (1 << 32) if x >> 31 else x, float=float, uint=lambda x: int(x) & 0xffffffff, int=int)
    py = expr.replace('0u', '0').replace('&&', 'and')
    py = re.sub(r'(0x[0-9a-f]+)u', r'\1', py)
    py = re.sub(r'\b(\d+)u\b', r'\1', py)
    py = re.sub(r'^isnan\((.*?)\) \? 0 : (.*)$', r'(0 if isnan(\1) else \2)', py)
    return dst, eval(py, {}, env)


ONE, TWO, HALF, THREE = f2h(1.0), f2h(2.0), f2h(0.5), f2h(3.0)
HIGH = 0xabcd0000
CASES = [
    ('v_add_f16 v3, v0, v1', [ONE | HIGH, TWO | HIGH], THREE),
    ('v_sub_f16 v3, v0, v1', [THREE, ONE], TWO),
    ('v_subrev_f16 v3, v0, v1', [ONE, THREE], TWO),
    ('v_mul_f16 v3, v0, v1', [THREE | HIGH, TWO], f2h(6.0)),
    ('v_min_f16 v3, v0, v1', [THREE, TWO], TWO),
    ('v_max_f16 v3, v0, v1', [THREE, TWO], THREE),
    ('v_fmac_f16 v2, v0, v1', [TWO, THREE, ONE], f2h(7.0)),
    ('v_fmamk_f16 v3, v0, 0x4000, v1', [THREE, ONE], f2h(7.0)),
    ('v_fmaak_f16 v3, v0, v1, 0x3c00', [THREE, TWO], f2h(7.0)),
    # Binary32 sum then binary16 would be one unit low on each (#2073); one rounding gives these words.
    ('v_fmac_f16 v2, v0, v1', [0x4a54, 0x1ef4, 0x4e2d], 0x4e33),
    ('v_fmac_f16 v2, v0, v1', [0x51ea, 0xbf3e, 0x1d5f], 0xd55b),
    ('v_fmac_f16 v2, v0, v1', [0x2bd5, 0xbe83, 0x4528], 0x450f),
    ('v_fmac_f16 v2, v0, v1', [0x3915, 0x387a, 0x4b71], 0x4b9f),
    ('v_add_f16 v3, 1.0, v1', [0, TWO], THREE),
    ('v_mul_f16 v3, 0.15915494, v1', [0, f2h(100.0)], f2h(h2f(f2h(0.15915494)) * 100.0)),
    ('v_mul_f16 v3, 0.15915494, v1', [0, 0x3002], 0x251b),
    ('v_cvt_f16_i16 v3, 1.0', [], f2h(15360.0)),
    ('v_cvt_f16_u16 v3, -1.0', [], f2h(48128.0)),
    ('v_add_f16 v3, -0x3c01, v1', [0, 0], 0xbc01),
    ('v_add_f16 v3, -v0, v1', [ONE, THREE], TWO),
    ('v_add_f16 v3, |v0|, v1', [f2h(-1.0), TWO], THREE),
    ('v_cvt_f16_i16 v3, v0', [0xfffd | HIGH], f2h(-3.0)),
    ('v_cvt_f16_u16 v3, v0', [0x8403 | HIGH], f2h(33795.0)),
    ('v_cvt_i16_f16 v3, v0', [f2h(-2.75) | HIGH], 0xfffe),
    ('v_cvt_i16_f16 v3, v0', [f2h(60000.0)], 0x7fff),
    ('v_cvt_u16_f16 v3, v0', [f2h(2.75) | HIGH], 2),
    ('v_cvt_u16_f16 v3, v0', [f2h(-1.5)], 0),
]
for asm, regs, expected in CASES:
    dst, got = run(asm, regs + [0] * (4 - len(regs)))
    check(asm + ' ' + str(regs), got == expected, 'got %#x want %#x' % (got, expected))
check('clamp saturates the binary16 result', run('v_add_f16 v3, v0, v1 clamp', [THREE, TWO, 0, 0])[1] == ONE)

for asm in ('v_add_f16 v3, v0, v1 op_sel:[1,0]', 'v_cvt_i16_f16 v3, v0 clamp', 'v_add_f16 v3, v0',
            'v_fmaak_f16 v3, v0, v1', 'v_mul_f16 v3, v0, v1 mul:2'):
    try:
        stage().one(asm.split()[0], asm)
        check('refuses ' + asm, False)
    except G.Unsupported:
        check('refuses ' + asm, True)

if failures:
    print('FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('f16 alu ok')
