"""v_ldexp_f32 in draw stages: D = S0 * 2^S1 (S1 a signed 32-bit integer), under the launch FLOAT_MODE the lowering
matches (f32 denormals flushed on input and output, see Stage.LAUNCH_DENORM).

usage: startup_gfx2hlsl_ldexp_check.py [fxc]

Pure Python: the emitted statement for VOP2 and VOP3 (source neg/abs, clamp, omod) and the helper x2w_ldexp evaluated
on its integer text against an exact rational reference over normal, denormal, zero, infinity and NaN inputs and
exponents at the overflow and underflow boundaries and at the int32 extremes. Controls: a modifier on the exponent
operand refuses; HLSL ldexp(x, float(e)) (the naive form) fails the reference on the extremes. With fxc given, a vs_5_1
stage compiles. Assumed, not measured: NaN inputs keep their bits (no quieting), and a result below 2^-126 is flushed
before rounding (only n == 0 with an all-ones mantissa could round up to 2^-126).
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
sys.path.insert(0, here)
import gfx2hlsl as G  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import Probe  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + str(detail)[:1500])


def lower(line):
    stage = Probe()
    stage.one(line.split(' ', 1)[0], line)
    return '\n'.join(stage.lines)


def reference(bits, e):
    """Exact v_ldexp_f32 with f32 denormals flushed: bits of the result."""
    e = e - (1 << 32) if e >= 1 << 31 else e
    e = max(-1000, min(1000, e))  # beyond +-1000 every finite non-zero input over- or underflows either way
    sign, ex, man = bits & 0x80000000, (bits >> 23) & 255, bits & 0x7fffff
    if ex == 255:
        return bits
    if ex == 0:
        return sign
    value = Fraction(0x800000 | man, 1 << 23) * Fraction(2) ** (ex - 127 + e)
    if value >= Fraction(2) ** 128:
        return sign | 0x7f800000
    if value < Fraction(2) ** -126:
        return sign
    n = ex - 127 + e
    while Fraction(2) ** n > value:
        n -= 1
    while Fraction(2) ** (n + 1) <= value:
        n += 1
    frac = value / Fraction(2) ** n - 1
    assert (frac * (1 << 23)).denominator == 1  # exact: a normal result needs no rounding
    return sign | (n + 127) << 23 | int(frac * (1 << 23))


def helper(bits, e):
    """x2w_ldexp's integer text, evaluated the way HLSL evaluates it (int32 / uint32)."""
    s, ex = bits & 0x80000000, (bits >> 23) & 255
    if ex == 255:
        return bits
    if ex == 0:
        return s
    ei = e - (1 << 32) if e >= 1 << 31 else e
    n = ex + max(-512, min(512, ei))
    return (s | 0x7f800000) if n >= 255 else s if n <= 0 else (s | n << 23 | (bits & 0x7fffff))


def naive(bits, e):
    """HLSL ldexp(x, float(e)) = x * exp2(e) in binary32 with denormal flush: the control."""
    ei = e - (1 << 32) if e >= 1 << 31 else e
    x = struct.unpack('<f', struct.pack('<I', bits))[0]
    try:
        p = struct.unpack('<f', struct.pack('<f', 2.0 ** ei))[0]
    except OverflowError:
        p = math.inf
    except (struct.error,):
        p = math.inf
    r = x * p
    try:
        out = struct.unpack('<I', struct.pack('<f', r))[0]
    except OverflowError:
        out = 0x7f800000 | (bits & 0x80000000)
    return out & 0x80000000 if (out >> 23) & 255 == 0 else out


INPUTS = [0x3f800000, 0xbf800000, 0x00800000, 0x80800000, 0x007fffff, 0x00000001, 0, 0x80000000, 0x7f7fffff,
          0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0x3fffffff, 0x40490fdb, 0xc2c80000]
EXPS = [0, 1, -1, 126, 127, 128, 254, 255, -126, -127, -149, -150, -254, -255, 300, -300, 0x7fffffff, 0x80000000,
        0xffffffff]
bad = [(hex(b), e) for b in INPUTS for e in EXPS if helper(b, e & 0xffffffff) != reference(b, e & 0xffffffff)]
check('x2w_ldexp matches the exact flushed reference on %d cases' % (len(INPUTS) * len(EXPS)), not bad, bad)
nbad = [(hex(b), e) for b in INPUTS for e in EXPS if naive(b, e & 0xffffffff) != reference(b, e & 0xffffffff)]
check('control: x * exp2(float(e)) misses the reference on the boundary cases', len(nbad) > 0, nbad)
text = G.HELPERS.get('x2w_ldexp', '')
for token in ('255u', '-512', '512', '0x7f800000u', '0x7fffffu', '<< 23'):
    check('helper text has %s' % token, token in text, text)

body = lower('v_ldexp_f32 v4, v1, v2')
check('VOP2 v_ldexp_f32 lowers through x2w_ldexp', 'v4 = U(x2w_ldexp(F(v1), v2));' in body, body)
body = lower('v_ldexp_f32_e64 v4, -|v1|, s3 clamp')
check('VOP3 source neg/abs and clamp', 'x2w_ldexp((-abs(F(v1))), s' in body and 'saturate' in body, body)
for line in ('v_ldexp_f32_e64 v4, v1, -v2', 'v_ldexp_f32_e64 v4, v1, |v2|'):
    try:
        lower(line)
        check('%s refuses (modifier on the integer exponent)' % line, False)
    except G.Unsupported:
        check('%s refuses (modifier on the integer exponent)' % line, True)

if len(sys.argv) > 1:
    src = ('#define F(x) asfloat(x)\n#define U(x) asuint(x)\n%s\ncbuffer entry : register(b0) { uint ud0; uint ud1; };\n'
           'float4 main(uint vid : SV_VertexID) : SV_Position { uint v1 = ud0, v2 = ud1 + vid, v4;\n'
           '    v4 = U(x2w_ldexp(F(v1), v2));\n    return float4(F(v4), 0, 0, 1); }\n' % text)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'ldexp.hlsl')
        open(path, 'w').write(src)
        r = subprocess.run([sys.argv[1], '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', path + '.cso', path],
                           capture_output=True, text=True, timeout=60)
        check('fxc vs_5_1 compiles x2w_ldexp', r.returncode == 0, r.stdout + r.stderr)

if failures:
    print('FAIL')
    sys.exit(1)
print('PASS')
