"""16-bit vector ALU lowering in tools/startup/gfx2hlsl.py, drawn on a real D3D12 adapter (Part of #1723).

  (no arguments)                       translation controls and reference self-checks, pure Python
  binary <llvm-mc>                     reassemble the fixture bodies and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair (FXC ps_5_1 when given, DXC ps_6_0), draw one pixel
                                       per four input sets and compare the four result words with the CPU reference

The pixel fixture is the one of startup_gfx2hlsl_bit_alu_check.py: three flat parameters (v0..v3, v4..v7, v8..v11),
one operation per lane, results in v12..v15 exported to mrt0. The vertex stage places the inputs in the parameters, so
the pixel stage cannot fold them. Every source word carries garbage in its upper half: the 16-bit operations read the
low half only and write a result whose upper half is zero.

The v_pk_{mul,add,fma}_f16 keys (name:half:group) draw four selector pairs per key and isolate one half of the result;
their reference is tests/f16_packed_model.py (exact rationals, NaN/inf/-0/denormal/overflow/tie values), and one
wrong expected word per run must be rejected.

The expected words come from an exact rational model of binary16 arithmetic (Fraction, then one round to binary16),
not from the lowering's expressions.
"""
import math
import re
import struct
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import f16_packed_model as M  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_WORDS, PRE_WORDS, vs_source, vs_words, words_text)

K = 0x4900  # literal operand of v_fmamk_f16 / v_fmaak_f16: 10.0


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(value):
    return struct.unpack('<H', struct.pack('<e', value))[0]


def exact(bits):
    return Fraction(h2f(bits))


def round_half(value):
    """Nearest binary16 (ties to even) of an exact rational; overflow gives infinity."""
    try:
        return f2h(float(value))
    except OverflowError:
        return 0xfc00 if value < 0 else 0x7c00


def trunc_clamp(bits, lo, hi):
    value = h2f(bits)
    if math.isnan(value):
        return 0
    return int(max(lo, min(hi, math.trunc(value)))) if not math.isinf(value) else (hi if value > 0 else lo)


def signed16(v):
    v &= 0xffff
    return v - 0x10000 if v & 0x8000 else v


def f_min(a, b):
    return a if exact(a) < exact(b) else b


def f_max(a, b):
    return a if exact(a) > exact(b) else b


# mnemonic -> (source count, VOP1 spelling, reference over (s0, s1, s2) words)
OPS = {
    'v_add_f16': (2, False, lambda a, b, c: round_half(exact(a) + exact(b))),
    'v_sub_f16': (2, False, lambda a, b, c: round_half(exact(a) - exact(b))),
    'v_subrev_f16': (2, False, lambda a, b, c: round_half(exact(b) - exact(a))),
    'v_mul_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b))),
    'v_min_f16': (2, False, lambda a, b, c: f_min(a & 0xffff, b & 0xffff)),
    'v_max_f16': (2, False, lambda a, b, c: f_max(a & 0xffff, b & 0xffff)),
    'v_fmac_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(c))),
    'v_fmamk_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(K) + exact(b))),
    'v_fmaak_f16': (2, False, lambda a, b, c: round_half(exact(a) * exact(b) + exact(K))),
    'v_cvt_f16_i16': (1, True, lambda a, b, c: round_half(Fraction(signed16(a)))),
    'v_cvt_f16_u16': (1, True, lambda a, b, c: round_half(Fraction(a & 0xffff))),
    'v_cvt_i16_f16': (1, True, lambda a, b, c: trunc_clamp(a, -32768, 32767) & 0xffff),
    'v_cvt_u16_f16': (1, True, lambda a, b, c: trunc_clamp(a, 0, 65535)),
}

G_ = {name: f2h(value) for name, value in (
    ('1', 1.0), ('2', 2.0), ('3', 3.0), ('half', 0.5), ('m1', -1.0), ('m2.5', -2.5), ('100', 100.0), ('0.1', 0.1),
    ('1000', 1000.0), ('m3', -3.0), ('7', 7.0), ('2.75', 2.75), ('m2.75', -2.75), ('60000', 60000.0),
    ('m60000', -60000.0), ('3.25', 3.25), ('300', 300.0), ('0', 0.0), ('1.5', 1.5), ('0.3', 0.3))}
HIGH = (0xabcd0000, 0x5a5a0000, 0xdead0000, 0x12340000)


def sets(pairs, accumulators=None):
    """Eight lanes of (s0, s1, s2) with garbage in the upper half of every word."""
    out = []
    for i, (a, b) in enumerate(pairs):
        c = accumulators[i] if accumulators else 0
        out.append((a | HIGH[i % 4], b | HIGH[(i + 1) % 4], c | HIGH[(i + 2) % 4]))
    return out


F = G_
ARITH = [(F['1'], F['2']), (F['3'], F['m1']), (F['0.1'], F['0.3']), (F['100'], F['1000']),
         (F['m2.5'], F['half']), (F['1.5'], F['7']), (F['300'], F['m3']), (F['3.25'], F['2.75'])]
ACC = [F['1'], F['m1'], F['half'], F['100'], F['2'], F['0'], F['3'], F['7']]
# (a, b, c) binary16 triples where rounding the binary32 sum and then binary16 differs from one rounding (#2073).
# Measured on AMD RX 9070 XT and WARP: the earlier f16rn(a*b + c) lowering was one binary16 unit low on each.
FMAC_DOUBLE_ROUNDING = [(0x4a54, 0x1ef4, 0x4e2d), (0x51ea, 0xbf3e, 0x1d5f), (0x2bd5, 0xbe83, 0x4528),
                        (0x3915, 0x387a, 0x4b71)]
INTS = [(0x0003, 0), (0xfffd, 0), (0x8403, 0), (0x7fff, 0), (0x8000, 0), (0, 0), (0xffff, 0), (0x00ff, 0)]
FLOATS = [(F['2.75'], 0), (F['m2.75'], 0), (F['60000'], 0), (F['m60000'], 0), (F['half'], 0), (F['m1'], 0),
          (F['100'], 0), (f2h(0.99951171875), 0)]
DRAW = {}
for _name, (_arity, _vop1, _ref) in OPS.items():
    if _name.startswith('v_cvt_f16'):
        DRAW[_name] = sets(INTS)
    elif _name.startswith('v_cvt_'):
        DRAW[_name] = sets(FLOATS)
    else:
        DRAW[_name] = sets(ARITH, ACC if _name == 'v_fmac_f16' else None)
# Four double-rounding lanes replace the first four fixture lanes of v_fmac_f16 (eight lanes stay two draws).
DRAW['v_fmac_f16'] = [tuple(w | HIGH[(i + k) % 4] for k, w in enumerate(t)) for i, t in
                      enumerate(FMAC_DOUBLE_ROUNDING)] + DRAW['v_fmac_f16'][4:]


# v_pk_{mul,add,fma}_f16 keys 'name:half:group'. One key draws four lanes, each with its own op_sel / op_sel_hi pair
# (every pair for mul and add, a covering sample of sixteen for fma), then isolates the low or the high half of the
# result with an integer op, so the readback word is a 16-bit value (tagged below like the integer conversions).
PK_OPS = {'v_pk_mul_f16': ('mul', 2), 'v_pk_add_f16': ('add', 2), 'v_pk_fma_f16': ('fma', 3)}


def pk_parse(key):
    name, half, group = key.split(':')
    kind, arity = PK_OPS[name]
    pairs = M.combos(arity)[int(group) * 4:int(group) * 4 + 4]
    return name, kind, arity, half, pairs


def pk_lane_ok(kind, words, sel, hi, half):
    """A lane whose result half is a NaN must get it from a positive NaN operand: the sign of a NaN made by inf * 0 or
    inf - inf differs between adapters, so such a lane could not be compared word for word."""
    picks = [M.pick(w, (hi if half == 'hi' else sel)[i]) for i, w in enumerate(words)]
    result = M.half_op(kind, *picks)
    if not M.is_nan(result):
        return True
    return any(M.is_nan(h) for h in picks) and not any(M.is_nan(h) and M.sign(h) for h in picks)


def pk_lanes(key):
    name, kind, arity, half, pairs = pk_parse(key)
    seeds = M.sample(sum(map(ord, key)), 400, arity)
    lanes = []
    for i in range(8):
        sel, hi = pairs[i % 4]
        word = next(w for w in seeds if pk_lane_ok(kind, w, sel, hi, half) and w not in lanes)
        lanes.append(word)
        seeds = [w for w in seeds if w != word]
    return [tuple(w) + (0,) * (3 - arity) for w in lanes]


PK_KEYS = ['%s:%s:%d' % (n, h, g) for n in PK_OPS for g in range(4) for h in ('lo', 'hi')]
for _key in PK_KEYS:
    DRAW[_key] = pk_lanes(_key)


def pk_reference(key, x, lane):
    name, kind, arity, half, pairs = pk_parse(key)
    sel, hi = pairs[lane % 4]
    word = M.packed(kind, x[:arity], sel, hi)
    return word >> 16 if half == 'hi' else word & 0xffff


def pk_body_source(key):
    name, kind, arity, half, pairs = pk_parse(key)
    lines = []
    for lane, (sel, hi) in enumerate(pairs):
        dst = 12 + lane
        srcs = ', '.join('v%d' % (lane + 4 * i) for i in range(arity))
        lines.append('%s v%d, %s op_sel:[%s] op_sel_hi:[%s]\n' % (name, dst, srcs, ','.join(map(str, sel)), ','.join(map(str, hi))))
        lines.append('v_and_b32 v%d, 0xffff, v%d\n' % (dst, dst) if half == 'lo' else 'v_lshrrev_b32 v%d, 16, v%d\n' % (dst, dst))
    return ''.join(lines)


def body_source(op):
    if op in PK_KEYS:
        return pk_body_source(op)
    arity, vop1, _ = OPS[op]
    lines = []
    for lane in range(4):
        dst, a, b = 12 + lane, lane, 4 + lane
        if op == 'v_fmac_f16':
            lines.append('v_mov_b32 v%d, v%d\n' % (dst, 8 + lane))
        if op == 'v_fmamk_f16':
            lines.append('%s v%d, v%d, 0x%x, v%d\n' % (op, dst, a, K, b))
        elif op == 'v_fmaak_f16':
            lines.append('%s v%d, v%d, v%d, 0x%x\n' % (op, dst, a, b, K))
        elif vop1:
            lines.append('%s%s v%d, v%d\n' % (op, '_e64' if lane % 2 else '', dst, a))
        else:
            lines.append('%s v%d, v%d, v%d\n' % (op, dst, a, b))
    return ''.join(lines)


# Recorded llvm-mc (gfx1013) encodings of body_source(op); `binary <llvm-mc>` regenerates and compares them.
BODY_WORDS = {
    'v_add_f16': bytes.fromhex(
        '00091864010b1a64020d1c64030f1e64'),
    'v_sub_f16': bytes.fromhex(
        '00091866010b1a66020d1c66030f1e66'),
    'v_subrev_f16': bytes.fromhex(
        '00091868010b1a68020d1c68030f1e68'),
    'v_mul_f16': bytes.fromhex(
        '0009186a010b1a6a020d1c6a030f1e6a'),
    'v_min_f16': bytes.fromhex(
        '00091874010b1a74020d1c74030f1e74'),
    'v_max_f16': bytes.fromhex(
        '00091872010b1a72020d1c72030f1e72'),
    'v_fmac_f16': bytes.fromhex(
        '0803187e0009186c09031a7e010b1a6c0a031c7e020d1c6c0b031e7e030f1e6c'),
    'v_fmamk_f16': bytes.fromhex(
        '0009186e00490000010b1a6e00490000020d1c6e00490000030f1e6e00490000'),
    'v_fmaak_f16': bytes.fromhex(
        '0009187000490000010b1a7000490000020d1c7000490000030f1e7000490000'),
    'v_cvt_f16_i16': bytes.fromhex(
        '00a3187e0d00d1d50101000002a31c7e0f00d1d503010000'),
    'v_cvt_f16_u16': bytes.fromhex(
        '00a1187e0d00d0d50101000002a11c7e0f00d0d503010000'),
    'v_cvt_i16_f16': bytes.fromhex(
        '00a7187e0d00d3d50101000002a71c7e0f00d3d503010000'),
    'v_cvt_u16_f16': bytes.fromhex(
        '00a5187e0d00d2d50101000002a51c7e0f00d2d503010000'),
}


# Recorded llvm-mc (gfx1013) encodings of the v_pk_*_f16 fixtures (see pk_body_source).
BODY_WORDS.update({
    'v_pk_mul_f16:lo:0': bytes.fromhex(
        '0c4010cc00090200ff181836ffff00000d4010cc010b0208ff1a1a36ffff00000e4010cc020d0210ff1c1c36ffff00000f4010cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:0': bytes.fromhex(
        '0c4010cc000902009018182c0d4010cc010b0208901a1a2c0e4010cc020d0210901c1c2c0f4010cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:1': bytes.fromhex(
        '0c4810cc00090200ff181836ffff00000d4810cc010b0208ff1a1a36ffff00000e4810cc020d0210ff1c1c36ffff00000f4810cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:1': bytes.fromhex(
        '0c4810cc000902009018182c0d4810cc010b0208901a1a2c0e4810cc020d0210901c1c2c0f4810cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:2': bytes.fromhex(
        '0c5010cc00090200ff181836ffff00000d5010cc010b0208ff1a1a36ffff00000e5010cc020d0210ff1c1c36ffff00000f5010cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:2': bytes.fromhex(
        '0c5010cc000902009018182c0d5010cc010b0208901a1a2c0e5010cc020d0210901c1c2c0f5010cc030f0218901e1e2c'),
    'v_pk_mul_f16:lo:3': bytes.fromhex(
        '0c5810cc00090200ff181836ffff00000d5810cc010b0208ff1a1a36ffff00000e5810cc020d0210ff1c1c36ffff00000f5810cc030f0218ff1e1e36ffff0000'),
    'v_pk_mul_f16:hi:3': bytes.fromhex(
        '0c5810cc000902009018182c0d5810cc010b0208901a1a2c0e5810cc020d0210901c1c2c0f5810cc030f0218901e1e2c'),
    'v_pk_add_f16:lo:0': bytes.fromhex(
        '0c400fcc00090200ff181836ffff00000d400fcc010b0208ff1a1a36ffff00000e400fcc020d0210ff1c1c36ffff00000f400fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:0': bytes.fromhex(
        '0c400fcc000902009018182c0d400fcc010b0208901a1a2c0e400fcc020d0210901c1c2c0f400fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:1': bytes.fromhex(
        '0c480fcc00090200ff181836ffff00000d480fcc010b0208ff1a1a36ffff00000e480fcc020d0210ff1c1c36ffff00000f480fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:1': bytes.fromhex(
        '0c480fcc000902009018182c0d480fcc010b0208901a1a2c0e480fcc020d0210901c1c2c0f480fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:2': bytes.fromhex(
        '0c500fcc00090200ff181836ffff00000d500fcc010b0208ff1a1a36ffff00000e500fcc020d0210ff1c1c36ffff00000f500fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:2': bytes.fromhex(
        '0c500fcc000902009018182c0d500fcc010b0208901a1a2c0e500fcc020d0210901c1c2c0f500fcc030f0218901e1e2c'),
    'v_pk_add_f16:lo:3': bytes.fromhex(
        '0c580fcc00090200ff181836ffff00000d580fcc010b0208ff1a1a36ffff00000e580fcc020d0210ff1c1c36ffff00000f580fcc030f0218ff1e1e36ffff0000'),
    'v_pk_add_f16:hi:3': bytes.fromhex(
        '0c580fcc000902009018182c0d580fcc010b0208901a1a2c0e580fcc020d0210901c1c2c0f580fcc030f0218901e1e2c'),
    'v_pk_fma_f16:lo:0': bytes.fromhex(
        '0c000ecc0009221cff181836ffff00000d080ecc010b2604ff1a1a36ffff00000e500ecc020d2a0cff1c1c36ffff00000f180ecc030f2e14ff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:0': bytes.fromhex(
        '0c000ecc0009221c9018182c0d080ecc010b2604901a1a2c0e500ecc020d2a0c901c1c2c0f180ecc030f2e14901e1e2c'),
    'v_pk_fma_f16:lo:1': bytes.fromhex(
        '0c600ecc0009221cff181836ffff00000d680ecc010b2604ff1a1a36ffff00000e300ecc020d2a0cff1c1c36ffff00000f780ecc030f2e14ff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:1': bytes.fromhex(
        '0c600ecc0009221c9018182c0d680ecc010b2604901a1a2c0e300ecc020d2a0c901c1c2c0f780ecc030f2e14901e1e2c'),
    'v_pk_fma_f16:lo:2': bytes.fromhex(
        '0c080ecc00092204ff181836ffff00000d200ecc010b260cff1a1a36ffff00000e380ecc020d2a14ff1c1c36ffff00000f100ecc030f2e1cff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:2': bytes.fromhex(
        '0c080ecc000922049018182c0d200ecc010b260c901a1a2c0e380ecc020d2a14901c1c2c0f100ecc030f2e1c901e1e2c'),
    'v_pk_fma_f16:lo:3': bytes.fromhex(
        '0c680ecc00092204ff181836ffff00000d400ecc010b260cff1a1a36ffff00000e580ecc020d2a14ff1c1c36ffff00000f700ecc030f2e1cff1e1e36ffff0000'),
    'v_pk_fma_f16:hi:3': bytes.fromhex(
        '0c680ecc000922049018182c0d400ecc010b260c901a1a2c0e580ecc020d2a14901c1c2c0f700ecc030f2e1c901e1e2c'),
})


def ps_words(op):
    return PRE_WORDS + BODY_WORDS[op] + POST_WORDS


def reference(op, x, lane=0):
    if op in PK_KEYS:
        return pk_reference(op, x, lane)
    return OPS[op][2](*x) & 0xffffffff


def translate(op):
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_words(op), root_blob()).hlsl_ps()


# The target is R32G32B32A32_FLOAT, so a small integer result is a binary32 denormal and FXC drops it when it comes
# from a float to integer conversion. The integer-result operations therefore set bit 30 on the way out of the pixel
# stage and in the expected word; the translated statements themselves are unchanged.
INTEGER_RESULT = ('v_cvt_i16_f16', 'v_cvt_u16_f16', *PK_KEYS)
TAG = 0x40000000


def draw_text(op):
    text = translate(op)
    return re.sub(r'F\(v(1[2-5])\)', r'F(v\1 | 0x%xu)' % TAG, text) if op in INTEGER_RESULT else text


def draw_word(op, x, lane=0):
    return reference(op, x, lane) | (TAG if op in INTEGER_RESULT else 0)


def controls():
    failures = []

    def check(label, ok):
        print('%s: %s' % (label, 'ok' if ok else 'MISMATCH'))
        if not ok:
            failures.append(label)

    # The model itself, by hand: 0.1 + 0.3 in binary16, saturation, truncation toward zero, NaN reads 0.
    check('model: 1 + 2 = 3', OPS['v_add_f16'][2](F['1'], F['2'], 0) == F['3'])
    check('model: 60000 truncates to the i16 limit', OPS['v_cvt_i16_f16'][2](F['60000'], 0, 0) == 0x7fff)
    check('model: -60000 saturates to -32768', OPS['v_cvt_i16_f16'][2](F['m60000'], 0, 0) == 0x8000)
    check('model: a negative value converts to u16 zero', OPS['v_cvt_u16_f16'][2](F['m1'], 0, 0) == 0)
    check('model: NaN converts to zero', OPS['v_cvt_i16_f16'][2](0x7e00, 0, 0) == 0)
    check('model: 0xfffd is -3 as i16', OPS['v_cvt_f16_i16'][2](0xfffd, 0, 0) == F['m3'])
    for a, b, c, want in ((0x4a54, 0x1ef4, 0x4e2d, 0x4e33), (0x51ea, 0xbf3e, 0x1d5f, 0xd55b),
                          (0x2bd5, 0xbe83, 0x4528, 0x450f), (0x3915, 0x387a, 0x4b71, 0x4b9f)):
        check('model: fmac %04x*%04x+%04x is one rounding' % (a, b, c), OPS['v_fmac_f16'][2](a, b, c) == want)
    check('draw: fmac covers the double-rounding triples',
          all(tuple(w & 0xffff for w in x) in FMAC_DOUBLE_ROUNDING
                                            for x in DRAW['v_fmac_f16'][:4]))
    for op in OPS:
        check('%s: every draw reference has a zero upper half' % op,
              all(reference(op, x) >> 16 == 0 for x in DRAW[op]))
        check('%s: draw sets cover two draws of four lanes' % op, len(DRAW[op]) == 8)
    for key in PK_KEYS:
        check('%s: every draw reference has a zero upper half' % key,
              all(reference(key, x, i) >> 16 == 0 for i, x in enumerate(DRAW[key])))
        check('%s: draw sets cover two draws of four lanes' % key, len(DRAW[key]) == 8)
        check('%s: no lane is a generated NaN' % key,
              all(pk_lane_ok(pk_parse(key)[1], x[:pk_parse(key)[2]], *pk_parse(key)[4][i % 4], pk_parse(key)[3])
                  for i, x in enumerate(DRAW[key])))
    if BODY_WORDS:
        for op in (*OPS, *PK_KEYS):
            try:
                text = translate(op)
            except G.Unsupported as e:
                check('%s: translates (%s)' % (op, e), False)
                continue
            check('%s: translates to four lane results' % op,
                  all(('v%d = ' % r) in text for r in range(12, 16)))
    if failures:
        raise SystemExit('FAILED: ' + ', '.join(failures))


def check_binary(llvm_mc):
    for op in (*OPS, *PK_KEYS):
        built = assemble(llvm_mc, body_source(op))
        if built != BODY_WORDS[op]:
            raise AssertionError('%s: assembled %s, recorded %s' % (op, built.hex(), BODY_WORDS[op].hex()))
    for op in DRAW:
        for i in range(0, 8, 4):
            if assemble(llvm_mc, vs_source(DRAW[op][i:i + 4])) != vs_words(DRAW[op][i:i + 4]):
                raise AssertionError('vertex fixture for %s differs from llvm-mc' % op)
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for op, lanes in DRAW.items():
                ps = compile_hlsl(compiler, kind, draw_text(op), 'ps_%s_%s' % (op.replace(':', '_'), kind), ps_profile, directory)
                for index in range(0, len(lanes), 4):
                    chunk = lanes[index:index + 4]
                    vs = compile_hlsl(compiler, kind, G.Pipeline(vs_words(chunk), ps_words(op), root_blob()).hlsl_vs(),
                                      'vs_%s_%d_%s' % (op.replace(':', '_'), index, kind), vs_profile, directory)
                    expected = words_text(draw_word(op, x, index + n) for n, x in enumerate(chunk))
                    run = draw(renderer, adapter, vs, ps, expected)
                    if op == PK_KEYS[0] and index == 0:
                        # A wrong expected word must be reported: the comparison is not vacuous.
                        wrong = words_text([draw_word(op, chunk[0], 0) ^ 1] + [draw_word(op, x, n + 1) for n, x in enumerate(chunk[1:])])
                        if not draw(renderer, adapter, vs, ps, wrong).returncode:
                            failures.append('%s %s wrong-value control accepted' % (kind, op))
                    if run.returncode:
                        failures.append('%s %s set %d' % (kind, op, index // 4))
                        print('%s %s set %d MISMATCH inputs %s\n%s' % (kind, op, index // 4, chunk, run.stderr.strip()),
                              flush=True)
                        continue
                    print('%s %s set %d: %s' % (kind, op, index // 4, expected), flush=True)
    if failures:
        raise AssertionError('draw mismatches: ' + ', '.join(failures))


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
