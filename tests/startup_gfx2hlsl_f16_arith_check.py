"""binary16 arithmetic and compares in tools/startup/gfx2hlsl.py, emulated in f32 (draws compile with fxc 5.1).

usage: startup_gfx2hlsl_f16_arith_check.py [fxc <fxc.exe> | dispatch <fxc.exe> <offscreen_check> warp|hardware]

Lowered: v_pk_add_f16 / v_pk_mul_f16 / v_pk_fma_f16 (op_sel, op_sel_hi, neg_lo, neg_hi, clamp), v_add / sub / subrev /
mul / fma / fmac_f16 (VOP2, VOP3 with neg, abs, op_sel and clamp, SDWA for add / sub / subrev / mul), the f16 compares
(e32, e64, SDWA), v_cvt_pk_u16_u32 and v_mad_u32_u16 with op_sel.

No arguments: pure Python.
  1. The helper chain (x2w_f16_fma, then x2w_f32tof16_rne) is modelled step by step in exact binary32 arithmetic, with
     HLSL f32tof16 modelled both toward zero (MEASURED on WARP, startup_round_mode_check) and to nearest, and compared
     with the single nearest-even rounding of the exact a * b + c (fractions.Fraction) over constructed cases (ties,
     subnormal results, overflow, signed zeros, large exponent gaps, fma near-midpoints) and random binary16 triples.
     Control: the plain f32 `a * b + c` rounded once to binary16 differs from the exact result on some of them.
  2. Each instruction word is built here from the encodings tools/disasm_rdna2.py decodes, translated, and the emitted
     HLSL statement is evaluated (the helpers through the models of 1) on random register words and compared with a
     reference computed from the encoding fields alone. Refusals: toward-zero MODE, a constant's high half, 1/(2*pi),
     dst_sel DWORD, omod.
`fxc`: every translated statement and the helpers compile for vs_5_1 (the draw profile).
`dispatch`: the helpers run on the device (fxc cs_5_1) over 4096 triples per batch and the words are compared with the
exact reference; control: the plain f32 expression differs from it on the device.
"""
import math
import os
import random
import re
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []
INF = float('inf')


def check(label, condition, detail=''):
    print('%s: %s' % (label, 'ok' if condition else 'MISMATCH'))
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


# ---- exact binary rounding ------------------------------------------------------------------------------------
def rnd(x, p, emin, maxv, mode):
    """x (a Fraction, nonzero) rounded to a binary format with p bits, minimum exponent emin, largest finite maxv."""
    s = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, emin)
    q = Fraction(2) ** (e - p + 1)
    n = a / q
    fl = n.numerator // n.denominator
    rem = n - fl
    if mode == 'rne' and (rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2)):
        fl += 1
    r = fl * q
    if r > maxv:
        return s * (INF if mode == 'rne' else float(maxv))
    return s * float(r)


F32MAX = (2 - Fraction(2) ** -23) * Fraction(2) ** 127


def r32(x):
    return rnd(x, 24, -126, F32MAX, 'rne')


def add32(x, y):
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return x + y
    exact = Fraction(x) + Fraction(y)
    if exact == 0:
        return -0.0 if (math.copysign(1, x) < 0 and math.copysign(1, y) < 0) else 0.0
    return r32(exact)


def mul32(x, y):
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return x * y
    exact = Fraction(x) * Fraction(y)
    if exact == 0:
        return math.copysign(0.0, x) * math.copysign(1, y)
    return r32(exact)


def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def f32_from(u):
    return struct.unpack('<f', struct.pack('<I', u & 0xffffffff))[0]


def h2f(bits):
    return struct.unpack('<e', struct.pack('<H', bits & 0xffff))[0]


def f2h(x):
    """bits of a value exactly representable in binary16 (or inf)."""
    if math.isinf(x):
        return 0xfc00 if x < 0 else 0x7c00
    return struct.unpack('<H', struct.pack('<e', x))[0]


def hlsl_f32tof16(x, mode):
    """HLSL f32tof16 as modelled: toward zero (MEASURED on WARP) or nearest even."""
    if math.isnan(x):
        return 0x7fff
    if math.isinf(x) or x == 0:
        return f2h(x) if x != 0 else (0x8000 if math.copysign(1, x) < 0 else 0)
    return f2h(rnd(Fraction(x), 11, -14, Fraction(65504), mode))


def model_rne(x, mode):
    """x2w_f32tof16_rne step by step (HELPERS text), binary32 arithmetic exact-rounded."""
    t = hlsl_f32tof16(x, mode)
    a = abs(x)
    if abs(h2f(t)) > a:
        t -= 1
    if (t & 0x7c00) == 0x7c00:
        return t
    lo = abs(h2f(t))
    if a == lo:
        return t
    u = t + 1
    hi = 65536.0 if (u & 0x7fff) == 0x7c00 else abs(h2f(u))
    dl, dh = add32(a, -lo), add32(hi, -a)
    return u if (dh < dl or (dh == dl and (t & 1))) else t


def model_fma(a, b, c):
    """x2w_f16_fma step by step."""
    p = mul32(a, b)
    s = add32(p, c)
    if (f32_bits(s) & 0x7f800000) == 0x7f800000:
        return s
    bb = add32(s, -p)
    e = add32(add32(p, -add32(s, -bb)), add32(c, -bb))
    u = f32_bits(s)
    if (f32_bits(e) & 0x7fffffff) and not (u & 1):
        u = (u + (0xffffffff if (f32_bits(e) ^ u) & 0x80000000 else 1)) & 0xffffffff
    return f32_from(u)


def exact_f16(a, b, c):
    """bits of the nearest-even binary16 rounding of the exact a * b + c (one rounding)."""
    if any(math.isnan(v) or math.isinf(v) for v in (a, b, c)):
        return None
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        prod_neg = (math.copysign(1, a) * math.copysign(1, b)) < 0
        if Fraction(a) * Fraction(b) == 0 and c == 0:
            return 0x8000 if (prod_neg and math.copysign(1, c) < 0) else 0
        return 0
    return f2h(rnd(exact, 11, -14, Fraction(65504), 'rne'))


def finite_f16(rng):
    while True:
        exp = rng.choice([0, 1, 2, 13, 14, 15, 16, 17, 25, 28, 29, 30] + list(range(31)))
        bits = rng.getrandbits(1) << 15 | exp << 10 | rng.getrandbits(10)
        if exp != 31:
            return bits


def constructed():
    """(a, b, c) binary16 bit triples: ties, near-midpoints, subnormals, overflow, signed zeros, large gaps."""
    cases = [
        (0x3c01, 0x37fe, 0x6401),  # 1025 + (0.5 - 2^-21): f32 rounds onto the tie, nearest even then goes up
        (0x3c01, 0x37fe, 0x6400),
        (0x3c01, 0x37fe, 0xe401),
        (0x3c01, 0xb7fe, 0x6401),
        (0x3c00, 0x3c00, 0x0001),  # 1 + 2^-24: below half an ulp
        (0x0001, 0x0001, 0x0000),  # 2^-48: underflows to +0
        (0x8001, 0x0001, 0x0000),  # -2^-48 + 0: -0 or +0
        (0x8001, 0x0001, 0x8000),
        (0x0000, 0x8000, 0x8000),  # -0 * ... + -0 = -0
        (0x3c00, 0x3c00, 0xbc00),  # exact cancellation: +0
        (0x7bff, 0x3c00, 0x5000),  # 65504 + 32: tie at the overflow threshold -> inf
        (0x7bff, 0x3c00, 0x4fff),
        (0x7bff, 0x7bff, 0x0000),  # overflow of the product
        (0x0200, 0x3c00, 0x0200),  # subnormal sums
        (0x0001, 0x3800, 0x0000),  # 2^-25: tie between 0 and 2^-24 -> 0
        (0x0003, 0x3800, 0x0000),  # 1.5 * 2^-24 -> 2^-23
        (0x3555, 0x3555, 0x8001),
        (0x5bff, 0x5bff, 0xfbff),
        (0x2001, 0x2001, 0x6c01),
    ]
    # fma near-midpoints: c with an odd last bit, product half an ulp of c minus a little (1 - 2^-20 times)
    for e in range(-12, 15):
        c = 0x400 | 1 | (e + 15) << 10
        ulp_half = 2.0 ** (e - 11)
        if not 2.0 ** -14 <= ulp_half * 2 ** 1 <= 32768:
            continue
        a = 0x3c01
        b_val = ulp_half * (1 - 2.0 ** -10)
        try:
            b = f2h(b_val)
        except (OverflowError, struct.error):
            continue
        if h2f(b) != b_val:
            continue
        for sa in (0, 0x8000):
            for sc in (0, 0x8000):
                cases.append((a ^ sa, b, (c & 0x7fff) | sc))
    return cases


def check_helpers():
    rng = random.Random(20261125)
    cases = [tuple(c) for c in constructed()]
    for _ in range(12000):
        cases.append((finite_f16(rng), finite_f16(rng), finite_f16(rng)))
    for _ in range(3000):  # products near a fixed addend's half ulp
        c = finite_f16(rng)
        cases.append((finite_f16(rng), finite_f16(rng) & 0x83ff | (c >> 10 & 31) << 10, c))
    bad = {'tz': [], 'ne': []}
    naive_bad = 0
    for a_bits, b_bits, c_bits in cases:
        a, b, c = h2f(a_bits), h2f(b_bits), h2f(c_bits)
        want = exact_f16(a, b, c)
        for mode, tag in (('rtz', 'tz'), ('rne', 'ne')):
            got = model_rne(model_fma(a, b, c), mode)
            if got != want:
                bad[tag].append((a_bits, b_bits, c_bits, got, want))
        naive = model_rne(add32(mul32(a, b), c), 'rtz')
        naive_bad += naive != want
        # add and mul as the handlers emit them
        want_add = exact_f16(a, 1.0, c)
        got_add = model_rne(model_fma(a, 1.0, c), 'rtz')
        if got_add != want_add:
            bad['tz'].append(('add', a_bits, c_bits, got_add, want_add))
        want_mul = exact_f16(a, b, -0.0)
        if want_mul is not None and Fraction(a) * Fraction(b) == 0:
            want_mul = 0x8000 if math.copysign(1, a) * math.copysign(1, b) < 0 else 0
        got_mul = model_rne(mul32(a, b), 'rtz')
        if got_mul != want_mul:
            bad['tz'].append(('mul', a_bits, b_bits, got_mul, want_mul))
    for tag in bad:
        check('helper chain, f32tof16 modelled %s: %d cases equal the single rounding'
              % ('toward zero' if tag == 'tz' else 'to nearest', len(cases)), not bad[tag],
              'first: %s' % (bad[tag][:3],))
    check('control: plain f32 a * b + c then rounding differs from the single rounding (%d cases)' % naive_bad,
          naive_bad > 0)
    # specials pass through the f32 result
    for a, b, c in ((INF, 0.0, 1.0), (INF, 1.0, -INF), (float('nan'), 1.0, 1.0), (INF, 2.0, 1.0)):
        got = model_fma(a, b, c)
        ref = a * b + c
        check('helper special %r * %r + %r' % (a, b, c), (math.isnan(got) and math.isnan(ref)) or got == ref)
    return cases


# ---- instruction words ------------------------------------------------------------------------------------------
V = 256


def vop2(op, vdst, src0, vsrc1, literal=None):
    w = struct.pack('<I', op << 25 | vdst << 17 | vsrc1 << 9 | src0)
    return w + (struct.pack('<I', literal) if literal is not None else b'')


def vopc(op, src0, vsrc1):
    return struct.pack('<I', 0x3e << 25 | op << 17 | vsrc1 << 9 | src0)


def vop3(op, vdst, srcs, *, op_sel=(0, 0, 0, 0), abs_bits=(0, 0, 0), neg=(0, 0, 0), clamp=0, omod=0):
    srcs = list(srcs) + [0] * (3 - len(srcs))
    w0 = 0x35 << 26 | op << 16 | clamp << 15 | vdst
    for i in range(4):
        w0 |= op_sel[i] << (11 + i)
    for i in range(3):
        w0 |= abs_bits[i] << (8 + i)
    w1 = srcs[0] | srcs[1] << 9 | srcs[2] << 18 | omod << 27
    for i in range(3):
        w1 |= neg[i] << (29 + i)
    return struct.pack('<II', w0, w1)


def vop3p(op, vdst, srcs, *, op_sel=(0, 0, 0), op_sel_hi=(1, 1, 1), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0), clamp=0):
    srcs = list(srcs) + [0] * (3 - len(srcs))
    w0 = 0b110011000 << 23 | op << 16 | clamp << 15 | op_sel_hi[2] << 14 | vdst
    w1 = srcs[0] | srcs[1] << 9 | srcs[2] << 18 | op_sel_hi[0] << 27 | op_sel_hi[1] << 28
    for i in range(3):
        w0 |= op_sel[i] << (11 + i) | neg_hi[i] << (8 + i)
        w1 |= neg_lo[i] << (29 + i)
    return struct.pack('<II', w0, w1)


SEL = {'BYTE_0': 0, 'WORD_0': 4, 'WORD_1': 5, 'DWORD': 6}
UNUSED = {'UNUSED_PAD': 0, 'UNUSED_SEXT': 1, 'UNUSED_PRESERVE': 2}


def sdwa2(op, vdst, src0, vsrc1, *, sel0, sel1, dst_sel='WORD_0', unused='UNUSED_PRESERVE', neg=(0, 0),
          abs_bits=(0, 0), clamp=0, s0=0):
    w0 = op << 25 | vdst << 17 | vsrc1 << 9 | 0xf9
    w1 = (src0 & 0xff) | SEL[dst_sel] << 8 | UNUSED[unused] << 11 | clamp << 13 | SEL[sel0] << 16 | \
        neg[0] << 20 | abs_bits[0] << 21 | s0 << 23 | SEL[sel1] << 24 | neg[1] << 28 | abs_bits[1] << 29
    return struct.pack('<II', w0, w1)


def sdwac(op, src0, vsrc1, *, sel0, sel1, sdst=None, s0=0, neg=(0, 0), abs_bits=(0, 0)):
    w0 = 0x3e << 25 | op << 17 | vsrc1 << 9 | 0xf9
    w1 = (src0 & 0xff) | SEL[sel0] << 16 | neg[0] << 20 | abs_bits[0] << 21 | s0 << 23 | SEL[sel1] << 24 | \
        neg[1] << 28 | abs_bits[1] << 29
    if sdst is not None:
        w1 |= (sdst & 0x7f) << 8 | 1 << 15
    return struct.pack('<II', w0, w1)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""
    def __init__(self, round16=0):
        self.lines, self.sym = [], {}
        self.low16_only_v = set()
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()
        self.masks = set()
        self.round16 = round16


def translate(code, round16=0):
    decoded = G.decode(code)
    stage = Stage(round16)
    for _o, _s, name, asm in decoded:
        stage.one(name, asm)
    return ' | '.join(asm for _o, _s, _n, asm in decoded), stage.lines


# ---- evaluating an emitted statement -----------------------------------------------------------------------------
def py_expr(text):
    text = re.sub(r'\b(0x[0-9a-f]+)u\b', r'\1', text)
    text = re.sub(r'\b(\d+)u\b', r'\1', text)
    # `(C) ? 1 : 0` -> `(1 if (C) else 0)`, innermost first
    while ' ? 1 : 0' in text:
        at = text.index(' ? 1 : 0')
        depth, i = 0, at - 1
        while True:
            if text[i] == ')':
                depth += 1
            elif text[i] == '(':
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        text = text[:i] + '(1 if ' + text[i:at] + ' else 0)' + text[at + len(' ? 1 : 0'):]
    text = text.replace('!(', ' not (').replace('&&', ' and ')
    return text


def evaluate(expr, env, mode='rtz'):
    names = dict(env)
    names.update({
        'f16tof32': h2f, 'x2w_f32tof16_rne': lambda x: model_rne(x, mode), 'x2w_f16_fma': model_fma,
        'saturate': lambda x: 0.0 if math.isnan(x) else min(max(x, 0.0), 1.0), 'min': min, 'asuint': int})
    return eval(py_expr(expr), {'__builtins__': {}}, names) & 0xffffffff


def statement(lines):
    assert len(lines) == 1, lines
    m = re.fullmatch(r'(\w+) = (.*);', lines[0])
    assert m, lines[0]
    return m.group(1), m.group(2)


def lane(word, high):
    return (word >> 16 if high else word) & 0xffff


def ref_f16(kind, a, b, c=None):
    """bits of the exact single rounding of add / mul / fma on binary16 values; None for a NaN/inf case."""
    if kind == 'add':
        return exact_f16(a, 1.0, b) if not (a == 0 and b == 0) else (
            0x8000 if math.copysign(1, a) < 0 and math.copysign(1, b) < 0 else 0)
    if kind == 'mul':
        if any(math.isinf(v) or math.isnan(v) for v in (a, b)):
            return None
        if Fraction(a) * Fraction(b) == 0:
            return 0x8000 if math.copysign(1, a) * math.copysign(1, b) < 0 else 0
        return f2h(rnd(Fraction(a) * Fraction(b), 11, -14, Fraction(65504), 'rne'))
    return exact_f16(a, b, c)


def clamp16(bits):
    x = h2f(bits)
    return f2h(min(max(x, 0.0), 1.0)) if not math.isnan(x) else 0


def run_words(label, code, asm, ref, trials=400, regs=('v2', 'v3', 'v4', 'v1', 's5')):
    """Translate `code`, check the decode, then evaluate the statement on random words against ref(words)."""
    rng = random.Random(label)
    got_asm, lines = translate(code)
    check(label + ' (decode)', got_asm == asm, 'got %r' % got_asm)
    if got_asm != asm:
        return lines
    dst, expr = statement(lines)
    bad = []
    for _ in range(trials):
        words = {}
        for r in regs:
            lo, hi = finite_f16(rng), finite_f16(rng)
            if rng.random() < 0.1:
                lo = rng.choice([0, 0x8000, 0x3c00, 0x0001])
            words[r] = lo | hi << 16
        want = ref(words)
        if want is None:
            continue
        got = evaluate(expr, words)
        if got != want:
            bad.append((words, got, want))
    check('%s: %d evaluated words equal the reference' % (label, trials), not bad,
          'first: %s\n  %s' % (bad[:1], lines[0]))
    return lines


def pk_ref(kind, fields, dst_reg='v1'):
    def f(words):
        out = 0
        n = 3 if kind == 'fma' else 2
        for li, (sel, neg) in enumerate(((fields['op_sel'], fields['neg_lo']), (fields['op_sel_hi'], fields['neg_hi']))):
            vals = []
            for i in range(n):
                src = fields['srcs'][i]
                bits = fields['consts'][i] if isinstance(src, int) else lane(words[src], sel[i])
                vals.append(h2f(bits ^ (0x8000 if neg[i] else 0)))
            r = ref_f16(kind, *vals)
            if r is None:
                return None
            if fields.get('clamp'):
                r = clamp16(r)
            out |= r << (16 * li)
        return out
    return f


def main_checks():
    check_helpers()
    v1, v2, v3, v4, s5 = V + 1, V + 2, V + 3, V + 4, 5
    PK_FMA, PK_ADD, PK_MUL = 0x0e, 0x0f, 0x10
    lines = []

    # ---- VOP3P ----
    lines += run_words('v_pk_add_f16 neg_lo/neg_hi (a packed subtract)',
                       vop3p(PK_ADD, 1, (v2, v3), neg_lo=(0, 1, 0), neg_hi=(0, 1, 0)),
                       'v_pk_add_f16 v1, v2, v3 neg_lo:[0,1] neg_hi:[0,1]',
                       pk_ref('add', dict(srcs=['v2', 'v3'], consts=[0, 0], op_sel=(0, 0), op_sel_hi=(1, 1),
                                          neg_lo=(0, 1), neg_hi=(0, 1))))
    lines += run_words('v_pk_add_f16 plain',
                       vop3p(PK_ADD, 1, (v2, v3)), 'v_pk_add_f16 v1, v2, v3',
                       pk_ref('add', dict(srcs=['v2', 'v3'], consts=[0, 0], op_sel=(0, 0), op_sel_hi=(1, 1),
                                          neg_lo=(0, 0), neg_hi=(0, 0))))
    lines += run_words('v_pk_add_f16 inline 0.5 low half only, neg on src1',
                       vop3p(PK_ADD, 1, (240, v3), op_sel_hi=(0, 1, 1), neg_lo=(0, 1, 0), neg_hi=(0, 1, 0)),
                       'v_pk_add_f16 v1, 0.5, v3 op_sel_hi:[0,1] neg_lo:[0,1] neg_hi:[0,1]',
                       pk_ref('add', dict(srcs=[0, 'v3'], consts=[0x3800, 0], op_sel=(0, 0), op_sel_hi=(0, 1),
                                          neg_lo=(0, 1), neg_hi=(0, 1))))
    lines += run_words('v_pk_fma_f16 op_sel_hi:[1,0,1]',
                       vop3p(PK_FMA, 1, (v2, v3, v4), op_sel_hi=(1, 0, 1)),
                       'v_pk_fma_f16 v1, v2, v3, v4 op_sel_hi:[1,0,1]',
                       pk_ref('fma', dict(srcs=['v2', 'v3', 'v4'], consts=[0] * 3, op_sel=(0, 0, 0),
                                          op_sel_hi=(1, 0, 1), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0))))
    lines += run_words('v_pk_fma_f16 op_sel, neg_lo on src2, neg_hi on src0, clamp',
                       vop3p(PK_FMA, 1, (v2, v3, v4), op_sel=(1, 0, 1), op_sel_hi=(0, 1, 1), neg_lo=(0, 0, 1),
                             neg_hi=(1, 0, 0), clamp=1),
                       'v_pk_fma_f16 v1, v2, v3, v4 op_sel:[1,0,1] op_sel_hi:[0,1,1] neg_lo:[0,0,1] neg_hi:[1,0,0] '
                       'clamp',
                       pk_ref('fma', dict(srcs=['v2', 'v3', 'v4'], consts=[0] * 3, op_sel=(1, 0, 1),
                                          op_sel_hi=(0, 1, 1), neg_lo=(0, 0, 1), neg_hi=(1, 0, 0), clamp=1)))
    lines += run_words('v_pk_mul_f16 op_sel:[1,1] op_sel_hi:[0,0]',
                       vop3p(PK_MUL, 1, (v2, v3), op_sel=(1, 1, 0), op_sel_hi=(0, 0, 1)),
                       'v_pk_mul_f16 v1, v2, v3 op_sel:[1,1] op_sel_hi:[0,0]',
                       pk_ref('mul', dict(srcs=['v2', 'v3'], consts=[0, 0], op_sel=(1, 1), op_sel_hi=(0, 0),
                                          neg_lo=(0, 0), neg_hi=(0, 0))))
    lines += run_words('v_pk_fma_f16 inline 2.0 and -1.0 through their low halves',
                       vop3p(PK_FMA, 1, (v2, 244, 243), op_sel_hi=(1, 0, 0)),
                       'v_pk_fma_f16 v1, v2, 2.0, -1.0 op_sel_hi:[1,0,0]',
                       pk_ref('fma', dict(srcs=['v2', 0, 0], consts=[0, 0x4000, 0xbc00], op_sel=(0, 0, 0),
                                          op_sel_hi=(1, 0, 0), neg_lo=(0, 0, 0), neg_hi=(0, 0, 0))))

    # ---- VOP2 / VOP3 ----
    def keep_hi(words, bits):
        return None if bits is None else (words['v1'] & 0xffff0000) | bits

    lines += run_words('v_sub_f16_e32', vop2(0x33, 1, v2, 3), 'v_sub_f16_e32 v1, v2, v3',
                       lambda w: keep_hi(w, ref_f16('add', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 0) ^ 0x8000))))
    lines += run_words('v_subrev_f16_e32 with an SGPR', vop2(0x34, 1, s5, 3), 'v_subrev_f16_e32 v1, s5, v3',
                       lambda w: keep_hi(w, ref_f16('add', h2f(lane(w['v3'], 0)), h2f(lane(w['s5'], 0) ^ 0x8000))))
    lines += run_words('v_add_f16_e32 inline 1.0', vop2(0x32, 1, 242, 3), 'v_add_f16_e32 v1, 1.0, v3',
                       lambda w: keep_hi(w, ref_f16('add', 1.0, h2f(lane(w['v3'], 0)))))
    lines += run_words('v_mul_f16_e32', vop2(0x35, 1, v2, 3), 'v_mul_f16_e32 v1, v2, v3',
                       lambda w: keep_hi(w, ref_f16('mul', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 0)))))
    lines += run_words('v_fmac_f16_e32', vop2(0x36, 1, v2, 3), 'v_fmac_f16_e32 v1, v2, v3',
                       lambda w: keep_hi(w, ref_f16('fma', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 0)),
                                                    h2f(lane(w['v1'], 0)))))
    lines += run_words('v_fma_f16 op_sel:[0,1,0]', vop3(0x34b, 1, (v2, v3, v4), op_sel=(0, 1, 0, 0)),
                       'v_fma_f16 v1, v2, v3, v4 op_sel:[0,1,0]',
                       lambda w: keep_hi(w, ref_f16('fma', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 1)),
                                                    h2f(lane(w['v4'], 0)))))
    lines += run_words('v_fma_f16 src2 high, destination high, neg on src2',
                       vop3(0x34b, 1, (v2, v3, v4), op_sel=(0, 0, 1, 1), neg=(0, 0, 1)),
                       'v_fma_f16 v1, v2, v3, -v4 op_sel:[0,0,1,1]',
                       lambda w: None if (r := ref_f16('fma', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 0)),
                                                       h2f(lane(w['v4'], 1) ^ 0x8000))) is None
                       else (w['v1'] & 0xffff) | r << 16)
    lines += run_words('v_add_f16_e64 abs, neg, clamp', vop3(0x132, 1, (v2, v3), abs_bits=(0, 1, 0), neg=(1, 1, 0),
                                                             clamp=1),
                       'v_add_f16_e64 v1, -v2, -|v3| clamp',
                       lambda w: None if (r := ref_f16('add', h2f(lane(w['v2'], 0) ^ 0x8000),
                                                       h2f(lane(w['v3'], 0) | 0x8000))) is None
                       else keep_hi(w, clamp16(r)))

    # ---- SDWA ----
    lines += run_words('v_sub_f16_sdwa src1 WORD_1, dst WORD_0 preserve',
                       sdwa2(0x33, 1, 2, 3, sel0='DWORD', sel1='WORD_1'),
                       'v_sub_f16_sdwa v1, v2, v3 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD '
                       'src1_sel:WORD_1',
                       lambda w: keep_hi(w, ref_f16('add', h2f(lane(w['v2'], 0)), h2f(lane(w['v3'], 1) ^ 0x8000))))
    lines += run_words('v_sub_f16_sdwa inline 1.0 WORD_0',
                       sdwa2(0x33, 1, 242, 3, sel0='WORD_0', sel1='WORD_1', s0=1),
                       'v_sub_f16_sdwa v1, 1.0, v3 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0 '
                       'src1_sel:WORD_1',
                       lambda w: keep_hi(w, ref_f16('add', 1.0, h2f(lane(w['v3'], 1) ^ 0x8000))))
    lines += run_words('v_mul_f16_sdwa dst WORD_1 preserve',
                       sdwa2(0x35, 1, 2, 3, sel0='WORD_1', sel1='WORD_0', dst_sel='WORD_1'),
                       'v_mul_f16_sdwa v1, v2, v3 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 '
                       'src1_sel:WORD_0',
                       lambda w: None if (r := ref_f16('mul', h2f(lane(w['v2'], 1)), h2f(lane(w['v3'], 0)))) is None
                       else (w['v1'] & 0xffff) | r << 16)

    # ---- compares ----
    def cmp_ref(op, x, y):
        if math.isnan(x) or math.isnan(y):
            return {'lt': 0, 'neq': 1, 'nlt': 1, 'u': 1, 'o': 0, 'gt': 0, 'eq': 0}[op]
        return int({'lt': x < y, 'neq': x != y, 'nlt': not x < y, 'u': False, 'o': True, 'gt': x > y,
                    'eq': x == y}[op])

    def nan_words(label):
        rng = random.Random(label)
        return lambda: rng.choice([0x7e00, 0x7c01, 0xfe00, 0x7c00, 0xfc00, 0x0000, 0x8000])

    def run_cmp(label, code, asm, op, a, b, dst):
        got_asm, out = translate(code)
        check(label + ' (decode)', got_asm == asm, 'got %r' % got_asm)
        if got_asm != asm:
            return
        reg, expr = statement(out)
        check(label + ' destination', reg == dst, out[0])
        rng = random.Random(label)
        special = nan_words(label)
        bad = []
        for _ in range(400):
            words = {r: (special() if rng.random() < 0.3 else finite_f16(rng)) |
                     (special() if rng.random() < 0.3 else finite_f16(rng)) << 16 for r in ('v2', 'v3', 's5')}
            want = cmp_ref(op, a(words), b(words))
            got = evaluate(expr, words)
            if got != want:
                bad.append((words, got, want))
        check('%s: 400 evaluated compares (NaN, inf, zeros included)' % label, not bad,
              'first: %s\n  %s' % (bad[:1], out[0]))
        lines.extend(out)

    run_cmp('v_cmp_lt_f16_e64 into an SGPR', vop3(0xc9, 6, (128, v3)), 'v_cmp_lt_f16_e64 s6, 0, v3', 'lt',
            lambda w: 0.0, lambda w: h2f(lane(w['v3'], 0)), 's6')
    run_cmp('v_cmp_neq_f16_e32 inline 1.0', vopc(0xed, 242, 3), 'v_cmp_neq_f16_e32 vcc_lo, 1.0, v3', 'neq',
            lambda w: 1.0, lambda w: h2f(lane(w['v3'], 0)), 'vcc')
    run_cmp('v_cmp_nlt_f16_e64 -|v2| op_sel high halves', vop3(0xee, 6, (v2, v3), op_sel=(1, 1, 0, 0),
                                                               abs_bits=(1, 0, 0), neg=(1, 0, 0)),
            'v_cmp_nlt_f16_e64 s6, -|v2|, v3 op_sel:[1,1,0]', 'nlt',
            lambda w: h2f(lane(w['v2'], 1) | 0x8000), lambda w: h2f(lane(w['v3'], 1)), 's6')
    run_cmp('v_cmp_u_f16_e32', vopc(0xe8, v2, 3), 'v_cmp_u_f16_e32 vcc_lo, v2, v3', 'u',
            lambda w: h2f(lane(w['v2'], 0)), lambda w: h2f(lane(w['v3'], 0)), 'vcc')
    run_cmp('v_cmp_lt_f16_sdwa into an SGPR', sdwac(0xc9, 128, 3, sel0='WORD_0', sel1='WORD_1', sdst=6, s0=1),
            'v_cmp_lt_f16_sdwa s6, 0, v3 src0_sel:WORD_0 src1_sel:WORD_1', 'lt',
            lambda w: 0.0, lambda w: h2f(lane(w['v3'], 1)), 's6')
    run_cmp('v_cmp_neq_f16_sdwa inline 1.0', sdwac(0xed, 242, 3, sel0='WORD_0', sel1='WORD_1', s0=1),
            'v_cmp_neq_f16_sdwa vcc_lo, 1.0, v3 src0_sel:WORD_0 src1_sel:WORD_1', 'neq',
            lambda w: 1.0, lambda w: h2f(lane(w['v3'], 1)), 'vcc')

    # ---- integer packing ----
    for label, code, asm, ref in (
            ('v_cvt_pk_u16_u32 saturates', vop3(0x36a, 1, (v2, 128)), 'v_cvt_pk_u16_u32 v1, v2, 0',
             lambda w: min(w['v2'], 0xffff)),
            ('v_cvt_pk_u16_u32 two registers', vop3(0x36a, 1, (v2, v3)), 'v_cvt_pk_u16_u32 v1, v2, v3',
             lambda w: min(w['v2'], 0xffff) | min(w['v3'], 0xffff) << 16),
            ('v_mad_u32_u16 op_sel:[0,1,0]', vop3(0x373, 1, (v2, v3, v4), op_sel=(0, 1, 0, 0)),
             'v_mad_u32_u16 v1, v2, v3, v4 op_sel:[0,1,0]',
             lambda w: ((w['v2'] & 0xffff) * (w['v3'] >> 16) + w['v4']) & 0xffffffff)):
        got_asm, out = translate(code)
        check(label + ' (decode)', got_asm == asm, 'got %r' % got_asm)
        _reg, expr = statement(out)
        rng = random.Random(label)
        bad = []
        for _ in range(300):
            words = {r: rng.choice([rng.getrandbits(32), rng.getrandbits(16), 0xffff, 0x10000])
                     for r in ('v2', 'v3', 'v4')}
            if evaluate(expr, words) != ref(words):
                bad.append(words)
        check(label + ': 300 evaluated words', not bad, out[0])
        lines.extend(out)
    check('control: v_mad_u32_u16 without op_sel keeps its table text',
          translate(vop3(0x373, 1, (v2, v3, v4)))[1] == ['v1 = (v2 & 0xffffu) * (v3 & 0xffffu) + v4;'])

    # ---- refusals ----
    def refused(label, code, round16=0):
        try:
            _a, out = translate(code, round16)
        except G.Unsupported as e:
            check(label + ' refused', True)
            return str(e)
        check(label + ' refused', False, repr(out))
    refused('v_pk_add_f16 under MODE f16 toward zero', vop3p(PK_ADD, 1, (v2, v3)), round16=3)
    refused('v_sub_f16 under MODE f16 toward zero', vop2(0x33, 1, v2, 3), round16=3)
    refused('v_pk_add_f16 high half of an inline constant', vop3p(PK_ADD, 1, (240, v3)))
    refused('v_add_f16 1/(2*pi)', vop2(0x32, 1, 248, 3))
    refused('v_add_f16_sdwa dst_sel DWORD', sdwa2(0x32, 1, 2, 3, sel0='WORD_0', sel1='WORD_0', dst_sel='DWORD'))
    refused('v_add_f16_e64 omod', vop3(0x132, 1, (v2, v3), omod=1))
    refused('v_cmp_lt_f16_sdwa BYTE_0', sdwac(0xc9, 2, 3, sel0='BYTE_0', sel1='WORD_1'))
    return lines


def compile_lines(fxc, lines):
    """Every translated statement in one vs_5_1 shader with the helpers it calls."""
    regs = sorted(set(re.findall(r'\b([vs]\d+|vcc)\b', ' '.join(lines))))
    helpers = [t for n, t in G.HELPERS.items() if any(re.search(r'\b%s\(' % n, l) for l in lines)]
    body = ['#define F asfloat', '#define U asuint', 'cbuffer c : register(b0) { uint4 inp[4]; };'] + helpers + [
        'float4 main(uint id : SV_VertexID) : SV_Position {',
        '    uint exec = 1u; ' + ' '.join('uint %s = inp[%d].x ^ id;' % (r, i % 4) for i, r in enumerate(regs))]
    body += ['    { %s }' % l for l in lines]
    body += ['    return float4(F(%s), 0, 0, 1);' % ' ^ '.join(regs), '}']
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, 'f16.hlsl')
        with open(src, 'w') as fh:
            fh.write('\n'.join(body) + '\n')
        r = subprocess.run([fxc, '/nologo', '/T', 'vs_5_1', '/E', 'main', '/Fo', os.path.join(d, 'f16.cso'), src],
                           capture_output=True, text=True)
    check('fxc vs_5_1 compiles all %d translated statements' % len(lines), r.returncode == 0, r.stderr[-1500:])


def raw_fill(index):
    # Keep in step with raw_uav_fill_word in tests/spirv_window/raw_uav_dump.cpp.
    return ((index + 1) * 0x9E3779B1 & 0xffffffff) | 1


def dispatch(fxc, runner, adapter, cases):
    """The helpers on the device: per thread one triple, four result words: fma | add << 16, mul, plain."""
    runner = os.path.abspath(runner)
    batch = 2048
    out_dir = os.path.join(here, '..', 'out')
    os.makedirs(out_dir, exist_ok=True)
    total = bad = plain_diff = 0
    with tempfile.TemporaryDirectory(dir=out_dir) as d:
        for start in range(0, len(cases), batch):
            chunk = cases[start:start + batch]
            chunk += [(0, 0, 0)] * (batch - len(chunk))
            table = []
            for i, (a, b, c) in enumerate(chunk):
                table += [raw_fill(4 * i) ^ (a | b << 16), raw_fill(4 * i + 1) ^ c]
            src = '\n'.join([
                '#define F asfloat', '#define U asuint', 'RWByteAddressBuffer data : register(u0);',
                G.HELPERS['x2w_f32tof16_rne'], G.HELPERS['x2w_f16_fma'],
                'static const uint T[%d] = {%s};' % (len(table), ', '.join('0x%08xu' % t for t in table)),
                '[numthreads(1024, 1, 1)]',
                'void main(uint3 g : SV_GroupID, uint3 t : SV_GroupThreadID) {',
                '    uint i = g.x * 1024u + t.x;',
                '    uint ab = data.Load(16u * i) ^ T[2u * i], cw = data.Load(16u * i + 4u) ^ T[2u * i + 1u];',
                '    float a = f16tof32(ab), b = f16tof32(ab >> 16), c = f16tof32(cw);',
                '    uint fma = x2w_f32tof16_rne(x2w_f16_fma(a, b, c));',
                '    uint add = x2w_f32tof16_rne(x2w_f16_fma(a, 1.0, c));',
                '    uint mul = x2w_f32tof16_rne(a * b);',
                '    precise float p = a * b; precise float s = p + c;',
                '    data.Store(16u * i, fma | (add << 16)); data.Store(16u * i + 4u, mul);',
                '    data.Store(16u * i + 8u, x2w_f32tof16_rne(s));',
                '}']) + '\n'
            hlsl, cso, out = (os.path.join(d, 'f16%s' % e) for e in ('.hlsl', '.cso', '.bin'))
            with open(hlsl, 'w') as fh:
                fh.write(src)
            subprocess.run([fxc, '/nologo', '/T', 'cs_5_1', '/E', 'main', '/Fo', cso, hlsl], check=True,
                           capture_output=True)
            run = subprocess.run([runner, adapter, '--raw-uav-dump', cso, '2', str(batch * 16), out],
                                 capture_output=True, text=True)
            print(run.stdout, end='')
            if run.returncode == 77:
                sys.exit(77)
            if run.returncode or run.stderr:
                raise AssertionError(run.stderr or 'raw UAV dump exited with %d' % run.returncode)
            words = struct.unpack('<%dI' % (batch * 4), open(out, 'rb').read())
            for i, (a_bits, b_bits, c_bits) in enumerate(cases[start:start + batch]):
                a, b, c = h2f(a_bits), h2f(b_bits), h2f(c_bits)
                want = exact_f16(a, b, c)
                want_add = exact_f16(a, 1.0, c)
                want_mul = ref_f16('mul', a, b)
                got_fma, got_add, got_mul = words[4 * i] & 0xffff, words[4 * i] >> 16, words[4 * i + 1]
                total += 1
                if (got_fma, got_add, got_mul) != (want, want_add, want_mul):
                    bad += 1
                    if bad <= 3:
                        print('  %04x %04x %04x: got %04x %04x %04x want %04x %04x %04x'
                              % (a_bits, b_bits, c_bits, got_fma, got_add, got_mul, want, want_add, want_mul))
                plain_diff += words[4 * i + 2] != want
    check('%s: fma, add and mul words of %d triples equal the single rounding (fxc cs_5_1)' % (adapter, total),
          bad == 0, '%d differ' % bad)
    check('%s: control: the plain f32 a * b + c rounds differently in %d triples' % (adapter, plain_diff),
          plain_diff > 0)


if __name__ == '__main__':
    if len(sys.argv) not in (1, 3, 5) or (len(sys.argv) > 1 and sys.argv[1] not in ('fxc', 'dispatch')):
        print(__doc__)
        sys.exit(2)
    translated = main_checks()
    if len(sys.argv) == 3:
        compile_lines(sys.argv[2], translated)
    elif len(sys.argv) == 5:
        dispatch(sys.argv[2], sys.argv[3], sys.argv[4], [tuple(c) for c in constructed()] + [
            (finite_f16(r), finite_f16(r), finite_f16(r)) for r in [random.Random(7)] for _ in range(14000)])
    if failures:
        print('FAILED: %d check(s): %s' % (len(failures), ', '.join(failures)))
        sys.exit(1)
    print('all checks passed')
