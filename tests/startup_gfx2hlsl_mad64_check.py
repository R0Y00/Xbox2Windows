"""v_mad_u64_u32 and v_mad_i64_i32 in tools/startup/gfx2hlsl.py (Part of #1723).

D = S0 * S1 + S2 modulo 2^64 with two 32-bit factors and a 64-bit S2, written to a VGPR pair. The unsigned form
also writes the carry-out of the 64-bit add to a lane mask; the signed form is accepted only with `null`.

No arguments: translate each form, evaluate the emitted helpers in 32-bit arithmetic against an independent
Python 64-bit reference over a value grid, and check the refusals. Pure Python, no GPU and no assembler.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    MASK, POST_WORDS, PRE_WORDS, Expr, Probe, check, vs_words)

M64 = (1 << 64) - 1
VALUES = (0, 1, 2, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0x89abcdef, 0xfffffffe, 0xffffffff)
ADDENDS = (0, 1, M64, 1 << 63, 0x00000001ffffffff, 0xfedcba9876543210, 0x7fffffffffffffff)


def signed32(x):
    return x - (1 << 32) if x >> 31 else x


def reference(signed, a, b, c):
    """(result, carry) from plain Python integers: the exact product plus the 64-bit addend."""
    product = (signed32(a) * signed32(b) if signed else a * b) & M64
    total = product + c
    return total & M64, int(total > M64)


class Evaluator:
    """Runs the emitted mulhi32/mad64u/mad64i helpers with the 32-bit evaluator of the bit-ALU check."""

    def __init__(self, text):
        self.helpers = {}
        for name in ('mulhi32', 'mad64u', 'mad64i'):
            m = re.search(r'uint3? %s\((.*?)\) \{ (.*?) \}$' % name, text, re.M)
            assert m, '%s missing from the generated HLSL' % name
            self.helpers[name] = ([p.split()[1] for p in m.group(1).split(',')], m.group(2))
        self.original_call = Expr.call
        Expr.call = staticmethod(self.call)

    def close(self):
        Expr.call = staticmethod(self.original_call)

    def call(self, name, args):
        if name == 'mulhi32':
            return self.run('mulhi32', [a[0] for a in args])[0], False
        return self.original_call(name, args)

    def run(self, name, values):
        params, body = self.helpers[name]
        env = dict(zip(params, values))
        stmts = [s.strip() for s in body.split(';') if s.strip()]
        for stmt in stmts[:-1]:
            for item in re.split(r',\s*(?=\w+ = )', stmt[len('uint '):]):
                var, rhs = [x.strip() for x in item.split('=', 1)]
                env[var] = Expr(rhs).run(env)[0]
        ret = stmts[-1]
        inner = re.fullmatch(r'return (?:uint3\((.*)\)|(.*))', ret)
        if inner.group(2) is not None:
            return [Expr(inner.group(2)).run(env)[0]]
        parts, depth, start = [], 0, 0
        text = inner.group(1)
        for i, ch in enumerate(text):
            depth += ch == '('
            depth -= ch == ')'
            if ch == ',' and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        parts.append(text[start:])
        return [Expr(p).run(env)[0] for p in parts]


def translate(asm):
    probe = Probe()
    probe.masks, probe.varying_s = set(), set()
    probe.one(asm.split(' ', 1)[0], asm)
    return probe


def refused(asm):
    try:
        translate(asm)
    except G.Unsupported:
        return True
    return False


def helper_source():
    """The helpers as a pixel stage emits them (a program that only exports)."""
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), PRE_WORDS + POST_WORDS, root_blob()).hlsl_ps()


def controls():
    ev = Evaluator(helper_source())
    try:
        for signed, name in ((False, 'mad64u'), (True, 'mad64i')):
            bad = None
            for a in VALUES:
                for b in VALUES:
                    for c in ADDENDS:
                        lo, hi, carry = ev.run(name, [a, b, c & MASK, c >> 32])
                        want, want_carry = reference(signed, a, b, c)
                        if (hi << 32 | lo) != want or (not signed and carry != want_carry):
                            bad = bad or (hex(a), hex(b), hex(c), hex(hi << 32 | lo), hex(want), carry)
            check('%s: %d inputs match the 64-bit reference' % (name, len(VALUES) ** 2 * len(ADDENDS)), bad is None)
    finally:
        ev.close()

    u = translate('v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21]')
    check('unsigned form writes the pair and vcc',
          u.lines == ['{ uint3 r = mad64u(v18, v19, v20, v21);', 'v12 = r.x;', 'v13 = r.y;', 'vcc = r.z;', '}'])
    s = translate('v_mad_u64_u32 v[12:13], s60, v18, v19, -1')
    check('inline S2 is sign-extended and an SGPR mask is tracked',
          s.lines[0] == '{ uint3 r = mad64u(v18, v19, 4294967295u, 4294967295u);' and 60 in s.masks)
    i = translate('v_mad_i64_i32 v[12:13], null, v18, v19, s[46:47]')
    check('signed form with null writes only the pair',
          i.lines == ['{ uint3 r = mad64i(v18, v19, s46, s47);', 'v12 = r.x;', 'v13 = r.y;', '}'])
    overlap = translate('v_mad_u64_u32 v[20:21], s60, v18, v19, v[20:21]')
    check('a destination overlapping S2 reads S2 before it writes', overlap.lines[0].endswith('v20, v21);'))

    for asm in ('v_mad_i64_i32 v[12:13], vcc_lo, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], null, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v[20:21] clamp',
                'v_mad_u64_u32 v12, vcc_lo, v18, v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v[18:19], v19, v[20:21]',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19',
                'v_mad_u64_u32 v[12:13], vcc_lo, v18, v19, v20'):
        check('refused: %s' % asm, refused(asm))


if __name__ == '__main__':
    controls()
