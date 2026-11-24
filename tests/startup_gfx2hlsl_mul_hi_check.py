"""`v_mul_hi_u32` in tools/startup/gfx2hlsl.py (issue #1792).

usage: startup_gfx2hlsl_mul_hi_check.py

The high half of an unsigned 32x32 product had no lowering, so `v_mul_hi_u32` was refused - one of the
remaining refused shaders in the main-game loading-screen cache (#870). `tools/pixel_aot.py` already had the
halfword-product helper; this port lowers the VOP3A form to it.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch): prefix `0b110101`, opcode `0x16a` in `w0[25:16]`, vdst in `w0[7:0]`,
src0 in `w1[8:0]`, src1 in `w1[17:9]`, and source 255 followed by a literal dword.
"""
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


VOP3A_MUL_HI_U32 = 0x16A
V0, V1, V2 = 0x100, 0x101, 0x102
LITERAL_SOURCE = 255


def encode_vop3a(op, vdst, srcs, literal=None):
    """Encode a native VOP3A instruction; a source of 255 is followed by its literal dword."""
    w0 = (0b110101 << 26) | ((op & 0x3FF) << 16) | (vdst & 0xFF)
    w1 = 0
    for i, src in enumerate(srcs):
        w1 |= (src & 0x1FF) << (9 * i)
    code = struct.pack('<II', w0, w1)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


class Pipeline:
    """Just enough pipeline state for one scalarized multiply."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines)


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, [V1, V2]))
check('encodes and decodes the register form', asm == ['v_mul_hi_u32 v1, v1, v2'], repr(asm))

# 2. The lowering calls the halfword-product helper rather than a plain multiply (the low half of the product
#    is not the high half, so a `*` here would be silently wrong).
check('the register form uses the helper', emitted == 'v1 = mul_hi_u32(v1, v2);', emitted)

# 3. The census form carries a literal operand, which reaches the helper with the same suffix the other
#    uint operations use.
asm, emitted = translate(encode_vop3a(VOP3A_MUL_HI_U32, 1, [V0, LITERAL_SOURCE],
                                      literal=0xAAAAAAAB))
check('encodes and decodes the literal form', asm == ['v_mul_hi_u32 v1, v0, 0xaaaaaaab'], repr(asm))
check('the literal form uses the helper', emitted == 'v1 = mul_hi_u32(v0, 0xaaaaaaabu);', emitted)

# 4. The helper is in the preamble the generated shader is built with; without it the lowering would not
#    compile even though its text looks right.
pipe = object.__new__(G.Pipeline)
pipe.uds, pipe.resources, pipe.samplers, pipe.uses_sampler_heap = [], {}, {}, False
common = ' '.join(pipe.common())
check('the helper is emitted in the common preamble', 'uint mul_hi_u32(uint a, uint b)' in common, common[:200])
check('the helper uses halfword products, not a 64-bit multiply',
      'a & 0xffffu' in common and 'b >> 16' in common, common[:200])

# 5. A neighbouring operation is untouched: the low-half multiply still lowers to a plain product.
VOP3A_MUL_LO_U32 = 0x169
asm, emitted = translate(encode_vop3a(VOP3A_MUL_LO_U32, 1, [V1, V2]))
check('the low multiply is unchanged', emitted == 'v1 = v1 * v2;', emitted)

# ---- value-level check of the emitted helper --------------------------------------------------------------
# The reference is the exact product, so a wrong carry in the helper must fail even though the emitted text is
# unchanged. The evaluator covers the integer operators this helper uses, with C precedence.
TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+u?|[A-Za-z_]\w*|>>|<<|[&|^()*+-])')


def tokens(source):
    out, position = [], 0
    while position < len(source):
        match = TOKEN.match(source, position)
        if not match:
            raise ValueError('cannot tokenise %r' % source[position:position + 20])
        out.append(match.group(1))
        position = match.end()
    return out


class Expr:
    """A recursive-descent evaluator for the 32-bit integer HLSL the mul_hi helper emits."""

    LEVELS = [('|',), ('^',), ('&',), ('>>', '<<'), ('+', '-'), ('*',)]

    def __init__(self, source, variables):
        self.items, self.at, self.vars = tokens(source), 0, variables

    def peek(self):
        return self.items[self.at] if self.at < len(self.items) else ''

    def take(self):
        item = self.peek()
        self.at += 1
        return item

    def value(self):
        return self.binary(0)

    def binary(self, level):
        if level == len(self.LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in self.LEVELS[level]:
            operator = self.take()
            right = self.binary(level + 1)
            if operator == '*':
                left = (left * right) & 0xFFFFFFFF
            elif operator == '+':
                left = (left + right) & 0xFFFFFFFF
            elif operator == '-':
                left = (left - right) & 0xFFFFFFFF
            elif operator == '&':
                left &= right
            elif operator == '^':
                left ^= right
            elif operator == '|':
                left |= right
            elif operator == '>>':
                left >>= right
            elif operator == '<<':
                left = (left << right) & 0xFFFFFFFF
        return left

    def unary(self):
        item = self.take()
        if item == '(':
            value = self.value()
            if self.take() != ')':
                raise ValueError('missing )')
            return value
        if re.fullmatch(r'0x[0-9a-fA-F]+u?', item):
            return int(item.rstrip('u'), 16)
        if re.fullmatch(r'\d+u?', item):
            return int(item.rstrip('u'))
        return self.vars[item]


def evaluate_mul_hi(helper, a, b):
    """Run the emitted helper body for one pair, so what is checked is the value it returns."""
    body = helper[helper.index('{') + 1:helper.rindex('}')]
    variables = {'a': a, 'b': b}
    for statement in (piece.strip() for piece in body.split(';')):
        if not statement:
            continue
        if statement.startswith('uint '):
            for part in statement[4:].split(','):
                name, expr = (piece.strip() for piece in part.split('=', 1))
                variables[name] = Expr(expr, variables).value() & 0xFFFFFFFF
        elif statement.startswith('return '):
            return Expr(statement[7:], variables).value() & 0xFFFFFFFF
    raise ValueError('the helper returned nothing for %#x %#x' % (a, b))


_pipe = object.__new__(G.Pipeline)
_pipe.uds, _pipe.resources, _pipe.samplers, _pipe.uses_sampler_heap = [], {}, {}, False
mul_hi_text = [entry for entry in _pipe.common() if 'uint mul_hi_u32(' in entry][0]
edges = [0, 1, 2, 0xFFFF, 0x10000, 0x10001, 0x7FFFFFFF, 0x80000000,
         0x80000001, 0xFFFEFFFF, 0xFFFF0000, 0xFFFFFFFF, 0x12345678]
pairs = [(a, b) for a in edges for b in edges]
state = 0x2545F491
for _ in range(20000):
    state = (state * 1103515245 + 12345) & 0xFFFFFFFF
    first = state
    state = (state * 1103515245 + 12345) & 0xFFFFFFFF
    pairs.append((first, state))
mismatches = [(a, b, (a * b) >> 32, evaluate_mul_hi(mul_hi_text, a, b)) for a, b in pairs
              if evaluate_mul_hi(mul_hi_text, a, b) != ((a * b) >> 32)]
check('the emitted helper agrees with the exact product on %d pairs' % len(pairs), not mismatches,
      repr(mismatches[:4]))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')