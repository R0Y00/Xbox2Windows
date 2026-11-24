"""`v_cmp_class_f16` in tools/startup/gfx2hlsl.py (issue #1855).

usage: startup_gfx2hlsl_cmp_class_f16_check.py

The binary16 class comparison had no lowering, so the census form `v_cmp_class_f16_e64 s14, 0x7c01, v0` was
refused (#870, issue #1855). `tools/pixel_aot.py` already carries the ten-class binary16 helper; this port
reuses the class-compare branch that handles the binary32 form.

The instruction bytes are **constructed here** from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, VOP3A branch): prefix `0b110101`, opcode in `w0[25:16]`, scalar destination in
`w0[7:0]`, src0 in `w1[8:0]`, src1 in `w1[17:9]`, and source 255 followed by a literal dword.
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


VOPC_CMP_CLASS_F16 = 0x8F
VOPC_CMPX_CLASS_F16 = 0x9F
S14, V0, VCC_LO = 14, 0x100, 106
INLINE_MINUS_ONE = 193
INLINE_ONE_FLOAT = 242
LITERAL_SOURCE = 255


def encode_vopc(op, sdst, src0, src1, literal=None, abs_src0=0, neg_src0=0):
    """Encode a VOPC comparison in its VOP3A form; source 255 is followed by its literal dword."""
    w0 = ((0b110101 << 26) | ((op & 0x3FF) << 16) | (sdst & 0xFF) | ((abs_src0 & 1) << 8))
    w1 = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9) | ((neg_src0 & 1) << 29)
    code = struct.pack('<II', w0, w1)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


def encode_vopc_e32(op, src0, vsrc1, literal=None):
    """Encode a VOPC comparison in its 32-bit form; a cmpx writes EXEC implicitly."""
    w0 = (0b0111110 << 25) | ((op & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)
    code = struct.pack('<I', w0)
    if literal is not None:
        code += struct.pack('<I', literal)
    return code


class Pipeline:
    """Just enough pipeline state for one comparison."""

    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()


class Stage(G.Stage):
    """A Stage with no pipeline: enough state for a decoded instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.masks = set()
        self.pipe = Pipeline()


def translate(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], ' '.join(stage.lines), stage


# 1. The encoder produces the form the decoder reads; a layout change shows up here first.
asm, emitted, _stage = translate(encode_vopc(VOPC_CMP_CLASS_F16, S14, LITERAL_SOURCE, V0, literal=0x7C01))
check('encodes and decodes the census form', asm == ['v_cmp_class_f16_e64 s14, 0x7c01, v0'], repr(asm))
check('the census form lowers through the helper',
      emitted == 's14 = (((class16(0x00007c01u) & v0) != 0u) ? 1u : 0u);', emitted)

# 2. The execution-mask form writes EXEC and stops the straight-line assumption.
asm, emitted, stage = translate(encode_vopc(VOPC_CMPX_CLASS_F16, VCC_LO, V0, INLINE_MINUS_ONE))
check('the exec-mask form writes exec', emitted.startswith('exec = '), emitted)
check('the exec-mask form clears exec_known', stage.exec_known is False)

# 3. The _e32 cmpx spelling prints no destination; it still writes EXEC.
asm, emitted, stage = translate(encode_vopc_e32(VOPC_CMPX_CLASS_F16, V0, 5))
check('encodes and decodes the e32 exec-mask form', asm == ['v_cmpx_class_f16_e32 v0, v5'], repr(asm))
check('the e32 form lowers through the helper', 'class16(v0) & v5' in emitted and emitted.startswith('exec = '), emitted)

# 4. The modifiers on a binary16 value act on bit 15 of the half, not bit 31 of the word.
asm, emitted, _stage = translate(encode_vopc(VOPC_CMP_CLASS_F16, S14, V0, 5, abs_src0=1))
check('an absolute binary16 value clears bit 15', 'class16((v0 & 0x7fffu))' in emitted, emitted)
asm, emitted, _stage = translate(encode_vopc(VOPC_CMP_CLASS_F16, S14, V0, 5, neg_src0=1))
check('a negated binary16 value flips bit 15', 'class16((v0 ^ 0x8000u))' in emitted, emitted)

# 5. A float-printed operand would be lowered as its binary32 bits, so it is refused by name.
_refused = False
try:
    translate(encode_vopc(VOPC_CMP_CLASS_F16, S14, INLINE_ONE_FLOAT, 5))
except G.Unsupported:
    _refused = True
check('a float-printed binary16 operand is refused', _refused)


# ---- value-level check of the emitted helper --------------------------------------------------------------
# The reference below is written from the IEEE binary16 layout, not from the lowering's expressions.
def ref_class16(bits):
    """The LLVM class bit of a binary16 pattern: sNaN qNaN -inf -normal -subnormal -0 +0 +subnormal +normal +inf."""
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    if exponent == 0x1F:
        if mantissa == 0:
            return 0x4 if sign else 0x200
        return 0x2 if mantissa & 0x200 else 0x1
    if exponent == 0:
        if mantissa == 0:
            return 0x20 if sign else 0x40
        return 0x10 if sign else 0x80
    return 0x8 if sign else 0x100


TOKEN = re.compile(r'\s*(0x[0-9a-fA-F]+u?|\d+u?|[A-Za-z_]\w*|>=|<=|==|!=|&&|\|\||[&|<>?:()!~^+-])')


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
    """A recursive-descent evaluator for the integer HLSL the class16 helper emits."""

    LEVELS = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!='), ('>=', '<=', '>', '<'), ('+', '-')]

    def __init__(self, source, variables):
        self.items, self.at, self.vars = tokens(source), 0, variables

    def peek(self):
        return self.items[self.at] if self.at < len(self.items) else ''

    def take(self):
        item = self.peek()
        self.at += 1
        return item

    def value(self):
        return self.ternary()

    def ternary(self):
        condition = self.binary(0)
        if self.peek() == '?':
            self.take()
            yes = self.ternary()
            if self.take() != ':':
                raise ValueError('missing :')
            return yes if condition else self.ternary()
        return condition

    def binary(self, level):
        if level == len(self.LEVELS):
            return self.unary()
        value = self.binary(level + 1)
        while self.peek() in self.LEVELS[level]:
            operator = self.take()
            rhs = self.binary(level + 1)
            if operator == '&':
                value &= rhs
            elif operator == '|':
                value |= rhs
            elif operator == '^':
                value ^= rhs
            elif operator == '==':
                value = int(value == rhs)
            elif operator == '!=':
                value = int(value != rhs)
            elif operator == '>=':
                value = int(value >= rhs)
            elif operator == '<=':
                value = int(value <= rhs)
            elif operator == '>':
                value = int(value > rhs)
            elif operator == '<':
                value = int(value < rhs)
            elif operator == '&&':
                value = int(bool(value) and bool(rhs))
            elif operator == '||':
                value = int(bool(value) or bool(rhs))
            elif operator == '+':
                value = (value + rhs) & 0xFFFFFFFF
            elif operator == '-':
                value = (value - rhs) & 0xFFFFFFFF
        return value

    def unary(self):
        item = self.take()
        if item == '!':
            return int(not self.unary())
        if item == '~':
            return (~self.unary()) & 0xFFFFFFFF
        if item == '(':
            value = self.ternary()
            if self.take() != ')':
                raise ValueError('missing )')
            return value
        if re.fullmatch(r'0x[0-9a-fA-F]+u?', item):
            return int(item.rstrip('u'), 16)
        if re.fullmatch(r'\d+u?', item):
            return int(item.rstrip('u'))
        return self.vars[item]


def evaluate_class16(helper, bits):
    """Run the emitted helper body for one bit pattern, so what is checked is the value it returns."""
    body = helper[helper.index('{') + 1:helper.rindex('}')]
    variables = {'bits': bits}
    for statement in (piece.strip() for piece in body.split(';')):
        if not statement:
            continue
        if statement.startswith('uint '):
            for part in statement[4:].split(','):
                name, expr = (piece.strip() for piece in part.split('=', 1))
                variables[name] = Expr(expr, variables).value() & 0xFFFFFFFF
        elif statement.startswith('bool '):
            name, expr = (piece.strip() for piece in statement[5:].split('=', 1))
            variables[name] = int(bool(Expr(expr, variables).value()))
        elif statement.startswith('if ') and ') return ' in statement:
            condition, consequence = statement[3:].split(') return ', 1)
            if Expr(condition[1:], variables).value():
                return Expr(consequence, variables).value() & 0xFFFFFFFF
        elif statement.startswith('return '):
            return Expr(statement[7:], variables).value() & 0xFFFFFFFF
    raise ValueError('the helper returned nothing for %#x' % bits)


_pipe = object.__new__(G.Pipeline)
_pipe.uds, _pipe.resources, _pipe.samplers, _pipe.uses_sampler_heap = [], {}, {}, False
class16_text = [entry for entry in _pipe.common() if 'uint class16(' in entry][0]
grid = list(range(0x10000))
_mismatches = [(bits, ref_class16(bits), evaluate_class16(class16_text, bits)) for bits in grid
               if evaluate_class16(class16_text, bits) != ref_class16(bits)]
check('the emitted helper agrees with the binary16 classifier on all 65536 patterns',
      not _mismatches, repr(_mismatches[:4]))

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')