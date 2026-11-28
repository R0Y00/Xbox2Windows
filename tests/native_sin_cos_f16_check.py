"""Binary16 VOP1/VOP3/SDWA v_sin_f16 and v_cos_f16 in direct Compute DXIL (#3298, Refs #2935)."""
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

import f16_packed_model as M
import native_f16_alu_check as F
from native_compute_clear_check import write_request
from native_sdwa_f16_check import ABS, NEG, PAD, PRESERVE, SCALAR, WORD_0, WORD_1, DWORD, OLD, sdwa


TRIG_OPS = {'sin': 0x60, 'cos': 0x61}
TWO_PI_F32 = struct.unpack('>d', bytes.fromhex('401921FB60000000'))[0]


def _f32_literal(token):
    if token.startswith('0x'):
        raw = token[2:].zfill(16)
        return struct.unpack('<f', struct.pack('<f', struct.unpack('>d', bytes.fromhex(raw))[0]))[0]
    return struct.unpack('<f', struct.pack('<f', float(token)))[0]


def _f32_bits(val):
    return struct.unpack('<I', struct.pack('<f', val))[0]


def expected_trig_f16(op, bits):
    """Exact model matching direct_compute's trig_turns + packed_half_rne + specials."""
    bits &= 0xFFFF
    if (bits & 0x7FFF) >= 0x7C00:
        return 0x7E00
    if op == 'sin' and bits == 0x8000:
        return 0x8000
    a = struct.unpack('<f', struct.pack('<f', M.h2f(bits)))[0]
    part = a if a == 0.0 else struct.unpack('<f', struct.pack('<f', a - math.trunc(a)))[0]
    mag = abs(part)
    exact_zero = (mag == 0.25 or mag == 0.75) if op == 'cos' else (mag == 0.5)
    if exact_zero:
        widened = 0.0
    else:
        radians = struct.unpack('<f', struct.pack('<f', part * TWO_PI_F32))[0]
        widened = struct.unpack('<f', struct.pack('<f', math.cos(radians) if op == 'cos' else math.sin(radians)))[0]
    return F.packed_half_rne(_f32_bits(widened))


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    encoding: str  # 'vop1', 'vop3', 'inline', 'inline_int', 'literal', 'sgpr', 'overlap', 'sdwa'
    src_val: int = 0
    abs_mask: int = 0
    neg: int = 0
    inline_code: int = 240
    sdwa_dst: int = WORD_0
    sdwa_unused: int = PRESERVE
    sdwa_src0: int = WORD_0

    def source_half(self):
        if self.encoding == 'inline':
            return F.INLINE_F16[self.inline_code]
        if self.encoding == 'inline_int':
            return (192 - self.inline_code) & 0xFFFF if self.inline_code >= 193 else (self.inline_code - 128) & 0xFFFF
        if self.encoding == 'sdwa':
            sel = self.sdwa_src0 & 7
            picked = (self.src_val >> 16) if sel == WORD_1 else (self.src_val & 0xFFFF)
            return F.modified(picked, bool(self.sdwa_src0 & ABS), bool(self.sdwa_src0 & NEG), 0)
        return F.modified(self.src_val & 0xFFFF, self.abs_mask, self.neg, 0)

    def want(self):
        return expected_trig_f16(self.op, self.source_half())

    def emit(self):
        opcode = TRIG_OPS[self.op]
        if self.encoding == 'sdwa':
            return (F.mov(1, self.src_val) + F.mov(3, OLD) +
                    sdwa(opcode, self.sdwa_dst, self.sdwa_unused, self.sdwa_src0, unary=True))
        if self.encoding == 'overlap':
            return F.mov(3, F.HI_A | self.src_val) + F.vop1(opcode, 3, 259)
        if self.encoding == 'literal':
            return F.mov(3, F.HI_A) + F.vop1(opcode, 3, 255) + [self.src_val]
        if self.encoding == 'sgpr':
            return F.smov(8, F.HI_B | self.src_val) + F.mov(3, F.HI_A) + F.vop1(opcode, 3, 8)
        if self.encoding in ('inline', 'inline_int'):
            return F.mov(3, F.HI_A) + F.vop1(opcode, 3, self.inline_code)
        if self.encoding == 'vop3':
            return F.mov(1, F.HI_B | self.src_val) + F.mov(3, F.HI_A) + F.vop3(
                opcode, 3, 257, 0, self.abs_mask, self.neg, unary=True)
        return F.mov(1, F.HI_B | self.src_val) + F.mov(3, F.HI_A) + F.vop1(opcode, 3, 257)

    def matches(self, got, *, hardware=False):
        want_h = self.want()
        if self.encoding == 'sdwa':
            shift = 16 if self.sdwa_dst == WORD_1 else 0
            mask = 0xFFFF << shift
            other = OLD & ~mask if self.sdwa_unused == PRESERVE else 0
            if (got & ~mask) != other:
                return False
            got_h = (got & mask) >> shift
        else:
            if (got >> 16) != (F.HI_A >> 16):
                return False
            got_h = got & 0xFFFF
        if not hardware or M.is_nan(want_h) or want_h in (0x0000, 0x8000, 0x3C00, 0xBC00):
            return F.same_binary16(got_h, want_h)
        return abs(M.h2f(got_h) - M.h2f(want_h)) <= 0.001


def all_cases():
    cases = [
        # Exact specials & quarter/half turns across both VOP1 and VOP3.
        Case('sin_pos_zero',    'sin', 'vop1', 0x0000),
        Case('sin_neg_zero',    'sin', 'vop1', 0x8000),
        Case('cos_pos_zero',    'cos', 'vop1', 0x0000),
        Case('cos_neg_zero',    'cos', 'vop1', 0x8000),
        Case('sin_qtr_pos',     'sin', 'vop1', 0x3400),  # +0.25 -> +1.0
        Case('sin_qtr_neg',     'sin', 'vop1', 0xB400),  # -0.25 -> -1.0
        Case('cos_qtr_pos',     'cos', 'vop1', 0x3400),  # +0.25 -> +0.0
        Case('cos_qtr_neg',     'cos', 'vop1', 0xB400),  # -0.25 -> +0.0
        Case('sin_half_pos',    'sin', 'vop1', 0x3800),  # +0.5  -> +0.0
        Case('sin_half_neg',    'sin', 'vop1', 0xB800),  # -0.5  -> +0.0
        Case('cos_half_pos',    'cos', 'vop1', 0x3800),  # +0.5  -> -1.0
        Case('cos_half_neg',    'cos', 'vop1', 0xB800),  # -0.5  -> -1.0
        Case('sin_3qtr_pos',    'sin', 'vop3', 0x3A00),  # +0.75 -> -1.0
        Case('cos_3qtr_pos',    'cos', 'vop3', 0x3A00),  # +0.75 -> +0.0 (kills drop 0.75)
        Case('cos_3qtr_neg',    'cos', 'vop3', 0xBA00),  # -0.75 -> +0.0
        Case('cos_1_75_neg',    'cos', 'vop1', 0xBF00),  # -1.75 -> +0.0
        Case('sin_int_2048',    'sin', 'vop1', 0x6800),  # 2048.0 -> +0.0
        Case('cos_max_norm',    'cos', 'vop1', 0x7BFF),  # 65504.0 -> +1.0
        Case('sin_pos_inf',     'sin', 'vop1', 0x7C00),  # +Inf -> NaN
        Case('cos_neg_inf',     'cos', 'vop3', 0xFC00),  # -Inf -> NaN
        Case('sin_qnan',        'sin', 'vop1', 0x7E01),  # NaN -> NaN
        Case('cos_snan',        'cos', 'vop3', 0x7D02),  # sNaN -> NaN
        # General angles, modifiers (abs/neg), source kinds, overlap, and SDWA.
        Case('sin_0125',        'sin', 'vop1', 0x3000),  # 0.125
        Case('cos_0125_neg',    'cos', 'vop3', 0x3000, neg=1),
        Case('sin_0375_abs',    'sin', 'vop3', 0xB600, abs_mask=1),
        Case('cos_00625',       'cos', 'vop1', 0x2C00),  # 0.0625
        Case('sin_subnormal',   'sin', 'vop1', 0x0001),
        Case('cos_inline_f16',  'cos', 'inline', inline_code=240),     # 0.5 -> -1.0
        Case('sin_inline_int',  'sin', 'inline_int', inline_code=193), # -1 -> +0.0
        Case('cos_literal',     'cos', 'literal', 0x12343800),         # low half 0.5 -> -1.0
        Case('sin_sgpr',        'sin', 'sgpr', 0x3400),
        Case('cos_overlap',     'cos', 'overlap', 0x3800),
        Case('sin_sdwa_w1_pres', 'sin', 'sdwa', 0x34003800, sdwa_dst=WORD_1, sdwa_unused=PRESERVE, sdwa_src0=WORD_1),
        Case('cos_sdwa_w0_pad',  'cos', 'sdwa', 0x34003800, sdwa_dst=WORD_0, sdwa_unused=PAD, sdwa_src0=DWORD | ABS),
    ]
    assert len(cases) < F.STRIDE
    return cases


def program(cases):
    code = F.preamble()
    for slot, c in enumerate(cases):
        code += c.emit() + F.store(slot * 4, 3)
    return code + [0xBF810000]


def refusal_programs():
    return [
        (F.vop3(TRIG_OPS['sin'], 3, 257, 0, clamp=1, unary=True) + [0xBF810000], 'unsupported v_sin_f16 clamp or omod'),
        (F.vop3(TRIG_OPS['cos'], 3, 257, 0, omod=1, unary=True) + [0xBF810000],  'unsupported v_cos_f16 clamp or omod'),
        (F.vop3(TRIG_OPS['sin'], 3, 257, 0, opsel=1, unary=True) + [0xBF810000], 'unsupported v_sin_f16 high-half op_sel'),
        (F.vop1(TRIG_OPS['cos'], 3, 250) + [0x010101B1, 0xBF810000],             'unsupported v_cos_f16 DPP form'),
        (F.vop1(TRIG_OPS['sin'], 3, 233) + [0x01010101, 0xBF810000],             'unsupported v_sin_f16 DPP form'),
        ([0xBFA40004] + F.vop1(TRIG_OPS['sin'], 3, 257) + [0xBF810000],          'unsupported binary16 rounding or denormal mode'),
        ([0xBFA50000] + F.vop1(TRIG_OPS['cos'], 3, 257) + [0xBF810000],          'unsupported binary16 rounding or denormal mode'),
        (sdwa(TRIG_OPS['sin'], WORD_0, PRESERVE, WORD_1 | SCALAR, reg0=242, unary=True) + [0xBF810000],
         'unsupported SDWA binary16 inline constant selection'),
    ]


def _float_val(token, values):
    if token.startswith('%'):
        kind, bits = values[token]
        assert kind == 'float', (token, kind)
        return struct.unpack('<f', struct.pack('<I', bits))[0]
    return _f32_literal(token)


def interpret_trig(ir_text):
    body = ir_text.split('define void @main()', 1)[1].split('\n}', 1)[0]
    blocks, label, buf = {}, None, []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == '{':
            continue
        if line.endswith(':') and ' ' not in line:
            if label is not None:
                blocks[label] = buf
            label, buf = line[:-1], []
            continue
        buf.append(line)
    if label is not None:
        blocks[label] = buf
    values, memory, stores = {}, {}, []
    label, steps = 'entry', 0
    while steps < 200000:
        steps += 1
        followed = False
        for line in blocks[label]:
            if line.startswith('br '):
                label = F._branch(line, values)
                followed = True
                break
            if line == 'ret void':
                return stores
            if (m := re.fullmatch(r'(%\w+) = fcmp (\w+) float (\S+), (\S+)', line)):
                pred, a, b = m.group(2), _float_val(m.group(3), values), _float_val(m.group(4), values)
                if pred == 'oeq':
                    res = (not math.isnan(a)) and (not math.isnan(b)) and (a == b)
                elif pred == 'ogt':
                    res = (not math.isnan(a)) and (not math.isnan(b)) and (a > b)
                else:
                    raise AssertionError(line)
                values[m.group(1)] = ('i1', int(res))
            elif (m := re.fullmatch(r'(%\w+) = select i1 (\S+), float (\S+), float (\S+)', line)):
                val = _float_val(m.group(3) if F._i1(m.group(2), values) else m.group(4), values)
                values[m.group(1)] = ('float', _f32_bits(val))
            elif (m := re.fullmatch(r'(%\w+) = fmul float (\S+), (\S+)', line)):
                res = _float_val(m.group(2), values) * _float_val(m.group(3), values)
                values[m.group(1)] = ('float', _f32_bits(struct.unpack('<f', struct.pack('<f', res))[0]))
            elif (m := re.fullmatch(r'(%\w+) = call float @dx\.op\.unary\.f32\(i32 (\d+), float (\S+)\)', line)):
                op_id, x = int(m.group(2)), _float_val(m.group(3), values)
                if op_id == 6:
                    y = abs(x)
                elif op_id == 12:
                    y = math.cos(x) if math.isfinite(x) else math.nan
                elif op_id == 13:
                    y = math.sin(x) if math.isfinite(x) else math.nan
                elif op_id == 26:
                    y = float(math.trunc(x)) if math.isfinite(x) else x
                    if x == 0.0:
                        y = x
                else:
                    raise AssertionError(line)
                values[m.group(1)] = ('float', _f32_bits(struct.unpack('<f', struct.pack('<f', y))[0]))
            else:
                F._exec(line, values, memory, stores)
        if not followed:
            raise AssertionError('block fell through: ' + label)
    raise AssertionError('step limit')


def ir(dumper):
    items = all_cases()
    code = program(items)
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'sin-cos-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(interpret_trig(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(items))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(items[i].want()))
               for i, (_, got) in enumerate(stores) if not items[i].matches(got)]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_sin_cos_f16 IR: {len(items)} cases matched; {len(refusals)} refusals verified')
    return 0


def controls(cli, sdk, work, root):
    items = all_cases()
    sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x80 | (5 << 1))
    marker = write_request(work, program(items), root, sa)
    output = marker.with_suffix('.cso')
    res = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                         capture_output=True, text=True)
    assert res.returncode == 0 and output.is_file(), (res.stdout, res.stderr)
    refusals = refusal_programs()
    for idx, (refused, why) in enumerate(refusals):
        m = write_request(work, refused, root, sa)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o],
                           capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o
    print(f'native_sin_cos_f16 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = all_cases()
    sa = struct.pack('<7I', 0, 0, 0, 32 | (1 << 16), 1 | (1 << 20), 0, 0x80 | (5 << 1))
    marker = write_request(work, program(items), root, sa)
    dest = work / 'out.bin'
    run = subprocess.run([replay, sdk, marker, work / 'cache', '0', dest, '--direct'])
    if run.returncode == 77:
        return 77
    run.check_returncode()
    actual = struct.unpack(f'<{F.WORDS}I', dest.read_bytes())
    results = {F.BASE + group * F.STRIDE + idx: items[idx]
               for group in range(F.GROUPS) for idx in range(len(items))}
    bad = []
    for index, got in enumerate(actual):
        if index in results:
            if not results[index].matches(got, hardware=True):
                bad.append((index, results[index].name, hex(got), hex(results[index].want())))
        elif got != F.SENTINEL:
            bad.append((index, 'guard', hex(got), hex(F.SENTINEL)))
    assert not bad, bad[:10]
    print(f'native_sin_cos_f16 hardware: {len(items) * F.GROUPS} words matched with D[31:16] preserved and guards')
    return 0


def main(mode, tool, sdk=None):
    if mode == 'ir':
        return ir(tool)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, root = work / 'root.hlsl', work / 'root.bin'
        src.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, src], check=True)
        return {'controls': controls, 'hardware': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
