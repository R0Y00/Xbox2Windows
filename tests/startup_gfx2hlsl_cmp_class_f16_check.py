"""v_cmp_class_f16 and v_cmpx_class_f16 in tools/startup/gfx2hlsl.py (no game data).

  (no arguments)        translation controls, pure Python; the emitted statements are also compiled as C++ when a
                        compiler is found and run over all 65,536 binary16 values and a set of class masks
  binary <llvm-mc>      the spellings the checks use assemble to the recorded encodings (SRC0 the binary16 value,
                        SRC1 the class mask)

The operation tests which of ten classes the binary16 value in the low half of SRC0 falls in, against the mask in SRC1:
bit 0 signalling NaN, 1 quiet NaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf (AMD RDNA 2
ISA reference). The reference below classifies through Python's own binary16 conversion, not through the bit tests of the
translation; only the signalling bit (the top mantissa bit) is read from the encoding, because no float API keeps it.
Not covered: SDWA spellings and op_sel are refused, not translated; a draw on a GPU has not been run.
"""
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
import gfx2hlsl  # noqa: E402

CLASS_NAMES = ('signalling NaN', 'quiet NaN', '-inf', '-normal', '-denormal', '-0', '+0', '+denormal', '+normal', '+inf')
MASKS = [0, 0x3ff, 0xffffffff, 0x1, 0x2, 0x3, 0x204, 0x300, 0x60, 0x18, 0x90, 0x155, 0x2aa, 0xfffffc00, 0x400]
MASKS += [1 << bit for bit in range(10)]


def reference(half_bits):
    """The class bit of a binary16 encoding, 0 to 9."""
    value = struct.unpack('<e', struct.pack('<H', half_bits))[0]
    negative = bool(half_bits & 0x8000)
    if math.isnan(value):
        return 1 if half_bits & 0x0200 else 0
    if math.isinf(value):
        return 2 if negative else 9
    if value == 0:
        return 5 if negative else 6
    if abs(value) < 2.0 ** -14:
        return 4 if negative else 7
    return 3 if negative else 8


def stage(exec_known=True):
    s = gfx2hlsl.Stage.__new__(gfx2hlsl.Stage)  # no machine code: the translation state alone
    s.lines, s.sym, s.used_s, s.used_v, s.uniform_v, s.masks = [], {}, set(), set(), set(), set()
    s.exec_known = exec_known
    return s


def emit(name, ops, mods=(), exec_known=True):
    """The statements one instruction translates to (the dispatcher's own entry, name with its encoding suffix)."""
    s = stage(exec_known)
    s.one(name, '%s %s%s' % (name, ', '.join(ops), ''.join(' ' + m for m in mods)))
    return s.lines, s


def find_compiler():
    for name in ('g++', 'c++', 'clang++'):
        if shutil.which(name):
            return shutil.which(name)
    return None


HARNESS = r'''
#include <cstdint>
#include <cstdio>
typedef unsigned uint;
static const uint masks[] = {%(masks)s};
int main() {
    for (uint h = 0; h < 65536u; ++h) {
        for (uint k = 0; k < sizeof(masks) / sizeof(masks[0]); ++k) {
            uint v1 = h, v2 = masks[k], vcc = 0, exec = %(exec)s, s4 = 0;
%(body)s
            std::putchar((%(result)s) ? '1' : '0');
        }
    }
    return 0;
}
'''


def run_cpp(compiler, lines, result, work, exec_start='1u'):
    source = work / 'class.cpp'
    source.write_text(HARNESS % {'masks': ', '.join('%du' % m for m in MASKS), 'body': '\n'.join(lines),
                                 'result': result, 'exec': exec_start})
    binary = work / 'class.bin'
    subprocess.run([compiler, '-std=c++17', '-O1', str(source), '-o', str(binary)], check=True, capture_output=True)
    return subprocess.run([str(binary)], check=True, capture_output=True).stdout.decode()


def expected_text(transform, mask_sets=MASKS):
    out = []
    for h in range(65536):
        bit = reference(transform(h))
        out.extend('1' if mask & (1 << bit) else '0' for mask in mask_sets)
    return ''.join(out)


IDENTITY = lambda h: h  # noqa: E731
FORMS = (
    # (instruction, operands, result variable, transform of the encoding the instruction reads)
    ('v_cmp_class_f16_e32', ['vcc_lo', 'v1', 'v2'], 'vcc', IDENTITY),
    ('v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'], 'vcc', IDENTITY),
    ('v_cmp_class_f16_e64', ['s4', 'v1', 'v2'], 's4', IDENTITY),
    ('v_cmpx_class_f16_e32', ['v1', 'v2'], 'exec', IDENTITY),
    ('v_cmpx_class_f16_e64', ['v1', 'v2'], 'exec', IDENTITY),
    ('v_cmp_class_f16_e64', ['vcc_lo', '-v1', 'v2'], 'vcc', lambda h: h ^ 0x8000),
    ('v_cmp_class_f16_e64', ['vcc_lo', '|v1|', 'v2'], 'vcc', lambda h: h & 0x7fff),
    ('v_cmp_class_f16_e64', ['vcc_lo', '-|v1|', 'v2'], 'vcc', lambda h: (h & 0x7fff) ^ 0x8000),
)


def check_semantics(failures):
    compiler = find_compiler()
    if not compiler:
        print('no C++ compiler: the translated statements were not executed')
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, ops, result, transform in FORMS:
            lines, _ = emit(name, ops)
            got = run_cpp(compiler, lines, '%s != 0u' % result if result != 's4' else 's4 != 0u', work)
            want = expected_text(transform)
            if got != want:
                at = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
                failures.append('%s %s: first difference at half 0x%04x, mask 0x%x (got %s, expected %s)'
                                % (name, ', '.join(ops), at // len(MASKS), MASKS[at % len(MASKS)], got[at], want[at]))
        print('%d forms x 65,536 binary16 values x %d masks: translated statements match the reference'
              % (len(FORMS), len(MASKS)))

        # Controls: the check must fail for each of these planted faults in the emitted text.
        lines, _ = emit('v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'])
        text = '\n'.join(lines)
        faults = (
            ('signalling bit moved', '0x0200u', '0x0100u'),
            ('denormal threshold', '0x0400u', '0x0401u'),
            ('infinity threshold', '0x7c00u', '0x7bffu'),
            ('+denormal and -denormal swapped', '(c16n ? 0x10u : 0x80u)', '(c16n ? 0x80u : 0x10u)'),
            ('+0 and -0 swapped', '(c16n ? 0x20u : 0x40u)', '(c16n ? 0x40u : 0x20u)'),
            ('mask ignored', '(c16c & v2)', '(c16c | v2)'),
        )
        for label, old, new in faults:
            if old not in text:
                failures.append('control %s: the emitted text no longer contains %r' % (label, old))
                continue
            got = run_cpp(compiler, text.replace(old, new).split('\n'), 'vcc != 0u', work)
            if got == expected_text(IDENTITY):
                failures.append('control %s: the planted fault was not detected' % label)
        print('%d planted faults in the emitted text were each detected' % len(faults))


def check_translation(failures):
    lines, s = emit('v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'])
    text = ' '.join(lines)
    if 'vcc = ' not in text or '& v2)' not in text or '& 0xffffu' not in text:
        failures.append('e64 to vcc: unexpected statements %r' % lines)
    if 1 not in s.used_v or 2 not in s.used_v:
        failures.append('e64 to vcc: v1 and v2 not both recorded as read (%s)' % sorted(s.used_v))

    # A narrowed EXEC stays narrowed: the result is intersected with it.
    lines, _ = emit('v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'], exec_known=False)
    if 'vcc = (exec & ' not in ' '.join(lines):
        failures.append('narrowed EXEC: the result is not intersected with exec: %r' % lines)
    lines, s = emit('v_cmpx_class_f16_e32', ['v1', 'v2'])
    if 'exec = ' not in ' '.join(lines) or s.exec_known:
        failures.append('v_cmpx_class_f16: exec must be written and no longer known to equal the entry mask')
    # A scalar destination records a lane mask.
    lines, s = emit('v_cmp_class_f16_e64', ['s4', 'v1', 'v2'])
    if 's4 = ' not in ' '.join(lines) or 4 not in s.masks:
        failures.append('scalar destination: s4 must be written and recorded as a lane mask: %r' % lines)
    # The mask operand may be an inline constant: -1 means every class.
    lines, _ = emit('v_cmp_class_f16_e64', ['vcc_lo', 'v1', '-1'])
    if '0xffffffffu' not in ' '.join(lines):
        failures.append('inline mask -1 was not read as all ones: %r' % lines)
    print('translation controls: vcc, scalar and exec destinations, narrowed EXEC, inline mask')


REFUSED = (
    ('op_sel on the value', 'v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'], ['op_sel:[1,0]']),
    ('clamp', 'v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2'], ['clamp']),
    ('inline float value', 'v_cmp_class_f16_e64', ['vcc_lo', '1.0', 'v2'], []),
    ('literal value', 'v_cmp_class_f16_e64', ['vcc_lo', '0x3c00', 'v2'], []),
    ('unbalanced abs', 'v_cmp_class_f16_e64', ['vcc_lo', '|v1', 'v2'], []),
    ('too few operands', 'v_cmp_class_f16_e64', ['v1'], []),
    ('too many operands', 'v_cmp_class_f16_e64', ['vcc_lo', 'v1', 'v2', 'v3'], []),
    ('SDWA spelling', 'v_cmp_class_f16_sdwa', ['vcc_lo', 'v1', 'v2'], []),
)


def check_refusals(failures):
    for label, name, ops, mods in REFUSED:
        try:
            lines, _ = emit(name, ops, mods)
        except gfx2hlsl.Unsupported:
            continue
        failures.append('%s: accepted and translated to %r' % (label, lines))
    print('%d unmodelled forms are refused' % len(REFUSED))


ENCODINGS = (
    ('v_cmp_class_f16_e32 vcc_lo, v1, v2', '01051e7d'),
    ('v_cmp_class_f16_e64 s0, v1, v2', '00008fd401050200'),
    ('v_cmpx_class_f16_e32 v1, v2', '01053e7d'),
    ('v_cmp_class_f16_e64 vcc_lo, -|v1|, v2', '6a018fd401050220'),
    ('v_cmp_class_f16_e64 vcc_lo, v1, 0x3ff', '6a008fd401ff0100ff030000'),
)


def check_binary(llvm_mc):
    for source, recorded in ENCODINGS:
        run = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'], input=source + '\n',
                             capture_output=True, text=True, check=True)
        m = re.search(r'encoding: \[([^\]]*)\]', run.stdout)
        built = ''.join('%02x' % int(b, 16) for b in m.group(1).split(',')) if m else ''
        if built != recorded:
            raise AssertionError('%s: assembled %s, recorded %s' % (source, built, recorded))
    print('%d spellings assemble to the recorded encodings' % len(ENCODINGS))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
        return 0
    failures = []
    check_translation(failures)
    check_refusals(failures)
    check_semantics(failures)
    for failure in failures:
        print('FAIL: ' + failure)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
