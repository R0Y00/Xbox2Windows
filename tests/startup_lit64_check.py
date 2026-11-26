"""A 32-bit literal in a 64-bit integer operand: accepted when bit 31 is clear (high dword 0 under either extension).

Synthetic VOP3 words: v_lshrrev_b64 v[5:6], 0, <literal> (op 0x300, src0 = inline 0, src1 = 255 + literal). LLVM
AMDGPUDisassembler::decodeLiteralConstant zero-extends a literal for OPERAND_REG_IMM_INT64 (with a ToDo on
signedness) and the assembler encodes Lo_32; zero- and sign-extension agree exactly when bit 31 is clear.
Controls: 0x80000000 zero-extends (an unsigned shift, see startup_gfx2hlsl_shift64_literal_check.py); the inline constant 1 keeps its old lowering.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402


class Pipe:
    root = {'params': [], 'flags': 0}

    def ud(self, source):
        return 0


def one(words):
    s = G.Stage(Pipe(), 'ps', b'')
    s.lines = []
    _, n, name, text = G.decode(struct.pack('<%dI' % len(words), *words))[0]
    s.pc = 0
    s.one(name, text)
    return text, '\n'.join(s.lines)


def main():
    text, body = one([0xd7000005, 128 | 255 << 9, 0x3f800000])
    assert text == 'v_lshrrev_b64 v[5:6], 0, 0x3f800000', text
    assert 'x2w_lshr64_lo(0x3f800000u, 0x00000000u, 0x00000000u)' in body, body
    assert 'x2w_lshr64_hi(0x3f800000u, 0x00000000u, 0x00000000u)' in body, body
    # v_lshrrev_b64 reads S1.u64, so a literal with bit 31 set zero-extends too
    # (startup_gfx2hlsl_shift64_literal_check.py; other 64-bit operands keep the refusal there).
    text, body = one([0xd7000005, 128 | 255 << 9, 0x80000000])
    assert 'x2w_lshr64_lo(0x80000000u, 0x00000000u, 0x00000000u)' in body, body
    text, body = one([0xd7000005, 128 | 129 << 9])
    assert text == 'v_lshrrev_b64 v[5:6], 0, 1' and 'x2w_lshr64_lo(0x00000001u, 0x00000000u, 0x00000000u)' in body, body
    print('lit64: bit-31-clear literal -> high dword 0; 0x80000000 zero-extended; inline 1 unchanged')


if __name__ == '__main__':
    main()
