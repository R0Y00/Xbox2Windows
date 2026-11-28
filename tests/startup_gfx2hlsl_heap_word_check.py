"""A compute program whose resource heap record index is a word it reads from a table buffer.

usage: startup_gfx2hlsl_heap_word_check.py [--llvm-mc <llvm-mc>]

The program of startup_gfx2hlsl_heap_record_check.py with its index no longer the root constant itself: root constant
word 2 picks the byte offset of a word in the buffer of table byte 96, and that word is the index:

    s_load_dwordx4 s[28:31], s[12:13], 0x60
    s_mulk_i32 s6, 0x150
    s_add_u32 s6, s6, 0x120
    s_buffer_load_dword s6, s[28:31], s6
    s_lshl_b32 s6, s6, 5
    s_load_dwordx4 s[16:19], s[8:9], s6

gfx2hlsl.py names that input `srv 1 buf heap word table 1 96 at 288 rootconst 0 2 times 336`: the dispatch reads the
word at byte 288 + 336 * (root constant word 2) of that buffer, then the 32-byte heap record it indexes. The byte
offset is the same for every thread of the dispatch. A literal byte offset, an offset modifier, s_mul_i32 and
s_addk_i32 and a later word of a wider load give the same line with their own numbers; the HLSL and the manifest of
the first form are compared whole. Every other form keeps a refusal: an offset from the group ID, an offset read
from a buffer, a scale or a byte offset that is not a multiple of 4, a register added after the arithmetic, a
modified descriptor, and a branch target between the arithmetic and the load. --llvm-mc reassembles every
instruction used here and compares the words.
"""
import argparse

import startup_gfx2hlsl_heap_record_check as H
from startup_gfx2hlsl_compute_check import G, assemble, refused

TABLE = 's_load_dwordx4 s[28:31], s[12:13], 0x60'
SCALE, ADD = 's_mulk_i32 s6, 0x150', 's_add_u32 s6, s6, 0x120'
READ_INDEX = 's_buffer_load_dword s6, s[28:31], s6'
WAIT = 's_waitcnt lgkmcnt(0)'
INDEX = (TABLE, SCALE, ADD, WAIT, READ_INDEX, WAIT, H.SHIFT)
# Every instruction added here, recorded from llvm-mc 15.
ENCODED = {
    TABLE: '060708f4600000fa', SCALE: '500106b8', ADD: '06ff068020010000', READ_INDEX: '8e0120f40000000c',
    's_buffer_load_dwordx2 s[26:27], s[28:31], s6': '8e0624f40000000c', 's_lshl_b32 s6, s27, 5': '1b85068f',
    's_buffer_load_dword s6, s[28:31], 0x40': '8e0120f4400000fa', 's_buffer_load_dword s6, s[28:31], s14': '8e0120f40000001c',
    's_mulk_i32 s6, 0x152': '520106b8', 's_add_u32 s6, s6, 0x122': '06ff068022010000',
    's_and_b32 s29, s29, 0xffff': '1dff1d87ffff0000', 's_add_u32 s6, s6, s14': '060e0680',
    's_mul_i32 s6, s6, s5': '06050693', 's_mul_i32 s6, s6, 0x150': '06ff069350010000', 's_addk_i32 s6, 0x120': '200186b7',
    's_buffer_load_dword s6, s[28:31], s6 offset:0x10': '8e0120f41000000c',
}
ALL = {**H.ENCODED, **ENCODED}

MANIFEST = """\
x2w-manifest 2
stage cs
group 2 3 4
ud 3
udsrc 0 rootconst 0 0
udsrc 1 rootconst 0 1
udsrc 2 rootconst 0 2
srv 0 buf table 1 96
srv 1 buf heap word table 1 96 at %s
uav 0 table 1 16
"""
FORM = '288 rootconst 0 2 times 336'
BINDINGS = """\
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; };
ByteAddressBuffer r0 : register(t0);
ByteAddressBuffer r1 : register(t1);
RWByteAddressBuffer u0 : register(u0);
"""
# The heap record program's body with the index read through r0 (the table buffer) and the input through r1.
MAIN = H.MAIN.replace('    s6 = ud2 << (0x00000005u & 31);\n', """\
    s6 = ud2 * 0x00000150u;
    scc = (s6 + 0x00000120u) < (s6);
    s6 = s6 + 0x00000120u;
    s6 = r0.Load(s6 + 0);
    s6 = s6 << (0x00000005u & 31);
""").replace('v5 = r0.Load(a + 0);', 'v5 = r1.Load(a + 0);')
assert MAIN.count('r0.Load') == 1 and MAIN.count('r1.Load') == 1


def words(index=INDEX):
    """The heap record program with the index lines in place of its shift."""
    return b''.join(bytes.fromhex(ALL[line]) for original in H.PROGRAM
                    for line in (index if original == H.SHIFT else (original,)))


WORDS = words()


def pipeline(index=INDEX):
    return G.ComputePipeline(words(index), H.root_blob(), H.RSRC2, H.GROUP)


def replace(old, *new):
    assert old in INDEX
    return tuple(line for original in INDEX for line in (new if original == old else (original,)))


def check_lowering():
    pipe = pipeline()
    hlsl = pipe.hlsl_cs()
    assert pipe.manifest() == MANIFEST % FORM, pipe.manifest()
    assert hlsl.endswith(BINDINGS + MAIN), hlsl[hlsl.index('cbuffer'):]
    forms = (
        ('literal byte offset', (TABLE, WAIT, 's_buffer_load_dword s6, s[28:31], 0x40', WAIT, H.SHIFT), '64'),
        ('offset modifier', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], s6 offset:0x10'),
         '304 rootconst 0 2 times 336'),
        ('s_mul_i32 and s_addk_i32', replace(ADD, 's_addk_i32 s6, 0x120'), FORM),
        ('s_mul_i32 by a literal', replace(SCALE, 's_mul_i32 s6, s6, 0x150'), FORM),
        ('second word of a wider load', replace(H.SHIFT, 's_lshl_b32 s6, s27, 5')[:4] + (
            's_buffer_load_dwordx2 s[26:27], s[28:31], s6', WAIT, 's_lshl_b32 s6, s27, 5'), '292 rootconst 0 2 times 336'),
        # A branch target before the arithmetic changes nothing.
        ('branch target before the arithmetic', (TABLE,) + H.JUMP + INDEX[1:], FORM),
    )
    for name, index, form in forms:
        got = pipeline(index).manifest()
        assert got == MANIFEST % form, '%s:\n%s' % (name, got)
    return len(forms)


def check_refusals():
    other = ('s_load of a resource heap record whose index is read from a buffer other than a table buffer at a '
             'dispatch-constant offset')
    controls = (
        ('offset from the group ID', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], s14')),
        ('offset read from a buffer', replace(READ_INDEX, 's_buffer_load_dword s6, s[28:31], 0x40', WAIT, READ_INDEX)),
        ('scale not a multiple of 4', replace(SCALE, 's_mulk_i32 s6, 0x152')),
        ('byte offset not a multiple of 4', replace(ADD, 's_add_u32 s6, s6, 0x122')),
        ('a register added after the arithmetic', replace(ADD, ADD, 's_add_u32 s6, s6, s14')),
        ('scaled by a register', replace(SCALE, 's_mul_i32 s6, s6, s5')),
        ('modified descriptor', replace(TABLE, TABLE, WAIT, 's_and_b32 s29, s29, 0xffff')),
        ('branch target between the arithmetic and the load', replace(ADD, ADD, *H.JUMP)),
    )
    for name, index in controls:
        try:
            refused(other, lambda: pipeline(index).manifest())
        except AssertionError as error:
            raise AssertionError('%s: %s' % (name, error)) from None
    # A branch target between the load and the shift ends the word's mark, as for a root constant.
    refused('whose offset is not a root constant shifted left by 5',
            lambda: pipeline(replace(READ_INDEX, READ_INDEX, *H.JUMP)).manifest())
    # Only a buffer read is named in the manifest; another kind through such a record is refused there.
    pipe = pipeline()
    key = next(k for k in pipe.resources if k[0][:2] == ('resheap', 'word'))
    pipe.resources[key]['kind'] = 'tex2d'
    refused("resource through ('resheap', 'word'", pipe.manifest)
    return len(controls) + 2


def check_binary(llvm_mc):
    for line, encoded in ENCODED.items():
        assert assemble(llvm_mc, line + '\n') == bytes.fromhex(encoded), line
    text = [line for original in H.PROGRAM for line in (INDEX if original == H.SHIFT else (original,))]
    assert assemble(llvm_mc, '\n'.join(text) + '\n') == WORDS
    print('heap word fixture: %d instructions reassembled to the recorded words' % len(ENCODED))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llvm-mc')
    args = parser.parse_args()
    if args.llvm_mc:
        check_binary(args.llvm_mc)
    forms = check_lowering()
    count = check_refusals()
    print('startup_gfx2hlsl_heap_word: PASS (manifest and HLSL exact; %d more forms; %d refused controls)' % (forms, count))
