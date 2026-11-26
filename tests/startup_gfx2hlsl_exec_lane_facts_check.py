"""Conflict registers read where the value cannot matter, a back-edge-only loop head, and vcc as a spare pair for
descriptor words (tools/startup/gfx2hlsl.py). Pure Python.

- ANDN2: on one path into a join s16 holds a descriptor word, on the other a lane mask; the join computes
  `s_andn2_b64 s[32:33], s[16:17], exec` while this lane's exec bit is proven 1 (fall-through of s_cbranch_execz,
  exec not written since), so bit 0 of the result is 0 whatever s16 holds. Controls: no execz proof, and an AND
  (which exec 1 does not absorb), are refused.
- PRUNE: the lane-bit facts decide a branch condition (s_andn2 of a known 1 with exec 1 gives scc 0, so the
  s_cbranch_scc1 edge is ruled out); a second scc1 edge into the join keeps the ANDN2 proof.
- LOOP: a loop rotated so its head is entered only by the back edge; the head reads s16, which the body writes as a
  run-time number on every path. Controls: a body path that skips the write, and a body that reloads a descriptor.
- VCC: `s_mov_b64 vcc, s[20:21]` of descriptor words, s[20:21] reused, `s_mov_b64 s[20:21], vcc` back, then
  image_load through s[16:23]. Controls: a vcc read in between, and a path into the copy back that skips the copy in.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from startup_gfx2hlsl_join_symbols_check import EXEC, S_AND_B64, S_ANDN2_B64, cmp_ne, root, sop2, sopp  # noqa: E402
from startup_flat_param_check import G, VS_WORDS  # noqa: E402

LOAD = bytes.fromhex('04040cf4000000fa')  # s_load_dwordx8 s[16:23], s[8:9], 0x0
LOAD_BODY = bytes.fromhex('04040cf4200000fa')  # s_load_dwordx8 s[16:23], s[8:9], 0x20
ZERO_V = bytes.fromhex('8002007e' '8002027e')  # v0 = 0, v1 = 0
EXPORT = bytes.fromhex('0f1800f800000000' '000081bf')  # exp mrt0 v0 x4 done vm; s_endpgm
SCC1, VCCZ, EXECZ, BRANCH = 5, 6, 8, 2
VCC = 106


def sop1(op, dst, src):
    return struct.pack('<I', 0xbe800000 | dst << 16 | op << 8 | src)


def sopc(op, src0, src1):
    return struct.pack('<I', 0xbf000000 | op << 16 | src1 << 8 | src0)


S_MOV_B32, S_MOV_B64 = 3, 4
S_CMP_LG_U32, S_CMP_LT_U32 = 7, 10
S_ADD_U32 = 0


def ps(words):
    return G.Pipeline(VS_WORDS, words, root('t', 1)).hlsl_ps().splitlines()


def refused(fragment, words):
    try:
        ps(words)
    except G.Unsupported as reason:
        assert fragment in str(reason), str(reason)
        return
    raise AssertionError('accepted, expected a refusal naming %r' % fragment)


def andn2(op=S_ANDN2_B64, execz=True, prune=False):
    """s16 = desc; exec &= (v1 != 0); [execz -> end]; vcc = v1 != 0; vccz -> L; s16 = (v0 != 0); L: s32 = s16 op exec."""
    tail = sop2(op, 32, 16, EXEC) + bytes.fromhex('2002007e') + EXPORT  # v0 = s32
    fall = cmp_ne(16, 0)
    if prune:
        # s[34:35] = exec; s34 = s34 & ~exec (0, scc 0); s_cbranch_scc1 into L (a second forward edge)
        fall = cmp_ne(16, 0) + sop1(S_MOV_B64, 34, EXEC)
        fall += sop2(S_ANDN2_B64, 34, 34, EXEC) + sopp(SCC1, 0)
    head = LOAD + ZERO_V + cmp_ne(30, 1) + sop2(S_AND_B64, EXEC, EXEC, 30)
    skip = cmp_ne(VCC, 1) + sopp(VCCZ, len(fall) // 4) + fall
    body = skip + tail
    end = bytes.fromhex('000081bf')
    if execz:
        return head + sopp(EXECZ, len(body) // 4) + body + end
    return head + body + end


def check_andn2():
    words = andn2()
    names = [asm for *_, asm in G.decode(words)]
    assert names[5:10] == ['s_cbranch_execz label_0x48', 'v_cmp_ne_i32_e64 vcc_lo, 0, v1', 's_cbranch_vccz label_0x34',
                          'v_cmp_ne_i32_e64 s16, 0, v0', 's_andn2_b64 s[32:33], s[16:17], exec'], names
    hlsl = ps(words)
    line, = (l for l in hlsl if l.startswith('        s32 = '))
    assert line == '        s32 = (s16 & ~exec) & 1;', hlsl
    for word in (0, 1, 0xffffffff, 0x12345679):
        assert eval(line.split('= ', 1)[1].rstrip(';'), {'s16': word, 'exec': 1}) == 0
    refused('numeric use of s16 holding conflict', andn2(execz=False))
    refused('numeric use of s16 holding conflict', andn2(op=S_AND_B64))


def check_prune():
    words = andn2(prune=True)
    assert 's_cbranch_scc1' in [name for _o, _n, name, _a in G.decode(words)]
    assert '        s32 = (s16 & ~exec) & 1;' in ps(words)
    # the facts themselves: scc is 0 after s34 = (s34 & ~exec) & 1 with s34 = exec = 1, so the scc1 edge is ruled out
    facts = G.lane_facts_after(['s34 = exec;', 's34 = (s34 & ~exec) & 1;', 'scc = s34 != 0;'], {'exec': ('b', 1)})
    assert facts['s34'] == ('v', 0) and facts['scc'] == ('v', 0), facts
    taken, fall = G.lane_branch_edges('scc', facts)
    assert taken is None and fall is not None
    taken, fall = G.lane_branch_edges('(exec & 1) == 0', {})
    assert taken == {'exec': ('b', 0)} and fall == {'exec': ('b', 1)}
    # conditional and braced assignments forget, they are not evaluated
    assert 's5' not in G.lane_facts_after(['if (exec & 1) s5 = 1;'], {'s5': ('v', 0)})
    assert 's5' not in G.lane_facts_after(['{ s5 = 1; }'], {'s5': ('v', 0)})
    assert 's5' not in G.lane_facts_after(['s5 = s6 + 1;'], {'s5': ('v', 0), 's6': ('v', 0)})
    assert G.lane_facts_merge([{'exec': ('v', 1)}, {'exec': ('b', 1)}, None]) == {'exec': ('b', 1)}


def loop(skip_write=False, reload=False):
    """s16 = desc; s24 = 0; s_branch BODY; HEAD: s_cmp_lg_u32 s16, 0; BODY: s24 += 1; s16 = s24; s_cmp_lt_u32 s24, 4;
    s_cbranch_scc1 HEAD; v0 = s16."""
    write = sop1(S_MOV_B32, 16, 24)
    if skip_write:
        write = sopc(S_CMP_LG_U32, 24, 129) + sopp(SCC1, 1) + write
    if reload:
        write = LOAD_BODY + bytes.fromhex('7fc08cbf')
    body = sop2(S_ADD_U32, 24, 24, 129) + write + sopc(S_CMP_LT_U32, 24, 132)
    back = -(len(body) // 4 + 1 + 1)  # to HEAD, one word before BODY
    return (LOAD + bytes.fromhex('7fc08cbf') + sop1(S_MOV_B32, 24, 128) + sopp(BRANCH, 1) +
            sopc(S_CMP_LG_U32, 16, 128) + body + sopp(SCC1, back) + bytes.fromhex('1002007e') + EXPORT)


def check_loop():
    asm = [a for *_, a in G.decode(loop())]
    assert asm[3:9] == ['s_branch label_0x18', 's_cmp_lg_u32 s16, 0', 's_add_u32 s24, s24, 1', 's_mov_b32 s16, s24',
                        's_cmp_lt_u32 s24, 4', 's_cbranch_scc1 label_0x14'], asm
    hlsl = ps(loop())
    assert any(l.strip().startswith('scc = s16 != 0') for l in hlsl), hlsl
    refused('numeric use of s16 holding conflict', loop(skip_write=True))
    refused('numeric use of s16 holding', loop(reload=True))


def vcc_copy(read_between=False, skip_copy_in=False):
    """s[16:23] = desc; vcc = s[20:21]; s20 = 0; [v2 = vcc ? v1 : v0]; s[20:21] = vcc; image_load s[16:23]."""
    middle = sop1(S_MOV_B32, 20, 128)
    if read_between:
        middle += struct.pack('<I', 0x02000000 | 2 << 17 | 1 << 9 | 256)  # v_cndmask_b32_e32 v2, v0, v1, vcc_lo
    copy_in = sop1(S_MOV_B64, VCC, 20)
    pre = b''
    if skip_copy_in:
        pre = cmp_ne(30, 1) + sopc(S_CMP_LG_U32, 30, 128) + sopp(SCC1, len(copy_in) // 4)
    return (LOAD + bytes.fromhex('7fc08cbf') + ZERO_V + pre + copy_in + middle + sop1(S_MOV_B64, 20, VCC) +
            bytes.fromhex('080f00f000040400' '703f8cbf' '0f1800f804050607' '000081bf'))


def check_vcc():
    asm = [a for *_, a in G.decode(vcc_copy())]
    assert asm[4:7] == ['s_mov_b64 vcc, s[20:21]', 's_mov_b32 s20, 0', 's_mov_b64 s[20:21], vcc'], asm
    hlsl = ps(vcc_copy())
    assert not any('vcc' in l for l in hlsl if l.startswith('        ') and 'uint vcc' not in l), hlsl
    refused('mask from descriptor data', vcc_copy(read_between=True))
    refused('mask from descriptor data', vcc_copy(skip_copy_in=True))


def main():
    check_andn2()
    check_prune()
    check_loop()
    check_vcc()
    print('conflict registers absorbed by a proven exec bit, a back-edge-only loop head and vcc descriptor copies pass;'
          ' every control refuses')


if __name__ == '__main__':
    main()
