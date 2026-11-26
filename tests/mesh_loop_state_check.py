"""Loop-carried scalar state in the mesh-stage translator (mesh2hlsl MeshStage.loop_one / MeshPipeline.mesh_stage).

Hand-assembled NGG mesh programs (no cache bytes), each a uniform scalar loop (s_cbranch_scc1 back edge, RDNA 2 ISA:
SCC from s_cmp_lt_u32, the branch taken when SCC = 1):

  counter:  s0 = 0; L: s0 += 1; s_cmp_lt_u32 s0, 3; s_cbranch_scc1 L; v1 = float(s0); exp pos0 v1, ...
  midentry: s2 = 0; s1 = 5; s_branch E; H: s1 = s4; E: s4 = s1 + 1; s2 += 1; s_cmp_lt_u32 s2, 3; s_cbranch_scc1 H;
            v1 = float(s4)   (H is reached only by the back edge: the loop is entered in its body)

The console computes s0 = 3 and s4 = 8. A translation that keeps, at the loop head, the state of the instruction
above it folds the constant before the loop into the body (`s0 = 0x00000000u + 0x00000001u`: the HLSL loop never
ends and stops at its guard). Checks: the loop counters are read as variables in the loop body, and the program
exports the right value when the emitted body is evaluated by a small interpreter for the statement forms it
uses.

  taint:    v1 = 0; s0 = 0; H: v_cmp_lt_u32 vcc, 0, v1; s_cbranch_vccz +1; s_nop; v1 = v5; s0 += 1; s_cmp_lt_u32 s0, 3;
            s_cbranch_scc1 H   (v1 is uniform on the entry edge and the vertex id on the back edge)

The vccz branch reads a vcc that differs between lanes from the second pass on: the translation must refuse it
("lane-dependent vcc"), which it does only when the head's lane taint joins the back edge. (Since e1a2ad2a, the
base's loop-head rule alone keeps counter / midentry right; taint is what still needs the join.)
Negative control: with X2W_MESH_LOOP_JOIN=0 (the old linear state) the same checks FAIL (run in a child).
"""
import os
import re
import struct
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
import mesh2hlsl as M  # noqa: E402
from startup_image_3d_check import root_blob  # noqa: E402

TAIL = (0x7e020c00,                # v_cvt_f32_u32 v1, s<src>
        0x7e040280,                # v_mov_b32 v2, 0
        0xb07c1003,                # s_movk_i32 m0, 0x1003
        0xbf900009,                # s_sendmsg 0x9
        0xf80008cf, 0x02020201,    # exp pos0 v1, v2, v2, v2 done
        0xf8000941, 0x00000002,    # exp prim v2 done
        0xbf810000)                # s_endpgm


def program(words, src):
    tail = list(TAIL)
    tail[0] = 0x7e020c00 | src
    return struct.pack('<%dI' % (len(words) + len(tail)), *(list(words) + tail))


COUNTER = program([0xbe800380,     # 0x00 s_mov_b32 s0, 0
                   0x80008100,     # 0x04 s_add_u32 s0, s0, 1
                   0xbf0a8300,     # 0x08 s_cmp_lt_u32 s0, 3
                   0xbf85fffd],    # 0x0c s_cbranch_scc1 0x04
                  0)
MIDENTRY = program([0xbe820380,    # 0x00 s_mov_b32 s2, 0
                    0xbe810385,    # 0x04 s_mov_b32 s1, 5
                    0xbf820001,    # 0x08 s_branch 0x10
                    0xbe810304,    # 0x0c s_mov_b32 s1, s4
                    0x80048101,    # 0x10 s_add_u32 s4, s1, 1
                    0x80028102,    # 0x14 s_add_u32 s2, s2, 1
                    0xbf0a8302,    # 0x18 s_cmp_lt_u32 s2, 3
                    0xbf85fffb],   # 0x1c s_cbranch_scc1 0x0c
                   4)
TAINT = program([0x7e020280,      # 0x00 v_mov_b32 v1, 0
                 0xbe800380,      # 0x04 s_mov_b32 s0, 0
                 0x7d820280,      # 0x08 v_cmp_lt_u32 vcc, 0, v1
                 0xbf860001,      # 0x0c s_cbranch_vccz 0x14
                 0xbf800000,      # 0x10 s_nop 0
                 0x7e020305,      # 0x14 v_mov_b32 v1, v5
                 0x80008100,      # 0x18 s_add_u32 s0, s0, 1
                 0xbf0a8300,      # 0x1c s_cmp_lt_u32 s0, 3
                 0xbf85fff9],     # 0x20 s_cbranch_scc1 0x08
                0)
TEXT = {
    'counter': ['s_mov_b32 s0, 0', 's_add_u32 s0, s0, 1', 's_cmp_lt_u32 s0, 3', 's_cbranch_scc1 label_0x4'],
    'midentry': ['s_mov_b32 s2, 0', 's_mov_b32 s1, 5', 's_branch label_0x10', 's_mov_b32 s1, s4',
                 's_add_u32 s4, s1, 1', 's_add_u32 s2, s2, 1', 's_cmp_lt_u32 s2, 3', 's_cbranch_scc1 label_0xc'],
    'taint': ['v_mov_b32_e32 v1, 0', 's_mov_b32 s0, 0', 'v_cmp_lt_u32_e32 vcc_lo, 0, v1', 's_cbranch_vccz label_0x14',
              's_nop 0', 'v_mov_b32_e32 v1, v5', 's_add_u32 s0, s0, 1', 's_cmp_lt_u32 s0, 3', 's_cbranch_scc1 label_0x8'],
}


def run_body(hlsl):
    """Evaluates the scalar statements of the emitted pc loop (uint assignments, scc compares, pc dispatch) for
    one thread and returns the exported position x. Only the forms these programs produce are understood."""
    lines = [l.strip() for l in hlsl.split('uint pc = 0;', 1)[1].splitlines()]
    blocks, cur = {}, None
    for l in lines:
        if l.startswith('GroupMemoryBarrierWithGroupSync'):
            break
        m = re.match(r'(?:else )?if \(pc == (\d+)\) \{$', l)
        if m:
            cur = int(m.group(1))
            blocks[cur] = []
        elif l.startswith('[loop] for ('):   # a bounded pc loop around a loop's blocks (the dispatch below repeats)
            continue
        elif cur is not None and l != '}':
            blocks[cur].append(l)
    env = {'pc': 0, 'scc': 0, 'exec': 1}

    def ev(expr):
        e = re.sub(r'0x([0-9a-f]+)u', lambda mm: str(int(mm.group(1), 16)), expr)
        e = re.sub(r'\b(\d+)u\b', r'\1', e)
        e = e.replace('U(float(', '(float(').replace('F(', '(')
        e = re.sub(r'\((scc)\) \? (\d+) : (\d+)', r'(\2 if \1 else \3)', e)
        e = e.replace('&&', ' and ').replace('||', ' or ')
        return eval(e, {'float': float}, env)
    for _ in range(1000):
        pc = env['pc']
        if pc >= 0xffff or pc not in blocks:
            break
        for stmt in blocks[pc]:
            for part in [p.strip() for p in stmt.split(';') if p.strip()]:
                m = re.match(r'if \(exec & 1\) \{ o\.pos = float4\((.*?), .*$', part)
                if m:
                    env['posx'] = ev(m.group(1))
                    continue
                if part.startswith(('if (exec & 1) { x2w_prim', 'if (exec & 1) x2w_alloc', '}')):
                    continue
                m = re.match(r'(\w+) = (.*)$', part)
                assert m, 'statement form %r' % part
                value = ev(m.group(2))
                env[m.group(1)] = (int(value) & 0xffffffff) if isinstance(value, (int, bool)) else value
    else:
        raise AssertionError('the translated loop does not end')
    return env.get('posx')


def check():
    # A loop entered from its middle. The forward jump skips an alignment
    # NOP before the head; its only real predecessor is the back edge.
    padded = MIDENTRY[:8] + struct.pack('<II', 0xbf820002, 0xbf800000) + MIDENTRY[12:]
    pipeline = M.MeshPipeline(padded, None, root_blob())
    assert run_body(pipeline.hlsl_ms()) == 8.0
    if M.LOOP_JOIN:
        assert not pipeline.vs._loop_entry[0x10]
    for name, code, want, counter in (('counter', COUNTER, 3.0, 's0 = s0 + 0x00000001u;'),
                                      ('midentry', MIDENTRY, 8.0, 's2 = s2 + 0x00000001u;')):
        text = [x[3] for x in G.decode(code)][:len(TEXT[name])]
        assert text == TEXT[name], (name, text)
        hlsl = M.MeshPipeline(code, None, root_blob()).hlsl_ms()
        assert counter in hlsl, (name, 'loop counter folded', [l for l in hlsl.splitlines() if ' s0 =' in l or ' s2 =' in l])
        got = run_body(hlsl)
        assert got == want, (name, got, want)
    text = [x[3] for x in G.decode(TAINT)][:len(TEXT['taint'])]
    assert text == TEXT['taint'], ('taint', text)
    try:
        M.MeshPipeline(TAINT, None, root_blob()).hlsl_ms()
        raise AssertionError('taint: vccz on a vcc that is lane-dependent through the back edge was translated')
    except G.Unsupported as e:
        assert 'lane-dependent vcc' in str(e), ('taint', str(e))
    return 'counter -> 3.0, mid-body entry -> 8.0, back-edge taint refused'


def main():
    if os.environ.get('X2W_MESH_LOOP_JOIN') == '0':      # child of the negative control
        check()
        return
    summary = check()
    env = dict(os.environ, X2W_MESH_LOOP_JOIN='0')
    r = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'AssertionError' in r.stderr, ('negative control passed', r.stdout, r.stderr)
    print('mesh loop state: %s; X2W_MESH_LOOP_JOIN=0 fails (%s)' % (summary, r.stderr.strip().splitlines()[-1][:80]))


if __name__ == '__main__':
    main()
