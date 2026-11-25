"""Waterfall loops in the mesh-stage translator (mesh2hlsl MeshStage.waterfalls / waterfall_step).

The compiler's loop for a value that must be uniform but differs between lanes (LLVM AMDGPU
SIInstrInfo emitLoadScalarOpsFromVGPRLoop):

    s_mov_b64 s[6:7], exec; s_mov_b64 s[8:9], exec
 H: v_readfirstlane_b32 s4, v5          (the first remaining lane's value)
    v_cmpx_eq_u32 s4, v5                (exec = the remaining lanes holding it)
    v_cvt_f32_u32 v1, s4                (the body: uses s4 as a uniform value)
    s_andn2_wrexec_b64 s[8:9], s[8:9]   (the lanes done leave; SCC = some lane remains)
    s_cbranch_scc1 H
    s_mov_b64 exec, s[6:7]
    exp pos0 v1, ...

On the console each lane active at H makes exactly one pass, with s4 = its own v5 (v5 = the vertex id, different in
every lane), so it exports float(v5). Hand-assembled programs (no cache bytes); RDNA 2 ISA encodings. Checks:
  full     every lane on: lanes 0, 1, 37, 127 export float(v5)
  partial  exec narrowed first (v_cmpx_ne_u32 0, v5 & 3): lanes with v5 % 4 != 0 export float(v5), the
           others nothing (every wave keeps active lanes: s_sendmsg is guarded by exec here)
  refused  s4 read after the loop (the hardware leaves the last pass's value), no exec restore after the loop,
           an s_barrier in the body
and dxc ms_6_5 + ps_6_0 compile the full program. The emitted body is evaluated per lane by a small interpreter
for the statement forms it uses. Negative control: with X2W_MESH_WATERFALL=0 the same checks FAIL (run in a
child): the loops are refused as before (here at the lane-mask SCC publish inside the loop).
"""
import os
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
import mesh2hlsl as M  # noqa: E402
from startup_image_3d_check import root_blob  # noqa: E402

NARROW = [0x360c0a83,              # v_and_b32 v6, 3, v5
          0x7daa0c80]              # v_cmpx_ne_u32_e32 0, v6 (every fourth lane off, in every wave)
LOOP = [0xbe86047e,                 # s_mov_b64 s[6:7], exec
        0xbe88047e,                 # s_mov_b64 s[8:9], exec
        0x7e080505,                 # H: v_readfirstlane_b32 s4, v5
        0x7da40a04,                 # v_cmpx_eq_u32_e32 s4, v5
        0x7e020c04,                 # v_cvt_f32_u32_e32 v1, s4
        0xbe883a08,                 # s_andn2_wrexec_b64 s[8:9], s[8:9]
        0xbf85fffb,                 # s_cbranch_scc1 H
        0xbefe0406]                 # s_mov_b64 exec, s[6:7]
TAIL = [0x7e040280,                 # v_mov_b32 v2, 0
        0xb07c1003,                 # s_movk_i32 m0, 0x1003
        0xbf900009,                 # s_sendmsg 0x9
        0xf80008cf, 0x02020201,     # exp pos0 v1, v2, v2, v2 done
        0xf8000941, 0x00000002,     # exp prim v2 done
        0xbf810000]                 # s_endpgm
PS = struct.pack('<4I', 0x7e0002f2, 0xf800180f, 0x00000000, 0xbf810000)   # v_mov v0, 1.0; exp mrt0; s_endpgm


def words(*parts):
    w = [x for p in parts for x in p]
    return struct.pack('<%dI' % len(w), *w)


FULL = words(LOOP, TAIL)
PARTIAL = words(NARROW, LOOP, TAIL)
LIVE_AFTER = words(LOOP, [0x7e020c04], TAIL)                    # v_cvt_f32_u32 v1, s4 after the restore
NO_RESTORE = words(LOOP[:-1], [0xbf800000], TAIL)                # s_nop 0 in place of the restore
BARRIER = words(LOOP[:4], [0xbf8a0000], LOOP[4:6], [0xbf85fffa], LOOP[7:], TAIL)   # s_barrier in the body
TEXT = ['s_mov_b64 s[6:7], exec', 's_mov_b64 s[8:9], exec', 'v_readfirstlane_b32 s4, v5',
        'v_cmpx_eq_u32_e32 s4, v5', 'v_cvt_f32_u32_e32 v1, s4', 's_andn2_wrexec_b64 s[8:9], s[8:9]',
        's_cbranch_scc1 label_0x8', 's_mov_b64 exec, s[6:7]']


def run_lane(hlsl, v5):
    """Evaluates the emitted pc loop for one thread whose v5 (vertex id) is v5; returns the exported position x,
    or None when the thread exports nothing. Only the statement forms these programs produce are understood."""
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
    env = {'pc': 0, 'scc': 0, 'exec': 1, 'v5': v5}

    def ev(expr):
        e = re.sub(r'0x([0-9a-f]+)u', lambda mm: str(int(mm.group(1), 16)), expr)
        e = re.sub(r'\b(\d+)u\b', r'\1', e)
        e = re.sub(r'\b[UF]\(', '(', e)
        e = re.sub(r'\(\((.*?)\) \? (\d+) : (\d+)\)', r'(\2 if (\1) else \3)', e)
        e = re.sub(r'^\((\w+)\) \? (\d+) : (\d+)$', r'(\2 if \1 else \3)', e)
        e = re.sub(r'~(\w+)', r'(~\1 & 0xffffffff)', e)
        e = e.replace('&&', ' and ').replace('||', ' or ').replace('false', '0').replace('true', '1')
        return eval(e, {'float': float}, env)
    posx = None
    for _ in range(100):
        pc = env['pc']
        if pc >= 0xffff or pc not in blocks:
            break
        for stmt in blocks[pc]:
            guarded = stmt.startswith('if (exec & 1) ')
            body = stmt[len('if (exec & 1) '):] if guarded else stmt
            if guarded and not (env['exec'] & 1):
                continue
            body = body.strip('{} ')
            for part in [p.strip() for p in body.split(';') if p.strip()]:
                m = re.match(r'o\.pos = float4\((.*?), .*$', part)
                if m:
                    posx = ev(m.group(1))
                    continue
                if part.startswith(('x2w_prim', 'x2w_alloc')):
                    continue
                m = re.match(r'(\w+) = (.*)$', part)
                assert m, 'statement form %r' % part
                value = ev(m.group(2))
                env[m.group(1)] = (int(value) & 0xffffffff) if isinstance(value, (int, bool)) else value
    else:
        raise AssertionError('the translated program does not end')
    return posx


def hlsl_of(code):
    return M.MeshPipeline(code, None, root_blob()).hlsl_ms()


def refused(code, why):
    try:
        hlsl_of(code)
    except G.Unsupported as e:
        assert why in str(e), (why, str(e))
        return
    raise AssertionError('not refused: %s' % why)


def check():
    assert [x[3] for x in G.decode(FULL)][:len(TEXT)] == TEXT, [x[3] for x in G.decode(FULL)]
    full = hlsl_of(FULL)
    for v5 in (0, 1, 37, 127):
        got = run_lane(full, v5)
        assert got == float(v5), ('full', v5, got)
    partial = hlsl_of(PARTIAL)
    for v5 in (0, 1, 2, 3, 4, 64, 65, 127):
        got = run_lane(partial, v5)
        want = None if v5 % 4 == 0 else float(v5)
        assert got == want, ('partial', v5, got, want)
    refused(LIVE_AFTER, 's4 written in it is live after it')
    refused(NO_RESTORE, 'no s_mov_b64 exec restore')
    refused(BARRIER, 's_barrier')
    compiled = 'dxc not found'
    if os.path.isfile(M.DXC):
        with tempfile.TemporaryDirectory() as d:
            r = M.translate_to(d, 1, FULL, PS, root_blob())
            assert r == 'ok', r
        compiled = 'dxc ms_6_5 + ps_6_0 ok'
    return compiled


def main():
    if os.environ.get('X2W_MESH_WATERFALL') == '0':      # child of the negative control
        check()
        return
    compiled = check()
    env = dict(os.environ, X2W_MESH_WATERFALL='0')
    r = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'Unsupported' in r.stderr, ('negative control passed', r.stdout, r.stderr)
    print('mesh waterfall: one pass per lane, full exec lanes 0/1/37/127 and partial exec lanes v5 %% 4 != 0 export '
          'float(v5); live-after, no-restore and barrier refused; %s; X2W_MESH_WATERFALL=0 fails (%s)'
          % (compiled, r.stderr.strip().splitlines()[-1][:90]))


if __name__ == '__main__':
    main()
