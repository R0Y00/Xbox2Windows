"""Per-lane loops in the mesh-stage translator (mesh2hlsl MeshStage.perlane_loop).

The compiler keeps the lanes that still loop in a mask M and loops while some lane does (SCC = M != 0):

    v_and_b32 v6, 3, v5                 (each lane's own trip count - 1)
    v_mov_b32 v1, 0
    s_mov_b64 s[6:7], exec; s_mov_b64 s[8:9], exec; s_mov_b32 s4, 0
 H: s_mov_b64 exec, s[8:9]              (the lanes still looping)
    v_add_f32 v1, 1.0, v1
    v_cmp_eq_u32 vcc, s4, v6            (this lane is done after this iteration)
    s_add_u32 s4, s4, 1
    s_and_b64 exec, vcc, exec           (exec = the lanes that are done)
    s_andn2_b64 s[8:9], s[8:9], exec    (they leave M; SCC = some lane remains)
    s_cbranch_scc1 H
    s_mov_b64 exec, s[6:7]
    exp pos0 v1, ...

On the console each lane runs (v5 & 3) + 1 iterations of its vector instructions (the wave runs the most), so it
exports float((v5 & 3) + 1). Hand-assembled programs (no cache bytes); RDNA 2 ISA encodings. Checks:
  full     every lane on: lanes 0..7 and 127 export float((v5 & 3) + 1)
  partial  exec narrowed first (v_cmpx_ne_u32 0, v5 & 3): those lanes the same, the others nothing
  refused  s4 read after the loop (the hardware leaves the wave's last count, a thread its own), an s_barrier
           in the loop, no exec restore after it, an SCC not from `s_andn2_b64 M, M, exec` (s_xor)
and dxc ms_6_5 + ps_6_0 compile the full program. The emitted code is evaluated per lane (mesh_waterfall_check's
interpreter). Negative control: with X2W_MESH_PERLANE_LOOPS=0 the same checks FAIL (run in a child): the loop
is refused as before.
"""
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
import mesh2hlsl as M  # noqa: E402
from startup_image_3d_check import root_blob  # noqa: E402
from mesh_waterfall_check import run_lane, TAIL, PS, NARROW, words, refused  # noqa: E402

PRE = [0x360c0a83,                  # v_and_b32 v6, 3, v5
       0x7e020280,                  # v_mov_b32 v1, 0
       0xbe86047e,                  # s_mov_b64 s[6:7], exec
       0xbe88047e,                  # s_mov_b64 s[8:9], exec
       0xbe840380]                  # s_mov_b32 s4, 0
BODY = [0xbefe0408,                 # H: s_mov_b64 exec, s[8:9]
        0x060202f2,                 # v_add_f32 v1, 1.0, v1
        0x7d840c04,                 # v_cmp_eq_u32 vcc, s4, v6
        0x80048104,                 # s_add_u32 s4, s4, 1
        0x87fe7e6a]                 # s_and_b64 exec, vcc, exec
EXIT = 0x8a887e08                   # s_andn2_b64 s[8:9], s[8:9], exec
BACK = 0xbf85fff9                   # s_cbranch_scc1 H
RESTORE = 0xbefe0406                # s_mov_b64 exec, s[6:7]

FULL = words(PRE, BODY, [EXIT, BACK, RESTORE], TAIL)
PARTIAL = words(NARROW, PRE, BODY, [EXIT, BACK, RESTORE], TAIL)
LIVE_AFTER = words(PRE, BODY, [EXIT, BACK, RESTORE, 0x7e020c04], TAIL)      # v_cvt_f32_u32 v1, s4 after it
BARRIER = words(PRE, BODY[:1], [0xbf8a0000], BODY[1:], [EXIT, 0xbf85fff8, RESTORE], TAIL)
NO_RESTORE = words(PRE, BODY, [EXIT, BACK, 0xbf800000], TAIL)              # s_nop 0 in place of the restore
XOR = words(PRE, BODY, [0x89887e08, BACK, RESTORE], TAIL)                   # s_xor_b64 s[8:9], s[8:9], exec
TEXT = ['v_and_b32_e32 v6, 3, v5', 'v_mov_b32_e32 v1, 0', 's_mov_b64 s[6:7], exec', 's_mov_b64 s[8:9], exec',
        's_mov_b32 s4, 0', 's_mov_b64 exec, s[8:9]', 'v_add_f32_e32 v1, 1.0, v1', 'v_cmp_eq_u32_e32 vcc_lo, s4, v6',
        's_add_u32 s4, s4, 1', 's_and_b64 exec, vcc, exec', 's_andn2_b64 s[8:9], s[8:9], exec',
        's_cbranch_scc1 label_0x14', 's_mov_b64 exec, s[6:7]']


def hlsl_of(code):
    return M.MeshPipeline(code, None, root_blob()).hlsl_ms()


def check():
    assert [x[3] for x in G.decode(FULL)][:len(TEXT)] == TEXT, [x[3] for x in G.decode(FULL)]
    full = hlsl_of(FULL)
    for v5 in list(range(8)) + [127]:
        got = run_lane(full, v5)
        assert got == float((v5 & 3) + 1), ('full', v5, got)
    partial = hlsl_of(PARTIAL)
    for v5 in list(range(8)) + [64, 127]:
        got = run_lane(partial, v5)
        want = None if v5 % 4 == 0 else float((v5 & 3) + 1)
        assert got == want, ('partial', v5, got, want)
    refused(LIVE_AFTER, 's4 written in it is live after it')
    refused(BARRIER, 's_barrier')
    refused(NO_RESTORE, 'no s_mov_b64 exec restore')
    refused(XOR, 'SCC not from the remaining-lanes mask')
    compiled = 'dxc not found'
    if os.path.isfile(M.DXC):
        with tempfile.TemporaryDirectory() as d:
            r = M.translate_to(d, 1, FULL, PS, root_blob())
            assert r == 'ok', r
        compiled = 'dxc ms_6_5 + ps_6_0 ok'
    return compiled


def main():
    if os.environ.get('X2W_MESH_PERLANE_LOOPS') == '0':      # child of the negative control
        check()
        return
    compiled = check()
    env = dict(os.environ, X2W_MESH_PERLANE_LOOPS='0')
    r = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'Unsupported' in r.stderr, ('negative control passed', r.stdout, r.stderr)
    print('mesh per-lane loop: lanes 0..7/127 (full) and v5 %% 4 != 0 (partial) export float((v5 & 3) + 1); '
          'live-after, barrier, no-restore and xor refused; %s; X2W_MESH_PERLANE_LOOPS=0 fails (%s)'
          % (compiled, r.stderr.strip().splitlines()[-1][:90]))


if __name__ == '__main__':
    main()
