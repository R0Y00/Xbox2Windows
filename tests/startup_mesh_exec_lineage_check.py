"""Exec lineage in the mesh-stage translator (mesh2hlsl MeshStage.lineage_*, X2W_MESH_EXEC_LINEAGE).

A VGPR written from uniform values while exec is E holds one value in every lane of E (the other lanes keep their
old value). The epoch rule forgets that at the first exec write that may widen exec; the compiler's usual shape
restores a SAVED copy of an earlier exec, after which only lanes of E can be active again. Hand-assembled NGG mesh
programs (no cache bytes), RDNA 2 ISA encodings, v5 = the vertex id (different in every lane):

  0x00 v_cmp_lt_u32 vcc, 3, v5;   0x04 s_and_saveexec_b64 s[26:27], vcc   E1 = lanes v5 > 3 (s26 = entry exec)
  0x08 v_mov_b32 v1, 1.0                                                   v1 = 1.0 in every lane of E1
  0x0c v_cmp_lt_u32 vcc, 7, v5;   0x10 s_and_saveexec_b64 s[18:19], vcc   E2 = lanes v5 > 7 (s18 = E1)
  0x14 v_mov_b32 v2, 0;           0x18 s_mov_b64 exec, s[18:19]           back to E1 (restore)
  0x1c v_cmp_lt_u32 vcc, 5, v5;   0x20 s_and_b64 exec, vcc, exec          E3 = lanes v5 > 5, a subset of E1
  0x24 v_cmp_lt_f32 vcc, 0, v1;   0x28 s_cbranch_vccz 0x30; 0x2c s_nop 0  vcc = 1 in every active lane
  0x30 s_mov_b64 exec, s[26:27]; then v1 = float(s0), exp pos0 / prim, s_endpgm

Checks:
  subset   the program translates: v1 is the same in every lane of E3, so the per-lane vccz branch is the wave's
  parent   0x18 restores s[26:27] (the entry exec, a superset of E1) instead: refused ("lane-dependent vcc"),
           since lanes outside E1 may be active again and hold another v1
  copy     0x14 overwrites s18 (s_mov_b32 s18, 0) before the restore: refused (the restore no longer names E1)
  half     0x14 overwrites only s19, the high half of the saved pair: refused as well
The rule is off by default; this check turns it on (X2W_MESH_EXEC_LINEAGE=1, set before the import).
Negative control: with X2W_MESH_EXEC_LINEAGE=0 (the epoch rule only) `subset` is refused (run in a child).
"""
import os
import struct
import subprocess
import sys

os.environ.setdefault('X2W_MESH_EXEC_LINEAGE', '1')   # the child of the negative control passes 0

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
import mesh2hlsl as M  # noqa: E402
from startup_image_3d_check import root_blob  # noqa: E402

TAIL = (0x7e020c00,                # v_cvt_f32_u32 v1, s0
        0x7e040280,                # v_mov_b32 v2, 0
        0xb07c1003,                # s_movk_i32 m0, 0x1003
        0xbf900009,                # s_sendmsg 0x9
        0xf80008cf, 0x02020201,    # exp pos0 v1, v2, v2, v2 done
        0xf8000941, 0x00000002,    # exp prim v2 done
        0xbf810000)                # s_endpgm
BODY = [0x7d820a83,                # 0x00 v_cmp_lt_u32 vcc, 3, v5
        0xbe9a246a,                # 0x04 s_and_saveexec_b64 s[26:27], vcc
        0x7e0202f2,                # 0x08 v_mov_b32 v1, 1.0
        0x7d820a87,                # 0x0c v_cmp_lt_u32 vcc, 7, v5
        0xbe92246a,                # 0x10 s_and_saveexec_b64 s[18:19], vcc
        0x7e040280,                # 0x14 v_mov_b32 v2, 0
        0xbefe0412,                # 0x18 s_mov_b64 exec, s[18:19]
        0x7d820a85,                # 0x1c v_cmp_lt_u32 vcc, 5, v5
        0x87fe7e6a,                # 0x20 s_and_b64 exec, vcc, exec
        0x7c020280,                # 0x24 v_cmp_lt_f32 vcc, 0, v1
        0xbf860001,                # 0x28 s_cbranch_vccz 0x30
        0xbf800000,                # 0x2c s_nop 0
        0xbefe041a]                # 0x30 s_mov_b64 exec, s[26:27]
TEXT = ['v_cmp_lt_u32_e32 vcc_lo, 3, v5', 's_and_saveexec_b64 s[26:27], vcc', 'v_mov_b32_e32 v1, 1.0',
        'v_cmp_lt_u32_e32 vcc_lo, 7, v5', 's_and_saveexec_b64 s[18:19], vcc', 'v_mov_b32_e32 v2, 0',
        's_mov_b64 exec, s[18:19]', 'v_cmp_lt_u32_e32 vcc_lo, 5, v5', 's_and_b64 exec, vcc, exec',
        'v_cmp_lt_f32_e32 vcc_lo, 0, v1', 's_cbranch_vccz label_0x30', 's_nop 0', 's_mov_b64 exec, s[26:27]']


def program(**at):
    words = list(BODY)
    for off, word in at.items():
        words[int(off[1:], 16) // 4] = word
    return struct.pack('<%dI' % (len(words) + len(TAIL)), *(words + list(TAIL)))


SUBSET = program()
PARENT = program(x18=0xbefe041a)   # 0x18 s_mov_b64 exec, s[26:27]
COPY = program(x14=0xbe920380)     # 0x14 s_mov_b32 s18, 0
HALF = program(x14=0xbe930380)     # 0x14 s_mov_b32 s19, 0


def translate(code):
    try:
        M.MeshPipeline(code, None, root_blob()).hlsl_ms()
        return 'ok'
    except G.Unsupported as e:
        return str(e)


def check():
    text = [x[3] for x in G.decode(SUBSET)][:len(TEXT)]
    assert text == TEXT, text
    assert [x[3] for x in G.decode(PARENT)][6] == 's_mov_b64 exec, s[26:27]'
    assert [x[3] for x in G.decode(COPY)][5] == 's_mov_b32 s18, 0'
    assert [x[3] for x in G.decode(HALF)][5] == 's_mov_b32 s19, 0'
    got = translate(SUBSET)
    assert got == 'ok', ('subset', got)
    for name, code in (('parent', PARENT), ('copy', COPY), ('half', HALF)):
        got = translate(code)
        assert 'lane-dependent vcc' in got, (name, got)
    return 'subset translates; parent restore, overwritten copy and overwritten high half refused'


def main():
    if os.environ.get('X2W_MESH_EXEC_LINEAGE') == '0':      # child of the negative control
        check()
        return
    summary = check()
    env = dict(os.environ, X2W_MESH_EXEC_LINEAGE='0')
    r = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'AssertionError' in r.stderr, ('negative control passed', r.stdout, r.stderr)
    print('mesh exec lineage: %s; X2W_MESH_EXEC_LINEAGE=0 fails (%s)' % (summary, r.stderr.strip().splitlines()[-1][:90]))


if __name__ == '__main__':
    main()
