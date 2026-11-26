"""Table descriptor words read as numbers in the mesh-stage translator (mesh2hlsl MeshStage.dw_*,
X2W_MESH_FEATURES=descword; runtime: d3d12_draw.inl `udsrc <n> descword <param> <byte>`, X2W_MESH_DESCWORD=1).

Hand-assembled NGG mesh programs (no cache bytes), RDNA 2 ISA encodings; root parameter 0 is a descriptor table
(s16 = its pointer: the mesh stage's user SGPRs start at s16), parameter 1 a root constant:

  0x00 s_load_dwordx4 s[20:23], s[16:17], 0x20   a V# from table 0 entry 1 (byte 0x20): s22 = word 2 (NUM_RECORDS)
  0x08 s_waitcnt lgkmcnt(0)
  0x0c s_mul_i32 s24, s22, 3                       a number computed from word 2 (byte 0x28)
  0x10 s_lshr_b32 s25, s21, 16                     from word 1 (byte 0x24), never read: dead
  0x14 v_cvt_f32_u32 v1, s24                       the numeric use; then exp pos0 v1 / prim, s_endpgm

Checks:
  lazy     translates; the manifest has `udsrc <n> descword 0 40` and no entry value for byte 36; the HLSL computes
           s24 from that entry value (ud<n> * 3) and s25 from 0u
  edit     0x0c-0x10 become s_or_b32 s22, s22, 1 (an in-place edit) then s_mul_i32 s24, s22, 3: refused ("edited
           descriptor word": the table's word is no longer the register's value); the same with s_or_b32 s22, s22, s17
           (the other operand a root constant)
  reload   as edit, then the same s_load again before the multiply: translates (a fresh load is the table's word)
Negative control: without X2W_MESH_FEATURES=descword `lazy` is refused ("numeric use of s24 holding descnum"; run in
a child).
"""
import os
import re
import struct
import subprocess
import sys

if os.environ.get('X2W_DESCWORD_CHILD') != '1':
    os.environ['X2W_MESH_FEATURES'] = 'descword'

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'startup'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
import mesh2hlsl as M  # noqa: E402
from startup_image_3d_check import root_blob  # noqa: E402

LOAD = (0xf4080508, 0xfa000020)    # s_load_dwordx4 s[20:23], s[16:17], 0x20
WAIT = (0xbf8cc07f,)               # s_waitcnt lgkmcnt(0)
MUL = (0x93188316,)                # s_mul_i32 s24, s22, 3
DEAD = (0x90199015,)               # s_lshr_b32 s25, s21, 16
EDIT = (0x88168116,)               # s_or_b32 s22, s22, 1
EDIT_UD = (0x88161116,)            # s_or_b32 s22, s22, s17 (s17 = the root constant)
USE = (0x7e020c18,)                # v_cvt_f32_u32 v1, s24
TAIL = (0x7e040280,                # v_mov_b32 v2, 0
        0xb07c1003,                # s_movk_i32 m0, 0x1003
        0xbf900009,                # s_sendmsg 0x9
        0xf80008cf, 0x02020201,    # exp pos0 v1, v2, v2, v2 done
        0xf8000941, 0x00000002,    # exp prim v2 done
        0xbf810000)                # s_endpgm


def program(*parts):
    words = [w for p in parts for w in p] + list(TAIL)
    return struct.pack('<%dI' % len(words), *words)


LAZY = program(LOAD, WAIT, MUL, DEAD, USE)
EDITED = program(LOAD, WAIT, EDIT, MUL, USE)
EDITED_UD = program(LOAD, WAIT, EDIT_UD, MUL, USE)
RELOAD = program(LOAD, WAIT, EDIT, LOAD, WAIT, MUL, USE)
TEXT = ['s_load_dwordx4 s[20:23], s[16:17], 0x20', 's_waitcnt lgkmcnt(0)', 's_mul_i32 s24, s22, 3',
        's_lshr_b32 s25, s21, 16', 'v_cvt_f32_u32_e32 v1, s24']


def translate(code):
    try:
        pipe = M.MeshPipeline(code, None, root_blob())
        return pipe.hlsl_ms(), pipe.manifest()
    except G.Unsupported as e:
        return str(e), None


def check():
    text = [x[3] for x in G.decode(LAZY)][:len(TEXT)]
    assert text == TEXT, text
    assert [x[3] for x in G.decode(EDITED)][2] == 's_or_b32 s22, s22, 1'
    hlsl, manifest = translate(LAZY)
    assert manifest is not None, ('lazy', hlsl)
    used = re.search(r'^udsrc (\d+) descword 0 40$', manifest, re.M)
    assert used and not re.search(r'descword 0 36$', manifest, re.M), manifest
    assert re.search(r'\bs24 = ud%s \* 0x00000003u;' % used.group(1), hlsl), hlsl
    assert re.search(r'\bs25 = 0u >> ', hlsl) and 'X2W_DW' not in hlsl, hlsl
    assert [x[3] for x in G.decode(EDITED_UD)][2] == 's_or_b32 s22, s22, s17'
    for name, code in (('edit', EDITED), ('edit by a root constant', EDITED_UD)):
        got = translate(code)
        assert got[1] is None and 'edited descriptor word' in got[0], (name, got[0][:200])
    got = translate(RELOAD)
    assert got[1] is not None and re.search(r'^udsrc \d+ descword 0 40$', got[1], re.M), ('reload', got[0][:200])
    return 'word 2 an entry value, dead word 1 0u; in-place edit refused; reload after the edit translates'


def main():
    if os.environ.get('X2W_DESCWORD_CHILD') == '1':      # child of the negative control: the feature is off
        check()
        return
    summary = check()
    env = dict(os.environ, X2W_DESCWORD_CHILD='1', X2W_MESH_FEATURES='')
    r = subprocess.run([sys.executable, '-B', os.path.abspath(__file__)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and 'holding descnum' in r.stderr, ('negative control passed', r.stdout, r.stderr)
    print('mesh descword: %s; without X2W_MESH_FEATURES=descword refused (%s)'
          % (summary, r.stderr.strip().splitlines()[-1][:100]))


if __name__ == '__main__':
    main()
