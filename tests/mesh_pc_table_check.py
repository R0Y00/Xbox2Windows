"""A source-built mesh reads an embedded four-word position table through GETPC."""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import mesh2hlsl as M
from mesh_loop_state_check import TAIL, root_blob


def stage(declared_bytes, contents):
    # Assemble the low/high address in s[8:9], the byte size in s10, and the
    # raw byte-addressed buffer control word in s11. v5 selects the byte offset.
    words = [0xbe881f00,                       # s_getpc_b64 s[8:9]
             0x8008ff08, 0,                    # s_add_u32 s8, s8, pool - 4
             0xb00a0000 | declared_bytes,       # s_movk_i32 s10, declared_bytes
             0xbe8b03ff, 0x31016fac,           # s_mov_b32 s11, raw buffer control
             0xe0301000, 0x80020105,           # buffer_load_dword v1, v5, s[8:11], 0 offen
             *TAIL[1:]]
    words[2] = 4 * len(words) - 4
    words.extend(contents)
    return M.MeshPipeline(struct.pack('<%dI' % len(words), *words), None, root_blob())


if __name__ == '__main__':
    contents = (0x3f800000, 0x40000000, 0x40400000, 0x40800000)
    pipe = stage(16, contents)
    hlsl = pipe.hlsl_ms()
    (offset, size), = pipe.vs.literal_tables
    name = pipe.vs.literal_tables[(offset, size)]
    assert size == 16
    assert '%s[4] = {0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u}' % name in hlsl
    assert 'a = v5 >> 2u;' in hlsl
    assert '(a < 4u ? %s[min(a, 3u) + 0u] : 0u)' % name in hlsl
    try:
        stage(20, contents)
    except M.Unsupported as e:
        assert 'incomplete PC-relative raw buffer' in str(e), e
    else:
        raise AssertionError('accepted a literal table beyond the supplied program')
    print('Embedded four-word position table translated; truncated table refused')
