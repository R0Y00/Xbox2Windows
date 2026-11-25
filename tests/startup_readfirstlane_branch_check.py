"""v_readfirstlane_b32 in a program with branches: the entry block is proven in program order (#2183); a later block is
proven only inside itself, with no exec write anywhere and no loop back to the entry."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/startup"))
import gfx2hlsl as gfx


class TestPipeline:
    root = {"params": [{"type": 2}], "flags": 0}

    def resource(self, key, kind):
        return "r0"


S_BRANCH_NEXT = 0xbf820000     # s_branch +0: the next instruction becomes a block start
S_BRANCH_ENTRY = 0xbf820000 | 0xfffb   # s_branch back to offset 0 when placed at word 4 (after S_LOAD, CVT, READ)
S_EXEC_LO_ZERO = 0xbe800300 | 126 << 16 | 128   # s_mov_b32 exec_lo, 0
S_LOAD = [0xf4000000 | 8 << 18 | 7 << 6 | 4, 125 << 25 | 12]   # s_buffer_load_dword s7, s[8:11], 12
CVT_V4 = 0x7e000000 | 4 << 17 | 8 << 9 | 7                     # v_cvt_i32_f32 v4, s7
READ_V4 = 0x7e000000 | 6 << 17 | 2 << 9 | 256 + 4              # v_readfirstlane_b32 s6, v4
END = 0xbf810000


def translate(*words):
    words = [*S_LOAD, *words]
    return gfx.Stage(TestPipeline(), "ps", struct.pack(f"<{len(words)}I", *words))


def body(*words):
    return "\n".join(line for block in translate(*words).blocks for line in block[1])


# Accepted, one per rule: the entry block, and a later block whose proof is made inside that block.
assert "s6 = v4;" in body(CVT_V4, READ_V4, S_BRANCH_NEXT, END), "entry block read refused"
assert "s6 = v4;" in body(S_BRANCH_NEXT, CVT_V4, READ_V4, END), "later block read refused"

# Refused: a join between proof and read, an exec write before the read (even in a later block), an exec write
# in a later block when the read sits in a later block, and a loop back to the entry block.
for words in ([CVT_V4, S_BRANCH_NEXT, READ_V4, END],
              [S_EXEC_LO_ZERO, CVT_V4, READ_V4, S_BRANCH_NEXT, END],
              [S_BRANCH_NEXT, CVT_V4, READ_V4, S_EXEC_LO_ZERO, END],
              [CVT_V4, READ_V4, S_BRANCH_ENTRY, END]):
    try:
        translate(*words)
    except gfx.Unsupported as error:
        assert "requires a proven uniform source" in str(error), str(error)
    else:
        raise AssertionError("accepted an unproven readfirstlane: %r" % (words,))
print("readfirstlane with branches: entry and later-block proofs accepted; four unproven reads rejected")
