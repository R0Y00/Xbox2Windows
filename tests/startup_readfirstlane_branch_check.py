"""v_readfirstlane_b32 in a program with branches: proven inside one basic block, refused across a join or exec write."""
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
S_EXEC_LO_ZERO = 0xbe800300 | 126 << 16 | 128   # s_mov_b32 exec_lo, 0
S_LOAD = [0xf4000000 | 8 << 18 | 7 << 6 | 4, 125 << 25 | 12]   # s_buffer_load_dword s7, s[8:11], 12
CVT_V4 = 0x7e000000 | 4 << 17 | 8 << 9 | 7                     # v_cvt_i32_f32 v4, s7
READ_V4 = 0x7e000000 | 6 << 17 | 2 << 9 | 256 + 4              # v_readfirstlane_b32 s6, v4
END = 0xbf810000


def translate(*words):
    words = [*S_LOAD, *words]
    return gfx.Stage(TestPipeline(), "ps", struct.pack(f"<{len(words)}I", *words))


# A branch after the read does not stop the read from being proven in its own block.
body = "\n".join(line for block in translate(CVT_V4, READ_V4, S_BRANCH_NEXT, END).blocks for line in block[1])
assert "s6 = v4;" in body, body

# A join point between the proof and the read, or an exec write anywhere, leaves the source unproven.
for words in ([CVT_V4, S_BRANCH_NEXT, READ_V4, END],
              [CVT_V4, READ_V4, S_BRANCH_NEXT, S_EXEC_LO_ZERO, END],
              [S_EXEC_LO_ZERO, CVT_V4, READ_V4, S_BRANCH_NEXT, END]):
    try:
        translate(*words)
    except gfx.Unsupported as error:
        assert "requires a proven uniform source" in str(error), str(error)
    else:
        raise AssertionError("accepted an unproven readfirstlane: %r" % (words,))
print("readfirstlane with branches: block-local proof accepted; three unproven reads rejected")


# The whole-program scan: implicit exec writes, a per-lane SCC and m0 must reach an earlier block through a loop.
def scan(*asm):
    stage = gfx.Stage.__new__(gfx.Stage)
    stage.ins = [(4 * i, 1, line.split()[0], line) for i, line in enumerate(asm)]
    return stage.scan_branch_safety()


assert not scan("s_andn2_wrexec_b64 s[2:3], s[4:5]")[0]
assert not scan("s_andn2_wrexec_b32 s2, s4")[0]
tainted = scan("v_cmp_lt_f32 vcc_lo, v1, v2", "s_cmp_eq_u32 vcc_lo, 0", "s_cselect_b32 s2, 1, 0")[1]
assert 2 in tainted, tainted
assert 2 in scan("v_cmp_lt_f32 vcc_lo, v1, v2", "s_mov_b32 m0, vcc_lo", "s_mov_b32 s2, m0")[1]
assert 2 not in scan("s_mov_b32 s3, 5", "s_cmp_eq_u32 s3, 0", "s_cselect_b32 s2, 1, 0")[1]
print("readfirstlane branch scan: wrexec, per-lane SCC and m0 are tracked")
