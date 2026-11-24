"""v_readlane_b32 and v_permlanex16_b32 translate from a proven uniform VGPR and are refused otherwise."""
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/startup"))
import gfx2hlsl as gfx


class TestPipeline:
    root = {"params": [{"type": 2}], "flags": 0}

    def resource(self, key, kind):
        return "r0"


def vop1(opcode, dst, src):
    return 0x7e000000 | dst << 17 | opcode << 9 | src


def readlane(sdst, vsrc, select):
    return [0xd7600000 | sdst, 256 + vsrc | select << 9]


def permlanex16(vdst, vsrc, sel0, sel1):
    return [0xd7780000 | vdst, 256 + vsrc | sel0 << 9 | sel1 << 18]


def translate(*extra, src=7):
    # s_buffer_load_dword s7, s[8:11], 12; v_cvt_i32_f32 v4, src (uniform when src is s7); extra; s_endpgm
    words = [0xf4000000 | 8 << 18 | 7 << 6 | 4, 125 << 25 | 12, vop1(8, 4, src), *extra, 0xbf810000]
    return "\n".join(gfx.Stage(TestPipeline(), "ps", struct.pack(f"<{len(words)}I", *words)).blocks[0][1])


body = translate(*readlane(6, 4, 128 + 5), *permlanex16(5, 4, 1, 2))
assert "s6 = v4;" in body
assert "v5 = v4;" in body
# The permuted value is still uniform, so a read of it is accepted too.
assert "s9 = v5;" in translate(*permlanex16(5, 4, 1, 2), *readlane(9, 5, 3))
for words, args, text in [
    (readlane(6, 4, 128), {"src": 256}, "v_readlane_b32 requires a proven uniform source"),
    (permlanex16(5, 4, 1, 2), {"src": 256}, "v_permlanex16_b32 requires a proven uniform source"),
    (readlane(6, 3, 128), {}, "v_readlane_b32 requires a proven uniform source"),
    (readlane(6, 4, 256), {}, "v_readlane_b32 operands"),
    ([0xd7780000 | 5 | 1 << 11, 256 + 4 | 1 << 9 | 2 << 18], {}, "v_permlanex16_b32 with"),
]:
    try:
        translate(*words, **args)
    except gfx.Unsupported as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("accepted " + text)
print("readlane and permlanex16 translated from a uniform VGPR; five other forms rejected")
