"""Packed slot-67 control for implicit derivatives in a virtual wave64 group.

The virtual-wave scheduler needs the packed state, more than one guest wave64,
and a conditional branch plus a synchronizing instruction.  The container
fixture does not enter that path.  This source-built request keeps the explicit
LOD and ordinary implicit cases compiling, and requires the combination to be
refused at the image opcode instead of producing a shader with no scheduler
readback.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_compute_clear_check import write_request


WIDTH = 8
SAMPLER = (0x12, 0xFFF000, 0x04500000, 0)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def vop2(op, dst, src0, src1):
    return (op << 25) | (dst << 17) | (src1 << 9) | src0


def setup():
    # One guest table at s4: the texture descriptor is loaded from it before
    # s4:7 is reused for the sampler constants the image op consumes.
    code = [
        0xBEFE04C1,              # s_mov_b64 exec, -1 (wave64)
        0xBE850380,              # s5 = 0, the high half of the table address
        0xF40C0202, 0xFA000000,  # s_load_dwordx8 s[8:15], s[4:5], 0
    ]
    for reg, value in enumerate(SAMPLER, start=4):
        code += [0xBE8003FF | (reg << 16), value]
    return code


def program(implicit, control):
    code = setup()
    code += [
        vop2(22, 3, 130, 0), vop2(27, 3, 131, 3), vop2(19, 3, 130, 3),
        vop2(26, 4, 151, 3), vop2(37, 4, 255, 4), bits(1 / WIDTH),
        vop2(27, 1, 129, 0), 0x7E020D01, vop2(8, 1, 260, 1),
        vop2(27, 4, 136, 0), 0x7E080D04, vop2(8, 4, 255, 4), bits(0.0625),
        vop2(3, 1, 260, 1), 0x7E0402F0, 0x7E060D03,
    ]
    if control:
        code += [0xBF880004, 0xBF8A0000]  # s_cbranch_execz +4; s_barrier
    code += [0xBF8CC07F, 0xF0000108 | ((32 if implicit else 36) << 18), 0x00220401]
    code.append(0xBF810000)
    return code


def main(cli, sdk):
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        source, root = work / "root.hlsl", work / "root.bin"
        source.write_text('#define SIG "DescriptorTable(SRV(t0,numDescriptors=8)),'
                          'DescriptorTable(UAV(u0,numDescriptors=8))"', encoding="ascii")
        subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG",
                        "-Fo", root, source], check=True)
        # Tagged slot-67 state: dword 0 and dword 1 select the packed ABI, dword 3
        # is X|Y<<16, dword 4 holds Z and the wave size, and dword 6 is RSRC2.
        sa = struct.pack("<7I", 0x4005cf00, 0x6c090102, 0, 128 | (1 << 16), 1, 0, 0x8d8c)
        cases = (
            ("implicit", True, False, 0, None),
            ("explicit", False, True, 0, None),
            ("implicit-virtual", True, True, 1, "implicit derivatives with virtual wave64 groups"),
        )
        for name, implicit, control, expected, diagnostic in cases:
            marker = write_request(work, program(implicit, control), root.read_bytes(), sa)
            output = work / f"{name}.cso"
            run = subprocess.run([cli, sdk, work / "cache", "gfx1013-direct-cs@0", marker, output],
                                 capture_output=True, text=True, timeout=60)
            assert run.returncode == expected, (name, run.stdout, run.stderr)
            if diagnostic:
                assert diagnostic in run.stderr and not output.exists(), (name, run.stderr)
                print(f"{name}: refused at the image opcode: {diagnostic}")
            else:
                assert output.is_file(), (name, run.stdout, run.stderr)
                print(f"{name}: packed slot-67 request compiled")
    print("virtual-wave derivative control passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
