"""Direct DXIL VOP3 v_min3_f16, v_max3_f16, and v_med3_f16 lowering in Compute and Pixel shaders."""
from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_direct_container_check import patched_pixel
from native_dpp_mirror_check import vop2
from native_dpp_quad_check import request
import native_f16_alu_check as F
from native_packed_f16_inline_check import PROLOGUE, inline_word
from native_scalar_lds_check import sop1
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

OPS = {'min3': 0x351, 'max3': 0x354, 'med3': 0x357}
ALL_EXEC = 0xFFFFFFFF
DST_INIT = 0xDEADCAFE


def vop3_f16(op_name, dst, srcs, *, abs_=0, neg=0, op_sel=0, clamp=0, omod=0):
    s0, s1, s2 = srcs
    w0 = (0xD4000000 | (OPS[op_name] << 16) | ((clamp & 1) << 15) | ((op_sel & 15) << 11) |
          ((abs_ & 7) << 8) | (dst & 0xFF))
    w1 = ((neg & 7) << 29) | ((omod & 3) << 27) | ((s2 & 0x1FF) << 18) | ((s1 & 0x1FF) << 9) | (s0 & 0x1FF)
    return [w0, w1]


def f16_key(bits):
    bits &= 0xFFFF
    return (bits ^ 0xFFFF) if (bits & 0x8000) else (bits ^ 0x8000)


def is_nan_f16(bits):
    return (bits & 0x7FFF) > 0x7C00


def minmax2_f16(x, y, is_max):
    x &= 0xFFFF
    y &= 0xFFFF
    xn, yn = is_nan_f16(x), is_nan_f16(y)
    if xn and yn:
        return 0x7E00
    if xn:
        return y
    if yn:
        return x
    return (x if f16_key(x) > f16_key(y) else y) if is_max else (x if f16_key(x) < f16_key(y) else y)


def eval_3op(op_name, raw_words, *, abs_=0, neg=0, op_sel=0, clamp=0, dst_init=DST_INIT):
    vals = []
    for j in range(3):
        bits = (raw_words[j] >> (16 if ((op_sel >> j) & 1) else 0)) & 0xFFFF
        if (abs_ >> j) & 1:
            bits &= 0x7FFF
        if (neg >> j) & 1:
            bits ^= 0x8000
        vals.append(bits)
    a, b, c = vals
    if op_name == 'min3':
        res = minmax2_f16(minmax2_f16(a, b, False), c, False)
    elif op_name == 'max3':
        res = minmax2_f16(minmax2_f16(a, b, True), c, True)
    else:
        lo = minmax2_f16(a, b, False)
        hi = minmax2_f16(a, b, True)
        if is_nan_f16(a) or is_nan_f16(b) or is_nan_f16(c):
            res = minmax2_f16(lo, c, False)
        else:
            res = minmax2_f16(lo, minmax2_f16(hi, c, False), True)
    if clamp:
        if is_nan_f16(res) or (res & 0x8000):
            res = 0x0000
        elif res > 0x3BFF:
            res = 0x3C00
    if op_sel & 8:
        return ((res & 0xFFFF) << 16) | (dst_init & 0xFFFF)
    return (dst_init & 0xFFFF0000) | (res & 0xFFFF)


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    srcs: tuple
    vals: tuple
    abs_: int = 0
    neg: int = 0
    op_sel: int = 0
    clamp: int = 0
    exec_mask: int = ALL_EXEC

    def expected(self, lane=0):
        if ((self.exec_mask >> lane) & 1) == 0:
            return DST_INIT
        raw = [v & 0xFFFFFFFF if (s < 128 or s >= 255) else inline_word(s) for s, v in zip(self.srcs, self.vals)]
        return eval_3op(self.op, raw, abs_=self.abs_, neg=self.neg, op_sel=self.op_sel, clamp=self.clamp)


def all_cases():
    return [
        # 1. Basic ordering & all 6 distinct permutations of med3_f16.
        Case('min3_basic', 'min3', (257, 258, 259), (0x4200, 0xC000, 0x3C00)),
        Case('max3_basic', 'max3', (257, 258, 259), (0x3800, 0xC400, 0x4400)),
        Case('med3_abc', 'med3', (257, 258, 259), (0x3800, 0x3C00, 0x4000)),
        Case('med3_acb', 'med3', (257, 258, 259), (0x3800, 0x4000, 0x3C00)),
        Case('med3_bac', 'med3', (257, 258, 259), (0x3C00, 0x3800, 0x4000)),
        Case('med3_bca', 'med3', (257, 258, 259), (0x3C00, 0x4000, 0x3800)),
        Case('med3_cab', 'med3', (257, 258, 259), (0x4000, 0x3800, 0x3C00)),
        Case('med3_cba', 'med3', (257, 258, 259), (0x4000, 0x3C00, 0x3800)),
        # 2. Signed zeros, binary16 subnormals, and +/-Inf.
        Case('min3_signed_zero', 'min3', (257, 258, 259), (0x0000, 0x8000, 0x0000)),
        Case('max3_signed_zero', 'max3', (257, 258, 259), (0x8000, 0x0000, 0x8000)),
        Case('med3_subnormals_inf', 'med3', (257, 258, 259), (0x0001, 0xFC00, 0x7C00)),
        # 3. NaN rules: 1 NaN, 2 NaNs, 3 NaNs (med3 returns min of numeric inputs when any NaN is present).
        Case('min3_one_nan', 'min3', (257, 258, 259), (0x7E01, 0x4000, 0x3C00)),
        Case('max3_two_nans', 'max3', (257, 258, 259), (0x7E01, 0xC000, 0x7D00)),
        Case('med3_one_nan_uses_min', 'med3', (257, 258, 259), (0x4200, 0x7E01, 0x3800)),
        Case('med3_nan_src2', 'med3', (257, 258, 259), (0x4200, 0x3800, 0x7E01)),
        Case('med3_two_nans', 'med3', (257, 258, 259), (0x7E01, 0x3800, 0x7D02)),
        Case('med3_three_nans', 'med3', (257, 258, 259), (0x7E01, 0x7D02, 0x7F03)),
        # 4. abs and neg modifiers across all three operands.
        Case('min3_abs_neg', 'min3', (257, 258, 259), (0xC000, 0x4200, 0xC400), abs_=0b101, neg=0b110),
        Case('med3_abs_neg', 'med3', (257, 258, 259), (0xC000, 0xBC00, 0x3800), abs_=0b010, neg=0b001),
        # 5. clamp: negative/-0/NaN -> +0.0, subnormal preserved, > 1.0/+Inf -> 1.0.
        Case('min3_clamp_neg_subnormal', 'min3', (257, 258, 259), (0x0001, 0x3800, 0x3C00), clamp=1),
        Case('min3_clamp_negative', 'min3', (257, 258, 259), (0xBC00, 0x3800, 0x3C00), clamp=1),
        Case('max3_clamp_inf_to_one', 'max3', (257, 258, 259), (0x3800, 0x7C00, 0xC000), clamp=1),
        Case('med3_clamp_nan_to_zero', 'med3', (257, 258, 259), (0x7E01, 0x8000, 0x0000), clamp=1),
        Case('med3_three_nans_clamp', 'med3', (257, 258, 259), (0x7E01, 0x7D02, 0x7F03), clamp=1),
        # 6. op_sel[2:0] high-half VGPR reads and op_sel[3]=1 destination high half preserving D[15:0].
        Case('max3_opsel_dst_hi', 'max3', (257, 258, 259), (0x4000DEAD, 0x3800CAFE, 0x44001111), op_sel=0b1101),
        # 7. SGPR, literal (255), inline binary16 float (240/242/243) and inline integer (128, 193).
        Case('min3_sgpr_literal_inline', 'min3', (10, 255, 240), (0x3C00, 0x3400, 0)),
        Case('min3_inline_neg_int', 'min3', (193, 258, 259), (0, 0x3C00, 0x4000)),
        Case('med3_inline_int_float', 'med3', (128, 242, 243), (0, 0, 0)),
        # 8. Partial EXEC_LO mask: inactive lanes keep DST_INIT.
        Case('max3_partial_exec', 'max3', (257, 258, 259), (0x3C00, 0x4000, 0x4200), exec_mask=0xAAAAAAAA),
    ]


def emit_case(c: Case):
    code = F.mov(4, DST_INIT)
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 255), c.exec_mask]
    lit_tail = []
    for s, v in zip(c.srcs, c.vals):
        if s >= 256:
            code += F.mov(s - 256, v)
        elif s <= 105:
            code += F.smov(s, v)
        elif s == 255:
            lit_tail = [v]
    code += vop3_f16(c.op, 4, c.srcs, abs_=c.abs_, neg=c.neg, op_sel=c.op_sel, clamp=c.clamp)
    code += lit_tail
    if c.exec_mask != ALL_EXEC:
        code += [sop1(3, 126, 193)]
    return code


def compute_program(cases):
    code = list(PROLOGUE)
    for slot, c in enumerate(cases):
        code += emit_case(c) + F.store(slot * 4, 4)
    return code + [0xBF810000]


def refusal_programs():
    cond_mode = [0xBF840001, 0xBFA5000F]  # s_cbranch_scc0 +1; s_denorm_mode 15
    plain = vop3_f16('med3', 4, (257, 258, 259)) + [0xBF810000]
    return [
        # Output modifier (omod != 0).
        (vop3_f16('min3', 4, (257, 258, 259), omod=1) + [0xBF810000], 'unsupported v_min3_f16 omod'),
        (vop3_f16('med3', 4, (257, 258, 259), omod=2) + [0xBF810000], 'unsupported v_med3_f16 omod'),
        # High-half op_sel on non-VGPR source.
        (vop3_f16('max3', 4, (10, 258, 259), op_sel=0b0001) + [0xBF810000],
         'unsupported v_max3_f16 high-half op_sel on non-VGPR source'),
        (vop3_f16('min3', 4, (257, 242, 259), op_sel=0b0010) + [0xBF810000],
         'unsupported v_min3_f16 high-half op_sel on non-VGPR source'),
        # Special scalar source (vcc_lo).
        (vop3_f16('med3', 4, (106, 258, 259)) + [0xBF810000], 'unsupported v_med3_f16 source'),
        # Binary16 denormal flush (s_denorm_mode 3) and dynamic conditional MODE write.
        ([0xBFA50003] + plain, 'unsupported binary16 rounding or denormal mode'),
        (cond_mode + plain, 'unsupported binary16 rounding or denormal mode'),
    ]


def ir(dumper):
    items = all_cases()
    code = compute_program(items)
    want = [c.expected(lane=0) for c in items]
    refusals = refusal_programs()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run = F.dump(dumper, work, code, 'minmax-med3-f16')
        assert run.returncode == 0, run.stderr
        stores = sorted(F.interpret(run.stdout))
        assert [addr for addr, _ in stores] == [4 * i for i in range(len(want))], stores[:4]
        bad = [(i, items[i].name, hex(got), hex(exp))
               for i, ((_, got), exp) in enumerate(zip(stores, want)) if got != exp]
        assert not bad, bad[:8]
        for idx, (refused, why) in enumerate(refusals):
            r = F.dump(dumper, work, refused, f'refused-{idx}')
            assert r.returncode != 0 and why in r.stderr, (idx, why, r.returncode, r.stderr)
    print(f'native_minmax_med3_f16 IR: {len(want)} cases matched exact model; {len(refusals)} refusals matched')
    return 0


def controls(cli, sdk, work, root):
    items = all_cases()
    code = compute_program(items)
    marker = request(work, root, code)
    output = marker.with_suffix('.cso')
    result = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', marker, output],
                            capture_output=True, text=True)
    assert result.returncode == 0 and output.is_file(), (result.stdout, result.stderr)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', output], capture_output=True, text=True, check=True).stdout
    assert 'icmp ugt i32' in dump and 'icmp ult i32' in dump, dump

    refusals = refusal_programs()
    for idx, (prog, why) in enumerate(refusals):
        m = request(work, root, prog)
        o = m.with_suffix(f'.refused{idx}.cso')
        r = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-direct-cs@0', m, o], capture_output=True, text=True)
        assert r.returncode != 0 and why in r.stderr, (idx, why, r.stdout, r.stderr)
        assert not o.exists(), o

    print(f'native_minmax_med3_f16 controls: compiled {len(items)} cases to DXIL; verified {len(refusals)} refusals')
    return 0


def hardware(replay, sdk, work, root):
    items = all_cases()
    code = compute_program(items)
    marker = request(work, root, code)
    output = work / 'out.bin'
    run = subprocess.run([replay, sdk, marker, work / 'cache', '0', output, '--direct'])
    if run.returncode == 77:
        return 77
    run.check_returncode()
    words_out = list(struct.unpack(f'<{F.WORDS}I', output.read_bytes()))
    expected = {}
    for group in range(F.GROUPS):
        for lane in range(32):
            head = F.BASE + group * 1024 + lane * 32
            for idx, c in enumerate(items):
                expected[head + idx] = c.expected(lane=lane)
    bad = []
    for i, got in enumerate(words_out):
        if i in expected:
            if got != expected[i]:
                bad.append((i, hex(got), hex(expected[i])))
        elif got != F.SENTINEL:
            bad.append((i, hex(got), hex(F.SENTINEL)))
    assert not bad, bad[:10]
    print(f'native_minmax_med3_f16 compute hardware: {len(items)} cases matched across {F.GROUPS}x32 lanes and guards')
    return 0


def pixel_program(pair):
    code = []
    for idx, c in enumerate(pair):
        code += emit_case(c)
        lo_reg, hi_reg = 8 + 2 * idx, 9 + 2 * idx
        code += [vop2(0x1B, lo_reg, 255, 4), 65535, vop2(0x16, hi_reg, 144, 4)]
        # Each 16-bit half is exactly representable in binary32 as an integer in 0..65535.
        code += [0x7E000C00 | reg << 17 | (256 + reg) for reg in (lo_reg, hi_reg)]
    return patched_pixel(len(code), code, export=(0xF800180F, 0x0B0A0908))


def pixel(cli, sdk, renderer, adapter, work):
    items = [c for c in all_cases() if c.exec_mask == ALL_EXEC]
    if len(items) % 2:
        items.append(items[-1])
    vs = compile_hlsl(str(Path(sdk) / 'dxc.exe'), 'dxc', FULLSCREEN_VS, 'fullscreen', 'vs_6_0', work)
    pairs = [items[i:i + 2] for i in range(0, len(items), 2)]
    for idx, pair in enumerate(pairs):
        shader = work / f'minmax3-{idx}.ps'
        shader.write_bytes(pixel_program(pair))
        cso = shader.with_suffix('.cso')
        build = subprocess.run([cli, sdk, work / 'cache', 'gfx1013-container-ps', shader, cso],
                               capture_output=True, text=True, timeout=30)
        assert build.returncode == 0, ([c.name for c in pair], build.stdout, build.stderr)
        w0, w1 = pair[0].expected(), pair[1].expected()
        ref = work / 'expected.bin'
        ref.write_bytes(struct.pack('<4f', w0 & 0xFFFF, w0 >> 16, w1 & 0xFFFF, w1 >> 16))
        run = subprocess.run([renderer, adapter, '--startup-pixel-words', vs, cso, '1', '1', ref],
                             capture_output=True, text=True, timeout=30)
        if run.returncode == 77:
            return 77
        assert run.returncode == 0, (idx, [c.name for c in pair], run.stdout, run.stderr)
        if 'debug_layer=enabled' in run.stdout:
            assert 'debug_errors=0 debug_warnings=0' in run.stdout, run.stdout
    print(f'native_minmax_med3_f16 pixel ({adapter}): {len(pairs)} pixel draws ({len(items)} cases) matched')
    return 0


def main(mode, tool, sdk=None, *args):
    if mode == 'ir':
        return ir(tool)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if mode == 'pixel':
            return pixel(tool, sdk, args[0], args[1], work)
        shader = work / 'root.hlsl'
        shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
        root = work / 'root.bin'
        subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, shader], check=True)
        return {'controls': controls, 'compute': hardware}[mode](tool, sdk, work, root.read_bytes())


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
