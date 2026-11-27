"""Pixel-stage structured-to-raw buffer loads in tools/startup/gfx2hlsl.py.

  (no arguments)                         translation controls and the CPU model, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                         translate, compile and draw one pixel. The readback must match the
                                         CPU model of a raw byte-address load bounded by NUM_RECORDS * stride.
                                         Dropping that bound must read the word past the bound.

The fixture is source-built. A pixel stage loads one table descriptor, clears its stride, scales NUM_RECORDS
by the extracted stride and sets OOB_SELECT to raw, then reads four dwords with buffer_load_dword offen.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import gfx2hlsl as G  # noqa: E402

MODIFIED = 'buffer load through a modified descriptor'
INDEXED = 'indexed load through a modified descriptor'
STRIDE = 8
RECORDS = 4
# Nine words the resource holds, then padding so the raw view's element count is a multiple of four.
# Word 8 sits past the converted byte count (8 * 4 = 32) and must read back as zero.
SOURCE = (0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000, 0x40e00000, 0x41000000,
          0xdeadbeef, 0, 0, 0)
OFFSETS = (0, 12, 28, 32)
DESC = 16
TMP = 20

SOP2 = {'s_and_b32': 0x0e, 's_or_b32': 0x10, 's_max_u32': 0x09, 's_mul_i32': 0x26, 's_bfe_u32': 0x27}

# One descriptor table. Graphics user data starts at s8; the pixel stage loads s[16:19] from that pointer.
RTS0 = b''.join((struct.pack('<6I', 2, 1, 24, 0, 0, 0), struct.pack('<3I', 0, 0, 36),
                 struct.pack('<2I', 1, 44), struct.pack('<6I', 0, 1, 0, 0, 0, 0)))
PART = b'RTS0' + struct.pack('<I', len(RTS0)) + RTS0
ROOT = b'DXBC' + bytes(16) + struct.pack('<4I', 1, 36 + len(PART), 1, 36) + PART

# A covering triangle that exports a position and reads no buffer. The draw itself uses the desktop
# full-screen vertex shader below; this stage only has to satisfy the translator.
VS = bytes.fromhex('f202087e' '80020a7e' 'cf0800f805050504' '000081bf')

FULLSCREEN_VS = """\
float4 main(uint id : SV_VertexID) : SV_Position
{
    return float4(id == 1 ? 3.0 : -1.0, id == 2 ? -3.0 : 1.0, 0.0, 1.0);
}
"""


def sop2(op, sdst, ssrc0, ssrc1, literal=None):
    word = struct.pack('<I', (0b10 << 30) | (SOP2[op] << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0)
    return word if literal is None else word + struct.pack('<I', literal)


def encode_mubuf(op, *, imm=0, offen=0, idxen=0, vaddr=0, vdata=1, srsrc=DESC, soffset=128):
    w0 = (0b111000 << 26) | ((op & 0xff) << 18) | ((idxen & 1) << 13) | ((offen & 1) << 12) | (imm & 0xfff)
    w1 = ((soffset & 0xff) << 24) | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return struct.pack('<II', w0, w1)


def encode_mtbuf(op, fmt, *, imm=0, offen=1, vaddr=0, vdata=1, srsrc=DESC, soffset=128):
    # MTBUF: [31:26] is 0b111010, the format is [25:19], and the low three opcode bits are [18:16].
    # The high opcode bit is word 1 bit 21. BUF_FMT_32_FLOAT is format 22.
    w0 = 0xe8000000 | ((fmt & 0x7f) << 19) | ((op & 7) << 16) | ((offen & 1) << 12) | (imm & 0xfff)
    w1 = ((soffset & 0xff) << 24) | (((op >> 3) & 1) << 21) | (((srsrc // 4) & 0x1f) << 16)
    w1 |= ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return struct.pack('<II', w0, w1)


def smem_load_x4(sdst, sbase, offset=0):
    w0 = 0xf4000000 | (0x02 << 18) | (sdst << 6) | (sbase // 2)
    return struct.pack('<II', w0, (125 << 25) | (offset & 0x1fffff)) + struct.pack('<I', 0xbf8cc07f)


def v_mov(dst, inline):
    return struct.pack('<I', 0x7e000000 | (dst << 17) | (1 << 9) | (128 + inline))


def v_mov_lit(dst, word):
    return struct.pack('<II', 0x7e000000 | (dst << 17) | (1 << 9) | 255, word & 0xffffffff)


def s_mov(dst, src):
    return struct.pack('<I', 0xbe800000 | (dst << 16) | (3 << 8) | (src & 0xff))


def exp_mrt0(*regs):
    w1 = regs[0] | (regs[1] << 8) | (regs[2] << 16) | (regs[3] << 24)
    return struct.pack('<II', 0xf800180f, w1) + struct.pack('<I', 0xbf810000)


def conversion(clamp=False, base=DESC, tmp=TMP):
    words = [sop2('s_bfe_u32', tmp, base + 1, 255, 0xe0010)]
    if clamp:
        words.append(sop2('s_max_u32', tmp, tmp, 129))
    words += [sop2('s_and_b32', base + 1, base + 1, 255, 0xc000ffff),
              sop2('s_mul_i32', base + 2, base + 2, tmp),
              sop2('s_and_b32', base + 3, base + 3, 255, 0xcfffffff),
              sop2('s_or_b32', base + 3, base + 3, 255, 0x30000000)]
    return b''.join(words)


def loads(idxen=False, srsrc=DESC, offsets=OFFSETS, soffset=128):
    return b''.join(encode_mubuf(0x0c, imm=offset, offen=0 if idxen else 1, idxen=int(idxen),
                                 vdata=1 + i, srsrc=srsrc, soffset=soffset)
                    for i, offset in enumerate(offsets))


def finish(body, exports=(1, 2, 3, 4)):
    return body + struct.pack('<I', 0xbf8c3f70) + exp_mrt0(*exports)


def program(middle, idxen=False, v0=0, offsets=OFFSETS, srsrc=DESC, soffset=128):
    mov = v_mov(0, v0) if isinstance(v0, int) and 0 <= v0 <= 64 else v_mov_lit(0, v0)
    return finish(smem_load_x4(DESC, 8) + middle + mov + loads(idxen, srsrc, offsets, soffset))


def translate(ps):
    return G.Pipeline(VS, ps, ROOT)


def add_u32(parts):
    """Add byte-address terms the way the bounded load does: a wrapped term makes the address unusable."""
    base, ok = 0, True
    for part in parts:
        part &= 0xffffffff
        total = base + part
        ok = ok and total <= 0xffffffff and (total & 0xffffffff) >= base
        base = total & 0xffffffff
    return ok, base


def raw_dwords(parts, count, stride=STRIDE, records=RECORDS):
    ok, address = add_u32(parts)
    limit = (stride * records) & 0xffffffff
    words = []
    for i in range(count):
        end = (address + 4 * (i + 1)) & 0xffffffff
        if not ok or end < address or end > limit:
            words.append(0)
        else:
            words.append(SOURCE[address // 4 + i])
    return tuple(words)


def cpu_load(address, stride=STRIDE, records=RECORDS):
    """One raw dword: in range when the 32-bit end byte does not wrap and is at or before the product."""
    return raw_dwords((address,), 1, stride, records)[0]


def expected_words():
    return tuple(cpu_load(offset) for offset in OFFSETS)


def ud_index(manifest, kind):
    found = [int(line.split()[1]) for line in manifest.splitlines()
             if line.startswith('udsrc ') and line.split()[2] == kind]
    if len(found) != 1:
        raise AssertionError('udsrc %s: %s' % (kind, manifest))
    return found[0]


def must_refuse(label, ps, message):
    try:
        translate(ps)
    except G.Unsupported as error:
        if str(error) != message:
            raise AssertionError('%s: refused with %r, expected %r' % (label, str(error), message))
        print('%s: refused (%s)' % (label, message))
        return
    raise AssertionError('%s: translated' % label)


def bound_open(stride, records, clamped=False):
    factor = 'max(ud%d, 1u)' % stride if clamped else 'ud%d' % stride
    return '{ uint lim = %s * ud%d; uint base = 0; bool ok = true;' % (factor, records)


def assert_bounded(ps, stride, records, offsets, count=1, clamped=False, terms=None):
    assert bound_open(stride, records, clamped) in ps, ps
    assert 'a = v0 +' not in ps, ps
    for term in terms or ('v0',):
        assert 'ok = ok && base + (%s) >= base; base += %s;' % (term, term) in ps, ps
    for index, offset in enumerate(offsets):
        assert 'ok = ok && base + (%d) >= base; base += %d;' % (offset, offset) in ps, ps
        for lane in range(count):
            end = 4 * (lane + 1)
            line = 'v%d = (ok && a + %du >= a && a + %du <= lim) ? r0.Load(a + %d) : 0u;' % (
                1 + index * count + lane, end, end, 4 * lane)
            assert line in ps, ps


def controls():
    pipe = translate(program(conversion()))
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    stride, records = ud_index(manifest, 'stride'), ud_index(manifest, 'numrecords')
    assert 'ByteAddressBuffer r0 : register(t0);' in ps, ps
    assert 'srv 0 buf table 0 0' in manifest, manifest
    assert_bounded(ps, stride, records, OFFSETS)
    print('raw conversion: four offen loads bounded by ud%d * ud%d' % (stride, records))

    clamped = translate(program(conversion(clamp=True)))
    cps, cmanifest = clamped.hlsl_ps(), clamped.manifest()
    cstride, crecords = ud_index(cmanifest, 'stride'), ud_index(cmanifest, 'numrecords')
    assert_bounded(cps, cstride, crecords, OFFSETS, clamped=True)
    print('clamped raw conversion: bound is max(stride, 1) * NUM_RECORDS')

    stride0 = program(sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff))
    text = translate(stride0).hlsl_ps()
    assert 'v1 = r0.Load(a + 0);' in text and 'uint lim' not in text, text
    print('stride clear only: unbounded raw load')

    # A dwordx2 at byte 28 covers bytes 28 and 32. Only the first dword is inside the 32-byte limit.
    cross2 = finish(smem_load_x4(DESC, 8) + conversion() + v_mov(0, 0)
                    + encode_mubuf(0x0d, imm=28, offen=1, vdata=1))
    x2 = translate(cross2)
    assert_bounded(x2.hlsl_ps(), ud_index(x2.manifest(), 'stride'), ud_index(x2.manifest(), 'numrecords'),
                   (28,), count=2)
    assert raw_dwords((0, 28), 2) == (SOURCE[7], 0)
    print('dwordx2: the dword that crosses NUM_RECORDS * stride is zero')

    # A dwordx4 at byte 24 covers 24, 28, 32 and 36. The last two dwords are out.
    cross4 = finish(smem_load_x4(DESC, 8) + conversion() + v_mov(0, 0)
                    + encode_mubuf(0x0e, imm=24, offen=1, vdata=1))
    x4 = translate(cross4)
    assert_bounded(x4.hlsl_ps(), ud_index(x4.manifest(), 'stride'), ud_index(x4.manifest(), 'numrecords'),
                   (24,), count=4)
    assert raw_dwords((0, 24), 4) == (SOURCE[6], SOURCE[7], 0, 0)
    print('dwordx4: two dwords inside the limit, then two zeros')

    assert raw_dwords((0,), 1, stride=0, records=4) == (0,)
    assert raw_dwords((4,), 1, stride=0, records=4) == (0,)
    print('stride 0: the product is zero, so every positive end is out of range')

    # soffset is an inline 8 in s21. v0 is 0, so the addresses are 8+4 and 8+28.
    shifted = finish(smem_load_x4(DESC, 8) + conversion() + s_mov(21, 128 + 8) + v_mov(0, 0)
                     + encode_mubuf(0x0c, imm=4, offen=1, vdata=1, soffset=21)
                     + encode_mubuf(0x0c, imm=28, offen=1, vdata=2, soffset=21),
                     exports=(1, 2, 1, 2))
    shifted_ps = translate(shifted).hlsl_ps()
    assert 'base + (0x00000008u)' in shifted_ps and 'base + (4)' in shifted_ps and 'base + (28)' in shifted_ps, shifted_ps
    assert raw_dwords((0, 8, 4), 1) == (SOURCE[3],)
    assert raw_dwords((0, 8, 28), 1) == (0,)
    print('non-zero soffset: byte 12 is in range and byte 36 is not')

    # The conversion runs on a copy of the loaded words, not on the registers the load wrote.
    copied = b''.join(s_mov(24 + i, DESC + i) for i in range(4))
    copy_ps = translate(program(copied + conversion(base=24, tmp=28), srsrc=24)).hlsl_ps()
    assert 'uint lim' in copy_ps and 'r0.Load(a + 0)' in copy_ps, copy_ps
    print('copied descriptor: the bound follows the s_mov copies')

    # v0 is 0xfffffffc. Adding 12, 28 or 32 wraps before the range test, so those loads are zero
    # rather than the words at the truncated addresses 8, 24 and 28.
    wrapped = program(conversion(), v0=0xfffffffc, offsets=(12, 28, 32, 0))
    wrapped_ps = translate(wrapped).hlsl_ps()
    assert 'ok = ok && base + (v0) >= base; base += v0;' in wrapped_ps, wrapped_ps
    assert raw_dwords((0xfffffffc, 12), 1) == (0,)
    assert raw_dwords((0xfffffffc, 28), 1) == (0,)
    assert raw_dwords((0xfffffffc, 32), 1) == (0,)
    assert (SOURCE[2], SOURCE[6], SOURCE[7]) == (0x40400000, 0x40e00000, 0x41000000)
    print('address wrap: a term that overflows the 32-bit sum reads zero')

    typed = finish(smem_load_x4(DESC, 8) + conversion() + v_mov(0, 0)
                   + encode_mtbuf(0, 22, imm=32, offen=1, vdata=1), exports=(1, 1, 1, 1))
    must_refuse('typed buffer load', typed, MODIFIED)

    missing_count = (sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                     + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                     + sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff)
                     + sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000))
    missing_oob = (sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                   + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                   + sop2('s_mul_i32', DESC + 2, DESC + 2, TMP))
    wrong_mask = conversion().replace(struct.pack('<I', 0xc000ffff), struct.pack('<I', 0xc000fff0), 1)
    oob_select_2 = conversion().replace(struct.pack('<I', 0x30000000), struct.pack('<I', 0x20000000), 1)
    scaled_by_constant = (sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                          + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                          + sop2('s_mul_i32', DESC + 2, DESC + 2, 128 + 4)
                          + sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff)
                          + sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000))
    and_first = (sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                 + sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                 + sop2('s_mul_i32', DESC + 2, DESC + 2, TMP)
                 + sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff)
                 + sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000))
    # The stride comes from a second descriptor, and the scaled count is written into the first.
    other = (smem_load_x4(24, 8, 32)
             + sop2('s_bfe_u32', TMP, 25, 255, 0xe0010)
             + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
             + sop2('s_mul_i32', DESC + 2, DESC + 2, TMP)
             + sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff)
             + sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000))
    must_refuse('missing byte count', program(missing_count), MODIFIED)
    must_refuse('missing OOB step', program(missing_oob), MODIFIED)
    must_refuse('word overwritten with zero', program(sop2('s_and_b32', DESC + 2, DESC + 2, 128)), MODIFIED)
    must_refuse('wrong stride mask', program(wrong_mask), MODIFIED)
    must_refuse('OOB select 2', program(oob_select_2), MODIFIED)
    must_refuse('scaled by a constant', program(scaled_by_constant), MODIFIED)
    must_refuse('steps out of order', program(and_first), MODIFIED)
    must_refuse('stride from a second descriptor', finish(smem_load_x4(DESC, 8) + other + v_mov(0, 0) + loads()), MODIFIED)
    must_refuse('idxen load', program(conversion(), idxen=True), INDEXED)

    model = expected_words()
    assert model == (SOURCE[0], SOURCE[3], SOURCE[7], 0), model
    assert cpu_load(0xffffffff) == 0
    print('CPU model: %s' % ','.join('%08x' % word for word in model))
    check_udsrc()


def compile_hlsl(compiler, kind, source, out, profile, directory):
    path = directory / ('%s.hlsl' % out)
    path.write_text(source, encoding='ascii')
    if kind == 'fxc':
        args = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', str(directory / ('%s.cso' % out)), str(path)]
    else:
        args = [compiler, '-T', profile, '-E', 'main', '-Fo', str(directory / ('%s.cso' % out)), str(path)]
    subprocess.run(args, check=True, capture_output=True, text=True)
    return directory / ('%s.cso' % out)


def entry_constants(manifest):
    count = int(next(line.split()[1] for line in manifest.splitlines() if line.startswith('ud ')))
    words = [0] * count
    words[ud_index(manifest, 'stride')] = STRIDE
    words[ud_index(manifest, 'numrecords')] = RECORDS
    return words


def draw_once(renderer, adapter, vs, ps, constants, expected):
    args = [renderer, adapter, '--startup-raw-buffer', str(vs), str(ps),
            ','.join('%08x' % word for word in constants),
            ','.join('%08x' % word for word in SOURCE),
            ','.join('%08x' % word for word in expected)]
    run = subprocess.run(args, capture_output=True, text=True)
    if run.returncode == 77:
        print(run.stdout + run.stderr)
        raise SystemExit(77)
    return run


def check_draw(dxc, renderer, adapter, fxc=None):
    pipe = translate(program(conversion()))
    translated, manifest = pipe.hlsl_ps(), pipe.manifest()
    constants = entry_constants(manifest)
    expected = expected_words()
    unbound = translated.replace('a + 4u <= lim', 'a + 4u <= 0xffffffffu')
    assert unbound != translated
    compilers = [('dxc', dxc, 'vs_6_0', 'ps_6_0')] + ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else [])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs_%s' % kind, vs_profile, directory)
            ps = compile_hlsl(compiler, kind, translated, 'ps_%s' % kind, ps_profile, directory)
            run = draw_once(renderer, adapter, vs, ps, constants, expected)
            if run.returncode:
                raise AssertionError('%s\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s: readback %s' % (kind, ','.join('%08x' % word for word in expected)))
            mutant = compile_hlsl(compiler, kind, unbound, 'ps_unbound_%s' % kind, ps_profile, directory)
            failed = draw_once(renderer, adapter, vs, mutant, constants, expected)
            if failed.returncode == 0 or 'deadbeef' not in failed.stderr:
                raise AssertionError('%s unbound control\n%s\n%s' % (kind, failed.stdout, failed.stderr))
            print('%s: dropping the bound reads the word past NUM_RECORDS * stride' % kind)


def check_udsrc():
    """The draw stand-in accepts numrecords and refuses an unknown kind instead of skipping the rest."""
    root = HERE.parent
    header = (root / 'src/startup_stubs/udsrc_line.h').read_text(encoding='utf-8')
    draw = (root / 'src/startup_stubs/d3d12_draw.inl').read_text(encoding='utf-8')
    assert 'else if (!std::strcmp(source, "numrecords")) kind = udsrc_num_records;' in header
    assert 'record_counts[i] = words[2];' in draw
    assert 'records = words[2];' in draw
    assert 'record_counts[i] = view.records;' in draw
    # records is a UINT and upload is a bool after it. Assigning by name avoids narrowing the count into the bool.
    assert 'view.records = records;' in draw and 'view.upload = upload;' in draw
    assert 'return {allocation.resource' not in draw and 'return {native, offset, bytes, stride' not in draw
    compute = (root / 'src/startup_stubs/d3d12_compute_dispatch.inl').read_text(encoding='utf-8')
    assert 'return {allocation.resource' not in compute and 'return {native, offset, bytes, 0, false}' not in compute
    assert 'ud.kind == Translation::num_records && ud.a >= 0 && ud.a < 32 ? record_counts[ud.a]' in draw
    assert 'x2w-draw unsupported udsrc' in draw
    driver = r"""
#include "udsrc_line.h"
int main() {
    UdSrcParsed parsed{};
    if (!accept_udsrc_line("udsrc 0 rect", 0, false, &parsed) || !parsed.is_udsrc || parsed.kind != udsrc_rect)
        return 1;
    if (!accept_udsrc_line("udsrc 4 numrecords 0", 4, false, &parsed) || parsed.kind != udsrc_num_records || parsed.a != 0)
        return 2;
    if (accept_udsrc_line("udsrc 5 notakind 0", 5, false, &parsed))
        return 3;
    if (accept_udsrc_line("udsrc 6 stride 0", 5, false, &parsed))
        return 4;
    if (!accept_udsrc_line("udsrc 5 stride 0", 5, false, &parsed) || parsed.kind != udsrc_stride)
        return 5;
    if (!accept_udsrc_line("srv 0 buf table 0 0", 0, false, &parsed) || parsed.is_udsrc)
        return 6;
    if (accept_udsrc_line("udsrc 0 mrtbits", 0, false, &parsed))
        return 7;
    if (!accept_udsrc_line("udsrc 0 mrtbits", 0, true, &parsed) || parsed.kind != udsrc_mrt_bits)
        return 8;
    return 0;
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        source = directory / 'udsrc_main.cpp'
        source.write_text(driver, encoding='ascii')
        binary = directory / 'udsrc_main'
        built = subprocess.run(
            ['g++', '-std=c++17', '-Wall', '-Wextra', '-I', str(root / 'src/startup_stubs'), str(source), '-o', str(binary)],
            capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stderr)
        run = subprocess.run([str(binary)], capture_output=True, text=True)
        if run.returncode:
            raise AssertionError('udsrc driver exit %s' % run.returncode)
    print('udsrc: numrecords is stored, and an unknown kind or a gap refuses the manifest')


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'draw':
        controls()
        check_draw(*args)
        return
    raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
