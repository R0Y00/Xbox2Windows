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


def encode_mubuf(op, *, imm=0, offen=0, idxen=0, vaddr=0, vdata=1, srsrc=DESC):
    w0 = (0b111000 << 26) | ((op & 0xff) << 18) | ((idxen & 1) << 13) | ((offen & 1) << 12) | (imm & 0xfff)
    w1 = (128 << 24) | (((srsrc // 4) & 0x1f) << 16) | ((vdata & 0xff) << 8) | (vaddr & 0xff)
    return struct.pack('<II', w0, w1)


def smem_load_x4(sdst, sbase):
    w0 = 0xf4000000 | (0x02 << 18) | (sdst << 6) | (sbase // 2)
    return struct.pack('<II', w0, 125 << 25) + struct.pack('<I', 0xbf8cc07f)


def v_mov(dst, inline):
    return struct.pack('<I', 0x7e000000 | (dst << 17) | (1 << 9) | (128 + inline))


def exp_mrt0(*regs):
    w1 = regs[0] | (regs[1] << 8) | (regs[2] << 16) | (regs[3] << 24)
    return struct.pack('<II', 0xf800180f, w1) + struct.pack('<I', 0xbf810000)


def conversion(clamp=False):
    words = [sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)]
    if clamp:
        words.append(sop2('s_max_u32', TMP, TMP, 129))
    words += [sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff),
              sop2('s_mul_i32', DESC + 2, DESC + 2, TMP),
              sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff),
              sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000)]
    return b''.join(words)


def loads(idxen=False):
    return b''.join(encode_mubuf(0x0c, imm=offset, offen=0 if idxen else 1, idxen=int(idxen), vdata=1 + i)
                    for i, offset in enumerate(OFFSETS))


def program(middle, idxen=False):
    return smem_load_x4(DESC, 8) + middle + v_mov(0, 0) + loads(idxen) + struct.pack('<I', 0xbf8c3f70) + exp_mrt0(1, 2, 3, 4)


def translate(ps):
    return G.Pipeline(VS, ps, ROOT)


def cpu_load(address, stride=STRIDE, records=RECORDS):
    """One raw dword: in range when the 32-bit end byte does not wrap and is at or before the product."""
    limit = (stride * records) & 0xffffffff
    end = (address + 4) & 0xffffffff
    if end < address or end > limit:
        return 0
    return SOURCE[address // 4]


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


def controls():
    pipe = translate(program(conversion()))
    ps, manifest = pipe.hlsl_ps(), pipe.manifest()
    stride, records = ud_index(manifest, 'stride'), ud_index(manifest, 'numrecords')
    assert 'ByteAddressBuffer r0 : register(t0);' in ps, ps
    assert 'srv 0 buf table 0 0' in manifest, manifest
    limit = '{ uint lim = ud%d * ud%d;' % (stride, records)
    for index, offset in enumerate(OFFSETS):
        # The immediate is part of the address. A one-dword load then reads a + 0 inside the bound.
        line = 'v%d = (a + 4u >= a && a + 4u <= lim) ? r0.Load(a + 0) : 0u;' % (1 + index)
        assert limit in ps and ('a = v0 + %d;' % offset) in ps and line in ps, ps
    print('raw conversion: four offen loads bounded by ud%d * ud%d' % (stride, records))

    clamped = translate(program(conversion(clamp=True)))
    cps, cmanifest = clamped.hlsl_ps(), clamped.manifest()
    cstride, crecords = ud_index(cmanifest, 'stride'), ud_index(cmanifest, 'numrecords')
    assert '{ uint lim = max(ud%d, 1u) * ud%d;' % (cstride, crecords) in cps, cps
    print('clamped raw conversion: bound is max(stride, 1) * NUM_RECORDS')

    stride0 = program(sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff))
    text = translate(stride0).hlsl_ps()
    assert 'v1 = r0.Load(a + 0);' in text and 'uint lim' not in text, text
    print('stride clear only: unbounded raw load')

    missing_count = (sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                     + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                     + sop2('s_and_b32', DESC + 3, DESC + 3, 255, 0xcfffffff)
                     + sop2('s_or_b32', DESC + 3, DESC + 3, 255, 0x30000000))
    missing_oob = (sop2('s_bfe_u32', TMP, DESC + 1, 255, 0xe0010)
                   + sop2('s_and_b32', DESC + 1, DESC + 1, 255, 0xc000ffff)
                   + sop2('s_mul_i32', DESC + 2, DESC + 2, TMP))
    must_refuse('missing byte count', program(missing_count), MODIFIED)
    must_refuse('missing OOB step', program(missing_oob), MODIFIED)
    must_refuse('in-place word rewrite', program(sop2('s_and_b32', DESC + 2, DESC + 2, 128)), MODIFIED)
    must_refuse('idxen load', program(conversion(), idxen=True), INDEXED)

    model = expected_words()
    assert model == (SOURCE[0], SOURCE[3], SOURCE[7], 0), model
    assert cpu_load(32, stride=0, records=4) == 0
    assert cpu_load(0xffffffff) == 0
    print('CPU model: %s' % ','.join('%08x' % word for word in model))


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
