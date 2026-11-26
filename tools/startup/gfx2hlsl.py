"""gfx1013 machine code -> HLSL for a vertex+pixel pipeline (startup tools; see docs/startup-runtime.md).

Model: one lane. Registers are uint variables; exec/vcc/mask registers hold the lane's bit in bit 0.
Scalar registers are also tracked symbolically at translation time so that pointer and descriptor
values (tables, vertex-buffer table, root buffers) turn into HLSL resources instead of arithmetic.
The output is HLSL for both stages plus a manifest that tells the runtime what to bind where.
"""
import bisect
from contextlib import suppress
import functools
import os
import re
import struct
import sys
import sysconfig
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import disasm_rdna2 as D  # noqa: E402


class Unsupported(Exception):
    pass


class StoredSrv(Unsupported):
    """A descriptor first read as an SRV and later stored to; translate_to retries with it in rw_keys."""

    def __init__(self, key, message):
        super().__init__(message)
        self.key = key


PREFETCH = bytes.fromhex('0300a0bf')

# DPP16 quad pair broadcasts (quad_perm of one lane of every horizontal or vertical pair into the other). The lanes of
# a quad are the pixels 0 1 / 2 3, row by row; lane i reads lane quad_perm[i].
DPP_PAIR = frozenset(((1, 1, 3, 3), (0, 0, 2, 2), (2, 3, 2, 3), (0, 1, 0, 1)))
DPP_SCRATCH = 256

# SDWA (sub-dword addressing) selections, with the same meaning as tools/pixel_aot.py: a source
# selection is zero-extended, or sign-extended under sext(); a destination selection writes the low
# bits of the result into that field, UNUSED_PAD zeroes the others and UNUSED_PRESERVE keeps them.
SDWA_SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
            'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}


def sopk_imm(token, signed):
    """The 32-bit word a SOPK 16-bit immediate stands for.

    RDNA 2 ISA SOPK: s_movk_i32 and the s_cmpk_*_i32 forms use signext(SIMM16[15:0]) (LLVM s16imm); the s_cmpk_*_u32
    forms use SIMM16[15:0] zero-extended (LLVM u16imm). The disassembler prints the field as unsigned hex."""
    m = re.fullmatch(r'(-?)(?:0x([0-9a-f]+)|(\d+))', token)
    if not m:
        raise Unsupported('SOPK immediate %r' % token)
    word = int(m.group(2), 16) if m.group(2) else int(m.group(3))
    if m.group(1):
        word = -word
    if not (-0x8000 if signed else 0) <= word <= 0xffff:
        raise Unsupported('SOPK immediate %r outside 16 bits' % token)
    if signed:
        word = ((word & 0xffff) ^ 0x8000) - 0x8000
    return word & 0xffffffff


def sdwa_source(word, sel, sext):
    """The uint expression a source selection reads out of a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return word
    if sext:
        return 'asuint(asint(%s << %du) >> %du)' % (word, 32 - offset - width, 32 - width)
    return '((%s >> %du) & %#xu)' % (word, offset, (1 << width) - 1)


def sdwa_dest(result, sel, unused, old):
    """The uint expression a destination selection writes back into a 32-bit register."""
    offset, width = SDWA_SEL[sel]
    if width == 32:
        return result
    if unused == 'UNUSED_SEXT':
        raise Unsupported('SDWA UNUSED_SEXT')
    field = '((%s & %#xu) << %du)' % (result, (1 << width) - 1, offset)
    if unused == 'UNUSED_PAD':
        return field
    if unused != 'UNUSED_PRESERVE':
        raise Unsupported('SDWA dst_unused %s' % unused)
    keep = ~(((1 << width) - 1) << offset) & 0xffffffff
    return '((asuint(%s) & %#xu) | %s)' % (old, keep, field)


def gather_2d_profile_words(words):
    """The sampler words a 2D gather is checked with: fields a 2D gather cannot observe are set to the profile.

    SQ_IMG_SAMP_WORD0 bits 8:6 (CLAMP_Z) address the third coordinate, which a 2D view does not have; D3D11.3
    functional spec 22.4.2 (gather4): "Only the addressing modes of the sampler are used". For the same reason a
    point XY_MAG_FILTER or XY_MIN_FILTER (WORD2 bits 21:20 and 23:22, value 0) reads the same four texels as the
    bilinear one: gather4 "Gathers the four texels that would be used in a bi-linear filtering operation", whatever
    the sampler's filter, so a point filter is checked as bilinear. That point and bilinear read the same four
    texels is the D3D11.3 rule; it has not been measured on console hardware. Anisotropic filters (2, 3) keep
    their values and stay refused. WORD2 bit 28 (MIP_POINT_PRECLAMP) acts only on mip selection, and the _lz
    gather reads mip zero ("the top level of any mip pyramid is used", the same spec section), so it is cleared
    as well; the level-zero comparison path (pixel_aot.comparison_sampler_state) accepts it for the same reason.
    Bits 29-31 stay checked. The words bound as the static sampler stay the title's own.
    """
    w0, w1, w2, w3 = words
    for shift in (20, 22):
        if (w2 >> shift) & 3 == 0:
            w2 |= 1 << shift
    return (w0 & ~(7 << 6), w1, w2 & ~(1 << 28), w3)


def strip_header(code):
    at = code.find(PREFETCH, 0, 0x80)
    # A call (s_swappc_b64) just before the prefetch is the program's first instruction, not header: keep it.
    if at >= 4 and at % 4 == 0 and struct.unpack_from('<I', code, at - 4)[0] & 0xff80ff00 == 0xbe802100:
        at -= 4
    return code[at:] if at >= 0 and at % 4 == 0 else code


# A stream vertex shader whose inputs come from a fetch routine the driver builds from the input layout starts
# with this call; the routine's address arrives in s[4:5]. Its FSSM header lists, from +0x138, one 5-byte record
# per fetched input: kind (6), input-signature register, component mask, first VGPR, component count - 1.
# Checked offline against the input signatures of every readable vertex program of one title's shader archive
# that starts with the call: each record's register and mask match a signature input (the mask is the read
# mask; the count reaches its highest component), the first VGPR is v4 and each next record is 4 VGPRs on.
# Programs without the call have no records. The header's u32 at +0x1d8 holds the record count.
# Any other header size, record kind or count is refused.
FETCH_CALL = 0xbe842104  # s_swappc_b64 s[4:5], s[4:5]
FETCH_HEADER_BASE, FETCH_RECORDS, FETCH_KIND, FETCH_COUNT = 0x2f8, 0x138, 6, 0x1d8
FETCH_SLOTS = 8  # vertex-buffer slots the stand-in binds (D3D12_VERTEX_BUFFER_VIEW vb[8])
# DXGI format -> (BUF_FMT name in FETCH_FORMATS, bytes per element).
FETCH_DXGI = {2: ('BUF_FMT_32_32_32_32_FLOAT', 16), 6: ('BUF_FMT_32_32_32_FLOAT', 12), 16: ('BUF_FMT_32_32_FLOAT', 8),
              41: ('BUF_FMT_32_FLOAT', 4), 28: ('BUF_FMT_8_8_8_8_UNORM', 4), 49: ('BUF_FMT_8_8_UNORM', 2),
              61: ('BUF_FMT_8_UNORM', 1), 34: ('BUF_FMT_16_16_FLOAT', 4), 35: ('BUF_FMT_16_16_UNORM', 4),
              37: ('BUF_FMT_16_16_SNORM', 4), 30: ('BUF_FMT_8_8_8_8_UINT', 4)}
# Typed buffer loads: (word load or None, the four component expressions). Components a format lacks read as
# 0, 0, 0, 1 here, as in the existing tbuffer translation; that default is an assumption, not a measurement.
FETCH_FORMATS = {
    'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
    'BUF_FMT_8_8_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '(w >> 16) & 255u', 'w >> 24']),
    'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
    'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
    'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
    'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
    # Two unsigned 16-bit fields, x in the low half (LLVM 15: BUF_DATA_FORMAT_16_16 with BUF_NUM_FORMAT_UINT,
    # gfx10 unified format 27); a missing W of an integer format reads 1.
    'BUF_FMT_16_16_UINT': ('ldu({0}, a)', ['w & 65535u', 'w >> 16', '0u', '1u']),
    # SNORM: sign-extend each half, divide by 32767 and clamp to -1.0, so that -32768 and -32767 both give -1.0
    # (Direct3D / Vulkan SNORM rule; LLVM 15 and Mesa ac_formats name the format 16_16_SNORM, number 24).
    'BUF_FMT_16_16_SNORM': ('ldu({0}, a)', ['U(max(float(int(w << 16) >> 16) / 32767.0, -1.0))',
                                            'U(max(float(int(w) >> 16) / 32767.0, -1.0))', '0u', 'U(1.0)']),
    'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
    # Four half channels in one 8-byte element: x/y in the first dword, z/w in the second.
    'BUF_FMT_16_16_16_16_FLOAT': (None, ['U(f16tof32(ldu({0}, a)))', 'U(f16tof32(ldu({0}, a) >> 16))',
                                         'U(f16tof32(ldu({0}, a + 4)))', 'U(f16tof32(ldu({0}, a + 4) >> 16))']),
    'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
    'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
    'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
    'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}


# gfx10 buffer-descriptor FORMAT field (word 3, bits 12..18) for each FETCH_FORMATS entry: the same unified
# numbers as the tbuffer `format:` operand (tools/disasm_rdna2.py MTBUF_FORMATS; 1 is the omitted default).
# A MUBUF buffer_load_format_* has no format operand, so its translation switches on this field at run time.
# An invalid FORMAT reaches the switch default, which poisons the decoded components with this NaN pattern.
# Selectors for constants still return constants, so the caller must validate FORMAT before submission.
DESC_FORMAT_UNKNOWN = 0x7fc0f0f0
DESC_FORMAT_CODES = {
    'BUF_FMT_8_UNORM': 1, 'BUF_FMT_8_UINT': 5, 'BUF_FMT_8_8_UNORM': 14, 'BUF_FMT_32_FLOAT': 22,
    'BUF_FMT_16_16_UNORM': 23, 'BUF_FMT_16_16_SNORM': 24, 'BUF_FMT_16_16_UINT': 27, 'BUF_FMT_16_16_FLOAT': 29, 'BUF_FMT_8_8_8_8_UNORM': 56, 'BUF_FMT_8_8_8_8_UINT': 60,
    'BUF_FMT_32_32_FLOAT': 64, 'BUF_FMT_16_16_16_16_FLOAT': 71, 'BUF_FMT_32_32_32_FLOAT': 74,
    'BUF_FMT_32_32_32_32_FLOAT': 77}
DESC_FORMAT_SHIFT, DESC_FORMAT_MASK = 12, 127
# Integer formats read a DST_SEL constant 1 as the integer 1, every other format as 1.0.
DESC_ONE_INTEGER = ('BUF_FMT_8_UINT', 'BUF_FMT_16_16_UINT', 'BUF_FMT_8_8_8_8_UINT')

# Supported 1-channel formats for compute buffer_store_format_d16_x (RDNA 2 MUBUF 0x84):
# V# word 3 bits 18:12 carry the 7-bit GFX10 unified FORMAT when RESOURCE_LEVEL (bit 24) is 1,
# or legacy (DFMT << 3) | NFMT when bit 24 is 0. Bits checked zero: TYPE[31:30], reserved[27:25],
# ADD_TID_ENABLE[23], INDEX_STRIDE[22:21], reserved[20:19]. DST_SEL_X[2:0] must be 4 (SQ_SEL_X);
# OOB_SELECT[29:28] must be 0 (SQ_OOB_INDEX_AND_OFFSET, structured with stride != 0) or
# 3 (SQ_OOB_COMPLETE, raw byte bounds with stride == 0). Modes 1 (INDEX_ONLY) and 2
# (NUM_RECORDS_0 / disabled bounds check) are refused. Entry constant uav_stride carries the
# unmasked upper 16 bits of V# word 1 (DW1 >> 16), so stride > 0x3fffu rejects CACHE_SWIZZLE
# (bit 30) and SWIZZLE_ENABLE (bit 31).
STORE_D16_X_FORMATS = {
    7: ('BUF_FMT_16_UNORM', 2),
    8: ('BUF_FMT_16_SNORM', 2),
    11: ('BUF_FMT_16_UINT', 2),
    12: ('BUF_FMT_16_SINT', 2),
    13: ('BUF_FMT_16_FLOAT', 2),
    22: ('BUF_FMT_32_FLOAT', 4),
}
STORE_D16_X_LEGACY_FORMATS = {
    (2 << 3) | 0: 7,   # DFMT_16 + NFMT_UNORM
    (2 << 3) | 1: 8,   # DFMT_16 + NFMT_SNORM
    (2 << 3) | 4: 11,  # DFMT_16 + NFMT_UINT
    (2 << 3) | 5: 12,  # DFMT_16 + NFMT_SINT
    (2 << 3) | 7: 13,  # DFMT_16 + NFMT_FLOAT
    (4 << 3) | 7: 22,  # DFMT_32 + NFMT_FLOAT
}
STORE_D16_X_WORD3_ZERO_MASK = 0xcef80000  # bits 31:30 TYPE | 27:25 reserved | 23 ADD_TID_ENABLE | 22:21 INDEX_STRIDE | 20:19 reserved


def validate_store_d16_x_word3(word3):
    """Validate V# word 3 for compute buffer_store_format_d16_x and return (code, name, elem_bytes)."""
    if not isinstance(word3, int) or isinstance(word3, bool) or not 0 <= word3 <= 0xffffffff:
        raise Unsupported('buffer_store_format_d16_x invalid descriptor word 3 %r' % (word3,))
    if word3 & STORE_D16_X_WORD3_ZERO_MASK:
        raise Unsupported('buffer_store_format_d16_x unsupported descriptor word 3 flags 0x%08x' % word3)
    if (word3 & 7) != 4:
        raise Unsupported('buffer_store_format_d16_x non-identity DST_SEL_X %d' % (word3 & 7))
    oob_select = (word3 >> 28) & 3
    if oob_select not in (0, 3):
        raise Unsupported('buffer_store_format_d16_x unsupported OOB_SELECT %d' % oob_select)
    field = (word3 >> DESC_FORMAT_SHIFT) & DESC_FORMAT_MASK
    code = field if ((word3 >> 24) & 1) else STORE_D16_X_LEGACY_FORMATS.get(field)
    info = STORE_D16_X_FORMATS.get(code)
    if info is None:
        raise Unsupported('buffer_store_format_d16_x unsupported descriptor FORMAT %d' % field)
    return code, info[0], info[1]


# Compute buffer_store_format_xyzw (MUBUF 0x07) writes only formats with exactly four channels, the set and
# conversions of tools/compute_store_aot.py: a channel-count mismatch writes nothing, since what the format unit
# does with the extra components is not established here. Word 3 bounds and OOB_SELECT checks are those of
# buffer_store_format_d16_x above; DST_SEL must be the identity X, Y, Z, W (0xfac).
STORE_FORMAT_XYZW_FORMATS = {
    50: ('BUF_FMT_2_10_10_10_UNORM', 4),
    56: ('BUF_FMT_8_8_8_8_UNORM', 4),
    71: ('BUF_FMT_16_16_16_16_FLOAT', 8),
    75: ('BUF_FMT_32_32_32_32_UINT', 16),
    77: ('BUF_FMT_32_32_32_32_FLOAT', 16),
}
STORE_FORMAT_XYZW_LEGACY_FORMATS = {
    (9 << 3) | 0: 50,   # DFMT_2_10_10_10 + NFMT_UNORM
    (10 << 3) | 0: 56,  # DFMT_8_8_8_8 + NFMT_UNORM
    (12 << 3) | 7: 71,  # DFMT_16_16_16_16 + NFMT_FLOAT
    (14 << 3) | 4: 75,  # DFMT_32_32_32_32 + NFMT_UINT
    (14 << 3) | 7: 77,  # DFMT_32_32_32_32 + NFMT_FLOAT
}


def validate_store_format_xyzw_word3(word3):
    """Validate V# word 3 for compute buffer_store_format_xyzw and return (code, name, elem_bytes)."""
    name = 'buffer_store_format_xyzw'
    if not isinstance(word3, int) or isinstance(word3, bool) or not 0 <= word3 <= 0xffffffff:
        raise Unsupported('%s invalid descriptor word 3 %r' % (name, word3))
    if word3 & STORE_D16_X_WORD3_ZERO_MASK:
        raise Unsupported('%s unsupported descriptor word 3 flags 0x%08x' % (name, word3))
    if (word3 & 0xfff) != 0xfac:
        raise Unsupported('%s non-identity DST_SEL 0x%03x' % (name, word3 & 0xfff))
    oob_select = (word3 >> 28) & 3
    if oob_select not in (0, 3):
        raise Unsupported('%s unsupported OOB_SELECT %d' % (name, oob_select))
    field = (word3 >> DESC_FORMAT_SHIFT) & DESC_FORMAT_MASK
    code = field if ((word3 >> 24) & 1) else STORE_FORMAT_XYZW_LEGACY_FORMATS.get(field)
    info = STORE_FORMAT_XYZW_FORMATS.get(code)
    if info is None:
        raise Unsupported('%s descriptor FORMAT %d is not a handled 4-channel format' % (name, field))
    return code, info[0], info[1]

def starts_with_fetch_call(code):
    return len(code) >= 4 and struct.unpack_from('<I', code)[0] == FETCH_CALL


def parse_input_layout(text):
    """'ia1 <n>' then one '<semantic> <index> <format> <slot> <offset> <class> <step>' line per element."""
    lines = text.split('\n')
    if not re.fullmatch(r'ia1 ([1-9]|[12][0-9]|3[0-2])', lines[0]) or lines[-1] != '' or len(lines) != int(lines[0][4:]) + 2:
        raise Unsupported('invalid input layout request')
    elements, ends = [], {}
    for line in lines[1:-1]:
        m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]{0,62}) (\d{1,10}) (\d{1,10}) (\d{1,2}) (\d{1,10}) (\d) (\d{1,10})', line)
        if not m:
            raise Unsupported('invalid input layout element')
        name, index, fmt, slot, offset, cls, step = m.group(1).upper(), *map(int, m.groups()[1:])
        if offset == 0xffffffff:  # D3D12_APPEND_ALIGNED_ELEMENT: right after the previous element of the slot
            # shortcut: unknown when an earlier size is unknown or ends off a dword (packing not measured).
            offset = ends.get(slot, 0)
            if offset is not None and offset % 4:
                offset = None
        size = FETCH_DXGI[fmt][1] if fmt in FETCH_DXGI else None
        ends[slot] = None if offset is None or size is None else offset + size
        # Format, class and slot are checked in stream_fetch, only for the elements a record fetches.
        elements.append(dict(name=name, index=index, format=fmt, slot=slot, offset=offset, cls=cls, step=step))
    return elements


def signature_inputs(isg1):
    """ISG1 part -> {register: (semantic name, semantic index, system value)}."""
    count, first = struct.unpack_from('<II', isg1, 0)
    out = {}
    for i in range(count):
        _, name_at, index, system, _, register, _ = struct.unpack_from('<6IB', isg1, first + 32 * i)
        out[register] = (isg1[name_at:isg1.index(b'\0', name_at)].decode('ascii').upper(), index, system)
    return out


def stream_fetch(vs_container, layout_text):
    """The fetch a stream vertex shader expects before its first instruction: [dict(vgpr, count, ...layout)]."""
    from pixel_aot import container_image
    from resource_inventory import parts
    if layout_text is None:
        raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
    sections = parts(vs_container)
    image = container_image(sections[b'DXIL'], stage=1)
    if len(image) < FETCH_HEADER_BASE or struct.unpack_from('<Q', image, 192)[0] != FETCH_HEADER_BASE:
        raise Unsupported('unverified FSSM header for a vertex fetch table')
    inputs, layout = signature_inputs(sections[b'ISG1']), parse_input_layout(layout_text)
    fetch, at = [], FETCH_RECORDS
    while image[at] != 0:
        kind, register, mask, vgpr, last = image[at:at + 5]
        count = last + 1
        # The count reaches the highest component the shader reads (mask 0x4 comes with count 3): load all of them.
        if (kind != FETCH_KIND or mask >> last != 1 or not 1 <= count <= 4 or vgpr < 1 or vgpr + count > 256
                or len(fetch) == 16):
            raise Unsupported('unverified vertex fetch record %s' % image[at:at + 5].hex())
        if register not in inputs:
            raise Unsupported('vertex fetch record for input register %d outside the signature' % register)
        name, index, _ = inputs[register]
        match = [e for e in layout if (e['name'], e['index']) == (name, index)]
        if len(match) != 1:
            raise Unsupported('input layout has %d elements for vertex input register %d' % (len(match), register))
        e = match[0]
        if e['format'] not in FETCH_DXGI:
            raise Unsupported('input layout format %d' % e['format'])
        if e['cls'] != 0 or e['step'] != 0:
            raise Unsupported('per-instance input layout element')
        if e['slot'] >= FETCH_SLOTS:
            raise Unsupported('input layout slot %d: the stand-in binds vertex-buffer slots 0 to %d' % (e['slot'], FETCH_SLOTS - 1))
        if e['offset'] is None:
            raise Unsupported('appended input layout element after an element of unknown or unaligned size')
        fetch.append(dict(e, vgpr=vgpr, count=count))
        at += 5
    if not fetch:
        raise Unsupported('vertex stage calls a fetch routine and its container lists no fetched inputs')
    stored = struct.unpack_from('<I', image, FETCH_COUNT)[0]
    if stored != len(fetch):
        raise Unsupported('vertex fetch header counts %d records, the table holds %d' % (stored, len(fetch)))
    taken = [r for e in fetch for r in range(e['vgpr'], e['vgpr'] + e['count'])]
    if len(taken) != len(set(taken)):
        raise Unsupported('vertex fetch records overlap')
    return fetch


# The interface between the two stages of a stream pipeline, as both containers state it in their native headers.
# Vertex: from +0x1dc one 6-byte record per exported parameter (kind 6, output-signature register, parameter index,
# 0, 0, 0), count at +0x29c. Pixel (header 0xef8 bytes): from +0x138 one 8-byte record per attribute (kind 6,
# input-signature register, component mask, attribute index, 0, 0, the register again, interpolation 1 or 2), count
# at +0x238. Records with other values pack two signature registers into one parameter or attribute; that form is
# refused here.
IFACE_VS_RECORDS, IFACE_VS_COUNT, IFACE_PS_HEADER_BASE, IFACE_PS_RECORDS, IFACE_PS_COUNT = 0x1dc, 0x29c, 0xef8, 0x138, 0x238


def signature_entries(part):
    """ISG1/OSG1 part -> [(register, semantic name, semantic index, system value, mask)]."""
    count, first = struct.unpack_from('<II', part, 0)
    out = []
    for i in range(count):
        _, name_at, index, system, _, register, mask = struct.unpack_from('<6IB', part, first + 32 * i)
        out.append((register, part[name_at:part.index(b'\0', name_at)].decode('ascii').upper(), index, system, mask))
    return out


def stream_interface(vs_container, ps_container):
    """{pixel attribute: vertex parameter}, matched by semantic through both containers' interface tables."""
    from pixel_aot import container_image
    from resource_inventory import parts
    vs, ps = parts(vs_container), parts(ps_container)
    inputs = [entry for entry in signature_entries(ps[b'ISG1']) if not entry[3]] if b'ISG1' in ps else []
    if not inputs:
        return {}  # nothing to link: a read of any attribute is refused in note_attr
    vs_image, ps_image = container_image(vs[b'DXIL'], stage=1), container_image(ps[b'DXIL'], stage=0)
    if (struct.unpack_from('<Q', vs_image, 192)[0] != FETCH_HEADER_BASE or
            struct.unpack_from('<Q', ps_image, 192)[0] != IFACE_PS_HEADER_BASE):
        raise Unsupported('unverified FSSM header for the stage interface')
    attributes = {}
    count = struct.unpack_from('<I', ps_image, IFACE_PS_COUNT)[0]
    if count > (IFACE_PS_COUNT - IFACE_PS_RECORDS) // 8:
        raise Unsupported('pixel input table exceeds its header range')
    for i in range(count):
        kind, register, mask, attr, b4, b5, again, interp = ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8]
        if (kind != 6 or attr != i or b4 or b5 or again != register or interp not in (1, 2) or not mask
                or mask & ~0xf or register in attributes):
            raise Unsupported('unverified pixel input record %s' % ps_image[IFACE_PS_RECORDS + 8 * i:IFACE_PS_RECORDS + 8 * i + 8].hex())
        declared = 0
        for entry in inputs:
            if entry[0] == register:
                declared |= entry[4]
        if mask != declared:
            raise Unsupported('pixel input record mask 0x%x differs from the signature mask 0x%x of register %d'
                              % (mask, declared, register))
        attributes[register] = mask
    if not attributes:
        return {}
    params = {}
    count = struct.unpack_from('<I', vs_image, IFACE_VS_COUNT)[0]
    if count > (IFACE_VS_COUNT - IFACE_VS_RECORDS) // 6:
        raise Unsupported('vertex output table exceeds its header range')
    for i in range(count):
        kind, register, param, *rest = vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6]
        if kind != 6 or param != i or any(rest) or register in params:
            raise Unsupported('unverified vertex output record %s' % vs_image[IFACE_VS_RECORDS + 6 * i:IFACE_VS_RECORDS + 6 * i + 6].hex())
        params[register] = param
    outputs = {}
    for register, name, index, system, mask in signature_entries(vs[b'OSG1']):
        if system:
            continue
        if (name, index) in outputs:
            raise Unsupported('duplicate vertex output semantic %s%d' % (name, index))
        outputs[name, index] = register, mask
    mapping = {}
    for attr, (register, mask) in enumerate(attributes.items()):
        sources = set()
        for _, name, index, _, semantic_mask in (e for e in inputs if e[0] == register):
            output = outputs.get((name, index))
            if not output or output[1] != semantic_mask:
                raise Unsupported('pixel input %s%d has no vertex output with the same components' % (name, index))
            sources.add(output[0])
        if len(sources) != 1 or next(iter(sources)) not in params:
            raise Unsupported('pixel attribute %d does not come from exactly one exported vertex register' % attr)
        mapping[attr] = (params[next(iter(sources))], mask)
    return mapping


# Binary32 words for inline source encodings 240 through 248, in encoding order.
CNDMASK_INLINE_FLOAT_BITS = (0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000,
                            0x40000000, 0xc0000000, 0x40800000, 0xc0800000,
                            0x3e22f983)  # 1/(2*pi)


INTERP_DST_HIGH = 'op_sel:[0,0,0,1]'


LDS_UNUSED_W1_MASK = {
    'ds_write2_b32': 0xff000000, 'ds_write2st64_b32': 0xff000000, 'ds_read2st64_b32': 0x00ffff00,
    **dict.fromkeys(('ds_add_u32', 'ds_min_u32', 'ds_max_u32', 'ds_and_b32', 'ds_or_b32'), 0xffff0000),
    **dict.fromkeys(('ds_add_rtn_u32', 'ds_min_rtn_u32', 'ds_max_rtn_u32', 'ds_and_rtn_b32',
                     'ds_or_rtn_b32', 'ds_wrxchg_rtn_b32'), 0x00ff0000),
}


def check_encoding(code, off, n, name, asm):
    # These SOP1 forms have an unused field that the disassembler does not print.
    # Only the ordinary four-byte encodings (no literal or hidden operand) are used
    # by the calculated-branch proof.
    if name in ('s_getpc_b64', 's_setpc_b64'):
        word, = struct.unpack_from('<I', code, off)
        unused = word & (0xff if name == 's_getpc_b64' else 0x7f0000)
        if n != 4 or unused:
            raise Unsupported('s_setpc_b64: unsupported %s encoding at +0x%x' % (name, off))
    if name == 'image_load_mip':
        # The shared disassembler does not expose these gfx10 MIMG flags.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('image_load_mip tfe')
        if word1 & (1 << 30):
            raise Unsupported('image_load_mip a16')
    if name in LDS_UNUSED_W1_MASK:
        if n != 8 or off + 8 > len(code):
            raise Unsupported('%s encoding length %d at +0x%x' % (name, n, off))
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & (1 << 16):
            raise Unsupported('%s reserved bit 16 set at +0x%x' % (name, off))
        if word1 & LDS_UNUSED_W1_MASK[name]:
            raise Unsupported('%s unused operand field set at +0x%x' % (name, off))
    if re.fullmatch(r'v_cmpx?_(lt|eq|le|gt|ne|ge)_[iu]16_e64', name):
        # The decoder prints NEG on an inline integer as a sign (inline 1 with NEG reads `-1`, the same text as
        # inline -1), and LLVM 15 rejects `-v1` and `|v1|` on these sources: refuse either bit (VOP3 w0[10:8] ABS,
        # w1[31:29] NEG) so a printed -N is always a genuine inline integer.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 16-bit integer source modifier' % name)
    if name in ('v_sad_u8', 'v_sad_u16', 'v_sad_u32', *VOP3_ONLY_VEC_U):
        # VOP3A integer forms take no source abs/neg; refuse those bits here so neg on an inline
        # integer does not print as -N and get accepted as a negative constant.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs' % name)
        if name in VOP3_ONLY_VEC_U and (word1 >> 18) & 0x1ff:
            raise Unsupported('%s with unused src2 field set' % name)
    if name == 'v_cvt_off_f32_i4_e64':
        # Integer-nibble conversion: the instruction has one source, so only src0's VOP3A ABS (w0[8]) and NEG
        # (w1[29]) can carry a modifier. Masking all three bit positions of each field (w0[10:8], w1[31:29])
        # also refuses the unused src1/src2 bits, which keeps an inline constant from masquerading as a
        # negative integer and keeps a hidden modifier out of the printed operand.
        word0, word1 = struct.unpack_from('<2I', code, off)
        if word0 & 0x700 or word1 & 0xe0000000:
            raise Unsupported('%s with neg/abs: not a 4-bit integer source modifier' % name)
    if name == 'v_cndmask_b32_e64':
        # The printed -1 is ambiguous: it can be plain inline -1 or inline 1 with NEG.
        word0, word1 = struct.unpack_from('<2I', code, off)
        for i in range(2):
            source = (word1 >> (9 * i)) & 0x1ff
            if not (128 <= source <= 208 or 240 <= source <= 248 or source == 255):
                continue
            if word0 & (1 << (8 + i)):
                raise Unsupported('v_cndmask_b32_e64 ABS on an immediate')
            if not word1 & (1 << (29 + i)):
                continue
            if source <= 192:
                bits = source - 128
            elif source <= 208:
                bits = (192 - source) & 0xffffffff
            elif source <= 248:
                bits = CNDMASK_INLINE_FLOAT_BITS[source - 240]
            else:
                if n != 12 or off + n > len(code):
                    raise Unsupported('v_cndmask_b32_e64 missing literal dword')
                bits, = struct.unpack_from('<I', code, off + 8)
            # Preserve the other operand spellings and any modifiers on the final operand.
            fields = asm.split(',', 3)
            if len(fields) != 4:
                raise Unsupported('malformed v_cndmask_b32_e64 operands')
            operand = fields[i + 1]
            fields[i + 1] = operand[:len(operand) - len(operand.lstrip())] + '0x%08x' % (bits ^ 0x80000000)
            asm = ','.join(fields)
    if n >= 8 and struct.unpack_from('<I', code, off)[0] >> 26 == 0b110101:
        # Read the word from the encoding: NEG on inline 1 is printed as `-1`, indistinguishable
        # from the plain inline -1 selector.
        # LLVM AMDGPUAsmParser::applyInputFPModifiers folds NEG into an immediate by XOR with the
        # FP sign mask. RDNA 2 lists VOP3 NEG as a float input modifier. Hardware behavior for
        # these immediate sources has not been measured.
        word0, word1 = struct.unpack_from('<2I', code, off)
        base = name.removesuffix('_e64')
        float_sources = 0
        if base in VEC_F:
            float_sources = VEC_F[base][0]
        elif re.fullmatch(r'v_cmpx?_\w+_f32', base):
            float_sources = 1 if base in ('v_cmp_class_f32', 'v_cmpx_class_f32') else 2
        rewritten = {}
        for i in range(3):
            if name == 'v_cndmask_b32_e64' and i < 2 or not word1 & (1 << (29 + i)):
                continue
            source = (word1 >> (9 * i)) & 0x1ff
            if not (128 <= source <= 192 or source == 255):
                continue
            # The selected float forms are VOP3A, where w0[10:8] holds ABS. Keep the
            # refusal for NEG combined with ABS on this source.
            if i >= float_sources or word0 & (1 << (8 + i)):
                raise Unsupported('%s with NEG on an integer or literal source %d: the printed sign would read as inline -N'
                                  % (name, i))
            if source == 255:
                if n != 12 or off + n > len(code):
                    raise Unsupported('%s missing literal dword' % name)
                bits, = struct.unpack_from('<I', code, off + 8)
            else:
                bits = source - 128
            # f() bitcasts this hex word through F(), the asfloat macro.
            rewritten[i + 1] = '0x%08x' % (bits ^ 0x80000000)
        if rewritten:
            mnemonic, operands, mods = parse(asm)
            if len(operands) <= max(rewritten):
                raise Unsupported('malformed %s operands' % name)
            for index, token in rewritten.items():
                operands[index] = token
            asm = ' '.join([mnemonic, ', '.join(operands), *mods])
    return asm


def decode(code, refusals=None):
    out, off = [], 0
    while off + 4 <= len(code):
        try:
            n, name, asm, _ = D.decode_instruction(code, off)
        except Exception as error:
            raise Unsupported('undecodable word at +0x%x: %s' % (off, error))
        try:
            asm = check_encoding(code, off, n, name, asm)
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), 'decode'))
        else:
            out.append((off, n, name, asm))
        off += n
    last = max((i for i, x in enumerate(out) if x[2] == 's_endpgm'), default=len(out) - 1)
    return out[:last + 1]


def split_top(text, sep=','):
    parts, depth, cur = [], 0, ''
    for ch in text:
        if ch in '[(':
            depth += 1
        elif ch in '])':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


@functools.lru_cache(maxsize=None)
def parse_cached(asm):
    name, _, rest = asm.partition(' ')
    ops = split_top(rest)
    mods = []
    if ops:
        tail = split_top(ops[-1], ' ')
        ops[-1], mods = tail[0], tail[1:]
    return name, tuple(ops), tuple(mods)


def parse(asm):
    """-> (mnemonic, [operands], [modifier tokens]); the analysis passes parse each instruction many times."""
    name, ops, mods = parse_cached(asm)
    return name, list(ops), list(mods)


FLOATS = {'0.5': 0.5, '-0.5': -0.5, '1.0': 1.0, '-1.0': -1.0, '2.0': 2.0, '-2.0': -2.0, '4.0': 4.0, '-4.0': -4.0,
          '0.15915494': 0.15915494}


def fbits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def ptr_high_symbol(source, offset):
    """High word of a 64-bit pointer advanced by a literal offset: the original word, or the one a completed
    s_addc_u32 produced for exactly that accumulated offset (an advanced word never equals another offset's)."""
    return ('ptrhi', source) if not offset else ('ptrhi', source, offset)


# Sources whose pointer is a 64-bit register pair (the low word 'ptr', the high word 'ptrhi').
PTR64_SOURCES = ('root', 'vbtab')


@functools.lru_cache(maxsize=None)
def reg_range(tok):
    """'s[8:11]' -> ('s', 8, 11); 's8' -> ('s', 8, 8); 'v[1:3]' -> ('v', 1, 3); 'exec'/'vcc' pairs too."""
    m = re.fullmatch(r'([sv])\[(\d+):(\d+)\]', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r'([sv])(\d+)', tok)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2))
    return None


# Scalar state for the compute execz-block check (Stage.execz_refusals): registers as 's<n>', 'vcc', 'exec', 'm0'
# and 'scc'. Reads may be over-approximated and must-writes under-approximated; both only add refusals.
# The low half of a mask (vcc_lo, exec_lo) is 'vcc' / 'exec', the high half 'vcc_hi' / 'exec_hi', and a 64-bit
# operand both, so a write of one half never counts as writing the other.
SCALAR_TOKEN = re.compile(r'\bs\[(\d+):(\d+)\]|\bs(\d+)\b|\b(vcc|exec)(_lo|_hi)?\b|\b(m0)\b|\bsrc_(scc)\b')
EXEC_KEYS = {'exec', 'exec_hi'}
NO_SCALAR_EFFECT = ('s_waitcnt', 's_nop', 's_inst_prefetch', 's_clause', 's_cbranch_', 's_branch')
# Forms that print no destination: every printed scalar operand is read. v_cmpx writes EXEC implicitly; its e32 and
# SDWA forms print only sources, and the repository decoder prints the e64 form's unused SDST field first.
# s_setpc reads its address; s_movrel* address a register through m0, so their printed operand is not a plain write.
NO_DESTINATION = ('v_cmpx', 's_setpc', 's_movrel', 'v_movrel')
IMPLICIT_VCC_READ = ('v_div_fmas',)
SCC_READERS = ('s_cselect', 's_cmov', 's_addc', 's_subb', 's_cbranch_scc')
READ_MODIFY_WRITE = ('s_addk_i32', 's_mulk_i32', 's_cmov', 's_bitset', 'v_mac', 'v_fmac')
TWO_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Effects outside scalar registers, or control flow the wave takes as a whole: refused inside such a block.
EXECZ_REFUSED = ('s_endpgm', 's_store', 's_atomic', 's_buffer_store', 's_buffer_atomic', 's_barrier', 's_sendmsg',
                 's_setreg', 's_setpc', 's_swappc', 's_dcache', 's_icache', 's_gl1', 's_memtime', 's_memrealtime',
                 's_trap', 's_sleep', 'v_writelane', 's_movrel', 'v_movrel')


def scalar_tokens(text):
    out = set()
    for lo, hi, single, mask, half, m0, scc in SCALAR_TOKEN.findall(text):
        if lo:
            out.update('s%d' % n for n in range(int(lo), int(hi) + 1))
        elif mask:
            out.update({'_lo': {mask}, '_hi': {mask + '_hi'}}.get(half, {mask, mask + '_hi'}))
        else:
            out.add('s' + single if single else m0 or scc)
    return out


def is_exec(regs):
    return regs in ({'exec'}, EXEC_KEYS)


def mask_words(tok, count):
    """The scalar state keys holding a mask operand's words, low word first ('0' for a zero literal), or None when
    the operand is not `count` whole words of scalar state."""
    if tok == '0':
        return ['0'] * count
    m = re.fullmatch(r's\[(\d+):(\d+)\]|s(\d+)|(vcc|exec)(_lo|_hi)?', tok)
    if not m:
        return None
    if m.group(4):
        words = {'_lo': [m.group(4)], '_hi': [m.group(4) + '_hi']}.get(m.group(5), [m.group(4), m.group(4) + '_hi'])
    else:
        words = ['s%d' % n for n in range(int(m.group(1) or m.group(3)), int(m.group(2) or m.group(3)) + 1)]
    return words if len(words) == count else None


@functools.lru_cache(maxsize=None)
def clear_shape(name, asm):
    """What clear_after needs from one instruction's text, which the fixed point would otherwise parse again on every
    visit: 'cmpx', None for a form that carries no fact, or (operation, saving suffix, word count, operand words)."""
    base = re.sub(r'_e(32|64)$', '', name)
    if base.startswith('v_cmpx'):
        return 'cmpx'
    m = re.fullmatch(r's_(mov|and|or|xor|andn2)(_saveexec|_wrexec)?_b(32|64)', base)
    if not m or (m.group(2) and m.group(1) == 'mov'):
        return None
    count = 2 if m.group(3) == '64' else 1
    words = [mask_words(op, count) for op in parse(asm)[1]]
    if None in words or len(words) != (2 if m.group(2) or m.group(1) == 'mov' else 3):
        return None
    return m.group(1), m.group(2), count, tuple(tuple(word) for word in words)


def clear_after(name, asm, may, clear):
    """Forward step of the proof Stage.execz_refusals runs over a block that s_cbranch_execz skips. A fact (key, h)
    says: for every lane of wave half h (0: lanes 0 to 31, 1: lanes 32 to 63) whose exec bit was clear at the branch,
    that lane's bit of the scalar state word `key` is clear. Returns the facts after the instruction, given the
    facts before it.

    Only bitwise forms carry a fact, per word: a zero literal holds it for both halves; D = A & B holds what either
    source holds; A | B and A ^ B what both hold; A & ~B and a move what A holds. A saveexec or wrexec form is its
    operation on exec plus the copy it saves, and v_cmpx ANDs its result into exec. Any other write, a compare
    result included, drops the facts of what it may write."""
    out = {fact for fact in clear if fact[0] not in may - EXEC_KEYS}
    shape = clear_shape(name, asm)
    if shape == 'cmpx':
        return out
    if shape is None:
        return {fact for fact in out if fact[0] not in may}
    op, saving, count, words = shape
    halves = lambda word: {0, 1} if word == '0' else {h for key, h in clear if key == word}
    dest, srcs = words[0], list(words[1:])
    if saving:  # RDNA 2 ISA SOP1: the operation is S0 op EXEC, so andn2 is S0 & ~EXEC
        srcs.append(('exec', 'exec_hi')[:count])
    combine = {'and': set.union, 'or': set.intersection, 'xor': set.intersection}.get(op, lambda a, *b: a)
    value = [combine(*(halves(src[j]) for src in srcs)) for j in range(count)]
    writes = list(zip(dest, value))
    if saving:  # saveexec keeps the old exec in D; wrexec writes the result to both
        writes = list(zip(('exec', 'exec_hi'), value)) + (writes if saving == '_wrexec' else
                                                          [(word, halves(old)) for word, old in zip(dest, srcs[-1])])
    out = {fact for fact in out if fact[0] not in dict(writes)}
    return out | {(word, h) for word, held in writes for h in held}


def scalar_effects(name, asm):
    """(reads, must-writes, may-writes) of scalar state for one decoded instruction."""
    base = re.sub(r'_e(32|64)$', '', name)
    _, ops, _ = parse(asm)
    if base.startswith(NO_SCALAR_EFFECT) or base == 's_endpgm':
        reads = {'scc'} if base.startswith('s_cbranch_scc') else {'vcc'} if base.startswith('s_cbranch_vcc') else \
            {'exec'} if base.startswith('s_cbranch_exec') else set()
        return reads, set(), set()
    if base.startswith('v_cmpx'):
        return scalar_tokens(' '.join(ops)) | {'exec'}, {'exec'}, set(EXEC_KEYS)
    if '_store' in base or base.startswith(('ds_write', 'exp') + NO_DESTINATION) or not ops:
        return scalar_tokens(' '.join(ops)) | ({'m0'} if 'movrel' in base else set()), set(), set()
    if base.startswith(('s_cmp', 's_bitcmp')):
        return scalar_tokens(' '.join(ops)), {'scc'}, {'scc'}
    two = base.startswith(TWO_DESTINATIONS)
    dests, srcs = ops[:2] if two else ops[:1], ops[2:] if two else ops[1:]
    writes = scalar_tokens(' '.join(dests))
    reads = scalar_tokens(' '.join(srcs))
    if base.startswith(READ_MODIFY_WRITE):
        reads |= writes
    if base.startswith(SCC_READERS) or (base in SCA and 'scc' in SCA[base][0]):
        reads.add('scc')
    if base.startswith(IMPLICIT_VCC_READ):
        reads.add('vcc')
    may = set()
    if 'saveexec' in base or 'wrexec' in base:
        reads.add('exec')
        writes.add('exec')
        may |= EXEC_KEYS
    if (base in SCA and SCA[base][1]) or 'saveexec' in base or 'wrexec' in base:
        writes.add('scc')
    return reads, writes, writes | may | ({'scc'} if base.startswith('s_') else set())

# Forms with no scalar destination: their printed operands are all sources (v_cmpx writes only EXEC, which the
# symbolic tracking does not hold).
NO_SCALAR_DESTINATION = ('s_cbranch', 's_branch', 's_waitcnt', 's_nop', 's_endpgm', 's_cmp', 's_bitcmp', 's_setpc',
                         's_barrier', 's_sendmsg', 's_inst_prefetch', 's_clause', 's_setprio', 's_sleep', 's_setreg',
                         's_code_end', 's_dcache', 's_icache', 's_ttrace', 'exp', 'ds_write', 'v_cmpx')
TWO_SCALAR_DESTINATIONS = ('v_add_co', 'v_sub_co', 'v_subrev_co', 'v_div_scale', 'v_mad_u64_u32', 'v_mad_i64_i32')
# Symbols whose register also holds the value at run time: constants are assigned when set, entry root constants are
# assigned at entry when a join needs it, and the others are assigned when tracked.
RUN_TIME_SYMBOLS = ('const', 'ud', 'execcopy', 'fixed_buffer_word')


def run_time_symbol(sym):
    """True when the register also holds the symbol's value at run time (a constant marked unassigned does not)."""
    return sym[0] in RUN_TIME_SYMBOLS and not (sym[0] == 'const' and len(sym) > 2)


def names_one_value(sym):
    """True when two equal symbols stand for the same value, so a join where every path brings it can keep it.
    A modified descriptor word, descnum, a fixed buffer word, a sampler-heap word and an unplaced entry register
    compare equal for different values (what was written, or where it came from, is not in the tuple). An exec copy
    is taken only while exec holds the entry mask, so equal copies hold the same mask."""
    if sym is None:
        return False
    if sym[0] == 'desc':
        # ('desc', source, load offset, dword) and, in a compute stage, the offset of the s_load that read it: the
        # source does not record an addend on the pointer, so only the same load proves the same words.
        return len(sym) == 4 or (len(sym) == 5 and isinstance(sym[4], int))
    # An add on a pointer keeps its symbol and marks it 'added' (the offset is not in the tuple), so only an
    # untouched pointer, exactly (kind, source), names one value.
    return sym[0] in ('const', 'ud', 'execcopy') or (sym[0] in ('ptr', 'ptrhi') and len(sym) == 2)


def scalar_destinations(name, asm):
    """The SGPRs (and 'm0') an instruction may write; 'any' for s_movrel*, whose register comes from m0."""
    if name.startswith(NO_SCALAR_DESTINATION) or '_store' in name:
        return set()
    if 'movrel' in name:
        return {'any'}
    ops = parse(asm)[1]
    out = set()
    for tok in ops[:2] if name.startswith(TWO_SCALAR_DESTINATIONS) else ops[:1]:
        if tok == 'm0':
            out.add('m0')
        r = reg_range(tok)
        if r and r[0] == 's':
            out.update(range(r[1], r[2] + 1))
    return out


def desc_key(sym):
    """Resource key of the descriptor a ('desc', source, load offset, dword) word belongs to: the key descriptor()
    gives its first word, so a write into a later descriptor of a wide load marks that descriptor."""
    return sym[1], None if sym[2] is None else sym[2] + 4 * (sym[3] & ~3)


def desc_word(sym, dword):
    return sym[:3] + (dword,) + sym[4:]


def raw_descriptor_alu(base, syms, constants):
    """Recognize only the word expressions needed for a structured-to-raw buffer store."""
    # gfx10 V# layout in docs/pixel_mubuf_contract.md: stride is DW1[29:16],
    # NUM_RECORDS is DW2 and OOB_SELECT is DW3[29:28].
    # The extracted stride may be clamped to at least 1 (s_max_u32 with 1), so that a descriptor that is
    # already raw (stride 0) keeps NUM_RECORDS bytes.
    if base == 's_max_u32':
        for i, word in enumerate(syms):
            if word and word[:2] == ('descnum', 'stride') and constants[1 - i] == 1:
                return ('descnum', 'stride_min1', word[2])
        return None
    for i, word in enumerate(syms):
        if not word or word[0] != 'desc' or len(word) < 4:
            continue
        other = 1 - i
        field = word[3] % 4
        loaded = 5 if len(word) > 4 and isinstance(word[4], int) else 4  # with the compute load tag
        plain = len(word) == loaded  # a loaded word of one s_load
        if plain and field == 1:
            if base == 's_and_b32' and constants[other] == 0xc000ffff:
                return word + ('stride0',)
            if base == 's_bfe_u32' and i == 0 and constants[1] == 0xe0010:
                return ('descnum', 'stride', word)
        if plain and field == 2 and base == 's_mul_i32':
            if syms[other] in (('descnum', kind, desc_word(word, word[3] - 1)) for kind in ('stride', 'stride_min1')):
                return word + ('byte_count',)
        if field == 3:
            if plain and base == 's_and_b32' and constants[other] == 0xcfffffff:
                return word + ('oob0',)
            if (plain or word[loaded:] == ('oob0',)) and base == 's_or_b32' and constants[other] == 0x30000000:
                return word[:loaded] + ('oob3',)
    return None


def smem_offset(tok):
    """The SMEM offset operand as a token: 'off' (no offset operand) is the literal 0, anything else unchanged."""
    return '0' if tok == 'off' else tok


SGPR_NAMES = {'vcc_lo': (106,), 'vcc_hi': (107,), 'vcc': (106, 107), 'm0': (124,)}
# SALU forms whose first operand is written and not read (D = f(S0, S1)); others that name D first also read it
# (s_addk_i32, s_mulk_i32, s_cmovk_i32, s_cmov_*, s_bitset*, s_movrel*) or do not write it (compares, stores).
SALU_NO_DST = ('s_cmp', 's_bitcmp', 's_cbranch', 's_branch', 's_store', 's_buffer_store', 's_atomic', 's_buffer_atomic',
               's_scratch_store', 's_waitcnt', 's_nop', 's_setpc', 's_setreg', 's_sendmsg', 's_endpgm', 's_clause',
               's_inst_prefetch', 's_dcache', 's_barrier', 's_sleep', 's_set', 's_ttrace', 's_icache', 's_denorm',
               's_round', 's_trap', 's_decperflevel', 's_incperflevel', 's_code_end')
SALU_READS_DST = ('s_addk', 's_mulk', 's_cmovk', 's_cmov_', 's_bitset', 's_movrel', 's_getreg')


def sgpr_access(name, asm):
    """(SGPRs read, SGPRs written) by one instruction, vcc and m0 included, exec and SCC left out.

    Conservative where the form is not listed: a register operand that is not known to be written only counts as
    read, so a caller that refuses on a read never misses one."""
    def regs(tok):
        r = reg_range(tok)
        if r:
            return set(range(r[1], r[2] + 1)) if r[0] == 's' else set()
        return set(SGPR_NAMES.get(tok, ()))
    _, ops, _ = parse(asm)
    reads, writes = set(), set()
    for t in ops:
        reads |= regs(t)
    first = regs(ops[0]) if ops else set()
    base = re.sub(r'_(e32|e64|sdwa|dpp)$', '', name)
    if name.startswith('s_'):
        if not name.startswith(SALU_NO_DST) and not name.startswith(SALU_READS_DST):
            writes, reads = first, reads - first if ops[1:] and not any(regs(t) & first for t in ops[1:]) else reads
    elif (base in ('v_readfirstlane_b32', 'v_readlane_b32') or
          (base.startswith('v_cmp_') and name.endswith(('_e64', '_sdwa')))) and first:
        writes = first
        reads = set().union(*(regs(t) for t in ops[1:])) if ops[1:] else set()
    if name.startswith('v_cmp_') and name.endswith('_e32'):
        writes |= {106}
    if name.startswith('v_') and name.endswith('_e32') and re.search(r'cndmask|addc|subb|_co_|div_fmas', name):
        reads |= {106, 107}
    if name.startswith('s_cbranch_vcc'):
        reads |= {106, 107}
    return reads, writes


def sopk_simm(tok):
    """A SOPK SIMM16 operand sign-extended to 32 bits, as a hex token (upstream #2116, open). Refuses anything that
    is not a 16-bit literal."""
    if not re.fullmatch(r'0x[0-9a-f]{1,4}|\d+', tok) or int(tok, 0) > 0xffff:
        raise Unsupported('SOPK immediate %s' % tok)
    value = int(tok, 0)
    return '0x%08x' % ((value - 0x10000 if value & 0x8000 else value) & 0xffffffff)


def smem_imm_refuse(base, ops, mods):
    """Refuse a negative SMEM immediate offset (upstream #1984 / ed7d09a4). OFFSET[20:0] is signed on RDNA 2:
    disasm_rdna2 prints a negative one as -0x.., an older decoder as 0x100000..0x1fffff; either would otherwise
    be added as a positive byte offset."""
    toks = [ops[2]] if len(ops) > 2 else []
    toks += [m[7:] for m in mods if m.startswith('offset:')]
    for tok in toks:
        if re.fullmatch(r'-?(0[xX][0-9a-fA-F]+|\d+)', tok) and not 0 <= int(tok, 0) <= 0xfffff:
            raise Unsupported('%s: SMEM immediate offset %s' % (base, tok))
def smem_mods(mods):
    """Non-negative 21-bit signed immediate byte offset from SMEM modifiers, refusing unknown or negative forms."""
    imm = None
    for m in mods:
        if m in ('glc', 'dlc'):
            continue
        if m.startswith('offset:') and imm is None and re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', m[7:]):
            # Reject signed negative offsets and older unsigned spellings with bit 20 set.
            token = m[7:]
            value = int(token, 16 if '0x' in token.lower() else 10)
            if not 0 <= value <= 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % m)
            imm = value
        else:
            raise Unsupported('SMEM modifier %s' % m)
    return imm


# Each conversion uses only its input bits (including byte extraction for ubyte0).
# Multiply, fused multiply-add (including the literal form), reciprocal and median of three use
# only local arithmetic. Move copies its input bits. None reads a lane ID or another lane.
UNIFORM_VALU = frozenset((
    'v_cvt_u32_f32', 'v_cvt_i32_f32', 'v_cvt_f32_u32', 'v_cvt_f32_ubyte0', 'v_cvt_off_f32_i4',
    'v_cvt_pk_u16_u32', 'v_cvt_pk_i16_i32',
    'v_mul_f32', 'v_fma_f32', 'v_fmaak_f32', 'v_rcp_f32', 'v_med3_f32', 'v_mov_b32',
))


# DPP quad_perm selections that exchange lanes of a 2x2 pixel quad (lane = 2 * y + x), with the SM 6.0 intrinsic.
QUAD_SWAPS = {(1, 0, 3, 2): 'QuadReadAcrossX', (2, 3, 0, 1): 'QuadReadAcrossY', (3, 2, 1, 0): 'QuadReadAcrossDiagonal'}


# Opcodes whose first operand is a source, not a destination (interp_f16_scan must not read it as a write).
INTERP_STORE_PREFIXES = ('buffer_store', 'tbuffer_store', 'global_store', 'flat_store', 'scratch_store', 'ds_write',
                         'ds_store', 'image_store')


# VALU forms whose second operand is an SGPR destination (VOP3b carry-out or scale flag), not a source.
VOP3B_SDST = re.compile(r'v_(add|sub|subrev)_co(_ci)?_u32|v_div_scale_f(32|64)|v_mad_[iu]64_[iu]32')


def destinations(name, ops):
    """The operands an instruction writes: the first, and the SGPR destination of the VOP3b forms. Compares write none."""
    if not ops or name.startswith(('s_cmp', 's_bitcmp')):
        return []
    return [ops[0]] + ([ops[1]] if len(ops) > 1 and VOP3B_SDST.match(name) else [])


def writes_exec(name, ops):
    """Conservatively: whether the instruction can write EXEC, explicitly or implicitly."""
    for tok in destinations(name, ops):
        r = reg_range(tok)
        if tok in ('exec', 'exec_lo', 'exec_hi') or r and r[0] == 's' and r[1] <= 127 and r[2] >= 126:
            return True
    return 'saveexec' in name or 'wrexec' in name or name.startswith(('v_cmpx', 's_movreld'))


def pc_pair(token):
    """Only aligned pairs of ordinary SGPRs can carry a symbolic code address."""
    r = reg_range(token)
    if r and r[0] == 's' and r[2] == r[1] + 1 <= 105 and r[1] % 2 == 0:
        return r[1]
    return None


def pc_word(token):
    """A displacement word, not a register or an inline floating-point constant."""
    if re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)', token):
        value = int(token, 16 if '0x' in token else 10)
        if -0x80000000 <= value <= 0xffffffff:
            return ((value & 0xffffffff) ^ 0x80000000) - 0x80000000
    return None


def scalar_registers(text):
    """All explicit SGPR words in an operand, including modifiers and ranges."""
    out = set()
    for low, high, single in re.findall(r'\bs(?:\[(\d+):(\d+)\]|(\d+)\b)', text):
        if single:
            out.add(int(single))
        else:
            out.update(range(int(low), int(high) + 1))
    return out


def reads_scc(name, asm):
    # LLVM AMDGPU SOPInstructions.td lists implicit SCC uses. Cover each supported family here
    # (s_cmov includes s_cmovk_i32) so both the calculated-PC and descriptor checks see new readers.
    return 'src_scc' in asm or name.startswith(('s_cbranch_scc', 's_cmov', 's_cselect', 's_addc', 's_subb'))


def calculated_pc_proof(ins, kind):
    """Return offset -> target for proven SETPCs, and offset -> None for their address arithmetic.

    RDNA 2 GETPC returns the address of its successor. A low-word ADD_U32 and a
    high-word ADDC_U32 with its matching sign extension add one signed byte
    displacement, including any carry across the (unknown) loaded PC's low word.
    A completed pair is therefore the shader's load address plus a known byte
    offset, not an integer value that can be emitted to HLSL.

    At a CFG join retain only pair facts that agree on every incoming edge;
    retain *all* potentially address-dependent words and SCCs. A recognized
    address instruction can be elided only if it is recognized on every path.
    No other operation can observe an elided word or SCC. This also checks loops:
    a newly discovered SETPC successor participates in the same worklist.
    Ordinary reads of those SGPRs also need a single reaching non-PC definition
    matching the definition seen in the linear lowering order: a path around an
    elided write may still hold a symbolic root binding, not its zero-initialized
    HLSL local.
    """
    if not any(name == 's_setpc_b64' for _, _, name, _ in ins):
        return {}
    by_off = {off: (n, name, asm) for off, n, name, asm in ins}
    end = ins[-1][0] + ins[-1][1]
    tracked = {reg for _, _, name, asm in ins if name == 's_getpc_b64' for reg in scalar_registers(asm)}
    entry_defs = dict.fromkeys(tracked, -1)

    def refuse(off, reason):
        raise Unsupported('s_setpc_b64: %s at +0x%x' % (reason, off))

    def boundary(off, target):
        if target % 4:
            refuse(off, 'unaligned calculated branch target %#x' % target)
        if not 0 <= target < end:
            refuse(off, 'target %#x outside decoded program' % target)
        if target not in by_off:
            refuse(off, 'target %#x in the interior of an instruction or undecoded bytes' % target)

    if 0 not in by_off:
        refuse(0, 'missing entry instruction')
    if not any(name == 's_endpgm' for _, _, name, _ in ins):
        refuse(0, 'decoded program has no s_endpgm')

    def ordinary_accesses(name, ops, mods, refs):
        # These scalar operations and loads have explicit unconditional destinations.
        # Otherwise conservatively treat a mentioned PC word as a read, and any
        # possible SGPR destination as an invalidation of its reaching definition.
        written = (scalar_registers(ops[0]) if ops and
                   (name in SCA or name.startswith(('s_load_dword', 's_buffer_load_dword'))
                    or name in ('s_cselect_b32', 's_cselect_b64')) else set())
        reads = scalar_registers(', '.join(ops[1:]) + ' ' + ' '.join(mods)) if written else refs
        maybe_written = {reg for dst in destinations(name, ops) for reg in scalar_registers(dst)} - written
        return written, reads, maybe_written

    # Each state is (possibly elided SGPR words, definitely proven pairs, SCC,
    # definitely reaching ordinary definitions of SGPRs the lowering can erase).
    # pairs[low] = (target, None) for a complete pair, or (target, high immediate)
    # for a partial low-word sum. SCC is None for ordinary state, the pair's low
    # register for the corresponding carry, or -1 for an elided/unusable flag.
    incoming = {0: (set(), {}, None, entry_defs)}
    pending = [0]
    lowered = {}

    def propagate(off, words, pairs, scc, definitions):
        state = incoming.get(off)
        if state is None:
            merged = (set(words), dict(pairs), scc, dict(definitions))
        else:
            old_words, old_pairs, old_scc, old_defs = state
            merged = (old_words | words, {p: v for p, v in old_pairs.items() if pairs.get(p) == v},
                      old_scc if old_scc == scc else -1,
                      {reg: d for reg, d in old_defs.items() if definitions.get(reg) == d})
        if state != merged:
            incoming[off] = merged
            pending.append(off)

    while pending:
        off = pending.pop()
        n, name, asm = by_off[off]
        _, ops, mods = parse(asm)
        words, pairs, scc, definitions = incoming[off]
        words, pairs, definitions = set(words), dict(pairs), dict(definitions)
        refs = scalar_registers(asm)
        successors = [off + n]

        def clobber(written):
            nonlocal scc
            words.difference_update(written)
            for reg in written:
                definitions.pop(reg, None)
            for p in list(pairs):
                if p in written or p + 1 in written:
                    del pairs[p]
                    if scc == p:
                        scc = -1

        if name == 's_getpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported GETPC destination/form')
            clobber({p, p + 1})
            words.update((p, p + 1))
            pairs[p] = (off + n, None)
            lowered[off] = None
        elif name in ('s_add_u32', 's_addc_u32') and words & refs:
            row = reg_range(ops[0]) if ops else None
            p = row[1] if row and row[0] == 's' and row[1] == row[2] else None
            imm = pc_word(ops[2]) if len(ops) == 3 else None
            if p is None or imm is None or mods or n not in (4, 8) or ops[0] != ops[1]:
                refuse(off, 'unsupported arithmetic on calculated PC words')
            if name == 's_add_u32':
                fact = pairs.get(p)
                if fact is None or fact[1] is not None:
                    refuse(off, 'unknown or ambiguous input pair for calculated PC low-word addition')
                pairs[p] = (fact[0] + imm, -1 if imm < 0 else 0)
                scc = p
            else:
                fact = pairs.get(p - 1)
                if fact is None or fact[1] is None or scc != p - 1:
                    refuse(off, 'missing matching low-word carry for calculated PC high-word addition')
                if imm != fact[1]:
                    refuse(off, 'calculated PC high word is not the signed displacement extension')
                pairs[p - 1] = (fact[0], None)
                scc = -1
            lowered[off] = None
        elif name == 's_setpc_b64':
            p = pc_pair(ops[0]) if len(ops) == 1 else None
            if p is None or mods or n != 4:
                refuse(off, 'unsupported SETPC source/form')
            fact = pairs.get(p)
            if fact is None:
                refuse(off, 'unknown or ambiguous calculated target from %s' % ops[0])
            target, high = fact
            if high is not None:
                refuse(off, 'incomplete calculated target (missing high-word addition with carry)')
            boundary(off, target)
            lowered[off] = target
            successors = [target]
        else:
            written, reads, maybe_written = ordinary_accesses(name, ops, mods, refs)
            if words & reads or words and name.startswith(('s_movrels', 's_movreld')):
                refuse(off, 'calculated PC word escapes into %s' % name)
            clobber(written)
            definitions.update((reg, off) for reg in written & tracked)
            for reg in maybe_written:
                definitions.pop(reg, None)
            if scc is not None:
                if reads_scc(name, asm):
                    refuse(off, 'calculated PC SCC escapes into %s' % name)
                # These comparisons always emit a replacement SCC (or refuse).
                # Other scalar writers may not model SCC, so they cannot make
                # a tainted flag usable; they also invalidate any pending carry.
                if re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_[iu]32|s_bitcmp[01]_b32', name):
                    scc = None
                elif not name.startswith(('v_', 's_branch', 's_cbranch_', 's_mov_', 's_movk_', 's_load_',
                                          's_buffer_load_', 's_nop', 's_waitcnt', 's_inst_prefetch', 's_clause',
                                          'buffer_', 'tbuffer_', 'image_', 'exp', 'ds_')):
                    scc = -1

            if name in ('s_swappc_b64', 's_call_b64', 's_rfe_b64'):
                # The only pre-existing exception is the vertex entry fetch call.
                if not (name == 's_swappc_b64' and kind == 'vs' and off == 0
                        and ops == ['s[4:5]', 's[4:5]'] and not mods):
                    refuse(off, 'unsupported control transfer %s' % name)
            if name == 's_endpgm':
                successors = []
            elif name == 's_branch' or name.startswith('s_cbranch_'):
                try:
                    target = int(ops[0].split('_')[1], 16)
                except (IndexError, ValueError):
                    refuse(off, 'unsupported direct branch operand')
                boundary(off, target)
                successors = [target] if name == 's_branch' else [target, off + n]

        for successor in successors:
            if successor not in by_off:
                refuse(off, 'missing decoded successor %#x' % successor)
            propagate(successor, words, pairs, scc, definitions)

    for off, _, name, _ in ins:
        if name == 's_setpc_b64' and off not in lowered:
            refuse(off, 'unreachable SETPC has no proven target')

    # Stage's existing resource/constant symbol table is intentionally linear.
    # Do not substitute a symbolic address into it or pretend an erased symbol
    # was a runtime word. Every later ordinary read must see the same *ordinary*
    # definition in both the listing and on all reachable paths. Include writes
    # in unreachable code when tracking the listing because Stage lowers those too.
    linear_defs = dict(entry_defs)
    for off, _, name, asm in ins:
        _, ops, mods = parse(asm)
        if off in lowered:
            if lowered[off] is None:
                for reg in scalar_registers(ops[0]):
                    linear_defs.pop(reg, None)
            continue
        written, reads, maybe_written = ordinary_accesses(name, ops, mods, scalar_registers(asm))
        if off in incoming:
            reaching = incoming[off][3]
            for reg in sorted(reads & tracked):
                if reg not in linear_defs or reaching.get(reg) != linear_defs[reg]:
                    refuse(off, 'elided PC register s%d has no matching ordinary definition on every incoming path' % reg)
        for reg in maybe_written:
            linear_defs.pop(reg, None)
        linear_defs.update((reg, off) for reg in written & tracked)
    return lowered


def _lds_single(tok):
    """Single hardware VGPR vN (v0..v255) for bounded LDS exchange validation."""
    m = re.fullmatch(r'v(\d+)', tok)
    return int(m.group(1)) if m and int(m.group(1)) < 256 else None


def _lds_vgpr_span(tok, count):
    """Exact contiguous `count`-VGPR range v[n:n+count-1] (v0..v255) for bounded wide LDS exchanges."""
    m = re.fullmatch(r'v\[(\d+):(\d+)\]', tok)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo if hi == lo + count - 1 and hi < 256 else None


def _lds_pair(tok):
    """Exact contiguous 2-VGPR pair v[n:n+1] (v0..v255) for bounded 64-bit LDS exchanges."""
    return _lds_vgpr_span(tok, 2)


def _lds_quad(tok):
    """Exact contiguous 4-VGPR quad v[n:n+3] (v0..v255) for bounded 128-bit LDS exchanges."""
    return _lds_vgpr_span(tok, 4)


def lds_byte_offsets(name, mods):
    if name != 'ds_read2_b32':
        if not mods:
            return (0,)
        if len(mods) == 1 and re.fullmatch(r'offset:\d+', mods[0]):
            offset = int(mods[0][7:])
            if offset <= 0xffff:
                return (offset,)
        raise Unsupported('LDS exchange needs two VGPRs and an optional byte offset')
    # DS_READ2 encodes each offset independently in dwords. An omitted field is zero.
    offsets = {}
    for mod in mods:
        field = re.fullmatch(r'offset([01]):(\d+)', mod)
        if not field or field[1] in offsets or int(field[2]) > 255:
            raise Unsupported('LDS read2 needs ordinary dword offsets')
        offsets[field[1]] = 4 * int(field[2])
    first, second = offsets.get('0', 0), offsets.get('1', 0)
    if first == second:
        raise Unsupported('LDS read2 needs two distinct offsets')
    return first, second


# (is_write, byte_width, fill_shift, fill_unit_name)
LDS_EXCHANGE_OPS = {
    'ds_write_b32': (True, 4, '2', 'dword'),
    'ds_read_b32': (False, 4, None, 'dword'),
    'ds_read2_b32': (False, 4, None, 'dword'),
    'ds_write_b64': (True, 8, '3', 'qword'),
    'ds_read_b64': (False, 8, None, 'qword'),
    'ds_write_b128': (True, 16, '4', 'quad-dword'),
    'ds_read_b128': (False, 16, None, 'quad-dword'),
}


def check_lds_exchange(ins, group, words):
    """Prove one race-free 32/64/128-bit fill of a contiguous LDS span, one uniform barrier, then mixed-width reads of initialized dwords.

    RDNA 2 ISA (Ch. 9 DS): ds_write_b32 / ds_read_b32 / ds_read2_b32 access 4-byte dwords,
    ds_write_b64 / ds_read_b64 access two contiguous little-endian dwords (v[n] at addr+offset, v[n+1] at addr+offset+4), and
    ds_write_b128 / ds_read_b128 access four contiguous little-endian dwords (v[n]..v[n+3] at addr+offset..addr+offset+12).
    All widths share the dword-indexed groupshared uint lds[N] array and require 4-byte alignment (natural 8/16-byte alignment is not required on GFX10/RDNA2 LDS).
    LLVM 15 AMDGPUUsage defines COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE in 128-dword (512-byte) units on gfx10.
    """
    if not 0 < words <= 8192 or group[0] < 1 or group[1:] != (1, 1):
        raise Unsupported('LDS exchange needs a 1D group and a nonzero allocation of at most 32 KiB')
    fill_span = None
    published = False
    read = False
    prev = (None, [], [])
    for i, (_, _, _, asm) in enumerate(ins):
        name, ops, mods = parse(asm)
        base = name.removesuffix('_e32').removesuffix('_e64')
        if (base.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc')) or writes_exec(base, ops)
                or base == 's_endpgm' and i + 1 < len(ins)):
            raise Unsupported('LDS exchange needs straight-line code and entry EXEC')
        if base == 's_barrier':
            if (fill_span is None or published or prev != ('s_waitcnt', ['lgkmcnt(0)'], [])
                    or mods or ops not in ([], ['0'], ['0x0'])):
                raise Unsupported('LDS exchange needs one barrier after a completed fill')
            published = True
        elif base in LDS_EXCHANGE_OPS:
            write, width, shift, unit_name = LDS_EXCHANGE_OPS[base]
            dwords = width // 4
            addr_op = ops[0] if (write and len(ops) == 2) else (ops[1] if len(ops) == 2 else '')
            val_op = ops[1] if (write and len(ops) == 2) else (ops[0] if len(ops) == 2 else '')
            if base == 'ds_read2_b32':
                if len(ops) != 2 or _lds_pair(val_op) is None or _lds_single(addr_op) is None:
                    raise Unsupported('LDS read2 needs a VGPR pair and a byte address VGPR')
            elif (len(ops) != 2 or _lds_single(addr_op) is None
                  or (_lds_single(val_op) is None if dwords == 1 else _lds_vgpr_span(val_op, dwords) is None)):
                need_val = ('a matching data VGPR' if dwords == 1
                            else 'an aligned %d-VGPR %s' % (dwords, 'pair' if dwords == 2 else 'quad'))
                raise Unsupported('LDS exchange needs %s, %s and an optional byte offset'
                                  % ('two VGPRs' if dwords == 1 else 'a VGPR address', need_val))
            offsets = lds_byte_offsets(base, mods)
            producer, args, modifiers = prev
            if write:
                offset, = offsets
                # The prefix below preserves the entry X ID in v0. Each lane writes exactly one distinct element.
                # 8-byte / 16-byte alignment is not required for LDS dword-addressed writes.
                if (fill_span is not None or producer != 'v_lshlrev_b32' or args != [addr_op, shift, 'v0']
                        or modifiers or offset % 4 or offset + group[0] * width > words * 4):
                    raise Unsupported('LDS exchange needs one bounded %s per local X ID' % unit_name)
                fill_span = (offset, offset + group[0] * width)
            else:
                if not published or modifiers or not args or args[0] != addr_op:
                    raise Unsupported('LDS read needs a bounded address after the fill barrier')
                if producer == 'v_mov_b32' and len(args) == 2 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1]):
                    low = high = int(args[1], 0)
                elif (producer == 'v_and_b32' and len(args) == 3 and re.fullmatch(r'0x[0-9a-f]+|\d+', args[1])
                      and _lds_single(args[2]) is not None):
                    low, high = 0, int(args[1], 0)
                else:
                    raise Unsupported('LDS read needs a literal or literal-masked byte address')
                fill_start, fill_end = fill_span
                # Every byte offset in [a + offset, a + offset + width) must be 4-byte aligned and inside the filled span.
                # Mixed 32/64/128-bit reads share the same dword-indexed LDS array once all read dwords lie in [fill_start, fill_end).
                if (low % 4 or high % 4
                        or any(offset % 4 or low + offset < fill_start or high + offset + width > fill_end
                               for offset in offsets)):
                    raise Unsupported('LDS read outside the aligned, initialized span')
                if i + 1 >= len(ins) or ins[i + 1][3] != 's_waitcnt lgkmcnt(0)':
                    raise Unsupported('LDS read needs a completed LGKM wait')
                read = True
        elif base.startswith('ds_') or 'lds' in mods:
            raise Unsupported('LDS exchange does not cover ' + asm)
        elif fill_span is None and not base.startswith('s_'):
            # Simple VALU instructions only write their single destination. Other forms, e.g. v_swap_b32,
            # can also change a source. After the fill, each read proves its own address without the entry X ID.
            if (base not in VEC_U and base not in VEC_F and base != 'v_mov_b32'
                    or not ops or not re.fullmatch(r'v[0-9]+', ops[0]) or ops[0] == 'v0'):
                raise Unsupported('LDS fill needs the unchanged entry local X ID')
        prev = base, ops, mods
    if not read:
        raise Unsupported('LDS exchange needs a fill, a barrier and a completed read')


# ---- slot 57 mesh stage: the group-memory count exchange (#2413) --------------------------------
# A group is two wave64 waves and thread = wave * 64 + lane. 128 is also D3D12's limit for the numthreads of a mesh
# shader, and the declared size of both output arrays.
MESH_THREADS = 128
MESH_VREG = re.compile(r'\bv(?:\[(\d+):(\d+)\]|(\d+))')
MESH_EXCHANGE = [
    r's_mov_b64 exec, 1',
    r's_mov_b64 (?P<lane>s\[\d+:\d+\]), exec',
    r's_and_b32 (?P<a_bits>s\d+), s3, 0xf000000',
    r's_cbranch_scc1 label_(?P<a_other>\w+)',
    r'v_lshrrev_b32_e32 v5, 7, v5',
    r'v_mov_b32_e32 (?P<a_zero>v\d+), 0',
    r'ds_write_b32 {a_zero}, v5 offset:8',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_branch label_(?P<a_join>\w+)',
    r's_barrier 0x0',
    r'v_mov_b32_e32 (?P<a_zero2>v\d+), 0',
    r'ds_read_b32 v5, {a_zero2} offset:8',
]
MESH_COUNTS = [
    r's_barrier 0x0',
    r's_cmp_eq_i32 (?P<wave>s\d+), 0',
    r's_cbranch_scc0 label_(?P<b_reader>\w+)',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_wskip>\w+)',
    r'v_mov_b32_e32 (?P<b_p>v\d+), (?P<prims>s\d+)',
    r'v_mov_b32_e32 (?P<b_zero>v\d+), 0',
    r'v_mov_b32_e32 (?P<b_v>v\d+), (?P<verts>s\d+)',
    r'ds_write_b32 {b_zero}, {b_p}',
    r'ds_write_b32 {b_zero}, {b_v} offset:4',
    r's_mov_b64 exec, -1',
    r's_waitcnt_depctr 0xffe3',
    r's_barrier 0x0',
    r's_lshl_b32 (?P<b_tmp>s\d+), {prims}, 12',
    r's_or_b32 m0, {b_tmp}, {verts}',
    r's_sendmsg 0x9',
    r's_branch label_(?P<b_join>\w+)',
    r's_barrier 0x0',
    r's_mov_b64 exec, {lane}',
    r's_cbranch_execz label_(?P<b_rskip>\w+)',
    r'v_mov_b32_e32 (?P<b_zero2>v\d+), 0',
    r'ds_read_b32 (?P<b_r0>v\d+), {b_zero2}',
    r'ds_read_b32 (?P<b_r1>v\d+), {b_zero2} offset:4',
    r's_waitcnt lgkmcnt\(1\)',
    r'v_readfirstlane_b32 {prims}, {b_r0}',
    r's_waitcnt lgkmcnt\(0\)',
    r'v_readfirstlane_b32 {verts}, {b_r1}',
    r's_mov_b64 exec, -1',
]


def mesh_vregs(tokens):
    out = set()
    for m in MESH_VREG.finditer(' '.join(tokens)):
        out.update(range(int(m[1] or m[3]), int(m[2] or m[3]) + 1))
    return out


def mesh_run(ins, at, what, patterns, env):
    """Match consecutive instructions; {name} in a pattern is the text an earlier (?P<name>...) bound."""
    for k, pattern in enumerate(patterns):
        text = ins[at + k][3] if at + k < len(ins) else ''
        m = re.fullmatch(re.sub(r'\{(\w+)\}', lambda ref: re.escape(env[ref[1]]), pattern), text)
        if not m:
            raise Unsupported('slot 57 mesh %s: instruction %d of the accepted shape (%s) is absent'
                              % (what, k + 1, pattern.split()[0]))
        env.update(m.groupdict())
    return at + len(patterns)


MESH_VCC_BRANCHES = ('s_cbranch_vccz', 's_cbranch_vccnz')
VCC_KEYS = {'vcc', 'vcc_hi'}


def mesh_skeleton(ins, vcc_branches, wave_limits):
    """Prove the one accepted group-memory shape of a slot 57 mesh program and replace it by per-thread instructions.

    Accepted shape, in program order (anything else is refused with its own message):
    - Exchange: exec = 1 (lane 0 of each wave), the mask saved in an SGPR pair; wave 0 (s3 bits 24..27 zero) stores
      entry v5 >> 7 at byte 8, waits and passes a barrier; every other wave passes a barrier and reads byte 8 to v5.
    - Code up to the next barrier. Its only read of v5 may be one v_readfirstlane_b32 under exec = -1.
    - Counts: one barrier; the wave whose index (s_bfe_u32 of s3, 4 bits at 24) is zero stores SGPR P at byte 0 and
      SGPR V at byte 4 under the saved lane-0 mask, waits, passes a barrier and sends one allocation request with
      m0 = P << 12 | V; every other wave passes a barrier and reads both words back into the same P and V.
    - Code to s_endpgm. Exports only here: pos0 and param under exec = -1 narrowed by one
      v_cmpx_gt_i32 V, thread; the primitive export likewise with P. thread is the v_mbcnt pair over wave << 6.
    Each path therefore passes three barriers, each store has one writer (one lane of one wave), and a wait and a
    barrier separate it from every read. The lowering keeps one writer and one barrier: thread 0 stores, all read.

    Registers the two arms leave different, that the lowering does not compute, or that hold an entry value the
    model does not define, are refused when read: VGPRs by the scan below, SGPRs by a symbol (setup_entry and the
    op_x2w_mesh_* handlers), SCC and m0 after the counts. Of s3 only the wave index is modelled.
    """
    env = {}
    first = 1 if ins and ins[0][2] == 's_inst_prefetch' else 0
    start = next((i for i in range(first, len(ins)) if ins[i][3] == 's_mov_b64 exec, 1'), first)
    a_end = mesh_run(ins, start, 'group-index exchange', MESH_EXCHANGE, env)
    count_at = next((i for i in range(a_end, len(ins)) if ins[i][2] == 's_barrier'), len(ins))
    b_end = mesh_run(ins, count_at, 'count exchange', MESH_COUNTS, env)

    def lands(key, index, what):
        if index >= len(ins) or int(env[key], 16) != ins[index][0]:
            raise Unsupported('slot 57 mesh %s: a branch leaves the accepted shape' % what)
    lands('a_other', start + 10, 'group-index exchange')
    lands('a_join', a_end, 'group-index exchange')
    lands('b_reader', count_at + 17, 'count exchange')
    lands('b_wskip', count_at + 10, 'count exchange')
    lands('b_join', b_end, 'count exchange')
    lands('b_rskip', count_at + 27, 'count exchange')
    lane = set(range(*(lambda r: (r[1], r[2] + 1))(reg_range(env['lane']))))
    a_bits = int(env['a_bits'][1:])
    sreg = {key: int(env[key][1:]) for key in ('wave', 'prims', 'verts', 'b_tmp')}
    vreg = {key: int(env[key][1:]) for key in ('a_zero', 'a_zero2', 'b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
    if (len({vreg['b_p'], vreg['b_zero'], vreg['b_v']}) != 3 or len({vreg['b_r0'], vreg['b_r1']}) != 2
            or vreg['b_r0'] == vreg['b_zero2'] or 5 in (vreg['a_zero'], vreg['a_zero2'])
            or len({sreg['prims'], sreg['verts'], sreg['b_tmp']}) != 3 or {sreg['prims'], sreg['verts']} & lane
            or sreg['wave'] in lane | {3, sreg['prims'], sreg['verts']} or a_bits in lane | {3} or 3 in lane):
        raise Unsupported('slot 57 mesh count exchange: registers of the accepted shape overlap')
    if not ins or ins[-1][2] != 's_endpgm':
        raise Unsupported('slot 57 mesh program does not end with s_endpgm')

    offsets = {item[0]: i for i, item in enumerate(ins)}
    edges = {}
    for lo, hi in ((0, start), (a_end, count_at), (b_end, len(ins))):
        for i in range(lo, hi):
            off, _, name, asm = ins[i]
            if name.startswith(('s_setpc', 's_call', 's_swappc')):
                raise Unsupported('slot 57 mesh: indirect control flow (%s) is not covered' % name)
            if name == 's_endpgm' and i != len(ins) - 1:
                raise Unsupported('slot 57 mesh: an early s_endpgm is not covered')
            if not name.startswith(('s_cbranch', 's_branch')):
                continue
            target = int(parse(asm)[1][0].split('_')[1], 16)
            dest = offsets.get(target)
            if dest is None:
                raise Unsupported('slot 57 mesh: a branch target is not an instruction boundary')
            # The first instruction of an exchange is also the preceding region's join.
            # Branches must reconverge before that instruction.
            if not lo <= dest < hi:
                raise Unsupported('slot 57 mesh: a branch enters or crosses a group-memory exchange')
            if target <= off:
                raise Unsupported('slot 57 mesh: a loop outside the group-memory exchanges is not covered')
            if name not in ('s_branch', 's_cbranch_scc0', 's_cbranch_scc1', 's_cbranch_execz') + MESH_VCC_BRANCHES:
                raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
            edges[i] = dest

    targets = set(edges.values())
    no_effect = lambda i: ins[i][2].startswith(('s_nop', 's_waitcnt', 's_clause'))
    first_post_count_branch = next((i for i in edges if i >= b_end), None)
    shift_branch = None
    for source, target in edges.items():
        if ins[source][2] == 's_cbranch_execz':
            # A single scalar shift followed by a full EXEC restore has a separate wave-level proof below.
            if (source == first_post_count_branch and source >= b_end + 2 and target == source + 2
                    and ins[source + 1][2] == 's_lshl_b32' and ins[target][3] == 's_mov_b64 exec, -1'):
                shift_branch = source
                continue
            # The model skips per lane, while the guest skips per wave. The difference is invisible when
            # lanes with clear EXEC can only bypass predicated writes before the paths meet. A skipped
            # s_branch may land at the execz target or past it over instructions without effect (the
            # `s_branch .else` over `.empty: s_nop` of an if/else), so both paths meet with the same state.
            for i in range(source + 1, target):
                _, _, name, asm = ins[i]
                if not (no_effect(i) or name == 'exp'
                        or (name == 's_branch' and edges[i] >= target
                            and all(no_effect(k) for k in range(target, edges[i])))
                        or (name.startswith(('v_', 'buffer_load')) and not name.startswith(EXECZ_REFUSED)
                            and not scalar_effects(name, asm)[2])):
                    raise Unsupported('slot 57 mesh: s_cbranch_execz skips code with effects outside active lanes')
        elif any(ins[i][2] == 'exp' for i in range(source + 1, target)):
            raise Unsupported('slot 57 mesh: a branch bypasses an export')

    saved_masks = set().union(*(scalar_registers(parse(asm)[1][0]) for _, _, name, asm in ins
                              if name == 's_and_saveexec_b64'))
    out = list(ins[:first])
    span = lambda first, last: (ins[first][0], ins[last][0] + ins[last][1] - ins[first][0])
    # State of the walk over the code outside the two exchanges.
    stale = set(range(256))   # VGPRs the model does not hold for every lane; no entry VGPR is modelled
    pending = set()           # of those, the ones written only for the lanes of a narrowed exec
    mask = None           # None (all lanes), an output count, or another narrowed mask
    wave_known = False     # sreg['wave'] holds s_bfe_u32(s3, 4 bits at 24)
    thread = base = None   # VGPR of the thread index, SGPR holding wave << 6
    thread_high = None    # second instruction of a proven adjacent v_mbcnt pair
    scc_stale = False
    # Possible last full-EXEC compare offsets. An empty set means some path lacks such a definition.
    # The exchanges preserve VCC. Stage.translate checks the sources against its SGPR facts at each definition.
    vcc_compares = frozenset()
    exports = []
    incoming = {}
    falls = True
    for index in (list(range(first, start)) + [start] + list(range(a_end, count_at)) + [None]
                  + list(range(b_end, len(ins)))):
        if index in targets:
            paths = incoming[index]
            if falls:
                paths.append((stale, pending, mask, wave_known, thread, base, scc_stale, vcc_compares))
            stale = set().union(*(p[0] for p in paths))
            pending = set().union(*(p[1] for p in paths))
            mask = paths[0][2]
            if any(p[2] != mask for p in paths):
                stale |= pending
                pending.clear()
                mask = 'unknown'
            wave_known = all(p[3] for p in paths)
            thread, base = (paths[0][k] if all(p[k] == paths[0][k] for p in paths) else None for k in (4, 5))
            scc_stale = any(p[6] for p in paths)
            definitions = [p[7] for p in paths]
            vcc_compares = frozenset().union(*definitions) if all(definitions) else frozenset()
            falls = True
        if not falls:
            raise Unsupported('slot 57 mesh: unreachable code after a branch is not covered')
        if index == start:
            out.append(span(start, start + 1) + ('x2w_mesh_lane0', 'x2w_mesh_lane0 ' + env['lane']))
            out.append(ins[start + 2])
            out.append(span(start + 3, a_end - 1) + ('s_nop', 's_nop 0'))
            stale |= pending | {5, vreg['a_zero'], vreg['a_zero2']}
            pending.clear()
            mask, thread = 'lane0', None
            wave_known = wave_known and sreg['wave'] != a_bits
            if base == a_bits or base in lane:
                base = None
            continue
        if index is None:
            if not wave_known:
                raise Unsupported('slot 57 mesh count exchange: the compared register is not the wave index')
            out.append(span(count_at, b_end - 2) + (
                'x2w_mesh_counts', 'x2w_mesh_counts %s, %s, %s' % (env['prims'], env['verts'], env['b_tmp'])))
            out.append(ins[b_end - 1])
            stale |= pending | {vreg[k] for k in ('b_p', 'b_zero', 'b_v', 'b_zero2', 'b_r0', 'b_r1')}
            pending.clear()
            mask, scc_stale = None, True
            wave_known = wave_known and sreg['wave'] != sreg['b_tmp']
            if base in {sreg['prims'], sreg['verts'], sreg['b_tmp']}:
                base = None
            continue
        off, size, name, asm = ins[index]
        after = index >= b_end
        _, ops, _ = parse(asm)
        base_name = re.sub(r'_e(32|64)$', '', name)
        complement = (name == 's_andn2_b64' and type(mask) is int and len(ops) == 3
                      and ops == ['exec', parse(ins[mask][3])[1][0], 'exec'])
        if name != 's_and_saveexec_b64' and not complement and scalar_registers(asm) & saved_masks:
            raise Unsupported('slot 57 mesh: an unproven use of a saved EXEC mask is not covered')
        if index in edges:
            if scc_stale and reads_scc(name, asm):
                raise Unsupported('slot 57 mesh: an unmodelled SCC is read by a branch')
            if index == shift_branch:
                # Under full EXEC the preceding thread < limit comparison selects a contiguous prefix.
                # A guest wave64 has any active lane exactly when its first thread is below the limit.
                cmp = re.fullmatch(r'v_cmp_gt_u32_e32 vcc_lo, (0x[0-9a-f]+|[0-9]+), v(\d+)', ins[index - 2][3])
                if mask != index - 1 or not cmp or int(cmp[2]) != thread:
                    raise Unsupported('slot 57 mesh: scalar shift needs a proven wave EXEC predicate')
                wave_limits[off] = int(cmp[1], 0)
            if name in MESH_VCC_BRANCHES:
                if mask is not None or not vcc_compares:
                    raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
                vcc_branches[off] = vcc_compares
            incoming.setdefault(edges[index], []).append(
                (stale.copy(), pending.copy(), mask, wave_known, thread, base, scc_stale, vcc_compares))
            out.append(ins[index])
            falls = name != 's_branch'
            continue
        if name.startswith('ds_') or name in ('s_barrier', 's_sendmsg', 's_sendmsghalt'):
            raise Unsupported('slot 57 mesh: %s outside the group-memory exchanges' % name)
        if name.startswith(('v_readlane', 'v_writelane')):
            raise Unsupported('slot 57 mesh: lane spills are not covered')
        keep = ins[index]
        if base_name == 'v_readfirstlane_b32':
            if (not a_end <= index < count_at or asm != 'v_readfirstlane_b32 %s, v5' % ops[0]
                    or mask is not None or 5 not in stale):
                raise Unsupported('slot 57 mesh: v_readfirstlane_b32 outside the group-index read is not covered')
            keep = (off, size, 'x2w_mesh_group', 'x2w_mesh_group ' + ops[0])
            ops = ops[:1]
        writes_s = set()
        if not base_name.startswith(('v_cmpx', 'exp', 'ds_write', 'buffer_store')):
            for tok in destinations(name, ops):
                r = reg_range(tok)
                if r and r[0] == 's':
                    writes_s.update(range(r[1], r[2] + 1))
        if 3 in writes_s or writes_s & lane:
            raise Unsupported('slot 57 mesh: the wave word s3 or the saved lane-0 mask is written')
        if writes_s and name.startswith('v_') and keep[2] != 'x2w_mesh_group':
            raise Unsupported('slot 57 mesh: %s into an SGPR is not covered' % base_name)
        # Only the wave index of s3 is modelled.
        if not re.fullmatch(r's_bfe_u32 s\d+, s3, 0x40018', asm) and any(
                r and r[0] == 's' and r[1] <= 3 <= r[2] for r in map(reg_range, ops)):
            raise Unsupported('slot 57 mesh: s3 is read other than as the wave index')
        if after and writes_s & {sreg['prims'], sreg['verts']}:
            raise Unsupported('slot 57 mesh: an output count is written after the count exchange')
        if after and re.search(r'\bm0\b', asm):
            raise Unsupported('slot 57 mesh: m0 is used after the allocation request')
        if scc_stale and reads_scc(name, asm):
            raise Unsupported('slot 57 mesh: SCC of the count exchange is read')
        if name.startswith(('s_cmp', 's_bitcmp')):
            scc_stale = False
        if scalar_effects(name, asm)[2] & VCC_KEYS:
            vcc_compares = frozenset({off}) if (
                mask is None and base_name.startswith('v_cmp_') and name.endswith(('_e32', '_e64'))
                and len(ops) == 3 and ops[0] in ('vcc', 'vcc_lo')
            ) else frozenset()
        # Saveexec uses a fresh VCC from an adjacent compare. Only the matching complement can read the
        # saved mask. Hardware SCC (whether any bit is set) is not represented by a per-lane boolean.
        if asm == 's_mov_b64 exec, -1':
            mask = None
            stale |= pending
            pending.clear()
        elif (name == 's_and_saveexec_b64' and mask is None and index > 0 and index not in targets
              and re.fullmatch(r's_and_saveexec_b64 s\[\d+:\d+\], vcc', asm)
              and ins[index - 1][2].startswith('v_cmp_')
              and parse(ins[index - 1][3])[1][:1] in (['vcc'], ['vcc_lo'])):
            mask, scc_stale = index, True
        elif complement:
            stale |= pending
            pending.clear()
            mask, scc_stale = ('else', mask), True
        elif base_name.startswith('v_cmpx'):
            count = {env['verts']: 'verts', env['prims']: 'prims'}.get(ops[0] if ops else None)
            if (base_name != 'v_cmpx_gt_i32' or not after or mask is not None or count is None
                    or thread is None or ops[1:] != ['v%d' % thread]):
                raise Unsupported('slot 57 mesh: an EXEC compare other than output count > thread index under '
                                  'exec = -1 is not covered')
            mask = count
        elif writes_exec(name, ops) or re.search(r'\b(?:exec(?:_lo|_hi)?|src_execz|src_vccz)\b', asm) or (
                name.startswith('s_') and re.search(r'\bvcc(_lo|_hi)?\b', asm)):
            raise Unsupported('slot 57 mesh: %s on a wave mask is not covered' % base_name)
        # The thread index: v_mbcnt_lo(-1, wave << 6) then v_mbcnt_hi(-1, that) under exec = -1.
        if asm == 's_bfe_u32 %s, s3, 0x40018' % env['wave']:
            wave_known = True
        elif sreg['wave'] in writes_s:
            wave_known = False
        if wave_known and re.fullmatch(r's_lshl_b32 s\d+, %s, 6' % env['wave'], asm) and ops[0] != env['wave']:
            base = int(ops[0][1:])
        elif base in writes_s:
            base = None
        reads_v = mesh_vregs(ops if base_name.startswith(('v_cmp', 'exp', 'ds_write', 'buffer_store')) else ops[1:])
        dest = reg_range(ops[0]) if ops and not base_name.startswith(('v_cmp', 'exp', 'buffer_store')) else None
        writes_v = set(range(dest[1], dest[2] + 1)) if dest and dest[0] == 'v' else set()
        if base_name == 'v_mbcnt_lo_u32_b32' and base is not None and ops[1:] == ['-1', 's%d' % base] and mask is None \
                and index + 1 not in targets \
                and index + 1 < len(ins) and ins[index + 1][3] == 'v_mbcnt_hi_u32_b32 %s, -1, %s' % (ops[0], ops[0]):
            thread = dest[1]
            thread_high = index + 1
        elif thread in writes_v and index != thread_high:
            raise Unsupported('slot 57 mesh: the thread index register is rewritten')
        if keep[2] != 'x2w_mesh_group' and reads_v & stale:
            raise Unsupported('slot 57 mesh: v%d is read where it is not defined for every lane (an entry value, or '
                              'left different per wave by an exchange)' % min(reads_v & stale))
        if mask is not None:
            pending |= writes_v & stale
        else:
            pending -= writes_v
        stale -= writes_v
        if name == 'exp':
            target = asm.split()[1].rstrip(',')
            want = 'prims' if target == 'target_20' else 'verts' if re.fullmatch(r'pos0|param\d+', target) else None
            if want is None:
                raise Unsupported('slot 57 mesh: export target %s is not covered' % target)
            if not after or mask != want:
                raise Unsupported('slot 57 mesh: export %s is not under exec = -1 narrowed by its output count > '
                                  'thread index' % target)
            exports.append(target)
        out.append(keep)
    if exports.count('pos0') != 1 or exports.count('target_20') != 1:
        raise Unsupported('slot 57 mesh: one pos0 and one primitive export are required')
    return out


class Stage:
    """Translates one stage. Shares the resource and entry-constant registries of its pipeline."""
    uses_q = False
    uses_swap = False
    uses_xsave = False

    # Immutable fallbacks for unit probes that bypass __init__; __init__ and translate() assign fresh
    # per-instance sets, so nothing is shared across Stage instances.
    needs_sm6 = False         # the lowering uses a shader model 6.0 intrinsic, so the stage cannot compile as 5.1
    exec_known = False        # unit probes without __init__ customize this explicitly when needed
    low16_only_v = ()         # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
    pinned_low16_v = ()       # low16_only_v live across basic-block boundaries
    whole_read_v = ()         # VGPRs read outside a proven low-16-bit consumer
    used_atomic = ()          # atomic result types ('uint', 'int') needing declarations in main()
    # pixel: VGPR -> (attr, channel, high) of the v_interp_p1ll_f16 that wrote it. Created on first use, because
    # unit probes build a Stage without __init__.
    interp_p1 = None
    # The whole-program proof of interp_f16_scan depends only on the decoded program, so it runs once per Stage
    # (a class default keeps probes built without __init__ working).
    interp_scanned = False
    defined_uniform_s = None
    bad_sgprs = None
    _uniform_written = None
    raw_code = b''

    def gather_2d_setup(self, name, ops, mods, addr_count, addr_label, *, mip_preclamp):
        if self.kind not in ('ps', 'cs') or len(ops) != 4:
            raise Unsupported('%s requires a pixel or compute stage and four operands' % name)
        for modifier in mods:
            if not (modifier.startswith(('dim:', 'dmask:')) or modifier in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, modifier))
        if [m for m in mods if m.startswith('dim:')] != ['dim:SQ_RSRC_IMG_2D']:
            raise Unsupported('%s requires a 2D texture' % name)
        channels = {'dmask:0x1': ('Red', 0), 'dmask:0x2': ('Green', 1), 'dmask:0x4': ('Blue', 2), 'dmask:0x8': ('Alpha', 3)}
        masks = [m for m in mods if m.startswith('dmask:')]
        if len(masks) != 1 or masks[0] not in channels:
            raise Unsupported('%s requires a one-hot dmask' % name)
        dst, addr, sampler = (reg_range(ops[i]) for i in (0, 1, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
        if not addr or addr[0] != 'v' or addr[2] - addr[1] + 1 != addr_count or addr[2] > 255:
            raise Unsupported('%s requires %s' % (name, addr_label))
        words = self.const_words(ops[3]) if sampler and sampler[0] == 's' and sampler[2] - sampler[1] == 3 else None
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        profiled = gather_2d_profile_words(words)
        if not mip_preclamp and (words[2] & (1 << 28)):
            profiled = (profiled[0], profiled[1], profiled[2] | (1 << 28), profiled[3])
        try:
            # _lz reads mip zero of the bound view; _l checks all other sampler fields against
            # the 2D gather profile and enforces words[1] MAX_LOD plus view mip bounds in HLSL.
            check_gather_sampler(profiled, 1, comparison=False)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        key = self.fixed_table_image(ops[2], name) if self.kind == 'cs' else self.descriptor(ops[2])
        tex = self.pipe.resource(key, 'tex2d')
        return dst, addr, words, channels[masks[0]], tex

    def carry_in_out_op(self, ops, mods, kind):
        """v_{add,sub,subrev}_co_ci_u32 D, SDST, S0, S1, S2: 32-bit add/sub with carry or borrow in and out.

        AMD RDNA 2 ISA: V_ADD_CO_CI_U32 D = S0 + S1 + S2[lane] with carry out to SDST[lane];
        V_SUB_CO_CI_U32 D = S0 - S1 - S2[lane] with borrow out to SDST[lane]; V_SUBREV_CO_CI_U32 swaps S0 and S1.
        Both VOP2 (_e32, implicit vcc_lo) and VOP3b (_e64) forms arrive with five operands. Results go through
        scratch a and w first so D or SDST may overlap any source.
        """
        name = 'v_%s_co_ci_u32' % kind
        if mods:
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        if len(ops) != 5:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[1] != d[2] or d[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        sdst = reg_range(ops[1])
        if ops[1] not in ('vcc', 'vcc_lo') and not (sdst and sdst[0] == 's' and sdst[1] == sdst[2] and sdst[1] <= 105):
            raise Unsupported('%s carry-out destination %s' % (name, ops[1]))
        cin_tok = ops[4]
        if cin_tok in ('vcc_hi', 'exec_hi'):
            raise Unsupported('%s carry-in %s' % (name, cin_tok))
        cin_const = self.const_u32(cin_tok)
        if cin_const in (0, 0xffffffff):
            ci = '1u' if cin_const else '0u'
        elif self.is_lane_mask(cin_tok):
            ci = '(%s & 1u)' % self.u(cin_tok)
        else:
            raise Unsupported('%s carry-in %s' % (name, cin_tok))
        a, b = self.u(ops[2]), self.u(ops[3])
        if kind == 'subrev':
            a, b = b, a
        if kind == 'add':
            self.emit('a = %s + %s;' % (a, b))
            self.emit('w = a < %s ? 1u : 0u;' % a)
            self.emit('a = a + %s;' % ci)
            self.emit('w = w | (a < %s ? 1u : 0u);' % ci)
        else:
            self.emit('a = %s - %s;' % (a, b))
            self.emit('w = %s > %s ? 1u : 0u;' % (b, a))
            self.emit('w = w | (%s > a ? 1u : 0u);' % ci)
            self.emit('a = a - %s;' % ci)
        self.set_v(d[1], 'a')
        predicate = 'w' if self.exec_known else '(exec & w)'
        if ops[1] in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % predicate)
        else:
            self.set_s(sdst[1], predicate, uniform=False)
            self.masks.add(sdst[1])

    def op_v_cubetc_f32(self, ops, mods):
        self.cube('tc', ops, mods)

    def op_v_cubema_f32(self, ops, mods):
        self.cube('ma', ops, mods)

    def check_compare_sdwa_destination(self, base, writes_exec, ops):
        # v_cmpx_*_sdwa prints just its sources. v_cmp_*_sdwa prints the scalar destination first.
        if len(ops) != (2 if writes_exec else 3):
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        if not writes_exec and ops[0] not in ('vcc', 'vcc_lo', 'exec', 'exec_lo') and not (
                reg_range(ops[0]) and reg_range(ops[0])[0] == 's'):
            raise Unsupported('%s into %s' % (base, ops[0]))

    def lds_atomic(self, name, ops, mods):
        """Lower 32-bit unsigned/bitwise compute LDS atomics (non-rtn and _rtn forms) onto groupshared uint lds[N]."""
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported(name)
        method, has_rtn = LDS_ATOMIC_OPS[name]
        if has_rtn:
            if (len(ops) != 3 or _lds_single(ops[0]) is None
                    or _lds_single(ops[1]) is None or _lds_single(ops[2]) is None):
                raise Unsupported('%s needs three single VGPRs (dst, addr, data)' % name)
            dst, addr_op, data_op = _lds_single(ops[0]), ops[1], ops[2]
        else:
            if len(ops) != 2 or _lds_single(ops[0]) is None or _lds_single(ops[1]) is None:
                raise Unsupported('%s needs two single VGPRs (addr, data)' % name)
            dst, addr_op, data_op = None, ops[0], ops[1]
        # Reuse ds_write_b32 single-offset validation (optional 16-bit byte offset, no gds/unknown mods).
        off, = lds_byte_offsets('ds_write_b32', mods)
        if off % 4:
            raise Unsupported('%s requires 4-byte aligned immediate offset, got %d' % (name, off))
        lds_bytes = self._lds_require_alloc(name)
        addr = self.u(addr_op)
        data = self.u(data_op)
        self.emit('a = %s;' % addr)
        exec_prefix = '' if self.exec_known else '(exec & 1) && '
        if off < lds_bytes:
            if has_rtn:
                if not isinstance(getattr(self, 'used_atomic', None), set):
                    self.used_atomic = set()
                self.used_atomic.add('uint')
                self.emit('ao = 0u;')
                self.emit('if (%s((a & 3u) == 0u) && a < %du) Interlocked%s(lds[(a + %du) >> 2], %s, ao);'
                          % (exec_prefix, lds_bytes - off, method, off, data))
                self.set_v(dst, 'ao')
            else:
                self.emit('if (%s((a & 3u) == 0u) && a < %du) Interlocked%s(lds[(a + %du) >> 2], %s);'
                          % (exec_prefix, lds_bytes - off, method, off, data))
        elif has_rtn:
            self.set_v(dst, '0u')

    def op_v_cubesc_f32(self, ops, mods):
        self.cube('sc', ops, mods)

    def op_image_gather4_l(self, ops, mods):
        """image_gather4_l on a 2D texture with explicit LOD and a validated constant sampler."""
        dst, addr, words, (_, channel_index), tex = self.gather_2d_setup(
            'image_gather4_l', ops, mods, 3, 'three contiguous address registers', mip_preclamp=False)
        self.pipe.uses_gather_2d_lod = True
        max_lod = ((words[1] >> 12) & 0xfff) / 256.0
        u, v, lod = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        self.emit('t = gather_2d_lod(%s, float2(%s, %s), %s, %d, %sf);' % (tex, u, v, lod, channel_index, repr(max_lod)))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def compare_sdwa_sources(self, base, typ, ops, mods, *, float_src0=False, float_sources=False):
        fields = {}
        for m in mods:
            key, _, value = m.partition(':')
            if key not in ('src0_sel', 'src1_sel') or value not in SDWA_SEL or key in fields:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
            fields[key] = value
        sources = []
        for index, token in enumerate(ops):
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            sext = token.startswith('sext(') and token.endswith(')')
            if float_src0 and index == 0:
                if selection != 'DWORD' or sext:
                    raise Unsupported('%s requires a DWORD float source' % base)
                sources.append(self.f(token))
                continue
            if float_sources:
                if 'sext(' in token:
                    raise Unsupported('%s: sext() on a float source' % base)
                if selection != 'DWORD':
                    raise Unsupported('%s: f32 SDWA byte/word selection depends on the binary32 denormal mode'
                                      % base)
                # The whole word: the same text as the e64 form with the same modifiers.
                sources.append(self.f(token))
                continue
            if sext:
                token = token[len('sext('):-1]
            word = sdwa_source(self.u(token, low16_only=selection in ('BYTE_0', 'BYTE_1', 'WORD_0')), selection, sext)
            sources.append('int(%s)' % word if typ == 'i32' else word)
        return sources

    def op_s_bfe_i64(self, ops, mods):
        return self.scalar_wide_op('s_bfe_i64', ops, mods, True, True, 'bfe64', True)

    def joined_registers(self, with_cfg=False):
        """{branch target: registers whose program-order symbol may not hold on every path into it}.

        Program order carries the state after every instruction above the target into it. That state holds on every
        path only for a register whose writes above the target are all in blocks that dominate it (in dominance
        order), and which no block that reaches it from at or below it writes (a loop body, or a later block that
        branches back). Computed over the basic blocks, so a path that leaves a loop's textual range and comes back
        counts.

        With ``with_cfg=True``, returns ``(joined, forward_splits, div_targets)`` where ``forward_splits`` keeps only
        writes on non-dominating forward arms between a target's dominators and the target (excluding loop back edges)
        and ``div_targets`` maps each ``s_cbranch_*`` offset to ``(writes, affected_targets)`` inside its divergent
        CFG region (between the branch and its immediate post-dominator, or over the loop range for a backward edge)."""
        ins = self.ins
        if not ins:
            return ({}, {}, {}) if with_cfg else {}
        index = {off: i for i, (off, *_) in enumerate(ins)}
        calculated = getattr(self, 'calculated_pc', {})  # unit probes build a Stage without translate()
        branch = lambda name: name.startswith('s_cbranch') or name in ('s_branch', 's_setpc_b64')
        target_of = lambda off, name, asm: (calculated.get(off) if name == 's_setpc_b64'  # None: no proven target
                                            else int(parse(asm)[1][0].split('_')[1], 16))
        leaders = {0} | {target_of(off, name, asm) for off, _, name, asm in ins if branch(name)} - {None}
        for i, (off, n, name, asm) in enumerate(ins[:-1]):
            if branch(name) or name == 's_endpgm':
                leaders.add(ins[i + 1][0])
        starts = sorted(index[o] for o in leaders if o in index)
        bounds = list(zip(starts, starts[1:] + [len(ins)]))
        off_of = [ins[lo][0] for lo, _ in bounds]
        block_of = {}
        for b, (lo, hi) in enumerate(bounds):
            for i in range(lo, hi):
                block_of[i] = b
        succ = []
        for lo, hi in bounds:
            off, n, name, asm = ins[hi - 1]
            out = [] if name in ('s_branch', 's_setpc_b64', 's_endpgm') or hi == len(ins) else [block_of[hi]]
            if branch(name) and target_of(off, name, asm) in index:
                out.append(block_of[index[target_of(off, name, asm)]])
            succ.append(out)
        pred = [[a for a in range(len(bounds)) if b in succ[a]] for b in range(len(bounds))]
        writes = [set().union(*(scalar_destinations(*ins[i][2:]) for i in range(lo, hi))) for lo, hi in bounds]
        everything = set(range(106)) | {'m0'}
        writes = [everything if 'any' in w else w for w in writes]
        # Dominators of each block (iterative; the entry block is block 0).
        dom = [set(range(len(bounds))) for _ in bounds]
        dom[0] = {0}
        changed = True
        while changed:
            changed = False
            for b in range(1, len(bounds)):
                new = ({b} | set.intersection(*(dom[p] for p in pred[b]))) if pred[b] else {b}
                if new != dom[b]:
                    dom[b], changed = new, True
        all_targets = leaders - {0} | ({0} if any(0 in s for s in succ) else set())
        joined, forward_splits = {}, {}
        for target in all_targets:
            if target not in index:
                continue
            t = block_of[index[target]]
            reach, todo = set(), list(pred[t])  # blocks with a path of length >= 1 into the target
            while todo:
                b = todo.pop()
                if b not in reach:
                    reach.add(b)
                    todo += pred[b]
            regs = set()
            above_dominators = sorted(b for b in dom[t] - {t} if b < t)
            for b in range(len(bounds)):
                if b < t and b not in dom[t]:
                    regs |= writes[b]  # above the target but not on every path into it
                if b in reach and b >= t:
                    regs |= writes[b]  # a loop body or a later block that comes back
            # Writes in dominating blocks hold on every path only if their textual order is their execution order.
            non_chain = any(not (a in dom[c]) for a, c in zip(above_dominators, above_dominators[1:]))
            if non_chain:
                regs |= set().union(*(writes[b] for b in above_dominators))
            joined[target] = regs
            if with_cfg:
                above_set = dom[t] - {t}
                split_blks, stodo = set(), [p for p in pred[t] if p < t and p not in above_set]
                while stodo:
                    b = stodo.pop()
                    if b not in split_blks:
                        split_blks.add(b)
                        stodo += [p for p in pred[b] if p < t and p not in above_set]
                fwd = set().union(*(writes[b] for b in split_blks)) if split_blks else set()
                if non_chain:
                    fwd |= set().union(*(writes[b] for b in above_dominators))
                forward_splits[target] = fwd
        if not with_cfg:
            return joined
        pdom = [set(range(len(bounds))) if succ[b] else {b} for b in range(len(bounds))]
        changed = True
        while changed:
            changed = False
            for b in range(len(bounds) - 1, -1, -1):
                if not succ[b]:
                    continue
                new = {b} | set.intersection(*(pdom[s] for s in succ[b]))
                if new != pdom[b]:
                    pdom[b], changed = new, True
        div_targets = {}
        for b, (lo, hi) in enumerate(bounds):
            off, _n, name, asm = ins[hi - 1]
            if not name.startswith('s_cbranch'):
                continue
            tgt = target_of(off, name, asm)
            if tgt is None:
                continue
            if tgt <= off:
                t_blk = block_of[index[tgt]] if tgt in index else b
                loop_blks = range(min(t_blk, b), b + 1)
                regs = set().union(*(writes[w] for w in loop_blks))
                affected = {off_of[w] for w in loop_blks} & all_targets
            else:
                stop = pdom[b] - {b}
                ipd = max(stop, key=lambda x: len(pdom[x])) if stop else None
                div, dtodo = set(), list(succ[b])
                while dtodo:
                    w = dtodo.pop()
                    if w == ipd or w in div:
                        continue
                    div.add(w)
                    dtodo += succ[w]
                regs = set().union(*(writes[w] for w in div)) if div else set()
                affected = ({off_of[w] for w in div} | ({off_of[ipd]} if ipd is not None else set())) & all_targets
            div_targets[off] = (regs, affected)
        self.flow = (index, block_of, succ, dom, writes, {})
        return joined, forward_splits, div_targets

    def _lds_general_barrier_ok(self):
        """True when a compute stage uses dual/atomic LDS ops in straight-line code without legacy exchange ops."""
        ins = getattr(self, 'ins', None)
        if not ins:
            return False
        has_general = False
        for i, (_, _, name, _) in enumerate(ins):
            op = re.sub(r'_e(32|64)$', '', name)
            if op in LEGACY_LDS_EXCHANGE_OPS:
                return False
            if op in LDS_DUAL_ATOMIC_OPS:
                has_general = True
            if op.startswith(('s_branch', 's_cbranch', 's_setpc', 's_swappc')) or (op == 's_endpgm' and i + 1 < len(ins)):
                return False
        return has_general

    def op_s_ashr_i64(self, ops, mods):
        return self.scalar_wide_op('s_ashr_i64', ops, mods, True, True, 'shr64i')

    def op_buffer_store_vs(self, ops, mods, count):
        # Vertex-stage store in the `off` form: scalar byte offset plus the 12-bit immediate, through a
        # descriptor read from a fixed root-table slot, written to a RWByteAddressBuffer (issue #1723).
        # The manifest carries the `uav` line the runtime binds; a descriptor may not also be loaded.
        if self.kind != 'vs':
            raise Unsupported('buffer store in the %s stage' % self.kind)
        offset = 0
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            else:
                raise Unsupported('buffer store modifier %s' % m)
        if len(ops) != 4 or ops[1] != 'off':
            raise Unsupported('buffer store without the off address form')
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[2] - src[1] != count - 1 or src[2] > 255:
            raise Unsupported('buffer store data %s' % ops[0])
        key = self.descriptor(ops[2])
        if key[0][0] != 'table' or key[1] is None:
            raise Unsupported('buffer store through %s' % (key,))
        if key in self.pipe.modified:
            # Only the modifications a compute store accepts: the stride clear or the structured-to-raw conversion.
            self.store_descriptor(ops[2], 4, 'buffer store', stride0_ok=True, raw_conversion_ok=True)
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            offset += int(ops[3], 0)
            soffset = ''
            if offset % 4:
                raise Unsupported('buffer store at unaligned byte offset %d' % offset)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            if offset % 4:
                raise Unsupported('buffer store at unaligned immediate offset %d' % offset)
            soffset = '%s + ' % self.S(sreg[1])
        uav = self.pipe.uav(key, 'raw')
        words = ', '.join(self.u('v%d' % n) for n in range(src[1], src[2] + 1))
        store = '%s.Store%s(a, %s);' % (uav, '' if count == 1 else str(count), 'uint%d(%s)' % (count, words) if count > 1 else words)
        # Store* needs a multiple of four: the immediate is checked above; a register offset is rounded down.
        self.emit('a = (%s%d) & 0xfffffffcu;' % (soffset, offset) if soffset else 'a = %d;' % offset)
        self.emit(store if self.exec_known else 'if (exec & 1) %s' % store)

    def interp_f16_scan(self):
        """Whole-program proof for the f16 interpolation pair (no fixed-point analysis: any doubt is a refusal).

        The lowering reads the rasterizer's interpolant, so it is only faithful if (1) the entry I/J registers are
        never written (they still hold the entry barycentrics wherever the pair reads them), (2) v_interp_p1ll_f16,
        which emits nothing, has no consumer but the v_interp_p2_f16 that follows it in the same straight-line run
        and nothing reads the destination it leaves stale when p2 writes elsewhere, and (3) nothing reads the
        upper 16 bits of a plain p2 destination: whether gfx10 keeps or clears them is not established by a source
        (LLVM 15 SIInstrInfo/VOP3Instructions.td model the result as f16 only; the encoding states no
        high-half behaviour), so only a plain v_cvt_f32_f16 reading the low half may consume it, until a p2 with
        op_sel 0x8 writes the high half of that register (Mesa ACO keeps the other half, see check_encoding).
        """
        if self.interp_scanned:
            return
        pair = None
        targets = set()
        rows = []
        for off, _, name, asm in self.ins:
            n, ops, mods = parse(asm)
            n = re.sub(r'_e(32|64)$', '', n)
            rows.append((off, n, ops, mods))
            target = getattr(self, 'calculated_pc', {}).get(off)
            if n.startswith('s_cbranch') or n == 's_branch':
                target = int(ops[0].split('_')[1], 16)
            if target is not None:
                if target <= off:
                    raise Unsupported('a backward branch in a program with v_interp_p1ll_f16 / v_interp_p2_f16')
                targets.add(target)

        def regs(tok):
            r = reg_range(tok)
            return set(range(r[1], r[2] + 1)) if r and r[0] == 'v' else set()

        pending, stale, high_undefined = {}, set(), set()
        for off, n, ops, mods in rows:
            branch_boundary = off in targets or n.startswith('s_cbranch') or n in ('s_branch', 's_setpc_b64')
            if branch_boundary and pending:
                raise Unsupported('a branch between v_interp_p1ll_f16 %s and its v_interp_p2_f16: p2 could be '
                                  'reached without the matching p1ll' % ', '.join('v%d' % r for r in sorted(pending)))
            if branch_boundary and (stale or high_undefined):
                raise Unsupported('unmodelled v_interp_p2_f16 result across a branch')
            store = n == 'exp' or n.startswith(INTERP_STORE_PREFIXES)
            # A malformed pair member is refused by its own handler; the scan models only the plain spelling.
            interp = (n in ('v_interp_p1ll_f16', 'v_interp_p2_f16') and len(ops) == (3 if n[10] == '1' else 4) and
                      all(re.fullmatch(r'v\d+', o) for o in ops[:2] + ops[3:]))
            if n in ('v_interp_p1ll_f16', 'v_interp_p2_f16') and not interp:
                return  # its own handler refuses the program; nothing after it is worth proving
            written = regs(ops[0]) if ops and not store else set()
            if interp:
                read = regs(ops[1])
                pair = self.interp_f16_pair(n, ops)
            else:
                read = set().union(*(regs(o) for o in ops)) if ops else set()
            low_reader = (n == 'v_cvt_f32_f16' and not mods and len(ops) == 2 and
                          all(re.fullmatch(r'v\d+', o) for o in ops))
            for r in sorted(read):
                if r in pending:
                    raise Unsupported('v%d is the v_interp_p1ll_f16 result, which emits nothing: %s reads it before '
                                      'the matching v_interp_p2_f16' % (r, n))
                if r in stale:
                    raise Unsupported('v%d is the v_interp_p1ll_f16 result, which emits nothing: %s reads it after '
                                      'a v_interp_p2_f16 that wrote another register' % (r, n))
                if r in high_undefined and not (low_reader and r == int(ops[1][1:])):
                    raise Unsupported('%s reads the upper 16 bits of the v_interp_p2_f16 result v%d: not modelled' %
                                      (n, r))
            if n == 'v_interp_p1ll_f16' and interp:
                pending[int(ops[0][1:])] = (ops[2], 'high' in mods)
            elif n == 'v_interp_p2_f16' and interp:
                d, out = int(ops[3][1:]), int(ops[0][1:])
                if pending.get(d) != (ops[2], 'high' in mods):
                    raise Unsupported('v_interp_p2_f16 source %s is not the v_interp_p1ll_f16 result of %s' %
                                      (ops[3], ops[2]))
                del pending[d]
                if out in pending:
                    raise Unsupported('v_interp_p2_f16 overwrites the pending v_interp_p1ll_f16 result v%d' % out)
                stale.discard(out)
                if out != d:
                    stale.add(d)
                if INTERP_DST_HIGH in mods:
                    # Completes a 16-bit pair: the low half it keeps must be a plain p2 result still waiting for its
                    # upper half (no other writer is modelled), after which every reader sees defined halves.
                    if out not in high_undefined:
                        raise Unsupported('v_interp_p2_f16 op_sel 0x8 into v%d, which holds no v_interp_p2_f16 low '
                                          'half: the half it keeps is not modelled' % out)
                    high_undefined.discard(out)
                else:
                    high_undefined.add(out)
            elif low_reader:
                high_undefined.discard(int(ops[0][1:]))
                stale.discard(int(ops[0][1:]))
        # I and J: never written anywhere
        base = pair if pair is not None else (getattr(getattr(self, 'pipe', None), 'persp_center_vgpr', 0) or 0)
        for off, n, ops, mods in rows:
            store = n == 'exp' or n.startswith(INTERP_STORE_PREFIXES)
            if ops and not store and regs(ops[0]) & {base, base + 1}:
                raise Unsupported('%s writes v%d: the entry barycentric registers must still hold the entry values '
                                  'where v_interp_p1ll_f16 / v_interp_p2_f16 read them' % (n, min(
                                      regs(ops[0]) & {base, base + 1})))
            if n == 'v_swap_b32' and len(ops) == 2 and regs(ops[1]) & {base, base + 1}:
                raise Unsupported('v_swap_b32 writes the entry barycentric registers')
        self.interp_scanned = True

    def op_v_cubeid_f32(self, ops, mods):
        self.cube('id', ops, mods)

    def scalar_wide_op(self, name, ops, mods, wide_src, wide_dst, helper, signed=None):
        """SOP2 shifts and bitfield extracts with a 64-bit source and/or result.

        D = f(S0, S1) with S1 a 32-bit word; SCC = (D != 0). Every source is read before any destination
        register is written, so an overlapping destination cannot feed a half-written value back."""
        if mods or len(ops) != 3:
            raise Unsupported('%s with %d operands and modifiers %s' % (name, len(ops), mods))
        dst = reg_range(ops[0])
        width = 1 + wide_dst
        if (not dst or dst[0] != 's' or dst[2] != dst[1] + wide_dst or dst[2] > 105
                or (wide_dst and dst[1] % 2)):
            raise Unsupported('%s destination %s' % (name, ops[0]))
        s1 = reg_range(ops[2])
        if ('m0' in ops[1:] or any(self.is_lane_mask(t) for t in ops[1:])
                or any(r[0] == 's' and any(w in self.masks for w in range(r[1], r[2] + 1))
                       for r in map(reg_range, ops[1:]) if r) or ops[2] in FLOATS or ops[1] in FLOATS
                or (s1 and (s1[0] != 's' or s1[1] != s1[2] or s1[1] > 105))):
            raise Unsupported('%s source %s' % (name, ', '.join(ops[1:])))
        if wide_src:
            r0 = reg_range(ops[1])
            if r0 and r0[0] != 's':
                raise Unsupported('%s source %s' % (name, ops[1]))
            low, high = self.i64_words(ops[1])
            known = [self.scalar_const('s%d' % r0[1]), self.scalar_const('s%d' % r0[2])] if r0 else [
                int(low[:-1]), int(high[:-1])]
        else:
            low, high = self.u(ops[1]), None
            known = [self.scalar_const(ops[1])]
        amount = self.u(ops[2])
        amount_known = self.scalar_const(ops[2])
        d = dst[1]
        self.masks.difference_update(range(d, d + width))
        if None not in known and amount_known is not None:
            x = known[0] | (known[1] << 32 if wide_src else 0)
            value = scalar_wide_fold(helper, x, amount_known, signed)
            for i in range(width):
                word = (value >> (32 * i)) & 0xffffffff
                self.set_s(d + i, '0x%08xu' % word, ('const', word))
            self.set_scc('true' if value else 'false')
            return None
        uniform = (self.tracks_uniformity()
                   and self.scalar_expr_uniform('%s %s %s' % (low, high or '', amount)))
        sg = '' if signed is None else ', %du' % (1 if signed else 0)
        if wide_dst:
            call = '%s(%s, %s, %s%s)' % (helper, low, high if wide_src else '0u', amount, sg)
            self.emit('{ uint2 t = %s; s%d = t.x; s%d = t.y; scc = (t.x | t.y) != 0u; }' % (call, d, d + 1))
        else:
            self.emit('{ uint t = %s(%s, %s%s); s%d = t; scc = t != 0u; }' % (helper, low, amount, sg, d))
        for r in range(d, d + width):
            self.sym.pop(r, None)
            self.used_s.add(r)
            if self.tracks_uniformity():
                (self.varying_s.discard if uniform else self.varying_s.add)(r)
        self.scc_effect = 'emitted'
        if self.tracks_uniformity():
            self.scc_uniform = uniform
        return None

    def interp_f16(self, op, ops, mods, vsrc, n_ops):
        """Common checks of v_interp_p1ll_f16 / v_interp_p2_f16: -> (attr, field, high).

        The attribute channel holds two binary16 values; `field` is the interpolated float the pair reads
        (`param0.x`, `hparam0_lo.z`). Two vertex exports exist, with different mappings:
        - compr (see export()): the pair is split into float channels, so half h of attribute channel c is float
          channel 2 * c + h of the parameter: attrN.x = (param.x, param.y), attrN.y = (param.z, param.w). Channels
          z and w have no such pair.
        - plain (a 32-bit export, e.g. halves packed by v_cvt_pkrtz_f16_f32): half h of channel c is
          f16tof32(word_c >> 16 * h), carried by the extra outputs hparamN_lo (h = 0) and hparamN_hi (h = 1),
          channel c of each. All four channels exist.
        """
        allowed = ('high', INTERP_DST_HIGH) if op == 'v_interp_p2_f16' else ('high',)
        extra = [m for m in mods if m not in allowed]
        if extra:
            raise Unsupported('%s with %s: only the plain form and high are lowered' % (op, ' '.join(extra)))
        if len(ops) != n_ops:
            raise Unsupported('%s with %d operands, expected %d' % (op, len(ops), n_ops))
        for tok in (ops[0], ops[1]) + tuple(ops[3:]):
            if not re.fullmatch(r'v\d+', tok):
                raise Unsupported('%s operand %s: only plain VGPRs are lowered' % (op, tok))
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', ops[2])
        if not m:
            raise Unsupported('%s interpolant %s: expected attrN.x/.y/.z/.w' % (op, ops[2]))
        attr, component = self.note_attr(ops[2], 'f16')
        chan, high = 'xyzw'.index(component), int('high' in mods)
        vs = getattr(getattr(self, 'pipe', None), 'vs', None)
        if vs is not None and attr in vs.plain_params and attr not in vs.compr_params:
            self.f16_plain.setdefault(attr, set()).add(chan)
            return attr, 'hparam%d_%s.%s' % (attr, ('lo', 'hi')[high], component), high
        if chan > 1:
            raise Unsupported('%s on channel %s: a compr export fills only the x and y channels' % (op, m.group(2)))
        return attr, 'param%d.%s' % (attr, 'xyzw'[2 * chan + high]), high

    def join_writes(self, branches):
        """Branch target -> the scalar registers another path can write before it reaches the target: the loop body
        [target, branch] for a backward branch, and the skipped instructions (branch, target) for a forward one."""
        offs, written = [], []  # per instruction, once: a program has hundreds of branches over thousands of instructions
        for off, _n, name, asm in self.ins:
            regs = set(range(128)) if name.startswith('s_movreld') else set()  # M0 picks the destination
            for tok in destinations(name, parse(asm)[1]):
                r = reg_range(tok)
                if r and r[0] == 's':
                    regs.update(range(r[1], r[2] + 1))
                elif tok == 'm0':
                    regs.add('m0')
            offs.append(off)
            written.append(regs)
        writes = {}
        for branch, target in branches:
            lo, hi = ((bisect.bisect_left(offs, target), bisect.bisect_right(offs, branch)) if target <= branch else
                      (bisect.bisect_right(offs, branch), bisect.bisect_left(offs, target)))
            writes.setdefault(target, set()).update(*written[lo:hi])
        return writes

    def op_s_lshr_b64(self, ops, mods):
        return self.scalar_wide_op('s_lshr_b64', ops, mods, True, True, 'shr64u')

    def scalar_accumulate(self, base, ops, mods):
        """s_addk_i32 / s_mulk_i32: D = D op signext(SIMM16), the destination is also the source.

        RDNA 2 ISA 12.2 (SOPK 15, 16): addk sets SCC to the signed overflow of the sum, mulk leaves SCC
        alone. LLVM 15 SOPInstructions.td declares Defs = [SCC] for both, so after mulk the SCC proof is
        dropped (the emitted scc keeps its value, which is the ISA meaning).
        """
        if mods or len(ops) != 2:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('%s destination %s' % (base, ops[0]))
        if not re.fullmatch(r'0x[0-9a-f]{1,4}', ops[1]):
            raise Unsupported('%s immediate %s' % (base, ops[1]))
        d = dst[1]
        # D is read before it is written: refuse a mask or descriptor source while the old state is intact.
        if d in self.masks or self.sym.get(d, ('',))[0] == 'execcopy':
            raise Unsupported('%s on a lane mask' % base)
        sym = self.sym.get(d)
        if sym and sym[0] not in ('const', 'ud', 'fixed_buffer_word'):
            raise Unsupported('%s on %s data' % (base, sym[0]))
        simm = int(ops[1], 16)
        imm = (simm - 0x10000 if simm & 0x8000 else simm) & 0xffffffff
        imm_text = '0x%08xu' % imm
        old = self.S(d)
        add = base == 's_addk_i32'
        result = '%s %s %s' % (old, '+' if add else '*', imm_text)
        if add:
            # signed overflow: the operands share a sign that the sum does not keep
            overflow = '((%s ^ (%s)) & ~(%s ^ %s) & 0x80000000u) != 0u' % (old, result, old, imm_text)
            if sym and sym[0] == 'const':
                total = (sym[1] + imm) & 0xffffffff
                overflow = 'true' if ((sym[1] ^ total) & ~(sym[1] ^ imm) & 0x80000000) else 'false'
            self.set_scc(overflow)
        elif self.tracks_uniformity():
            self.scc_uniform = False
        if sym and sym[0] == 'const':
            value = (sym[1] + imm if add else sym[1] * imm) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
        else:
            self.set_s(d, result)
        self.masks.discard(d)
        return None

    def op_buffer_store_format_d16_x(self, ops, mods):
        """Lower compute MUBUF buffer_store_format_d16_x (opcode 0x84) using V# word 3, stride and num_records.

        Writes one converted 16-bit component from the low 16 bits of vdata into a raw UAV buffer:
        - BUF_FMT_32_FLOAT (22): binary16 -> binary32, 4-byte aligned Store.
        - BUF_FMT_16_FLOAT (13), BUF_FMT_16_UINT (11), BUF_FMT_16_SINT (12): low 16 bits stored at a
          2-byte aligned offset via a lock-free InterlockedCompareExchange CAS loop on the enclosing dword.
        - BUF_FMT_16_UNORM (7), BUF_FMT_16_SNORM (8): binary16 -> binary32 -> D3D half-away-from-zero
          16-bit UNORM/SNORM (NaN -> 0), stored via the same 2-byte atomic CAS loop.
        Per RDNA 2 ISA §8.1.5 Table 35, OOB_SELECT 0 (structured, stride != 0) checks index < num_records
        and (voff + imm_off) + elem_bytes <= stride while soffset adds to the base address; OOB_SELECT 3
        (raw, stride == 0) checks (soffset + voff + imm_off) + elem_bytes <= num_records. OOB_SELECT 1/2,
        mismatched stride/OOB_SELECT, nonzero DW1[31:30] swizzle flags (uav_stride carries DW1 >> 16),
        unsupported formats/flags, misaligned addresses and out-of-bounds lanes are no-ops.
        """
        name = 'buffer_store_format_d16_x'
        if self.kind != 'cs':
            raise Unsupported(name)
        raw = getattr(self, 'raw_code', b'')
        pc = getattr(self, 'pc', None)
        if raw and pc is not None and pc + 8 <= len(raw):
            w0, w1 = struct.unpack_from('<II', raw, pc)
            if (w0 & 0x00030000) or (w1 & 0x00200000):
                raise Unsupported('%s illegal MUBUF encoding bits (lds/reserved)' % name)
        if len(ops) != 4:
            raise Unsupported('%s with %d operands, expected 4' % (name, len(ops)))
        idxen = offen = False
        imm_offset = None
        for m in mods:
            if m == 'idxen' and not idxen:
                idxen = True
            elif m == 'offen' and not offen:
                offen = True
            elif m.startswith('offset:') and imm_offset is None and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                val = int(m[7:], 0)
                if val > 0xfff:
                    raise Unsupported('%s modifier %s' % (name, m))
                imm_offset = val
            else:
                raise Unsupported('%s modifier %s' % (name, m))
        imm_offset = 0 if imm_offset is None else imm_offset
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[1] != src[2] or src[1] > 255:
            raise Unsupported('%s data %s' % (name, ops[0]))
        if idxen and offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[2] - vaddr[1] != 1 or vaddr[2] > 255:
                raise Unsupported('%s vaddr %s for idxen offen' % (name, ops[1]))
            idx_expr, voff_expr = self.u('v%d' % vaddr[1]), self.u('v%d' % vaddr[2])
        elif idxen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for idxen' % (name, ops[1]))
            idx_expr, voff_expr = self.u(ops[1]), '0u'
        elif offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for offen' % (name, ops[1]))
            idx_expr, voff_expr = '0u', self.u(ops[1])
        else:
            if ops[1] != 'off':
                raise Unsupported('%s vaddr %s without idxen or offen' % (name, ops[1]))
            idx_expr, voff_expr = '0u', '0u'
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            soff_imm = int(ops[3], 0)
            if soff_imm > 0xffffffff:
                raise Unsupported('%s soffset %s' % (name, ops[3]))
            soff_expr = '%du' % soff_imm
        else:
            if self.is_lane_mask(ops[3]):
                raise Unsupported('%s soffset %s is a lane mask' % (name, ops[3]))
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('%s soffset %s' % (name, ops[3]))
            soff_expr = self.S(sreg[1])
        rsrc = reg_range(ops[2])
        if not rsrc or rsrc[0] != 's' or rsrc[2] - rsrc[1] != 3 or rsrc[2] > 105:
            raise Unsupported('%s descriptor %s' % (name, ops[2]))
        key = self.store_descriptor(ops[2], 4, name, stride0_ok=False)
        first = self.sym.get(rsrc[1])
        if getattr(self, 'past_entry', False) and not self.formatted_descriptor_from_block(rsrc, first):
            raise Unsupported('%s requires descriptor words proven in the entry block or a local table load' % name)
        uav = self.pipe.uav(key, 'raw')
        uav_index = self.pipe.uavs[key]
        word3 = 'ud%d' % self.pipe.ud(('uav_descword3', uav_index))
        stride = 'ud%d' % self.pipe.ud(('uav_stride', uav_index))
        num_records = 'ud%d' % self.pipe.ud(('uav_numrecords', uav_index))
        val_expr = self.u(ops[0], low16_only=True)
        self.pipe.uses_store_d16_x = True
        call = 'store_format_d16_x(%s, %s, %s, %s, %s, %s, %s, %du, %s);' % (
            uav, word3, stride, num_records, idx_expr, voff_expr, soff_expr, imm_offset, val_expr)
        self.emit(call if self.exec_known else 'if (exec & 1) ' + call)

    def cube(self, kind, ops, mods):
        """v_cube{id,sc,tc,ma}_f32: LLVM's ConstantFoldAMDGCNCubeIntrinsic.

        The major axis is z when |z| is at least |x| and |y|, otherwise y when |y| is at
        least |x|, otherwise x. A face is negative only for a nonzero, non-NaN negative
        coordinate. cubema returns twice the signed major coordinate.
        """
        if len(ops) != 4:
            raise Unsupported('v_cube%s_f32 with %d operands, expected 4' % (kind, len(ops)))
        if any(m != 'clamp' for m in mods):
            raise Unsupported('v_cube%s_f32 with %s' % (kind, ', '.join(mods)))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] != dst[1] or dst[2] > 255:
            raise Unsupported('v_cube%s_f32 destination %s' % (kind, ops[0]))
        x, y, z = [self.f(o) for o in ops[1:4]]
        temp = 'cube%d' % getattr(self, 'cube_temps', 0)
        self.cube_temps = getattr(self, 'cube_temps', 0) + 1
        self.emit('float3 %s = float3(%s, %s, %s);' % (temp, x, y, z))
        c = temp
        a = 'abs(%s)' % c
        zmajor = '(%s.z >= %s.x && %s.z >= %s.y)' % (a, a, a, a)
        ymajor = '(%s.y >= %s.x)' % (a, a)
        expression = {
            'id': '(%s ? (%s.z < 0.0f ? 5.0f : 4.0f) : %s ? (%s.y < 0.0f ? 3.0f : 2.0f) '
                  ': (%s.x < 0.0f ? 1.0f : 0.0f))' % (zmajor, c, ymajor, c, c),
            'sc': '(%s ? (%s.z < 0.0f ? -%s.x : %s.x) : %s ? %s.x '
                  ': (%s.x < 0.0f ? %s.z : -%s.z))' % (zmajor, c, c, c, ymajor, c, c, c, c),
            'tc': '(%s ? -%s.y : %s ? (%s.y < 0.0f ? -%s.z : %s.z) : -%s.y)' % (
                zmajor, c, ymajor, c, c, c, c),
            'ma': '((%s ? %s.z : %s ? %s.y : %s.x) * 2.0f)' % (zmajor, c, ymajor, c, c),
        }[kind]
        self.set_vf(dst[1], expression, mods)

    def buffer_atomic(self, name, ops, mods):
        """Lower 32-bit compute MUBUF atomics with optional glc pre-atomic return."""
        if self.kind != 'cs':
            raise Unsupported(name)
        method, signed, is_sub, data_count = MUBUF_ATOMIC_OPS[name]
        data, uav, address, glc = self.raw_table_uav_store(
            'buffer atomic', ops, mods, data_count, allow_glc=True, allow_scalar_address=name == 'buffer_atomic_add')
        words = [self.u('v%d' % n) for n in range(data[1], data[2] + 1)]
        self.emit('a = %s;' % address)
        if not isinstance(getattr(self, 'used_atomic', None), set):
            self.used_atomic = set()
        if name == 'buffer_atomic_cmpswap':
            # RDNA 2 puts replacement in vdata[0] and compare in vdata[1]; HLSL takes compare first.
            if glc:
                self.used_atomic.add('uint')
                call = '%s.InterlockedCompareExchange(a, %s, %s, ao);' % (uav, words[1], words[0])
            else:
                call = '%s.InterlockedCompareStore(a, %s, %s);' % (uav, words[1], words[0])
        elif name == 'buffer_atomic_swap':
            # HLSL RWByteAddressBuffer.InterlockedExchange always requires an output operand.
            self.used_atomic.add('uint')
            call = '%s.InterlockedExchange(a, %s, ao);' % (uav, words[0])
        else:
            arg = 'asint(%s)' % words[0] if signed else ('(0u - %s)' % words[0] if is_sub else words[0])
            if glc:
                out_var = 'ai' if signed else 'ao'
                self.used_atomic.add('int' if signed else 'uint')
                call = '%s.Interlocked%s(a, %s, %s);' % (uav, method, arg, out_var)
            else:
                call = '%s.Interlocked%s(a, %s);' % (uav, method, arg)
        self.emit(call if self.exec_known else 'if (exec & 1) %s' % call)
        if glc:
            self.set_v(data[1], 'asuint(ai)' if signed else 'ao')

    def op_s_bfe_i32(self, ops, mods):
        return self.scalar_wide_op('s_bfe_i32', ops, mods, False, False, 'bfe32', True)

    def scalar_const(self, tok):
        """Value of a 32-bit scalar source that is known at translation time, else None."""
        r = reg_range(tok)
        if r:
            return self.is_const(r[1]) if r[0] == 's' and r[1] == r[2] else None
        try:
            return int(tok, 0) & 0xffffffff
        except ValueError:
            return None

    def op_s_bfe_u64(self, ops, mods):
        return self.scalar_wide_op('s_bfe_u64', ops, mods, True, True, 'bfe64', False)

    def interp_f16_pair(self, op, ops):
        """The entry I/J registers: the PERSP_CENTER pair of the pixel inputs, I in the first, J in the second."""
        base = getattr(getattr(self, 'pipe', None), 'persp_center_vgpr', 0)
        if base is None:
            raise Unsupported('%s: the pixel inputs do not enable the PERSP_CENTER barycentrics' % op)
        want = 'v%d' % (base + (0 if op == 'v_interp_p1ll_f16' else 1))
        if ops[1] != want:
            raise Unsupported('%s barycentric operand %s: only the PERSP_CENTER entry %s (%s) is lowered' % (
                op, ops[1], 'I' if op == 'v_interp_p1ll_f16' else 'J', want))
        return base

    def op_s_lshl_b64(self, ops, mods):
        return self.scalar_wide_op('s_lshl_b64', ops, mods, True, True, 'shl64')

    def _lds_require_alloc(self, base):
        if getattr(self, 'pipe', None) is None:
            raise Unsupported('%s needs pipeline metadata' % base)
        words = self.pipe.lds_words
        if not 0 < words <= 8192:
            raise Unsupported('%s needs a nonzero LDS allocation of at most 32 KiB' % base)
        self.uses_lds = True
        return words * 4

    def lds_dual(self, name, ops, mods):
        """Lower 32-bit dual-offset and stride-64 LDS ops (ds_write2_b32, ds_write2st64_b32, ds_read2st64_b32)."""
        if getattr(self, 'kind', None) != 'cs':
            raise Unsupported(name)
        is_write, stride = LDS_DUAL_OPS[name]
        if is_write:
            if (len(ops) != 3 or _lds_single(ops[0]) is None
                    or _lds_single(ops[1]) is None or _lds_single(ops[2]) is None):
                raise Unsupported('%s needs three single VGPRs (addr, data0, data1)' % name)
        else:
            if len(ops) != 2 or _lds_pair(ops[0]) is None or _lds_single(ops[1]) is None:
                raise Unsupported('%s needs a 2-VGPR pair and a byte address VGPR' % name)
        # Reuse ds_read2_b32 modifier/aliasing validation (offset0/offset1 in 0..255, distinct, no gds/unknown mods).
        off0, off1 = lds_byte_offsets('ds_read2_b32', mods)
        scale = stride // 4
        off0, off1 = off0 * scale, off1 * scale
        lds_bytes = self._lds_require_alloc(name)
        exec_prefix = '' if self.exec_known else '(exec & 1) && '
        if is_write:
            addr = self.u(ops[0])
            d0, d1 = self.u(ops[1]), self.u(ops[2])
            self.emit('a = %s;' % addr)
            for off, val in ((off0, d0), (off1, d1)):
                if off < lds_bytes:
                    self.emit('if (%s((a & 3u) == 0u) && a < %du) lds[(a + %du) >> 2] = %s;'
                              % (exec_prefix, lds_bytes - off, off, val))
        else:
            dst = _lds_pair(ops[0])
            addr = self.u(ops[1])
            self.emit('a = %s;' % addr)
            for idx, off in enumerate((off0, off1)):
                if off < lds_bytes:
                    self.emit('w = 0u;')
                    self.emit('if (%s((a & 3u) == 0u) && a < %du) w = lds[(a + %du) >> 2];'
                              % (exec_prefix, lds_bytes - off, off))
                    self.set_v(dst + idx, 'w')
                else:
                    self.set_v(dst + idx, '0u')

    def op_s_ff1_i32_b32(self, ops, mods):
        if self.kind != 'cs' or mods or len(ops) != 2 or ops[1] != 'exec_lo':
            raise Unsupported('s_ff1_i32_b32 requires compute exec_lo')
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
            raise Unsupported('s_ff1_i32_b32 requires one SGPR destination')
        if self.past_entry or not self.exec_known:
            raise Unsupported('s_ff1_i32_b32 requires the compute entry EXEC mask')
        # At dispatch entry even the last partial wave starts at lane zero.
        # FF1 reports its index and leaves SCC unchanged.
        self.set_s(dst[1], '0u', ('const', 0))

    def __init__(self, pipe, kind, code, refusals=None, decoded=None):
        self.pipe, self.kind = pipe, kind
        self.raw = strip_header(code)  # instruction words, for encoding bits the decoder does not print
        self.refusals = refusals
        self.raw_code = self.raw
        if refusals is None:
            self.ins = decode(self.raw)
        else:
            self.ins = decode(self.raw_code, refusals) if decoded is None else decoded
        if kind == 'ms' and getattr(pipe, 'uses_mesh_skeleton', False):
            self.mesh_vcc_branches = {}
            self.mesh_wave_limits = {}
            self.ins = mesh_skeleton(self.ins, self.mesh_vcc_branches, self.mesh_wave_limits)
        # A loop header is translated before the back edges into it, so whether a back edge brings the symbol kept
        # there is known only afterwards: translate again with that register read at run time or refused there. A
        # register only moves from kept to 'run' to 'refuse', so this ends.
        self.force_joined = {}    # branch target -> register -> 'run' | 'refuse'
        first_refusal = len(refusals) if refusals is not None else 0
        while True:
            self.lines = []           # body statements (per block)
            self.sym = {}             # scalar register -> symbolic value
            self.used_s, self.used_v = set(), set()
            self.used_atomic = set()
            self.exec_known = True    # exec still equals the entry mask (the lane is active)
            self.wqm_exec = False     # exec is proven whole-quad: an s_wqm of exec into itself ran while exec_known
            self.exports = {}         # 'pos0' / 'paramN' / 'mrtN' -> [4 expressions]
            self.write_masks = {}     # 'mrtN' -> channels the export writes (bit c = channel c), see export()
            self.attrs = {}           # pixel: attr index -> 'linear' | 'flat' | 'f16'
            self.compr_params = set() # vertex: param indices exported with the compr flag (packed binary16 pairs)
            self.plain_params = {}    # vertex: param index -> channels exported as `off` by its latest plain export
            self.f16_plain = {}       # pixel: attr index -> channels read as binary16 halves of a plain export
            self.uses_pos = False
            self.masks = set()        # scalar registers currently holding lane masks
            self.fetched = False      # vertex: the entry call into the fetch routine was translated as loads
            self.defined_uniform_s = set()
            self.bad_sgprs = set()
            self._uniform_written = set()
            self.splat = set()
            self.uses_q = False
            self.used_h = set()
            self.written_v = set()
            self.entry_reads = {}
            self.pc = 0
            self.round16 = 0
            if kind == 'ps':
                self.launch_mode(getattr(pipe, 'ps_float_mode', None))
            self.setup_entry()
            self.translate()
            grown = False
            for t, regs in getattr(self, 'back_force', {}).items():
                have = self.force_joined.setdefault(t, {})
                for reg, kind in regs.items():
                    if have.get(reg) not in (kind, 'refuse'):
                        have[reg], grown = kind, True
            if not grown:
                break
            if refusals is not None:
                del refusals[first_refusal:]

    # ---- entry state ---------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        # Graphics root parameters start at s8. For compute, s4 is an inference, not a documented layout: the only
        # in-repo evidence is a local compute-record scan whose table bindings used s4 or s5 (docs/experiments.md).
        # s0-s3 are not modelled.
        reg = 4 if self.kind == 'cs' else 8
        for index, p in enumerate(root['params']):
            if p['type'] == 0:
                self.sym[reg] = ('ptr', ('table', index))
                reg += 1
            elif p['type'] == 1:
                for i in range(p['count']):
                    self.sym[reg] = ('ud', self.pipe.ud(('rootconst', index, i)))
                    reg += 1
            else:
                self.sym[reg] = ('ptr', ('root', index))
                self.sym[reg + 1] = ('ptrhi', ('root', index))
                reg += 2
        if root['flags'] & 0x400:
            self.sym[reg] = ('ptr', ('resheap',))
            reg += 1
        if root['flags'] & 0x800:
            self.sym[reg] = ('ptr', ('smpheap',))
            reg += 1
        if self.kind == 'cs':
            self.setup_compute_entry(reg)
        elif self.kind == 'ms':
            # The wave index is bits 24..27 of s3 (mesh_skeleton requires the program to read it that way).
            self.used_s.add(3)
            self.entry = ['s3 = (tid >> 6) << 24;']
            for n in range(105):
                if n != 3 and n not in self.sym:
                    self.sym[n] = ('an entry value the mesh model does not define',)
        elif self.kind == 'vs':
            self.sym[0] = ('ptr', ('vbtab',))
            self.sym[1] = ('ptrhi', ('vbtab',))
            self.sym[2] = ('ud', self.pipe.ud(('basevertex',)))
            self.sym[3] = ('ud', self.pipe.ud(('startinstance',)))
        else:
            self.sym[reg] = ('const', 0)  # primitive mask: only ever moved to m0

    def setup_compute_entry(self, root_end):
        """Entry registers of a compute wave as compute_store_aot.entry_registers (#1608) gives them: the user SGPRs,
        then one SGPR per enabled group-ID axis (SV_GroupID), and v0..v2 holding SV_GroupThreadID. Where the root
        parameters sit inside the user SGPRs (from s4) is this translator's inference. User SGPRs the root does not
        place (s0-s3 and any past the root's span) have no known value, so any numeric read of them is refused."""
        user, groups, threads = self.pipe.entry
        if root_end > user:
            raise Unsupported('root signature needs s4-s%d, COMPUTE_PGM_RSRC2 gives %d user SGPRs' % (root_end - 1, user))
        for n in list(range(4)) + list(range(root_end, user)):
            self.sym[n] = ('unknown user sgpr',)
        self.entry = []
        self.entry_uniform_s = set()
        for axis, n in sorted(groups.items()):
            self.used_s.add(n)
            self.entry_uniform_s.add(n)
            self.entry.append('s%d = gid.%s;' % (n, 'xyz'[axis]))
        for i in range(threads):
            self.used_v.add(i)
            self.entry.append('v%d = tid.%s;' % (i, 'xyz'[i]))

    # ---- operand helpers ---------------------------------------------------------------------
    def S(self, n):
        """Runtime expression of scalar register n (uint)."""
        s = self.sym.get(n)
        if s:
            if s[0] == 'const':
                return '0x%08xu' % (s[1] & 0xffffffff)
            if s[0] == 'ud':
                return 'ud%d' % s[1]
            if s[0] in ('execcopy', 'fixed_buffer_word'):
                self.used_s.add(n)
                return 's%d' % n
            raise Unsupported('numeric use of s%d holding %s' % (n, s[0]))
        self.used_s.add(n)
        return 's%d' % n

    def note_whole_read(self, reg):
        if reg in self.low16_only_v:
            raise Unsupported('whole-word read of v%d after odd d16 image_load' % reg)
        if isinstance(self.whole_read_v, set):
            self.whole_read_v.add(reg)

    def u(self, tok, *, low16_only=False):
        """uint expression for a 32-bit operand."""
        if tok in ('vcc_lo', 'vcc'):
            self.read_vcc()
            return 'vcc'
        if tok in ('exec_lo', 'exec'):
            return 'exec'
        if tok == 'm0':
            m = self.sym.get('m0')
            if m:
                raise Unsupported('numeric use of m0 holding %s' % m[0])
            return 'm0'
        if tok == 'src_scc':
            # SCC read as a 32-bit operand (0 or 1)
            return '(scc ? 1u : 0u)'
        if tok in getattr(self, 'dpp_names', ()):
            return tok  # the quad-read source of a DPP instruction (quad_dpp)
        r = reg_range(tok)
        if r:
            if r[0] == 'v':
                written = getattr(self, 'written_v', None)  # absent on the tests' bare one-instruction stages
                if written is not None and r[1] not in written:
                    self.entry_reads.setdefault(r[1], self.pc)
                self.used_v.add(r[1])
                if not low16_only:
                    for reg in range(r[1], r[2] + 1):
                        self.note_whole_read(reg)
                return 'v%d' % r[1]
            return self.S(r[1])
        if tok in FLOATS:
            return '0x%08xu' % fbits(FLOATS[tok])
        try:
            return '0x%08xu' % (int(tok, 0) & 0xffffffff)
        except ValueError:
            raise Unsupported('operand %r' % tok)

    def is_lane_mask(self, tok):
        reg = reg_range(tok)
        return (tok in ('exec', 'exec_lo', 'exec_hi', 'vcc', 'vcc_lo', 'vcc_hi')
                or bool(reg and reg[0] == 's' and reg[1] in getattr(self, 'masks', ())))

    @staticmethod
    def split_neg(tok):
        """(negated, operand) for the `-x` sign modifier and the `neg(x)` spelling of it."""
        # LLVM prints a lone negation of a numeric operand as neg(c), so that `-1` is not read as the
        # integer: it is the sign bit flipped, never an integer negation.
        m = re.fullmatch(r'neg\((.+)\)', tok)
        if m:
            return True, m.group(1)
        if tok.startswith('-') and not re.match(r'-[\d.]', tok):
            return True, tok[1:]
        return False, tok

    def reject_lane_mask_data(self, op, tok):
        """The lane bit cannot represent a mask's word when it is read as data."""
        source = tok
        while True:
            neg, operand = self.split_neg(source)
            if neg:
                source = operand
            elif source.startswith('|') and source.endswith('|') and len(source) >= 2:
                source = source[1:-1]
            elif source.startswith(('abs(', 'sext(')) and source.endswith(')'):
                source = source[source.index('(') + 1:-1]
            else:
                break
        if self.is_lane_mask(source):
            raise Unsupported('%s data operand %s is a lane mask' % (op, tok))

    def u_src(self, tok):
        """uint expression for an integer-context source that the disassembler may print negated."""
        # In an integer instruction the `-` modifier is the source's sign bit flipped: the same bit
        # pattern a float negation produces. Only handlers that model it ask for this; other integer
        # sources keep using u(), so an unexpected negation stays a refusal rather than a silent flip.
        neg, tok = self.split_neg(tok)
        return '(%s ^ 0x80000000u)' % self.u(tok) if neg else self.u(tok)

    def f(self, tok):
        """float expression, with the inline neg/abs operand modifiers."""
        neg, tok = self.split_neg(tok)
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(FLOATS[tok])
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'F(%s)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def u_mod(self, tok, sign=0x80000000):
        """uint expression for an operand with VOP3 neg/abs applied as sign-bit operations (no float rounding)."""
        neg = tok.startswith('-') and not re.match(r'-[\d.]', tok)
        if neg:
            tok = tok[1:]
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        text = self.u(tok)
        if absolute:
            text = '(%s & %#xu)' % (text, ~sign & 0xffffffff)
        if neg:
            text = '(%s ^ %#xu)' % (text, sign)
        return text

    def h(self, tok):
        """uint expression whose low 16 bits are a binary16 operand: inline float constants take their half form."""
        if tok in FLOATS:
            return '0x%04xu' % struct.unpack('<H', struct.pack('<e', FLOATS[tok]))[0]
        return self.u(tok)

    def is_const(self, n):
        s = self.sym.get(n)
        return s[1] if s and s[0] == 'const' else None

    def const_u32(self, tok):
        r = reg_range(tok)
        if r:
            if r[0] == 's' and r[1] == r[2]:
                c = self.is_const(r[1])
                return None if c is None else (c & 0xffffffff)
            return None
        if tok in FLOATS:
            return fbits(FLOATS[tok])
        try:
            return int(tok, 0) & 0xffffffff
        except ValueError:
            return None

    def read_vcc(self):
        if getattr(self, 'has_branches', False):
            self.vcc_events.setdefault(self.pc, []).append(('read_vcc', None))
        elif getattr(self, 'vcc_lost', False):
            raise Unsupported('vcc read after a descriptor load into s106/s107')

    def emit(self, text):
        # The address temporary `a` is lane-invariant while its last assignment was an unconditional one from a
        # lane-invariant expression (a buffer load with no VGPR address, or one computed from uniform values).
        if hasattr(self, 'lane_s') and re.search(r'\ba = ', text):
            m = re.fullmatch(r'a = (.*);', text)
            self.a_invariant = bool(m) and self.lane_invariant(m.group(1))
        if re.search(r'\bexec = ', text):
            self.exec_writes = getattr(self, 'exec_writes', 0) + 1
        if re.search(r'\bt = ', text):
            self.t_invariant = False  # image() marks its own result again (see there)
        if 'vcc' in text:
            assignment = re.match(r'vcc = (.*);$', text)
            rhs = assignment.group(1) if assignment else text
            if re.search(r'\bvcc\b', rhs):
                self.read_vcc()
            if assignment:
                if getattr(self, 'has_branches', False):
                    self.vcc_events.setdefault(self.pc, []).append(('write_vcc', None))
                if getattr(self, 'vcc_lost', False):
                    # The scalar symbols are provisional across branches, checked after translation.
                    self.vcc_lost = False
                    self.vcc_descriptor_overwritten = True
                    if not getattr(self, 'has_branches', False):
                        self.sym.pop(106, None)
                        self.sym.pop(107, None)
        self.lines.append(text)

    def check_vcc_paths(self):
        """An overlapping descriptor must reach its use on every path, and VCC must be numeric at a mask read."""
        if not self.has_branches or not any(event == 'load_descriptor' for events in self.vcc_events.values()
                                           for event, _ in events):
            return
        offsets = [off for off, _, _ in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        by_block = [[] for _ in offsets]
        block = 0
        for off, events in sorted(self.vcc_events.items()):
            while block + 1 < len(offsets) and offsets[block + 1] <= off:
                block += 1
            by_block[block].extend(events)
        todo = [(0, None)]
        seen = set()
        while todo:
            block, state = todo.pop()
            if (block, state) in seen:
                continue
            seen.add((block, state))
            for event, value in by_block[block]:
                if event == 'read_vcc' and state is not None:
                    raise Unsupported('vcc read after a descriptor load into s106/s107')
                if event == 'read_descriptor':
                    if state is None or any(state[reg] != symbol for reg, symbol in value):
                        raise Unsupported('descriptor operand overlaps s106/s107 without a matching load on every path')
                if event == 'load_descriptor':
                    state = value
                elif event == 'write_vcc':
                    state = None
            term = self.blocks[block][2]
            if term and term[0] in ('jump', 'branch'):
                if term[-1] in index:
                    todo.append((index[term[-1]], state))
            if (not term or term[0] == 'branch') and block + 1 < len(offsets):
                todo.append((block + 1, state))

    def clear_low16(self, n):
        if isinstance(self.low16_only_v, set) and self.exec_known and n not in self.pinned_low16_v:
            self.low16_only_v.discard(n)

    def lane_invariant(self, expr):
        """True when the HLSL expression has the same value in every lane of the wave.

        It may name only entry constants (udN), resources (rN, read at any address the expression itself
        computes), pure functions and literals, SGPR variables that hold a scalar number (not a lane mask: the
        one-lane model keeps only the lane's bit of a mask) and VGPRs proven lane-invariant. vcc, exec, scc, m0,
        the temporaries (a, w, t, q) and the interpolants (i.) depend on the lane or on state not tracked here.
        """
        for m in re.finditer(r'\b[A-Za-z_]\w*', expr):
            name = m.group(0)
            if name in ('x', 'y', 'z', 'w') and expr[m.start() - 1:m.start()] == '.' and \
                    re.search(r'\bt\.$', expr[:m.start()]):
                continue  # a component of the image result t (lane-invariant only when t is, checked as 't')
            if name in LANE_INVARIANT_NAMES or name.startswith('x2w_'):
                continue
            if name == 'a' and getattr(self, 'a_invariant', False):
                continue
            if name == 't' and getattr(self, 't_invariant', False):
                continue
            if re.fullmatch(r'xv\d+', name):  # assigned in every lane from a lane-invariant value (readfirstlane)
                continue
            reg = re.fullmatch(r'(ud|r|s|v)(\d+)', name)
            if reg and reg.group(1) in ('ud', 'r'):
                continue
            if reg and reg.group(1) == 's' and int(reg.group(2)) not in self.lane_s and int(reg.group(2)) not in self.masks:
                continue
            if reg and reg.group(1) == 'v' and int(reg.group(2)) in self.uniform_v:
                continue
            return False
        return True

    def set_v(self, n, expr, *, low16_only=False):
        if hasattr(self, 'written_v'):
            self.written_v.add(n)
        if getattr(self, 'interp_p1', None):
            self.interp_p1.pop(n, None)
        self.last_v = (n, expr)
        if low16_only:
            if not isinstance(self.low16_only_v, set):
                self.low16_only_v = set()
            self.low16_only_v.add(n)
        else:
            self.clear_low16(n)
        # A VGPR written in every lane (exec still the entry mask) from lane-invariant values holds one value in
        # all of them, so a later v_readfirstlane_b32 reads that value whatever exec is then.
        invariant = self.exec_known and hasattr(self, 'lane_s') and self.lane_invariant(expr)
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(n)
        if invariant:
            self.uniform_v.add(n)
        partial = getattr(self, 'partial_uniform', None)
        if partial is not None:
            shadow = None if self.exec_known else self.partial_shadow(expr)
            partial.pop(n, None)
        self.used_v.add(n)
        self.emit('v%d = %s;' % (n, expr) if self.exec_known else 'if (exec & 1) v%d = %s;' % (n, expr))
        if partial is not None and shadow is not None:
            # Written in the active lanes only, from a lane-invariant value: every lane active now holds that value.
            # Kept with the emitted line so v_readfirstlane_b32 can take the value for every lane (see there).
            partial[n] = {'lines': self.lines, 'at': len(self.lines) - 1, 'n': n, 'expr': shadow[0], 'deps': shadow[1],
                          'epoch': getattr(self, 'exec_writes', 0), 'done': False}

    def partial_shadow(self, expr):
        """(shadow expression, entries it reads) when expr has one value in every lane active now; else None.

        Lane-invariant names qualify, and so does a VGPR written in the active lanes from such a value with exec
        not written since (partial_uniform, same epoch): that VGPR is replaced by its shadow xvN, which holds the
        value in every lane (a lane inactive at the write kept an older VGPR value; its shadow is computed from
        the same scalars). This is the scalar recompute of upstream PR #2203 in the one-lane model: the value the
        first active lane reads is rebuilt from the scalar expression chain that defined it, so a chain such as
        the unsigned-division reciprocal (v_cvt_f32_u32 v1, sN; v_rcp_f32 v1, v1; v_mul_f32 v1, k, v1;
        v_cvt_u32_f32 v1, v1; v_readfirstlane_b32) under a narrowed exec is read by every lane."""
        partial = self.partial_uniform
        epoch = getattr(self, 'exec_writes', 0)
        deps = []

        def sub(m):
            k = int(m.group(1))
            if k in self.uniform_v:
                return m.group(0)
            entry = partial.get(k)
            if entry is None or entry['epoch'] != epoch:
                raise Unsupported('not partial')
            deps.append(entry)
            return 'xv%d' % k
        try:
            shadow = re.sub(r'\bv(\d+)\b', sub, expr)
        except Unsupported:
            return None
        return (shadow, deps) if self.lane_invariant(shadow) else None

    def materialize_shadow(self, entry):
        """Rewrite a partial_uniform write (and the writes its shadow reads) to also assign its shadow xvN in every
        lane, in program order, at the write itself."""
        if entry['done']:
            return
        for dep in entry['deps']:
            self.materialize_shadow(dep)
        n = entry['n']
        entry['lines'][entry['at']] = 'xv%d = %s; if (exec & 1) v%d = xv%d;' % (n, entry['expr'], n, n)
        self.shadow_v.add(n)
        entry['done'] = True

    def f_mods(self, expr, mods=()):
        # The hardware applies the output modifier before the clamp, but the disassembler prints `clamp`
        # first, so the order of `mods` is not the order of the operations: scale first, saturate last.
        scale = 1.0
        clamp = False
        for m in mods:
            if m == 'clamp':
                clamp = True
            elif m in ('mul:2', 'mul:4'):
                scale *= float(m[-1])
            elif m == 'div:2':
                scale *= 0.5
            elif m.startswith('op_sel'):
                # op_sel selects which half of a packed source the operation reads. This path does not
                # model that, so dropping it would change the operand: refuse instead of ignoring it.
                raise Unsupported('modifier %s' % m)
            else:
                raise Unsupported('modifier %s' % m)
        if scale != 1.0:
            factor = ('%d.0' % int(scale)) if scale == int(scale) else ('%g' % scale)
            expr = '(%s) * %s' % (expr, factor)
        if clamp:
            expr = 'saturate(%s)' % expr
        return expr

    def set_vf(self, n, expr, mods=()):
        self.set_v(n, 'U(%s)' % self.f_mods(expr, mods))

    def scalar_expr_uniform(self, expr):
        # These identifiers are runtime inputs in the scalar expressions emitted below.
        # In particular, a compare can carry a lane mask into an otherwise scalar select.
        for token in re.findall(r'\b(?:s\d+|v\d+|vcc|exec|m0|scc)\b', expr):
            if token == 'scc':
                if not self.scc_uniform:
                    return False
            elif not self.uniform_source(token):
                return False
        return True

    def set_scc(self, expr, proof=None):
        """Emit an SCC assignment and track its uniformity from `proof` (an expression over the
        sources) when the emitted text names a register, such as m0, that the proof cannot judge."""
        self.scc_effect = 'emitted'
        if self.tracks_uniformity():
            self.scc_uniform = self.scalar_expr_uniform(expr if proof is None else proof)
        self.emit('scc = %s;' % expr)

    def set_s(self, n, expr, sym=None, scc=None, uniform=None):
        """Assign scalar register n at run time and/or symbolically.

        Any write ends n's life as a lane mask; a caller that writes a mask marks n again after this call.
        The uniformity of expr is judged first, while a source that is n itself still has its old mark."""
        if self.tracks_uniformity() and (sym is None or sym[0] == 'const'):
            if uniform is None:
                uniform = self.scalar_expr_uniform(expr)
            if uniform:
                self.varying_s.discard(n)
                if self.defined_uniform_s is not None:
                    self.defined_uniform_s.add(n)
                if self._uniform_written is not None:
                    self._uniform_written.add(n)
            else:
                self.varying_s.add(n)
                if self.defined_uniform_s is not None:
                    self.defined_uniform_s.discard(n)
                if self._uniform_written is not None:
                    self._uniform_written.discard(n)
                if self.bad_sgprs is not None:
                    self.bad_sgprs.add(n)
            # Invariant: since_join gains an SGPR only here, when set_s proves it uniform; forget_writes drops any SGPR
            # an instruction writes first, so a write through an unmodelled path never keeps a stale mark.
            if getattr(self, 'since_join', None) is not None:
                (self.since_join.add if uniform else self.since_join.discard)(n)
        getattr(self, 'masks', set()).discard(n)  # unit probes build a Stage without __init__
        getattr(self, 'splat', set()).discard(n)
        if sym is not None:
            self.sym[n] = sym
            if sym[0] != 'const':
                return
        else:
            self.sym.pop(n, None)
        if hasattr(self, 'lane_s'):
            if self.lane_invariant(expr):
                self.lane_s.discard(n)
            else:
                self.lane_s.add(n)
            self.s_written_at.setdefault(self.pc, set()).add(n)
        self.used_s.add(n)
        self.emit('s%d = %s;' % (n, expr))
        if scc:
            self.set_scc(scc.replace('$', 's%d' % n))

    def descriptor(self, tok):
        """Resource key for a descriptor operand like s[8:11]: (source, byte offset or None)."""
        r = reg_range(tok)
        s = self.sym.get(r[1]) if r else None
        if r and r[0] == 's' and r[1] <= 107 and r[2] >= 106:
            if getattr(self, 'has_branches', False):
                expected = tuple((i - 106, self.sym.get(i)) for i in range(max(106, r[1]), min(107, r[2]) + 1))
                self.vcc_events.setdefault(self.pc, []).append(('read_descriptor', expected))
            elif getattr(self, 'vcc_descriptor_overwritten', False):
                raise Unsupported('descriptor operand %s overlaps a redefined vcc' % tok)
        if s and s[0] == 'desc':
            # A descriptor can start inside a wider load (s_load_dwordx16 fetches two image
            # descriptors): its offset is the load's offset plus the first register's dword index.
            return s[1], (None if s[2] is None else s[2] + 4 * s[3])
        if s and s[0] == 'ptr':
            if len(s) > 2 and s[2]:
                raise Unsupported('descriptor operand %s is a pointer with offset %s' % (tok, hex(s[2]) if isinstance(s[2], int) else s[2]))
            return s[1], None
        raise Unsupported('descriptor operand %s holds %s' % (tok, s))

    def heap_sampler(self, tok):
        """'qh[...]' when the four sampler words of tok came from one run-time sampler heap load."""
        r = reg_range(tok)
        if not r or r[2] - r[1] != 3:
            return None
        first = self.sym.get(r[1])
        if not first or first[0] != 'smpheap':
            return None
        if any(self.sym.get(r[1] + i) != ('smpheap', first[1], i) for i in range(4)):
            raise Unsupported('sampler operand %s mixes sampler heap words' % tok)
        self.pipe.uses_sampler_heap = True
        # Sampler descriptors are 16 bytes; x2w_sampler_entry() maps the byte offset to its entry and
        # sends unaligned or out-of-range offsets to the fallback entry.
        return 'qh[x2w_sampler_entry(s%d)]' % first[1]

    def const_words(self, tok):
        r = reg_range(tok)
        fconst = getattr(self, 'float_const_s', {})
        words = [self.is_const(n) if self.is_const(n) is not None else fconst.get(n)
                 for n in range(r[1], r[2] + 1)]
        return None if any(w is None for w in words) else words

    def execz_refusals(self):
        """{offset: reason} for compute blocks that s_cbranch_execz skips and the one-lane model cannot run.

        The wave runs such a block when any lane is active, so its scalar results are the same for every lane; the
        model skips the block for an inactive lane. That lane can differ from the wave only by reading, after the
        join, scalar state the block defines (no lane active means the wave skipped the block too). The block is
        therefore accepted unchanged when every scalar register, mask and SCC it may define is dead where it joins,
        and it has no effect outside scalar registers and no branch out of it. An exec write is accepted when it
        cannot set the bit of a lane whose bit was clear at the branch: clear_after carries, over every path through
        the block (its branches and loops included), which masks have that lane's bit clear, and the write must
        leave exec among them. Such a lane's bit then stays clear at every instruction the wave runs in the block,
        so the wave changes nothing of that lane there and its exec at the join is the model's. The proof starts at
        the branch, so a block that a branch from outside enters keeps its exec writes refused. A live definition,
        an unproven exec write, a refused effect, a branch leaving the block or a backward s_cbranch_execz is
        refused.
        One live definition is accepted: the i1-phi lane mask of the compiler's if-conversion, a 32-bit M (vcc_lo
        or one sN) zeroed before the branch (`s_mov_b32 M, 0`) and set in the block from exec only (`s_mov_b32 M,
        exec_lo`, `s_and_b32 M, X, exec_lo` in either operand order). Every write of M in the block yields a value
        whose bit is 0 for any lane whose exec bit is clear there: the clear-mask proof keeps exec clear for that lane, and a copy of exec, or X & exec, has that bit clear. M is 0
        at the branch on every path (nothing branches or lands between the zeroing and the branch). So for a lane
        the model skips, the wave's M has that lane's bit 0 whether the wave ran the block or not, which is the
        model's M (0); a lane that runs the block runs it in the model too. This holds only while M is read per
        lane. A whole-mask read (s_cbranch_vccz/vccnz, s_cmp, a plain copy, a data operand) sees the other lanes'
        bits, which the one-lane model does not have, so any such read of M reachable after the join, before M is
        rewritten, keeps the refusal. The reads accepted are the lane select of v_cndmask, the carry-in of v_addc /
        v_subb / v_add_co_ci style ops, and s_and_saveexec / s_and_b32 with exec."""
        offsets = [ins[0] for ins in self.ins]
        index = {off: i for i, off in enumerate(offsets)}
        effects = [scalar_effects(name, asm) for _, _, name, asm in self.ins]
        succ = []
        jumps = []  # (instruction, target) of every branch, the proven calculated s_setpc_b64 ones included
        for i, (off, n, name, asm) in enumerate(self.ins):
            to = int(parse(asm)[1][0].split('_')[1], 16) if name.startswith(('s_cbranch', 's_branch')) else None
            if name == 's_setpc_b64':  # a proven calculated jump reaches its target
                to = getattr(self, 'calculated_pc', {}).get(off)
            nxt = [i + 1] if i + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm') else []
            if to is not None:
                jumps.append((i, to))
            succ.append(nxt + ([index[to]] if to in index else []))
        live_in = [set() for _ in self.ins]
        changed = True
        while changed:
            changed = False
            for i in reversed(range(len(self.ins))):
                out = set().union(*(live_in[s] for s in succ[i])) if succ[i] else set()
                reads, writes, _ = effects[i]
                value = reads | (out - writes)
                if value != live_in[i]:
                    live_in[i], changed = value, True
        refused = {}
        for b, (off, n, name, asm) in enumerate(self.ins):
            if name != 's_cbranch_execz':
                continue
            t = index.get(int(parse(asm)[1][0].split('_')[1], 16))
            if t is None or t <= b:
                refused.setdefault(off, 's_cbranch_execz to a block before it')
                continue
            live = live_in[t] | {'m0'}  # m0 is also read implicitly; treat it as live
            # Facts before each instruction the wave can reach from the branch: the largest sets every path keeps.
            entry = frozenset({('exec', 0), ('exec_hi', 1)})
            before = {b + 1: entry} if b + 1 < t else {}
            entered = any(index.get(target) in range(b + 1, t) for k, target in jumps if not b < k < t)
            changed = not entered
            while changed:
                changed = False
                for i in range(b + 1, t):
                    if i in before:
                        after = frozenset(clear_after(*self.ins[i][2:], effects[i][2], before[i])) | entry
                        for s in succ[i]:
                            if b < s < t and before.get(s, after) & after != before.get(s):
                                before[s], changed = before.get(s, after) & after, True
            targets = [target for _, target in jumps]
            straight = not any(self.ins[k][2].startswith(('s_cbranch', 's_branch')) for k in range(b + 1, t)) and \
                not any(index.get(target) in range(b + 1, t) for target in targets)
            cand = {k for i in range(b + 1, t) for k in (effects[i][2] - EXEC_KEYS) & live if re.fullmatch(r'vcc|s\d+', k)}
            ok_masks = {k for k in cand if straight and self.phi_mask_ok(k, b, t, effects, succ, targets, index)}
            for i in range(b + 1, t):
                at, _, inner, inner_asm = self.ins[i]
                may = effects[i][2]
                reason = None
                if inner.startswith(EXECZ_REFUSED):
                    reason = 'effect outside scalar registers'
                elif inner.startswith(('s_cbranch', 's_branch')) and not (
                        b < index.get(int(parse(inner_asm)[1][0].split('_')[1], 16), -1) <= t):
                    reason = 'branch out of the block'
                elif may & EXEC_KEYS and (entered or i in before and
                                          not entry <= clear_after(inner, inner_asm, may, before[i])):
                    reason = 'exec write'
                elif (may - EXEC_KEYS - ok_masks) & live:
                    reason = 'definition of %s, live where the block joins' % ', '.join(
                        sorted((may - EXEC_KEYS - ok_masks) & live))
                if reason:
                    refused.setdefault(at, '%s inside a block s_cbranch_execz skips per lane: %s' % (inner, reason))
        return refused

    def phi_mask_ok(self, key, b, t, effects, succ, targets, index):
        """True when lane mask `key` ('vcc' for vcc_lo, or 'sN'), live at the join t of the s_cbranch_execz at b, is
        the zeroed-then-set-from-exec shape execz_refusals accepts (see its docstring): every write of it in the
        block is a 32-bit move of exec_lo or an s_and with exec_lo into it alone, the last write before the branch
        is `s_mov_b32 M, 0` with no branch and no branch target after it up to the branch, and after the join it is
        only read per lane. The caller has checked that the block is straight-line."""
        name = 'vcc_lo' if key == 'vcc' else key
        for i in range(b + 1, t):
            if key not in effects[i][2]:
                continue
            _, ops, _ = parse(self.ins[i][3])
            base = re.sub(r'_e(32|64)$', '', self.ins[i][2])
            if not ops or ops[0] != name:
                return False
            if base == 's_mov_b32' and ops[1:] == ['exec_lo']:
                continue
            if base == 's_and_b32' and len(ops) == 3 and 'exec_lo' in ops[1:]:
                continue
            return False
        # The zeroing: the last instruction before the branch that may write M.
        z = next((k for k in range(b - 1, -1, -1) if key in effects[k][2]), None)
        if z is None:
            return False
        _, ops, _ = parse(self.ins[z][3])
        if self.ins[z][2] != 's_mov_b32' or ops != [name, '0']:
            return False
        if any(self.ins[k][2].startswith(('s_cbranch', 's_branch', 's_setpc')) for k in range(z + 1, b)) or \
                any(index.get(x) in range(z + 1, b + 1) for x in targets):
            return False
        # After the join, until M is rewritten, every read must be per lane.
        seen, todo = set(), [t]
        while todo:
            i = todo.pop()
            if i in seen:
                continue
            seen.add(i)
            reads, writes, _ = effects[i]
            if key in reads and not self.per_lane_read(i, key):
                return False
            if key not in writes:
                todo.extend(succ[i])
        return True

    def per_lane_read(self, i, key):
        """True when instruction i reads mask `key` only as a lane select or carry, or into exec (s_and_saveexec,
        s_and with exec): the lane's own bit is all it uses."""
        _, _, name, asm = self.ins[i]
        base = re.sub(r'_e(32|64)$', '', name)
        _, ops, _ = parse(asm)
        if base.startswith(('v_cndmask', 'v_addc', 'v_subb', 'v_add_co_ci', 'v_sub_co_ci', 'v_subrev_co_ci')):
            # the mask is the last operand and appears nowhere else (not as a data word, not as the destination)
            return len(ops) >= 4 and key in scalar_tokens(ops[-1]) and \
                all(key not in scalar_tokens(op) for op in ops[:-1])
        if base == 's_and_saveexec_b32':
            return len(ops) == 2 and key in scalar_tokens(ops[1]) and key not in scalar_tokens(ops[0])
        if base == 's_and_b32':
            return len(ops) == 3 and key not in scalar_tokens(ops[0]) and any(
                is_exec(scalar_tokens(op)) for op in ops[1:])
        return False

    # ---- instruction translation -------------------------------------------------------------
    # Symbolic values that are also assigned to their HLSL variable, so a path where the register holds one of them
    # and a path where it holds a plain run-time number agree on the variable's value.
    RUNTIME_BACKED = ('const', 'fixed_buffer_word', 'execcopy')

    def snapshot(self):
        return dict(self.sym), set(self.masks), self.exec_known, set(self.lane_s)

    def merge(self, states):
        """The symbolic state at a join of forward edges: what every incoming path agrees on.

        A register whose paths disagree keeps working as a number when every path also assigned it at run time;
        otherwise it becomes ('conflict',), which any later use refuses (a descriptor on one path and a loaded
        word on the other has no single translation). A mask on any path stays a mask, and exec is the entry
        mask only when it is on every path.
        """
        sym = {}
        self.execzero = {}  # conflict registers symbolic only on s_cbranch_execz edges -> exec_writes at the join
        for reg in set().union(*(s[0] for s in states)):
            values = [s[0].get(reg) for s in states]
            if all(v == values[0] for v in values):
                sym[reg] = values[0]
            elif not all(v is None or v[0] in self.RUNTIME_BACKED for v in values):
                # Symbolic only on s_cbranch_execz edges: the run-time variable is right wherever exec is 1, so an
                # AND with exec before exec is written again may read it (see scalar); any other use refuses.
                symbolic = [s for s, v in zip(states, values) if not (v is None or v[0] in self.RUNTIME_BACKED)]
                sym[reg] = ('conflict',)
                if all(any(s is e for e in getattr(self, 'execz_snaps', ())) for s in symbolic):
                    self.execzero[reg] = getattr(self, 'exec_writes', 0)
        self.sym = sym
        self.masks = set().union(*(s[1] for s in states))
        self.exec_known = all(s[2] for s in states)
        self.lane_s = set().union(*(s[3] for s in states))  # lane-dependent on any incoming path

    # Instructions whose SGPR result is a lane mask (bit i = lane i): compares into an SGPR pair, save-exec forms,
    # wrexec forms and a copy of exec.
    LOOP_MASK_WRITER = re.compile(r'^v_cmpx?_|saveexec|wrexec')

    def loop_writes(self, head, last):
        """(SGPRs written, SGPRs written by something other than a lane-mask writer, exec written) over the
        instructions at offsets [head, last]."""
        written, non_mask, exec_written = set(), set(), False
        for off, n, name, asm in self.ins:
            if not head <= off <= last:
                continue
            _, writes = sgpr_access(name, asm)
            ops = parse(asm)[1]
            first = ops[0] if ops else ''
            if writes:
                written |= writes
                mask = self.LOOP_MASK_WRITER.search(name) or (
                    name.startswith('s_mov_b') and len(ops) > 1 and ops[1].startswith('exec'))
                if not mask:
                    non_mask |= writes
            if (first.startswith('exec') and name.startswith('s_')) or re.search(r'saveexec|wrexec|^v_cmpx_', name):
                exec_written = True
        return written, non_mask, exec_written

    def forget_loop_writes(self, head, last):
        """At a loop head: drop what the entry state says about registers the loop body writes (see translate)."""
        written, non_mask, exec_written = self.loop_writes(head, last)
        for reg in written:
            value = self.sym.pop(reg, None)
            if value is not None and value[0] not in self.RUNTIME_BACKED:
                # The entry value lives only in the symbolic state (a descriptor, a pointer): its HLSL variable was
                # never assigned, so a first-pass read of the run-time variable would be wrong. As at a forward
                # join of a symbolic and a run-time value, any use refuses.
                self.sym[reg] = ('conflict',)
        self.masks -= non_mask
        if exec_written:
            self.exec_known = False

    def unjoined(self, target, last_write, regs):
        """The registers of regs whose program-order state is not proven at a branch target. A lane-mask mark or a
        constant follows program order, so it holds on every path into the target only if the block of the register's
        last write dominates the target and no block on a path from that write to the target writes the register
        again."""
        index, block_of, succ, dom, writes, reached = self.flow
        def after(block):  # blocks reached from block over at least one edge
            if block not in reached:
                seen, todo = set(), list(succ[block])
                while todo:
                    b = todo.pop()
                    if b not in seen:
                        seen.add(b)
                        todo += succ[b]
                reached[block] = seen
            return reached[block]
        t = block_of[index[target]]
        into = {b for b in range(len(succ)) if t in after(b)}
        out = set()
        for reg in regs:
            w = block_of[index[last_write[reg]]] if reg in last_write else 0
            if w not in dom[t] or any(reg in writes[b] for b in (after(w) & into) - {w}):
                out.add(reg)
        return out

    def translate_mesh_program(self):
        refusals = getattr(self, 'refusals', None)  # unit probes build a Stage without __init__
        vcc_branches = getattr(self, 'mesh_vcc_branches', {})
        vcc_compares = set().union(*vcc_branches.values())
        try:
            self.calculated_pc = calculated_pc_proof(self.ins, getattr(self, 'kind', None))
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
            self.calculated_pc = {}
        targets, sources = set(), {}
        skipped = []  # compute: (branch, target) of each s_cbranch_execz
        targets.update(target for target in self.calculated_pc.values() if target is not None)
        for at, target in self.calculated_pc.items():
            if target is not None:
                sources.setdefault(target, []).append(at)
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                target = int(parse(asm)[1][0].split('_')[1], 16)
                targets.add(target)
                sources.setdefault(target, []).append(off)
                if name == 's_cbranch_execz' and self.kind == 'cs':
                    skipped.append((off, target))
        # The first branch or branch target: before it the program is one straight-line path (op_s_round_mode).
        self.uses_round_mode = any(name == 's_round_mode' for off, n, name, asm in self.ins)
        # X2W_PS_QUAD_DIVERGENT=1 (default off, read when the stage is built): quad reads (DPP quad_perm) and
        # image_sample_b are also lowered where a lane may skip them in forward control flow. That is exact only
        # where the four lanes of each quad take the same path (D3D leaves quad reads and implicit derivatives
        # undefined otherwise, while the hardware reads the inactive lanes, fi:1); loops stay refused. Each use is
        # listed in quad_divergent_used and the manifest says `quaddivergent 1`.
        self.quad_divergent = self.kind == 'ps' and os.environ.get('X2W_PS_QUAD_DIVERGENT') == '1'
        self.quad_divergent_used = []
        self.has_loop = any(src >= t for t, srcs in sources.items() for src in srcs)
        self.cf_first = min([off for off, n, name, asm in self.ins if name.startswith('s_cbranch') or name == 's_branch']
                            + list(targets) + [1 << 32])
        if self.kind == 'ms' and self.uses_round_mode:
            # A mesh request supplies no launch FLOAT_MODE. Limit the first supported profile to
            # the common mixed-precision emitters that consult round16. Mesh-specific d16 image
            # packing and the other half-precision forms require their own audit.
            for _, _, op, _ in self.ins:
                if re.search(r'[fd]16|f64', op) and op not in ('v_fma_mixlo_f16', 'v_fma_mixhi_f16'):
                    raise Unsupported('mesh round mode with %s' % op)
            self.launch_unknown = True
        self.mode_at = self.round_mode_flow() if self.kind in ('ps', 'ms') and self.uses_round_mode else {}
        self.blocks = []          # (offset, [lines], terminator)
        self.waterfall_back = set()  # back edges of proven waterfall loops, emitted as fall-through (waterfall)
        self.execz_snaps = []    # the snapshots on taken s_cbranch_execz edges (see merge)
        self.uniform_v = set()    # VGPRs proven to hold one value in every lane (see set_v)
        self.partial_uniform = {}  # VGPRs holding one value in every lane active at their write (see set_v)
        self.shadow_v = set()     # shadow variables xvN that v_readfirstlane_b32 introduced
        self.lane_s = set()       # SGPR variables that may differ between lanes in the one-lane model (see set_s)
        self.s_written_at = {}    # instruction offset -> SGPRs it assigns at run time
        self.varying_s = set()
        self.scc_uniform = True
        self.low16_only_v = set()   # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
        self.pinned_low16_v = set() # low16_only_v live across basic-block boundaries
        self.whole_read_v = set()   # VGPRs read outside a proven low-16-bit consumer
        self.has_branches = bool(targets)
        self.uniform_expr = {}    # proven VGPR -> uniform HLSL expression over scalars that defined it
        self.since_join = None    # SGPRs proven uniform since the last branch target (None: none reached)
        start = 0
        incoming = {}             # branch target -> symbolic states of the forward branches seen so far
        previous = None
        self.execz_fallthrough_epoch = None
        scc_written, scc_unknown = set(), set()
        self.entry_table_pointers = {reg: sym[1] for reg, sym in self.sym.items()
                                     if isinstance(reg, int) and sym[0] == 'ptr' and sym[1][0] == 'table'}
        self.descriptor_block_first = 0
        for instruction_index, (off, n, name, asm) in enumerate(self.ins):
            self.descriptor_instruction_index = instruction_index
            if off in targets or off == start:
                self.descriptor_block_first = instruction_index
            if off in targets and off != start:
                self.pinned_low16_v.update(self.low16_only_v)
                self.blocks.append((start, self.lines, None))
                self.lines, start = [], off
            # The scan is linear, so at a join the state left by the instruction above is only one of the
            # incoming paths (and none after an unconditional branch). Merge the states of every incoming
            # edge when they are all forward; a loop head merges its entry edges and forgets the body's writes.
            if off in targets and all(src < off for src in sources[off]):
                states = list(incoming.get(off, []))
                if previous not in (None, 's_branch', 's_endpgm'):
                    states.append(self.snapshot())
                if states:
                    self.merge(states)
            elif off in targets:
                # A loop head: the entry edges (fall-through and forward branches) are merged as at any join, and
                # the back edges carry whatever the loop body left. Every SGPR the body writes anywhere in
                # [head, last back-edge source] loses its symbolic value (a constant known on entry is the first
                # pass's value only: a counter `s14 = 0` must not fold into `0 + 1` on every pass), a register the
                # body writes with anything but a lane mask stops being a mask, and exec is no longer the entry
                # mask if the body writes it.
                states = list(incoming.get(off, []))
                if previous not in (None, 's_branch', 's_endpgm'):
                    states.append(self.snapshot())
                if states:
                    self.merge(states)
                self.forget_loop_writes(off, max(sources[off]))
            if off in targets:
                # Lane invariance is not merged across edges for VGPRs; at a loop head the linear state misses the
                # back edges, so every SGPR becomes suspect until it is written again.
                self.uniform_v.clear()
                self.splat.clear()
                self.partial_uniform.clear()
                self.a_invariant = False
                self.t_invariant = False
                if not all(src < off for src in sources[off]):
                    self.lane_s = set(range(128))
                else:
                    # In the one-lane model a branch condition (scc, vccz, execz) may differ between lanes, so an
                    # SGPR assigned on only some paths into this join may too, however invariant each value is.
                    first = min(sources[off])
                    for at, regs in self.s_written_at.items():
                        if first <= at < off:
                            self.lane_s |= regs
            if off in targets:
                self.execz_fallthrough_epoch = None
            self.pc = off
            self.past_entry = bool(self.blocks) or 0 in targets
            if off in targets:
                self.since_join, self.scc_uniform = set(), False
            before = len(self.lines)
            try:
                term = self.one(name, asm)
            except Unsupported as reason:
                if refusals is None:
                    raise
                refusals.append((str(reason), self.kind))
                continue
            except Exception as error:
                if refusals is None:
                    raise
                refusals.append((handler_error(name, error, refusals), self.kind))
                continue
            if any('scc = ' in line for line in self.lines[before:]):
                scc_written.add(off)
            elif re.sub(r'_e(32|64)$', '', name) in SCC_UNMODELLED or (name in SCA and SCA[name][1]):
                # a hardware SCC write the HLSL did not make (or, for an SCA op with an SCC rule, one that a
                # descriptor or m0 path skipped)
                scc_unknown.add(off)
            previous = name
            if name.startswith('s_cbranch') or name == 's_branch':
                snap = self.snapshot()
                if name == 's_cbranch_execz':
                    self.execz_snaps.append(snap)  # the taken edge: this lane's exec bit is 0 there (merge)
                incoming.setdefault(int(parse(asm)[1][0].split('_')[1], 16), []).append(snap)
            if term:
                self.pinned_low16_v.update(self.low16_only_v)
                self.blocks.append((start, self.lines, term))
                self.lines, start = [], off + n
        if self.lines:
            self.blocks.append((start, self.lines, None))
        self.check_scc(scc_written, scc_unknown)
        self.check_quad_ops()
        self.pinned_low16_v.update(self.low16_only_v)
        leak = self.pinned_low16_v & self.whole_read_v
        if leak:
            reason = 'whole-word read of v%d after odd d16 image_load' % min(leak)
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))

    def translate(self):
        if getattr(self, 'kind', None) == 'ms' and not getattr(getattr(self, 'pipe', None), 'uses_mesh_skeleton', False):
            return self.translate_mesh_program()

        refusals = getattr(self, 'refusals', None)  # unit probes build a Stage without __init__
        vcc_branches = getattr(self, 'mesh_vcc_branches', {})
        wave_limits = getattr(self, 'mesh_wave_limits', {})
        vcc_compares = set().union(*vcc_branches.values())
        try:
            self.calculated_pc = calculated_pc_proof(self.ins, getattr(self, 'kind', None))
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
            self.calculated_pc = {}
        targets = set()
        targets.update(target for target in self.calculated_pc.values() if target is not None)
        # (branch, target) of every branch, the proven s_setpc_b64 jumps included
        branches = [(pc, target) for pc, target in self.calculated_pc.items() if target is not None]
        for off, n, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                target = int(parse(asm)[1][0].split('_')[1], 16)
                targets.add(target)
                branches.append((off, target))
        # Targets that only forward s_cbranch_execz branches (and the fall-through) enter: on a taken edge the lane's
        # exec bit is 0 at the target. Compute stages have their own execz handling and are left out.
        sources = {}
        for pc, target in branches:
            sources.setdefault(target, []).append(pc)
        # The first branch or branch target: before it the program is one straight-line path (op_s_round_mode).
        self.uses_round_mode = any(name == 's_round_mode' for off, n, name, asm in self.ins)
        # X2W_PS_QUAD_DIVERGENT=1 (default off, read when the stage is built): quad reads (DPP quad_perm) and
        # image_sample_b are also lowered where a lane may skip them in forward control flow. That is exact only
        # where the four lanes of each quad take the same path (D3D leaves quad reads and implicit derivatives
        # undefined otherwise, while the hardware reads the inactive lanes, fi:1); loops stay refused. Each use is
        # listed in quad_divergent_used and the manifest says `quaddivergent 1`.
        self.quad_divergent = getattr(self, 'kind', None) == 'ps' and os.environ.get('X2W_PS_QUAD_DIVERGENT') == '1'
        self.quad_divergent_used = []
        self.has_loop = any(src >= t for t, srcs in sources.items() for src in srcs)
        self.cf_first = min([off for off, n, name, asm in self.ins if name.startswith('s_cbranch') or name == 's_branch']
                            + list(targets) + [1 << 32])
        if getattr(self, 'kind', None) == 'ms' and self.uses_round_mode:
            # A mesh request supplies no launch FLOAT_MODE. Limit the first supported profile to
            # the common mixed-precision emitters that consult round16. Mesh-specific d16 image
            # packing and the other half-precision forms require their own audit.
            for _, _, op, _ in self.ins:
                if re.search(r'[fd]16|f64', op) and op not in ('v_fma_mixlo_f16', 'v_fma_mixhi_f16'):
                    raise Unsupported('mesh round mode with %s' % op)
            self.launch_unknown = True
        self.mode_at = self.round_mode_flow() if getattr(self, 'kind', None) in ('ps', 'ms') and self.uses_round_mode else {}
        names = {off: name for off, _n, name, _asm in self.ins}
        execz_targets = set() if getattr(self, 'kind', None) == 'cs' else {
            t for t in targets if all(pc < t and names[pc] == 's_cbranch_execz' for pc, to in branches if to == t)}
        self.mesh_allocation_block = None
        splits = set(targets)
        if getattr(self, 'mesh_vcc_branches', None) is not None and targets:
            at = next(i for i, item in enumerate(self.ins) if item[2] == 'x2w_mesh_counts')
            self.mesh_allocation_block = self.ins[at][0]
            # The exchange and its EXEC restore always run. Keep the group barrier and allocation outside
            # the pc dispatch even when the preceding region took different paths.
            splits.update((self.mesh_allocation_block, self.ins[at + 2][0]))
        join_writes = self.join_writes(branches)
        refused = self.execz_refusals() if getattr(self, 'kind', None) == 'cs' else {}  # probes skip __init__
        joined, forward_splits, div_targets = self.joined_registers(with_cfg=True)
        init_sym = dict(self.sym)
        init_exec_known = getattr(self, 'exec_known', True)
        init_wqm_exec = getattr(self, 'wqm_exec', False)
        init_masks = set(getattr(self, 'masks', ()))
        init_attrs = dict(getattr(self, 'attrs', {}))
        init_used_s = set(getattr(self, 'used_s', ()))
        init_used_v = set(getattr(self, 'used_v', ()))
        init_entry_uniform = set(getattr(self, 'entry_uniform_s', ())) | {
            k for k, v in init_sym.items()
            if isinstance(k, int) and v and (
                (v[0] == 'const' and len(v) == 2) or v[0] in ('ud', 'fixed_buffer_word')
            )
        }
        entry_ud = {reg: sym for reg, sym in init_sym.items() if sym[0] == 'ud'}
        forced = getattr(self, 'force_joined', {})
        force_unproven = set()
        force_at = {}
        first_refusal = len(refusals) if refusals is not None else 0
        every_sgpr = set(range(106)) | {'m0'}
        init_entry = [r for r in set(init_sym) | init_used_s if isinstance(r, int)]
        init_written = set(range(max(init_entry + [-1 if getattr(self, 'kind', None) == 'cs' else 15]) + 1)) | {'m0'}

        for _ in range(max(4, len(self.ins) + 2)):
            scc_effects = {}
            uniform_vcc_compares = set()
            self.lines = []
            self.blocks = []          # (offset, [lines], terminator)
            self.sym = dict(init_sym)
            self.exec_known = init_exec_known
            self.wqm_exec = init_wqm_exec
            self.masks = set(init_masks)
            self.join_masks = set()   # marked SGPRs whose mark is not proven on every path (unjoined)
            last_write = {}           # SGPR -> offset of its last write in program order
            self.attrs = dict(init_attrs)
            self.used_s, self.used_v = set(init_used_s), set(init_used_v)
            self.exports = {}
            self.write_masks = {}
            self.compr_params = set()
            self.plain_params = {}
            self.f16_plain = {}
            self.uses_pos = False
            self.fetched = False
            self.waterfall_back = set()
            self.execz_snaps = []
            self.partial_uniform = {}
            self.shadow_v = set()
            self.lane_s = set()
            self.s_written_at = {}
            self.execz_fallthrough_epoch = None
            self.exec_writes = 0
            self.a_invariant = self.t_invariant = False
            self.splat = set()
            self.quad_divergent_used = []
            scc_written, scc_unknown = set(), set()
            self.uniform_v = set()
            self.varying_s = set()
            self.scc_uniform = True
            self.low16_only_v = set()   # VGPRs whose upper 16 bits are undefined after odd-dmask d16 image_load
            self.pinned_low16_v = set() # low16_only_v live across basic-block boundaries
            self.whole_read_v = set()   # VGPRs read outside a proven low-16-bit consumer
            self.float_const_s = {}     # SGPR -> 32-bit pattern from s_mov_b32 of an inline float constant
            self.has_branches = bool(targets)
            self.vcc_events = {}
            self.entry_table_pointers = {
                reg: sym[1] for reg, sym in init_sym.items()
                if isinstance(reg, int) and sym[0] == 'ptr' and sym[1][0] == 'table'
            }
            self.descriptor_block_first = 0
            self.uniform_expr = {}    # proven VGPR -> uniform HLSL expression over scalars that defined it
            self.since_join = None    # SGPRs proven uniform since the last branch target (None: none reached)
            self.defined_uniform_s = set(init_entry_uniform)
            self.bad_sgprs = set()
            bad_at_target = {}
            tentative_kept = set()
            materialize = {}
            incoming = {}
            kept = {}
            self.back_force = {}
            written = set(init_written)
            falls = True
            start = 0
            raised_unsupported = None

            for instruction_index, (off, n, name, asm) in enumerate(self.ins):
                if off in splits and off != start:
                    self.pinned_low16_v.update(self.low16_only_v)
                    self.blocks.append((start, self.lines, None))
                    self.lines, start = [], off
                    self.descriptor_block_first = instruction_index
                if off in targets:
                    self.partial_uniform.clear()
                    self.splat.clear()
                    self.a_invariant = self.t_invariant = False
                    self.execz_fallthrough_epoch = None
                    if any(src >= off for src in sources[off]):
                        self.lane_s = set(range(128))
                    else:
                        for at, regs in self.s_written_at.items():
                            if min(sources[off]) <= at < off:
                                self.lane_s.update(regs)
                self.pc = off
                self.descriptor_instruction_index = instruction_index
                self.scc_effect = None
                # Before the first branch target or terminator every path has run exactly the instructions above, so
                # the program-order tracking is exact there even in a program with branches. Code after s_endpgm in a
                # branch-free program is past it too (dead code).
                self.past_entry = bool(self.blocks) or 0 in targets
                if off in targets:
                    pre_uniform = {
                        r for r in (self.defined_uniform_s if self.since_join is None else self.since_join)
                        if r not in self.varying_s and r not in self.masks
                    }
                    kept[off] = {}
                    here = joined.get(off, set())
                    arriving = ([({reg: self.sym.get(reg) for reg in here}, written & here)] if falls else []) + \
                        incoming.get(off, [])
                    written = (written - here) | set().union(*(w for _, w in arriving))
                    for reg in here:
                        paths = [s[reg] for s, w in arriving if reg in w]
                        force = forced.get(off, {}).get(reg)
                        if any(reg not in w for _, w in arriving) and any(p and not run_time_symbol(p) for p in paths):
                            self.sym[reg] = ('joined at a branch target',)
                            continue
                        if force is None and paths and names_one_value(paths[0]) and all(p == paths[0] for p in paths):
                            self.sym[reg] = kept[off][reg] = paths[0]
                            continue
                        if force == 'refuse' or any(p and not run_time_symbol(p) for p in paths):
                            self.sym[reg] = ('joined at a branch target',)
                            continue
                        if reg in entry_ud:
                            materialize[reg] = 'ud%d' % entry_ud[reg][1]
                        self.sym.pop(reg, None)
                        if reg != 'm0':
                            self.used_s.add(reg)
                    # A joined register the fall-through wrote as a plain run-time number: every other path into
                    # this target has exec bit 0, so the instruction at the target may AND it with exec (scalar()). A register that still holds a plain
                    # variable reads the same either way.
                    self.exec_zero_join = (off, {
                        reg for reg in here
                        if falls and off in execz_targets and reg in arriving[0][1] and arriving[0][0][reg] is None})
                    clobber = here if off in joined else every_sgpr
                    self.join_masks |= self.unjoined(off, last_write, self.masks)
                    fwd_clobber = forward_splits.get(off, clobber)
                    blocked = force_unproven | force_at.get(off, set())
                    kept_u = {
                        r for r in pre_uniform
                        if r not in clobber or (r not in fwd_clobber and r not in blocked)
                    }
                    tentative_kept.update((kept_u & clobber) - blocked)
                    self.uniform_v.clear()
                    self.since_join, self.scc_uniform = kept_u, False
                    # A constant is only known here if no other path into the join writes the register: a loop counter
                    # set before the loop and incremented in it reaches the loop head with a new value on every pass.
                    # set_s also assigns constants at run time, so the register reads its real value once it is dropped.
                    # clobber also holds the writes of every earlier block off the dominator chain; a constant written
                    # after those, by a block every path runs, with no later write on any path, is still known.
                    consts = {reg for reg in join_writes.get(off, set()) | (clobber & every_sgpr)
                              if (self.sym.get(reg) or ('',))[0] == 'const'}
                    for reg in self.unjoined(off, last_write, consts):
                        del self.sym[reg]
                    fconsts = (join_writes.get(off, set()) | (clobber & every_sgpr)) & set(self.float_const_s)
                    for reg in self.unjoined(off, last_write, fconsts):
                        self.float_const_s.pop(reg, None)
                    self.defined_uniform_s = set(kept_u) | {
                        k for k, v in self.sym.items()
                        if isinstance(k, int) and v and (
                            (v[0] == 'const' and len(v) == 2) or v[0] in ('ud', 'fixed_buffer_word')
                        )
                    }
                if name.startswith('s_cbranch') or name == 's_branch' or (
                        name == 's_setpc_b64' and self.calculated_pc.get(off) is not None):
                    b_target = self.calculated_pc[off] if name == 's_setpc_b64' else int(parse(asm)[1][0].split('_')[1], 16)
                    there = joined.get(b_target, set())
                    if b_target > off:
                        incoming.setdefault(b_target, []).append(({reg: self.sym.get(reg) for reg in there}, written & there))
                    else:
                        for reg in there & written:
                            back, sym = self.sym.get(reg), kept.get(b_target, {}).get(reg)
                            if back and not run_time_symbol(back) and back != sym:
                                self.back_force.setdefault(b_target, {})[reg] = 'refuse'
                            elif sym is not None and back != sym:
                                self.back_force.setdefault(b_target, {}).setdefault(
                                    reg, 'run' if run_time_symbol(sym) else 'refuse')
                falls = name not in ('s_branch', 's_setpc_b64', 's_endpgm')
                dests = scalar_destinations(name, asm)
                written |= every_sgpr if 'any' in dests else dests
                last_write.update(dict.fromkeys(every_sgpr if 'any' in dests else dests, off))
                self._pre_fconst = dict(self.float_const_s)
                if 'any' in dests:
                    self.float_const_s.clear()
                else:
                    for reg in dests:
                        self.float_const_s.pop(reg, None)
                try:
                    if off in refused:
                        raise Unsupported(refused[off])
                    base = re.sub(r'_e(32|64)$', '', name)
                    if off in vcc_compares and all(
                        not mesh_vregs([tok]) and self.uniform_source(tok)
                        for tok in parse(asm)[1][1:]
                    ):
                        uniform_vcc_compares.add(off)
                    if base.startswith('s_cbranch_'):
                        cond = base[10:]
                        cond_uniform = (cond in ('scc0', 'scc1') and self.scc_uniform) or off in wave_limits
                        if off in vcc_branches:
                            if not vcc_branches[off] <= uniform_vcc_compares:
                                raise Unsupported('slot 57 mesh: %s on a wave-wide condition is not covered' % name)
                            cond_uniform = True
                        if not cond_uniform:
                            b_target = int(parse(asm)[1][0].split('_')[1], 16)
                            regs, affected = div_targets.get(off, (set(), set()))
                            regs = {r for r in regs if isinstance(r, int)}
                            if b_target <= off:
                                regs.update(r for r in joined.get(b_target, ()) if isinstance(r, int))
                            for tgt_off in affected:
                                bad_at_target.setdefault(tgt_off, set()).update(regs)
                    before = len(self.lines)
                    term = self.one(name, asm)
                    if any('scc = ' in line for line in self.lines[before:]):
                        self.scc_effect = 'emitted'
                        scc_written.add(off)
                    elif base in SCC_UNMODELLED or (name in SCA and SCA[name][1]):
                        scc_unknown.add(off)
                    if off in wave_limits:
                        term = ('branch', '(tid & ~63u) >= %du' % wave_limits[off], term[2])
                except Unsupported as reason:
                    if refusals is None:
                        if not hasattr(reason, 'instruction'):
                            reason.instruction = name  # for fail_reason(); the message itself is unchanged
                        raised_unsupported = reason
                        break
                    refusals.append((str(reason), self.kind))
                    continue
                except Exception as error:
                    if refusals is None:
                        raise
                    refusals.append((handler_error(name, error, refusals), self.kind))
                    continue
                # A write proves the mark again: every path from here on runs it. s_movreld can write any SGPR.
                self.join_masks = (self.join_masks | every_sgpr) if 'any' in dests else self.join_masks - dests
                if self.scc_effect:
                    scc_effects[off] = self.scc_effect
                if term:
                    self.pinned_low16_v.update(self.low16_only_v)
                    self.blocks.append((start, self.lines, term))
                    self.lines, start = [], off + n
                    self.descriptor_block_first = instruction_index + 1

            new_bad = self.bad_sgprs - force_unproven
            new_at = {k: v - force_at.get(k, set()) for k, v in bad_at_target.items() if v - force_at.get(k, set())}
            if (new_bad or new_at) and tentative_kept:
                force_unproven.update(new_bad)
                for k, v in new_at.items():
                    force_at.setdefault(k, set()).update(v)
                if refusals is not None:
                    del refusals[first_refusal:]
                continue
            if raised_unsupported is not None:
                raise raised_unsupported
            break
        else:
            reason = 'v_readfirstlane_b32 uniformity proof exceeded pass limit'
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))
        if self.lines:
            self.blocks.append((start, self.lines, None))
        self.check_quad_ops()
        self.materialized = materialize  # declarations() assigns these before the body (and before any loop)
        if set(scc_effects.values()) - {'emitted'}:
            for kind, name in self.descriptor_scc_reads(dict(branches), scc_effects):
                reason = 'scc from %s in %s' % ('descriptor data' if kind == 'descriptor' else kind, name)
                if refusals is None:
                    raise Unsupported(reason)
                refusals.append((reason, self.kind))
        self.pinned_low16_v.update(self.low16_only_v)
        try:
            self.check_vcc_paths()
        except Unsupported as reason:
            if refusals is None:
                raise
            refusals.append((str(reason), self.kind))
        leak = self.pinned_low16_v & self.whole_read_v
        if leak:
            reason = 'whole-word read of v%d after odd d16 image_load' % min(leak)
            if refusals is None:
                raise Unsupported(reason)
            refusals.append((reason, self.kind))


    def descriptor_scc_reads(self, branches, effects):
        """Find reads reachable with an elided SCC write, including through joins and back edges."""
        instructions = {off: (size, name, asm) for off, size, name, asm in self.ins}
        pending = [(0, None)]  # stale: None, or the scc_effect of the write that left the emitted scc behind
        seen = set()
        while pending:
            off, stale = pending.pop()
            if (off, stale) in seen or off not in instructions:
                continue
            seen.add((off, stale))
            size, name, asm = instructions[off]
            if (stale and off not in self.calculated_pc and off not in getattr(self, 'ptr_addc_pcs', ())
                    and reads_scc(name, asm)):
                yield stale, name
            if off in effects:
                stale = effects[off] if effects[off] != 'emitted' else None
            if off in branches:
                pending.append((branches[off], stale))
            if name not in ('s_branch', 's_setpc_b64', 's_endpgm'):
                pending.append((off + size, stale))

    def check_scc(self, written, unknown):
        """Refuse when an SCC value the translation does not compute can be read.

        The instructions in `unknown` write SCC in hardware but the HLSL leaves `scc` as it was. That is only
        a wrong translation if some path from one of them reaches an SCC reader before an instruction whose
        translation assigns `scc` (one in `written`). Paths follow fall-through and branch targets, loops
        included; s_endpgm and s_branch end the fall-through.
        """
        if not unknown:
            return
        order = [ins[0] for ins in self.ins]
        info = {off: (name, asm) for off, n, name, asm in self.ins}
        following = dict(zip(order, order[1:]))
        calc = getattr(self, 'calculated_pc', None) or {}

        def successors(off):
            name, asm = info[off]
            if off in calc:
                return [calc[off]] if calc[off] in info else [following[off]] if off in following else []
            if name == 's_endpgm':
                return []
            nxt = [following[off]] if off in following else []
            if name == 's_branch':
                nxt = []
            if name.startswith('s_cbranch') or name == 's_branch':
                nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
            return [o for o in nxt if o in info]
        for start in sorted(unknown):
            if start in calc:
                continue
            seen, todo = set(), list(successors(start))
            while todo:
                off = todo.pop()
                if off in seen:
                    continue
                seen.add(off)
                name, asm = info[off]
                if reads_scc(name, asm) and off not in calc:
                    raise Unsupported('SCC of %s at %#x (not modelled) is read by %s at %#x'
                                      % (info[start][0], start, name, off))
                if off in written or off in unknown:
                    continue
                todo.extend(successors(off))

    def tracks_uniformity(self):
        return all(hasattr(self, name) for name in ('uniform_v', 'varying_s', 'scc_uniform'))

    def uniform_source(self, tok):
        r = reg_range(tok)
        if r:
            kind, lo, hi = r
            if lo != hi:
                return False
            if kind == 'v':
                return lo in self.uniform_v
            if lo in getattr(self, 'masks', ()):
                return False
            if getattr(self, 'since_join', None) is not None and lo not in self.since_join:
                return False  # written before the last join: program-order tracking is not exact there
            sym = self.sym.get(lo)
            if sym:
                return sym[0] in ('const', 'ud', 'fixed_buffer_word')
            return lo not in self.varying_s
        if tok in FLOATS:
            return True
        try:
            int(tok, 0)
        except ValueError:
            return False
        return True

    def one(self, name, asm):
        _, ops, mods = parse(asm)
        if name.endswith('_e32') and name[:-4] in VOP3_ONLY_VEC_U:
            raise Unsupported(name)
        base = re.sub(r'_e(32|64)$', '', name)
        if base.startswith('s_setreg') and 'HW_REG_MODE' in asm and (self.uses_round_mode or self.round16):
            raise Unsupported('%s writes MODE in a program that uses s_round_mode or a non-zero launch mode' % base)
        # Compute the proof before lowering, since a destination can also be a source.
        dst = None
        self._uniform_written = set()
        if self.tracks_uniformity() and hasattr(self, 'uniform_expr'):
            wrote_s = {r for r in self.forget_writes(name, ops, drop_since=False) if isinstance(r, int)}
        else:
            wrote_s = set()
        # Unit probes build a Stage without __init__: without the tracking state, prove nothing.
        # Past the entry block of a program with branches the read recomputes the defining expression, so the
        # proof no longer needs a known exec (see op_v_readfirstlane_b32).
        if (base in UNIFORM_VALU and ops and self.tracks_uniformity() and
                (getattr(self, 'past_entry', False) and hasattr(self, 'uniform_expr') or
                 not getattr(self, 'past_entry', True) and getattr(self, 'exec_known', False)) and
                (not mods or (base in VEC_F or base in VEC_U_TO_F) and
                 all(m in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods))):
            candidate = reg_range(ops[0])
            if candidate and candidate[0] == 'v' and candidate[1] == candidate[2]:
                sources = ops[1:]
                if base in VEC_F:
                    sources = [re.sub(r'^-(?=[^\d.])', '', tok).strip('|') for tok in sources]
                if sources and all(self.uniform_source(tok) for tok in sources):
                    dst = candidate[1]
        self.last_v = None
        term = self.lower(base, asm, ops, mods, name)
        uniform_written = self._uniform_written
        if self.tracks_uniformity() and getattr(self, 'since_join', None) is not None and wrote_s:
            self.since_join -= (wrote_s - uniform_written)
        if wrote_s and self.bad_sgprs is not None:
            for r in wrote_s - uniform_written:
                sym = self.sym.get(r)
                if (r in self.varying_s or r in getattr(self, 'masks', ()) or
                        (sym is not None and sym[0] not in ('const', 'ud', 'fixed_buffer_word')) or
                        (self.defined_uniform_s is not None and r not in self.defined_uniform_s)):
                    if self.defined_uniform_s is not None:
                        self.defined_uniform_s.discard(r)
                    self.bad_sgprs.add(r)
        if dst is not None:
            self.uniform_v.add(dst)
            if hasattr(self, 'uniform_expr'):
                # Proven sources are scalars, constants and VGPRs; each VGPR needs its own current expression.
                expr = self.last_v[1] if self.last_v and self.last_v[0] == dst else None
                vs = [int(n) for n in re.findall(r'\bv(\d+)\b', expr or '')]
                if expr is None or any(n not in self.uniform_expr for n in vs):
                    self.uniform_expr.pop(dst, None)
                else:
                    self.uniform_expr[dst] = re.sub(r'\bv(\d+)\b', lambda m: '(%s)' % self.uniform_expr[int(m[1])],
                                                    expr)
        return term

    def dpp_pair(self, base, asm, ops, mods):
        """A DPP16 spelling whose SRC0 is a quad pair broadcast (#2125 item C).

        RDNA 2 ISA section 12.19 (DPP16): lane i of a quad reads SRC0 of lane quad_perm[i] of the same quad. A pair
        broadcast ([1,1,3,3], [0,0,2,2], [2,3,2,3], [0,1,0,1]) moves one lane of every horizontal or vertical pair
        into the other and leaves the first lane alone. The word of each of the four lanes is read with the shader
        model 6.0 QuadReadLaneAt (its lane argument must be a constant, so all four reads are unconditional) and this
        lane keeps the one quad_perm names for its own position in the quad. That needs all four lanes of the quad to
        run the instruction in the same flow, as in a pixel shader with helper lanes: straight-line code and a
        whole-quad exec mask (proved by s_wqm exec, exec on the entry mask; a restored exact mask loses it). Every
        other control (other permutations, row_*, wave_*, a partial row or bank mask, a vertex stage) is refused by
        name, since the single lane cannot read a neighbour."""
        perms = [m for m in mods if m.startswith('quad_perm:')]
        rest = [m for m in mods if not m.startswith('quad_perm:')]
        if len(perms) != 1:
            raise Unsupported('%s_dpp with %s' % (base, ' '.join(mods)))
        for m in rest:
            if m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        # With all four lanes active no source lane is disabled, so bound_ctrl has nothing to replace.
        perm = tuple(int(c) for c in re.findall(r'\d', perms[0]))
        if perm not in DPP_PAIR:
            raise Unsupported('%s_dpp with %s' % (base, perms[0]))
        if (self.kind != 'ps' or getattr(self, 'has_branches', True) or not getattr(self, 'exec_known', False) or
                not getattr(self, 'wqm_exec', False) or getattr(self, 'fetched', False)):
            raise Unsupported('%s_dpp %s outside straight-line pixel code with a whole-quad exec mask (s_wqm exec, exec)'
                              % (base, perms[0]))
        if len(ops) < 2:
            raise Unsupported('%s_dpp with %d operands' % (base, len(ops)))
        # The permuted word goes to a scratch register (v256, beyond the 256 hardware registers), so a second source
        # naming the same register, or a destination that is the source, still reads the right lane.
        src = re.fullmatch(r'([-|]*)v(\d+)(\|?)', ops[1])
        if not src:
            raise Unsupported('%s_dpp source %s' % (base, ops[1]))
        word = self.u('v%s' % src.group(2))  # records the original source as a used VGPR
        self.needs_sm6 = True
        self.used_v.add(DPP_SCRATCH)
        self.uniform_v.discard(DPP_SCRATCH)
        self.emit('{ uint x2w_q0 = QuadReadLaneAt(%s, 0), x2w_q1 = QuadReadLaneAt(%s, 1),'
                  ' x2w_q2 = QuadReadLaneAt(%s, 2), x2w_q3 = QuadReadLaneAt(%s, 3);'
                  ' uint x2w_lane = ((uint(i.pos.y) & 1u) << 1u) | (uint(i.pos.x) & 1u);'
                  ' v%d = x2w_lane == 0u ? x2w_q%d : x2w_lane == 1u ? x2w_q%d : x2w_lane == 2u ? x2w_q%d : x2w_q%d; }'
                  % ((word,) * 4 + (DPP_SCRATCH,) + perm))
        ops = [ops[0], '%sv%d%s' % (src.group(1), DPP_SCRATCH, src.group(3))] + ops[2:]
        return self.lower(base, asm, ops, [])

    def forget_writes(self, name, ops, drop_since=True):
        """Before an instruction runs: an EXEC write ends every recomputable expression, and an SGPR write ends the
        expressions that read it and (when drop_since is set) that SGPR's since-join mark (set_s marks it again when it proves the new value).
        The VGPR proofs stay: the entry-block read copies the VGPR, which neither write changes."""
        if writes_exec(name, ops):
            self.uniform_expr.clear()
        regs = set(range(128)) if name.startswith('s_movreld') else set()  # M0 picks the destination
        for tok in destinations(name, ops):
            r = reg_range(tok)
            if r and r[0] == 's':
                regs.update(range(r[1], r[2] + 1))
        if not regs:
            return set()
        if drop_since and self.since_join is not None:
            self.since_join -= regs
        for v in [v for v, expr in self.uniform_expr.items()
                  if regs & {int(n) for n in re.findall(r'\bs(\d+)\b', expr)}]:
            del self.uniform_expr[v]
        return regs

    def lower(self, base, asm, ops, mods, name=None):
        if name is None:
            name = base
        pc = getattr(self, 'pc', None)
        calculated = getattr(self, 'calculated_pc', {})
        if pc in calculated:
            target = calculated[pc]
            if target is not None:
                return ('jump', target)
            # The proof checked that the code address is never read as data.
            # Do not materialize it as a fictional integer or keep an older
            # entry constant/resource/lane-mask proof for the overwritten SGPR.
            for reg in scalar_registers(ops[0]):
                self.sym.pop(reg, None)
                self.masks.discard(reg)
                self.varying_s.add(reg)
            if base in ('s_add_u32', 's_addc_u32'):
                self.scc_uniform = False
            return None
        if getattr(self, 'fetched', False):  # unit probes build a Stage without __init__
            # Behind a fetch call the definite-assignment guard relies on exec_known: every exec write must clear it.
            # s_wqm of exec into itself only adds lanes, so it stays a no-op; other s_wqm forms (a no-op here) and
            # exec writes outside the handlers that clear exec_known are refused. Other programs keep their handling.
            if base in ('s_wqm_b32', 's_wqm_b64') and not (ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1]):
                raise Unsupported('%s %s after a fetch call' % (base, ', '.join(ops)))
            if ops and ops[0] in ('exec', 'exec_lo', 'exec_hi') and not base.startswith('s_wqm') and (
                    ops[0] == 'exec_hi' or not (base in SCA or base.startswith('v_cmp') or
                                                (base in ('s_cselect_b32', 's_cselect_b64') and ops[0] == 'exec'))):
                raise Unsupported('%s into %s after a fetch call' % (base, ops[0]))
        if base in ('s_wqm_b32', 's_wqm_b64'):
            if ops[0] in ('exec', 'exec_lo') and ops[1:] == ops[:1] and getattr(self, 'exec_known', False):
                self.wqm_exec = True   # every lane of a touched quad is now active, helper lanes included
            if hasattr(self, 'scc_uniform'):
                self.scc_uniform = False  # the hardware SCC result is not modelled
        if base == 's_waitcnt_depctr':
            if len(ops) != 1 or not re.fullmatch(r'0x[0-9a-f]+|[0-9]+', ops[0]) or int(ops[0], 0) > 65535 or mods:
                raise Unsupported('s_waitcnt_depctr operands')
            return None
        if base in ('s_inst_prefetch', 's_waitcnt', 's_nop', 'v_nop', 's_clause', 's_sopp_0f', 's_wqm_b64', 's_wqm_b32',
                    's_setreg_imm32_b32', 's_code_end', 's_setprio'):
            return None
        if base == 's_endpgm':
            return ('end',)
        if base == 's_branch':
            return ('jump', int(ops[0].split('_')[1], 16))
        if base == 's_cbranch_scc1' and getattr(self, 'pc', None) in getattr(self, 'waterfall_back', ()):
            self.lines.append('// waterfall back edge: not taken for this lane (one pass, see waterfall)')
            return None
        if base.startswith('s_cbranch_'):
            if base[10:].startswith('vcc'):
                self.read_vcc()
            if (getattr(getattr(self, 'pipe', None), 'uses_mesh_skeleton', False) and self.kind == 'ms'
                    and base in ('s_cbranch_scc0', 's_cbranch_scc1') and not self.scc_uniform):
                raise Unsupported('slot 57 mesh: a branch reads SCC that is not proven wave-uniform')
            cond = {'scc0': '!scc', 'scc1': 'scc', 'vccz': '(vcc & 1) == 0', 'vccnz': '(vcc & 1) != 0',
                    'execz': '(exec & 1) == 0', 'execnz': '(exec & 1) != 0'}[base[10:]]
            return ('branch', cond, int(ops[0].split('_')[1], 16))
        if base == 'exp':
            return self.export(asm)
        modes = self.mode_at.get(self.pc) if self.mode_at else None
        if modes is not None and self.rounds16(name):
            # the MODE[3:2] values that reach this instruction on every path (round_mode_flow)
            if '?' in modes:
                raise Unsupported('%s: f16/f64 rounding with the launch FLOAT_MODE not passed (.psin third field)'
                                  % name)
            if modes != {self.round16}:
                raise Unsupported('%s at %#x reached under MODE f16 roundings %s (the translation uses %d)'
                                  % (name, self.pc, sorted(modes, key=str), self.round16))
        if self.round16:
            self.check_round16(name)
        elif self.launch_unknown and modes is None:
            self.check_launch_known(name)
        if base.endswith('_dpp'):
            return self.quad_dpp(base[:-len('_dpp')], ops, mods)
        if base in (*LDS_EXCHANGE_OPS, 's_barrier') and getattr(self, 'kind', None) == 'cs':
            # The refusal census continues past failed instructions. Scan this program only once.
            if not hasattr(self, 'lds_error'):
                self.lds_error = None
                try:
                    if getattr(self, 'pipe', None) is None:
                        raise Unsupported('LDS exchange needs pipeline metadata')
                    check_lds_exchange(self.ins, self.pipe.group, self.pipe.lds_words)
                    self.uses_lds = True
                except Unsupported as reason:
                    self.lds_error = str(reason)
            if self.lds_error is not None:
                raise Unsupported(self.lds_error)
            if base == 's_barrier':
                self.emit('GroupMemoryBarrierWithGroupSync();')
            else:
                write = base.startswith('ds_write')
                address = self.u(ops[0 if write else 1])
                offsets = lds_byte_offsets(base, mods)
                if base in ('ds_write_b32', 'ds_read_b32', 'ds_read2_b32'):
                    if write:
                        self.emit('lds[(%s + %du) >> 2] = %s;' % (address, offsets[0], self.u(ops[1])))
                    else:
                        if base == 'ds_read2_b32':
                            # Either destination can overlap the incoming address VGPR.
                            self.emit('a = %s;' % address)
                            address = 'a'
                        start = reg_range(ops[0])[1]
                        for index, offset in enumerate(offsets):
                            self.set_v(start + index, 'lds[(%s + %du) >> 2]' % (address, offset))
                else:
                    dwords = 4 if base.endswith('_b128') else 2
                    lo = _lds_vgpr_span(ops[1 if write else 0], dwords)
                    if write:
                        vals = [self.u('v%d' % (lo + k)) for k in range(dwords)]
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        for k, val in enumerate(vals):
                            self.emit('lds[%s] = %s;' % ('a' if k == 0 else 'a + %du' % k, val))
                    else:
                        self.emit('a = (%s + %du) >> 2;' % (address, offsets[0]))
                        for k in range(dwords):
                            self.set_v(lo + k, 'lds[%s]' % ('a' if k == 0 else 'a + %du' % k))
            return None
        if base == 's_ff1_i32_b32' and ops[1:] == ['exec_lo'] and getattr(self, 'kind', None) == 'cs':
            dst = reg_range(ops[0])
            if mods or not dst or dst[0] != 's' or dst[1] != dst[2] or dst[1] > 105:
                raise Unsupported('s_ff1_i32_b32 requires one unmodified SGPR destination')
            if self.past_entry or not self.exec_known:
                raise Unsupported('s_ff1_i32_b32 requires the compute entry EXEC mask')
            # Even the final partial group starts with lane zero active. FF1 preserves SCC.
            self.set_s(dst[1], '0u', ('const', 0))
            return None
        handler = getattr(self, 'op_' + base, None)
        if handler:
            return handler(ops, mods)
        if base.startswith('v_cmp'):
            return self.compare(base, ops, mods)
        if base.endswith('_sdwa'):
            return self.sdwa(base[:-len('_sdwa')], ops, mods)
        if base in VEC_H:
            return self.half_alu(base, ops, mods)
        if base in VEC_F:
            arity, text = VEC_F[base]
            # A missing source used to raise IndexError and an extra one was dropped without a word.
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            args = [self.f(o) for o in ops[1:1 + arity]]
            if base in ('v_fmac_f32', 'v_mac_f32'):
                args.append(self.f(ops[0]))
            if base in FMA_FUSED or base in FMA_UNFUSED:
                # The multiply-add rule (lead 000800 answer; the same rule as mesh2hlsl's): a fused op is one
                # rounding, an unfused one rounds the product first; both made deterministic so a depth pre-pass and
                # a colour pass of the same geometry agree bit for bit. Recorded in the manifest as `fma <rule>`.
                pipe = getattr(self, 'pipe', None)  # the tests' bare stages have none
                if hasattr(pipe, 'fma_rule'):
                    pipe.fma_rule = DRAW_FMA
                helper = 'x2w_fma_exact' if base in FMA_FUSED and DRAW_FMA == 'exact' else 'x2w_mad_precise'
                if DRAW_FMA != 'legacy':
                    return self.set_vf(reg_range(ops[0])[1], '%s(%s, %s, %s)' % ((helper,) + tuple(args)), mods)
            return self.set_vf(reg_range(ops[0])[1], text.format(*args), mods)
        if base in VEC_U_TO_F:
            arity, text_f = VEC_U_TO_F[base]
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            # Only the output modifiers a conversion can carry; op_sel and the source modifiers are not
            # accepted here, because ignoring them would silently change the operand.
            if any(m not in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods):
                raise Unsupported('%s with %s' % (base, mods))
            dst = reg_range(ops[0])[1]
            low16 = base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1', 'v_cvt_off_f32_i4')
            self.set_vf(dst, text_f.format(*[self.u(o, low16_only=low16) for o in ops[1:1 + arity]]), mods)
            return None
        if base in VEC_U:
            arity, text = VEC_U[base]
            if len(ops) != arity + 1:
                raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
            if base in VOP3_ONLY_VEC_U:
                dst = reg_range(ops[0])
                if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
                    raise Unsupported('%s destination %s' % (base, ops[0]))
                for src in ops[1:1 + arity]:
                    if self.is_lane_mask(src):
                        raise Unsupported('%s on a lane mask' % base)
                    sr = reg_range(src)
                    if sr is not None:
                        if sr[1] != sr[2] or (sr[0] == 'v' and sr[1] > 255) or (sr[0] == 's' and sr[1] > 105):
                            raise Unsupported('%s source %s' % (base, src))
                    elif src not in FLOATS and src != 'm0':
                        try:
                            val = int(src, 0)
                        except ValueError:
                            raise Unsupported('operand %r' % src)
                        if not -0x80000000 <= val <= 0xffffffff:
                            raise Unsupported('%s source %s' % (base, src))
            if base == 'v_and_b32':
                c0, c1 = self.const_u32(ops[1]), self.const_u32(ops[2])
                args = [self.u(ops[1], low16_only=c1 is not None and (c1 & 0xffff0000) == 0),
                        self.u(ops[2], low16_only=c0 is not None and (c0 & 0xffff0000) == 0)]
            else:
                args = [self.u(o) for o in ops[1:1 + arity]]
            if base in ('v_sad_u8', 'v_sad_u16', 'v_sad_u32'):
                pipe = getattr(self, 'pipe', None)
                if pipe is not None and isinstance(getattr(pipe, 'uses_sad', None), set):
                    pipe.uses_sad.add(base)
            result = text.format(*args)
            if mods:
                # VOP3 output modifiers (omod, clamp) act on a binary32 result, so they are only taken on the
                # integer-source conversions that produce one; set_vf applies omod and then the clamp.
                if base not in FLOAT_RESULT_U or any(m not in ('clamp', 'mul:2', 'mul:4', 'div:2') for m in mods):
                    raise Unsupported('%s with %s' % (base, mods))
                return self.set_vf(reg_range(ops[0])[1], 'F(%s)' % result, mods)
            return self.set_v(reg_range(ops[0])[1], result)
        if base in SCA:
            return self.scalar(base, ops)
        if base == 's_addc_u32':
            # Only the high word of an advanced descriptor pointer; the carry is not modelled.
            hi = reg_range(ops[1]) if len(ops) == 3 else None
            if hasattr(self, 'scc_uniform') and self.tracks_uniformity():
                self.scc_uniform = False
            if hi and hi[0] == 's' and self.pointer_add(base, ops[0], ops[1:], [self.sym.get(hi[1]), None]):
                return None
            raise Unsupported(base)
        if base == 's_getpc_b64':
            # RDNA 2 GETPC returns the address of its successor: the shader's load address plus a known byte
            # offset, which is not an integer this model can emit. It is elided only when the program also
            # calculates a branch target, which calculated_pc_proof() proves and records. Reaching here means no
            # such proof was found, so name the reason instead of reporting the mnemonic alone.
            raise Unsupported('%s: the shader load address is only elided for a program that also calculates a branch target' % base)
        raise Unsupported(base)

    # DPP16 quad permutations in a pixel stage: src0 is read from lane perm[lane & 3] of the lane's 2x2 quad, the
    # rest of the instruction is the plain VOP1/VOP2 operation. LLVM AMDGPUModifierSyntax.md: quad_perm is a "full
    # permute of 4 threads"; fi:1 "fetch pre-existing values from inactive lanes"; bound_ctrl only acts on invalid
    # lanes, which a quad permutation never names. Quad lane 1 is +x and lane 2 is +y (Mesa ACO
    # aco_select_nir_intrinsics.cpp: ddx_coarse = v_subrev_f32 with quad_perm(0,0,0,0) then v_mov_b32
    # quad_perm(1,1,1,1), ddy_coarse with (2,2,2,2)), the order of D3D QuadReadLaneAt (0 top-left, 1 top-right,
    # 2 bottom-left, 3 bottom-right). The HLSL quad reads need SM 6.0 (the mesh pipelines' dxc ps_6_0, which sets
    # pipe.quad_ops; draws compile with fxc ps_5_1 and refuse), and every lane of the quad must execute them:
    # each read is its own statement outside the exec guard, and check_quad_ops refuses one that a lane can skip.
    QUAD_READS = {(1, 0, 3, 2): 'QuadReadAcrossX({0})', (2, 3, 0, 1): 'QuadReadAcrossY({0})',
                  (3, 2, 1, 0): 'QuadReadAcrossDiagonal({0})'}

    def quad_dpp(self, base, ops, mods):
        if base in VOP3_ONLY_VEC_U:
            raise Unsupported('DPP form of %s' % base)
        if self.kind != 'ps' or not getattr(self.pipe, 'quad_ops', False):
            raise Unsupported('%s_dpp' % base)  # as before: no lane reads outside an SM 6.0 pixel stage
        control = mods[0] if mods else 'no control'
        m = re.fullmatch(r'quad_perm:\[([0-3]),([0-3]),([0-3]),([0-3])\]', control)
        if not m:
            raise Unsupported('%s_dpp with %s (not a quad permutation)' % (base, control))
        rest = set(mods[1:])
        if len(rest) != len(mods) - 1 or not {'row_mask:0xf', 'bank_mask:0xf', 'fi:1'} <= rest \
                or rest - {'row_mask:0xf', 'bank_mask:0xf', 'fi:1', 'bound_ctrl:1'}:
            raise Unsupported('%s_dpp with %s' % (base, ' '.join(mods[1:])))
        perm = tuple(int(x) for x in m.groups())
        src = re.fullmatch(r'(-?\|?)(v\d+)(\|?)', ops[1]) if len(ops) > 1 else None
        if not src:
            raise Unsupported('%s_dpp source %s' % (base, ops[1] if len(ops) > 1 else None))
        value = self.u(src.group(2))
        if len(set(perm)) == 1:
            expr = 'QuadReadLaneAt(%s, %du)' % (value, perm[0])
        elif perm in self.QUAD_READS:
            expr = self.QUAD_READS[perm].format(value)
        else:
            raise Unsupported('%s_dpp with %s' % (base, control))
        names = self.__dict__.setdefault('dpp_names', [])
        name = 'dpp%d' % len(names)
        names.append(name)
        self.__dict__.setdefault('dpp_at', []).append(self.pc)
        statement = 'uint %s = %s;' % (name, expr)
        self.__dict__.setdefault('dpp_statements', set()).add(statement)
        self.emit(statement)
        return self.one(base, '%s %s' % (base, ', '.join([ops[0], src.group(1) + name + src.group(3)] + ops[2:])))

    def check_quad_ops(self):
        """Refuse a quad read (quad_dpp) that some lane of the HLSL may not execute: the program has a loop (the
        dispatch loop lets lanes reach a block in different iterations), or a path from the entry to the end
        avoids the instruction (a per-lane branch around it)."""
        at = getattr(self, 'dpp_at', ())
        if not at:
            return
        info = {off: (name, asm) for off, n, name, asm in self.ins}
        order = [ins[0] for ins in self.ins]
        following = dict(zip(order, order[1:]))
        def successors(off):
            name, asm = info[off]
            if name == 's_endpgm':
                return []
            nxt = [] if name == 's_branch' else [following[off]] if off in following else []
            if name.startswith('s_cbranch') or name == 's_branch':
                nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
            return [o for o in nxt if o in info]
        uniform = os.environ.get('X2W_PS_QUAD_UNIFORM') == '1'
        last_quad = max(at)
        for off in order:
            if any(t <= off and (not uniform or t <= last_quad) for t in successors(off) if t != following.get(off)):
                raise Unsupported('quad read (DPP) at %#x in a program with a loop (branch at %#x)' % (at[0], off))
        if uniform:
            # Forward dispatch visits each block once, in guest instruction order. Leave
            # VGPR writes and consumers predicated, but read every lane's retained VGPR
            # outside the guest PC guard (fi:1). A prior discard can disable a source lane.
            killed = False
            for _, lines, _ in self.blocks:
                depth = 0
                for line in lines:
                    if line in self.dpp_statements and (killed or depth):
                        raise Unsupported('uniform quad read after discard or inside a nested scope')
                    killed |= bool(re.search(r'\b(discard|return)\b', line))
                    depth += line.count('{') - line.count('}')
            self.uniform_dpp = self.dpp_statements
            return
        for target in at:
            seen, todo = set(), [order[0]]
            while todo:
                off = todo.pop()
                if off in seen or off == target:
                    continue
                seen.add(off)
                nxt = successors(off)
                if not nxt:
                    if self.quad_divergent:
                        self.quad_divergent_used.append(target)
                        break
                    raise Unsupported('quad read (DPP) at %#x is skipped on a path to %#x' % (target, off))
                todo.extend(nxt)
    # VOP1/VOP2 with DPP16 quad broadcast (#2125 item A): quad_perm:[k,k,k,k] makes every lane read SRC0 from
    # lane k of its own quad (RDNA 2 ISA, "DPP": a lane reads lane (lane & ~3) | perm[lane & 3] of the row; LLVM 15
    # AMDGPU DPP quad_perm). A pixel stage's four consecutive lanes are a 2x2 pixel quad (lane = x & 1 | (y & 1) << 1),
    # which is what SM 6.0 QuadReadLaneAt(value, k) reads, so the stage is compiled as ps_6_0 (Stage.needs_sm6).
    # Accepted only where the whole quad is proven active: a straight-line pixel stage that ran s_wqm exec, exec and
    # whose EXEC is still that mask (helper lanes are active, so no source lane is off). bound_ctrl then has
    # nothing to decide (the source lane is never disabled), so it is accepted and ignored. Everything else a DPP word can say stays refused: other controls (pair
    # broadcasts, row shifts, mirrors, DPP8), a masked row or bank, and a vertex stage (dpp() drops fi:1 first).
    DPP_TEMP = 256   # hidden VGPR that holds the quad read; the hardware has v0..v255 only
    DPP_OUTPUT_MODS = ('clamp', 'mul:2', 'mul:4', 'div:2')

    def dpp_broadcast(self, base, ops, mods):
        fields, rest = {}, []
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('quad_perm', 'row_mask', 'bank_mask', 'bound_ctrl') and value and key not in fields:
                fields[key] = value
            elif m in self.DPP_OUTPUT_MODS:
                rest.append(m)
            else:
                raise Unsupported('%s_dpp with modifier %s' % (base, m))
        lanes = re.fullmatch(r'\[([0-3]),([0-3]),([0-3]),([0-3])\]', fields.get('quad_perm', ''))
        if not lanes or len(set(lanes.groups())) != 1:
            raise Unsupported('%s_dpp %s: only quad_perm:[k,k,k,k] broadcasts are lowered'
                              % (base, fields.get('quad_perm', 'without quad_perm')))
        if fields.get('row_mask') != '0xf' or fields.get('bank_mask') != '0xf':
            raise Unsupported('%s_dpp with a partial row_mask or bank_mask' % base)
        if fields.get('bound_ctrl', '1') != '1':
            raise Unsupported('%s_dpp bound_ctrl:%s' % (base, fields['bound_ctrl']))
        if (self.kind != 'ps' or getattr(self, 'has_branches', True) or getattr(self, 'blocks', [])
                or not getattr(self, 'exec_known', False)):
            raise Unsupported('%s_dpp outside a straight-line pixel stage with the entry EXEC mask' % base)
        if not getattr(self, 'wqm_exec', False):
            # The entry EXEC can leave out lanes of a partly covered quad, and the guest then reads an inactive
            # source lane (bound_ctrl / fi), while QuadReadLaneAt would return a helper lane's value.
            raise Unsupported('%s_dpp without s_wqm exec, exec proving whole-quad activity' % base)
        if base != 'v_mov_b32' and base not in VEC_F and base not in VEC_U and base not in VEC_U_TO_F:
            raise Unsupported('DPP form of %s' % base)
        if len(ops) < 2:
            raise Unsupported('%s_dpp with %d operands' % (base, len(ops)))
        token = ops[1]
        neg = token.startswith('-') and base in VEC_F
        token = token[1:] if neg else token
        absolute = token.startswith('|') and token.endswith('|') and base in VEC_F
        token = token[1:-1] if absolute else token
        source = reg_range(token)
        if not source or source[0] != 'v' or source[1] != source[2]:
            raise Unsupported('%s_dpp source %s is not a VGPR' % (base, ops[1]))
        # Modifiers apply to the value the lane read, so they stay on the rewritten operand.
        rewritten = 'v%d' % self.DPP_TEMP
        rewritten = '|%s|' % rewritten if absolute else rewritten
        rewritten = '-' + rewritten if neg else rewritten
        self.needs_sm6 = True
        self.set_v(self.DPP_TEMP, 'QuadReadLaneAt(%s, %su)' % (self.u(token), lanes.group(1)))
        operands = [ops[0], rewritten] + ops[2:]
        if base == 'v_mov_b32':
            if rest or len(operands) != 2:
                raise Unsupported('v_mov_b32_dpp with %d operands or an output modifier' % len(ops))
            return self.lower(base, '%s %s' % (base, ', '.join(operands)), operands, [])
        return self.lower(base, '%s %s %s' % (base, ', '.join(operands), ' '.join(rest)), operands, rest)

    # VOP1/VOP2 with a DPP16 quad_perm source (AMD RDNA 2 ISA, "Data-Parallel Primitives"): lane i of
    # each aligned group of four reads src0 from lane quad_perm[i] of that group. The three exchanges
    # map onto the SM 6.0 quad intrinsics, which read the other lane of the 2x2 pixel quad and keep
    # helper lanes active. Only a pixel stage at entry EXEC qualifies, and only after `s_wqm exec, exec`
    # proved whole-quad EXEC: the hardware can return 0 or the old value for a source lane whose EXEC bit
    # is off (an uncovered pixel of an edge quad), and the intrinsics have no such case. Other permutations, fi:1 without that proof, DPP8, row shifts and partial masks stay refused.
    def dpp(self, base, ops, mods):
        if base in VOP3_ONLY_VEC_U:
            raise Unsupported('DPP form of %s' % base)
        # fi:1 (fetch inactive, DPP16 dword bit 18; LLVM AMDGPU `fi`, Mesa ACO fetch_inactive) only decides what a
        # disabled source lane returns. Every lowering below is taken only with the whole quad proven active, so no
        # source lane is disabled and it decides nothing.
        mods = [m for m in mods if m != 'fi:1']
        if any(re.fullmatch(r'quad_perm:\[([0-3]),\1,\1,\1\]', mod) for mod in mods):
            return self.dpp_broadcast(base, ops, mods)
        perm = None
        for m in mods:
            match = re.fullmatch(r'quad_perm:\[([0-3]),([0-3]),([0-3]),([0-3])\]', m)
            if match and perm is None:
                perm = tuple(int(x) for x in match.groups())
            elif m not in ('row_mask:0xf', 'bank_mask:0xf', 'bound_ctrl:0', 'bound_ctrl:1'):
                raise Unsupported('%s_dpp with %s' % (base, m))
        if perm in DPP_PAIR:
            return self.dpp_pair(base, '%s %s' % (base, ', '.join(ops)), ops, mods)
        intrinsic = QUAD_SWAPS.get(perm)
        if not intrinsic:
            raise Unsupported('%s_dpp quad_perm %s: only the swaps [1,0,3,2], [2,3,0,1] and [3,2,1,0] are lowered'
                              % (base, perm))
        if base == 'v_mov_b32':
            arity, text, kind = 1, '{0}', 'u'
        elif base in VEC_U:
            (arity, text), kind = VEC_U[base], 'u'
        elif base in VEC_F and base not in ('v_fmac_f32', 'v_mac_f32'):
            (arity, text), kind = VEC_F[base], 'f'
        else:
            raise Unsupported('%s_dpp' % base)
        src_neg_abs = []
        if kind == 'f':
            # Float quad swaps accept the source neg/abs of the plain operation (RDNA 2 ISA 12.19;
            # LLVM 15 AMDGPU DPP prints them on the _dpp spelling). The permuted word is read first,
            # then neg/abs apply, so -vN negates the neighbour lane value. VGPR-only is kept: an SGPR
            # source stays refused below, as does a non-VGPR base.
            if len(ops) != arity + 1 or not re.fullmatch(r'v\d+', ops[0]):
                raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
            for token in ops[1:1 + arity]:
                neg, rest = self.split_neg(token)
                absolute = rest.startswith('|') and rest.endswith('|')
                base_tok = rest[1:-1] if absolute else rest
                if not re.fullmatch(r'v\d+', base_tok):
                    raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
                src_neg_abs.append((neg, absolute, base_tok))
        elif len(ops) != arity + 1 or any(not re.fullmatch(r'v\d+', o) for o in ops):
            raise Unsupported('%s_dpp needs VGPR operands without modifiers' % base)
        if (getattr(self, 'kind', None) != 'ps' or getattr(self, 'has_branches', True) or getattr(self, 'blocks', True)
                or not getattr(self, 'exec_known', False)):
            raise Unsupported('%s_dpp requires a straight-line pixel shader with entry EXEC' % base)
        if not getattr(self, 'wqm_exec', False):
            raise Unsupported('%s_dpp requires whole-quad EXEC (s_wqm of exec first): a partly covered quad has '
                              'inactive source lanes that the hardware masks and QuadReadAcross does not' % base)
        self.needs_sm6 = True
        # src0 is the exchanged lane; a VOP2 src1 stays this lane's own register.
        if kind == 'f':
            neg0, abs0, base0 = src_neg_abs[0]
            swapped = '%s(%s)' % (intrinsic, self.u(base0))
            first = 'F(%s)' % swapped
            if abs0:
                first = 'abs(%s)' % first
            if neg0:
                first = '(-%s)' % first
            args = [first] + [self.f(o) for o in ops[2:1 + arity]]
        else:
            swapped = '%s(%s)' % (intrinsic, self.u(ops[1]))
            args = [swapped] + [self.u(o) for o in ops[2:1 + arity]]
        expr = text.format(*args)
        self.set_v(reg_range(ops[0])[1], 'U(%s)' % expr if kind == 'f' else expr)

    def half_operand(self, tok, kind):
        """Float expression for a 16-bit operand: the low half of the register, or an inline constant."""
        if kind in ('i16', 'u16'):
            if tok in FLOATS:  # an inline float constant is a binary16 word at 16-bit width, not binary32
                word = struct.unpack('<H', struct.pack('<e', FLOATS[tok]))[0]
                src = '%du' % word
            else:
                src = self.u(tok)
            if kind == 'i16':
                return 'float(asint((%s & 0xffffu) << 16u) >> 16)' % src
            return 'float(%s & 0xffffu)' % src
        neg = False
        if tok.startswith('-') and (not re.match(r'-[\d.]', tok) or re.match(r'-0x', tok)):
            neg, tok = True, tok[1:]
        absolute = tok.startswith('|') and tok.endswith('|')
        if absolute:
            tok = tok[1:-1]
        if tok in FLOATS:
            text = repr(struct.unpack('<e', struct.pack('<e', FLOATS[tok]))[0])  # inline constants are binary16 here
        elif tok == '-0':
            text = '0.0'
        else:
            text = 'f16tof32(%s & 0xffffu)' % self.u(tok)
        if absolute:
            text = 'abs(%s)' % text
        return '(-%s)' % text if neg else text

    def half_alu(self, base, ops, mods):
        """16-bit VALU: operands are the low 16 bits, the result fills the low 16 bits and zeroes the rest (RDNA 2)."""
        arity, kind, text, result = VEC_H[base]
        if len(ops) != arity + 1:
            raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), arity + 1))
        if any(m != 'clamp' for m in mods) or ('clamp' in mods and result != 'h'):
            raise Unsupported('%s with %s' % (base, mods))
        args = [self.half_operand(o, kind) for o in ops[1:1 + arity]]
        if base in ('v_fmac_f16', 'v_mac_f16'):
            args.append(self.half_operand(ops[0], 'h'))
        expr = text.format(*args)
        if result == 'h':
            if 'clamp' in mods:
                expr = 'saturate(%s)' % expr
            expr = 'f16rn(%s)' % expr
        self.set_v(reg_range(ops[0])[1], expr)

    # VOP1/VOP2 with sub-dword addressing: select the source fields, run the same operation, then place
    # the low bits of the result in the destination field. The integer forms reuse VEC_U unchanged,
    # v_mov_b32 is the same path with the selected source as its result,
    # VEC_U_TO_F conversions apply output modifiers before dst_sel, VEC_F forms stay refused, and
    # dst_unused UNUSED_SEXT has no lowering.
    def sdwa(self, base, ops, mods):
        if base in ('v_cvt_off_f32_i4', *VOP3_ONLY_VEC_U):
            raise Unsupported('SDWA form of %s' % base)
        fields = {}
        outmods = []
        for m in mods:
            key, _, value = m.partition(':')
            if key in ('src0_sel', 'src1_sel', 'src2_sel', 'dst_sel', 'dst_unused') and value:
                fields[key] = value
            elif m in ('clamp', 'mul:2', 'mul:4', 'div:2') and base in VEC_U_TO_F:
                # The SDWA9 encoding (gfx9+, LLVM VOPInstructions.td VOP_SDWA9 has omod and clamp for VOP1/VOP2)
                # carries the output modifiers; on a float-result conversion they act on the binary32 result as
                # in VOP3. Taken only with dst_sel:DWORD (below), where no field placement follows them.
                outmods.append(m)
            else:
                raise Unsupported('%s with SDWA modifier %s' % (base, m))
        if base == 'v_cndmask_b32':
            # VOP2 SDWA select: the mask is always vcc; the selected source fields are moved as bits.
            if len(ops) != 4 or ops[3] not in ('vcc', 'vcc_lo'):
                raise Unsupported('SDWA v_cndmask_b32 form %s' % ops)
            arity, text = 2, '(vcc & 1) ? {1} : {0}'
        elif base == 'v_mov_b32':
            # VOP1 SDWA move (LLVM VOP1Instructions.td V_MOV_B32 has an SDWA form): the selected source field,
            # placed by dst_sel / dst_unused like every integer SDWA result.
            arity, text = 1, '{0}'
            # A lane mask is one bit per lane here, so a byte or word of it is not the hardware's field.
            if fields.get('src0_sel') == 'DWORD':
                self.reject_lane_mask_data(base, ops[1])
            elif self.is_lane_mask(ops[1].removeprefix('sext(').removesuffix(')')):
                raise Unsupported('v_mov_b32 SDWA %s of lane mask %s' % (fields.get('src0_sel'), ops[1]))
        elif base in VEC_U_TO_F:
            # The float-result conversions moved out of VEC_U (#1707) keep their SDWA form: the same expression
            # as the bits, as VEC_U had it ('U(float({0}))' and so on). No output modifier is accepted above.
            arity, text_f = VEC_U_TO_F[base]
            text = 'U(%s)' % text_f
        elif base not in VEC_U:
            raise Unsupported('SDWA form of %s' % base)
        else:
            arity, text = VEC_U[base]
        sources = []
        for index in range(arity):
            token = ops[1 + index]
            sext = token.startswith('sext(') and token.endswith(')')
            if sext:
                token = token[len('sext('):-1]
            selection = fields.get('src%d_sel' % index)
            if not selection:
                raise Unsupported('%s SDWA without src%d_sel' % (base, index))
            low16 = selection in ('BYTE_0', 'BYTE_1', 'WORD_0') or (
                selection == 'DWORD' and not sext and base in ('v_cvt_f32_f16', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1'))
            sources.append(sdwa_source(self.u(token, low16_only=low16), selection, sext))
        result = text.format(*sources)
        select = fields.get('dst_sel')
        if not select:
            raise Unsupported('%s SDWA without dst_sel' % base)
        if outmods:
            if select != 'DWORD':
                raise Unsupported('%s SDWA with %s and dst_sel:%s' % (base, ' '.join(outmods), select))
            return self.set_vf(reg_range(ops[0])[1], text_f.format(*sources), outmods)
        # The operation's HLSL carries its own operators, so it has to be grouped before the
        # destination field masks it: "v7 | v8 & 0xffff" would otherwise bind as "v7 | (v8 & 0xffff)".
        if SDWA_SEL[select][1] < 32:
            result = '(%s)' % result
        destination = reg_range(ops[0])[1]
        unused = fields.get('dst_unused', 'UNUSED_PAD')
        # UNUSED_PRESERVE reads the destination's other bits: read it as an operand, so a pixel stage's entry
        # (SPI input) VGPR read is recorded like any other read
        old = self.u(ops[0]) if unused == 'UNUSED_PRESERVE' and SDWA_SEL[select][1] < 32 else 'v%d' % destination
        keep_low16 = (unused == 'UNUSED_PRESERVE' and select not in ('DWORD', 'WORD_1') and
                      destination in self.low16_only_v)
        self.set_v(destination, sdwa_dest(result, select, unused, old), low16_only=keep_low16)

    def pointer_add(self, base, dname, srcs, syms):
        """Propagate a descriptor pointer through s_add_u32 sD, sP, <literal> / s_addc_u32 sD+1, sP+1, 0.

        The pair advances a table or root pointer by a literal byte offset, into the same or
        another register pair. The offset rides in the low word as ('ptr', source, offset); the
        high word keeps ('ptrhi', source). The low-word carry is taken as zero: the base is a
        table or root address and the offset is below 1 MiB. Anything else returns False.
        """
        if base not in ('s_add_u32', 's_addc_u32') or len(srcs) != 2 or not syms[0]:
            return False
        if getattr(self, 'has_branches', False):
            # The symbolic update runs once in disassembly order; a branch could skip it or repeat it at run time.
            return False
        reg = reg_range(dname)
        if not reg or reg[0] != 's' or reg[1] != reg[2] or reg[1] > 105:
            return False
        d = reg[1]
        if base == 's_add_u32':
            if syms[0][0] != 'ptr' or not re.fullmatch(r'(?:0[xX][0-9a-fA-F]+|\d+)', srcs[1]):
                return False
            added = int(srcs[1], 0)
            if added > 0xfffff:
                raise Unsupported('descriptor pointer offset %s' % srcs[1])
            before = syms[0][2] if len(syms[0]) > 2 else 0
            self.sym[d] = ('ptr', syms[0][1], before + added)
            if syms[0][1][0] in PTR64_SOURCES and d + 1 <= 105:
                # The carry-in of an s_addc_u32 is only the carry of this very add: remember where the next
                # instruction starts so that the high-word add is accepted right behind it and nowhere else.
                size = next((n for off, n, _name, _asm in getattr(self, 'ins', ()) if off == self.pc), 0)
                self.ptr_low_add = (d, self.pc + size, syms[0][1], before, before + added)
                # The pair stays open until the matching s_addc_u32 completes it. The high register is left as it
                # is (in place it is still the original high source); a load through an open pair is refused.
                self.__dict__.setdefault('ptr_open_pairs', set()).add(d)
            return True
        low = self.sym.get(d - 1)
        mark = getattr(self, 'ptr_low_add', None)
        hi = reg_range(srcs[0])
        # The high operand is judged by what it holds, not by where it sits: a copied high word still counts.
        if (not mark or srcs[1] != '0' or not low or low[0] != 'ptr' or not hi or hi[0] != 's'
                or mark[:2] != (d - 1, getattr(self, 'pc', 0)) or low[1] != mark[2]
                or self.sym.get(hi[1]) != ptr_high_symbol(mark[2], mark[3])):
            return False
        self.sym[d] = ptr_high_symbol(mark[2], mark[4])
        getattr(self, 'ptr_open_pairs', set()).discard(d - 1)
        # This s_addc_u32 is the only reader of the low add's elided carry; any other SCC reader stays refused.
        self.__dict__.setdefault('ptr_addc_pcs', set()).add(self.pc)
        return True

    # scalar ALU with descriptor-word bookkeeping
    def exec_absorbed_source(self, base, srcs):
        """Index of a joined source of the instruction at a branch target whose stale value cannot reach bit 0 of
        the result, or None.

        `s_and_b64 D, S, exec` (either order) and `s_orn2_b64 D, S, exec` are lane-mask operations: the emitted
        result is bit 0 of `S & exec` or `S | ~exec`. When S was written as a run-time number on the fall-through
        and every other path into the target is a taken s_cbranch_execz edge (exec_zero_join), the lane's exec bit
        is 0 on exactly the paths where the variable of S may not hold the register's value, and there the result
        is 0 (or 1 for orn2) whatever S holds. Only the instruction at the target qualifies: exec is unchanged."""
        off, regs = getattr(self, 'exec_zero_join', (None, ()))
        if off is None or off != getattr(self, 'pc', None) or len(srcs) != 2:
            return None
        for i in {'s_and_b64': (0, 1), 's_orn2_b64': (0,)}.get(base, ()):
            reg = reg_range(srcs[i])
            if srcs[1 - i] == 'exec' and reg and reg[0] == 's' and reg[1] in regs:
                return i
        return None

    def scalar(self, base, ops):
        # Every SALU form in SCA is dst, src (the two templates below use {0} and {1}); a short list would
        # index past the end and leave `translator crash in <op>: IndexError` in the census instead of a
        # refusal. s_movk_i32 keeps its own exact check for the immediate it folds.
        if base != 's_movk_i32' and len(ops) < 2:
            raise Unsupported('%s with %d operands, expected 2' % (base, len(ops)))
        if base == 's_movk_i32' and len(ops) != 2:
            raise Unsupported('s_movk_i32 with %d operands, expected 2' % len(ops))
        text, scc = SCA[base]
        if base in SCC_UNMODELLED and self.tracks_uniformity():
            self.scc_uniform = False
        dst = reg_range(ops[0])
        pair = base.endswith('_b64')
        srcs = ops[1:]
        # 64-bit operands name the low register; exec/vcc name themselves.
        def low(tok):
            r = reg_range(tok)
            return ('s%d' % r[1]) if r and r[0] == 's' and r[1] != r[2] else tok
        srcs = [low(t) for t in srcs]
        if base == 's_movk_i32':
            # D.i32 = signext(SIMM16) (RDNA 2 ISA SOPK; LLVM s16imm); disasm_rdna2 prints the field unsigned.
            srcs = [sopk_simm(srcs[0])]
        dname = low(ops[0])
        # Upstream: s_sext_i32_i8 / s_sext_i32_i16 (SCA) need the whole word of their source, not a lane mask.
        pack_or_absdiff = base in PACK_B16 or base == 's_absdiff_i32'
        if pack_or_absdiff or base in ('s_sext_i32_i8', 's_sext_i32_i16'):
            if self.is_lane_mask(dname) or any(self.is_lane_mask(t) for t in srcs):
                raise Unsupported('%s on a lane mask' % base)
            if dname != 'm0' and (dst is None or dst[0] != 's' or dst[1] > 105):
                raise Unsupported('%s destination %s' % (base, dname))
            for source in srcs:
                reg = reg_range(source)
                if reg and (reg[0] != 's' or reg[1] > 105):
                    raise Unsupported('%s source %s' % (base, source))
        word_complement = base in ('s_nand_b32', 's_nor_b32', 's_xnor_b32')
        if word_complement:
            # A lane mask operand or destination is the whole wave's mask, and SCC is set when any bit of the
            # complemented 32-bit result is set; the one-lane projection keeps only this lane's bit.
            def lane_mask(t):
                r = reg_range(t)
                return t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or bool(r and r[0] == 's' and r[1] in self.masks)
            def per_lane(t):
                # a value whose bit i is lane i's: a mask, or 0 / 0xffffffff (every bit the same)
                r = reg_range(t)
                return (lane_mask(t) or t in ('0', '-1') or
                        bool(r and r[0] == 's' and r[1] in getattr(self, 'splat', ())))
            if lane_mask(dname) or any(lane_mask(t) for t in srcs):
                if not all(per_lane(t) for t in srcs):
                    raise Unsupported('%s on a lane mask' % base)
                # Every operand's bit i is lane i's, so the lane's own bit of ~(a op b) is ~(a_i op b_i), which the
                # one-lane value below gives (as for s_not_b32). SCC is not: lanes outside the wave's active set
                # (0 in a compare result) are 1 after the complement. No SCC is written, so check_scc refuses any
                # read of it.
                scc = None
        if (base == 's_andn2_b64' and dst and dst[0] == 's' and dst[2] == dst[1] + 1
                and ops[0] == ops[1] and ops[2] == 'exec' and self.sym.get(dst[1]) == ('conflict',)
                and getattr(self, 'execz_fallthrough_epoch', None) == getattr(self, 'exec_writes', 0)):
            # Each executing lane computes D_i & ~1. Lanes skipping the block already retain
            # their old bit. Leave the wave-wide SCC unmodelled; check_scc rejects a live read.
            self.set_s(dst[1], '0u')
            self.set_s(dst[2], '0u')
            self.masks.add(dst[1])
            return None
        if base in ('s_and_b32', 's_and_b64') and any(t in ('exec', 'exec_lo') for t in srcs) and reg_range(dname):
            # D = D & exec where D is symbolic only on s_cbranch_execz edges into a join (merge) and exec has not
            # been written since: on those paths this lane's exec bit is 0, so the result is 0 whatever D held,
            # and on the others D's run-time variable is its value.
            d0 = reg_range(dname)[1]
            regs = (d0, d0 + 1) if pair else (d0,)
            def execzero(r):
                return (self.sym.get(r) == ('conflict',) and
                        getattr(self, 'execzero', {}).get(r, -1) == getattr(self, 'exec_writes', 0))
            if dname in srcs and all(self.sym.get(r) is None or self.sym[r][0] in self.RUNTIME_BACKED
                                     or execzero(r) for r in regs):
                for r in regs:
                    if execzero(r):
                        self.sym.pop(r)
        syms = [self.sym.get('m0') if t == 'm0' else self.sym.get(reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else None for t in srcs]
        if any(s and s[0] in ('desc', 'descnum', 'ptr', 'ptrhi') for s in syms):
            if pack_or_absdiff:
                raise Unsupported('%s on descriptor data' % base)
            if scc or base in SCC_UNMODELLED:
                self.scc_effect = 'descriptor'
                if self.tracks_uniformity():
                    self.scc_uniform = False
            if word_complement:
                # the descriptor path below records the word symbolically and emits no SCC
                raise Unsupported('scc from descriptor data')
            if self.pointer_add(base, dname, srcs, syms):
                return None
            if dname == 'm0':
                if scc:
                    raise Unsupported('scc from descriptor data')
                self.sym['m0'] = syms[0] if base == 's_mov_b32' else ('descnum',)
                return None
            if dname in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
                raise Unsupported('mask from descriptor data')
            d = reg_range(dname)[1]
            own = self.sym.get(d)
            raw = (raw_descriptor_alu(base, syms, [self.const_u32(t) for t in srcs])
                   if getattr(self, 'kind', None) in ('cs', 'vs') and len(syms) == 2 else None)
            if base in ('s_mov_b32', 's_mov_b64'):
                self.sym[d] = syms[0]
                if pair:
                    high = reg_range(ops[1])[2]
                    hi = self.sym.get(high)
                    if hi and not run_time_symbol(hi):
                        self.sym[d + 1] = hi  # a descriptor word, a pointer or an unassigned constant: symbol only
                    else:
                        # A value the high word holds at run time is copied at run time too, so a join that later
                        # reads the register at run time finds it (a constant keeps its symbol for folding).
                        self.set_s(d + 1, self.S(high))
                        if hi and hi[0] in ('const', 'ud'):
                            self.sym[d + 1] = hi
                        elif self.tracks_uniformity():
                            self.varying_s.add(d + 1)
            elif (own and own[0] == 'ptr' and base in ('s_add_u32', 's_addc_u32')
                  and not getattr(self, 'has_branches', False)):
                # With branches the offset is not tracked: the 'added' mark below stays, and the join refuses it.
                raise Unsupported('%s adds a non-literal to a descriptor pointer' % base)
            elif raw is not None:
                self.sym[d] = raw
                if raw[0] == 'desc':
                    self.pipe.modified.add(desc_key(raw))
            elif own and own[0] in ('desc', 'ptr', 'ptrhi') and any(s == own for s in syms):
                if own[0] == 'desc':
                    self.pipe.modified.add(desc_key(own))
                    self.sym[d] = own[:4] + ('modified',)
                    getattr(self.pipe, 'other_modified', set()).add(desc_key(own))
                    # which word changed and how (buffer_store accepts edits that keep the base address)
                    other = [t for t, s in zip(srcs, syms) if s != own]
                    self.pipe.__dict__.setdefault('modified_words', {}).setdefault((own[1], own[2]), []).append(
                        (own[3], base, tuple(other)))
                else:
                    self.sym[d] = own if len(own) > 2 else own + ('added',)
            else:
                self.sym[d] = ('descnum',)
            return None
        if base.startswith(('s_bcnt', 's_ff0', 's_ff1')) and any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or
                                             (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                                             for t in srcs):
            # a lane count or a lane index: the one-lane model keeps only this lane's bit of a mask
            raise Unsupported('%s of a lane mask' % base)
        # The table templates name their sources as {0}, {1}, ...; a short operand list would make
        # str.format raise IndexError, which the single-shader path records as a translator crash
        # rather than a refusal. The arity guards above cover the table entries that spell a count, so
        # this one covers the templates themselves: every placeholder the template uses must have a
        # source, and a template may not read a source the instruction does not carry.
        needed = max((int(i) for i in re.findall(r'\{(\d+)\}', text)), default=-1) + 1
        if len(srcs) < needed:
            raise Unsupported('%s with %d operands, expected %d' % (base, len(ops), needed + 1))
        absorbed = self.exec_absorbed_source(base, srcs)
        if absorbed is not None:
            self.used_s.add(reg_range(srcs[absorbed])[1])
        values = [t if i == absorbed else self.u(t) for i, t in enumerate(srcs)]
        expr = text.format(*values)
        if dname == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            if scc:
                # Through set_scc, so an SCC derived from a lane mask (s_and_b32 m0, vcc_lo, -1)
                # is tracked as varying like any other scalar result.
                self.set_scc(scc.replace('$', 'm0'), proof=scc.replace('$', '(%s)' % expr))
            return None
        if dname in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            var = dname[:4].rstrip('_') if dname.startswith('exec') else 'vcc'
            if var == 'exec':
                copy = base in ('s_mov_b64', 's_mov_b32') and self.exec_known and (syms[0] == ('execcopy',))
                if copy:
                    # A restored copy can re-activate lanes a WQM added: drop every VGPR proof.
                    self.wqm_exec = False  # and the copy can be older than the WQM
                    if hasattr(self, 'uniform_v'):
                        self.uniform_v.clear()
                    return None
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (var, expr))
            if scc:
                self.set_scc(scc.replace('$', var))
            return None
        d = reg_range(dname)[1]
        is_mask = pair or any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and reg_range(t)[1] in self.masks)
                              for t in srcs) and base.split('_')[1] in ('mov', 'and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not', 'cselect')
        if base in ('s_mov_b32', 's_mov_b64', 's_movk_i32') and re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', srcs[0]) and srcs[0] not in FLOATS:
            value = int(srcs[0], 0)
            self.set_s(d, values[0], ('const', value & 0xffffffff))
            if pair:
                self.set_s(d + 1, '0x%08xu' % ((value >> 32) & 0xffffffff), ('const', (value >> 32) & 0xffffffff))
            self.masks.discard(d)
            return None
        if base in SCALAR_CONST_FOLD:
            # Constant operands give a known word, so a descriptor built from them (an inline sampler: s_movk_i32,
            # s_bfm_b32 (op_s_bfm_b32), s_mov_b32, s_brev_b32 into s[N:N+3]) is a constant descriptor.
            def known(tok):
                r = reg_range(tok)
                if r and r[0] == 's' and r[1] == r[2]:
                    c = self.is_const(r[1])
                    return c if c is not None else getattr(self, '_pre_fconst', getattr(self, 'float_const_s', {})).get(r[1])
                if tok in FLOATS:
                    return fbits(FLOATS[tok])
                if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', tok) and tok not in FLOATS:
                    return int(tok, 0) & 0xffffffff
                return None
            operands = [known(t) for t in srcs]
            if None not in operands:
                value = SCALAR_CONST_FOLD[base](*operands) & 0xffffffff
                self.set_s(d, '0x%08xu' % value, ('const', value))
                return None
        if base in ('s_mov_b64', 's_mov_b32') and srcs[0] in ('exec', 'exec_lo') and self.exec_known:
            self.set_s(d, 'exec')
            self.sym[d] = ('execcopy',)
            self.masks.add(d)
            if pair:
                self.sym.pop(d + 1, None)
                if self.tracks_uniformity():
                    self.varying_s.add(d + 1)
            return None
        if is_mask:
            expr = '(%s) & 1' % expr
        if base == 's_mov_b64':
            r = reg_range(ops[1])
            high = r[2] if r and r[1] != r[2] else None
            # Classify the source high word before a destination write can change its status.
            high_source = 's%d' % high if high is not None else srcs[0]
            high_uniform = (self.tracks_uniformity() and high != d and
                            self.uniform_source(high_source))
        self.set_s(d, expr, scc=scc)
        if is_mask:
            self.masks.add(d)
        if pair and base != 's_mov_b64':
            self.set_s(d + 1, '0u')
        elif pair:
            self.set_s(d + 1, self.S(high) if high is not None else '0u', uniform=high_uniform)
        return None

    def set_exec_scc(self, expr):
        self.exec_known = False
        self.scc_effect = 'emitted'
        self.emit('exec = (%s) & 1; scc = exec != 0;' % expr)
        if hasattr(self, 'scc_uniform'):
            self.scc_uniform = False

    def op_s_and_saveexec_b64(self, ops, mods, text='exec & {0}'):
        if len(ops) != 2:
            raise Unsupported('saveexec with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('saveexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        # S0 is read before D is written: with D == S0 (s_and_saveexec_b64 s[6:7], s[6:7]) the new exec is
        # exec & old S0, not exec & exec (RDNA 2 ISA SOP1: D = EXEC; EXEC = S0 & EXEC, with S0 the value read at issue).
        if reg_range(ops[1]) and reg_range(ops[1])[0] == 's' and reg_range(ops[1])[1] == d:
            self.uses_xsave = True
            self.lines.append('xsave = %s;' % self.u(src))  # every lane: a temporary read only below
            value = 'xsave'
        else:
            value = self.u(src)
        self.set_s(d, 'exec')
        self.masks.add(d)
        self.exec_known = False
        self.emit('exec = (%s) & 1; scc = exec != 0;' % text.format(value))
        if hasattr(self, 'scc_uniform'):
            self.scc_uniform = False

    op_s_and_saveexec_b32 = op_s_and_saveexec_b64

    def op_s_or_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, 'exec | {0}')

    op_s_or_saveexec_b32 = op_s_or_saveexec_b64

    def op_s_andn2_saveexec_b64(self, ops, mods):
        self.op_s_and_saveexec_b64(ops, mods, '{0} & ~exec')

    def op_s_andn2_wrexec_b64(self, ops, mods):
        if len(ops) != 2:
            raise Unsupported('wrexec with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's':
            raise Unsupported('wrexec destination %s' % ops[0])
        d = dst[1]
        src = ops[1] if not reg_range(ops[1]) else 's%d' % reg_range(ops[1])[1]
        self.set_exec_scc('%s & ~exec' % self.u(src))
        self.set_s(d, 'exec')
        self.masks.add(d)

    op_s_andn2_wrexec_b32 = op_s_andn2_wrexec_b64

    # SOPK s_addk_i32 / s_mulk_i32: D = D op sext(simm16). LLVM's SIShrinkInstructions forms them from s_add_i32 /
    # s_mul_i32 only when isInt<16> of the constant, so the immediate is signed. s_addk_i32 defines SCC (signed
    # overflow, not modelled: see SCC_UNMODELLED); s_mulk_i32 does not.
    def sopk_arith(self, ops, op):
        d = reg_range(ops[0])
        if not d or d[0] != 's' or d[1] != d[2]:
            raise Unsupported('SOPK destination %s' % ops[0])
        imm = int(ops[1], 0) & 0xffff
        imm = (imm - 0x10000 if imm & 0x8000 else imm) & 0xffffffff
        self.masks.discard(d[1])
        self.set_s(d[1], '%s %s 0x%08xu' % (self.S(d[1]), op, imm))

    # s_bitset0_b32 / s_bitset1_b32: D[S0[4:0]] = 0 / 1, the other bits of D kept (AMD RDNA 2 ISA SOP1; LLVM
    # SOPInstructions.td defines them with D tied as an input and no SCC def). A lane mask or symbolic D is refused.
    def bitset(self, ops, one):
        d = reg_range(ops[0])
        if not d or d[0] != 's' or d[1] != d[2] or d[1] in self.masks:
            raise Unsupported('s_bitset%d_b32 destination %s' % (one, ops[0]))
        bit = '(1u << (%s & 31))' % self.u(ops[1])
        old = self.S(d[1])
        expr = '%s | %s' % (old, bit) if one else '%s & ~%s' % (old, bit)
        a, b = self.is_const(d[1]), self.is_const(reg_range(ops[1])[1]) if reg_range(ops[1]) else None
        if b is None and re.fullmatch(r'(0x[0-9a-f]+|\d+)', ops[1]):
            b = int(ops[1], 0)
        if a is not None and b is not None:  # both known: keep the constant (descriptor words are built this way)
            value = (a | 1 << (b & 31)) if one else (a & ~(1 << (b & 31)))
            self.set_s(d[1], '0x%08xu' % (value & 0xffffffff), ('const', value & 0xffffffff))
            return
        self.set_s(d[1], expr)

    def op_s_bitset0_b32(self, ops, mods):
        self.bitset(ops, 0)

    def op_s_bitset1_b32(self, ops, mods):
        self.bitset(ops, 1)

    def op_s_addk_i32(self, ops, mods):
        return self.scalar_accumulate('s_addk_i32', ops, mods)

    def op_s_mulk_i32(self, ops, mods):
        return self.scalar_accumulate('s_mulk_i32', ops, mods)

    def op_s_cselect_b32(self, ops, mods):
        self.scalar_select(ops, False)

    def op_s_cselect_b64(self, ops, mods):
        self.scalar_select(ops, True)

    def op_s_cmov_b32(self, ops, mods):
        self.scalar_cmov(ops, mods, False)

    def op_s_cmov_b64(self, ops, mods):
        self.scalar_cmov(ops, mods, True)

    def op_s_cmovk_i32(self, ops, mods):
        # SOPK form: the 16-bit immediate is sign-extended to 32 bits.
        if len(ops) != 2 or mods:
            raise Unsupported('s_cmovk_i32 requires a destination and a 16-bit immediate')
        value = sopk_imm(ops[1], True)
        self.scalar_cmov([ops[0], '0x%x' % value], mods, False, 's_cmovk_i32')

    def scalar_cmov(self, ops, mods, pair, name=None):
        """D = SCC ? S0 : D on plain SGPR destinations; SCC is unchanged."""
        name = name or 's_cmov_b%d' % (64 if pair else 32)
        if len(ops) != 2 or mods:
            raise Unsupported('%s requires two unmodified operands' % name)
        dst, src = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] > 105 or dst[2] - dst[1] != int(pair) or (pair and dst[1] % 2):
            raise Unsupported('%s destination %s' % (name, ops[0]))
        if pair and (not src or src[0] != 's' or src[2] > 105 or src[1] % 2 or src[2] != src[1] + 1):
            raise Unsupported('%s source %s' % (name, ops[1]))
        # A pair holds two words: a lane mask in either half cannot be moved as a full 32-bit scalar.
        halves = 2 if pair else 1
        regs = [('s%d' % (dst[1] + i)) for i in range(halves)]
        if src and src[0] == 's':
            regs += [('s%d' % (src[1] + i)) for i in range(halves)]
        if self.is_lane_mask(ops[0]) or self.is_lane_mask(ops[1]) or any(self.is_lane_mask(r) for r in regs):
            raise Unsupported('%s on a lane mask' % name)
        d = dst[1]
        low = ('s%d' % src[1]) if src and src[0] == 's' else ops[1]
        old = [self.S(d + i) for i in range(2 if pair else 1)]
        new = [self.u(low)]
        if pair:
            new.append(self.S(src[1] + 1))
        for i, (was, now) in enumerate(zip(old, new)):
            self.set_s(d + i, 'scc ? %s : %s' % (now, was))
        self.masks.discard(d)

    def scalar_select(self, ops, pair):
        if len(ops) != 3:
            raise Unsupported('scalar select with %d operands, expected 3' % len(ops))
        low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
        if pair and 'm0' in low:
            raise Unsupported('64-bit scalar operand m0')
        expr = 'scc ? %s : %s' % (self.u(low[1]), self.u(low[2]))
        if low[0] == 'm0':
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return
        if low[0] in ('exec', 'vcc', 'exec_lo', 'vcc_lo'):
            # In wave32, exec_lo and vcc_lo hold the whole lane mask, as the generic scalar path reads them.
            mask = 'exec' if low[0].startswith('exec') else 'vcc'
            if mask == 'exec':
                self.exec_known = False
            self.emit('%s = (%s) & 1;' % (mask, expr))
            return
        r = reg_range(low[0])
        if not r or r[0] != 's':
            raise Unsupported('s_cselect into %s' % low[0])
        d = r[1]
        self.set_s(d, '(%s) & 1' % expr if pair else expr)
        if pair:
            self.masks.add(d)
            self.set_s(d + 1, '0u')

    def smem_soffset(self, tok, record_use=True):
        """Runtime expression for an SMEM scalar offset operand (s0..s105 or m0)."""
        if tok == 'm0':
            return self.u(tok)
        r = reg_range(tok)
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            if record_use:
                return self.S(r[1])
            saved = set(self.used_s)
            try:
                return self.S(r[1])
            finally:
                self.used_s = saved
        raise Unsupported('operand %r' % tok)

    def const_soffset(self, tok):
        """The constant held by an SMEM offset register s0..s105, or None (m0 and unknown values)."""
        r = reg_range(tok)
        # Symbols are not merged at control-flow joins, so a constant proves the runtime value only without branches.
        if getattr(self, 'has_branches', True):
            return None
        if r and r[0] == 's' and r[1] == r[2] and r[1] <= 105:
            # A tagged constant (the ('const', 0, 'unassigned') default-sampler placeholder) is not a run-time value.
            sym = self.sym.get(r[1])
            value = self.is_const(r[1]) if sym and len(sym) == 2 else None
            if value is not None and 0 <= value <= 0xfffff:
                return value
        return None

    def op_s_load_dwordx4(self, ops, mods, count=4):
        imm = smem_mods(mods)
        if len(ops) != 3:
            raise Unsupported('s_load with %d operands' % len(ops))
        dst, base = reg_range(ops[0]), reg_range(ops[1])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[1] > 105 or dst[2] > 107:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        # s106/s107 are vcc: only a descriptor load, which emits no code, may end there (s[104:107]).
        into_vcc = dst[2] > 105
        if into_vcc and (dst[1] > 104 or dst[2] != 107):
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if not base or base[0] != 's' or base[2] - base[1] != 1 or base[2] > 105:
            raise Unsupported('s_load base %s' % ops[1])
        # A loaded descriptor or word is not a lane mask; some forms below only record a symbol, not set_s.
        for n in range(dst[1], dst[2] + 1):
            getattr(self, 'masks', set()).discard(n)
        p = self.sym.get(base[1])
        if not p or p[0] != 'ptr':
            raise Unsupported('s_load through s%d holding %s' % (base[1], p))
        if len(p) > 2 and not isinstance(p[2], int):
            # The 'added' mark (an add whose offset is not tracked, as in a program with branches): no byte offset.
            raise Unsupported('s_load through s%d: the pointer was advanced by an untracked offset' % base[1])
        base_off = p[2] if len(p) > 2 else 0   # literal bytes added by s_add_u32 on the pointer
        if (base_off or len(p) > 2) and p[1][0] in PTR64_SOURCES:
            # An advanced 64-bit pointer needs its high word completed by the matching s_addc_u32;
            # a table pointer is 32 bits wide and has no high word.
            if base[1] in getattr(self, 'ptr_open_pairs', ()) or self.sym.get(base[1] + 1) != ptr_high_symbol(p[1], base_off):
                raise Unsupported('s_load through s%d: its high word is not the matching advanced pair' % base[1])
        if p[1] == ('resheap',) and 'heap' in getattr(self.pipe, 'features', ()) and count in (4, 8) and not mods and not base_off and not into_vcc:
            k = len(self.pipe.heapvars)
            self.pipe.heapvars.append(None)
            self.used_h.add(k)
            self.emit('h%d = %s;' % (k, self.u(ops[2]) if reg_range(ops[2]) else '0x%xu' % (int(ops[2], 0) & 0xffffffff)))
            for i in range(count):
                self.sym[dst[1] + i] = ('desc', ('heapdyn', k), 0, i)
            return
        # 'off' names no offset operand at all, so the byte offset is zero, as for the literal 0.
        token = smem_offset(ops[2])
        known = self.const_soffset(token) if p[1] != ('smpheap',) else None
        if known is not None:
            # A register offset that holds a known constant (s_movk before the load) is that literal byte
            # offset; the modifier adds to it. This reaches descriptors in a bindless heap (#1723).
            total = known + (imm or 0)
            if total > 0xfffff:
                raise Unsupported('SMEM register offset %#x' % total)
            token, imm = str(total), None
        if re.fullmatch(r'-?(?:0[xX][0-9a-fA-F]+|\d+)', token):
            off = int(token, 0)
            if off < 0 or off > 0xfffff:
                raise Unsupported('SMEM immediate offset %s' % ops[2])
            if imm is not None:
                raise Unsupported('SMEM literal offset %s with offset:%#x' % (ops[2], imm))
            if p[1][0] == 'root':
                if into_vcc:
                    raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
                buf = self.pipe.resource((p[1], None), 'buf')
                for i in range(count):
                    self.set_s(dst[1] + i, '%s.Load(%d)' % (buf, base_off + off + 4 * i))
                return
            # The pointer symbol does not record an addend or a rewritten high word, so two loads through such a
            # pointer may read different words under the same (source, offset). In a compute or vertex stage (the
            # stages that accept a raw descriptor conversion before a store) those loads are told apart by the
            # offset of the s_load; loads through the untouched pair keep one identity.
            # A 32-bit table pointer carries a high word the program sets to a constant: loads through the same
            # untouched low word and the same constant read the same words (tag -1 - constant).
            hi = self.sym.get(base[1] + 1)
            if getattr(self, 'kind', None) not in ('cs', 'vs') or (p == ('ptr', p[1]) and hi == ('ptrhi', p[1])):
                load = ()
            elif p == ('ptr', p[1]) and hi and hi[0] == 'const' and len(hi) == 2:
                load = (-1 - hi[1],)
            else:
                load = (self.pc,)
            for i in range(count):
                self.sym[dst[1] + i] = ('desc', p[1], base_off + off, i) + load
            if into_vcc:
                self.vcc_lost = True
                self.vcc_descriptor_overwritten = False
                if getattr(self, 'has_branches', False):
                    value = tuple(self.sym.get(i) for i in (106, 107))
                    self.vcc_events.setdefault(self.pc, []).append(('load_descriptor', value))
            return
        if into_vcc:
            raise Unsupported('s_load destination %s for %d dwords' % (ops[0], count))
        if base_off:
            raise Unsupported('s_load through %s advanced by %#x' % (p[1], base_off))
        if p[1] == ('smpheap',):
            if self.pipe.sampler_heap and count == 4:
                # A sampler picked at run time from the sampler heap: keep the byte offset in the
                # first destination register; image() indexes the heap table with it.
                d = dst[1]
                self.used_s.add(d)
                expr = self.smem_soffset(token)
                if imm:
                    expr = '%s + %d' % (expr, imm)
                self.emit('s%d = %s;' % (d, expr))
                for i in range(count):
                    self.sym[d + i] = ('smpheap', d, i)
                return
            self.smem_soffset(token, record_use=False)
            if count != 4:
                raise Unsupported('s_load of %d dwords from %s' % (count, p[1]))
            # Shortcut (default unless X2W_SAMPLER_HEAP=1): a fixed default sampler.
            for i in range(count):
                # Never assigned at run time (the third field says so): a join that reads it refuses (run_time_symbol).
                self.sym[dst[1] + i] = ('const', 0, 'unassigned')
            self.pipe.notes.add('dynamic sampler index replaced by a default sampler')
            return
        raise Unsupported('s_load with a register offset through %s' % (p[1],))


    def op_s_bfm_b32(self, ops, mods):
        """Fold constant bitfield masks: titles build inline sampler words with them."""
        if len(ops) != 3:
            raise Unsupported('s_bfm_b32 with %d operands, expected 3' % len(ops))
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        if ops[0] == 'm0':
            if None not in vals:
                value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
                expr = '0x%08xu' % value
            else:
                expr = '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2]))
            self.sym.pop('m0', None)
            self.emit('m0 = %s;' % expr)
            return None
        d = reg_range(ops[0])[1]
        if None not in vals:
            value = (((1 << (vals[0] & 31)) - 1) << (vals[1] & 31)) & 0xffffffff
            self.set_s(d, '0x%08xu' % value, ('const', value))
            return None
        self.set_s(d, '(((1u << (%s & 31)) - 1) << (%s & 31))' % (self.u(ops[1]), self.u(ops[2])))
        return None

    def op_s_bfm_b64(self, ops, mods):
        """D[63:0] = ((1 << S0[5:0]) - 1) << S1[5:0]; SCC unchanged. Folded when both counts are constant."""
        d = reg_range(ops[0])[1]
        vals = []
        for o in ops[1:3]:
            r = reg_range(o)
            if r and r[0] == 's':
                vals.append(self.is_const(r[1]))
            else:
                try:
                    vals.append(int(o, 0))
                except ValueError:
                    vals.append(None)
        if None in vals:
            raise Unsupported('s_bfm_b64 with a run-time count')
        value = (((1 << (vals[0] & 63)) - 1) << (vals[1] & 63)) & 0xffffffffffffffff
        self.masks.discard(d)
        self.set_s(d, '0x%08xu' % (value & 0xffffffff), ('const', value & 0xffffffff))
        self.set_s(d + 1, '0x%08xu' % (value >> 32), ('const', value >> 32))
        return None

    def bitcmp(self, ops, want):
        """s_bitcmp{0,1}_b32: SCC = (S0[S1[4:0]] == want); no register is written."""
        self.emit('scc = ((%s >> (%s & 31u)) & 1u) == %du;' % (self.u(ops[0]), self.u(ops[1]), want))

    def op_s_bitcmp0_b32(self, ops, mods):
        self.bitcmp(ops, 0)

    def op_s_bitcmp1_b32(self, ops, mods):
        self.bitcmp(ops, 1)

    def op_s_load_dwordx8(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 8)

    def op_s_load_dwordx16(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 16)

    def op_s_load_dwordx2(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 2)

    def op_s_load_dword(self, ops, mods):
        self.op_s_load_dwordx4(ops, mods, 1)

    def op_s_buffer_load_dword(self, ops, mods, count=1):
        smem_imm_refuse('s_buffer_load', ops, mods)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] - dst[1] + 1 != count or dst[2] > 105:
            raise Unsupported('s_buffer_load destination %s for %d dwords' % (ops[0], count))
        buf = self.pipe.resource(self.descriptor(ops[1]), 'buf')
        token = smem_offset(ops[2])
        offset = self.u(token)
        # An SGPR offset can carry an immediate as well (disasm_rdna2 prints "s4 offset:0x10"): the byte offset is
        # the sum. glc/dlc are cache policy. Anything else is refused.
        extra = 0
        for m in mods:
            if m.startswith('offset:'):
                extra += int(m[7:], 0)
            elif m not in ('glc', 'dlc'):
                raise Unsupported('s_buffer_load modifier %s' % m)
        # Every word's address uses the offset register as it was before the load, so when the destination range
        # holds that register it is written last (s_buffer_load_dwordx2 s[4:5], s[16:19], s4).
        order = list(range(count))
        r = reg_range(token)
        if r and r[0] == 's' and dst[1] <= r[1] < dst[1] + count:
            order.remove(r[1] - dst[1])
            order.append(r[1] - dst[1])
        for i in order:
            self.set_s(dst[1] + i, '%s.Load(%s + %d)' % (buf, offset, extra + 4 * i))
            if not mods and re.fullmatch(r'0x[0-9a-f]+|\d+', token):
                self.sym[dst[1] + i] = ('fixed_buffer_word',)

    def op_s_buffer_load_dwordx2(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 2)

    def op_s_buffer_load_dwordx4(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 4)

    def op_s_buffer_load_dwordx8(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 8)

    def op_s_buffer_load_dwordx16(self, ops, mods):
        self.op_s_buffer_load_dword(ops, mods, 16)

    @staticmethod
    def check_buffer_load_arity(ops, mods):
        for modifier in mods:
            if modifier in ('lds', 'tfe', 'lwe'):
                raise Unsupported('buffer modifier %s' % modifier)
        if len(ops) != 4:
            raise Unsupported('buffer load with %d operands, expected 4' % len(ops))

    def buffer_address(self, ops, mods):
        Stage.check_buffer_load_arity(ops, mods)
        key = self.descriptor(ops[2])
        rw = key in getattr(self.pipe, 'rw_keys', ()) or key in getattr(self.pipe, 'uavs', {})
        if rw:
            # A buffer this program also stores to: read through its UAV (in place, so the program sees its own
            # stores), at the same uavbase-shifted address the stores use. No stride source for a UAV.
            if 'idxen' in mods:
                raise Unsupported('indexed load through a buffer the program stores to')
            buf = self.pipe.uav(key)
        else:
            buf = self.pipe.resource(key, 'buf')
        soff = ops[3]
        if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', soff):
            offset, extra = int(soff, 0), ''
        else:
            offset, extra = 0, '%s + ' % self.u(soff)  # an SGPR soffset adds its value
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
        if rw:
            extra = 'ud%d + %s' % (self.pipe.ud(('uavbase', int(buf[1:]))), extra)
        idxen, offen = 'idxen' in mods, 'offen' in mods
        if idxen or offen:
            regs = reg_range(ops[1])
            mode = 'idxen offen' if idxen and offen else 'idxen' if idxen else 'offen'
            if not regs or regs[0] != 'v' or regs[2] > 255 or regs[2] - regs[1] != int(idxen and offen):
                raise Unsupported('buffer vaddr %s for %s' % (ops[1], mode))
        elif ops[1] != 'off':
            raise Unsupported('buffer vaddr %s without idxen or offen' % ops[1])
        terms = []
        if idxen:
            if key in self.pipe.modified:
                raise Unsupported('indexed load through a modified descriptor')
            if key[0][0] == 'heapdyn':
                stride = 'x2w_vstride(h%d)' % key[0][1]
            else:
                stride = 'ud%d' % self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            terms.append('%s * %s' % (self.u('v%d' % regs[1]), stride))
        if offen:
            terms.append(self.u('v%d' % regs[2]))
        address = ' + '.join(terms + [str(offset)])
        return buf, extra + address

    def op_buffer_load_dword(self, ops, mods, count=1):
        # ops[0] is read before buffer_address() can refuse a short operand list, so an instruction with no
        # operands at all would reach it as an index error. Everything longer is left to the shared check,
        # which sees a modifier first: the lds form decodes to three operands and must still refuse on it.
        if not ops:
            raise Unsupported('buffer load with 0 operands, expected 4')
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, '%s.Load(a + %d)' % (buf, 4 * i))

    def op_buffer_load_dwordx2(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 2)

    def op_buffer_load_dwordx3(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 3)

    def op_buffer_load_dwordx4(self, ops, mods):
        self.op_buffer_load_dword(ops, mods, 4)

    def tbuffer(self, ops, mods, count):
        dst = reg_range(ops[0])
        formats = [m for m in mods if m.startswith('format:')]
        fmt = formats[0][8:-1] if formats else 'BUF_FMT_8_UNORM'  # the disassembler omits the default format
        buf, address = self.buffer_address(ops, [m for m in mods if not m.startswith('format:')])
        if buf.startswith('u'):
            raise Unsupported('typed load through a buffer the program stores to')  # ldu() takes a ByteAddressBuffer
        self.emit('a = %s;' % address)
        table = {'BUF_FMT_8_8_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', 'U(((w >> 16) & 255) / 255.0)', 'U((w >> 24) / 255.0)']),
                 'BUF_FMT_8_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', 'U(((w >> 8) & 255) / 255.0)', '0u', 'U(1.0)']),
                 'BUF_FMT_8_UNORM': ('ldu({0}, a)', ['U((w & 255) / 255.0)', '0u', '0u', 'U(1.0)']),
                 'BUF_FMT_8_UINT': ('{0}.Load(a & ~3u) >> ((a & 3u) * 8u)', ['w & 255u', '0u', '0u', '1u']),
                 # integer fields as unsigned values; a missing component of an integer format reads 1
                 'BUF_FMT_8_8_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '(w >> 16) & 255u', 'w >> 24']),
                 'BUF_FMT_16_16_UINT': ('ldu({0}, a)', ['w & 65535u', 'w >> 16', '0u', '1u']),
                 # x in bits [9:0], y [19:10], z [29:20], w [31:30] (the R10G10B10A2 layout, x lowest). ASSUMED from
                 # the GCN data-format naming; not checked against a source or a console readback.
                 'BUF_FMT_2_10_10_10_UNORM': ('ldu({0}, a)', ['U((w & 1023) / 1023.0)', 'U(((w >> 10) & 1023) / 1023.0)',
                                                             'U(((w >> 20) & 1023) / 1023.0)', 'U((w >> 30) / 3.0)']),
                 'BUF_FMT_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)', '0u', 'U(1.0)']),
                 # SNORM n: max(c / (2^(n-1) - 1), -1.0), so both -32768 and -32767 read -1.0 (the D3D11.3 functional
                 # spec's SNORM -> FLOAT rule; that the console's buffer fetch clamps the same way is ASSUMED)
                 'BUF_FMT_16_SNORM': ('ldu({0}, a)', ['U(max(float(asint(w << 16) >> 16) / 32767.0, -1.0))',
                                                     '0u', '0u', 'U(1.0)']),
                 'BUF_FMT_16_16_SNORM': ('ldu({0}, a)', ['U(max(float(asint(w << 16) >> 16) / 32767.0, -1.0))',
                                                        'U(max(float(asint(w) >> 16) / 32767.0, -1.0))', '0u', 'U(1.0)']),
                 'BUF_FMT_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))', '0u', 'U(1.0)']),
                 # four binary16 values in eight bytes: x, y in the first dword (low half first), z, w in the second
                 'BUF_FMT_16_16_16_16_FLOAT': ('ldu({0}, a)', ['U(f16tof32(w))', 'U(f16tof32(w >> 16))',
                                                              'U(f16tof32(ldu({0}, a + 4)))', 'U(f16tof32(ldu({0}, a + 4) >> 16))']),
                 'BUF_FMT_32_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', '{0}.Load(a + 12)']),
                 'BUF_FMT_32_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '{0}.Load(a + 8)', 'U(1.0)']),
                 'BUF_FMT_32_32_FLOAT': (None, ['{0}.Load(a)', '{0}.Load(a + 4)', '0u', 'U(1.0)']),
                 'BUF_FMT_32_FLOAT': (None, ['{0}.Load(a)', '0u', '0u', 'U(1.0)'])}
        if fmt not in table:
            raise Unsupported('buffer format %s' % fmt)
        load, comps = table[fmt]
        if load:
            self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(dst[1] + i, comps[i].format(buf))

    def load_format(self, fmt, buf, first, count):
        if fmt not in FETCH_FORMATS:
            raise Unsupported('buffer format %s' % fmt)
        load, comps = FETCH_FORMATS[fmt]
        if load:
            self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(first + i, comps[i].format(buf))

    def op_s_swappc_b64(self, ops, mods):
        # Only the entry call into the driver's fetch routine is translated: as the loads that routine makes.
        if self.kind != 'vs' or self.pc != 0 or ops != ['s[4:5]', 's[4:5]']:
            raise Unsupported('s_swappc_b64 %s at +0x%x' % (', '.join(ops), self.pc))
        if not self.pipe.vs_fetch:
            raise Unsupported('vertex stage calls a fetch routine and the pipeline declares no input layout')
        base = self.pipe.ud(('basevertex',))
        for e in self.pipe.vs_fetch:
            key = (('vbtab',), 16 * e['slot'])
            buf = self.pipe.resource(key, 'buf')
            stride = self.pipe.ud(('stride', self.pipe.resources[key]['index']))
            self.emit('a = (vid + ud%d) * ud%d + %d;' % (base, stride, e['offset']))
            self.load_format(FETCH_DXGI[e['format']][0], buf, e['vgpr'], e['count'])
        self.fetched = True

    def op_tbuffer_load_format_x(self, ops, mods):
        self.tbuffer(ops, mods, 1)

    def op_tbuffer_load_format_xy(self, ops, mods):
        self.tbuffer(ops, mods, 2)

    def op_tbuffer_load_format_xyz(self, ops, mods):
        self.tbuffer(ops, mods, 3)

    def op_tbuffer_load_format_xyzw(self, ops, mods):
        self.tbuffer(ops, mods, 4)

    @staticmethod
    def descriptor_scalar_writes(name, ops):
        regs = set()
        for token in destinations(name, ops):
            if token in ('vcc', 'vcc_lo', 'vcc_hi'):
                regs.update((106, 107))
            parsed = reg_range(token)
            if parsed and parsed[0] == 's':
                regs.update(range(parsed[1], parsed[2] + 1))
        if name.startswith('v_cmp') and len(ops) == 2:
            regs.update((106, 107))
        return regs

    def formatted_pointer_reaches(self, pointer, before):
        """Refuse a write to the original root pointer on any predecessor path of this read."""
        if not hasattr(self, 'descriptor_predecessors'):
            by_offset = {off: i for i, (off, *_rest) in enumerate(self.ins)}
            self.descriptor_predecessors = [[] for _ in self.ins]
            self.descriptor_writes_by_instruction = []
            for index, (offset, _, name, asm) in enumerate(self.ins):
                _, args, _ = parse(asm)
                self.descriptor_writes_by_instruction.append(self.descriptor_scalar_writes(name, args))
                calculated_target = self.calculated_pc.get(offset)
                if calculated_target is not None:
                    self.descriptor_predecessors[by_offset[calculated_target]].append(index)
                elif index + 1 < len(self.ins) and name not in ('s_branch', 's_endpgm'):
                    self.descriptor_predecessors[index + 1].append(index)
                if name == 's_branch' or name.startswith('s_cbranch_'):
                    target = by_offset.get(int(args[0].split('_')[1], 16))
                    if target is not None:
                        self.descriptor_predecessors[target].append(index)
        seen = set()
        pending = list(self.descriptor_predecessors[before])
        while pending:
            earlier = pending.pop()
            if earlier in seen:
                continue
            seen.add(earlier)
            if pointer in self.descriptor_writes_by_instruction[earlier]:
                return False
            pending.extend(self.descriptor_predecessors[earlier])
        return True

    def formatted_pointer_copy(self, pointer, before):
        """(source register, instruction index) of the one s_mov_b32 that is the last write to this pointer on
        every path from the entry to this read, or None.

        One writer found walking back, with no path that reaches the entry without a write, means that the copy is
        on every path to the read and that no path from the copy to the read writes the pointer again."""
        self.formatted_pointer_reaches(pointer, before)  # builds the predecessor tables
        writers, seen, expand = set(), set(), [before]
        while expand:
            at = expand.pop()
            if at == 0 or not self.descriptor_predecessors[at]:
                return None  # the entry value of the register reaches the read
            for earlier in self.descriptor_predecessors[at]:
                if earlier in seen:
                    continue
                seen.add(earlier)
                _, _, name, asm = self.ins[earlier]
                wrote = self.descriptor_writes_by_instruction[earlier] | scalar_destinations(name, asm)
                if pointer in wrote or 'any' in wrote:
                    writers.add(earlier)
                else:
                    expand.append(earlier)
        if len(writers) != 1:
            return None
        index, = writers
        _, _, name, asm = self.ins[index]
        _, args, _ = parse(asm)
        source = reg_range(args[1]) if name == 's_mov_b32' and len(args) == 2 else None
        if args[0] != 's%d' % pointer or not source or source[0] != 's' or source[1] != source[2]:
            return None
        return source[1], index

    def formatted_descriptor_from_block(self, reg, first):
        """Require one untouched local load and a table pointer that no incoming path can rewrite."""
        start, end = self.descriptor_block_first, self.descriptor_instruction_index
        block = self.ins[start:end]
        words = set(range(reg[1], reg[2] + 1))
        for index in range(len(block) - 1, -1, -1):
            _, _, name, asm = block[index]
            _, args, _ = parse(asm)
            if not (self.descriptor_scalar_writes(name, args) & words):
                continue
            m = re.fullmatch(r's_load_dwordx(4|8|16)', name)
            load = reg_range(args[0])
            if not m or not load or load[1] + first[3] != reg[1] or int(m.group(1)) < first[3] + 4:
                return False
            base = reg_range(args[1])
            if not base or base[0] != 's':
                return False
            pointer = base[1]
            read = start + index
            for previous in range(index - 1, -1, -1):
                _, _, earlier, prior = block[previous]
                _, src, _ = parse(prior)
                if pointer not in self.descriptor_scalar_writes(earlier, src):
                    continue
                copied_from = reg_range(src[1]) if earlier == 's_mov_b32' and len(src) == 2 else None
                if not copied_from or copied_from[0] != 's' or copied_from[1] != copied_from[2]:
                    return False
                pointer = copied_from[1]
                read = start + previous
                break
            else:
                copy = getattr(self, 'kind', None) == 'cs' and self.formatted_pointer_copy(pointer, read)
                if copy:
                    pointer, read = copy
            return self.entry_table_pointers.get(pointer) == first[1] and self.formatted_pointer_reaches(pointer, read)
        return False

    def mubuf_format(self, ops, mods, count):
        """buffer_load_format_x/xy/xyz/xyzw: FORMAT and DST_SEL come from word 3 of the descriptor, known at draw time.

        The caller supplies that word as an entry constant. Each supported format becomes one case of a switch
        on the FORMAT field that decodes the four components as the tbuffer path does; DST_SEL then picks, for
        each destination register, a component or the constant 0 or 1. The draw builds the word from a hardware
        descriptor or from the title's typed buffer view (src/startup_stubs/buffer_format_word3.h) and refuses the
        draw when it has none or the FORMAT has no case here, so the default arm is not reached from a draw.
        For compute, the caller must validate the native descriptor and reachable byte addresses, and bind the
        SRV and entry constants from that same descriptor snapshot.
        """
        buf, address = self.buffer_address(ops, mods)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != count or dst[2] > 255:
            raise Unsupported('formatted buffer load destination %s for %d components' % (ops[0], count))
        # FORMAT and DST_SEL are read from the table word, so word 3 must still be the loaded one.
        r = reg_range(ops[2])
        first = self.sym.get(r[1]) if r else None
        if (not r or r[0] != 's' or r[2] - r[1] != 3 or not first or first[0] != 'desc'
                or not names_one_value(first) or any(
                    self.sym.get(r[1] + i) != desc_word(first, first[3] + i) for i in range(4))):
            raise Unsupported('formatted buffer load through a descriptor whose words were rewritten')
        if first[1][0] != 'table':
            raise Unsupported('formatted buffer load requires a hardware table descriptor')
        if self.past_entry and not self.formatted_descriptor_from_block(r, first):
            raise Unsupported('formatted buffer load requires descriptor words proven in the entry block or a local table load')
        word3 = 'ud%d' % self.pipe.ud(('descword3', self.pipe.resources[self.descriptor(ops[2])]['index']))
        self.emit('a = %s;' % address)
        self.emit('{ uint c0 = 0, c1 = 0, c2 = 0, c3 = 0, one = U(1.0);')
        self.emit('switch ((%s >> %d) & %du) {' % (word3, DESC_FORMAT_SHIFT, DESC_FORMAT_MASK))
        for fmt, code in sorted(DESC_FORMAT_CODES.items(), key=lambda kv: kv[1]):
            load, comps = FETCH_FORMATS[fmt]
            body = ['w = %s;' % load.format(buf)] if load else []
            body += ['c%d = %s;' % (i, comps[i].format(buf)) for i in range(4)]
            if fmt in DESC_ONE_INTEGER:
                body.append('one = 1u;')
            self.emit('case %du: %s break;' % (code, ' '.join(body)))
        self.emit('default: c0 = c1 = c2 = c3 = %du; break; }' % DESC_FORMAT_UNKNOWN)
        pick = ('((%s >> %d) & 7u)' % (word3, 3 * i) for i in range(count))
        for i, sel in enumerate(pick):
            self.set_v(dst[1] + i, '%s == 4u ? c0 : %s == 5u ? c1 : %s == 6u ? c2 : %s == 7u ? c3 : %s == 1u ? one : 0u'
                       % (sel, sel, sel, sel, sel))
        self.emit('}')

    def format_load(self, ops, mods, count, name):
        """buffer_load_format_*: the format and swizzle are in the descriptor's word 3, known only at draw time.

        Behind X2W_DRAW_FEATURES=bufword3: the draw runtime passes srv i's word 3 as an entry constant (manifest
        `udsrc <n> bufword3 <i>`: the real word of a title-written descriptor, the synthesized one of a typed
        buffer view, 0 for raw-copy, vertex and root sources) and x2w_fmt_load decodes the element. The idxen
        stride is `udsrc stride` (element size of a typed view, descriptor stride otherwise; d3d12_draw.inl)."""
        if self.kind == 'cs' or getattr(self.pipe, 'uses_mesh_skeleton', False):
            return self.mubuf_format(ops, mods, count)
        if 'bufword3' not in getattr(self.pipe, 'features', ()):
            raise Unsupported(name)
        for m in mods:
            if not (m in ('idxen', 'offen', 'glc', 'slc', 'dlc') or m.startswith('offset:')):
                raise Unsupported('%s modifier %s' % (name, m))
        key = self.descriptor(ops[2])
        if key[0][0] == 'heapdyn' and {'heap', 'heapwords'} <= set(getattr(self.pipe, 'features', ())):
            for m in mods:
                if not (m in ('idxen', 'offen', 'glc', 'slc', 'dlc') or m.startswith('offset:')):
                    raise Unsupported('%s modifier %s' % (name, m))
            dst = reg_range(ops[0])
            buf, address = self.buffer_address(ops, mods)
            if buf.startswith('u'):
                raise Unsupported('%s through a buffer the program stores to' % name)
            self.pipe.draw_heapwords = True
            self.emit('a = %s;' % address)
            for i in range(count):
                self.set_v(dst[1] + i, 'x2w_fmt_load(%s, a, x2w_hword(h%d, 3u), %du)' % (buf, key[0][1], i))
            return
        if key[0][0] != 'table':
            # word 3 reaches the shader only for table entries (root and vertex sources give 0)
            raise Unsupported('%s through %s' % (name, key[0][0]))
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        if buf.startswith('u'):
            raise Unsupported('%s through a buffer the program stores to' % name)  # bufword3 names an srv
        w3 = self.pipe.ud(('bufword3', self.pipe.resources[key]['index']))
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, 'x2w_fmt_load(%s, a, ud%d, %du)' % (buf, w3, i))

    def buffer_store(self, ops, mods, count, name):
        """buffer_store_dword{,x2,x3,x4} to a raw buffer UAV, behind X2W_DRAW_FEATURES=uav.

        The descriptor becomes `uav <i> buf table|root ...` (RWByteAddressBuffer u<i>, draw-rt RUNTIME_DONE.md) and the
        address adds `udsrc <n> uavbase <i>` (the bytes between the 16-aligned view start the runtime binds and the
        descriptor base; without it a misaligned buffer would get a null UAV). Refused: idxen (no stride line for a
        UAV), a descriptor this program also reads as an SRV (D3D12 has no SRV + UAV on one buffer in a draw), and
        any modifier other than offen, offset:N and the cache-policy bits. The lane's store runs only where exec holds.
        """
        if 'uav' not in getattr(getattr(self, 'pipe', None), 'features', ()):
            raise Unsupported(name)
        for m in mods:
            if not (m in ('offen', 'glc', 'slc', 'dlc') or m.startswith('offset:')):
                raise Unsupported('%s modifier %s' % (name, m))
        key = self.descriptor(ops[2])
        if key[0][0] not in ('table', 'root'):
            raise Unsupported('%s through %s' % (name, key[0][0]))
        if key in self.pipe.modified:
            # A raw store uses only the base address (word 0, word 1 [15:0]); the runtime binds the entry's bytes in
            # place. Accepted edits: words 2 and 3 (record count, format, OOB select: bounds and typing a raw
            # store ignores; assumed in range) and an AND of word 1 that keeps [15:0] (clearing the stride).
            for word, op, other in getattr(self.pipe, 'modified_words', {}).get(key, [('?', '?', ())]):
                keeps_base = word == 1 and op == 's_and_b32' and len(other) == 1 and \
                    re.fullmatch(r'0x[0-9a-f]+', other[0]) and int(other[0], 16) & 0xffff == 0xffff
                if word not in (2, 3) and not keeps_base:
                    raise Unsupported('%s through a modified descriptor' % name)
        u = self.pipe.uav(key)
        base = self.pipe.ud(('uavbase', int(u[1:])))
        offset = sum(int(m[7:], 0) for m in mods if m.startswith('offset:'))
        soff = ops[3]
        if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', soff):
            offset += int(soff, 0)
            parts = []
        else:
            parts = [self.u(soff)]  # an SGPR soffset adds its value
        if 'offen' in mods:
            parts.insert(0, self.u(ops[1]))
        address = ' + '.join(['ud%d' % base] + parts + ['%d' % offset])
        data = reg_range(ops[0])
        if data[2] - data[1] + 1 != count:
            raise Unsupported('%s data %s' % (name, ops[0]))
        values = ', '.join(self.u('v%d' % (data[1] + i)) for i in range(count))
        call = '%s.Store%s(a, %s);' % (u, '' if count == 1 else count, values if count == 1 else 'uint%d(%s)' % (count, values))
        self.emit('a = %s;' % address)
        self.emit(call if self.exec_known else 'if (exec & 1) %s' % call)

    def format_store(self, ops, mods, count, name):
        """buffer_store_format_{x,xy,xyz,xyzw} to a typed buffer UAV, behind X2W_DRAW_FEATURES=uav,uavfmt.

        The format is in the V# word 3 [18:12], known only at draw time: `udsrc <n> uavword3 <i>` gives it, and
        `udsrc <n> uavstride <i>` the V# stride (word 1 [29:16]) that idxen multiplies the index by (RDNA 2 ISA,
        MUBUF addressing: base + soffset + offset + stride * index (+ the offen VGPR)). x2w_fmt_store writes data
        component c to format channel c; only formats whose channels are all 32 bits (codes 20-22, 62-64, 72-77:
        the bits are stored as given) are written, any other format stores nothing. ASSUMED, not measured: DST_SEL
        does not reorder a store (LLVM lowers the store intrinsic with the data in channel order), and the index
        is in range (num_records not checked, as for the raw stores). The lane's store runs only where exec holds."""
        if not {'uav', 'uavfmt'} <= set(getattr(getattr(self, 'pipe', None), 'features', ())):
            raise Unsupported(name)
        for m in mods:
            if not (m in ('idxen', 'offen', 'glc', 'slc', 'dlc') or m.startswith('offset:')):
                raise Unsupported('%s modifier %s' % (name, m))
        key = self.descriptor(ops[2])
        if key[0][0] not in ('table', 'root'):
            raise Unsupported('%s through %s' % (name, key[0][0]))
        if key in self.pipe.modified:
            raise Unsupported('%s through a modified descriptor' % name)
        u = self.pipe.uav(key)
        index = int(u[1:])
        base, w3 = self.pipe.ud(('uavbase', index)), self.pipe.ud(('uavword3', index))
        offset = sum(int(m[7:], 0) for m in mods if m.startswith('offset:'))
        soff = ops[3]
        parts = ['ud%d' % base]
        if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', soff):
            offset += int(soff, 0)
        else:
            parts.append(self.u(soff))
        vaddr = reg_range(ops[1])
        if 'idxen' in mods and 'offen' in mods:
            if not vaddr or vaddr[2] != vaddr[1] + 1:
                raise Unsupported('%s idxen offen address %s' % (name, ops[1]))
            parts += ['v%d * ud%d' % (vaddr[1], self.pipe.ud(('uavstride', index))), self.u('v%d' % vaddr[2])]
        elif 'idxen' in mods:
            parts.append('%s * ud%d' % (self.u(ops[1]), self.pipe.ud(('uavstride', index))))
        elif 'offen' in mods:
            parts.append(self.u(ops[1]))
        data = reg_range(ops[0])
        if not data or data[2] - data[1] + 1 != count:
            raise Unsupported('%s data %s' % (name, ops[0]))
        values = [self.u('v%d' % (data[1] + i)) for i in range(count)] + ['0u'] * (4 - count)
        self.pipe.notes.add('typed UAV store (x2w_fmt_store: 32-bit channel formats only)')
        call = 'x2w_fmt_store(%s, a, ud%d, uint4(%s), %du);' % (u, w3, ', '.join(values), count)
        self.emit('a = %s;' % ' + '.join(parts + ['%d' % offset]))
        self.emit(call if self.exec_known else 'if (exec & 1) %s' % call)

    def op_buffer_store_format_x(self, ops, mods):
        self.format_store(ops, mods, 1, 'buffer_store_format_x')

    def op_buffer_store_format_xy(self, ops, mods):
        self.format_store(ops, mods, 2, 'buffer_store_format_xy')

    def op_buffer_store_format_xyz(self, ops, mods):
        self.format_store(ops, mods, 3, 'buffer_store_format_xyz')

    def op_buffer_store_format_xyzw(self, ops, mods):
        if self.kind == 'cs':
            return self.compute_store_format_xyzw(ops, mods)
        self.format_store(ops, mods, 4, 'buffer_store_format_xyzw')

    def store_descriptor(self, tok, words, what, stride0_ok=False, raw_conversion_ok=False):
        """Resource key for a single loaded descriptor or a proven raw transformation of it."""
        rsrc = reg_range(tok)
        first = self.sym.get(rsrc[1]) if rsrc and rsrc[0] == 's' and rsrc[2] - rsrc[1] == words - 1 else None
        if not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or first[3] % 4:
            raise Unsupported('%s through %s holding %s' % (what, tok, first))
        key = desc_key(first)
        loaded = tuple(desc_word(first, first[3] + k) for k in range(words))
        actual = tuple(self.sym.get(rsrc[1] + k) for k in range(words))
        valid = names_one_value(first) and actual == loaded
        if names_one_value(first) and stride0_ok and words == 4:
            cleared = loaded[1] + ('stride0',)
            valid = valid or actual == (loaded[0], cleared, *loaded[2:])
            if not valid and raw_conversion_ok:
                valid = actual == (loaded[0], cleared, loaded[2] + ('byte_count',), loaded[3] + ('oob3',))
        if (not valid or key in getattr(self.pipe, 'other_modified', ())
                or (key in self.pipe.modified and not stride0_ok)):
            raise Unsupported('%s through a modified descriptor' % what)
        return key

    def raw_table_uav_store(self, kind_label, ops, mods, data_count, *, allow_glc=False, allow_scalar_address=False,
                            raw_conversion_ok=False):
        """Validate a fixed-table raw UAV descriptor, data range, byte address, and 4-byte alignment."""
        offset = 0
        glc = False
        for m in mods:
            if m.startswith('offset:') and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                offset += int(m[7:], 0)
            elif m == 'glc' and allow_glc:
                glc = True
            elif m != 'offen':
                raise Unsupported('%s modifier %s' % (kind_label, m))
        offen = 'offen' in mods
        if len(ops) != 4 or (not offen and not (allow_scalar_address and glc and ops[1] == 'off')):
            raise Unsupported('%s without the offen address form' % kind_label)
        data = reg_range(ops[0])
        if not data or data[0] != 'v' or data[2] - data[1] != data_count - 1 or data[2] > 255:
            raise Unsupported('%s data %s' % (kind_label, ops[0]))
        if offen:
            vaddr = reg_range(ops[1])
            if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
                raise Unsupported('%s vaddr %s for offen' % (kind_label, ops[1]))
        key = self.store_descriptor(ops[2], 4, kind_label, stride0_ok=True, raw_conversion_ok=raw_conversion_ok)
        terms = [self.u(ops[1])] if offen else []
        if re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            if not offen:
                raise Unsupported('%s off requires a scalar register soffset' % kind_label)
            offset += int(ops[3], 0)
        else:
            sreg = reg_range(ops[3])
            if not sreg or sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 105:
                raise Unsupported('buffer soffset %s' % ops[3])
            terms.append(self.S(sreg[1]))
        if offset % 4:
            raise Unsupported('%s at unaligned immediate offset %d' % (kind_label, offset))
        terms.append(str(offset))
        return data, self.pipe.uav(key), '(%s) & 0xfffffffcu' % ' + '.join(terms), glc

    def compute_buffer_store(self, ops, mods, count=1):
        # Compute-stage store in the offen form: per-lane byte VGPR plus the scalar offset and the 12-bit immediate,
        # through a descriptor read from a fixed root-table slot, to a RWByteAddressBuffer (manifest `uav` line).
        # Like the loads, this assumes a raw descriptor (no swizzle or stride). Out-of-range stores: assumes a future
        # runtime sizes the view to the descriptor's range so both sides drop them; nothing binds or checks that yet.
        if self.kind != 'cs':
            raise Unsupported('buffer_store_dword' + ('x%d' % count if count > 1 else ''))  # the unknown-op refusal
        src, uav, address, _ = self.raw_table_uav_store('buffer store', ops, mods, count, raw_conversion_ok=True)
        words = [self.u('v%d' % n) for n in range(src[1], src[2] + 1)]
        data = words[0] if count == 1 else 'uint%d(%s)' % (count, ', '.join(words))
        # Store* needs a multiple of four. As in #2059, the run-time part (VGPR and register soffset) is rounded down:
        # an assumption that a dword access ignores the low two address bits, not a measurement.
        self.emit('a = %s;' % address)
        store = '%s.Store%s(a, %s);' % (uav, '' if count == 1 else count, data)
        self.emit(store if self.exec_known else 'if (exec & 1) %s' % store)

    def compute_store_format_xyzw(self, ops, mods):
        """Lower compute MUBUF buffer_store_format_xyzw (opcode 0x07) in the idxen form with an inline soffset.

        FORMAT, stride and num_records are read at dispatch from the same entry constants as
        buffer_store_format_d16_x, with its bounds and OOB_SELECT rules. Only 4-channel formats write
        (STORE_FORMAT_XYZW_FORMATS); any other descriptor makes the store a no-op.
        """
        name = 'buffer_store_format_xyzw'
        if self.kind != 'cs':
            raise Unsupported(name)  # the unknown-op refusal
        raw = getattr(self, 'raw_code', b'')
        pc = getattr(self, 'pc', None)
        if raw and pc is not None and pc + 8 <= len(raw):
            w0, w1 = struct.unpack_from('<II', raw, pc)
            if (w0 & 0x00030000) or (w1 & 0x00200000):
                raise Unsupported('%s illegal MUBUF encoding bits (lds/reserved)' % name)
        if len(ops) != 4:
            raise Unsupported('%s with %d operands, expected 4' % (name, len(ops)))
        idxen = False
        imm_offset = None
        for m in mods:
            if m == 'idxen' and not idxen:
                idxen = True
            elif m == 'offen':
                raise Unsupported('%s offen address form (only idxen is modelled)' % name)
            elif m.startswith('offset:') and imm_offset is None and re.fullmatch(r'0x[0-9a-f]+|\d+', m[7:]):
                imm_offset = int(m[7:], 0)
                if imm_offset > 0xfff:
                    raise Unsupported('%s modifier %s' % (name, m))
            else:
                raise Unsupported('%s modifier %s' % (name, m))
        if not idxen:
            raise Unsupported('%s without idxen (only the idxen address form is modelled)' % name)
        src = reg_range(ops[0])
        if not src or src[0] != 'v' or src[2] - src[1] != 3 or src[2] > 255:
            raise Unsupported('%s data %s' % (name, ops[0]))
        vaddr = reg_range(ops[1])
        if not vaddr or vaddr[0] != 'v' or vaddr[1] != vaddr[2] or vaddr[1] > 255:
            raise Unsupported('%s vaddr %s for idxen' % (name, ops[1]))
        if not re.fullmatch(r'0x[0-9a-f]+|\d+', ops[3]):
            raise Unsupported('%s soffset %s (only a non-negative inline constant is modelled)' % (name, ops[3]))
        rsrc = reg_range(ops[2])
        if not rsrc or rsrc[0] != 's' or rsrc[2] - rsrc[1] != 3 or rsrc[2] > 105:
            raise Unsupported('%s descriptor %s' % (name, ops[2]))
        key = self.store_descriptor(ops[2], 4, name, stride0_ok=False)
        first = self.sym.get(rsrc[1])
        if getattr(self, 'past_entry', False) and not self.formatted_descriptor_from_block(rsrc, first):
            raise Unsupported('%s requires descriptor words proven in the entry block or a local table load' % name)
        idx = self.u(ops[1])
        data = 'uint4(%s)' % ', '.join(self.u('v%d' % n) for n in range(src[1], src[2] + 1))
        uav = self.pipe.uav(key, 'raw')
        uav_index = self.pipe.uavs[key]
        word3 = 'ud%d' % self.pipe.ud(('uav_descword3', uav_index))
        stride = 'ud%d' % self.pipe.ud(('uav_stride', uav_index))
        num_records = 'ud%d' % self.pipe.ud(('uav_numrecords', uav_index))
        self.pipe.uses_store_format_xyzw = self.pipe.uses_pack16_rne = True
        call = 'store_format_xyzw(%s, %s, %s, %s, %s, 0u, %du, %du, %s);' % (
            uav, word3, stride, num_records, idx, int(ops[3], 0), imm_offset or 0, data)
        self.emit(call if self.exec_known else 'if (exec & 1) ' + call)


    def op_buffer_store_dword(self, ops, mods, count=1):
        if getattr(self, 'kind', None) == 'cs':
            return self.compute_buffer_store(ops, mods, count)
        Stage.buffer_store(self, ops, mods, count, 'buffer_store_dword' + ('x%d' % count if count > 1 else ''))

    def op_buffer_store_dwordx2(self, ops, mods):
        if getattr(self, 'kind', None) == 'cs':
            return self.compute_buffer_store(ops, mods, 2)
        self.buffer_store(ops, mods, 2, 'buffer_store_dwordx2')

    def op_buffer_store_dwordx3(self, ops, mods):
        if getattr(self, 'kind', None) == 'cs':
            return self.compute_buffer_store(ops, mods, 3)
        self.buffer_store(ops, mods, 3, 'buffer_store_dwordx3')

    def op_buffer_store_dwordx4(self, ops, mods):
        if getattr(self, 'kind', None) == 'cs':
            return self.compute_buffer_store(ops, mods, 4)
        self.buffer_store(ops, mods, 4, 'buffer_store_dwordx4')

    def op_buffer_load_format_x(self, ops, mods):
        self.format_load(ops, mods, 1, 'buffer_load_format_x')

    def op_buffer_load_format_xy(self, ops, mods):
        self.format_load(ops, mods, 2, 'buffer_load_format_xy')

    def op_buffer_load_format_xyz(self, ops, mods):
        self.format_load(ops, mods, 3, 'buffer_load_format_xyz')

    def op_buffer_load_format_xyzw(self, ops, mods):
        self.format_load(ops, mods, 4, 'buffer_load_format_xyzw')

    def image(self, ops, mods, how):
        packed_load = how in ('load', 'load_mip') and 'd16' in mods
        if len(ops) < 3:
            raise Unsupported('image instruction requires three operands')
        if self.kind == 'cs' and packed_load:
            raise Unsupported('compute d16 image load requires validated SRV formats')
        if packed_load and self.round16:
            raise Unsupported('d16 image load requires validated SRV formats and round-to-nearest')
        dst = reg_range(ops[0])
        dims = [m for m in mods if m.startswith('dim:')]
        dmasks = [m for m in mods if m.startswith('dmask:')]
        if not dims or not dmasks:
            raise Unsupported('image instruction without %s' % ('dim:' if not dims else 'dmask:'))
        dim = dims[0][4:]
        dmask = int(dmasks[0][6:], 0)
        kinds = {'SQ_RSRC_IMG_1D': 'tex1d', 'SQ_RSRC_IMG_2D': 'tex2d', 'SQ_RSRC_IMG_3D': 'tex3d',
                 'SQ_RSRC_IMG_2D_ARRAY': 'tex2da'}
        if dim not in kinds:
            raise Unsupported('image dimension %s' % dim)
        # A 2D array takes the slice as its third address component, where a volume takes z. Sample and
        # Load on a Texture2DArray take it in the same place, so only the declared type differs. A 1D image has
        # the one coordinate (x for a load, u for a sample), bound as a Texture1D (manifest kind tex1d).
        axes = {'SQ_RSRC_IMG_1D': 1, 'SQ_RSRC_IMG_2D': 2}.get(dim, 3)
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        # Every form takes only dim, dmask, the cache-policy bits and, for the loads, unorm (a load addresses texels by
        # integer coordinates either way) and a16. a16 packs the 16-bit address components two to a dword, the first
        # in the low half (LLVM SIISelLowering packImage16bitOpsToDwords): unsigned texel coordinates for a load and
        # binary16 coordinates for a sample. d16, tfe, lwe and r128 change the result or resource layout and are
        # refused, as is any modifier not named here.
        # (image_sample_l and image_load_mip check their own modifier lists below.)
        allowed = ('dlc', 'glc', 'slc', 'a16') + (('unorm', 'd16') if how == 'load' else ('d16',) if how in ('sample', 'lz') else ())
        if how == 'get_lod':
            allowed = ('dlc', 'glc', 'slc')
        for m in mods:
            if how in ('sample', 'lz', 'load', 'b', 'd', 'get_lod') and not (m.startswith(('dim:', 'dmask:')) or m in allowed):
                raise Unsupported('image_%s modifier %s' % ({'b': 'sample_b', 'd': 'sample_d'}.get(how, how), m))
        a16 = 'a16' in mods
        bias = None
        if how == 'b':
            # image_sample_b: the LOD bias is the first address dword, before the coordinates (LLVM
            # AMDGPUImageDimIntrinsics sample_b: ExtraAddrArgs [bias]). With a16 it would be a 16-bit value packed
            # into its own dword, not modelled here.
            if dim == 'SQ_RSRC_IMG_1D':
                raise Unsupported('unverified 1D image sample address or modifiers')
            if a16:
                raise Unsupported('image_sample_b modifier a16')
            if dmask == 0:
                raise Unsupported('image_sample_b with an empty dmask')
            if len(cs) != axes + 1:
                raise Unsupported('image_sample_b with %d address registers' % len(cs))
            if self.kind != 'ps' or ((not self.exec_known or self.blocks) and
                                     (not getattr(self, 'quad_divergent', False) or self.has_loop)):
                raise Unsupported('image_sample_b needs implicit derivatives outside straight-line code with every lane active')
            bias, cs = cs[0], cs[1:]
        grads = None
        if how == 'd':
            # image_sample_d: the gradients come first, d<c>dh for each coordinate c (s, t, r) then d<c>dv, and then
            # the coordinates and the slice (LLVM IntrinsicsAMDGPU.td AMDGPUDimProps.GradientArgs, AddrArgs order
            # ExtraAddrArgs, GradientArgs, CoordSliceArgs). h is the horizontal (x) and v the vertical (y) screen
            # direction: SampleGrad's ddx and ddy. A 2D array has 2D gradients. With a16 the gradients would be
            # packed too, not modelled here.
            n = {'SQ_RSRC_IMG_1D': 1, 'SQ_RSRC_IMG_2D': 2, 'SQ_RSRC_IMG_3D': 3, 'SQ_RSRC_IMG_2D_ARRAY': 2}[dim]
            if n == 3 and len(cs) == 16 and not coords.startswith('['):
                cs = cs[:9]
            if a16:
                raise Unsupported('image_sample_d modifier a16')
            if dim == 'SQ_RSRC_IMG_1D':
                raise Unsupported('unverified 1D image sample address or modifiers')
            if len(cs) != 2 * n + axes:
                raise Unsupported('image_sample_d with %d address registers' % len(cs))
            if a16 or dmask == 0:
                raise Unsupported('image_sample_d form %s %s' % (', '.join(ops), ' '.join(mods)))
            grads, cs = (cs[:n], cs[n:2 * n]), cs[2 * n:]
        if a16:
            if len(cs) < (axes + 1) // 2:
                raise Unsupported('a16 image address with %d registers' % len(cs))
            # component i is the (i % 2) half of address dword i // 2; unpacked to one 32-bit value each
            cs = [('a16', cs[i // 2], i % 2) for i in range(axes)]
        lod = None
        if how == 'l':
            # Address dwords are the coordinates and then the explicit level; offsets, bias, clamp and derivatives
            # would add dwords or modifiers this lowering does not read, so they are refused. Only the cache-policy
            # bits (glc, slc, dlc), which never change the sampled value, are accepted next to dim and dmask;
            # unorm, r128, tfe, lwe, a16, d16 and reserved bits would each change the address, resource or result
            # layout and are refused by name.
            for m in mods:
                if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                    raise Unsupported('image_sample_l modifier %s' % m)
            if dmask == 0:
                raise Unsupported('image_sample_l with an empty dmask')
            if len(cs) != axes + 1:
                raise Unsupported('image_sample_l with %d address registers' % len(cs))
            lod, cs = cs[-1], cs[:-1]
        desc_key = self.descriptor(ops[2])
        tex = self.pipe.resource(desc_key, kinds[dim])
        if packed_load:
            if desc_key not in self.pipe.resources:
                raise Unsupported('d16 load requires a bound SRV')
            res = self.pipe.resources[desc_key]
            if not dst or dst[0] != 'v' or dst[2] > 255 or dst[2] - dst[1] + 1 != (dmask.bit_count() + 1) // 2:
                raise Unsupported('image_%s d16 destination %s for dmask %#x' % (how, ops[0], dmask))
            fmt = self.pipe.srv_formats.get(res['index'])
            if fmt is not None and fmt not in D16_EXACT_SRV_FORMATS:
                raise Unsupported('image_%s d16 unsupported SRV format %d' % (how, fmt))

        if how in ('load', 'load_mip'):
            # image_load_mip addresses x, y, [slice or z] and then the mip, so a 2D array is four registers
            # and Texture2DArray.Load takes int4(x, y, slice, mip) like a volume's int4(x, y, z, mip); a Texture1D
            # takes int2(x, mip).
            mip = '0'
            if how == 'load_mip':
                unsupported = [m for m in mods if not m.startswith(('dim:', 'dmask:'))
                               and m not in ('unorm', 'glc', 'slc', 'dlc', 'd16')]
                if unsupported or len(cs) != axes + 1 or not dmask:
                    raise Unsupported('image_load_mip address or modifiers: %s' % ' '.join(mods))
                mip = self.u(cs[axes])
            if len(cs) < axes:
                raise Unsupported('image_%s with %d address registers' % (how, len(cs)))
            address = [self.a16_u(c) if isinstance(c, tuple) else self.u(c) for c in cs[:axes]] + [mip]
            call = '%s.Load(int%d(%s))' % (tex, axes + 1, ', '.join(address))
        else:
            heap = self.heap_sampler(ops[3])
            words = None if heap else self.const_words(ops[3])
            name = 'image_sample' + ('' if how == 'sample' else '_' + how)
            if how == 'b':
                # The guest clamps the biased LOD to the sampler's MIN_LOD and MAX_LOD. Only a constant descriptor
                # can be read here, and only its limits are carried to the static sampler (line smpl), so a sampler
                # from the heap or from a register is refused: its limits and bias cannot be shown to be irrelevant.
                words = self.sample_b_sampler(heap, words)
                smp = self.pipe.sampler(tuple(words), lod_limits=True)
            elif how == 'get_lod':
                # The clamped result depends on the sampler's MIN_LOD, MAX_LOD and bias, so the sampler must be a
                # constant descriptor under the same validation as image_sample_b, with its limits carried.
                if heap or not words or self.default_sampler_words(words):
                    raise Unsupported('image_get_lod with a sampler that is not constant (LOD limits and bias unknown)')
                smp = self.pipe.sampler(tuple(self.lod_sampler_words('image_get_lod', words)), lod_limits=True)
            elif how in ('sample', 'l', 'lz', 'd') and (heap or (words and self.default_sampler_words(words))):
                # A heap sampler, or the fixed default that stands in for one, keeps its unbounded static LOD range: the
                # descriptor is read at run time, so its MIN_LOD, MAX_LOD and bias are not validated here (#2221).
                self.pipe.lod_unvalidated.add(name)
                smp = heap or self.pipe.sampler(tuple(words))
            elif how in ('sample', 'l', 'lz', 'd') and words:
                # A constant descriptor carries its LOD limits like image_sample_b, under the same bias and range
                # validation. A mip filter of 0 (base level only) keeps the ordinary sampler, as before: refusing it
                # would drop every pipeline with an all-zero inline sampler, and only the bias form needs it refused.
                self.lod_sampler_words(name, words, mip_filter_none_ok=True)
                smp = self.pipe.sampler(tuple(words), lod_limits=bool((words[2] >> 26) & 3))
            else:
                smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3]))
            if len(cs) < axes:
                raise Unsupported('image_%s with %d address registers' % (how, len(cs)))
            fs = [self.a16_f(c) if isinstance(c, tuple) else self.f(c) for c in cs[:axes]]
            uv = fs[0] if axes == 1 else 'float%d(%s)' % (axes, ', '.join(fs))
            if how == 'sample' and self.kind == 'cs':
                raise Unsupported('image_sample with implicit LOD in a compute program')
            if how == 'l':
                call = '%s.SampleLevel(%s, %s, %s)' % (tex, smp, uv, self.f(lod))
            elif how == 'b':
                # SampleBias takes implicit derivatives like Sample, so it is lowered only where Sample is (below);
                # the level-0 fallback Sample uses elsewhere would ignore the bias.
                if self.kind != 'ps':
                    raise Unsupported('image_sample_b outside straight-line code with every lane active')
                if not self.exec_known or len(self.blocks) > 0:
                    # X2W_PS_QUAD_DIVERGENT=1 (see quad_divergent): taken in forward control flow, not in a loop
                    if not self.quad_divergent or self.has_loop:
                        raise Unsupported('image_sample_b outside straight-line code with every lane active')
                    self.quad_divergent_used.append(self.pc)
                call = '%s.SampleBias(%s, %s, %s)' % (tex, smp, uv, self.f(bias))
            elif how == 'get_lod':
                # image_get_lod: [0] the LOD the sample would use, clamped, [1] the computed LOD before the clamp
                # (Mesa nir_texop_lod, which ACO emits as image_get_lod and returns as is; GLSL textureQueryLod x/y).
                # HLSL CalculateLevelOfDetail / CalculateLevelOfDetailUnclamped (HLSL intrinsics, SM 4.1+). Both use implicit
                # derivatives, so the control-flow rule of image_sample_b applies. Which clamps each side applies
                # (sampler MIN/MAX_LOD, the view's mip range) is ASSUMED to agree; the runtime carries the sampler's
                # LOD fields with X2W_SAMPLER_LOD=1.
                if self.kind != 'ps' or dim != 'SQ_RSRC_IMG_2D' or dmask & ~3 or not dmask or len(cs) != 2 or a16:
                    raise Unsupported('image_get_lod form %s %s' % (', '.join(ops), ' '.join(mods)))
                if not self.exec_known or len(self.blocks) > 0:
                    if not self.quad_divergent or self.has_loop:
                        raise Unsupported('image_get_lod outside straight-line code with every lane active')
                    self.quad_divergent_used.append(self.pc)
                call = 'float4(%s.CalculateLevelOfDetail(%s, %s), %s.CalculateLevelOfDetailUnclamped(%s, %s), 0.0, 0.0)' % (
                    tex, smp, uv, tex, smp, uv)
            elif how == 'd':
                # explicit gradients: no implicit derivatives, so any control flow and any exec state
                ddx, ddy = (', '.join(self.f(c) for c in g) for g in grads)
                if len(grads[0]) > 1:
                    ddx, ddy = 'float%d(%s)' % (len(grads[0]), ddx), 'float%d(%s)' % (len(grads[0]), ddy)
                call = '%s.SampleGrad(%s, %s, %s, %s)' % (tex, smp, uv, ddx, ddy)
            elif how == 'lz' or not self.exec_known or self.kind != 'ps' or len(self.blocks) > 0:
                call = '%s.SampleLevel(%s, %s, 0)' % (tex, smp, uv)
            else:
                call = '%s.Sample(%s, %s)' % (tex, smp, uv)
        self.emit('t = %s;' % call)
        # An image read with no implicit derivatives (load, level 0, explicit level) of a bound texture (rN) through a
        # fixed sampler (qN), at lane-invariant coordinates, returns the same texel or filtered value in every lane.
        if how in ('load', 'load_mip', 'lz', 'l') and hasattr(self, 'lane_s') and re.match(r'r\d+\b', call):
            expr = re.sub(r'\bq\d+\b', '0', call)
            expr = re.sub(r'\.(Load|SampleLevel)\(', '(', expr)
            expr = re.sub(r'\b(int|uint|float)[234]\(', '(', expr)
            self.t_invariant = self.lane_invariant(expr)
        n = dst[1]
        if packed_load:
            res = self.pipe.resources[desc_key]
            res['d16_exact'] = True
            mode_arg = 'ud%d' % self.pipe.ud(('d16unorm16', res['index']))
            self.pipe.uses_pack16_rne = True
            halves = ['pack16_d16(t.%s, %s)' % (comp, mode_arg) for bit, comp in enumerate('xyzw') if dmask & (1 << bit)]
            for i in range(0, len(halves), 2):
                odd_tail = i + 1 == len(halves)
                expr = halves[i] if odd_tail else '%s | (%s << 16)' % (halves[i], halves[i + 1])
                self.set_v(dst[1] + i // 2, expr, low16_only=odd_tail)
            return
        elif 'd16' in mods:
            if self.round16:
                raise Unsupported('image_%s d16 under s_round_mode f16/f64 toward zero' % how)
            # Packed D16 (gfx9+, LLVM GCNSubtarget hasPackedD16VMem; SIISelLowering adjustLoadValueType): each enabled
            # channel of the filtered result is converted to binary16 and two go in one VGPR, the first channel in
            # the low half; the high half after an odd channel count is written as 0 here (undefined in hardware).
            # Only the sampling forms take it: their result is a float whatever the texel format.
            comps = ['f32tof16(t.%s)' % c for bit, c in enumerate('xyzw') if dmask & (1 << bit)]
            if not comps or dst[2] - dst[1] + 1 != (len(comps) + 1) // 2:
                raise Unsupported('image_%s d16 with dmask %#x into %s' % (how, dmask, ops[0]))
            for i in range(0, len(comps), 2):
                self.set_v(n, comps[i] if i + 1 == len(comps) else '%s | (%s << 16)' % (comps[i], comps[i + 1]))
                n += 1
            return
        for bit, comp in enumerate('xyzw'):
            if dmask & (1 << bit):
                self.set_v(n, 'U(t.%s)' % comp)
                n += 1

    def a16_u(self, component):
        _, reg, half = component
        return '(%s >> 16)' % self.u(reg) if half else '(%s & 0xffffu)' % self.u(reg)

    def a16_f(self, component):
        _, reg, half = component
        return 'f16tof32(%s)' % ('%s >> 16' % self.u(reg) if half else self.u(reg))

    def image_compare(self, ops, mods, how):
        """image_sample_c and image_sample_c_lz: a 2D or 2D-array depth comparison returning one float.

        The address dwords are the reference, then u and v (and the slice of a 2D array), which is what
        SampleCmp* takes as its third argument and its location. The comparison function and the filter live in the sampler descriptor,
        so they are read from its constant words by the decoder the pixel AOT path already checks against
        D3D12 (pixel_aot.comparison_sampler_state) and bound as a SamplerComparisonState. HLSL has no
        SampleCmpLevel before shader model 6.7 (the stages here compile as 5.1 and 6.0), so
        image_sample_c_l is refused instead of being sampled at level zero.
        """
        base = 'image_sample_' + how
        if how == 'c_l':
            raise Unsupported('image_sample_c_l: SampleCmpLevel needs shader model 6.7; only c and c_lz are lowered')
        for m in mods:
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (base, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % base)
        dim, dmask = dims[0], int(masks[0], 0)
        if dim not in ('SQ_RSRC_IMG_2D', 'SQ_RSRC_IMG_2D_ARRAY'):
            raise Unsupported('%s on %s' % (base, dim))
        array = dim == 'SQ_RSRC_IMG_2D_ARRAY'
        if dmask != 1:
            raise Unsupported('%s with dmask %#x: a comparison returns one channel' % (base, dmask))
        if len(ops) != 4:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        # The reference, u and v; a 2D array adds the slice as the last address register.
        if len(cs) != (4 if array else 3):
            raise Unsupported('%s with %d address registers' % (base, len(cs)))
        if self.heap_sampler(ops[3]):
            raise Unsupported('%s through a sampler heap' % base)
        words = self.const_words(ops[3])
        if words is None:
            raise Unsupported('%s with a sampler that is not constant' % base)
        import pixel_aot
        checked = list(words)
        if how == 'c_lz' and (words[2] >> 28) == 1:
            # Word 2 bit 28 is MIP_POINT_PRECLAMP (Mesa gfx10-rsrc.json SQ_IMG_SAMP_WORD2): it changes how a point
            # mip filter clamps the LOD before rounding it to a level. c_lz samples level zero with MIN_LOD 0 (the
            # decoder refuses any other MIN_LOD), so no LOD reaches the mip selection and the bit cannot change the

            # Mesa (bb59ce45 ac_build_sampler_descriptor) never sets the bit, so no open driver documents more.
            checked[2] &= ~(1 << 28)
        clamp_xy = (words[0] & 7, words[0] >> 3 & 7)
        if 6 in clamp_xy and set(clamp_xy) <= {2, 6}:
            # SQ_TEX_CLAMP_BORDER (6, Mesa radv_sampler.c maps VK CLAMP_TO_BORDER to it) in X and/or Y: outside
            # [0, 1] the texel is the border colour, BORDER_COLOR_TYPE (word 3 bits 31:30; 0 transparent black,
            # 1 opaque black, 2 opaque white, Mesa ac_descriptors.c) and the comparison runs against its red
            # channel; D3D12 BORDER with that static colour does the same (Vulkan orders texel replacement before
            # the depth compare as well). Type 3 (a border colour table entry, BORDER_COLOR_PTR) is refused. The
            # runtime carries the colour with X2W_SAMPLER_LOD=1 (sampler_lod_fields); without it the static
            # sampler's border is transparent black, which is exact for type 0 only. The smpc manifest line keeps
            # the original words (the runtime's decode_sampler maps 6 to D3D12_TEXTURE_ADDRESS_MODE_BORDER); the
            # other fields of the contract (filters, compare, LOD, TRUNC_COORD) are checked with CLAMP_LAST_TEXEL.
            if words[3] >> 30 == 3:
                raise Unsupported('%s sampler: border colour from the border colour table (BORDER_COLOR_TYPE 3)' % base)
            checked[0] = checked[0] & ~0x3f | 2 | 2 << 3
        try:
            pixel_aot.comparison_sampler_state(tuple(checked))
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (base, error))
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2da' if array else 'tex2d')
        smp = self.pipe.sampler(tuple(words), comparison=True)
        # Texture2DArray takes the slice as the third location component, like Sample and Load on the same type.
        location = ('float3(%s, %s, %s)' % (self.f(cs[1]), self.f(cs[2]), self.f(cs[3])) if array
                    else 'float2(%s, %s)' % (self.f(cs[1]), self.f(cs[2])))
        args = '%s, %s, %s' % (smp, location, self.f(cs[0]))
        # MAX_LOD (word 1 bits 23:12) is not carried into the static sampler, and the comparison-state mapping ignores
        # the mip filter and uses level zero. Only a source limited to level zero is therefore exact for the implicit
        # form; any other limit would need mip selection and blending that is not lowered in this slice, so it is refused.
        max_lod = (words[1] >> 12) & 0xfff
        if how == 'c' and max_lod:
            raise Unsupported('%s with implicit mip selection (sampler MAX_LOD %#x)' % (base, max_lod))
        call = '%s.SampleCmpLevelZero(%s)' % (tex, args)
        self.set_v(reg_range(ops[0])[1], 'U(%s)' % call)

    def op_image_gather4_lz(self, ops, mods):
        """image_gather4_lz on a 2D texture: the one channel dmask names, from the 2x2 footprint at level 0.

        The four result dwords are the texels in the order Gather*() returns them, (u0,v1), (u1,v1), (u1,v0),
        (u0,v0): gather4 is the D3D Gather operation, and compilers map Gather's four results straight onto the
        four destination VGPRs. Gather always reads the most detailed level, which is what _lz names. Offsets
        (_o), compare (_c), an explicit or biased level, other dimensions and result-layout modifiers (d16, tfe,
        lwe, a16, unorm, r128) would change the address or the result and are refused by name.
        """
        for m in mods:
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc')):
                raise Unsupported('image_gather4_lz modifier %s' % m)
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [int(m[6:], 0) for m in mods if m.startswith('dmask:')]
        if dims != ['SQ_RSRC_IMG_2D'] or len(masks) != 1:
            raise Unsupported('image_gather4_lz on %s' % (dims or 'no dim'))
        channel = {1: 'Red', 2: 'Green', 4: 'Blue', 8: 'Alpha'}.get(masks[0])
        if channel is None:
            raise Unsupported('image_gather4_lz dmask %#x: gather reads exactly one channel' % masks[0])
        dst = reg_range(ops[0])
        if len(ops) != 4 or not dst or dst[0] != 'v' or dst[2] - dst[1] != 3:
            raise Unsupported('image_gather4_lz operands %s' % ', '.join(ops))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)]
        if len(cs) != 2:
            raise Unsupported('image_gather4_lz with %d address registers' % len(cs))
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2d')
        heap = self.heap_sampler(ops[3])
        words = None if heap else self.const_words(ops[3])
        smp = heap or self.pipe.sampler(tuple(words) if words else self.descriptor(ops[3]))
        self.emit('t = %s.Gather%s(%s, float2(%s, %s));' % (tex, channel, smp, self.f(cs[0]), self.f(cs[1])))
        for i, comp in enumerate('xyzw'):
            self.set_v(dst[1] + i, 'U(t.%s)' % comp)

    def op_image_sample_c(self, ops, mods):
        self.image_compare(ops, mods, 'c')

    def op_image_sample_c_lz(self, ops, mods):
        self.image_compare(ops, mods, 'c_lz')

    def op_image_sample_c_l(self, ops, mods):
        self.image_compare(ops, mods, 'c_l')

    def op_image_sample(self, ops, mods):
        self.image(ops, mods, 'sample')

    def op_image_sample_b(self, ops, mods):
        self.image(ops, mods, 'b')

    def op_image_sample_d(self, ops, mods):
        self.image(ops, mods, 'd')

    def op_image_sample_lz(self, ops, mods):
        self.image(ops, mods, 'lz')

    def op_image_get_lod(self, ops, mods):
        self.image(ops, mods, 'get_lod')

    def op_image_sample_l(self, ops, mods):
        self.image(ops, mods, 'l')

    def sample_b_sampler(self, heap, words):
        """The constant sampler words of an image_sample_b, or Unsupported when the guest sampler state is not kept.

        SQ_IMG_SAMP_WORD1 bits 11:0 MIN_LOD and 23:12 MAX_LOD are u4.8 and are carried to MinLOD and MaxLOD of the
        static sampler (translated_sampler_bindings.h, line smpl). SQ_IMG_SAMP_WORD2 bits 13:0 LOD_BIAS (s6.8) and
        19:14 LOD_BIAS_SEC (s2.4) are not carried: the ISA scales them by T#.perf_modulation, which the D3D12 sampler
        has no field for, so a nonzero bias is refused. Mip filter 0 disables mipmapping (ISA: "0=none (disable
        mipmapping, use base-level)"), which a D3D12 sampler cannot express and which would make the bias count, so
        it is refused too. Field layout: Mesa 24.0.0 src/amd/registers/gfx10-rsrc.json, SQ_IMG_SAMP_WORD1/2."""
        if heap:
            raise Unsupported('image_sample_b through a sampler heap (LOD limits and bias unknown)')
        if not words:
            raise Unsupported('image_sample_b with a sampler that is not constant')
        if self.default_sampler_words(words):
            raise Unsupported('image_sample_b through a sampler heap (replaced by a default sampler)')
        return self.lod_sampler_words('image_sample_b', words)

    def default_sampler_words(self, words):
        """True for the four zero words that the default-sampler shortcut of a run-time heap load leaves behind."""
        return list(words) == [0, 0, 0, 0] and 'dynamic sampler index replaced by a default sampler' in self.pipe.notes

    def lod_sampler_words(self, name, words, mip_filter_none_ok=False):
        """The words of a constant sampler whose LOD limits are carried to the static sampler, or Unsupported.

        Shared by image_sample_b and by image_sample, _l and _lz (#2177). Only the bias form refuses a mip filter of
        0 (mip_filter_none_ok False): there the bias would count; the other forms leave that sampler as it was."""
        if words[2] & 0xfffff:
            raise Unsupported('%s with a sampler LOD bias (word 2 %#x)' % (name, words[2]))
        if (words[2] >> 26) & 3 == 0 and not mip_filter_none_ok:
            raise Unsupported('%s with mipmapping disabled (sampler mip filter 0)' % name)
        if words[1] & 0xfff > (words[1] >> 12) & 0xfff:
            raise Unsupported('%s with sampler MIN_LOD %#x above MAX_LOD %#x'
                              % (name, words[1] & 0xfff, (words[1] >> 12) & 0xfff))
        return words

    def op_image_sample_c_d(self, ops, mods):
        raise Unsupported('image_sample_c_d: SampleCmpGrad needs shader model 6.8; only c and c_lz are lowered')

    def op_image_load(self, ops, mods):
        self.image(ops, mods, 'load')

    def op_image_load_mip(self, ops, mods):
        self.image(ops, mods, 'load_mip')

    def fixed_table_image(self, operand, name):
        """Return the table location if the manifest can bind the original eight-word image view."""
        image = reg_range(operand)
        first = self.sym.get(image[1]) if image and image[0] == 's' and image[2] - image[1] == 7 else None
        if not first or first[0] != 'desc' or first[1][0] != 'table' or first[2] is None or first[3] % 8:
            raise Unsupported('%s requires a fixed table image descriptor' % name)
        for i in range(8):
            word = desc_word(first, first[3] + i)
            # A write through a second load of this descriptor can leave these SGPR symbols intact.
            if self.sym.get(image[1] + i) != word or desc_key(word) in self.pipe.modified:
                raise Unsupported('%s through a modified image descriptor' % name)
        return self.descriptor(operand)

    def op_image_store(self, ops, mods):
        """Write one four-channel 2D texel from pixel FP32 or compute packed FP16 data.

        MIMG opcode 0x08 (LLVM AMDGPU MIMGInstructions.td, IMAGE_STORE via MIMG_Store): VDATA is the
        source, VADDR the integer texel address (x, y for dim 2D) and SRSRC the image descriptor; the
        channels selected by DMASK are written in order from consecutive VDATA registers. HLSL's typed UAV
        store writes every channel of the view, so only DMASK 0xf is exact here; a partial mask would need a
        read-modify-write and is refused. The data is stored as float4, so the bound view must be of a
        float or normalized format (the runtime side is not part of this lowering). The store runs under the
        lane's exec bit, like the other per-lane writes, and D3D drops UAV writes from helper lanes as the
        hardware drops them for lanes outside exec.
        """
        name = 'image_store'
        if self.kind not in ('ps', 'cs'):
            raise Unsupported('%s outside a pixel stage' % name)
        d16 = self.kind == 'cs' and mods.count('d16') == 1
        if len(ops) != 3:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        for m in mods:
            if d16 and m == 'd16':
                continue
            # IMAGE_STORE always uses integer texel coordinates, so unorm is a no-op like on image_load.
            if not (m.startswith(('dim:', 'dmask:')) or m in ('dlc', 'glc', 'slc', 'unorm')):
                raise Unsupported('%s modifier %s' % (name, m))
        dims = [m[4:] for m in mods if m.startswith('dim:')]
        masks = [m[6:] for m in mods if m.startswith('dmask:')]
        if len(dims) != 1 or len(masks) != 1:
            raise Unsupported('%s without exactly one dim and dmask' % name)
        if dims[0] != 'SQ_RSRC_IMG_2D':
            raise Unsupported('%s on %s: only 2D images are lowered' % (name, dims[0]))
        try:
            mask = int(masks[0], 0)
        except ValueError:
            raise Unsupported('%s with dmask %s: not an integer' % (name, masks[0])) from None
        if mask != 0xf:
            raise Unsupported('%s with dmask %s: a typed UAV store writes all four channels' % (name, masks[0]))
        data = reg_range(ops[0])
        count = 2 if d16 else 4
        if not data or data[0] != 'v' or data[2] - data[1] + 1 != count or data[2] > 255:
            raise Unsupported('%s data %s: %s VGPRs expected' % (name, ops[0], 'two' if d16 else 'four'))
        coords = ops[1]
        if coords.startswith('['):
            cs = [c.strip() for c in coords[1:-1].split(',')]
        else:
            r = reg_range(coords)
            cs = ['v%d' % n for n in range(r[1], r[2] + 1)] if r and r[0] == 'v' else []
        if len(cs) != 2:
            raise Unsupported('%s with %d address registers: x and y expected' % (name, len(cs)))
        image = self.pipe.image_uav(self.descriptor(ops[2]), 'tex2d')
        value = 'asfloat(uint4(%s))' % ', '.join(self.u('v%d' % n) for n in range(data[1], data[1] + 4))
        body = '%s[uint2(%s, %s)] = %s;' % (image, self.u(cs[0]), self.u(cs[1]), value)
        self.emit(body if self.exec_known else 'if (exec & 1) ' + body)

    def op_image_gather4_c_lz(self, ops, mods):
        """image_gather4_c_lz: four red-channel depth comparisons at view mip zero.

        The address dwords are the reference, then u and v, as on the hardware gather. The comparison
        function and the depth compare live in the constant sampler, so the guest words are bound as a
        SamplerComparisonState and the sample profile is the one pixel_aot.check_gather_sampler already
        checks for comparison gathers. Only red is in contract: gather4_c compares the resource's red
        component and D3D's gather4_c takes a single-component view.
        """
        name = 'image_gather4_c_lz'
        if self.kind != 'ps' or len(ops) != 4:
            raise Unsupported('%s requires a pixel stage and four operands' % name)
        for modifier in mods:
            if not (modifier.startswith(('dim:', 'dmask:')) or modifier in ('dlc', 'glc', 'slc')):
                raise Unsupported('%s modifier %s' % (name, modifier))
        if [m for m in mods if m.startswith('dim:')] != ['dim:SQ_RSRC_IMG_2D']:
            raise Unsupported('%s requires a 2D texture' % name)
        masks = [m for m in mods if m.startswith('dmask:')]
        if masks != ['dmask:0x1']:
            raise Unsupported('%s requires red dmask 0x1' % name)
        dst, addr, image, sampler = (reg_range(ops[i]) for i in (0, 1, 2, 3))
        if not dst or dst[0] != 'v' or dst[2] - dst[1] != 3 or dst[2] > 255:
            raise Unsupported('%s requires four destination registers' % name)
        if not addr or addr[0] != 'v' or addr[2] - addr[1] != 2 or addr[2] > 255:
            raise Unsupported('%s requires three contiguous address registers' % name)
        if not image or image[0] != 's' or image[2] - image[1] != 7 or image[2] > 105:
            raise Unsupported('%s requires eight image descriptor registers' % name)
        if not sampler or sampler[0] != 's' or sampler[2] - sampler[1] != 3 or sampler[2] > 105:
            raise Unsupported('%s requires four sampler descriptor registers' % name)
        words = self.const_words(ops[3])
        if words is None:
            raise Unsupported('%s requires a constant sampler' % name)
        from pixel_aot import check_gather_sampler
        try:
            # Both the guest _lz and D3D gather compare at mip zero of the bound view.
            check_gather_sampler(words, 1, comparison=True)
        except ValueError as error:
            raise Unsupported('%s sampler: %s' % (name, error)) from error
        tex = self.pipe.resource(self.descriptor(ops[2]), 'tex2d')
        smp = self.pipe.sampler(tuple(words), comparison=True)
        reference, u, v = (self.f('v%d' % i) for i in range(addr[1], addr[2] + 1))
        # The gather is read before any destination register is written, so an alias between the address
        # and destination ranges keeps the original operands.
        self.emit('t = %s.GatherCmp(%s, float2(%s, %s), %s);' % (tex, smp, u, v, reference))
        for i, component in enumerate('xyzw', dst[1]):
            self.set_v(i, 'U(t.%s)' % component)

    def op_v_cvt_i32_f32(self, ops, mods):
        if mods:
            raise Unsupported('v_cvt_i32_f32 with modifiers')
        if len(ops) != 2:
            raise Unsupported('v_cvt_i32_f32 with %d operands, expected 2' % len(ops))
        dst = reg_range(ops[0])[1]
        self.set_v(dst, 'uint(int(F(%s)))' % self.u(ops[1]))

    def op_v_pack_b32_f16(self, ops, mods):
        """D[15:0] = S0.f16, D[31:16] = S1.f16 (AMD RDNA 2 ISA, VOP3 V_PACK_B32_F16).

        op_sel[i] selects the high half of source i (LLVM 15 VOP3Instructions.td: V_PACK_B32_F16 is VOP3_OPSEL).
        The neg and abs source modifiers act on the sign bit (bit 15) of the selected half, abs first. The halves
        are copied as bits, like pixel_aot's lowering of the same instruction: no binary16 denormal flush and no
        NaN quieting is modelled (an assumption, not measured). Register sources only: an inline float constant
        on a binary16 operand is a binary16 value, not the binary32 word u() would give. A destination op_sel bit
        and clamp are refused, and so is any modifier other than the one op_sel:[a,b,c] spelling the decoder
        prints."""
        sel = [0, 0, 0]
        for m in mods:
            match = re.fullmatch(r'op_sel:\[([01]),([01]),([01])\]', m)
            if not match:
                raise Unsupported('v_pack_b32_f16 modifier %s' % m)
            sel = [int(x) for x in match.groups()]
        dst = reg_range(ops[0]) if ops else None
        if sel[2] or len(ops) != 3 or not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_pack_b32_f16 form %s %s' % (', '.join(ops), ' '.join(mods)))
        halves = []
        for i, tok in enumerate(ops[1:3]):
            neg = tok.startswith('-') and not re.match(r'-[\d.]', tok)
            tok = tok[1:] if neg else tok
            absolute = tok.startswith('|') and tok.endswith('|')
            tok = tok[1:-1] if absolute else tok
            reg = reg_range(tok)
            if not reg and self.kind == 'ms' and not sel[i]:
                half = '(%s & 0xffffu)' % self.h(tok)
            elif not reg or reg[1] != reg[2] or reg[0] not in ('v', 's'):
                raise Unsupported('v_pack_b32_f16 source %s' % ops[1 + i])
            else:
            # The high half needs the whole word; the low half reads only bits 15:0, so a register whose high half
            # is undefined (after an odd-count d16 load) stays readable here.
                half = '(%s >> 16)' % self.u(tok) if sel[i] else '(%s & 0xffffu)' % self.u(tok, low16_only=True)
            if absolute:
                half = '(%s & 0x7fffu)' % half
            if neg:
                half = '(%s ^ 0x8000u)' % half
            halves.append(half)
        self.set_v(dst[1], '%s | (%s << 16)' % (halves[0], halves[1]))

    def op_v_cvt_pk_u8_f32(self, ops, mods):
        # VOP3A opcode 350 of the RDNA 1.0 and RDNA 2 ISA guides:
        #   D = (S2 & ~(0xff << 8 * S1[1:0])) | ((flt32_to_uint8(S0) & 0xff) << 8 * S1[1:0])
        # The guides leave flt32_to_uint8 undefined. Measured on RDNA 2 hardware (RX 6900 XT, gfx1030, 5432 rows,
        # see docs/experiments.md): round to nearest even, saturated to [0, 255], NaN and negatives (-0.0, -inf
        # included) give 0, +inf gives 255. MODE[1:0] = 3 (round toward zero) truncates instead, which is why a write
        # of those bits is refused in s_setreg_imm32_b32. s_denorm_mode 0 changed nothing.
        # Lowering: (bits - 1u) < 0x7f800000u holds exactly from the smallest positive subnormal to +inf and fails
        # for both zeros, every negative and every NaN, so those take the 0u arm. It is an integer test on the raw
        # word: a float compare may be folded under fast-math (DXC emits `fcmp fast ogt`, so NaN would depend on
        # the driver), and min/max treat a NaN operand differently between back ends.
        # Above zero, min(S0, 255.0) saturates (+inf included) and HLSL round() is round_ne (DXIL Round_ne), so the
        # conversion is exact and in range.
        if mods:
            raise Unsupported('v_cvt_pk_u8_f32 with modifiers')
        if len(ops) != 4:
            raise Unsupported('v_cvt_pk_u8_f32 with %d operands, expected 4' % len(ops))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('v_cvt_pk_u8_f32 destination %s' % ops[0])
        if reg_range(ops[1]):
            src = reg_range(ops[1])
            if src[1] != src[2]:
                raise Unsupported('v_cvt_pk_u8_f32 source %s' % ops[1])
        bits, select, rest = self.u(ops[1]), self.u(ops[2]), self.u(ops[3])
        value = 'F(%s)' % bits
        shift = '((%s & 3u) << 3u)' % select
        self.set_v(dst[1], '((%s & ~(0xffu << %s)) | (((((%s - 1u) < 0x7f800000u) ? uint(round(min(%s, 255.0))) : 0u) & 0xffu) << %s))'
                   % (rest, shift, bits, value, shift))

    WATERFALL_CROSS_LANE = re.compile(r'_dpp$|permlane|readlane|writelane|readfirstlane|^s_store|^s_buffer_store|'
                                      r'^s_atomic|^s_buffer_atomic|^s_scratch|^s_sendmsg|^s_endpgm|^s_setpc|^s_getpc|'
                                      r'^s_swappc|^s_dcache|^exp|^s_movrel|^v_movrel|^s_setreg|^ds_gws|^s_barrier')

    def waterfall(self, s, v):
        """Check that the v_readfirstlane_b32 s<s>, v<v> at self.pc heads a waterfall loop the one-lane model runs"""
        index = {off: i for i, (off, _, _, _) in enumerate(self.ins)}
        h = index[self.pc]
        if h + 1 >= len(self.ins):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        _, _, cname, casm = self.ins[h + 1]
        _, cops, cmods = parse(casm)
        if (not re.fullmatch(r'v_cmpx_eq_[iu]32_e(32|64)', cname) or cmods or
                sorted(cops) != sorted(['s%d' % s, 'v%d' % v])):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        # The loop head is the readfirstlane or a run of waits / nops just above it (LLVM places the wait for the
        # index VGPR's load at the head when the load is outstanding).
        t = h
        while t > 0 and self.ins[t - 1][2] in ('s_waitcnt', 's_nop'):
            t -= 1
        heads = {off for off, _, _, _ in self.ins[t:h + 1]}
        back = [i for i, (off, _, name, asm) in enumerate(self.ins)
                if i > h and name.startswith(('s_cbranch', 's_branch'))
                and int(parse(asm)[1][0].split('_')[1], 16) in heads]
        if len(back) != 1 or self.ins[back[0]][2] != 's_cbranch_scc1':
            raise Unsupported('waterfall loop without one s_cbranch_scc1 back edge')
        b = back[0]
        t = index[int(parse(self.ins[b][3])[1][0].split('_')[1], 16)]
        wname, wasm = self.ins[b - 1][2], self.ins[b - 1][3]
        wops = parse(wasm)[1]
        wide = wname == 's_andn2_wrexec_b64'
        if (wname not in ('s_andn2_wrexec_b32', 's_andn2_wrexec_b64') or len(wops) != 2 or wops[0] != wops[1] or
                not reg_range(wops[0]) or reg_range(wops[0])[0] != 's' or
                reg_range(wops[0])[2] - reg_range(wops[0])[1] != int(wide)):
            raise Unsupported('waterfall loop not closed by s_andn2_wrexec')
        pending = set(range(reg_range(wops[0])[1], reg_range(wops[0])[2] + 1))
        inner = {off for off, _, _, _ in self.ins[t + 1:b + 1]}
        for off, _, name, asm in self.ins:
            if name.startswith(('s_cbranch', 's_branch')):
                target = int(parse(asm)[1][0].split('_')[1], 16)
                if target in inner or (off in inner and off != self.ins[b][0]):
                    raise Unsupported('waterfall loop with another branch')
        body = self.ins[t:b + 1]
        body_writes = set().union(*(sgpr_access(n, a)[1] for _, _, n, a in body))
        written = set()
        for off, _, name, asm in body:
            if off != self.pc and self.WATERFALL_CROSS_LANE.search(name):
                raise Unsupported('waterfall loop body with %s' % name)
            reads, writes = sgpr_access(name, asm)
            if 124 in reads | writes:
                raise Unsupported('waterfall loop body uses m0')
            carried = (reads - written) & body_writes
            if carried - pending:
                raise Unsupported('waterfall loop carries s%d' % min(carried - pending))
            written |= writes
        live = written - pending
        # Every path from the loop exit: each of these SGPRs is written before any read.
        stack, seen = [(b + 1, frozenset(live))], set()
        while stack:
            i, regs = stack.pop()
            while regs and i < len(self.ins) and (i, regs) not in seen:
                seen.add((i, regs))
                off, _, name, asm = self.ins[i]
                reads, writes = sgpr_access(name, asm)
                if reads & regs:
                    raise Unsupported('s%d read after its waterfall loop' % min(reads & regs))
                regs = regs - writes
                if name == 's_endpgm' or name.startswith(('s_setpc', 's_swappc')):
                    break
                if name.startswith(('s_cbranch', 's_branch')):
                    target = index.get(int(parse(asm)[1][0].split('_')[1], 16))
                    if target is None:
                        raise Unsupported('waterfall exit path leaves the program')
                    stack.append((target, regs))
                    if name == 's_branch':
                        break
                i += 1
        # One pass: the lane's own iteration is the only one in which it is active, and it leaves the loop with its
        # exec bit 0, so for this lane the back edge is never taken. Emitting it as a branch would leave a loop
        # whose model condition repeats the first pass exactly (the one-lane state is the same each time), so a
        # body that does not restore exec would spin until a guard; without the edge it runs once.
        if not hasattr(self, 'waterfall_back'):
            self.waterfall_back = set()
        self.waterfall_back.add(self.ins[b][0])

    # ---- slot 57 mesh stage: instructions mesh_skeleton() puts in place of the group-memory exchanges ----
    def op_x2w_mesh_lane0(self, ops, mods):
        # `s_mov_b64 exec, 1` leaves lane 0 of each wave running. The saved copy of that mask is not modelled.
        r = reg_range(ops[0])
        for n in range(r[1], r[2] + 1):
            self.sym[n] = ('the saved mesh lane-0 mask',)
        self.emit('exec = (tid & 63u) == 0u ? 1u : 0u;')
        self.exec_known = False

    def op_x2w_mesh_group(self, ops, mods):
        # The word is entry v5 >> 7 of wave 0, lane 0. What v5 holds at entry is not verified, so a use is refused.
        self.sym[reg_range(ops[0])[1]] = ('the mesh group index',)

    def op_x2w_mesh_counts(self, ops, mods):
        prims, verts, temporary = (reg_range(tok)[1] for tok in ops)
        self.emit('if (tid == 0u) { lds[0] = %s; lds[1] = %s; }' % (self.S(prims), self.S(verts)))
        self.emit('GroupMemoryBarrierWithGroupSync();')
        # Thread t writes vertex t and primitive t, so a count above the thread count names outputs that no
        # thread writes. The clamp keeps the declared arrays in bounds and changes nothing a thread writes.
        for n, word in ((prims, 0), (verts, 1)):
            self.set_s(n, 'min(lds[%d], %du)' % (word, MESH_THREADS), uniform=True)
        # Without a vertex no index is valid, so no primitive is.
        self.emit('if (%s == 0u) %s = 0u;' % (self.S(verts), self.S(prims)))
        self.emit('SetMeshOutputCounts(%s, %s);' % (self.S(verts), self.S(prims)))
        self.mesh_verts = self.S(verts)
        self.sym[temporary] = ('a mesh allocation temporary',)

    def mesh_export(self, asm, target, vals, flags):
        if target == 'target_20':
            # First word: three 9-bit vertex indices at bits 0, 10 and 20, an edge flag above each (not modelled:
            # triangles are filled) and the null bit 31 (Mesa ac_nir_pack_ngg_prim_exp_arg).
            if vals[1:] != ['off'] * 3 or [f for f in flags if f != 'done']:
                raise Unsupported('mesh primitive export form %r' % asm)
            # An index at or past the vertex count has no defined vertex; the clamp keeps it inside the array.
            # A null primitive names one vertex three times, which covers no pixel.
            self.emit('if (exec & 1) { w = %s; tris[tid] = w >> 31 != 0u ? (uint3)0u : '
                      'min(uint3(w, w >> 10, w >> 20) & 511u, max(%s, 1u) - 1u); }'
                      % (self.u(vals[0]), self.mesh_verts))
            return None
        first = len(self.lines)
        self.in_mesh_export = True
        try:
            self.export(asm)
        finally:
            self.in_mesh_export = False
        self.lines[first:] = ['if (exec & 1) { %s mesh_vertex = true; }' % ' '.join(self.lines[first:])]
        return None

    def op_v_mbcnt_lo_u32_b32(self, ops, mods, high=False):
        name = 'v_mbcnt_%s_u32_b32' % ('hi' if high else 'lo')
        if self.kind == 'ms':
            # V_MBCNT counts the mask bits strictly below the lane, in the low or high half of a wave64 mask.
            dst = reg_range(ops[0]) if len(ops) == 3 else None
            if mods or not dst or dst[0] != 'v' or dst[1] != dst[2] or ops[1] != '-1':
                raise Unsupported('%s in a mesh stage requires a VGPR, the mask -1 and a base' % name)
            below = 'max(tid & 63u, 32u) - 32u' if high else 'min(tid & 63u, 32u)'
            self.set_v(dst[1], '(%s + %s)' % (self.u(ops[2]), below))
            return
        if self.kind != 'cs':
            raise Unsupported(name)
        if mods or len(ops) != 3:
            raise Unsupported('%s requires three unmodified operands' % name)
        x, y, z = self.pipe.group
        if x * y * z > 32:
            raise Unsupported('%s requires at most 32 threads per group' % name)
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2] or dst[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        # Both halves use the tested constant-mask profile, even though HI only copies the base.
        mask = self.const_u32(ops[1])
        if mask is None:
            raise Unsupported('%s requires a known mask' % name)
        base = reg_range(ops[2])
        base_is_vgpr = base and base[0] == 'v' and base[1] == base[2] and base[2] <= 255
        if not base_is_vgpr and self.const_u32(ops[2]) is None:
            raise Unsupported('%s requires a VGPR or known base' % name)
        result = self.u(ops[2])
        # RDNA 2 V_MBCNT counts mask bits strictly below the lane. All threads of
        # this group fit in the low half of either guest wave size, including lane 31.
        if not high:
            lane = '(tid.x + %du * (tid.y + %du * tid.z))' % (x, y)
            result = '(countbits(0x%08xu & ((1u << %s) - 1u)) + %s)' % (mask, lane, result)
        self.set_v(dst[1], result)

    def op_v_mbcnt_hi_u32_b32(self, ops, mods):
        self.op_v_mbcnt_lo_u32_b32(ops, mods, high=True)

    def op_v_readfirstlane_b32(self, ops, mods):
        # D = S0 of the first active lane (AMD RDNA 2 ISA; LLVM selects it for a VGPR-to-SGPR copy of a value the
        # divergence analysis found uniform, SIFixSGPRCopies). With a lane-invariant source (see set_v) every
        # lane's own value is that value. Branches are allowed: invariance is cleared at every branch target.
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        offsets = [off for off, _, _, _ in getattr(self, 'ins', ())]
        here = getattr(self, 'pc', None)
        following = self.ins[offsets.index(here) + 1][2] if here in offsets[:-1] else ''
        if (not mods and dst and dst[0] == 's' and dst[1] == dst[2] and src and src[0] == 'v' and src[1] == src[2]
                and src[1] not in getattr(self, 'uniform_v', ()) and getattr(self, 'partial_uniform', {}).get(src[1]) is None
                and following.startswith('v_cmpx_eq_')):
            # a waterfall loop head (see waterfall): this lane's own value
            self.waterfall(dst[1], src[1])
            # An inactive lane's vM is stale (hardware never uses it: no iteration serves that lane), but the HLSL
            # evaluates the body's descriptor reads in every lane, and the heap memory is a root SRV with no bounds
            # check; index 0 keeps that read inside the heap. Active lanes read their own vM as hardware does.
            self.set_s(dst[1], '((exec & 1) != 0 ? %s : 0u)' % self.u(ops[1]))
            return
        partial = getattr(self, 'partial_uniform', {}).get(src[1]) if src and src[0] == 'v' else None
        if (not mods and dst and dst[0] == 's' and partial is not None and
                partial['epoch'] == getattr(self, 'exec_writes', 0)):
            # The source was written, in the lanes active then, from a lane-invariant value (or a chain of such
            # writes, see partial_shadow), and exec has not been written since: the first active lane holds that
            # value. Lanes inactive now kept an older VGPR value, but the SGPR is the wave's, so every lane takes the
            # value itself, from shadows assigned in all lanes at the writes. (With no active lane the hardware
            # reads lane 0; that case is not modelled.)
            self.materialize_shadow(partial)
            self.set_s(dst[1], 'xv%d' % src[1])
            return
        if (mods or not self.exec_known or not dst or dst[0] != 's' or
                not src or src[0] != 'v' or src[1] not in self.uniform_v):
            raise Unsupported('v_readfirstlane_b32 requires a proven uniform source')
        self.set_s(reg_range(ops[0])[1], self.u(ops[1]))

    # ---- SGPR spill to VGPR lanes (#2125, item E) --------------------------------------------
    # The compiler parks a scalar in one lane of a VGPR: `v_writelane_b32 vN, sM, k` writes sM into lane k
    # and `v_readlane_b32 sD, vN, k` reads it back (AMD RDNA 2 ISA reference, VOP3 lane ops; LLVM 15
    # AMDGPU spill lowering). Both ignore EXEC. With a constant k and a VGPR that no other instruction
    # touches, vN is only a table of scalar slots, so one scalar variable per (vN, k) is an exact model
    # and no other lane is ever read. Anything else keeps the one-lane model out of reach and is refused.
    SPILL_LANES = 32

    def spill_operands(self, name, ops, mods):
        if mods or len(ops) != 3:
            raise Unsupported('%s with modifiers or %d operands' % (name, len(ops)))
        vreg, sreg = ((reg_range(ops[0]), reg_range(ops[1])) if name == 'v_writelane_b32'
                      else (reg_range(ops[1]), reg_range(ops[0])))
        if (not vreg or vreg[0] != 'v' or vreg[1] != vreg[2] or vreg[1] > 255 or
                (sreg is not None and (sreg[0] != 's' or sreg[1] != sreg[2] or sreg[1] > 103))):
            raise Unsupported('%s requires one VGPR and one plain SGPR' % name)
        try:
            lane = int(ops[2], 0)
        except ValueError:
            raise Unsupported('%s requires a constant lane, got %s' % (name, ops[2])) from None
        if not 0 <= lane < self.SPILL_LANES:
            raise Unsupported('%s lane %d outside 0..%d' % (name, lane, self.SPILL_LANES - 1))
        if getattr(self, 'kind', None) != 'ps':
            raise Unsupported('%s is lowered in pixel stages only' % name)
        if getattr(self, 'has_branches', True) or not self.tracks_uniformity():
            raise Unsupported('%s needs straight-line code to prove write-before-read' % name)
        self.require_spill_only(vreg[1], name)
        return vreg[1], sreg, lane

    def require_spill_only(self, n, name):
        """Refuse unless every instruction naming v<n> is a lane write or read of it."""
        for _, _, op, asm in getattr(self, 'ins', ()):
            if re.sub(r'_e(32|64)$', '', op) in ('v_writelane_b32', 'v_readlane_b32'):
                continue
            for lo, hi, single in re.findall(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', asm):
                a, b = (int(lo), int(hi)) if lo else (int(single), int(single))
                if a <= n <= b:
                    raise Unsupported('%s: v%d is also used as a vector (%s)' % (name, n, op))

    def op_v_writelane_b32(self, ops, mods):
        n, src, lane = self.spill_operands('v_writelane_b32', ops, mods)
        if src is None:
            if ops[1] in ('vcc', 'vcc_lo', 'vcc_hi', 'exec', 'exec_lo', 'exec_hi', 'm0'):
                raise Unsupported('v_writelane_b32 source %s is a lane mask or m0' % ops[1])
            value, sym, uniform = self.u(ops[1]), None, True
            if re.fullmatch(r'0x[0-9a-f]{8}u', value):
                sym = ('const', int(value[2:-1], 16))
        else:
            if self.is_lane_mask(ops[1]) or (self.sym.get(src[1]) or ('',))[0] not in ('', 'const', 'ud', 'fixed_buffer_word'):
                raise Unsupported('v_writelane_b32 source s%d holds a lane mask or descriptor' % src[1])
            value = self.S(src[1])
            sym = self.sym.get(src[1]) if (self.sym.get(src[1]) or ('',))[0] == 'const' else None
            uniform = self.uniform_source(ops[1])
        slots = self.__dict__.setdefault('spill_slots', {})
        slots[(n, lane)] = (sym, uniform)
        self.emit('sp%d_%d = %s;' % (n, lane, value))

    def op_v_readlane_b32(self, ops, mods):
        if len(ops) != 3 or reg_range(ops[1]) is None or reg_range(ops[1])[0] != 'v':
            raise Unsupported('v_readlane_b32 requires a VGPR source')
        n, dst, lane = self.spill_operands('v_readlane_b32', ops, mods)
        if dst is None:
            raise Unsupported('v_readlane_b32 requires an SGPR destination')
        slot = getattr(self, 'spill_slots', {}).get((n, lane))
        if slot is None:
            raise Unsupported('v_readlane_b32 reads v%d lane %d before any v_writelane_b32 wrote it' % (n, lane))
        sym, uniform = slot
        self.set_s(dst[1], 'sp%d_%d' % (n, lane), sym=sym, uniform=uniform)

    def pair(self, tok, *, unsigned_valu=False):
        """(low, high) uint expressions of a 64-bit integer operand: a register pair or an inline integer constant
        (sign-extended to 64 bits). Unsigned VALU sources zero-extend a 32-bit literal."""
        r = reg_range(tok)
        if r:
            if r[2] != r[1] + 1:
                raise Unsupported('64-bit operand %s' % tok)
            if r[0] == 'v':
                self.used_v.update((r[1], r[2]))
                return 'v%d' % r[1], 'v%d' % r[2]
            return self.S(r[1]), self.S(r[2])
        try:
            value = int(tok, 0)
        except ValueError:
            raise Unsupported('64-bit operand %r' % tok)
        if tok.startswith('0x') and 0 <= value <= (0xffffffff if unsigned_valu else 0x7fffffff):
            # RDNA 2 ISA 6.2.1, Literal Expansion to 64 bits. Both logical shifts read S1.u64 (12.9).
            return '0x%08xu' % value, '0x00000000u'
        if not -16 <= value <= 64 or tok.startswith('0x'):
            raise Unsupported('64-bit literal operand %s' % tok)
        value &= 0xffffffffffffffff
        return '0x%08xu' % (value & 0xffffffff), '0x%08xu' % (value >> 32)

    def shift64(self, ops, mods, helper):
        """v_l{sh,sh}rrev_b64 D, S0, S1: D = S1 shifted by S0[5:0]; the shift count comes first ("rev")."""
        if mods:
            raise Unsupported('64-bit shift with %s' % mods)
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[2] != d[1] + 1:
            raise Unsupported('64-bit shift destination %s' % ops[0])
        low, high = self.pair(ops[2], unsigned_valu=True)
        count = self.u(ops[1])
        self.uses_q = True
        # Read both source words into q before either destination word is written: D and S1 may overlap.
        self.emit('q = uint2(%s_lo(%s, %s, %s), %s_hi(%s, %s, %s));' % (helper, low, high, count, helper, low, high, count))
        self.set_v(d[1], 'q.x')
        self.set_v(d[1] + 1, 'q.y')

    def carry_op(self, ops, mods, kind, carry_in):
        """v_{add,sub,subrev}_co[_ci]_u32 D, SDST, S0, S1[, CI]: 32-bit add or subtract with a carry or borrow out.

        AMD RDNA 2 ISA: V_ADD_CO_CI_U32 D = S0 + S1 + CI, carry = the 33rd bit; V_SUB_CO_CI_U32 D = S0 - S1 - CI,
        borrow = S1 + CI > S0 (as 33-bit values); SUBREV swaps S0 and S1; the _co forms have no carry-in. LLVM
        VOP2Instructions.td defines V_ADD_CO_CI_U32 as gfx10's name for v_addc_u32 (VOP2b, carry-out an SGPR mask).
        The carry is the lane's bit of SDST, as v_cmp writes it; CI must be a lane mask (vcc, exec or an SGPR the
        translation holds as a mask). Both results go through q, so D or SDST may overlap a source.
        """
        if mods:
            raise Unsupported('v_%s_co%s_u32 with %s' % (kind, '_ci' if carry_in else '', ' '.join(mods)))
        if len(ops) != (5 if carry_in else 4):
            raise Unsupported('v_%s_co%s_u32 with %d operands' % (kind, '_ci' if carry_in else '', len(ops)))
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[1] != d[2]:
            raise Unsupported('carry operation destination %s' % ops[0])
        a, b = self.u(ops[2]), self.u(ops[3])
        if kind == 'subrev':
            a, b = b, a
        c = '0u'
        if carry_in:
            r = reg_range(ops[4])
            if not (ops[4] in ('vcc', 'vcc_lo', 'exec', 'exec_lo') or (r and r[0] == 's' and r[1] in self.masks)):
                raise Unsupported('carry-in %s is not a lane mask' % ops[4])
            c = '(%s & 1u)' % self.u(ops[4] if not r else 's%d' % r[1])
        if kind == 'add':
            value, carry = '%s + %s + %s' % (a, b, c), '((%s + %s) < %s || (%s + %s + %s) < (%s + %s))' % (
                a, b, a, a, b, c, a, b)
        else:
            value, carry = '%s - %s - %s' % (a, b, c), '(%s > %s || (%s - %s) < %s)' % (b, a, a, b, c)
        self.uses_q = True
        self.emit('q = uint2(%s, %s ? 1u : 0u);' % (value, carry))
        self.set_v(d[1], 'q.x')
        sdst = ops[1]
        r = reg_range(sdst)
        self.write_compare(False, sdst if not r else 's%d' % r[1], 'q.y')

    def op_v_add_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'add')

    def op_v_sub_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'sub')

    def op_v_subrev_co_ci_u32(self, ops, mods):
        self.carry_in_out_op(ops, mods, 'subrev')

    def op_v_add_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'add')

    def op_v_sub_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'sub')

    def op_v_subrev_co_u32(self, ops, mods):
        self.carry_out_op(ops, mods, 'subrev')

    def op_v_lshrrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'x2w_lshr64')

    def op_v_lshlrev_b64(self, ops, mods):
        self.shift64(ops, mods, 'x2w_lshl64')

    def half_op(self, ops, mods, fn):
        """v_{min,max}_f16: the binary16 result goes to D[15:0]; gfx10 keeps D[31:16] (LLVM
        GCNSubtarget::zeroesHigh16BitsOfDest: "On gfx10, all 16-bit instructions preserve the high bits")."""
        if mods:
            raise Unsupported('f16 operation with %s' % mods)
        if any(t.startswith(('-', '|')) and not re.match(r'-[\d.]', t) for t in ops[1:3]):
            raise Unsupported('f16 operation with source modifiers')
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        a, b = ('f16tof32(%s)' % self.h(t) for t in ops[1:3])
        self.set_v(d, '(v%d & 0xffff0000u) | f32tof16(%s(%s, %s))' % (d, fn, a, b))

    def half_arithmetic(self, ops, mods, form, arity):
        if len(ops) != arity + 1 or mods:
            raise Unsupported('binary16 arithmetic operands or modifiers: %s %s' % (ops, mods))
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[1] != dst[2]:
            raise Unsupported('binary16 arithmetic destination %s' % ops[0])
        d = dst[1]
        self.used_v.add(d)
        args = ['f16tof32(%s)' % self.h(t) for t in ops[1:]]
        expr = form.format(*args, dest='f16tof32(v%d)' % d)
        self.set_v(d, '(v%d & 0xffff0000u) | %s' % (d, self.to_f16(expr)))

    def op_v_mul_f16(self, ops, mods):
        # A finite binary16 product fits exactly in binary32 before rounding.
        self.half_arithmetic(ops, mods, '{0} * {1}', 2)

    def op_v_fmac_f16(self, ops, mods):
        self.half_arithmetic(ops, mods, 'x2w_half_madd_odd({0}, {1}, {dest})', 2)

    def op_v_rsq_f16(self, ops, mods):
        self.half_arithmetic(ops, mods, 'rsqrt({0})', 1)

    def op_v_min_f16(self, ops, mods):
        self.half_op(ops, mods, 'min')

    def op_v_max_f16(self, ops, mods):
        self.half_op(ops, mods, 'max')

    def packed_half(self, ops, mods, fn):
        """v_pk_{min,max}_f16: two binary16 lanes. Result half i reads, from each source, the half picked by
        op_sel (low result) or op_sel_hi (high result); the disassembler prints only non-default selections."""
        sel, sel_hi = [0, 0], [1, 1]
        for m in mods:
            if m.startswith('op_sel:['):
                sel = [int(x) for x in m[8:-1].split(',')]
            elif m.startswith('op_sel_hi:['):
                sel_hi = [int(x) for x in m[11:-1].split(',')]
            else:
                raise Unsupported('packed f16 operation with %s' % m)
        if len(sel) != 2 or len(sel_hi) != 2:
            raise Unsupported('packed f16 selection %s' % mods)
        words = []
        for index, tok in enumerate(ops[1:3]):
            r = reg_range(tok)
            if r:
                words.append(self.u(tok))
            elif re.fullmatch(r'0x[0-9a-f]+', tok) and not sel[index] and not sel_hi[index]:
                words.append(self.u(tok))  # a literal read only through its low half
            elif tok == '0':
                words.append('0x00000000u')  # the inline integer 0: every bit is 0, so each half reads +0.0
            else:
                raise Unsupported('packed f16 operand %s with selection %s/%s' % (tok, sel[index], sel_hi[index]))
        def half(word, high):
            return 'f16tof32(%s >> 16)' % word if high else 'f16tof32(%s)' % word
        low = 'f32tof16(%s(%s, %s))' % (fn, half(words[0], sel[0]), half(words[1], sel[1]))
        high = 'f32tof16(%s(%s, %s))' % (fn, half(words[0], sel_hi[0]), half(words[1], sel_hi[1]))
        self.set_v(reg_range(ops[0])[1], '%s | (%s << 16)' % (low, high))

    def op_v_pk_min_f16(self, ops, mods):
        self.packed_half(ops, mods, 'min')

    def op_v_pk_max_f16(self, ops, mods):
        self.packed_half(ops, mods, 'max')

    def op_v_cvt_f16_f32(self, ops, mods):
        # gfx10 keeps D[31:16] for 16-bit results (see half_op); the SDWA form keeps its own dst_sel handling.
        if mods:
            raise Unsupported('v_cvt_f16_f32 with %s' % mods)
        if len(ops) != 2:
            raise Unsupported('v_cvt_f16_f32 with %d operands, expected 2' % len(ops))
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        self.set_v(d, '(v%d & 0xffff0000u) | %s' % (d, self.to_f16('F(%s)' % self.u(ops[1]))))

    def op_v_mov_b32(self, ops, mods):
        if len(ops) != 2:
            raise Unsupported('v_mov_b32 with %d operands, expected 2' % len(ops))
        self.reject_lane_mask_data('v_mov_b32', ops[1])
        self.set_v(reg_range(ops[0])[1], self.u(ops[1]))

    def op_v_swap_b32(self, ops, mods):
        # RDNA 2 ISA V_SWAP_B32 (VOP1 0x65, LLVM VOP1Instructions.td: outs $vdst, $vdst1 tied to $src0):
        # tmp = D; D = S0; S0 = tmp, in the active lanes. Both operands are single VGPRs; there is no VOP3,
        # DPP or SDWA form (the decoder reports those as unsupported).
        d, s = reg_range(ops[0]) if ops else None, reg_range(ops[1]) if len(ops) == 2 else None
        if mods or len(ops) != 2 or not d or not s or d[0] != 'v' or s[0] != 'v' or d[1] != d[2] or s[1] != s[2]:
            raise Unsupported('v_swap_b32 form %s %s' % (ops, mods))
        if d[1] == s[1]:
            return  # a register swapped with itself keeps its value
        self.uses_swap = True
        self.lines.append('xswap = %s;' % self.u(ops[0]))  # every lane: a temporary read only below
        self.set_v(d[1], self.u(ops[1]))
        self.set_v(s[1], 'xswap')

    def op_v_cndmask_b32(self, ops, mods):
        # The VOP3 form takes neg/abs on its two data operands; the select moves bits, so they are sign-bit
        # operations on the selected word.
        if mods:
            raise Unsupported('v_cndmask_b32 with %s' % mods)
        if len(ops) not in (3, 4):
            raise Unsupported('v_cndmask_b32 with %d operands, expected 3 or 4' % len(ops))
        for source in ops[1:3]:
            self.reject_lane_mask_data('v_cndmask_b32', source)
        mask = self.u(ops[3]) if len(ops) > 3 else 'vcc'
        self.set_v(reg_range(ops[0])[1], '(%s & 1) ? %s : %s' % (mask, self.u_mod(ops[2]), self.u_mod(ops[1])))

    def op_v_interp_p1_f32(self, ops, mods):
        if hasattr(self, 'uniform_v'):
            self.uniform_v.discard(reg_range(ops[0])[1])
        self.note_attr(ops[2], 'linear')

    def op_v_interp_p2_f32(self, ops, mods):
        attr, comp = self.note_attr(ops[2], 'linear')
        self.set_v(reg_range(ops[0])[1], 'U(i.param%d.%s)' % (attr, comp))

    def op_v_interp_mov_f32(self, ops, mods):
        # Only p0 is the flat copy. p10 (P1 - P0) and p20 (P2 - P0) need the per-vertex attribute
        # values, which the FXC path does not have, so they are refused rather than read as p0.
        if ops[1] != 'p0':
            raise Unsupported('v_interp_mov_f32 parameter selector %s: only p0 is lowered' % ops[1])
        attr, comp = self.note_attr(ops[2], 'flat')
        self.set_v(reg_range(ops[0])[1], 'i.fparam%d.%s' % (attr, comp))

    def interp_f16_mods(self, name, mods):
        if any(m != 'high' for m in mods):
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        return 'hi' if 'high' in mods else 'lo'

    def op_v_interp_p1ll_f16(self, ops, mods):
        # First pass of a binary16 attribute (Mesa ACO emit_interp_instr: p1ll then p2_f16). Like p1_f32 it
        # produces no value of its own here: the interpolated half comes from p2.
        self.interp_f16_mods('v_interp_p1ll_f16', mods)
        if len(ops) != 3:
            raise Unsupported('v_interp_p1ll_f16 with %d operands' % len(ops))
        self.note_attr(ops[2], 'half')

    def op_v_interp_p2_f16(self, ops, mods):
        """The f16 attribute half (`high`: bits 31:16 of the exported word) interpolated, rounded to binary16.

        The vertex/mesh stage exports a second and third interpolated copy of the attribute, each half converted to
        f32 (loparamN / hiparamN, see export); the result is f32tof16 of the interpolant. The hardware
        interpolates in f32 and rounds once, as this does (LLVM SIInstrInfo / ACO: p1ll is f32 out, p2 f16 out).
        A 16-bit write: the other half of the VGPR is kept (Mesa aco_ir.cpp instr_is_16bit lists v_interp_p2_f16
        and v_interp_p2_hi_f16 for GFX9+). op_sel bit 3 (VOP3 word 0 bit 14, which LLVM 15 and the decoder do not
        print) writes the high half: Mesa aco_assembler.cpp emits v_interp_p2_hi_f16 as v_interp_p2_f16 with
        opsel 0x8. Other op_sel bits are refused."""
        half = self.interp_f16_mods('v_interp_p2_f16', mods)
        if len(ops) != 4:
            raise Unsupported('v_interp_p2_f16 with %d operands' % len(ops))
        attr, comp = self.note_attr(ops[2], 'half')
        opsel = struct.unpack_from('<I', self.raw, self.pc)[0] >> 11 & 15
        if opsel not in (0, 8):
            raise Unsupported('v_interp_p2_f16 with op_sel %#x' % opsel)
        d = reg_range(ops[0])[1]
        self.used_v.add(d)
        value = self.to_f16('i.%sparam%d.%s' % (half, attr, comp))
        if opsel:
            self.set_v(d, '(v%d & 0xffffu) | (%s << 16)' % (d, value))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | %s' % (d, value))

    def note_attr(self, tok, how):
        m = re.fullmatch(r'attr(\d+)\.([xyzw])', tok)
        attr, component = int(m.group(1)), m.group(2)
        links = getattr(getattr(self, 'pipe', None), 'ps_attr_map', None)  # unit probes may have no pipe
        if links is not None:  # stream pipelines: the vertex parameter the attribute is linked to
            if attr not in links:
                raise Unsupported('pixel stage reads attribute %d, which its interface table does not list' % attr)
            parameter, mask = links[attr]
            if not mask & (1 << 'xyzw'.index(component)):
                raise Unsupported('pixel stage reads undeclared component %s of attribute %d' % (component, attr))
            attr = parameter
        self.attrs.setdefault(attr, set()).add(how)
        return attr, component

    def mix(self, ops, mods, dest):
        """v_fma_mix{_f32,lo_f16,hi_f16}: fma of operands that are f32 or f16 halves.

        LLVM AMDGPUISelDAGToDAG::SelectVOP3PMadMixModsImpl: op_sel_hi[i] makes operand i an f16 converted to f32
        and op_sel[i] then picks its high half; abs and neg act on the converted value (abs first). The result is
        f32 (dest 'f32') or rounds to f16 in the low or high half, the other half kept.
        """
        sel, high = [0, 0, 0], [0, 0, 0]
        for m in mods:
            if m.startswith('op_sel_hi:'):
                sel = [int(x) for x in m[len('op_sel_hi:['):-1].split(',')]
            elif m.startswith('op_sel:'):
                high = [int(x) for x in m[len('op_sel:['):-1].split(',')]
            elif m != 'clamp':
                raise Unsupported('modifier %s on fma_mix' % m)
        if len(ops) != 4 or len(sel) != 3 or len(high) != 3:
            raise Unsupported('fma_mix form %s %s' % (ops, mods))
        def operand(tok, is_half, is_high):
            if not is_half:
                if is_high:
                    raise Unsupported('op_sel on an f32 fma_mix operand')
                if tok.lstrip('-') == '0.15915494':
                    raise Unsupported('fma_mix inline constant 1/(2*pi)')
                # "-0" is the inline 0 with the neg modifier: -0.0 (f() reads it as +0.0)
                return '(-0.0)' if tok == '-0' else self.f(tok)
            if tok in ('0', '-0'):
                return '0.0' if tok == '0' else '(-0.0)'
            if tok in FLOATS and not is_high and tok != '0.15915494':
                # An inline float constant read as an f16 operand is that constant's binary16 form: 0.5 is 0x3800,
                # not the low half of the f32 0x3f000000 (LLVM VOP3PInstructions.td, VOP3P_Mix_Profile: "The inline
                # immediate values are really f16 converted to f32"; its fadd pattern passes FP16_ONE with op_sel_hi).
                # 1/(2*pi) is refused: its f16 and f32 forms differ and only the f16 one is documented here.
                return 'f16tof32(%s)' % self.h(tok)
            neg = tok.startswith('-')
            body = tok[1:] if neg else tok
            absolute = body.startswith('|') and body.endswith('|')
            if absolute:
                body = body[1:-1]
            r = reg_range(body)
            if re.fullmatch(r'0x[0-9a-f]{1,8}', body):
                # A 32-bit literal read as an f16 operand: op_sel picks its half as for a register (LLVM
                # SIFoldOperands::tryFoldImmWithOpSel reads VOP3P literal halves as ImmVal >> (OP_SEL_0 ? 16 : 0)).
                value = int(body, 16)
                text = 'f16tof32(0x%04xu)' % ((value >> 16) if is_high else (value & 0xffff))
            elif not r or r[1] != r[2]:
                raise Unsupported('f16 fma_mix operand %s' % tok)
            else:
                text = 'f16tof32(%s%s)' % (self.u(body), ' >> 16' if is_high else '')
            if absolute:
                text = 'abs(%s)' % text
            return '(-%s)' % text if neg else text
        expr = '%s * %s + %s' % tuple(operand(t, s, h) for t, s, h in zip(ops[1:4], sel, high))
        if 'clamp' in mods:
            expr = 'saturate(%s)' % expr
        d = reg_range(ops[0])[1]
        if dest == 'f32':
            return self.set_v(d, 'U(%s)' % expr)
        self.used_v.add(d)
        if dest == 'hi':
            self.set_v(d, '(v%d & 0xffffu) | (%s << 16)' % (d, self.to_f16(expr)))
        else:
            self.set_v(d, '(v%d & 0xffff0000u) | %s' % (d, self.to_f16(expr)), low16_only=d in self.low16_only_v)

    # s_round_mode writes MODE[3:0] from its immediate: bits 1:0 round f32 results, bits 3:2 round f16 and f64 results
    # (LLVM SIDefines.h FP_ROUND_MODE_SP / FP_ROUND_MODE_DP; Mesa aco_ir.h struct float_mode, round32 : 2 then
    # round16_64 : 2, "matches encoding of the MODE register"), each 0 nearest even, 1 +inf, 2 -inf, 3 toward zero
    # (FP_ROUND_ROUND_TO_*, aco fp_round_*). The denorm bits are s_denorm_mode's and stay refused. Taken: f32 to
    # nearest even (what the HLSL arithmetic does) with f16/f64 to nearest even or toward zero, in the straight-line
    # entry of the program, so the mode at every later instruction is known. Under a nonzero f16/f64 mode the
    # instructions that round an f16 result go through to_f16; check_round16 refuses every other f16/f64 one.
    ROUND16_EXACT = ('v_cvt_f32_f16', 'v_cmp_', 'v_cmpx_', 'v_interp_p1ll_f16', 'v_fma_mix_f32', 'v_min_f16',
                     'v_max_f16', 'v_pk_min_f16', 'v_pk_max_f16', 'v_cvt_pkrtz_f16_f32')
    ROUND16_LOWERED = ('v_mul_f16', 'v_fmac_f16', 'v_rsq_f16', 'v_fma_mixlo_f16', 'v_fma_mixhi_f16', 'v_cvt_f16_f32', 'v_interp_p2_f16')
    round16 = 0  # the class defaults, for a Stage built without __init__ (the operand-count probe)
    uses_round_mode = False
    launch_unknown = False  # a pixel stage built without its launch FLOAT_MODE (launch_mode)
    mode_at = {}            # offset -> MODE[3:2] values reaching it (round_mode_flow), for stages with s_round_mode
    quad_divergent = has_loop = False

    def op_s_round_mode(self, ops, mods):
        value = int(ops[0], 0) if ops and re.fullmatch(r'(0x[0-9a-f]+|\d+)', ops[0]) else None
        if value is None or mods:
            raise Unsupported('s_round_mode %s' % ' '.join(ops + mods))
        if self.kind not in ('ps', 'ms') or (self.kind == 'ms' and not self.mode_at):
            raise Unsupported('s_round_mode in a %s stage' % self.kind)
        # After control flow the layout-order value set here can differ from what some path brings to a later
        # instruction; round_mode_flow finds the values per instruction and one() refuses a mismatch.
        if value & ~0xf or value & 3 or (value >> 2) not in (0, 3):
            raise Unsupported('s_round_mode %#x (only f32 nearest even with f16/f64 nearest even or toward zero)'
                              % value)
        self.round16 = value >> 2
        self.launch_unknown = False  # MODE[3:2] is known from here on

    # FLOAT_MODE denorm bits [7:4] the lowering matches: f32 denormals flushed ([5:4] = 0), f16/f64 kept ([7:6] = 3).
    # D3D flushes f32 denormals in shader arithmetic and keeps f16 ones; the 1,500 measured mesh sets are all 0xc_.
    LAUNCH_DENORM = 0xc0

    def launch_mode(self, mode):
        """Start the pixel stage's MODE from its launch FLOAT_MODE (SPI_SHADER_PGM_RSRC1_PS bits 19:12; the layout"""
        if mode is None:
            self.launch_unknown = True
            return
        if mode & 0xf0 != self.LAUNCH_DENORM or mode & 3 or (mode >> 2 & 3) not in (0, 3):
            raise Unsupported('launch FLOAT_MODE %#04x (only f32 nearest even, f16/f64 nearest even or toward zero, '
                              'denorm %#04x)' % (mode, self.LAUNCH_DENORM))
        self.round16 = mode >> 2 & 3

    def round_mode_flow(self):
        """offset -> the set of MODE[3:2] values (0, 3, or '?' for the unknown launch value, 'x' after an
        s_round_mode op_s_round_mode refuses) that reach it on some path from the entry. Paths follow fall-through
        and branch targets, loops included; only s_round_mode changes the value (s_setreg of MODE is refused)."""
        info = {off: (name, asm) for off, n, name, asm in self.ins}
        order = [ins[0] for ins in self.ins]
        following = dict(zip(order, order[1:]))
        at = {order[0]: {'?' if self.launch_unknown else self.round16}} if order else {}
        todo = list(at)
        while todo:
            off = todo.pop()
            name, asm = info[off]
            out = at[off]
            if name == 's_round_mode':
                ops = parse(asm)[1]
                value = int(ops[0], 0) if ops and re.fullmatch(r'(0x[0-9a-f]+|\d+)', ops[0]) else -1
                out = {value >> 2} if 0 <= value <= 0xf and not value & 3 and value >> 2 in (0, 3) else {'x'}
            if name == 's_endpgm':
                continue
            nxt = [] if name == 's_branch' else [following[off]] if off in following else []
            if name.startswith('s_cbranch') or name == 's_branch':
                nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
            for t in nxt:
                if t in info and not out <= at.get(t, set()):
                    at[t] = at.get(t, set()) | out
                    todo.append(t)
        return at

    def check_round16(self, name):
        """Under f16/f64 round toward zero: refuse an f16/f64 instruction that rounds and is not lowered for it.

        Exact whatever the mode: the f16 -> f32 widening, compares (no result rounding), v_interp_p1ll_f16 and
        v_fma_mix_f32 (f32 results, MODE[1:0]), f16 min/max (the result is an operand), v_cvt_pkrtz_f16_f32 (its
        rounding is toward zero by definition)."""
        if not self.rounds16(name):
            return
        if re.sub(r'_(e32|e64)$', '', name) in self.ROUND16_LOWERED:
            return
        raise Unsupported('%s under s_round_mode f16/f64 toward zero (not lowered)' % name)

    def rounds16(self, name):
        """An f16/f64 instruction whose result depends on MODE[3:2] (not in ROUND16_EXACT)."""
        if not re.search(r'f16|f64', name):
            return False
        return not re.sub(r'_(e32|e64|sdwa|dpp)$', '', name).startswith(self.ROUND16_EXACT)

    def check_launch_known(self, name):
        """A pixel stage whose launch FLOAT_MODE was not passed: refuse an instruction whose result depends on
        MODE[3:2] before an s_round_mode sets it. The launch value is not 0 in general (MEASURED: 0xcc, f16 toward
        zero, in 262 of 1412 mesh pixel stages and 1015 of 3124 draw requests), so neither rounding can be assumed."""
        if self.rounds16(name):
            raise Unsupported('%s: f16/f64 rounding with the launch FLOAT_MODE not passed (.psin third field)' % name)

    def to_f16(self, expr):
        """An f32 value rounded to binary16 under the current MODE[3:2] (see op_s_round_mode)."""
        # HLSL f32tof16 rounds toward zero on WARP and on an RX 9070 XT (MEASURED, startup_round_mode_check), so
        # nearest even (MODE[3:2] = 0) needs its own helper.
        return ('x2w_f32tof16_rtz(%s)' if self.round16 == 3 else 'x2w_f32tof16_rne(%s)') % expr

    def op_v_fma_mix_f32(self, ops, mods):
        self.mix(ops, mods, 'f32')

    def op_v_fma_mix_f32(self, ops, mods):
        self.mix(ops, mods, 'f32')

    def op_v_fma_mixlo_f16(self, ops, mods):
        self.mix(ops, mods, 'lo')

    def op_v_fma_mixhi_f16(self, ops, mods):
        self.mix(ops, mods, 'hi')

    def i64_words(self, tok):
        """(low, high) uint expressions of a 64-bit integer source: a VGPR/SGPR pair or an inline int."""
        pair = reg_range(tok)
        if pair and pair[2] == pair[1] + 1:
            if pair[0] == 'v' and pair[2] <= 255:
                return self.u('v%d' % pair[1]), self.u('v%d' % pair[2])
            if pair[0] == 's' and pair[1] % 2 == 0 and pair[2] <= 105:
                return self.u('s%d' % pair[1]), self.u('s%d' % pair[2])
            raise Unsupported('64-bit source %s' % tok)
        if re.fullmatch(r'0|-?[1-9]\d*', tok) and -16 <= int(tok) <= 64:
            value = int(tok) & 0xffffffffffffffff
            return '%du' % (value & 0xffffffff), '%du' % (value >> 32)
        raise Unsupported('64-bit source %s' % tok)

    def carry_out_op(self, ops, mods, kind):
        """v_{add,sub,subrev}_co_u32 D, SDST, S0, S1: a 32-bit add or subtract with a carry or borrow out.

        AMD RDNA 2 ISA: V_ADD_CO_U32 D = S0 + S1 with the carry out of bit 31 written to the lane's bit of SDST;
        V_SUB_CO_U32 D = S0 - S1 with a borrow when S1 > S0; V_SUBREV_CO_U32 swaps S0 and S1. These forms are VOP3b
        only and take no carry-in. The result and the carry go through the scratch registers a and w first, so D or
        SDST may overlap a source.
        """
        name = 'v_%s_co_u32' % kind
        if mods:
            raise Unsupported('%s with %s' % (name, ' '.join(mods)))
        if len(ops) != 4:
            raise Unsupported('%s with %d operands' % (name, len(ops)))
        d = reg_range(ops[0])
        if not d or d[0] != 'v' or d[1] != d[2] or d[1] > 255:
            raise Unsupported('%s destination %s' % (name, ops[0]))
        sdst = reg_range(ops[1])
        if ops[1] not in ('vcc', 'vcc_lo') and not (sdst and sdst[0] == 's' and sdst[1] == sdst[2] and sdst[1] <= 105):
            raise Unsupported('%s carry-out destination %s' % (name, ops[1]))
        a, b = self.u(ops[2]), self.u(ops[3])
        if kind == 'subrev':
            a, b = b, a
        if kind == 'add':
            self.emit('a = %s + %s;' % (a, b))
            self.emit('w = (%s + %s) < %s ? 1u : 0u;' % (a, b, a))
        else:
            self.emit('a = %s - %s;' % (a, b))
            self.emit('w = %s > %s ? 1u : 0u;' % (b, a))
        self.set_v(d[1], 'a')
        predicate = 'w' if self.exec_known else '(exec & w)'
        if ops[1] in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % predicate)
        else:
            # The carry is a per-lane bit: never a uniform scalar, whatever the scratch name hides.
            self.set_s(sdst[1], predicate, uniform=False)
            self.masks.add(sdst[1])

    def op_v_ashrrev_i64(self, ops, mods):
        raise Unsupported('v_ashrrev_i64 (local shift64 has no arithmetic 64-bit form)')
    def op_v_cvt_pkrtz_f16_f32(self, ops, mods):
        # Round toward zero whatever MODE says (the "rtz" of the name; LLVM int_amdgcn_cvt_pkrtz; Mesa
        # nir_opt_algebraic.py turns pack_half_2x16_split into pack_half_2x16_rtz_split only for backends with
        # has_pack_half_2x16_rtz, the rounding being the difference). HLSL f32tof16 does not promise that rounding (MEASURED toward zero
        # on WARP, startup_round_mode_check), so both halves go through x2w_f32tof16_rtz.
        if len(ops) != 3:
            raise Unsupported('v_cvt_pkrtz_f16_f32 with %d operands, expected 3' % len(ops))
        self.set_v(reg_range(ops[0])[1], 'x2w_f32tof16_rtz(%s) | (x2w_f32tof16_rtz(%s) << 16)'
                   % (self.f(ops[1]), self.f(ops[2])))

    def pknorm(self, ops, mods, kind):
        # S0 packs into D[15:0] and S1 into D[31:16] (per Mesa ACO selection: pack_[us]norm_2x16 of (x, y) is
        # v_cvt_pknorm_[ui]16_f32 x, y); each half is x2w_pknorm_[ui]16, see HELPERS. Source neg/abs apply first.
        if mods or len(ops) != 3:
            raise Unsupported('v_cvt_pknorm_%s16_f32 with %s' % (kind, mods or ops))
        helper = 'x2w_pknorm_%s16' % kind
        self.set_v(reg_range(ops[0])[1], '%s(%s) | (%s(%s) << 16)' % (helper, self.f(ops[1]), helper, self.f(ops[2])))

    def op_v_cvt_pknorm_u16_f32(self, ops, mods):
        self.pknorm(ops, mods, 'u')

    def op_v_cvt_pknorm_i16_f32(self, ops, mods):
        self.pknorm(ops, mods, 'i')

    def _low16_source(self, base, tok, signed):
        """A source of a 16-bit integer compare: bits 15:0 of the operand, sign- or zero-extended (RDNA 2 VOPC
        S0.i16 / S0.u16). A float token is refused: on a 16-bit operand the inline constants 240-248 are binary16
        values (LLVM 15 prints inline 1.0 there as 0x3c00), while u() would give the binary32 word."""
        if tok in FLOATS:
            raise Unsupported('%s source %s: inline float constants are binary16 on 16-bit operands' % (base, tok))
        reg = reg_range(tok)
        if reg:
            if reg[1] != reg[2]:
                raise Unsupported('%s source %s' % (base, tok))
            x = self.u(tok, low16_only=True)
            return '(asint(%s << 16u) >> 16u)' % x if signed else '(%s & 0xffffu)' % x
        try:
            value = int(tok, 0)
        except ValueError:
            # vcc, exec, m0 and modifier spellings: not a source this lowering reads.
            raise Unsupported('%s source %s' % (base, tok))
        # The decoder prints inline integers -16..64 in decimal and a literal in hex; LLVM 15 accepts only 16-bit
        # literals on these operands.
        if not -16 <= value <= 0xffff:
            raise Unsupported('%s source %s: wider than 16 bits' % (base, tok))
        low = value & 0xffff
        return '%d' % (low - 0x10000 if low & 0x8000 else low) if signed else '0x%04xu' % low

    def _low16_source(self, base, tok, signed):
        """A source of a 16-bit integer compare: bits 15:0 of the operand, sign- or zero-extended (RDNA 2 VOPC
        S0.i16 / S0.u16). A float token is refused: on a 16-bit operand the inline constants 240-248 are binary16
        values (LLVM 15 prints inline 1.0 there as 0x3c00), while u() would give the binary32 word."""
        if tok in FLOATS:
            raise Unsupported('%s source %s: inline float constants are binary16 on 16-bit operands' % (base, tok))
        reg = reg_range(tok)
        if reg:
            if reg[1] != reg[2]:
                raise Unsupported('%s source %s' % (base, tok))
            x = self.u(tok, low16_only=True)
            return '(asint(%s << 16u) >> 16u)' % x if signed else '(%s & 0xffffu)' % x
        try:
            value = int(tok, 0)
        except ValueError:
            # vcc, exec, m0 and modifier spellings: not a source this lowering reads.
            raise Unsupported('%s source %s' % (base, tok))
        # The decoder prints inline integers -16..64 in decimal and a literal in hex; LLVM 15 accepts only 16-bit
        # literals on these operands.
        if not -16 <= value <= 0xffff:
            raise Unsupported('%s source %s: wider than 16 bits' % (base, tok))
        low = value & 0xffff
        return '%d' % (low - 0x10000 if low & 0x8000 else low) if signed else '0x%04xu' % low

    def compare(self, base, ops, mods=()):
        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|u32|i32|f16|u16|i16)(_sdwa)?', base)
        if not m or (m.group(4) and m.group(3) not in ('i32', 'u32') and m.group(2) != 'class') \
                or (m.group(3) == 'f16' and m.group(2) not in ('class', 'gt', 'lt')):
            if m and m.group(4) and m.group(3) == 'f32':
                # The f32 predicates are verified on the plain and e64 forms only (#1920); the SDWA forms
                # stay refused until they are verified too.
                raise Unsupported('%s: f32 SDWA compares are not lowered' % base)
            raise Unsupported(base)
        writes_exec, cond, typ = m.group(1) == 'x', m.group(2), m.group(3)
        if typ in ('u16', 'i16'):
            if m.group(4) or cond not in ('lt', 'eq', 'le', 'gt', 'ne', 'ge'):
                raise Unsupported(base)
            if mods:
                raise Unsupported('%s with %s' % (base, mods))
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        else:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        if m.group(4):
            # VOPC SDWA on integers: each source is its selected field (zero- or, with sext(), sign-extended)
            # compared as a 32-bit value; the result goes to vcc, an SGPR mask or (cmpx) exec as for e32.
            fields = dict(mod.partition(':')[::2] for mod in mods)
            if set(fields) != {'src0_sel', 'src1_sel'} or any(v not in SDWA_SEL for v in fields.values()):
                raise Unsupported('%s with %s' % (base, mods))
            if cond == 'class' and typ == 'f16':
                # v_cmp_class_f16 SDWA: S0 is the selected field of a VGPR or SGPR, zero-extended, of which the op
                # reads bits [15:0] as the binary16 value (SDWA source selection, AMD RDNA ISA 6.3.3; for WORD_1 that
                # is the high half); neg()/abs() act on bit 15 of the selected value. S1 is the integer class mask
                # through its own selection, as for f32. A constant S0 is refused: its f16 inline form is not modelled.
                # The disassembly spells the modifiers -vN and |vN| on registers (neg(N)/abs(N) on constants).
                # -|vN| is abs then neg (SDWA applies abs before neg, as VOP3): bit 15 set.
                mm = re.fullmatch(r'(-)?(?:\|(\w+)\||(\w+))', a)
                src = (mm.group(2) or mm.group(3)) if mm else a
                if src.startswith('sext(') or not reg_range(src):
                    raise Unsupported('%s with f16 value operand %s' % (base, a))
                value = sdwa_source(self.u(src), fields['src0_sel'], False)
                if mm and mm.group(2):
                    value = '(%s %s)' % (value, '| 0x8000u' if mm.group(1) else '& 0x7fffu')
                elif mm and mm.group(1):
                    value = '(%s ^ 0x8000u)' % value
                if b.startswith(('sext(', 'neg(', 'abs(')):
                    raise Unsupported('%s with a modified class mask %s' % (base, b))
                expr = 'x2w_class_f16(%s, %s)' % (value, sdwa_source(self.u(b), fields['src1_sel'], False))
                return self.write_compare(writes_exec, dst, expr)
            if cond == 'class':
                # v_cmp_class_f32 SDWA: S0 is the f32 value (whole dword only; neg()/abs() act on its sign bit as
                # for the e64 form), S1 the integer class mask read through its selection (a compiler uses
                # src1_sel:BYTE_n to test bits of one byte against constant classes such as -0.0).
                if fields['src0_sel'] != 'DWORD':
                    raise Unsupported('%s with a sub-dword f32 value' % base)
                mm = re.fullmatch(r'(neg|abs)\((.*)\)', a)
                value = self.u(mm.group(2) if mm else a)
                if mm:
                    value = '(%s %s)' % (value, '^ 0x80000000u' if mm.group(1) == 'neg' else '& 0x7fffffffu')
                if b.startswith(('sext(', 'neg(', 'abs(')):
                    raise Unsupported('%s with a modified class mask %s' % (base, b))
                expr = 'x2w_class_f32(%s, %s)' % (value, sdwa_source(self.u(b), fields['src1_sel'], False))
                return self.write_compare(writes_exec, dst, expr)
            words = []
            for token, key in ((a, 'src0_sel'), (b, 'src1_sel')):
                sext = token.startswith('sext(') and token.endswith(')')
                words.append(sdwa_source(self.u(token[5:-1] if sext else token), fields[key], sext))
            x, y = words if typ == 'u32' else ['int(%s)' % w for w in words]
            return self.compare_result(base, cond, writes_exec, dst, x, y)
        elif mods:
            raise Unsupported('%s with %s' % (base, mods))
        if cond == 'class':
            # v_cmp_class: S0 is the value, S1 the class mask (llvm.amdgcn.class(x, mask)); neg/abs on S0 act on
            # its sign bit, and an f16 value sits in the low half with inline float constants in half form.
            if typ == 'f16':
                value = self.u_mod(a, 0x8000) if reg_range(a.strip('-|')) else self.h(a)
                expr = 'x2w_class_f16(%s, %s)' % (value, self.u(b))
            else:
                expr = 'x2w_class_f32(%s, %s)' % (self.u_mod(a), self.u(b))
            return self.write_compare(writes_exec, dst, expr)
        if typ == 'f16':
            words = [self.u_mod(t, 0x8000) if reg_range(t.strip('-|')) else self.h(t) for t in (a, b)]
            op = '>' if cond == 'gt' else '<'
            expr = '((%s & 0x7fffu) <= 0x7c00u && (%s & 0x7fffu) <= 0x7c00u && f16tof32(%s) %s f16tof32(%s))'
            return self.write_compare(writes_exec, dst, expr % (words[0], words[1], words[0], op, words[1]))
        if typ == 'f32':
            x, y = self.f(a), self.f(b)
        elif typ in ('i16', 'u16'):
            if not writes_exec and dst not in ('vcc', 'vcc_lo') and not re.fullmatch(r's[0-9]+', dst):
                raise Unsupported('%s into %s' % (base, dst))
            x, y = (self._low16_source(base, t, typ == 'i16') for t in (a, b))
        elif typ == 'i32':
            x, y = 'int(%s)' % self.u(a), 'int(%s)' % self.u(b)
        else:
            x, y = self.u(a), self.u(b)
        ordered = ''
        if typ == 'f32':
            def word(tok):
                # A float source's word before its neg/abs modifiers, as f() reads them; they keep a NaN a NaN.
                if tok.startswith('-') and not re.match(r'-[\d.]', tok):
                    tok = tok[1:]
                if tok.startswith('|') and tok.endswith('|'):
                    tok = tok[1:-1]
                return self.u(tok)
            ordered = ' && '.join('(%s & 0x7fffffffu) <= 0x7f800000u' % word(t) for t in (a, b))
        return self.compare_result(base, cond, writes_exec, dst, x, y, ordered)

    def compare_result(self, base, cond, writes_exec, dst, x, y, ordered=''):
        # f32 lg is ordered and not equal (false when either source is NaN) and nlg its negation (#1920).
        # HLSL != is unordered, so lg reads as less or greater, behind {2}: both source words not NaN,
        # tested on their bits. DXC marks float compares fast unless -Gis, and a driver may then fold
        # less-or-greater into an unordered not-equal.
        # FXC without /Gis compiles !(a < b) to a >= b, false on NaN, and DXC without -Gis folds x != x
        # and x == x to constants (#1936). So nlt, nle, ngt and nge negate an ordered relation behind {2},
        # and o and u read {2} alone; the startup cache compiles with FXC and no flags.
        table = {'lt': '{0} < {1}', 'eq': '{0} == {1}', 'le': '{0} <= {1}', 'gt': '{0} > {1}',
                 'lg': '({2} && ({0} < {1} || {0} > {1}))', 'ne': '{0} != {1}',
                 'ge': '{0} >= {1}', 'neq': '!({0} == {1})', 'nlt': '!({2} && {0} < {1})',
                 'nle': '!({2} && {0} <= {1})', 'ngt': '!({2} && {0} > {1})', 'nge': '!({2} && {0} >= {1})',
                 'nlg': '!({2} && ({0} < {1} || {0} > {1}))', 'o': '({2})', 'u': '!({2})',
                 't': 'true', 'f': 'false'}
        if cond not in table:
            raise Unsupported(base)
        if '{2}' in table[cond] and not ordered:
            # Only the plain and e64 f32 forms supply the source words the ordered test needs.
            raise Unsupported('%s: no f32 source words for the ordered test' % base)
        return self.write_compare(writes_exec, dst, '((%s) ? 1u : 0u)' % table[cond].format(x, y, ordered))

    def write_compare(self, writes_exec, dst, expr):
        """Store a 0/1 compare result as the lane's bit of exec, vcc or an SGPR mask."""
        if not self.exec_known:
            expr = '(exec & %s)' % expr
        if writes_exec or dst in ('exec', 'exec_lo'):
            self.exec_known = False
            self.emit('exec = %s;' % expr)
        elif dst in ('vcc', 'vcc_lo'):
            self.emit('vcc = %s;' % expr)
        else:
            r = reg_range(dst)
            if not r or r[0] != 's':
                # e.g. vcc_hi or exec_hi as a carry-out / compare destination: not an SGPR the one-lane model holds
                raise Unsupported('compare or carry destination %s' % dst)
            d = r[1]
            self.set_s(d, expr)
            self.masks.add(d)

    def export(self, asm):
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm| en:0x[0-9a-f]+)*)', asm)
        if not m:
            # e.g. a compressed export with an unpaired enable mask, which the disassembler prints as en:<mask>
            raise Unsupported('export form %r' % asm)
        target, vals, flags = m.group(1), [v.strip() for v in m.group(2).split(',')], m.group(3).split()
        if getattr(self, 'kind', None) == 'ms' and not getattr(self, 'in_mesh_export', False):
            return self.mesh_export(asm, target, vals, flags)
        if any(f.startswith('en:') for f in flags):
            raise Unsupported('compressed export enable mask %s does not select whole channel pairs'
                              % next(f for f in flags if f.startswith('en:'))[3:])
        if 'vm' in flags and not self.exec_known:
            self.emit('if (!(exec & 1)) discard;')
        if target == 'null':
            return None  # the null target writes no render target; its only effect is the vm discard above
        if target == 'mrtz':
            # Z export: x is depth, y stencil, z the sample mask, w alpha for alpha-to-coverage. Only a depth-only
            # export (SPI_SHADER_32_R) has an HLSL output, SV_Depth; the others are refused by channel.
            if self.kind != 'ps' or 'compr' in flags or vals[0] == 'off' or any(v != 'off' for v in vals[1:]):
                raise Unsupported('mrtz export other than depth alone: %s' % asm)
            self.exports['depth'] = True
            self.emit('o.depth = F(%s);' % self.u(vals[0]))
            return None
        if target == 'pos1' and self.kind == 'vs' and getattr(self.pipe, 'vs_out_cntl', None) is not None \
                and 'compr' not in flags:
            # With VS_OUT_MISC_VEC_ENA clear and VS_OUT_CCDIST0_VEC_ENA set, the export after pos0 is the first
            # clip/cull distance vector (see clip_vector_width); D3D: SV_ClipDistance0.
            n = clip_vector_width(self.pipe.vs_out_cntl, vals)
            self.exports['clip'] = n
            parts = ', '.join('F(%s)' % self.u(v) for v in vals[:n])
            self.emit('o.clip0 = %s;' % (parts if n == 1 else 'float%d(%s)' % (n, parts)))
            return None
        name = {'pos0': 'pos'}.get(target, target)
        if not re.fullmatch(r'pos|param\d+|mrt[0-7]', name):
            raise Unsupported('export target %s' % target)
        kind = self.pipe.mrt_kind(int(name[3:])) if name.startswith('mrt') else 'float'
        if len(vals) != 4:
            raise Unsupported('export with %d operands' % len(vals))
        # The channels the export writes. An 'off' channel is not written, so a render target keeps
        # what it held there; the pipeline's write mask carries that (manifest 'wmask', #1674).
        mask = sum(1 << c for c, v in enumerate(vals) if v != 'off')
        if 'compr' in flags:
            if name.startswith('param'):
                self.__dict__.setdefault('compr_params', set()).add(int(name[5:]))
            if kind in ('uint', 'int'):
                raise Unsupported('packed integer export %s' % name)
            # A compressed export takes x, y from the two halves of its first register and z, w from the second.
            # The enable mask is per pair (0x3 and 0xc), and the disassembler prints each enabled register twice
            # and a disabled pair as "off, off"; a disabled pair reads 0 like an "off" channel of a plain export.
            if vals[0] != vals[1] or vals[2] != vals[3]:
                raise Unsupported('compressed export operands %s' % ', '.join(vals))
            exprs = []
            for v in (vals[0], vals[2]):
                if v == 'off':
                    exprs += ['0.0', '0.0']
                else:
                    word = self.u(v)
                    exprs += ['f16tof32(%s)' % word, 'f16tof32(%s >> 16)' % word]
            words = ['0u' if e == '0.0' else 'U(%s)' % e for e in exprs]
        elif kind in ('uint', 'int'):
            exprs = ['0u' if v == 'off' else self.u(v) for v in vals]
            words = exprs
        elif kind == 'bits':
            ud = self.pipe.ud(('mrtbits',))
            exprs = ['0.0' if v == 'off' else 'F((%s) ^ ud%d)' % (self.u(v), ud) for v in vals]
            words = None
        else:
            exprs = ['0.0' if v == 'off' else 'F(%s)' % self.u(v) for v in vals]
            words = ['0u' if v == 'off' else self.u(v) for v in vals]
            if name.startswith('param'):
                self.__dict__.setdefault('plain_params', {})[int(name[5:])] = {c for c, v in enumerate(vals) if v == 'off'}
        if name.startswith('mrt'):
            # One pipeline write mask per target: an export of no channel, or two exports of one target
            # with different channels, cannot be represented and is refused.
            if not mask:
                raise Unsupported('export to %s writes no channel' % name)
            known = self.__dict__.setdefault('write_masks', {}).setdefault(name, mask)
            if known != mask:
                raise Unsupported('exports to %s write channel masks 0x%x and 0x%x' % (name, known, mask))
        self.exports[name] = True
        expr = '%s(%s)' % ('uint4' if kind in ('uint', 'int') else 'float4', ', '.join(exprs))
        self.emit('o.%s = %s;' % (name, 'asint(%s)' % expr if kind == 'int' else expr))
        if name.startswith('param'):
            # The flat copy read by v_interp_mov_f32 carries the register words. A float copy would let
            # FXC fold asfloat() of a constant and flush integer bit patterns that are binary32
            # denormals (#1259). hlsl_vs() drops the line for attributes the pixel stage reads only
            # interpolated.
            self.emit('o.f%s = uint4(%s);' % (name, ', '.join(words)))
            if 'compr' in flags:
                # The words of a compressed export are not the registers' halves (each half went to its own
                # channel), so a binary16 read of this attribute is refused (Pipeline.check_half_reads).
                self.__dict__.setdefault('compr_params', set()).add(int(name[5:]))
                return None
            # The binary16 halves of each word as f32, interpolated for v_interp_p2_f16 (the hardware interpolates
            # each half of the parameter word). Dropped like the flat copy when the pixel stage does not read them.
            self.emit('o.lo%s = float4(%s);' % (name, ', '.join(
                '0.0' if w == '0u' else 'f16tof32(%s)' % w for w in words)))
            self.emit('o.hi%s = float4(%s);' % (name, ', '.join(
                '0.0' if w == '0u' else 'f16tof32((%s) >> 16)' % w for w in words)))
        return None

    # ---- output ------------------------------------------------------------------------------
    LOOP_PASSES = 64

    def body(self):
        """Statements of the whole program as structured HLSL."""
        out = []
        offsets = [b[0] for b in self.blocks]
        index = {off: i for i, off in enumerate(offsets)}
        if len(self.blocks) == 1:
            return ['    ' + line for line in self.blocks[0][1]]

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        out.append('    uint pc = 0;')
        # Only the blocks between a loop head and its back edge run inside a dispatch loop: the intervals
        # [head, source] of the back edges, merged where they overlap. A branch out of a region goes forward (a
        # backward one would be a back edge inside the region), so the blocks outside the regions, tested in offset
        # order, run each at most once like the loop-free form. Same execution as one loop over all blocks; FXC
        # takes minutes on a large dispatch loop, so the loop is kept as small as the control flow allows.
        regions = []
        for j, (_, _, term) in enumerate(self.blocks):
            if term and term[0] in ('jump', 'branch') and goto(term[-1]) <= j:
                regions.append([goto(term[-1]), j])
        regions.sort()
        merged = []
        for lo, hi in regions:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        starts = {lo: hi for lo, hi in merged}
        inside = None
        for i, (off, lines, term) in enumerate(self.blocks):
            if i in starts:
                inside = starts[i]
                # A hard bound so that a wrong lowering costs a wrong pixel, never a GPU timeout: LOOP_PASSES
                # passes through the region (one block visit per guard step). ASSUMED enough for the title's loops.
                out.append('    [loop] for (uint guard = 0; guard < %du && pc >= %d && pc <= %d; ++guard) {'
                           % (self.LOOP_PASSES * (inside - i + 1), i, inside))
            indent = '    ' if inside is not None else ''
            first = inside is not None and i in starts
            out.append('    %s%sif (pc == %d) {' % (indent, 'else ' if inside is not None and not first else '', i))
            for line in lines:
                if line in getattr(self, 'uniform_dpp', ()):
                    out.extend(['    }', '    ' + line, '    if (pc == %d) {' % i])
                else:
                    out.append('        ' + indent + line)
            nxt = i + 1 if i + 1 < len(self.blocks) else 0xffff
            if not term:
                out.append('        %spc = %d;' % (indent, nxt))
            elif term[0] == 'end':
                out.append('        %spc = 0xffffu;' % indent)
            elif term[0] == 'jump':
                out.append('        %spc = %d;' % (indent, goto(term[1])))
            else:
                out.append('        %spc = (%s) ? %d : %d;' % (indent, term[1], goto(term[2]), nxt))
            out.append('    %s}' % indent)
            if inside is not None and i == inside:
                out.append('    }')
                inside = None
        return out

    def declarations(self):
        lines = []
        if self.used_s:
            lines.append('    uint %s;' % ', '.join('s%d = 0' % n for n in sorted(self.used_s)))
        if self.used_v:
            lines.append('    uint %s;' % ', '.join('v%d = 0' % n for n in sorted(self.used_v)))
        if getattr(self, 'shadow_v', None):
            lines.append('    uint %s;' % ', '.join('xv%d = 0' % n for n in sorted(self.shadow_v)))
        packed = getattr(self.pipe, 'packed_position_vgpr', None)  # the tests' bare pipelines have none
        if self.kind == 'ps' and packed is not None and packed in self.used_v:
            lines.append('    v%d = (uint(i.pos.x) & 0xffffu) | ((uint(i.pos.y) & 0xffffu) << 16);' % packed)
        if self.kind == 'ps':
            for reg, bit in sorted(getattr(self.pipe, 'ps_input_vgprs', {}).items()):
                lines.append('    v%d = %s;  // SPI input %s' % (reg, PS_INPUT_HLSL[bit], PS_INPUTS[bit][0]))
        if getattr(self, 'spill_slots', None):
            lines.append('    uint %s;' % ', '.join('sp%d_%d = 0' % k for k in sorted(self.spill_slots)))
        lines.append('    uint vcc = 0, exec = 1, m0 = 0, a = 0, w = 0; bool scc = false; float4 t = 0;')
        if self.used_h:
            lines.append('    uint %s;' % ', '.join('h%d = 0' % k for k in sorted(self.used_h)))
        if self.uses_q:
            lines.append('    uint2 q = 0;')
        if self.uses_swap:
            lines.append('    uint xswap = 0;')
        if self.uses_xsave:
            lines.append('    uint xsave = 0;')
        return lines


# Identifiers Stage.lane_invariant accepts in an expression besides registers: pure HLSL functions and the
# translation's own pure helpers (x2w_* are accepted by prefix), the buffer Load method and literals.
LANE_INVARIANT_NAMES = frozenset((
    'F', 'U', 'asuint', 'asfloat', 'asint', 'uint', 'int', 'float', 'uint2', 'float2', 'float3', 'float4',
    'min', 'max', 'abs', 'saturate', 'sqrt', 'rsqrt', 'exp2', 'log2', 'floor', 'ceil', 'trunc', 'frac', 'round',
    'sin', 'cos', 'countbits', 'firstbitlow', 'firstbithigh', 'reversebits', 'f16tof32', 'f32tof16',
    'mul_legacy', 'med3', 'clamp', 'Load', 'true', 'false'))
# The multiply-add rule for draw stages, X2W_DRAW_FMA: exact (default) = fused ops through fma in binary64 then one
# rounding (the binary32 product is exact in binary64, so this is the fused result except in rare double-rounding
# cases; needs DoublePrecisionFloatShaderOps), unfused ops as a `precise` multiply then add; precise = both through
# the precise multiply-add (deterministic, not the console's fused value); legacy = `a * b + c` as before (the
# compiler may fuse or not). Names differ from mesh2hlsl's x2w_fma/x2w_madp (same bodies) so a mesh pipeline, which
# emits its own, never gets two definitions.
FMA_FUSED = ('v_fma_f32', 'v_fmac_f32', 'v_fmamk_f32', 'v_fmaak_f32')
FMA_UNFUSED = ('v_mad_f32', 'v_mac_f32', 'v_madmk_f32', 'v_madak_f32')
DRAW_FMA = os.environ.get('X2W_DRAW_FMA', 'exact')
if DRAW_FMA not in ('exact', 'precise', 'legacy'):
    raise SystemExit('X2W_DRAW_FMA: exact, precise or legacy, not %r' % DRAW_FMA)
# name -> (source operand count, expression); operands are floats
VEC_F = {
    'v_add_f32': (2, '{0} + {1}'), 'v_sub_f32': (2, '{0} - {1}'), 'v_subrev_f32': (2, '{1} - {0}'),
    'v_mul_f32': (2, '{0} * {1}'), 'v_mul_legacy_f32': (2, 'mul_legacy({0}, {1})'),
    'v_min_f32': (2, 'min({0}, {1})'), 'v_max_f32': (2, 'max({0}, {1})'),
    'v_fmac_f32': (2, '{0} * {1} + {2}'), 'v_mac_f32': (2, '{0} * {1} + {2}'),
    'v_fma_f32': (3, '{0} * {1} + {2}'), 'v_mad_f32': (3, '{0} * {1} + {2}'), 'v_mad_legacy_f32': (3, 'mul_legacy({0}, {1}) + {2}'),
    'v_fmamk_f32': (3, '{0} * {1} + {2}'), 'v_fmaak_f32': (3, '{0} * {1} + {2}'),
    'v_madmk_f32': (3, '{0} * {1} + {2}'), 'v_madak_f32': (3, '{0} * {1} + {2}'),
    'v_rcp_f32': (1, '1.0 / {0}'), 'v_rcp_iflag_f32': (1, '1.0 / {0}'), 'v_rsq_f32': (1, 'rsqrt({0})'), 'v_sqrt_f32': (1, 'sqrt({0})'),
    'v_exp_f32': (1, 'exp2({0})'), 'v_log_f32': (1, 'log2({0})'), 'v_floor_f32': (1, 'floor({0})'), 'v_ceil_f32': (1, 'ceil({0})'),
    'v_trunc_f32': (1, 'trunc({0})'), 'v_fract_f32': (1, 'fract32({0})'), 'v_rndne_f32': (1, 'round({0})'),
    'v_sin_f32': (1, 'sin({0} * 6.283185307)'), 'v_cos_f32': (1, 'cos({0} * 6.283185307)'),
    'v_med3_f32': (3, 'med3({0}, {1}, {2})'), 'v_min3_f32': (3, 'min({0}, min({1}, {2}))'), 'v_max3_f32': (3, 'max({0}, max({1}, {2}))'),
    # Cube-map face selection; x2w_cube() returns (face id, sc, tc, 2 * major axis), see HELPERS.
    'v_cubeid_f32': (3, 'x2w_cube({0}, {1}, {2}).x'), 'v_cubesc_f32': (3, 'x2w_cube({0}, {1}, {2}).y'),
    'v_cubetc_f32': (3, 'x2w_cube({0}, {1}, {2}).z'), 'v_cubema_f32': (3, 'x2w_cube({0}, {1}, {2}).w'),
}
# 16-bit VALU: (sources, source kind, float or uint expression of the sources, result kind). 'h' sources are
# binary16 read from the low half and computed in binary32; an 'h' result is rounded back to binary16. The
# i16/u16 conversions to binary16 read the low half as an integer; the conversions to integers truncate toward
# zero, saturate to the 16-bit range and read NaN as 0.
VEC_H = {
    'v_add_f16': (2, 'h', '{0} + {1}', 'h'), 'v_sub_f16': (2, 'h', '{0} - {1}', 'h'),
    'v_subrev_f16': (2, 'h', '{1} - {0}', 'h'), 'v_mul_f16': (2, 'h', '{0} * {1}', 'h'),
    'v_min_f16': (2, 'h', 'min({0}, {1})', 'h'), 'v_max_f16': (2, 'h', 'max({0}, {1})', 'h'),
    'v_fmac_f16': (2, 'h', 'fma16o({0}, {1}, {2})', 'h'), 'v_fmamk_f16': (3, 'h', 'fma16o({0}, {1}, {2})', 'h'),
    'v_fmaak_f16': (3, 'h', 'fma16o({0}, {1}, {2})', 'h'),
    'v_cvt_f16_i16': (1, 'i16', '{0}', 'h'), 'v_cvt_f16_u16': (1, 'u16', '{0}', 'h'),
    'v_cvt_i16_f16': (1, 'h', 'isnan({0}) ? 0u : (uint(int(clamp(trunc({0}), -32768.0, 32767.0))) & 0xffffu)', 'u'),
    'v_cvt_u16_f16': (1, 'h', 'isnan({0}) ? 0u : uint(clamp(trunc({0}), 0.0, 65535.0))', 'u'),
}
# Conversions whose result is binary32. The expression is the float value, so an output modifier
# (clamp, mul:2, mul:4, div:2) can wrap it the way it wraps any other float result. The uint-result
# conversions stay in VEC_U.
VEC_U_TO_F = {
    'v_cvt_f32_u32': (1, 'float({0})'), 'v_cvt_f32_i32': (1, 'float(int({0}))'),
    'v_cvt_f32_ubyte0': (1, 'float({0} & 255)'), 'v_cvt_f32_ubyte1': (1, 'float(({0} >> 8) & 255)'),
    'v_cvt_f32_ubyte2': (1, 'float(({0} >> 16) & 255)'), 'v_cvt_f32_ubyte3': (1, 'float({0} >> 24)'),
    'v_cvt_f32_f16': (1, 'f16tof32({0})'),
    'v_cvt_off_f32_i4': (1, '(float(int({0} << 28) >> 28) * 0.0625)'),
}
# operands are uints; the result is a uint expression
VEC_U = {
    'v_add_nc_u32': (2, '{0} + {1}'), 'v_sub_nc_u32': (2, '{0} - {1}'), 'v_subrev_nc_u32': (2, '{1} - {0}'),
    'v_mul_lo_u32': (2, '{0} * {1}'), 'v_mul_hi_u32': (2, 'mul_hi_u32({0}, {1})'), 'v_and_b32': (2, '{0} & {1}'), 'v_or_b32': (2, '{0} | {1}'), 'v_xor_b32': (2, '{0} ^ {1}'),
    'v_not_b32': (1, '~{0}'), 'v_lshlrev_b32': (2, '{1} << ({0} & 31)'), 'v_lshrrev_b32': (2, '{1} >> ({0} & 31)'),
    'v_ashrrev_i32': (2, 'uint(int({1}) >> ({0} & 31))'), 'v_min_u32': (2, 'min({0}, {1})'), 'v_max_u32': (2, 'max({0}, {1})'),
    'v_min_i32': (2, 'uint(min(int({0}), int({1})))'), 'v_max_i32': (2, 'uint(max(int({0}), int({1})))'),
    # Three-operand integer forms (RDNA2 ISA: V_MIN3/V_MAX3/V_MED3 _I32/_U32; median = max(min(a, b), min(max(a, b), c)),
    # the identity the f32 med3 helper uses).
    'v_min3_u32': (3, 'min({0}, min({1}, {2}))'), 'v_max3_u32': (3, 'max({0}, max({1}, {2}))'),
    'v_min3_i32': (3, 'uint(min(int({0}), min(int({1}), int({2}))))'),
    'v_max3_i32': (3, 'uint(max(int({0}), max(int({1}), int({2}))))'),
    'v_med3_u32': (3, 'max(min({0}, {1}), min(max({0}, {1}), {2}))'),
    'v_med3_i32': (3, 'uint(max(min(int({0}), int({1})), min(max(int({0}), int({1})), int({2}))))'),
    'v_cvt_u32_f32': (1, 'uint(max(F({0}), 0.0))'),
    'v_cvt_f16_f32': (1, 'x2w_f32tof16_rne(F({0}))'),  # nearest even (MODE 0; check_round16 refuses other modes)
    'v_bfe_u32': (3, '({0} >> ({1} & 31)) & ((1u << ({2} & 31)) - 1)'), 'v_mad_u32_u24': (3, '({0} & 0xffffffu) * ({1} & 0xffffffu) + {2}'),
    'v_mad_i32_i24': (3, 'asuint(asint({0} << 8) >> 8) * asuint(asint({1} << 8) >> 8) + {2}'),
    'v_lshl_add_u32': (3, '({0} << ({1} & 31)) + {2}'), 'v_add_lshl_u32': (3, '({0} + {1}) << ({2} & 31)'), 'v_add3_u32': (3, '{0} + {1} + {2}'),
    'v_or3_b32': (3, '{0} | {1} | {2}'), 'v_xor3_b32': (3, '{0} ^ {1} ^ {2}'),'v_and_or_b32': (3, '({0} & {1}) | {2}'), 'v_lshl_or_b32': (3, '({0} << ({1} & 31)) | {2}'),
    'v_xad_u32': (3, '({0} ^ {1}) + {2}'), 'v_bfi_b32': (3, '({0} & {1}) | (~{0} & {2})'),
    'v_mul_u32_u24': (2, '({0} & 0xffffffu) * ({1} & 0xffffffu)'),
    # v_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0] (RDNA 2 ISA VOP3; same formula as s_bfm_b32).
    'v_bfm_b32': (2, '(((1u << ({0} & 31)) - 1) << ({1} & 31))'),
    'v_mul_i32_i24': (2, 'asuint(asint({0} << 8) >> 8) * asuint(asint({1} << 8) >> 8)'),
    # Per-lane bit operations (AMD RDNA 2 ISA reference). The bitscan results count from the most significant
    # bit for the "h" forms and read 0xffffffff when no bit qualifies; firstbitlow(0) is already 0xffffffff.
    # v_bfe_i32 extracts a field at S1[4:0] of width S2[4:0] (0 reads 0) and sign-extends it from its top bit,
    # taking the bits above bit 31 as copies of bit 31; v_bcnt_u32_b32 adds S1 to the population count of S0.
    'v_bfe_i32': (3, '(({2} & 31u) == 0u ? 0u : asuint(asint({0} << (32u - min(({1} & 31u) + ({2} & 31u), 32u))) '
                     '>> (32u - min(({1} & 31u) + ({2} & 31u), 32u) + ({1} & 31u))))'),
    'v_bcnt_u32_b32': (2, 'countbits({0}) + {1}'), 'v_ffbl_b32': (1, 'firstbitlow({0})'),
    'v_ffbh_u32': (1, '({0} == 0u ? 0xffffffffu : 31u - firstbithigh({0}))'),
    'v_ffbh_i32': (1, '({0} == 0u || {0} == 0xffffffffu ? 0xffffffffu : 31u - firstbithigh(asint({0}) < 0 ? ~{0} : {0}))'),
    'v_bfrev_b32': (1, 'reversebits({0})'),
    # The high word of the unsigned 64-bit product; shader model 5.1 has no 64-bit integers (see HELPERS).
    'v_mul_hi_u32': (2, 'x2w_mul_hi_u32({0}, {1})'),
    # D = S0[15:0] * S1[15:0] + S2 (op_sel, which picks other halves, arrives as a modifier and is refused).
    'v_mad_u32_u16': (3, '({0} & 0xffffu) * ({1} & 0xffffu) + {2}'),
    'v_sad_u8': (3, 'sad_u8({0}, {1}, {2})'),
    'v_sad_u16': (3, 'sad_u16({0}, {1}, {2})'),
    'v_sad_u32': (3, 'sad_u32({0}, {1}, {2})'),
    # Saturating 32-to-16-bit integer packs (AMD RDNA 2 ISA reference, VOP3A opcodes 0x36a and 0x36b; saturation
    # bounds per Mesa NIR pack_uvec2_to_uint / pack_ivec2_to_uint selected in ACO): src0 -> low 16 bits, src1 -> high 16 bits.
    'v_cvt_pk_u16_u32': (2, 'min({0}, 0xffffu) | (min({1}, 0xffffu) << 16u)'),
    'v_cvt_pk_i16_i32': (2, '(asuint(clamp(asint({0}), -32768, 32767)) & 0xffffu) | '
                            '(asuint(clamp(asint({1}), -32768, 32767)) << 16u)'),
    'v_cvt_pk_u16_u32': (2, 'min({0}, 0xffffu) | (min({1}, 0xffffu) << 16u)'),
    'v_cvt_pk_i16_i32': (2, '(asuint(clamp(asint({0}), -32768, 32767)) & 0xffffu) | '
                            '(asuint(clamp(asint({1}), -32768, 32767)) << 16u)'),
}


def _mul24_hi_i32():
    """High 32 bits of the 48-bit product of two signed 24-bit fields, in 32-bit integer arithmetic.

    With a = ah * 2^12 + al (al in 0..4095, ah the sign-extended top 12 bits) the product is
    hh * 2^24 + mid * 2^12 + lo, so p >> 12 = hh * 2^12 + t with t = mid + (lo >> 12), and
    p >> 32 = (hh + (t >> 12)) >> 8 exactly (floor division nests), which stays inside 32 bits.
    """
    def top(n):
        return '(asint({%d} << 8) >> 20)' % n

    def low(n):
        return 'asint({%d} & 0xfffu)' % n
    hh = '(%s * %s)' % (top(0), top(1))
    mid = '(%s * %s + %s * %s)' % (top(0), low(1), low(0), top(1))
    t = '(%s + ((%s * %s) >> 12))' % (mid, low(0), low(1))
    return 'asuint(((%s + (%s >> 12)) >> 8))' % (hh, t)


VEC_U.update({
    # AMD RDNA 2 ISA reference: the 24-bit multiplies read the low 24 bits of each source, the signed forms as a
    # two's complement field. The mul and mad forms keep the low 32 bits of the product; the hi forms keep bits 63..32
    # of the 48-bit product. The 24-bit field is sign-extended with a left shift of 8 and an arithmetic right shift.
    # v_mul_hi_u32_u24 is deliberately not lowered. Its result is at most 0xffff, so as a float output word it is
    # always a binary32 denormal, and FXC 10.0.26100 at ps_5_1 (the startup path) folds a float output whose
    # provable range lies below 0x00800000 to a constant zero (any form tried, even a plain shift without a
    # product). The same words through a uint4 output survive, and DXC keeps them. v_mul_hi_i32_i24 can reach
    # the whole 32-bit range, so FXC keeps it. A refusal is visible; zeros are not.
    'v_mul_hi_i32_i24': (2, _mul24_hi_i32()),
    'v_xnor_b32': (2, '~({0} ^ {1})'),
    # v_alignbit_b32 takes bits [S2[4:0] + 31 : S2[4:0]] of the 64-bit value {S0, S1}; a zero shift returns S1 and
    # the guard avoids a shift by 32. v_alignbyte_b32 does the same with S2[1:0] * 8 bits.
    'v_alignbit_b32': (3, '(({2} & 31u) == 0u ? {1} : (({1} >> ({2} & 31u)) | ({0} << (32u - ({2} & 31u)))))'),
    'v_alignbyte_b32': (3, '(({2} & 3u) == 0u ? {1} : (({1} >> (({2} & 3u) * 8u)) | ({0} << (32u - ({2} & 3u) * 8u))))'),
})

VOP3_ONLY_VEC_U = frozenset(('v_cvt_pk_u16_u32', 'v_cvt_pk_i16_i32'))
# HLSL helpers, each emitted only by pipelines whose code calls it. The integer helpers are single expressions so
# tests/startup_gfx2hlsl_coverage_check.py can evaluate their text against plain integer references.
# x2w_mul_hi_u32: bits 63:32 of the unsigned product, from four 16-bit partial products; the carry into bit 32
# comes from the high half of the low product plus the low halves of the two cross products (each sum < 2^18).
MUL_HI_U32 = ('(a >> 16) * (b >> 16) + (((a & 0xffffu) * (b >> 16)) >> 16) + (((a >> 16) * (b & 0xffffu)) >> 16) + '
              '(((((a & 0xffffu) * (b & 0xffffu)) >> 16) + (((a & 0xffffu) * (b >> 16)) & 0xffffu) + '
              '(((a >> 16) * (b & 0xffffu)) & 0xffffu)) >> 16)')
# v_cmp_class: the bit index of the IEEE class of x in the S1 mask (LLVM SIDefines.h ClassFlags): 0 signaling NaN,
# 1 quiet NaN, 2 -inf, 3 -normal, 4 -subnormal, 5 -0, 6 +0, 7 +subnormal, 8 +normal, 9 +inf. Classified on the bits,
# so no float compare or denormal flush can change the answer.
CLASS_F32 = ('(((x >> 23) & 255u) == 255u ? ((x & 0x7fffffu) != 0u ? ((x >> 22) & 1u) : ((x >> 31) != 0u ? 2u : 9u)) : '
             '((x >> 23) & 255u) != 0u ? ((x >> 31) != 0u ? 3u : 8u) : '
             '(x & 0x7fffffu) != 0u ? ((x >> 31) != 0u ? 4u : 7u) : ((x >> 31) != 0u ? 5u : 6u))')
CLASS_F16 = ('(((x >> 10) & 31u) == 31u ? ((x & 0x3ffu) != 0u ? ((x >> 9) & 1u) : (((x >> 15) & 1u) != 0u ? 2u : 9u)) : '
             '((x >> 10) & 31u) != 0u ? (((x >> 15) & 1u) != 0u ? 3u : 8u) : '
             '(x & 0x3ffu) != 0u ? (((x >> 15) & 1u) != 0u ? 4u : 7u) : (((x >> 15) & 1u) != 0u ? 5u : 6u))')
LSHR64_LO = ('((n & 63u) == 0u ? l : (n & 63u) < 32u ? (l >> (n & 63u)) | (h << (32u - (n & 63u))) : '
             'h >> ((n & 63u) - 32u))')
LSHR64_HI = '((n & 63u) < 32u ? h >> (n & 63u) : 0u)'
LSHL64_LO = '((n & 63u) < 32u ? l << (n & 63u) : 0u)'
LSHL64_HI = ('((n & 63u) == 0u ? h : (n & 63u) < 32u ? (h << (n & 63u)) | (l >> (32u - (n & 63u))) : '
             'l << ((n & 63u) - 32u))')
# SCA operations folded when every operand is a known constant (Stage.scalar), as their SCA texts compute them
# (AMD RDNA ISA: S_BREV_B32 D.u[31:0] = S0.u[0:31], no SCC write). s_bfm_b32 has its own folding handler.
SCALAR_CONST_FOLD = {
    's_brev_b32': lambda a: int('{:032b}'.format(a)[::-1], 2),
}

HELPERS = {
    # The product of binary16 inputs is exact in f32. Use FastTwoSum for the addition
    # residual, then round to odd at f32 precision. The final conversion to f16 can
    # decide ties and directed rounding without double rounding.
    'x2w_half_madd_odd': '''float x2w_half_madd_odd(float a, float b, float c) {
        precise float p = a * b;
        precise float sum = p + c;
        if (!isfinite(sum)) return sum;
        precise float hi = abs(p) >= abs(c) ? p : c;
        precise float lo = abs(p) >= abs(c) ? c : p;
        precise float tail = lo - (sum - hi);
        uint u = asuint(sum);
        if (tail == 0.0 || (u & 1u)) return sum;
        return asfloat((tail > 0.0) == (sum > 0.0) ? u + 1u : u - 1u);
    }''',
    'x2w_mul_hi_u32': 'uint x2w_mul_hi_u32(uint a, uint b) { return %s; }' % MUL_HI_U32,
    'x2w_class_f32': 'uint x2w_class_f32(uint x, uint mask) { return (mask >> %s) & 1u; }' % CLASS_F32,
    # binary16 rounding toward zero (MODE[3:2] = 3, see Stage.op_s_round_mode): f32tof16, then one step toward zero
    # when that rounded away from zero. Exact if f32tof16 rounds to nearest or toward zero: overflow to inf steps
    # back to 0x7bff; a NaN or an exactly representable value stays (the compare is false).
    'x2w_f32tof16_rtz': ('uint x2w_f32tof16_rtz(float x) { uint h = f32tof16(x); '
                         'return abs(f16tof32(h)) > abs(x) ? h - 1u : h; }'),
    # binary16 rounding to nearest even (MODE[3:2] = 0): t = the toward-zero result (as x2w_f32tof16_rtz, inlined so
    # the helper stands alone), u = the next value away from zero (0x7c00 counts as 65536, the first value past the
    # range, so 65520 and above round to inf); the nearer wins, a tie goes to the even one. The two distances are
    # exact in f32 (Sterbenz: |x| lies within [lo, 2 lo] of the f16 neighbours, or lo is 0). inf and NaN stay.
    'x2w_f32tof16_rne': ('uint x2w_f32tof16_rne(float x) { uint t = f32tof16(x); float a = abs(x); '
                         'if (abs(f16tof32(t)) > a) t -= 1u; if ((t & 0x7c00u) == 0x7c00u) return t; '
                         'float lo = abs(f16tof32(t)); if (a == lo) return t; uint u = t + 1u; '
                         'float hi = (u & 0x7fffu) == 0x7c00u ? 65536.0 : abs(f16tof32(u)); '
                         'float dl = a - lo, dh = hi - a; return (dh < dl || (dh == dl && (t & 1u) != 0u)) ? u : t; }'),
    'x2w_class_f16':'uint x2w_class_f16(uint x, uint mask) { return (mask >> %s) & 1u; }' % CLASS_F16,
    # The words of a 64-bit shift of (l, h) by n[5:0]. HLSL masks shift counts to five bits, so a count of 0
    # (where 32 - n would wrap to a shift by 0) and counts of 32..63 are separate cases.
    'x2w_lshr64_lo': 'uint x2w_lshr64_lo(uint l, uint h, uint n) { return %s; }' % LSHR64_LO,
    'x2w_lshr64_hi': 'uint x2w_lshr64_hi(uint l, uint h, uint n) { return %s; }' % LSHR64_HI,
    'x2w_lshl64_lo': 'uint x2w_lshl64_lo(uint l, uint h, uint n) { return %s; }' % LSHL64_LO,
    'x2w_lshl64_hi': 'uint x2w_lshl64_hi(uint l, uint h, uint n) { return %s; }' % LSHL64_HI,
    # v_cube{id,sc,tc,ma}_f32 as LLVM's constant folder defines them (ConstantFoldAMDGCNCubeIntrinsic): the
    # major axis is z when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|, else x; a negative major axis is one
    # that is below zero (not -0, not NaN). ma is twice the major-axis component.
    'x2w_cube': ('float4 x2w_cube(float a, float b, float c) { '
                 'if (abs(c) >= abs(a) && abs(c) >= abs(b)) return float4(c < 0.0 ? 5.0 : 4.0, c < 0.0 ? -a : a, -b, c + c); '
                 'if (abs(b) >= abs(a)) return float4(b < 0.0 ? 3.0 : 2.0, a, b < 0.0 ? -c : c, b + b); '
                 'return float4(a < 0.0 ? 1.0 : 0.0, a < 0.0 ? c : -c, -b, a + a); }'),
    # One 16-bit half of v_cvt_pknorm_{u16,i16}_f32, per Mesa ACO selection (Mesa main 13ebbc00,
    # src/amd/compiler/instruction_selection/aco_select_nir_alu.cpp selects nir pack_[us]norm_2x16 to these
    # opcodes) with the NIR definition (src/compiler/nir/nir_constant_expressions.py pack_[us]norm_1x16):
    # (uint16_t)(int)roundeven(CLAMP(x, lo, 1) * 65535 or 32767). Mesa's CLAMP (src/util/macros.h) is
    # "X > MIN ? (X > MAX ? MAX : X) : MIN", so NaN gives MIN: unorm 0, snorm -32767 (0x8001). The NaN case is
    # decided on the bits so no compiler min/max rewrite can change it; HLSL round() rounds half to even.
    'x2w_pknorm_u16': ('uint x2w_pknorm_u16(float x) { if ((asuint(x) & 0x7fffffffu) > 0x7f800000u) return 0u; '
                       'return uint(round(clamp(x, 0.0, 1.0) * 65535.0)); }'),
    'x2w_pknorm_i16': ('uint x2w_pknorm_i16(float x) { if ((asuint(x) & 0x7fffffffu) > 0x7f800000u) return 0x8001u; '
                       'return uint(int(round(clamp(x, -1.0, 1.0) * 32767.0))) & 0xffffu; }'),
}

# gfx10 buffer FORMAT codes (V# word 3 [18:12]) the draw runtime can synthesize for a typed buffer view
# (gfx10_view_words.h gfx10_buffer_words; raw and structured views get 22, 32_FLOAT) and their channels, x first
# from bit 0: (bit widths, numeric kind). Mixed widths list x, y, z, w as the bits rise: 10_11_11_FLOAT is x 11, y
# 11, z 10; 2_10_10_10 is x 10, y 10, z 10, w 2. Codes from Mesa's gfx10 format table, as in gfx10_view_words.h.
BUF_FORMATS = {
    1: ((8,), 'unorm'), 2: ((8,), 'snorm'), 5: ((8,), 'uint'), 6: ((8,), 'sint'),
    7: ((16,), 'unorm'), 8: ((16,), 'snorm'), 11: ((16,), 'uint'), 12: ((16,), 'sint'), 13: ((16,), 'float'),
    14: ((8, 8), 'unorm'), 15: ((8, 8), 'snorm'), 18: ((8, 8), 'uint'), 19: ((8, 8), 'sint'),
    20: ((32,), 'uint'), 21: ((32,), 'sint'), 22: ((32,), 'float'),
    23: ((16, 16), 'unorm'), 24: ((16, 16), 'snorm'), 27: ((16, 16), 'uint'), 28: ((16, 16), 'sint'),
    29: ((16, 16), 'float'),
    36: ((11, 11, 10), 'float'),
    50: ((10, 10, 10, 2), 'unorm'), 54: ((10, 10, 10, 2), 'uint'),
    56: ((8, 8, 8, 8), 'unorm'), 57: ((8, 8, 8, 8), 'snorm'), 60: ((8, 8, 8, 8), 'uint'), 61: ((8, 8, 8, 8), 'sint'),
    62: ((32, 32), 'uint'), 63: ((32, 32), 'sint'), 64: ((32, 32), 'float'),
    65: ((16,) * 4, 'unorm'), 66: ((16,) * 4, 'snorm'), 69: ((16,) * 4, 'uint'), 70: ((16,) * 4, 'sint'),
    71: ((16,) * 4, 'float'),
    72: ((32,) * 3, 'uint'), 73: ((32,) * 3, 'sint'), 74: ((32,) * 3, 'float'),
    75: ((32,) * 4, 'uint'), 76: ((32,) * 4, 'sint'), 77: ((32,) * 4, 'float'),
}


def buf_format_channel(widths, kind, k):
    """HLSL for channel k (a Python int) of an element at byte address a of ByteAddressBuffer b, as a uint."""
    pos, n = sum(widths[:k]), widths[k]
    field = 'b.Load(a + %du)' % (pos // 8) if n == 32 else \
        '((ldu(b, a + %du) >> %du) & 0x%xu)' % (4 * (pos // 32), pos % 32, (1 << n) - 1)
    if kind == 'uint' or (kind == 'float' and n == 32):
        return field
    if kind == 'sint':
        return field if n == 32 else 'uint(int(%s << %du) >> %du)' % (field, 32 - n, 32 - n)
    if kind == 'unorm':
        return 'U(float(%s) / %d.0)' % (field, (1 << n) - 1)
    if kind == 'snorm':
        return 'U(max(float(int(%s << %du) >> %du) / %d.0, -1.0))' % (field, 32 - n, 32 - n, (1 << (n - 1)) - 1)
    # binary16, and the unsigned 11-bit (5-bit exponent, 6-bit mantissa) and 10-bit (5, 5) floats, which are the
    # upper bits of a binary16 without its sign
    return 'U(f16tof32(%s))' % (field if n == 16 else '(%s << %du)' % (field, 15 - n))


def buf_format_helper():
    """x2w_fmt_load(b, a, w3, c): channel c of a buffer_load_format_* result, from V# word 3 w3 (udsrc bufword3).

    DST_SEL_X/Y/Z/W ([2:0] [5:3] [8:6] [11:9]): 0 -> 0, 1 -> 1 (1.0 for a float format, 1 for an integer one),
    4..7 -> element channel x..w. A channel the format lacks reads 0 (w reads 1, as the fill 0001); a FORMAT code
    not in BUF_FORMATS (0 = invalid, which the runtime gives for raw-copy, vertex and root sources) reads 0 in
    every channel, as do selectors 2 and 3. These fills are assumptions, not measured."""
    lines = ['uint x2w_fmt_chan(ByteAddressBuffer b, uint a, uint f, uint k, uint one) {']
    for code, (widths, kind) in sorted(BUF_FORMATS.items()):
        chans = ' '.join('if (k == %du) return %s;' % (k, buf_format_channel(widths, kind, k)) for k in range(len(widths)))
        lines.append('    if (f == %du) { %s return k == 3u ? one : 0u; }' % (code, chans))
    lines.append('    return 0u; }')
    known = ' || '.join('f == %du' % c for c in sorted(BUF_FORMATS))
    ints = ' || '.join('f == %du' % c for c, (_, kind) in sorted(BUF_FORMATS.items()) if kind in ('uint', 'sint'))
    lines.append('uint x2w_fmt_load(ByteAddressBuffer b, uint a, uint w3, uint c) {')
    lines.append('    uint f = (w3 >> 12) & 127u, s = (w3 >> (3u * c)) & 7u;')
    lines.append('    if (!(%s) || s == 0u || s == 2u || s == 3u) return 0u;' % known)
    lines.append('    uint one = (%s) ? 1u : 0x3f800000u;' % ints)
    lines.append('    return s == 1u ? one : x2w_fmt_chan(b, a, f, s - 4u, one); }')
    return '\n'.join(lines)


FMT_KIND = {'uint': 0, 'sint': 1, 'unorm': 2, 'snorm': 3, 'float': 4}


def buf_format_word(code):
    """x2w_fmt_tab entry: channel widths x, y, z, w in 6-bit fields [5:0] [11:6] [17:12] [23:18] (0 = absent), kind
    in [26:24] (FMT_KIND); 0 for a code not in BUF_FORMATS."""
    if code not in BUF_FORMATS:
        return 0
    widths, kind = BUF_FORMATS[code]
    word = FMT_KIND[kind] << 24
    for k, n in enumerate(widths):
        word |= n << (6 * k)
    return word


def buf_format_helper_table():
    """The same x2w_fmt_load as buf_format_helper, written once for every format: the format code indexes a constant
    table of channel widths and kind, and one channel decoder serves all formats. Values are the same as the per-format
    chain's (same field, sign extension, UNORM/SNORM divisor, binary16 widening), except that the UNORM/SNORM divisor
    is a run-time value here rather than a literal, so the compiler cannot replace the division by a multiply."""
    size = max(BUF_FORMATS) + 1
    table = ', '.join('0x%08xu' % buf_format_word(c) for c in range(size))
    return '\n'.join([
        'static const uint x2w_fmt_tab[%d] = { %s };' % (size, table),
        'uint x2w_fmt_chan(ByteAddressBuffer b, uint a, uint d, uint k, uint one) {',
        '    uint n = (d >> (6u * k)) & 63u;',
        '    if (n == 0u) return k == 3u ? one : 0u;',
        '    uint pos = (k > 0u ? (d & 63u) : 0u) + (k > 1u ? ((d >> 6) & 63u) : 0u) + (k > 2u ? ((d >> 12) & 63u) : 0u);',
        '    uint kind = (d >> 24) & 7u, sh = 32u - n;',
        '    uint raw = n == 32u ? b.Load(a + pos / 8u) : ((ldu(b, a + 4u * (pos / 32u)) >> (pos % 32u)) & ((1u << n) - 1u));',
        '    int sv = int(raw << sh) >> sh;',
        '    if (kind == 0u || (kind == 4u && n == 32u)) return raw;',
        '    if (kind == 1u) return uint(sv);',
        '    if (kind == 2u) return U(float(raw) / float((1u << n) - 1u));',
        '    if (kind == 3u) return U(max(float(sv) / float((1u << (n - 1u)) - 1u), -1.0));',
        '    return U(f16tof32(n == 16u ? raw : raw << (15u - n))); }',
        'uint x2w_fmt_load(ByteAddressBuffer b, uint a, uint w3, uint c) {',
        '    uint f = (w3 >> 12) & 127u, s = (w3 >> (3u * c)) & 7u;',
        '    uint d = f < %du ? x2w_fmt_tab[f] : 0u;' % size,
        '    if (d == 0u || s == 0u || s == 2u || s == 3u) return 0u;',
        '    uint one = ((d >> 24) & 7u) <= 1u ? 1u : 0x3f800000u;',
        '    return s == 1u ? one : x2w_fmt_chan(b, a, d, s - 4u, one); }'])


# The table form: fxc 10.0.26100 took 432.6 s for a vertex stage with two calls of the per-format chain and 0.6 s with
# this one (translator_work NOTES round 5). buf_format_helper stays as the reference the check compares against.
HELPERS['x2w_fmt_load'] = buf_format_helper_table()
HELPERS['x2w_fmt_store'] = '\n'.join([
    'void x2w_fmt_store(RWByteAddressBuffer b, uint a, uint w3, uint4 v, uint m) {',
    '    uint f = (w3 >> 12) & 127u;',
    '    uint n = (f >= 20u && f <= 22u) ? 1u : (f >= 62u && f <= 64u) ? 2u : (f >= 72u && f <= 74u) ? 3u :',
    '             (f >= 75u && f <= 77u) ? 4u : 0u;',
    '    n = min(n, m);',
    '    if (n > 0u) b.Store(a, v.x);',
    '    if (n > 1u) b.Store(a + 4u, v.y);',
    '    if (n > 2u) b.Store(a + 8u, v.z);',
    '    if (n > 3u) b.Store(a + 12u, v.w); }'])
HELPERS['x2w_fma_exact'] = ('float x2w_fma_exact(float a, float b, float c) '
                            '{ return (float)fma((double)a, (double)b, (double)c); }')
HELPERS['x2w_mad_precise'] = ('float x2w_mad_precise(float a, float b, float c) '
                              '{ precise float p = a * b; precise float r = p + c; return r; }')
# VEC_U conversions whose result is a binary32 value, so the VOP3 output modifiers apply to it.
FLOAT_RESULT_U = frozenset(('v_cvt_f32_u32', 'v_cvt_f32_i32', 'v_cvt_f32_ubyte0', 'v_cvt_f32_ubyte1',
                            'v_cvt_f32_ubyte2', 'v_cvt_f32_ubyte3', 'v_cvt_f32_f16'))
# scalar ops: (expression over uint sources, SCC rule with $ for the result or None)
# Scalar ops that write SCC on the hardware (carry, overflow, or which source was selected) but have
# no SCC template here: the emitted scc keeps its previous value, so the uniformity proof stops
# trusting it.
SCC_UNMODELLED = frozenset(('s_add_i32', 's_add_u32', 's_sub_i32', 's_sub_u32',
                            's_min_i32', 's_min_u32', 's_max_i32', 's_max_u32'))

PACK_B16 = frozenset(('s_pack_ll_b32_b16', 's_pack_lh_b32_b16', 's_pack_hh_b32_b16'))
SCA = {
    's_mov_b32': ('{0}', None), 's_mov_b64': ('{0}', None), 's_movk_i32': ('{0}', None),
    's_and_b32': ('{0} & {1}', '$ != 0'), 's_or_b32': ('{0} | {1}', '$ != 0'), 's_xor_b32': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b32': ('{0} & ~{1}', '$ != 0'), 's_orn2_b32': ('{0} | ~{1}', '$ != 0'), 's_not_b32': ('~{0}', '$ != 0'),
    's_nand_b32': ('~({0} & {1})', '$ != 0'), 's_nor_b32': ('~({0} | {1})', '$ != 0'), 's_xnor_b32': ('~({0} ^ {1})', '$ != 0'),
    's_and_b64': ('{0} & {1}', '$ != 0'), 's_or_b64': ('{0} | {1}', '$ != 0'), 's_xor_b64': ('{0} ^ {1}', '$ != 0'),
    's_andn2_b64': ('{0} & ~{1}', '$ != 0'), 's_orn2_b64': ('{0} | ~{1}', '$ != 0'), 's_not_b64': ('~{0}', '$ != 0'),
    's_nand_b64': ('~({0} & {1})', '$ != 0'), 's_nor_b64': ('~({0} | {1})', '$ != 0'), 's_xnor_b64': ('~({0} ^ {1})', '$ != 0'),
    's_lshl_b32': ('{0} << ({1} & 31)', '$ != 0'), 's_lshr_b32': ('{0} >> ({1} & 31)', '$ != 0'),
    's_ashr_i32': ('uint(int({0}) >> ({1} & 31))', '$ != 0'),
    's_add_i32': ('{0} + {1}', None), 's_add_u32': ('{0} + {1}', None), 's_sub_i32': ('{0} - {1}', None), 's_sub_u32': ('{0} - {1}', None),
    's_mul_i32': ('{0} * {1}', None), 's_min_u32': ('min({0}, {1})', None), 's_max_u32': ('max({0}, {1})', None),
    's_min_i32': ('uint(min(int({0}), int({1})))', None), 's_max_i32': ('uint(max(int({0}), int({1})))', None),
    's_bfe_u32': ('({0} >> ({1} & 31)) & ((1u << (({1} >> 16) & 127)) - 1)', '$ != 0'),
    's_brev_b32': ('reversebits({0})', None),
    # s_bcnt0/1_i32_b32: the count of zero / one bits of S0; SCC = (D != 0) (AMD RDNA 2 ISA SOP1, LLVM
    # SOPInstructions.td: Defs = [SCC]).
    's_bcnt1_i32_b32': ('countbits({0})', '$ != 0'), 's_bcnt0_i32_b32': ('32u - countbits({0})', '$ != 0'),
    # s_ff1/s_ff0_i32_b32: the index of the lowest one / zero bit of S0, -1 when there is none; SCC unchanged (AMD
    # RDNA 2 ISA SOP1; LLVM SOPInstructions.td has no SCC def; it selects AMDGPUffbl_b32 = cttz_zero_undef with
    # the -1 result). HLSL firstbitlow returns 0xffffffff for 0 as well.
    's_ff1_i32_b32': ('firstbitlow({0})', None), 's_ff0_i32_b32': ('firstbitlow(~{0})', None),
    # s_sext_i32_iN: sign-extend the low N bits of S0; SCC unchanged.
    's_sext_i32_i8': ('uint(int({0} << 24) >> 24)', None),
    's_sext_i32_i16': ('uint(int({0} << 16) >> 16)', None),
    's_pack_lh_b32_b16': ('({0} & 0xffffu) | ({1} & 0xffff0000u)', None),
    # s_bfm_b32: a mask of S0[4:0] ones shifted left by S1[4:0]; SCC unchanged.
    's_bfm_b32': ('(((1u << ({0} & 31)) - 1) << ({1} & 31))', None),
    # s_absdiff_i32: D = |S0 - S1| with the 32-bit wrapping difference (|INT_MIN| stays INT_MIN); SCC = (D != 0)
    # (RDNA 2 ISA SOP2; LLVM SOPInstructions.td Defs = [SCC]).
    's_absdiff_i32': ('uint(abs(int({0} - {1})))', '$ != 0'),
    's_mul_hi_u32': ('x2w_mul_hi_u32({0}, {1})', None),
    # LLVM SOPInstructions.td: s_lshlN_add_u32 selects shlN_add, (S0 << N) + S1; it defines SCC, not modelled.
    's_lshl1_add_u32': ('({0} << 1) + {1}', None), 's_lshl2_add_u32': ('({0} << 2) + {1}', None),
    's_lshl3_add_u32': ('({0} << 3) + {1}', None), 's_lshl4_add_u32': ('({0} << 4) + {1}', None),
    # s_abs_i32 is what LLVM selects for smax(x, -x) (so abs(INT_MIN) stays INT_MIN); it defines SCC, not modelled.
    's_abs_i32': ('uint(max(int({0}), -int({0})))', None),
    's_pack_ll_b32_b16': ('({0} & 0xffffu) | ({1} << 16)', None),
    's_pack_hh_b32_b16': ('({0} >> 16) | ({1} & 0xffff0000u)', None),
    # LLVM SOPInstructions.td S_SEXT_I32_I8 / S_SEXT_I32_I16: sext_inreg of S0 to i8 / i16; no SCC def.
    's_sext_i32_i8': ('uint(int({0} << 24) >> 24)', None), 's_sext_i32_i16': ('uint(int({0} << 16) >> 16)', None),
}
# Scalar ops whose SCC result the translation does not compute (LLVM SOPInstructions.td lists SCC in their Defs).
# Stage.check_scc refuses a program only when that SCC can reach a reader.
SCC_UNMODELLED = frozenset(('s_add_u32', 's_add_i32', 's_sub_u32', 's_sub_i32', 's_min_u32', 's_max_u32', 's_min_i32',
                            's_max_i32', 's_abs_i32', 's_addk_i32', 's_lshl1_add_u32', 's_lshl2_add_u32',
                            's_lshl3_add_u32', 's_lshl4_add_u32', 's_lshl_b64', 's_lshr_b64', 's_ashr_i64'))
SCC_READERS = ('s_cbranch_scc0', 's_cbranch_scc1', 's_cselect_b32', 's_cselect_b64', 's_cmov_b32', 's_cmov_b64',
               's_cmovk_i32', 's_addc_u32', 's_subb_u32')


def reads_scc(name, asm):
    return re.sub(r'_e(32|64)$', '', name) in SCC_READERS or 'src_scc' in parse(asm)[1]

for _name, _op in (('eq', '=='), ('lg', '!='), ('gt', '>'), ('ge', '>='), ('lt', '<'), ('le', '<=')):
    for _typ, _cast in (('i32', 'int({0}) %s int({1})'), ('u32', '{0} %s {1}')):
        def _make(op=_op, cast=_cast, name='s_cmp_%s_%s' % (_name, _typ)):
            def handler(self, ops, mods):
                # A scalar compare reads exactly two sources; refusing any other count keeps a malformed
                # instruction a refusal rather than an index error in the batch translator.
                if len(ops) != 2:
                    raise Unsupported('%s with %d operands, expected 2' % (name, len(ops)))
                self.set_scc((cast % op).format(self.u(ops[0]), self.u(ops[1])))
            return handler
        setattr(Stage, 'op_s_cmp_%s_%s' % (_name, _typ), _make())

        def _make_k(op=_op, cast=_cast, signed=_typ == 'i32', name='s_cmpk_%s_%s' % (_name, _typ)):
            # s_cmpk_*_i32 compares with signext(SIMM16), s_cmpk_*_u32 with SIMM16 zero-extended (LLVM s16imm /
            # u16imm); the disassembler prints the field unsigned either way.
            def handler(self, ops, mods):
                # A wrong operand count is refused by count, like s_cmp_* (upstream handler arity check).
                if len(ops) != 2:
                    raise Unsupported('%s with %d operands, expected 2' % (name, len(ops)))
                if mods:
                    raise Unsupported('s_cmpk with %s %s' % (ops, mods))
                imm = sopk_simm(ops[1]) if signed else ops[1]
                self.emit('scc = %s;' % (cast % op).format(self.u(ops[0]), self.u(imm)))
            return handler
        setattr(Stage, 'op_s_cmpk_%s_%s' % (_name, _typ), _make_k())


def _cmp_u64(equal):
    # s_cmp_eq_u64 / s_cmp_lg_u64: SCC = (S0 == S1) / (S0 != S1) on 64-bit values (RDNA 2 ISA, SOPC). Lowered on
    # data pairs only (a 64-bit pointer or constant read from memory, an inline constant): a lane mask compared
    # with anything is a question about the whole wave, which this one-lane model holds as this lane's bit only,
    # so exec, vcc and any pair holding a lane mask (or a copy of exec) are refused.
    def handler(self, ops, mods):
        if len(ops) != 2 or mods:
            raise Unsupported('s_cmp_%s_u64 form %s %s' % ('eq' if equal else 'lg', ops, mods))
        for tok in ops:
            r = reg_range(tok)
            if r and r[0] == 's' and any(n in getattr(self, 'masks', ()) or (self.sym.get(n) or ('',))[0] == 'execcopy'
                                         for n in (r[1], r[2])):
                raise Unsupported('s_cmp_%s_u64 on a lane mask' % ('eq' if equal else 'lg'))
        (a0, a1), (b0, b1) = self.pair(ops[0]), self.pair(ops[1])
        if equal:
            self.emit('scc = %s == %s && %s == %s;' % (a0, b0, a1, b1))
        else:
            self.emit('scc = %s != %s || %s != %s;' % (a0, b0, a1, b1))
    return handler


Stage.op_s_cmp_eq_u64 = _cmp_u64(True)
Stage.op_s_cmp_lg_u64 = _cmp_u64(False)


def scalar_wide_fold(helper, x, amount, signed):
    """Translation-time value of one SOP2 shift or bitfield extract (the helper name picks the operation)."""
    m64 = 0xffffffffffffffff
    if helper == 'shl64':
        return (x << (amount & 63)) & m64
    if helper == 'shr64u':
        return x >> (amount & 63)
    if helper == 'shr64i':
        return ((x - (1 << 64) if x >> 63 else x) >> (amount & 63)) & m64
    bits = 32 if helper == 'bfe32' else 64
    off = amount & (bits - 1)
    wd = min((amount >> 16) & 127, bits - off)  # bits past the source MSB do not exist
    field = (x >> off) & ((1 << wd) - 1)
    if signed and 0 < wd < bits and field >> (wd - 1) & 1:
        field |= ((1 << bits) - 1) ^ ((1 << wd) - 1)
    return 0 if signed and wd == 0 else field


def parse_root(blob):
    if not blob or blob[:4] != b'DXBC':
        raise Unsupported('root signature blob missing')

    def read(fmt, data, offset):
        if offset < 0 or offset + struct.calcsize(fmt) > len(data):
            raise Unsupported('root signature blob truncated')
        return struct.unpack_from(fmt, data, offset)

    n, = read('<I', blob, 28)
    for o in read('<%dI' % n, blob, 32):
        if blob[o:o + 4] == b'RTS0':
            d = blob[o + 8:]
            ver, nparam, poff, nsamp, soff, flags = read('<6I', d, 0)
            params = []
            for i in range(nparam):
                ptype, vis, off = read('<3I', d, poff + 12 * i)
                p = dict(type=ptype, vis=vis)
                if ptype == 1:
                    p['reg'], p['space'], p['count'] = read('<3I', d, off)
                params.append(p)
            return dict(flags=flags, params=params)
    raise Unsupported('no RTS0 part')


# Run-time sampler heap indexing: an s_load from the sampler heap with a register offset becomes a
# read of a bounded table, SamplerState qh[SAMPLER_HEAP_SIZE] : register(s0, space1), the form a
# desktop SM 5.1 / 6.0 shader can index at run time. Opt-in with X2W_SAMPLER_HEAP=1 until the
# stand-in binds that table; otherwise such a load keeps the default-sampler shortcut.
#
# Contract at the manifest boundary ('smpheap <count> <space> fallback <entry>'):
# - entries 0 .. SAMPLER_HEAP_FALLBACK - 1 hold the title's sampler-heap entries, entry n at byte
#   offset 16 * n;
# - entry SAMPLER_HEAP_FALLBACK is reserved: the host fills it with a fallback sampler;
# - the shader computes the entry itself (x2w_sampler_entry): a byte offset that is not a multiple
#   of 16, or that lies at or past 16 * SAMPLER_HEAP_FALLBACK, reads the fallback entry. The host
#   cannot check the offset, which is computed in the shader, so the guard is in the shader.
SAMPLER_HEAP_SIZE = 2048  # D3D12's limit for a shader-visible sampler heap
SAMPLER_HEAP_SPACE = 1
SAMPLER_HEAP_FALLBACK = SAMPLER_HEAP_SIZE - 1


# Draw lowerings that need the draw runtime to parse a manifest line, named in X2W_DRAW_FEATURES (comma list):
# heap (`heap 1`), bufword3 (`udsrc <n> bufword3 <srv>` for buffer_load_format_*), uav (`uav <i> buf table ...`
# with `udsrc <n> uavbase <i>` for buffer stores), heapwords (`heapwords 1`: a resource-heap view's synthesized
# descriptor words from vdesc, for buffer_load_format_* through a heap descriptor; needs heap and bufword3).
# Off by default until a build with the runtime side is announced.
KNOWN_DRAW_FEATURES = frozenset(('heap', 'bufword3', 'uav', 'heapwords', 'uavfmt'))


def draw_features():
    named = frozenset(f for f in os.environ.get('X2W_DRAW_FEATURES', '').replace(' ', '').split(',') if f)
    unknown = named - KNOWN_DRAW_FEATURES
    if unknown:
        raise SystemExit('X2W_DRAW_FEATURES: unknown %s' % ', '.join(sorted(unknown)))
    return named


# Resource-heap views of a draw (manifest `heap 1`, layout agreed with draw-rt): the title heap memory heapmem
# (t0 space1, a view token per 32-byte slot: HEAP_TOKEN then the view id), vinfo (t1 space1, 16 bytes per view id,
# dword 0 the buffer stride), and one array per kind over the view-id region, index 8 + 8 * id + kind for ids below
# HEAP_IDS and kind alone (an empty view) otherwise. Bounded arrays, so FXC needs no unbounded-table flag.
HEAP_TOKEN = 0x56573258
HEAP_IDS = 90000
HEAP_KINDS = {'tex2d': ('heaptex2d', 0, 'Texture2D<float4>'), 'tex2da': ('heaptex2da', 1, 'Texture2DArray<float4>'),
              'tex3d': ('heaptex3d', 2, 'Texture3D<float4>'), 'tex1d': ('heaptex1d', 3, 'Texture1D<float4>'),
              'buf': ('heapbuf', 4, 'ByteAddressBuffer')}


# DXGI_FORMAT values for integer RTVs and the binary32 float targets.
UINT_RTVS = frozenset((3, 7, 12, 17, 25, 30, 36, 42, 50, 57, 62))
SINT_RTVS = frozenset((4, 8, 14, 18, 32, 38, 43, 52, 59, 64))
FLOAT32_RTVS = frozenset((2, 6, 16, 41))
# DXGI_FORMAT values independently verified exact for D16 image_load and image_load_mip, matching
# d16_srv_mode() in src/startup_stubs/d16_image_format.h (#2023): 32-bit float and 8-bit UNORM views (exact via
# pack16_rne) plus 16-bit UNORM views (exact via pack16_unorm16).
D16_FLOAT_EXACT_SRV_FORMATS = frozenset((2, 6, 16, 21, 28, 41, 49, 61, 65, 87, 88))
D16_UNORM16_SRV_FORMATS = frozenset((11, 35, 56))
D16_EXACT_SRV_FORMATS = D16_FLOAT_EXACT_SRV_FORMATS | D16_UNORM16_SRV_FORMATS
# Uncompressed exports keep their 32-bit shader values. Narrow integer RTVs clamp
# each channel to the destination range in the output merger.
VERIFIED_UINT_RTVS = frozenset((3, 12, 17, 25, 30, 36, 42, 50, 57, 62))
VERIFIED_SINT_RTVS = frozenset((4, 14, 18, 32, 38, 43, 52, 59, 64))


# SPI pixel-stage inputs in SPI_PS_INPUT_ENA/ADDR bit order and the VGPRs each takes (Mesa
# src/amd/common/ac_shader_util.c ac_get_fs_input_vgpr_cnt): the IJ pairs, the pull model's three, then
# one VGPR each for line stipple, POS_X/Y/Z/W_FLOAT, FRONT_FACE, ANCILLARY, SAMPLE_COVERAGE, POS_FIXED_PT.
PS_INPUTS = (('persp_sample', 2), ('persp_center', 2), ('persp_centroid', 2), ('persp_pull_model', 3),
             ('linear_sample', 2), ('linear_center', 2), ('linear_centroid', 2), ('line_stipple', 1),
             ('pos_x_float', 1), ('pos_y_float', 1), ('pos_z_float', 1), ('pos_w_float', 1),
             ('front_face', 1), ('ancillary', 1), ('sample_coverage', 1), ('pos_fixed_pt', 1))
POS_FIXED_PT = 15
# SPI inputs lowered besides POS_FIXED_PT (bit -> HLSL uint expression over PSIn i). Mesa
# ac_nir_lower_intrinsics_to_args.c reads frag_coord x/y/z from POS_X/Y/Z_FLOAT and frag_coord_w_rcp from
# POS_W_FLOAT; nir_builder.c nir_build_frag_coord makes gl_FragCoord.w = frcp(frag_coord_w_rcp), and D3D
# SV_Position.w is 1 / gl_FragCoord.w (vkd3d-shader spirv.c frag_coord_fixup), so POS_W_FLOAT is SV_Position.w.
PS_INPUT_HLSL = {8: 'U(i.pos.x)', 9: 'U(i.pos.y)', 10: 'U(i.pos.z)', 11: 'U(i.pos.w)',
                 # FRONT_FACE is the float +1.0 (front) or -1.0 (back): Mesa nir_intrinsics.py front_face_fsign
                 # "front_face ? 1.0 : -1.0", which ac_nir_lower_intrinsics_to_args.c loads as the front_face argument
                 # itself (and load_front_face as fgt(arg, 0)). SV_IsFrontFace follows the PSO's FrontCounterClockwise,
                 # which the runtime sets from the cull register (pipeline_state_decode.h); the manifest says
                 # `frontface 1` so these keys can be found if that convention changes (X2W_DRAW_STATE_CULL_FLIP).
                 12: 'U(i.ff ? 1.0 : -1.0)'}
FRONT_FACE = 12
# The SPI position inputs, which a per-sample stage would read at the sample (see PS_SAMPLE_RATE).
PS_POSITION_INPUTS = (8, 9, 10, 11)
# The *_sample barycentrics: the stage runs per sample and the position inputs are at the sample, which a plain
# SV_Position input is not.
PS_SAMPLE_RATE = (1 << 0) | (1 << 4)


def ps_input_layout(ena, addr):
    """VGPR -> SPI input bit for the given SPI_PS_INPUT_ENA/ADDR, or Unsupported.

    The VGPRs follow ADDR, the inputs the code was compiled for; ENA names the ones the hardware initializes
    and must lie inside ADDR (Mesa ac_print_spi_ps_input_vgpr_list walks the ADDR bits to number v0, v1, ... and
    marks an input "initialized" only when its ENA bit is also set). Both are 16-bit masks.
    """
    if not 0 <= ena <= 0xffff or not 0 <= addr <= 0xffff or ena & ~addr:
        raise Unsupported('unverified pixel input enable/address combination %#x/%#x' % (ena, addr))
    layout, vgpr = {}, 0
    for bit, (_, width) in enumerate(PS_INPUTS):
        if addr >> bit & 1:
            for _ in range(width):
                layout[vgpr] = bit
                vgpr += 1
    return layout


class Pipeline:
    vertex_kind = 'vs'
    uses_pack16_rne = False
    uses_gather_2d_lod = False
    uses_store_d16_x = False
    uses_sad = ()
    uses_store_format_xyzw = False
    srv_formats = ()
    f16_plain = ()

    def __init__(self, vs_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), ps_hw_inputs=None,
                 features=None, vs_out_cntl=None, rw_keys=(), vs_fetch=None, refusals=None, decoded=None,
                 ps_attr_map=None, srv_formats=(), pk_u8_s0_contract=False):
        # pk_u8_s0_contract: the caller guarantees that every v_cvt_pk_u8_f32 S0 register not proven in domain
        # holds +0.0 or an integer in 1..255 (fixtures only; ordinary translation never selects it).
        self.pk_u8_s0_contract = pk_u8_s0_contract
        self.root = parse_root(root_blob)
        self.rw_keys = frozenset(rw_keys)  # buffers the program stores to and reads: every access through the UAV
        # PA_CL_VS_OUT_CNTL of the vertex stage (request .vsout), or None: says what a pos1 export carries.
        self.vs_out_cntl = vs_out_cntl
        self.features = draw_features() if features is None else frozenset(features)
        self.heapvars = []    # heap descriptor k -> resource kind it is used as (None until used)
        self.fma_rule = None  # set by the first multiply-add lowered (manifest `fma exact|precise`)
        self.draw_heapwords = False  # x2w_hword read (X2W_DRAW_FEATURES heapwords, manifest `heapwords 1`)
        self.spi_inputs(ps_hw_inputs)
        self.ps_attr_map = ps_attr_map  # stream_interface(): pixel attribute -> (vertex parameter, component mask); None pairs by number
        self.vs_fetch = vs_fetch  # stream_fetch(): the loads the vertex stage's entry call stands for
        # (packed_position_vgpr is set by spi_inputs above.)
        # First VGPR of the PERSP_CENTER I/J entry pair (SPI_PS_INPUT_ADDR bit 1, after the PERSP_SAMPLE pair when that
        # one is enabled). Without hardware inputs the model assumes the center pair alone: v0, v1.
        self.persp_center_vgpr = 0
        self.rtv_formats = rtv_formats
        self.srv_formats = dict(srv_formats) if isinstance(srv_formats, dict) else dict(enumerate(srv_formats))
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.uses_pack16_rne = False
        self.uses_gather_2d_lod = False
        self.uses_store_d16_x = False
        self.uses_sad = set()
        self.uses_store_format_xyzw = False
        self.resources = {}   # key -> dict(index, kind)
        self.samplers = {}    # key -> index
        self.uds = []         # entry-constant sources
        self.notes = set()    # shortcuts taken (written to the manifest as comments)
        self.lod_unvalidated = set()  # image_sample forms read through a heap sampler (LOD fields unchecked, #2221)
        self.modified = set()
        self.helper_text = None  # both stage bodies, read once by common() to pick the HELPERS to emit
        self.rect = self.ud(('rect',))
        if refusals is None:
            self.vs = Stage(self, self.vertex_kind, vs_code)
            self.ps = Stage(self, 'ps', ps_code) if ps_code is not None else None
        else:
            decoded = {} if decoded is None else decoded
            self.vs = Stage(self, self.vertex_kind, vs_code, refusals=refusals, decoded=decoded.get('vs'))
            self.ps = Stage(self, 'ps', ps_code, refusals=refusals, decoded=decoded.get('ps')) if ps_code is not None else None
        if self.ps is None:
            self.params, self.flat, self.half, self.f16_plain = [], [], [], []
            self.ps_entry_reads = {}
            if 'pos' not in self.vs.exports:
                raise Unsupported('vertex stage exports no position')
            return
        self.spi_reads()
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
        self.half = sorted(a for a, how in self.ps.attrs.items() if 'half' in how)
        self.check_half_reads()
        for a, how in self.ps.attrs.items():
            if 'f16' not in how:
                continue
            compr, plain = a in self.vs.compr_params, a in self.vs.plain_params
            if how != {'f16'} or not (compr or plain):
                raise Unsupported('attribute %d is read as binary16 by the pixel stage but not only so, or not exported '
                                  'by the vertex stage with compr or as a plain 32-bit export' % a)
            if compr and plain:
                raise Unsupported('attribute %d is exported both with compr and plain by the vertex stage: the two '
                                  'binary16 layouts differ' % a)
            if plain and self.ps.f16_plain.get(a, set()) & self.vs.plain_params[a]:
                raise Unsupported('attribute %d is read as binary16 from channels the vertex stage exports as off' % a)
        self.f16_plain = sorted(a for a in self.ps.f16_plain if a in self.vs.plain_params and a not in self.vs.compr_params)
        missing = [a for a in self.ps.attrs if a not in self.params]
        if missing:
            raise Unsupported('pixel stage reads attributes %s the vertex stage does not export' % missing)
        if 'pos' not in self.vs.exports:
            raise Unsupported('vertex stage exports no position')

    def spi_inputs(self, ps_hw_inputs):
        """Before the pixel stage is built: the SPI input layout from its SPI_PS_INPUT_ENA/ADDR, or none.

        Shared by the draw and mesh pipelines. With (ENA, ADDR), entry VGPRs map to SPI inputs. Only POS_FIXED_PT
        is lowered: the integer pixel X in [15:0] and Y in [31:16] (Mesa nir_intrinsics.py pixel_coord, "16-bit
        integer vec2 of the pixel X/Y in the framebuffer", which ac_nir_lower_intrinsics_to_args.c builds as
        unpack_32_2x16 of the pos_fixed_pt argument). D3D SV_Position.xy is the pixel centre, so its integer part
        is that pixel."""
        self.packed_position_vgpr = None
        self.ps_input_layout = None
        # The pixel stage's launch FLOAT_MODE (SPI_SHADER_PGM_RSRC1_PS bits 19:12), the third .psin field, or None
        # when the request does not carry it (then f16 rounding before s_round_mode is refused; Stage.launch_mode).
        # (None, None, mode) passes the mode without an SPI input layout.
        self.ps_float_mode = ps_hw_inputs[2] if ps_hw_inputs is not None and len(ps_hw_inputs) > 2 else None
        if ps_hw_inputs is not None and ps_hw_inputs[0] is None:
            ps_hw_inputs = None
        if ps_hw_inputs is not None:
            ena, addr = ps_hw_inputs[:2]
            self.ps_input_layout = layout = ps_input_layout(ena, addr)
            self.ps_input_ena = ena
            if ena >> POS_FIXED_PT & 1:
                self.packed_position_vgpr = [v for v, bit in layout.items() if bit == POS_FIXED_PT][0]

    def spi_reads(self):
        """After the pixel stage is built: check its entry-VGPR reads against the SPI input layout.

        A pixel stage starts with its SPI inputs in the VGPRs (barycentrics, position, face, ancillary data). The
        interpolation instructions name them without reading them (they take the attribute); any other read before
        a write sees 0 in the HLSL instead of the hardware input. Without the layout this is reported, not refused
        (existing fixtures read entry VGPRs as zero): ps_entry_reads, VGPR -> offset of the first read; the
        translate_to callers refuse a pipeline that has any. With the layout, a read of an input that is not
        lowered is kept in ps_input_refusal and raised by hlsl_ps(), so a pipeline can still be built to ask for its
        layout (packed_position_vgpr), but no pixel HLSL is ever written for it."""
        self.ps_entry_reads = dict(self.ps.entry_reads)
        self.ps_input_refusal = None
        self.ps_input_vgprs = {}  # entry VGPR -> PS_INPUT_HLSL bit, initialized by Stage.declarations()
        if self.ps_input_layout is None:
            return
        for reg, off in sorted(self.ps_entry_reads.items(), key=lambda item: item[1]):
            bit = self.ps_input_layout.get(reg)
            if bit is None:
                self.ps_input_refusal = ('pixel stage reads entry VGPR v%d at %#x, past the SPI inputs (ADDR allocates %d)'
                                         % (reg, off, len(self.ps_input_layout)))
            elif not self.ps_input_ena >> bit & 1:
                self.ps_input_refusal = ('pixel stage reads SPI input %s (v%d) at %#x, allocated but not enabled'
                                         % (PS_INPUTS[bit][0], reg, off))
            elif bit in PS_POSITION_INPUTS and self.ps_input_ena & PS_SAMPLE_RATE:
                self.ps_input_refusal = ('pixel stage reads SPI input %s (v%d) at %#x with per-sample barycentrics '
                                         'enabled (%#x)' % (PS_INPUTS[bit][0], reg, off, self.ps_input_ena))
            elif bit in PS_INPUT_HLSL:
                self.ps_input_vgprs[reg] = bit
            elif bit != POS_FIXED_PT:
                self.ps_input_refusal = ('pixel stage reads SPI input %s (v%d) at %#x: not lowered'
                                         % (PS_INPUTS[bit][0], reg, off))
            if self.ps_input_refusal:
                break
        self.ps_entry_reads = {}  # every entry read is a lowered input (or refused above)

    def check_half_reads(self):
        """Refuse a binary16 read (v_interp_p2_f16) of an attribute the vertex/mesh stage exports with compr: its
        loparam/hiparam copies are not written (Stage.export)."""
        compr = sorted(set(self.half) & set(getattr(self.vs, 'compr_params', ())))
        if compr:
            raise Unsupported('binary16 interpolation of attribute %s exported with compr' % compr)

    def mrt_kind(self, slot):
        if slot >= len(self.rtv_formats):
            return 'float'
        fmt = self.rtv_formats[slot]
        # R32_UINT / R32_SINT (42 / 43) are verified by a draw (tests/startup_mrt_check.py). The other formats with
        # 32 bits per channel (R32G32B32A32 3 / 4, R32G32B32 7 / 8, R32G32 17 / 18) take the same uncompressed
        # export words unconverted, only with more channels (D3D drops the components a format lacks, as the
        # colour buffer does): ASSUMED equal, not drawn. Narrower integer targets convert in the colour buffer
        # (clamp or wrap, not checked) and stay refused.
        if fmt in UINT_RTVS:
            if fmt not in (3, 7, 17, 42):
                raise Unsupported('unverified UINT RTV format %d' % fmt)
            return 'uint'
        if fmt in SINT_RTVS:
            if fmt not in (4, 8, 18, 43):
                raise Unsupported('unverified SINT RTV format %d' % fmt)
            return 'int'
        return 'bits' if fmt in FLOAT32_RTVS else 'float'

    def ud(self, source):
        if source not in self.uds:
            self.uds.append(source)
        return self.uds.index(source)

    def resource(self, key, kind):
        if key[0][0] == 'heapdyn':
            k = key[0][1]
            if key[1] not in (0, None):
                raise Unsupported('heap descriptor at word offset %s' % key[1])
            if self.heapvars[k] not in (None, kind):
                raise Unsupported('heap descriptor used as %s and %s' % (self.heapvars[k], kind))
            if kind not in HEAP_KINDS:
                raise Unsupported('heap %s' % kind)
            self.heapvars[k] = kind
            return '%s[NonUniformResourceIndex(x2w_hix(h%d, %du))]' % (HEAP_KINDS[kind][0], k, HEAP_KINDS[kind][1])
        if key in getattr(self, 'uavs', {}) or key in getattr(self, 'rw_keys', ()):
            raise Unsupported('descriptor %s read as an SRV and stored to as a UAV' % (key,))
        if key in Pipeline.stored_images(self):
            raise Unsupported('resource %s is both read and stored to' % (key,))
        r = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('resource %s used as %s and %s' % (key, r['kind'], kind))
        return 'r%d' % r['index']

    def uav(self, key):
        """u<i> for a raw buffer UAV (manifest `uav <i> buf ...`); indices count from 0 apart from the SRVs."""
        if key in self.resources:
            # translate_to builds the pipeline again with the key in rw_keys, so its vector loads go through the UAV
            raise StoredSrv(key, 'descriptor %s read as an SRV and stored to as a UAV' % (key,))
        if getattr(self, 'image_uavs', None):
            raise Unsupported('buffer UAV %s in a program with stored images (u registers are shared)' % (key,))
        uavs = self.__dict__.setdefault('uavs', {})
        if key not in uavs:
            if len(uavs) == 8:
                raise Unsupported('more than 8 UAVs')
            uavs[key] = len(uavs)
        return 'u%d' % uavs[key]

    def stored_images(self):
        return getattr(self, 'image_uavs', None) or {}

    def image_uav(self, key, kind):
        if key in self.resources:
            raise Unsupported('resource %s is both read and stored to' % (key,))
        if getattr(self, 'uavs', None):
            raise Unsupported('stored image %s in a program with buffer UAVs (u registers are shared)' % (key,))
        if getattr(self, 'image_uavs', None) is None:
            self.image_uavs = {}
        r = self.image_uavs.setdefault(key, dict(index=len(self.image_uavs), kind=kind))
        if r['kind'] != kind:
            raise Unsupported('stored image %s used as %s and %s' % (key, r['kind'], kind))
        return ('u%d' if kind == 'raw' else 'w%d') % r['index']

    def sampler(self, key, comparison=False, lod_limits=False):
        if not isinstance(key[0], int):
            raise Unsupported('sampler from a descriptor')
        # A comparison sampler, or one that carries the descriptor's LOD limits (image_sample_b), is its own binding
        # even when its words equal an ordinary sampler's.
        key = ('cmp',) + tuple(key) if comparison else ('lod',) + tuple(key) if lod_limits else key
        return 'q%d' % self.samplers.setdefault(key, len(self.samplers))

    def sampler_lines(self):
        """The manifest lines of the static samplers, graphics and compute: smp, smpc or smpl by the key's prefix."""
        lines = []
        for key, i in self.samplers.items():
            prefixed = key[0] in ('cmp', 'lod')
            lines.append('%s %d %s' % ({'cmp': 'smpc', 'lod': 'smpl'}.get(key[0], 'smp'), i,
                                       ' '.join('0x%x' % w for w in key[prefixed:])))
        return lines

    def common(self):
        import pixel_aot
        out = ['#define F(x) asfloat(x)', '#define U(x) asuint(x)',
               'float mul_legacy(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }',
               'float med3(float a, float b, float c) { return max(min(a, b), min(max(a, b), c)); }',
               # High half of an unsigned 32x32 product, by halfword products so it fits 32-bit HLSL.
               'uint mul_hi_u32(uint a, uint b) { uint la = a & 0xffffu, lb = b & 0xffffu;'
               ' uint middle = (a >> 16) * lb + ((la * lb) >> 16);'
               ' return (a >> 16) * (b >> 16) + (middle >> 16) + (((middle & 0xffffu) + la * (b >> 16)) >> 16); }',
               # a * b + c for binary16 inputs with one rounding (#2073). The first form recovered the binary32 sum error
               # with a floating-point TwoSum; an NVIDIA driver (RTX 3070 Ti, 32.0.16.2101) simplified that error term to
               # zero despite `precise`, so ties rounded the wrong way. fma16_rne works on the encodings in 32-bit integer
               # arithmetic, so no driver floating-point rewrite can reach it. Infinities and NaNs keep float semantics.
               pixel_aot.SALU_HELPERS['fma16_rne'],
               'float fma16o(float a, float b, float c) {',
               '    if (!isfinite(a) || !isfinite(b) || !isfinite(c)) return a * b + c;',
               '    return f16tof32(fma16_rne(f32tof16(a), f32tof16(b), f32tof16(c)));',
               '}',
               # binary32 to binary16 rounded to nearest even: f32tof16 truncates, a binary16 result must round (#1723).
               'uint f16rn(float v) { uint b = asuint(v), s = (b >> 16) & 0x8000u, a = b & 0x7fffffffu;'
               ' if (a > 0x7f800000u) return s | 0x7e00u; if (a >= 0x47800000u) return s | 0x7c00u;'
               ' if (a < 0x38800000u) return s | uint(round(asfloat(a) * 16777216.0));'
               ' uint r = a + 0xfffu + ((a >> 13) & 1u); return s | min((r - 0x38000000u) >> 13, 0x7c00u); }',
               # The ten binary32 classes v_cmp_class_f32 tests, from the raw encoding (#1795).
               'uint class32(float value) { uint bits = asuint(value), magnitude = bits & 0x7fffffffu;'
               ' bool negative = bits >= 0x80000000u;'
               ' if (magnitude > 0x7f800000u) return (bits & 0x00400000u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7f800000u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x00800000u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               # The ten binary16 classes v_cmp_class_f16 tests, from the raw encoding (issue #1855):
               # signaling NaN, quiet NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
               'uint class16(uint bits) { uint h = bits & 0xffffu, magnitude = h & 0x7fffu;'
               ' bool negative = h >= 0x8000u;'
               ' if (magnitude > 0x7c00u) return (h & 0x0200u) != 0u ? 0x2u : 0x1u;'
               ' if (magnitude == 0x7c00u) return negative ? 0x4u : 0x200u;'
               ' if (magnitude >= 0x0400u) return negative ? 0x8u : 0x100u;'
               ' if (magnitude != 0u) return negative ? 0x10u : 0x80u;'
               ' return negative ? 0x20u : 0x40u; }',
               # The 64-bit shifts (issue #1815): a pair of words, so the count never reaches 32 in a 32-bit
               # shift and an in-place destination keeps the original value (the caller passes both words in).
               'uint2 shl64(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (lo >> sc);'
               ' return uint2(sh < 32u ? (lo << sb) : 0u, sh < 32u ? ((hi << sb) | cy) : (lo << sb)); }',
               'uint2 shr64u(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = hi >> sb;'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : 0u); }',
               'uint2 shr64i(uint lo, uint hi, uint sh) { sh &= 63u; uint sb = sh & 31u, sc = (32u - sb) & 31u;'
               ' uint cy = sb == 0u ? 0u : (hi << sc); uint h = asuint(asint(hi) >> sb);'
               ' return uint2(sh < 32u ? ((lo >> sb) | cy) : h, sh < 32u ? h : asuint(asint(hi) >> 31u)); }',
               'uint2 bfe64(uint lo, uint hi, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 64u - (s1 & 63u)); uint2 e = shr64u(lo, hi, s1);'
               ' e = uint2(e.x & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u)),'
               ' e.y & (w >= 64u ? 0xffffffffu : (w > 32u ? ((1u << ((w - 32u) & 31u)) - 1u) : 0u)));'
               ' uint sh = (64u - w) & 63u; uint2 t = shl64(e.x, e.y, sh);'
               ' return sg != 0u ? shr64i(t.x, t.y, sh) : e; }',
               'uint bfe32(uint x, uint s1, uint sg) { uint w = min((s1 >> 16u) & 127u, 32u - (s1 & 31u));'
               ' uint e = (x >> (s1 & 31u)) & (w >= 32u ? 0xffffffffu : ((1u << (w & 31u)) - 1u));'
               ' uint sh = (32u - w) & 31u; return sg != 0u ? asuint(asint(e << sh) >> sh) : e; }',
               'uint ldu(ByteAddressBuffer b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; '
               'return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }']
        if getattr(self, 'helper_text', None) is None:  # upstream unit probes build a Pipeline without __init__
            # The draw stages, or the compute stage of upstream's ComputePipeline (slot 67).
            stages = [getattr(self, k) for k in ('vs', 'ps', 'cs') if getattr(self, k, None) is not None]
            self.helper_text = '\n'.join(line for s in stages for line in s.body())
        out += [text for name, text in HELPERS.items() if re.search(r'\b%s\(' % name, self.helper_text)]
        out += ['cbuffer entry : register(b0) { %s };' % ' '.join('uint ud%d;' % i for i in range(len(self.uds)))]
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            out.append('%s r%d : register(t%d);' % ({'buf': 'ByteAddressBuffer', 'tex1d': 'Texture1D<float4>', 'tex2d': 'Texture2D<float4>', 'tex2da': 'Texture2DArray<float4>', 'tex3d': 'Texture3D<float4>'}[r['kind']], r['index'], r['index']))
        for key, i in getattr(self, 'uavs', {}).items():
            out.append('RWByteAddressBuffer u%d : register(u%d);' % (i, i))
        for key, r in sorted(Pipeline.stored_images(self).items(), key=lambda kv: kv[1]['index']):
            out.append('%s w%d : register(u%d);' % ({'tex2d': 'RWTexture2D<float4>'}[r['kind']], r['index'], r['index']))
        for key, i in self.samplers.items():
            out.append('%s q%d : register(s%d);' % ('SamplerComparisonState' if key[0] == 'cmp' else 'SamplerState', i, i))
        if any(getattr(self, 'heapvars', ())):  # tests build bare pipelines without __init__
            out += ['ByteAddressBuffer heapmem : register(t0, space1);', 'ByteAddressBuffer vinfo : register(t1, space1);',
                    'uint x2w_vid(uint h) { return heapmem.Load(h) == 0x%08xu ? heapmem.Load(h + 4) : 0xffffffffu; }'
                    % HEAP_TOKEN,
                    'uint x2w_hix(uint h, uint k) { uint i = x2w_vid(h); return i < %du ? 8u + 8u * i + k : k; }' % HEAP_IDS,
                    'uint x2w_vstride(uint h) { uint i = x2w_vid(h); return i < %du ? vinfo.Load(i * 16u) : 0u; }' % HEAP_IDS]
            for kind in sorted(set(k for k in self.heapvars if k), key=lambda k: HEAP_KINDS[k][1]):
                name, slot, hlsl_type = HEAP_KINDS[kind]
                out.append('%s %s[%d] : register(t0, space%d);' % (hlsl_type, name, 8 + 8 * HEAP_IDS, 2 + slot))
            if getattr(self, 'draw_heapwords', False):
                out += ['ByteAddressBuffer x2w_vdesc : register(t2, space1);',
                        'uint x2w_hword(uint h, uint i) { uint id = x2w_vid(h); return id < %du ? '
                        'x2w_vdesc.Load(id * 32u + 4u * i) : 0u; }' % HEAP_IDS]
        if self.uses_sampler_heap:
            out.append('SamplerState qh[%d] : register(s0, space%d);' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE))
            out.append('uint x2w_sampler_entry(uint offset) { return (offset & 15u) != 0 || offset >= %du ? %du : offset >> 4; }'
                       % (16 * SAMPLER_HEAP_FALLBACK, SAMPLER_HEAP_FALLBACK))
        if getattr(self, 'uses_pack16_rne', False):
            # IEEE binary32 to binary16 round-to-nearest-even (quieting NaN while preserving its sign,
            # matching RDNA 2 hardware on binary32 inputs), plus exact UNORM16 ratio-to-half conversion
            # (avoiding binary32 double-rounding at 0xffaf -> 0x3bfd and 0xffef -> 0x3bff).
            out.extend((
                'uint pack16_rne(float value) {',
                '    uint bits = asuint(value), sign = (bits >> 16) & 0x8000u, abs_bits = bits & 0x7fffffffu;',
                '    if (abs_bits > 0x7f800000u) return sign | 0x7e00u;',
                '    if (abs_bits >= 0x477ff000u) return sign | 0x7c00u;',
                '    if (abs_bits >= 0x38800000u) {',
                '        uint shifted = abs_bits - 0x38000000u;',
                '        return sign | ((shifted + 0x00000fffu + ((shifted >> 13) & 1u)) >> 13);',
                '    }',
                '    if (abs_bits < 0x33000000u) return sign;',
                '    uint sig = 0x00800000u | (abs_bits & 0x007fffffu), shift = 126u - (abs_bits >> 23);',
                '    uint half_m = sig >> shift, rem = sig & ((1u << shift) - 1u), mid = 1u << (shift - 1u);',
                '    if (rem > mid || (rem == mid && (half_m & 1u) != 0u)) half_m += 1u;',
                '    return sign | half_m;',
                '}',
                'uint pack16_unorm16(float value) {',
                '    uint u = uint(saturate(value) * 65535.0 + 0.5);',
                '    if (u <= 3u) return u << 8;',
                '    if (u >= 65535u) return 0x3c00u;',
                '    uint msb = firstbithigh(u);',
                '    uint num = u << (26u - msb);',
                '    uint q = num / 65535u;',
                '    if (num - q * 65535u > 32767u) q += 1u;',
                '    return ((msb - 1u) << 10) + (q - 1024u);',
                '}',
                'uint pack16_d16(float value, uint unorm16) {',
                '    return unorm16 != 0u ? pack16_unorm16(value) : pack16_rne(value);',
                '}',
            ))
        if self.uses_gather_2d_lod:
            out.extend((
                'float4 gather_2d_lod(Texture2D<float4> tex, float2 uv, float lod, uint component, float max_lod) {',
                '    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);',
                '    if ((asuint(uv.x) & 0x7f800000u) != 0x7f800000u &&',
                '        (asuint(uv.y) & 0x7f800000u) != 0x7f800000u &&',
                '        (asuint(lod) & 0x7f800000u) != 0x7f800000u) {',
                '        uint width, height, levels;',
                '        tex.GetDimensions(0u, width, height, levels);',
                '        if (levels != 0u) {',
                '            uint mip = min(uint(floor(clamp(lod, 0.0f, max_lod) + 0.5f)), levels - 1u);',
                '            tex.GetDimensions(mip, width, height, levels);',
                '            int2 edge = int2(width, height) - 1;',
                '            precise float2 scaled = uv * float2(width, height) - 0.5f;',
                '            int2 origin = int2(floor(clamp(scaled, float2(-1.0f, -1.0f), float2(width, height))));',
                '            result = float4(tex.Load(int3(clamp(origin + int2(0, 1), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin + int2(1, 1), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin + int2(1, 0), int2(0, 0), edge), int(mip)))[component],',
                '                            tex.Load(int3(clamp(origin,              int2(0, 0), edge), int(mip)))[component]);',
                '        }',
                '    }',
                '    return result;',
                '}',
            ))
        if self.uses_store_d16_x:
            out.extend((
                'uint d16_to_unorm16(uint h) {',
                '    float x = f16tof32(h);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = saturate(x);',
                '    precise float scaled = saturated * 65535.0f;',
                '    return (uint)floor(scaled + 0.5f);',
                '}',
                'uint d16_to_snorm16(uint h) {',
                '    float x = f16tof32(h);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = clamp(x, -1.0f, 1.0f);',
                '    precise float scaled = saturated * 32767.0f;',
                '    precise float rounded = scaled >= 0.0f ? floor(scaled + 0.5f) : ceil(scaled - 0.5f);',
                '    return ((uint)(int)rounded) & 0xffffu;',
                '}',
                'void store_format_d16_x(RWByteAddressBuffer buf, uint word3, uint stride, uint num_records, '
                'uint idx, uint voff, uint soff, uint imm_off, uint val) {',
                '    if ((word3 & 0xcef80000u) != 0u || (word3 & 7u) != 4u) return;',
                '    uint oob_sel = (word3 >> 28u) & 3u;',
                '    if ((oob_sel != 0u && oob_sel != 3u) || stride > 0x3fffu) return;',
                '    uint field = (word3 >> 12u) & 127u;',
                '    uint fmt = 0u;',
                '    if (((word3 >> 24u) & 1u) != 0u) {',
                '        fmt = field;',
                '    } else {',
                '        if (field == 16u) fmt = 7u;',
                '        else if (field == 17u) fmt = 8u;',
                '        else if (field == 20u) fmt = 11u;',
                '        else if (field == 21u) fmt = 12u;',
                '        else if (field == 23u) fmt = 13u;',
                '        else if (field == 39u) fmt = 22u;',
                '    }',
                '    uint elem_bytes = fmt == 22u ? 4u : (fmt == 7u || fmt == 8u || fmt == 11u || fmt == 12u || fmt == 13u ? 2u : 0u);',
                '    if (elem_bytes == 0u) return;',
                '    if (voff > 0xffffffffu - imm_off) return;',
                '    uint rec_off = voff + imm_off;',
                '    uint addr = 0u;',
                '    if (oob_sel == 3u) {',
                '        if (stride != 0u) return;',
                '        if (soff > 0xffffffffu - rec_off) return;',
                '        addr = soff + rec_off;',
                '        if (elem_bytes > num_records || addr > num_records - elem_bytes) return;',
                '    } else {',
                '        if (stride == 0u) return;',
                '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;',
                '        if (idx > (0xffffffffu - rec_off) / stride) return;',
                '        uint elem_addr = idx * stride + rec_off;',
                '        if (soff > 0xffffffffu - elem_addr) return;',
                '        addr = elem_addr + soff;',
                '    }',
                '    uint uav_bytes = 0u;',
                '    buf.GetDimensions(uav_bytes);',
                '    uint h = val & 0xffffu;',
                '    if (fmt == 22u) {',
                '        if ((addr & 3u) != 0u || uav_bytes < 4u || addr > uav_bytes - 4u) return;',
                '        buf.Store(addr, asuint(f16tof32(h)));',
                '        return;',
                '    }',
                '    if ((addr & 1u) != 0u) return;',
                '    uint out16 = h;',
                '    if (fmt == 7u) out16 = d16_to_unorm16(h);',
                '    else if (fmt == 8u) out16 = d16_to_snorm16(h);',
                '    uint base = addr & 0xfffffffcu;',
                '    if (uav_bytes < 4u || base > uav_bytes - 4u) return;',
                '    uint shift = (addr & 2u) << 3u;',
                '    uint keep = ~(0xffffu << shift);',
                '    uint ins = (out16 & 0xffffu) << shift;',
                '    uint cur = buf.Load(base);',
                '    for (;;) {',
                '        uint next = (cur & keep) | ins;',
                '        uint prev = 0u;',
                '        buf.InterlockedCompareExchange(base, cur, next, prev);',
                '        if (prev == cur) break;',
                '        cur = prev;',
                '    }',
                '}',
            ))
        if 'v_sad_u8' in self.uses_sad:
            out.append(
                'uint sad_u8(uint a, uint b, uint c) {'
                ' return uint(abs(int(a & 0xffu) - int(b & 0xffu)))'
                ' + uint(abs(int((a >> 8) & 0xffu) - int((b >> 8) & 0xffu)))'
                ' + uint(abs(int((a >> 16) & 0xffu) - int((b >> 16) & 0xffu)))'
                ' + uint(abs(int((a >> 24) & 0xffu) - int((b >> 24) & 0xffu))) + c; }')
        if 'v_sad_u16' in self.uses_sad:
            out.append(
                'uint sad_u16(uint a, uint b, uint c) {'
                ' return uint(abs(int(a & 0xffffu) - int(b & 0xffffu)))'
                ' + uint(abs(int(a >> 16) - int(b >> 16))) + c; }')
        if 'v_sad_u32' in self.uses_sad:
            out.append(
                'uint sad_u32(uint a, uint b, uint c) {'
                ' return (a > b ? a - b : b - a) + c; }')
        if self.uses_store_format_xyzw:
            legacy = ['        %sif (field == %du) fmt = %du;' % ('' if i == 0 else 'else ', field, code)
                      for i, (field, code) in enumerate(sorted(STORE_FORMAT_XYZW_LEGACY_FORMATS.items()))]
            out.extend((
                'uint store_unorm(uint bits, float scale) {',
                '    float x = asfloat(bits);',
                '    if (isnan(x)) return 0u;',
                '    precise float saturated = saturate(x);',
                '    precise float scaled = saturated * scale;',
                '    return (uint)floor(scaled + 0.5f);',
                '}',
                'void store_format_xyzw(RWByteAddressBuffer buf, uint word3, uint stride, uint num_records, '
                'uint idx, uint voff, uint soff, uint imm_off, uint4 v) {',
                '    if ((word3 & 0xcef80000u) != 0u || (word3 & 0xfffu) != 0xfacu) return;',
                '    uint oob_sel = (word3 >> 28u) & 3u;',
                '    if ((oob_sel != 0u && oob_sel != 3u) || stride > 0x3fffu) return;',
                '    uint field = (word3 >> 12u) & 127u;',
                '    uint fmt = 0u;',
                '    if (((word3 >> 24u) & 1u) != 0u) {',
                '        fmt = field;',
                '    } else {',
                *legacy,
                '    }',
                '    uint elem_bytes = (fmt == 75u || fmt == 77u) ? 16u : fmt == 71u ? 8u : (fmt == 50u || fmt == 56u) ? 4u : 0u;',
                '    if (elem_bytes == 0u) return;',
                '    if (voff > 0xffffffffu - imm_off) return;',
                '    uint rec_off = voff + imm_off;',
                '    uint addr = 0u;',
                '    if (oob_sel == 3u) {',
                '        if (stride != 0u) return;',
                '        if (soff > 0xffffffffu - rec_off) return;',
                '        addr = soff + rec_off;',
                '        if (elem_bytes > num_records || addr > num_records - elem_bytes) return;',
                '    } else {',
                '        if (stride == 0u) return;',
                '        if (idx >= num_records || elem_bytes > stride || rec_off > stride - elem_bytes) return;',
                '        if (idx > (0xffffffffu - rec_off) / stride) return;',
                '        uint elem_addr = idx * stride + rec_off;',
                '        if (soff > 0xffffffffu - elem_addr) return;',
                '        addr = elem_addr + soff;',
                '    }',
                '    if ((addr & 3u) != 0u) return;',
                '    uint uav_bytes = 0u;',
                '    buf.GetDimensions(uav_bytes);',
                '    if (uav_bytes < elem_bytes || addr > uav_bytes - elem_bytes) return;',
                '    if (elem_bytes == 16u) {',
                '        buf.Store4(addr, v);',
                '    } else if (fmt == 71u) {',
                '        buf.Store2(addr, uint2(pack16_rne(asfloat(v.x)) | (pack16_rne(asfloat(v.y)) << 16u), '
                'pack16_rne(asfloat(v.z)) | (pack16_rne(asfloat(v.w)) << 16u)));',
                '    } else if (fmt == 56u) {',
                '        buf.Store(addr, store_unorm(v.x, 255.0f) | (store_unorm(v.y, 255.0f) << 8u) '
                '| (store_unorm(v.z, 255.0f) << 16u) | (store_unorm(v.w, 255.0f) << 24u));',
                '    } else {',
                '        buf.Store(addr, store_unorm(v.x, 1023.0f) | (store_unorm(v.y, 1023.0f) << 10u) '
                '| (store_unorm(v.z, 1023.0f) << 20u) | (store_unorm(v.w, 3.0f) << 30u));',
                '    }',
                '}',
            ))
        return out

    def interface(self, for_pixel):
        fields = []
        for p in range(max(self.params) + 1 if self.params else 0):
            fields.append('    float4 param%d : PARAM%d;' % (p, p))
        # Attributes read without interpolation get a second, flat copy of the vertex output.
        for p in self.flat:
            fields.append('    nointerpolation uint4 fparam%d : FPARAM%d;' % (p, p))
        # Attributes read as binary16 halves (v_interp_p2_f16) get the two halves as interpolated f32 copies.
        for p in getattr(self, 'half', ()):
            fields.append('    float4 loparam%d : LOPARAM%d;' % (p, p))
            fields.append('    float4 hiparam%d : HIPARAM%d;' % (p, p))
        for p in self.f16_plain:
            fields.append('    float4 hparam%d_lo : HPARAML%d;' % (p, p))
            fields.append('    float4 hparam%d_hi : HPARAMH%d;' % (p, p))
        fields.append('    float4 pos : SV_Position;')
        if for_pixel and FRONT_FACE in getattr(self, 'ps_input_vgprs', {}).values():
            fields.append('    bool ff : SV_IsFrontFace;')  # system-generated: pixel input only
        clip = self.vs.exports.get('clip')
        if clip:
            fields.append('    float%s clip0 : SV_ClipDistance0;' % ('' if clip == 1 else clip))
        return fields

    def hlsl_vs(self):
        s = self.vs
        out = self.common() + ['struct VSOut {'] + self.interface(False) + ['};', 'VSOut body(uint vid) {', '    VSOut o = (VSOut)0;']
        s.used_v.add(0)
        flat = re.compile(r'\s*o\.(f|lo|hi)param(\d+) = ')

        def unread(line):
            m = flat.match(line)
            return m and int(m.group(2)) not in (self.flat if m.group(1) == 'f' else getattr(self, 'half', ()))
        body = [line for line in s.body() if not unread(line)]
        if s.fetched:
            # Behind a fetch call a register read as 0 by default would hide a missing input, so every read must be
            # of a register defined on all paths. Proven here only for the entry block (straight-line code from the
            # call to the first branch or branch target): v0, the fetched registers, and registers an unpredicated
            # write in that block assigned before the read. A write under an exec predicate or in any later block
            # does not define a register.
            written = {0}
            for index, (start, lines, term) in enumerate(s.blocks):
                for line in lines + ([term[1]] if term and term[0] == 'branch' else []):
                    target = re.match(r'v(\d+) = (.*)', line)
                    unwritten = {int(n) for n in re.findall(r'\bv(\d+)\b', target.group(2) if target else line)} - written
                    if unwritten:
                        raise Unsupported('vertex stage reads v%d, which neither the fetch nor an unconditional write '
                                          'before the first branch defines' % min(unwritten))
                    if target and index == 0 and start == 0:
                        written.add(int(target.group(1)))
        out += s.declarations() + ['    v0 = vid;'] + body + ['    return o;', '}',
            '// Rectangle lists arrive as six vertices per rectangle: corners 0 1 2, 0 2 and the fourth corner 0 + 2 - 1.',
            'VSOut main(uint vid : SV_VertexID) {',
            '    if (ud%d == 0) return body(vid);' % self.rect,
            '    uint first = (vid / 6) * 3, corner = vid % 6;',
            '    if (corner < 3) return body(first + corner);',
            '    if (corner == 3) return body(first);',
            '    if (corner == 4) return body(first + 2);',
            '    VSOut p = body(first), q = body(first + 1), r = body(first + 2), o;']
        for p in range(max(self.params) + 1 if self.params else 0):
            out.append('    o.param%d = p.param%d + r.param%d - q.param%d;' % (p, p, p, p))
        out += ['    o.fparam%d = p.fparam%d;' % (p, p) for p in self.flat]
        for p in getattr(self, 'half', ()):
            out += ['    o.%sparam%d = p.%sparam%d + r.%sparam%d - q.%sparam%d;' % ((h, p) * 4) for h in ('lo', 'hi')]
        if self.vs.exports.get('clip'):
            out.append('    o.clip0 = p.clip0 + r.clip0 - q.clip0;')  # linear in the corner, like the params
        for p in self.f16_plain:
            out += ['    o.hparam%d_%s = p.hparam%d_%s + r.hparam%d_%s - q.hparam%d_%s;' % (p, h, p, h, p, h, p, h)
                    for h in ('lo', 'hi')]
        out += ['    o.pos = p.pos + r.pos - q.pos;', '    return o;', '}']
        return '\n'.join(out) + '\n'

    def ps_profile(self):
        """The shader profile the pixel stage compiles with: ps_6_0 once it holds a quad read, else ps_5_1."""
        return 'ps_6_0' if self.ps.needs_sm6 else 'ps_5_1'

    def hlsl_ps(self):
        refusal = getattr(self, 'ps_input_refusal', None)  # the tests' bare pipelines have none
        if refusal:
            raise Unsupported(refusal)
        s = self.ps
        targets = sorted(int(k[3:]) for k in s.exports if k.startswith('mrt'))
        out = self.common() + ['struct PSIn {'] + self.interface(True) + ['};']
        depth = 'depth' in s.exports
        if not targets and not depth:  # depth-only: no output struct, so the pipeline declares no SV_Target
            out += ['void main(PSIn i) {'] + s.declarations() + s.body() + ['}']
            return '\n'.join(out) + '\n'
        out.append('struct PSOut {')
        for t in targets:
            kind = self.mrt_kind(t)
            out.append('    %s mrt%d : SV_Target%d;' % ({'int': 'int4', 'uint': 'uint4'}.get(kind, 'float4'), t, t))
        if depth:
            out.append('    float depth : SV_Depth;')
        out += ['};', 'PSOut main(PSIn i) {', '    PSOut o = (PSOut)0;']
        out += s.declarations() + s.body() + ['    return o;', '}']
        return '\n'.join(out) + '\n'

    def manifest(self):
        # A manifest that names a writable view is version 2, so a runtime that reads only version 1 lines can tell.
        lines = ['x2w-manifest %d' % (2 if Pipeline.stored_images(self) else 1), 'ud %d' % len(self.uds)]
        for i, source in enumerate(self.uds):
            lines.append('udsrc %d %s' % (i, ' '.join(str(x) for x in source)))
        for key, r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            source, off = key
            where = {'table': 'table %d %d' % (source[1] if len(source) > 1 else 0, off or 0), 'vbtab': 'vb %d' % ((off or 0) // 16),
                     'root': 'root %d' % (source[1] if len(source) > 1 else 0)}.get(source[0])
            if where is None:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
            if r.get('d16_exact'):
                lines.append('d16srv %d exact' % r['index'])
        for (source, off), i in sorted(getattr(self, 'uavs', {}).items(), key=lambda kv: kv[1]):
            if source[0] == 'table':
                lines.append('uav %d buf table %d %d' % (i, source[1] if len(source) > 1 else 0, off or 0))
            elif source[0] == 'root' and not off:
                lines.append('uav %d buf root %d 0' % (i, source[1] if len(source) > 1 else 0))
            else:
                raise Unsupported('uav through %s at %s' % (source, off))
        if any(getattr(self, 'heapvars', ())):
            lines.append('heap 1')
            if getattr(self, 'draw_heapwords', False):
                lines.append('heapwords 1')
        for key, r in sorted(Pipeline.stored_images(self).items(), key=lambda kv: kv[1]['index']):
            source, off = key
            if source[0] != 'table':
                raise Unsupported('stored image through %s' % (source,))
            lines.append('uav %d %s table %d %d' % (r['index'], r['kind'], source[1] if len(source) > 1 else 0, off or 0))
        lines += self.sampler_lines()
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        if getattr(self, 'fma_rule', None) and self.fma_rule != 'legacy':
            lines.append('fma %s' % self.fma_rule)
        if getattr(self.ps, 'quad_divergent_used', None):
            # X2W_PS_QUAD_DIVERGENT lowered a quad read or image_sample_b a lane may skip (Stage.translate)
            lines.append('quaddivergent 1')
        if FRONT_FACE in getattr(self, 'ps_input_vgprs', {}).values():
            lines.append('frontface 1')  # reads SV_IsFrontFace: depends on the PSO winding (see PS_INPUT_HLSL)
        lines.append('params %d' % (max(self.params) + 1 if self.params else 0))
        if self.ps is None:
            lines.append('ps none')
            return '\n'.join(lines) + '\n'
        lines.append('targets %s' % ' '.join(k[3:] for k in sorted(self.ps.exports) if k.startswith('mrt')))
        # 'wmask <target> <mask>': the export leaves the channels outside the mask unwritten, so the
        # runtime keeps them out of that target's RenderTargetWriteMask. Full masks are not written.
        for name, mask in sorted(getattr(self.ps, 'write_masks', {}).items(), key=lambda kv: int(kv[0][3:])):
            if mask != 0xF:
                lines.append('wmask %d 0x%x' % (int(name[3:]), mask))
        return '\n'.join(lines) + '\n'


class ComputePipeline(Pipeline):
    """One compute program with the root and resource model of Pipeline. COMPUTE_PGM_RSRC2 and the thread group
    size are pipeline register state the caller supplies; COMPUTE_PGM_RSRC2 is decoded as compute_store_aot does
    (#1608). Root parameters from s4 are an inference (see Stage.setup_entry)."""

    def stored_images(self):
        return super().stored_images()

    def __init__(self, cs_code, root_blob, rsrc2, group, sampler_heap=None, refusals=None, decoded=None):
        from compute_store_aot import checked_group, entry_registers
        try:
            self.entry = entry_registers(rsrc2)  # (user SGPRs, {axis: group-ID SGPR}, thread-ID VGPRs)
            self.group = checked_group(group)
        except ValueError as reason:
            raise Unsupported(str(reason)) from None
        self.root = parse_root(root_blob)
        self.vs_fetch, self.packed_position_vgpr, self.rtv_formats = None, None, ()
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.uses_gather_2d_lod = False
        self.uses_store_d16_x = False
        self.uses_sad = set()
        self.resources, self.uavs, self.samplers, self.uds = {}, {}, {}, []
        self.notes, self.modified, self.other_modified = set(), set(), set()  # other: unproved descriptor writes
        self.lod_unvalidated = set()
        # LLVM AMDGPUUsage, COMPUTE_PGM_RSRC2.GRANULATED_LDS_SIZE on gfx10: 128 dwords per unit.
        self.lds_words = ((rsrc2 >> 15) & 0x1ff) * 128
        self.cs = Stage(self, 'cs', cs_code, refusals=refusals, decoded=decoded)

    def resource(self, key, kind):
        if key in self.uavs:
            raise Unsupported('descriptor %s used for a load and a store' % (key,))
        return super().resource(key, kind)

    def uav(self, key, kind='raw'):
        if key in self.resources:
            raise Unsupported('descriptor %s used for a load and a store' % (key,))
        return super().uav(key) if kind == 'raw' else self.image_uav(key, kind)

    def common(self):
        return [line for line in super().common() if self.uds or not line.startswith('cbuffer entry')]

    def hlsl_cs(self):
        s = self.cs
        if s.exports:
            raise Unsupported('export in a compute program')
        body = s.body()
        shared = ['groupshared uint lds[%d];' % self.lds_words] if getattr(s, 'uses_lds', False) else []
        return '\n'.join(self.common() + shared + ['[numthreads(%d, %d, %d)]' % self.group,
                                          'void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {']
                         + s.declarations() + ['    ' + line for line in s.entry] + body + ['}']) + '\n'

    def manifest(self):
        # Version 2: `stage cs`, `group` and `uav` lines, which a graphics-only reader must not take for a draw.
        lines = ['x2w-manifest 2', 'stage cs', 'group %d %d %d' % self.group, 'ud %d' % len(self.uds)]
        lines += ['udsrc %d %s' % (i, ' '.join(str(x) for x in source)) for i, source in enumerate(self.uds)]
        for (source, off), r in sorted(self.resources.items(), key=lambda kv: kv[1]['index']):
            if source[0] == 'table':
                where = 'table %d %d' % (source[1], off or 0)
            elif source[0] == 'root':
                where = 'root %d' % source[1]
            else:
                raise Unsupported('resource through %s' % (source,))
            lines.append('srv %d %s %s' % (r['index'], r['kind'], where))
        for (source, off), index in self.uavs.items():
            lines.append('uav %d table %d %d' % (index, source[1], off))
        for (source, off), r in self.stored_images().items():
            lines.append('uav %d tex2d float table %d %d channels 4' % (r['index'], source[1], off))
        lines += self.sampler_lines()
        if self.uses_sampler_heap:
            lines.append('smpheap %d %d fallback %d' % (SAMPLER_HEAP_SIZE, SAMPLER_HEAP_SPACE, SAMPLER_HEAP_FALLBACK))
        return '\n'.join(lines) + '\n'


class MeshPipeline(Pipeline):
    """A slot 57 mesh pipeline with an optional pixel stage. The mesh stage is self.vs."""
    vertex_kind = 'ms'
    uses_mesh_skeleton = True

    def hlsl_ms(self):
        s = self.vs
        outputs = {'f': self.flat, 'lo': self.half, 'hi': self.half}
        def linked_output(match):
            return match[0] if int(match[2]) in outputs[match[1]] else ''

        lines = [re.sub(r' o\.(f|lo|hi)param(\d+) = [^;]*;', linked_output,
                        re.sub(r' o\.hparam(\d+)_(?:lo|hi) = [^;]*;',
                               lambda m: m[0] if int(m[1]) in self.f16_plain else '', line))
                 for line in s.body()]
        if self.ps is None:
            # Nothing reads a parameter without a pixel stage: drop its writes but keep the rest of the export
            # line (its exec guard and mesh_vertex = true), as for unread fparam/hparam fields above.
            lines = [re.sub(r' o\.param\d+ = [^;]*;', '', line) for line in lines]
        return '\n'.join(
            self.common() + ['struct VSOut {'] + self.interface(False) + [
                '};', 'groupshared uint lds[2];', '[numthreads(%d, 1, 1)]' % MESH_THREADS, '[outputtopology("triangle")]',
                'void main(uint tid : SV_GroupIndex, out vertices VSOut verts[%d], out indices uint3 tris[%d]) {'
                % (MESH_THREADS, MESH_THREADS), '    VSOut o = (VSOut)0; bool mesh_vertex = false;']
            + s.declarations() + ['    ' + line for line in s.entry] + lines
            + ['    if (mesh_vertex) verts[tid] = o;', '}']) + '\n'


def fnv(data, h=1469598103934665603):
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def pipeline_key(vs_code, ps_code, root_blob, layout=b''):
    """Hash the startup translation revision and the three blobs in runtime order, then a stream's input layout."""
    key = fnv(root_blob, fnv(ps_code, fnv(vs_code, fnv(b'x2w-startup-shader-3'))))
    return fnv(layout, key) if layout else key


def compute_key(cs_code, data, root_blob, desc, state_a=b'', state_b=b''):
    """The runtime key of a device slot 67 compute request (req_<key>.c67, .data, .root, .desc, .sa, .sb): the
    graphics revision followed by "cs", then each part preceded by its length as a little-endian u64: the code,
    the words 7-8 blob, the root signature, the 12 description words with their pointers zeroed and the two
    state blocks."""
    key = fnv(b'x2w-startup-shader-3cs')
    for part in (cs_code, data, root_blob, desc, state_a, state_b):
        key = fnv(part, fnv(struct.pack('<Q', len(part)), key))
    return key


S_ENDPGM = 0xbf810000


def compute_entry(state_a):
    """(COMPUTE_PGM_RSRC2, group size) from state block A. A hypothesis from a local observation, not decoded
    by the stand-in (docs/startup-runtime.md#compute-pipeline-translation): dword 3 = X | Y << 16, dword 4
    bits 15:0 = Z, dword 6 = COMPUTE_PGM_RSRC2."""
    if len(state_a) != 28:
        raise Unsupported('compute state block A is %d bytes, not 28' % len(state_a))
    words = struct.unpack('<7I', state_a)
    return words[6], (words[3] & 0xffff, words[3] >> 16, words[4] & 0xffff)


def load_compute_request(out_dir, key, refusals=None):
    """The ComputePipeline of a slot 67 request, after checking the files against .desc and the key."""
    name = os.path.join(out_dir, 'req_%016x' % key)
    try:
        parts = {ext: open(name + ext, 'rb').read() for ext in ('.c67', '.data', '.root', '.desc', '.sa', '.sb')}
    except OSError:
        raise Unsupported('unreadable shader request') from None  # the error text would name the cache path
    desc = parts['.desc']
    if len(desc) != 96:
        raise Unsupported('compute description is %d bytes, not 96' % len(desc))
    words = struct.unpack('<12Q', desc)
    for ext, word in (('.c67', 6), ('.data', 8), ('.sa', 2), ('.sb', 4)):
        if len(parts[ext]) != words[word]:
            raise Unsupported('compute request %s is %d bytes, the description says %d' % (ext, len(parts[ext]), words[word]))
    if compute_key(parts['.c67'], parts['.data'], parts['.root'], desc, parts['.sa'], parts['.sb']) != key:
        raise Unsupported('compute request files do not match the key in their name')
    code = parts['.c67']
    if not code or len(code) % 4:
        raise Unsupported('compute code of %d bytes is not whole dwords' % len(code))
    if S_ENDPGM not in struct.unpack('<%dI' % (len(code) // 4), code):
        raise Unsupported('compute code without s_endpgm')
    rsrc2, group = compute_entry(parts['.sa'])
    return ComputePipeline(code, parts['.root'], rsrc2, group, refusals=refusals)


def compute_refusals(out_dir, key):
    """Refusal sites of one slot 67 compute request, without publishing a translation."""
    refusals, op = [], 'request loading'
    try:
        pipe = load_compute_request(out_dir, key, refusals)
        op = 'hlsl_cs'
        pipe.hlsl_cs()
        op = 'manifest'
        pipe.manifest()
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
    except Exception as error:  # noqa: BLE001 - one request must not end the census
        refusals.append((handler_error(op, error, refusals), 'pipeline'))
    return refusals


def log_unvalidated_lod(key, pipe):
    """Once per translated module: the sample forms that read a heap sampler keep its LOD fields unvalidated (#2221)."""
    forms = getattr(pipe, 'lod_unvalidated', None)  # test stand-ins for Pipeline do not set it
    if forms:
        print('%016x: %s through a sampler heap: sampler MIN_LOD, MAX_LOD and LOD bias are not validated (#2221)'
              % (key, ', '.join(sorted(forms))), file=sys.stderr, flush=True)


def translate_compute(out_dir, key):
    """Writes <key>.cs.hlsl, then <key>.cs.cso (DXC cs_6_0) and <key>.txt last, or <key>.fail with the reason.
    X2W_FXC set to the empty string publishes the manifest alone, as for graphics."""
    name = os.path.join(out_dir, '%016x' % key)
    with suppress(FileNotFoundError):
        os.remove(name + '.txt')  # the result is unpublished until validation and compilation finish
    try:
        pipe = load_compute_request(out_dir, key)
        hlsl, manifest = pipe.hlsl_cs(), pipe.manifest()
    except Unsupported as reason:
        reason = fail_reason(reason)
        open(name + '.fail', 'w').write(reason + '\n')
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.cs.hlsl', 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a compute stage needs shader model 6.0' % os.environ.get('X2W_DXC', '<unset>')
            open(name + '.fail', 'w').write(message + '\n')
            return message
        import subprocess
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            bytecode = os.path.join(staging, 'cs.cso')
            r = subprocess.run([compiler, '-Gis', '-T', 'cs_6_0', '-E', 'main', '-Fo', bytecode, name + '.cs.hlsl'],
                               capture_output=True, text=True)
            if r.returncode or not has_bytecode(bytecode):
                # DXC names the source path; keep only the diagnostic after it.
                lines = [re.sub(r'^.*?\.cs\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                detail = lines[0] if lines else ('?' if r.returncode else 'no bytecode produced')
                open(name + '.fail', 'w').write('dxc cs: %s\n' % detail[:500])
                return 'dxc cs failed: %s' % detail[:500]
            os.replace(bytecode, name + '.cs.cso')
    temporary = '%s.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.txt')
    with suppress(FileNotFoundError):
        os.remove(name + '.fail')
    return 'ok'


DEFAULT_FXC = r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe'
KITS_BIN = r'C:\Program Files (x86)\Windows Kits\10\bin'


def find_fxc():
    """The fxc.exe to compile with, '' when X2W_FXC is the empty string (manifest only), or None when none exists.

    Order: X2W_FXC, the 10.0.26100.0 SDK, the newest KITS_BIN/<version>/x64, then PATH."""
    configured = os.environ.get('X2W_FXC')
    if configured == '':
        return ''
    if configured and os.path.isfile(configured):
        return configured
    if os.path.isfile(DEFAULT_FXC):
        return DEFAULT_FXC
    versions = []
    try:
        for entry in os.listdir(KITS_BIN):
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', entry):
                versions.append((tuple(map(int, entry.split('.'))), entry))
    except OSError:
        pass
    for _, entry in sorted(versions, reverse=True):
        candidate = os.path.join(KITS_BIN, entry, 'x64', 'fxc.exe')
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which('fxc.exe')


def find_dxc():
    """The dxc.exe for a pixel stage that needs shader model 6.0, or None. Order: X2W_DXC, the newest
    KITS_BIN/<version>/x64, then PATH."""
    configured = os.environ.get('X2W_DXC')
    if configured and os.path.isfile(configured):
        return configured
    versions = []
    try:
        for entry in os.listdir(KITS_BIN):
            if re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', entry):
                versions.append((tuple(map(int, entry.split('.'))), entry))
    except OSError:
        pass
    for _, entry in sorted(versions, reverse=True):
        candidate = os.path.join(KITS_BIN, entry, 'x64', 'dxc.exe')
        if os.path.isfile(candidate):
            return candidate
    import shutil
    return shutil.which('dxc.exe') or shutil.which('dxc')


def has_bytecode(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def has_complete_mesh57_bytecode(path):
    """Check the DXContainer header and length without reading the shader body."""
    try:
        with open(path, 'rb') as shader:
            header = shader.read(32)
            if len(header) != 32 or header[:4] != b'DXBC':
                return False
            size, parts = struct.unpack_from('<II', header, 24)
            return size == os.fstat(shader.fileno()).st_size and 0 < parts <= (size - 32) // 4
    except OSError:
        return False


def translate_to(out_dir, key, vs_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='',
                 ps_hw_inputs=None, vs_out_cntl=None, vs_fetch=None, ps_attr_map=None):
    """Writes <key>.vs.hlsl, <key>.ps.hlsl and <key>.txt (last), or <key>.fail with the reason.

    With check, the pair is compiled to .vs.cso and .ps.cso first and a missing fxc is a failure.
    check=False, or X2W_FXC set to the empty string, publishes the manifest without bytecode."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        rw_keys = set()
        while True:
            try:
                pipe = Pipeline(vs_code, ps_code, root_blob, rtv_formats=rtv_formats, ps_hw_inputs=ps_hw_inputs,
                                vs_out_cntl=vs_out_cntl, rw_keys=rw_keys,
                                vs_fetch=vs_fetch, ps_attr_map=ps_attr_map)
                break
            except StoredSrv as conflict:
                # read before the first store: build again with every access to that buffer through its UAV
                if conflict.key in rw_keys or len(rw_keys) == 8:
                    raise
                rw_keys.add(conflict.key)
        if pipe.ps_entry_reads:
            # A request does not carry the SPI input layout, so the HLSL would read that input as 0 (for example
            # the packed pixel position a dither or a full-screen copy addresses with). Not written as a result.
            reg, off = min(pipe.ps_entry_reads.items(), key=lambda item: item[1])
            raise Unsupported('pixel stage reads entry VGPR v%d (an SPI input; layout not in the request) at %#x'
                              % (reg, off))
        vs, ps, manifest = pipe.hlsl_vs(), pipe.hlsl_ps(), pipe.manifest()
        if rtv_request:
            manifest += rtv_request + '\n'
    except Unsupported as reason:
        reason = fail_reason(reason)
        open(name + '.fail', 'w').write(reason + '\n')
        return 'unsupported: %s' % reason
    log_unvalidated_lod(key, pipe)
    open(name + '.vs.hlsl', 'w').write(vs)
    open(name + '.ps.hlsl', 'w').write(ps)
    fxc = find_fxc() if check else ''
    if fxc is None:
        message = 'fxc.exe not found (X2W_FXC=%s): set X2W_FXC to its path or install the Windows SDK' % os.environ.get('X2W_FXC', '<unset>')
        open(name + '.fail', 'w').write(message + '\n')
        return message
    manifest_path = name + '.txt'
    write_manifest = True
    # A quad intrinsic exists from shader model 6.0 only, and a pipeline does not mix DXBC and DXIL stages.
    sm6 = any(getattr(getattr(pipe, stage, None), 'needs_sm6', False) for stage in ('vs', 'ps'))
    if fxc and sm6:
        compiler = find_dxc()
        if compiler is None:
            message = 'dxc not found (X2W_DXC=%s): a DPP quad read needs shader model 6.0' % os.environ.get('X2W_DXC', '<unset>')
            open(name + '.fail', 'w').write(message + '\n')
            return message
        profiles = (('vs', 'vs_6_0'), ('ps', 'ps_6_0'))
    else:
        compiler, profiles = fxc, (('vs', 'vs_5_1'), ('ps', 'ps_5_1'))
    if fxc:
        import subprocess
        # X2W_FXC_FLAGS: extra fxc options, for example /Od for a stage the default optimiser takes minutes on
        # (the bytecode then runs unoptimised; the HLSL and so the result are the same). Default: none.
        extra = os.environ.get('X2W_FXC_FLAGS', '').split()
        if any(not re.fullmatch(r'/(Od|O[0-3]|Gfp)', flag) for flag in extra):
            # (options that change results or layouts, /Gis /Gpp /Zpr /Zpc, are not taken)
            raise SystemExit('X2W_FXC_FLAGS: only /Od, /O0-/O3 and /Gfp')
        if not sm6:
            for stage, profile in profiles:
                r = subprocess.run([fxc, '/nologo', '/T', profile, '/E', 'main'] + extra +
                                   ['/Fo', name + '.%s.cso' % stage, name + '.%s.hlsl' % stage],
                                   capture_output=True, text=True)
                if r.returncode:
                    open(name + '.fail', 'w').write('fxc %s: %s\n' % (stage, r.stderr[-2000:]))
                    return 'fxc %s failed: %s' % (stage, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?')
        else:  # upstream: a shader model 6.0 pair (dxc), staged and published together
            # An existing manifest can already point at these paths. Publish only bytecode from a successful pair.
            with tempfile.TemporaryDirectory(prefix='.fxc-', dir=out_dir) as staging:
                for stage, profile in profiles:
                    bytecode = os.path.join(staging, stage + '.cso')
                    if sm6:
                        command = [compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, name + '.%s.hlsl' % stage]
                    else:
                        command = [compiler, '/nologo', '/T', profile, '/E', 'main', '/Fo', bytecode, name + '.%s.hlsl' % stage]
                    r = subprocess.run(command, capture_output=True, text=True)
                    if r.returncode or not has_bytecode(bytecode):
                        detail = r.stderr.strip() or ('?' if r.returncode else 'no bytecode produced')
                        open(name + '.fail', 'w').write('fxc %s: %s\n' % (stage, detail[-2000:]))
                        return 'fxc %s failed: %s' % (stage, detail.splitlines()[-1])
                with suppress(FileNotFoundError, UnicodeError):
                    with open(manifest_path) as existing:
                        write_manifest = existing.read() != manifest
                # If the bindings changed, withdraw the old manifest before publishing new bytecode.
                if write_manifest:
                    with suppress(FileNotFoundError):
                        os.remove(manifest_path)
                for stage in ('vs', 'ps'):
                    os.replace(os.path.join(staging, stage + '.cso'), name + '.%s.cso' % stage)
    if not sm6:
        temporary = '%s.txt.%d.tmp' % (name, os.getpid())
        with open(temporary, 'w') as result:
            result.write(manifest)
        os.replace(temporary, name + '.txt')
    else:
        if write_manifest:
            temporary = '%s.%d.tmp' % (manifest_path, os.getpid())
            try:
                with open(temporary, 'w') as result:
                    result.write(manifest)
                os.replace(temporary, manifest_path)
            finally:
                with suppress(FileNotFoundError):
                    os.remove(temporary)
        if fxc:
            with suppress(FileNotFoundError):
                os.remove(name + '.fail')
    return 'ok'


def classify_decoded(ins):
    text = ' '.join(x[3] for x in ins if x[2] == 'exp')
    return 'vs' if 'pos' in text else 'ps' if 'mrt' in text or 'null' in text or not text else 'other'  # no export: a depth-only pixel shader


def classify(blob):
    return classify_decoded(decode(strip_header(blob)))


def load_request(out_dir, key, refusals=None, pair_info=None, *, validate_base_key=False):
    """Read a base request or an RTV variant and classify the source stages."""
    decoded = {}

    def classify_stage(blob):
        if refusals is None:
            return classify(blob)
        ins = decode(strip_header(blob), refusals)
        kind = classify_decoded(ins)
        decoded[kind] = ins
        return kind

    name = os.path.join(out_dir, 'req_%016x' % key)
    request = ''
    formats = ()
    base = key
    if os.path.isfile(name + '.rtv'):
        with open(name + '.rtv', encoding='ascii') as file:
            request = file.read(256)
        if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or fnv(request.encode('ascii')) != key:
            raise Unsupported('invalid RTV variant request')
        base, *formats = map(int, request.split()[1:])
        if base > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
            raise Unsupported('invalid RTV variant values')
        name = os.path.join(out_dir, 'req_%016x' % base)
    a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    if pair_info is not None and not request:
        import hashlib
        pair_info.append((hashlib.sha256(struct.pack('<Q', len(a)) + a + b).digest(), root))
    # A stream with an input layout publishes it as .ia (before .a) and hashes it into the key.
    layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
    if (request or validate_base_key) and pipeline_key(a, b, root, layout) != base:
        raise Unsupported('RTV variant source key mismatch' if request else 'shader request source key mismatch')
    fetch = attr_map = None
    if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
        # Slot 86 records complete containers in canonical (VS, PS) order.
        # Keep the container bytes in the cache key; translate only the bounded native code.
        import zlib
        from pixel_aot import container_code
        from resource_inventory import parts
        try:
            container, ps_container = a, b
            a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                    for blob, stage in ((a, 1), (b, 0)))
            attr_map = stream_interface(container, ps_container)
            if starts_with_fetch_call(a):
                fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
        except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
            raise Unsupported('invalid stream shader: %s' % error) from error
        if classify_stage(a) != 'vs' or classify_stage(b) != 'ps':
            raise Unsupported('stream stage and shader export disagree')
        kinds = {'vs': a, 'ps': b}
    else:
        kinds = {classify_stage(a): a, classify_stage(b): b}
    return kinds, root, formats, request, fetch, decoded, attr_map


def mesh57_key(ms, ps, root):
    value = fnv(b'x2w-startup-mesh-57-1')
    for part in (ms, ps, root):
        value = fnv(part, fnv(struct.pack('<Q', len(part)), value))
    return value


def translate_mesh57(out_dir, key):
    """Translate a fixed-descriptor mesh request of the accepted shape (mesh_skeleton), or raise Unsupported.

    Writes <key>.ms57.hlsl and the compiled .ms57.cso, their .ps57 counterparts if present, and <key>.ms57.txt (last).
    No runtime path loads these names yet."""
    source = os.path.join(out_dir, 'req_%016x' % key)
    ms, ps, root = (open(source + ext, 'rb').read() for ext in ('.m57', '.ps57', '.root'))
    if mesh57_key(ms, ps, root) != key:
        raise Unsupported('slot 57 mesh request key mismatch')
    parse_root(root)
    mesh = decode(strip_header(ms))
    exports = [assembly.split()[1] for _, _, op, assembly in mesh if op == 'exp']
    if not mesh or mesh[-1][2] != 's_endpgm' or not any(
        op == 's_sendmsg' and assembly == 's_sendmsg 0x9' for _, _, op, assembly in mesh
    ) or not any(target.startswith('pos') for target in exports) or 'target_20' not in exports:
        raise Unsupported('slot 57 mesh stage has no verified allocation and primitive export')
    if ps:
        pixel = decode(strip_header(ps))
        # A pixel shader with no colour output ends in a null-target export (LLVM SILateBranchLowering, Mesa ACO create_fs_null_export): depth-only.
        null_only = {a.split()[1] for _, _, op, a in pixel if op == 'exp'} == {'null'}
        if not pixel or pixel[-1][2] != 's_endpgm' or classify_decoded(pixel) != 'ps' and not null_only:
            raise Unsupported('slot 57 pixel stage has an incompatible export or termination')
        if null_only:
            raise Unsupported('slot 57 mesh with a depth-only pixel stage is not covered')
    pipe = MeshPipeline(ms, ps or None, root)
    sources = [('ms', 'ms_6_5', pipe.hlsl_ms())]
    if ps:
        sources.append(('ps', 'ps_6_0', pipe.hlsl_ps()))
    manifest = pipe.manifest()
    name = os.path.join(out_dir, '%016x' % key)
    for stage, _, hlsl in sources:
        open('%s.%s57.hlsl' % (name, stage), 'w').write(hlsl)
    if os.environ.get('X2W_FXC') != '':
        compiler = find_dxc()
        if compiler is None:
            raise Unsupported('dxc not found (X2W_DXC=%s): a mesh stage needs shader model 6.5' % os.environ.get('X2W_DXC', '<unset>'))
        import subprocess
        # Publish bytecode only after all requested stages compile successfully.
        with tempfile.TemporaryDirectory(prefix='.dxc-', dir=out_dir) as staging:
            for stage, profile, _ in sources:
                bytecode = os.path.join(staging, stage + '.cso')
                r = subprocess.run([compiler, '-Gis', '-T', profile, '-E', 'main', '-Fo', bytecode, '%s.%s57.hlsl' % (name, stage)],
                                   capture_output=True, text=True)
                if r.returncode or not has_complete_mesh57_bytecode(bytecode):
                    # DXC names the source path; keep only the diagnostic after it.
                    lines = [re.sub(r'^.*?57\.hlsl(:\d+:\d+)?:\s*', '', line) for line in r.stderr.strip().splitlines()]
                    raise Unsupported('dxc %s: %s' % (stage, (lines[0] if lines else 'missing or incomplete bytecode')[:500]))
            for stage, _, _ in sources:
                os.replace(os.path.join(staging, stage + '.cso'), '%s.%s57.cso' % (name, stage))
    temporary = '%s.ms57.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.ms57.txt')
    with suppress(FileNotFoundError):
        os.remove(name + '.fail')
    return 'ok'


def translate_request(out_dir, key):
    """Translate a base shader request, an RTV variant referencing its source blobs, or a slot 67 compute request."""
    if is_compute_request(out_dir, key):
        return translate_compute(out_dir, key)
    if os.path.isfile(os.path.join(out_dir, 'req_%016x.m57' % key)):
        try:
            return translate_mesh57(out_dir, key)
        except (Unsupported, OSError, UnicodeError, struct.error) as reason:
            open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
            return 'unsupported: %s' % reason
    name = os.path.join(out_dir, 'req_%016x' % key)
    request = ''
    formats = ()
    try:
        base = key
        if os.path.isfile(name + '.rtv'):
            with open(name + '.rtv', encoding='ascii') as file:
                request = file.read(256)
            if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or fnv(request.encode('ascii')) != key:
                raise Unsupported('invalid RTV variant request')
            base, *formats = map(int, request.split()[1:])
            if base > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
                raise Unsupported('invalid RTV variant values')
            name = os.path.join(out_dir, 'req_%016x' % base)
        a, b, root = (open(name + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
        # Upstream: a stream pipeline's input layout (.ia) is part of its key (pipeline_key, draw.inl).
        layout = open(name + '.ia', 'rb').read() if os.path.isfile(name + '.ia') else b''
        if request and pipeline_key(a, b, root, layout) != base:
            raise Unsupported('RTV variant source key mismatch')
        fetch = attr_map = None
        ps_hw_inputs = read_ps_inputs(name + '.psin')
        vs_out_cntl = read_vs_out(name + '.vsout')
        dumped = dumped_state(name) if ps_hw_inputs is None or vs_out_cntl is None else (None, None)
        if ps_hw_inputs is None and dumped[0] is not None:
            ps_hw_inputs = (dumped[0], dumped[0])  # ADDR assumed equal to ENA (no ADDR field located)
        if ps_hw_inputs is None or len(ps_hw_inputs) == 2:
            mode = dumped_float_mode(name)  # the launch FLOAT_MODE, when the .psin does not carry it
            if mode is not None:
                ps_hw_inputs = tuple(ps_hw_inputs or (None, None)) + (mode,)
        if vs_out_cntl is None:
            vs_out_cntl = dumped[1]
        if a.startswith(b'DXBC') or b.startswith(b'DXBC'):
            # Slot 86 records complete containers in canonical (VS, PS) order.
            # Keep the container bytes in the cache key; translate only the bounded native code.
            import zlib
            from pixel_aot import container_code
            from resource_inventory import parts
            try:
                container, ps_container = a, b
                a, b = (container_code(parts(blob)[b'DXIL'], stage=stage)
                        for blob, stage in ((a, 1), (b, 0)))
                # Upstream: the stream's attribute pairing and the loads its vertex fetch call stands for.
                attr_map = stream_interface(container, ps_container)
                if starts_with_fetch_call(a):
                    fetch = stream_fetch(container, layout.decode('ascii') if layout else None)
            except (KeyError, ValueError, IndexError, struct.error, zlib.error) as error:
                raise Unsupported('invalid stream shader: %s' % error) from error
            if classify(a) != 'vs' or classify(b) != 'ps':
                raise Unsupported('stream stage and shader export disagree')
            kinds = {'vs': a, 'ps': b}
        else:
            kinds = {classify(a): a, classify(b): b}
    except (Unsupported, OSError, UnicodeError) as reason:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    if set(kinds) != {'vs', 'ps'}:
        open(os.path.join(out_dir, '%016x.fail' % key), 'w').write('stages %s\n' % sorted(kinds))
        return 'stages %s' % sorted(kinds)
    return translate_to(out_dir, key, kinds['vs'], kinds['ps'], root, rtv_formats=formats, rtv_request=request,
                        ps_hw_inputs=ps_hw_inputs, vs_out_cntl=vs_out_cntl, vs_fetch=fetch, ps_attr_map=attr_map)


VSREGS_PS_INPUT_ENA = 0x58
VSREGS_VS_OUT_CNTL = 0x98


def dumped_state(name):
    """(ENA or None, PA_CL_VS_OUT_CNTL or None) from <name>.vsregs, else from the per-state dumps
    <name>.<8 hex>.vsregs when they all agree. (None, None) when nothing is dumped."""
    paths = [name + '.vsregs'] if os.path.isfile(name + '.vsregs') else []
    if not paths:
        folder, base = os.path.split(name)
        try:
            paths = [os.path.join(folder, n) for n in os.listdir(folder or '.')
                     if re.fullmatch(re.escape(os.path.basename(base)) + r'\.[0-9a-f]{8}\.vsregs', n)]
        except OSError:
            paths = []
    enas, cntls = set(), set()
    for path in paths:
        with open(path, 'rb') as file:
            blob = file.read(0x400)
        enas.add(struct.unpack_from('<H', blob, VSREGS_PS_INPUT_ENA)[0] if len(blob) >= VSREGS_PS_INPUT_ENA + 2 else None)
        cntls.add(struct.unpack_from('<I', blob, VSREGS_VS_OUT_CNTL)[0] if len(blob) >= VSREGS_VS_OUT_CNTL + 4 else None)
    return (enas.pop() if len(enas) == 1 else None), (cntls.pop() if len(cntls) == 1 else None)


VSREGS_PS_RSRC1 = 0x4c


def dumped_float_mode(name):
    """The pixel stage's launch FLOAT_MODE from <name>.vsregs (or the agreeing per-state dumps), else None."""
    paths = [name + '.vsregs'] if os.path.isfile(name + '.vsregs') else []
    if not paths:
        folder, base = os.path.split(name)
        try:
            paths = [os.path.join(folder, n) for n in os.listdir(folder or '.')
                     if re.fullmatch(re.escape(os.path.basename(base)) + r'\.[0-9a-f]{8}\.vsregs', n)]
        except OSError:
            paths = []
    modes = set()
    for path in paths:
        with open(path, 'rb') as file:
            blob = file.read(0x400)
        modes.add(struct.unpack_from('<I', blob, VSREGS_PS_RSRC1)[0] >> 12 & 0xff if len(blob) >= VSREGS_PS_RSRC1 + 4
                  else None)
    return modes.pop() if len(modes) == 1 else None


def read_vs_out(path):
    """PA_CL_VS_OUT_CNTL from a request's optional .vsout file, 'pa_cl_vs_out_cntl 1 <hex8>', or None."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding='ascii') as file:
        text = file.read(64)
    m = re.fullmatch(r'pa_cl_vs_out_cntl 1 ([0-9a-f]{8})\n', text)
    if not m:
        raise Unsupported('invalid vertex output control file')
    return int(m.group(1), 16)


# PA_CL_VS_OUT_CNTL fields (Mesa src/amd/registers/gfx10.json): CLIP_DIST_ENA_0..7 [7:0], CULL_DIST_ENA_0..7 [15:8],
# USE_VTX_POINT_SIZE 16, _EDGE_FLAG 17, _RENDER_TARGET_INDX 18, _VIEWPORT_INDX 19, _KILL_FLAG 20,
# VS_OUT_MISC_VEC_ENA 21, VS_OUT_CCDIST0_VEC_ENA 22, VS_OUT_CCDIST1_VEC_ENA 23, VS_OUT_MISC_SIDE_BUS_ENA 24,
# USE_VTX_GS_CUT_FLAG 25, USE_VTX_SHD_OBJPRIM_ID 26, USE_VTX_LINE_WIDTH 27.
VS_OUT_CCDIST0 = 1 << 22
VS_OUT_SIDE_BUS = 1 << 24


def clip_vector_width(cntl, vals):
    """Channels of a pos1 export that are clip distances 0..n-1; refuses every other PA_CL_VS_OUT_CNTL use.

    Accepted only when pos1 is the first clip/cull distance vector (no misc vector, CCDIST0 on, CCDIST1 off,
    no per-vertex misc values), the written channels are x.. contiguously, each is a clip distance, and the
    cull enables are either none or the same channels. A distance that is both clip and cull distance is
    lowered as SV_ClipDistance alone: the cull removes a primitive only when every vertex is below 0, which the
    clip at 0 removes already. VS_OUT_MISC_SIDE_BUS_ENA (a transport choice) is allowed.
    """
    written = [i for i, v in enumerate(vals) if v != 'off']
    n = len(written)
    clip, cull = cntl & 0xff, cntl >> 8 & 0xff
    if (cntl & ~(0xffff | VS_OUT_CCDIST0 | VS_OUT_SIDE_BUS) or not cntl & VS_OUT_CCDIST0 or not n
            or written != list(range(n)) or clip != (1 << n) - 1 or cull not in (0, clip)):
        raise Unsupported('pos1 export with PA_CL_VS_OUT_CNTL %#010x (channels %s)' % (cntl, written))
    return n


def read_ps_inputs(path):
    """(ENA, ADDR[, FLOAT_MODE]) from a request's optional .psin file,
    'spi_ps_input 1 <ENA hex8> <ADDR hex8>[ <FLOAT_MODE hex8>]', or None. FLOAT_MODE is the pixel stage's launch
    MODE[7:0] (SPI_SHADER_PGM_RSRC1_PS bits 19:12; for a mesh pipeline u32 at state block A + 0x04c, bits 19:12);
    without it MODE 0 rounding is assumed (Stage.launch_mode).

    The runtime writes it beside the request (it is not part of the key); without one the pixel stage's SPI
    input layout is unknown and a read of an input VGPR is refused.
    """
    if not os.path.isfile(path):
        return None
    with open(path, encoding='ascii') as file:
        text = file.read(256)
    m = re.fullmatch(r'spi_ps_input 1 ([0-9a-f]{8}) ([0-9a-f]{8})( [0-9a-f]{8})?\n(?:vs_input 1 [0-3] [0-9a-f]{8}\n)?',
                     text)
    if not m:
        raise Unsupported('invalid SPI input file')
    if m.group(3) is None:
        return int(m.group(1), 16), int(m.group(2), 16)
    # optional third field: the pixel stage's launch FLOAT_MODE (SPI_SHADER_PGM_RSRC1_PS bits 19:12), 8 hex digits
    mode = int(m.group(3), 16)
    if mode > 0xff:
        raise Unsupported('invalid SPI input file')
    return int(m.group(1), 16), int(m.group(2), 16), mode


REFUSALS_CAVEAT = ('Counts after the first refusal are approximate. Skipped instructions leave later state '
                   'unmodelled and can cause follow-on errors. A pipeline-scope refusal ends the pipeline, '
                   'so Only counts at pipeline scope reflect a lower bound on the reasons reached.')


def request_keys(out_dir, compute=False):
    """Graphics request keys (ready marker .a or .rtv), or with compute, slot 67 compute keys (.c67)."""
    pattern = r'req_([0-9a-f]{16})\.c67' if compute else r'req_([0-9a-f]{16})\.(?:a|rtv)'
    for name in sorted(os.listdir(out_dir)):
        match = re.fullmatch(pattern, name)
        if match:
            yield int(match.group(1), 16)


def is_compute_request(out_dir, key):
    return os.path.isfile(os.path.join(out_dir, 'req_%016x.c67' % key))


def normalize_refusal(reason):
    # Opaque decoder/container details can contain input bytes or disassembly.
    for label in ('undecodable word', 'invalid stream shader', 'unverified vertex fetch record',
                  'unreadable shader request', 'invalid binary shader request'):
        if reason.startswith(label):
            return label
    reason = re.sub(r'\b([sv])\[\d+(?::(\d+))?\]', lambda m: m[1] + ('[N:N]' if m[2] else '[N]'), reason)
    reason = re.sub(r'\b([sv])\d+\b', r'\1N', reason)
    reason = re.sub(r'\b(pos|param|mrt)\d+\b', r'\1N', reason)
    reason = re.sub(r'\blabel_[0-9a-f]+\b', 'label_N', reason, flags=re.IGNORECASE)
    reason = re.sub(r'\bword_0x[0-9a-f]+\b', 'word_N', reason, flags=re.IGNORECASE)
    # Preserve cause numbers. Redact hex operands and offsets, and decimal operands
    # in instruction operand lists, symbolic values, and named operand fields.
    number = r'[+-]?(?:0x[0-9a-f]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)'
    reason = re.sub(r'(?<!\w)[+-]?0x[0-9a-f]+\b', 'N', reason, flags=re.IGNORECASE)
    reason = re.sub(r'(^[a-z]\w*_\w+\s+)(' + number + r')(?=\s*(?:,|$))', r'\1N', reason)
    reason = re.sub(r'([,\[(]\s*|\b(?:offset|operand|destination|source|amount|selector|vaddr|soffset|register|attributes|dmask|op_sel|offset0|offset1)[: ]+)([\'"]?)'
                    + number + r'(?![\w.-])', r'\1\2N', reason, flags=re.IGNORECASE)
    return reason


def handler_error(op, error, refusals):
    label = 'internal error' if refusals else 'translator crash'
    return '%s in %s: %s' % (label, op, type(error).__name__)


def request_refusals(out_dir, key, pair_info=None):
    """Collect reached refusal sites for one request without publishing a translation."""
    refusals = []
    try:
        kinds, root, formats, _, fetch, decoded, attr_map = load_request(out_dir, key, refusals, pair_info)
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
        return refusals
    except struct.error:
        refusals.append(('invalid binary shader request', 'pipeline'))
        return refusals
    except (OSError, UnicodeError):
        # File errors can include the cache path and request key.
        refusals.append(('unreadable shader request', 'pipeline'))
        return refusals
    except Exception as error:
        refusals.append((handler_error('request loading', error, refusals), 'pipeline'))
        return refusals

    op = 'Pipeline'
    try:
        if set(kinds) != {'vs', 'ps'}:
            raise Unsupported('stages %s' % sorted(kinds))
        pipe = Pipeline(kinds['vs'], kinds['ps'], root, rtv_formats=formats, vs_fetch=fetch,
                        refusals=refusals, decoded=decoded, ps_attr_map=attr_map)
        op = 'hlsl_vs'
        pipe.hlsl_vs()
        op = 'hlsl_ps'
        pipe.hlsl_ps()
        op = 'manifest'
        pipe.manifest()
    except Unsupported as reason:
        refusals.append((str(reason), 'pipeline'))
    except Exception as error:
        refusals.append((handler_error(op, error, refusals), 'pipeline'))
    return refusals


def refusals_report(out_dir):
    histogram, rows, pairs = {}, {}, {}
    for key in dict.fromkeys(request_keys(out_dir)):
        pair_info = []
        sites = [(normalize_refusal(reason), scope) for reason, scope in request_refusals(out_dir, key, pair_info)]
        count = len(sites)
        histogram[count] = histogram.get(count, 0) + 1
        distinct = len({reason for reason, _ in sites})
        for reason, scope in set(sites):
            row = rows.setdefault((reason, scope), dict(reason=reason, scope=scope, containing=0, first=0, only=0))
            row['containing'] += 1
            row['first'] += (reason, scope) == sites[0]
            row['only'] += distinct == 1
        if pair_info:
            pair, root = pair_info[0]
            pairs.setdefault(pair, {}).setdefault(root, set()).update(sites)

    pair_rows = {}
    for roots in pairs.values():
        every = set.intersection(*roots.values())
        for reason, scope in set.union(*roots.values()):
            row = pair_rows.setdefault((reason, scope), dict(reason=reason, scope=scope,
                                                            under_every_root=0, under_some_roots=0))
            column = 'under_every_root' if (reason, scope) in every else 'under_some_roots'
            row[column] += 1

    return dict(per_request=dict(examined=sum(histogram.values()), no_refusal=histogram.get(0, 0),
                                 refusal_sites_per_request=[dict(sites=count, requests=histogram[count])
                                                            for count in sorted(histogram)],
                                 reasons=sorted(rows.values(), key=lambda row: (-row['containing'], row['reason'], row['scope']))),
                per_code_pair=dict(examined=len(pairs),
                                   reasons=sorted(pair_rows.values(), key=lambda row: (row['reason'], row['scope']))),
                compute=compute_refusals_report(out_dir),
                caveat=REFUSALS_CAVEAT)


def compute_refusals_report(out_dir):
    """Slot 67 compute requests, kept apart so the graphics statistics above do not move."""
    examined, clean, rows = 0, 0, {}
    for key in request_keys(out_dir, compute=True):
        examined += 1
        sites = [normalize_refusal(reason) for reason, _ in compute_refusals(out_dir, key)]
        clean += not sites
        for reason in set(sites):
            row = rows.setdefault(reason, dict(reason=reason, containing=0, first=0))
            row['containing'] += 1
            row['first'] += reason == sites[0]
    return dict(examined=examined, no_refusal=clean,
                reasons=sorted(rows.values(), key=lambda row: (-row['containing'], row['reason'])))


def refusals_main(argv):
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only diagnostic for all cached requests.',
                                     epilog=REFUSALS_CAVEAT, allow_abbrev=False)
    parser.add_argument('--refusals', required=True, metavar='CACHE_DIR')
    parser.add_argument('--json', action='store_true', help='emit the report as JSON')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.refusals):
        parser.error('CACHE_DIR must be an existing directory')
    report = refusals_report(args.refusals)
    if args.json:
        print(json.dumps(report))
        return
    requests = report['per_request']
    print('Per request key')
    print('Requests examined: %d' % requests['examined'])
    print('No refusal recorded (not compiled): %d' % requests['no_refusal'])
    print('Refusal sites per request (sites, requests):')
    for bucket in requests['refusal_sites_per_request']:
        print('%d\t%d' % (bucket['sites'], bucket['requests']))
    print('Containing\tFirst\tOnly\tScope\tReason')
    for row in requests['reasons']:
        print('%(containing)d\t%(first)d\t%(only)d\t%(scope)s\t%(reason)s' % row)
    pairs = report['per_code_pair']
    print('Per code pair (excluding RTV variants)')
    print('Code pairs examined: %d' % pairs['examined'])
    print('Under every root\tUnder some roots\tScope\tReason')
    for row in pairs['reasons']:
        print('%(under_every_root)d\t%(under_some_roots)d\t%(scope)s\t%(reason)s' % row)
    compute = report['compute']
    print('Compute requests (device slot 67)')
    print('Compute requests examined: %d' % compute['examined'])
    print('No refusal recorded (not compiled): %d' % compute['no_refusal'])
    print('Containing\tFirst\tReason')
    for row in compute['reasons']:
        print('%(containing)d\t%(first)d\t%(reason)s' % row)
    print(REFUSALS_CAVEAT)


def main():
    if any(arg == '--refusals' or arg.startswith('--refusals=') for arg in sys.argv[1:]):
        return refusals_main(sys.argv[1:])
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    if sys.argv[1] == '--capture':
        # --capture <dir> <capture.bin>: write request files for every pipeline the capture binds
        import capture
        cap = capture.Capture(sys.argv[3])
        done = set()
        for c in cap.calls:
            w = c['args'][0]
            if c['slot'] != 25 or w in done:
                continue
            done.add(w)
            p = cap.psos.get(w)
            seen = {w}
            while p and p['path'] == 55 and p['base']:
                if p['base'] in seen:
                    print('pso %x: wrapper chain loops back to %x' % (w & 0xffffff, p['base'] & 0xffffff))
                    p = None
                    break
                seen.add(p['base'])
                p = cap.psos.get(p['base'])
            if not p or p['path'] != 57:
                continue
            if len(p['desc']) < 96:
                print('pso %x: pipeline desc is %d bytes, need 96' % (w & 0xffffff, len(p['desc'])))
                continue
            q = struct.unpack_from('<12Q', p['desc'], 0)
            a, b = cap.memory.read(q[5], q[6]), cap.memory.read(q[9], q[10])
            root = cap.roots.get(q[0], b'')
            if not a or not b or len(a) != q[6] or len(b) != q[10]:
                print('pso %x: code not captured' % (w & 0xffffff))
                continue
            key = pipeline_key(a, b, root)
            for ext, blob in (('.a', a), ('.b', b), ('.root', root)):
                open(os.path.join(out_dir, 'req_%016x%s' % (key, ext)), 'wb').write(blob)
            print('pso %x -> request %016x' % (w & 0xffffff, key))
    if sys.argv[1] == '--watch':
        # --watch <dir> [seconds]: translate requests as a running title writes them, until stopped
        interval = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        # The first pass retries earlier failures once (the runtime never rewrites a request, so a
        # translator fix would otherwise not reach them); later passes skip failures already written.
        skip_failed = False
        while True:
            if watch_pass(out_dir, skip_failed):
                skip_failed = True
            time.sleep(interval)
    translate_pending(out_dir)




def watch_pass(out_dir, skip_failed):
    """One --watch pass. A worker killed from outside (out of memory) breaks the pool; the pass is
    reported and its requests are retried on the next pass instead of ending the watcher."""
    from concurrent.futures.process import BrokenProcessPool
    try:
        translate_pending(out_dir, skip_failed=skip_failed)
        return True
    except BrokenProcessPool as error:
        print('translation pass interrupted, retrying: %s' % error, file=sys.stderr, flush=True)
        return False


def fail_reason(reason):
    """The .fail text of a refusal. An operand alone ("operand '-v30'") does not say which lowering refused it, so
    the instruction is added when the translation loop recorded it. The record is an attribute on the exception, so
    a refusal raised outside that loop, or rebuilt as a new Unsupported on the way here, gives the message alone."""
    text = str(reason)
    if text.startswith('operand ') and getattr(reason, 'instruction', None):
        text += ' in %s' % reason.instruction
    return text


def raise_site(error):
    """The traceback frame that raised error, skipping standard-library frames so an error raised inside a library
    call (json, struct helpers, ...) names the code that made the call. None without a traceback."""
    frames = traceback.extract_tb(error.__traceback__)
    roots = {os.path.normcase(os.path.abspath(sysconfig.get_paths()[name])) + os.sep
             for name in ('stdlib', 'platstdlib')}
    for frame in reversed(frames):
        if not os.path.normcase(os.path.abspath(frame.filename)).startswith(tuple(roots)):
            return frame
    return frames[-1] if frames else None


def translate_one(out_dir, key):
    """translate_request, with an unexpected translator error recorded as <key>.fail instead of raised."""
    try:
        return translate_request(out_dir, key)
    except Exception as error:  # noqa: BLE001 - one bad request must not stop a scan or the watcher
        reason = 'translator error: %s: %s' % (type(error).__name__, error)
        # Where it was raised (function, file name, line), so the .fail points at the fault without the program
        # that hit it. The line number moves with unrelated edits, so this text is for reading, not a stable key.
        frame = raise_site(error)
        if frame:
            reason += ' (in %s, %s line %d)' % (frame.name, os.path.basename(frame.filename), frame.lineno)
        with open(os.path.join(out_dir, '%016x.fail' % key), 'w') as fail:
            fail.write(reason + '\n')
        return reason


def translation_jobs():
    try:
        jobs = int(os.environ.get('X2W_TRANSLATE_JOBS') or 0)
    except ValueError:
        jobs = 0
    jobs = jobs if jobs > 0 else os.cpu_count() or 1
    return min(jobs, 61) if os.name == 'nt' else jobs  # ProcessPoolExecutor refuses more than 61 on Windows


def translate_pending(out_dir, skip_failed=False, jobs=None):
    """Translate every request that has no result yet. The runtime writes the ready markers last.

    Requests are independent (each writes only its own <key>.* files), so several run in parallel
    processes; X2W_TRANSLATE_JOBS (default: the processor count, at most 61 on Windows) sets how many.
    """
    keys = pending_keys(out_dir, skip_failed)
    jobs = min(jobs or translation_jobs(), len(keys))
    if jobs < 2:
        for key in keys:
            print('%016x: %s' % (key, translate_one(out_dir, key)), flush=True)
        return
    from concurrent.futures import ProcessPoolExecutor
    chunk = max(1, len(keys) // (jobs * 4))  # small batches keep every worker busy
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for key, result in zip(keys, pool.map(translate_one, [out_dir] * len(keys), keys, chunksize=chunk)):
            print('%016x: %s' % (key, result), flush=True)


def pending_keys(out_dir, skip_failed):
    keys = []
    manifest_only = os.environ.get('X2W_FXC') == ''
    for name in sorted(os.listdir(out_dir)):
        m = re.fullmatch(r'req_([0-9a-f]{16})\.(a|rtv|c67|m57)', name)
        if not m:
            continue
        key = int(m.group(1), 16)
        result = os.path.join(out_dir, '%016x' % key)
        mesh = m.group(2) == 'm57'
        stages = ('.cs.cso',) if m.group(2) == 'c67' else ('.ms57.cso', '.ps57.cso') if mesh else ('.vs.cso', '.ps.cso')
        ready_bytecode = has_complete_mesh57_bytecode if mesh else has_bytecode
        manifest = result + ('.ms57.txt' if mesh else '.txt')
        # A published manifest with bytecode (or manifest-only mode) is done; otherwise resume (#2008).
        if os.path.isfile(manifest):
            if manifest_only or all(ready_bytecode(result + s) for s in stages):
                continue
            if mesh and has_complete_mesh57_bytecode(result + '.ms57.cso'):
                with suppress(OSError), open(manifest, 'rb') as published:
                    if b'ps none' in published.read().splitlines():
                        continue
        fail = os.path.join(out_dir, '%016x.fail' % key)
        # A repeated pass retries a failure only once its request is rewritten.
        if skip_failed and os.path.exists(fail) and os.path.getmtime(fail) >= os.path.getmtime(os.path.join(out_dir, name)):
            continue
        keys.append(key)
    return keys

if __name__ == '__main__':
    main()
