"""`v_cvt_off_f32_i4` (VOP1 0x0e and VOP3A 0x18e) in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation, decoder, evaluator, provenance, uniformity, refusal, and
                                       unlowered-control checks, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pairs with FXC ps_5_1 (when given) and DXC ps_6_0,
                                       draw all 16 nibble values, upper-bit garbage, and e64 output modifiers on the
                                       adapter, and require mutated lowerings to fail with their mutated words

RDNA 2 ISA semantics:
  `v_cvt_off_f32_i4` reads bits `[3:0]` of SRC0 as a signed 4-bit two's-complement integer in `[-8, +7]` and returns
  `float(i4) * 0.0625` (exact steps of `1/16`, `[-0.5, +0.4375]`). As a float-result conversion in `VEC_U_TO_F`, it
  accepts output modifiers (`clamp`, `mul:2`, `mul:4`, `div:2`, applied with scaling before `clamp`), reads only bits
  `[3:0]` (`low16_only=True`), and preserves uniformity (`UNIFORM_VALU`). Source `abs`/`neg`, `op_sel`, wrong operand
  counts, and SDWA forms stay refused.

Every instruction tested here is constructed as synthetic gfx1013 machine words from the layouts in
`tools/disasm_rdna2.py` and decoded via `gfx2hlsl.decode()` (no game data).
"""
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import POST_WORDS, PRE_WORDS, vs_words, words_text  # noqa: E402
from startup_image_3d_check import VS_WORDS as FULLSCREEN_VS_WORDS, root_blob as table_root_blob  # noqa: E402

MASK = 0xffffffff
VOP1_CVT_OFF_F32_I4 = 0x0e
VOP1_CVT_F32_UBYTE2 = 0x13
VOP1_E64_BASE = 0x180
VOP2_ADD_NC_U32 = 0x25
VOP1_READFIRSTLANE_B32 = 0x02
VOPC_CMPX_EQ_U32 = 0xd2

OMOD = {None: 0, '': 0, 'mul:2': 1, 'mul:4': 2, 'div:2': 3}
OMOD_SCALE = {None: 1.0, '': 1.0, 'mul:2': 2.0, 'mul:4': 4.0, 'div:2': 0.5}
INLINE_INT = {n: 128 + n for n in range(65)}
INLINE_INT.update({-n: 192 + n for n in range(1, 17)})
LITERAL = 255
S_LOAD_DESC_X8 = bytes.fromhex('04040cf4000000fa')   # s_load_dwordx8 s[16:23], s[8:9], 0x0
S_WAITCNT_LGKMCNT0 = bytes.fromhex('7fc08cbf')         # s_waitcnt lgkmcnt(0)
S_WAITCNT_VMCNT0 = bytes.fromhex('703f8cbf')           # s_waitcnt vmcnt(0)
S_ENDPGM = 0xbf810000
EXP_MRT0_V12_V15 = bytes.fromhex('0f1800f80c0d0e0f')


def V(n):
    return 256 + n


def S(n):
    return n


# ---- Synthetic gfx1013 encoders -------------------------------------------------------------------------------------

def encode_vop1_e32(op, vdst, src0, *, literal=None):
    w0 = 0x7e000000 | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0) + (b'' if literal is None else struct.pack('<I', literal & MASK))


def encode_vop1_e64(op, vdst, src0, *, literal=None, clamp=0, omod=None, opsel=(0, 0, 0, 0), abs0=0, neg0=0):
    w0 = ((0b110101 << 26) | ((VOP1_E64_BASE + op) << 16) | ((clamp & 1) << 15)
          | ((abs0 & 1) << 8) | (vdst & 0xff))
    for i, bit in enumerate(opsel):
        w0 |= (bit & 1) << (11 + i)
    w1 = (src0 & 0x1ff) | ((OMOD[omod] & 0x3) << 27) | ((neg0 & 1) << 29)
    return struct.pack('<II', w0, w1) + (b'' if literal is None else struct.pack('<I', literal & MASK))


def encode_vop1_sdwa(op, vdst, src0, *, dst_sel=6, dst_unused=0, src0_sel=1, clamp=0, omod=0,
                     sext=0, neg=0, abs_=0, sgpr=0):
    w0 = (0b0111111 << 25) | ((vdst & 0xff) << 17) | ((op & 0xff) << 9) | 249
    w1 = ((src0 & 0xff) | ((dst_sel & 0x7) << 8) | ((dst_unused & 0x3) << 11)
          | ((clamp & 1) << 13) | ((omod & 0x3) << 14) | ((src0_sel & 0x7) << 16)
          | ((sext & 1) << 19) | ((neg & 1) << 20) | ((abs_ & 1) << 21) | ((sgpr & 1) << 23))
    return struct.pack('<II', w0, w1)


def encode_vop2_e32(op, vdst, src0, vsrc1, *, literal=None):
    w0 = ((op & 0x3f) << 25) | ((vdst & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0) + (b'' if literal is None else struct.pack('<I', literal & MASK))


def encode_vopc_e32(op, src0, vsrc1):
    w0 = (0b0111110 << 25) | ((op & 0xff) << 17) | ((vsrc1 & 0xff) << 9) | (src0 & 0x1ff)
    return struct.pack('<I', w0)


def encode_mimg_load_d16(vdata, vaddr, srsrc, dmask=0x1):
    w0 = (0xf0000000 | (0x00 << 18) | (1 << 3) | ((dmask & 0xf) << 8))
    w1 = (vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc >> 2) & 0x1f) << 16) | (1 << 31)
    return struct.pack('<II', w0, w1)


def encode_smem_buffer_load(sdata, sbase, offset_imm=0):
    w0 = (0b111101 << 26) | (S_BUFFER_LOAD_DWORD << 18) | (1 << 16) | ((sdata & 0x7f) << 6) | ((sbase >> 1) & 0x3f)
    w1 = offset_imm & 0x1fffff
    return struct.pack('<II', w0, w1)


def encode_s_waitcnt_lgkmcnt0():
    return struct.pack('<I', S_WAITCNT | 0x3f7f)


def encode_endpgm():
    return struct.pack('<I', S_ENDPGM)


def encode_exp_mrt0(v0, v1, v2, v3):
    return struct.pack('<II', 0xf800180f, (v0 & 0xff) | ((v1 & 0xff) << 8) | ((v2 & 0xff) << 16) | ((v3 & 0xff) << 24))


# ---- Independent RDNA 2 ISA reference & HLSL expression evaluator ---------------------------------------------------

def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def bits_to_f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & MASK))[0]


def isa_cvt_off_f32_i4(word, *, omod='', clamp=False):
    """Independent RDNA 2 ISA reference: signed 4-bit nibble bits[3:0] * 0.0625, then omod, then clamp."""
    nibble = word & 0xf
    i4 = nibble - 16 if nibble >= 8 else nibble
    val = i4 * 0.0625
    if omod:
        val *= OMOD_SCALE[omod]
    if clamp:
        if val <= 0.0:
            val = 0.0
        elif val >= 1.0:
            val = 1.0
    return f32_bits(val)


def wrong_unsigned_nibble(word, *, omod='', clamp=False):
    """Mutated reference: treats bits[3:0] as unsigned 0..15 instead of signed -8..+7."""
    val = (word & 0xf) * 0.0625
    if omod:
        val *= OMOD_SCALE[omod]
    if clamp:
        val = min(max(val, 0.0), 1.0)
    return f32_bits(val)


def wrong_5bit_nibble(word, *, omod='', clamp=False):
    """Mutated reference: sign-extends 5 bits (<< 27 >> 27) instead of 4 bits (<< 28 >> 28)."""
    low5 = word & 0x1f
    i5 = low5 - 32 if low5 >= 16 else low5
    val = i5 * 0.0625
    if omod:
        val *= OMOD_SCALE[omod]
    if clamp:
        val = min(max(val, 0.0), 1.0)
    return f32_bits(val)


class U32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & MASK)

    def __lshift__(self, n):
        return U32((int(self) << (int(n) & 31)) & MASK)

    def __rshift__(self, n):
        return U32((int(self) & MASK) >> (int(n) & 31))

    def __and__(self, other):
        return U32(int(self) & int(other))

    def __rand__(self, other):
        return U32(int(other) & int(self))


class I32(int):
    def __new__(cls, value):
        u = int(value) & MASK
        return super().__new__(cls, u - 0x100000000 if u & 0x80000000 else u)

    def __rshift__(self, n):
        return I32(int(self) >> (int(n) & 31))


class F32:
    def __init__(self, bits):
        self.bits = int(bits) & MASK

    @classmethod
    def from_float(cls, value):
        return cls(f32_bits(value))

    def to_float(self):
        return bits_to_f32(self.bits)

    def __mul__(self, factor):
        return F32.from_float(self.to_float() * float(factor))


def hlsl_saturate(val):
    f = val.to_float()
    if f <= 0.0:
        return F32(0)
    if f >= 1.0:
        return F32(0x3f800000)
    return val


HLSL_ENV = {
    'U32': U32,
    'uint': U32,
    'int': I32,
    'float': lambda x: F32.from_float(int(x)),
    'saturate': hlsl_saturate,
    'U': lambda f: U32(f.bits),
}


def eval_hlsl_expr(expr, regs):
    py_expr = re.sub(r'\b(0x[0-9a-fA-F]+|\d+)u\b', r'U32(\1)', expr)
    return int(eval(py_expr, {'__builtins__': {}}, {**HLSL_ENV, **{k: U32(v) for k, v in regs.items()}}))


# ---- Test harness helpers -------------------------------------------------------------------------------------------

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}", flush=True)
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail, flush=True)


def make_pipeline(ps_body_bytes, *, use_table_root=False):
    """Build a full G.Pipeline from synthetic gfx1013 bytes (calls G.decode() and G.Stage)."""
    if use_table_root:
        return G.Pipeline(FULLSCREEN_VS_WORDS, ps_body_bytes, table_root_blob())
    return G.Pipeline(vs_words([(0, 0, 0)] * 4), ps_body_bytes, root_blob())


def decode_and_one(code, *, sym=None, low16_v=()):
    """Decode bytes with G.decode() and execute them on a straight-line Stage."""
    ins = G.decode(code)
    stage = G.Stage.__new__(G.Stage)
    stage.kind = 'ps'
    stage.lines = []
    stage.sym = dict(sym or {})
    stage.used_s, stage.used_v, stage.uniform_v = set(), set(), set()
    stage.varying_s = set()
    stage.scc_uniform = True
    stage.uniform_expr = {}
    stage.past_entry = False
    stage.has_branches = False
    stage.exec_known = True
    stage.masks = set()
    stage.exports = {}
    stage.low16_only_v = set(low16_v)
    stage.pinned_low16_v = set()
    stage.whole_read_v = set()
    stage.byte_float_v = set()
    stage.bad_sgprs = set()
    stage.defined_uniform_s = None
    stage.since_join = None
    pipe = type('Pipe', (), {'resources': {}, 'notes': set(), 'modified': set(), 'sampler_heap': False})()
    stage.pipe = pipe
    for _off, _sz, name, asm in ins:
        stage.one(name, asm)
    return ins, stage


def is_refused_bytes(code, expected_substr='', *, via_pipeline=False):
    try:
        if via_pipeline:
            G.Pipeline(vs_words([(0, 0, 0)] * 4), code, root_blob())
        else:
            decode_and_one(code)
    except G.Unsupported as err:
        return expected_substr in str(err)
    return False


def is_refused_asm(asm, expected_substr=''):
    stage = G.Stage.__new__(G.Stage)
    stage.lines, stage.sym = [], {}
    stage.used_s, stage.used_v, stage.uniform_v = set(), set(), set()
    stage.exec_known = True
    try:
        stage.one(asm.split(' ', 1)[0], asm)
    except G.Unsupported as err:
        return expected_substr in str(err)
    return False


# ---- Pure-Python controls -------------------------------------------------------------------------------------------

def check_decode_and_lowering():
    # 1. VOP1 _e32 and VOP3A _e64 decoding via G.decode()
    decode_vectors = (
        (encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, V(2)), 'v_cvt_off_f32_i4_e32 v4, v2'),
        (encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, S(2)), 'v_cvt_off_f32_i4_e32 v4, s2'),
        (encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, INLINE_INT[3]), 'v_cvt_off_f32_i4_e32 v4, 3'),
        (encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0x12345678),
         'v_cvt_off_f32_i4_e32 v4, 0x12345678'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2)), 'v_cvt_off_f32_i4_e64 v4, v2'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), clamp=1), 'v_cvt_off_f32_i4_e64 v4, v2 clamp'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), omod='mul:2'), 'v_cvt_off_f32_i4_e64 v4, v2 mul:2'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), omod='mul:4'), 'v_cvt_off_f32_i4_e64 v4, v2 mul:4'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), omod='div:2'), 'v_cvt_off_f32_i4_e64 v4, v2 div:2'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), clamp=1, omod='mul:4'),
         'v_cvt_off_f32_i4_e64 v4, v2 clamp mul:4'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, S(2), omod='div:2'), 'v_cvt_off_f32_i4_e64 v4, s2 div:2'),
        (encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0xfffffff8),
         'v_cvt_off_f32_i4_e64 v4, 0xfffffff8'),
    )
    for code, expected_asm in decode_vectors:
        ins, st = decode_and_one(code)
        check(f'decode+lower {expected_asm}',
              len(ins) == 1 and ins[0][3] == expected_asm and len(st.lines) == 1,
              f'got ins={ins!r} lines={st.lines!r}')

    # 2. All 16 nibble values 0..15 + upper-bit garbage across e32 and e64 * (omod x clamp)
    garbage_prefixes = (0x00000000, 0x12345670, 0xfffffff0, 0x80000000, 0xdeadbee0)
    samples = [prefix | nib for prefix in garbage_prefixes for nib in range(16)]
    samples.extend((0x12345678, 0xfffffff8))

    eval_count = 0
    eval_mismatches = []
    for enc_name, code in (
        ('e32', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, V(2))),
        ('e64', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2))),
    ):
        _, st = decode_and_one(code)
        expr = st.lines[0][len('v4 = '):-1]
        for word in samples:
            eval_count += 1
            got = eval_hlsl_expr(expr, {'v2': word})
            want = isa_cvt_off_f32_i4(word)
            if got != want and len(eval_mismatches) < 3:
                eval_mismatches.append((enc_name, hex(word), hex(got), hex(want)))

    for omod in ('', 'mul:2', 'mul:4', 'div:2'):
        for clamp in (0, 1):
            code = encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), omod=omod, clamp=clamp)
            _, st = decode_and_one(code)
            expr = st.lines[0][len('v4 = '):-1]
            for word in samples:
                eval_count += 1
                got = eval_hlsl_expr(expr, {'v2': word})
                want = isa_cvt_off_f32_i4(word, omod=omod, clamp=bool(clamp))
                if got != want and len(eval_mismatches) < 3:
                    eval_mismatches.append((omod, clamp, hex(word), hex(got), hex(want)))

    check(f'all {eval_count} nibble/garbage/modifier evaluations match RDNA2 ISA',
          not eval_mismatches, repr(eval_mismatches))

    # 3. Verify hardware order: omod before clamp (5 -> 0.3125 * 4 = 1.25 -> saturate = 1.0, not 1.25;
    #    8 -> -0.5 * 2 = -1.0 -> saturate = 0.0)
    _, st_cm4 = decode_and_one(encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), omod='mul:4', clamp=1))
    expr_cm4 = st_cm4.lines[0][len('v4 = '):-1]
    check('clamp applies after mul:4 on nibble 5 (0.3125 * 4 -> 1.0)',
          eval_hlsl_expr(expr_cm4, {'v2': 0x12345675}) == f32_bits(1.0), expr_cm4)
    check('clamp clamps negative nibble 8 (-0.5 * 4 -> 0.0)',
          eval_hlsl_expr(expr_cm4, {'v2': 0xfffffff8}) == f32_bits(0.0), expr_cm4)

    # 4. Literal and inline constant sources via G.decode()
    for desc, code, want_val in (
        ('e32 literal 0x12345678', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0x12345678), -0.5),
        ('e32 literal 0xfffffff8', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0xfffffff8), -0.5),
        ('e64 literal 0x12345678', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0x12345678), -0.5),
        ('e64 literal 0xfffffff8', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, LITERAL, literal=0xfffffff8), -0.5),
        ('e32 inline 7', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, INLINE_INT[7]), 0.4375),
        ('e32 inline -1', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, INLINE_INT[-1]), -0.0625),
        ('e64 inline -8', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, INLINE_INT[-8]), -0.5),
    ):
        _, st = decode_and_one(code)
        expr = st.lines[0][len('v4 = '):-1]
        got = eval_hlsl_expr(expr, {})
        check(f'constant source {desc}', got == f32_bits(want_val), f'{expr} -> {hex(got)}')


def check_low16_provenance():
    # Build full synthetic pixel shaders starting with an odd-dmask d16 image_load into v4,
    # making v4 low16_only_v via real decoded MIMG words through G.Pipeline / G.decode().
    load_prefix = (S_LOAD_DESC_X8 + S_WAITCNT_LGKMCNT0
                   + encode_vop1_e32(0x01, 0, INLINE_INT[0])
                   + encode_vop1_e32(0x01, 1, INLINE_INT[0])
                   + encode_mimg_load_d16(vdata=4, vaddr=0, srsrc=16, dmask=0x1)
                   + S_WAITCNT_VMCNT0)
    exp_v5 = encode_exp_mrt0(5, 5, 5, 5) + encode_endpgm()

    # Accepted: v_cvt_off_f32_i4_e32 and _e64 reading low16_only v4 into v5
    for tag, cvt in (
        ('e32', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 5, V(4))),
        ('e64', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 5, V(4), omod='mul:2', clamp=1)),
    ):
        ps_bytes = load_prefix + cvt + exp_v5
        try:
            pipe = make_pipeline(ps_bytes, use_table_root=True)
            hlsl = pipe.hlsl_ps()
            ok = 4 in pipe.ps.low16_only_v and 5 not in pipe.ps.low16_only_v and '<< 28' in hlsl
            detail = ''
        except G.Unsupported as err:
            ok = False
            detail = str(err)
        check(f'low16_only accepted by v_cvt_off_f32_i4_{tag}', ok, detail)

    # Accepted: in-place overwrite v4 <- v_cvt_off_f32_i4(v4) clears low16_only on v4 so whole-word read succeeds
    ps_inplace = (load_prefix + encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, V(4))
                  + encode_vop2_e32(VOP2_ADD_NC_U32, 6, INLINE_INT[1], V(4))
                  + encode_exp_mrt0(6, 6, 6, 6) + encode_endpgm())
    try:
        pipe = make_pipeline(ps_inplace, use_table_root=True)
        ok = 4 not in pipe.ps.low16_only_v
        detail = ''
    except G.Unsupported as err:
        ok = False
        detail = str(err)
    check('in-place v_cvt_off_f32_i4 v4, v4 clears low16_only for subsequent whole-word read', ok, detail)

    # Negative control: reading v4 whole-word AFTER v_cvt_off_f32_i4 v5, v4 must still fail
    ps_leak = (load_prefix + encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 5, V(4))
               + encode_vop2_e32(VOP2_ADD_NC_U32, 6, INLINE_INT[1], V(4))
               + encode_exp_mrt0(6, 6, 6, 6) + encode_endpgm())
    try:
        make_pipeline(ps_leak, use_table_root=True)
    except G.Unsupported as err:
        ok = 'whole-word read of v4' in str(err)
        detail = str(err)
    else:
        ok = False
        detail = 'unexpectedly accepted'
    check('low16_only still catches whole-word read of v4 after v_cvt_off_f32_i4 v5, v4', ok, detail)

    # Negative control: ubyte2 reads bits[23:16] of v4 and must still be refused on odd d16 image_load
    ps_ubyte2 = load_prefix + encode_vop1_e32(VOP1_CVT_F32_UBYTE2, 5, V(4)) + exp_v5
    try:
        make_pipeline(ps_ubyte2, use_table_root=True)
    except G.Unsupported as err:
        ok = 'whole-word read of v4' in str(err)
        detail = str(err)
    else:
        ok = False
        detail = 'unexpectedly accepted'
    check('low16_only negative control: v_cvt_f32_ubyte2 v5, v4 refused', ok, detail)


def check_uniform_valu():
    # table_root_blob gives s9 = root constant 1 (uniform user-data scalar).
    # Translate real pixel programs via G.Pipeline -> G.decode() -> G.Stage with v_readfirstlane_b32.
    rfl = encode_vop1_e32(VOP1_READFIRSTLANE_B32, 10, V(7))
    copy_to_v5 = encode_vop1_e32(0x01, 5, S(10))
    tail = copy_to_v5 + encode_exp_mrt0(5, 5, 5, 5) + encode_endpgm()

    for tag, body in (
        ('e32 uniform s9', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 7, S(9)) + rfl),
        ('e64 uniform s9 mul:2 clamp',
         encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 7, S(9), omod='mul:2', clamp=1) + rfl),
        ('chained uniform v7 -> v8 -> readfirstlane',
         encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 7, S(9))
         + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 8, V(7), omod='div:2')
         + encode_vop1_e32(VOP1_READFIRSTLANE_B32, 10, V(8))),
    ):
        try:
            pipe = make_pipeline(body + tail, use_table_root=True)
            lines = [line for _, blk, _ in pipe.ps.blocks for line in blk]
            ok = any(line.startswith('s10 = v') for line in lines)
            detail = repr(lines)
        except G.Unsupported as err:
            ok = False
            detail = str(err)
        check(f'UNIFORM_VALU accepted ({tag})', ok, detail)

    # Negative control 1: varying VGPR v0 -> v7 -> v_readfirstlane_b32 s10, v7 must be refused
    ps_varying = encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 7, V(0)) + rfl + tail
    try:
        make_pipeline(ps_varying, use_table_root=True)
    except G.Unsupported as err:
        ok = 'v_readfirstlane_b32 requires a proven uniform source' in str(err)
        detail = str(err)
    else:
        ok = False
        detail = 'unexpectedly accepted'
    check('UNIFORM_VALU refuses varying VGPR source in v_readfirstlane_b32', ok, detail)

    # Negative control 2: unknown exec after v_cmpx_eq_u32 -> v_cvt_off_f32_i4 v7, s9 -> v_readfirstlane_b32
    ps_exec_unknown = (encode_vopc_e32(VOPC_CMPX_EQ_U32, INLINE_INT[0], 0)
                       + encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 7, S(9)) + rfl + tail)
    try:
        make_pipeline(ps_exec_unknown, use_table_root=True)
    except G.Unsupported as err:
        ok = 'v_readfirstlane_b32 requires a proven uniform source' in str(err)
        detail = str(err)
    else:
        ok = False
        detail = 'unexpectedly accepted'
    check('UNIFORM_VALU refuses v_readfirstlane_b32 when exec is unknown', ok, detail)


def check_dpp_broadcast():
    """The generic DPP16 `quad_perm:[k,k,k,k]` broadcast path accepts this VOP1 op, and only where it is sound.

    The merge base refused every `v_cvt_off_f32_i4` form because the op was not lowered at all, so it also
    refused the DPP form. The op is now in `VEC_U_TO_F`, and `Stage.dpp_broadcast` is op-agnostic: it proves
    whole-quad activity, reads the named lane of the quad with `QuadReadLaneAt`, and then lowers the base
    operation on that value (for a unary op the broadcast happens before the conversion, which is the ISA
    order). Accepting the broadcast is therefore intended rather than incidental, and the rows below pin the
    acceptance, the lane it reads, the shader-model bump, and the four refusals that keep the rest out.
    """
    def probe(**state):
        stage = G.Stage.__new__(G.Stage)
        stage.kind = 'ps'
        stage.lines, stage.sym = [], {}
        stage.used_s, stage.used_v, stage.uniform_v = set(), set(), set()
        stage.varying_s = set()
        stage.scc_uniform = True
        stage.uniform_expr = {}
        stage.past_entry = False
        stage.has_branches = False
        stage.exec_known = True
        stage.masks = set()
        stage.exports = {}
        stage.low16_only_v = set()
        stage.pinned_low16_v = set()
        stage.whole_read_v = set()
        stage.byte_float_v = set()
        stage.bad_sgprs = set()
        stage.defined_uniform_s = None
        stage.since_join = None
        stage.wqm_exec = True
        stage.needs_sm6 = False
        stage.pipe = type('Pipe', (), {'resources': {}, 'notes': set(), 'modified': set(),
                                       'sampler_heap': False})()
        for key, value in state.items():
            setattr(stage, key, value)
        return stage

    def translate(asm, **state):
        stage = probe(**state)
        try:
            stage.one(asm.split(' ', 1)[0], asm)
        except G.Unsupported as error:
            return None, str(error)
        return stage, None

    accepted = 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf'
    stage, reason = translate(accepted)
    check('dpp_broadcast accepts quad_perm:[2,2,2,2] for v_cvt_off_f32_i4',
          stage is not None, str(reason))
    check('dpp_broadcast reads lane 2 of the quad with QuadReadLaneAt',
          stage is not None and any('QuadReadLaneAt(v1, 2u)' in line for line in stage.lines),
          str(stage and stage.lines))
    check('dpp_broadcast lowers the conversion on the broadcast value (nibble sign-extend x 1/16)',
          stage is not None and any('int(v256 << 28) >> 28' in line and '0.0625' in line
                                    for line in stage.lines),
          str(stage and stage.lines))
    check('dpp_broadcast requires shader model 6 for QuadReadLaneAt',
          stage is not None and stage.needs_sm6 is True)
    # The same broadcast at a different lane, to prove the lane is not hard-coded.
    lane3, reason = translate('v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf')
    check('dpp_broadcast reads the lane quad_perm names, not a fixed one',
          lane3 is not None and any('QuadReadLaneAt(v1, 3u)' in line for line in lane3.lines),
          str(reason) or str(lane3 and lane3.lines))
    for label, asm, state, message in (
        ('a non-broadcast quad_perm', 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf',
         {}, ''),
        ('a mixed quad_perm', 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,3] row_mask:0xf bank_mask:0xf',
         {}, 'only the swaps'),
        ('a partial row_mask', 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,2] row_mask:0x7 bank_mask:0xf',
         {}, 'partial row_mask'),
        ('bound_ctrl:0', 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf bound_ctrl:0',
         {}, 'bound_ctrl'),
        ('an entry mask not proven whole-quad',
         'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf', {'wqm_exec': False},
         'whole-quad'),
        ('a vertex stage', 'v_cvt_off_f32_i4_dpp v2, v1 quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf',
         {'kind': 'vs'}, 'straight-line pixel stage'),
    ):
        refused, detail = translate(asm, **state)
        check('dpp_broadcast refuses %s' % label,
              refused is None and message in (detail or ''), str(detail))


def check_refusals_and_unlowered():
    # Refusals: abs, neg, op_sel, SDWA (all decoded from synthetic gfx1013 bytes via G.decode())
    byte_refusals = (
        ('e64 abs on v2', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), abs0=1), ''),
        ('e64 neg on v2', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), neg0=1), ''),
        ('e64 neg on s2', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, S(2), neg0=1), ''),
        ('e64 neg on inline 3', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, INLINE_INT[3], neg0=1), 'neg/abs'),
        ('e64 op_sel src0', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), opsel=(1, 0, 0, 0)), 'op_sel'),
        ('e64 op_sel src1', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), opsel=(0, 1, 0, 0)), 'op_sel'),
        ('e64 op_sel dst', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2), opsel=(0, 0, 0, 1)), 'op_sel'),
        ('SDWA BYTE_1', encode_vop1_sdwa(VOP1_CVT_OFF_F32_I4, 4, 2, src0_sel=1), 'SDWA form of v_cvt_off_f32_i4'),
        ('SDWA DWORD clamp', encode_vop1_sdwa(VOP1_CVT_OFF_F32_I4, 4, 2, src0_sel=6, clamp=1),
         'SDWA form of v_cvt_off_f32_i4'),
        ('SDWA neg', encode_vop1_sdwa(VOP1_CVT_OFF_F32_I4, 4, 2, src0_sel=1, neg=1), ''),
        ('SDWA abs', encode_vop1_sdwa(VOP1_CVT_OFF_F32_I4, 4, 2, src0_sel=1, abs_=1), ''),
    )
    for label, code, msg in byte_refusals:
        check(f'refused: {label}', is_refused_bytes(code, msg))

    # Arity refusals
    for asm, expected_err in (
        ('v_cvt_off_f32_i4 v4', 'with 1 operands, expected 2'),
        ('v_cvt_off_f32_i4 v4, v2, v3', 'with 3 operands, expected 2'),
        ('v_cvt_off_f32_i4_e64 v4', 'with 1 operands, expected 2'),
        ('v_cvt_off_f32_i4_e64 v4, v2, 0', 'with 3 operands, expected 2'),
    ):
        check(f'refused arity: {asm}', is_refused_asm(asm, expected_err))

    # Unlowered control: temporarily remove 'v_cvt_off_f32_i4' from G.VEC_U_TO_F and assert both e32 and e64 refuse
    saved = G.VEC_U_TO_F.pop('v_cvt_off_f32_i4')
    try:
        for tag, code in (
            ('e32', encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 4, V(2))),
            ('e64', encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 4, V(2))),
        ):
            check(f'unlowered control ({tag})', is_refused_bytes(code, 'v_cvt_off_f32_i4'))
    finally:
        G.VEC_U_TO_F['v_cvt_off_f32_i4'] = saved


def controls():
    check_decode_and_lowering()
    check_low16_provenance()
    check_uniform_valu()
    check_dpp_broadcast()
    check_refusals_and_unlowered()
    if failures:
        print('FAILED:', ', '.join(failures))
        sys.exit(1)
    print('startup_gfx2hlsl_cvt_off_f32_i4_check: all controls passed')


# ---- Draw fixtures and mutated controls -----------------------------------------------------------------------------

# Base pixel body: 4 lanes alternating VOP1 _e32 (lanes 0, 2) and VOP3A _e64 (lanes 1, 3), reading v0..v3 -> v12..v15.
PLAIN_BODY = (
    encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 12, V(0))
    + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 13, V(1))
    + encode_vop1_e32(VOP1_CVT_OFF_F32_I4, 14, V(2))
    + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 15, V(3))
)

# Output-modifier pixel body: 4 lanes in VOP3A _e64 carrying clamp, mul:2, mul:4, and clamp + div:2.
MOD_SPEC = (
    ('', True),       # lane 0: clamp
    ('mul:2', False), # lane 1: mul:2
    ('mul:4', True),  # lane 2: clamp mul:4 (tests scaling before clamp)
    ('div:2', False), # lane 3: div:2
)
MOD_BODY = (
    encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 12, V(0), clamp=1)
    + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 13, V(1), omod='mul:2')
    + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 14, V(2), omod='mul:4', clamp=1)
    + encode_vop1_e64(VOP1_CVT_OFF_F32_I4, 15, V(3), omod='div:2')
)

# 20 inputs (5 draws x 4 lanes): all 16 nibble values 0..15 plus upper-bit garbage including 0x12345678 and 0xfffffff8.
PLAIN_DRAWS = [
    # Draw 0: exercises sign bit (0x8), upper-bit garbage (0x12345678, 0xfffffff8), and positive + negative nibbles
    [(0x12345678, 0, 0), (0xfffffff8, 0, 0), (0x80000007, 0, 0), (0xdeadbeef, 0, 0)],
    # Nibbles 0..3 with distinct upper-bit patterns
    [(0x00000000, 0, 0), (0x11111111, 0, 0), (0x22222222, 0, 0), (0x33333333, 0, 0)],
    # Nibbles 4..7 with distinct upper-bit patterns
    [(0x44444444, 0, 0), (0x55555555, 0, 0), (0x66666666, 0, 0), (0x77777777, 0, 0)],
    # Nibbles 8..11 with distinct upper-bit patterns
    [(0x88888888, 0, 0), (0x99999999, 0, 0), (0xaaaaaaa0 | 0xa, 0, 0), (0xbbbbbbb0 | 0xb, 0, 0)],
    # Nibbles 12..15 with distinct upper-bit patterns
    [(0xccccccc0 | 0xc, 0, 0), (0xddddddd0 | 0xd, 0, 0), (0xeeeeeee0 | 0xe, 0, 0), (0xffffffff, 0, 0)],
]

MOD_DRAWS = [
    # Lane 0 (clamp): -0.5 -> 0.0; Lane 1 (mul:2): -0.5 -> -1.0;
    # Lane 2 (clamp mul:4): 5 (0.3125 * 4 = 1.25 -> 1.0); Lane 3 (div:2): 7 -> 0.21875
    [(0xfffffff8, 0, 0), (0x12345678, 0, 0), (0x12345675, 0, 0), (0x80000007, 0, 0)],
    # Lane 0 (clamp): +0.4375 -> 0.4375; Lane 1 (mul:2): +0.4375 -> 0.875;
    # Lane 2 (clamp mul:4): -1 -> 0.0; Lane 3 (div:2): -8 -> -0.25
    [(0x00000007, 0, 0), (0xabcdef07, 0, 0), (0xffffffff, 0, 0), (0x00000008, 0, 0)],
]


def mutate_plain_unsigned(text):
    mutated, count = re.subn(r'int\((v\d+) << 28\) >> 28', r'(\1 & 15u)', text)
    assert count == 4 and mutated != text, count
    return mutated


def mutate_plain_5bit(text):
    mutated, count = re.subn(r'int\((v\d+) << 28\) >> 28', r'int(\1 << 27) >> 27', text)
    assert count == 4 and mutated != text, count
    return mutated


def mutate_mod_early_clamp(text):
    # Swap saturate((...) * 4.0) into (saturate(...)) * 4.0 on lane 2 (v14)
    pattern = r'saturate\(\(\(float\(int\(v2 << 28\) >> 28\) \* 0\.0625\)\) \* 4\.0\)'
    replacement = r'(saturate((float(int(v2 << 28) >> 28) * 0.0625))) * 4.0'
    mutated, count = re.subn(pattern, replacement, text)
    assert count == 1 and mutated != text, (count, text)
    return mutated


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    assert {x[0] & 0xf for chunk in PLAIN_DRAWS for x in chunk} == set(range(16))

    plain_ps_words = PRE_WORDS + PLAIN_BODY + POST_WORDS
    mod_ps_words = PRE_WORDS + MOD_BODY + POST_WORDS
    plain_ps_hlsl = G.Pipeline(vs_words([(0, 0, 0)] * 4), plain_ps_words, root_blob()).hlsl_ps()
    mod_ps_hlsl = G.Pipeline(vs_words([(0, 0, 0)] * 4), mod_ps_words, root_blob()).hlsl_ps()

    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    draws = mutations = 0
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            plain_cso = compile_hlsl(compiler, kind, plain_ps_hlsl, f'ps_plain_{kind}', ps_profile, directory)
            for idx, chunk in enumerate(PLAIN_DRAWS):
                vs_hlsl = G.Pipeline(vs_words(chunk), plain_ps_words, root_blob()).hlsl_vs()
                vs_cso = compile_hlsl(compiler, kind, vs_hlsl, f'vs_plain_{idx}_{kind}', vs_profile, directory)
                want = words_text([isa_cvt_off_f32_i4(x[0]) for x in chunk])
                run = draw(renderer, adapter, vs_cso, plain_cso, want)
                if run.returncode:
                    raise AssertionError(f'{kind} plain draw {idx} {chunk}\n{run.stdout}\n{run.stderr}')
                draws += 1
                print(f'{kind} plain[{idx}] {[hex(x[0]) for x in chunk]}: {want}', flush=True)

                if idx == 0:
                    for mname, mut_fn, ref_fn in (
                        ('unsigned_nibble', mutate_plain_unsigned, wrong_unsigned_nibble),
                        ('5bit_signext', mutate_plain_5bit, wrong_5bit_nibble),
                    ):
                        wrong = words_text([ref_fn(x[0]) for x in chunk])
                        assert wrong != want
                        bad_cso = compile_hlsl(compiler, kind, mut_fn(plain_ps_hlsl),
                                               f'ps_mut_{mname}_{kind}', ps_profile, directory)
                        mrun = draw(renderer, adapter, vs_cso, bad_cso, want)
                        if mrun.returncode == 0 or f'actual={wrong}' not in mrun.stderr:
                            raise AssertionError(
                                f'{kind} mutation {mname} did not read {wrong}\n{mrun.stdout}\n{mrun.stderr}')
                        mutations += 1
                        print(f'{kind} mutation {mname}: fails as expected, reads {wrong}', flush=True)

            mod_cso = compile_hlsl(compiler, kind, mod_ps_hlsl, f'ps_mod_{kind}', ps_profile, directory)
            for idx, chunk in enumerate(MOD_DRAWS):
                vs_hlsl = G.Pipeline(vs_words(chunk), mod_ps_words, root_blob()).hlsl_vs()
                vs_cso = compile_hlsl(compiler, kind, vs_hlsl, f'vs_mod_{idx}_{kind}', vs_profile, directory)
                want = words_text([
                    isa_cvt_off_f32_i4(x[0], omod=omod, clamp=clamp)
                    for x, (omod, clamp) in zip(chunk, MOD_SPEC)
                ])
                run = draw(renderer, adapter, vs_cso, mod_cso, want)
                if run.returncode:
                    raise AssertionError(f'{kind} mod draw {idx} {chunk}\n{run.stdout}\n{run.stderr}')
                draws += 1
                print(f'{kind} mod[{idx}] {[hex(x[0]) for x in chunk]}: {want}', flush=True)

                if idx == 0:
                    # Early clamp on lane 2: saturate(0.3125) * 4.0 = 1.25 (0x3fa00000) instead of 1.0 (0x3f800000)
                    wrong_words = [
                        isa_cvt_off_f32_i4(x[0], omod=omod, clamp=clamp)
                        for x, (omod, clamp) in zip(chunk, MOD_SPEC)
                    ]
                    wrong_words[2] = f32_bits(bits_to_f32(isa_cvt_off_f32_i4(chunk[2][0], clamp=True)) * 4.0)
                    wrong = words_text(wrong_words)
                    assert wrong != want
                    bad_cso = compile_hlsl(compiler, kind, mutate_mod_early_clamp(mod_ps_hlsl),
                                           f'ps_mut_early_clamp_{kind}', ps_profile, directory)
                    mrun = draw(renderer, adapter, vs_cso, bad_cso, want)
                    if mrun.returncode == 0 or f'actual={wrong}' not in mrun.stderr:
                        raise AssertionError(
                            f'{kind} mutation early_clamp did not read {wrong}\n{mrun.stdout}\n{mrun.stderr}')
                    mutations += 1
                    print(f'{kind} mutation early_clamp: fails as expected, reads {wrong}', flush=True)

    print(f'{adapter}: {draws} draws matched, {mutations} mutated draws read their mutated words')


def main():
    if len(sys.argv) == 1:
        controls()
        return
    if sys.argv[1] == 'draw' and len(sys.argv) in (5, 6):
        check_draw(*sys.argv[2:])
        return
    raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
