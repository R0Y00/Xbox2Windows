"""Compare native FBLD and FBSTP code with the x87 leaf-function profile.

Usage:
    python tests/cpu/x87_bcd.py asm <asm> <def>             regenerate the MASM source and exports
    python tests/cpu/x87_bcd.py emit <dll> <cpp> [control]   translate (and optionally mutate) the fixture
    python tests/cpu/x87_bcd.py check <harness> <native> <lifted> [controls...]

Each export reads values, 64-bit integers and 80-bit packed BCD operands from its input block, converts
them with FBLD and FBSTP, and stores the results in its output block. The exports in FIXED have no FBLD
and no FLDCW and are translated by the fixed profile, lower(..., x87=True); the others by the checked
entry, lower(..., x87=Checked(REJECT)). Both are wrapped as uint32_t name(word input, word output,
word* result), which returns the profile-rejection status (always 0 for the fixed profile) and writes
*result only for status 0. The native exports run through x87_call of tests/cpu/x87_harness.asm.

A packed BCD operand holds 18 decimal digits in bytes 0 to 8, two a byte with the lower digit in the low
nibble, and the sign in bit 7 of byte 9. FBSTP rounds ST(0) to an integer by the rounding control and
stores it with the sign of ST(0), or stores the packed BCD indefinite for a NaN, an infinity, an
unsupported encoding or a rounded magnitude of 10^18 or more. FBLD pushes the digits exactly. The SDM
leaves FBLD undefined for a digit above 9: the checked entry leaves with status "bcddigit" there, and
those inputs run natively only as an observation, which is compared with the sum of the digits times
their powers of ten but does not decide the test.

The model works with Fraction arithmetic and shares no code with tools/x87_soft.h.
"""
from fractions import Fraction
import ctypes
from pathlib import Path
import platform
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import x87_add as base  # noqa: E402  (tests/cpu is the script directory)
from x87_add import J, classify, floor_log2, m80_bytes  # noqa: E402

INPUT, OUTPUT = base.INPUT, base.OUTPUT
A, B, C, CW, RESTORE = 0, 16, 32, 48, 50  # input offsets: two operands, an integer or value, two control words
STATUS = {"ok": 0, "bcddigit": 6}  # x87_lower.STATUS, and 0 for success
SENTINEL = 0x5A5A5A5A5A5A5A5A  # *result before the call; the checked entry keeps it on a rejection
REJECT = "{{ x87_status = {status}; return 0; }}"
WORDS = [0x007F | pc << 8 | rc << 10 for pc in (0, 2, 3) for rc in range(4)]  # the twelve masked words
INDEFINITE_BCD = bytes(7) + b"\xc0\xff\xff"
LIMIT = 10 ** 18

# name: (input kind, steps). Every export ends with MOV EAX, 0 and RET.
SEQUENCES = {
    # The fixed profile: control word 0x027F, so FBSTP rounds to nearest-even.
    "bstp": ("value", [("fld", A), ("fbstp", 0)]),
    "bstp_int": ("int", [("fild", C), ("fbstp", 0)]),
    # FBSTP of ST(0) at depth 8, then a store of the next value.
    "bstp_depth8": ("value", [("fld", A), *[("fld_st", 0)] * 7, ("fbstp", 0), ("fstp", 16), *[("pop",)] * 6]),
    # The checked entry: FBSTP under each of the twelve masked control words.
    "bstp_cw": ("cw", [("fldcw", CW), ("fld", A), ("fbstp", 0), ("fldcw", RESTORE)]),
    "bld": ("bcd", [("fbld", "rcx", A), ("fstp", 0)]),
    "bld_bstp": ("bcd", [("fbld", "rcx", A), ("fbstp", 0)]),
    # A round trip through the output block: the indefinite FBSTP stores reads back as an invalid digit.
    "bstp_bld": ("value", [("fld", A), ("fbstp", 0), ("fbld", "rdx", 0), ("fstp", 16)]),
    # The sum of two loaded integers, rounded to 53 bits by FADDP.
    "bld_add": ("bcd2", [("fbld", "rcx", A), ("fbld", "rcx", B), ("faddp",), ("fstp", 0)]),
    # FBLD as the eighth value.
    "bld_depth8": ("bcdc", [("fld", C), *[("fld_st", 0)] * 6, ("fbld", "rcx", A), ("fstp", 0), ("fstp", 16),
                            *[("pop",)] * 6]),
}
FIXED = {"bstp", "bstp_int", "bstp_depth8"}


# ---- MASM generation and decoding ------------------------------------------------------------------

def asm_line(step):
    op, *args = step
    return {"fld": lambda: f"    fld tbyte ptr [rcx+{args[0]}]",
            "fild": lambda: f"    fild qword ptr [rcx+{args[0]}]",
            "fld_st": lambda: f"    fld st({args[0]})",
            "fbld": lambda: f"    fbld tbyte ptr [{args[0]}+{args[1]}]",
            "fbstp": lambda: f"    fbstp tbyte ptr [rdx+{args[0]}]",
            "fstp": lambda: f"    fstp tbyte ptr [rdx+{args[0]}]",
            "pop": lambda: "    fstp st(0)",
            "faddp": lambda: "    faddp st(1), st",
            "fldcw": lambda: f"    fldcw word ptr [rcx+{args[0]}]"}[op]()


def expected_forms(step):
    """The x87 instruction of a step as (mnemonic, operand): m16, m64, m80, bcd, st0 to st7 or none."""
    op, *args = step
    return {"fld": ("fld", "m80"), "fild": ("fild", "m64"), "fld_st": ("fld", f"st{args[0] if args else 0}"),
            "fbld": ("fbld", "bcd"), "fbstp": ("fbstp", "bcd"), "fstp": ("fstp", "m80"), "pop": ("fstp", "st0"),
            "faddp": ("faddp", "st1"), "fldcw": ("fldcw", "m16")}[op]


def decoded_form(i):
    from iced_x86 import MemorySize as S, MemorySizeExt, Mnemonic as M, OpKind as K, Register as R
    name = next(n for n, v in vars(M).items() if v == i.mnemonic).lower()
    if i.op_count == 0:
        return name, ""
    if i.op0_kind == K.MEMORY:
        return name, "bcd" if i.memory_size == S.BCD else f"m{8 * MemorySizeExt.size(i.memory_size)}"
    return name, f"st{i.op0_register - R.ST0}"


def write_asm(asm_path, def_path):
    lines = ["; coverage: source cpu_aot_x87_bcd",
             "; Generated by tests/cpu/x87_bcd.py asm; edit SEQUENCES there and regenerate.",
             "; Source-owned differential fixture for FBLD and FBSTP in the x87 leaf-function profile. Each",
             "; export reads an input block at RCX, writes an output block at RDX and returns with the x87",
             "; stack empty.", ".code", ""]
    for name, (_, steps) in SEQUENCES.items():
        lines += [f"{name} PROC FRAME", "    .endprolog", *(asm_line(s) for s in steps), "    mov eax, 0", "    ret",
                  f"{name} ENDP", ""]
    lines.append("END")
    Path(asm_path).write_text("\n".join(lines) + "\n", encoding="ascii")
    Path(def_path).write_text("EXPORTS\n" + "".join(f"    {name}\n" for name in SEQUENCES), encoding="ascii")


# ---- Translation -----------------------------------------------------------------------------------

def entry(name, code):
    return (f'extern "C" __declspec(dllexport) uint32_t {name}(word a, word b, word* result) {{\n'
            "    uint32_t x87_status = 0;\n"
            f"    const word value = [&]() -> word {{\n{code}\n    }}();\n"
            "    if (x87_status == 0) *result = value;\n"
            "    return x87_status;\n}\n")


RANGE = "n >= 1000000000000000000ull) {  // bcd range"
CONTROLS = {
    # FBSTP rounds to nearest-even under every control word.
    "rcnearest": (re.escape("x87_to_integer(v, 64, false, cw)"), "x87_to_integer(v, 64, false, 0x027f)"),
    # The range check reads the value truncated, before rounding.
    "rangebefore": (re.escape(RANGE), "(x87_sign(v) ? 0 - x87_to_integer(v, 64, true, cw) : "
                                      "x87_to_integer(v, 64, true, cw)) >= 1000000000000000000ull) {  // bcd range"),
    # 10^18 itself is still stored as digits.
    "threshold": (re.escape(RANGE), "n > 1000000000000000000ull) {  // bcd range"),
    # A negative value that rounds to zero is stored as +0.
    "zerosign": (re.escape("const uint16_t sign = x87_sign(v) ? 0x8000 : 0;  // bcd sign"),
                 "const uint16_t sign = x87_sign(v) && r != 0 ? 0x8000 : 0;  // bcd sign"),
    # The indefinite has the integer-indefinite top byte 0x80 instead of 0xc0.
    "indefinite": (re.escape("low = 0xc000000000000000ull;"), "low = 0x8000000000000000ull;"),
    # FBSTP stores the two digits of each byte in the other nibbles.
    "bstpnibbles": (re.escape("low |= (n % 10) << (4 * k);"), "low |= (n % 10) << (4 * (k ^ 1));"),
    # FBLD reads the other nibble of each byte first.
    "bldnibbles": (re.escape("(4 * (k & 15))) & 15;  // bcd digit"), "(4 * ((k & 15) ^ 1))) & 15;  // bcd digit"),
    # FBLD reads the sign from every bit of byte 9.
    "signbits": (re.escape("(high & 0x8000) != 0;  // bcd sign bit"), "(high & 0xff00) != 0;  // bcd sign bit"),
    # FBLD accepts a digit above 9 as its value.
    "nodigitcheck": (re.escape("if (digit > 9) return false;"), ""),
}


def emit(image, output, control="lifted"):
    import pefile
    from cpu_leaf_aot import export_rva
    from cpu_ranges import function_ranges
    from ffx_version_query import INTEGER_HEADER, lower, region
    from x87_lower import X87_HEADER, Checked, is_x87

    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (_, steps) in SEQUENCES.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            decoded = [decoded_form(i) for i in body if is_x87(i)]
            wanted = [expected_forms(s) for s in steps]
            if decoded != wanted:
                raise AssertionError(f"{name} assembled as {decoded}, expected {wanted}")
            profile = True if name in FIXED else Checked(REJECT)
            parts.append(entry(name, lower(body, ["rcx = a; rdx = b;"], {}, {}, {}, x87=profile)))
    text = "".join(parts)
    if control != "lifted":
        pattern, replacement = CONTROLS[control]
        text, count = re.subn(pattern, lambda _: replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


# ---- Independent model -----------------------------------------------------------------------------
# Values are raw 80-bit pairs (sig, se); packed BCD operands are 10-byte strings.

def round_integer(sign, x, rc):
    """|value| x >= 0 with the given sign rounded to an integer magnitude by rounding control rc."""
    floor = x.numerator // x.denominator
    if floor == x:
        return floor
    if rc == 0:
        rest = x - floor
        return floor + (1 if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and floor & 1) else 0)
    if rc == 3 or rc == (2 if sign else 1):  # toward zero, or toward the infinity on the other side
        return floor
    return floor + 1


def to_bcd(v, rc):
    """FBSTP of v under rounding control rc."""
    c = classify(v)
    if c[0] in ("unsupported", "nan", "inf"):
        return INDEFINITE_BCD
    sign = v[1] >> 15
    q = 0 if c[0] == "zero" else round_integer(sign, c[2], rc)
    if q >= LIMIT:
        return INDEFINITE_BCD
    digits = [q // 10 ** k % 10 for k in range(18)]
    return bytes(digits[2 * k] | digits[2 * k + 1] << 4 for k in range(9)) + bytes([sign << 7])


def bcd_bytes(n, sign_byte=None):
    """The packed BCD encoding of the integer n (|n| < 10^18), with byte 9 given or from the sign of n."""
    m = abs(n)
    digits = bytes((m // 10 ** (2 * k) % 10) | (m // 10 ** (2 * k + 1) % 10) << 4 for k in range(9))
    return digits + bytes([(0x80 if n < 0 else 0) if sign_byte is None else sign_byte])


def from_bcd(b):
    """FBLD of b: (valid, value). For a digit above 9 the value is the sum of the digits times their powers
    of ten, which the native observation is compared with."""
    digits = [b[k // 2] >> (4 * (k % 2)) & 15 for k in range(18)]
    n = sum(d * 10 ** k for k, d in enumerate(digits))
    sign = b[9] >> 7
    return all(d <= 9 for d in digits), (0, sign << 15) if n == 0 else base.m80_exact(sign, Fraction(n))


class Outcome:
    def __init__(self, status, out, observed=False):
        self.status, self.out, self.observed = status, bytes(out), observed


def run(name, data, native=False):
    """The model of an export, for the translation or (native set) for a native run, which continues past
    a digit above 9 with the observed digit sum and marks the outcome as an observation."""
    steps = SEQUENCES[name][1]
    out, slots, rc, observed = bytearray(b"\xcc" * OUTPUT), [], 0, False
    for op, *args in steps:
        if op == "fld":
            raw = int.from_bytes(data[args[0]:args[0] + 10], "little")
            slots.append((raw & (2 ** 64 - 1), raw >> 64))
        elif op == "fild":
            n = int.from_bytes(data[args[0]:args[0] + 8], "little", signed=True)
            slots.append((0, 0) if n == 0 else base.m80_exact(int(n < 0), Fraction(abs(n))))
        elif op == "fld_st":
            slots.append(slots[-1 - args[0]])
        elif op == "fbld":
            source = data if args[0] == "rcx" else out
            valid, value = from_bcd(bytes(source[args[1]:args[1] + 10]))
            if not valid:
                if not native:
                    return Outcome(STATUS["bcddigit"], out)
                observed = True
            slots.append(value)
        elif op == "fbstp":
            out[args[0]:args[0] + 10] = to_bcd(slots.pop(), rc)
        elif op in ("fstp", "pop"):
            v = slots.pop()
            if op == "fstp":
                out[args[0]:args[0] + 10] = m80_bytes(v)[:10]
        elif op == "faddp":
            slots[-2] = base.add(slots[-2], slots[-1])
            slots.pop()
        elif op == "fldcw":
            rc = (int.from_bytes(data[args[0]:args[0] + 2], "little") >> 10) & 3
        else:
            raise AssertionError(op)
    return Outcome(STATUS["ok"], out, observed)


def check_model():
    """Spot checks of the model against conversions worked by hand."""
    half, one_half, two_half = (J, 0x3FFE), (J | J >> 1, 0x3FFF), (J | J >> 2, 0x4000)
    rows = [
        (to_bcd(one_half, 0), bytes.fromhex("02000000000000000000")),
        (to_bcd(two_half, 0), bytes.fromhex("02000000000000000000")),   # the tie to even
        (to_bcd(two_half, 2), bytes.fromhex("03000000000000000000")),
        (to_bcd((J, 0xBFFE), 0), bytes.fromhex("00000000000000000080")),  # -0.5 rounds to -0
        (to_bcd((J, 0xBFFE), 1), bytes.fromhex("01000000000000000080")),  # toward -inf: -1
        (to_bcd(base.m80_exact(0, Fraction(LIMIT - 1)), 0), bytes.fromhex("99999999999999999900")),
        (to_bcd(base.m80_exact(0, Fraction(LIMIT)), 0), INDEFINITE_BCD),
        (to_bcd(base.m80_exact(1, Fraction(1234567)), 3), bytes.fromhex("67452301000000000080")),
        (to_bcd((J, 0x7FFF), 0), INDEFINITE_BCD),
        (to_bcd(half, 2), bytes.fromhex("01000000000000000000")),
        (from_bcd(bytes.fromhex("67452301000000000080")), (True, base.m80_exact(1, Fraction(1234567)))),
        (from_bcd(bytes(9) + b"\x80"), (True, (0, 0x8000))),
        (from_bcd(bytes(9) + b"\x7f"), (True, (0, 0))),
        (from_bcd(b"\x0a" + bytes(9))[0], False),
    ]
    for got, want in rows:
        if got != want:
            raise AssertionError(f"model gave {got}, expected {want}")
    return len(rows)


# ---- Cases -----------------------------------------------------------------------------------------

def values(rng):
    """m80 values: the special encodings, then values chosen to reach each rounding and range branch."""
    sign = lambda: rng.getrandbits(1)  # noqa: E731
    rows = list(base.special_m80())
    for _ in range(600):  # magnitudes from 2^-70 to 2^70
        rows.append((J | rng.getrandbits(63), (0x3FFF + rng.randint(-70, 70)) | sign() << 15))
    for _ in range(80):  # the rest of the exponent range
        rows.append((J | rng.getrandbits(63), rng.randint(0x3FFF + 70, 0x7FFE) | sign() << 15))
    for k in range(-40, 41):  # around 10^18 in steps of 1/16, the quantum there
        rows.append(base.m80_exact(sign(), Fraction(LIMIT) + Fraction(k, 16)))
    for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100, LIMIT - 2, LIMIT - 1, LIMIT, LIMIT + 1):  # ties and neighbours
        for f in (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(0)):
            if n + f:
                rows.append(base.m80_exact(sign(), n + f))
    for _ in range(300):  # integers and halves of every size below the limit
        n = rng.randrange(10 ** rng.randint(1, 18))
        f = rng.choice((Fraction(0), Fraction(1, 2), Fraction(1, 2 ** rng.randint(1, 8))))
        if n + f and (n + f).denominator <= 2 ** (63 - floor_log2(n + f)):  # exact in 64 significand bits
            rows.append(base.m80_exact(sign(), n + f))
    for _ in range(40):  # denormals and pseudo-denormals
        rows.append((rng.getrandbits(rng.randint(1, 63)) | rng.choice((0, J)), sign() << 15))
    for _ in range(120):  # raw encodings: unnormals, pseudo-NaNs and pseudo-infinities among them
        rows.append((rng.getrandbits(64), rng.getrandbits(16)))
    return rows


def integers(rng):
    rows = [0, 1, -1, LIMIT - 1, -(LIMIT - 1), LIMIT, -LIMIT, LIMIT + 1, -(LIMIT + 1), 2 ** 63 - 1, -2 ** 63]
    rows += [rng.randrange(-2 ** 63, 2 ** 63) >> rng.randint(0, 63) for _ in range(200)]
    rows += [LIMIT + rng.randint(-1000, 1000) for _ in range(20)] + [-LIMIT + rng.randint(-1000, 1000) for _ in range(20)]
    return rows


def bcds(rng):
    """Packed BCD operands: valid digits with each kind of byte 9, then a digit above 9 at each position."""
    rows = [bcd_bytes(0), bcd_bytes(0, 0x80), bcd_bytes(LIMIT - 1), bcd_bytes(-(LIMIT - 1)), bcd_bytes(1),
            bcd_bytes(-1), bcd_bytes(0, 0x7F), bcd_bytes(5, 0xFF), INDEFINITE_BCD]
    for _ in range(400):
        n = rng.randrange(10 ** rng.randint(1, 18))
        rows.append(bcd_bytes(n, rng.choice((0, 0x80, rng.getrandbits(8)))))
    for k in range(18):  # one digit from 10 to 15 at each digit position
        for _ in range(3):
            b = bytearray(bcd_bytes(rng.randrange(LIMIT), rng.choice((0, 0x80))))
            b[k // 2] = b[k // 2] & (0xF0 if k % 2 == 0 else 0x0F) | rng.randint(10, 15) << (4 * (k % 2))
            rows.append(bytes(b))
    rows += [bytes(rng.getrandbits(8) for _ in range(10)) for _ in range(40)]  # mostly invalid
    return rows


def cases():
    rng = random.Random(406)
    vs, ns, bs = values(rng), integers(rng), bcds(rng)

    def block(a=b"", b=b"", c=b"", cw=0x027F):
        data = bytearray(INPUT)
        data[A:A + len(a)], data[B:B + len(b)], data[C:C + len(c)] = a, b, c
        data[CW:CW + 2], data[RESTORE:RESTORE + 2] = cw.to_bytes(2, "little"), (0x027F).to_bytes(2, "little")
        return bytes(data)

    valid = [b for b in bs if from_bcd(b)[0]]
    for name, (kind, _) in SEQUENCES.items():
        if kind == "value":
            rows = [block(m80_bytes(v)) for v in vs]
        elif kind == "int":
            rows = [block(c=(n % 2 ** 64).to_bytes(8, "little")) for n in ns]
        elif kind == "cw":
            rows = [block(m80_bytes(v), cw=w) for v in vs for w in WORDS]
        elif kind == "bcd":
            rows = [block(b) for b in bs]
        elif kind == "bcd2":
            rows = [block(rng.choice(valid), rng.choice(valid if k % 8 else bs)) for k in range(400)]
        else:  # bcdc
            rows = [block(b, c=m80_bytes(rng.choice(vs))) for b in bs]
        for data in rows:
            yield name, data


# ---- Check -----------------------------------------------------------------------------------------

class EntryRunner:
    """Calls the wrapped entries of a translated DLL with guarded buffers."""

    def __init__(self, harness_path, path):
        self.buffers = base.Runner(harness_path, path)
        self.lib = self.buffers.lib

    def __call__(self, name, data):
        guard, source, target = self.buffers._buffers(data)
        result = ctypes.c_uint64(SENTINEL)
        function = getattr(self.lib, name)
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
        function.restype = ctypes.c_uint32
        status = function(ctypes.addressof(source) + base.GUARD, ctypes.addressof(target) + base.GUARD,
                          ctypes.byref(result))
        return status, self.buffers._output(name, data, guard, source, target), result.value


def expected_lifted(model):
    return model.status, model.out, 0 if model.status == STATUS["ok"] else SENTINEL


def check(harness_path, native_path, lifted_path, *control_paths):
    print(f"the model agreed with {check_model()} conversions worked by hand")
    expected = [(name, data, run(name, data)) for name, data in cases()]
    lifted = EntryRunner(harness_path, lifted_path)
    statuses = dict.fromkeys(STATUS.values(), 0)
    indefinite = bad = 0
    for name, data, model in expected:
        got = lifted(name, data)
        statuses[model.status] += 1
        indefinite += any(model.out[k:k + 10] == INDEFINITE_BCD for k in (0, 16))
        if got != expected_lifted(model):
            if bad < 3:
                print(f"lifted mismatch: {name}(input {data.hex()})\n  got      status {got[0]}, {got[1][:26].hex()}, "
                      f"{got[2]:#x}\n  expected status {model.status}, {model.out[:26].hex()}")
            bad += 1
    if bad:
        raise AssertionError(f"lifted: {bad}/{len(expected)} mismatches")
    print(f"{len(expected)} cases over {len(SEQUENCES)} exports ({len(FIXED)} in the fixed profile) matched the model "
          "in the translation (status, output block and guards, and *result, which only status 0 writes)")
    print("translated statuses: " + ", ".join(f"{name} {statuses[code]}" for name, code in STATUS.items())
          + f"; {indefinite} cases stored the packed BCD indefinite")
    native = base.Runner(harness_path, native_path)
    compared = observed = observed_agree = 0
    for name, data, _ in expected:
        want = run(name, data, native=True)
        got, state = native(name, data)
        if (state & 0xFFFF, (state >> 32) & 0xFFFF, (state >> 16) & 0x3800) != (0x027F, 0xFFFF, 0):
            raise AssertionError(f"native {name} left state {state:#x}; the profile requires an empty stack at TOP 0")
        if want.observed:
            observed += 1
            observed_agree += got == want.out
            continue
        if got != want.out:
            raise AssertionError(f"native {name}(input {data.hex()}) gave {got[:26].hex()}, expected {want.out[:26].hex()}")
        compared += 1
    print(f"native: {compared} cases matched the model, with control 0x027f and an empty stack after each export")
    print(f"native FBLD with a digit above 9 (observation, rejected by the checked entry): {observed_agree} of "
          f"{observed} cases matched the sum of the digits times their powers of ten")
    print(f"processor: {platform.processor()}")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        run_control = EntryRunner(harness_path, path)
        bad = sum(1 for name, data, model in expected if run_control(name, data) != expected_lifted(model))
        print(f"control {control}: {bad}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")
    check_rejections()


def check_rejections():
    from iced_x86 import Decoder
    from ffx_version_query import lower
    from x87_lower import Checked

    def attempt(hex_bytes, checked=True):
        body = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(body, [], {}, {}, {}, x87=Checked(REJECT) if checked else True)
        except ValueError as exc:
            return str(exc)
        return None

    fld, pop, fbld, fbstp, fstp = "db 29", "dd d8", "df 21", "df 32", "db 3a"  # [rcx] loads, [rdx] stores
    rejected = {
        "fbld in the fixed profile": (attempt(f"{fbld} {fbstp} c3", False), "unsupported x87 operation"),
        "fbld at depth 8": (attempt(f"{fld * 1} " + f"{fld} " * 7 + f"{fbld} " + f"{pop} " * 9 + "c3"),
                            "x87 stack overflow"),
        "fbstp on an empty stack": (attempt(f"{fbstp} c3", False), "x87 stack underflow"),
        "a store of the value fbstp popped": (attempt(f"{fld} {fbstp} {fstp} c3", False), "x87 stack underflow"),
        "fbld with a value left at the return": (attempt(f"{fbld} c3"), "x87 stack not empty at return"),
    }
    for case, (message, needle) in rejected.items():
        if message is None or needle not in message:
            raise AssertionError(f"{case}: expected a rejection containing {needle!r}, got {message!r}")
    print(f"rejected {len(rejected)} sequences: " + "; ".join(rejected))
    accepted = {
        "fbstp in the fixed profile": attempt(f"{fld} {fbstp} c3", False),
        "fbld and fbstp in the checked entry": attempt(f"{fbld} {fbstp} c3"),
        "fbld as the eighth value": attempt(f"{fld} " * 7 + f"{fbld} " + f"{pop} " * 8 + "c3"),
    }
    for case, message in accepted.items():
        if message is not None:
            raise AssertionError(f"{case}: expected translation, got {message!r}")
    print(f"translated {len(accepted)} sequences: " + "; ".join(accepted))


if __name__ == "__main__":
    mode, first, second, *rest = sys.argv[1:]
    if mode == "asm":
        write_asm(first, second)
    elif mode == "emit":
        emit(Path(first), Path(second), *rest)
    elif mode == "check":
        check(first, second, *rest)
    else:
        raise ValueError(mode)
