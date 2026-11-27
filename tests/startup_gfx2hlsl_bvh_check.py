"""image_bvh_intersect_ray: the FP32 box-node (kind 5) translation slice.

The instruction is RDNA 2 MIMG opcode 230. This check covers the translator path and the
box model only: a constant inline BVH T#[12:15], an 11-dword packed address, a constant
node base whose low three bits are 5, and the four child results. Triangle nodes, 16-bit
box nodes, the 64-bit form, A16, NSA and every dynamic descriptor or node base are refused
by name before any HLSL is emitted. The independent CPU box reference is
tests/gfx1030_bvh_box/bvh_model.py; the Windows dispatch exercises the actual emitted helper.
"""
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "startup"))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
import startup_gfx2hlsl_compute_check as C  # noqa: E402
from gfx1030_bvh_box import bvh_model as B  # noqa: E402

INVALID = 0xffffffff
SORT_ENABLE = 1 << 31
RESOURCE_TYPE = 1 << 31
MASK = 0xffffffff
FLT_MAX = 3.4028234663852886e38


def box(child, depth, x=-8):
    return child, (x, -8, depth), (x + 16, 8, depth + 0.5)


NODES = {
    0: (box(0x25, 4), box(0x1d, 1), box(0x2d, 2, x=32), box(INVALID, 3)),
    256: (box(0x45, 7), box(0x3d, 5), box(0x35, 3), box(0x0d, 1)),
    384: (box(0x25, 4), box(INVALID, 1), box(0x2d, 2), box(0x35, 3)),
    # A finite hit at FLT_MAX must sort ahead of misses (key +infinity).
    512: (box(INVALID, 1), (0x4d, (FLT_MAX,) * 3, (FLT_MAX,) * 3),
          box(0x55, 2, x=32), box(INVALID, 3)),
}
ORIGIN = (0.5, -0.5, -1.0)
FORWARD = (0.25, -0.125, 0.5)
RAYS = (
    ("sorted", 0, 19.0, ORIGIN, FORWARD, (0x1d, 0x25, INVALID, INVALID)),
    ("clipped", 0, 6.0, ORIGIN, FORWARD, (0x1d, INVALID, INVALID, INVALID)),
    ("all_miss", 0, 2.0, ORIGIN, FORWARD, (INVALID,) * 4),
    ("all_four", 256, 19.0, ORIGIN, FORWARD, (0x0d, 0x35, 0x3d, 0x45)),
    ("reverse", 0, 21.0, (0.0, 0.0, 10.0), (0.25, -0.125, -0.5), (0x25, 0x1d, INVALID, INVALID)),
    ("finite_max", 512, math.inf, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0x4d, INVALID, INVALID, INVALID)),
    ("encoded_base_low_bits", 0, 19.0, ORIGIN, FORWARD, (0x1d, 0x25, INVALID, INVALID)),
    ("encoded_base_high_byte", 256, 19.0, ORIGIN, FORWARD, (0x0d, 0x35, 0x3d, 0x45)),
)
UNSORTED_RAYS = (
    ("unsorted_three_hits", 384, 19.0, ORIGIN, FORWARD, (0x25, INVALID, 0x2d, 0x35)),
    ("unsorted_mixed", 0, 19.0, ORIGIN, FORWARD, (0x25, 0x1d, INVALID, INVALID)),
    ("unsorted_finite_max", 512, math.inf, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
     (INVALID, 0x4d, INVALID, INVALID)),
)
DESCRIPTOR_BASES = {
    "encoded_base_low_bits": (0x1008, 0),
    "encoded_base_high_byte": (0, 0x81),
}
RESULT_OFFSET = max(NODES) + 128
RESULT_WORD = RESULT_OFFSET // 4
BUFFER_SIZE = RESULT_OFFSET + 16 * (len(RAYS) + len(UNSORTED_RAYS))

failures = []


def check(label, ok, detail=""):
    print("%s: %s" % (label, "ok" if ok else "MISMATCH"), flush=True)
    if not ok:
        failures.append(label)
        if detail:
            print("  " + detail)


def node_bytes():
    data = bytearray(RESULT_OFFSET)
    for offset, nodes in NODES.items():
        struct.pack_into("<4I", data, offset, *(child for child, _, _ in nodes))
        for i, (_, lo, hi) in enumerate(nodes):
            struct.pack_into("<6f", data, offset + 16 + 24 * i, *lo, *hi)
    return data


def reference(offset, extent, origin, direction, sort_boxes):
    """Use the independent model on bytes relative to the bound allocation."""
    return tuple(B.intersect(
        node_bytes(), B.descriptor(sort=sort_boxes), (offset >> 3) | 5,
        extent=extent, origin=origin, direction=direction,
        inverse=tuple(1.0 / c for c in direction)))


def model(nodes, extent, origin, direction, sort_boxes):
    """The emitted HLSL helper, transcribed: four slabs, keys, five compare-exchange stages."""
    inv = [max(-FLT_MAX, min(1.0 / c, FLT_MAX)) for c in direction]
    items = []
    for child, lo, hi in nodes:
        if child == INVALID:
            items.append([INVALID, math.inf])
            continue
        bound0 = [(lo[i] - origin[i]) * inv[i] for i in range(3)]
        bound1 = [(hi[i] - origin[i]) * inv[i] for i in range(3)]
        tmin = max(min(bound0[i], bound1[i]) for i in range(3))
        tmax = min(max(bound0[i], bound1[i]) for i in range(3))
        hit = not math.isnan(lo[0]) and tmax >= max(0.0, tmin) and tmin < extent
        if hit:
            items.append([child, tmin if sort_boxes else 0.0])
        else:
            items.append([INVALID, math.inf])
    if not sort_boxes:
        return tuple(child for child, _key in items)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
        if items[b][1] < items[a][1]:
            items[a], items[b] = items[b], items[a]
    return tuple(item[0] for item in items)


def translate_program(*, node=0x05, sort_boxes=True, word0=0x1000, word1=None, word2=0x0b,
                      word3=RESOURCE_TYPE, loaded_descriptor=False, modifier="", opcode="image_bvh_intersect_ray",
                      constant_node=True, stage_kind="cs"):
    """A straight-line compute program with a constant inline T# and a constant node pointer."""
    if word1 is None:
        word1 = SORT_ENABLE if sort_boxes else 0
    if stage_kind != "cs":
        class Probe:
            kind = stage_kind
        try:
            G.Stage.op_image_bvh_intersect_ray(Probe(), ("v[32:35]", "v[16:31]", "s[12:15]"), [])
        except G.Unsupported as error:
            return None, None, [str(error)]
        return None, None, ["accepted outside compute"]
    decoded = []
    at = 0

    def add(name, asm, length=8):
        nonlocal at
        decoded.append((at, length, name, asm))
        at += length

    if loaded_descriptor:
        add("s_mov_b32", "s_mov_b32 s10, s6")
        add("s_mov_b32", "s_mov_b32 s11, 0")
        add("s_load_dwordx4", "s_load_dwordx4 s[12:15], s[10:11], 0x0", 16)
        add("s_waitcnt", "s_waitcnt lgkmcnt(0)", 4)
    else:
        for reg, word in ((12, word0), (13, word1), (14, word2), (15, word3)):
            add("s_mov_b32", "s_mov_b32 s%d, 0x%08x" % (reg, word))
    for reg, word in ((16, node), (17, 0x41a00000), (18, 0x3f000000), (19, 0xbf000000),
                      (20, 0xbf800000), (21, 0x3e800000), (22, 0xbe000000), (23, 0x3f000000),
                      (24, 0x40800000), (25, 0xc1000000), (26, 0x40000000)):
        source = "s4" if reg == 16 and not constant_node else "0x%08x" % word
        add("v_mov_b32", "v_mov_b32 v%d, %s" % (reg, source))
    add(opcode, "%s v[32:35], v[16:31], s[12:15]%s" % (opcode, (" " + modifier) if modifier else ""))
    add("s_endpgm", "s_endpgm", 4)
    refusals = []
    pipe = G.ComputePipeline(b"", C.root_blob(), C.USER7, (1, 1, 1), refusals=refusals, decoded=decoded)
    return pipe, pipe.hlsl_cs(), [reason for reason, _kind in refusals]


def controls():
    bad = None
    for name, offset, extent, origin, direction, expected in RAYS:
        actual = reference(offset, extent, origin, direction, True)
        if actual != expected:
            bad = (name, actual, expected)
            break
    check("independent public box model matches sorted fixture expectations", bad is None, str(bad))

    bad = None
    for name, offset, extent, origin, direction, expected in RAYS:
        actual = model(NODES[offset], extent, origin, direction, True)
        if actual != expected:
            bad = (name, actual, expected)
            break
    check("transcribed helper model matches independent CPU reference", bad is None, str(bad))

    # Negative control: sort disabled keeps child order. The sorted ray's first word changes.
    unsorted = reference(0, 19.0, ORIGIN, FORWARD, False)
    sorted_result = reference(0, 19.0, ORIGIN, FORWARD, True)
    check("sort bit negative control changes word 0",
          unsorted[0] == 0x25 and sorted_result[0] == 0x1d, str((unsorted, sorted_result)))
    check("model without the sort bit matches the unsorted reference",
          model(NODES[0], 19.0, ORIGIN, FORWARD, False) == unsorted,
          str((model(NODES[0], 19.0, ORIGIN, FORWARD, False), unsorted)))
    for name, offset, extent, origin, direction, expected in UNSORTED_RAYS:
        actual = reference(offset, extent, origin, direction, False)
        check("CPU reference keeps child order with the sort bit clear (%s)" % name, actual == expected,
              str((actual, expected)))
        actual = model(NODES[offset], extent, origin, direction, False)
        check("helper keeps child order with the sort bit clear (%s)" % name, actual == expected,
              str((actual, expected)))
    check("all-miss result is four 0xffffffff words",
          reference(0, 2.0, ORIGIN, FORWARD, True) == (INVALID,) * 4, "")

    pipe, hlsl, refusals = translate_program()
    check("constant box32 program translates", pipe is not None and not refusals, str(refusals))
    if pipe is not None:
        manifest = pipe.manifest()
        check("manifest names the node allocation", "srv 0 bvh base 0x100000 size 0xb" in manifest, manifest)
        check("HLSL declares the node buffer", "ByteAddressBuffer r0 : register(t0);" in hlsl, "")
        check("HLSL contains the four-result helper",
              "uint4 x2w_bvh_intersect_box32" in hlsl and "x2w_bvh_box32_slot" in hlsl, "")
        check("the call carries the sort-enabled T# word and four child results",
              "0x80000000u" in hlsl and all("v%d = bt" % reg in hlsl for reg in range(32, 36)),
              "")
        call_line = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_box32(r0" in line), "")
        check("constant T# passes literals, not descriptor SGPRs",
              "0x00001000u, 0x80000000u, 0x0000000bu, 0x80000000u" in call_line
              and "s12" not in call_line and "s13" not in call_line, call_line)

    pipe, hlsl, refusals = translate_program(sort_boxes=False)
    check("sort-disabled program translates", pipe is not None and not refusals, str(refusals))
    if pipe is not None:
        call = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_box32(r0" in line), "")
        check("sort-disabled call uses a zero sort word",
              "0x00001000u, 0x00000000u, 0x0000000bu, 0x80000000u" in call, call)

    # Independently decode the 40-bit unit field; test the same raw DWORDs in
    # the emitted call and the decoded byte-address binding in the manifest.
    for units, high, base in ((0x1000, 0, 0x100000), (0x1008, 0, 0x100800),
                              (0, 0x81, 0x810000000000)):
        words = (units, high | SORT_ENABLE, 0x0b, RESOURCE_TYPE)
        check("public model decodes base %#x" % base,
              B.node_address(words, 5) == (5, base), str(B.node_address(words, 5)))
        pipe, hlsl, refusals = translate_program(node=0x25, word0=words[0], word1=words[1])
        check("encoded base %#x translates" % base, pipe is not None and not refusals, str(refusals))
        if pipe is not None:
            manifest = pipe.manifest()
            call = next((line for line in hlsl.splitlines() if "x2w_bvh_intersect_box32(r0" in line), "")
            literal_words = ", ".join("0x%08xu" % word for word in words)
            check("manifest uses byte base %#x" % base, "srv 0 bvh base %#x size 0xb" % base in manifest, manifest)
            check("call preserves all four descriptor DWORDs at %#x" % base, literal_words in call, call)

    # The model wraps the absolute node address at 48 bits. The raw-bound
    # helper cannot represent that wrap, so the translator must refuse it.
    words = (0xffffffff, SORT_ENABLE | 0xff, 0x0b, RESOURCE_TYPE)
    check("public model wraps the node address", B.node_address(words, 0x25) == (5, 0),
          str(B.node_address(words, 0x25)))
    pipe, boundary_hlsl, refusals = translate_program(node=0x15, word0=words[0], word1=words[1])
    check("node ending exactly at the 48-bit boundary translates",
          pipe is not None and not refusals, str(refusals))
    pipe, wrap_hlsl, refusals = translate_program(node=0x25, word0=words[0], word1=words[1])
    check("refuses 48-bit wrap before resource binding",
          refusals == ["image_bvh_intersect_ray node address wraps the 48-bit address space"]
          and " bvh base " not in pipe.manifest()
          and "x2w_bvh_intersect_box32" not in wrap_hlsl and "uint4 bt" not in wrap_hlsl, str(refusals))

    # Portable checks ensure the actual emitted helper and every negative
    # mutation are present even when a GPU dispatch is unavailable.
    helpers = "\n".join(G.BVH_HELPERS)
    check("translated shader embeds the dispatch-tested helper", hlsl is not None and helpers in hlsl, "")
    for mutation in MUTATIONS:
        dispatch_hlsl(mutation)

    # Every refusal must happen before a helper call or a result VGPR is emitted.
    cases = [
        ("triangle node", dict(node=0x00), "triangle node (kind 0)"),
        ("16-bit box node", dict(node=0x04), "16-bit box node (kind 4)"),
        ("unknown node kind", dict(node=0x06), "node kind 6"),
        ("unaligned node", dict(node=0x0d), "not an aligned 128-byte node"),
        ("non-constant node", dict(constant_node=False), "non-constant node base"),
        ("64-bit form", dict(opcode="image_bvh64_intersect_ray"), "image_bvh64_intersect_ray"),
        ("A16", dict(modifier="a16"), "modifier a16"),
        ("NSA address", dict(modifier=""), "non-sequential"),
        ("loaded descriptor", dict(loaded_descriptor=True), "loaded at run time"),
        ("descriptor type", dict(word3=0x60000000), "descriptor type 6 is not 8"),
        ("box growing", dict(word1=SORT_ENABLE | (2 << 23)), "box growing amount 2"),
        ("reserved word 1", dict(word1=SORT_ENABLE | 0x100), "reserved word 1"),
        ("size zero", dict(word2=0), "does not cover a 128-byte node"),
        ("node outside size", dict(node=0x25, word2=3), "outside descriptor size"),
        ("absolute base", dict(word0=0, word1=0), "absolute node base"),
        ("size high bits", dict(word3=RESOURCE_TYPE | 0x100), "size above 4 GiB"),
    ]
    for label, kwargs, reason in cases:
        if label == "NSA address":
            # A non-sequential address is a text form the packed decoder never prints.
            pipe, hlsl, refusals = None, None, []
            try:
                G.Stage.op_image_bvh_intersect_ray(
                    type("P", (), {"kind": "cs", "const_u32": lambda self, tok: 5})(),
                    ("v[32:35]", "[v0, v2, v4, v6, v8, v10, v12, v14, v16, v18, v20]", "s[12:15]"), [])
            except G.Unsupported as error:
                refusals = [str(error)]
        else:
            pipe, hlsl, refusals = translate_program(**kwargs)
        ok = len(refusals) == 1 and reason in refusals[0]
        check("refuses %s" % label, ok, str(refusals))
        if hlsl is not None:
            check("  no partial output for %s" % label,
                  "uint4 bt" not in hlsl and "x2w_bvh_intersect_box32(r" not in hlsl, "")

    pipe, hlsl, refusals = translate_program(stage_kind="ps")
    check("refuses a pixel stage", len(refusals) == 1 and "outside a compute stage" in refusals[0], str(refusals))


def find_dxc():
    named = os.environ.get("X2W_DXC")
    if named and Path(named).is_file():
        return str(Path(named))
    found = shutil.which("dxc")
    if found:
        return found
    kits = os.environ.get("ProgramFiles(x86)", "")
    if kits:
        candidates = sorted((Path(kits) / "Windows Kits" / "10" / "bin").glob("*/x64/dxc.exe"), reverse=True)
        if candidates:
            return str(candidates[0])
    return None


def compile_check(hlsl):
    dxc = find_dxc()
    if dxc is None:
        print("dxc compile: skip (no dxc on this host)")
        return None
    with tempfile.TemporaryDirectory(dir=ROOT / "out" if (ROOT / "out").is_dir() else None) as temporary:
        source = Path(temporary) / "bvh_box32.hlsl"
        source.write_text(hlsl, encoding="ascii")
        result = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", str(source.with_suffix(".cso")), str(source)],
                                capture_output=True, text=True)
    if result.returncode or result.stderr:
        check("dxc compiles the translated helper", False, result.stderr or "exit %d" % result.returncode)
        return False
    print("dxc compile: ok (%s)" % dxc)
    return True




def ffloat(value):
    return "asfloat(0x%08xu)" % struct.unpack("<I", struct.pack("<f", value))[0]


def fbits(value):
    return "0x%08xu" % struct.unpack("<I", struct.pack("<f", value))[0]


MUTATIONS = {
    "miss_fill": ("all_miss", tuple(
        ("c%d = h%d ? c%d : X2W_BVH_MISS;" % (i, i, i), "c%d = c%d;" % (i, i)) for i in range(4))),
    "compact": ("unsorted_three_hits", (
        ("if (!sort_boxes) return uint4(c0, c1, c2, c3);",
         "if (!sort_boxes) { uint4 ordered = miss; uint n = 0u; "
         "if (h0) ordered[n++] = c0; if (h1) ordered[n++] = c1; "
         "if (h2) ordered[n++] = c2; if (h3) ordered[n++] = c3; return ordered; }"),)),
    "finite_miss": ("finite_max", tuple(
        ("float k%d = h%d ? d%d : asfloat(0x7f800000u);" % (i, i, i),
         "float k%d = h%d ? d%d : 3.402823466e+38;" % (i, i, i)) for i in range(4))),
    "base_alignment": ("encoded_base_low_bits", (
        ("return (word1 & 0x7fffff00u) == 0u &&",
         "return (word0 & 0xffu) == 0u && (word1 & 0x7fffff00u) == 0u &&"),)),
    "high_byte": ("encoded_base_high_byte", (
        ("(word0 != 0u || (word1 & 0x000000ffu) != 0u)", "(word0 != 0u)"),)),
}


def dispatch_hlsl(mutation=None):
    helpers = "\n".join(G.BVH_HELPERS).replace("ByteAddressBuffer", "RWByteAddressBuffer")
    if mutation:
        for old, new in MUTATIONS[mutation][1]:
            assert helpers.count(old) == 1, (mutation, old)
            helpers = helpers.replace(old, new)
    lines = [helpers, "RWByteAddressBuffer data : register(u0);", "[numthreads(1, 1, 1)]", "void main() {"]
    for base_offset, children in NODES.items():
        for i, (child, lo, hi) in enumerate(children):
            lines.append("    data.Store(%du, 0x%08xu);" % (base_offset + 4 * i, child))
            for j in range(3):
                lines.append("    data.Store(%du, %s);" % (base_offset + 16 + 24 * i + 4 * j, fbits(lo[j])))
                lines.append("    data.Store(%du, %s);" % (base_offset + 16 + 24 * i + 12 + 4 * j, fbits(hi[j])))
        for i in range(4):
            lines.append("    data.Store(%du, 0u);" % (base_offset + 112 + 4 * i))
    lines.append("    DeviceMemoryBarrierWithGroupSync();")
    for case, (_name, offset, extent, origin, direction, _expected) in enumerate(RAYS + UNSORTED_RAYS):
        inv = tuple(1.0 / c for c in direction)
        base0, base1 = DESCRIPTOR_BASES.get(_name, (0x1000, 0))
        word1 = base1 | (SORT_ENABLE if case < len(RAYS) else 0)
        lines.append(
            "    uint4 r%d = x2w_bvh_intersect_box32(data, 0x%08xu, 0x%08xu, 0x%08xu, "
            "0x0000000bu, 0x80000000u, %s, float3(%s, %s, %s), float3(%s, %s, %s));" % (
                case, (offset >> 3) | 5, base0, word1, ffloat(extent), ffloat(origin[0]), ffloat(origin[1]),
                ffloat(origin[2]), ffloat(inv[0]), ffloat(inv[1]), ffloat(inv[2])))
        for word in range(4):
            lines.append("    data.Store(%du, r%d.%s);" % (RESULT_OFFSET + 16 * case + 4 * word, case, "xyzw"[word]))
    lines.append("}")
    return "\n".join(lines) + "\n"


def run_dispatch(dxc, runner, adapter, directory, mutation=None):
    name = "bvh_dispatch_" + mutation if mutation else "bvh_dispatch"
    source = directory / (name + ".hlsl")
    cso = directory / (name + ".cso")
    output = directory / (name + ".bin")
    source.write_text(dispatch_hlsl(mutation), encoding="ascii")
    compiled = subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", str(cso), str(source)],
                              capture_output=True, text=True)
    if compiled.returncode or compiled.stderr:
        raise AssertionError(compiled.stderr or "dxc exited with %d" % compiled.returncode)
    result = subprocess.run([runner, adapter, "--raw-uav-dump", str(cso), "1", str(BUFFER_SIZE), str(output)],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77, None
    if result.returncode or result.stderr:
        raise AssertionError(result.stderr or "offscreen_check exited with %d" % result.returncode)
    data = output.read_bytes()
    assert len(data) == BUFFER_SIZE, (len(data), BUFFER_SIZE)
    return 0, struct.unpack("<%dI" % (len(data) // 4), data)


def check_dispatch(dxc, runner, adapter):
    controls()
    if failures:
        return 1
    if not dxc or not runner:
        print("dispatch: skip (no dxc or offscreen_check)")
        return 77
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        directory = Path(temporary)
        status, words = run_dispatch(dxc, runner, adapter, directory)
        if status == 77:
            print("dispatch: skip (adapter unavailable)")
            return 77
        cases = RAYS + UNSORTED_RAYS
        for case, (_name, _offset, _extent, _origin, _direction, expected) in enumerate(cases):
            start = RESULT_WORD + 4 * case
            got = tuple(words[start:start + 4])
            check("dispatch case %d %s" % (case, _name), got == expected,
                  "got %s expected %s" % (["%08x" % w for w in got], ["%08x" % w for w in expected]))
        for mutation, (witness, _replacements) in MUTATIONS.items():
            status, mutant = run_dispatch(dxc, runner, adapter, directory, mutation=mutation)
            if status == 77:
                print("dispatch control: skip (adapter unavailable)")
                return 77
            case = next(i for i, ray in enumerate(cases) if ray[0] == witness)
            start = RESULT_WORD + 4 * case
            changed = tuple(mutant[start:start + 4])
            check("dispatch %s control changes %s" % (mutation, witness), changed != cases[case][-1],
                  "got %s" % ["%08x" % w for w in changed])
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("dispatch: all ok on %s" % adapter)
    return 0

def main():
    if len(sys.argv) == 5 and sys.argv[1] == "dispatch":
        return check_dispatch(sys.argv[2], sys.argv[3], sys.argv[4])
    if len(sys.argv) != 1:
        sys.exit(__doc__)
    controls()
    pipe, hlsl, refusals = translate_program()
    assert not refusals, refusals
    compile_check(hlsl)
    if failures:
        print("FAILED: %s" % ", ".join(failures))
        return 1
    print("startup gfx2hlsl bvh box32: all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
