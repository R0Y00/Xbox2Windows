"""Check gfx1013 DWORD buffer atomics against a raw UAV readback."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from compute_store_check import aot, assemble, disassemble, rejects, store_container, variant

HERE = Path(__file__).resolve().parent
ADD = (HERE / "compute_buffer_atomic_add.s").read_text(encoding="ascii").splitlines()
CAS = (HERE / "compute_buffer_atomic_cmpswap.s").read_text(encoding="ascii").splitlines()
ADD_ATOMIC = next(line for line in ADD if line.startswith("buffer_atomic_add"))
ADD_RECORD = next(line for line in ADD if line.startswith("buffer_store_dword"))
SHIFT = "v_lshlrev_b32_e32 v2, 3, v0"
WAIT = "s_waitcnt vmcnt(0)"
INDEX_RECORD = ADD_RECORD.replace("v2,", "v0,").replace("offen", "idxen")
MAX_ATOMIC = "buffer_atomic_umax v4, v0, s[24:27], 0 idxen glc"
MAX = variant(ADD, (SHIFT, ""), ("v_add_nc_u32_e32 v4, 1, v0", "v_mov_b32_e32 v4, 0x80012345"),
              (ADD_ATOMIC, MAX_ATOMIC), (ADD_RECORD, INDEX_RECORD))
MAX_NO_GLC = variant(MAX, (MAX_ATOMIC, MAX_ATOMIC.removesuffix(" glc")), (WAIT, ""), (INDEX_RECORD, ""))
SHARED = variant(ADD, (SHIFT, ""), (ADD_ATOMIC, "buffer_atomic_add v4, off, s[24:27], 0 glc"),
                 (ADD_RECORD, INDEX_RECORD))
PREFIX = "070394be 800395be 0a0608f4300000fa 000046d70a0a0104 "
CASES = {
    "add_offen": (ADD, 8, "83000434 8100084a 7fc08cbf 0050c8e002040680 703f8cbf 041070e002040680"),
    "umax_idxen": (MAX, 8, "ff02087e45230180 7fc08cbf 0060e0e000040680 703f8cbf 042070e000040680"),
    "umax_no_glc": (MAX_NO_GLC, 8, "ff02087e45230180 7fc08cbf 0020e0e000040680"),
    "shared_add": (SHARED, 8, "8100084a 7fc08cbf 0040c8e000040680 703f8cbf 042070e000040680"),
    "cmpswap": (CAS, 16, "ff00083800003412 050069d500090100 810a0a4a 050069d505ff0100b179379e "
                 "810a0a38 81000c36 060b0a3a 7fc08cbf 0060c4e000040680 703f8cbf 042070e000040680 082070e000050680"),
}
GROUP = (32, 1, 1)
LANES = GROUP[0]
DWORDS = 320
WORD_MASK = 0xFFFFFFFF
BINDING = dict(table_reg=7, group_reg=10, descriptor_offset=0x30)


def controls():
    rejected = (
        (ADD_ATOMIC, ADD_ATOMIC.removesuffix(" glc"), "Store data is not a known word"),
        (WAIT, "", "before s_waitcnt vmcnt(0)"),
        (WAIT, "s_waitcnt lgkmcnt(0)", "before s_waitcnt vmcnt(0)"),
        (WAIT, "v_mov_b32_e32 v8, v4\n" + WAIT, "before s_waitcnt vmcnt(0)"),
        (ADD_RECORD, ADD_RECORD.removesuffix(" offset:4"), "Overlapping raw UAV stores"),
        (ADD_ATOMIC, ADD_ATOMIC.replace("v2,", "v0,"), "Unverified per-thread atomic address"),
        (SHIFT, SHIFT.replace(", 3,", ", 31,"), "Unverified per-thread atomic address"),
    )
    for old, new, message in rejected:
        rejects(message, aot.lower, variant(ADD, (old, new)), GROUP, **BINDING, stride=8)
    # Every other 32-bit MUBUF atomic the gfx1030 profile knows stays refused on gfx1013.
    refused_ops = sorted(set(aot.MUBUF_ATOMIC_METHODS) - {"add", "umax", "cmpswap"})
    assert {"sub", "smin", "umin", "smax", "and", "or", "xor", "swap", "inc", "dec", "csub"} <= set(refused_ops)
    for op in refused_ops:
        rejects("require 32-bit add, umax or cmpswap", aot.lower,
                variant(ADD, (ADD_ATOMIC, ADD_ATOMIC.replace("atomic_add", f"atomic_{op}"))),
                GROUP, **BINDING, stride=8)
    # The 64-bit forms of the accepted ops, with every data VGPR defined, so only the width gate can refuse them.
    defined = "v_add_nc_u32_e32 v4, 1, v0\n" + "\n".join(f"v_mov_b32_e32 v{r}, {r}" for r in (5, 6, 7))
    for op, data in (("add", "v[4:5]"), ("umax", "v[4:5]"), ("cmpswap", "v[4:7]")):
        rejects("require 32-bit add, umax or cmpswap", aot.lower,
                variant(ADD, ("v_add_nc_u32_e32 v4, 1, v0", defined),
                        (ADD_ATOMIC, ADD_ATOMIC.replace("buffer_atomic_add v4", f"buffer_atomic_{op}_x2 {data}"))),
                GROUP, **BINDING, stride=8)
    srv_atomic =variant(ADD, (ADD_ATOMIC, ADD_ATOMIC.replace("s[24:27]", "s[28:31]")),
                         ("s_waitcnt lgkmcnt(0)", "s_load_dwordx4 s[28:31], s[20:21], 0x40\ns_waitcnt lgkmcnt(0)"))
    rejects("Unverified atomic reduction descriptor", aot.lower, srv_atomic, GROUP,
            **BINDING, stride=8, source_descriptor=0x40)
    rejects("Return record can overlap the fixed atomic address", aot.lower,
            variant(SHARED, (INDEX_RECORD, INDEX_RECORD.removesuffix(" offset:4"))), GROUP, **BINDING, stride=8)
    # The emitted text pins what only the Windows draw would otherwise show.
    lowered = {name: aot.lower(lines, GROUP, **BINDING, stride=stride) for name, (lines, stride, _) in CASES.items()}
    # cmpswap: VDATA (v4) is the replacement and VDATA+1 (v5) the compare; HLSL takes the compare first.
    compare = "((0x00000001u & tid.x) ^ (0x00000001u | ((0x00000001u + (tid.x * 0x00000004u)) * 0x9e3779b1u)))"
    assert (f"outputData.InterlockedCompareExchange(tid.x * 16u + 0u, {compare}, (0x12340000u | tid.x), atomicOld);"
            in lowered["cmpswap"]), lowered["cmpswap"]
    # glc returns the old value, and that value is what the record stores.
    assert "outputData.InterlockedAdd((tid.x << 3u) + 0u, (0x00000001u + tid.x), atomicOld);" in lowered["add_offen"]
    assert "outputData.Store((tid.x << 3u) + 4u, atomicOld);" in lowered["add_offen"]
    # umax is the unsigned InterlockedMax, with and without glc.
    assert ("    uint atomicOld;\n    outputData.InterlockedMax(tid.x * 8u + 0u, 0x80012345u, atomicOld);"
            in lowered["umax_idxen"]), lowered["umax_idxen"]
    assert ("    uint ignoredAtomic;\n    outputData.InterlockedMax(tid.x * 8u + 0u, 0x80012345u, ignoredAtomic);"
            in lowered["umax_no_glc"]), lowered["umax_no_glc"]
    # Without glc nothing is returned.
    assert "atomicOld" not in lowered["umax_no_glc"]
    # A nonzero per-thread offset reaches the address, here with the record moved to offset 0.
    offset = aot.lower(variant(ADD, (ADD_ATOMIC, ADD_ATOMIC.replace("0 offen glc", "0 offen offset:4 glc")),
                               (ADD_RECORD, ADD_RECORD.replace(" offset:4", ""))), GROUP, **BINDING, stride=8)
    assert "outputData.InterlockedAdd((tid.x << 3u) + 4u, (0x00000001u + tid.x), atomicOld);" in offset, offset
    assert "outputData.Store((tid.x << 3u) + 0u, atomicOld);" in offset, offset
    # A lane guard before or after the atomic is refused.
    guard = "v_cmp_gt_u32_e32 vcc_lo, 16, v0\ns_and_saveexec_b32 s30, vcc_lo"
    rejects("gfx1013 requires an unguarded raw-buffer program", aot.lower,
            variant(ADD, (ADD_ATOMIC, guard + "\n" + ADD_ATOMIC)), GROUP, **BINDING, stride=8)
    rejects("gfx1013 buffer atomics require an unguarded program", aot.lower,
            variant(ADD, (ADD_ATOMIC, ADD_ATOMIC + "\n" + guard)), GROUP, **BINDING, stride=8)
    check_serialization_control()
    print("gfx1013 atomic translation and serialization controls passed", flush=True)


def binary(directory, assembler, objcopy, library, name, lines, stride, recorded, source):
    code = assemble(directory, assembler, objcopy, lines)
    assert code == bytes.fromhex(PREFIX + recorded + " 000081bf"), f"{name}: machine words differ ({code.hex()})"
    print(f"{name}: gfx1013 machine words match llvm-mc", flush=True)
    if not library:
        return
    decoded = aot.labelled(disassemble(code, library, offsets=True, cpu=b"gfx1013"))
    assert aot.lower(decoded, GROUP, **BINDING, stride=stride) == source
    container, output = directory / f"{name}.dxbc", directory / f"{name}-cli.hlsl"
    container.write_bytes(store_container(code, GROUP))
    options = [arg for key, value in (BINDING | {"stride": stride}).items()
               for arg in ("--" + key.replace("_", "-"), str(value))]
    subprocess.run([sys.executable, "-B", aot.__file__, container, "--output", output, "--llvm", library,
                    *options], check=True)
    assert output.read_text() == source


def fill_word(index):
    # The raw UAV dump harness starts with a distinct deterministic sentinel per word.
    return (((index + 1) * 0x9E3779B1) | 1) & WORD_MASK


def reference(name):
    words = [fill_word(index) for index in range(DWORDS)]
    if name == "shared_add":
        words[0] += sum(range(1, LANES + 1))
        words[1:2 * LANES:2] = [None] * LANES  # checked by the serialization oracle
        return words
    for lane in range(LANES):
        index = lane * (4 if name == "cmpswap" else 2)
        old = words[index]
        if name == "cmpswap":
            compare = old ^ (lane & 1)
            words[index] = (0x12340000 | lane) if compare == old else old
            words[index + 2] = compare  # VDATA+1 retains its input after the GLC result.
        elif name == "add_offen":
            words[index] = (old + lane + 1) & WORD_MASK
        else:
            words[index] = max(old, 0x80012345)
        if name != "umax_no_glc":
            words[index + 1] = old
    return words


def check_serialization(old_values, final):
    # Each lane adds a different strictly positive value and the total cannot wrap.
    # Sorting the (old value, lane operand) pairs therefore gives the only possible
    # order of this observed trace. Reconstruct each edge rather than assuming lane order.
    current = fill_word(0)
    for old, operand in sorted((old, lane + 1) for lane, old in enumerate(old_values)):
        assert old == current, f"shared add trace breaks at operand {operand}: {old:08x} != {current:08x}"
        current += operand
    assert current == final, f"shared add trace ends at {current:08x}, UAV has {final:08x}"


def check_serialization_control():
    old_values, current = [0] * LANES, fill_word(0)
    for lane in reversed(range(LANES)):
        old_values[lane] = current
        current += lane + 1
    check_serialization(old_values, current)
    # The sorted old values still look the same, but two now belong to the wrong lanes.
    old_values[0], old_values[1] = old_values[1], old_values[0]
    try:
        check_serialization(old_values, current)
    except AssertionError:
        return
    raise AssertionError("The serialization oracle accepted a scrambled lane/operand pair")


def draw(directory, dxc, renderer, adapter, name, source):
    expected = reference(name)
    variants = [("translated", source, None)]
    if name == "add_offen":
        variants += [
            ("add lowered as OR", source.replace("InterlockedAdd", "InterlockedOr"),
             (0, 0x9E3779B2, 0x9E3779B1)),
            ("record contains the addend", source.replace("+ 4u, atomicOld);", "+ 4u, tid.x + 1u);"),
             (1, 0x9E3779B1, 1)),
        ]
    hlsl, cso, dump = (directory / f"{name}.{ext}" for ext in ("hlsl", "cso", "bin"))
    for label, code, predicted in variants:
        assert (code == source) == (predicted is None), label
        hlsl.write_text(code, encoding="ascii")
        subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        result = subprocess.run([renderer, adapter, "--raw-uav-dump", cso, "1", str(4 * DWORDS), dump],
                                capture_output=True, text=True, timeout=15)
        print(f"{name} {label}:\n{result.stdout}{result.stderr}", end="", flush=True)
        if result.returncode == 77:
            return 77
        result.check_returncode()
        if "debug_layer=enabled" in result.stdout:
            assert "debug_errors=0 debug_warnings=0" in result.stdout, name
        actual = struct.unpack(f"<{DWORDS}I", dump.read_bytes())
        first = next(((i, want, got) for i, (want, got) in enumerate(zip(expected, actual, strict=True))
                      if want is not None and want != got), None)
        assert first == predicted, f"{name} {label}: first mismatch {first}, expected {predicted}"
        if predicted is not None:
            index, want, got = first
            print(f"control caught at word {index}: expected={want:08x} actual={got:08x}", flush=True)
            continue
        if name == "shared_add":
            check_serialization(actual[1:2 * LANES:2], actual[0])
            print(f"shared add: 32 old values serialize from {fill_word(0):08x} to {actual[0]:08x}", flush=True)
        print(f"{name}: compared all {DWORDS} words", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--binary", nargs=2, metavar=("LLVM_MC", "LLVM_OBJCOPY"))
    modes.add_argument("--draw", nargs=3, metavar=("DXC", "RENDERER", "ADAPTER"))
    parser.add_argument("--llvm", metavar="LIBLLVM", help="Also check disassembly and the container CLI")
    args = parser.parse_args()
    controls()
    if not (args.binary or args.draw):
        for lines, stride, _ in CASES.values():
            aot.lower(lines, GROUP, **BINDING, stride=stride)
        return 0
    out = HERE.parent / "out"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compute_buffer_atomics_", dir=out) as tmp:
        directory = Path(tmp)
        for name, (lines, stride, words) in CASES.items():
            source = aot.lower(lines, GROUP, **BINDING, stride=stride)
            if args.binary:
                binary(directory, *args.binary, args.llvm, name, lines, stride, words, source)
            elif draw(directory, *args.draw, name, source) == 77:
                return 77
    if args.draw and (record := os.environ.get("FORM_COVERAGE_RECORD")):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        accepted = {line.split()[0] for lines, _, _ in CASES.values() for line in lines}
        path.write_text("".join(f"{op}\n" for op in sorted(accepted)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
