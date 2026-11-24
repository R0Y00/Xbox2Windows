"""Batch N0 preflight: the local gfx1013 compute translator on native machine code (#1412).

Fixture only. No runtime code, no slot 67 handler, no native cache, no change to the frozen slot 62
slice. Two questions:

1. Does `compute_store_aot.lower()` lower a two-dimensional guarded read-compute-write program?
2. Does the *assembled* machine code decode back to the same program and lower identically?

The LLVM stages run inside Linux (WSL) because no LLVM exists on Windows here:

    python3 tests/compute_native_check.py --linux-worker --out <dir> --program both

The Windows stages compile, validate and probe the same HLSL:

    out\\pe-env\\Scripts\\python.exe -B tests/compute_native_check.py --out out\\native-compute\\preflight `
        --wsl-distro Ubuntu-22.04 --dxc <dxc.exe> --root-probe root_probe.exe --device-probe root_device_probe.exe

Stages: translate, controls, reference, assemble, disassemble, compile, validate, probe.
Exit codes: 0 all requested stages passed, 77 a capability skip, 1 a failure.
"""
import argparse
import json
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import compute_store_aot as aot  # noqa: E402
from pixel_aot import disassemble  # noqa: E402
from resource_inventory import parts  # noqa: E402

SKIP = 77
LLVM15_MC = "/usr/lib/llvm-15/bin/llvm-mc"
LLVM15_OBJCOPY = "/usr/lib/llvm-15/bin/llvm-objcopy"
LLVM15_LIB = "/usr/lib/llvm-15/lib/libLLVM-15.so.1"

# Stage 3J's proposed program, kept verbatim as the control that the local translator refuses: the
# lane guard compares against a linear index that v_mad_u32_u24 computed, and a guard operand must be
# a dispatch-thread word, not an ALU result.
STAGE3J_LINEAR = [
    "s_mov_b32 s1, 0",
    "s_load_dwordx4 s[4:7], s[0:1], 0x20",
    "v_lshl_add_u32 v0, s2, 3, v0",
    "v_lshl_add_u32 v1, s3, 2, v1",
    "v_mad_u32_u24 v0, v1, 24, v0",
    "s_waitcnt lgkmcnt(0)",
    "s_buffer_load_dword s16, s[4:7], 0x0",
    "s_buffer_load_dword s17, s[4:7], 0x4",
    "s_waitcnt lgkmcnt(0)",
    "s_load_dwordx8 s[8:15], s[0:1], 0x0",
    "v_cmp_gt_u32_e32 vcc_lo, s16, v0",
    "s_and_saveexec_b32 s2, vcc_lo",
    "s_cbranch_execz .LBB0_2",
    "v_lshlrev_b32_e32 v2, 2, v0",
    "s_waitcnt lgkmcnt(0)",
    "buffer_load_dword v1, v2, s[12:15], 0 offen",
    "s_waitcnt vmcnt(0)",
    "v_and_b32_e32 v3, 0xffff, v1",
    "v_lshrrev_b32_e32 v1, 16, v1",
    "v_mad_u32_u24 v1, v3, v1, v0",
    "v_min_u32_e32 v1, s17, v1",
    "buffer_store_dword v1, v0, s[8:11], 0 idxen",
    ".LBB0_2:",
    "s_endpgm",
]

# The corrected two-dimensional program. The store uses the translator's proven two-axis layout
# (idxen = row, offen = column bytes), so the linear index the refused program computed is not needed.
# The guard is the column bound, which is the only kind of lane guard the translator verifies.
PROGRAM_2D = [
    "s_mov_b32 s1, 0",
    "s_load_dwordx4 s[4:7], s[0:1], 0x20",
    "v_lshl_add_u32 v0, s2, 3, v0",
    "v_lshl_add_u32 v1, s3, 2, v1",
    "s_waitcnt lgkmcnt(0)",
    "s_buffer_load_dword s16, s[4:7], 0x0",
    "s_buffer_load_dword s17, s[4:7], 0x4",
    "s_waitcnt lgkmcnt(0)",
    "s_load_dwordx8 s[8:15], s[0:1], 0x0",
    "v_lshlrev_b32_e32 v2, 2, v0",
    "v_cmp_gt_u32_e32 vcc_lo, s16, v0",
    "s_and_saveexec_b32 s2, vcc_lo",
    "s_cbranch_execz .LBB0_2",
    "s_waitcnt lgkmcnt(0)",
    "buffer_load_dword v3, v2, s[12:15], 0 offen",
    "s_waitcnt vmcnt(0)",
    "v_and_b32_e32 v4, 0xffff, v3",
    "v_lshrrev_b32_e32 v3, 16, v3",
    "v_mad_u32_u24 v3, v4, v3, v0",
    "v_min_u32_e32 v3, s17, v3",
    "buffer_store_dword v3, v[1:2], s[8:11], 0 idxen offen",
    ".LBB0_2:",
    "s_endpgm",
]

GROUP_2D = (8, 4, 1)
DISPATCH_2D = (3, 2, 1)
STRIDE_2D = 96            # 24 words per row of the 24x8 dispatch
BIND_2D = dict(table_reg=0, descriptor_offset=0, constant_descriptor=32, source_descriptor=16,
               stride=STRIDE_2D, group_reg=2, group_reg_y=3)

GROUP_1D = (32, 1, 1)
DISPATCH_1D = (6, 1, 1)
CONSTANTS = 16            # RootConstants(num32BitConstants=16,b0)
CAP = 100000
SENTINEL = 0xC5C5C5C5
WORDS = 320               # UAV size; the kernel writes at most words 0..191

ROOT_SIGNATURE = 'SRV(t0),UAV(u0),RootConstants(num32BitConstants=16,b0)'


def read_compute_1d():
    """The frozen fixture's guarded read-compute-write program and binding, imported, not copied."""
    sys.path.insert(0, str(ROOT / "tests"))
    import compute_store_check as frozen

    return frozen.READ_COMPUTE, frozen.READ_BINDING


def source_word(index):
    """The fixture's input: a 16-bit low half and a small high half, so the product never wraps."""
    return ((0x1234 + 0x101 * index) & 0xFFFF) | ((1 + index % 7) << 16)


def reference_1d(active, cap=CAP):
    """Words for the 1-D program: out[i] = min(cap, (low * high + i) mod 2^32)."""
    words = [SENTINEL] * WORDS
    for i in range(active):
        value = source_word(i)
        product = (value & 0xFFFF) * (value >> 16)
        words[i] = min(cap, (product + i) & 0xFFFFFFFF)
    return words


def row_index(row, column):
    return row * 24 + column


def reference_2d(columns, cap=CAP):
    """Words for the 2-D program: out[row*24 + col] = min(cap, low(col) * high(col) + col).

    The read is a column broadcast: the translator's load has no row term, so every row reads the
    same 24 source words. That is the documented limitation of this fixture.
    """
    words = [SENTINEL] * WORDS
    for row in range(DISPATCH_2D[1] * GROUP_2D[1]):
        for col in range(DISPATCH_2D[0] * GROUP_2D[0]):
            if col >= columns:
                continue
            value = source_word(col)
            product = (value & 0xFFFF) * (value >> 16)
            words[row * 24 + col] = min(cap, (product + col) & 0xFFFFFFFF)
    return words


def page(text):
    return text if text.endswith("\n") else text + "\n"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page(text), encoding="ascii")
    return path


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def translate_text(out):
    """Stage: lower both programs from authored text and check the emitted contract."""
    result = {}
    source = aot.lower(PROGRAM_2D, GROUP_2D, **BIND_2D)
    write(out / "native_tile_2d.hlsl", source)
    write(out / "native_tile_2d.s", "\n".join(PROGRAM_2D))
    (out / "native_tile_2d.json").write_text(json.dumps({
        "program": "native_tile_2d",
        "assembled_from": "source-built gfx1013 assembly (tools/compute_store_aot.py lower())",
        "group": list(GROUP_2D),
        "dispatch": list(DISPATCH_2D),
        "threads": DISPATCH_2D[0] * DISPATCH_2D[1] * GROUP_2D[0] * GROUP_2D[1],
        "binding": BIND_2D,
        "constants": CONSTANTS,
        "root_signature": ROOT_SIGNATURE,
        "text_sha256": hashlib.sha256("\n".join(PROGRAM_2D).encode("ascii")).hexdigest(),
        "hlsl_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "hlsl_lines": len(source.splitlines()),
    }, indent=2, sort_keys=True) + "\n", encoding="ascii")

    required = [
        "cbuffer Constants : register(b0) { uint4 params[4]; };",
        "ByteAddressBuffer inputData : register(t0);",
        "RWByteAddressBuffer outputData : register(u0);",
        "[numthreads(8,4,1)]",
        "void main(uint3 tid : SV_DispatchThreadID)",
        "if (params[0].x > tid.x) {",
        "outputData.Store(tid.y * 96u + (tid.x << 2u) + 0u,",
        "inputData.Load((tid.x << 2u) + 0u)",
        "min(params[0].y,",
        "(0x00000010u & 31u)",
        "& 0xffffffu",           # v_mad_u32_u24 keeps its 24-bit operand masks
    ]
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        raise AssertionError(f"generated HLSL is missing {missing}:\n{source}")
    result["2d"] = source

    one, binding = read_compute_1d()
    fallback = aot.lower(one, GROUP_1D[0], **binding)
    write(out / "native_tile_1d.hlsl", fallback)
    if "[numthreads(32,1,1)]" not in fallback or "outputData.Store" not in fallback:
        raise AssertionError(f"1-D fallback did not lower as expected:\n{fallback}")
    result["1d"] = fallback
    return result


def controls():
    """Refusals must be explicit and must name the reason, not fall back to a dummy output."""
    cases = [
        ("stage3j linear index", STAGE3J_LINEAR, GROUP_2D, BIND_2D, "Unverified lane guard compare"),
        ("truncated stream", PROGRAM_2D[:-1], GROUP_2D, BIND_2D, "Missing final program exit"),
        ("LDS opcode", [line.replace("v_min_u32_e32 v3, s17, v3", "ds_read_b32 v3, v0")
                        for line in PROGRAM_2D], GROUP_2D, BIND_2D, "Unverified private LDS read"),
        ("missing source binding", PROGRAM_2D, GROUP_2D,
         dict(BIND_2D, source_descriptor=None), "Unresolved compute descriptor load"),
        ("wrong table register", PROGRAM_2D, GROUP_2D,
         dict(BIND_2D, table_reg=6), "Unresolved compute descriptor load"),
        ("group too large", PROGRAM_2D, (64, 32, 1), BIND_2D, "Unsupported thread group"),
        ("group not a power of two", PROGRAM_2D, (6, 4, 1), BIND_2D, "Unverified dispatch index"),
    ]
    results = []
    for label, program, group, binding, message in cases:
        try:
            aot.lower(program, group, **binding)
        except ValueError as error:
            if message not in str(error):
                raise AssertionError(f"{label}: expected {message!r}, got {error!r}") from error
            results.append((label, message, str(error)))
        else:
            raise AssertionError(f"{label}: the translator accepted an unsupported program")
    # The translator now accepts COMPUTE_PGM_RSRC2, but this adapter records the field without
    # interpreting it: the profile below names its group registers explicitly. Both upstream
    # refusals that keep that contract reachable are checked, because neither the record schema
    # nor a group dimension may silently come from the field.
    rsrc2_cases = [
        ("rsrc2 with explicit group registers", dict(BIND_2D, rsrc2=0x984),
         "COMPUTE_PGM_RSRC2 places the group IDs"),
        ("rsrc2 TG_SIZE_EN", dict(BIND_2D, rsrc2=0x400, group_reg=None, group_reg_y=None),
         "COMPUTE_PGM_RSRC2 TG_SIZE_EN is not supported"),
    ]
    for label, binding, message in rsrc2_cases:
        try:
            aot.lower(PROGRAM_2D, GROUP_2D, **binding)
        except ValueError as error:
            if message not in str(error):
                raise AssertionError(f"{label}: expected {message!r}, got {error!r}") from error
            results.append((label, message, str(error)))
        else:
            raise AssertionError(f"{label}: the translator accepted the unsupported binding")
    return results


# ---------------------------------------------------------------- Linux (LLVM) worker

def llvm_version(mc):
    result = run([mc, "--version"])
    return result.stdout.splitlines()[0].strip() if result.returncode == 0 else "unknown"


def assemble(program, mc, objcopy, stem):
    """Source-built machine code through llvm-mc, exactly as the repository's fixtures do."""
    source, obj, raw = stem.with_suffix(".s"), stem.with_suffix(".o"), stem.with_suffix(".bin")
    source.write_text("\n".join(program) + "\n", encoding="ascii")
    subprocess.run([mc, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj",
                    str(source), "-o", str(obj)], check=True, capture_output=True, text=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", str(obj), str(raw)],
                   check=True, capture_output=True, text=True)
    return raw.read_bytes()


def decode(code, library):
    """Machine bytes to (offset, text) pairs, rejecting an unknown encoding."""
    return disassemble(code, library, offsets=True, cpu=b"gfx1013")


def boundary_check(code, decoded):
    """Every instruction is 4-byte aligned with a 4- or 8-byte encoding, and covers all bytes."""
    if not decoded or decoded[0][0] != 0:
        raise AssertionError("decoding does not start at byte 0")
    offsets = [where for where, _ in decoded]
    sizes = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    sizes.append(len(code) - offsets[-1])
    for where, size in zip(offsets, sizes):
        if where % 4 or size not in (4, 8):
            raise AssertionError(f"instruction at byte {where} has encoding size {size}")
    if sum(sizes) != len(code):
        raise AssertionError(f"decoded {sum(sizes)} bytes of {len(code)}")
    return sizes


def branch_check(decoded):
    """The single conditional branch must target the final s_endpgm offset."""
    branch = [(where, text) for where, text in decoded if text.startswith("s_cbranch_execz")]
    if len(branch) != 1:
        raise AssertionError(f"expected one s_cbranch_execz, found {len(branch)}")
    where, text = branch[0]
    operand = text.split()[-1]
    try:
        # gfx10 prints the target as a signed dword offset from the next instruction.
        target = where + 4 + 4 * int(operand, 0)
    except ValueError as error:
        raise AssertionError(f"branch operand is not a relative offset: {text!r}") from error
    end = [at for at, line in decoded if line == "s_endpgm"]
    if not end:
        raise AssertionError("no s_endpgm in the decoded program")
    if target <= where or target != end[-1]:
        raise AssertionError(f"branch target {target} is not the forward exit at {end[-1]}")
    return {"at": where, "operand": int(operand, 0), "target": target, "exit": end[-1]}


def worker_program(out, name, program, group, binding, mc, objcopy, library):
    stem = out / f"native_{name}"
    code = assemble(program, mc, objcopy, stem)
    (out / f"native_{name}.sha256").write_text(hashlib.sha256(code).hexdigest() + "\n", encoding="ascii")
    decoded = decode(code, library)
    sizes = boundary_check(code, decoded)
    branch = branch_check(decoded)
    (out / f"native_{name}.disasm.txt").write_text(
        "\n".join(f"{where:08x}: {text}" for where, text in decoded) + "\n", encoding="ascii")
    text_hlsl = aot.lower(program, group, **binding)
    decoded_hlsl = aot.lower(aot.labelled(decoded), group, **binding)
    (out / f"native_{name}.text.hlsl").write_text(text_hlsl, encoding="ascii")
    (out / f"native_{name}.decoded.hlsl").write_text(decoded_hlsl, encoding="ascii")
    if text_hlsl != decoded_hlsl:
        raise AssertionError(f"{name}: the decoded program lowers to different HLSL")
    return {
        "bytes": len(code),
        "sha256": hashlib.sha256(code).hexdigest(),
        "words": len(code) // 4,
        "instructions": len(decoded),
        "encodings": {"4-byte": sizes.count(4), "8-byte": sizes.count(8)},
        "termination": decoded[-1][1],
        "branch": branch,
        "text_hlsl_sha256": hashlib.sha256(text_hlsl.encode("ascii")).hexdigest(),
        "decoded_hlsl_sha256": hashlib.sha256(decoded_hlsl.encode("ascii")).hexdigest(),
        "hlsl_identical": True,
    }


def worker_controls(out, mc, objcopy, library):
    """Byte-level refusals: the same decode -> lower path the accepted program takes."""
    stem = out / "control"
    results = []

    def attempt(label, code, group, binding, message):
        try:
            decoded = decode(code, library)
        except ValueError as error:
            actual = f"{type(error).__name__}: {error}"
        else:
            try:
                aot.lower(aot.labelled(decoded), group, **binding)
            except ValueError as error:
                actual = f"{type(error).__name__}: {error}"
            else:
                actual = "ACCEPTED"
        ok = message in actual
        results.append({"control": label, "expected": message, "actual": actual, "ok": ok})
        if not ok:
            raise AssertionError(f"{label}: expected {message!r}, got {actual!r}")

    good = assemble(PROGRAM_2D, mc, objcopy, stem.with_name("control_good"))
    attempt("truncated stream (exit removed)", good[:-4], GROUP_2D, BIND_2D,
            "Branch outside decoded instruction boundaries")
    attempt("truncated stream (partial instruction)", good[:-2], GROUP_2D, BIND_2D,
            "Unknown instruction at byte")
    attempt("invalid instruction encoding", bytes.fromhex("ffffffff") + good[4:], GROUP_2D, BIND_2D,
            "Unknown instruction at byte 0")
    lds = assemble([line.replace("v_min_u32_e32 v3, s17, v3", "ds_read_b32 v3, v0")
                    for line in PROGRAM_2D], mc, objcopy, stem.with_name("control_lds"))
    attempt("unsupported opcode", lds, GROUP_2D, BIND_2D, "Unverified private LDS read")
    no_exit = assemble(PROGRAM_2D[:-1], mc, objcopy, stem.with_name("control_noexit"))
    attempt("missing final exit", no_exit, GROUP_2D, BIND_2D,
            "Branch outside decoded instruction boundaries")
    linear = assemble(STAGE3J_LINEAR, mc, objcopy, stem.with_name("control_linear"))
    attempt("computed-index guard", linear, GROUP_2D, BIND_2D, "Unverified lane guard compare")

    # Binding and group refusals against the *decoded* text, not the authored text.
    decoded_good = aot.labelled(decode(good, library))
    for label, group, binding, message in (
            ("decoded: wrong table register", GROUP_2D, dict(BIND_2D, table_reg=6),
             "Unresolved compute descriptor load"),
            ("decoded: missing source binding", GROUP_2D, dict(BIND_2D, source_descriptor=None),
             "Unresolved compute descriptor load"),
            ("decoded: group too large", (64, 32, 1), BIND_2D, "Unsupported thread group")):
        try:
            aot.lower(decoded_good, group, **binding)
        except ValueError as error:
            actual = f"{type(error).__name__}: {error}"
        else:
            actual = "ACCEPTED"
        ok = message in actual
        results.append({"control": label, "expected": message, "actual": actual, "ok": ok})
        if not ok:
            raise AssertionError(f"{label}: expected {message!r}, got {actual!r}")
    return results


def linux_worker(argv):
    parser = argparse.ArgumentParser(prog="compute_native_check.py --linux-worker")
    parser.add_argument("--linux-worker", action="store_true", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--program", choices=("2d", "1d", "both"), default="both")
    parser.add_argument("--llvm-mc", default=LLVM15_MC)
    parser.add_argument("--objcopy", default=LLVM15_OBJCOPY)
    parser.add_argument("--llvm", default=LLVM15_LIB)
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "llvm": {"llvm_mc": args.llvm_mc, "objcopy": args.objcopy, "libLLVM": args.llvm,
                 "version": llvm_version(args.llvm_mc), "target": "amdgcn--amdpal", "cpu": "gfx1013"},
        "python": sys.version.split()[0],
    }
    if args.program in ("both", "2d"):
        report["2d"] = worker_program(out, "2d", PROGRAM_2D, GROUP_2D, BIND_2D,
                                      args.llvm_mc, args.objcopy, args.llvm)
    if args.program in ("both", "1d"):
        one, binding = read_compute_1d()
        report["1d"] = worker_program(out, "1d", one, GROUP_1D, binding,
                                      args.llvm_mc, args.objcopy, args.llvm)
    report["controls"] = worker_controls(out, args.llvm_mc, args.objcopy, args.llvm)
    (out / "machine.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                      encoding="ascii")
    print(f"linux worker: {report['llvm']['version']}, "
          f"2d={report.get('2d', {}).get('bytes', 0)} B, 1d={report.get('1d', {}).get('bytes', 0)} B, "
          f"{len(report['controls'])} byte controls")
    return 0


# ---------------------------------------------------------------- Windows stages

def to_wsl_path(path):
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}" + resolved.as_posix()[len(resolved.drive):]


def check_container(blob):
    if blob[:4] != b"DXBC":
        raise AssertionError(f"DXC output is not a DXBC container: {blob[:4]!r}")
    tags = [tag.decode("ascii", "replace") for tag in parts(blob)]
    if "DXIL" not in tags:
        raise AssertionError(f"DXC output has no DXIL part: {tags}")
    return tags


def compile_hlsl(dxc, source, output, directory, *, root):
    """Compile one HLSL file; with root=True, add the fixture root signature attribute."""
    path = directory / output.replace(".cso", ".hlsl")
    text = source
    if root:
        index = text.index("[numthreads(")
        text = text[:index] + f'[RootSignature("{ROOT_SIGNATURE}")]\n' + text[index:]
    path.write_text(text, encoding="ascii")
    result = run([str(dxc), "-T", "cs_6_0", "-E", "main", "-Fo", str(directory / output), str(path)])
    if result.returncode:
        raise AssertionError(f"dxc failed for {output}:\n{result.stdout}\n{result.stderr}")
    return (directory / output).read_bytes(), result.returncode


def compare_hlsl(out, machine_dir):
    """Decoded-code HLSL against the accepted text-mode HLSL; a newline difference must be the only one."""
    result = {}
    for name in ("2d", "1d"):
        text_path = out / f"native_tile_{name}.hlsl"
        decoded_path = machine_dir / f"native_{name}.decoded.hlsl"
        if not (text_path.exists() and decoded_path.exists()):
            continue
        text_bytes, decoded_bytes = text_path.read_bytes(), decoded_path.read_bytes()
        normalized_text = text_bytes.replace(b"\r\n", b"\n")
        normalized_decoded = decoded_bytes.replace(b"\r\n", b"\n")
        result[name] = {
            "text_bytes": len(text_bytes),
            "decoded_bytes": len(decoded_bytes),
            "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "decoded_sha256": hashlib.sha256(decoded_bytes).hexdigest(),
            "byte_identical": text_bytes == decoded_bytes,
            "line_endings_only": normalized_text == normalized_decoded and text_bytes != decoded_bytes,
            "normalized_sha256": hashlib.sha256(normalized_text).hexdigest(),
        }
        if not (text_bytes == decoded_bytes or normalized_text == normalized_decoded):
            raise AssertionError(f"{name}: the decoded program lowers to different HLSL")
    (out / "hlsl-comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                              encoding="ascii")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "native-compute" / "preflight")
    parser.add_argument("--linux-worker", action="store_true", help="run the LLVM stages in Linux")
    parser.add_argument("--program", choices=("2d", "1d", "both"), default="both")
    parser.add_argument("--wsl-distro", help="run the LLVM worker through this WSL distribution")
    parser.add_argument("--llvm-mc")
    parser.add_argument("--objcopy")
    parser.add_argument("--llvm")
    parser.add_argument("--dxc")
    parser.add_argument("--root-probe")
    parser.add_argument("--device-probe")
    parser.add_argument("--warp-only", action="store_true")
    args = parser.parse_args(argv)

    if args.linux_worker:
        return linux_worker(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "command.txt").write_text(
        "python -B tests/compute_native_check.py " + " ".join(sys.argv[1:]) + "\n", encoding="ascii")
    skips, failures = [], []
    summary = []

    def record(stage, outcome, detail):
        summary.append((stage, outcome, detail))
        print(f"N0 stage={stage:<12} {outcome:<6} {detail}")

    # -- translate (authored text)
    translated = None
    try:
        translated = translate_text(out)
        record("translate", "PASS", f"2-D {len(translated['2d'].splitlines())} HLSL lines, "
                                    f"1-D {len(translated['1d'].splitlines())} HLSL lines")
    except Exception as error:  # noqa: BLE001 - the report names the failure
        failures.append(("translate", str(error)))
        record("translate", "FAIL", str(error))

    # -- controls (authored text)
    text_controls = []
    try:
        text_controls = controls()
        write(out / "controls.txt", "\n".join(
            f"{label}: {expected}\n  {actual}" for label, expected, actual in text_controls))
        record("controls", "PASS", f"{len(text_controls)} refusals, all named")
    except Exception as error:  # noqa: BLE001
        failures.append(("controls", str(error)))
        record("controls", "FAIL", str(error))

    # -- reference
    try:
        cases = [
            ("1d-n192", reference_1d(192), 192, 0),
            ("1d-n173", reference_1d(173), 173, 19),
            ("1d-n0", reference_1d(0), 0, 192),
            ("2d-n24", reference_2d(24), 192, 0),
            ("2d-n13", reference_2d(13), 104, 88),
            ("2d-n0", reference_2d(0), 0, 192),
        ]
        payload = {}
        for name, words, active, inactive in cases:
            if len(words) != WORDS:
                raise AssertionError(f"{name}: {len(words)} words")
            payload[name] = {"active": active, "inactive": inactive, "words": words}
        assert payload["1d-n192"]["words"][0] == 0x1234, payload["1d-n192"]["words"][0]
        assert payload["1d-n192"]["words"][1] == 0x266B, payload["1d-n192"]["words"][1]
        assert payload["1d-n192"]["words"][191] == CAP, payload["1d-n192"]["words"][191]
        assert payload["1d-n173"]["words"][173] == SENTINEL
        assert payload["2d-n13"]["words"][row_index(0, 12)] != SENTINEL, "column 12 must be written"
        assert payload["2d-n13"]["words"][row_index(0, 13)] == SENTINEL, "column 13 must stay untouched"
        assert payload["2d-n13"]["words"][row_index(1, 13)] == SENTINEL
        # The 2-D read is a column broadcast: every row's active prefix is identical.
        for row in range(1, 8):
            assert payload["2d-n24"]["words"][row_index(row, 5)] == payload["2d-n24"]["words"][row_index(0, 5)]
        (out / "reference.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")

        # Lower-cap case: the cap binds for every active element.
        low_cap = 4000
        for name, words, active in (("1d", reference_1d(192, cap=low_cap), 192),
                                    ("2d", reference_2d(24, cap=low_cap), 192)):
            if any(word != low_cap for word in words[:active]):
                raise AssertionError(f"{name}: a lower cap did not bind on every active element")
            if any(word != SENTINEL for word in words[active:]):
                raise AssertionError(f"{name}: a lower cap changed an inactive word")

        # Row-stride sensitivity: a record with the wrong stride changes the address map, so the
        # reference built for 96 bytes cannot describe it.
        wrong = aot.lower(PROGRAM_2D, GROUP_2D, **dict(BIND_2D, stride=92))
        if wrong == translated["2d"]:
            raise AssertionError("a wrong row stride produced identical HLSL")
        accepted_row1 = STRIDE_2D // 4
        wrong_row1 = 92 // 4
        if accepted_row1 == wrong_row1:
            raise AssertionError("the stride control does not separate row 1")
        (out / "sensitivity.json").write_text(json.dumps({
            "lower_cap": {"cap": low_cap, "active_elements": 192,
                          "every_active_word": low_cap, "inactive_words": "sentinel"},
            "wrong_row_stride": {"accepted": STRIDE_2D, "wrong": 92,
                                 "row1_col0_word_accepted": accepted_row1,
                                 "row1_col0_word_wrong": wrong_row1,
                                 "hlsl_differs": True},
            "row_broadcast": {"rows_identical": True,
                              "reason": "the load has no row term; only the store separates rows"},
        }, indent=2, sort_keys=True) + "\n", encoding="ascii")
        record("reference", "PASS", "192/173/0 active (1-D), 24/13/0 columns (2-D), cap and stride controls")
    except Exception as error:  # noqa: BLE001
        failures.append(("reference", str(error)))
        record("reference", "FAIL", str(error))

    # -- assemble + disassemble (Linux worker, or a native LLVM if one is supplied)
    machine = None
    machine_dir = out / "llvm"
    if args.wsl_distro:
        try:
            machine_dir.mkdir(parents=True, exist_ok=True)
            command = ["wsl", "-d", args.wsl_distro, "-u", "root", "--", "python3",
                       to_wsl_path(ROOT / "tests" / "compute_native_check.py"),
                       "--linux-worker", "--out", to_wsl_path(machine_dir), "--program", "both"]
            result = run(command)
            (machine_dir / "linux-worker.log").write_text(
                f"$ {' '.join(command)}\nexit={result.returncode}\n{result.stdout}\n{result.stderr}\n",
                encoding="ascii")
            if result.returncode:
                raise AssertionError(f"linux worker failed:\n{result.stdout}\n{result.stderr}")
            machine = json.loads((machine_dir / "machine.json").read_text(encoding="ascii"))
            comparison = compare_hlsl(out, machine_dir)
            record("assemble", "PASS",
                   f"2d {machine['2d']['bytes']} B sha256 {machine['2d']['sha256'][:16]}..., "
                   f"1d {machine['1d']['bytes']} B sha256 {machine['1d']['sha256'][:16]}...")
            record("disassemble", "PASS",
                   f"2d {machine['2d']['instructions']} instructions, branch "
                   f"{machine['2d']['branch']['at']}->{machine['2d']['branch']['target']}, "
                   f"1d {machine['1d']['instructions']} instructions, HLSL "
                   + ", ".join(f"{name}:{'identical' if entry['byte_identical'] else 'newlines-only'}"
                               for name, entry in comparison.items()))
            byte_controls = machine["controls"]
            (machine_dir / "byte-controls.json").write_text(
                json.dumps(byte_controls, indent=2) + "\n", encoding="ascii")
            with (out / "controls.txt").open("a", encoding="ascii") as handle:
                for entry in byte_controls:
                    handle.write(f"bytes: {entry['control']}: {entry['expected']}\n  {entry['actual']}\n")
            record("controls", "PASS",
                   f"{len(text_controls)} text refusals + {len(byte_controls)} byte refusals")
        except Exception as error:  # noqa: BLE001
            failures.append(("assemble", str(error)))
            record("assemble", "FAIL", str(error))
    elif args.llvm_mc and args.objcopy:
        try:
            machine_dir.mkdir(parents=True, exist_ok=True)
            # A native LLVM: assemble here, no worker process.
            machine = {}
            one, one_binding = read_compute_1d()
            for name, program, group, binding in (
                    ("2d", PROGRAM_2D, GROUP_2D, BIND_2D),
                    ("1d", one, GROUP_1D, one_binding)):
                machine[name] = worker_program(machine_dir, name, program, group, binding,
                                               args.llvm_mc, args.objcopy, args.llvm)
            machine["controls"] = worker_controls(machine_dir, args.llvm_mc, args.objcopy, args.llvm)
            (machine_dir / "machine.json").write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n",
                                                      encoding="ascii")
            record("assemble", "PASS", "native llvm-mc/objcopy")
            record("disassemble", "PASS", "native libLLVM, HLSL identical")
        except Exception as error:  # noqa: BLE001
            failures.append(("assemble", str(error)))
            record("assemble", "FAIL", str(error))
    else:
        skips.append(("assemble", "no --wsl-distro and no --llvm-mc/--objcopy"))
        record("assemble", "SKIP", "no LLVM toolchain was available")
        skips.append(("disassemble", "no libLLVM"))
        record("disassemble", "SKIP", "no libLLVM was available")

    # -- compile: the HLSL generated from the decoded machine code when it exists
    programs = {}
    if translated is not None:
        programs["2d"] = translated["2d"]
        programs["1d"] = translated["1d"]
    if machine:
        for name in ("2d", "1d"):
            path = machine_dir / f"native_{name}.decoded.hlsl"
            if path.exists():
                programs[name] = path.read_text(encoding="ascii")
    containers = {}
    if args.dxc and programs:
        try:
            for name, source in programs.items():
                plain, _ = compile_hlsl(args.dxc, source, f"native_{name}.cso", out, root=False)
                rooted, _ = compile_hlsl(args.dxc, source, f"native_{name}.root.cso", out, root=True)
                tags_plain, tags_root = check_container(plain), check_container(rooted)
                if b"xbox." in plain:
                    raise AssertionError(f"{name}: the compiled module names an xbox intrinsic")
                if "RTS0" not in tags_root:
                    raise AssertionError(f"{name}: the root variant carries no RTS0 part")
                containers[name] = {"plain": plain, "root": rooted,
                                    "tags_plain": tags_plain, "tags_root": tags_root}
            (out / "containers.json").write_text(json.dumps({
                name: {
                    "source": "decoded machine code" if machine else "authored text",
                    "plain": {"parts": entry["tags_plain"], "bytes": len(entry["plain"]),
                              "sha256": hashlib.sha256(entry["plain"]).hexdigest()},
                    "root": {"parts": entry["tags_root"], "bytes": len(entry["root"]),
                             "sha256": hashlib.sha256(entry["root"]).hexdigest()},
                } for name, entry in containers.items()
            }, indent=2) + "\n", encoding="ascii")
            record("compile", "PASS", "; ".join(
                f"{name} {len(entry['plain'])}/{len(entry['root'])} B" for name, entry in containers.items()))
        except Exception as error:  # noqa: BLE001
            failures.append(("compile", str(error)))
            record("compile", "FAIL", str(error))
    else:
        skips.append(("compile", "no --dxc"))
        record("compile", "SKIP", "dxc was not supplied")

    # -- validate
    checks = {}
    if args.root_probe and containers and not failures:
        try:
            import dxil_retarget as rt
            from probe_roots import validate_shader

            evidence = {}
            for name, entry in containers.items():
                for variant in ("plain", "root"):
                    blob = entry[variant]
                    key = f"{name}.{variant}"
                    checks[key] = validate_shader(Path(args.root_probe), blob, parts(blob)[b"DXIL"])
                    status = checks[key]["container"]["status"]
                    module_status = checks[key]["module"]["status"]
                    if status != "0x00000000" or module_status != "0x00000000":
                        raise AssertionError(f"{key}: validation {status} / {module_status}")
                    if checks[key]["ir"]["xbox_call_sites"]:
                        raise AssertionError(f"{key}: xbox calls {checks[key]['ir']['xbox_call_sites']}")
                    signed = rt.sign_container(Path(args.root_probe), blob)
                    check_container(signed)
                    try:
                        rt.sign_container(Path(args.root_probe), blob[:-4])
                    except Exception:  # noqa: BLE001 - any refusal is the expected outcome
                        tamper = "rejected"
                    else:
                        raise AssertionError(f"{key}: the signer accepted a truncated container")
                    evidence[key] = {
                        "validation": checks[key],
                        "signed_bytes": len(signed),
                        "signed_sha256": hashlib.sha256(signed).hexdigest(),
                        "signed_digest_nonzero": signed[4:20] != bytes(16),
                        "signer_identical_to_dxc_output": signed == blob,
                        "truncated_container": tamper,
                    }
                    (out / f"native_{name}.{variant}.signed.cso").write_bytes(signed)
            (out / "validation.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                                                 encoding="ascii")
            record("validate", "PASS", "dxil.dll 0x00000000 for container and module, signed, both variants")
        except Exception as error:  # noqa: BLE001
            failures.append(("validate", str(error)))
            record("validate", "FAIL", str(error))
    else:
        skips.append(("validate", "no --root-probe"))
        record("validate", "SKIP", "root_probe was not supplied")

    # -- probe
    if args.device_probe and checks and not failures:
        try:
            import dxil_retarget as rt

            results = {}
            for name in containers:
                blob = containers[name]["root"]
                inventory = checks[f"{name}.root"]["ir"]["xbox_intrinsics"]
                warp = rt.check_compute_pso(Path(args.device_probe), blob, warp_only=True,
                                            intrinsics=inventory)
                both = rt.check_compute_pso(Path(args.device_probe), blob, warp_only=False,
                                            intrinsics=inventory)
                results[name] = {"warp": warp, "both": both}
                (out / f"native_{name}.pso.json").write_text(
                    json.dumps(both, indent=2, sort_keys=True) + "\n", encoding="ascii")
            (out / "pso.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                          encoding="ascii")
            plain = rt.check_compute_pso(Path(args.device_probe), containers["2d"]["plain"],
                                         warp_only=True,
                                         intrinsics=checks["2d.root"]["ir"]["xbox_intrinsics"])
            (out / "pso-plain-container.json").write_text(
                json.dumps({"embedded_root_missing": plain}, indent=2, sort_keys=True) + "\n",
                encoding="ascii")
            record("probe", "PASS", "; ".join(
                f"{name} warp={results[name]['warp'].get('warp_pso')} "
                f"hw={results[name]['both'].get('hardware_pso')}" for name in results))
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode("utf-8", "replace").strip() or f"exit {error.returncode}"
            if error.returncode == 5:
                skips.append(("probe", detail))
                record("probe", "SKIP", detail)
            else:
                failures.append(("probe", detail))
                record("probe", "FAIL", detail)
        except Exception as error:  # noqa: BLE001
            failures.append(("probe", str(error)))
            record("probe", "FAIL", str(error))
    else:
        skips.append(("probe", "no --device-probe or no validated container"))
        record("probe", "SKIP", "root_device_probe was not supplied")

    (out / "summary.json").write_text(json.dumps({
        "stages": [{"stage": s, "outcome": o, "detail": d} for s, o, d in summary],
        "failures": failures, "skips": skips,
        "machine_code": {name: {key: machine[name][key] for key in
                                ("bytes", "sha256", "instructions", "hlsl_identical")}
                         for name in ("2d", "1d")} if machine else None,
    }, indent=2) + "\n", encoding="ascii")

    if failures:
        print(f"N0 result: FAIL ({len(failures)} stage(s))")
        return 1
    if skips:
        print(f"N0 result: PARTIAL ({len(skips)} capability skip(s))")
        return SKIP if len(skips) == len(summary) - 1 else 0
    print("N0 result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
