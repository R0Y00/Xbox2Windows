"""Offline native gfx1013 compute adapter and cache (#1412, Stage 4B.2).

Fixture-only toolchain: it consumes a bounded native compute record, translates the machine code
through the existing translator, compiles and validates desktop DXIL, probes the PSO and publishes an
integrity-checked cache entry. It does not touch the runtime, the slot 67 table or the slot 62 slice.

Stages, so a machine with LLVM and a machine with DXC can split the work:

    --stage request    write req_<key>.csn and <key>.csn.src from record JSON files
    --stage translate  --llvm LIB                 machine code -> HLSL   (needs LLVM 15 + gfx1013)
    --stage compile    --dxc ... --probe ... --pso-probe ...            HLSL -> validated signed DXIL

Cache family: x2w-startup-compute-native-2
    req_<key>.csn          canonical record bytes
    <key>.csn.src          request manifest
    <key>.csn.hlsl         translated source (an intermediate, not a validated entry)
    <key>.csn.cso          signed desktop DXIL
    <key>.csn.manifest     response manifest, published last
    <key>.csn.error        retryable infrastructure failure
    <key>.csn.corrupt      quarantined malformed request
    <key>.csn.lock         single-writer lock with an ownership token

Single writer: a lock is acquired with O_EXCL and holds a random ownership token. It is never
reclaimed because of age; every cache mutation asserts the token first and release only removes a
lock that still holds this writer's token. An abandoned lock is a manual-repair condition: the
operator must establish that the original writer has stopped before deleting it.

Exit codes: 0 published, revalidated or a cache hit; 1 a permanent program refusal; 2 retryable,
busy or a toolchain mismatch; 3 a record or usage error; 4 a quarantined corrupt request.
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import secrets
import struct
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

MAGIC = b"X2WN"
VERSION = 1
STAGE_COMPUTE = 5
ISA_GFX1013 = 1
GUEST_WAVE32 = 32
REVISION = "x2w-startup-compute-native-2"
TRANSLATOR_NAME = "compute_store_aot"
DECODER_NAME = "pixel_aot"
TARGET = "dxil-ms-dx"
DXC_PROFILE = "cs_6_0"
DXC_FLAGS = "cs_6_0 -E main"
MAX_CODE = 64 * 1024
MAX_RECORD = MAX_CODE + 4096
MAX_MANIFEST = 1 << 16
MAX_OUTPUT = 1 << 20
MAX_HLSL = MAX_CODE * 4
MAX_ROOT_PARAMETERS = 8
NO_REGISTER = 0xFFFFFFFF

D3D12_32BIT_CONSTANTS = 1
D3D12_SRV = 3
D3D12_UAV = 4
SHADER_VISIBILITY_ALL = 0
ROOT_DESCRIPTOR_FLAG_MASK_SUPPORTED = 0

PROFILE_1D = 1
PROFILE_2D = 2

ROOT_FIXTURE = (
    (D3D12_SRV, 0, 1),
    (D3D12_UAV, 0, 1),
    (D3D12_32BIT_CONSTANTS, 0, 16),
)
ROOT_SIGNATURE = 'SRV(t0),UAV(u0),RootConstants(num32BitConstants=16,b0)'
ROOT_CONSTANTS = 16

# Deliberately pinned, content-addressed toolchain identities. The translate stage hashes the
# libLLVM file it is given and refuses a different one; the compile stage hashes dxcompiler.dll and
# dxil.dll next to the dxc it is given. A toolchain change is an explicit edit here and invalidates
# every entry through the manifest identity, never silently through a stale acceptance.
PINNED_LLVM = "de2e35a4f9b3f6a06d2a8a3342b3f62a3842b1923b8dfc2a6ce48e0cc2d1e85d"
PINNED_COMPILER = "495033935c058fc6d318ac257afdfa910795bf41b7e5ba905a0e7191dc8bd4fa"
PINNED_VALIDATOR = "db108ffef7bcecb984bdccf2b4725719d65b2ce7f1ebc40019dbe6c5a7016e09"

# Repository tools whose content is part of the produced artifact's identity. Names only: no
# absolute path ever enters a record or a manifest.
PIPELINE_TOOLS = (
    ("compute_store_aot", "tools/compute_store_aot.py"),
    ("pixel_aot", "tools/pixel_aot.py"),
    ("resource_inventory", "tools/resource_inventory.py"),
    ("dxil_retarget", "tools/dxil_retarget.py"),
    ("probe_roots", "tools/probe_roots.py"),
    ("xbox_intrinsic_gate", "tools/xbox_intrinsic_gate.py"),
    ("compute_native", "tools/startup/compute_native.py"),
)

# Exactly the fields of serialize(), in serialization order. The magic precedes them; the root
# parameter list, the code digest and the code bytes follow them.
SCALARS = (
    "version", "stage", "isa", "profile", "guest_wave",
    "group_x", "group_y", "group_z",
    "dispatch_x", "dispatch_y", "dispatch_z",
    "user_sgpr_count", "table_reg", "group_reg", "group_reg_y",
    "descriptor_offset", "constant_descriptor", "source_descriptor",
    "source_stride", "u0_stride",
    "address_layout", "column_shift", "element_bytes",
    "guard_axis", "guard_op", "bound_constant", "cap_constant",
    "translator_revision", "decoder_id", "dxc_profile", "dxc_flags",
    "rsrc2_interpretation", "recorded_rsrc2", "code_length",
)

GUARD_OPS = {1: "gt"}
FIXED_JSON_FIELDS = ("version", "stage", "isa", "guest_wave", "translator_revision", "decoder_id",
                     "dxc_profile", "dxc_flags", "rsrc2_interpretation", "guard_op")
BUILD_FIELDS = ("profile", "user_sgpr_count", "table_reg", "group_reg", "group_reg_y",
                "descriptor_offset", "constant_descriptor", "source_descriptor", "source_stride",
                "u0_stride", "address_layout", "column_shift", "element_bytes", "guard_axis",
                "bound_constant", "cap_constant", "recorded_rsrc2")
JSON_FIELDS = frozenset(BUILD_FIELDS + FIXED_JSON_FIELDS +
                        ("group", "dispatch", "root_parameters", "code_hex"))

MANIFEST_HEADER = "x2w-native-compute 2"
MANIFEST_COMMON = ("key", "revision", "status", "record_version", "record_sha256")
MANIFEST_REQUEST = MANIFEST_COMMON + ("code_bytes", "code_sha256")
MANIFEST_REFUSED = MANIFEST_COMMON + ("code_bytes", "code_sha256", "reason")
MANIFEST_VALIDATED = MANIFEST_COMMON + (
    "code_bytes", "code_sha256", "group", "hlsl_sha256", "output_bytes", "output_sha256",
    "translator", "decoder", "pipeline", "llvm", "compiler", "validator", "target", "dxc_flags",
    "validation", "rts0", "gate", "signed",
)
MANIFEST_TRANSLATED = MANIFEST_COMMON + (
    "code_bytes", "code_sha256", "hlsl_sha256", "translator", "decoder", "pipeline", "llvm",
)
MANIFEST_IDENTITY = ("translator", "decoder", "pipeline", "llvm", "compiler", "validator", "target",
                     "dxc_flags")
MANIFEST_EVIDENCE = ("validation", "rts0", "gate", "signed")
MANIFEST_KNOWN = frozenset(MANIFEST_VALIDATED + ("reason",))

VALIDATION_OK = "container=0x00000000 module=0x00000000"
SIGNED_OK = "dxil-digest-nonzero"


class RecordError(ValueError):
    """The record is malformed or outside the supported schema."""


class Refusal(Exception):
    """The program is unsupported. Publishing this is permanent."""


class Retryable(Exception):
    """Infrastructure could not complete the job. The request stays queued."""


class ToolchainMismatch(Exception):
    """The local toolchain is not the pinned, verified one. Nothing is mutated."""


def fnv_bytes(data, current=1469598103934665603):
    for byte in data:
        current = ((current ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return current


def digest_hex(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return digest_hex(Path(path).read_bytes())


# ---------------------------------------------------------------- identity

@lru_cache(maxsize=1)
def expected_identity():
    """The pinned identity every validated manifest must declare exactly."""
    digests = []
    for name, relative in PIPELINE_TOOLS:
        path = ROOT / relative
        try:
            digest = file_sha256(path)
        except OSError as error:
            raise ToolchainMismatch(f"cannot read dependency {relative}: {error}") from error
        digests.append(f"{name}={digest}\n")
    pipeline = digest_hex("".join(sorted(digests)).encode("ascii"))[:16]
    translator = f"{TRANSLATOR_NAME}/{file_sha256(ROOT / 'tools' / 'compute_store_aot.py')[:16]}"
    decoder = f"{DECODER_NAME}/{file_sha256(ROOT / 'tools' / 'pixel_aot.py')[:16]}"
    return {
        "translator": translator,
        "decoder": decoder,
        "pipeline": pipeline,
        "llvm": PINNED_LLVM,
        "compiler": PINNED_COMPILER,
        "validator": PINNED_VALIDATOR,
        "target": TARGET,
        "dxc_flags": DXC_FLAGS,
    }


def verify_translate_toolchain(library):
    """Refuse a different libLLVM before any cache mutation."""
    try:
        observed = file_sha256(library)
    except OSError as error:
        raise ToolchainMismatch(f"cannot read {library}: {error}") from error
    if observed != PINNED_LLVM:
        raise ToolchainMismatch(f"libLLVM identity {observed[:16]} is not the pinned "
                                f"{PINNED_LLVM[:16]}")


def verify_compile_toolchain(dxc, validator=None):
    """Refuse a different compiler or validator before any cache mutation."""
    dxc = Path(dxc)
    compiler = dxc.parent / "dxcompiler.dll"
    validator_path = Path(validator) if validator else dxc.parent / "dxil.dll"
    for label, path, pin in (("dxcompiler.dll", compiler, PINNED_COMPILER),
                             ("dxil.dll", validator_path, PINNED_VALIDATOR)):
        try:
            observed = file_sha256(path)
        except OSError as error:
            raise ToolchainMismatch(f"cannot read {label} next to dxc: {error}") from error
        if observed != pin:
            raise ToolchainMismatch(f"{label} identity {observed[:16]} is not the pinned {pin[:16]}")


# ---------------------------------------------------------------- record

def build_record(*, profile, code, group, dispatch, user_sgpr_count, table_reg, group_reg,
                 group_reg_y, descriptor_offset, constant_descriptor, source_descriptor,
                 source_stride, u0_stride, address_layout, column_shift, element_bytes,
                 guard_axis, bound_constant, cap_constant, recorded_rsrc2=0,
                 root_parameters=ROOT_FIXTURE):
    """Build a record; the two approved fixture profiles are the only ones validate() accepts."""
    return {
        "version": VERSION,
        "stage": STAGE_COMPUTE,
        "isa": ISA_GFX1013,
        "profile": profile,
        "guest_wave": GUEST_WAVE32,
        "group_x": group[0], "group_y": group[1], "group_z": group[2],
        "dispatch_x": dispatch[0], "dispatch_y": dispatch[1], "dispatch_z": dispatch[2],
        "user_sgpr_count": user_sgpr_count,
        "table_reg": table_reg,
        "group_reg": group_reg,
        "group_reg_y": group_reg_y,
        "descriptor_offset": descriptor_offset,
        "constant_descriptor": constant_descriptor,
        "source_descriptor": source_descriptor,
        "source_stride": source_stride,
        "u0_stride": u0_stride,
        "address_layout": address_layout,
        "column_shift": column_shift,
        "element_bytes": element_bytes,
        "guard_axis": guard_axis,
        "guard_op": 1,
        "bound_constant": bound_constant,
        "cap_constant": cap_constant,
        "translator_revision": 1,
        "decoder_id": 1,
        "dxc_profile": 0,
        "dxc_flags": 0,
        "rsrc2_interpretation": 0,
        "recorded_rsrc2": recorded_rsrc2,
        "code_length": len(code),
        "root_parameters": tuple(tuple(parameter) for parameter in root_parameters),
        "code_sha256": hashlib.sha256(code).digest(),
        "code": code,
    }


def serialize(record):
    """Canonical bytes: magic, the scalar block, the root list, the code digest, the code."""
    out = bytearray(MAGIC)
    for name in SCALARS:
        value = record[name]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF:
            raise RecordError(f"Record field {name} is not a u32")
        out += struct.pack("<I", value)
    parameters = record["root_parameters"]
    out += struct.pack("<I", len(parameters))
    for kind, register, count in parameters:
        out += struct.pack("<III", kind, register, count)
    out += record["code_sha256"]
    out += record["code"]
    return bytes(out)


def parse(blob):
    """Strict inverse of serialize(). Trailing bytes, a short read or a bad magic are errors."""
    if len(blob) > MAX_RECORD:
        raise RecordError(f"Record is larger than {MAX_RECORD} bytes")
    if len(blob) < 4 + 4 * len(SCALARS) + 4 + 32:
        raise RecordError("Record is truncated")
    if blob[:4] != MAGIC:
        raise RecordError("Record magic is not X2WN")
    record = {}
    offset = 4
    for name in SCALARS:
        record[name], = struct.unpack_from("<I", blob, offset)
        offset += 4
    count, = struct.unpack_from("<I", blob, offset)
    offset += 4
    if count > MAX_ROOT_PARAMETERS:
        raise RecordError(f"Record declares {count} root parameters")
    if len(blob) < offset + 12 * count + 32:
        raise RecordError("Record is truncated inside the root parameter list")
    parameters = []
    for _ in range(count):
        kind, register, size = struct.unpack_from("<III", blob, offset)
        parameters.append((kind, register, size))
        offset += 12
    record["root_parameters"] = tuple(parameters)
    record["code_sha256"] = blob[offset:offset + 32]
    offset += 32
    if record["code_length"] > MAX_CODE:
        raise RecordError(f"Record declares {record['code_length']} code bytes")
    if len(blob) != offset + record["code_length"]:
        raise RecordError("Record length does not match its code length")
    record["code"] = blob[offset:]
    return record


def record_digest(record):
    return hashlib.sha256(serialize(record)).digest()


def cache_key(record):
    """FNV-1a 64 over the revision and the canonical record digest, as the repository does."""
    return fnv_bytes(record_digest(record), fnv_bytes(REVISION.encode("ascii")))


def key_text(key):
    return f"{key:016x}"


def validate(record):
    """Every rejection names its cause. No field is defaulted or guessed."""
    if record["version"] != VERSION:
        raise RecordError(f"Unsupported record version {record['version']}")
    if record["stage"] != STAGE_COMPUTE:
        raise RecordError(f"Unsupported shader stage {record['stage']}")
    if record["isa"] != ISA_GFX1013:
        raise RecordError(f"Unsupported ISA {record['isa']}")
    if record["profile"] not in (PROFILE_1D, PROFILE_2D):
        raise RecordError(f"Unsupported native compute profile {record['profile']}")
    if record["guest_wave"] != GUEST_WAVE32:
        raise RecordError(f"Unsupported guest wave contract {record['guest_wave']}")
    if record["translator_revision"] != 1:
        raise RecordError(f"Unsupported translator revision {record['translator_revision']}")
    if record["decoder_id"] != 1:
        raise RecordError(f"Unsupported decoder identity {record['decoder_id']}")
    if record["dxc_profile"] != 0:
        raise RecordError(f"Unsupported DXC profile {record['dxc_profile']}")
    if record["dxc_flags"] != 0:
        raise RecordError(f"Unsupported DXC flags {record['dxc_flags']}")
    if record["rsrc2_interpretation"] != 0:
        raise RecordError("COMPUTE_PGM_RSRC2 is recorded, not interpreted: TG_SIZE_EN stays unsupported")
    if record["recorded_rsrc2"] != 0:
        raise RecordError(f"recorded COMPUTE_PGM_RSRC2 0x{record['recorded_rsrc2']:08x} is not "
                          "supported: the local translator cannot interpret it")

    group = (record["group_x"], record["group_y"], record["group_z"])
    dispatch = (record["dispatch_x"], record["dispatch_y"], record["dispatch_z"])
    for axis, size in zip("xyz", group):
        if not 1 <= size <= (64 if axis == "z" else 1024):
            raise RecordError(f"Unsupported thread group axis {axis}={size}")
        if size & (size - 1):
            raise RecordError(f"Thread group axis {axis}={size} is not a power of two")
    if group[0] * group[1] * group[2] > 1024:
        raise RecordError(f"Unsupported thread group product {group[0] * group[1] * group[2]}")
    if not all(1 <= size <= 0xFFFF for size in dispatch):
        raise RecordError("Unsupported dispatch dimensions")

    user = record["user_sgpr_count"]
    if not 2 <= user <= 16:
        raise RecordError(f"Unsupported user SGPR count {user}")
    if not 0 <= record["table_reg"] <= 105 or record["table_reg"] + 1 >= user:
        raise RecordError("The table pointer does not fit the user SGPRs")
    if not 0 <= record["group_reg"] < user or record["group_reg"] == record["table_reg"]:
        raise RecordError("The group X register is outside the user SGPRs")
    if record["group_reg_y"] != NO_REGISTER:
        if not 0 <= record["group_reg_y"] < user:
            raise RecordError("The group Y register is outside the user SGPRs")
        if record["group_reg_y"] in (record["table_reg"], record["group_reg"]):
            raise RecordError("The group Y register collides with another entry register")

    for name in ("descriptor_offset", "constant_descriptor", "source_descriptor"):
        value = record[name]
        if value % 4 or value >= 0x1000:
            raise RecordError(f"{name} {value} is not a bounded DWORD table offset")
    offsets = (record["descriptor_offset"], record["constant_descriptor"], record["source_descriptor"])
    if len(set(offsets)) != 3:
        raise RecordError("The bound descriptors share a table offset")
    if record["source_stride"] and (record["source_stride"] % 4 or record["source_stride"] >= 0x1000):
        raise RecordError(f"source_stride {record['source_stride']} is not a bounded DWORD stride")

    if record["address_layout"] not in (1, 2):
        raise RecordError(f"Unsupported address layout {record['address_layout']}")
    if record["element_bytes"] != 4:
        raise RecordError(f"Unsupported element width {record['element_bytes']}")
    if record["column_shift"] != 2:
        raise RecordError(f"Unsupported column shift {record['column_shift']}")
    if record["address_layout"] == 1:
        if record["u0_stride"] != 4:
            raise RecordError("The index layout requires a four-byte u0 stride")
        if group[1] != 1 or group[2] != 1:
            raise RecordError("The index layout requires a one-dimensional group")
        if record["group_reg_y"] != NO_REGISTER:
            raise RecordError("The index layout does not use a group Y register")
    else:
        if record["u0_stride"] < group[0] << record["column_shift"] or record["u0_stride"] % 4:
            raise RecordError("The row stride does not hold one group row")
        if record["group_reg_y"] == NO_REGISTER:
            raise RecordError("The index-offset layout requires a group Y register")

    if record["guard_axis"] not in (0, 1):
        raise RecordError(f"Unsupported guard axis {record['guard_axis']}")
    if record["guard_op"] not in GUARD_OPS:
        raise RecordError(f"Unsupported guard compare {record['guard_op']}")
    for name in ("bound_constant", "cap_constant"):
        if record[name] >= 16:
            raise RecordError(f"{name} {record[name]} is outside the 16-DWORD constant contract")

    parameters = record["root_parameters"]
    if len(parameters) != len(ROOT_FIXTURE) or tuple(parameters) != ROOT_FIXTURE:
        raise RecordError("The root signature contract is not the approved fixture contract")

    profile = record["profile"]
    if profile == PROFILE_1D and (group != (32, 1, 1) or record["address_layout"] != 1):
        raise RecordError("Profile 1d requires group (32,1,1) and the index layout")
    if profile == PROFILE_2D and (group != (8, 4, 1) or record["address_layout"] != 2
                                  or record["u0_stride"] != 96):
        raise RecordError("Profile 2d requires group (8,4,1), the index-offset layout and stride 96")
    if profile == PROFILE_2D and dispatch[0] * group[0] > (record["u0_stride"] // 4):
        raise RecordError("The dispatch row overruns the declared stride")

    code = record["code"]
    if len(code) % 4:
        raise RecordError("The native code is not a whole number of words")
    if not 4 <= len(code) <= MAX_CODE:
        raise RecordError(f"Unsupported native code length {len(code)}")
    if record["code_length"] != len(code):
        raise RecordError("code_length does not match the native code")
    if hashlib.sha256(code).digest() != record["code_sha256"]:
        raise RecordError("The native code digest does not match its bytes")
    return record


def record_to_json(record):
    """The accepted JSON form: constructor fields plus the fixed scalars, nothing derived."""
    payload = {name: record[name] for name in BUILD_FIELDS}
    payload.update({name: record[name] for name in FIXED_JSON_FIELDS})
    payload["group"] = [record["group_x"], record["group_y"], record["group_z"]]
    payload["dispatch"] = [record["dispatch_x"], record["dispatch_y"], record["dispatch_z"]]
    payload["root_parameters"] = [list(parameter) for parameter in record["root_parameters"]]
    payload["code_hex"] = record["code"].hex()
    return payload


def record_from_json(payload):
    """Strict JSON intake: unknown, duplicate, missing and contradictory fields are named errors."""
    if not isinstance(payload, dict):
        raise RecordError("A record description must be a JSON object")
    unknown = sorted(set(payload) - JSON_FIELDS)
    if unknown:
        raise RecordError(f"Unknown record field {unknown[0]!r}")
    required = set(BUILD_FIELDS) | set(FIXED_JSON_FIELDS) | {"group", "dispatch", "root_parameters", "code_hex"}
    missing = sorted(required - set(payload))
    if missing:
        raise RecordError(f"Missing record field {missing[0]!r}")
    fields = dict(payload)
    code_hex = fields.pop("code_hex")
    if not isinstance(code_hex, str):
        raise RecordError("code_hex must be a string")
    try:
        code = bytes.fromhex(code_hex)
    except ValueError as error:
        raise RecordError("code_hex is not hexadecimal") from error
    group = fields["group"]
    dispatch = fields["dispatch"]
    for name, value in (("group", group), ("dispatch", dispatch)):
        if not isinstance(value, list) or len(value) != 3:
            raise RecordError(f"{name} must be a three-element list")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
            raise RecordError(f"{name} must contain integers")
    parameters = fields["root_parameters"]
    if not isinstance(parameters, list) or any(
            not isinstance(parameter, list) or len(parameter) != 3 for parameter in parameters):
        raise RecordError("root_parameters must be a list of three-element lists")
    record = build_record(code=code, group=tuple(group), dispatch=tuple(dispatch),
                          root_parameters=tuple(tuple(parameter) for parameter in parameters),
                          **{name: fields[name] for name in BUILD_FIELDS})
    if tuple(tuple(parameter) for parameter in parameters) != record["root_parameters"]:
        raise RecordError("root_parameters does not round-trip")
    for name in FIXED_JSON_FIELDS:
        if fields[name] != record[name]:
            raise RecordError(f"Record field {name} is fixed at {record[name]}")
    return record


# ---------------------------------------------------------------- translation

def boundary_check(code, decoded):
    """4-byte aligned 4- or 8-byte encodings covering every byte, starting at zero."""
    if not decoded or decoded[0][0] != 0:
        raise Refusal("Decoding does not start at byte 0")
    offsets = [where for where, _ in decoded]
    sizes = [later - earlier for earlier, later in zip(offsets, offsets[1:])]
    sizes.append(len(code) - offsets[-1])
    for where, size in zip(offsets, sizes):
        if where % 4 or size not in (4, 8):
            raise Refusal(f"Invalid instruction boundary at byte {where}")
    if sum(sizes) != len(code):
        raise Refusal(f"Decoded {sum(sizes)} bytes of {len(code)}")


def branch_check(decoded):
    """Branches must target a decoded boundary, forward, and the program must end in s_endpgm."""
    starts = {where for where, _ in decoded}
    for where, text in decoded:
        if not text.startswith("s_cbranch_execz"):
            continue
        operand = text.split()[-1]
        try:
            target = where + 4 + 4 * int(operand, 0)
        except ValueError as error:
            raise Refusal(f"Unreadable branch target {operand!r}") from error
        if target <= where:
            raise Refusal(f"Backward branch at byte {where}")
        if target not in starts:
            raise Refusal(f"Branch target {target} is not an instruction boundary")
    if not decoded or decoded[-1][1] != "s_endpgm":
        raise Refusal("Missing final program exit")


def translation_kwargs(record):
    return dict(table_reg=record["table_reg"],
                descriptor_offset=record["descriptor_offset"],
                constant_descriptor=record["constant_descriptor"],
                source_descriptor=record["source_descriptor"],
                stride=record["u0_stride"],
                source_stride=record["source_stride"] or None,
                group_reg=record["group_reg"],
                group_reg_y=None if record["group_reg_y"] == NO_REGISTER else record["group_reg_y"],
                cpu="gfx1013")


def translate(record, library):
    """Native bytes -> decoded instructions -> HLSL, with no assembly-text fallback anywhere."""
    import compute_store_aot as aot
    from pixel_aot import disassemble

    validate(record)
    try:
        decoded = disassemble(record["code"], library, offsets=True, cpu=b"gfx1013")
    except ValueError as error:
        raise Refusal(f"Undecodable native code: {error}") from error
    boundary_check(record["code"], decoded)
    branch_check(decoded)
    group = (record["group_x"], record["group_y"], record["group_z"])
    try:
        return aot.lower(aot.labelled(decoded), group, **translation_kwargs(record))
    except ValueError as error:
        raise Refusal(f"Unsupported program: {error}") from error


# ---------------------------------------------------------------- compilation and probing

def embed_root(hlsl):
    """Insert the approved fixture root signature before the entry attribute; nothing else changes."""
    marker = b"[numthreads("
    if marker not in hlsl:
        raise Refusal("The translated source has no [numthreads] entry")
    index = hlsl.index(marker)
    return hlsl[:index] + f'[RootSignature("{ROOT_SIGNATURE}")]\n'.encode("ascii") + hlsl[index:]


def compile_dxil(hlsl, dxc, work):
    """Normal DXC path, cs_6_0 -E main; output must be DXBC carrying DXIL and the embedded root."""
    source = work / "shader.root.hlsl"
    output = work / "shader.cso"
    source.write_bytes(embed_root(hlsl))
    result = subprocess.run([str(dxc), "-T", DXC_PROFILE, "-E", "main", "-Fo", str(output), str(source)],
                            capture_output=True, text=True)
    if result.returncode:
        raise Retryable(f"dxc failed ({result.returncode}): {result.stderr.strip()[:200]}")
    blob = output.read_bytes()
    from resource_inventory import parts

    if blob[:4] != b"DXBC":
        raise Retryable("DXC output is not a DXBC container")
    tags = [tag.decode("ascii", "replace") for tag in parts(blob)]
    if "DXIL" not in tags:
        raise Retryable(f"DXC output has no DXIL part: {tags}")
    if "RTS0" not in tags:
        raise Retryable("DXC output carries no embedded root signature")
    return blob, tags


def validate_and_sign(blob, probe):
    """dxil.dll container and module validation, the intrinsic gate, then the repository signer."""
    import dxil_retarget as rt
    from probe_roots import validate_shader
    from resource_inventory import parts
    from xbox_intrinsic_gate import decide

    checks = validate_shader(Path(probe), blob, parts(blob)[b"DXIL"])
    container, module = checks["container"], checks["module"]
    if container["status"] != "0x00000000" or module["status"] != "0x00000000":
        raise Retryable(f"Validation failed: {container['status']} / {module['status']}")
    if checks["ir"]["target_triple"] != TARGET:
        raise Retryable(f"Unexpected target triple {checks['ir']['target_triple']}")
    if checks["ir"]["xbox_call_sites"]:
        raise Refusal(f"Executable Xbox intrinsics remain: {sorted(checks['ir']['xbox_call_sites'])}")
    gate = decide(checks["ir"]["xbox_intrinsics"])
    if gate["verdict"] != "ALLOW":
        raise Refusal(gate["message"])
    signed = rt.sign_container(Path(probe), blob)
    if signed[4:20] == bytes(16):
        raise Retryable("The signer left the container digest zeroed")
    return checks, signed, gate


def collect_backends(blob, inventory, pso_probe, required):
    """Every required backend must report a root and a PSO HRESULT; a missing key is not a pass."""
    import dxil_retarget as rt

    def run(warp_only):
        try:
            result = rt.check_compute_pso(Path(pso_probe), blob, warp_only=warp_only,
                                          intrinsics=inventory)
        except subprocess.CalledProcessError as error:
            raise Retryable(f"PSO probe failed: exit {error.returncode}") from error
        except ValueError as error:
            raise Retryable(f"PSO probe output is not JSON: {error}") from error
        if not isinstance(result, dict):
            raise Retryable("PSO probe returned malformed JSON")
        if result.get("gate", {}).get("submitted") is not True:
            raise Retryable("PSO probe did not submit the module")
        return result

    warp = run(True)
    backends = {"warp": {"root": warp.get("warp_root"), "pso": warp.get("warp_pso"),
                         "adapter": "warp", "shader_model": warp.get("warp_shader_model")}}
    both = None
    if "hardware" in required:
        both = run(False)
        backends["hardware"] = {"root": both.get("hardware_root"), "pso": both.get("hardware_pso"),
                                "adapter": both.get("adapter_id", ""),
                                "shader_model": both.get("hardware_shader_model")}
    for name, entry in backends.items():
        if entry["root"] != "0x00000000":
            raise Retryable(f"{name}: root signature creation {entry['root']}")
        if entry["pso"] != "0x00000000":
            raise Retryable(f"{name}: compute PSO creation {entry['pso']}")
        if not entry["adapter"] or not entry["shader_model"]:
            raise Retryable(f"{name}: probe did not identify the adapter and shader model")
        if name == "warp" and entry["adapter"] != "warp":
            raise Retryable(f"warp: probe reported adapter {entry['adapter']!r}")
        if name == "hardware" and entry["adapter"] == "warp":
            raise Retryable("hardware: the probe ran WARP, not a physical adapter")
    return {"backends": backends, "raw": {"warp": warp, "both": both}}


# ---------------------------------------------------------------- root signature

def parse_rts0(payload):
    """Read the DXC-serialized versioned root signature (version 1.1 layout)."""
    if len(payload) < 24:
        raise RecordError("The embedded RTS0 part is truncated")
    version, count, parameter_offset, samplers, sampler_offset, flags = struct.unpack_from("<6I", payload, 0)
    if version != 2:
        raise RecordError(f"Unsupported embedded root signature version {version}")
    if count > MAX_ROOT_PARAMETERS:
        raise RecordError(f"The embedded root signature declares {count} parameters")
    if samplers > 8:
        raise RecordError(f"The embedded root signature declares {samplers} static samplers")
    if sampler_offset > len(payload):
        raise RecordError("The embedded sampler table is outside the part")
    parameters = []
    for index in range(count):
        at = parameter_offset + 12 * index
        if at + 12 > len(payload):
            raise RecordError("A root parameter entry is truncated")
        kind, visibility, offset = struct.unpack_from("<3I", payload, at)
        if kind == D3D12_32BIT_CONSTANTS:
            if offset + 12 > len(payload):
                raise RecordError("A root constants entry is truncated")
            register, space, count_values = struct.unpack_from("<3I", payload, offset)
            parameters.append({"type": kind, "visibility": visibility, "register": register,
                               "space": space, "count": count_values, "flags": None})
        elif kind in (D3D12_SRV, D3D12_UAV):
            if offset + 12 > len(payload):
                raise RecordError("A root descriptor entry is truncated")
            register, space, descriptor_flags = struct.unpack_from("<3I", payload, offset)
            parameters.append({"type": kind, "visibility": visibility, "register": register,
                               "space": space, "count": None, "flags": descriptor_flags})
        else:
            raise RecordError(f"Unsupported embedded root parameter type {kind}")
    return {"version": version, "parameters": parameters, "samplers": samplers, "flags": flags}


def verify_rts0(payload, record):
    """The embedded signature must be exactly the supported fixture contract."""
    parsed = parse_rts0(payload)
    if parsed["samplers"] != 0:
        raise RecordError("The supported root signature has no static samplers")
    if parsed["flags"] != 0:
        raise RecordError(f"The supported root signature has flags 0x{parsed['flags']:x}")
    expected = record["root_parameters"]
    if len(parsed["parameters"]) != len(expected):
        raise RecordError("The embedded root signature parameter count differs from the record")
    for index, (parameter, (kind, register, count)) in enumerate(zip(parsed["parameters"], expected)):
        if parameter["type"] != kind or parameter["register"] != register:
            raise RecordError(f"Root parameter {index} differs from the record contract")
        if parameter["visibility"] != SHADER_VISIBILITY_ALL:
            raise RecordError(f"Root parameter {index} is not visible to all stages")
        if parameter["space"] != 0:
            raise RecordError(f"Root parameter {index} uses register space {parameter['space']}")
        if kind == D3D12_32BIT_CONSTANTS:
            if parameter["count"] != count or count != ROOT_CONSTANTS:
                raise RecordError(f"Root constants parameter {index} has {parameter['count']} values")
        else:
            if parameter["flags"] != ROOT_DESCRIPTOR_FLAG_MASK_SUPPORTED:
                raise RecordError(f"Root descriptor {index} has flags 0x{parameter['flags']:x}")
    return parsed


# ---------------------------------------------------------------- cache files

def cache_paths(directory, key):
    stem = key_text(key)
    return {name: directory / f"{prefix}{stem}{suffix}" for name, prefix, suffix in (
        ("request", "req_", ".csn"),
        ("request_manifest", "", ".csn.src"),
        ("hlsl", "", ".csn.hlsl"),
        ("output", "", ".csn.cso"),
        ("manifest", "", ".csn.manifest"),
        ("error", "", ".csn.error"),
        ("corrupt", "", ".csn.corrupt"),
        ("lock", "", ".csn.lock"),
    )}


def write_atomic(path, data):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                # A file published by the other side of the split toolchain can still be briefly
                # held across the /mnt boundary.
                if attempt == 3:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


class Writer:
    """A lock handle. Every mutation asserts that this token is still the one on disk."""

    def __init__(self, path, token):
        self.path = path
        self.token = token

    def assert_owner(self):
        try:
            current = self.path.read_text(encoding="ascii").strip()
        except FileNotFoundError as error:
            raise Retryable("the writer lock disappeared; no mutation was made") from error
        except OSError as error:
            raise Retryable(f"the writer lock is unreadable: {error}") from error
        if current != self.token:
            raise Retryable("the writer lock is held by another writer; no mutation was made")


def release_lock(path, token):
    """Remove the lock only when it still holds this writer's token."""
    try:
        current = path.read_text(encoding="ascii").strip()
    except OSError:
        return
    if current != token:
        return
    try:
        path.unlink()
    except OSError:
        pass


@contextlib.contextmanager
def writer_lock(directory, key):
    """Exclusive acquisition with an ownership token. A held lock is never reclaimed by age."""
    path = cache_paths(directory, key)["lock"]
    token = f"{os.getpid():d}-{secrets.token_hex(16)}"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise Retryable("another writer holds the lock; an abandoned lock is a manual-repair "
                        "condition") from error
    try:
        os.write(handle, (token + "\n").encode("ascii"))
    finally:
        os.close(handle)
    writer = Writer(path, token)
    try:
        yield writer
    finally:
        release_lock(path, token)


def read_bounded(path, limit):
    if path.stat().st_size > limit:
        raise RecordError(f"{path.name} is larger than {limit} bytes")
    return path.read_bytes()


def manifest_text(fields):
    return "\n".join([MANIFEST_HEADER] + [f"{name} {value}" for name, value in fields]) + "\n"


def parse_manifest(text):
    """Strict manifest parsing: known names only, no duplicates, no control characters."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != MANIFEST_HEADER:
        raise RecordError("Not a native compute manifest")
    fields = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        name, separator, value = line.strip().partition(" ")
        if not separator or not value:
            raise RecordError(f"Manifest line {line.strip()!r} has no value")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise RecordError(f"Manifest field name {name!r} is not supported")
        if name in fields:
            raise RecordError(f"Contradictory or repeated manifest field {name!r}")
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
            raise RecordError(f"Manifest field {name!r} has a non-printable value")
        if name.startswith("pso-"):
            if name[len("pso-"):] not in ("warp", "hardware"):
                raise RecordError(f"Unsupported probe backend {name!r}")
        elif name not in MANIFEST_KNOWN:
            raise RecordError(f"Unsupported manifest field {name!r}")
        fields[name] = value
    return fields


def verified_backends(fields):
    """The backends a manifest actually proves, from its own lines."""
    found = {}
    for name, value in fields.items():
        if not name.startswith("pso-"):
            continue
        backend = name[len("pso-"):]
        parts = dict(piece.split("=", 1) for piece in value.split() if "=" in piece)
        if not {"root", "pso", "adapter", "shader_model"} <= set(parts):
            continue
        if not parts["root"].startswith("0x") or not parts["pso"].startswith("0x"):
            continue
        if parts["root"] != "0x00000000" or parts["pso"] != "0x00000000":
            continue
        if backend == "warp" and parts["adapter"] != "warp":
            continue
        if backend == "hardware" and (parts["adapter"] == "warp" or ":" not in parts["adapter"]):
            continue
        found[backend] = parts
    return found


def _problems_add(problems, name, detail):
    problems.append(f"{name}: {detail}")


def inspect_entry(directory, key, required=(), *, expected=None, want_artifacts=True):
    """The one authoritative acceptance decision.

    Returns {state, detail, problems, fields, record, output, hlsl} and never raises for malformed
    cache content. States: hit, backend-missing, refused, error, corrupt-request, corrupt, stale,
    missing.
    """
    result = {"state": "missing", "detail": "", "problems": [], "fields": None, "record": None,
              "output": None, "hlsl": None}
    paths = cache_paths(directory, key)
    if expected is None:
        try:
            expected = expected_identity()
        except ToolchainMismatch as error:
            result.update(state="error", detail=str(error))
            return result
    if paths["corrupt"].exists():
        try:
            detail = read_bounded(paths["corrupt"], 4096).decode("ascii", "replace").strip()
        except OSError as error:
            detail = str(error)
        result.update(state="corrupt-request", detail=detail)
        return result
    if paths["error"].exists():
        try:
            detail = read_bounded(paths["error"], 4096).decode("ascii", "replace").strip()
        except OSError as error:
            detail = str(error)
        result.update(state="error", detail=detail)
        return result
    if not paths["manifest"].exists():
        return result
    try:
        fields = parse_manifest(read_bounded(paths["manifest"], MAX_MANIFEST).decode("utf-8", "replace"))
    except (RecordError, OSError) as error:
        result.update(state="corrupt", detail=str(error))
        return result
    result["fields"] = fields
    problems = result["problems"]
    for name in MANIFEST_COMMON:
        if name not in fields:
            _problems_add(problems, "missing", name)
    if problems:
        result.update(state="corrupt", detail="; ".join(problems))
        return result
    if fields["key"] != key_text(key):
        result.update(state="stale", detail="manifest key does not match the filename")
        return result
    if fields["revision"] != REVISION:
        result.update(state="stale", detail=f"manifest revision {fields['revision']!r} is not supported")
        return result
    if fields["status"] == "refused":
        try:
            record = _load_record(paths, key)
        except (RecordError, OSError) as error:
            result.update(state="corrupt", detail=f"refusal record: {error}")
            return result
        result["record"] = record
        if fields.get("record_sha256") != record_digest(record).hex():
            result.update(state="corrupt", detail="refusal record digest does not match")
            return result
        if fields.get("record_version") != str(record["version"]):
            result.update(state="corrupt", detail="refusal record version does not match")
            return result
        if fields.get("code_bytes") != str(len(record["code"])) or \
                fields.get("code_sha256") != digest_hex(record["code"]):
            result.update(state="corrupt", detail="refusal record code identity does not match")
            return result
        result.update(state="refused", detail=fields.get("reason", ""))
        return result
    if fields["status"] != "validated":
        result.update(state="corrupt", detail=f"unknown manifest status {fields['status']!r}")
        return result

    for name in MANIFEST_VALIDATED:
        if name not in fields:
            _problems_add(problems, "missing", name)
    for name in MANIFEST_IDENTITY:
        if name in fields and fields[name] != expected[name]:
            result.update(state="stale", detail=f"manifest {name} {fields[name]!r} is not the "
                                                f"pinned {expected[name]!r}")
            return result
    for name in MANIFEST_EVIDENCE:
        if name not in fields:
            continue
        valid = {"validation": VALIDATION_OK, "rts0": "present", "gate": "ALLOW",
                 "signed": SIGNED_OK}[name]
        if fields[name] != valid:
            _problems_add(problems, name, f"{fields[name]!r} is not {valid!r}")
    if problems:
        result.update(state="corrupt", detail="; ".join(problems))
        return result

    try:
        record = _load_record(paths, key)
    except (RecordError, OSError) as error:
        result.update(state="corrupt", detail=f"record: {error}")
        return result
    result["record"] = record
    if fields["record_version"] != str(record["version"]):
        _problems_add(problems, "record_version", "does not match the record")
    if fields["record_sha256"] != record_digest(record).hex():
        _problems_add(problems, "record_sha256", "does not match the record")
    if fields["code_bytes"] != str(len(record["code"])):
        _problems_add(problems, "code_bytes", "does not match the record")
    if fields["code_sha256"] != digest_hex(record["code"]):
        _problems_add(problems, "code_sha256", "does not match the record")
    expected_group = f'{record["group_x"]}x{record["group_y"]}x{record["group_z"]}'
    if fields["group"] != expected_group:
        _problems_add(problems, "group", "does not match the record")
    if problems:
        result.update(state="corrupt", detail="; ".join(problems))
        return result

    try:
        hlsl = read_bounded(paths["hlsl"], MAX_HLSL)
    except OSError as error:
        result.update(state="corrupt", detail=f"hlsl: {error}")
        return result
    if fields["hlsl_sha256"] != digest_hex(hlsl):
        result.update(state="corrupt", detail="hlsl digest does not match")
        return result
    result["hlsl"] = hlsl
    try:
        output = read_bounded(paths["output"], MAX_OUTPUT)
    except OSError as error:
        result.update(state="corrupt", detail=f"output: {error}")
        return result
    if fields["output_bytes"] != str(len(output)):
        result.update(state="corrupt", detail="output size does not match")
        return result
    if fields["output_sha256"] != digest_hex(output):
        result.update(state="corrupt", detail="output digest does not match")
        return result
    if output[:4] != b"DXBC":
        result.update(state="corrupt", detail="output is not a DXBC container")
        return result
    try:
        from resource_inventory import parts

        sections = parts(output)
    except ValueError as error:
        result.update(state="corrupt", detail=f"output container: {error}")
        return result
    if b"DXIL" not in sections:
        result.update(state="corrupt", detail="output carries no DXIL part")
        return result
    if b"RTS0" not in sections:
        result.update(state="corrupt", detail="output carries no embedded root signature")
        return result
    if output[4:20] == bytes(16):
        result.update(state="corrupt", detail="output container digest is zero (unsigned)")
        return result
    try:
        verify_rts0(sections[b"RTS0"], record)
    except RecordError as error:
        result.update(state="corrupt", detail=f"root contract: {error}")
        return result
    result["output"] = output
    backends = verified_backends(fields)
    missing = [backend for backend in required if backend not in backends]
    if missing:
        result.update(state="backend-missing", detail="missing required backend " + ",".join(missing))
        return result
    result.update(state="hit", detail="")
    return result


def _load_record(paths, key):
    if not paths["request"].exists():
        raise RecordError("the request record is missing")
    record = parse(read_bounded(paths["request"], MAX_RECORD))
    validate(record)
    if cache_key(record) != key:
        raise RecordError("the request bytes do not hash to the request key")
    return record


def entry_state(directory, key, required=()):
    result = inspect_entry(directory, key, required)
    return result["state"], result["detail"]


def verify_entry(directory, key, required):
    """Consumer-side acceptance. The returned bytes are exactly the bytes that were verified."""
    result = inspect_entry(directory, key, required)
    return {"accepted": result["state"] == "hit", "state": result["state"],
            "detail": result["detail"], "problems": result["problems"],
            "fields": result["fields"], "record": result["record"],
            "output": result["output"], "hlsl": result["hlsl"]}


# ---------------------------------------------------------------- publication

def publish_error(writer, directory, key, reason):
    writer.assert_owner()
    write_atomic(cache_paths(directory, key)["error"],
                 ((" ".join(reason.split())[:300]) + "\n").encode("ascii"))


def publish_corrupt(writer, directory, key, reason):
    writer.assert_owner()
    write_atomic(cache_paths(directory, key)["corrupt"],
                 ((" ".join(reason.split())[:300]) + "\n").encode("ascii"))
    cache_paths(directory, key)["error"].unlink(missing_ok=True)


def publish_refusal(writer, directory, key, record, reason):
    """Publish the permanent refusal before removing obsolete successful artifacts."""
    paths = cache_paths(directory, key)
    writer.assert_owner()
    fields = [
        ("key", key_text(key)), ("revision", REVISION), ("status", "refused"),
        ("record_version", record["version"]), ("record_sha256", record_digest(record).hex()),
        ("code_bytes", len(record["code"])), ("code_sha256", digest_hex(record["code"])),
        ("reason", " ".join(reason.split())[:300]),
    ]
    paths["error"].unlink(missing_ok=True)
    paths["corrupt"].unlink(missing_ok=True)
    write_atomic(paths["manifest"], manifest_text(fields).encode("ascii"))
    for name in ("hlsl", "output"):
        try:
            paths[name].unlink(missing_ok=True)
        except OSError:
            # The refusal is already authoritative; leftover artifacts are ignored by acceptance.
            pass


def publish_validated(writer, directory, key, record, hlsl, output, checks, gate, backends, expected):
    """Write artifacts first, clear obsolete markers, publish the manifest last."""
    paths = cache_paths(directory, key)
    verify_rts0_from_container(output, record)
    fields = [
        ("key", key_text(key)), ("revision", REVISION), ("status", "validated"),
        ("record_version", record["version"]), ("record_sha256", record_digest(record).hex()),
        ("code_bytes", len(record["code"])), ("code_sha256", digest_hex(record["code"])),
        ("group", f'{record["group_x"]}x{record["group_y"]}x{record["group_z"]}'),
        ("hlsl_sha256", digest_hex(hlsl)),
        ("output_bytes", len(output)), ("output_sha256", digest_hex(output)),
    ]
    fields += [(name, expected[name]) for name in MANIFEST_IDENTITY]
    fields += [
        ("validation", f'container={checks["container"]["status"]} module={checks["module"]["status"]}'),
        ("rts0", "present"), ("gate", gate["verdict"]), ("signed", SIGNED_OK),
    ]
    for backend, entry in sorted(backends.items()):
        fields.append((f"pso-{backend}",
                       f'adapter={entry["adapter"]} root={entry["root"]} pso={entry["pso"]} '
                       f'shader_model={entry["shader_model"]}'))
    writer.assert_owner()
    write_atomic(paths["hlsl"], hlsl)
    writer.assert_owner()
    write_atomic(paths["output"], output)
    paths["error"].unlink(missing_ok=True)
    paths["corrupt"].unlink(missing_ok=True)
    writer.assert_owner()
    write_atomic(paths["manifest"], manifest_text(fields).encode("ascii"))


def verify_rts0_from_container(output, record):
    from resource_inventory import parts

    return verify_rts0(parts(output)[b"RTS0"], record)


def verify_translation_receipt(directory, key, record, hlsl, expected):
    """The split stages agree: the HLSL being compiled is the one the translate stage produced."""
    paths = cache_paths(directory, key)
    try:
        receipt = parse_manifest(read_bounded(paths["request_manifest"], MAX_MANIFEST)
                                 .decode("utf-8", "replace"))
    except (RecordError, OSError) as error:
        raise Retryable(f"translation receipt is unreadable: {error}") from error
    if receipt.get("status") != "translated":
        raise Retryable("translation has not run for this key")
    if receipt.get("key") != key_text(key) or receipt.get("revision") != REVISION:
        raise Retryable("translation receipt does not belong to this key")
    if receipt.get("record_sha256") != record_digest(record).hex():
        raise Retryable("translation receipt does not describe this record")
    if receipt.get("hlsl_sha256") != digest_hex(hlsl):
        raise Retryable("translation receipt does not describe this HLSL")
    for name in ("translator", "decoder", "pipeline", "llvm"):
        if receipt.get(name) != expected[name]:
            raise Retryable(f"translation receipt {name} is not the pinned identity")


# ---------------------------------------------------------------- jobs

def record_from_request(directory, key):
    return _load_record(cache_paths(directory, key), key)


def stage_request(directory, records, required, evidence=None):
    results = []
    for record in records:
        validate(record)
        key = cache_key(record)
        body = serialize(record)
        if cache_key(parse(body)) != key:
            raise RecordError("The canonical record does not round-trip")
        try:
            with writer_lock(directory, key) as writer:
                state = inspect_entry(directory, key, required)["state"]
                if state in ("hit", "refused"):
                    results.append((key, state, ""))
                    continue
                writer.assert_owner()
                paths = cache_paths(directory, key)
                write_atomic(paths["request"], body)
                write_atomic(paths["request_manifest"], manifest_text([
                    ("key", key_text(key)), ("revision", REVISION), ("status", "request"),
                    ("record_version", record["version"]),
                    ("record_sha256", record_digest(record).hex()),
                    ("code_bytes", len(record["code"])),
                    ("code_sha256", digest_hex(record["code"])),
                ]).encode("ascii"))
                paths["corrupt"].unlink(missing_ok=True)
                if evidence is not None:
                    evidence(key, "request", {"bytes": len(body), "record_sha256":
                                              record_digest(record).hex()})
                results.append((key, "requested", ""))
        except Retryable as error:
            results.append((key, "busy", str(error)))
        except OSError as error:
            results.append((key, "error", str(error)))
    return results


def stage_translate(directory, library, required, evidence=None):
    verify_translate_toolchain(library)
    expected = expected_identity()
    results = []
    for request in sorted(directory.glob("req_*.csn")):
        key = int(request.name[4:-4], 16)
        try:
            with writer_lock(directory, key) as writer:
                state = inspect_entry(directory, key, required)["state"]
                if state in ("hit", "refused", "backend-missing"):
                    results.append((key, state, ""))
                    continue
                try:
                    record = record_from_request(directory, key)
                except RecordError as error:
                    try:
                        publish_corrupt(writer, directory, key, f"malformed request: {error}")
                    except (Retryable, OSError):
                        results.append((key, "error", f"quarantine failed: {error}"))
                    else:
                        results.append((key, "quarantined", str(error)))
                    continue
                except OSError as error:
                    try:
                        publish_error(writer, directory, key, str(error))
                    except (Retryable, OSError):
                        pass
                    results.append((key, "error", str(error)))
                    continue
                try:
                    hlsl = translate(record, library).encode("ascii")
                except Refusal as error:
                    try:
                        publish_refusal(writer, directory, key, record, str(error))
                    except (Retryable, OSError) as failure:
                        results.append((key, "error", f"refusal publication failed: {failure}"))
                    else:
                        results.append((key, "refused", str(error)))
                    continue
                paths = cache_paths(directory, key)
                writer.assert_owner()
                write_atomic(paths["hlsl"], hlsl)
                writer.assert_owner()
                write_atomic(paths["request_manifest"], manifest_text([
                    ("key", key_text(key)), ("revision", REVISION), ("status", "translated"),
                    ("record_version", record["version"]),
                    ("record_sha256", record_digest(record).hex()),
                    ("code_bytes", len(record["code"])),
                    ("code_sha256", digest_hex(record["code"])),
                    ("hlsl_sha256", digest_hex(hlsl)),
                    ("translator", expected["translator"]),
                    ("decoder", expected["decoder"]),
                    ("pipeline", expected["pipeline"]),
                    ("llvm", expected["llvm"]),
                ]).encode("ascii"))
                if evidence is not None:
                    evidence(key, "translation", {"bytes": len(hlsl), "hlsl_sha256": digest_hex(hlsl),
                                                  "identity": expected["translator"]})
                results.append((key, "translated", len(hlsl)))
        except Retryable as error:
            results.append((key, "busy", str(error)))
        except OSError as error:
            results.append((key, "error", str(error)))
    return results


def stage_compile(directory, dxc, probe, pso_probe, required, *, validator=None, evidence=None):
    verify_compile_toolchain(dxc, validator)
    expected = expected_identity()
    results = []
    for request in sorted(directory.glob("req_*.csn")):
        key = int(request.name[4:-4], 16)
        try:
            with writer_lock(directory, key) as writer:
                finding = inspect_entry(directory, key, required)
                state = finding["state"]
                if state in ("hit", "refused"):
                    results.append((key, state, ""))
                    continue
                try:
                    record = record_from_request(directory, key)
                except RecordError as error:
                    try:
                        publish_corrupt(writer, directory, key, f"malformed request: {error}")
                    except (Retryable, OSError):
                        results.append((key, "error", f"quarantine failed: {error}"))
                    else:
                        results.append((key, "quarantined", str(error)))
                    continue
                except OSError as error:
                    try:
                        publish_error(writer, directory, key, str(error))
                    except (Retryable, OSError):
                        pass
                    results.append((key, "error", str(error)))
                    continue
                try:
                    if state == "backend-missing" and finding["output"] is not None:
                        output = finding["output"]
                        hlsl = finding["hlsl"]
                        checks, signed, gate = validate_and_sign(output, probe)
                        if signed != output:
                            raise Retryable("the published output does not match its re-signed form")
                        if evidence is not None:
                            evidence(key, "validation", checks)
                        collected = collect_backends(signed, checks["ir"]["xbox_intrinsics"],
                                                     pso_probe, required)
                        if evidence is not None:
                            evidence(key, "pso", collected["raw"])
                        publish_validated(writer, directory, key, record, hlsl, signed, checks, gate,
                                          collected["backends"], expected)
                        results.append((key, "revalidated", ",".join(sorted(collected["backends"]))))
                        continue
                    hlsl = finding["hlsl"]
                    if hlsl is None:
                        paths = cache_paths(directory, key)
                        if not paths["hlsl"].exists():
                            raise Retryable("translation has not run for this key")
                        hlsl = read_bounded(paths["hlsl"], MAX_HLSL)
                    verify_translation_receipt(directory, key, record, hlsl, expected)
                    work = directory / f"work-{key_text(key)}"
                    work.mkdir(exist_ok=True)
                    output, tags = compile_dxil(hlsl, dxc, work)
                    if evidence is not None:
                        evidence(key, "compile", {"tags": tags, "bytes": len(output),
                                                  "sha256": digest_hex(output)})
                    checks, signed, gate = validate_and_sign(output, probe)
                    if evidence is not None:
                        evidence(key, "validation", checks)
                    collected = collect_backends(signed, checks["ir"]["xbox_intrinsics"],
                                                 pso_probe, required)
                    if evidence is not None:
                        evidence(key, "pso", collected["raw"])
                    publish_validated(writer, directory, key, record, hlsl, signed, checks, gate,
                                      collected["backends"], expected)
                    results.append((key, "validated", ",".join(sorted(collected["backends"]))))
                except Refusal as error:
                    try:
                        publish_refusal(writer, directory, key, record, str(error))
                    except (Retryable, OSError) as failure:
                        results.append((key, "error", f"refusal publication failed: {failure}"))
                    else:
                        results.append((key, "refused", str(error)))
                except (Retryable, RecordError, OSError) as error:
                    try:
                        publish_error(writer, directory, key, str(error))
                    except (Retryable, OSError):
                        pass
                    results.append((key, "error", str(error)))
        except Retryable as error:
            results.append((key, "busy", str(error)))
        except OSError as error:
            results.append((key, "error", str(error)))
    return results


def load_records(paths):
    records = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        try:
            payload = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except json.JSONDecodeError as error:
            raise RecordError(f"{path}: not JSON: {error}") from error
        records.append(record_from_json(payload))
    return records


def _no_duplicate_keys(pairs):
    seen = {}
    for name, value in pairs:
        if name in seen:
            raise RecordError(f"Duplicate JSON field {name!r}")
        seen[name] = value
    return seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pending", type=Path, help="cache directory")
    parser.add_argument("--stage", choices=("request", "translate", "compile"))
    parser.add_argument("--record", type=Path, action="append", default=[],
                        help="record description JSON (request stage)")
    parser.add_argument("--llvm")
    parser.add_argument("--dxc")
    parser.add_argument("--validator")
    parser.add_argument("--probe")
    parser.add_argument("--pso-probe")
    parser.add_argument("--require-backend", action="append", default=[], choices=("warp", "hardware"))
    parser.add_argument("--print-key", type=Path, help="print the canonical key of a record JSON")
    args = parser.parse_args(argv)

    required = tuple(args.require_backend or ("warp", "hardware"))
    try:
        if args.print_key:
            record = load_records([args.print_key])[0]
            validate(record)
            print(key_text(cache_key(record)))
            return 0
        if not args.pending or not args.stage:
            parser.error("--pending and --stage are required")
        directory = args.pending.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if args.stage == "request":
            records = load_records(args.record)
            results = stage_request(directory, records, required)
        elif args.stage == "translate":
            if not args.llvm:
                parser.error("--llvm is required for the translate stage")
            results = stage_translate(directory, args.llvm, required)
        else:
            if not (args.dxc and args.probe and args.pso_probe):
                parser.error("--dxc, --probe and --pso-probe are required for the compile stage")
            results = stage_compile(directory, args.dxc, args.probe, args.pso_probe, required,
                                    validator=args.validator)
    except ToolchainMismatch as error:
        print(f"native compute: {error}")
        return 2
    except RecordError as error:
        print(f"native compute: {error}")
        return 3
    except OSError as error:
        print(f"native compute: {error}")
        return 3

    for key, outcome, detail in results:
        print(f"{key_text(key)} {outcome} {detail}")
    if not results:
        print("native compute: nothing to do")
    outcomes = {outcome for _, outcome, _ in results}
    if "quarantined" in outcomes:
        return 4
    if "error" in outcomes or "busy" in outcomes:
        return 2
    if "refused" in outcomes:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
