"""Stage 4B.2 checks: the hardened native compute adapter and cache (#1412).

Runs the real split toolchain: assembly and translation in WSL (LLVM 15), compilation, validation and
PSO probes in Windows (DXC, dxil.dll, WARP and the physical adapter). Infrastructure failure controls
call the real functions with clearly labelled stub tools; every other control exercises the real
acceptance, publication and locking code against a clone of the published cache.

    out\\pe-env\\Scripts\\python.exe -B tests\\compute_native_cache_check.py `
        --out out\\native-compute\\cache-suite --wsl-distro Ubuntu-22.04 `
        --dxc <dxc.exe> --validator <dxil.dll> --probe root_probe.exe --pso-probe root_device_probe.exe

Exit codes: 0 all checks passed, 1 a failure.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "startup"))

import compute_native as native  # noqa: E402

LLVM15_LIB = "/usr/lib/llvm-15/lib/libLLVM-15.so.1"
REQUIRED = ("warp", "hardware")
IDENTITY_MUTATIONS = ("translator", "decoder", "pipeline", "llvm", "compiler", "validator", "target",
                      "dxc_flags")
EVIDENCE_MUTATIONS = ("validation", "rts0", "gate", "signed")


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def to_wsl(path):
    resolved = Path(path).resolve()
    return "/mnt/" + resolved.drive.rstrip(":").lower() + resolved.as_posix()[len(resolved.drive):]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def part_offset(blob, tag):
    count = struct.unpack_from("<I", blob, 28)[0]
    for index in range(count):
        offset = struct.unpack_from("<I", blob, 32 + 4 * index)[0]
        if blob[offset:offset + 4] == tag:
            return offset
    raise LookupError(tag)


VARY_FIELDS = ("profile", "user_sgpr_count", "table_reg", "group_reg", "group_reg_y",
               "descriptor_offset", "constant_descriptor", "source_descriptor", "source_stride",
               "u0_stride", "address_layout", "column_shift", "element_bytes", "guard_axis",
               "bound_constant", "cap_constant")


def variant(record, code, **changes):
    fields = {name: record[name] for name in VARY_FIELDS}
    fields["group"] = (record["group_x"], record["group_y"], record["group_z"])
    fields["dispatch"] = (record["dispatch_x"], record["dispatch_y"], record["dispatch_z"])
    fields["code"] = code
    fields.update(changes)
    return native.build_record(**fields)


def write_stub_tools(directory, python):
    """Labelled mocks: .cmd wrappers around tiny Python stubs, standing in for failing tools."""
    directory.mkdir(parents=True, exist_ok=True)
    stubs = {
        "dxc_fail": "import sys\nsys.stderr.write('mock dxc: forced failure\\n')\nsys.exit(1)\n",
        "dxc_garbage": ("import sys\nfrom pathlib import Path\na = sys.argv\n"
                        "Path(a[a.index('-Fo') + 1]).write_bytes(b'NOTDXBC')\n"),
        "probe_fail": ("import sys, json\n"
                       "if '--disassemble' in sys.argv:\n"
                       "    print('target triple = \"dxil-ms-dx\"')\n"
                       "else:\n"
                       "    print(json.dumps({'status': '0x80004005', 'call_hr': '0x80004005'}))\n"),
        "pso_root_fail": "import sys, json\nprint(json.dumps({'warp_root': '0x80004005', 'warp_shader_model': '0x65'}))\n",
        "pso_missing": "import sys, json\nprint(json.dumps({'warp_root': '0x00000000', 'warp_shader_model': '0x65'}))\n",
        "pso_fail": ("import sys, json\nprint(json.dumps({'warp_root': '0x00000000', 'warp_pso': '0x887a0005',"
                     " 'warp_shader_model': '0x65'}))\n"),
        "pso_bad_json": "print('this is not json')\n",
    }
    paths = {}
    for name, body in stubs.items():
        script = directory / f"{name}.py"
        script.write_text(body, encoding="ascii")
        wrapper = directory / f"{name}.cmd"
        wrapper.write_text(f'@echo off\r\n"{python}" "{script}" %*\r\n', encoding="ascii")
        paths[name] = wrapper
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "native-compute" / "cache-suite")
    parser.add_argument("--wsl-distro", default="Ubuntu-22.04")
    parser.add_argument("--llvm", default=LLVM15_LIB)
    parser.add_argument("--dxc", required=True)
    parser.add_argument("--validator", default=None)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--pso-probe", required=True)
    args = parser.parse_args(argv)

    out = args.out.resolve()
    cache, evidence, llvm_dir, scratch = out / "cache", out / "evidence", out / "llvm", out / "scratch"
    for directory in (cache, evidence, llvm_dir, scratch, evidence / "validation", evidence / "pso",
                      evidence / "compile", evidence / "translation", evidence / "request"):
        directory.mkdir(parents=True, exist_ok=True)

    checks, failures = [], []

    def check(name, condition, detail=""):
        checks.append({"check": name, "ok": bool(condition), "detail": str(detail)[:300]})
        if not condition:
            failures.append((name, str(detail)[:300]))
        print(f"B2 {'PASS' if condition else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        return bool(condition)

    def evidence_writer(key, name, payload):
        path = evidence / name / f"{native.key_text(key)}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")

    # -- toolchain pins ---------------------------------------------------------------
    try:
        native.verify_compile_toolchain(args.dxc, args.validator)
        compile_pin = True
        compile_pin_detail = ""
    except native.ToolchainMismatch as error:
        compile_pin = False
        compile_pin_detail = str(error)
    check("toolchain: compiler and validator match the pinned identity", compile_pin, compile_pin_detail)
    identity = native.expected_identity()
    check("toolchain: translator and decoder identities are computed from source",
          identity["translator"].startswith("compute_store_aot/") and
          identity["decoder"].startswith("pixel_aot/") and len(identity["pipeline"]) == 16,
          identity)
    dxc = Path(args.dxc)
    validator = Path(args.validator) if args.validator else dxc.parent / "dxil.dll"
    (evidence / "toolchain.json").write_text(json.dumps({
        "dxc_version": (run([str(dxc), "--version"]).stdout.strip().splitlines() or ["unknown"])[0],
        "dxcompiler_sha256": sha_file(dxc.parent / "dxcompiler.dll"),
        "dxil_sha256": sha_file(validator),
        "probe_sha256": sha_file(args.probe),
        "pso_probe_sha256": sha_file(args.pso_probe),
        "pinned_llvm_sha256": native.PINNED_LLVM,
        "pinned_compiler_sha256": native.PINNED_COMPILER,
        "pinned_validator_sha256": native.PINNED_VALIDATOR,
        "identity": identity,
        "python": sys.version.split()[0],
    }, indent=2, sort_keys=True) + "\n", encoding="ascii")

    # -- assembly through the N0 worker ------------------------------------------------
    worker = run(["wsl", "-d", args.wsl_distro, "-u", "root", "--", "python3",
                  to_wsl(ROOT / "tests" / "compute_native_check.py"), "--linux-worker",
                  "--out", to_wsl(llvm_dir), "--program", "both"])
    (evidence / "linux-worker.log").write_text(
        f"$ {' '.join(worker.args)}\nexit={worker.returncode}\n{worker.stdout}\n{worker.stderr}\n",
        encoding="utf-8")
    if not check("linux worker assembles both programs", worker.returncode == 0,
                 worker.stderr.strip()[-200:]):
        print("B2 result: FAIL")
        return 1
    machine = json.loads((llvm_dir / "machine.json").read_text(encoding="utf-8"))
    (evidence / "machine.json").write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n",
                                           encoding="ascii")

    code_1d = (llvm_dir / "native_1d.bin").read_bytes()
    code_2d = (llvm_dir / "native_2d.bin").read_bytes()
    golden_1d = (llvm_dir / "native_1d.decoded.hlsl").read_bytes()
    golden_2d = (llvm_dir / "native_2d.decoded.hlsl").read_bytes()

    record_1d = native.build_record(
        profile=native.PROFILE_1D, code=code_1d, group=(32, 1, 1), dispatch=(6, 1, 1),
        user_sgpr_count=2, table_reg=0, group_reg=1, group_reg_y=native.NO_REGISTER,
        descriptor_offset=0, constant_descriptor=32, source_descriptor=16, source_stride=0,
        u0_stride=4, address_layout=1, column_shift=2, element_bytes=4,
        guard_axis=0, bound_constant=0, cap_constant=1)
    record_2d = native.build_record(
        profile=native.PROFILE_2D, code=code_2d, group=(8, 4, 1), dispatch=(3, 2, 1),
        user_sgpr_count=4, table_reg=0, group_reg=2, group_reg_y=3,
        descriptor_offset=0, constant_descriptor=32, source_descriptor=16, source_stride=0,
        u0_stride=96, address_layout=2, column_shift=2, element_bytes=4,
        guard_axis=0, bound_constant=0, cap_constant=1)
    native.validate(record_1d)
    native.validate(record_2d)
    record_by_name = {"1d": record_1d, "2d": record_2d}
    keys = {name: native.cache_key(record) for name, record in record_by_name.items()}

    # -- canonical serialization and JSON round trip ------------------------------------
    serialized = {name: native.serialize(record) for name, record in record_by_name.items()}
    header = 4 + 4 * len(native.SCALARS) + 4
    expected_length = {name: 144 + 36 + 32 + len(record["code"])
                       for name, record in record_by_name.items()}
    check("canonical: 144-byte prefix, 36-byte root list, 32-byte digest, code at 212",
          header == 144 and all(len(serialized[name]) == expected_length[name] == 212 + len(record_by_name[name]["code"])
                                for name in serialized),
          [(name, header, len(serialized[name])) for name in serialized])
    check("canonical: idempotent round trip",
          all(native.serialize(native.parse(serialized[name])) == serialized[name] for name in serialized))
    check("canonical: digest is stable",
          all(native.record_digest(native.parse(serialized[name])) == native.record_digest(record_by_name[name])
              for name in serialized))
    check("canonical: JSON round trip preserves the record and key",
          all(native.cache_key(native.record_from_json(native.record_to_json(record_by_name[name])))
              == keys[name] for name in keys))
    check("canonical: keys differ per profile", keys["1d"] != keys["2d"],
          f'{native.key_text(keys["1d"])} vs {native.key_text(keys["2d"])}')
    check("canonical: key covers the dispatch contract",
          native.cache_key(dict(record_2d, dispatch_x=4)) != keys["2d"])

    (evidence / "golden-vectors.json").write_text(json.dumps({
        "revision": native.REVISION,
        "manifest_header": native.MANIFEST_HEADER,
        "identity": identity,
        "prefix_bytes": 144,
        "root_bytes": 36,
        "digest_bytes": 32,
        "code_offset": 212,
        "scalars": list(native.SCALARS),
        "records": {name: {"record_hex": serialized[name].hex(),
                           "record_sha256": native.record_digest(record).hex(),
                           "cache_key": native.key_text(keys[name]),
                           "code_sha256": sha_bytes(record["code"]),
                           "code_bytes": len(record["code"]),
                           "record_bytes": len(serialized[name]),
                           "group": [record["group_x"], record["group_y"], record["group_z"]],
                           "dispatch": [record["dispatch_x"], record["dispatch_y"], record["dispatch_z"]]}
                    for name, record in record_by_name.items()},
        "key_sensitivity": {"2d_dispatch_x_4": native.key_text(native.cache_key(dict(record_2d, dispatch_x=4)))},
    }, indent=2, sort_keys=True) + "\n", encoding="ascii")

    # -- request -> translate (WSL) -> compile (Windows) ---------------------------------
    requested = native.stage_request(cache, [record_1d, record_2d], REQUIRED, evidence_writer)
    check("request stage writes both requests",
          all(outcome in ("requested", "hit") for _, outcome, _ in requested), requested)

    translate = run(["wsl", "-d", args.wsl_distro, "-u", "root", "--", "python3",
                     to_wsl(ROOT / "tools" / "startup" / "compute_native.py"), "--pending", to_wsl(cache),
                     "--stage", "translate", "--llvm", args.llvm,
                     "--require-backend", "warp", "--require-backend", "hardware"])
    (evidence / "translate.log").write_text(
        f"$ {' '.join(translate.args)}\nexit={translate.returncode}\n{translate.stdout}\n{translate.stderr}\n",
        encoding="utf-8")
    check("translate stage in WSL", translate.returncode == 0,
          (translate.stdout + translate.stderr).strip()[-200:])
    for name in ("1d", "2d"):
        receipt = native.parse_manifest(
            native.cache_paths(cache, keys[name])["request_manifest"].read_text(encoding="utf-8"))
        evidence_writer(keys[name], "translation", receipt)

    compiled = native.stage_compile(cache, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator, evidence=evidence_writer)
    check("compile stage validates and publishes",
          all(outcome in ("validated", "revalidated", "hit") for _, outcome, _ in compiled), compiled)

    for name in ("1d", "2d"):
        paths = native.cache_paths(cache, keys[name])
        finding = native.inspect_entry(cache, keys[name], REQUIRED)
        check(f"{name}: authoritative acceptance is a hit", finding["state"] == "hit",
              f'state={finding["state"]} {finding["detail"]}')
        manifest = native.parse_manifest(paths["manifest"].read_text(encoding="utf-8"))
        missing = sorted(set(native.MANIFEST_VALIDATED) - set(manifest))
        check(f"{name}: manifest declares every mandatory field", not missing, missing)
        check(f"{name}: manifest identity is the pinned identity",
              all(manifest[field] == identity[field] for field in native.MANIFEST_IDENTITY))
        check(f"{name}: manifest evidence is green",
              manifest["validation"] == native.VALIDATION_OK and manifest["rts0"] == "present"
              and manifest["gate"] == "ALLOW" and manifest["signed"] == native.SIGNED_OK)
        check(f"{name}: manifest digests match the files",
              manifest["record_sha256"] == native.record_digest(record_by_name[name]).hex()
              and manifest["output_sha256"] == sha_file(paths["output"])
              and manifest["hlsl_sha256"] == sha_file(paths["hlsl"]))
        check(f"{name}: published HLSL is the accepted translation",
              paths["hlsl"].read_bytes() == (golden_1d if name == "1d" else golden_2d))
        verdict = native.verify_entry(cache, keys[name], REQUIRED)
        check(f"{name}: consumer verification returns the verified bytes",
              verdict["accepted"] and verdict["output"] == paths["output"].read_bytes()
              and verdict["hlsl"] == paths["hlsl"].read_bytes(), verdict["problems"])
        check(f"{name}: embedded root matches the supported contract",
              native.verify_rts0_from_container(paths["output"].read_bytes(), record_by_name[name])
              == {"version": 2, "samplers": 0, "flags": 0, "parameters": [
                  {"type": 3, "visibility": 0, "register": 0, "space": 0, "count": None, "flags": 0},
                  {"type": 4, "visibility": 0, "register": 0, "space": 0, "count": None, "flags": 0},
                  {"type": 1, "visibility": 0, "register": 0, "space": 0, "count": 16, "flags": None}]})
        validation = json.loads((evidence / "validation" / f"{native.key_text(keys[name])}.json")
                                .read_text(encoding="utf-8"))
        check(f"{name}: raw validator report is green",
              validation["container"]["status"] == "0x00000000"
              and validation["module"]["status"] == "0x00000000"
              and validation["ir"]["target_triple"] == "dxil-ms-dx"
              and not validation["ir"]["xbox_call_sites"])
        pso = json.loads((evidence / "pso" / f"{native.key_text(keys[name])}.json")
                         .read_text(encoding="utf-8"))
        check(f"{name}: raw PSO probe reports both backends",
              pso["warp"]["warp_root"] == "0x00000000" and pso["warp"]["warp_pso"] == "0x00000000"
              and pso["both"]["adapter_id"] == manifest["pso-hardware"].split()[0].split("=")[1]
              and pso["both"]["hardware_root"] == "0x00000000"
              and pso["both"]["hardware_pso"] == "0x00000000")

    # -- cache hit must not translate or compile again ------------------------------------
    before = {name: (sha_file(native.cache_paths(cache, keys[name])["hlsl"]),
                     sha_file(native.cache_paths(cache, keys[name])["output"])) for name in keys}
    again = run(["wsl", "-d", args.wsl_distro, "-u", "root", "--", "python3",
                 to_wsl(ROOT / "tools" / "startup" / "compute_native.py"), "--pending", to_wsl(cache),
                 "--stage", "translate", "--llvm", args.llvm])
    recompiled = native.stage_compile(cache, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                      validator=args.validator)
    after = {name: (sha_file(native.cache_paths(cache, keys[name])["hlsl"]),
                    sha_file(native.cache_paths(cache, keys[name])["output"])) for name in keys}
    check("cache hit skips translation", again.returncode == 0 and "translated" not in again.stdout,
          again.stdout.strip()[:160])
    check("cache hit skips compilation", before == after
          and all(outcome == "hit" for _, outcome, _ in recompiled), recompiled)

    # ------------------------------------------------------------------ control helpers
    def fresh(name):
        directory = scratch / name
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        return directory

    def clone(name):
        target = scratch / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(cache, target)
        return target

    def clone_of(source, name):
        target = scratch / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return target

    def manifest_of(directory, key):
        return native.parse_manifest(native.cache_paths(directory, key)["manifest"].read_text(encoding="utf-8"))

    def rewrite_manifest(directory, key, fields):
        native.cache_paths(directory, key)["manifest"].write_bytes(
            native.manifest_text(sorted(fields.items())).encode("ascii"))

    def state(name, condition, detail=""):
        checks.append({"check": name, "ok": bool(condition), "detail": str(detail)[:300]})
        if not condition:
            failures.append((name, str(detail)[:300]))
        print(f"B2 {'PASS' if condition else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        return bool(condition)

    stubs = write_stub_tools(out / "mocks", sys.executable)
    validation_2d = json.loads((evidence / "validation" / f"{native.key_text(keys['2d'])}.json")
                               .read_text(encoding="utf-8"))
    inventory = validation_2d["ir"]["xbox_intrinsics"]
    gate = {"verdict": "ALLOW"}
    verified_cache = native.verify_entry(cache, keys["2d"], REQUIRED)
    blob = native.cache_paths(cache, keys["2d"])["output"].read_bytes()
    hlsl = native.cache_paths(cache, keys["2d"])["hlsl"].read_bytes()

    # ------------------------------------------------------------------ real-byte refusals
    def translate_refusal(name, code, record, message, directory):
        mutated = variant(record, code)
        native.validate(mutated)
        key = native.cache_key(mutated)
        native.stage_request(directory, [mutated], REQUIRED)
        result = run(["wsl", "-d", args.wsl_distro, "-u", "root", "--", "python3",
                      to_wsl(ROOT / "tools" / "startup" / "compute_native.py"), "--pending",
                      to_wsl(directory), "--stage", "translate", "--llvm", args.llvm,
                      "--require-backend", "warp", "--require-backend", "hardware"])
        manifest = native.cache_paths(directory, key)["manifest"]
        fields = native.parse_manifest(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        reason = fields.get("reason", (result.stdout + result.stderr).strip())
        state(name, result.returncode == 1 and fields.get("status") == "refused" and message in reason,
              reason[:200])

    translate_refusal("bytes: invalid encoding refused", b"\xff\xff\xff\xff" + code_2d[4:], record_2d,
                      "Unknown instruction at byte 0", fresh("bytes-invalid"))
    translate_refusal("bytes: branch target outside the program refused", code_2d[:-4], record_2d,
                      "not an instruction boundary", fresh("bytes-branch"))
    translate_refusal("bytes: missing final exit refused", code_2d[:4], record_2d,
                      "Missing final program exit", fresh("bytes-exit"))
    translate_refusal("bytes: unsupported opcode refused", (llvm_dir / "control_lds.bin").read_bytes(),
                      record_2d, "Unverified private LDS read", fresh("bytes-lds"))
    translate_refusal("bytes: computed-index guard refused",
                      (llvm_dir / "control_linear.bin").read_bytes(), record_2d,
                      "Unverified lane guard compare", fresh("bytes-linear"))

    # ------------------------------------------------------------------ H4 strict metadata
    def expect_record_error(name, record, message):
        try:
            native.validate(record)
        except native.RecordError as error:
            state(name, message in str(error), str(error))
        else:
            state(name, False, "accepted")

    for value in (0x984, 1, 0xFFFFFFFF):
        expect_record_error(f"record: recorded_rsrc2 0x{value:x} is refused",
                            dict(record_2d, recorded_rsrc2=value), "recorded COMPUTE_PGM_RSRC2")
    expect_record_error("record: unsupported interpretation mode",
                        dict(record_2d, rsrc2_interpretation=1), "recorded, not interpreted")
    expect_record_error("record: unsupported guest wave", dict(record_2d, guest_wave=64),
                        "Unsupported guest wave contract")
    expect_record_error("record: unsupported version", dict(record_2d, version=2),
                        "Unsupported record version")
    expect_record_error("record: invalid stage", dict(record_2d, stage=4), "Unsupported shader stage")
    expect_record_error("record: invalid ISA", dict(record_2d, isa=2), "Unsupported ISA")
    expect_record_error("record: unsupported profile", dict(record_2d, profile=3),
                        "Unsupported native compute profile")
    expect_record_error("record: stale translator revision", dict(record_2d, translator_revision=2),
                        "Unsupported translator revision")
    expect_record_error("record: missing binding metadata", dict(record_2d, source_descriptor=0),
                        "share a table offset")
    expect_record_error("record: invalid group dimension", dict(record_2d, group_x=48),
                        "not a power of two")
    expect_record_error("record: incorrect root contract", dict(record_2d, root_parameters=((2, 0, 1),)),
                        "not the approved fixture contract")
    expect_record_error("record: code digest mismatch", dict(record_2d, code_sha256=bytes(32)),
                        "digest does not match its bytes")

    def expect_parse_error(name, blob, message):
        try:
            native.parse(blob)
        except native.RecordError as error:
            state(name, message in str(error), str(error))
        else:
            state(name, False, "accepted")

    expect_parse_error("record: truncated", serialized["2d"][:-4], "length does not match")
    expect_parse_error("record: oversized", b"X2WN" + bytes(native.MAX_RECORD), "larger than")
    expect_parse_error("record: bad magic", b"ZZZZ" + serialized["2d"][4:], "magic is not X2WN")

    # JSON intake
    payload = native.record_to_json(record_2d)
    for label, mutation, message in (
            ("unknown field", dict(payload, bogus=1), "Unknown record field"),
            ("missing field", {k: v for k, v in payload.items() if k != "profile"}, "Missing record field"),
            ("derived field", dict(payload, record_sha256="00"), "Unknown record field"),
            ("fixed field", dict(payload, version=2), "is fixed at 1")):
        try:
            native.record_from_json(mutation)
        except native.RecordError as error:
            state(f"json: {label} is a named error", message is None or message in str(error), str(error))
        else:
            state(f"json: {label} is a named error", False, "accepted")
    try:
        native.validate(native.record_from_json(dict(payload, group=[8, 4, 2])))
    except native.RecordError as error:
        state("json: a semantically invalid group is refused", "Profile 2d" in str(error), str(error))
    else:
        state("json: a semantically invalid group is refused", False, "accepted")
    try:
        native._no_duplicate_keys([("version", 1), ("version", 1)])
    except native.RecordError as error:
        state("json: duplicate field is a named error", "Duplicate" in str(error), str(error))
    else:
        state("json: duplicate field is a named error", False, "accepted")

    # ------------------------------------------------------------------ H1 manifest acceptance
    base = clone("acceptance")
    paths = native.cache_paths(base, keys["2d"])
    for field in native.MANIFEST_VALIDATED:
        target = clone_of(base, f"drop-{field}")
        fields = manifest_of(target, keys["2d"])
        fields.pop(field)
        rewrite_manifest(target, keys["2d"], fields)
        finding = native.inspect_entry(target, keys["2d"], REQUIRED)
        state(f"manifest: missing {field} is not a hit", finding["state"] == "corrupt",
              f'state={finding["state"]} {finding["detail"]}')

    for field in IDENTITY_MUTATIONS:
        target = clone_of(base, f"identity-{field}")
        fields = manifest_of(target, keys["2d"])
        fields[field] = "foreign/0000000000000000"
        rewrite_manifest(target, keys["2d"], fields)
        state(f"manifest: foreign {field} is stale",
              native.entry_state(target, keys["2d"], REQUIRED)[0] == "stale")

    for field, bad in (("record_version", "9"), ("validation", "container=0x80004005 module=0x00000000"),
                       ("rts0", "a test marker"), ("gate", "DENY"), ("signed", "unsigned"),
                       ("output_bytes", "1"), ("code_sha256", "00")):
        target = clone_of(base, f"evidence-{field}")
        fields = manifest_of(target, keys["2d"])
        fields[field] = bad
        rewrite_manifest(target, keys["2d"], fields)
        finding = native.inspect_entry(target, keys["2d"], REQUIRED)
        state(f"manifest: invalid {field} is not a hit", finding["state"] == "corrupt",
              f'state={finding["state"]} {finding["detail"]}')

    target = clone_of(base, "manifest-key")
    fields = manifest_of(target, keys["2d"])
    fields["key"] = native.key_text(keys["2d"] ^ 1)
    rewrite_manifest(target, keys["2d"], fields)
    state("manifest: a foreign cache key is stale",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "stale")

    target = clone_of(base, "failed-hresult")
    fields = manifest_of(target, keys["2d"])
    fields["pso-warp"] = fields["pso-warp"].replace("root=0x00000000", "root=0x80004005")
    rewrite_manifest(target, keys["2d"], fields)
    state("manifest: a failed HRESULT is not a hit",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "backend-missing")

    target = clone_of(base, "missing-cso")
    native.cache_paths(target, keys["2d"])["output"].unlink()
    state("manifest: missing CSO is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")
    target = clone_of(base, "missing-hlsl")
    native.cache_paths(target, keys["2d"])["hlsl"].unlink()
    state("manifest: missing HLSL is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")
    target = clone_of(base, "missing-request")
    native.cache_paths(target, keys["2d"])["request"].unlink()
    state("manifest: missing request is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")

    target = clone_of(base, "tampered-cso")
    tampered_blob = bytearray(native.cache_paths(target, keys["2d"])["output"].read_bytes())
    tampered_blob[-1] ^= 0xFF
    native.cache_paths(target, keys["2d"])["output"].write_bytes(bytes(tampered_blob))
    state("manifest: tampered CSO is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")
    target = clone_of(base, "tampered-hlsl")
    tampered_hlsl = bytearray(native.cache_paths(target, keys["2d"])["hlsl"].read_bytes())
    tampered_hlsl[-1] ^= 0xFF
    native.cache_paths(target, keys["2d"])["hlsl"].write_bytes(bytes(tampered_hlsl))
    state("manifest: tampered HLSL is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")

    target = clone_of(base, "unsigned")
    signed = bytearray(native.cache_paths(target, keys["2d"])["output"].read_bytes())
    signed[4:20] = bytes(16)
    native.cache_paths(target, keys["2d"])["output"].write_bytes(bytes(signed))
    fields = manifest_of(target, keys["2d"])
    fields["output_sha256"] = sha_bytes(bytes(signed))
    fields["output_bytes"] = str(len(signed))
    rewrite_manifest(target, keys["2d"], fields)
    finding = native.inspect_entry(target, keys["2d"], REQUIRED)
    state("manifest: zeroed signing digest is not a hit", finding["state"] == "corrupt",
          finding["detail"])

    target = clone_of(base, "root-contract")
    root_blob = bytearray(native.cache_paths(target, keys["2d"])["output"].read_bytes())
    offset = part_offset(bytes(root_blob), b"RTS0") + 8
    struct.pack_into("<I", root_blob, offset + 20, 1)
    native.cache_paths(target, keys["2d"])["output"].write_bytes(bytes(root_blob))
    fields = manifest_of(target, keys["2d"])
    fields["output_sha256"] = sha_bytes(bytes(root_blob))
    fields["output_bytes"] = str(len(root_blob))
    rewrite_manifest(target, keys["2d"], fields)
    finding = native.inspect_entry(target, keys["2d"], REQUIRED)
    state("manifest: foreign root flags are not a hit",
          finding["state"] == "corrupt" and "root contract" in finding["detail"], finding["detail"])

    target = clone_of(base, "duplicate")
    text = native.cache_paths(target, keys["2d"])["manifest"].read_text(encoding="utf-8")
    native.cache_paths(target, keys["2d"])["manifest"].write_text(
        text + "status validated\n", encoding="utf-8")
    state("manifest: contradictory field is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")
    target = clone_of(base, "bad-header")
    text = native.cache_paths(target, keys["2d"])["manifest"].read_text(encoding="utf-8")
    native.cache_paths(target, keys["2d"])["manifest"].write_text(
        text.replace(native.MANIFEST_HEADER, "x2w-native-compute 1", 1), encoding="utf-8")
    state("manifest: foreign header is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")
    target = clone_of(base, "unknown-field")
    text = native.cache_paths(target, keys["2d"])["manifest"].read_text(encoding="utf-8")
    native.cache_paths(target, keys["2d"])["manifest"].write_text(text + "bogus 1\n", encoding="utf-8")
    state("manifest: unknown field is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")

    target = clone_of(base, "refusal-digest")
    native.cache_paths(target, keys["2d"])["manifest"].write_bytes(native.manifest_text(sorted({
        "key": native.key_text(keys["2d"]), "revision": native.REVISION, "status": "refused",
        "record_version": "1", "record_sha256": "00" * 32, "code_bytes": "1", "code_sha256": "00",
        "reason": "mock refusal",
    }.items())).encode("ascii"))
    state("manifest: refusal with a foreign record digest is corrupt",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt")

    target = clone_of(base, "error-marker")
    native.cache_paths(target, keys["2d"])["error"].write_text("mock: retryable\n", encoding="ascii")
    state("manifest: an error marker dominates a valid entry",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "error")
    target = clone_of(base, "corrupt-marker")
    native.cache_paths(target, keys["2d"])["corrupt"].write_text("mock: malformed request\n", encoding="ascii")
    state("manifest: a corrupt-request marker dominates",
          native.entry_state(target, keys["2d"], REQUIRED)[0] == "corrupt-request")

    # ------------------------------------------------------------------ H2 writer ownership
    locked = clone("locked")
    locking = native.writer_lock(locked, keys["2d"])
    writer = locking.__enter__()
    manifest_before = sha_file(native.cache_paths(locked, keys["2d"])["manifest"])
    outcomes = native.stage_compile(locked, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator)
    state("lock: a fresh held lock makes the stage busy",
          any(outcome == "busy" for _, outcome, _ in outcomes), outcomes)
    state("lock: contention does not create an error marker",
          not native.cache_paths(locked, keys["2d"])["error"].exists())
    state("lock: contention does not mutate the manifest",
          sha_file(native.cache_paths(locked, keys["2d"])["manifest"]) == manifest_before)
    state("lock: the live writer still owns the lock", writer.assert_owner() is None)
    old = time.time() - 10000
    os.utime(native.cache_paths(locked, keys["2d"])["lock"], (old, old))
    stale_outcomes = native.stage_compile(locked, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                          validator=args.validator)
    state("lock: a lock older than 600 s is never reclaimed",
          any(outcome == "busy" for _, outcome, _ in stale_outcomes)
          and native.cache_paths(locked, keys["2d"])["lock"].exists(), stale_outcomes)
    state("lock: contention leaves the entry a hit",
          native.entry_state(locked, keys["2d"], REQUIRED)[0] == "hit")
    native.cache_paths(locked, keys["2d"])["lock"].write_text("other-writer-token\n", encoding="ascii")
    locking.__exit__(None, None, None)
    state("lock: release does not remove another writer's lock",
          native.cache_paths(locked, keys["2d"])["lock"].read_text(encoding="ascii").strip()
          == "other-writer-token")
    native.cache_paths(locked, keys["2d"])["lock"].unlink()
    retried = native.stage_compile(locked, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                   validator=args.validator)
    state("lock: a released lock lets the next run proceed",
          all(outcome == "hit" for _, outcome, _ in retried), retried)

    vanished = clone("vanished")
    vanishing = native.writer_lock(vanished, keys["2d"])
    vanished_writer = vanishing.__enter__()
    native.cache_paths(vanished, keys["2d"])["lock"].unlink()
    try:
        vanished_writer.assert_owner()
        vanished_ok = False
    except native.Retryable:
        vanished_ok = True
    vanishing.__exit__(None, None, None)
    state("lock: a lock that disappears during the job stops the writer", vanished_ok)
    state("lock: release after a vanished lock does not raise",
          not native.cache_paths(vanished, keys["2d"])["lock"].exists())

    nonowner = clone("nonowner")
    bad_writer = native.Writer(native.cache_paths(nonowner, keys["2d"])["lock"], "not-the-owner")
    for function, label in ((lambda: native.publish_error(bad_writer, nonowner, keys["2d"], "mock"),
                             "error"),
                            (lambda: native.publish_refusal(bad_writer, nonowner, keys["2d"],
                                                            record_2d, "mock"), "refusal")):
        try:
            function()
            state(f"lock: a non-owner cannot publish an {label}", False, "accepted")
        except native.Retryable:
            state(f"lock: a non-owner cannot publish an {label}", True)
    state("lock: the non-owner wrote nothing",
          not native.cache_paths(nonowner, keys["2d"])["error"].exists()
          and native.entry_state(nonowner, keys["2d"], REQUIRED)[0] == "hit")

    # ------------------------------------------------------------------ H3 publication
    interrupted = clone("interrupted")
    native.cache_paths(interrupted, keys["2d"])["manifest"].unlink()
    with native.writer_lock(interrupted, keys["2d"]) as writer:
        with mock.patch.object(native, "write_atomic", side_effect=[None, None, OSError("mock: interrupted")]):
            try:
                native.publish_validated(writer, interrupted, keys["2d"], verified_cache["record"],
                                         verified_cache["hlsl"], verified_cache["output"],
                                         validation_2d, gate,
                                         native.verified_backends(verified_cache["fields"]),
                                         native.expected_identity())
            except OSError:
                pass
    state("publication: an interrupted validated publish is never a hit",
          native.entry_state(interrupted, keys["2d"], REQUIRED)[0] == "missing",
          native.entry_state(interrupted, keys["2d"], REQUIRED))

    refusal_interrupted = clone("refusal-interrupted")
    with native.writer_lock(refusal_interrupted, keys["2d"]) as writer:
        with mock.patch.object(native, "write_atomic", side_effect=OSError("mock: interrupted")):
            try:
                native.publish_refusal(writer, refusal_interrupted, keys["2d"], record_2d, "mock refusal")
            except OSError:
                pass
    state("publication: an interrupted refusal leaves the old entry valid",
          native.entry_state(refusal_interrupted, keys["2d"], REQUIRED)[0] == "hit")

    refused = clone("refused")
    with native.writer_lock(refused, keys["2d"]) as writer:
        native.publish_refusal(writer, refused, keys["2d"], record_2d, "mock refusal")
    state("publication: a refusal is visible before its artifacts are removed",
          native.entry_state(refused, keys["2d"], REQUIRED)[0] == "refused"
          and not native.cache_paths(refused, keys["2d"])["output"].exists()
          and not native.cache_paths(refused, keys["2d"])["hlsl"].exists())

    stale_error = clone("stale-error")
    native.cache_paths(stale_error, keys["2d"])["error"].write_text("mock: stale\n", encoding="ascii")
    state("publication: a stale error marker blocks acceptance",
          native.entry_state(stale_error, keys["2d"], REQUIRED)[0] == "error")
    with native.writer_lock(stale_error, keys["2d"]) as writer:
        native.publish_validated(writer, stale_error, keys["2d"], verified_cache["record"],
                                 verified_cache["hlsl"], verified_cache["output"], validation_2d,
                                 gate, native.verified_backends(verified_cache["fields"]),
                                 native.expected_identity())
    state("publication: a successful publish clears the error marker",
          native.entry_state(stale_error, keys["2d"], REQUIRED)[0] == "hit"
          and not native.cache_paths(stale_error, keys["2d"])["error"].exists())

    incomplete = clone("incomplete")
    native.cache_paths(incomplete, keys["2d"])["manifest"].unlink()
    state("reader: artifacts without a manifest are not a hit",
          native.entry_state(incomplete, keys["2d"], REQUIRED)[0] == "missing")

    mixed = clone("mixed")
    native.cache_paths(mixed, keys["2d"])["output"].write_bytes(
        native.cache_paths(mixed, keys["1d"])["output"].read_bytes())
    finding = native.inspect_entry(mixed, keys["2d"], REQUIRED)
    state("reader: a mixed-generation artifact is rejected",
          finding["state"] == "corrupt" and "output" in finding["detail"], finding["detail"])

    # ------------------------------------------------------------------ backend revalidation
    backend = clone("backend")
    fields = manifest_of(backend, keys["2d"])
    fields.pop("pso-hardware")
    rewrite_manifest(backend, keys["2d"], fields)
    state("backend: a WARP-only entry does not satisfy a hardware request",
          native.entry_state(backend, keys["2d"], ("hardware",))[0] == "backend-missing")
    verdict = native.verify_entry(backend, keys["2d"], REQUIRED)
    state("backend: consumer verification refuses a backend-missing entry",
          not verdict["accepted"] and verdict["state"] == "backend-missing")
    state("backend: the same entry satisfies a WARP-only request",
          native.entry_state(backend, keys["2d"], ("warp",))[0] == "hit")
    revalidated = native.stage_compile(backend, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                       validator=args.validator, evidence=evidence_writer)
    state("backend: the missing backend is revalidated without recompiling",
          any(outcome == "revalidated" for _, outcome, _ in revalidated), revalidated)
    state("backend: revalidation restores the hardware evidence",
          native.entry_state(backend, keys["2d"], REQUIRED)[0] == "hit")

    revalidation_failed = clone("revalidation-failed")
    fields = manifest_of(revalidation_failed, keys["2d"])
    fields.pop("pso-hardware")
    rewrite_manifest(revalidation_failed, keys["2d"], fields)
    failed = native.stage_compile(revalidation_failed, args.dxc, args.probe, stubs["pso_fail"],
                                  REQUIRED, validator=args.validator)
    state("backend: a failed revalidation is retryable, never a hit",
          any(outcome == "error" for _, outcome, _ in failed)
          and native.entry_state(revalidation_failed, keys["2d"], REQUIRED)[0] == "error", failed)

    # ------------------------------------------------------------------ H6 corrupt requests
    corrupt = clone("corrupt-request")
    native.cache_paths(corrupt, keys["2d"])["request"].write_bytes(b"X2WN" + bytes(200))
    outcomes = native.stage_compile(corrupt, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator)
    state("request: a malformed request is quarantined, not retried",
          any(outcome == "quarantined" for _, outcome, _ in outcomes)
          and native.cache_paths(corrupt, keys["2d"])["corrupt"].exists()
          and not native.cache_paths(corrupt, keys["2d"])["error"].exists()
          and native.entry_state(corrupt, keys["2d"], REQUIRED)[0] == "corrupt-request", outcomes)
    state("request: a quarantined request is not published as a refusal",
          not (native.cache_paths(corrupt, keys["2d"])["manifest"].exists()
               and manifest_of(corrupt, keys["2d"]).get("status") == "refused"))

    foreign = clone("foreign-request")
    other = native.cache_paths(foreign, keys["1d"])["request"].read_bytes()
    native.cache_paths(foreign, keys["2d"])["request"].write_bytes(other)
    outcomes = native.stage_compile(foreign, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator)
    state("request: a foreign request filename key is quarantined",
          any(outcome == "quarantined" for _, outcome, _ in outcomes)
          and native.entry_state(foreign, keys["2d"], REQUIRED)[0] == "corrupt-request", outcomes)

    # ------------------------------------------------------------------ split-stage receipt
    receipt_missing = clone("receipt-missing")
    native.cache_paths(receipt_missing, keys["2d"])["manifest"].unlink()
    native.cache_paths(receipt_missing, keys["2d"])["request_manifest"].write_bytes(
        native.manifest_text([
            ("key", native.key_text(keys["2d"])), ("revision", native.REVISION), ("status", "request"),
            ("record_version", "1"),
            ("record_sha256", native.record_digest(record_2d).hex()),
            ("code_bytes", len(record_2d["code"])), ("code_sha256", native.digest_hex(record_2d["code"])),
        ]).encode("ascii"))
    outcomes = native.stage_compile(receipt_missing, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator)
    state("receipt: compilation without a translation receipt is retryable",
          any(outcome == "error" and "translation has not run" in detail for _, outcome, detail in outcomes)
          and not native.cache_paths(receipt_missing, keys["2d"])["manifest"].exists(), outcomes)

    receipt_tampered = clone("receipt-tampered")
    native.cache_paths(receipt_tampered, keys["2d"])["manifest"].unlink()
    hlsl_path = native.cache_paths(receipt_tampered, keys["2d"])["hlsl"]
    hlsl_path.write_bytes(hlsl_path.read_bytes() + b"// tampered after translation\n")
    outcomes = native.stage_compile(receipt_tampered, args.dxc, args.probe, args.pso_probe, REQUIRED,
                                    validator=args.validator)
    state("receipt: an HLSL changed after translation is refused",
          any(outcome == "error" and "does not describe this HLSL" in detail
              for _, outcome, detail in outcomes)
          and not native.cache_paths(receipt_tampered, keys["2d"])["manifest"].exists(), outcomes)

    # ------------------------------------------------------------------ refusal publication failure
    refusal_failure = clone("refusal-failure")
    native.cache_paths(refusal_failure, keys["2d"])["manifest"].unlink()
    with mock.patch.object(native, "compile_dxil", return_value=(b"DXBC" + bytes(28), ["DXIL", "RTS0"])), \
         mock.patch.object(native, "validate_and_sign", side_effect=native.Refusal("mock: intrinsic gate")), \
         mock.patch.object(native, "write_atomic", side_effect=OSError("mock: publication failed")):
        outcomes = native.stage_compile(refusal_failure, args.dxc, args.probe, args.pso_probe,
                                        REQUIRED, validator=args.validator)
    state("publication: a refusal that could not be published is an error, not a refusal",
          any(outcome == "error" and "refusal publication failed" in detail
              for _, outcome, detail in outcomes), outcomes)

    # ------------------------------------------------------------------ toolchain identity
    mismatch = clone("toolchain")
    foreign = fresh("foreign-toolchain")
    (foreign / "dxc.exe").write_bytes(b"not a compiler")
    (foreign / "dxcompiler.dll").write_bytes(b"not the pinned compiler")
    (foreign / "dxil.dll").write_bytes(b"not the pinned validator")
    try:
        native.stage_compile(mismatch, foreign / "dxc.exe", args.probe, args.pso_probe, REQUIRED)
        state("toolchain: a foreign compiler hash is refused before any mutation", False, "accepted")
    except native.ToolchainMismatch as error:
        state("toolchain: a foreign compiler hash is refused before any mutation", True, str(error))
    state("toolchain: the refusal mutated nothing",
          native.entry_state(mismatch, keys["2d"], REQUIRED)[0] == "hit")

    # ------------------------------------------------------------------ infrastructure mocks
    work = fresh("mock-work")
    for label, call, message in (
            ("dxc failure", lambda: native.compile_dxil(hlsl, stubs["dxc_fail"], work), "dxc failed"),
            ("non-DXBC output", lambda: native.compile_dxil(hlsl, stubs["dxc_garbage"], work), "DXBC"),
            ("validator failure", lambda: native.validate_and_sign(blob, stubs["probe_fail"]),
             "Validation failed"),
            ("root creation failure", lambda: native.collect_backends(
                blob, inventory, stubs["pso_root_fail"], ("warp",)), "root signature creation"),
            ("missing PSO result", lambda: native.collect_backends(
                blob, inventory, stubs["pso_missing"], ("warp",)), "compute PSO creation"),
            ("PSO creation failure", lambda: native.collect_backends(
                blob, inventory, stubs["pso_fail"], ("warp",)), "compute PSO creation"),
            ("malformed probe JSON", lambda: native.collect_backends(
                blob, inventory, stubs["pso_bad_json"], ("warp",)), "not JSON")):
        try:
            call()
        except native.Retryable as error:
            state(f"mock: {label} is retryable", message in str(error), str(error))
        else:
            state(f"mock: {label} is retryable", False, "accepted")

    # -- evidence ---------------------------------------------------------------------
    for name in ("1d", "2d"):
        paths = native.cache_paths(cache, keys[name])
        (evidence / f"{name}.manifest").write_bytes(paths["manifest"].read_bytes())
        (evidence / f"{name}.hlsl").write_bytes(paths["hlsl"].read_bytes())
        (evidence / f"{name}.cso").write_bytes(paths["output"].read_bytes())
        (evidence / f"{name}.request").write_bytes(paths["request"].read_bytes())
        (evidence / f"{name}.receipt").write_bytes(paths["request_manifest"].read_bytes())
    (evidence / "controls.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="ascii")
    summary = {"checks": len(checks), "passed": sum(1 for item in checks if item["ok"]),
               "failed": failures}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    if failures:
        print(f"B2 result: FAIL ({len(failures)} of {summary['checks']} checks)")
        return 1
    print(f"B2 result: PASS ({summary['checks']} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
