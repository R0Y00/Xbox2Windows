"""Dispatch condstore_256 compiled with and without DXC's GVN and compare the whole raw UAV.

Follow-up to #775: the minimized `condstore_N` family from tools/dxc_compile_time.py
spends most of its DXC time in Global Value Numbering, and `-opt-disable gvn` removes
that time. This check takes one small member, compiles it once with default
optimization and once with `-opt-disable gvn`, dispatches both blobs on one adapter
against identical inputs and compares the complete raw UAV bit for bit:

- with each other;
- with an expected buffer computed here, independently of the GPU.

Inputs (identical for every blob): the sampler fixture's bindings, with a 2x1 R32_FLOAT
texture (0.25, 0.75), wrap/clamp point samplers and LESS_EQUAL comparison samplers. The
70,656-byte raw UAV is pre-filled with `((i + 1) * 0x9E3779B1) | 1` in word i, and there
is one dispatch of 1 group and one of 2 groups.

Wrong-output controls, which must be detected:
- a mutated store: the same source with store k=101 redirected into the unused tail of
  the UAV, compiled with default optimization, must differ from the expected buffer at
  exactly those two words in both dispatches;
- a corrupted expected buffer, with one bit flipped, must make the comparison fail.

The shader target is cs_6_0 without native 16-bit types. The #775 compile-time
measurements use cs_6_6, but the Windows 10 in-box D3D12 runtime exposes neither SM 6.6
on WARP nor on the tested hardware adapter. The DXC version and the exact flags are printed.

Usage: dxc_gvn_output_check.py <dxc.exe> <offscreen_check.exe> <warp|hardware> [work-dir]
Exit 0 pass, 1 fail, 77 when the adapter reports a capability skip.
"""

import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import dxc_compile_time as tool  # noqa: E402

SKIP = 77
VARIANT = "condstore_256"
N = 256
TARGET = "cs_6_0"
BASE_WORD = 65536 // 4          # condstore store k lands in word BASE_WORD + k
TAIL_WORDS = 1024
UAV_BYTES = 65536 + 4 * N + 4 * TAIL_WORDS
MUTATED_K = 101
MUTATED_WORD = BASE_WORD + N + 64   # inside the tail, which no correct store writes
FILL_MULTIPLIER = 0x9E3779B1     # keep in step with raw_uav_fill_word in raw_uav_dump.cpp
CONFIGS = {
    "default": [],
    "no_gvn": ["-opt-disable", "gvn"],
}


def fill_word(index):
    return (((index + 1) * FILL_MULTIPLIER) | 1) & 0xFFFFFFFF


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def expected_words(groups):
    """The raw UAV after one dispatch of `groups` groups, from the HLSL semantics alone."""
    words = [fill_word(i) for i in range(UAV_BYTES // 4)]
    for x in range(groups):
        # Control stores: SampleLevel through _3[x] at u=1.25 (wrap for x=0, clamp for x=1)
        # and SampleCmpLevelZero through _6[x] with reference 0.5 and LESS_EQUAL.
        words[x] = f32_bits(0.25 if x == 0 else 0.75)
        words[x + 2] = f32_bits(0.0 if x == 0 else 1.0)
        for k in range(N):
            if ((x + k) & 1) != 0:
                words[BASE_WORD + k] = 0     # gl_WorkGroupID.y of a 1-row dispatch
    return words


def mutated_source():
    source = tool.variant_info(VARIANT)[2]
    old = f"_4.Store({65536 + 4 * MUTATED_K}u, gl_WorkGroupID.y);"
    new = f"_4.Store({4 * MUTATED_WORD}u, gl_WorkGroupID.y);"
    if source.count(old) != 1:
        raise RuntimeError("mutation site not found exactly once")
    return source.replace(old, new)


def words_of(blob):
    return list(struct.unpack(f"<{len(blob) // 4}I", blob))


def mismatches(actual, expected):
    return [i for i, (a, e) in enumerate(zip(actual, expected)) if a != e] + (
        [min(len(actual), len(expected))] if len(actual) != len(expected) else [])


def compile_blob(dxc, work, name, source, extra):
    hlsl = work / f"{name}.hlsl"
    hlsl.write_text(source, encoding="utf-8", newline="\n")
    flags = ["-T", TARGET, "-E", tool.DEFAULT_ENTRY, *extra]
    run = tool.run_once(dxc, hlsl, work / f"{name}.cso", flags, 120)
    print(f"compile {name}: {' '.join(flags)} -> {run['outcome']} {run['cso_bytes']} B "
          f"{run['wall_s']:.3f} s")
    if run["outcome"] != "dxil":
        raise RuntimeError(f"{name} did not compile: {run['first_diagnostic']}")
    return work / f"{name}.cso"


def dispatch(offscreen, adapter, cso, groups, out):
    result = subprocess.run([str(offscreen), adapter, "--raw-uav-dump", str(cso), str(groups),
                             str(UAV_BYTES), str(out)], capture_output=True, text=True)
    text = (result.stdout + result.stderr).strip()
    if result.returncode == SKIP:
        print(text)
        return None
    if result.returncode != 0:
        raise RuntimeError(f"offscreen_check failed ({result.returncode}) for {cso.name}:\n{text}")
    for line in text.splitlines():
        if line.startswith(("adapter=", "driver=")):
            print(line)
    return words_of(out.read_bytes())


def main(argv):
    if len(argv) not in (4, 5):
        print(__doc__)
        return 2
    dxc, offscreen, adapter = Path(argv[1]), Path(argv[2]), argv[3]
    print(f"DXC: {tool.dxc_version(dxc)}")
    print(f"variant {VARIANT}, target {TARGET}, UAV {UAV_BYTES} bytes, adapter {adapter}")
    with tempfile.TemporaryDirectory(prefix="dxc-gvn-output-") as temp:
        work = Path(argv[4]) if len(argv) == 5 else Path(temp)
        work.mkdir(parents=True, exist_ok=True)
        source = tool.variant_info(VARIANT)[2]
        blobs = {name: compile_blob(dxc, work, name, source, extra) for name, extra in CONFIGS.items()}
        blobs["mutated"] = compile_blob(dxc, work, "mutated", mutated_source(), [])
        digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in blobs.items()}
        for name, digest in digests.items():
            print(f"sha256 {name}: {digest}")
        failures = []
        if digests["default"] == digests["no_gvn"]:
            failures.append("default and -opt-disable gvn emitted identical containers; nothing to compare")

        for groups in (1, 2):
            expected = expected_words(groups)
            results = {}
            for name, cso in blobs.items():
                words = dispatch(offscreen, adapter, cso, groups, work / f"{name}_{groups}.bin")
                if words is None:
                    return SKIP
                results[name] = words
            written = sum(1 for i, w in enumerate(results["default"]) if w != fill_word(i))
            print(f"groups={groups}: {written} of {len(expected)} words differ from the fill")
            for name in CONFIGS:
                bad = mismatches(results[name], expected)
                print(f"groups={groups} {name} vs expected: {len(bad)} mismatching words")
                if bad:
                    i = bad[0]
                    failures.append(f"groups={groups} {name}: word {i} = 0x{results[name][i]:08x}, "
                                    f"expected 0x{expected[i]:08x} ({len(bad)} words)")
            bad = mismatches(results["default"], results["no_gvn"])
            print(f"groups={groups} default vs no_gvn: {len(bad)} mismatching words")
            if bad:
                failures.append(f"groups={groups}: default and no_gvn differ at word {bad[0]}")

            # Wrong-output control 1: the mutated store must be caught at exactly its two words.
            bad = mismatches(results["mutated"], expected)
            print(f"groups={groups} control mutated-store vs expected: mismatching words {bad}")
            if bad != [BASE_WORD + MUTATED_K, MUTATED_WORD]:
                failures.append(f"groups={groups}: mutated store control gave {bad}, "
                                f"expected [{BASE_WORD + MUTATED_K}, {MUTATED_WORD}]")
            # Wrong-output control 2: a one-bit corruption of the expected buffer must be caught.
            corrupted = list(expected)
            corrupted[BASE_WORD + 1] ^= 1
            bad = mismatches(results["no_gvn"], corrupted)
            print(f"groups={groups} control corrupted-expected: mismatching words {bad}")
            if bad != [BASE_WORD + 1]:
                failures.append(f"groups={groups}: corrupted expected control gave {bad}")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    print("dxc_gvn_output_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
