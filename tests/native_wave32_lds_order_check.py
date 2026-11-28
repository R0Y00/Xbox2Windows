"""LDS order inside one host wave32: each lane reads the word that another lane wrote one instruction earlier.

Usage: native_wave32_lds_order_check.py controls|hardware|warp COMPILE_DXIL NATIVE_STORE_REPLAY DXC_DIR

One guest wave32 per 32-thread group. Lane l writes LDS word l, then reads word 31 - l with no barrier in
between, and stores what it read. The same words also run as a narrow guest wave64 (a wave64 request of 32
threads), which the direct path translates on the wave32 model. On that model one host wave is one guest
wave and no group barrier stands before an LDS access, so each access takes its lane predicate from a
WaveActiveBallot: a wave operation brings every lane of the host wave to that point.

controls needs no GPU and checks that ballot in both modules. hardware replays with --direct, warp with
--direct-warp; both return 77 when replay does. The in-box WARP has 4-lane waves and skips. App-local WARP
1.0.22 runs it (see docs/experiments/native-compute-warp.md), and read unwritten words without the ballot.
"""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_wave64_lane_pairs_check import (
    GROUPS, OUTPUT_WORDS, blank, compile_batch, dumpbin, first_mismatch, put, read_words, root_signature, value)

# RSRC2: 5 user SGPRs (s4 = UAV table), TGID x in s5; LDS 1 x 128 words.
RSRC2 = (5 << 1) | 0x80 | (1 << 15)
THREADS = 32

# Assembled with LLVM 15.0.7: clang -cc1as -triple amdgcn-amd-amdhal -target-cpu gfx1013.
PROGRAM = (
    0xBE940304,              # s_mov_b32 s20, s4
    0xBE950380,              # s_mov_b32 s21, 0
    0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
    0x8F078B05,              # s_lshl_b32 s7, s5, 11          (group byte offset)
    0x34100084,              # v_lshlrev_b32_e32 v8, 4, v0     (thread byte offset)
    0x4A101007,              # v_add_nc_u32_e32 v8, s7, v8
    0xD7650001, 0x000100C1,  # v_mbcnt_lo_u32_b32 v1, -1, 0    (lane)
    0x34120282,              # v_lshlrev_b32_e32 v9, 2, v1
    0x34040088,              # v_lshlrev_b32_e32 v2, 8, v0
    0x38040485,              # v_or_b32_e32 v2, 5, v2
    0xD8340000, 0x00000209,  # ds_write_b32 v9, v2              (word l)
    0x4C14029F,              # v_sub_nc_u32_e32 v10, 31, v1
    0x34141482,              # v_lshlrev_b32_e32 v10, 2, v10
    0xBF8CC07F,              # s_waitcnt lgkmcnt(0)
    0xD8D80000, 0x0300000A,  # ds_read_b32 v3, v10              (word 31 - l, no barrier)
    0xBF8CC07F,              # s_waitcnt lgkmcnt(0)
    0xE0701000, 0x80080308,  # buffer_store_dword v3, v8, s[32:35], 0 offen
    0xBF810000,              # s_endpgm
)

# name: dword 4 of the request (z = 1; bit 20 set for a wave32 request)
REQUESTS = {"wave32": 1 | (1 << 20), "narrow_wave64": 1}

BALLOT = re.compile(r"call %dx\.types\.fouri32 @dx\.op\.waveActiveBallot\(")
LDS = re.compile(r"(?:\bload i32, |\bstore i32 \S+, )i32 addrspace\(3\)\*")


def model(source=lambda lane: 31 - lane):
    out = blank()
    for group in range(GROUPS):
        for lane in range(THREADS):
            put(out, group, lane, 0, value(source(lane)))
    return out


def build(work):
    requests = work / "requests"
    requests.mkdir()
    root = (work / "root.bin").read_bytes()
    markers = {}
    for name, dword4 in REQUESTS.items():
        sa = struct.pack("<7I", 0, 0, 0, THREADS | (1 << 16), dword4, 0, RSRC2)
        markers[name] = write_request(requests, PROGRAM, root, sa)
    return requests, markers


def controls(cli, tool, sdk, work):
    del tool
    requests, markers = build(work)
    result, _, batch = compile_batch(cli, sdk, work, "controls", requests)
    assert result.returncode == 0, (result.stdout, result.stderr)
    for name, marker in markers.items():
        ir = dumpbin(sdk, batch / f"{marker.stem[4:]}.cs.cso")
        assert "NumThreads=(32,1,1)" in ir and "waveExchange" not in ir, (name, ir)
        # The entry ballot (EXEC), then one ballot before the LDS write and one before the LDS read.
        events = ["ballot" if BALLOT.search(line) else "lds" for line in ir.splitlines()
                  if BALLOT.search(line) or LDS.search(line)]
        assert events == ["ballot", "ballot", "lds", "ballot", "lds"], (name, events)
        print(f"{name}, {THREADS}x1x1 threads: a ballot before each of the two LDS accesses")
    return 0


def run(cli, tool, sdk, work, mode):
    requests, markers = build(work)
    result, cache, _ = compile_batch(cli, sdk, work, "run", requests)
    assert result.returncode == 0, (result.stdout, result.stderr)
    readbacks = {}
    for name, marker in markers.items():
        output = work / f"{name}.bin"
        replay = subprocess.run([tool, sdk, marker, cache, "0", output, mode], capture_output=True, text=True)
        if replay.returncode == 77:
            print(f"{name}: replay {mode} returned 77 (adapter lacks the required capabilities)")
            return 77
        assert replay.returncode == 0, (name, replay.stdout, replay.stderr)
        if not readbacks:  # name the adapter once; --direct-warp does not report it
            print("\n".join(line for line in replay.stderr.splitlines() if line.startswith(("adapter=", "driver="))))
        readbacks[name] = read_words(output)
    for name, readback in readbacks.items():
        mismatch = first_mismatch(readback, model())
        if mismatch:
            index, got, want = mismatch
            raise AssertionError(f"{name} ({mode}): mismatch at word {index}, got {got:#010x}, expected {want:#010x}")
        print(f"{name}, {THREADS}x1x1 threads ({mode}): {OUTPUT_WORDS} words with guards matched the model")
        # Negative control: each lane reading its own word must not match.
        own = first_mismatch(readback, model(lambda lane: lane))
        assert own, (name, mode, "own-word model matched")
        index, got, want = own
        print(f"negative, {name}, own word: mismatch at word {index} (got {got:#010x}, expected {want:#010x})")
    return 0


def main(mode, cli, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        root_signature(sdk, work)
        cli, tool = Path(cli).resolve(), Path(tool).resolve()
        if mode == "controls":
            return controls(cli, tool, sdk, work)
        return run(cli, tool, sdk, work, {"hardware": "--direct", "warp": "--direct-warp"}[mode])


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
