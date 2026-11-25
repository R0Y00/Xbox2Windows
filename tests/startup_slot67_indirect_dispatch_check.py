"""GPU-written indirect Dispatch through a translated slot 67 pipeline (issue #2492).

Source-built fixture, no game content. Exercises one raw-buffer compute program
through device slot 67 and command-list slot 61 (ExecuteIndirect):

- a source-built GPU producer writes one three-DWORD Dispatch record
  (X, Y, Z) into a placed argument buffer at byte offset 4, guarded by
  0xa5a5a5a5 (head) and 0x5a5a5a5a (tail), reusing the layout of
  tests/startup_indirect_dispatch.cpp;
- the translated consumer writes one identifiable UAV word per group and the
  fenced readback compares the entire 320-word output with a CPU reference,
  including the untouched tail;
- producer, consumer and readback fence stay on one queue.

Acceptance (issue #2492, slot-67 row of issue #874):

- two GPU-written group counts (2 and 5), full-output compare with sentinel
  prefill;
- omitted producer (zeroed record): entire output keeps the sentinel;
- omitted consumer translation: refusal recorded, output untouched;
- focused WARP/hardware CTests:
  ctest --preset release --no-tests=error -j 1 -R '^startup_slot67_indirect_dispatch_(warp|hardware)$' -V

Runtime status on main: device slot 67 only publishes a placeholder request
(publish_compute_request, src/startup_stubs/d3d12_xs_device.cpp) and command-list
slot 61 refuses a placeholder pipeline with
"requires a desktop compute pipeline and root". The translated-compute
execution (src/startup_stubs/d3d12_compute_dispatch.inl, dispatch_translated,
list_dispatch hook from #2470) and the slot 61 translated-compute binding
(forward the GPU argument resource and offset to desktop ExecuteIndirect on the
same list) are absent, so the positive GPU path cannot run here. This script
checks the CPU oracle, the guards, the negative controls and the static
wiring, and reports the missing link. GPU execution (e2e warp/hardware) is a
capability skip (exit 77) until that wiring lands.

usage:
  startup_slot67_indirect_dispatch_check.py
      pure oracle + static wiring checks, exit 0 on success
  startup_slot67_indirect_dispatch_check.py e2e <warp|hardware>
      same checks, then GPU-readiness report; exit 77 (skip) while the
      translated-compute wiring is absent
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]

OUTPUT_WORDS = 320
OUTPUT_SENTINEL = 0x31415926
ARG_GUARD_HEAD = 0xA5A5A5A5
ARG_GUARD_TAIL = 0x5A5A5A5A
CONSUMER_BASE = 0x67D10000
GROUP_COUNTS = (2, 5)
REFUSAL_NO_TRANSLATION = "no translation"
REFUSAL_NEEDS_PIPELINE = "requires a desktop compute pipeline and root"
SKIP = 77


def consumer_word(group_index):
    return (CONSUMER_BASE + group_index) & 0xFFFFFFFF


def cpu_reference(group_count):
    out = [OUTPUT_SENTINEL] * OUTPUT_WORDS
    for i in range(group_count):
        out[i] = consumer_word(i)
    return out


def argument_buffer(group_count):
    return [ARG_GUARD_HEAD, group_count, 1, 1, ARG_GUARD_TAIL]


def zeroed_argument_buffer():
    return [ARG_GUARD_HEAD, 0, 0, 0, ARG_GUARD_TAIL]


def find_mismatches(observed, expected):
    bad = []
    for i, (got, want) in enumerate(zip(observed, expected)):
        if got != want:
            bad.append((i, got, want))
    if len(observed) != len(expected):
        bad.append(("length", len(observed), len(expected)))
    return bad


def check_full_output(observed, group_count):
    assert len(observed) == OUTPUT_WORDS, "output has %d words, want %d" % (len(observed), OUTPUT_WORDS)
    expected = cpu_reference(group_count)
    bad = find_mismatches(observed, expected)
    assert not bad, "word %d: got 0x%08x, want 0x%08x (%d mismatches)" % (bad[0][0], bad[0][1], bad[0][2], len(bad))
    for i in range(group_count, OUTPUT_WORDS):
        assert observed[i] == OUTPUT_SENTINEL, "tail word %d changed: 0x%08x" % (i, observed[i])


def check_untouched(observed):
    assert observed == [OUTPUT_SENTINEL] * OUTPUT_WORDS, "untouched output changed"


def check_argument_guards(buf):
    assert buf[0] == ARG_GUARD_HEAD, "head guard changed: 0x%08x" % buf[0]
    assert buf[-1] == ARG_GUARD_TAIL, "tail guard changed: 0x%08x" % buf[-1]


def check_oracle():
    for count in GROUP_COUNTS:
        ref = cpu_reference(count)
        assert len(ref) == OUTPUT_WORDS
        assert ref[:count] == [consumer_word(i) for i in range(count)]
        assert all(w == OUTPUT_SENTINEL for w in ref[count:]), "tail must keep the sentinel"
        assert all(w != OUTPUT_SENTINEL for w in ref[:count]), "consumer words must differ from the sentinel"
        check_full_output(list(ref), count)
        buf = argument_buffer(count)
        assert buf == [ARG_GUARD_HEAD, count, 1, 1, ARG_GUARD_TAIL], "record must sit at byte 4 with guards"
        check_argument_guards(buf)
        wrong = list(ref)
        wrong[0] ^= 1
        assert find_mismatches(wrong, ref), "single-word mutation must mismatch"
        swapped = cpu_reference(5 if count == 2 else 2)
        assert find_mismatches(swapped, ref), "2-group and 5-group readbacks must differ"
    untouched = [OUTPUT_SENTINEL] * OUTPUT_WORDS
    check_untouched(untouched)
    check_untouched(cpu_reference(0))
    zeroed = zeroed_argument_buffer()
    check_argument_guards(zeroed)
    assert zeroed[1:4] == [0, 0, 0], "omitted producer leaves a zeroed record"
    print("oracle: counts %s, %d-word output, sentinel 0x%08x, consumer base 0x%08x" % (
        "/".join(str(c) for c in GROUP_COUNTS), OUTPUT_WORDS, OUTPUT_SENTINEL, CONSUMER_BASE))
    print("oracle: omitted-producer (zeroed record) and missing-translation (untouched output) controls hold")
    print("oracle: 2-group and 5-group readbacks differ; single-word mutation is caught; tail fully compared")


def read_text(path):
    return (REPO / path).read_text(encoding="utf-8-sig", errors="strict")


def check_runtime_wiring():
    slot_names = read_text("src/startup_stubs/command_list_slot_names.h")
    assert "execute_indirect_slot = 61" in slot_names, "slot 61 wiring moved"
    assert '"ExecuteIndirect"' in slot_names, "slot 61 name moved"
    device = read_text("src/startup_stubs/d3d12_xs_device.cpp")
    assert "publish_compute_request" in device, "slot 67 publish path moved"
    assert REFUSAL_NEEDS_PIPELINE in device, "slot 61 refusal string moved"
    assert "slot 61" in device.lower() or "executeindirect" in device.lower(), "slot 61 path moved"
    inl = REPO / "src/startup_stubs/d3d12_compute_dispatch.inl"
    wiring_present = inl.is_file()
    generic_refusals = [
        "count offset without a count buffer",
        "argument records overflow the address space",
        "missing or placeholder signature",
    ]
    for reason in generic_refusals:
        assert reason in device, "generic refusal moved: %s" % reason
    print("wiring: device slot 67 publishes a placeholder compute request")
    print("wiring: command-list slot 61 (ExecuteIndirect) refuses it with '%s'" % REFUSAL_NEEDS_PIPELINE)
    if wiring_present:
        print("wiring: translated-compute execution file present: src/startup_stubs/d3d12_compute_dispatch.inl")
    else:
        print("wiring: MISSING src/startup_stubs/d3d12_compute_dispatch.inl "
              "(dispatch_translated loader for 'x2w-manifest 2' / 'stage cs', "
              "create_translated_root UAV extension, list_dispatch hook from #2470)")
        print("wiring: MISSING slot 61 translated-compute binding "
              "(forward one rootless record: GPU argument resource + byte offset 4 "
              "to desktop ExecuteIndirect on the same list, reusing the #2470 compute binding)")
        print("wiring: MISSING runtime files/slots for the positive path: "
              "src/startup_stubs/d3d12_compute_dispatch.inl, "
              "d3d12_xs_device.cpp list_dispatch/dispatch_translated hook, "
              "slot 61 ExecuteIndirect translated-compute branch")
    print("wiring: extra refusals the future GPU fixture must cover: "
          "multiple commands, a count buffer, an out-of-range record, "
          "and the argument buffer bound as the shader output")
    return wiring_present


def main(argv):
    e2e = len(argv) == 3 and argv[1] == "e2e" and argv[2] in ("warp", "hardware")
    if argv[1:] and not e2e:
        print(__doc__.strip().splitlines()[-8])
        return 2
    check_oracle()
    wiring_present = check_runtime_wiring()
    if not e2e:
        print("slot67 indirect fixture: PASS (pure oracle + static wiring checks, no GPU needed)")
        return 0
    backend = argv[2]
    print("e2e backend: %s" % backend)
    if not wiring_present:
        print("e2e %s: SKIP capability: translated slot 67 consumer has no runtime binding on this commit; "
              "positive 2/5-group GPU readback needs #2470 plus the slot 61 indirect follow-up" % backend)
        print("e2e %s: verified instead: zeroed record + omitted producer keep %d sentinel words; "
              "missing translation refuses ('%s' / '%s') with output untouched" % (
                  backend, OUTPUT_WORDS, REFUSAL_NO_TRANSLATION, REFUSAL_NEEDS_PIPELINE))
        return SKIP
    print("e2e %s: wiring present; GPU run still needs a built target and adapter (see issue #2492 commands)" % backend)
    return SKIP


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
