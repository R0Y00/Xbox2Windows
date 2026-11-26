"""Exercise buffer heap indexing with synthetic free slots and mapped buffers."""
from pathlib import Path
import re
import sys
import tempfile

from startup_draw_uav_check import run_mode

IDENTITY = bytes((0, 0, 0x80, 0x3F))
VALUE = bytes((99, 0, 0, 0))
DECOY = bytes((55, 0, 0, 0))
ON = {"X2W_BUFFER_HEAP_INDEX": "1", "X2W_BUFFER_ADDRESS_INDEX": "check"}
REWRITE_LOG = dict(ON, X2W_MARKER_REWRITE_LOG="1")
MODES = {"membuf": ON, "membuf-off": {}, "membuf-link": ON, "membuf-structured": ON, "membuf-typed": ON, "membuf-refused": REWRITE_LOG,
         "membuf-unaligned": ON, "membuf-unaligned-off": {}, "membuf-rewrite": REWRITE_LOG}


def columns(image):
    """Pixels (x, y) for x < 2 of the 8x4 RGBA8 read-back: what the two strips drew."""
    return {image[(y * 8 + x) * 4:(y * 8 + x) * 4 + 4] for y in range(4) for x in range(2)}


def check(directory, consumer, backend):
    results = {}
    for mode, env in MODES.items():
        got = run_mode(directory, consumer, backend, mode, dict(env, X2W_DRAW_NULL_IDENTITY="1"))
        if got is None:
            print("no existing-heap support or no adapter: skip")
            return 77
        results[mode] = got
    for mode, (_, image, log) in results.items():
        if DECOY in columns(image):
            raise AssertionError(f"{mode}: the draw read the decoy at slot N (55)\n{log[-3000:]}")
    _, image, log = results["membuf"]
    if columns(image) != {VALUE}:
        raise AssertionError(f"membuf: pixels {image[:32].hex()} (expected 99 = {VALUE.hex()})\n{log[-4000:]}")
    # The switch-on modes run X2W_BUFFER_ADDRESS_INDEX=check: the index answers as the walk (here: no buffer).
    for mode in ("membuf", "membuf-structured", "membuf-typed"):
        lines = re.findall(r"^x2w-memsrv buffer address index checked (\d+) lookups against the walk: (\d+) differ$", results[mode][2], re.M)
        if not lines or any(d != "0" for _, d in lines) or "DIFFERS" in results[mode][2]:
            raise AssertionError(f"{mode}: address index check lines {lines}\n{results[mode][2][-3000:]}")
    for mode, layout in (("membuf", "stride 0, records 256, word 3 0x"), ("membuf-structured", "stride 16, records 4, word 3 0x"),
                         ("membuf-typed", "stride 4, records 64, word 3 0x"),
                         ("membuf-unaligned", "stride 12, records 4, word 3 0x")):
        _, image, log = results[mode]
        words = re.search(r"^membuf entry words: (\w+) (\w+) (\w+) (\w+)$", log, re.M)
        if columns(image) != {VALUE} or not words or words.group(1) in ("00000001", "56573258", "52463258") or words.group(4) == "00000000":
            raise AssertionError(f"{mode}: pixels {image[:32].hex()} (expected 99), entry {words.group(0) if words else None}\n{log[-4000:]}")
        written = re.search(r"^x2w-memsrv buffer descriptor written at .*$", log, re.M)
        if "X2W_BUFFER_HEAP_INDEX=1" not in log or not written or layout not in written.group(0) or "stand-in buffer at" in log:
            raise AssertionError(f"{mode}: descriptor line {written.group(0) if written else None} (expected {layout!r}), or a stand-in\n{log[-4000:]}")
    _, image, log = results["membuf-unaligned"]
    if "(a buffer is there, at an offset the view cannot name)" not in log \
            or "buffer descriptors: written 1 (no buffer 0, buffer at an offset the view cannot name 1)" not in log \
            or not re.search(r"per written call: buffer lookup \d+\.\d\d us, destination checks \d+\.\d\d us, shadow clear \d+\.\d\d us "
                             r"\(lock taken [01]\)", log):
        raise AssertionError(f"membuf-unaligned: the found-buffer case is not named\n{log[-4000:]}")
    _, image, log = results["membuf-unaligned-off"]
    if columns(image) != {IDENTITY} or "reason=buffer_unavailable" not in log or "buffer descriptor written" in log:
        raise AssertionError(f"membuf-unaligned-off: pixels {image[:32].hex()} (expected the identity fill)\n{log[-4000:]}")
    _, image, log = results["membuf-refused"]
    line = re.search(r"^x2w-frame srv-heapbuf .*$", log, re.M)
    if columns(image) != {IDENTITY} or "membuf entry words: 52463258" not in log or not line \
            or "(refused: null_address)" not in line.group(0) \
            or not re.search(r"refused view: fmt \d+ layout 0x[0-9a-f]+ dim 1 mips \d+ [1-9]\d*x1x1 address 0x0 unreadable, detile: -$",
                             line.group(0)):
        raise AssertionError(f"membuf-refused: pixels {image[:32].hex()}, line {line.group(0) if line else None}\n{log[-4000:]}")
    # membuf-refused is also the negative control of X2W_MARKER_REWRITE_LOG: an intact marker is not flagged.
    if "rewritten refused-view marker" in log or "X2W_MARKER_REWRITE_LOG=1" not in log:
        raise AssertionError(f"membuf-refused: an intact marker was flagged as rewritten, or the switch line is missing\n{log[-4000:]}")
    # membuf-rewrite: the title patches word 0 / word 1 bits 0-15 of the refused destination with a V# base after the
    # call; the copied entry is flagged, with the description and the base read back from the rewritten words.
    _, image, log = results["membuf-rewrite"]
    entry = re.search(r"^membuf entry words: ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8}) ([0-9a-f]{8})$", log, re.M)
    line = re.search(r"^x2w-draw rewritten refused-view marker .*$", log, re.M)
    if not entry or not line:
        raise AssertionError(f"membuf-rewrite: entry {entry.group(0) if entry else None}, line {line.group(0) if line else None}\n{log[-4000:]}")
    base = int(entry.group(1), 16) | (int(entry.group(2), 16) & 0xFFFF) << 32
    if entry.group(1) == "52463258" or f"as a V# 0x{base:x}" not in line.group(0) \
            or not re.search(r"described as fmt \d+ layout 0x[0-9a-f]+ dim 1 mips \d+ [1-9]\d*x1x1 address 0x0 unreadable, detile: -;",
                             line.group(0)) \
            or "x2w-draw rewritten refused-view markers 1" not in log:
        raise AssertionError(f"membuf-rewrite: base 0x{base:x}, line {line.group(0)}\n{log[-4000:]}")
    _, image, log = results["membuf-off"]
    if columns(image) != {IDENTITY} or "membuf entry words: 00000001 00000000 00000000 00000000" not in log \
            or "reason=buffer_unavailable" not in log or "X2W_BUFFER_HEAP_INDEX" in log:
        raise AssertionError(f"membuf-off: pixels {image[:32].hex()} (expected the identity fill {IDENTITY.hex()})\n{log[-4000:]}")
    _, image, log = results["membuf-link"]
    line = re.search(r"^x2w-frame srv-heapbuf .*$", log, re.M)
    if columns(image) != {IDENTITY} or not line or "(free link)" not in line.group(0) \
            or "slot N-1 0x0 words 00000001" not in line.group(0) or "slot N 0x1 token 56573258" not in line.group(0):
        raise AssertionError(f"membuf-link: pixels {image[:32].hex()}, line {line.group(0) if line else None}\n{log[-4000:]}")
    print(f"{backend}: slot-70 buffer without a resource read through a copied entry (99) with X2W_BUFFER_HEAP_INDEX; "
          f"switch off: link 1 copied, identity fill; free link: bound null and logged with slots N-1 / N; decoy at slot N never read; structured and typed views 99; a refused call's marker is carried, bound null and named")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        sys.exit(check(Path(temporary), *sys.argv[1:]))
