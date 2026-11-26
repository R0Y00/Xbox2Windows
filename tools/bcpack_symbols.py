"""Decode bounded MDU rANS streams into lane symbols, without texture reconstruction."""
import argparse
import json
from pathlib import Path
import stat

from _bcpack_rans import BCPackError, decode_lanes

MAX_INPUT = 2 << 20
MAX_SYMBOLS = 1 << 20
MAX_LANES = 16


def _validate_shape(alphabet_bits, symbols):
    """Validate lane widths (1..9) and bounded symbol count per lane."""
    widths = tuple(alphabet_bits)
    if not 1 <= len(widths) <= MAX_LANES:
        raise ValueError("Expected 1..16 lane widths")
    if any(type(bits) is not int or not 1 <= bits <= 9 for bits in widths):
        raise ValueError("Expected integer symbol widths in 1..9")
    if type(symbols) is not int or not 1 <= symbols <= MAX_SYMBOLS // len(widths):
        raise ValueError("Expected a positive count, at most 1048576 symbols across all lanes")
    return widths


def decode(data, alphabet_bits, symbols):
    """Decode one stream with caller-established lane widths and symbols per lane.

    alphabet_bits contains actual symbol widths (1..9), not hardware indices.
    Return symbols, consumed byte count and lane state/fragment diagnostics.
    Zero padding may follow the stream; other trailing data is rejected.
    """
    widths = _validate_shape(alphabet_bits, symbols)
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_INPUT:
        raise ValueError("Expected 1..2097152 bytes")
    result = decode_lanes(data, [bits - 1 for bits in widths], [symbols] * len(widths))
    if any(data[result["consumed"]:]):
        raise BCPackError("Non-zero trailing data after byte %d" % result["consumed"])
    for lane, bits in zip(result["syms"], widths):
        if len(lane) != symbols or any(not 0 <= value < 1 << bits for value in lane):
            raise BCPackError("Decoded symbols do not match the requested lane shape")
    # The callback-only order trace is not part of the public result.
    result.pop("order")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--alphabet-bits", type=int, nargs="+", required=True,
                        help="Actual symbol widths, one per lane (1..9)")
    parser.add_argument("--symbols", type=int, required=True, help="Symbols per lane")
    args = parser.parse_args(argv)
    try:
        info = args.packet.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Expected a regular packet file: {args.packet}")
        _validate_shape(args.alphabet_bits, args.symbols)
        if not 1 <= info.st_size <= MAX_INPUT:
            raise ValueError(f"Expected 1..{MAX_INPUT} bytes")
        with args.packet.open("rb") as stream:
            data = stream.read(MAX_INPUT + 1)
        result = decode(data, args.alphabet_bits, args.symbols)
    except (OSError, ValueError, BCPackError) as error:
        parser.exit(1, f"BCPack symbols: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
