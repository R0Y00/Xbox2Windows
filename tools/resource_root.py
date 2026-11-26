"""Check the root instance of a relocated PRD0 v0x7a resource in one zlib member."""
import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import struct
from typing import BinaryIO
import zlib

from paged_resource import PagedResource, Segment


def inspect_root(data: bytes | bytearray) -> tuple[int, int]:
    header_size = 0x60
    if len(data) < header_size or data[:8] != bytes(8) or data[16:22] != b"0DRPz\0":
        raise ValueError("Expected a relocated PRD0 v0x7a resource header")

    def span(pointer: int, size: int) -> tuple[int, int]:
        start = pointer - 0x50000000
        if not (pointer < 0x58000000 and header_size <= start <= len(data) - size):
            raise ValueError("Resource extent outside the decoded member")
        return start, start + size

    block_map, = struct.unpack_from("<Q", data, 8)
    block_map_ptr = span(block_map, 1)
    index, = struct.unpack_from("<I", data, 32)
    schema_ptr, _, blocks_ptr = struct.unpack_from("<3Q", data, 40)
    schema_count, _, block_count = struct.unpack_from("<3H", data, 72)
    if not schema_count or not 1 <= index <= block_count:
        raise ValueError("Invalid schema count or one-based root index")
    schemas = span(schema_ptr, 32 * schema_count)
    blocks = span(blocks_ptr, 16 * block_count)
    kind, size, ptr = struct.unpack_from("<IIQ", data, blocks[0] + (index - 1) * 16)
    if not size:
        raise ValueError("Empty root instance")
    root = span(ptr, size)
    extents = sorted((block_map_ptr, schemas, blocks, root))
    if any(left[1] > right[0] for left, right in zip(extents, extents[1:])):
        raise ValueError("Overlapping schema, block directory, block map pointer or root instance")

    sizes = [struct.unpack_from("<I", data, pos + 24)[0] for pos in range(*schemas, 32)
             if struct.unpack_from("<I", data, pos)[0] == kind]
    if sizes != [size]:
        raise ValueError("Expected one schema matching the root type and byte size")
    return kind, size


def read_member(stream: BinaryIO, offset: int, stored: int, decoded: int) -> tuple[int, int]:
    budget = 2 << 20
    if max(stored, decoded) > budget:
        raise ValueError("Member exceeds 2 MiB stored or decoded budget")
    resource = PagedResource(stream, [Segment(offset, stored, decoded, "zlib")])
    return inspect_root(resource.read(0, decoded, limit=budget))


def integer(text: str) -> int:
    return int(text, 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--offset", type=integer, required=True, help="Byte offset of the zlib member")
    parser.add_argument("--stored", type=integer, required=True, help="Exact compressed size in bytes")
    parser.add_argument("--decoded", type=integer, required=True, help="Exact uncompressed size in bytes")
    args = parser.parse_args(argv)
    try:
        if args.offset < 0:
            raise ValueError(f"Expected non-negative --offset, got {args.offset}")
        if args.stored <= 0:
            raise ValueError(f"Expected positive --stored, got {args.stored}")
        if args.decoded <= 0:
            raise ValueError(f"Expected positive --decoded, got {args.decoded}")
        if args.archive.is_dir():
            raise ValueError(f"Archive path is a directory: {args.archive}")
        with args.archive.open("rb") as stream:
            kind, size = read_member(stream, args.offset, args.stored, args.decoded)
    except (OSError, ValueError, struct.error, zlib.error) as error:
        parser.exit(1, f"{args.archive}: {error}\n")
    print(json.dumps({"root_type_hash": f"0x{kind:08x}", "root_bytes": size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
