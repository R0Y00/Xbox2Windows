"""Read logical byte ranges from a seekable archive with an explicit segment map."""
from bisect import bisect_right
from dataclasses import dataclass
import os
import zlib


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class Segment:
    offset: int         # Physical byte offset in the open file
    stored: int         # Bytes excluding alignment padding
    size: int           # Decoded bytes in logical order
    codec: str

    def __post_init__(self) -> None:
        _require_int(self.offset, "segment.offset")
        _require_int(self.stored, "segment.stored")
        _require_int(self.size, "segment.size")


class PagedResource:
    """The caller owns the binary file and supplies segments in logical order."""

    def __init__(self, stream, segments):
        self.stream = stream
        self.segments = tuple(segments)
        file_size = stream.seek(0, os.SEEK_END)
        starts = []
        physical_end = logical_end = 0
        for segment in self.segments:
            offset = _require_int(segment.offset, "segment.offset")
            stored = _require_int(segment.stored, "segment.stored")
            size = _require_int(segment.size, "segment.size")
            if not isinstance(segment.codec, str) or not segment.codec:
                raise ValueError(f"Unsupported segment codec {segment.codec!r}")
            if offset < physical_end or min(stored, size) <= 0:
                raise ValueError("Overlapping, descending or empty segment")
            physical_end = offset + stored
            if physical_end > file_size or (segment.codec == "raw" and stored != size):
                raise ValueError("Segment extent disagrees with file or raw size")
            starts.append(logical_end)
            logical_end += size
        self.starts = tuple(starts)
        self.size = logical_end

    def _read_at(self, offset: int, size: int) -> bytes:
        self.stream.seek(offset)
        first = self.stream.read(size)
        if not first or len(first) == size:
            return first or b""
        data = bytearray(first)
        while len(data) < size:
            chunk = self.stream.read(size - len(data))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)

    def read(self, offset, length, *, limit=64 << 20):
        offset = _require_int(offset, "offset")
        length = _require_int(length, "length")
        limit = _require_int(limit, "limit")
        if limit < 0 or offset < 0 or length < 0 or length > limit or offset + length > self.size:
            raise ValueError("Logical range outside resource or read budget")
        index = bisect_right(self.starts, offset) - 1
        result = bytearray()
        while len(result) < length:
            segment = self.segments[index]
            inside = offset + len(result) - self.starts[index]
            take = min(length - len(result), segment.size - inside)
            if segment.codec == "raw":
                chunk = self._read_at(segment.offset + inside, take)
            elif segment.codec == "zlib":
                if max(segment.size, segment.stored) > limit:
                    raise ValueError("Compressed segment exceeds read budget")
                data = self._read_at(segment.offset, segment.stored)
                decoder = zlib.decompressobj()
                try:
                    decoded = decoder.decompress(data, segment.size + 1)
                except zlib.error as exc:
                    raise ValueError("Invalid compressed segment or decoded extent") from exc
                if (len(data) != segment.stored or not decoder.eof or decoder.unused_data
                        or len(decoded) != segment.size):
                    raise ValueError("Invalid compressed segment or decoded extent")
                chunk = decoded[inside:inside + take]
            else:
                raise ValueError(f"Unsupported segment codec {segment.codec}")
            if len(chunk) != take:
                raise ValueError("Truncated segment")
            result.extend(chunk)
            index += 1
        return bytes(result)
