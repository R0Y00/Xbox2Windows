#!/usr/bin/env python3
"""Pure-Python unit, edge-case, and mutation suite for tools/paged_resource.py."""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
TOOL = TOOLS_DIR / "paged_resource.py"
sys.path.insert(0, str(TOOLS_DIR))
import paged_resource as pr  # noqa: E402


class _ShortReadStream:
    """Seekable stream returning at most max_chunk bytes per read() and optionally truncating EOF."""

    def __init__(self, data: bytes, *, max_chunk: int = 3, reported_size: int | None = None) -> None:
        self._data = data
        self._max_chunk = max_chunk
        self._reported_size = len(data) if reported_size is None else reported_size
        self._pos = 0

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_END:
            self._pos = self._reported_size + offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = offset
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= len(self._data):
            return b""
        if size < 0:
            take = len(self._data) - self._pos
        else:
            take = min(size, self._max_chunk, len(self._data) - self._pos)
        chunk = self._data[self._pos : self._pos + take]
        self._pos += len(chunk)
        return chunk


def _build_multi_segment_archive() -> tuple[bytes, list[pr.Segment], bytes]:
    raw0 = b"RAW_HEAD_0"  # 10 bytes
    dec1 = b"ZLIB_PAYLOAD_SEG_ONE" * 2  # 40 bytes
    cmp1 = zlib.compress(dec1)
    raw2 = b"MIDDLE_RAW_BYTES"  # 16 bytes
    dec3 = b"TAIL_ZLIB_DATA!" * 3  # 45 bytes
    cmp3 = zlib.compress(dec3)

    buf = bytearray(b"\xaa" * 4)
    off0 = len(buf)
    buf.extend(raw0)
    buf.extend(b"\xbb" * 6)  # physical alignment padding
    off1 = len(buf)
    buf.extend(cmp1)
    buf.extend(b"\xcc" * 5)
    off2 = len(buf)
    buf.extend(raw2)
    buf.extend(b"\xdd" * 3)
    off3 = len(buf)
    buf.extend(cmp3)
    buf.extend(b"\xee" * 4)

    segments = [
        pr.Segment(off0, len(raw0), len(raw0), "raw"),
        pr.Segment(off1, len(cmp1), len(dec1), "zlib"),
        pr.Segment(off2, len(raw2), len(raw2), "raw"),
        pr.Segment(off3, len(cmp3), len(dec3), "zlib"),
    ]
    logical = raw0 + dec1 + raw2 + dec3
    return bytes(buf), segments, logical


class PagedResourcePositiveTests(unittest.TestCase):
    def test_multi_segment_raw_and_zlib_spanning_boundaries(self) -> None:
        blob, segments, logical = _build_multi_segment_archive()
        resource = pr.PagedResource(io.BytesIO(blob), segments)

        self.assertEqual(resource.size, len(logical))
        self.assertEqual(resource.starts, (0, 10, 50, 66))
        self.assertEqual(resource.read(0, len(logical)), logical)

        # Cross-boundary spans across raw -> zlib -> raw -> zlib.
        for offset, length in (
            (0, 10),
            (5, 20),
            (10, 40),
            (45, 25),
            (0, 0),
            (resource.size, 0),
            (0, 0),
            (len(logical) - 12, 12),
        ):
            with self.subTest(offset=offset, length=length):
                self.assertEqual(resource.read(offset, length), logical[offset : offset + length])

        self.assertEqual(resource.read(0, 0, limit=0), b"")

        # Empty segment table is valid with size 0.
        empty = pr.PagedResource(io.BytesIO(b"unused"), [])
        self.assertEqual(empty.size, 0)
        self.assertEqual(empty.starts, ())
        self.assertEqual(empty.read(0, 0), b"")

    def test_short_read_reassembly(self) -> None:
        blob, segments, logical = _build_multi_segment_archive()
        stream = _ShortReadStream(blob, max_chunk=3)
        resource = pr.PagedResource(stream, segments)

        self.assertEqual(resource._read_at(segments[0].offset, segments[0].stored), logical[:10])
        self.assertEqual(resource.read(0, len(logical)), logical)
        self.assertEqual(resource.read(7, 65), logical[7:72])


class PagedResourceNegativeTests(unittest.TestCase):
    def test_segment_and_init_type_and_layout_rejections(self) -> None:
        stream = io.BytesIO(b"\x00" * 64)

        # Bool / non-int Segment fields rejected at construction and in PagedResource.__init__.
        for kwargs in (
            {"offset": True, "stored": 4, "size": 4, "codec": "raw"},
            {"offset": 0, "stored": False, "size": 4, "codec": "raw"},
            {"offset": 0, "stored": 4, "size": True, "codec": "raw"},
            {"offset": 1.5, "stored": 4, "size": 4, "codec": "raw"},
            {"offset": 0, "stored": "4", "size": 4, "codec": "raw"},
            {"offset": 0, "stored": 4, "size": None, "codec": "raw"},
        ):
            with self.subTest(bad_segment=kwargs):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    pr.Segment(**kwargs)
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    pr.PagedResource(stream, [SimpleNamespace(**kwargs)])

        # Overlapping, descending, negative, empty segments.
        bad_layout = (
            ("negative_offset", [pr.Segment(-1, 4, 4, "raw")]),
            ("empty_stored", [pr.Segment(0, 0, 4, "raw")]),
            ("empty_size", [pr.Segment(0, 4, 0, "raw")]),
            ("negative_stored", [pr.Segment(0, -4, 4, "raw")]),
            ("negative_size", [pr.Segment(0, 4, -4, "raw")]),
            ("overlapping", [pr.Segment(0, 10, 10, "raw"), pr.Segment(9, 4, 4, "raw")]),
            ("descending", [pr.Segment(16, 4, 4, "raw"), pr.Segment(8, 4, 4, "raw")]),
        )
        for label, segs in bad_layout:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, "Overlapping, descending or empty segment"):
                    pr.PagedResource(stream, segs)

        # Oversized and mismatched raw segments.
        bad_extents = (
            ("past_eof", [pr.Segment(60, 8, 8, "raw")]),
            ("raw_stored_gt_size", [pr.Segment(0, 8, 4, "raw")]),
            ("raw_stored_lt_size", [pr.Segment(0, 4, 8, "raw")]),
        )
        for label, segs in bad_extents:
            with self.subTest(case=label):
                with self.assertRaisesRegex(ValueError, "Segment extent disagrees"):
                    pr.PagedResource(stream, segs)

        # Non-str / empty codecs are rejected upfront in PagedResource.__init__.
        for bad_codec in (None, 123, b"raw", ""):
            with self.subTest(codec=bad_codec):
                seg = pr.Segment(0, 4, 4, bad_codec)  # type: ignore[arg-type]
                with self.assertRaisesRegex(ValueError, "Unsupported segment codec"):
                    pr.PagedResource(stream, [seg])

        # Unknown string codecs stay deferred to read() so later segments remain addressable.
        for unknown_codec in ("other", "lz4", "zstd"):
            with self.subTest(unknown_codec=unknown_codec):
                mixed = pr.PagedResource(
                    io.BytesIO(b"abcdef"),
                    [
                        pr.Segment(0, 2, 2, "raw"),
                        pr.Segment(2, 2, 2, unknown_codec),
                        pr.Segment(4, 2, 2, "raw"),
                    ],
                )
                self.assertEqual(mixed.read(0, 2), b"ab")
                self.assertEqual(mixed.read(4, 2), b"ef")
                with self.assertRaisesRegex(ValueError, f"Unsupported segment codec {unknown_codec}"):
                    mixed.read(1, 2)

    def test_zlib_error_and_limit_cases(self) -> None:
        payload = b"ABCD" * 8  # 32 bytes
        compressed = zlib.compress(payload)

        cases = (
            ("corrupt_zlib", b"\x78\x9c" + b"\xff" * 16, len(payload)),
            ("truncated_zlib", compressed[:-3], len(payload)),
            ("trailing_data", compressed + b"\x00", len(payload)),
            ("decoded_smaller_than_declared", compressed, len(payload) + 1),
            ("decoded_larger_than_declared", compressed, len(payload) - 1),
        )
        for label, data, declared_size in cases:
            with self.subTest(case=label):
                res = pr.PagedResource(io.BytesIO(data), [pr.Segment(0, len(data), declared_size, "zlib")])
                with self.assertRaisesRegex(ValueError, "Invalid compressed segment"):
                    res.read(0, 4)

        # Over-limit zlib (both decoded size > limit and stored size > limit).
        over_dec = pr.PagedResource(io.BytesIO(compressed), [pr.Segment(0, len(compressed), len(payload), "zlib")])
        with self.assertRaisesRegex(ValueError, "Compressed segment exceeds read budget"):
            over_dec.read(0, 4, limit=len(payload) - 1)

        incompressible = os.urandom(32) if False else bytes((i * 37) & 0xFF for i in range(32))
        big_cmp = zlib.compress(incompressible)
        self.assertGreater(len(big_cmp), len(incompressible))
        over_stored = pr.PagedResource(io.BytesIO(big_cmp), [pr.Segment(0, len(big_cmp), len(incompressible), "zlib")])
        with self.assertRaisesRegex(ValueError, "Compressed segment exceeds read budget"):
            over_stored.read(0, 4, limit=len(big_cmp) - 1)

        # Truncated physical read detected cleanly.
        trunc_raw = pr.PagedResource(_ShortReadStream(b"AB", reported_size=8), [pr.Segment(0, 8, 8, "raw")])
        with self.assertRaisesRegex(ValueError, "Truncated segment"):
            trunc_raw.read(0, 8)

        trunc_zlib = pr.PagedResource(
            _ShortReadStream(compressed[:4], reported_size=len(compressed)),
            [pr.Segment(0, len(compressed), len(payload), "zlib")],
        )
        with self.assertRaisesRegex(ValueError, "Invalid compressed segment"):
            trunc_zlib.read(0, 4)

    def test_read_bounds_and_type_rejections(self) -> None:
        res = pr.PagedResource(io.BytesIO(b"0123456789"), [pr.Segment(0, 10, 10, "raw")])

        for args, kwargs in (
            ((-1, 1), {}),
            ((0, -1), {}),
            ((0, 0), {"limit": -1}),
            ((0, 5), {"limit": 4}),
            ((8, 3), {}),
            ((11, 0), {}),
        ):
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "Logical range outside resource or read budget"):
                    res.read(*args, **kwargs)

        for args, kwargs in (
            ((True, 1), {}),
            ((0, False), {}),
            ((0, 1), {"limit": True}),
            ((0.0, 1), {}),
            ((0, "1"), {}),
            ((0, 1), {"limit": None}),
        ):
            with self.subTest(bad_types=(args, kwargs)):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    res.read(*args, **kwargs)

    def test_fast_temp_mutations(self) -> None:
        source = TOOL.read_text(encoding="utf-8")
        payload = b"MUTATION_TEST_DATA"
        cmp_plus_trail = zlib.compress(payload) + b"\x00"

        mutations = (
            (
                "allow_bool_segment_offset",
                "    if isinstance(value, bool) or not isinstance(value, int):\n",
                "    if not isinstance(value, int):\n",
                lambda mod: mod.Segment(True, 1, 1, "raw"),
            ),
            (
                "skip_upfront_codec_type_check",
                '            if not isinstance(segment.codec, str) or not segment.codec:\n'
                '                raise ValueError(f"Unsupported segment codec {segment.codec!r}")\n',
                "            pass\n",
                lambda mod: mod.PagedResource(io.BytesIO(b"abcd"), [mod.Segment(0, 4, 4, "")]),
            ),
            (
                "allow_negative_limit",
                "        if limit < 0 or offset < 0 or length < 0 or length > limit or offset + length > self.size:\n",
                "        if offset < 0 or length < 0 or (limit >= 0 and length > limit) or offset + length > self.size:\n",
                lambda mod: mod.PagedResource(io.BytesIO(b"abcd"), [mod.Segment(0, 4, 4, "raw")]).read(
                    0, 0, limit=-1
                ),
            ),
            (
                "ignore_trailing_zlib_data",
                "                if (len(data) != segment.stored or not decoder.eof or decoder.unused_data\n",
                "                if (len(data) != segment.stored or not decoder.eof\n",
                lambda mod: mod.PagedResource(
                    io.BytesIO(cmp_plus_trail),
                    [mod.Segment(0, len(cmp_plus_trail), len(payload), "zlib")],
                ).read(0, len(payload)),
            ),
            (
                "ignore_raw_size_mismatch",
                '            if physical_end > file_size or (segment.codec == "raw" and stored != size):\n',
                "            if physical_end > file_size:\n",
                lambda mod: mod.PagedResource(io.BytesIO(b"abcd"), [mod.Segment(0, 4, 2, "raw")]),
            ),
        )

        for label, target, replacement, trigger in mutations:
            with self.subTest(mutation=label):
                self.assertEqual(source.count(target), 1, target)
                mod = types.ModuleType(f"paged_resource_mut_{label}")
                mod.__file__ = str(TOOL)
                sys.modules[mod.__name__] = mod
                try:
                    exec(compile(source.replace(target, replacement, 1), str(TOOL), "exec"), mod.__dict__)
                    trigger(mod)
                finally:
                    sys.modules.pop(mod.__name__, None)


if __name__ == "__main__":
    unittest.main()
