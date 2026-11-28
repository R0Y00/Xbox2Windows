"""Verify GFX10 texture computer, micro-tile detiling, layout records and implicit-layout gating."""
import ctypes as ct
import struct
import sys

DESC = struct.Struct("<I4xQQIHHIIIII4x")
LAYOUT_BYTES = 0x18 + 4 * 0x6C8 + 8
ENTRY = struct.Struct("<IHHI")

PTR = ct.c_void_p
UINT = ct.c_uint32
UINT64 = ct.c_uint64
HRESULT = ct.c_int32

LAYOUT = ct.WINFUNCTYPE(HRESULT, PTR, PTR)
SIZE = ct.WINFUNCTYPE(UINT64, PTR)
ALIGN = ct.WINFUNCTYPE(UINT64, PTR)
OFFSET = ct.WINFUNCTYPE(UINT64, PTR, UINT, UINT)
ELEMENT = ct.WINFUNCTYPE(UINT64, PTR, UINT, UINT, UINT64, UINT, UINT, UINT, PTR)
COPY_INTO = ct.WINFUNCTYPE(HRESULT, PTR, PTR, UINT, UINT, PTR, UINT, UINT)
# Scarlett copy-out (slot 9) passes an extra option UINT between the subresource and the tiled
# base (DirectXTex XboxDetile); slot 8 keeps the shorter copy-in prototype.
COPY_FROM = ct.WINFUNCTYPE(HRESULT, PTR, PTR, UINT, UINT, UINT, PTR, UINT, UINT)
RELEASE = ct.WINFUNCTYPE(UINT64, PTR)
TILING = ct.WINFUNCTYPE(HRESULT, PTR, PTR, PTR, PTR, PTR, UINT, PTR)

MIP_HEADER_FMT = "<QQQIIIII"  # span, offset, slice_span, pitch_px, pitch_bytes, align, padw, padh
MIP_EXT_FMT = "<IIIIII"       # slices, cols, rows, mip_w, mip_h, slices2


def test_format(create, name, fmt, width, height, mips, slices, layout_code, bpe, block_dim):
    desc = ct.create_string_buffer(
        DESC.pack(3, 0, width, height, slices, mips, fmt, 1, 0, layout_code, 0)
    )
    computer = PTR()
    hr = create(desc, ct.byref(computer), 0, 0)
    assert hr == 0 and computer, f"Failed to create computer for {name}"
    slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]

    try:
        mode = layout_code & 0xFF
        is_linear = mode == 0
        expected_align = 65536 if mode in (9, 10) else (4096 if mode in (5, 6) else 256)

        # Slot 3 (size) & Slot 4 (alignment)
        total_size = SIZE(slots[3])(computer)
        alignment = ALIGN(slots[4])(computer)
        assert total_size > 0, f"{name}: size must be > 0"
        assert alignment == expected_align, f"{name}: align={alignment}, expected={expected_align}"

        # Slot 2 (layout)
        layout_buf = ct.create_string_buffer(LAYOUT_BYTES)
        assert LAYOUT(slots[2])(computer, layout_buf) == 0
        l_size, l_align, l_mips, l_planes = struct.unpack_from("<QQII", layout_buf, 0)
        assert l_size == total_size, f"{name}: layout size {l_size} != {total_size}"
        assert l_align == alignment, f"{name}: layout align {l_align} != {alignment}"
        assert l_mips == mips, f"{name}: layout mips {l_mips} != {mips}"
        assert l_planes == 1, f"{name}: layout planes {l_planes} != 1"

        plane_off = 0x18
        p_total, p_stride, p_align, p_bpe, p_std, p_packed = struct.unpack_from(
            "<QQ8xQIII", layout_buf, plane_off + 8
        )
        assert p_total == total_size, f"{name}: plane total {p_total} != {total_size}"
        assert p_stride * slices == total_size, f"{name}: slice stride {p_stride} * {slices} != {total_size}"
        assert p_align == alignment, f"{name}: plane align {p_align} != {alignment}"
        assert p_bpe == bpe, f"{name}: plane bpe {p_bpe} != {bpe}"
        assert p_std + p_packed == mips, f"{name}: plane std={p_std} + packed={p_packed} != {mips}"

        # Slot 10 (tiling)
        tiles = UINT()
        packed = ct.create_string_buffer(12)
        shape = (UINT * 3)()
        count = UINT(mips * slices)
        entries = ct.create_string_buffer(count.value * ENTRY.size)
        assert TILING(slots[10])(computer, ct.byref(tiles), packed, shape, ct.byref(count), 0, entries) == 0
        num_standard, num_packed, num_packed_tiles, _ = struct.unpack_from("<BB2xII", packed)
        assert num_standard == p_std and num_packed == p_packed, f"{name}: slot10 packed mismatch"
        assert num_standard + num_packed == mips, f"{name}: standard={num_standard} + packed={num_packed} != {mips}"
        assert count.value == num_standard * slices, f"{name}: slot10 entry count {count.value}"
        tile_w_px, tile_h_px, tile_d = shape
        assert tile_w_px > 0 and tile_h_px > 0 and tile_d == 1, f"{name}: invalid tile shape {tuple(shape)}"
        if is_linear:
            assert tiles.value * 256 == total_size, f"{name}: linear tiles {tiles.value} * 256 != {total_size}"
        else:
            assert tiles.value * alignment == total_size, f"{name}: tiles {tiles.value} * {alignment} != {total_size}"

        # Verify Slot 5 (offset) and Slot 2 per-mip records
        mip_offsets = []
        for m in range(mips):
            mip_off = plane_off + 0x38 + m * 0x70
            span, offset, slice_span, pitch_px, pitch_bytes, mip_align, padw, padh = struct.unpack_from(
                MIP_HEADER_FMT, layout_buf, mip_off
            )
            m_slices, cols, rows, mip_w, mip_h, _ = struct.unpack_from(MIP_EXT_FMT, layout_buf, mip_off + 0x2C)
            tail_flag = struct.unpack_from("<I", layout_buf, mip_off + 0x64)[0]

            expected_cols = (max(1, width >> m) + block_dim - 1) // block_dim
            expected_rows = (max(1, height >> m) + block_dim - 1) // block_dim
            assert cols == expected_cols and rows == expected_rows, f"{name} mip {m}: cols/rows {(cols, rows)}"
            assert mip_w == max(1, width >> m) and mip_h == max(1, height >> m), f"{name} mip {m}: ext"
            assert m_slices == slices, f"{name} mip {m}: slices {m_slices} != {slices}"
            assert padw >= cols and padh >= rows, f"{name} mip {m}: padw={padw}, padh={padh}"
            assert pitch_px == block_dim * padw, f"{name} mip {m}: pitch_px {pitch_px}"
            assert pitch_bytes == bpe * padw, f"{name} mip {m}: pitch_bytes {pitch_bytes}"
            assert span == bpe * padw * padh and slice_span == span, f"{name} mip {m}: span {span}"
            if is_linear:
                assert pitch_bytes % 256 == 0 and pitch_bytes >= cols * bpe, f"{name} mip {m}: linear pitch {pitch_bytes}"
                assert mip_align == 256 and tail_flag == 0
            else:
                expected_tail = 1 if m >= p_std else 0
                assert tail_flag == expected_tail, f"{name} mip {m}: tail_flag {tail_flag}"
                assert mip_align == (256 if tail_flag else alignment), f"{name} mip {m}: mip_align {mip_align}"

            # Slot 5: GetSubresourceOffset matches layout mip offset
            slot5_off = OFFSET(slots[5])(computer, 0, m)
            assert slot5_off == offset, f"{name} mip {m}: slot5 {slot5_off} != layout {offset}"
            assert offset < p_stride, f"{name} mip {m}: offset {offset} >= slice stride {p_stride}"
            mip_offsets.append(offset)

            if m < num_standard:
                tw, th, td, start_tile = ENTRY.unpack_from(entries, m * ENTRY.size)
                assert tw > 0 and th > 0 and td == 1, f"{name} mip {m}: bad tiling entry {(tw, th, td, start_tile)}"
                assert start_tile * (256 if is_linear else alignment) == offset, (
                    f"{name} mip {m}: start_tile={start_tile} != offset={offset}"
                )

        assert OFFSET(slots[5])(computer, 1, 0) == 0, f"{name}: out-of-range plane offset should be 0"
        assert OFFSET(slots[5])(computer, 0, mips) == 0, f"{name}: out-of-range mip offset should be 0"

        # Test subresource tile & detile roundtrip + slot 6 element address consistency
        tiled_buf = ct.create_string_buffer(total_size)
        for s in range(slices):
            for m in range(mips):
                mw = (max(1, width >> m) + block_dim - 1) // block_dim
                mh = (max(1, height >> m) + block_dim - 1) // block_dim
                row_pitch = mw * bpe + 16  # padded pitch

                linear_src = bytearray(mh * row_pitch)
                for y in range(mh):
                    for x in range(mw):
                        for b in range(bpe):
                            linear_src[y * row_pitch + x * bpe + b] = (
                                (m * 41 + s * 17 + x * 7 + y * 13 + b) & 0xFF
                            )

                src_buf = ct.create_string_buffer(bytes(linear_src), len(linear_src))
                subresource = s * mips + m
                # Slot 8: CopyIntoSubresource (linear -> tiled)
                rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, subresource, src_buf, row_pitch, len(linear_src))
                assert rc == 0, f"{name} mip {m} CopyIntoSubresource failed: 0x{rc & 0xFFFFFFFF:08x}"

                # Slot 9: CopyFromSubresource (tiled -> linear)
                dest_buf = ct.create_string_buffer(b"\xaa" * len(linear_src), len(linear_src))
                copy_from_slice_pitch = 0 if (m & 1) else len(linear_src)
                rc = COPY_FROM(slots[9])(computer, dest_buf, 0, subresource, 0, tiled_buf, row_pitch, copy_from_slice_pitch)
                assert rc == 0, f"{name} mip {m} CopyFromSubresource failed: 0x{rc & 0xFFFFFFFF:08x}"

                for y in range(mh):
                    row_orig = linear_src[y * row_pitch: y * row_pitch + mw * bpe]
                    row_detiled = dest_buf.raw[y * row_pitch: y * row_pitch + mw * bpe]
                    assert row_detiled == row_orig, f"{name} slice {s} mip {m} row {y} mismatch after detile!"

                # Slot 6: verify first_addr, bounds, and exact byte match with Slot 8 output
                bit = UINT(99)
                first_addr = ELEMENT(slots[6])(computer, 0, m, 0, 0, s, 0, ct.byref(bit))
                assert bit.value == 0
                expected_first_min = s * p_stride + mip_offsets[m]
                assert first_addr == expected_first_min or (m >= p_std and first_addr >= s * p_stride), (
                    f"{name} s={s} m={m}: first_addr={first_addr} vs expected={expected_first_min}"
                )
                assert first_addr + bpe <= total_size
                sample_coords = [(0, 0), (mw - 1, mh - 1)]
                if mw > 1:
                    sample_coords.append((1, 0))
                if mh > 1:
                    sample_coords.append((0, 1))
                sample_coords = list(dict.fromkeys(sample_coords))
                seen_addrs = set()
                for sx, sy in sample_coords:
                    addr = ELEMENT(slots[6])(computer, 0, m, sx, sy, s, 0, ct.byref(bit))
                    assert bit.value == 0
                    assert addr + bpe <= total_size, f"{name}: addr {addr} out of bounds"
                    assert addr not in seen_addrs, f"{name}: duplicate addr {addr} at ({sx},{sy})"
                    seen_addrs.add(addr)
                    expected_elem = bytes(linear_src[sy * row_pitch + sx * bpe: sy * row_pitch + (sx + 1) * bpe])
                    actual_elem = tiled_buf.raw[addr: addr + bpe]
                    assert actual_elem == expected_elem, (
                        f"{name} s={s} m={m} ({sx},{sy}): slot6 addr 0x{addr:x} mismatch with slot8"
                    )

    finally:
        RELEASE(slots[1])(computer)


def test_implicit_and_unsupported_routing(create):
    """Implicit layouts (UNKNOWN=0, ROW_MAJOR=1) and depth formats must stay on the desktop footprint path."""
    for name, fmt, w, h, layout_code in (
        ("RGBA8 64x64 UNKNOWN", 28, 64, 64, 0),
        ("R8 64x64 UNKNOWN", 61, 64, 64, 0),
        ("RG8 64x64 UNKNOWN", 49, 64, 64, 0),
        ("BC3 64x64 UNKNOWN", 77, 64, 64, 0),
        ("BC7 64x64 UNKNOWN", 98, 64, 64, 0),
        ("RGBA8 64x64 ROW_MAJOR", 28, 64, 64, 1),
        ("D24S8 64x64 0x109", 45, 64, 64, 0x109),
        ("D32F 64x64 0x109", 40, 64, 64, 0x109),
    ):
        desc = ct.create_string_buffer(DESC.pack(3, 0, w, h, 1, 1, fmt, 1, 0, layout_code, 0))
        computer = PTR()
        hr = create(desc, ct.byref(computer), 0, 0)
        assert hr == 0 and computer, f"{name}: XGCreateTextureComputer failed: 0x{hr & 0xFFFFFFFF:08x}"
        slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]
        try:
            # Desktop table leaves slots 3, 5, 6, 9 as no-op stubs returning 0 (unlike Gfx10 bc_table)
            assert SIZE(slots[3])(computer) == 0, f"{name}: unexpectedly routed to GFX10 slot 3"
            assert OFFSET(slots[5])(computer, 0, 0) == 0, f"{name}: unexpectedly routed to GFX10 slot 5"
            bit = UINT(99)
            assert ELEMENT(slots[6])(computer, 0, 0, 0, 0, 0, 0, ct.byref(bit)) == 0
            # Slot 9 (CopyFromSubresource) on Gfx10 returns E_INVALIDARG for null/zero pitch; desktop no-op returns 0
            assert COPY_FROM(slots[9])(computer, None, 0, 0, 0, None, 0, 0) == 0, (
                f"{name}: unexpectedly routed to GFX10 slot 9"
            )
            if fmt == 28 and layout_code == 0:
                layout_buf = ct.create_string_buffer(LAYOUT_BYTES)
                assert LAYOUT(slots[2])(computer, layout_buf) == 0
                l_size, l_align, l_mips, l_planes = struct.unpack_from("<QQII", layout_buf, 0)
                mip0_span, _, _, mip0_pitch = struct.unpack_from("<QQQ4xI", layout_buf, 0x18 + 0x38)
                assert l_size == 65536 and l_align == 65536 and l_mips == 1 and l_planes == 1
                assert mip0_span == 16384 and mip0_pitch == 256, (
                    f"{name}: expected desktop footprint span=16384 pitch=256, got {(mip0_span, mip0_pitch)}"
                )
        finally:
            RELEASE(slots[1])(computer)
    print("  Implicit layouts & depth formats: PASS (routed to desktop path, not GFX10 tiler)")


def test_negative_controls(create):
    desc = ct.create_string_buffer(
        DESC.pack(3, 0, 64, 64, 1, 1, 98, 1, 0, 0x105, 0)
    )
    computer = PTR()
    hr = create(desc, ct.byref(computer), 0, 0)
    assert hr == 0 and computer, "Failed to create computer for negative controls"
    slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]

    try:
        total_size = SIZE(slots[3])(computer)
        tiled_buf = ct.create_string_buffer(total_size)
        linear_src = ct.create_string_buffer(16 * 16 * 16)
        row_pitch = 16 * 16  # 256 bytes

        # 1. pitch < row_bytes
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 0, linear_src, row_pitch - 1, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, linear_src, 0, 0, 0, tiled_buf, row_pitch - 1, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 2. Out-of-range plane
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 1, 0, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, linear_src, 1, 0, 0, tiled_buf, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 3. Out-of-range subresource
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 1, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, linear_src, 0, 1, 0, tiled_buf, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 4. Overlapping source and destination buffers
        overlap_buf = ct.create_string_buffer(total_size)
        rc = COPY_INTO(slots[8])(computer, overlap_buf, 0, 0, overlap_buf, row_pitch, total_size)
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, overlap_buf, 0, 0, 0, overlap_buf, row_pitch, total_size)
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 5. Null pointers
        rc = COPY_INTO(slots[8])(computer, None, 0, 0, linear_src, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 0, None, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, None, 0, 0, 0, tiled_buf, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, linear_src, 0, 0, 0, None, row_pitch, len(linear_src))
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 6. Non-zero slice_pitch < linear_bytes
        rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, 0, linear_src, row_pitch, len(linear_src) - 1)
        assert (rc & 0xFFFFFFFF) == 0x80070057
        rc = COPY_FROM(slots[9])(computer, linear_src, 0, 0, 0, tiled_buf, row_pitch, len(linear_src) - 1)
        assert (rc & 0xFFFFFFFF) == 0x80070057

        # 7. Non-zero scarlett_option in CopyFromSubresource (slot 9) -> E_NOTIMPL without touching dest
        sentinel = b"\xad" * len(linear_src)
        refused_buf = ct.create_string_buffer(sentinel, len(sentinel))
        for opt in (1, 0xFFFFFFFF):
            rc = COPY_FROM(slots[9])(computer, refused_buf, 0, 0, opt, tiled_buf, row_pitch, 0)
            assert (rc & 0xFFFFFFFF) == 0x80004001, f"opt={opt}: expected E_NOTIMPL, got 0x{rc & 0xFFFFFFFF:08x}"
            assert refused_buf.raw == sentinel, "unsupported copy-out option changed destination"
        rc = COPY_FROM(slots[9])(computer, None, 0, 1, 1, None, 0, 0)
        assert (rc & 0xFFFFFFFF) == 0x80004001, "non-zero option must fail with E_NOTIMPL before arg checks"

    finally:
        RELEASE(slots[1])(computer)

    # Also verify non-zero scarlett_option refusal on an uncompressed GFX10 surface (RGBA8 64x64 4KB_S)
    desc_rgba = ct.create_string_buffer(DESC.pack(3, 0, 64, 64, 1, 1, 28, 1, 0, 0x105, 0))
    comp_rgba = PTR()
    hr = create(desc_rgba, ct.byref(comp_rgba), 0, 0)
    assert hr == 0 and comp_rgba, "Failed to create RGBA8 computer for slot 9 option check"
    s_rgba = ct.cast(comp_rgba, ct.POINTER(ct.POINTER(PTR)))[0]
    try:
        rgba_size = SIZE(s_rgba[3])(comp_rgba)
        rgba_tiled = ct.create_string_buffer(rgba_size)
        rgba_sentinel = b"\xad" * (64 * 64 * 4)
        rgba_refused = ct.create_string_buffer(rgba_sentinel, len(rgba_sentinel))
        rc = COPY_FROM(s_rgba[9])(comp_rgba, rgba_refused, 0, 0, 1, rgba_tiled, 64 * 4, 0)
        assert (rc & 0xFFFFFFFF) == 0x80004001
        assert rgba_refused.raw == rgba_sentinel, "RGBA8 unsupported copy-out option changed destination"
    finally:
        RELEASE(s_rgba[1])(comp_rgba)

    # 8. Descriptor with MipLevels > 16 rejected from GFX10 tiling (falls back to desktop path)
    desc_over_mips = ct.create_string_buffer(
        DESC.pack(3, 0, 64, 64, 1, 17, 98, 1, 0, 0x109, 0)
    )
    comp_over = PTR()
    hr = create(desc_over_mips, ct.byref(comp_over), 0, 0)
    if hr == 0 and comp_over:
        s = ct.cast(comp_over, ct.POINTER(ct.POINTER(PTR)))[0]
        try:
            assert SIZE(s[3])(comp_over) == 0, "MipLevels=17 unexpectedly routed to GFX10 slot 3"
            assert COPY_FROM(s[9])(comp_over, None, 0, 0, 0, None, 0, 0) == 0, (
                "MipLevels=17 unexpectedly routed to GFX10 slot 9"
            )
            layout_buf = ct.create_string_buffer(LAYOUT_BYTES)
            assert LAYOUT(s[2])(comp_over, layout_buf) == 0
            l_mips = struct.unpack_from("<I", layout_buf, 0x10)[0]
            assert l_mips == 15, f"MipLevels=17 expected desktop clamp to 15, got {l_mips}"
        finally:
            RELEASE(s[1])(comp_over)
    print("  Negative controls: PASS (pitch < row_bytes, plane/subresource out of range, overlap, null, slice_pitch, scarlett_option, mips > 16)")


def test_pinned_addrlib_offsets(create):
    """Pin expected element byte offsets per element size (1, 2, 4, 8, 16) from Mesa 25.1.0 addrlib."""
    # Offsets collected from xg_gfx10_tiling_reference (AMD addrlib, navi21) so any swizzle bit
    # permutation (such as swapping PATTERN_S/PATTERN_D bits or mip-tail origins) fails even when
    # addrlib is not configured in the build.
    pinned_cases = [
        # bpe=1 (_S and _D)
        ("R8 256x256 4-mip 64KB_S", 61, 256, 256, 4, 0x109, 1, 1, {
            (0, 0, 0): 0x10000, (0, 1, 0): 0x10001, (0, 2, 0): 0x10002, (0, 4, 0): 0x10004,
            (0, 8, 0): 0x10008, (0, 16, 0): 0x10200,
            (0, 0, 1): 0x10010, (0, 0, 2): 0x10020, (0, 0, 4): 0x10040, (0, 0, 8): 0x10080,
            (0, 0, 16): 0x10100,
        }),
        ("R8 64x64 1-mip 4KB_D", 61, 64, 64, 1, 0x106, 1, 1, {
            (0, 0, 0): 0x0, (0, 1, 0): 0x1, (0, 2, 0): 0x2, (0, 4, 0): 0x4, (0, 8, 0): 0x40,
            (0, 0, 1): 0x10, (0, 0, 2): 0x8, (0, 0, 4): 0x20, (0, 0, 8): 0x80,
        }),
        # bpe=2 (_S and _D)
        ("R16 64x64 3-mip 4KB_S", 56, 64, 64, 3, 0x105, 2, 1, {
            (0, 0, 0): 0x1000, (0, 1, 0): 0x1002, (0, 2, 0): 0x1004, (0, 4, 0): 0x1008,
            (0, 8, 0): 0x1080,
            (0, 0, 1): 0x1010, (0, 0, 2): 0x1020, (0, 0, 4): 0x1040, (0, 0, 8): 0x1100,
        }),
        ("RG8 64x64 3-mip 4KB_D", 49, 64, 64, 3, 0x106, 2, 1, {
            (0, 0, 0): 0x1000, (0, 1, 0): 0x1002, (0, 2, 0): 0x1004, (0, 4, 0): 0x1008,
            (0, 8, 0): 0x1080,
            (0, 0, 1): 0x1010, (0, 0, 2): 0x1020, (0, 0, 4): 0x1040,
        }),
        # bpe=4 (_S and _D)
        ("BGRA8 64x64 3-mip 4KB_S", 87, 64, 64, 3, 0x105, 4, 1, {
            (0, 0, 0): 0x2000, (0, 1, 0): 0x2004, (0, 2, 0): 0x2008, (0, 4, 0): 0x2080,
            (0, 8, 0): 0x2200,
            (0, 0, 1): 0x2010, (0, 0, 2): 0x2020, (0, 0, 4): 0x2040, (0, 0, 8): 0x2100,
        }),
        ("RGBA8 128x128 4-mip 64KB_S", 28, 128, 128, 4, 0x109, 4, 1, {
            (0, 0, 0): 0x10000, (0, 1, 0): 0x10004, (0, 2, 0): 0x10008, (0, 4, 0): 0x10080,
            (0, 8, 0): 0x10200,
            (0, 0, 1): 0x10010, (0, 0, 2): 0x10020, (0, 0, 4): 0x10040, (0, 0, 8): 0x10100,
        }),
        ("R32 64x64 3-mip 256B_D", 41, 64, 64, 3, 0x102, 4, 1, {
            (0, 0, 0): 0x1400, (0, 1, 0): 0x1404, (0, 2, 0): 0x1408, (0, 4, 0): 0x1440,
            (0, 0, 1): 0x1410, (0, 0, 2): 0x1420, (0, 0, 4): 0x1480,
        }),
        # bpe=8 (_S and _D)
        ("BC1 256x256 5-mip 64KB_S", 71, 256, 256, 5, 0x109, 8, 4, {
            (0, 0, 0): 0x8000, (0, 1, 0): 0x8008, (0, 2, 0): 0x8040, (0, 4, 0): 0x8080,
            (0, 8, 0): 0x8200,
            (0, 0, 1): 0x8010, (0, 0, 2): 0x8020, (0, 0, 4): 0x8100,
        }),
        ("RGBA16 64x64 3-mip 64KB_D", 10, 64, 64, 3, 0x10A, 8, 1, {
            (0, 0, 0): 0x8000, (0, 1, 0): 0x8008, (0, 2, 0): 0x8020, (0, 4, 0): 0x8040,
            (0, 0, 1): 0x8010, (0, 0, 2): 0x8080, (0, 0, 4): 0x8100,
        }),
        ("RGBA16 100x10 3-mip LINEAR", 10, 100, 10, 3, 0x100, 8, 1, {
            (0, 0, 0): 3328, (0, 3, 2): 3328 + 2 * 1024 + 3 * 8,
            (1, 0, 0): 768, (1, 3, 2): 768 + 2 * 512 + 3 * 8,
            (2, 0, 0): 0, (2, 3, 1): 0 + 1 * 256 + 3 * 8,
        }),
        # bpe=16 (_S, _D, and mip tail)
        ("RGBA32 32x32 3-mip 4KB_S", 2, 32, 32, 3, 0x105, 16, 1, {
            (0, 0, 0): 0x2000, (0, 1, 0): 0x2040, (0, 2, 0): 0x2080, (0, 4, 0): 0x2200,
            (0, 0, 1): 0x2010, (0, 0, 2): 0x2020, (0, 0, 4): 0x2100,
        }),
        ("BC5 256x256 4-mip 64KB_D", 83, 256, 256, 4, 0x10A, 16, 4, {
            (0, 0, 0): 0x10000, (0, 1, 0): 0x10010, (0, 2, 0): 0x10040, (0, 4, 0): 0x10200,
            (0, 0, 1): 0x10020, (0, 0, 2): 0x10080, (0, 0, 4): 0x10100,
        }),
        ("BC7 64x64 7-mip 4KB_S", 98, 64, 64, 7, 0x105, 16, 4, {
            (0, 0, 0): 0x1000, (0, 1, 0): 0x1040, (0, 2, 0): 0x1080,
            (0, 0, 1): 0x1010, (0, 0, 2): 0x1020,
            (2, 0, 0): 0x600, (2, 1, 0): 0x640, (2, 0, 1): 0x610,
            (3, 0, 0): 0x500, (3, 1, 0): 0x540, (3, 0, 1): 0x510,
            (4, 0, 0): 0x400, (5, 0, 0): 0x300, (6, 0, 0): 0x200,
        }),
    ]

    for name, fmt, w, h, mips, layout, bpe, bdim, expected_map in pinned_cases:
        desc = ct.create_string_buffer(DESC.pack(3, 0, w, h, 1, mips, fmt, 1, 0, layout, 0))
        computer = PTR()
        hr = create(desc, ct.byref(computer), 0, 0)
        assert hr == 0 and computer, f"{name}: create failed"
        slots = ct.cast(computer, ct.POINTER(ct.POINTER(PTR)))[0]
        try:
            total_size = SIZE(slots[3])(computer)
            tiled_buf = ct.create_string_buffer(total_size)
            bit = UINT(0)
            by_mip = {}
            for (mip, x, y), exp_addr in expected_map.items():
                got_addr = ELEMENT(slots[6])(computer, 0, mip, x, y, 0, 0, ct.byref(bit))
                assert bit.value == 0 and got_addr == exp_addr, (
                    f"{name} mip {mip} ({x},{y}): slot6 addr=0x{got_addr:x}, expected 0x{exp_addr:x}"
                )
                by_mip.setdefault(mip, []).append((x, y, exp_addr))

            for mip, coords in by_mip.items():
                mw = (max(1, w >> mip) + bdim - 1) // bdim
                mh = (max(1, h >> mip) + bdim - 1) // bdim
                row_pitch = mw * bpe
                linear_src = bytearray(mh * row_pitch)
                for y in range(mh):
                    for x in range(mw):
                        for b in range(bpe):
                            linear_src[y * row_pitch + x * bpe + b] = ((mip * 31 + x * 13 + y * 29 + b) & 0xFF)
                src_buf = ct.create_string_buffer(bytes(linear_src), len(linear_src))
                rc = COPY_INTO(slots[8])(computer, tiled_buf, 0, mip, src_buf, row_pitch, len(linear_src))
                assert rc == 0, f"{name} mip {mip}: CopyIntoSubresource failed"
                for x, y, exp_addr in coords:
                    expected_elem = bytes(linear_src[y * row_pitch + x * bpe: y * row_pitch + (x + 1) * bpe])
                    actual_elem = tiled_buf.raw[exp_addr: exp_addr + bpe]
                    assert actual_elem == expected_elem, (
                        f"{name} mip {mip} ({x},{y}): slot8 wrote wrong bytes at pinned offset 0x{exp_addr:x}"
                    )
        finally:
            RELEASE(slots[1])(computer)
    print(f"  Pinned addrlib offsets: PASS ({len(pinned_cases)} surfaces across bpe 1/2/4/8/16)")


def main(dll_path):
    create = ct.WinDLL(dll_path).XGCreateTextureComputer
    create.argtypes = (PTR, ct.POINTER(PTR), ct.c_uint64, ct.c_uint64)
    create.restype = HRESULT

    cases = [
        ("BC1 256x256 5-mip 64KB_S", 71, 256, 256, 5, 1, 0x109, 8, 4),
        ("BC1 128x128 4-mip 4KB_S", 71, 128, 128, 4, 1, 0x105, 8, 4),
        ("BC2 128x128 3-mip 4KB_S", 74, 128, 128, 3, 1, 0x105, 16, 4),
        ("BC3 128x128 3-mip 4KB_S", 77, 128, 128, 3, 1, 0x105, 16, 4),
        ("BC4 512x256 6-mip 64KB_S", 80, 512, 256, 6, 1, 0x109, 8, 4),
        ("BC5 256x256 4-mip 64KB_D", 83, 256, 256, 4, 1, 0x10A, 16, 4),
        ("BC6H 256x256 4-mip 64KB_S", 95, 256, 256, 4, 1, 0x109, 16, 4),
        ("BC7 64x64 7-mip 4KB_S", 98, 64, 64, 7, 1, 0x105, 16, 4),
        ("BC7 81x69 1-mip 4KB_D", 98, 81, 69, 1, 1, 0x106, 16, 4),
        ("BC7 128x128 3-mip 2-slice 64KB_S", 98, 128, 128, 3, 2, 0x109, 16, 4),
        ("BC7 128x64 8-mip 4KB_S", 98, 128, 64, 8, 1, 0x105, 16, 4),
        ("R8 256x256 4-mip 64KB_S", 61, 256, 256, 4, 1, 0x109, 1, 1),
        ("R8 64x64 1-mip 256B_S", 61, 64, 64, 1, 1, 0x101, 1, 1),
        ("R8 64x64 1-mip 4KB_D", 61, 64, 64, 1, 1, 0x106, 1, 1),
        ("RG8 64x64 3-mip 4KB_D", 49, 64, 64, 3, 1, 0x106, 2, 1),
        ("R16 64x64 3-mip 4KB_S", 56, 64, 64, 3, 1, 0x105, 2, 1),
        ("R32 64x64 3-mip 256B_D", 41, 64, 64, 3, 1, 0x102, 4, 1),
        ("RGBA8 128x128 4-mip 64KB_S", 28, 128, 128, 4, 1, 0x109, 4, 1),
        ("BGRA8 64x64 3-mip 4KB_S", 87, 64, 64, 3, 1, 0x105, 4, 1),
        ("RGBA16 64x64 3-mip 64KB_D", 10, 64, 64, 3, 1, 0x10A, 8, 1),
        ("RGBA32 32x32 3-mip 4KB_S", 2, 32, 32, 3, 1, 0x105, 16, 1),
        ("RGBA8 64x64 1-mip LINEAR", 28, 64, 64, 1, 1, 0x100, 4, 1),
        ("RGBA8 256x256 3-mip LINEAR", 28, 256, 256, 3, 1, 0x100, 4, 1),
        ("RGBA16 100x10 3-mip 2-slice LINEAR", 10, 100, 10, 3, 2, 0x100, 8, 1),
    ]

    for name, fmt, w, h, mips, slices, layout, bpe, bdim in cases:
        test_format(create, name, fmt, w, h, mips, slices, layout, bpe, bdim)
        print(f"  {name}: PASS (layout, offset, element, tiling, slot 8 tile, slot 9 detile exact)")

    test_pinned_addrlib_offsets(create)
    test_implicit_and_unsupported_routing(create)
    test_negative_controls(create)

    print(f"All {len(cases)} GFX10 texture computer test cases, implicit routing checks and negative controls passed!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: startup_xg_gfx10_check.py <path_to_xg_xs.dll>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
