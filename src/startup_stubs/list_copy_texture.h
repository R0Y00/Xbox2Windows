// Experimental (docs/startup-runtime.md, #870): the argument checks for command list slot 16,
// CopyTextureRegion(dst, x, y, z, src, box). Plain standard C++ so a Linux fixture can run it; the
// handler (list_copy_texture.inl) maps the title's resources and forwards only what passes here.
// A call the desktop runtime would reject is refused with a named reason instead.
#pragma once

#include <cstdint>
#include <initializer_list>

namespace x2w_list_copy_texture {

enum class Kind : std::uint32_t { subresource = 0, placed_footprint = 1 };

// What the check needs to know about one side of the copy. For a subresource side, width/height/depth are
// the subresource's size in blocks (texels for an uncompressed format). For a footprint side, the footprint
// size in blocks, the row pitch in bytes, the byte offset and the bytes per block are used, and
// resource_bytes is the size of the buffer.
struct Side {
    Kind kind;
    std::uint32_t width, height, depth;
    std::uint64_t offset;
    std::uint32_t row_pitch, bytes_per_block;
    std::uint64_t resource_bytes;
    bool multisampled;
    std::uint32_t block_w, block_h;  // texels per block (4 x 4 for block-compressed formats, else 1 x 1)
};

struct Box {
    std::uint32_t left, top, front, right, bottom, back;
};

enum class Verdict {
    ok,
    buffer_to_buffer,   // a footprint on both sides: the desktop wants CopyBufferRegion
    bad_box,            // an empty or inverted source box
    source_overflow,    // the source box leaves the source subresource
    dest_overflow,      // the destination origin plus box size leaves the destination subresource
    bad_pitch,          // a footprint row pitch below one row, or not a multiple of 256
    bad_offset,         // a footprint offset that is not a multiple of 512
    footprint_overflow, // the footprint reaches past the end of its buffer
    sample_mismatch,    // a multisampled side paired with a single-sample one
    bad_alignment       // a block-compressed copy that does not start on a block
};

inline const char* name(Verdict v) noexcept
{
    switch (v) {
    case Verdict::ok: return "ok";
    case Verdict::buffer_to_buffer: return "buffer_to_buffer";
    case Verdict::bad_box: return "bad_box";
    case Verdict::source_overflow: return "source_overflow";
    case Verdict::dest_overflow: return "dest_overflow";
    case Verdict::bad_pitch: return "bad_pitch";
    case Verdict::bad_offset: return "bad_offset";
    case Verdict::footprint_overflow: return "footprint_overflow";
    case Verdict::sample_mismatch: return "sample_mismatch";
    case Verdict::bad_alignment: return "bad_alignment";
    }
    return "unknown";
}

// Bytes a footprint reaches for a w x h x d copy that starts at texel (ox, oy, oz) inside it, from its offset
// (the last row is not padded; slices are as tall as the footprint, not the copy).
inline bool footprint_end(const Side& s, std::uint32_t w, std::uint32_t h, std::uint32_t d, std::uint32_t ox, std::uint32_t oy,
                          std::uint32_t oz, std::uint64_t& end) noexcept
{
    if (!w || !h || !d) { end = s.offset; return true; }
    const std::uint64_t row = std::uint64_t{(w + s.block_w - 1) / s.block_w} * s.bytes_per_block;
    const std::uint64_t rows = (h + s.block_h - 1) / s.block_h;
    const std::uint64_t start_x = std::uint64_t{ox / s.block_w} * s.bytes_per_block;
    const std::uint64_t start_row = oy / s.block_h;
    const std::uint64_t slice_rows = (std::uint64_t{s.height} + s.block_h - 1) / s.block_h;
    if (start_x + row > s.row_pitch) return false;
    const std::uint64_t slice = std::uint64_t{s.row_pitch} * (start_row + rows - 1) + start_x + row;
    const std::uint64_t total = std::uint64_t{s.row_pitch} * slice_rows * (std::uint64_t{oz} + d - 1) + slice;
    end = s.offset + total;
    return end >= s.offset;
}

// box == nullptr copies the whole source (for a footprint source, the whole footprint).
inline Verdict check(const Side& dst, std::uint32_t x, std::uint32_t y, std::uint32_t z, const Side& src, const Box* box) noexcept
{
    if (dst.kind == Kind::placed_footprint && src.kind == Kind::placed_footprint) return Verdict::buffer_to_buffer;
    if (dst.multisampled != src.multisampled) return Verdict::sample_mismatch;
    const Box whole{0, 0, 0, src.width, src.height, src.depth};
    const Box use = box ? *box : whole;
    if (use.left >= use.right || use.top >= use.bottom || use.front >= use.back) return Verdict::bad_box;
    if (use.right > src.width || use.bottom > src.height || use.back > src.depth) return Verdict::source_overflow;
    const Side& tex = dst.kind == Kind::subresource ? dst : src;  // the side that knows the block size
    if (x % tex.block_w || use.left % tex.block_w || y % tex.block_h || use.top % tex.block_h) return Verdict::bad_alignment;
    const std::uint32_t w = use.right - use.left, h = use.bottom - use.top, d = use.back - use.front;
    if (std::uint64_t{x} + w > dst.width || std::uint64_t{y} + h > dst.height || std::uint64_t{z} + d > dst.depth)
        return Verdict::dest_overflow;
    for (const Side* s : {&dst, &src}) {
        if (s->kind != Kind::placed_footprint) continue;
        if (!s->row_pitch || s->row_pitch % 256) return Verdict::bad_pitch;
        if (s->offset % 512) return Verdict::bad_offset;
        std::uint64_t end = 0;
        const bool is_dst = s == &dst;
        if (!footprint_end(*s, w, h, d, is_dst ? x : use.left, is_dst ? y : use.top, is_dst ? z : use.front, end))
            return Verdict::bad_pitch;
        if (end > s->resource_bytes) return Verdict::footprint_overflow;
    }
    return Verdict::ok;
}

}  // namespace x2w_list_copy_texture
