#pragma once
#include <cstdint>

namespace x2w_startup {

constexpr std::uint32_t heap_raw_token_magic = 0x56573258;  // "X2WV" (d3d12_capture.inl view_token_magic)
// Translated draw heap layout (d3d12_draw.inl ensure_draw_heap): ring [0, 2^18), 8 empty views, 8 views per view id
// for 90000 ids, then the raw windows.
constexpr std::uint32_t heap_raw_window_first = (1u << 18) + 8 + 8 * 90000;  // 982152
constexpr std::uint32_t heap_raw_windows_max = 256;
// The entry's window field (heap_raw_prelude.hlsli hbuf): 0..255 a window, base 4-aligned; heap_raw_window_unaligned |
// window, a base that is not (the program reads each dword from two); heap_raw_window_paged, a range whose 64 KiB pages
// are in different windows or at scattered offsets of one: the offset field is its first page record (8 bytes each,
// window index and the window offset of the page's start) in the view at heap_raw_pages_view. A program from the
// earlier prelude (window < 256 only) reads both as a miss, as before.
constexpr std::uint32_t heap_raw_window_unaligned = 0x100, heap_raw_window_paged = 0x200;
constexpr std::uint32_t heap_raw_pages_view = heap_raw_window_first + heap_raw_windows_max;  // 982408
constexpr std::uint32_t heap_raw_page_records = 1u << 20;
                                                           // split ranges of 57-182 pages by frame 8573, still growing)
constexpr std::uint32_t heap_raw_paged_max_pages = 4096;   // a paged range spans at most 256 MiB
constexpr std::uint32_t heap_raw_map_entries = 1u << 19;
constexpr std::uint32_t heap_raw_probes = 16;  // the same in heap_raw_prelude.hlsli vraw
constexpr std::uint32_t heap_raw_resolve_budget = 8192;  // heap-sweep finds resolved per drain (the rest wait for the next)
constexpr std::uint32_t heap_raw_retry_frames = 120;     // an unresolved range is tried again after this many frames
constexpr std::uint32_t heap_raw_sweep_frames = 16;      // the heap sweep reads the whole heap in this many frames
constexpr std::uint32_t heap_raw_revalidate_passes = 4;  // a resolved entry is resolved again after this many sweep passes
constexpr std::uint64_t heap_raw_window_view_bytes = 4ull << 27;  // a raw buffer SRV names at most 2^27 dwords

struct RawBufferWords {
    std::uint64_t base = 0;
    std::uint32_t stride = 0, num_records = 0, bytes = 0, format = 0, oob = 0;
};
// True for a buffer V# the table can bound: not a view token, TYPE 0, base != 0, no SWIZZLE_ENABLE, no ADD_TID_ENABLE,
// OOB_SELECT 0, 1 or 3, and a nonzero size by the rule above. bytes saturates at 0xffffffff (no table entry is that
// large: it must fit in one window).
inline bool raw_buffer_words(const std::uint32_t w[4], RawBufferWords& out)
{
    if (w[0] == heap_raw_token_magic || (w[3] >> 30) != 0 || (w[1] >> 31) || ((w[3] >> 23) & 1)) return false;
    out.base = std::uint64_t{w[0]} | (std::uint64_t{w[1] & 0xffff} << 32);
    if (!out.base) return false;
    out.stride = (w[1] >> 16) & 0x3fff;
    out.num_records = w[2];
    out.format = (w[3] >> 12) & 127;
    out.oob = (w[3] >> 28) & 3;
    if (out.oob == 2) return false;
    const std::uint64_t bytes = out.oob == 3 ? std::uint64_t{w[2]} : std::uint64_t{out.stride} * w[2];
    if (!bytes) return false;
    out.bytes = bytes > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(bytes);
    return true;
}
// The shape of the V#s the runtime writes (gfx10_buffer_words): words 4-7 zero, word 3 only DST_SEL, FORMAT and
// OOB_SELECT (TYPE 0). The heap reads (d3d12_heap_raw.inl heap_raw_sweep_range) take only slots of this shape, so arena
// bytes and other data in the heap range rarely look like one; one that does is a read-only range of title memory.
inline bool runtime_vdesc_shape(const std::uint32_t w[8])
{
    return !(w[4] | w[5] | w[6] | w[7]) && !(w[3] & 0xcff80000u);
}
inline std::uint32_t raw_map_hash(std::uint32_t base_lo, std::uint32_t base_hi, std::uint32_t bytes)
{
    std::uint32_t x = (base_lo * 0x9e3779b1u) ^ (base_hi * 0x85ebca6bu) ^ (bytes * 0xc2b2ae35u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x;
}

}  // namespace x2w_startup
