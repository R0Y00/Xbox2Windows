// LOCAL (compute translation runtime, manifest `heapraw 1`): the heap prelude of a translated compute program whose
// heap SRV buffers resolve raw GFX10 V# words as well as view tokens. Replaces the `heap 1` lines heapmem / vinfo /
// vid / vix / vstride / vfmt; a heap SRV buffer H<k> is `#define H<k> hbuf(h<k>)` instead of hres<k>. The runtime
// side, the table layout and the hash are in gfx10_raw_vdesc.h (raw_map_hash == vrawh) and d3d12_heap_raw.inl.
// The constants: 0x56573258 "X2WV" token, 90000 view ids, 262144 empty views, 262152 first view id view,
// 982152 first raw window view (heap_raw_window_first), 256 windows, 982408 page-record view (heap_raw_pages_view),
// 16 probes (linear; with 8, about 1% of inserts found every probe occupied at 45% load in rawheap-bench).
// The entry's window field (gfx10_raw_vdesc.h): 0..255 a window; 0x100 | window, a base that is not 4-aligned; 0x200
// paged (each dword through its 64 KiB page's record); 0xfffe / 0xffff misses. The earlier prelude took < 256 only.
ByteAddressBuffer heapmem : register(t0, space1);
ByteAddressBuffer vinfo : register(t1, space1);
ByteAddressBuffer rawmap : register(t2, space1);
uint vid(uint h) { return heapmem.Load(h) == 0x56573258u ? heapmem.Load(h + 4) : 0xffffffffu; }
uint vix(uint h, uint k) { uint i = vid(h); return i < 90000u ? 262152u + i * 8u + k : 262144u + k; }
// A buffer V# the table can bound and its bytes (gfx10_raw_vdesc.h raw_buffer_words, the same rule): not a token,
// TYPE 0, base != 0, no SWIZZLE_ENABLE (word 1 bit 31), no ADD_TID_ENABLE (word 3 bit 23), OOB_SELECT 0, 1 or 3;
// bytes NUM_RECORDS for OOB_SELECT 3, else STRIDE x NUM_RECORDS (saturated at 0xffffffff), and not 0.
uint vrawbytes(uint4 w)
{
    uint s = (w.y >> 16) & 0x3fffu;
    if (((w.w >> 28) & 3u) == 3u) return w.z;
    return s == 0u ? 0u : (w.z > 0xffffffffu / s ? 0xffffffffu : s * w.z);
}
bool vrawok(uint4 w)
{
    return w.x != 0x56573258u && (w.w >> 30) == 0u && (w.y >> 31) == 0u && ((w.w >> 23) & 1u) == 0u && ((w.w >> 28) & 3u) != 2u &&
           (w.x | (w.y & 0xffffu)) != 0u && vrawbytes(w) != 0u;
}
uint vrawh(uint lo, uint hi, uint b) { uint x = (lo * 0x9e3779b1u) ^ (hi * 0x85ebca6bu) ^ (b * 0xc2b2ae35u); x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; return x; }
// The window index of the V# in slot h (0xffffffff: no table entry), its byte offset in that window and its bytes.
uint vraw(uint h, out uint o, out uint n)
{
    o = 0u;
    n = 0u;
    uint4 w = heapmem.Load4(h);
    if (!vrawok(w)) return 0xffffffffu;
    uint b = vrawbytes(w), hi = w.y & 0xffffu, cap;
    rawmap.GetDimensions(cap);
    cap /= 16u;
    if (cap == 0u || (cap & (cap - 1u)) != 0u) return 0xffffffffu;  // no table bound (switch off): a null view
    uint i = vrawh(w.x, hi, b);
    [loop] for (uint p = 0u; p < 16u; ++p) {
        uint4 e = rawmap.Load4(((i + p) & (cap - 1u)) * 16u);
        if (e.z == 0u) break;
        if (e.x == w.x && (e.y & 0xffffu) == hi && e.z == b) { o = e.w; n = b; return e.y >> 16; }
    }
    return 0xffffffffu;
}
// The stride and format of a V# slot only when the table resolves it (vraw hits): a miss binds the empty view, and
// its stride and format must be those of the empty view (0), as before `heapraw` and with X2W_HEAP_RAW=0, so a typed
// load of a miss reads raw 0 and not a decoded default (W = 1.0).
bool vrawhit(uint h) { uint o, n; return vid(h) == 0xffffffffu && vraw(h, o, n) <= 0x200u; }
uint vstride(uint h)
{
    uint i = vid(h);
    if (i < 90000u) return vinfo.Load(i * 16u);
    return vrawhit(h) ? (heapmem.Load(h + 4) >> 16) & 0x3fffu : 0u;
}
// FORMAT of a resolved V# as vinfo word 1 holds it for a view: 0 for raw (OOB_SELECT 3) and 32_FLOAT (22) V#s, which
// the runtime's own V#s of raw and structured views carry (vinfo: 0 for both).
uint vfmt(uint h)
{
    uint i = vid(h), n;
    vinfo.GetDimensions(n);
    if (i < 90000u) return i * 16u + 8u <= n ? vinfo.Load(i * 16u + 4u) : 0u;
    if (!vrawhit(h)) return 0u;
    uint w3 = heapmem.Load(h + 12), f = (w3 >> 12) & 127u;
    return ((w3 >> 28) & 3u) != 3u && f != 22u ? f : 0u;
}
// The dword at byte x (4-aligned) of a paged range, x counted from the start of its first page; `first` its first
// page record (window index, window offset of the page's start).
uint hpage(uint first, uint x)
{
    ByteAddressBuffer t = ResourceDescriptorHeap[982408u];
    uint2 r = t.Load2((first + (x >> 16)) * 8u);
    ByteAddressBuffer b = ResourceDescriptorHeap[NonUniformResourceIndex(982152u + r.x)];
    return b.Load(r.y + (x & 0xffffu));
}
// A heap SRV buffer: a view token's raw view (bounded by D3D12), or a resolved V#'s window with its offset and its
// size (a dword at byte a is read when a + 4 <= n, else 0: the ISA's offset + payload > NUM_RECORDS), or a paged
// range (q: its base's offset in its first page + 1, o its first page record), or the empty view. A base that is not
// 4-aligned is read byte-exact, from the two dwords around each one (both inside the range).
struct HBuf {
    uint d;
    uint o;
    uint n;
    uint q;
    uint Word(uint x)
    {
        if (q != 0u) return hpage(o, x);
        ByteAddressBuffer b = ResourceDescriptorHeap[d];
        return b.Load(x);
    }
    uint Load(uint a)
    {
        if (!(a <= n - 4u && n >= 4u)) return 0u;
        uint x = (q != 0u ? q - 1u : o) + a, k = (x & 3u) * 8u, lo = Word(x & ~3u);
        if (k == 0u) return lo;  // not ?: (both sides evaluated before HLSL 2021: a second load on every aligned read)
        return (lo >> k) | (Word((x & ~3u) + 4u) << (32u - k));
    }
};
HBuf hbuf(uint h)
{
    HBuf r;
    r.d = vix(h, 4u);
    r.o = 0u;
    r.n = 0xffffffffu;
    r.q = 0u;
    if (vid(h) == 0xffffffffu) {
        uint o, n, wi = vraw(h, o, n);
        if (wi <= 0x200u) {
            if (wi == 0x200u) r.q = (heapmem.Load(h) & 0xffffu) + 1u;
            else r.d = 982152u + (wi & 255u);
            r.o = o;
            r.n = n;
        }
    }
    return r;
}
uint ldu(HBuf b, uint a) { uint k = (a & 3) * 8, lo = b.Load(a & ~3u); if (k == 0) return lo; return (lo >> k) | (b.Load((a & ~3u) + 4) << (32 - k)); }
