// LOCAL: the translated program of startup_cx_async.cpp modes rawheap / rawheap-off / rawheap-reset (manifest
// `heap 1` + `heapraw 1`, d3d12_heap_raw.inl). The heap prelude is the one the compute translator emits for
// `heapraw 1`; H0 is a heap SRV buffer as the translator declares it. Stores to the title buffer (root UAV 0):
//   16..31  slot 1 (raw V#, 16 bytes, window 1): dwords 0 and 3, dword 4 (past the 16 bytes: 0), vstride (0)
//   32..47  slot 2 (rawheap: the title's memcpy of slot 1; reset: the last rewritten V#): dwords 0 and 3, the
//           unaligned dword at byte 2, 0
//   48..51  slot 3 (a V# whose base is not 4-aligned: read byte-exact) dword 0
//   64..79  slot 4 (structured V#, stride 12 x 3 records, window 2): dword at byte 32, at byte 36 (past 36 bytes: 0),
//           vstride (12), vfmt (32_FLOAT: 0)
//   80..95  slot 5 (typed R32_UINT V#, 4 elements): dword 0, vstride (4), vfmt (the V#'s FORMAT), dword at byte 16 (0)
//   96..111 slot 6 (a view token over the title buffer) dword 0, then slot 1 dword 0 again (one program reading a token
//           slot and a V# slot), vstride / vfmt of slot 3 (raw: 0, 0)
//   112..127 slot 7 (32 bytes over two windows, a paged entry): dwords 0, 3 and 4 (the second window), the dword at
//           byte 14 (two bytes from each window)
RWByteAddressBuffer counter : register(u0);
#include "heap_raw_prelude.hlsli"
#define H0 hbuf(h0)
[numthreads(1, 1, 1)]
void RawHeapMain()
{
    uint h0 = 1u << 5;
    counter.Store4(16, uint4(H0.Load(0), H0.Load(12), H0.Load(16), vstride(h0)));
    h0 = 2u << 5;
    counter.Store4(32, uint4(H0.Load(0), H0.Load(12), ldu(H0, 2), 0u));
    h0 = 3u << 5;
    counter.Store(48, H0.Load(0));
    h0 = 4u << 5;
    counter.Store4(64, uint4(H0.Load(32), H0.Load(36), vstride(h0), vfmt(h0)));
    h0 = 5u << 5;
    counter.Store4(80, uint4(H0.Load(0), vstride(h0), vfmt(h0), H0.Load(16)));
    h0 = 6u << 5;
    uint token = H0.Load(0);
    h0 = 1u << 5;
    uint again = H0.Load(0);
    h0 = 3u << 5;
    counter.Store4(96, uint4(token, again, vstride(h0), vfmt(h0)));
    h0 = 7u << 5;
    counter.Store4(112, uint4(H0.Load(0), H0.Load(12), H0.Load(16), ldu(H0, 14)));
}

// LOCAL: startup_cx_async.cpp rawheap-bench. Title words: [1] first ring index, [2] count, [3] ring size (V# slots
// from heap slot 8). Each checked slot's V# names title data the bench filled with dword (address >> 2): the program
// reads dword 0 through the slot and counts it in [6]; a wrong or empty read (a miss: 0) counts in [5].
[numthreads(1, 1, 1)]
void RawBenchMain()
{
    uint first = counter.Load(4), count = counter.Load(8), ring = counter.Load(12), bad = 0u;
    [loop] for (uint i = 0u; i < count; ++i) {
        uint h0 = (8u + (first + i) % ring) << 5;
        uint4 w = heapmem.Load4(h0);
        uint want = (w.x >> 2) | ((w.y & 3u) << 30);
        if (H0.Load(0) != want) ++bad;
    }
    counter.Store(20, counter.Load(20) + bad);
    counter.Store(24, counter.Load(24) + count);
}
