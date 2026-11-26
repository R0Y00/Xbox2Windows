// LOCAL isa-gaps: the translated program of startup_cx_async.cpp mode heapwords (manifest `heap 1` + `heapwords 1`,
// d3d12_compute.inl / d3d12_vdesc.inl). The helper is the one the compute translator emits for `heapwords 1`
// (board 045000): a slot without the view-token magic reads the title's words from heapmem as they are (a raw
// descriptor: here the V# the runtime wrote for a slot-70 memory SRV, d3d12_heap_raw.inl); a token slot reads the
// view's words from vdesc (t3 space1, 32 bytes per view id, bounded: an id past it reads 0). Stores to root UAV 0:
//   bytes 16..31: slot 1 (runtime-written V#, 16 bytes) words 0, 2, 3; slot 2 (the title's memcpy of slot 1) word 2
//   bytes 32..47: slot 2 word 3; slot 3 (a token with id 0xffffffff) words 0 and 2; slot 0 (zeros) word 0
//   bytes 48..51: the title idiom on slot 1: w = word 2 & 0x3fff; w == 0 ? 0 : ((w << 2) + 4) >> 2
//   bytes 52..63: slot 6 (a view token: a slot-18 raw SRV of 16 bytes of the title buffer) words 1, 2, 3 from vdesc
RWByteAddressBuffer counter : register(u0);
ByteAddressBuffer heapmem : register(t0, space1);
ByteAddressBuffer vinfo : register(t1, space1);
ByteAddressBuffer x2w_vdesc : register(t3, space1);
uint x2w_hword(uint h, uint i) { if (heapmem.Load(h) != 0x56573258u) return heapmem.Load(h + 4u * i); uint id = heapmem.Load(h + 4u); return id < 0x08000000u ? x2w_vdesc.Load(id * 32u + 4u * i) : 0u; }
[numthreads(1, 1, 1)]
void WordsMain()
{
    counter.Store4(16, uint4(x2w_hword(32u, 0u), x2w_hword(32u, 2u), x2w_hword(32u, 3u), x2w_hword(64u, 2u)));
    counter.Store4(32, uint4(x2w_hword(64u, 3u), x2w_hword(96u, 0u), x2w_hword(96u, 2u), x2w_hword(0u, 0u)));
    uint w = x2w_hword(32u, 2u) & 0x3fffu;
    counter.Store(48, w == 0u ? 0u : ((w << 2) + 4u) >> 2);
    counter.Store3(52, uint3(x2w_hword(192u, 1u), x2w_hword(192u, 2u), x2w_hword(192u, 3u)));
}
