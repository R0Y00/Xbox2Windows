RWByteAddressBuffer counter : register(u0);

ByteAddressBuffer heapmem : register(t0, space1);
[numthreads(1, 1, 1)]
void HeapMain()
{
    counter.Store2(4, uint2(heapmem.Load(32), heapmem.Load(128)));
}

[numthreads(1, 1, 1)]
void CountMain()
{
    uint before;
    counter.InterlockedAdd(0, 1, before);
}
