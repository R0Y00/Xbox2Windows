#define ROOT "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),RootConstants(num32BitConstants=2,b0),SRV(t254,space=1),UAV(u253,space=1),UAV(u0)"
cbuffer Params : register(b0) { uint heapBase; uint page; };
ByteAddressBuffer directory : register(t254, space1);
RWByteAddressBuffer faults : register(u253, space1);
RWByteAddressBuffer output : register(u0);
[RootSignature(ROOT)]
[numthreads(1,1,1)]
void MemoryMain(uint3 dt : SV_DispatchThreadID)
{
    uint p = page + dt.x;
    uint2 e = directory.Load2(8 * p);
    uint copy = e.x & 0x7fffffff;
    if (!copy) faults.Store((1 + p % 4095) * 4, p);
    RWByteAddressBuffer window = ResourceDescriptorHeap[NonUniformResourceIndex(heapBase + copy)];
    output.Store(dt.x * 4, window.Load(e.y + 16));
    output.Store((dt.x + 2) * 4, e.x);
}
