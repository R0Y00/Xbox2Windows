// startup_gpu_heap: fixture shader using directly indexed descriptor heaps (ResourceDescriptorHeap by index, SM 6.6),
// reading or writing one slot of the mirror heap (tests/startup_gpu_heap.cpp). Not a title shader.
cbuffer Pick : register(b0)
{
    uint mode;   // 0 raw read, 1 raw write, 2 constant read, 3 typed (uint) read
    uint slot;   // the heap slot
    uint count;  // dwords
    uint mark;   // mode 1: the first value written
};
RWByteAddressBuffer result : register(u0);

struct Sixteen { uint4 v[16]; };

[numthreads(1, 1, 1)]
void PickMain()
{
    for (uint i = 0; i < count; ++i) {
        uint value = 0;
        if (mode == 0) {
            ByteAddressBuffer b = ResourceDescriptorHeap[slot];
            value = b.Load(i * 4);
        } else if (mode == 1) {
            RWByteAddressBuffer b = ResourceDescriptorHeap[slot];
            b.Store(i * 4, mark + i);
            value = mark + i;
        } else if (mode == 2) {
            ConstantBuffer<Sixteen> c = ResourceDescriptorHeap[slot];
            value = c.v[i / 4][i % 4];
        } else if (mode == 8) {
            Texture2DArray<float> t = ResourceDescriptorHeap[slot];
            value = asuint(t.Load(int4(1, 1, i >> 1, i & 1)));
        } else if (mode == 7) {
            Texture2D<float4> t = ResourceDescriptorHeap[slot];
            value = (uint)round(t.Load(int3(0, 0, 0)).r * 255);
        } else if (mode == 6) {
            RWByteAddressBuffer b = ResourceDescriptorHeap[slot];
            b.Store(i * 4, mark + i);
            value = b.Load(i * 4);
        } else if (mode == 4) {
            Texture2DArray<uint> t = ResourceDescriptorHeap[slot];
            value = t.Load(int4(i, 0, 0, 0));
        } else if (mode == 5) {
            RWTexture2DArray<uint> t = ResourceDescriptorHeap[slot];
            uint3 at = uint3(i, 0, 0);
            t[at] = mark + i;
            value = t[at];
        } else {
            Buffer<uint> t = ResourceDescriptorHeap[slot];
            value = t[i];
        }
        result.Store(i * 4, value);
    }
}
