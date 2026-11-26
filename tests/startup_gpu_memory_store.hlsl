#define ROOT "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED),RootConstants(num32BitConstants=3,b0),SRV(t254,space=1),UAV(u253,space=1),UAV(u0),SRV(t0)"
cbuffer Params : register(b0) { uint base; uint page; uint value; };
ByteAddressBuffer directory : register(t254, space1);
RWByteAddressBuffer faults : register(u253, space1);
RWByteAddressBuffer result : register(u0);
ByteAddressBuffer ordinary : register(t0);
[RootSignature(ROOT)]
[numthreads(1,1,1)]
void StoreMain() {
    uint2 e = directory.Load2(page*8);
    uint number = e.x & 0x7fffffff;
    if (!number) faults.Store((1+page%4095)*4,page);
    RWByteAddressBuffer readWindow = ResourceDescriptorHeap[NonUniformResourceIndex(base + number)];
    if (value == 0x1234eeee) {
        result.Store2(0, uint2(readWindow.Load(e.y + 16), ordinary.Load(0)));
        return;
    }
    result.Store2(0,readWindow.Load2(e.y+16));
    if (value) {
        uint writable = (int(e.x)<0) ? 0 : number;
        RWByteAddressBuffer writeWindow = ResourceDescriptorHeap[NonUniformResourceIndex(base+writable)];
        writeWindow.Store(e.y+16,value);
    }
}
