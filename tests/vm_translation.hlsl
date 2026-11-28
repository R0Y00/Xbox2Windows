// Copies 1024 words from u0 to u1 with a fixed XOR, addressed only through root UAV GPU addresses.
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    dst.Store(id.x * 4, src.Load(id.x * 4) ^ 0x5a5a5a5au);
}
