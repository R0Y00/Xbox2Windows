// WP3 vbind: synthetic compute shaders for the binding ABI described in the upstream emitter, src/emit/mod.rs.
//   user data: b0 space0, 32-bit root constants;  tables: ResourceDescriptorHeap[byte offset / 32 + entry];
//   root SRV t3 space1 and root UAV u5 space1 as one-view tables of raw views, read at the bias the host passes.
struct Constants {
    uint4 v;
};
cbuffer UserData : register(b0, space0)
{
    uint ud_table;     // the table's byte offset in its heap, console units (32 a view)
    uint ud_srv_bias;  // USER_VIEW_BIAS of root parameter 3
    uint ud_uav_bias;  // USER_VIEW_BIAS of root parameter 5
    uint ud_count;     // bounds: dwords to store
};
ByteAddressBuffer root_srv : register(t3, space1);
RWByteAddressBuffer root_uav : register(u5, space1);

// Entry 0 of the table: a raw SRV; entry 1: a CBV (made afresh: the title's is not on 256 bytes).
[numthreads(1, 1, 1)]
void bind_main()
{
    const uint base = ud_table / 32;
    ByteAddressBuffer listed = ResourceDescriptorHeap[base + 0];
    ConstantBuffer<Constants> constants = ResourceDescriptorHeap[base + 1];
    const uint a = listed.Load(0);
    const uint b = root_srv.Load(ud_srv_bias);
    const uint4 c = constants.v;
    root_uav.Store4(ud_uav_bias, uint4(a, b, c.x, c.w));
}

// ud_count dwords from the root UAV's address on: the view's bounds decide what lands.
[numthreads(64, 1, 1)]
void bounds_main(uint i : SV_DispatchThreadID)
{
    if (i < ud_count) root_uav.Store(ud_uav_bias + i * 4, 0xc0de0000u | i);
}
