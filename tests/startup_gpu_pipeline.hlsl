// Fixture shaders for the binding ABI (the upstream emitter, src/emit/mod.rs), for
// startup_gpu_pipeline: user data at b0 space0, the title's root views in space1 (b2 a root
// CBV, t5 a root SRV, u7 a root UAV, t255 the strides slot), an immediate sampler at s0
// space0, and views and samplers picked from the heaps by the user data (byte offsets: views 32
// bytes apart, samplers 16, as the console's descriptor sizes). Not title shaders.
cbuffer UserData : register(b0, space0)
{
    uint4 ud[2];
};
struct RootData
{
    uint4 v;
};
ConstantBuffer<RootData> root_cbv : register(b2, space1);
ByteAddressBuffer root_srv : register(t5, space1);
RWByteAddressBuffer root_uav : register(u7, space1);
ByteAddressBuffer strides : register(t255, space1);
SamplerState immediate : register(s0, space0);

// ud[0].x: output view (bytes), ud[0].y: texture view, ud[0].z: heap sampler (bytes);
// ud[1].x: a value to store, ud[1].y: the root UAV's value.
[numthreads(1, 1, 1)]
void CsMain()
{
    RWByteAddressBuffer heap_out = ResourceDescriptorHeap[ud[0].x / 32];
    Texture2D<float4> tex = ResourceDescriptorHeap[ud[0].y / 32];
    SamplerState heap_sampler = SamplerDescriptorHeap[ud[0].z / 16];
    const float a = tex.SampleLevel(immediate, float2(0.5, 0.5), 0).x;
    const float b = tex.SampleLevel(heap_sampler, float2(0.5, 0.5), 0).x;
    heap_out.Store4(0, uint4(ud[1].x, root_cbv.v.x, root_srv.Load(0), strides.Load(0)));
    heap_out.Store2(16, uint2(asuint(a), asuint(b)));
    root_uav.Store(0, ud[1].y);
}

struct VsOut
{
    float4 pos : SV_Position;
};

// ud[1].z: the scale of the position (1.0 as bits). The vertex stage's user data is its own.
VsOut VsMain(float2 p : POSITION)
{
    VsOut o;
    o.pos = float4(p * asfloat(ud[1].z), 0, 1);
    return o;
}

// One triangle over the whole target, as VsMain draws it.
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void MsMain(out vertices VsOut v[3], out indices uint3 t[1])
{
    SetMeshOutputCounts(3, 1);
    const float s = asfloat(ud[1].z);
    v[0].pos = float4(-1 * s, -1 * s, 0, 1);
    v[1].pos = float4(-1 * s, 3 * s, 0, 1);
    v[2].pos = float4(3 * s, -1 * s, 0, 1);
    t[0] = uint3(0, 1, 2);
}

// ud[0].x: a raw buffer view (bytes); ud[1].x: a value. The pixel stage's user data is its own.
uint4 PsMain() : SV_Target
{
    ByteAddressBuffer heap_in = ResourceDescriptorHeap[ud[0].x / 32];
    return uint4(ud[1].x, root_cbv.v.x, heap_in.Load(0), 0x600d);
}
