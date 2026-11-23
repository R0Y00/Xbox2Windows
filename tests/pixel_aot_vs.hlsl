struct Vertex { float4 uv : TEXCOORD0; float4 position : SV_Position; };
[RootSignature("DescriptorTable(SRV(t0,numDescriptors=2),visibility=SHADER_VISIBILITY_PIXEL),CBV(b0,visibility=SHADER_VISIBILITY_PIXEL),"
               "CBV(b1,visibility=SHADER_VISIBILITY_PIXEL),CBV(b2,visibility=SHADER_VISIBILITY_PIXEL),StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_POINT,"
               "addressU=TEXTURE_ADDRESS_CLAMP,addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP)")]
Vertex main(uint id : SV_VertexID)
{
    Vertex v;
    // Vertices 3 to 5 draw the partial triangle. Vertices 6 to 8 draw the full triangle with attribute
    // values 1 + uv / 64, which need more significand bits than binary16 holds at every pixel center.
    const bool fine = id >= 6;
    const bool partial = id >= 3 && !fine;
    const uint corner = fine ? id - 6 : partial ? id - 3 : id;
    float2 xy = float2((corner << 1) & 2, corner & 2) * (partial ? 7.75f / 16.0f : 1.0f);
    v.uv = float4(xy, 0.25f + 0.5f * xy.x, 1.0f - xy.y);
    if (fine) v.uv = 1.0f + v.uv * 0.015625f;
    v.position = float4(xy * float2(2,-2) + float2(-1,1), 0, 1);
    return v;
}
