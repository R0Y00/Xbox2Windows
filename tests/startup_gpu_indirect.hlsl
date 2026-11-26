// tests/startup_gpu_indirect.cpp: the mesh program the clamped mesh records run (our own, never a title's).
// Each group counts itself and notes the largest group number in each dimension; it emits one degenerate
// triangle (no pixels; the pipeline has no pixel shader or targets).
#define RS "UAV(u0)"
RWByteAddressBuffer seen : register(u0);

struct Vertex {
    float4 position : SV_Position;
};

[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void MeshMain(uint3 group : SV_GroupID, out vertices Vertex verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);
    for (uint i = 0; i < 3; ++i) verts[i].position = float4(0, 0, 0, 1);
    tris[0] = uint3(0, 1, 2);
    uint was;
    seen.InterlockedAdd(0, 1, was);
    seen.InterlockedMax(4, group.x, was);
    seen.InterlockedMax(8, group.y, was);
    seen.InterlockedMax(12, group.z, was);
}
