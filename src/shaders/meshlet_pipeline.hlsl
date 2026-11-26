struct MeshletVertex {
    float3 position;
    float pad0;
    float3 normal;
    float pad1;
    float4 color;
};

struct MeshletDescriptor {
    uint vertex_count;
    uint vertex_offset;
    uint primitive_count;
    uint primitive_offset;
};

struct MeshletBounds {
    float3 center;
    float radius;
    float3 cone_apex;
    float cone_cutoff;
    float3 cone_axis;
    float pad0;
};

cbuffer SceneConstants : register(b0) {
    row_major float4x4 view_proj;
    float4 frustum_planes[6];
    float4 camera_pos; // xyz = camera world position, w > 0.5 enables AS culling
    uint meshlet_count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<MeshletVertex> g_vertices : register(t0);
StructuredBuffer<MeshletDescriptor> g_meshlets : register(t1);
StructuredBuffer<MeshletBounds> g_bounds : register(t2);
StructuredBuffer<uint> g_unique_vertex_indices : register(t3);
StructuredBuffer<uint> g_packed_primitives : register(t4);
RWStructuredBuffer<uint> g_cull_stats : register(u0); // [0]=tested, [1]=frustum_culled, [2]=backface_culled, [3]=dispatched

struct MeshletPayload {
    uint meshlet_indices[32];
};

struct MeshOutput {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
};

groupshared MeshletPayload s_payload;
groupshared uint s_visible_count;

bool is_frustum_culled(MeshletBounds b) {
    [unroll]
    for (uint i = 0; i < 6; ++i) {
        if (dot(frustum_planes[i].xyz, b.center) + frustum_planes[i].w < -b.radius) {
            return true;
        }
    }
    return false;
}

bool is_backface_culled(MeshletBounds b) {
    if (b.cone_cutoff >= 1.5f) {
        return false;
    }
    float3 to_apex = b.cone_apex - camera_pos.xyz;
    float len_sq = dot(to_apex, to_apex);
    if (len_sq <= 1e-12f) {
        return false;
    }
    float3 view_dir = to_apex * rsqrt(len_sq);
    return dot(view_dir, b.cone_axis) >= b.cone_cutoff;
}

[numthreads(32, 1, 1)]
void ASMain(uint dtid : SV_DispatchThreadID, uint gtid : SV_GroupIndex) {
    if (gtid == 0) {
        s_visible_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (dtid < meshlet_count) {
        InterlockedAdd(g_cull_stats[0], 1);
        MeshletBounds b = g_bounds[dtid];
        bool culled = false;

        if (camera_pos.w > 0.5f) {
            if (is_frustum_culled(b)) {
                InterlockedAdd(g_cull_stats[1], 1);
                culled = true;
            } else if (is_backface_culled(b)) {
                InterlockedAdd(g_cull_stats[2], 1);
                culled = true;
            }
        }

        if (!culled) {
            uint slot = 0;
            InterlockedAdd(s_visible_count, 1, slot);
            s_payload.meshlet_indices[slot] = dtid;
            InterlockedAdd(g_cull_stats[3], 1);
        }
    }

    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(s_visible_count, 1, 1, s_payload);
}

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void MSMain(
    uint gtid : SV_GroupIndex,
    uint gid : SV_GroupID,
    in payload MeshletPayload payload,
    out vertices MeshOutput verts[64],
    out indices uint3 tris[126]) {
    uint meshlet_id = payload.meshlet_indices[gid];
    MeshletDescriptor desc = g_meshlets[meshlet_id];

    SetMeshOutputCounts(desc.vertex_count, desc.primitive_count);

    if (gtid < desc.vertex_count) {
        uint v_idx = g_unique_vertex_indices[desc.vertex_offset + gtid];
        MeshletVertex v = g_vertices[v_idx];
        MeshOutput o;
        o.position = mul(float4(v.position, 1.0f), view_proj);
        o.color = v.color;
        verts[gtid] = o;
    }

    for (uint p = gtid; p < desc.primitive_count; p += 64) {
        uint packed = g_packed_primitives[desc.primitive_offset + p];
        tris[p] = uint3(packed & 0xFFu, (packed >> 8) & 0xFFu, (packed >> 16) & 0xFFu);
    }
}

MeshOutput VSMain(VSInput input) {
    MeshOutput o;
    o.position = mul(float4(input.position, 1.0f), view_proj);
    o.color = input.color;
    return o;
}

float4 PSMain(MeshOutput input) : SV_Target {
    return input.color;
}
