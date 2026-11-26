#pragma once
#include "startup_stubs/gpu_abi.h"
#include "startup_stubs/gpu_pipeline_state.h"
namespace x2w_gpu {
static_assert(sizeof(x2w_gpu::GpuStaticSampler) == sizeof(x2w_gpu::VpStaticSampler),
              "the export's decoded sampler and the builder's are the same fields");

// A record as the root builder reads it (the record's spans live for the process).
inline VpShaderView vp_record_view(const GpuShader* s, std::uint32_t stage)
{
    x2w_gpu::VpShaderView v;
    v.stage = stage;
    v.tag = s;
    if (!s) return v;
    v.dxil = s->dxil;
    v.dxil_size = s->dxil_bytes;
    v.user_data_dwords = s->user_data_dwords();
    v.samplers = reinterpret_cast<const x2w_gpu::VpStaticSampler*>(s->static_samplers.data);
    v.sampler_count = s->static_samplers.count;
    v.root_cbvs = s->root_cbvs.data;
    v.root_cbv_count = s->root_cbvs.count;
    v.root_srvs = s->root_srvs.data;
    v.root_srv_count = s->root_srvs.count;
    v.root_uavs = s->root_uavs.data;
    v.root_uav_count = s->root_uavs.count;
    return v;
}

}
