#pragma once

#include "meshlet_builder.h"

#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>

namespace x2w::meshlet {

struct MeshShaderCapabilities {
    D3D12_MESH_SHADER_TIER mesh_shader_tier = D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
    D3D_SHADER_MODEL highest_shader_model = D3D_SHADER_MODEL_5_1;
    bool has_device2 = false;
    bool has_command_list6 = false;

    bool supported() const noexcept {
        return mesh_shader_tier >= D3D12_MESH_SHADER_TIER_1 &&
               highest_shader_model >= D3D_SHADER_MODEL_6_5 &&
               has_device2;
    }
};

MeshShaderCapabilities query_mesh_shader_capabilities(ID3D12Device* device);

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Value>
struct alignas(void*) PipelineSubobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Value value{};
};

// D3D12_PIPELINE_STATE_STREAM_DESC layout matching D3DX12_MESH_SHADER_PIPELINE_STATE_DESC
// (AS + MS + PS) without requiring an external d3dx12.h dependency.
struct MeshAmplifiedPipelineStream {
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root_signature;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, D3D12_SHADER_BYTECODE> as;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> rasterizer;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth_stencil;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtv_formats;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv_format;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> sample_desc;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

// Variant without AS subobject (an empty AS subobject is invalid in D3D12).
struct MeshOnlyPipelineStream {
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root_signature;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT> sample_mask;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> rasterizer;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth_stencil;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> rtv_formats;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> dsv_format;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> sample_desc;
    PipelineSubobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

enum class MeshShaderPipelineMode {
    Auto,
    ForceMeshShader,
    ForceVertexFallback,
};

struct MeshShaderBytecode {
    D3D12_SHADER_BYTECODE as{};
    D3D12_SHADER_BYTECODE ms{};
    D3D12_SHADER_BYTECODE vs{};
    D3D12_SHADER_BYTECODE ps{};
};

struct alignas(256) SceneConstantsGPU {
    float view_proj[16]{};
    float frustum_planes[6][4]{};
    float camera_pos[4]{};
    std::uint32_t meshlet_count = 0;
    std::uint32_t pad0 = 0;
    std::uint32_t pad1 = 0;
    std::uint32_t pad2 = 0;
};
static_assert(sizeof(SceneConstantsGPU) == 256, "SceneConstantsGPU must be 256-byte CBV aligned");

struct MeshletGpuResources {
    Microsoft::WRL::ComPtr<ID3D12Resource> vertices;
    Microsoft::WRL::ComPtr<ID3D12Resource> indices;
    Microsoft::WRL::ComPtr<ID3D12Resource> meshlets;
    Microsoft::WRL::ComPtr<ID3D12Resource> bounds;
    Microsoft::WRL::ComPtr<ID3D12Resource> unique_vertex_indices;
    Microsoft::WRL::ComPtr<ID3D12Resource> packed_primitives;
    D3D12_VERTEX_BUFFER_VIEW vbv{};
    D3D12_INDEX_BUFFER_VIEW ibv{};
    std::uint32_t meshlet_count = 0;
    std::uint32_t index_count = 0;
};

class MeshShaderPipeline {
public:
    // Returns false only when mode == ForceMeshShader and the device lacks MeshShaderTier_1 / SM 6.5 / ID3D12Device2.
    bool initialize(
        ID3D12Device* device,
        const MeshShaderBytecode& shaders,
        MeshShaderPipelineMode mode,
        DXGI_FORMAT rtv_format,
        DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN);

    // The mesh path accepts up to 64 vertices and 126 primitives per meshlet.
    MeshletGpuResources create_gpu_resources(ID3D12Device* device, const MeshletBufferData& data) const;

    void record_draw(
        ID3D12GraphicsCommandList* list,
        D3D12_GPU_VIRTUAL_ADDRESS scene_cb_va,
        D3D12_GPU_VIRTUAL_ADDRESS stats_uav_va,
        const MeshletGpuResources& resources) const;

    bool using_mesh_shaders() const noexcept { return using_mesh_shaders_; }
    const MeshShaderCapabilities& capabilities() const noexcept { return caps_; }

private:
    MeshShaderCapabilities caps_{};
    bool using_mesh_shaders_ = false;
    bool has_amplification_ = false;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_state_;
};

} // namespace x2w::meshlet
