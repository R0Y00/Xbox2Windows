#include "mesh_shader_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(OFFSCREEN_AGILITY_620)
extern "C" const UINT D3D12SDKVersion;
#endif

namespace x2w::meshlet {

namespace {

using Microsoft::WRL::ComPtr;

void throw_if_failed(HRESULT hr, const char* context) {
    if (FAILED(hr)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s (HRESULT 0x%08lx)", context, static_cast<unsigned long>(hr));
        throw std::runtime_error(buf);
    }
}

HRESULT check_feature_compat(ID3D12Device* device, D3D12_FEATURE feature, void* data, UINT size) {
#if defined(OFFSCREEN_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
    ComPtr<ID3D12Device16> versioned;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&versioned)))) {
        return versioned->CheckFeatureSupport1(feature, data, size, D3D12SDKVersion);
    }
#endif
    return device->CheckFeatureSupport(feature, data, size);
}

D3D12_BLEND_DESC default_opaque_blend() {
    D3D12_BLEND_DESC desc{};
    desc.AlphaToCoverageEnable = FALSE;
    desc.IndependentBlendEnable = FALSE;
    auto& rt = desc.RenderTarget[0];
    rt.BlendEnable = FALSE;
    rt.LogicOpEnable = FALSE;
    rt.SrcBlend = D3D12_BLEND_ONE;
    rt.DestBlend = D3D12_BLEND_ZERO;
    rt.BlendOp = D3D12_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D12_BLEND_ONE;
    rt.DestBlendAlpha = D3D12_BLEND_ZERO;
    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rt.LogicOp = D3D12_LOGIC_OP_NOOP;
    rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    return desc;
}

D3D12_RASTERIZER_DESC default_rasterizer() {
    D3D12_RASTERIZER_DESC desc{};
    desc.FillMode = D3D12_FILL_MODE_SOLID;
    desc.CullMode = D3D12_CULL_MODE_BACK;
    desc.FrontCounterClockwise = FALSE;
    desc.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    desc.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    desc.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    desc.DepthClipEnable = TRUE;
    desc.MultisampleEnable = FALSE;
    desc.AntialiasedLineEnable = FALSE;
    desc.ForcedSampleCount = 0;
    desc.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
    return desc;
}

D3D12_DEPTH_STENCIL_DESC default_depth_stencil(DXGI_FORMAT dsv_format) {
    D3D12_DEPTH_STENCIL_DESC desc{};
    if (dsv_format != DXGI_FORMAT_UNKNOWN) {
        desc.DepthEnable = TRUE;
        desc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    } else {
        desc.DepthEnable = FALSE;
        desc.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    }
    desc.StencilEnable = FALSE;
    desc.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK;
    desc.StencilWriteMask = D3D12_DEFAULT_STENCIL_WRITE_MASK;
    const D3D12_DEPTH_STENCILOP_DESC noop{
        D3D12_STENCIL_OP_KEEP,
        D3D12_STENCIL_OP_KEEP,
        D3D12_STENCIL_OP_KEEP,
        D3D12_COMPARISON_FUNC_ALWAYS,
    };
    desc.FrontFace = noop;
    desc.BackFace = noop;
    return desc;
}

template <typename Stream>
void populate_common_stream_fields(
    Stream& stream,
    ID3D12RootSignature* root_sig,
    D3D12_SHADER_BYTECODE ps,
    DXGI_FORMAT rtv_format,
    DXGI_FORMAT dsv_format) {
    stream.root_signature.value = root_sig;
    stream.ps.value = ps;
    stream.blend.value = default_opaque_blend();
    stream.sample_mask.value = UINT_MAX;
    stream.rasterizer.value = default_rasterizer();
    stream.depth_stencil.value = default_depth_stencil(dsv_format);

    D3D12_RT_FORMAT_ARRAY rtv_array{};
    rtv_array.NumRenderTargets = 1;
    rtv_array.RTFormats[0] = rtv_format;
    stream.rtv_formats.value = rtv_array;

    stream.dsv_format.value = dsv_format;
    stream.sample_desc.value = DXGI_SAMPLE_DESC{1, 0};
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

ComPtr<ID3D12RootSignature> serialize_and_create_root(
    ID3D12Device* device,
    const D3D12_ROOT_SIGNATURE_DESC& desc,
    const char* label) {
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error_blob;
    const HRESULT ser_hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error_blob);
    if (FAILED(ser_hr)) {
        std::string msg = label;
        if (error_blob) {
            msg += ": ";
            msg.append(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize());
        }
        throw_if_failed(ser_hr, msg.c_str());
    }
    ComPtr<ID3D12RootSignature> root;
    throw_if_failed(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
                    label);
    return root;
}

ComPtr<ID3D12Resource> upload_typed_buffer(
    ID3D12Device* device,
    const void* data,
    std::size_t byte_size,
    const char* label) {
    const UINT64 alloc_size = std::max<UINT64>(4, static_cast<UINT64>(byte_size));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = alloc_size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ComPtr<ID3D12Resource> buffer;
    throw_if_failed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&buffer)),
                    label);
    if (byte_size > 0 && data != nullptr) {
        void* mapped = nullptr;
        const D3D12_RANGE empty{0, 0};
        throw_if_failed(buffer->Map(0, &empty, &mapped), label);
        std::memcpy(mapped, data, byte_size);
        buffer->Unmap(0, nullptr);
    }
    return buffer;
}

} // namespace

MeshShaderCapabilities query_mesh_shader_capabilities(ID3D12Device* device) {
    MeshShaderCapabilities caps{};
    if (!device) return caps;

    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7{};
    if (SUCCEEDED(check_feature_compat(device, D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7)))) {
        caps.mesh_shader_tier = options7.MeshShaderTier;
    }

    for (const D3D_SHADER_MODEL candidate : {
             D3D_SHADER_MODEL_6_6,
             D3D_SHADER_MODEL_6_5,
             D3D_SHADER_MODEL_6_4,
             D3D_SHADER_MODEL_6_0,
             D3D_SHADER_MODEL_5_1,
         }) {
        D3D12_FEATURE_DATA_SHADER_MODEL sm{candidate};
        const HRESULT hr = check_feature_compat(device, D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm));
        if (SUCCEEDED(hr)) {
            caps.highest_shader_model = sm.HighestShaderModel;
            break;
        }
    }

    ComPtr<ID3D12Device2> device2;
    caps.has_device2 = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device2)));
    caps.has_command_list6 = caps.has_device2 && (caps.mesh_shader_tier >= D3D12_MESH_SHADER_TIER_1);
    return caps;
}

bool MeshShaderPipeline::initialize(
    ID3D12Device* device,
    const MeshShaderBytecode& shaders,
    MeshShaderPipelineMode mode,
    DXGI_FORMAT rtv_format,
    DXGI_FORMAT dsv_format) {
    if (!device) throw std::invalid_argument("MeshShaderPipeline::initialize requires a valid ID3D12Device");

    caps_ = query_mesh_shader_capabilities(device);
    has_amplification_ = shaders.as.pShaderBytecode != nullptr && shaders.as.BytecodeLength > 0;

    if (mode == MeshShaderPipelineMode::ForceMeshShader) {
        if (!caps_.supported()) {
            using_mesh_shaders_ = false;
            return false;
        }
        using_mesh_shaders_ = true;
    } else if (mode == MeshShaderPipelineMode::ForceVertexFallback) {
        using_mesh_shaders_ = false;
    } else {
        using_mesh_shaders_ = caps_.supported();
    }

    if (using_mesh_shaders_) {
        D3D12_ROOT_PARAMETER params[7]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[0].Descriptor = {0, 0};
        params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        for (UINT i = 0; i < 5; ++i) {
            params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            params[1 + i].Descriptor = {i, 0};
            params[1 + i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }

        params[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[6].Descriptor = {0, 0};
        params[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.NumParameters = 7;
        root_desc.pParameters = params;
        root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        root_signature_ = serialize_and_create_root(device, root_desc, "Create mesh shader root signature");

        ComPtr<ID3D12Device2> device2;
        throw_if_failed(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query ID3D12Device2");

        if (has_amplification_) {
            MeshAmplifiedPipelineStream stream{};
            populate_common_stream_fields(stream, root_signature_.Get(), shaders.ps, rtv_format, dsv_format);
            stream.as.value = shaders.as;
            stream.ms.value = shaders.ms;
            D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
            throw_if_failed(device2->CreatePipelineState(&stream_desc, IID_PPV_ARGS(&pipeline_state_)),
                            "Create amplified mesh shader PSO");
        } else {
            MeshOnlyPipelineStream stream{};
            populate_common_stream_fields(stream, root_signature_.Get(), shaders.ps, rtv_format, dsv_format);
            stream.ms.value = shaders.ms;
            D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
            throw_if_failed(device2->CreatePipelineState(&stream_desc, IID_PPV_ARGS(&pipeline_state_)),
                            "Create mesh shader PSO");
        }
        return true;
    }

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    param.Descriptor = {0, 0};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 1;
    root_desc.pParameters = &param;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    root_signature_ = serialize_and_create_root(device, root_desc, "Create vertex fallback root signature");

    static const D3D12_INPUT_ELEMENT_DESC kInputLayout[]{
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root_signature_.Get();
    pso_desc.VS = shaders.vs;
    pso_desc.PS = shaders.ps;
    pso_desc.BlendState = default_opaque_blend();
    pso_desc.SampleMask = UINT_MAX;
    pso_desc.RasterizerState = default_rasterizer();
    pso_desc.DepthStencilState = default_depth_stencil(dsv_format);
    pso_desc.InputLayout = {kInputLayout, static_cast<UINT>(std::size(kInputLayout))};
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = rtv_format;
    pso_desc.DSVFormat = dsv_format;
    pso_desc.SampleDesc = {1, 0};

    throw_if_failed(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pipeline_state_)),
                    "Create vertex fallback PSO");
    return true;
}

MeshletGpuResources MeshShaderPipeline::create_gpu_resources(ID3D12Device* device, const MeshletBufferData& data) const {
    if (!device) throw std::invalid_argument("MeshShaderPipeline::create_gpu_resources requires a valid ID3D12Device");

    if (using_mesh_shaders_ && std::any_of(data.meshlets.begin(), data.meshlets.end(), [](const MeshletDescriptor& meshlet) {
            return meshlet.vertex_count > kMaxMeshletVertices || meshlet.primitive_count > kMaxMeshletPrimitives;
        })) {
        throw std::invalid_argument("Meshlet exceeds the mesh shader output capacity (64 vertices, 126 primitives)");
    }

    MeshletGpuResources res;
    const std::size_t vb_bytes = data.vertices.size() * sizeof(MeshletVertex);
    const std::size_t ib_bytes = data.source_indices.size() * sizeof(std::uint32_t);

    res.vertices = upload_typed_buffer(device, data.vertices.data(), vb_bytes, "Create meshlet vertices buffer");
    res.indices = upload_typed_buffer(device, data.source_indices.data(), ib_bytes, "Create fallback index buffer");
    res.meshlets = upload_typed_buffer(device, data.meshlets.data(),
                                       data.meshlets.size() * sizeof(MeshletDescriptor), "Create meshlets buffer");
    res.bounds = upload_typed_buffer(device, data.bounds.data(),
                                     data.bounds.size() * sizeof(MeshletBounds), "Create meshlet bounds buffer");
    res.unique_vertex_indices = upload_typed_buffer(device, data.unique_vertex_indices.data(),
                                                    data.unique_vertex_indices.size() * sizeof(std::uint32_t),
                                                    "Create unique vertex indices buffer");
    res.packed_primitives = upload_typed_buffer(device, data.packed_primitives.data(),
                                                data.packed_primitives.size() * sizeof(std::uint32_t),
                                                "Create packed primitives buffer");

    res.vbv.BufferLocation = res.vertices->GetGPUVirtualAddress();
    res.vbv.SizeInBytes = static_cast<UINT>(vb_bytes);
    res.vbv.StrideInBytes = sizeof(MeshletVertex);

    res.ibv.BufferLocation = res.indices->GetGPUVirtualAddress();
    res.ibv.SizeInBytes = static_cast<UINT>(ib_bytes);
    res.ibv.Format = DXGI_FORMAT_R32_UINT;

    res.meshlet_count = static_cast<std::uint32_t>(data.meshlets.size());
    res.index_count = static_cast<std::uint32_t>(data.source_indices.size());
    return res;
}

void MeshShaderPipeline::record_draw(
    ID3D12GraphicsCommandList* list,
    D3D12_GPU_VIRTUAL_ADDRESS scene_cb_va,
    D3D12_GPU_VIRTUAL_ADDRESS stats_uav_va,
    const MeshletGpuResources& resources) const {
    if (!list || !root_signature_ || !pipeline_state_) {
        throw std::runtime_error("MeshShaderPipeline::record_draw called before initialize");
    }

    list->SetGraphicsRootSignature(root_signature_.Get());
    list->SetPipelineState(pipeline_state_.Get());
    list->SetGraphicsRootConstantBufferView(0, scene_cb_va);

    if (using_mesh_shaders_) {
        if (resources.meshlet_count == 0) return;
        list->SetGraphicsRootShaderResourceView(1, resources.vertices->GetGPUVirtualAddress());
        list->SetGraphicsRootShaderResourceView(2, resources.meshlets->GetGPUVirtualAddress());
        list->SetGraphicsRootShaderResourceView(3, resources.bounds->GetGPUVirtualAddress());
        list->SetGraphicsRootShaderResourceView(4, resources.unique_vertex_indices->GetGPUVirtualAddress());
        list->SetGraphicsRootShaderResourceView(5, resources.packed_primitives->GetGPUVirtualAddress());
        list->SetGraphicsRootUnorderedAccessView(6, stats_uav_va);

        ComPtr<ID3D12GraphicsCommandList6> mesh_list;
        throw_if_failed(list->QueryInterface(IID_PPV_ARGS(&mesh_list)), "Query ID3D12GraphicsCommandList6");
        const UINT group_count = has_amplification_ ? (resources.meshlet_count + 31u) / 32u : resources.meshlet_count;
        mesh_list->DispatchMesh(group_count, 1, 1);
    } else {
        if (resources.index_count == 0) return;
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->IASetVertexBuffers(0, 1, &resources.vbv);
        list->IASetIndexBuffer(&resources.ibv);
        list->DrawIndexedInstanced(resources.index_count, 1, 0, 0, 0);
    }
}

} // namespace x2w::meshlet
