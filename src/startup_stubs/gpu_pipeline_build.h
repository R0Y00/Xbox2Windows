// Host root signature and pipeline construction for translated shaders. Tests supply a device and exercise the ABI
// with compiled HLSL on WARP and hardware.
//
// Each stage gets user_data_dwords 32-bit root constants (multiple of four) at b0 space0. Resource and sampler
// heaps are directly indexed. Immediate samplers become static samplers at s<n> space0 in list order. Title root
// CBVs use b<p> space1. Title root SRVs and UAVs use bounded one-view tables at t<p> or u<p> space1 with volatile
// descriptors and data. The stride table (t255), memory directory (t254) and fault buffer (u253) stay root
// descriptors. Disabling bounded_root_views makes every root SRV/UAV direct.
//
// Noncompute roots, including mesh, allow the input assembler. Use root signature version 1.1. DXIL declares any
// required wave size.
#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "startup_stubs/gpu_pipeline_state.h"
#include "startup_stubs/gpu_rect.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // stream subobjects are padded to pointer alignment on purpose
#endif

namespace x2w_gpu {

constexpr UINT vp_user_data_space = 0;
constexpr UINT vp_sampler_space = 0;
constexpr UINT vp_root_cbv_space = 1;
constexpr std::uint8_t vp_slot_strides = 255;
constexpr std::uint8_t vp_slot_memory = 254;
constexpr std::uint8_t vp_slot_faults = 253;

// A decoded immediate sampler; the integers are D3D12_FILTER, D3D12_TEXTURE_ADDRESS_MODE, D3D12_COMPARISON_FUNC and
// D3D12_STATIC_BORDER_COLOR.
struct VpStaticSampler {
    std::uint32_t filter;
    std::uint32_t address[3];
    float mip_lod_bias;
    std::uint32_t max_anisotropy;
    std::uint32_t comparison;
    std::uint32_t border;
    float min_lod;
    float max_lod;
};

// Anisotropic point-mip filtering is optional (OPTIONS19); without it the filter keeps the
// linear-mip approximation (StaticSampler::filter_for_host).
inline std::uint32_t vp_filter_for_host(std::uint32_t filter, bool aniso_point_mip_supported)
{
    return !aniso_point_mip_supported && (filter & ~0x180u) == 0x54 ? filter | 1 : filter;
}

// What the root builder reads of a translated shader (HostShader).
struct VpShaderView {
    std::uint32_t stage = vp_stage_none;  // console stage number (vp_stage_*)
    const void* dxil = nullptr;
    std::size_t dxil_size = 0;
    std::uint32_t user_data_dwords = 0;   // HostShader::user_data_dwords(): round4(max index + 1)
    const VpStaticSampler* samplers = nullptr;
    std::size_t sampler_count = 0;
    const std::uint8_t* root_cbvs = nullptr;
    std::size_t root_cbv_count = 0;
    const std::uint8_t* root_srvs = nullptr;
    std::size_t root_srv_count = 0;
    const std::uint8_t* root_uavs = nullptr;
    std::size_t root_uav_count = 0;
    const void* tag = nullptr;            // the caller's record, kept in VpBoundStage
};

// HostShader::user_data_dwords from the user-data indices.
inline std::uint32_t vp_user_data_dwords(const std::uint32_t* indices, std::size_t count)
{
    std::uint32_t n = 0;
    for (std::size_t i = 0; i < count; ++i) n = (std::max)(n, indices[i] + 1);
    return (n + 3) & ~3u;
}

// A title root view and the host root parameter it goes to.
struct VpRootView {
    std::uint8_t slot;    // the title's root parameter index (= the register in space1)
    std::uint32_t param;  // ours
    bool table;           // a one-view descriptor table (else a root descriptor)
};

// One stage of a host pipeline and where its bindings go.
struct VpBoundStage {
    std::uint32_t stage = vp_stage_none;
    const void* tag = nullptr;
    bool has_user_data = false;
    std::uint32_t user_data_param = 0;
    std::uint32_t user_data_dwords = 0;
    std::vector<VpRootView> root_cbvs, root_srvs, root_uavs;
};

struct VpRootOptions {
    bool bounded_root_views = true;     // Bound title root SRVs and UAVs by their view size.
    bool aniso_point_mip = false;       // OPTIONS19 AnisoFilterWithPointMipSupported
    bool heap_indexing = true;          // false only in the tests' negative control
};

// The root's description, with storage that stays put while it lives.
struct VpRootLayout {
    std::vector<D3D12_ROOT_PARAMETER1> params;
    std::vector<D3D12_DESCRIPTOR_RANGE1> ranges;  // reserved up front: params point into it
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    std::vector<VpBoundStage> stages;
};

inline D3D12_SHADER_VISIBILITY vp_visibility(std::uint32_t stage)
{
    switch (stage) {
    case vp_stage_vertex: return D3D12_SHADER_VISIBILITY_VERTEX;
    case vp_stage_pixel: return D3D12_SHADER_VISIBILITY_PIXEL;
    case vp_stage_mesh: return D3D12_SHADER_VISIBILITY_MESH;
    default: return D3D12_SHADER_VISIBILITY_ALL;
    }
}

// The root parameters the stages' bindings define, in binding order: per stage its user data, then root CBVs, SRVs,
// UAVs; static samplers per stage. `stages` is the compute stage alone, or the vertex or mesh stage then the pixel
// stage.
inline VpRootLayout vp_root_layout(const VpShaderView* stages, std::size_t count, bool compute, const VpRootOptions& o)
{
    VpRootLayout r;
    std::size_t views = 0;
    for (std::size_t s = 0; s < count; ++s) views += stages[s].root_srv_count + stages[s].root_uav_count;
    r.ranges.reserve(views);
    for (std::size_t s = 0; s < count; ++s) {
        const VpShaderView& t = stages[s];
        const D3D12_SHADER_VISIBILITY visibility = compute ? D3D12_SHADER_VISIBILITY_ALL : vp_visibility(t.stage);
        VpBoundStage bound;
        bound.stage = t.stage;
        bound.tag = t.tag;
        if (t.user_data_dwords > 0) {
            bound.has_user_data = true;
            bound.user_data_param = static_cast<std::uint32_t>(r.params.size());
            bound.user_data_dwords = t.user_data_dwords;
            D3D12_ROOT_PARAMETER1 p{};
            p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p.Constants = {0, vp_user_data_space, t.user_data_dwords};
            p.ShaderVisibility = visibility;
            r.params.push_back(p);
        }
        struct Kind {
            const std::uint8_t* slots;
            std::size_t count;
            D3D12_ROOT_PARAMETER_TYPE type;
            std::vector<VpRootView>* out;
        };
        const Kind kinds[] = {
            {t.root_cbvs, t.root_cbv_count, D3D12_ROOT_PARAMETER_TYPE_CBV, &bound.root_cbvs},
            {t.root_srvs, t.root_srv_count, D3D12_ROOT_PARAMETER_TYPE_SRV, &bound.root_srvs},
            {t.root_uavs, t.root_uav_count, D3D12_ROOT_PARAMETER_TYPE_UAV, &bound.root_uavs},
        };
        for (const Kind& k : kinds) {
            for (std::size_t i = 0; i < k.count; ++i) {
                const std::uint8_t slot = k.slots[i];
                const bool uav = k.type == D3D12_ROOT_PARAMETER_TYPE_UAV;
                const bool ours = uav ? slot == vp_slot_faults : (slot == vp_slot_strides || slot == vp_slot_memory);
                const std::uint32_t param = static_cast<std::uint32_t>(r.params.size());
                D3D12_ROOT_PARAMETER1 p{};
                p.ShaderVisibility = visibility;
                if ((uav || k.type == D3D12_ROOT_PARAMETER_TYPE_SRV) && !ours && o.bounded_root_views) {
                    D3D12_DESCRIPTOR_RANGE1 range{};
                    range.RangeType = uav ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                    range.NumDescriptors = 1;
                    range.BaseShaderRegister = slot;
                    range.RegisterSpace = vp_root_cbv_space;
                    range.Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE;
                    range.OffsetInDescriptorsFromTableStart = 0;
                    r.ranges.push_back(range);
                    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                    p.DescriptorTable = {1, &r.ranges.back()};
                    k.out->push_back({slot, param, true});
                } else {
                    p.ParameterType = k.type;
                    p.Descriptor = {slot, vp_root_cbv_space, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE};
                    k.out->push_back({slot, param, false});
                }
                r.params.push_back(p);
            }
        }
        for (std::size_t n = 0; n < t.sampler_count; ++n) {
            const VpStaticSampler& ss = t.samplers[n];
            D3D12_STATIC_SAMPLER_DESC d{};
            d.Filter = static_cast<D3D12_FILTER>(vp_filter_for_host(ss.filter, o.aniso_point_mip));
            d.AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.address[0]);
            d.AddressV = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.address[1]);
            d.AddressW = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(ss.address[2]);
            d.MipLODBias = ss.mip_lod_bias;
            d.MaxAnisotropy = ss.max_anisotropy;
            d.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(ss.comparison);
            d.BorderColor = static_cast<D3D12_STATIC_BORDER_COLOR>(ss.border);
            d.MinLOD = ss.min_lod;
            d.MaxLOD = ss.max_lod;
            d.ShaderRegister = static_cast<UINT>(n);
            d.RegisterSpace = vp_sampler_space;
            d.ShaderVisibility = visibility;
            r.samplers.push_back(d);
        }
        r.stages.push_back(std::move(bound));
    }
    r.flags = compute ? D3D12_ROOT_SIGNATURE_FLAG_NONE : D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    if (o.heap_indexing)
        r.flags |= D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    return r;
}

// Serializes the layout (version 1.1). On failure `why` holds the serializer's text.
inline HRESULT vp_serialize_root(const VpRootLayout& r, std::vector<unsigned char>& blob, std::string& why)
{
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = static_cast<UINT>(r.params.size());
    desc.Desc_1_1.pParameters = r.params.empty() ? nullptr : r.params.data();
    desc.Desc_1_1.NumStaticSamplers = static_cast<UINT>(r.samplers.size());
    desc.Desc_1_1.pStaticSamplers = r.samplers.empty() ? nullptr : r.samplers.data();
    desc.Desc_1_1.Flags = r.flags;
    ID3DBlob* out = nullptr;
    ID3DBlob* error = nullptr;
    const HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &out, &error);
    if (error) {
        why.assign(static_cast<const char*>(error->GetBufferPointer()), error->GetBufferSize());
        error->Release();
    }
    if (SUCCEEDED(hr) && out) {
        const auto* p = static_cast<const unsigned char*>(out->GetBufferPointer());
        blob.assign(p, p + out->GetBufferSize());
    }
    if (out) out->Release();
    return SUCCEEDED(hr) && blob.empty() ? E_FAIL : hr;
}

// Query device support for mesh shaders, depth bounds and OPTIONS19 once per device.
struct VpDeviceCaps {
    bool mesh_shaders = false;     // MeshShaderTier >= 1 and shader model 6.6
    bool depth_bounds = false;     // OPTIONS2 DepthBoundsTestSupported
    bool aniso_point_mip = false;  // OPTIONS19 AnisoFilterWithPointMipSupported
};

inline VpDeviceCaps vp_device_caps(ID3D12Device* device)
{
    VpDeviceCaps caps;
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    caps.mesh_shaders = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7))) &&
        o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED &&
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) &&
        model.HighestShaderModel >= D3D_SHADER_MODEL_6_6;
    D3D12_FEATURE_DATA_D3D12_OPTIONS2 o2{};
    caps.depth_bounds = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &o2, sizeof(o2))) &&
        o2.DepthBoundsTestSupported;
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 o19{};
    caps.aniso_point_mip = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &o19, sizeof(o19))) &&
        o19.AnisoFilterWithPointMipSupported;
    return caps;
}

// A stream subobject: its type, then its description, pointer-aligned.
template <class T>
struct alignas(alignof(void*)) VpSub {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
    T value;
};

template <class S>
HRESULT vp_stream_pipeline(ID3D12Device* device, S& stream, ID3D12PipelineState** out)
{
    ID3D12Device2* device2 = nullptr;
    HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device2));
    if (FAILED(hr)) return hr;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(S), &stream};
    hr = device2->CreatePipelineState(&desc, IID_PPV_ARGS(out));
    device2->Release();
    return hr;
}

// Mesh pipeline state streams omit the input layout and specify triangle topology. The library requires an explicit
// topology and translated mesh shaders emit triangles.
template <class D>
struct VpMeshStream {
    VpSub<ID3D12RootSignature*> root;
    VpSub<D3D12_SHADER_BYTECODE> ms;
    VpSub<D3D12_SHADER_BYTECODE> ps;
    VpSub<D3D12_BLEND_DESC> blend;
    VpSub<UINT> mask;
    VpSub<D3D12_RASTERIZER_DESC> raster;
    VpSub<D> depth;
    VpSub<D3D12_RT_FORMAT_ARRAY> targets;
    VpSub<DXGI_FORMAT> dsv;
    VpSub<DXGI_SAMPLE_DESC> samples;
    VpSub<D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

inline HRESULT vp_mesh_pipeline(ID3D12Device* device, ID3D12RootSignature* root, D3D12_SHADER_BYTECODE ms,
                                D3D12_SHADER_BYTECODE ps, const VpGraphicsState& g, bool depth_bounds,
                                ID3D12PipelineState** out)
{
    const auto fill = [&](auto& s) {
        s.root = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root};
        s.ms = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms};
        s.ps = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps};
        s.blend = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, vp_host_blend(g.blend, g.targets)};
        s.mask = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask};
        s.raster = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer};
        s.targets = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets};
        s.dsv = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format};
        s.samples = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples};
        s.topology = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE};
    };
    if (depth_bounds) {
        VpMeshStream<D3D12_DEPTH_STENCIL_DESC1> s{};
        fill(s);
        s.depth = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, g.depth1()};
        return vp_stream_pipeline(device, s, out);
    }
    VpMeshStream<D3D12_DEPTH_STENCIL_DESC> s{};
    fill(s);
    s.depth = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth};
    return vp_stream_pipeline(device, s, out);
}

// A graphics pipeline enabling the depth bounds test: the plain description has no field for it, so it is made as a
// stream with DEPTH_STENCIL1.
inline HRESULT vp_graphics_depth_bounds_pipeline(ID3D12Device* device, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& d,
                                                 const D3D12_DEPTH_STENCIL_DESC1& depth, ID3D12PipelineState** out)
{
    struct Stream {
        VpSub<ID3D12RootSignature*> root;
        VpSub<D3D12_SHADER_BYTECODE> vs;
        VpSub<D3D12_SHADER_BYTECODE> ps;
        VpSub<D3D12_BLEND_DESC> blend;
        VpSub<UINT> mask;
        VpSub<D3D12_RASTERIZER_DESC> raster;
        VpSub<D3D12_DEPTH_STENCIL_DESC1> depth;
        VpSub<D3D12_INPUT_LAYOUT_DESC> input;
        VpSub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE> strip_cut;
        VpSub<D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
        VpSub<D3D12_RT_FORMAT_ARRAY> targets;
        VpSub<DXGI_FORMAT> dsv;
        VpSub<DXGI_SAMPLE_DESC> samples;
        VpSub<D3D12_SHADER_BYTECODE> gs;  // left out of the size when there is no GS
    } s{};
    D3D12_RT_FORMAT_ARRAY targets{};
    for (int i = 0; i < 8; ++i) targets.RTFormats[i] = d.RTVFormats[i];
    targets.NumRenderTargets = d.NumRenderTargets;
    s.root = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, d.pRootSignature};
    s.vs = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, d.VS};
    s.ps = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, d.PS};
    s.blend = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, d.BlendState};
    s.mask = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, d.SampleMask};
    s.raster = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, d.RasterizerState};
    s.depth = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth};
    s.input = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, d.InputLayout};
    s.strip_cut = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, d.IBStripCutValue};
    s.topology = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, d.PrimitiveTopologyType};
    s.targets = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, targets};
    s.dsv = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, d.DSVFormat};
    s.samples = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, d.SampleDesc};
    s.gs = {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, d.GS};
    ID3D12Device2* device2 = nullptr;
    HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device2));
    if (FAILED(hr)) return hr;
    const std::size_t bytes = d.GS.pShaderBytecode ? sizeof(s) : offsetof(Stream, gs);
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{bytes, &s};
    hr = device2->CreatePipelineState(&desc, IID_PPV_ARGS(out));
    device2->Release();
    return hr;
}

// A made host pipeline. Holds one reference on pso and root.
struct VpHostPipeline {
    bool compute = false;
    bool mesh = false;
    ID3D12PipelineState* pso = nullptr;
    ID3D12RootSignature* root = nullptr;
    std::vector<unsigned char> root_blob;
    // Vertex or mesh, then pixel; or the compute stage.
    std::vector<VpBoundStage> stages;
    std::uint32_t slots = 0;           // vertex buffer slots read (bit per slot)
    std::vector<VpInput> inputs;
    bool rects = false;                // host PSO has the rectangle geometry stage
    bool depth_bounds = false;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;  // the title's (19/32 kept)

    VpHostPipeline() = default;
    VpHostPipeline(const VpHostPipeline&) = delete;
    VpHostPipeline& operator=(const VpHostPipeline&) = delete;
    ~VpHostPipeline()
    {
        if (pso) pso->Release();
        if (root) root->Release();
    }
};

struct VpMakeOptions {
    VpRootOptions root;
    HRESULT (*load_compute)(ID3D12Device*, const std::vector<unsigned char>&,
                            const D3D12_COMPUTE_PIPELINE_STATE_DESC&, ID3D12PipelineState**) = nullptr;
    bool allow_mesh = true;  // X2W_GPU_MESH=0 turns it off
};

// Makes the host pipeline for translated `shaders` with the graphics state `g` (null: none was kept, as for the
// plain graphics description), or for one compute shader. Returns false with `why` set when it is refused.
inline bool vp_make_host_pipeline(ID3D12Device* device, const VpDeviceCaps& caps, const VpShaderView* shaders,
                                  std::size_t count, const VpGraphicsState* g, const VpMakeOptions& options,
                                  VpHostPipeline& out, std::string& why)
{
    const auto find = [&](std::uint32_t stage) -> const VpShaderView* {
        for (std::size_t i = 0; i < count; ++i)
            if (shaders[i].stage == stage) return &shaders[i];
        return nullptr;
    };
    const bool compute = count == 1 && shaders[0].stage == vp_stage_compute;
    if (!compute) {
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t s = shaders[i].stage;
            if (s != vp_stage_vertex && s != vp_stage_pixel && s != vp_stage_mesh) {
                why = "only vertex, mesh, pixel and compute stages are translated";
                return false;
            }
        }
    }
    const VpShaderView* mesh = find(vp_stage_mesh);
    std::vector<VpShaderView> stages;
    if (compute) {
        stages.push_back(shaders[0]);
    } else {
        const VpShaderView* first = mesh ? mesh : find(vp_stage_vertex);
        if (!first) {
            why = "no vertex or mesh shader";
            return false;
        }
        stages.push_back(*first);
        if (const VpShaderView* ps = find(vp_stage_pixel)) stages.push_back(*ps);
    }
    if (mesh && !options.allow_mesh) {
        why = "mesh pipelines are off (X2W_GPU_MESH=0)";
        return false;
    }
    if (mesh && !caps.mesh_shaders) {
        why = "the host GPU has no mesh shaders (or no shader model 6.6)";
        return false;
    }
    for (const auto& s : stages) {
        if (!s.dxil || !s.dxil_size) {
            why = "a stage has no translated DXIL";
            return false;
        }
    }
    if (!compute && !g) {
        why = "graphics state not kept (plain description)";
        return false;
    }
    if (g && g->bad_inputs) {
        why = "the input layout was not readable or had more than 32 elements";
        return false;
    }

    VpRootOptions root_options = options.root;
    root_options.aniso_point_mip = caps.aniso_point_mip;
    VpRootLayout layout = vp_root_layout(stages.data(), stages.size(), compute, root_options);
    std::vector<unsigned char> blob;
    std::string text;
    HRESULT hr = vp_serialize_root(layout, blob, text);
    if (FAILED(hr)) {
        char line[64];
        std::snprintf(line, sizeof(line), "host root signature refused 0x%08lx: ", static_cast<unsigned long>(hr));
        why = line + text;
        return false;
    }
    ID3D12RootSignature* root = nullptr;
    hr = device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        char line[64];
        std::snprintf(line, sizeof(line), "CreateRootSignature 0x%08lx", static_cast<unsigned long>(hr));
        why = line;
        return false;
    }
    const auto code = [&](std::size_t i) {
        return i < stages.size() ? D3D12_SHADER_BYTECODE{stages[i].dxil, stages[i].dxil_size} : D3D12_SHADER_BYTECODE{};
    };
    ID3D12PipelineState* pso = nullptr;
    const char* what = "host pipeline";
    bool depth_bounds = false;
    if (compute) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = root;
        desc.CS = code(0);
        bool create = true;
        if (options.load_compute) {
            hr = options.load_compute(device, blob, desc, &pso);
            create = FAILED(hr) && SUCCEEDED(device->GetDeviceRemovedReason());
        }
        if (create) hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
    } else {
        depth_bounds = g->depth_bounds && caps.depth_bounds;
        if (mesh) {
            what = "host mesh pipeline";
            hr = vp_mesh_pipeline(device, root, code(0), code(1), *g, depth_bounds, &pso);
        } else {
            const std::vector<D3D12_INPUT_ELEMENT_DESC> inputs = g->input_layout();
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = root;
            desc.VS = code(0);
            desc.PS = code(1);
            std::vector<unsigned char> rect_gs;
            if (static_cast<int>(g->topology) == vp_topology_rectangles) {
                rect_gs = vp_rectangle_stage(desc.VS, desc.PS);
                if (!rect_gs.empty()) desc.GS = {rect_gs.data(), rect_gs.size()};
            }
            desc.BlendState = vp_host_blend(g->blend, g->targets);
            desc.SampleMask = g->sample_mask;
            desc.RasterizerState = g->rasterizer;
            desc.DepthStencilState = g->depth;
            desc.InputLayout = {inputs.empty() ? nullptr : inputs.data(), static_cast<UINT>(inputs.size())};
            desc.IBStripCutValue = g->strip_cut;
            desc.PrimitiveTopologyType = vp_host_topology_type(g->topology);
            desc.NumRenderTargets = g->targets.NumRenderTargets;
            for (int i = 0; i < 8; ++i) desc.RTVFormats[i] = g->targets.RTFormats[i];
            desc.DSVFormat = g->depth_format;
            desc.SampleDesc = g->samples;
            const auto make = [&] {
                return depth_bounds ? vp_graphics_depth_bounds_pipeline(device, desc, g->depth1(), &pso)
                                    : device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso));
            };
            hr = make();
            if (FAILED(hr) && desc.GS.pShaderBytecode) {
                std::fprintf(stderr, "x2w-gpu pipeline: host refused rectangle geometry stage 0x%08lx, retrying without it\n",
                             static_cast<unsigned long>(hr));
                desc.GS = {};
                hr = make();
            }
            out.rects = SUCCEEDED(hr) && desc.GS.pShaderBytecode != nullptr;
        }
    }
    if (FAILED(hr) || !pso) {
        char line[96];
        std::snprintf(line, sizeof(line), "%s refused 0x%08lx", what, static_cast<unsigned long>(hr));
        why = line;
        root->Release();
        return false;
    }
    out.compute = compute;
    out.mesh = mesh != nullptr;
    out.pso = pso;
    out.root = root;
    out.root_blob = std::move(blob);
    out.stages = std::move(layout.stages);
    out.depth_bounds = depth_bounds;
    if (g && !compute && !mesh) {
        out.slots = g->slots();
        out.inputs = g->elements();
    }
    if (g) out.topology = g->topology;
    return true;
}

}  // namespace x2w_gpu

#ifdef _MSC_VER
#pragma warning(pop)
#endif
