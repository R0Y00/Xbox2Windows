// startup_gpu_pipeline <warp|hardware> <root|compute|graphics|mesh> [--mutate no-heap-flags] [--debug]
//
// The host root signatures and pipelines of src/startup_stubs/gpu_pipeline_build.h (WP2, adapted from the upstream
// HostPipeline::new) from synthetic HostShader records over fixture HLSL using the shader binding ABI
// (tests/startup_gpu_pipeline.hlsl, dxc SM 6.6):
// - root: the serialized root of a compute record (8 dwords of user data, root CBV b2, root SRVs t5
//   and t255, root UAV u7, one static sampler) and of a vertex+pixel record has the expected parameter
//   list: b0 space0 constants sized user_data_dwords, a root CBV b2 space1, a one-view volatile SRV
//   table t5 space1, a root SRV t255 space1 (the strides slot stays a root descriptor), a one-view UAV
//   table u7 space1, the static sampler s0 space0, per-stage visibility, and the flags
//   CBV_SRV_UAV/SAMPLER_HEAP_DIRECTLY_INDEXED (+ ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT for graphics).
//   The anisotropic point-mip filter falls back to linear mip without OPTIONS19 support.
// - compute: the compute pipeline is made and a dispatch through the returned parameter indices
//   reads user data, the root CBV, the root SRV table, the root SRV descriptor, a texture and a
//   sampler from the heaps and the static sampler, and writes a heap UAV and the root UAV table.
// - graphics: a console stream (input layout, blend on an integer target, no depth) is parsed and
//   made into a graphics pipeline; a draw writes the pixel stage's user data, root CBV and a heap
//   view to an R32G32B32A32_UINT target. The vertex stage's own user data scales the positions.
// - mesh: the same with a console mesh stream (MS = 24): a mesh pipeline, DispatchMesh(1, 1, 1).
// --mutate no-heap-flags leaves the DIRECTLY_INDEXED flags off: the root check or the pipeline
// creation must fail (negative control, WILL_FAIL).
// Exit 0 pass, 77 capability skip (no adapter; no shader model 6.6 / resource binding tier 3; no
// mesh shaders), other values fail.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "startup_stubs/gpu_pipeline_pending.h"
#include "startup_gpu_pipeline_cs.h"
#include "startup_gpu_pipeline_ms.h"
#include "startup_gpu_pipeline_ps.h"
#include "startup_gpu_pipeline_vs.h"

using Microsoft::WRL::ComPtr;
using namespace x2w_gpu;

namespace {

bool mutate = false;
bool reference_miss = false;
unsigned reference_attempts = 0;
int failures = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);            \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

VpMakeOptions options()
{
    VpMakeOptions o;
    o.root.heap_indexing = !mutate;
    if (reference_miss) o.load_compute = [](ID3D12Device*, const std::vector<unsigned char>& blob,
                                           const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc, ID3D12PipelineState** out) -> HRESULT {
        ++reference_attempts;
        EXPECT(!blob.empty() && desc.pRootSignature && desc.CS.BytecodeLength);
        *out = nullptr;
        return E_INVALIDARG;
    };
    return o;
}

const std::uint8_t cs_cbvs[] = {2};
const std::uint8_t cs_srvs[] = {5, 255};
const std::uint8_t cs_uavs[] = {7};
const std::uint8_t ps_cbvs[] = {2};
// Point filtering, clamp: D3D12_FILTER_MIN_MAG_MIP_POINT, ADDRESS_MODE_CLAMP, NEVER, opaque black.
const VpStaticSampler point_clamp{0, {3, 3, 3}, 0.f, 1, 1, 1, 0.f, 3.402823466e+38f};

VpShaderView compute_record()
{
    VpShaderView v;
    v.stage = vp_stage_compute;
    v.dxil = startup_gpu_pipeline_cs;
    v.dxil_size = sizeof(startup_gpu_pipeline_cs);
    const std::uint32_t used[] = {0, 1, 2, 4, 5};  // dwords the shader reads (ud[0].xyz, ud[1].xy)
    v.user_data_dwords = vp_user_data_dwords(used, 5);
    v.samplers = &point_clamp;
    v.sampler_count = 1;
    v.root_cbvs = cs_cbvs;
    v.root_cbv_count = 1;
    v.root_srvs = cs_srvs;
    v.root_srv_count = 2;
    v.root_uavs = cs_uavs;
    v.root_uav_count = 1;
    return v;
}

VpShaderView stage_record(std::uint32_t stage, const void* dxil, std::size_t size, std::uint32_t dwords, bool cbv)
{
    VpShaderView v;
    v.stage = stage;
    v.dxil = dxil;
    v.dxil_size = size;
    v.user_data_dwords = dwords;
    if (cbv) {
        v.root_cbvs = ps_cbvs;
        v.root_cbv_count = 1;
    }
    return v;
}

VpShaderView pixel_record() { return stage_record(vp_stage_pixel, startup_gpu_pipeline_ps, sizeof(startup_gpu_pipeline_ps), 8, true); }

bool open_device(const char* backend, ComPtr<ID3D12Device>& device, bool debug)
{
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    if (debug) {
        ComPtr<ID3D12Debug> layer;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&layer)))) layer->EnableDebugLayer();
    }
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(backend, "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            if (FAILED(factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate)))) return false;
            DXGI_ADAPTER_DESC1 desc{};
            candidate->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                std::printf("adapter=%ls\n", desc.Description);
                return true;
            }
        }
    }
    if (std::strcmp(backend, "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return false;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return false;
    std::printf("adapter=WARP\n");
    return true;
}

// SM 6.6 dynamic resources need shader model 6.6 and resource binding tier 3.
bool heap_indexing_supported(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))) &&
        model.HighestShaderModel >= D3D_SHADER_MODEL_6_6 &&
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o))) &&
        o.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;
}

// ---- root ----

int check_root(ID3D12Device* device)
{
    const VpShaderView cs = compute_record();
    EXPECT(cs.user_data_dwords == 8);
    const VpRootLayout layout = vp_root_layout(&cs, 1, true, options().root);
    std::vector<unsigned char> blob;
    std::string why;
    if (FAILED(vp_serialize_root(layout, blob, why))) {
        std::printf("FAIL serialize: %s\n", why.c_str());
        return 1;
    }
    ComPtr<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root))));
    ComPtr<ID3D12VersionedRootSignatureDeserializer> reader;
    if (FAILED(D3D12CreateVersionedRootSignatureDeserializer(blob.data(), blob.size(), IID_PPV_ARGS(&reader)))) return 1;
    const D3D12_VERSIONED_ROOT_SIGNATURE_DESC* desc = nullptr;
    if (FAILED(reader->GetRootSignatureDescAtVersion(D3D_ROOT_SIGNATURE_VERSION_1_1, &desc))) return 1;
    const D3D12_ROOT_SIGNATURE_DESC1& d = desc->Desc_1_1;
    EXPECT(d.Flags == (D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED));
    EXPECT(d.NumParameters == 5);
    if (d.NumParameters == 5) {
        const auto& p = d.pParameters;
        EXPECT(p[0].ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS && p[0].Constants.ShaderRegister == 0 &&
               p[0].Constants.RegisterSpace == 0 && p[0].Constants.Num32BitValues == 8 && p[0].ShaderVisibility == D3D12_SHADER_VISIBILITY_ALL);
        EXPECT(p[1].ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV && p[1].Descriptor.ShaderRegister == 2 && p[1].Descriptor.RegisterSpace == 1 &&
               p[1].Descriptor.Flags == D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE);
        EXPECT(p[2].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE && p[2].DescriptorTable.NumDescriptorRanges == 1);
        if (p[2].DescriptorTable.NumDescriptorRanges == 1) {
            const auto& r = p[2].DescriptorTable.pDescriptorRanges[0];
            EXPECT(r.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV && r.NumDescriptors == 1 && r.BaseShaderRegister == 5 && r.RegisterSpace == 1 &&
                   r.Flags == (D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE) &&
                   r.OffsetInDescriptorsFromTableStart == 0);
        }
        EXPECT(p[3].ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV && p[3].Descriptor.ShaderRegister == 255 && p[3].Descriptor.RegisterSpace == 1);
        EXPECT(p[4].ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE && p[4].DescriptorTable.NumDescriptorRanges == 1);
        if (p[4].DescriptorTable.NumDescriptorRanges == 1) {
            const auto& r = p[4].DescriptorTable.pDescriptorRanges[0];
            EXPECT(r.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_UAV && r.BaseShaderRegister == 7 && r.RegisterSpace == 1);
        }
    }
    EXPECT(d.NumStaticSamplers == 1);
    if (d.NumStaticSamplers == 1) {
        const auto& s = d.pStaticSamplers[0];
        EXPECT(s.ShaderRegister == 0 && s.RegisterSpace == 0 && s.AddressU == D3D12_TEXTURE_ADDRESS_MODE_CLAMP &&
               s.ComparisonFunc == D3D12_COMPARISON_FUNC_NEVER && s.ShaderVisibility == D3D12_SHADER_VISIBILITY_ALL);
    }
    // The binding record the bind package reads.
    EXPECT(layout.stages.size() == 1);
    if (layout.stages.size() == 1) {
        const VpBoundStage& b = layout.stages[0];
        EXPECT(b.has_user_data && b.user_data_param == 0 && b.user_data_dwords == 8);
        EXPECT(b.root_cbvs.size() == 1 && b.root_cbvs[0].slot == 2 && b.root_cbvs[0].param == 1 && !b.root_cbvs[0].table);
        EXPECT(b.root_srvs.size() == 2 && b.root_srvs[0].param == 2 && b.root_srvs[0].table && b.root_srvs[1].slot == 255 &&
               b.root_srvs[1].param == 3 && !b.root_srvs[1].table);
        EXPECT(b.root_uavs.size() == 1 && b.root_uavs[0].param == 4 && b.root_uavs[0].table);
    }
    // X2W_SHADERCFG_ROOT_SRV_DIRECT=1: every root view a root descriptor.
    VpRootOptions direct = options().root;
    direct.bounded_root_views = false;
    const VpRootLayout plain = vp_root_layout(&cs, 1, true, direct);
    EXPECT(plain.params.size() == 5 && plain.params[2].ParameterType == D3D12_ROOT_PARAMETER_TYPE_SRV &&
           plain.params[4].ParameterType == D3D12_ROOT_PARAMETER_TYPE_UAV && plain.ranges.empty());

    // Vertex + pixel: per-stage visibility, the input-layout flag, ids in order.
    const VpShaderView g[2] = {stage_record(vp_stage_vertex, startup_gpu_pipeline_vs, sizeof(startup_gpu_pipeline_vs), 8, false),
                               pixel_record()};
    const VpRootLayout gl = vp_root_layout(g, 2, false, options().root);
    EXPECT(gl.params.size() == 3);
    if (gl.params.size() == 3) {
        EXPECT(gl.params[0].ShaderVisibility == D3D12_SHADER_VISIBILITY_VERTEX && gl.params[0].ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS);
        EXPECT(gl.params[1].ShaderVisibility == D3D12_SHADER_VISIBILITY_PIXEL && gl.params[1].ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS);
        EXPECT(gl.params[2].ShaderVisibility == D3D12_SHADER_VISIBILITY_PIXEL && gl.params[2].ParameterType == D3D12_ROOT_PARAMETER_TYPE_CBV);
    }
    EXPECT(gl.flags & D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    std::vector<unsigned char> gblob;
    EXPECT(SUCCEEDED(vp_serialize_root(gl, gblob, why)));
    ComPtr<ID3D12RootSignature> groot;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, gblob.data(), gblob.size(), IID_PPV_ARGS(&groot))));
    // No user data: no constants parameter. A mesh stage sees its own.
    VpShaderView ms = stage_record(vp_stage_mesh, startup_gpu_pipeline_ms, sizeof(startup_gpu_pipeline_ms), 0, false);
    const VpRootLayout ml = vp_root_layout(&ms, 1, false, options().root);
    EXPECT(ml.params.empty() && !ml.stages[0].has_user_data);
    ms.user_data_dwords = 4;
    EXPECT(vp_root_layout(&ms, 1, false, options().root).params[0].ShaderVisibility == D3D12_SHADER_VISIBILITY_MESH);

    // Anisotropic point-mip (0x54) keeps its filter only with OPTIONS19 support; comparison and
    // reduction bits stay.
    EXPECT(vp_filter_for_host(0x54, false) == 0x55 && vp_filter_for_host(0x54, true) == 0x54);
    EXPECT(vp_filter_for_host(0xd4, false) == 0xd5 && vp_filter_for_host(0x55, false) == 0x55 && vp_filter_for_host(0x15, false) == 0x15);
    EXPECT(vp_user_data_dwords(nullptr, 0) == 0);
    const std::uint32_t nine[] = {8};
    EXPECT(vp_user_data_dwords(nine, 1) == 12);
    return failures ? 1 : 0;
}

// ---- GPU helpers ----

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    UINT64 value = 0;

    bool open(D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT)
    {
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = type;
        return SUCCEEDED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) &&
            SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))) &&
            SUCCEEDED(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) &&
            SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }

    bool run()
    {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++value);
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(value, event);
        WaitForSingleObject(event, 10000);
        CloseHandle(event);
        return fence->GetCompletedValue() >= value && SUCCEEDED(device->GetDeviceRemovedReason());
    }

    ComPtr<ID3D12Resource> buffer(D3D12_HEAP_TYPE heap, UINT64 bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES hp{heap};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = d.DepthOrArraySize = d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = flags;
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    ComPtr<ID3D12Resource> texture(DXGI_FORMAT format, UINT size, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = d.Height = size;
        d.DepthOrArraySize = d.MipLevels = 1;
        d.Format = format;
        d.SampleDesc.Count = 1;
        d.Flags = flags;
        ComPtr<ID3D12Resource> r;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }
};

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, from, to};
    list->ResourceBarrier(1, &b);
}

void write(ID3D12Resource* upload, std::size_t at, const void* data, std::size_t bytes)
{
    void* p = nullptr;
    upload->Map(0, nullptr, &p);
    std::memcpy(static_cast<char*>(p) + at, data, bytes);
    upload->Unmap(0, nullptr);
}

void put32(ID3D12Resource* upload, std::size_t at, std::uint32_t v) { write(upload, at, &v, 4); }

void raw_srv(ID3D12Device* device, ID3D12Resource* r, UINT64 first, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_R32_TYPELESS;
    v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Buffer.FirstElement = first;
    v.Buffer.NumElements = 64;
    v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    device->CreateShaderResourceView(r, &v, at);
}

void raw_uav(ID3D12Device* device, ID3D12Resource* r, UINT64 first, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_R32_TYPELESS;
    v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    v.Buffer.FirstElement = first;
    v.Buffer.NumElements = 64;
    v.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    device->CreateUnorderedAccessView(r, nullptr, &v, at);
}

struct Heaps {
    ComPtr<ID3D12DescriptorHeap> views, samplers;
    UINT view_step = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT i) const
    {
        auto h = views->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(i) * view_step;
        return h;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT i) const
    {
        auto h = views->GetGPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<UINT64>(i) * view_step;
        return h;
    }
    bool open(ID3D12Device* device)
    {
        D3D12_DESCRIPTOR_HEAP_DESC v{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        D3D12_DESCRIPTOR_HEAP_DESC s{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        view_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (FAILED(device->CreateDescriptorHeap(&v, IID_PPV_ARGS(&views))) || FAILED(device->CreateDescriptorHeap(&s, IID_PPV_ARGS(&samplers))))
            return false;
        // Heap sampler 1: point, clamp.
        D3D12_SAMPLER_DESC sd{};
        sd.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sd.MaxLOD = D3D12_FLOAT32_MAX;
        auto h = samplers->GetCPUDescriptorHandleForHeapStart();
        h.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
        device->CreateSampler(&sd, h);
        return true;
    }
};

// Root constants: a stage's user data, console byte offsets in the view and sampler heaps.
void constants(ID3D12GraphicsCommandList* list, bool compute, const VpBoundStage& stage, const std::uint32_t (&ud)[8])
{
    if (!stage.has_user_data) return;
    if (compute) list->SetComputeRoot32BitConstants(stage.user_data_param, stage.user_data_dwords, ud, 0);
    else list->SetGraphicsRoot32BitConstants(stage.user_data_param, stage.user_data_dwords, ud, 0);
}

constexpr std::uint32_t one_float = 0x3f800000u;

// ---- compute ----

int check_compute(Gpu& gpu)
{
    ID3D12Device* device = gpu.device.Get();
    const VpShaderView cs = compute_record();
    VpHostPipeline host;
    std::string why;
    if (!vp_make_host_pipeline(device, vp_device_caps(device), &cs, 1, nullptr, options(), host, why)) {
        std::printf("FAIL compute pipeline: %s\n", why.c_str());
        return 1;
    }
    EXPECT(host.compute && !host.mesh && host.stages.size() == 1);
    Heaps heaps;
    if (!heaps.open(device)) return 1;
    auto ro = gpu.buffer(D3D12_HEAP_TYPE_UPLOAD, 0x10000, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto rw = gpu.buffer(D3D12_HEAP_TYPE_DEFAULT, 0x1000, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto back = gpu.buffer(D3D12_HEAP_TYPE_READBACK, 0x1000, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto tex = gpu.texture(DXGI_FORMAT_R32_FLOAT, 1, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!ro || !rw || !back || !tex) return 1;
    put32(ro.Get(), 0x000, 0x5a5a0005u);  // root SRV table t5
    put32(ro.Get(), 0x200, 0x51515151u);  // root SRV descriptor t255
    put32(ro.Get(), 0x1000, 0xcb000002u); // root CBV b2
    const float texel = 0.75f;
    write(ro.Get(), 0x2000, &texel, 4);   // texture upload (256-aligned row)
    raw_uav(device, rw.Get(), 0, heaps.cpu(3));          // heap output view
    raw_srv(device, ro.Get(), 0, heaps.cpu(5));          // root SRV table's view
    raw_uav(device, rw.Get(), 0x400 / 4, heaps.cpu(7));  // root UAV table's view
    D3D12_SHADER_RESOURCE_VIEW_DESC tv{};
    tv.Format = DXGI_FORMAT_R32_FLOAT;
    tv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    tv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(tex.Get(), &tv, heaps.cpu(1));

    auto* list = gpu.list.Get();
    D3D12_TEXTURE_COPY_LOCATION dst{tex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{ro.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    src.PlacedFootprint = {0x2000, {DXGI_FORMAT_R32_FLOAT, 1, 1, 1, 256}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(list, tex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* both[] = {heaps.views.Get(), heaps.samplers.Get()};
    list->SetDescriptorHeaps(2, both);  // before the root: its heaps are directly indexed
    list->SetComputeRootSignature(host.root);
    list->SetPipelineState(host.pso);
    const VpBoundStage& stage = host.stages[0];
    const std::uint32_t ud[8] = {3 * 32, 1 * 32, 1 * 16, 0, 0x1234abcdu, 0x7777u, 0, 0};
    constants(list, true, stage, ud);
    for (const auto& v : stage.root_cbvs) list->SetComputeRootConstantBufferView(v.param, ro->GetGPUVirtualAddress() + 0x1000);
    for (const auto& v : stage.root_srvs) {
        if (v.table) list->SetComputeRootDescriptorTable(v.param, heaps.gpu(5));
        else list->SetComputeRootShaderResourceView(v.param, ro->GetGPUVirtualAddress() + 0x200);
    }
    for (const auto& v : stage.root_uavs) {
        if (v.table) list->SetComputeRootDescriptorTable(v.param, heaps.gpu(7));
        else list->SetComputeRootUnorderedAccessView(v.param, rw->GetGPUVirtualAddress() + 0x400);
    }
    list->Dispatch(1, 1, 1);
    barrier(list, rw.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(back.Get(), 0, rw.Get(), 0, 0x1000);
    if (!gpu.run()) {
        std::printf("FAIL dispatch did not complete (device removed 0x%08lx)\n", static_cast<unsigned long>(gpu.device->GetDeviceRemovedReason()));
        return 1;
    }
    std::uint32_t* out = nullptr;
    back->Map(0, nullptr, reinterpret_cast<void**>(&out));
    float a = 0, b = 0;
    std::memcpy(&a, out + 4, 4);
    std::memcpy(&b, out + 5, 4);
    std::printf("compute out %08x %08x %08x %08x %g %g root uav %08x\n", out[0], out[1], out[2], out[3], a, b, out[0x100]);
    EXPECT(out[0] == 0x1234abcdu);  // user data
    EXPECT(out[1] == 0xcb000002u);  // root CBV b2 space1
    EXPECT(out[2] == 0x5a5a0005u);  // root SRV t5 space1 (one-view table)
    EXPECT(out[3] == 0x51515151u);  // root SRV t255 space1 (descriptor)
    EXPECT(a == 0.75f && b == 0.75f);  // static sampler s0 and heap sampler
    EXPECT(out[0x100] == 0x7777u);  // root UAV u7 space1 (one-view table)
    back->Unmap(0, nullptr);

    // Lazy make: two title pipelines of the same stage share one host pipeline; a stage without a
    // record is refused at first use.
    VpPipelineCache cache;
    VpPending pa, pb, missing;
    pa.add(cs);
    pb.add(cs);
    VpShaderView none = cs;
    none.dxil = nullptr;
    none.dxil_size = 0;
    missing.add(none);
    const VpDeviceCaps caps = vp_device_caps(device);
    const VpHostPipeline* ha = vp_ready(pa, device, caps, cache, options());
    const VpHostPipeline* hb = vp_ready(pb, device, caps, cache, options());
    EXPECT(ha && ha == hb && vp_ready(pa, device, caps, cache, options()) == ha);
    EXPECT(cache.made_count == 1 && cache.shared_count == 1);
    // The same DXIL from another record is a pipeline of its own: the bind path reads the record's mapping.
    static const int other_record = 0;
    VpShaderView other = cs;
    other.tag = &other_record;
    VpPending pc;
    pc.add(other);
    const VpHostPipeline* hc = vp_ready(pc, device, caps, cache, options());
    EXPECT(hc && hc != ha && hc->stages[0].tag == &other_record && ha->stages[0].tag == nullptr && cache.made_count == 2);
    EXPECT(!vp_ready(missing, device, caps, cache, options()) && missing.why == "a stage has no translated record in the export");
    return failures ? 1 : 0;
}

// ---- graphics and mesh ----

// A console stream builder: subobjects at 8-byte steps, the console's numbering.
struct StreamBytes {
    std::vector<unsigned char> bytes;
    template <class T>
    void add(std::uint32_t kind, const T& value, std::size_t data_at = 4)
    {
        const std::size_t size = vp_subobject(kind).size;
        const std::size_t at = bytes.size();
        bytes.resize(at + size, 0);
        std::memcpy(&bytes[at], &kind, 4);
        std::memcpy(&bytes[at + data_at], &value, sizeof(T));
    }
};

int check_draw(Gpu& gpu, bool mesh, bool rectangles = false)
{
    ID3D12Device* device = gpu.device.Get();
    const VpDeviceCaps caps = vp_device_caps(device);
    if (mesh && !caps.mesh_shaders) {
        std::printf("no mesh shaders (tier or shader model 6.6): capability skip\n");
        return 77;
    }
    // The title's stream (console numbering: the mesh stage is 24).
    D3D12_INPUT_ELEMENT_DESC position{"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D12_APPEND_ALIGNED_ELEMENT,
                                      D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    StreamBytes s;
    s.add<void*>(0, nullptr, 8);
    if (mesh) s.add(24, D3D12_SHADER_BYTECODE{startup_gpu_pipeline_ms, sizeof(startup_gpu_pipeline_ms)}, 8);
    else s.add(1, D3D12_SHADER_BYTECODE{startup_gpu_pipeline_vs, sizeof(startup_gpu_pipeline_vs)}, 8);
    s.add(2, D3D12_SHADER_BYTECODE{startup_gpu_pipeline_ps, sizeof(startup_gpu_pipeline_ps)}, 8);
    D3D12_BLEND_DESC blend = VpGraphicsState().blend;
    blend.RenderTarget[0].BlendEnable = TRUE;  // on an integer target: host_blend turns it off
    s.add(8, blend);
    D3D12_RASTERIZER_DESC raster = VpGraphicsState().rasterizer;
    raster.CullMode = D3D12_CULL_MODE_NONE;
    s.add(10, raster);
    D3D12_DEPTH_STENCIL_DESC depth{};
    s.add(11, depth);
    if (!mesh) s.add(12, D3D12_INPUT_LAYOUT_DESC{&position, 1}, 8);
    s.add(14, rectangles ? static_cast<std::uint32_t>(vp_topology_rectangles)
                         : static_cast<std::uint32_t>(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE));
    D3D12_RT_FORMAT_ARRAY targets{};
    targets.NumRenderTargets = 1;
    targets.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_UINT;
    s.add(15, targets);
    s.add(16, static_cast<std::uint32_t>(DXGI_FORMAT_UNKNOWN));
    const VpStream stream = vp_parse_stream(s.bytes.data(), s.bytes.size(), VpFormatMap{});
    EXPECT(!stream.stopped && !stream.truncated && stream.shaders.size() == 2);
    if (stream.shaders.size() != 2) return 1;
    std::vector<VpShaderView> views;
    for (const auto& sh : stream.shaders) {
        if (sh.stage == vp_stage_pixel) views.push_back(pixel_record());
        else views.push_back(stage_record(sh.stage, sh.code, sh.size, 8, false));
    }
    VpHostPipeline host;
    std::string why;
    if (!vp_make_host_pipeline(device, caps, views.data(), views.size(), &stream.graphics, options(), host, why)) {
        std::printf("FAIL %s pipeline: %s\n", mesh ? "mesh" : "graphics", why.c_str());
        return 1;
    }
    EXPECT(host.mesh == mesh && !host.compute && host.stages.size() == 2);
    EXPECT(host.rects == rectangles);
    if (rectangles && caps.depth_bounds) {
        auto depth_state = stream.graphics;
        depth_state.depth_bounds = true;
        depth_state.depth_format = DXGI_FORMAT_D32_FLOAT;
        depth_state.depth.DepthEnable = TRUE;
        depth_state.depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        VpHostPipeline bounded;
        EXPECT(vp_make_host_pipeline(device, caps, views.data(), views.size(), &depth_state, options(), bounded, why));
        EXPECT(bounded.depth_bounds && bounded.rects);
    }
    EXPECT(mesh ? host.slots == 0 && host.inputs.empty() : host.slots == 1 && host.inputs.size() == 1);
    // X2W_GPU_MESH=0 refuses mesh pipelines.
    if (mesh) {
        VpMakeOptions off = options();
        off.allow_mesh = false;
        VpHostPipeline refused;
        std::string reason;
        EXPECT(!vp_make_host_pipeline(device, caps, views.data(), views.size(), &stream.graphics, off, refused, reason) &&
               reason == "mesh pipelines are off (X2W_GPU_MESH=0)");
    }
    // Reject a graphics pipeline description with no state.
    VpHostPipeline none;
    std::string plain;
    EXPECT(!vp_make_host_pipeline(device, caps, views.data(), views.size(), nullptr, options(), none, plain) &&
           plain == "graphics state not kept (plain description)");

    Heaps heaps;
    if (!heaps.open(device)) return 1;
    auto ro = gpu.buffer(D3D12_HEAP_TYPE_UPLOAD, 0x10000, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto back = gpu.buffer(D3D12_HEAP_TYPE_READBACK, 0x1000, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto target = gpu.texture(DXGI_FORMAT_R32G32B32A32_UINT, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (!ro || !back || !target) return 1;
    put32(ro.Get(), 0x000, 0xbeef0002u);   // heap view read by the pixel stage
    put32(ro.Get(), 0x1000, 0xcb000002u);  // root CBV b2
    // Rectangle lists supply three corners; the GS must generate the top-right corner.
    const float edge = rectangles ? 1.0f : 3.0f;
    const float tri[6] = {-1, -1, -1, edge, edge, -1};
    write(ro.Get(), 0x2000, tri, sizeof(tri));
    raw_srv(device, ro.Get(), 0, heaps.cpu(2));
    ComPtr<ID3D12DescriptorHeap> rtvs;
    D3D12_DESCRIPTOR_HEAP_DESC rd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    if (FAILED(device->CreateDescriptorHeap(&rd, IID_PPV_ARGS(&rtvs)))) return 1;
    const auto rtv = rtvs->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);

    auto* list = gpu.list.Get();
    const float clear[4] = {0, 0, 0, 0};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    ID3D12DescriptorHeap* both[] = {heaps.views.Get(), heaps.samplers.Get()};
    list->SetDescriptorHeaps(2, both);
    list->SetGraphicsRootSignature(host.root);
    list->SetPipelineState(host.pso);
    const std::uint32_t first_ud[8] = {0, 0, 0, 0, 0, 0, one_float, 0};  // the vertex or mesh stage: scale 1.0
    const std::uint32_t pixel_ud[8] = {2 * 32, 0, 0, 0, 0x9999abcdu, 0, 0, 0};
    constants(list, false, host.stages[0], first_ud);
    constants(list, false, host.stages[1], pixel_ud);
    for (const auto& stage : host.stages)
        for (const auto& v : stage.root_cbvs) list->SetGraphicsRootConstantBufferView(v.param, ro->GetGPUVirtualAddress() + 0x1000);
    D3D12_VIEWPORT viewport{0, 0, 4, 4, 0, 1};
    D3D12_RECT scissor{0, 0, 4, 4};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    if (mesh) {
        ComPtr<ID3D12GraphicsCommandList6> list6;
        if (FAILED(gpu.list.As(&list6))) return 1;
        list6->DispatchMesh(1, 1, 1);
    } else {
        const D3D12_VERTEX_BUFFER_VIEW vb{ro->GetGPUVirtualAddress() + 0x2000, sizeof(tri), 8};
        list->IASetVertexBuffers(0, 1, &vb);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
    }
    barrier(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION dst{back.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32G32B32A32_UINT, 4, 4, 1, 256}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (!gpu.run()) {
        std::printf("FAIL draw did not complete (device removed 0x%08lx)\n", static_cast<unsigned long>(gpu.device->GetDeviceRemovedReason()));
        return 1;
    }
    std::uint32_t* out = nullptr;
    back->Map(0, nullptr, reinterpret_cast<void**>(&out));
    // Pixel (3, 3): row 3 at 3 * 256 bytes, texel 3 at 48.
    const std::uint32_t* corner = out + (3 * 256 + 3 * 16) / 4;
    if (rectangles) {
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 4; ++x)
                EXPECT(out[(y * 256 + x * 16) / 4] == 0x9999abcdu);
    }
    std::printf("%s pixel (0,0) %08x %08x %08x %08x (3,3) %08x %08x %08x %08x\n", mesh ? "mesh" : "graphics", out[0], out[1], out[2],
                out[3], corner[0], corner[1], corner[2], corner[3]);
    for (const std::uint32_t* p : std::initializer_list<const std::uint32_t*>{out, corner}) {
        EXPECT(p[0] == 0x9999abcdu);  // the pixel stage's user data
        EXPECT(p[1] == 0xcb000002u);  // root CBV b2 space1 (pixel visibility)
        EXPECT(p[2] == 0xbeef0002u);  // heap view at the user data's offset
        EXPECT(p[3] == 0x600du);
    }
    back->Unmap(0, nullptr);
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    bool debug = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mutate") == 0 && i + 1 < argc && std::strcmp(argv[i + 1], "no-heap-flags") == 0) {
            mutate = true;
            ++i;
        } else if (std::strcmp(argv[i], "--reference-miss") == 0) {
            reference_miss = true;
        } else if (std::strcmp(argv[i], "--debug") == 0) {
            debug = true;
        } else {
            return 2;
        }
    }
    const std::string mode = argv[2];
    Gpu gpu;
    if (!open_device(argv[1], gpu.device, debug)) {
        std::printf("no %s adapter: capability skip\n", argv[1]);
        return 77;
    }
    if (!heap_indexing_supported(gpu.device.Get())) {
        std::printf("no shader model 6.6 or resource binding tier 3: capability skip\n");
        return 77;
    }
    int result = 2;
    if (mode == "root") result = check_root(gpu.device.Get());
    else if (!gpu.open()) result = 4;
    else if (mode == "compute") result = check_compute(gpu);
    else if (mode == "graphics") result = check_draw(gpu, false);
    else if (mode == "rectangles") result = check_draw(gpu, false, true);
    else if (mode == "mesh") result = check_draw(gpu, true);
    // --debug: the debug layer's errors fail the run too.
    ComPtr<ID3D12InfoQueue> queue;
    if (debug && result == 0 && SUCCEEDED(gpu.device.As(&queue))) {
        int errors = 0;
        for (UINT64 i = 0, n = queue->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T bytes = 0;
            queue->GetMessage(i, nullptr, &bytes);
            std::vector<char> buffer(bytes);
            auto* m = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
            if (FAILED(queue->GetMessage(i, m, &bytes))) continue;
            if (m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::printf("debug layer: %.*s\n", static_cast<int>(m->DescriptionByteLength), m->pDescription);
                ++errors;
            }
        }
        std::printf("debug layer errors: %d\n", errors);
        if (errors) result = 1;
    }
    std::printf("startup_gpu_pipeline %s %s%s: %s (%d failed)\n", argv[1], mode.c_str(), mutate ? " --mutate no-heap-flags" : "",
                result == 0 ? "pass" : result == 77 ? "skip" : "FAIL", failures);
    if (reference_miss && result == 0 && reference_attempts == 0) return 1;
    return result;
}
