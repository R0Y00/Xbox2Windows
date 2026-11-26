// device creation, root signatures, compute, graphics and mesh pipelines, the stub shaders, and the test
// drivers Host::draw and Host::dispatch.
//
// stub_vs/stub_ps (the HostShader wrappers) compile once x2w_shader/native.h exists; the DXIL
// they carry (stub_vs_dxil/stub_ps_dxil) is built now.

#include "x2w_shader/emit/host.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <thread>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/signature.h"

#if __has_include("x2w_shader/native.h")
#define X2W_VS_HOST_STUBS 1
#include "x2w_shader/emit/emit.h"
#else
#define X2W_VS_HOST_STUBS 0
#endif

namespace x2w::vs::host {

namespace {

[[noreturn]] void fail(HRESULT hr) { throw Error(message(hr)); }
void check(HRESULT hr) {
    if (FAILED(hr)) fail(hr);
}

std::size_t round_up(std::size_t x, std::size_t to) { return (x + to - 1) / to * to; }

bool env_is(const char* name, const char* value) {
    const char* v = std::getenv(name);
    return v && std::strcmp(v, value) == 0;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    // A borrowed pointer; the barrier never releases it.
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

D3D12_RESOURCE_DESC buffer_desc(std::uint64_t size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC texture_desc(std::uint32_t w, std::uint32_t h, D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc = {1, 0};
    d.Flags = flags;
    return d;
}

ComPtr<ID3D12Resource> make(ID3D12Device* dev, D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear = nullptr) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    ComPtr<ID3D12Resource> r;
    check(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, clear, IID_PPV_ARGS(&r)));
    if (!r) throw Error("no resource");
    return r;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* r, UINT index = 0) {
    D3D12_TEXTURE_COPY_LOCATION l{};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    l.SubresourceIndex = index;
    return l;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* r, DXGI_FORMAT format, std::uint32_t w, std::uint32_t h,
                                      std::uint32_t pitch) {
    D3D12_TEXTURE_COPY_LOCATION l{};
    l.pResource = r;
    l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    l.PlacedFootprint.Offset = 0;
    l.PlacedFootprint.Footprint = {format, w, h, 1, pitch};
    return l;
}

// Copies w x h RGBA32F texels (rows of w) into an upload buffer of pitch-byte rows.
void fill_rows(ID3D12Resource* upload, const std::array<float, 4>* texels, std::uint32_t w, std::uint32_t h,
               std::size_t pitch) {
    void* p = nullptr;
    check(upload->Map(0, nullptr, &p));
    for (std::size_t y = 0; y < h; ++y)
        std::memcpy(static_cast<std::uint8_t*>(p) + y * pitch, texels + y * w, std::size_t(w) * 16);
    upload->Unmap(0, nullptr);
}

// Runs list on queue and waits for it; throws when the device is removed.
void run(ID3D12Device* dev, ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* list, bool check_after) {
    check(list->Close());
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    check(queue->Signal(fence.Get(), 1));
    while (fence->GetCompletedValue() < 1) {
        HRESULT reason = dev->GetDeviceRemovedReason();
        if (FAILED(reason)) throw Error("device removed: " + message(reason));
        std::this_thread::yield();
    }
    if (check_after) {
        HRESULT reason = dev->GetDeviceRemovedReason();
        if (FAILED(reason)) throw Error("device removed: " + message(reason));
    }
}

struct Queue {
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
};

Queue direct_queue(ID3D12Device* dev, ID3D12PipelineState* pso) {
    Queue q;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q.queue)));
    check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&q.alloc)));
    check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, q.alloc.Get(), pso, IID_PPV_ARGS(&q.list)));
    return q;
}

ComPtr<ID3D12DescriptorHeap> heap(ID3D12Device* dev, D3D12_DESCRIPTOR_HEAP_TYPE ty, std::size_t n) {
    D3D12_DESCRIPTOR_HEAP_DESC d{};
    d.Type = ty;
    d.NumDescriptors = static_cast<UINT>(std::max<std::size_t>(n, 1));
    d.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> h;
    check(dev->CreateDescriptorHeap(&d, IID_PPV_ARGS(&h)));
    return h;
}

}  // namespace

std::string message(HRESULT hr) {
    char* text = nullptr;
    DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                             nullptr, static_cast<DWORD>(hr), 0, reinterpret_cast<char*>(&text), 0, nullptr);
    std::string s;
    if (n && text) {
        s.assign(text, n);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
            s.pop_back();
    }
    if (text) LocalFree(text);
    if (s.empty()) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "HRESULT 0x%08x", static_cast<unsigned>(hr));
        s = buf;
    }
    return s;
}

Host::Host(bool warp) {
    warp = warp || env_is("X2W_SHADERCFG_WARP", "1");
    // X2W_SHADERCFG_D3D12_DEBUG: the debug layer, whose messages a refused pipeline's error then
    // carries.
    if (std::getenv("X2W_SHADERCFG_D3D12_DEBUG")) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))) && debug) debug->EnableDebugLayer();
    }
    if (warp) {
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> adapter;
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
    } else {
        check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        // A machine without a graphics card gives the software adapter as its default device (a
        // hosted test runner, whose first adapter is a display stub that cannot make a device):
        // that is no hardware device, and the tests that ask for one skip it, as they do where
        // there is none.
        if (device) {
            ComPtr<IDXGIFactory4> factory;
            check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
            ComPtr<IDXGIAdapter1> adapter;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))) &&
                SUCCEEDED(adapter->GetDesc1(&desc)) &&
                ((desc.Flags & 2) != 0 || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)))
                throw Error("the default device is the software adapter");
        }
    }
    if (!device) throw Error("no device");
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    check(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof sm));
    if (sm.HighestShaderModel < D3D_SHADER_MODEL_6_6) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "the device supports SM 0x%x only", static_cast<unsigned>(sm.HighestShaderModel));
        throw Error(buf);
    }
}

ComPtr<ID3D12RootSignature> Host::root_signature(const Stage& vs, const Stage& ps) const {
    return root_signature_for({{vs, D3D12_SHADER_VISIBILITY_VERTEX}, {ps, D3D12_SHADER_VISIBILITY_PIXEL}});
}

ComPtr<ID3D12RootSignature> Host::root_signature_for(
    std::span<const std::pair<Stage, D3D12_SHADER_VISIBILITY>> stages) const {
    std::vector<D3D12_ROOT_PARAMETER1> params;
    for (const auto& [s, vis] : stages) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p.Constants.ShaderRegister = 0;
        p.Constants.RegisterSpace = shader::USER_DATA_SPACE;
        p.Constants.Num32BitValues = std::max<std::uint32_t>(s.user_data_dwords, 1);
        p.ShaderVisibility = vis;
        params.push_back(p);
    }
    // By slot, then kind: 0 CBV, 1 SRV, 2 UAV.
    std::vector<std::pair<std::uint8_t, std::uint8_t>> views;
    for (const auto& [s, vis] : stages) {
        for (std::uint8_t r : s.root_cbvs) views.emplace_back(r, std::uint8_t{0});
        for (std::uint8_t r : s.root_srvs) views.emplace_back(r, std::uint8_t{1});
        for (std::uint8_t r : s.root_uavs) views.emplace_back(r, std::uint8_t{2});
    }
    std::sort(views.begin(), views.end());
    views.erase(std::unique(views.begin(), views.end()), views.end());
    for (const auto& [slot, kind] : views) {
        D3D12_ROOT_PARAMETER1 p{};
        p.ParameterType = kind == 1   ? D3D12_ROOT_PARAMETER_TYPE_SRV
                          : kind == 2 ? D3D12_ROOT_PARAMETER_TYPE_UAV
                                      : D3D12_ROOT_PARAMETER_TYPE_CBV;
        p.Descriptor.ShaderRegister = slot;
        p.Descriptor.RegisterSpace = shader::ROOT_CBV_SPACE;
        p.Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;
        p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        params.push_back(p);
    }
    // Older runtimes may not recognize OPTIONS19; that means no support.
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 options{};
    const bool aniso_point_mip_supported =
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &options, sizeof options)) &&
        options.AnisoFilterWithPointMipSupported;
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    for (const auto& [s, vis] : stages) {
        for (std::size_t n = 0; n < s.static_samplers.size(); ++n) {
            const sampler::StaticSampler& d = s.static_samplers[n];
            D3D12_STATIC_SAMPLER_DESC sd{};
            sd.Filter = static_cast<D3D12_FILTER>(d.filter_for_host(aniso_point_mip_supported));
            sd.AddressU = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(d.address[0]);
            sd.AddressV = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(d.address[1]);
            sd.AddressW = static_cast<D3D12_TEXTURE_ADDRESS_MODE>(d.address[2]);
            sd.MipLODBias = d.mip_lod_bias;
            sd.MaxAnisotropy = d.max_anisotropy;
            sd.ComparisonFunc = static_cast<D3D12_COMPARISON_FUNC>(d.comparison);
            sd.BorderColor = static_cast<D3D12_STATIC_BORDER_COLOR>(d.border);
            sd.MinLOD = d.min_lod;
            sd.MaxLOD = d.max_lod;
            sd.ShaderRegister = static_cast<UINT>(n);
            sd.RegisterSpace = shader::SAMPLER_SPACE;
            sd.ShaderVisibility = vis;
            samplers.push_back(sd);
        }
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.NumParameters = static_cast<UINT>(params.size());
    desc.Desc_1_1.pParameters = params.data();
    desc.Desc_1_1.NumStaticSamplers = static_cast<UINT>(samplers.size());
    desc.Desc_1_1.pStaticSamplers = samplers.data();
    desc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                          D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
                          D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> error;
    check(D3D12SerializeVersionedRootSignature(&desc, &blob, &error));
    if (!blob) throw Error("no root signature blob");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
    return root;
}

Pipeline Host::compute_pipeline(const Stage& cs) const {
    auto root = root_signature_for({{cs, D3D12_SHADER_VISIBILITY_ALL}});
    auto pso = compute_pipeline_with(root.Get(), cs.dxil);
    return {std::move(root), std::move(pso)};
}

ComPtr<ID3D12PipelineState> Host::compute_pipeline_with(ID3D12RootSignature* root,
                                                        std::span<const std::uint8_t> cs) const {
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    // A borrowed pointer the descriptor never releases.
    desc.pRootSignature = root;
    desc.CS = {cs.data(), cs.size()};
    ComPtr<ID3D12PipelineState> pso;
    HRESULT hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) throw Error(explained(hr));
    return pso;
}

std::string Host::explained(HRESULT hr) const {
    std::string text = message(hr);
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device.As(&queue))) return text;
    // Each message is read into a buffer of the size the queue asks for, aligned for the message
    // header.
    const UINT64 count = queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T len = 0;
        (void)queue->GetMessage(i, nullptr, &len);
        std::vector<std::uint64_t> buffer((len + 7) / 8);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
        if (SUCCEEDED(queue->GetMessage(i, m, &len))) {
            const SIZE_T n = m->DescriptionByteLength ? m->DescriptionByteLength - 1 : 0;
            text += " | ";
            text.append(m->pDescription, n);
        }
    }
    queue->ClearStoredMessages();
    return text;
}

std::vector<std::array<float, 4>> Host::draw(ID3D12RootSignature* root, ID3D12PipelineState* pso,
                                             const Draw& d) const {
    ID3D12Device* dev = device.Get();
    const auto [w, h] = d.size;
    // D3D12 calls with descriptions that outlive them; resources are kept alive until the fence
    // says the GPU is done.
    Queue q = direct_queue(dev, pso);
    ID3D12GraphicsCommandList* list = q.list.Get();
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    auto target = make(dev, D3D12_HEAP_TYPE_DEFAULT,
                       texture_desc(w, h, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, DXGI_FORMAT_R32G32B32A32_FLOAT),
                       D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 1;
        check(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap)));
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(target.Get(), nullptr, rtv);
    auto views = heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, d.textures.size());
    auto sampler_heap = heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, d.samplers.size());
    const std::size_t view_size = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const std::size_t sampler_size = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    std::vector<ComPtr<ID3D12Resource>> keep;
    for (std::size_t i = 0; i < d.textures.size(); ++i) {
        const Texture& t = d.textures[i];
        if (t.texels.size() < std::size_t(t.width) * t.height) bitcode::bug("host: texture texels too short");
        auto tex = make(dev, D3D12_HEAP_TYPE_DEFAULT,
                        texture_desc(t.width, t.height, D3D12_RESOURCE_FLAG_NONE, DXGI_FORMAT_R32G32B32A32_FLOAT),
                        D3D12_RESOURCE_STATE_COPY_DEST);
        const std::size_t pitch = round_up(std::size_t(t.width) * 16, 256);
        auto upload = make(dev, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(pitch * t.height, D3D12_RESOURCE_FLAG_NONE),
                           D3D12_RESOURCE_STATE_GENERIC_READ);
        fill_rows(upload.Get(), t.texels.data(), t.width, t.height, pitch);
        const auto dst = subresource(tex.Get());
        const auto src = footprint(upload.Get(), DXGI_FORMAT_R32G32B32A32_FLOAT, t.width, t.height,
                                   static_cast<std::uint32_t>(pitch));
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        const auto barrier = transition(tex.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE |
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &barrier);
        D3D12_CPU_DESCRIPTOR_HANDLE at = views->GetCPUDescriptorHandleForHeapStart();
        at.ptr += i * view_size;
        dev->CreateShaderResourceView(tex.Get(), nullptr, at);
        keep.push_back(std::move(tex));
        keep.push_back(std::move(upload));
    }
    for (std::size_t i = 0; i < d.samplers.size(); ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE at = sampler_heap->GetCPUDescriptorHandleForHeapStart();
        at.ptr += i * sampler_size;
        dev->CreateSampler(&d.samplers[i], at);
    }
    ComPtr<ID3D12Resource> vb;
    if (!d.vertices.empty()) {
        const std::size_t bytes = d.vertices.size() * 4;
        vb = make(dev, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE),
                  D3D12_RESOURCE_STATE_GENERIC_READ);
        void* p = nullptr;
        check(vb->Map(0, nullptr, &p));
        std::memcpy(p, d.vertices.data(), bytes);
        vb->Unmap(0, nullptr);
    }
    ID3D12DescriptorHeap* heaps[] = {views.Get(), sampler_heap.Get()};
    list->SetDescriptorHeaps(2, heaps);
    list->SetGraphicsRootSignature(root);
    for (const auto& [param, arg] : d.roots) {
        switch (arg.kind) {
            case Root::Constants:
                list->SetGraphicsRoot32BitConstants(param, static_cast<UINT>(arg.data.size()), arg.data.data(), 0);
                break;
            case Root::Table: {
                ID3D12DescriptorHeap* hp = arg.sampler ? sampler_heap.Get() : views.Get();
                const std::size_t size = arg.sampler ? sampler_size : view_size;
                D3D12_GPU_DESCRIPTOR_HANDLE at = hp->GetGPUDescriptorHandleForHeapStart();
                at.ptr += static_cast<UINT64>(arg.index * size);
                list->SetGraphicsRootDescriptorTable(param, at);
                break;
            }
            case Root::View:
                throw Error("root views in a draw");
        }
    }
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT viewport{0, 0, static_cast<float>(w), static_cast<float>(h), 0, 1};
    list->RSSetViewports(1, &viewport);
    D3D12_RECT scissor{0, 0, static_cast<LONG>(w), static_cast<LONG>(h)};
    list->RSSetScissorRects(1, &scissor);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (vb) {
        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), static_cast<UINT>(d.vertices.size() * 4), d.stride};
        list->IASetVertexBuffers(0, 1, &vbv);
    }
    if (d.mesh_groups > 0) {
        ComPtr<ID3D12GraphicsCommandList6> list6;
        check(q.list.As(&list6));
        list6->DispatchMesh(d.mesh_groups, 1, 1);
    } else {
        list->DrawInstanced(3, 1, 0, 0);
    }
    {
        const auto barrier =
            transition(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
    }
    const std::size_t pitch = round_up(std::size_t(w) * 16, 256);
    auto readback = make(dev, D3D12_HEAP_TYPE_READBACK, buffer_desc(pitch * h, D3D12_RESOURCE_FLAG_NONE),
                         D3D12_RESOURCE_STATE_COPY_DEST);
    {
        const auto dst =
            footprint(readback.Get(), DXGI_FORMAT_R32G32B32A32_FLOAT, w, h, static_cast<std::uint32_t>(pitch));
        const auto src = subresource(target.Get());
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    run(dev, q.queue.Get(), list, true);
    void* p = nullptr;
    check(readback->Map(0, nullptr, &p));
    std::vector<std::array<float, 4>> out;
    out.reserve(std::size_t(w) * h);
    for (std::size_t y = 0; y < h; ++y) {
        const auto* row = reinterpret_cast<const std::array<float, 4>*>(static_cast<const std::uint8_t*>(p) + y * pitch);
        out.insert(out.end(), row, row + w);
    }
    readback->Unmap(0, nullptr);
    keep.clear();
    return out;
}

std::vector<std::vector<std::uint32_t>> Host::dispatch(ID3D12RootSignature* root, ID3D12PipelineState* pso,
                                                       const Dispatch& d) const {
    ID3D12Device* dev = device.Get();
    // D3D12 calls with descriptions that outlive them; resources stay alive until the fence says
    // the GPU is done.
    Queue q = direct_queue(dev, pso);
    ID3D12GraphicsCommandList* list = q.list.Get();
    // An upload buffer holding data (sized for a constant buffer).
    auto upload = [&](std::span<const std::uint32_t> data) {
        const std::size_t size = round_up(data.size() * 4, 256);
        auto b = make(dev, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(size, D3D12_RESOURCE_FLAG_NONE),
                      D3D12_RESOURCE_STATE_GENERIC_READ);
        void* p = nullptr;
        check(b->Map(0, nullptr, &p));
        if (!data.empty()) std::memcpy(p, data.data(), data.size() * 4);
        b->Unmap(0, nullptr);
        return b;
    };
    auto views = heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, d.views.size());
    auto samplers = heap(dev, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, d.samplers.size());
    const std::size_t view_size = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const std::size_t sampler_size = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    std::vector<ComPtr<ID3D12Resource>> keep;
    // UAVs: (resource, texture size or none, dwords or texel words, readback buffer).
    struct Uav {
        ComPtr<ID3D12Resource> r;
        std::optional<std::pair<std::uint32_t, std::uint32_t>> size;
        std::uint32_t n = 0;
        ComPtr<ID3D12Resource> rb;
    };
    std::vector<Uav> uavs;
    for (std::size_t i = 0; i < d.views.size(); ++i) {
        const View& v = d.views[i];
        D3D12_CPU_DESCRIPTOR_HANDLE at = views->GetCPUDescriptorHandleForHeapStart();
        at.ptr += i * view_size;
        switch (v.kind) {
            case View::Texture:
            case View::Cube: {
                const std::uint32_t w = v.w, h = v.h;
                const std::uint32_t faces = v.kind == View::Cube ? 6 : 1;
                const std::uint16_t mips = v.kind == View::Cube ? v.mips : 1;
                std::span<const std::array<float, 4>> texels = v.texels;
                auto desc = texture_desc(w, h, D3D12_RESOURCE_FLAG_NONE, DXGI_FORMAT_R32G32B32A32_FLOAT);
                desc.DepthOrArraySize = static_cast<UINT16>(faces);
                desc.MipLevels = mips;
                auto tex = make(dev, D3D12_HEAP_TYPE_DEFAULT, desc, D3D12_RESOURCE_STATE_COPY_DEST);
                for (std::uint32_t sub = 0; sub < std::uint32_t(mips) * faces; ++sub) {
                    const std::uint32_t mip = sub % std::uint32_t(mips);
                    const std::uint32_t mw = std::max<std::uint32_t>(w >> mip, 1);
                    const std::uint32_t mh = std::max<std::uint32_t>(h >> mip, 1);
                    const std::size_t pitch = round_up(std::size_t(mw) * 16, 256);
                    auto up = make(dev, D3D12_HEAP_TYPE_UPLOAD, buffer_desc(pitch * mh, D3D12_RESOURCE_FLAG_NONE),
                                   D3D12_RESOURCE_STATE_GENERIC_READ);
                    const std::size_t count = std::size_t(mw) * mh;
                    if (texels.size() < count) bitcode::bug("host: view texels too short");
                    fill_rows(up.Get(), texels.data(), mw, mh, pitch);
                    texels = texels.subspan(count);
                    const auto dst = subresource(tex.Get(), sub);
                    const auto src = footprint(up.Get(), DXGI_FORMAT_R32G32B32A32_FLOAT, mw, mh,
                                               static_cast<std::uint32_t>(pitch));
                    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                    keep.push_back(std::move(up));
                }
                const auto barrier = transition(tex.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                list->ResourceBarrier(1, &barrier);
                D3D12_SHADER_RESOURCE_VIEW_DESC cube{};
                cube.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                cube.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                cube.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                cube.TextureCube.MipLevels = mips;
                dev->CreateShaderResourceView(tex.Get(), faces == 6 ? &cube : nullptr, at);
                keep.push_back(std::move(tex));
                break;
            }
            case View::RawUav:
            case View::HalfUav: {
                std::uint32_t n, elements;
                DXGI_FORMAT format;
                D3D12_BUFFER_UAV_FLAGS flags;
                if (v.kind == View::HalfUav) {
                    n = 2 * v.n;
                    format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    elements = v.n;
                    flags = D3D12_BUFFER_UAV_FLAG_NONE;
                } else {
                    n = v.n;
                    format = DXGI_FORMAT_R32_TYPELESS;
                    elements = v.n;
                    flags = D3D12_BUFFER_UAV_FLAG_RAW;
                }
                const std::uint64_t bytes = std::uint64_t(n) * 4;
                auto b = make(dev, D3D12_HEAP_TYPE_DEFAULT,
                              buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                              D3D12_RESOURCE_STATE_COMMON);
                D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
                desc.Format = format;
                desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                desc.Buffer.NumElements = elements;
                desc.Buffer.Flags = flags;
                dev->CreateUnorderedAccessView(b.Get(), nullptr, &desc, at);
                auto rb = make(dev, D3D12_HEAP_TYPE_READBACK, buffer_desc(bytes, D3D12_RESOURCE_FLAG_NONE),
                               D3D12_RESOURCE_STATE_COPY_DEST);
                uavs.push_back({std::move(b), std::nullopt, n, std::move(rb)});
                break;
            }
            case View::TextureUav:
            case View::UintTextureUav: {
                const bool uint = v.kind == View::UintTextureUav;
                auto t = make(dev, D3D12_HEAP_TYPE_DEFAULT,
                              texture_desc(v.w, v.h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           uint ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R32G32B32A32_FLOAT),
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                dev->CreateUnorderedAccessView(t.Get(), nullptr, nullptr, at);
                const std::uint32_t pitch = static_cast<std::uint32_t>(round_up(std::size_t(v.w) * (uint ? 4 : 16), 256));
                auto rb = make(dev, D3D12_HEAP_TYPE_READBACK,
                               buffer_desc(std::uint64_t(pitch * v.h), D3D12_RESOURCE_FLAG_NONE),
                               D3D12_RESOURCE_STATE_COPY_DEST);
                uavs.push_back({std::move(t), std::pair{v.w, v.h}, uint ? 1u : 4u, std::move(rb)});
                break;
            }
            case View::Cbv: {
                auto b = upload(v.data);
                D3D12_CONSTANT_BUFFER_VIEW_DESC desc{b->GetGPUVirtualAddress(),
                                                     static_cast<UINT>(round_up(v.data.size() * 4, 256))};
                dev->CreateConstantBufferView(&desc, at);
                keep.push_back(std::move(b));
                break;
            }
            case View::Buffer: {
                auto b = upload(v.data);
                DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
                std::size_t elements = 0;
                std::uint32_t stride = 0;
                D3D12_BUFFER_SRV_FLAGS flags = D3D12_BUFFER_SRV_FLAG_NONE;
                switch (v.stride.kind) {
                    case Stride::Raw:
                        format = DXGI_FORMAT_R32_TYPELESS;
                        elements = v.data.size();
                        flags = D3D12_BUFFER_SRV_FLAG_RAW;
                        break;
                    case Stride::Typed:
                        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
                        elements = v.data.size() / 4;
                        break;
                    case Stride::Structured:
                        if (v.stride.bytes == 0) bitcode::bug("host: structured stride 0");
                        format = DXGI_FORMAT_UNKNOWN;
                        elements = v.data.size() * 4 / v.stride.bytes;
                        stride = v.stride.bytes;
                        break;
                    case Stride::Format:
                        if (v.stride.bytes == 0) bitcode::bug("host: format stride 0");
                        format = v.stride.format;
                        elements = v.data.size() * 4 / v.stride.bytes;
                        break;
                }
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                desc.Format = format;
                desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                desc.Buffer.FirstElement = 0;
                desc.Buffer.NumElements = static_cast<UINT>(elements);
                desc.Buffer.StructureByteStride = stride;
                desc.Buffer.Flags = flags;
                dev->CreateShaderResourceView(b.Get(), &desc, at);
                keep.push_back(std::move(b));
                break;
            }
        }
    }
    for (std::size_t i = 0; i < d.samplers.size(); ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE at = samplers->GetCPUDescriptorHandleForHeapStart();
        at.ptr += i * sampler_size;
        dev->CreateSampler(&d.samplers[i], at);
    }
    ID3D12DescriptorHeap* heaps[] = {views.Get(), samplers.Get()};
    list->SetDescriptorHeaps(2, heaps);
    list->SetComputeRootSignature(root);
    for (const auto& [param, arg] : d.roots) {
        switch (arg.kind) {
            case Root::Constants:
                list->SetComputeRoot32BitConstants(param, static_cast<UINT>(arg.data.size()), arg.data.data(), 0);
                break;
            case Root::Table: {
                ID3D12DescriptorHeap* hp = arg.sampler ? samplers.Get() : views.Get();
                const std::size_t size = arg.sampler ? sampler_size : view_size;
                D3D12_GPU_DESCRIPTOR_HANDLE at = hp->GetGPUDescriptorHandleForHeapStart();
                at.ptr += static_cast<UINT64>(arg.index * size);
                list->SetComputeRootDescriptorTable(param, at);
                break;
            }
            case Root::View: {
                auto b = upload(arg.data);
                const D3D12_GPU_VIRTUAL_ADDRESS va = b->GetGPUVirtualAddress();
                if (arg.srv)
                    list->SetComputeRootShaderResourceView(param, va);
                else
                    list->SetComputeRootConstantBufferView(param, va);
                keep.push_back(std::move(b));
                break;
            }
        }
    }
    list->Dispatch(d.groups[0], d.groups[1], d.groups[2]);
    for (const Uav& u : uavs) {
        // (Buffers were promoted from the common state by the dispatch.)
        const auto barrier =
            transition(u.r.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        if (u.size) {
            const auto [w, h] = *u.size;
            const auto dst = footprint(u.rb.Get(), u.n == 1 ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R32G32B32A32_FLOAT,
                                       w, h, static_cast<std::uint32_t>(round_up(std::size_t(w) * 4 * u.n, 256)));
            const auto src = subresource(u.r.Get());
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        } else {
            list->CopyBufferRegion(u.rb.Get(), 0, u.r.Get(), 0, std::uint64_t(u.n) * 4);
        }
    }
    run(dev, q.queue.Get(), list, false);
    std::vector<std::vector<std::uint32_t>> out;
    for (const Uav& u : uavs) {
        void* p = nullptr;
        check(u.rb->Map(0, nullptr, &p));
        std::vector<std::uint32_t> words;
        if (u.size) {
            const auto [w, h] = *u.size;
            const std::size_t per = u.n;
            const std::size_t pitch = round_up(std::size_t(w) * 4 * per, 256);
            for (std::size_t y = 0; y < h; ++y) {
                const auto* row = reinterpret_cast<const std::uint32_t*>(static_cast<const std::uint8_t*>(p) + y * pitch);
                words.insert(words.end(), row, row + std::size_t(w) * per);
            }
        } else {
            const auto* all = static_cast<const std::uint32_t*>(p);
            words.assign(all, all + u.n);
        }
        u.rb->Unmap(0, nullptr);
        out.push_back(std::move(words));
    }
    keep.clear();
    return out;
}


using container::SigElement;

namespace {

// What the stage declares that a pipeline must match.
std::vector<SigElement> sig(std::span<const std::uint8_t> dxil, dxbc::FourCC fourcc) {
    try {
        const dxbc::Container c = dxbc::Container::parse(dxil);
        const dxbc::Part* p = c.part(fourcc);
        if (!p) return {};
        return vs::signature(p->data);
    } catch (const std::exception& e) {
        throw Error(e.what());
    }
}

DXGI_FORMAT input_format(const SigElement& e) {
    const std::size_t i = std::size_t(std::clamp<std::uint8_t>(e.cols, 1, 4)) - 1;
    static constexpr DXGI_FORMAT F[4] = {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32B32_FLOAT,
                                         DXGI_FORMAT_R32G32B32A32_FLOAT};
    static constexpr DXGI_FORMAT S[4] = {DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32G32_SINT, DXGI_FORMAT_R32G32B32_SINT,
                                         DXGI_FORMAT_R32G32B32A32_SINT};
    static constexpr DXGI_FORMAT U[4] = {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32_UINT,
                                         DXGI_FORMAT_R32G32B32A32_UINT};
    switch (e.comp) {
        case 9: return F[i];
        case 4: return S[i];
        default: return U[i];
    }
}

DXGI_FORMAT target_format(const SigElement& e) {
    switch (e.comp) {
        case 9: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case 4: return DXGI_FORMAT_R32G32B32A32_SINT;
        default: return DXGI_FORMAT_R32G32B32A32_UINT;
    }
}

D3D12_SHADER_BYTECODE code(std::span<const std::uint8_t> b) { return {b.data(), b.size()}; }

D3D12_RASTERIZER_DESC rasterizer() {
    D3D12_RASTERIZER_DESC r{};
    r.FillMode = D3D12_FILL_MODE_SOLID;
    r.CullMode = D3D12_CULL_MODE_NONE;
    r.DepthClipEnable = TRUE;
    return r;
}

// The targets, depth and stencil the pixel shader's outputs ask for.
struct Targets {
    D3D12_RT_FORMAT_ARRAY rts{};
    D3D12_DEPTH_STENCIL_DESC depth{};
    DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN;
};

Targets targets(const std::vector<SigElement>& outputs) {
    Targets t;
    for (const SigElement& o : outputs) {
        if (o.kind != 16) continue;
        const std::size_t n = o.row.value_or(0);
        if (n >= 8) bitcode::bug("host: render target index out of bounds");
        t.rts.RTFormats[n] = target_format(o);
        t.rts.NumRenderTargets = std::max<UINT>(t.rts.NumRenderTargets, static_cast<UINT>(n) + 1);
    }
    if (std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind >= 17 && o.kind <= 19; })) {
        t.dsv = DXGI_FORMAT_D32_FLOAT;
        t.depth.DepthEnable = TRUE;
        t.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        t.depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    }
    if (std::any_of(outputs.begin(), outputs.end(), [](const SigElement& o) { return o.kind == 20; })) {
        t.dsv = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        t.depth.StencilEnable = TRUE;
    }
    return t;
}

}  // namespace

Pipeline Host::pipeline(const Stage& vs, const Stage& ps) const {
    auto root = root_signature(vs, ps);
    auto pso = pipeline_with(root.Get(), vs.dxil, ps.dxil);
    return {std::move(root), std::move(pso)};
}

ComPtr<ID3D12PipelineState> Host::pipeline_with(ID3D12RootSignature* root, std::span<const std::uint8_t> vs,
                                                std::span<const std::uint8_t> ps) const {
    return pipeline_with_gs(root, vs, {}, ps);
}

ComPtr<ID3D12PipelineState> Host::pipeline_with_gs(ID3D12RootSignature* root, std::span<const std::uint8_t> vs,
                                                   std::span<const std::uint8_t> gs,
                                                   std::span<const std::uint8_t> ps) const {
    const std::vector<SigElement> inputs = sig(vs, dxbc::FourCC::ISG1);
    std::vector<D3D12_INPUT_ELEMENT_DESC> layout;
    for (const SigElement& e : inputs) {
        if (e.kind != 0) continue;
        D3D12_INPUT_ELEMENT_DESC l{};
        l.SemanticName = e.name.c_str();
        l.SemanticIndex = e.index;
        l.Format = input_format(e);
        l.InputSlot = 0;
        l.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT;
        l.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
        l.InstanceDataStepRate = 0;
        layout.push_back(l);
    }
    const std::vector<SigElement> outputs = sig(ps, dxbc::FourCC::OSG1);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    // A borrowed pointer the descriptor never releases.
    desc.pRootSignature = root;
    desc.VS = code(vs);
    desc.PS = code(ps);
    if (!gs.empty()) desc.GS = code(gs);
    desc.SampleMask = UINT32_MAX;
    desc.RasterizerState = rasterizer();
    desc.InputLayout = {layout.data(), static_cast<UINT>(layout.size())};
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.SampleDesc = {1, 0};
    for (auto& rt : desc.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    const Targets t = targets(outputs);
    for (UINT i = 0; i < 8; ++i) desc.RTVFormats[i] = t.rts.RTFormats[i];
    desc.NumRenderTargets = t.rts.NumRenderTargets;
    desc.DSVFormat = t.dsv;
    desc.DepthStencilState = t.depth;
    ComPtr<ID3D12PipelineState> pso;
    HRESULT hr = device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) throw Error(message(hr));
    return pso;
}

Pipeline Host::mesh_pipeline(const Stage& ms, const Stage& ps) const {
    auto root = root_signature_for({{ms, D3D12_SHADER_VISIBILITY_MESH}, {ps, D3D12_SHADER_VISIBILITY_PIXEL}});
    auto pso = mesh_pipeline_with(root.Get(), ms.dxil, ps.dxil);
    return {std::move(root), std::move(pso)};
}

namespace {
// A stream subobject: its type, then its description, pointer-aligned (padding intended).
#pragma warning(push)
#pragma warning(disable : 4324)
template <class T>
struct alignas(void*) Sub {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
    T value;
};
#pragma warning(pop)
}  // namespace

ComPtr<ID3D12PipelineState> Host::mesh_pipeline_with(ID3D12RootSignature* root, std::span<const std::uint8_t> ms,
                                                     std::span<const std::uint8_t> ps) const {
    struct Stream {
        Sub<ID3D12RootSignature*> root;
        Sub<D3D12_SHADER_BYTECODE> ms;
        Sub<D3D12_SHADER_BYTECODE> ps;
        Sub<D3D12_BLEND_DESC> blend;
        Sub<UINT> mask;
        Sub<D3D12_RASTERIZER_DESC> raster;
        Sub<D3D12_DEPTH_STENCIL_DESC> depth;
        Sub<D3D12_RT_FORMAT_ARRAY> targets;
        Sub<DXGI_FORMAT> dsv;
        Sub<DXGI_SAMPLE_DESC> samples;
    };
    const std::vector<SigElement> outputs = sig(ps, dxbc::FourCC::OSG1);
    D3D12_BLEND_DESC blend{};
    for (auto& rt : blend.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    const Targets t = targets(outputs);
    Stream stream{
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(ms)},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(ps)},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, UINT32_MAX},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, rasterizer()},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, t.depth},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, t.rts},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, t.dsv},
        {D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC{1, 0}},
    };
    D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(Stream), &stream};
    ComPtr<ID3D12Device2> device2;
    HRESULT hr = device.As(&device2);
    if (FAILED(hr)) throw Error(message(hr));
    // The stream points at live data and outlives the call.
    ComPtr<ID3D12PipelineState> pso;
    hr = device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) throw Error(message(hr));
    return pso;
}

namespace {

std::vector<std::uint8_t> stub(bool pixel, std::vector<SigElement> outputs) {
    shader::Shader s(pixel);
    const std::size_t b = s.entry;
    for (std::size_t id = 0; id < outputs.size(); ++id) {
        const SigElement& e = outputs[id];
        for (std::uint8_t col = 0; col < e.cols; ++col) {
            // Allocate zero before the id and column operands.
            const char* suffix;
            bitcode::V zero;
            if (e.comp == 9) {
                suffix = ".f32";
                zero = s.m.f32(0);
            } else {
                suffix = ".i32";
                zero = s.m.i32(0);
            }
            const bitcode::V a0 = s.m.i32(static_cast<std::uint32_t>(id));
            const bitcode::V a1 = s.m.i32(0);
            const bitcode::V a2 = s.m.i8(col);
            const bitcode::V args[] = {a0, a1, a2, zero};
            s.op(b, std::string("storeOutput") + suffix, bitcode::Attr::NoUnwind, s.t.void_, 5, args);
        }
    }
    for (SigElement& e : outputs) e.usage = static_cast<std::uint8_t>((1u << e.cols) - 1);
    s.outputs = std::move(outputs);
    s.m.terminate(b, bitcode::Term::ret());
    return s.finish();
}

}  // namespace

std::vector<std::uint8_t> stub_vs_dxil(std::span<const std::uint8_t> ps) {
    std::vector<SigElement> outputs;
    for (SigElement& e : sig(ps, dxbc::FourCC::ISG1))
        if (e.kind == 0 || e.kind == 3 || e.kind == 6 || e.kind == 7) outputs.push_back(std::move(e));
    if (std::none_of(outputs.begin(), outputs.end(), [](const SigElement& e) { return e.kind == 3; })) {
        std::uint32_t row = 0;
        for (const SigElement& e : outputs)
            if (e.row) row = std::max(row, *e.row + 1);
        SigElement pos;
        pos.name = "SV_Position";
        pos.index = 0;
        pos.kind = 3;
        pos.comp = 9;
        pos.interp = 4;
        pos.row = row;
        pos.col = 0;
        pos.cols = 4;
        pos.usage = 0;
        pos.half = false;
        outputs.push_back(std::move(pos));
    }
    return stub(false, std::move(outputs));
}

std::vector<std::uint8_t> stub_ps_dxil() {
    SigElement target;
    target.name = "SV_Target";
    target.index = 0;
    target.kind = 16;
    target.comp = 9;
    target.interp = 0;
    target.row = 0;
    target.col = 0;
    target.cols = 4;
    target.usage = 0;
    target.half = false;
    return stub(true, {target});
}

#if X2W_VS_HOST_STUBS

namespace {
HostShader stub_shader(bool pixel, std::vector<std::uint8_t> dxil) {
    HostShader h;
    h.stage = pixel ? native::HwStage::Ps : native::HwStage::Vs;
    h.dxil = std::move(dxil);
    return h;
}
}  // namespace

HostShader stub_vs(const HostShader& ps) { return stub_shader(false, stub_vs_dxil(ps.dxil)); }

HostShader stub_ps() { return stub_shader(true, stub_ps_dxil()); }

#endif  // X2W_VS_HOST_STUBS

}  // namespace x2w::vs::host
