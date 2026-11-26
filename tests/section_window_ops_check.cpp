#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT64 page = 65536, bytes = 4 * page;

void drain(ID3D12Device* device)
{
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&queue)))) return;
    for (UINT64 i = 0; i < queue->GetNumStoredMessages() && i < 16; ++i) {
        SIZE_T size = 0;
        queue->GetMessage(i, nullptr, &size);
        std::string storage(size, '\0');
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(queue->GetMessage(i, m, &size)) && m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
            std::printf("debug sev=%d id=%d %s\n", m->Severity, m->ID, m->pDescription);
    }
    queue->ClearStoredMessages();
}
D3D12_RESOURCE_DESC buffer_desc(UINT64 width, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = width;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}
ComPtr<ID3D12Resource> committed(ID3D12Device* device, D3D12_HEAP_TYPE type, UINT64 width, D3D12_RESOURCE_STATES state,
                                 D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    const D3D12_HEAP_PROPERTIES heap{type};
    const D3D12_RESOURCE_DESC desc = buffer_desc(width, flags);
    ComPtr<ID3D12Resource> r;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}
const char* const shader = R"(
RWByteAddressBuffer u : register(u0);
ByteAddressBuffer t : register(t0);
cbuffer C : register(b0) { uint4 c; };
[numthreads(1, 1, 1)] void main() { u.Store(0, t.Load(0) + c.x + 1); }
)";
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    const std::string op = argv[2];
    const bool debug = !(argc > 3 && std::strcmp(argv[3], "nodebug") == 0);
    if (debug) {
        ComPtr<ID3D12Debug> d;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&d)))) d->EnableDebugLayer();
    }
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
    if (warp ? FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))
             : FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))))
        return 77;
    ComPtr<ID3D12Device3> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 77;
    std::printf("op %s on %s, debug layer %s\n", op.c_str(), argv[1], debug ? "on" : "off");

    // The window: as section_va makes it.
    const HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(bytes), nullptr);
    auto* view = section ? static_cast<unsigned*>(MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes)) : nullptr;
    if (!view) return 3;
    view[0] = 41;
    ComPtr<ID3D12Heap> heap;
    if (FAILED(device->OpenExistingHeapFromFileMapping(section, IID_PPV_ARGS(&heap)))) return 4;
    std::printf("heap flags 0x%x\n", heap->GetDesc().Flags);
    const D3D12_RESOURCE_DESC wdesc =
        buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
    ComPtr<ID3D12Resource> window;
    if (FAILED(device->CreatePlacedResource(heap.Get(), 0, &wdesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&window)))) return 5;

    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC qdesc{};
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return 6;
    auto plain = committed(device.Get(), D3D12_HEAP_TYPE_DEFAULT, bytes, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!plain) return 7;

    // A compute pipeline: root CBV b0, SRV t0, UAV u0, or one table (u0, t0) for the table ops.
    const bool table = op == "table_uav" || op == "table_srv";
    ComPtr<ID3DBlob> cs, error, sig;
    if (FAILED(D3DCompile(shader, std::strlen(shader), nullptr, nullptr, nullptr, "main", "cs_5_1", 0, 0, &cs, &error))) return 8;
    D3D12_DESCRIPTOR_RANGE ranges[2] = {{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0}, {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 1}};
    D3D12_ROOT_PARAMETER params[3]{};
    UINT nparams = 0;
    params[nparams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[nparams++].Descriptor = {0, 0};
    if (table) {
        params[nparams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[nparams++].DescriptorTable = {2, ranges};
    } else {
        params[nparams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[nparams++].Descriptor = {0, 0};
        params[nparams].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[nparams++].Descriptor = {0, 0};
    }
    const D3D12_ROOT_SIGNATURE_DESC rsd{nparams, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &error)) ||
        FAILED(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&root))))
        return 9;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pdesc{};
    pdesc.pRootSignature = root.Get();
    pdesc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateComputePipelineState(&pdesc, IID_PPV_ARGS(&pso)))) return 10;
    ComPtr<ID3D12DescriptorHeap> gpu_heap, cpu_heap;
    const D3D12_DESCRIPTOR_HEAP_DESC hv{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    const D3D12_DESCRIPTOR_HEAP_DESC hc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (FAILED(device->CreateDescriptorHeap(&hv, IID_PPV_ARGS(&gpu_heap))) || FAILED(device->CreateDescriptorHeap(&hc, IID_PPV_ARGS(&cpu_heap))))
        return 11;
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu_at = [&](ID3D12DescriptorHeap* h, UINT i) { auto c = h->GetCPUDescriptorHandleForHeapStart(); c.ptr += i * step; return c; };
    auto gpu_at = [&](UINT i) { auto g = gpu_heap->GetGPUDescriptorHandleForHeapStart(); g.ptr += std::uint64_t{i} * step; return g; };

    ID3D12Resource* uav_target = op == "root_uav" || op == "table_uav" ? window.Get() : plain.Get();
    ID3D12Resource* srv_source = op == "root_srv" || op == "table_srv" ? window.Get() : plain.Get();
    ID3D12Resource* cbv_source = op == "root_cbv" ? window.Get() : plain.Get();
    if (op == "copy_in") {
        list->CopyBufferRegion(window.Get(), 0, plain.Get(), 0, page);
    } else if (op == "copy_out") {
        list->CopyBufferRegion(plain.Get(), 0, window.Get(), 0, page);
    } else if (op == "clear_uav") {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_UINT;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = static_cast<UINT>(bytes / 4);
        device->CreateUnorderedAccessView(window.Get(), nullptr, &u, cpu_at(gpu_heap.Get(), 0));
        device->CreateUnorderedAccessView(window.Get(), nullptr, &u, cpu_at(cpu_heap.Get(), 0));
        ID3D12DescriptorHeap* hs[] = {gpu_heap.Get()};
        list->SetDescriptorHeaps(1, hs);
        const UINT values[4] = {7, 7, 7, 7};
        list->ClearUnorderedAccessViewUint(gpu_at(0), cpu_at(cpu_heap.Get(), 0), window.Get(), values, 0, nullptr);
    } else if (op == "barrier_uav") {
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_UAV};
        b.UAV.pResource = window.Get();
        list->ResourceBarrier(1, &b);
    } else if (op == "transition") {
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
        b.Transition = {window.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
        list->ResourceBarrier(1, &b);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        list->ResourceBarrier(1, &b);
    } else if (op == "root_uav" || op == "root_srv" || op == "root_cbv" || table) {
        list->SetComputeRootSignature(root.Get());
        list->SetPipelineState(pso.Get());
        list->SetComputeRootConstantBufferView(0, cbv_source->GetGPUVirtualAddress());
        if (table) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
            u.Format = DXGI_FORMAT_R32_TYPELESS;
            u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            u.Buffer.NumElements = static_cast<UINT>(bytes / 4);
            u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device->CreateUnorderedAccessView(uav_target, nullptr, &u, cpu_at(gpu_heap.Get(), 0));
            D3D12_SHADER_RESOURCE_VIEW_DESC s{};
            s.Format = DXGI_FORMAT_R32_TYPELESS;
            s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.Buffer.NumElements = static_cast<UINT>(bytes / 4);
            s.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            device->CreateShaderResourceView(srv_source, &s, cpu_at(gpu_heap.Get(), 1));
            ID3D12DescriptorHeap* hs[] = {gpu_heap.Get()};
            list->SetDescriptorHeaps(1, hs);
            list->SetComputeRootSignature(root.Get());
            list->SetComputeRootDescriptorTable(1, gpu_at(0));
        } else {
            list->SetComputeRootShaderResourceView(1, srv_source->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(2, uav_target->GetGPUVirtualAddress());
        }
        list->Dispatch(1, 1, 1);
    } else if (op == "srv_past_end" || op == "uav_past_end" || op == "plain_srv_past_end") {
        // A raw view that runs 64 KiB past the buffer's end, created into a shader-visible heap (no list work).
        ID3D12Resource* target = op == "plain_srv_past_end" ? plain.Get() : window.Get();
        if (op == "uav_past_end") {
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
            u.Format = DXGI_FORMAT_R32_TYPELESS;
            u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            u.Buffer.NumElements = static_cast<UINT>((bytes + page) / 4);
            u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            device->CreateUnorderedAccessView(target, nullptr, &u, cpu_at(gpu_heap.Get(), 0));
        } else {
            D3D12_SHADER_RESOURCE_VIEW_DESC v{};
            v.Format = DXGI_FORMAT_R32_TYPELESS;
            v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.Buffer.NumElements = static_cast<UINT>((bytes + page) / 4);
            v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            device->CreateShaderResourceView(target, &v, cpu_at(gpu_heap.Get(), 0));
        }
        std::printf("after the view: device removed reason 0x%08lx\n", static_cast<unsigned long>(device->GetDeviceRemovedReason()));
        drain(device.Get());
    } else if (op == "cbv_big" || op == "cbv_unaligned" || op == "cbv_past_end" || op == "null_uav_zero" ||
               op == "tex_uav_bad_mip" || op == "tex3d_uav_bad_w" || op == "tex_srv_bad_mip") {
        // Other descriptions the mirror can make, created into a shader-visible heap.
        if (op == "cbv_big" || op == "cbv_unaligned" || op == "cbv_past_end") {
            D3D12_CONSTANT_BUFFER_VIEW_DESC c{plain->GetGPUVirtualAddress(), 256};
            if (op == "cbv_big") c.SizeInBytes = 65536 + 256;
            if (op == "cbv_unaligned") c.BufferLocation += 16;
            if (op == "cbv_past_end") c = {plain->GetGPUVirtualAddress() + bytes - 256, 1024};
            device->CreateConstantBufferView(&c, cpu_at(gpu_heap.Get(), 0));
        } else if (op == "null_uav_zero") {
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
            u.Format = DXGI_FORMAT_R32_UINT;
            u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            device->CreateUnorderedAccessView(nullptr, nullptr, &u, cpu_at(gpu_heap.Get(), 0));
        } else {
            const bool three = op == "tex3d_uav_bad_w";
            D3D12_RESOURCE_DESC t{};
            t.Dimension = three ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            t.Width = 64;
            t.Height = 64;
            t.DepthOrArraySize = three ? 8 : 1;
            t.MipLevels = 2;
            t.Format = DXGI_FORMAT_R32_UINT;
            t.SampleDesc.Count = 1;
            t.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            const D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
            ComPtr<ID3D12Resource> tex;
            if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &t, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&tex))))
                return 12;
            if (op == "tex_srv_bad_mip") {
                D3D12_SHADER_RESOURCE_VIEW_DESC v{};
                v.Format = DXGI_FORMAT_R32_UINT;
                v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                v.Texture2D.MostDetailedMip = 1;
                v.Texture2D.MipLevels = 4;
                device->CreateShaderResourceView(tex.Get(), &v, cpu_at(gpu_heap.Get(), 0));
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
                u.Format = DXGI_FORMAT_R32_UINT;
                if (three) {
                    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                    u.Texture3D = {1, 0, 8};  // mip 1 has depth 4
                } else {
                    u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                    u.Texture2D.MipSlice = 3;
                }
                device->CreateUnorderedAccessView(tex.Get(), nullptr, &u, cpu_at(gpu_heap.Get(), 0));
            }
            std::printf("after the view: device removed reason 0x%08lx\n", static_cast<unsigned long>(device->GetDeviceRemovedReason()));
            drain(device.Get());
            list->Close();
            return device->GetDeviceRemovedReason() == S_OK ? 0 : 1;
        }
        std::printf("after the view: device removed reason 0x%08lx\n", static_cast<unsigned long>(device->GetDeviceRemovedReason()));
        drain(device.Get());
    } else {
        std::printf("unknown op\n");
        return 2;
    }
    const HRESULT closed = list->Close();
    std::printf("Close -> 0x%08lx\n", static_cast<unsigned long>(closed));
    drain(device.Get());
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence.Get(), 1);
    for (const ULONGLONG deadline = GetTickCount64() + 20000; fence->GetCompletedValue() < 1;) {
        if (GetTickCount64() >= deadline || device->GetDeviceRemovedReason() != S_OK) break;
        Sleep(1);
    }
    drain(device.Get());
    const HRESULT removed = device->GetDeviceRemovedReason();
    std::printf("op %s: device removed reason 0x%08lx; window word 0 = %u\n", op.c_str(), static_cast<unsigned long>(removed), view[0]);
    return removed == S_OK ? 0 : 1;
}
