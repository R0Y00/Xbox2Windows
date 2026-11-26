#include "startup_stubs/gpu_depth_twins.h"
#include <dxgi1_6.h>
#include <cstdio>
#include <cstring>
#include <map>

using namespace x2w_depth_twins;
namespace {
ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE kind, D3D12_RESOURCE_STATES initial)
{
    const D3D12_HEAP_PROPERTIES props{kind};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> r;
    device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr, IID_PPV_ARGS(&r));
    return r;
}
bool run(ID3D12Device* device, DXGI_FORMAT family, bool stencil)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 17; desc.Height = 7; desc.DepthOrArraySize = 2; desc.MipLevels = 3;
    desc.Format = family;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> original;
    if (FAILED(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&original)))) return false;
    desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Twins twins;
    if (!twins.create(device, original.Get(), desc)) return false;
    D3D12_UNORDERED_ACCESS_VIEW_DESC wanted{}, chosen{};
    wanted.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
    wanted.Texture2DArray.ArraySize = 2;
    wanted.Texture2DArray.PlaneSlice = stencil ? 1 : 0;
    if (twins.view(&wanted, chosen) != twins.colour.Get() || chosen.Texture2DArray.PlaneSlice) return false;
    if (stencil && (chosen.Format != DXGI_FORMAT_R8_UINT || twins.view(nullptr, chosen) != twins.depth_plane.Get())) return false;

    constexpr UINT n = 6;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout[n]{};
    UINT64 row[n]{}, bytes = 0;
    auto colour = twins.colour->GetDesc();
    device->GetCopyableFootprints(&colour, 0, n, 0, layout, nullptr, row, &bytes);
    auto upload = buffer(device, bytes * 2, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!upload || !readback) return false;
    BYTE* src = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&src)))) return false;
    std::memset(src, 0, static_cast<size_t>(bytes) * 2);
    for (UINT i = 0; i < n; ++i)
        for (UINT y = 0; y < layout[i].Footprint.Height; ++y)
            std::memset(src + layout[i].Offset + y * layout[i].Footprint.RowPitch, 11 + i * 13 + y, static_cast<size_t>(row[i]));
    upload->Unmap(0, nullptr);
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC q{D3D12_COMMAND_LIST_TYPE_COMPUTE};
    if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    std::map<ID3D12Resource*, D3D12_RESOURCE_STATES> states;
    const auto set = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES next) {
        auto& current = states[r];
        if (current == next) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, current, next};
        list->ResourceBarrier(1, &b);
        current = next;
    };
    auto copy = [&](ID3D12Resource* staging, bool write, bool clear) {
        set(twins.colour.Get(), write ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (UINT i = 0; i < n; ++i) {
            D3D12_TEXTURE_COPY_LOCATION host{staging, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            host.PlacedFootprint = layout[i];
            // All inputs are positive patterns, so replacing the source by zero before copying it back
            // catches a missing transfer in either direction.
            if (clear) host.PlacedFootprint.Offset = (bytes + 511) & ~UINT64{511};
            D3D12_TEXTURE_COPY_LOCATION tex{twins.colour.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            tex.SubresourceIndex = i;
            list->CopyTextureRegion(write ? &tex : &host, 0, 0, 0, write ? &host : &tex, nullptr);
        }
    };
    copy(upload.Get(), true, false);
    twins.copy_colour(list.Get(), false, set);
    copy(upload.Get(), true, true);
    twins.copy_colour(list.Get(), true, set);
    copy(readback.Get(), false, false);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[]{list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.Get(), 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr)) ||
        FAILED(device->GetDeviceRemovedReason())) return false;
    BYTE* result = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&result)))) return false;
    bool ok = true;
    for (UINT i = 0; i < n; ++i)
        for (UINT y = 0; y < layout[i].Footprint.Height; ++y)
            for (UINT x = 0; x < row[i]; ++x)
                ok &= result[layout[i].Offset + y * layout[i].Footprint.RowPitch + x] == 11 + i * 13 + y;
    readback->Unmap(0, nullptr);
    return ok;
}
}
int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> f;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) return 77;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (FAILED(warp ? f->EnumWarpAdapter(IID_PPV_ARGS(&adapter))
                    : f->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 77;
    for (auto f : {DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R32G8X24_TYPELESS}) {
        const bool ok = run(device.Get(), f, f == DXGI_FORMAT_R24G8_TYPELESS || f == DXGI_FORMAT_R32G8X24_TYPELESS);
        std::printf("format %d, 3 mips x 2 slices, both copy directions: %s\n", f, ok ? "pass" : "fail");
        if (!ok) return 1;
    }
}
