// Each case makes an invalid view description, puts it through the check, creates what the check lets through (or a null view, as the mirror does)
// and requires the device to stay. --mutate skips the checks: the device must then be removed (negative control).
//
//   startup_gpu_heap_fit_check <warp|hardware> [--mutate]
// Exit 0 pass, 1 a check failed, 77 no adapter, other values a setup step.
#include "startup_stubs/gpu_heap.inl"
#include "startup_stubs/gpu_read_states.h"
#include <d3d12sdklayers.h>

#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
void print_line(const char* text) { std::printf("  log: %s\n", text); }
ComPtr<ID3D12Resource> make(ID3D12Device* device, D3D12_RESOURCE_DIMENSION dim, UINT64 width, UINT height, UINT16 depth,
                            UINT16 mips, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Layout = dim == D3D12_RESOURCE_DIMENSION_BUFFER ? D3D12_TEXTURE_LAYOUT_ROW_MAJOR : D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> r;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    return r;
}
bool srgb_uav_roundtrip(ID3D12Device* device, ID3D12DescriptorHeap* visible)
{
    D3D12_RESOURCE_DESC original{};
    original.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    original.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    original.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const DXGI_FORMAT backing = x2w_srgb::backing_format(original);
    if (backing != DXGI_FORMAT_R8G8B8A8_TYPELESS) return false;
    auto texture = make(device, original.Dimension, 8, 8, 1, 1, backing, original.Flags);
    if (!texture) return false;
    const auto rd = texture->GetDesc();
    auto srv = x2w_srgb::srv(rd, original.Format);
    auto uav = x2w_srgb::uav(rd, original.Format);
    auto rtv = x2w_srgb::rtv(rd, original.Format);
    uav.Format = original.Format;  // an explicit sRGB UAV requests format normalization
    if (!gpu_mirror::fits(texture.Get(), &uav.Format, true) || uav.Format != DXGI_FORMAT_R8G8B8A8_UNORM ||
        !gpu_mirror::fits(texture.Get(), &srv.Format, false) || srv.Format != original.Format ||
        rtv.Format != original.Format || x2w_srgb::uav(rd, original.Format).Format != uav.Format) return false;
    ComPtr<ID3D12DescriptorHeap> cpu, targets;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2};
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&cpu)))) return false;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&targets)))) return false;
    auto hc = cpu->GetCPUDescriptorHandleForHeapStart();
    auto hs = hc;
    hs.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    device->CreateShaderResourceView(texture.Get(), &srv, hs);
    device->CreateRenderTargetView(texture.Get(), &rtv, targets->GetCPUDescriptorHandleForHeapStart());
    device->CreateUnorderedAccessView(texture.Get(), nullptr, &uav, hc);
    device->CopyDescriptorsSimple(1, visible->GetCPUDescriptorHandleForHeapStart(), hc, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (FAILED(device->GetDeviceRemovedReason())) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&rd, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    D3D12_RESOURCE_DESC bd{};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = bytes; bd.Height = bd.DepthOrArraySize = bd.MipLevels = 1;
    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES readback_props{D3D12_HEAP_TYPE_READBACK};
    ComPtr<ID3D12Resource> readback;
    if (FAILED(device->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback)))) return false;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC q{};
    if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {texture.Get(), 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    list->ResourceBarrier(1, &barrier);
    list->SetDescriptorHeaps(1, &visible);
    const FLOAT color[4]{0, 1, 0, 1};
    list->ClearUnorderedAccessViewFloat(visible->GetGPUDescriptorHandleForHeapStart(), hc, texture.Get(), color, 0, nullptr);
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION dest{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dest.PlacedFootprint = footprint;
    D3D12_TEXTURE_COPY_LOCATION src{texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dest, 0, 0, 0, &src, nullptr);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[]{list.Get()};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.Get(), 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr)) ||
        FAILED(device->GetDeviceRemovedReason())) return false;
    UINT* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    bool pixels_match = true;
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x)
            pixels_match &= mapped[y * footprint.Footprint.RowPitch / sizeof(UINT) + x] == 0xff00ff00u;
    readback->Unmap(0, nullptr);
    return pixels_match;
}
bool queue_read_states(ID3D12Device* device)
{
    auto texture = make(device, D3D12_RESOURCE_DIMENSION_TEXTURE2D, 16, 16, 1, 1, DXGI_FORMAT_R32_UINT,
                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto buffer = make(device, D3D12_RESOURCE_DIMENSION_BUFFER, 4096, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12InfoQueue> info;
    ComPtr<ID3D12Fence> fence;
    if (!texture || !buffer || FAILED(device->QueryInterface(IID_PPV_ARGS(&info))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
    info->ClearStoredMessages();
    UINT64 sequence = 0;
    for (auto type : {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE}) {
        const auto texture_read = x2w_gpu::texture_read_state(type);
        const auto buffer_read = x2w_gpu::buffer_read_state(type);
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = type;
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
        const auto transition = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER b{};
            b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
            list->ResourceBarrier(1, &b);
        };
        transition(texture.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        transition(texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, texture_read);
        transition(texture.Get(), texture_read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        transition(buffer.Get(), D3D12_RESOURCE_STATE_COMMON, buffer_read);
        transition(buffer.Get(), buffer_read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.Get(), ++sequence)) || FAILED(fence->SetEventOnCompletion(sequence, nullptr)) ||
            FAILED(device->GetDeviceRemovedReason())) return false;
    }
    for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T size = 0;
        info->GetMessage(i, nullptr, &size);
        std::vector<unsigned char> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(info->GetMessage(i, message, &size)) && message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
            std::printf("D3D12: %s\n", message->pDescription);
            return false;
        }
    }
    return true;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    const bool mutate = argc > 2 && std::strcmp(argv[2], "--mutate") == 0;
    const bool states = argc > 2 && std::strcmp(argv[2], "--read-states") == 0;
    if (states) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
        debug->EnableDebugLayer();
    }
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
    if (warp ? FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))
             : FAILED(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter))))
        return 77;
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) return 77;
    if (states) {
        check(queue_read_states(device.Get()), "direct/compute tracked upload, shader read, UAV write and return to COMMON");
        return failures ? 1 : 0;
    }
    gpu_mirror::fit_log() = print_line;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)))) return 3;
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto at = [&](UINT i) { auto c = heap->GetCPUDescriptorHandleForHeapStart(); c.ptr += SIZE_T{i} * step; return c; };

    constexpr UINT64 bytes = 1 << 16;
    auto buffer = make(device.Get(), D3D12_RESOURCE_DIMENSION_BUFFER, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN,
                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto tex2d = make(device.Get(), D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, 1, 2, DXGI_FORMAT_R32_UINT,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto tex3d = make(device.Get(), D3D12_RESOURCE_DIMENSION_TEXTURE3D, 64, 64, 8, 2, DXGI_FORMAT_R32_UINT,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    if (!buffer || !tex2d || !tex3d) return 4;
    const auto removed = [&] { return device->GetDeviceRemovedReason() != S_OK; };
    const auto null_srv = [&](UINT i) { gpu_mirror::null_buffer_view(device.Get(), at(i)); };

    if (!mutate) check(srgb_uav_roundtrip(device.Get(), heap.Get()), "sRGB SRV and RTV, linear UAV writes all 64 texels on a typeless backing");
    if (!mutate) {
        auto depth = make(device.Get(), D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, 1, 1, DXGI_FORMAT_R24G8_TYPELESS,
                          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        gpu_mirror::Env env{};
        env.device = device.Get();
        auto host = gpu_mirror::HostHeap::create(env, false, 1, {64, 0}, 0, 32, 4);
        if (!depth || !host) return 4;
        gpu_mirror::View view{};
        view.kind = gpu_mirror::Kind::unordered;
        view.has_desc = true;
        view.uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        view.uav.Format = DXGI_FORMAT_R32_UINT;
        view.resource = tex2d.Get();
        host->make_at(1, view);
        check(host->kept_at(1) == tex2d.Get(), "writable texture slot targets materialized texture");
        view.resource = depth.Get();
        host->make_at(1, view);
        check(!host->kept_at(1), "reusing a slot for a refused depth UAV leaves no resource to transition");
    }
    // 1. A raw SRV 16 KiB past the buffer's end, from 1 KiB in: cut to the end.
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = 256;
        d.Buffer.NumElements = static_cast<UINT>((bytes + (16 << 10)) / 4) - 256;
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        const bool ok = mutate || gpu_mirror::fit_buffer(buffer.Get(), d, "case 1");
        if (!mutate) check(ok && d.Buffer.NumElements == bytes / 4 - 256, "raw SRV past the end: cut to the buffer's end");
        if (ok) gpu_view_trace::MirrorCall{device.Get(), "fit control"}.srv(buffer.Get(), &d, at(0));
        check(!removed(), "raw SRV past the end: device kept");
    }
    // 2. A structured UAV that starts past the end: refused (null view).
    if (!removed()) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = bytes / 16 + 4;
        d.Buffer.NumElements = 8;
        d.Buffer.StructureByteStride = 16;
        const bool ok = mutate || gpu_mirror::fit_buffer(buffer.Get(), d, "case 2");
        if (!mutate) check(!ok, "structured UAV starting past the end: refused");
        if (ok) device->CreateUnorderedAccessView(buffer.Get(), nullptr, &d, at(1));
        else null_srv(1);
        check(!removed(), "structured UAV starting past the end: device kept");
    }
    // 3. A constant view of 64 KiB + 256 and one off 256 bytes: refused.
    if (!removed()) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC big{buffer->GetGPUVirtualAddress(), (64u << 10) + 256};
        D3D12_CONSTANT_BUFFER_VIEW_DESC off{buffer->GetGPUVirtualAddress() + 16, 256};
        const bool ok_big = mutate || gpu_mirror::fit_cbv(big, "case 3");
        const bool ok_off = mutate || gpu_mirror::fit_cbv(off, "case 3");
        if (!mutate) check(!ok_big && !ok_off, "constant views over 64 KiB and off 256 bytes: refused");
        if (ok_big) device->CreateConstantBufferView(&big, at(2));
        if (ok_off) device->CreateConstantBufferView(&off, at(3));
        check(!removed(), "constant views: device kept");
    }
    // 4. A 2D texture SRV of mips 1..4 of a 2-mip texture: cut to mip 1; a UAV of mip 3: refused.
    if (!removed()) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s{};
        s.Format = DXGI_FORMAT_R32_UINT;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Texture2D.MostDetailedMip = 1;
        s.Texture2D.MipLevels = 4;
        const bool ok_s = mutate || gpu_mirror::fit_srv(tex2d.Get(), &s, "case 4");
        if (!mutate) check(ok_s && s.Texture2D.MipLevels == 1, "texture SRV of mips past the texture: cut");
        if (ok_s) device->CreateShaderResourceView(tex2d.Get(), &s, at(4));
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_UINT;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        u.Texture2D.MipSlice = 3;
        const bool ok_u = mutate || gpu_mirror::fit_uav(tex2d.Get(), &u, "case 4");
        if (!mutate) check(!ok_u, "texture UAV of a mip the texture lacks: refused");
        if (ok_u) device->CreateUnorderedAccessView(tex2d.Get(), nullptr, &u, at(5));
        check(!removed(), "texture views: device kept");
    }
    // 5. A 3D UAV of mip 1 (depth 4) with 8 W slices: cut to 4.
    if (!removed()) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_UINT;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        u.Texture3D = {1, 0, 8};
        const bool ok = mutate || gpu_mirror::fit_uav(tex3d.Get(), &u, "case 5");
        if (!mutate) check(ok && u.Texture3D.WSize == 4, "3D UAV with W slices past its mip's depth: cut");
        if (ok) device->CreateUnorderedAccessView(tex3d.Get(), nullptr, &u, at(6));
        check(!removed(), "3D UAV: device kept");
    }
    // 6. Descriptions that fit are left as they are.
    if (!removed()) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = bytes / 4;
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        D3D12_UNORDERED_ACCESS_VIEW_DESC u{};
        u.Format = DXGI_FORMAT_R32_UINT;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        u.Texture3D = {0, 0, ~0u};
        const D3D12_CONSTANT_BUFFER_VIEW_DESC c{buffer->GetGPUVirtualAddress(), 256};
        const bool kept = gpu_mirror::fit_buffer(buffer.Get(), d, "case 6") && d.Buffer.NumElements == bytes / 4 &&
                          gpu_mirror::fit_uav(tex3d.Get(), &u, "case 6") && u.Texture3D.WSize == ~0u &&
                          gpu_mirror::fit_cbv(c, "case 6");
        if (!mutate) check(kept, "descriptions that fit: unchanged");
    }
    std::printf("cut %llu, refused %llu\n", static_cast<unsigned long long>(gpu_mirror::counted(gpu_mirror::c_fit_cut)),
                static_cast<unsigned long long>(gpu_mirror::counted(gpu_mirror::c_fit_refused)));
    if (mutate && removed()) std::printf("--mutate: the device was removed without the checks (expected for the control)\n");
    if (failures) std::printf("startup_gpu_heap_fit_check: %d failures\n", failures);
    else std::printf("startup_gpu_heap_fit_check: pass\n");
    return failures ? 1 : 0;
}
