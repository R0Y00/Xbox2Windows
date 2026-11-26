// WP3 vbind on WARP: the root bindings that gpu_bind.inl makes, with the pure parts it uses
// (src/startup_stubs/gpu_roots.h: fill_user_data, raw_window, raw_elements, constant_view_bytes, find_bounded), run
// against our own ABI shaders (tests/startup_gpu_bind.hlsl, cs_6_6), not title shaders.
//
//   startup_gpu_bind_check bind [--mutate]
//     A table at title slot 10 (user-data dword = its byte offset in console units, 320) whose entry 0 is a raw SRV and
//     entry 1 a CBV the title placed on 16 bytes (made afresh in 256-byte steps); a root SRV at t3 space1 whose title
//     address is 8 bytes past 16 (raw view from the 16 bytes below, the bias in user data); the result through a root
//     UAV at u5 space1, 4 bytes past 16. --mutate: the table dword as index x 64 (a desktop increment), the SRV bias
//     left out, the CBV made in place on the 256 bytes below: the checks must fail.
//   startup_gpu_bounds_check bounds [--mutate]
//     A root UAV into a 256-byte title buffer of a 64 KiB window stores 512 bytes: what lies past the buffer's end (the
//     canaries) must stay; a root UAV into memory no title buffer record holds gets a null view (nothing stored); a
//     title buffer running past its window's end is cut there. --mutate: bounded by the window alone (the runs
//     129/132 hazard): the canaries must be overwritten, so the checks fail.
//
// Exit codes: 0 pass, 1 failed checks, 2 usage, 77 skipped (no WARP, no SM 6.6 or no resource binding tier 3).
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "startup_stubs/gpu_roots.h"
#include "startup_gpu_bind_bind_main.h"
#include "startup_gpu_bind_bounds_main.h"

using Microsoft::WRL::ComPtr;
using namespace x2w_vbind;
using x2w_gpu::GpuUserData;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

constexpr std::uint64_t window_bytes = 64 * 1024;
constexpr std::uint64_t title_read = 0x00007f0000000000ull;   // title GPU address of the read window
constexpr std::uint64_t title_write = 0x00007f0100000000ull;  // title GPU address of the write window
constexpr std::uint64_t title_heap = 0x00007e0000000000ull;   // title GPU start of the view heap
constexpr std::uint32_t heap_slots = 64;
std::uint32_t pattern(std::uint64_t offset) { return 0x10000000u + static_cast<std::uint32_t>(offset / 4); }
constexpr std::uint32_t canary = 0xaaaaaaaau;

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    UINT increment = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(std::uint32_t i) const
    {
        auto h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += std::size_t{i} * increment;
        return h;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(std::uint32_t i) const
    {
        auto h = heap->GetGPUDescriptorHandleForHeapStart();
        h.ptr += std::uint64_t{i} * increment;
        return h;
    }
    ComPtr<ID3D12Resource> buffer(D3D12_HEAP_TYPE type, std::uint64_t bytes, bool uav)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = type;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = d.DepthOrArraySize = d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES state = type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                            : type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST
                                                                               : D3D12_RESOURCE_STATE_COMMON;
        ComPtr<ID3D12Resource> r;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
        return r;
    }
    // descriptor/host.rs raw_at: an R32_TYPELESS raw view of `bytes` from byte `first` (a multiple of 16).
    void raw_srv(std::uint32_t slot, ID3D12Resource* b, std::uint64_t first, std::uint64_t bytes)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC v{};
        v.Format = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        v.Buffer.FirstElement = first / 4;
        v.Buffer.NumElements = raw_elements(bytes);
        v.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(b, &v, cpu(slot));
    }
    void raw_uav(std::uint32_t slot, ID3D12Resource* b, std::uint64_t first, std::uint64_t bytes)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        v.Buffer.FirstElement = first / 4;
        v.Buffer.NumElements = raw_elements(bytes);
        v.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(b, nullptr, &v, cpu(slot));
    }
    void null_raw_uav(std::uint32_t slot)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
        v.Format = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        v.Buffer.NumElements = 1;
        v.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(nullptr, nullptr, &v, cpu(slot));
    }
    void cbv(std::uint32_t slot, D3D12_GPU_VIRTUAL_ADDRESS at, std::uint32_t bytes)
    {
        const D3D12_CONSTANT_BUFFER_VIEW_DESC v{at, bytes};
        device->CreateConstantBufferView(&v, cpu(slot));
    }
    bool run()
    {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        static std::uint64_t value = 0;
        ++value;
        if (FAILED(queue->Signal(fence.Get(), value))) return false;
        HANDLE e = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(value, e);
        WaitForSingleObject(e, 30000);
        CloseHandle(e);
        if (fence->GetCompletedValue() < value) return false;
        return SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.Get(), nullptr));
    }
};

void transition(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    l->ResourceBarrier(1, &b);
}

// 0 ok, 77 skip, 1 failure.
int make(Gpu& g)
{
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 77;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g.device)))) return 77;
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
        FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o))) ||
        o.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
        std::printf("skip: no SM 6.6 / resource binding tier 3\n");
        return 77;
    }
    const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue))) ||
        FAILED(g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.allocator))) ||
        FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.allocator.Get(), nullptr, IID_PPV_ARGS(&g.list))) ||
        FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence))))
        return 1;
    const D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, heap_slots, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.heap)))) return 1;
    g.increment = g.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // The ABI's host root (WP2 builds it from the export): b0 space0 constants, then the one-view tables of t3 / u5
    // space1, the heap directly indexed.
    D3D12_DESCRIPTOR_RANGE1 ranges[2]{};
    ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 3, 1, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0};
    ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 5, 1, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE, 0};
    D3D12_ROOT_PARAMETER1 params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 4};
    for (int i = 0; i < 2; ++i) {
        params[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[1 + i].DescriptorTable = {1, &ranges[i]};
    }
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rd{};
    rd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rd.Desc_1_1 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED};
    ComPtr<ID3DBlob> blob, error;
    if (FAILED(D3D12SerializeVersionedRootSignature(&rd, &blob, &error)) ||
        FAILED(g.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&g.root)))) {
        std::printf("root signature: %s\n", error ? static_cast<const char*>(error->GetBufferPointer()) : "?");
        return 1;
    }
    return 0;
}

ComPtr<ID3D12PipelineState> pipeline(Gpu& g, const void* code, std::size_t bytes)
{
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};
    d.pRootSignature = g.root.Get();
    d.CS = {code, bytes};
    ComPtr<ID3D12PipelineState> p;
    if (FAILED(g.device->CreateComputePipelineState(&d, IID_PPV_ARGS(&p)))) return nullptr;
    return p;
}

// A window of title memory: `fill(offset)` per dword, uploaded on the list (left in COPY_DEST).
ComPtr<ID3D12Resource> window(Gpu& g, ComPtr<ID3D12Resource>& upload, std::uint32_t (*fill)(std::uint64_t), bool uav)
{
    auto w = g.buffer(D3D12_HEAP_TYPE_DEFAULT, window_bytes, uav);
    upload = g.buffer(D3D12_HEAP_TYPE_UPLOAD, window_bytes, false);
    if (!w || !upload) return nullptr;
    void* p = nullptr;
    upload->Map(0, nullptr, &p);
    for (std::uint64_t o = 0; o < window_bytes; o += 4) static_cast<std::uint32_t*>(p)[o / 4] = fill(o);
    upload->Unmap(0, nullptr);
    g.list->CopyBufferRegion(w.Get(), 0, upload.Get(), 0, window_bytes);
    return w;
}
std::vector<std::uint32_t> read_back(Gpu& g, ID3D12Resource* w, D3D12_RESOURCE_STATES state)
{
    auto rb = g.buffer(D3D12_HEAP_TYPE_READBACK, window_bytes, false);
    std::vector<std::uint32_t> out;
    if (!rb) return out;
    transition(g.list.Get(), w, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g.list->CopyBufferRegion(rb.Get(), 0, w, 0, window_bytes);
    if (!g.run()) return out;
    void* p = nullptr;
    const D3D12_RANGE all{0, window_bytes};
    if (FAILED(rb->Map(0, &all, &p))) return out;
    out.assign(static_cast<const std::uint32_t*>(p), static_cast<const std::uint32_t*>(p) + window_bytes / 4);
    rb->Unmap(0, nullptr);
    return out;
}

// One window from `base` (title address), `buffer` its desktop resource: gpu_find's window lookup.
auto window_lookup(std::uint64_t base, ID3D12Resource* buffer)
{
    return [=](std::uint64_t a, std::uint64_t n, void** b, std::uint64_t* o) {
        if (a < base || a + n > base + window_bytes || a + n < a) return false;
        *b = buffer;
        *o = a - base;
        return true;
    };
}
// The title buffer records (resource_address_index): [first, end) pairs.
auto extent_lookup(std::vector<std::pair<std::uint64_t, std::uint64_t>> records, bool ignore)
{
    return [=](std::uint64_t a, std::uint64_t* base, std::uint64_t* end) {
        if (ignore) return false;
        for (const auto& r : records)
            if (a >= r.first && a < r.second) return *base = r.first, *end = r.second, true;
        return false;
    };
}

int bind_mode(Gpu& g, bool mutate)
{
    auto pso = pipeline(g, startup_gpu_bind_bind_main, sizeof(startup_gpu_bind_bind_main));
    if (!pso) return 1;
    ComPtr<ID3D12Resource> up_read, up_write;
    auto read = window(g, up_read, pattern, false);
    auto write = window(g, up_write, [](std::uint64_t) { return canary; }, true);
    if (!read || !write) return 1;
    transition(g.list.Get(), read.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(g.list.Get(), write.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto in_read = window_lookup(title_read, read.Get());
    const auto in_write = window_lookup(title_write, write.Get());

    // The title's table at slot 10, mirrored in place (host slot = title slot): entry 0 a raw SRV of window offset
    // 0x400; entry 1 the title's CBV at title_read + 0x2010 (16-aligned, not 256): made afresh in the upload buffer.
    Roots roots;
    roots.set_table(true, 0, title_heap + 10 * abi::view_increment);
    g.raw_srv(10, read.Get(), 0x400, 256);
    const std::uint64_t cbv_title = title_read + 0x2010;
    auto copies = g.buffer(D3D12_HEAP_TYPE_UPLOAD, 64 * 1024, false);
    if (!copies) return 1;
    const std::uint32_t cbv_size = constant_view_bytes(16);
    check(cbv_size == 256, "a 16-byte CBV is made in 256 bytes");
    {
        void* p = nullptr;
        copies->Map(0, nullptr, &p);
        auto* d = static_cast<std::uint32_t*>(p) + 512 / 4;  // the ring places at 512 bytes
        for (std::uint32_t i = 0; i < cbv_size / 4; ++i) d[i] = i < 4 ? pattern(cbv_title - title_read + i * 4) : 0;
        copies->Unmap(0, nullptr);
    }
    if (!mutate) g.cbv(11, copies->GetGPUVirtualAddress() + 512, cbv_size);
    else g.cbv(11, read->GetGPUVirtualAddress() + ((cbv_title - title_read) & ~std::uint64_t{255}), 256);  // in place, aligned down
    // What the mutated table dword (index x 64) selects instead: other views, so the reads differ.
    g.raw_srv(20, read.Get(), 0x800, 256);
    g.cbv(21, copies->GetGPUVirtualAddress() + 4096, 256);

    // Root SRV, root parameter 3: title address 8 past 16 inside the title buffer [read+0x1000, read+0x1100).
    const auto read_records = extent_lookup({{title_read + 0x1000, title_read + 0x1100}}, false);
    Found f;
    RawWindow srv_raw, uav_raw;
    const bool srv_found = find_bounded(title_read + 0x1008, 1, ~0ull, 0, read_records, in_read, &f) &&
                           raw_window(f.offset, f.left, true, &srv_raw);
    check(srv_found && srv_raw.first == 0x1000 && srv_raw.bias == 8 && srv_raw.bytes == 0x100,
          "root SRV: raw view from the 16 bytes below, bias 8, to the title buffer's end");
    if (!srv_found) return 1;
    g.raw_srv(40, read.Get(), srv_raw.first, srv_raw.bytes);
    roots.set_view(true, 3, title_read + 0x1008, GpuRootKind::srv);
    // Root UAV, root parameter 5: 4 past 16 in [write+0x3000, write+0x3100).
    const auto write_records = extent_lookup({{title_write + 0x3000, title_write + 0x3100}}, false);
    const bool uav_found = find_bounded(title_write + 0x3004, 1, ~0ull, 0, write_records, in_write, &f) &&
                           raw_window(f.offset, f.left, true, &uav_raw);
    check(uav_found && uav_raw.first == 0x3000 && uav_raw.bias == 4, "root UAV: raw view from the 16 bytes below, bias 4");
    if (!uav_found) return 1;
    g.raw_uav(41, write.Get(), uav_raw.first, uav_raw.bytes);
    roots.set_view(true, 5, title_write + 0x3004, GpuRootKind::uav);

    // The user data as bind() fills it: the table's byte offset, the two biases of the bound views.
    const HeapSpan spans[2] = {{title_heap, heap_slots, false}, {}};
    const GpuUserData words[] = {
        {0, 0, GpuRootType::table, 0, 0},
        {1, 3, GpuRootType::other, abi::user_view_bias, 0},
        {2, 5, GpuRootType::other, abi::user_view_bias, 0},
    };
    const std::vector<BoundBias> biases = {{3, mutate ? 0u : srv_raw.bias}, {5, uav_raw.bias}};
    auto values = fill_user_data(roots, true, words, 3, 4, {}, spans, 2, biases, 0,
                                 [](std::uint32_t, std::uint32_t) -> std::uint64_t { return 0; });
    check(values[0] == 320, "table dword = byte offset in console units (10 x 32)");
    if (mutate) values[0] = 10 * 64;

    ID3D12DescriptorHeap* heaps[] = {g.heap.Get()};
    g.list->SetDescriptorHeaps(1, heaps);
    g.list->SetComputeRootSignature(g.root.Get());
    g.list->SetPipelineState(pso.Get());
    g.list->SetComputeRoot32BitConstants(0, 4, values.data(), 0);
    g.list->SetComputeRootDescriptorTable(1, g.gpu(40));
    g.list->SetComputeRootDescriptorTable(2, g.gpu(41));
    g.list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    g.list->ResourceBarrier(1, &uav);
    const auto out = read_back(g, write.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (out.size() != window_bytes / 4) return 1;
    const std::uint32_t* r = &out[0x3004 / 4];
    char what[160];
    std::snprintf(what, sizeof(what), "table entry 0 (raw SRV at 0x400): 0x%08x, want 0x%08x", r[0], pattern(0x400));
    check(r[0] == pattern(0x400), what);
    std::snprintf(what, sizeof(what), "root SRV at the biased address: 0x%08x, want 0x%08x", r[1], pattern(0x1008));
    check(r[1] == pattern(0x1008), what);
    std::snprintf(what, sizeof(what), "table entry 1 (CBV made afresh) x: 0x%08x, want 0x%08x; w: 0x%08x, want 0x%08x", r[2],
                  pattern(0x2010), r[3], pattern(0x201c));
    check(r[2] == pattern(0x2010) && r[3] == pattern(0x201c), what);
    check(out[0x3000 / 4] == canary && out[0x3014 / 4] == canary, "root UAV stores at the title address, not the view's start");
    return 0;
}

int bounds_mode(Gpu& g, bool mutate)
{
    auto pso = pipeline(g, startup_gpu_bind_bounds_main, sizeof(startup_gpu_bind_bounds_main));
    if (!pso) return 1;
    ComPtr<ID3D12Resource> up;
    auto write = window(g, up, [](std::uint64_t) { return canary; }, true);
    if (!write) return 1;
    transition(g.list.Get(), write.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto in_write = window_lookup(title_write, write.Get());
    // Title buffers: [0x4000, 0x4100) and one running past the window's end [0xf000, 0x11000). Nothing at 0x8000.
    // The mutation bounds by the window alone (no records; any length allowed).
    const auto records = extent_lookup({{title_write + 0x4000, title_write + 0x4100}, {title_write + 0xf000, title_write + 0x11000}}, mutate);
    const std::uint64_t unknown = mutate ? ~0ull : 0;  // root UAVs: no allowance for unknown memory
    ID3D12DescriptorHeap* heaps[] = {g.heap.Get()};
    g.list->SetDescriptorHeaps(1, heaps);
    g.list->SetComputeRootSignature(g.root.Get());
    g.list->SetPipelineState(pso.Get());
    g.raw_srv(40, write.Get(), 0, 16);  // t3 is not read here; a valid view all the same
    struct Case {
        std::uint64_t offset;
        std::uint32_t slot;
    };
    const Case cases[] = {{0x4008, 41}, {0x8000, 42}, {0xff00, 43}};
    char what[160];
    for (const Case& c : cases) {
        Found f;
        RawWindow raw;
        const bool found = find_bounded(title_write + c.offset, 1, ~0ull, unknown, records, in_write, &f) &&
                           raw_window(f.offset, f.left, true, &raw);
        if (c.offset == 0x4008) {
            std::snprintf(what, sizeof(what), "root UAV at 0x4008: %llu bytes left (want 248: the title buffer's end)",
                          static_cast<unsigned long long>(f.left));
            check(found && f.known_extent && f.left == 0xf8 && raw.bias == 8, what);
        } else if (c.offset == 0x8000) {
            check(!found, "root UAV in memory no title buffer holds: not found (a null view)");
        } else {
            std::snprintf(what, sizeof(what), "root UAV at 0xff00: %llu bytes left, cut by the window (want 256)",
                          static_cast<unsigned long long>(f.left));
            check(found && f.left == 0x100 && f.cut_by_window, what);
        }
        if (found) g.raw_uav(c.slot, write.Get(), raw.first, raw.bytes);
        else g.null_raw_uav(c.slot);
        const std::uint32_t values[4] = {0, 0, found ? raw.bias : 0, 128};  // 128 dwords: 512 bytes
        g.list->SetComputeRoot32BitConstants(0, 4, values, 0);
        g.list->SetComputeRootDescriptorTable(1, g.gpu(40));
        g.list->SetComputeRootDescriptorTable(2, g.gpu(c.slot));
        g.list->Dispatch(2, 1, 1);
    }
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    g.list->ResourceBarrier(1, &uav);
    const auto out = read_back(g, write.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (out.size() != window_bytes / 4) return 1;
    const auto stored = [&](std::uint64_t from, std::uint64_t to, std::uint64_t at) {
        for (std::uint64_t o = from; o < to; o += 4)
            if (out[o / 4] != (0xc0de0000u | static_cast<std::uint32_t>((o - at) / 4))) return false;
        return true;
    };
    const auto untouched = [&](std::uint64_t from, std::uint64_t to) {
        for (std::uint64_t o = from; o < to; o += 4)
            if (out[o / 4] != canary) return false;
        return true;
    };
    check(stored(0x4008, 0x4100, 0x4008), "the stores inside the title buffer landed");
    check(untouched(0x4000, 0x4008), "the 8 bytes below the title address (the bias) kept");
    check(untouched(0x4100, 0x4300), "the canaries past the title buffer's end kept (stores past it dropped)");
    check(untouched(0x8000, 0x8200), "nothing stored through the null view");
    check(stored(0xff00, 0x10000, 0xff00), "the stores up to the window's end landed");
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const bool bounds = std::strcmp(argv[1], "bounds") == 0;
    if (!bounds && std::strcmp(argv[1], "bind") != 0) return 2;
    const bool mutate = argc > 2 && std::strcmp(argv[2], "--mutate") == 0;
    Gpu g;
    if (const int rc = make(g)) return rc;
    const int rc = bounds ? bounds_mode(g, mutate) : bind_mode(g, mutate);
    std::printf("gpu %s%s: %d failure(s)%s\n", argv[1], mutate ? " (mutated control)" : "", failures, rc ? " (setup failed)" : "");
    return rc || failures ? 1 : 0;
}
