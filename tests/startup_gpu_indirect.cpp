// WP4 (vindirect): the indirect execution core (src/startup_stubs/gpu_indirect_clamp.h) on a
// desktop device, WARP or hardware. Ported from the upstream Rust tests in src/indirect.rs
// (rejects_unaligned_or_undersized_layouts, warp_count_buffer_ignores_garbage_tail_and_preserves_source,
// warp_window_leaves_unfilled_slots_empty_and_uncharged, warp_budget_saturates_without_wraparound_...,
// warp_a_group_charges_its_records_in_lane_order, warp_a_record_that_does_not_fit_may_take_nothing,
// host_list_budget_starts_empty_at_each_submission, card_runs_wide_indirect_dispatches), plus the console
// signature type 0x69 run as a dispatch and mesh records (kind 10) run as DispatchMesh(x, 1, 1).
//
// Usage: startup_gpu_indirect <warp|hardware> <case> [--mutate]
//   cases: layout count window budget lanes restart restart-nozero pool dispatch69 mesh wide
//   --mutate: the clamp's group-count bound is removed from the HLSL; `count` must then fail.
//   restart-nozero: the budget slot is not zeroed after the reset; must fail (control).
// Exit 0 pass, 1 fail, 77 skipped (no such adapter, or it lacks mesh shaders, or `wide` off AMD).
#include "gpu_indirect_clamp.h"
#include "gpu_indirect_capture.h"

#include "startup_gpu_indirect_ms.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace vi = x2w_gpu_indirect;
using vi::u32;
using vi::u64;

namespace {

int failures = 0;
#define EXPECT(cond, ...)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);      \
            std::printf(__VA_ARGS__);                                        \
            std::printf("\n");                                               \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

constexpr char worker_cs[] = R"hlsl(
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 group : SV_GroupID) {
    uint was;
    hits.InterlockedAdd(0, 1, was);
    hits.InterlockedMax(4, group.x, was);
}
)hlsl";

struct Mapped {
    ComPtr<ID3D12Resource> buffer;
    unsigned char* data = nullptr;
    u64 gpu = 0;
    u32* words() const { return reinterpret_cast<u32*>(data); }
};

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> info;
    HANDLE event = nullptr;
    u64 submitted = 0;
    vi::Clamp clamp;
    vi::Clamp worker;
    vi::HostSignatures signatures;
    std::vector<ComPtr<ID3D12Resource>> kept;

    // A CPU-visible buffer the GPU may write as a UAV (custom heap, write-back, L0), mapped and zeroed.
    Mapped shared(u64 size)
    {
        Mapped m;
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&m.buffer))) ||
            FAILED(m.buffer->Map(0, nullptr, reinterpret_cast<void**>(&m.data)))) {
            std::printf("FAIL: shared buffer of %llu bytes\n", static_cast<unsigned long long>(size));
            std::exit(1);
        }
        std::memset(m.data, 0, static_cast<std::size_t>(size));
        m.gpu = m.buffer->GetGPUVirtualAddress();
        return m;
    }
    ComPtr<ID3D12Resource> device_local(u64 size, D3D12_HEAP_TYPE type = D3D12_HEAP_TYPE_DEFAULT)
    {
        const D3D12_HEAP_PROPERTIES heap{type};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = size;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        const D3D12_RESOURCE_STATES state = type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COMMON;
        ComPtr<ID3D12Resource> r;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&r)))) {
            std::printf("FAIL: buffer of %llu bytes\n", static_cast<unsigned long long>(size));
            std::exit(1);
        }
        return r;
    }
    // Closes, runs and waits for the list, then reopens it. Fails on any debug-layer error.
    void run()
    {
        if (FAILED(list->Close())) {
            std::printf("FAIL: list close\n");
            std::exit(1);
        }
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++submitted);
        if (fence->GetCompletedValue() < submitted) {
            fence->SetEventOnCompletion(submitted, event);
            WaitForSingleObject(event, 60000);
        }
        if (fence->GetCompletedValue() < submitted || FAILED(device->GetDeviceRemovedReason())) {
            std::printf("FAIL: GPU did not finish (removed 0x%08lx)\n", static_cast<unsigned long>(device->GetDeviceRemovedReason()));
            std::exit(1);
        }
        debug_errors();
        allocator->Reset();
        list->Reset(allocator.Get(), nullptr);
    }
    void debug_errors()
    {
        if (!info) return;
        const UINT64 n = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> bytes(size);
            auto* m = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
            if (SUCCEEDED(info->GetMessage(i, m, &size)) &&
                (m->Severity == D3D12_MESSAGE_SEVERITY_ERROR || m->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)) {
                std::printf("FAIL debug layer: %.*s\n", static_cast<int>(m->DescriptionByteLength), m->pDescription);
                ++failures;
            }
        }
        info->ClearStoredMessages();
    }
    void copy_out(ID3D12Resource* from, D3D12_RESOURCE_STATES state, u64 at, const Mapped& to, u64 bytes)
    {
        const D3D12_RESOURCE_BARRIER b = vi::transition(from, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &b);
        list->CopyBufferRegion(to.buffer.Get(), 0, from, at, bytes);
        const D3D12_RESOURCE_BARRIER back = vi::transition(from, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        list->ResourceBarrier(1, &back);
    }
    void bind_worker(u64 hits_gpu)
    {
        list->SetComputeRootSignature(worker.root);
        list->SetPipelineState(worker.pso);
        list->SetComputeRootUnorderedAccessView(0, hits_gpu);
    }
};

// 0 ok, 77 no adapter.
int open_gpu(Gpu& g, bool hardware)
{
    ComPtr<ID3D12Debug> debug;
    char wanted[8]{};
    const bool debug_off = GetEnvironmentVariableA("X2W_TEST_NO_DEBUG_LAYER", wanted, sizeof(wanted)) && wanted[0] == '1';
    if (!debug_off && SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return 77;
    ComPtr<IDXGIAdapter1> adapter;
    if (hardware) {
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g.device))))
                break;
            adapter.Reset();
        }
        if (!g.device) return 77;
    } else {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
            FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g.device))))
            return 77;
    }
    if (debug) g.device.As(&g.info);
    std::printf("debug layer %s\n", g.info ? "on" : "off");
    const D3D12_COMMAND_QUEUE_DESC queue{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(g.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&g.queue))) ||
        FAILED(g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.allocator))) ||
        FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.allocator.Get(), nullptr, IID_PPV_ARGS(&g.list))) ||
        FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence)))) {
        std::printf("FAIL: device objects\n");
        return 1;
    }
    g.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::string errors;
    if (FAILED(vi::make_clamp(g.device.Get(), worker_cs, &g.worker, &errors))) {
        std::printf("FAIL: worker: %s\n", errors.c_str());
        return 1;
    }
    return 0;
}

bool make_clamp(Gpu& g, bool mutate, bool rest = true)
{
    std::string text = vi::clamp_text(vi::widest(vi::is_amd(g.device.Get())), rest);
    if (mutate) {
        const std::string bound = "g = min(g, uint3(g.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));";
        const auto at = text.find(bound);
        if (at == std::string::npos) {
            std::printf("FAIL: the mutation found no bound to remove\n");
            return false;
        }
        text.erase(at, bound.size());
    }
    std::string errors;
    const HRESULT hr = vi::make_clamp(g.device.Get(), text, &g.clamp, &errors);
    if (FAILED(hr)) std::printf("FAIL: the clamp did not build (0x%08lx): %s\n", static_cast<unsigned long>(hr), errors.c_str());
    return SUCCEEDED(hr);
}

void put(unsigned char* at, std::initializer_list<u32> words)
{
    for (u32 w : words) {
        std::memcpy(at, &w, 4);
        at += 4;
    }
}

// ---- cases -----------------------------------------------------------------------------------------------

int case_draw_capture(Gpu& g, bool indexed)
{
    // A pixel UAV records the actual 16x16 pixel coordinates touched by the indirect draw.
    constexpr char shaders[] = R"(
        float4 vs(uint id:SV_VertexID):SV_Position { return float4(-0.5625,0.1875,0.5,1); }
        RWByteAddressBuffer hits:register(u0);
        void ps(float4 xy:SV_Position) { uint ignored; hits.InterlockedAdd(4*(uint(xy.x)+16*uint(xy.y)),1,ignored); }
    )";
    ComPtr<ID3DBlob> vs, ps;
    if (FAILED(D3DCompile(shaders, sizeof(shaders), nullptr, nullptr, nullptr, "vs", "vs_5_1", 0, 0, &vs, nullptr)) ||
        FAILED(D3DCompile(shaders, sizeof(shaders), nullptr, nullptr, nullptr, "ps", "ps_5_1", 0, 0, &ps, nullptr))) return 1;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = g.worker.root;
    pso.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pso.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.SampleMask = UINT_MAX;
    pso.SampleDesc.Count = 1;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    ComPtr<ID3D12PipelineState> pipeline;
    if (FAILED(g.device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)))) return 1;
    constexpr UINT stride = 32, records = 10;
    auto args = g.shared(stride * records), count = g.shared(16), budget = g.shared(256), pixels = g.shared(1024);
    auto indices = g.shared(32);
    put(indices.data, {0,0,1,2,3});
    if (indexed) put(args.data, {3,2,2,UINT_MAX,5});
    else put(args.data, {3,2,7,5});
    put(args.data + stride, {9,1,0,0});
    put(count.data, {1});
    auto scratch = g.device_local(4096);
    vi::ClampCall c;
    c.kind = indexed ? vi::kind_indexed : vi::kind_draw;
    c.stride = stride; c.records = records; c.dst_offset = 256;
    c.src = args.buffer.Get(); c.count = count.buffer.Get(); c.count_out = true;
    c.dst = scratch.Get(); c.budget_gpu = budget.gpu;
    if (!vi::record_clamp(g.list.Get(), g.clamp, c)) return 1;
    auto* signature = g.signatures.get(g.device.Get(), c.kind, stride);
    if (!signature) return 1;
    x2w_compute_trace::IndirectArguments capture;
    if (!capture.record(g.device.Get(), g.list.Get(), c.dst, c.dst_offset, c.kind, stride, records, true)) return 1;
    x2w_compute_trace::IndirectArguments uncounted;
    if (!uncounted.record(g.device.Get(), g.list.Get(), c.dst, c.dst_offset, c.kind, stride, 1, false)) return 1;
    g.list->SetGraphicsRootSignature(g.worker.root);
    g.list->SetGraphicsRootUnorderedAccessView(0, pixels.gpu);
    g.list->SetPipelineState(pipeline.Get());
    g.list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    if (indexed) {
        const D3D12_INDEX_BUFFER_VIEW view{indices.gpu, 32, DXGI_FORMAT_R32_UINT};
        g.list->IASetIndexBuffer(&view);
    }
    const D3D12_VIEWPORT viewport{0,0,16,16,0,1};
    const D3D12_RECT scissor{0,0,16,16};
    g.list->RSSetViewports(1, &viewport);
    g.list->RSSetScissorRects(1, &scissor);
    g.list->ExecuteIndirect(signature, records, c.dst, c.dst_offset, c.dst, c.dst_offset + records * stride);
    g.run();
    x2w_compute_trace::IndirectArguments::Snapshot sample{};
    EXPECT(capture.read(sample) && sample.count == 1 && capture.copied == 8, "effective count and sample bound");
    const size_t bytes = indexed ? sizeof(D3D12_DRAW_INDEXED_ARGUMENTS) : sizeof(D3D12_DRAW_ARGUMENTS);
    EXPECT(!std::memcmp(sample.records[0].data(), args.data, bytes), "counts, start locations and indexed base vertex");
    EXPECT(uncounted.read(sample) && !uncounted.counted && sample.count == 1 &&
        !std::memcmp(sample.records[0].data(), args.data, bytes), "draw without a count buffer");
    for (UINT i=0; i<256; ++i)
        EXPECT(pixels.words()[i] == (i == 6*16+3 ? 6u : 0u), "indirect coverage pixel %u count %u", i, pixels.words()[i]);
    return 0;
}

int case_layout()
{
    u64 bytes = 0;
    EXPECT(vi::validate_layout(0, 16, 4, 0, false, 0, &bytes) && bytes == 64, "draw");
    EXPECT(vi::validate_layout(1, 20, 4, 64, true, 128, &bytes) && bytes == 80, "indexed with count");
    EXPECT(vi::validate_layout(2, 12, 0, 0, true, 0, &bytes) && bytes == 0, "dispatch, no records");
    EXPECT(vi::validate_layout(10, 16, 2, 8, false, 0, &bytes) && bytes == 32, "mesh");
    EXPECT(!vi::validate_layout(0, 12, 1, 0, false, 0, nullptr), "undersized draw");
    EXPECT(!vi::validate_layout(1, 16, 1, 0, false, 0, nullptr), "undersized indexed");
    EXPECT(!vi::validate_layout(2, 8, 1, 0, false, 0, nullptr), "undersized dispatch");
    EXPECT(!vi::validate_layout(10, 8, 1, 0, false, 0, nullptr), "undersized mesh");
    EXPECT(!vi::validate_layout(3, 16, 1, 0, false, 0, nullptr), "unknown kind");
    EXPECT(!vi::validate_layout(0, 18, 1, 0, false, 0, nullptr), "unaligned stride");
    EXPECT(!vi::validate_layout(0, 16, 1, 2, false, 0, nullptr), "unaligned source");
    EXPECT(!vi::validate_layout(0, 16, 1, 0, true, 6, nullptr), "unaligned count");
    EXPECT(!vi::validate_layout(1, 20, 0xffffffffu, 0, false, 0, nullptr), "over 32 bits of bytes");
    u32 kind = 99;
    const u32 single[] = {vi::console_single_dispatch};
    EXPECT(vi::signature_kind(single, 1, &kind) && kind == vi::kind_dispatch, "0x69 runs as a dispatch (kind %u)", kind);
    for (u32 type : {0u, 1u, 2u, 10u}) {
        const u32 one[] = {type};
        EXPECT(vi::signature_kind(one, 1, &kind) && kind == type, "type %u", type);
    }
    for (u32 type : {3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 0x68u, 0x6au}) {
        const u32 one[] = {type};
        EXPECT(!vi::signature_kind(one, 1, &kind), "type %u is not run", type);
    }
    const u32 constants_then_dispatch[] = {5, 2};
    EXPECT(!vi::signature_kind(constants_then_dispatch, 2, &kind), "a signature that changes root arguments is dropped");
    EXPECT(!vi::signature_kind(nullptr, 0, &kind), "no arguments");
    EXPECT(vi::mesh_groups(70000) == 65535 && vi::mesh_groups(12) == 12, "direct mesh dispatches are cut to 65,535");
    const u32 threads[3] = {8847, 1, 1}, size[3] = {64, 1, 1};
    u32 groups[3];
    vi::dispatch_groups(threads, size, groups);
    EXPECT(groups[0] == 139 && groups[1] == 1 && groups[2] == 1, "threads in groups: %u %u %u", groups[0], groups[1], groups[2]);
    const u32 threads2[3] = {0, 12, 9}, size2[3] = {64, 4, 4};
    vi::dispatch_groups(threads2, size2, groups);
    EXPECT(groups[0] == 0 && groups[1] == 3 && groups[2] == 3, "threads in groups (2)");
    EXPECT(vi::widest(true) == (1u << 20) && vi::widest(false) == 65535 && vi::widest(true, false) == 65535, "widest");
    return 0;
}

// The count bounds the records, the garbage tail is neither run nor charged, the source and count are left as
// they were, and whole strides are copied. The effective count is read from the scratch buffer (count_out).
int case_count(Gpu& g)
{
    constexpr u32 kind = vi::kind_dispatch, stride = 24, max = 4096;
    constexpr std::size_t args_offset = 64, count_1 = 128, count_2 = 192;
    const std::size_t src_bytes = args_offset + std::size_t{stride} * max + 64;
    Mapped src = g.shared(src_bytes), count = g.shared(512), budget = g.shared(256), hits = g.shared(256);
    Mapped back = g.shared(stride * 5 + 16);
    const u64 dst_bytes = u64{stride} * max + 4;
    ComPtr<ID3D12Resource> dst = g.device_local(dst_bytes);
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), kind, stride);
    std::memset(src.data, 0xa5, src_bytes);
    auto record = [&](std::size_t i, std::initializer_list<u32> w) { put(src.data + args_offset + i * stride, w); };
    record(0, {3, 2, 1, 0x11112222, 0x33334444, 0x55556666});
    record(1, {5, 1, 1, 0x77778888, 0x9999aaaa, 0xbbbbcccc});
    record(2, {(1u << 20) + 1, 1, 1, 0xddddeeee, 0xffff0001, 0x01234567});
    for (std::size_t i = 3; i < max; ++i) record(i, {0xffffffffu, 16, 1, 0xdeadbeef, 0xbad0c0de, static_cast<u32>(i)});
    std::memset(count.data, 0x5a, 512);
    put(count.data + count_1, {3});
    put(count.data + count_2, {2});
    const std::vector<unsigned char> src_before(src.data, src.data + src_bytes), count_before(count.data, count.data + 512);

    vi::ClampCall c;
    c.kind = kind;
    c.stride = stride;
    c.records = max;
    c.src = src.buffer.Get();
    c.src_offset = args_offset;
    c.count = count.buffer.Get();
    c.count_offset = count_1;
    c.dst = dst.Get();
    c.budget_gpu = budget.gpu;
    c.count_out = true;
    EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "first clamp recorded");
    g.bind_worker(hits.gpu);
    g.list->ExecuteIndirect(sig, max, dst.Get(), 0, dst.Get(), u64{stride} * max);
    // Snapshot of the first five records and the count after the records.
    const D3D12_RESOURCE_BARRIER b = vi::transition(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    g.list->ResourceBarrier(1, &b);
    g.list->CopyBufferRegion(back.buffer.Get(), 0, dst.Get(), 0, stride * 5);
    g.list->CopyBufferRegion(back.buffer.Get(), stride * 5, dst.Get(), u64{stride} * max, 4);
    const D3D12_RESOURCE_BARRIER b2 = vi::transition(dst.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    g.list->ResourceBarrier(1, &b2);
    // A second execution in the same list, the scratch buffer straight from arguments back to unordered access.
    c.count_offset = count_2;
    c.dst_before = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "second clamp recorded");
    g.bind_worker(hits.gpu);
    g.list->ExecuteIndirect(sig, max, dst.Get(), 0, dst.Get(), u64{stride} * max);
    g.run();

    const u32 expected = (6 + 5 + 65535) + (6 + 5);
    EXPECT(hits.words()[0] == expected, "valid dispatches all run: %u groups, expected %u", hits.words()[0], expected);
    EXPECT(budget.words()[0] == expected * 64, "budget charged %u, expected %u", budget.words()[0], expected * 64);
    EXPECT(std::memcmp(src_before.data(), src.data, src_bytes) == 0, "source never written");
    EXPECT(std::memcmp(count_before.data(), count.data, 512) == 0, "count never written");
    const u32* w = back.words();
    const u32 r0[] = {3, 2, 1, 0x11112222, 0x33334444, 0x55556666}, r1[] = {5, 1, 1, 0x77778888, 0x9999aaaa, 0xbbbbcccc},
              r2[] = {65535, 1, 1, 0xddddeeee, 0xffff0001, 0x01234567};
    EXPECT(std::memcmp(w, r0, 24) == 0, "record 0: %u %u %u", w[0], w[1], w[2]);
    EXPECT(std::memcmp(w + 6, r1, 24) == 0, "record 1: %u %u %u", w[6], w[7], w[8]);
    EXPECT(std::memcmp(w + 12, r2, 24) == 0, "record 2 clamped to 65535: %u %u %u", w[12], w[13], w[14]);
    EXPECT(w[21] != 0xdeadbeef, "inactive record 3 not copied");
    EXPECT(w[30] == 3, "effective count after the records: %u", w[30]);
    return 0;
}

// Mesh records clamped ahead and one by one: filled ones as they are, all-ones slots empty and free.
int case_window(Gpu& g)
{
    constexpr u32 stride = 32, records = 16;
    const u32 filled[3] = {5, 0, 519};
    const std::size_t total = std::size_t{stride} * records;
    for (bool ahead : {true, false}) {
        Mapped src = g.shared(total), budget = g.shared(256), back = g.shared(total);
        ComPtr<ID3D12Resource> dst = g.device_local(total);
        std::memset(src.data, 0xff, total);
        for (u32 i = 0; i < 3; ++i) put(src.data + i * stride, {filled[i], 1, 1, 0xab00 + i});
        vi::ClampCall c;
        c.kind = vi::kind_mesh;
        c.stride = stride;
        c.records = records;
        c.src = src.buffer.Get();
        c.dst = dst.Get();
        c.budget_gpu = budget.gpu;
        c.ahead = ahead;
        EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "clamp recorded");
        g.copy_out(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, 0, back, total);
        g.run();
        const u32* out = back.words();
        for (u32 i = 0; i < 3; ++i)
            EXPECT(out[i * 8] == filled[i] && out[i * 8 + 1] == 1 && out[i * 8 + 2] == 1 && out[i * 8 + 3] == 0xab00 + i,
                   "filled record %u (ahead %d): %u %u %u %x", i, ahead, out[i * 8], out[i * 8 + 1], out[i * 8 + 2], out[i * 8 + 3]);
        for (u32 i = 3; i < records; ++i) {
            EXPECT(out[i * 8] == 0 && out[i * 8 + 1] == 0 && out[i * 8 + 2] == 0, "unfilled slot %u empty (ahead %d)", i, ahead);
            EXPECT(out[i * 8 + 3] == 0xffffffffu, "words after the record copied (slot %u)", i);
        }
        EXPECT(budget.words()[0] == (5 + 519) * 128, "unfilled slots charge nothing: %u", budget.words()[0]);
        EXPECT(budget.words()[1] == 519, "largest asked %u", budget.words()[1]);
    }
    return 0;
}

// The budget saturates at 2^27 without 32-bit wraparound; draw fields past the counts are kept.
int case_budget(Gpu& g)
{
    constexpr u32 stride = 24, records = 128;
    const std::size_t total = std::size_t{stride} * records;
    Mapped src = g.shared(total), budget = g.shared(256), back = g.shared(total);
    ComPtr<ID3D12Resource> dst = g.device_local(total);
    for (u32 i = 0; i < records; ++i)
        put(src.data + i * stride, {i + 1 < records ? (1u << 22) : (1u << 23), 0xffffffffu, 1000 + i, static_cast<u32>(-200 + static_cast<int>(i)),
                                    5000 + i, 0xcafe0000u | i});
    const std::vector<unsigned char> before(src.data, src.data + total);
    vi::ClampCall c;
    c.kind = vi::kind_indexed;
    c.stride = stride;
    c.records = records;
    c.src = src.buffer.Get();
    c.dst = dst.Get();
    c.budget_gpu = budget.gpu;
    EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "clamp recorded");
    g.copy_out(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, 0, back, total);
    g.run();
    EXPECT(budget.words()[0] == vi::budget, "saturates at the budget: %u", budget.words()[0]);
    EXPECT(std::memcmp(before.data(), src.data, total) == 0, "source untouched");
    u32 kept = 0, zeroed = 0;
    const u32* out = back.words();
    for (u32 i = 0; i < records; ++i) {
        const u32* r = out + i * 6;
        if (r[0] == (1u << 22) && r[1] == 16) ++kept;
        else if (r[0] == 0 && r[1] == 0) ++zeroed;
        EXPECT(r[2] == 1000 + i && r[3] == static_cast<u32>(-200 + static_cast<int>(i)) && r[4] == 5000 + i && r[5] == (0xcafe0000u | i),
               "fields of record %u kept", i);
    }
    EXPECT(kept == 2 && zeroed == records - 2, "kept %u, zeroed %u", kept, zeroed);
    return 0;
}

// Draws of 100, 250, 50, 100 with 300 left: in lane order; with REST the one that does not fit takes the rest.
int case_lanes(Gpu& g, bool rest)
{
    vi::Clamp own;
    std::string errors;
    if (FAILED(vi::make_clamp(g.device.Get(), vi::clamp_text(vi::widest(vi::is_amd(g.device.Get())), rest), &own, &errors))) {
        std::printf("FAIL: clamp (rest %d): %s\n", rest, errors.c_str());
        return 1;
    }
    constexpr u32 stride = 16;
    const u32 counts[4] = {100, 250, 50, 100};
    Mapped src = g.shared(64), budget = g.shared(256), back = g.shared(64);
    ComPtr<ID3D12Resource> dst = g.device_local(64);
    for (u32 i = 0; i < 4; ++i) put(src.data + i * stride, {counts[i], 1, 7, 9});
    budget.words()[0] = vi::budget - 300;
    vi::ClampCall c;
    c.kind = vi::kind_draw;
    c.stride = stride;
    c.records = 4;
    c.src = src.buffer.Get();
    c.dst = dst.Get();
    c.budget_gpu = budget.gpu;
    EXPECT(vi::record_clamp(g.list.Get(), own, c), "clamp recorded");
    g.copy_out(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, 0, back, 64);
    g.run();
    own.release();
    const u32 left = vi::budget - budget.words()[0];
    const u32* out = back.words();
    const u32 expect_rest[4] = {100, 0, 0, 0}, expect_take_nothing[4] = {100, 0, 50, 100};
    const u32* expected = rest ? expect_rest : expect_take_nothing;
    for (u32 i = 0; i < 4; ++i)
        EXPECT(out[i * 4] == expected[i] && out[i * 4 + 1] == (expected[i] ? 1u : 0u) && out[i * 4 + 2] == 7 && out[i * 4 + 3] == 9,
               "record %u (rest %d): %u %u", i, rest, out[i * 4], out[i * 4 + 1]);
    EXPECT(left == (rest ? 0u : 50u), "left %u (rest %d)", left, rest);
    return 0;
}

// The budget restarts at each recording: zeroed on the GPU at the first clamp after the reset (as the glue
// does, from a pool slot). A recording whose clamps spend it all leaves a small record after them empty; the
// next recording's small record runs. `zero` false: the control, the slot is not zeroed again.
int case_restart(Gpu& g, bool zero)
{
    Mapped big = g.shared(64), little = g.shared(64), hits = g.shared(256);
    // Three records of 65535 x 16 groups (67,107,840 threads each): two fit 2^27, the third takes the rest.
    for (u32 i = 0; i < 3; ++i) put(big.data + i * 12, {65535, 16, 1});
    put(little.data, {1, 1, 1});
    ComPtr<ID3D12Resource> pool = g.device_local(64 * vi::budget_slot_bytes);
    ComPtr<ID3D12Resource> zeros = g.device_local(256, D3D12_HEAP_TYPE_UPLOAD);
    {
        void* p = nullptr;
        zeros->Map(0, nullptr, &p);
        std::memset(p, 0, 256);
        zeros->Unmap(0, nullptr);
    }
    const u64 slot = 5 * vi::budget_slot_bytes;
    ComPtr<ID3D12Resource> dst = g.device_local(64 << 10);
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), vi::kind_dispatch, 12);
    for (int round = 0; round < 2; ++round) {
        if (round == 0 || zero) vi::record_budget_zero(g.list.Get(), pool.Get(), slot, zeros.Get());
        vi::ClampCall c;
        c.kind = vi::kind_dispatch;
        c.stride = 12;
        c.dst = dst.Get();
        c.budget_gpu = pool->GetGPUVirtualAddress() + slot;
        c.budget_resource = pool.Get();
        if (round == 0) {
            c.records = 3;
            c.src = big.buffer.Get();
            EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "big clamp");
            c.dst_before = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        }
        c.records = 1;
        c.src = little.buffer.Get();
        EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "small clamp");
        g.bind_worker(hits.gpu);
        g.list->ExecuteIndirect(sig, 1, dst.Get(), 0, nullptr, 0);
        const D3D12_RESOURCE_BARRIER rest = vi::transition(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
        g.list->ResourceBarrier(1, &rest);
        g.run();
        EXPECT(hits.words()[0] == static_cast<u32>(round), "round %d: the small record ran %u times, expected %d", round,
               hits.words()[0], round);
    }
    return 0;
}

// Two lists in one ExecuteCommandLists, each zeroing and charging its own slot of one budget pool, as the glue
// does: both run their record, and the debug layer finds no state the pool is not in (the pool rests in COMMON
// outside each clamp, since buffers decay only at the end of the whole call).
int case_pool(Gpu& g)
{
    Mapped one = g.shared(64), hits = g.shared(256);
    put(one.data, {3, 1, 1});
    ComPtr<ID3D12Resource> pool = g.device_local(64 * vi::budget_slot_bytes);
    ComPtr<ID3D12Resource> zeros = g.device_local(256, D3D12_HEAP_TYPE_UPLOAD);
    {
        void* p = nullptr;
        zeros->Map(0, nullptr, &p);
        std::memset(p, 0, 256);
        zeros->Unmap(0, nullptr);
    }
    ComPtr<ID3D12Resource> dst = g.device_local(64 << 10);
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), vi::kind_dispatch, 12);
    ComPtr<ID3D12CommandAllocator> allocator2;
    ComPtr<ID3D12GraphicsCommandList> list2;
    if (FAILED(g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator2))) ||
        FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator2.Get(), nullptr, IID_PPV_ARGS(&list2)))) {
        std::printf("FAIL: second list\n");
        return 1;
    }
    ID3D12GraphicsCommandList* lists[2] = {g.list.Get(), list2.Get()};
    for (int i = 0; i < 2; ++i) {
        ID3D12GraphicsCommandList* l = lists[i];
        const u64 slot = (7 + i) * vi::budget_slot_bytes;
        vi::record_budget_zero(l, pool.Get(), slot, zeros.Get());
        for (int call = 0; call < 2; ++call) {
            vi::ClampCall c;
            c.kind = vi::kind_dispatch;
            c.stride = 12;
            c.records = 1;
            c.src = one.buffer.Get();
            c.dst = dst.Get();
            c.dst_offset = 256 * (2 * i + call);
            c.budget_gpu = pool->GetGPUVirtualAddress() + slot;
            c.budget_resource = pool.Get();
            EXPECT(vi::record_clamp(l, g.clamp, c), "clamp %d.%d", i, call);
            l->SetComputeRootSignature(g.worker.root);
            l->SetPipelineState(g.worker.pso);
            l->SetComputeRootUnorderedAccessView(0, hits.gpu);
            l->ExecuteIndirect(sig, 1, dst.Get(), c.dst_offset, nullptr, 0);
            const D3D12_RESOURCE_BARRIER rest = vi::transition(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
            l->ResourceBarrier(1, &rest);
        }
    }
    list2->Close();
    // g.run() closes and runs g.list; list2 goes in the same call.
    if (FAILED(g.list->Close())) return 1;
    ID3D12CommandList* both[2] = {g.list.Get(), list2.Get()};
    g.queue->ExecuteCommandLists(2, both);
    g.queue->Signal(g.fence.Get(), ++g.submitted);
    g.fence->SetEventOnCompletion(g.submitted, g.event);
    WaitForSingleObject(g.event, 60000);
    g.debug_errors();
    EXPECT(hits.words()[0] == 12, "both lists ran both records: %u groups, expected 12", hits.words()[0]);
    g.allocator->Reset();
    g.list->Reset(g.allocator.Get(), nullptr);
    return 0;
}

// A signature of the console's type 0x69 runs its records as dispatches through a host signature of kind 2.
int case_dispatch69(Gpu& g)
{
    const D3D12_INDIRECT_ARGUMENT_DESC title[1] = {{static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(0x69)}};
    u32 kind = 99;
    EXPECT(vi::signature_kind_of(title, 1, &kind) && kind == vi::kind_dispatch, "0x69 decoded as %u", kind);
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), kind, 12);
    EXPECT(sig != nullptr, "host signature for kind 2, stride 12");
    EXPECT(sig == g.signatures.get(g.device.Get(), kind, 12), "made once");
    if (!sig) return 1;
    Mapped src = g.shared(64), budget = g.shared(256), hits = g.shared(256);
    put(src.data, {7, 2, 1, 0xffffffffu, 1, 1});
    ComPtr<ID3D12Resource> dst = g.device_local(64 << 10);
    vi::ClampCall c;
    c.kind = kind;
    c.stride = 12;
    c.records = 2;
    c.src = src.buffer.Get();
    c.dst = dst.Get();
    c.dst_offset = 256;
    c.budget_gpu = budget.gpu;
    EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "clamp recorded");
    g.bind_worker(hits.gpu);
    g.list->ExecuteIndirect(sig, 2, dst.Get(), 256, nullptr, 0);
    g.run();
    // (7, 2, 1) runs 14 groups; (2^32 - 1, 1, 1) is cut to 65,535 in x (past WIDEST on any card).
    const u32 expected = 14 + 65535;
    EXPECT(hits.words()[0] == expected, "groups run %u, expected %u", hits.words()[0], expected);
    return 0;
}

// Mesh records (kind 10) with a count: run as DispatchMesh(x, 1, 1); all-ones and over-65,535 records empty.
int case_mesh(Gpu& g)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    if (FAILED(g.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))) ||
        options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
        std::printf("SKIP: no mesh shaders on this adapter\n");
        return 77;
    }
    ComPtr<ID3D12Device2> device2;
    ComPtr<ID3D12GraphicsCommandList6> list6;
    if (FAILED(g.device.As(&device2)) || FAILED(g.list.As(&list6))) {
        std::printf("SKIP: no ID3D12Device2 / ID3D12GraphicsCommandList6\n");
        return 77;
    }
    ComPtr<ID3D12RootSignature> root;
    if (FAILED(g.device->CreateRootSignature(0, startup_gpu_indirect_ms, sizeof(startup_gpu_indirect_ms), IID_PPV_ARGS(&root)))) {
        std::printf("FAIL: mesh root signature\n");
        return 1;
    }
    struct alignas(void*) Stream {
        struct alignas(void*) {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE;
            ID3D12RootSignature* value;
        } root;
        struct alignas(void*) {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS;
            D3D12_SHADER_BYTECODE value;
        } ms;
        struct alignas(void*) {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER;
            D3D12_RASTERIZER_DESC value;
        } raster;
        struct alignas(void*) {
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL;
            D3D12_DEPTH_STENCIL_DESC value;
        } depth;
    } stream{};
    stream.root.value = root.Get();
    stream.ms.value = {startup_gpu_indirect_ms, sizeof(startup_gpu_indirect_ms)};
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthEnable = FALSE;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device2->CreatePipelineState(&desc, IID_PPV_ARGS(&pso)))) {
        std::printf("FAIL: mesh pipeline\n");
        return 1;
    }
    constexpr u32 stride = 12, max = 8;
    Mapped src = g.shared(stride * max), count = g.shared(16), budget = g.shared(256), seen = g.shared(256);
    std::memset(src.data, 0xff, stride * max);
    put(src.data + 0 * stride, {5, 1, 1});
    // record 1 stays all ones (an unfilled slot)
    put(src.data + 2 * stride, {70000, 1, 1});
    put(src.data + 3 * stride, {3, 7, 9});
    put(count.data, {4});
    ComPtr<ID3D12Resource> dst = g.device_local(64 << 10);
    vi::ClampCall c;
    c.kind = vi::kind_mesh;
    c.stride = stride;
    c.records = max;
    c.src = src.buffer.Get();
    c.count = count.buffer.Get();
    c.dst = dst.Get();
    c.budget_gpu = budget.gpu;
    c.count_out = true;
    EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "clamp recorded");
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), vi::kind_mesh, stride);
    EXPECT(sig != nullptr, "mesh host signature");
    if (!sig) return 1;
    g.list->SetGraphicsRootSignature(root.Get());
    g.list->SetPipelineState(pso.Get());
    g.list->SetGraphicsRootUnorderedAccessView(0, seen.gpu);
    const D3D12_VIEWPORT viewport{0, 0, 16, 16, 0, 1};
    const D3D12_RECT scissor{0, 0, 16, 16};
    g.list->RSSetViewports(1, &viewport);
    g.list->RSSetScissorRects(1, &scissor);
    g.list->ExecuteIndirect(sig, max, dst.Get(), 0, dst.Get(), u64{stride} * max);
    g.run();
    const u32* s = seen.words();
    EXPECT(s[0] == 5 + 3, "mesh groups run %u, expected 8", s[0]);
    EXPECT(s[1] == 4 && s[2] == 0 && s[3] == 0, "one-dimensional: largest group %u %u %u", s[1], s[2], s[3]);
    EXPECT(budget.words()[0] == 8 * 128, "charged %u", budget.words()[0]);
    EXPECT(budget.words()[1] == 5, "largest asked %u (the over-65,535 and all-ones records are not counted)", budget.words()[1]);
    return 0;
}

// An AMD card runs every group of an indirect dispatch of more than 65,535 in x (the title's tile lists).
int case_wide(Gpu& g, bool hardware)
{
    if (!hardware || !vi::is_amd(g.device.Get())) {
        std::printf("SKIP: wide dispatches are run as asked only on an AMD card\n");
        return 77;
    }
    Mapped src = g.shared(64), budget = g.shared(256), hits = g.shared(256);
    ComPtr<ID3D12Resource> dst = g.device_local(64 << 10);
    ID3D12CommandSignature* sig = g.signatures.get(g.device.Get(), vi::kind_dispatch, 12);
    for (u32 asked : {67440u, 115200u, 1u << 20, (1u << 20) + 1}) {
        const u32 runs = asked > (1u << 20) ? 65535 : asked;
        std::memset(hits.data, 0, 256);
        std::memset(budget.data, 0, 256);
        put(src.data, {asked, 1, 1});
        vi::ClampCall c;
        c.kind = vi::kind_dispatch;
        c.stride = 12;
        c.records = 1;
        c.src = src.buffer.Get();
        c.dst = dst.Get();
        c.budget_gpu = budget.gpu;
        EXPECT(vi::record_clamp(g.list.Get(), g.clamp, c), "clamp recorded");
        g.bind_worker(hits.gpu);
        g.list->ExecuteIndirect(sig, 1, dst.Get(), 0, nullptr, 0);
        const D3D12_RESOURCE_BARRIER rest = vi::transition(dst.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
        g.list->ResourceBarrier(1, &rest);
        g.run();
        EXPECT(hits.words()[0] == runs && hits.words()[1] == runs - 1, "asked %u: ran %u, largest group %u", asked, hits.words()[0],
               hits.words()[1]);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::printf("usage: startup_gpu_indirect <warp|hardware> <case> [--mutate]\n");
        return 2;
    }
    const std::string adapter = argv[1], name = argv[2];
    const bool mutate = argc > 3 && std::string(argv[3]) == "--mutate";
    if (name == "layout") {
        case_layout();
        std::printf("%s layout\n", failures ? "FAIL" : "PASS");
        return failures ? 1 : 0;
    }
    Gpu g;
    const int opened = open_gpu(g, adapter == "hardware");
    if (opened) {
        if (opened == 77) std::printf("SKIP: no %s adapter\n", adapter.c_str());
        return opened;
    }
    if (!make_clamp(g, mutate)) return 1;
    int result = 0;
    if (name == "capture") result = case_draw_capture(g, false) | case_draw_capture(g, true);
    else if (name == "count") result = case_count(g);
    else if (name == "window") result = case_window(g);
    else if (name == "budget") result = case_budget(g);
    else if (name == "lanes") result = case_lanes(g, true) | case_lanes(g, false);
    else if (name == "restart") result = case_restart(g, true);
    else if (name == "restart-nozero") result = case_restart(g, false);
    else if (name == "dispatch69") result = case_dispatch69(g);
    else if (name == "pool") result = case_pool(g);
    else if (name == "mesh") result = case_mesh(g);
    else if (name == "wide") result = case_wide(g, adapter == "hardware");
    else {
        std::printf("unknown case %s\n", name.c_str());
        return 2;
    }
    if (result == 77) return 77;
    if (result || failures) {
        std::printf("FAIL %s %s%s (%d failures)\n", adapter.c_str(), name.c_str(), mutate ? " --mutate" : "", failures);
        return 1;
    }
    std::printf("PASS %s %s%s\n", adapter.c_str(), name.c_str(), mutate ? " --mutate" : "");
    return 0;
}
