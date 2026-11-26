// startup_gpu_heap: the descriptor mirror (src/startup_stubs/gpu_heap.inl) on a real device, with a fixture
// title heap, title memory and window of our own, read by our own SM 6.6 shader (startup_gpu_heap.hlsl).
//
//   startup_gpu_heap <warp|hardware> twins   the raw twin of a typed, a structured and a raw view reads the
//                                                view's bytes, bounded; a UAV twin writes them and nothing else; a
//                                                CBV's twin is the CBV; the twin of a texture slot, of an empty slot
//                                                and of the last slot reads zero; a dynamic slot's twin
//   startup_gpu_heap <warp|hardware> views   a structured view is made raw (raw_window), the table of strides,
//                                                an unaligned CBV is null at the title slot, a V# record is a view,
//                                                a view off its window's elements reads a copy of its own bytes
//   startup_gpu_heap <warp|hardware> submit  records the title copied as bytes are mirrored by submit(), a
//                                                cleared one goes null, own-byte copies follow title memory, the
//                                                texture of a made slot is due for upload
//
// WARP takes no heap of 2^22 views (debug layer id 645), so the WARP fixture halves the production heap sizes
// (twins at 2^20 in a heap of 2^21: Limits{2^20, 2^20}); on hardware it is the probed one, which must be 2^22 with
// twins at 2^21 (exit 77 if the host takes no 2^22).
#include "startup_stubs/gpu_heap.inl"

#include <dxgi1_6.h>
#include <wrl/client.h>

#include <map>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "startup_gpu_heap_cs.h"
#include "startup_gpu_memory_cs.h"
#include "startup_gpu_memory_store_cs.h"
#include "startup_stubs/gpu_memory_arena.h"
#include "startup_stubs/gpu_memory.h"
#include "startup_stubs/gpu_packed_light.h"
#include "startup_stubs/small_texture_heap.h"
#include "startup_stubs/gpu_list_states.h"
#include "startup_stubs/gpu_transition.h"
#include "startup_stubs/gpu_compute_trace.h"
#include "startup_stubs/gpu_ambient.h"
#include "startup_stubs/gpu_resident_mips.h"
#include "startup_stubs/slice_footprints.h"
#include <d3dcompiler.h>

using Microsoft::WRL::ComPtr;
using namespace gpu_heap;
using namespace gpu_mirror;

namespace {

int failures = 0;
#define CHECK(c)                                                                        \
    do {                                                                                \
        if (!(c)) {                                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                    \
            ++failures;                                                                 \
        }                                                                               \
    } while (0)

constexpr u32 title_slots = 4096;
constexpr u64 memory_bytes = 1u << 20;
constexpr u64 window_offset = 256;  // where title memory starts in the window

struct Fixture {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    u64 fence_value = 0;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12Resource> window, result, readback, texture;
    unsigned char* memory = nullptr;  // title memory
    unsigned char* heap_memory = nullptr;  // the title's descriptor memory
    std::vector<Record> records;
    std::vector<ResourceFacts> resources;  // wrapper = index + 1
    std::vector<u64> uploads;
    bool coherent = false;
    bool find_fails = false;  // find answers nothing (a window not known yet)
};

Fixture fx;
u64 mapped_texture = 0;
unsigned deferred_probes = 0;
unsigned resolved_probes = 0;
unsigned fixture_decline = 8;
u64 fixture_room = UINT64_MAX, fixture_epoch = 0, fixture_blocked_address = 0;
unsigned allocation_probes = 0;
bool fixture_needs_allocation = false;
bool fixture_remap_on_probe = false;
UINT fixture_creations = 0;
float partial_clamp = 0;
UINT vendor = 0;

u64 title(u64 offset) { return reinterpret_cast<u64>(fx.memory) + offset; }
std::uint32_t pattern(u64 dword) { return 0x10000u + static_cast<std::uint32_t>(dword); }

bool find(void*, u64 address, u64 bytes, ID3D12Resource** resource, u64* offset, u64* left)
{
    const u64 base = reinterpret_cast<u64>(fx.memory);
    if (fx.find_fails) return false;
    if (address < base || address - base >= memory_bytes || bytes > memory_bytes - (address - base)) return false;
    *resource = fx.window.Get();
    *offset = address - base + window_offset;
    *left = memory_bytes - (address - base);
    return true;
}
bool record(void*, u32 id, Record* out)
{
    if (id >= fx.records.size()) return false;
    *out = fx.records[id];
    return true;
}
bool resource(void*, u64 wrapper, ResourceFacts* out)
{
    if (!wrapper || wrapper > fx.resources.size()) return false;
    *out = fx.resources[wrapper - 1];
    return true;
}
bool upload(void*, void*, u64 wrapper, ID3D12Resource*)
{
    fx.uploads.push_back(wrapper);
    return true;
}

u64 buffer_wrapper(u64 offset, u64 bytes)
{
    ResourceFacts f;
    f.desktop = fx.window.Get();  // (the window stands in for the buffer's own copy)
    f.buffer = true;
    f.title_memory = title(offset);
    f.bytes = bytes;
    fx.resources.push_back(f);
    return fx.resources.size();
}
u64 texture_wrapper()
{
    ResourceFacts f;
    f.desktop = fx.texture.Get();
    fx.resources.push_back(f);
    return fx.resources.size();
}
// A view record and the token for it written into title slot `index` (note_view's).
u32 note(u32 index, u32 slot, u64 wrapper, const void* desc, u32 bytes)
{
    Record r;
    r.id = static_cast<u32>(fx.records.size());
    r.slot = slot;
    r.resource = wrapper;
    r.desc_bytes = bytes;
    std::memcpy(r.desc, desc, bytes);
    fx.records.push_back(r);
    const std::uint32_t token[8] = {token_magic, r.id, slot, 0, static_cast<std::uint32_t>(wrapper),
                                    static_cast<std::uint32_t>(wrapper >> 32), 0, 0};
    std::memcpy(fx.heap_memory + u64{index} * 32, token, 32);
    return r.id;
}
D3D12_SHADER_RESOURCE_VIEW_DESC buffer_srv(DXGI_FORMAT format, u64 first, u32 n, u32 stride, bool raw)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = format;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.FirstElement = first;
    d.Buffer.NumElements = n;
    d.Buffer.StructureByteStride = stride;
    d.Buffer.Flags = raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
    return d;
}

void wait()
{
    fx.queue->Signal(fx.fence.Get(), ++fx.fence_value);
    HANDLE e = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    fx.fence->SetEventOnCompletion(fx.fence_value, e);
    WaitForSingleObject(e, 10000);
    CloseHandle(e);
}
ComPtr<ID3D12Resource> buffer(u64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap{type};
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    ComPtr<ID3D12Resource> r;
    if (FAILED(fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}
void begin()
{
    fx.allocator->Reset();
    fx.list->Reset(fx.allocator.Get(), nullptr);
}
void end_and_wait()
{
    fx.list->Close();
    ID3D12CommandList* lists[] = {fx.list.Get()};
    fx.queue->ExecuteCommandLists(1, lists);
    wait();
}
// The window gets title memory's bytes (from `window_offset` on).
void fill_window()
{
    auto up = buffer(memory_bytes + window_offset, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* data = nullptr;
    up->Map(0, nullptr, &data);
    std::memset(data, 0xee, window_offset);
    std::memcpy(static_cast<unsigned char*>(data) + window_offset, fx.memory, memory_bytes);
    up->Unmap(0, nullptr);
    begin();
    fx.list->CopyBufferRegion(fx.window.Get(), 0, up.Get(), 0, memory_bytes + window_offset);
    end_and_wait();
}
// The window's dwords from title offset `offset`.
std::vector<std::uint32_t> read_window(u64 offset, u32 dwords)
{
    auto back = buffer(u64{dwords} * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    begin();
    fx.list->CopyBufferRegion(back.Get(), 0, fx.window.Get(), window_offset + offset, u64{dwords} * 4);
    end_and_wait();
    std::vector<std::uint32_t> out(dwords);
    void* data = nullptr;
    back->Map(0, nullptr, &data);
    std::memcpy(out.data(), data, out.size() * 4);
    back->Unmap(0, nullptr);
    return out;
}
// Runs the shader on `heap` slot `slot`: what it read (or wrote) of `count` dwords.
std::vector<std::uint32_t> pick(ID3D12DescriptorHeap* heap, u32 mode, u32 slot, u32 count, u32 mark = 0)
{
    begin();
    ID3D12DescriptorHeap* heaps[] = {heap};
    fx.list->SetDescriptorHeaps(1, heaps);
    fx.list->SetComputeRootSignature(fx.root.Get());
    fx.list->SetPipelineState(fx.pso.Get());
    const std::uint32_t constants[4] = {mode, slot, count, mark};
    fx.list->SetComputeRoot32BitConstants(0, 4, constants, 0);
    fx.list->SetComputeRootUnorderedAccessView(1, fx.result->GetGPUVirtualAddress());
    fx.list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = fx.result.Get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    fx.list->ResourceBarrier(1, &b);
    fx.list->CopyBufferRegion(fx.readback.Get(), 0, fx.result.Get(), 0, 256);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    fx.list->ResourceBarrier(1, &b);
    end_and_wait();
    std::vector<std::uint32_t> out(count);
    void* data = nullptr;
    fx.readback->Map(0, nullptr, &data);
    std::memcpy(out.data(), data, u64{count} * 4);
    fx.readback->Unmap(0, nullptr);
    return out;
}
std::vector<std::uint32_t> pick(HostHeap& heap, u32 mode, u32 slot, u32 count, u32 mark = 0)
{
    return pick(heap.heap(), mode, slot, count, mark);
}
bool all_zero(const std::vector<std::uint32_t>& v)
{
    for (auto x : v)
        if (x) return false;
    return true;
}
// The title's dwords from byte `offset`, then zeros, `count` in all with `n` of them the title's.
bool title_then_zero(const std::vector<std::uint32_t>& v, u64 offset, u32 n)
{
    for (u32 i = 0; i < v.size(); ++i) {
        std::uint32_t want = 0;
        if (i < n) std::memcpy(&want, fx.memory + offset + u64{i} * 4, 4);
        if (v[i] != want) {
            std::printf("  dword %u: 0x%08x, expected 0x%08x\n", i, v[i], want);
            return false;
        }
    }
    return true;
}

int setup(bool warp)
{
    if (warp) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
    }
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return 77;
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
    } else {
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) == S_OK; ++i) {
            DXGI_ADAPTER_DESC1 d;
            adapter->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter.Reset();
        }
    }
    if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&fx.device)))) return 77;
    {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        vendor = d.VendorId;
        std::printf("adapter=%ls\n", d.Description);
    }
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
    fx.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o));
    if (FAILED(fx.device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) || sm.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
        o.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
        std::printf("no SM 6.6 / binding tier 3: skipped\n");
        return 77;
    }
    D3D12_COMMAND_QUEUE_DESC q{D3D12_COMMAND_LIST_TYPE_DIRECT};
    fx.device->CreateCommandQueue(&q, IID_PPV_ARGS(&fx.queue));
    fx.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&fx.allocator));
    fx.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, fx.allocator.Get(), nullptr, IID_PPV_ARGS(&fx.list));
    fx.list->Close();
    fx.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fx.fence));

    D3D12_ROOT_PARAMETER1 params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 4;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rs{D3D_ROOT_SIGNATURE_VERSION_1_1};
    rs.Desc_1_1.NumParameters = 2;
    rs.Desc_1_1.pParameters = params;
    rs.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
    ComPtr<ID3DBlob> blob, error;
    if (FAILED(D3D12SerializeVersionedRootSignature(&rs, &blob, &error)) ||
        FAILED(fx.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&fx.root)))) {
        std::printf("root signature refused\n");
        return 1;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC p{};
    p.pRootSignature = fx.root.Get();
    p.CS = {startup_gpu_heap_cs, sizeof(startup_gpu_heap_cs)};
    if (FAILED(fx.device->CreateComputePipelineState(&p, IID_PPV_ARGS(&fx.pso)))) {
        std::printf("pipeline refused\n");
        return 1;
    }
    fx.window = buffer(memory_bytes + window_offset, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    fx.result = buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    fx.readback = buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    {
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC t{};
        t.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        t.Width = 4;
        t.Height = 4;
        t.DepthOrArraySize = 1;
        t.MipLevels = 1;
        t.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        t.SampleDesc.Count = 1;
        fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &t, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&fx.texture));
    }
    fx.memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, memory_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    fx.heap_memory = static_cast<unsigned char*>(VirtualAlloc(nullptr, u64{title_slots} * 32, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    for (u64 i = 0; i < memory_bytes / 4; ++i) {
        const std::uint32_t v = pattern(i);
        std::memcpy(fx.memory + i * 4, &v, 4);
    }
    fill_window();
    return 0;
}

Env env()
{
    Env e;
    e.device = fx.device.Get();
    e.coherent_window = [](u64 address, u64 bytes, ID3D12Resource* expected, u64 offset) {
        return fx.coherent && address >= title(0) && address - title(0) < memory_bytes &&
               bytes <= memory_bytes - (address - title(0)) && offset == address - title(0) + window_offset &&
               expected == fx.window.Get();
    };
    e.find = find;
    e.record = record;
    e.resource = resource;
    e.placement = [](u64 wrapper, const D3D12_SHADER_RESOURCE_VIEW_DESC& srv, u32 (&words)[8]) {
        if (!wrapper || wrapper > fx.resources.size()) return false;
        const auto& placed = fx.resources[wrapper - 1];
        if (!placed.desktop || placed.buffer || !placed.title_memory) return false;
        auto desc = placed.desktop->GetDesc();
        desc.Flags = D3D12_RESOURCE_FLAGS(0x300000);
        desc.Layout = D3D12_TEXTURE_LAYOUT(0x105);
        desc.Alignment = 0;
        return x2w_late_texture::encode(desc, srv, placed.title_memory, words);
    };
    e.local_room = [](ID3D12Device*) { return fixture_room; };
    e.copied_bytes = []() -> u64 { return fx.uploads.size() * 4096; };
    e.mapping_epoch = [] { return fixture_epoch; };
    e.allocation_size = [](ID3D12Device*, const u32*) -> u64 { ++allocation_probes; return 65536; };
    e.upload = upload;
    e.late_texture = [](const u32* w, D3D12_SHADER_RESOURCE_VIEW_DESC& srv, bool& pending, bool allow_create) -> u64 {
        D3D12_RESOURCE_DESC desc{};
        x2w_late_texture::decode(w, desc, srv);
        if (fixture_remap_on_probe) {
            ++fixture_epoch;
            fixture_remap_on_probe = false;
        }
        if (fixture_needs_allocation) {
            if (!allow_create) return 0;
            ++fixture_creations;
        }
        if (fixture_blocked_address && (u64{w[2]} << 32 | w[1]) == fixture_blocked_address) {
            pending = true;
            return 0;
        }
        x2w_late_texture::Resolution why{};
        const auto resource = x2w_late_texture::resolve([] { return mapped_texture; },
            [] { ++deferred_probes; return fixture_decline; }, [] { return u64{}; }, why);
        pending = why == x2w_late_texture::Resolution::unavailable || (resource && partial_clamp > 0);
        if (resource) {
            ++resolved_probes;
            srv.Texture2D.ResourceMinLODClamp = partial_clamp;
        }
        return resource;
    };
    return e;
}

#include "startup_gpu_memory_store.inl"
#include "startup_gpu_compute_trace.inl"
#include "startup_gpu_ambient.inl"
#include "startup_gpu_resident_mips.inl"
#include "startup_gpu_small_texture_heap.inl"
#include "startup_gpu_list_states.inl"
bool packed_light_test(DXGI_FORMAT reader_format)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 17;
    desc.Height = 7;
    desc.MipLevels = desc.DepthOrArraySize = 2;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32_UINT;
    const bool render_target = reader_format == DXGI_FORMAT_R32_FLOAT;
    const auto writer_state = render_target ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    desc.Flags = render_target ? D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    ComPtr<ID3D12Resource> writer, other_writer, reader;
    if (FAILED(fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&writer)))) return false;
    if (FAILED(fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&other_writer)))) return false;
    desc.Format = reader_format;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(fx.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr, IID_PPV_ARGS(&reader)))) return false;
    const auto writer_desc = writer->GetDesc();
    auto invalid = desc;
    invalid.Format = DXGI_FORMAT_D32_FLOAT;
    if (x2w_packed_light::compatible(writer_desc, invalid)) return false;
    invalid.Format = DXGI_FORMAT_NV12;
    if (x2w_packed_light::compatible(writer_desc, invalid)) return false;
    invalid.Format = DXGI_FORMAT_R16_UNORM;
    if (x2w_packed_light::compatible(writer_desc, invalid)) return false;
    invalid = desc;
    invalid.MipLevels = 1;
    if (x2w_packed_light::compatible(writer_desc, invalid)) return false;
    x2w_packed_light::Copy copy;
    if (!copy.init(fx.device.Get(), writer.Get(), reader.Get()) || copy.init(fx.device.Get(), reader.Get(), writer.Get()))
        return false;
    const auto bytes = copy.scratch->GetDesc().Width;
    auto tiny = buffer(4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    x2w_packed_light::Copy larger, shared, alternate;
    if (!tiny || !larger.init(fx.device.Get(), writer.Get(), reader.Get(), tiny.Get()) ||
        larger.scratch.Get() == tiny.Get() ||
        !shared.init(fx.device.Get(), writer.Get(), reader.Get(), copy.scratch.Get()) ||
        shared.scratch.Get() != copy.scratch.Get()) return false;
    if (!alternate.init(fx.device.Get(), other_writer.Get(), reader.Get(), copy.scratch.Get())) return false;
    auto input = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!input || !output) return false;
    auto pattern = [](UINT part, UINT row, UINT x) { return 0x88402010u + part * 8192u + row * 32u + x; };
    void* cpu = nullptr;
    if (FAILED(input->Map(0, nullptr, &cpu))) return false;
    for (UINT i = 0; i < copy.source_rows.size(); ++i) {
        const auto& r = copy.source_rows[i];
        for (UINT y = 0; y < r.Footprint.Height; ++y) {
            auto* row = reinterpret_cast<UINT*>(static_cast<unsigned char*>(cpu) + r.Offset + y*r.Footprint.RowPitch);
            for (UINT x = 0; x < r.Footprint.Width; ++x) row[x] = pattern(i,y,x);
        }
    }
    input->Unmap(0, nullptr);
    begin();
    std::map<ID3D12Resource*, D3D12_RESOURCE_STATES> states;
    auto require = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES after) {
        const auto before = states[resource];
        x2w_gpu::transition_all(fx.list.Get(), resource, before, after);
        states[resource] = after;
    };
    require(writer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    auto transfer = [&](bool upload) {
        const auto& rows = upload ? copy.source_rows : copy.dest_rows;
        for (UINT i = 0; i < rows.size(); ++i) {
            D3D12_TEXTURE_COPY_LOCATION buf{}, tex{};
            buf.pResource = upload ? input.Get() : output.Get();
            buf.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            buf.PlacedFootprint = rows[i];
            tex.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            tex.pResource = upload ? writer.Get() : reader.Get();
            tex.SubresourceIndex = i;
            if (upload) fx.list->CopyTextureRegion(&tex,0,0,0,&buf,nullptr);
            else fx.list->CopyTextureRegion(&buf,0,0,0,&tex,nullptr);
        }
    };
    x2w_packed_light::BoundCopy bound;
    const auto record = [&](x2w_packed_light::Copy& c) {
        if (bound.contains(c.source.Get(), c.destination.Get())) return false;
        c.record(fx.list.Get(), require, states[c.source.Get()]);
        bound = {c.source.Get(), c.destination.Get()};
        return true;
    };
    if (!record(alternate) || record(alternate)) return false;
    if (!record(copy) || record(shared)) return false;
    bound.invalidate(writer.Get());
    transfer(true);
    require(writer.Get(), writer_state);
    if (!record(copy) || record(shared)) return false;
    if (states[writer.Get()] != writer_state) return false;
    require(writer.Get(), D3D12_RESOURCE_STATE_COMMON);
    bound.invalidate(reader.Get());
    if (!record(shared)) return false;
    require(reader.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
    transfer(false);
    // At list close the packed-light source and scratch already rest in COMMON.
    // Only the reader should need a return barrier.
    unsigned close_barriers = 0;
    for (const auto& entry : states)
        if (x2w_gpu::transition_all(fx.list.Get(), entry.first, entry.second, D3D12_RESOURCE_STATE_COMMON))
            ++close_barriers;
    if (close_barriers != 1) return false;
    end_and_wait();
    if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(output->Map(0, nullptr, &cpu))) return false;
    bool match = states[writer.Get()] == D3D12_RESOURCE_STATE_COMMON;
    for (UINT i = 0; i < copy.dest_rows.size(); ++i) {
        const auto& r = copy.dest_rows[i];
        for (UINT y = 0; y < r.Footprint.Height; ++y) {
            auto* row = reinterpret_cast<const UINT*>(static_cast<unsigned char*>(cpu) + r.Offset + y*r.Footprint.RowPitch);
            for (UINT x = 0; x < r.Footprint.Width; ++x) match &= row[x] == pattern(i,y,x);
        }
    }
    output->Unmap(0, nullptr);
    if (FAILED(input->Map(0, nullptr, &cpu))) return false;
    constexpr UINT replacement = 0x77ab1122u;
    *static_cast<UINT*>(cpu) = replacement;
    input->Unmap(0, nullptr);
    begin();
    states.clear();
    x2w_packed_light::BoundCopy next_list;
    if (next_list.contains(writer.Get(), reader.Get())) return false;
    require(writer.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
    transfer(true);
    require(writer.Get(), D3D12_RESOURCE_STATE_COMMON);
    shared.record(fx.list.Get(), require, D3D12_RESOURCE_STATE_COMMON);
    require(reader.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
    transfer(false);
    require(reader.Get(), D3D12_RESOURCE_STATE_COMMON);
    end_and_wait();
    if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(output->Map(0, nullptr, &cpu))) return false;
    match &= *static_cast<const UINT*>(cpu) == replacement;
    output->Unmap(0, nullptr);
    if (reader_format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
        x2w_compute_trace::Bindings capture;
        x2w_compute_trace::Binding binding{reader.Get(), 1, 16};
        binding.slice = binding.mip = 1;
        binding.view_format = reader_format;
        capture.add(binding, false, true);
        x2w_compute_trace::Image image;
        if (capture.count != 1 || !image.init(fx.device.Get(), capture.values[0])) return false;
        begin();
        require(reader.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        image.record(fx.list.Get(), false);
        require(reader.Get(), D3D12_RESOURCE_STATE_COMMON);
        end_and_wait();
        if (FAILED(fx.device->GetDeviceRemovedReason()) || FAILED(image.pre->Map(0, nullptr, &cpu))) return false;
        for (UINT y = 0; y < image.height(); ++y) {
            auto* words = reinterpret_cast<const UINT*>(image.row(static_cast<const unsigned char*>(cpu), y, 0));
            for (UINT x = 0; x < image.width(); ++x) match &= words[x] == pattern(3, y + image.box.top, x + image.box.left);
        }
        image.pre->Unmap(0, nullptr);
    }
    if (reader_format == DXGI_FORMAT_R32_FLOAT) {
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        ComPtr<ID3D12DescriptorHeap> views;
        if (FAILED(fx.device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&views)))) return false;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = reader_format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2DArray.ArraySize = srv.Texture2DArray.MipLevels = 2;
        fx.device->CreateShaderResourceView(reader.Get(), &srv, views->GetCPUDescriptorHandleForHeapStart());
        const auto observed = pick(views.Get(), 8, 0, 4);
        for (UINT i = 0; i < observed.size(); ++i) match &= observed[i] == pattern(i, 1, 1);
    }
    return match;
}

}  // namespace

int main(int argc, char** argv)
{
    const std::string backend = argc > 1 ? argv[1] : "warp";
    const std::string mode = argc > 2 ? argv[2] : "twins";
    const bool warp = backend == "warp";
    if (const int r = setup(warp)) return r;
    if (mode == "small_texture_heap") return small_texture_heap_test() ? 0 : 1;
    if (mode == "list_states") return list_states_test() ? 0 : 1;
    if (mode == "resident_mips") return resident_mip_test() ? 0 : 1;
    if (mode == "ambient") return ambient_lighting_test() ? 0 : 1;
    if (mode == "compute_priority_depth_capture") return compute_priority_depth_capture_test() ? 0 : 1;
    if (mode == "compute_volume_capture") return compute_texture_capture_test(33) ? 0 : 1;
    if (mode == "compute_sparse_volume_capture") return compute_texture_capture_test(65) ? 0 : 1;
    if (mode == "compute_texture_capture") return compute_texture_capture_test() ? 0 : 1;
    if (mode == "material_record") return material_record_test() ? 0 : 1;
    if (mode == "pixel_subresource_capture") return pixel_subresource_capture_test() ? 0 : 1;
    if (mode == "packed_light") return packed_light_test(DXGI_FORMAT_R9G9B9E5_SHAREDEXP) && packed_light_test(DXGI_FORMAT_R32_FLOAT) ? 0 : 1;
    if (mode == "views" && vendor == 0x1002) {
        const D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                                       D3D12_MEMORY_POOL_L0, 1, 1};
        const auto desc = fx.window->GetDesc();
        ComPtr<ID3D12Resource> shared;
        if (FAILED(fx.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr, IID_PPV_ARGS(&shared)))) return 1;
        void* data = nullptr;
        if (FAILED(shared->Map(0, nullptr, &data))) return 1;
        auto* title_bytes = static_cast<unsigned char*>(data) + window_offset;
        std::memcpy(title_bytes, fx.memory, memory_bytes);
        VirtualFree(fx.memory, 0, MEM_RELEASE);
        fx.memory = title_bytes;
        fx.window = shared;
        fx.coherent = true;
    }
    HostHeap::env_log() = [](const char* text) { std::printf("%s\n", text); };

    Limits limits = probe_view_limit(fx.device.Get(), true);
    std::printf("probe: views %u raw %u\n", limits.views, limits.raw);
    if (warp) {
        CHECK(limits.raw == 0);  // WARP refuses 2^22 views
        limits = Limits{1u << 20, 1u << 20};  // half the production heap sizes (fixtures only)
    } else if (limits.raw != raw_twins_first) {
        std::printf("the host takes no heap of 2^22 views: skipped\n");
        return 77;
    }
    auto heap = HostHeap::create(env(), false, title_slots, limits, reinterpret_cast<u64>(fx.heap_memory), 32);
    if (!heap) {
        std::printf("mirror heap refused\n");
        return 1;
    }
    const Layout& l = heap->layout();
    std::printf("layout: size %u count %u dynamic %u null %u raw %u twins %u\n", l.size, l.count, l.dynamic, l.null, l.raw, l.twins);
    CHECK(l.size == 2 * limits.raw && l.null == limits.raw - 1 && l.count == title_slots);
    if (!warp) CHECK(l.size == (1u << 22) && l.raw == (1u << 21));

    // Fixture views, written as the title's view calls leave them (a token in the title slot), then mirrored.
    const u64 b0 = buffer_wrapper(0, 4096), b1 = buffer_wrapper(0x10000, 4096), b2 = buffer_wrapper(0x4008, 48);
    const u64 tex = texture_wrapper();
    D3D12_SHADER_RESOURCE_VIEW_DESC as{};
    as.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    as.RaytracingAccelerationStructure.Location = 0x123400000ull;
    note(0, 18, 0, &as, sizeof(as));
    std::uint32_t words[8];
    std::memcpy(words, fx.heap_memory, sizeof(words));
    u64 address = 0;
    CHECK(title_as_address(env(), words, &address) && address == as.RaytracingAccelerationStructure.Location);
    View native{};
    CHECK(!view_of_record(env(), words, false, native));
    CHECK(x2w_as_address::annotate(words, 0x432100000ull));
    std::uint32_t copied[8];
    std::memcpy(copied, words, sizeof(copied));
    CHECK(title_as_address(env(), copied, &address) && address == 0x432100000ull);
    CHECK(x2w_as_address::annotate(words, 0));
    CHECK(title_as_address(env(), words, &address) && address == 0);
    auto d0 = buffer_srv(static_cast<DXGI_FORMAT>(42), 4, 10, 0, false);   // R32_UINT, dwords 4..13
    note(0, 18, b0, &d0, sizeof(d0));
    std::memcpy(words, fx.heap_memory, sizeof(words));
    CHECK(!title_as_address(env(), words, &address));
    auto d1 = buffer_srv(static_cast<DXGI_FORMAT>(57), 8, 7, 0, false);    // R16_UINT, bytes 16..29
    note(1, 18, b0, &d1, sizeof(d1));
    auto d2 = buffer_srv(DXGI_FORMAT_UNKNOWN, 2, 3, 24, false);            // structured 24: bytes 48..119
    note(2, 18, b0, &d2, sizeof(d2));
    D3D12_SHADER_RESOURCE_VIEW_DESC dt{};
    dt.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dt.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    dt.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    dt.Texture2D.MipLevels = 1;
    note(3, 18, tex, &dt, sizeof(dt));
    D3D12_UNORDERED_ACCESS_VIEW_DESC du{};
    du.Format = DXGI_FORMAT_R32_TYPELESS;
    du.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    du.Buffer.FirstElement = 16;
    du.Buffer.NumElements = 16;
    du.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    note(4, 19, b1, &du, sizeof(du));                                        // raw UAV: bytes 64..127 of b1
    D3D12_CONSTANT_BUFFER_VIEW_DESC c5{title(0x2000), 256};
    note(5, 17, 0, &c5, sizeof(c5));                                         // on 256 in the window
    D3D12_CONSTANT_BUFFER_VIEW_DESC c6{title(0x2010), 256};
    note(6, 17, 0, &c6, sizeof(c6));                                         // not on 256: null
    // slot 7: nothing
    {
        x2w_startup::BufferViewLayout bl = x2w_startup::buffer_view_layout(DXGI_FORMAT_R32_TYPELESS, 0, 16, 0, 1);
        std::uint32_t w[8];
        x2w_startup::gfx10_buffer_words(bl, w);
        const u64 base = title(0x3000);
        w[0] = static_cast<std::uint32_t>(base);
        w[1] |= static_cast<std::uint32_t>(base >> 32) & 0xffff;
        std::memcpy(fx.heap_memory + 8 * 32, w, 32);                         // slot 8: a V# of 64 raw bytes
    }
    auto d9 = buffer_srv(DXGI_FORMAT_UNKNOWN, 0, 4, 12, false);              // structured 12 at 0x4008: off elements
    note(9, 18, b2, &d9, sizeof(d9));
    auto d10 = buffer_srv(static_cast<DXGI_FORMAT>(42), 1, 10, 0, false);  // R32_UINT from byte 4: off 16
    note(10, 18, b0, &d10, sizeof(d10));
    for (u32 i = 0; i < 11; ++i) heap->mirror(i);

    if (mode == "memory_store") {
        store_fixture(*heap);
    } else if (mode == "memory" || mode == "memory_empty") {
        // Raw misses at the front of GPU feedback must not consume the
        // mapping budget before the first reachable page.
        x2w_vmemory::FaultPages feedback;
        std::vector<UINT> faults;
        for (UINT page = 65536; page <= 65536 + 2048; ++page) faults.push_back(page);
        CHECK(feedback.report(faults) == 0);
        feedback.seed(1, [&](UINT page) { return page == faults.back(); });
        CHECK(feedback.report({}) == 1);
        faults.resize(129);
        feedback.report(faults);
        feedback.seed(2, [](UINT) { return true; });
        CHECK(feedback.report({}) == 129);
        x2w_vmemory::Recording memory;
        if (!memory.create(fx.device.Get())) return 1;
        memory.base = 48;
        heap->null_raw_uav_at(memory.base);
        ComPtr<ID3D12RootSignature> root;
        CHECK(SUCCEEDED(fx.device->CreateRootSignature(0, startup_gpu_memory_cs, sizeof(startup_gpu_memory_cs), IID_PPV_ARGS(&root))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.Get();
        pd.CS = {startup_gpu_memory_cs, sizeof(startup_gpu_memory_cs)};
        ComPtr<ID3D12PipelineState> pso;
        CHECK(SUCCEEDED(fx.device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso))));
        if (failures) return 1;
        begin();
        std::unordered_map<ID3D12Resource*, D3D12_RESOURCE_STATES> states;
        states[fx.result.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        const auto state = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES next) {
            auto& before = states[r];
            if (before == next) return;
            D3D12_RESOURCE_BARRIER b{};
            b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, next};
            fx.list->ResourceBarrier(1, &b);
            before = next;
        };
        const UINT first = 65552;
        // Simulate residue from a previous submission of this same native list.
        auto previous = buffer(8, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        UINT* residue = nullptr;
        CHECK(previous && SUCCEEDED(previous->Map(0, nullptr, reinterpret_cast<void**>(&residue))));
        if (!residue) return 1;
        residue[0] = 0x80000001;
        residue[1] = window_offset;
        previous->Unmap(0, nullptr);
        state(memory.directory.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        fx.list->CopyBufferRegion(memory.directory.Get(), UINT64{first + 1} * 8, previous.Get(), 0, 8);
        state(memory.faults.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        fx.list->CopyBufferRegion(memory.faults.Get(), 4, previous.Get(), 0, 8);
        if (mode == "memory") {
            memory.map(fx.list.Get(), first, [&](UINT64 at, ID3D12Resource** w, UINT64* o) {
                *w = fx.window.Get(); *o = window_offset; return at == UINT64{first} * 65536;
            }, [&](UINT i, auto* w, UINT64 size) { heap->raw_uav_at(i, w, 0, size); }, state);
        }
        memory.ready(fx.list.Get(), state);
        auto* descriptor_heap = heap->heap();
        fx.list->SetDescriptorHeaps(1, &descriptor_heap);
        fx.list->SetComputeRootSignature(root.Get());
        fx.list->SetPipelineState(pso.Get());
        UINT params[]{memory.base, first};
        fx.list->SetComputeRoot32BitConstants(0, 2, params, 0);
        fx.list->SetComputeRootShaderResourceView(1, memory.directory->GetGPUVirtualAddress());
        fx.list->SetComputeRootUnorderedAccessView(2, memory.faults->GetGPUVirtualAddress());
        fx.list->SetComputeRootUnorderedAccessView(3, fx.result->GetGPUVirtualAddress());
        fx.list->Dispatch(2, 1, 1);
        memory.close(fx.list.Get(), state);
        state(fx.result.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        fx.list->CopyBufferRegion(fx.readback.Get(), 0, fx.result.Get(), 0, 16);
        end_and_wait();
        UINT* out = nullptr;
        CHECK(SUCCEEDED(fx.readback->Map(0, nullptr, reinterpret_cast<void**>(&out))));
        if (!out) return 1;
        CHECK(out[0] == (mode == "memory" ? pattern(4) : 0) && out[1] == 0);
        CHECK(out[2] == (mode == "memory" ? 0x80000001u : 0u) && out[3] == 0);
        fx.readback->Unmap(0, nullptr);
        const std::vector<UINT> expected = mode == "memory" ? std::vector<UINT>{first + 1}
                                                            : std::vector<UINT>{first, first + 1};
        CHECK(memory.read_faults() == expected);
    } else if (mode == "twins") {
        CHECK(title_then_zero(pick(*heap, 0, l.twin(0), 16), 16, 10));      // typed R32: dwords 4..13, bounded
        CHECK(title_then_zero(pick(*heap, 0, l.twin(1), 8), 16, 4));        // typed R16: 4 dwords from byte 16
        CHECK(title_then_zero(pick(*heap, 0, l.twin(2), 24), 48, 18));      // structured: its 72 bytes
        CHECK(all_zero(pick(*heap, 0, l.twin(3), 8)));                       // a texture slot's twin
        CHECK(all_zero(pick(*heap, 0, l.twin(7), 8)));                       // an empty slot's twin
        CHECK(all_zero(pick(*heap, 0, l.size - 1, 8)));                      // the last slot (past every twin)
        CHECK(all_zero(pick(*heap, 0, l.twin(title_slots + l.dynamic - 1), 8)));  // the last twin, of no view
        CHECK(title_then_zero(pick(*heap, 2, l.twin(5), 16), 0x2000, 16));  // a CBV's twin: the CBV
        CHECK(title_then_zero(pick(*heap, 0, l.twin(8), 20), 0x3000, 16));  // a V#'s twin
        // The UAV's twin writes its 16 dwords and nothing beside them.
        const auto before = read_window(0x10000, 64);
        pick(*heap, 1, l.twin(4), 16, 0xa0000000u);
        const auto after = read_window(0x10000, 64);
        u32 changed = 0, outside = 0;
        for (u32 i = 0; i < 64; ++i) {
            if (after[i] != before[i]) ++changed;
            if (after[i] != before[i] && (i < 16 || i >= 32)) ++outside;
            if (i >= 16 && i < 32) CHECK(after[i] == 0xa0000000u + (i - 16));
        }
        std::printf("uav twin: %u dwords changed, %u outside the view\n", changed, outside);
        CHECK(changed == 16 && outside == 0);
        // A dynamic slot made from slot 0's view has its twin too.
        const auto dyn = heap->dynamic(4);
        CHECK(dyn && *dyn == title_slots);
        if (dyn) {
            View v;
            std::uint32_t words[8];
            std::memcpy(words, fx.heap_memory, 32);
            CHECK(view_of_record(env(), words, false, v));
            heap->make_at(*dyn + 1, v);
            CHECK(title_then_zero(pick(*heap, 0, l.twin(*dyn + 1), 16), 16, 10));
            heap->raw_at(*dyn + 2, fx.window.Get(), window_offset + 0x100, 32);
            CHECK(title_then_zero(pick(*heap, 0, *dyn + 2, 12), 0x100, 8));  // a root SRV with a size
            CHECK(all_zero(pick(*heap, 0, l.twin(*dyn + 2), 4)));            // bound by handle: no twin
            heap->release(*dyn, 4);
            CHECK(heap->dynamic(4).has_value());
        }
        std::printf("twins %llu, moved %llu, unmoved %llu, null %llu\n", counted(c_twins), counted(c_twins_moved),
                    counted(c_twins_unmoved), counted(c_twins_null));
        // A view that begins between two multiples of 16 (slot 10, from byte 4): its twin is moved on where the
        // host's raw descriptor is its address and bytes (host.rs move_raw: AMD), else null.
        if (counted(c_twins_moved)) {
            CHECK(title_then_zero(pick(*heap, 0, l.twin(10), 16), 4, 10));
        } else {
            CHECK(vendor != 0x1002);  // AMD must move it
            CHECK(all_zero(pick(*heap, 0, l.twin(10), 16)));
        }
    } else if (mode == "views") {
        // The structured view is raw (raw_window) at the title slot: read as bytes.
        CHECK(title_then_zero(pick(*heap, 0, 2, 20), 48, 18));
        CHECK(title_then_zero(pick(*heap, 3, 0, 12), 16, 10));               // typed R32_UINT as it is
        CHECK(title_then_zero(pick(*heap, 0, 8, 16), 0x3000, 16));           // the V# record: a raw view
        CHECK(title_then_zero(pick(*heap, 2, 5, 16), 0x2000, 16));           // the aligned CBV
        CHECK(all_zero(pick(*heap, 2, 6, 16)));                              // the unaligned CBV: null
        CHECK(counted(c_cbv_unaligned) >= 1);
        CHECK(title_then_zero(pick(*heap, 0, 9, 12), 0x4008, 12));           // off elements: its own bytes
        if (counted(c_direct_offset)) {
            CHECK(heap->kept_at(9) == fx.window.Get());
            // Write only the desktop buffer after mirroring. A snapshot of the title bytes
            // would still show the pattern read above; both aliases must observe the GPU copy.
            auto update = buffer(64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            UINT* mapped = nullptr;
            CHECK(update && SUCCEEDED(update->Map(0, nullptr, reinterpret_cast<void**>(&mapped))));
            if (mapped) {
                for (UINT i = 0; i < 16; ++i) mapped[i] = 0xaedd0000u + i;
                update->Unmap(0, nullptr);
                begin();
                fx.list->CopyBufferRegion(fx.window.Get(), window_offset + 0x4008, update.Get(), 0, 64);
                end_and_wait();
                for (const auto slot : {9u, l.twin(9)}) {
                    const auto data = pick(*heap, 0, slot, 16);
                    for (UINT i = 0; i < 16; ++i) CHECK(data[i] == (i < 12 ? 0xaedd0000u + i : 0));
                }
                const UINT cpu_write = 0xad008800;
                std::memcpy(fx.memory + 0x4008, &cpu_write, sizeof(cpu_write));
                CHECK(pick(*heap, 0, 9, 1)[0] == cpu_write);
            }
        } else {
            CHECK(vendor != 0x1002 && counted(c_exact) >= 1);
        }
        // A cache hit is a real resource lease. Destroying the heap's cache drops its ownership.
        Microsoft::WRL::ComPtr<ID3D12Resource> lease;
        {
            Exacts local;
            lease = local.get(env(), title(0x4008), 48, 1);
            CHECK(lease && local.get(env(), title(0x4008), 48, 2).Get() == lease.Get());
            CHECK(!local.get(env(), title(0x4010), 64ull << 20, 3) && counted(c_exact_capacity));
            CHECK(!local.get(env(), 1, 32, 3) && counted(c_exact_unreadable));
        }
        if (lease) {
            lease->AddRef();
            CHECK(lease->Release() == 1);
        }
        // The table of strides: typed 4 (integer), R16 2 (integer), structured 24, raw UAV 1; none for others.
        CHECK(heap->stride_word(0) == (4 | view_integer));
        CHECK(heap->stride_word(1) == (2 | view_integer));
        CHECK(heap->stride_word(2) == 24);
        CHECK(heap->stride_word(4) == 1);
        CHECK(heap->stride_word(3) == 0 && heap->stride_word(5) == 0 && heap->stride_word(7) == 0);
        CHECK(heap->stride_word(8) == 1);
        CHECK(heap->strides_gpu() != 0);
        const auto slots = heap->buffer_slots();
        CHECK(slots.size() == 7);  // 0, 1, 2, 4, 8, 9, 10
    } else if (mode == "typed_null") {
        auto srv = D3D12_SHADER_RESOURCE_VIEW_DESC{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        srv.Format = DXGI_FORMAT_R32_UINT;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2DArray.MipLevels = 1;
        srv.Texture2DArray.ArraySize = 1;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        uav.Format = DXGI_FORMAT_R32_UINT;
        uav.Texture2DArray.ArraySize = 1;
        note(20, 18, 0, &srv, sizeof(srv));
        const auto id = note(21, 19, 0, &uav, sizeof(uav));
        for (u32 i = 20; i <= 21; ++i) {
            std::memcpy(words, fx.heap_memory + i * 32, sizeof(words));
            View v{};
            CHECK(view_of_record(env(), words, false, v) && v.explicit_null);
            if (i == 20) CHECK(v.kind == Kind::shader && v.srv.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
            else CHECK(v.kind == Kind::unordered && v.uav.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
            CHECK(heap->mirror(i));
            CHECK(heap->make_listed(l.count + i, words).has_value());
            CHECK(all_zero(pick(*heap, i == 20 ? 4 : 5, i, 4)));
            CHECK(all_zero(pick(*heap, i == 20 ? 4 : 5, l.count + i, 4)));
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC buffer_null{};
        buffer_null.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        buffer_null.Format = DXGI_FORMAT_R32_UINT;
        buffer_null.Buffer.NumElements = 4;
        note(22, 19, 0, &buffer_null, sizeof(buffer_null));
        CHECK(heap->mirror(22));
        CHECK(all_zero(pick(*heap, 6, l.twin(22), 4)));
        fx.records[id].materialized = false;
        View rejected{};
        CHECK(!view_of_record(env(), words, false, rejected));
        fx.records[id].materialized = true;
        fx.records[id].resource = 0x2208759ad90ull;
        CHECK(!view_of_record(env(), words, false, rejected));
        CHECK(SUCCEEDED(fx.device->GetDeviceRemovedReason()));
    } else if (mode == "submit") {
        // The title copies slot 0's record into slot 20 and slot 2's into 21 as bytes, and clears slot 1's.
        std::memcpy(fx.heap_memory + 20 * 32, fx.heap_memory, 32);
        std::memcpy(fx.heap_memory + 21 * 32, fx.heap_memory + 2 * 32, 32);
        CHECK(all_zero(pick(*heap, 0, l.twin(20), 8)));                      // not mirrored yet
        CHECK(title_then_zero(pick(*heap, 0, l.twin(1), 8), 16, 4));
        std::memset(fx.heap_memory + 1 * 32, 0, 32);
        const u64 changed = heap->submit();
        std::printf("submit: %llu records changed\n", changed);
        CHECK(changed == 3);  // 20 and 21 copied, 1 cleared (the others were mirrored by their calls)
        CHECK(title_then_zero(pick(*heap, 0, l.twin(20), 16), 16, 10));
        CHECK(title_then_zero(pick(*heap, 0, 21, 20), 48, 18));
        CHECK(all_zero(pick(*heap, 0, l.twin(1), 8)));
        CHECK(heap->submit() == 0);                                          // nothing new
        // Title memory under the own-bytes copy of slot 9 changes: the next submit carries it.
        const std::uint32_t mark = 0xc0ffee00u;
        std::memcpy(fx.memory + 0x4008, &mark, 4);
        heap->submit();
        CHECK(title_then_zero(pick(*heap, 0, 9, 12), 0x4008, 12));
        // Six submits on, with the record unchanged, the copy is still refreshed (a slot shows it).
        for (int i = 0; i < 6; ++i) heap->submit();
        const std::uint32_t mark2 = 0xbeef0000u;
        std::memcpy(fx.memory + 0x400c, &mark2, 4);
        heap->submit();
        {
            const auto got = pick(*heap, 0, 9, 12);
            CHECK(got.size() >= 2 && got[0] == mark && got[1] == mark2);
        }
        // A V# whose memory has no window yet when mirrored (slot 11) shows null, and is made at a later submit.
        {
            x2w_startup::BufferViewLayout bl = x2w_startup::buffer_view_layout(DXGI_FORMAT_R32_TYPELESS, 0, 16, 0, 1);
            std::uint32_t w[8];
            x2w_startup::gfx10_buffer_words(bl, w);
            const u64 base = title(0x3100);
            w[0] = static_cast<std::uint32_t>(base);
            w[1] |= static_cast<std::uint32_t>(base >> 32) & 0xffff;
            std::memcpy(fx.heap_memory + 11 * 32, w, 32);
        }
        fx.find_fails = true;
        heap->mirror(11);
        fx.find_fails = false;
        CHECK(all_zero(pick(*heap, 0, 11, 16)));
        CHECK(heap->shows_none(11));
        heap->submit();
        CHECK(!heap->shows_none(11));
        CHECK(title_then_zero(pick(*heap, 0, 11, 16), 0x3100, 16));
        // The texture of slot 3 is due for upload.
        CHECK(heap->upload_due(nullptr, 16) >= 1);
        bool texture_uploaded = false;
        for (u64 w : fx.uploads) texture_uploaded |= w == tex;
        CHECK(texture_uploaded);
        // The texture isn't mapped until after its descriptor is copied. A slot the
        // title clears before the mapping appears must remain empty.
        D3D12_RESOURCE_DESC placed = fx.texture->GetDesc();
        placed.Flags = D3D12_RESOURCE_FLAGS(0x300000);
        placed.Layout = D3D12_TEXTURE_LAYOUT(0x105);
        placed.Alignment = 0;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = placed.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        u32 late[8]{};
        CHECK(x2w_late_texture::encode(placed, srv, title(0x7000), late));
        for (u32 index : {28u, 29u}) std::memcpy(fx.heap_memory + 32 * index, late, 32);
        heap->submit();
        CHECK(heap->shows_none(28) && heap->shows_none(29));
        std::memset(fx.heap_memory + 32 * 29, 0, 32);
        CHECK(deferred_probes == 2); // first attempt for each copied slot, once per submit
        const unsigned before = deferred_probes;
        mapped_texture = tex;
        heap->submit();
        CHECK(heap->kept_at(28) == fx.texture.Get());
        CHECK(deferred_probes == before);
        // The same title VA and unchanged descriptor can acquire a new host object.
        // Re-reading the slot must replace the descriptor and its retained resource.
        const auto replacement_desc = fx.texture->GetDesc();
        D3D12_HEAP_PROPERTIES props{};
        props.Type = D3D12_HEAP_TYPE_DEFAULT;
        ComPtr<ID3D12Resource> replacement;
        CHECK(SUCCEEDED(fx.device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &replacement_desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&replacement))));
        if (replacement) {
            const auto fill = [&](ID3D12Resource* texture, unsigned char red) {
                auto pixels = buffer(1024, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
                void* cpu = nullptr;
                if (!pixels || FAILED(pixels->Map(0, nullptr, &cpu))) return false;
                std::memset(cpu, red, 1024);
                pixels->Unmap(0, nullptr);
                D3D12_TEXTURE_COPY_LOCATION src{pixels.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
                src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 256};
                D3D12_TEXTURE_COPY_LOCATION dst{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                begin();
                x2w_gpu::transition_all(fx.list.Get(), texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                fx.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                x2w_gpu::transition_all(fx.list.Get(), texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
                end_and_wait();
                return true;
            };
            CHECK(fill(fx.texture.Get(), 51));
            CHECK(fill(replacement.Get(), 204));
            fx.resources[tex - 1].title_memory = title(0x7000);
            note(28, 18, tex, &srv, sizeof(srv));
            CHECK(heap->mirror(28));
            std::memcpy(fx.heap_memory + 32 * 28, late, sizeof(late));
            const auto first_wrapper = mapped_texture;
            mapped_texture = 0;
            CHECK(heap->mirror(28));
            CHECK(heap->kept_at(28) == fx.texture.Get());
            CHECK(heap->make_listed(l.count + 28, late, 28).has_value());
            CHECK(heap->kept_at(l.count + 28) == fx.texture.Get());
            CHECK(pick(*heap, 7, 28, 1)[0] == 51);
            CHECK(pick(*heap, 7, l.count + 28, 1)[0] == 51);
            fixture_decline = 5;
            CHECK(!heap->mirror(28));
            CHECK(!heap->kept_at(28));
            fixture_decline = 8;
            mapped_texture = first_wrapper;
            CHECK(heap->mirror(28));
            mapped_texture = 0;
            // Change the address, resource dimensions, view format or selected mip.
            for (unsigned field : {1u, 3u, 4u, 6u}) {
                u32 changed[8];
                std::copy_n(late, 8, changed);
                changed[field] ^= field == 4 ? 0x100 : 1;
                CHECK(heap->make_listed(l.count + 29, changed, 28).has_value());
                CHECK(!heap->kept_at(l.count + 29));
                CHECK(!heap->mirror_words(28, changed));
                CHECK(!heap->kept_at(28));
                mapped_texture = first_wrapper;
                CHECK(heap->mirror(28));
                mapped_texture = 0;
            }
            CHECK(heap->mirror(28));
            mapped_texture = first_wrapper;
            fx.resources[mapped_texture - 1].desktop = replacement.Get();
            heap->submit();
            CHECK(heap->kept_at(28) == replacement.Get());
            CHECK(heap->make_listed(l.count + 28, late, 28).has_value());
            CHECK(heap->kept_at(l.count + 28) == replacement.Get());
            CHECK(pick(*heap, 7, 28, 1)[0] == 204);
            CHECK(pick(*heap, 7, l.count + 28, 1)[0] == 204);
            u32 lazy[8];
            std::copy_n(late, 8, lazy);
            lazy[0] = x2w_late_texture::lazy;
            {
                u32 malformed[8];
                std::copy_n(lazy, 8, malformed);
                malformed[3] = 2048u | (1024u << 16);
                malformed[4] = (malformed[4] & 0xffffffu) | (193u << 24);
                std::memcpy(fx.heap_memory + 32 * 29, malformed, sizeof(malformed));
                const unsigned allocated = allocation_probes, resolved = resolved_probes;
                CHECK(!heap->mirror(29, false) && !heap->warm_slot(nullptr, 29, true, 1));
                CHECK(!heap->mirror(29, true) && !heap->kept_at(29));
                CHECK(allocated == allocation_probes && resolved == resolved_probes);
            }
            const unsigned before_stage = resolved_probes;
            for (u32 slot : {30u, 31u, 32u, 33u}) {
                std::memcpy(fx.heap_memory + slot * 32, lazy, 32);
                heap->mirror(slot, false);
            }
            heap->submit();
            CHECK(resolved_probes == before_stage);
            CHECK(!heap->kept_at(30));
            CHECK(heap->make_listed(l.count + 30, lazy, 30).has_value());
            CHECK(pick(*heap, 7, l.count + 30, 1)[0] == 204);
            LazyBudget budget;
            std::memset(fx.heap_memory + 32 * 32, 0, 32);
            heap->warm_lazy(nullptr, 1, budget, 1);
            CHECK(heap->kept_at(30) == replacement.Get() && !heap->kept_at(31));
            heap->warm_lazy(nullptr, 1, budget, 1);
            CHECK(!heap->kept_at(31));
            heap->warm_lazy(nullptr, 2, budget, 4);
            CHECK(pick(*heap, 7, 31, 1)[0] == 204 && !heap->kept_at(32));
            CHECK(heap->kept_at(33) == replacement.Get());
            // Resident textures can pass the old 16-slot ceiling. Empty budget and byte exhaustion
            // preserve queued descriptors. A newly mapped page retries in the same presented frame.
            for (u32 slot = 150; slot < 184; ++slot) {
                std::memcpy(fx.heap_memory + slot * 32, lazy, 32);
                heap->mirror(slot, false);
            }
            fixture_room = 0;
            heap->warm_lazy(nullptr, 3, budget);
            CHECK(!budget.visits && !heap->kept_at(150));
            fixture_room = 32768;
            heap->warm_lazy(nullptr, 4, budget, 4096, 1ull << 30, 1000);
            CHECK(budget.visits == 34 && !budget.bytes && !heap->kept_at(150));
            fixture_room = UINT64_MAX;
            heap->warm_lazy(nullptr, 5, budget, 4096, 1ull << 30, 0);
            CHECK(!budget.visits);
            heap->warm_lazy(nullptr, 5, budget, 4096, 20 * 4096, 1000);
            CHECK(budget.visits == 20 && budget.bytes == 20 * 4096);
            CHECK(heap->kept_at(169) == replacement.Get() && !heap->kept_at(170));
            heap->warm_lazy(nullptr, 5, budget, 4096, 20 * 4096, 1000);
            CHECK(budget.visits == 20);
            const auto mapped = mapped_texture;
            mapped_texture = 0;
            heap->warm_lazy(nullptr, 6, budget, 4096, 1ull << 30, 1000);
            CHECK(budget.visits == 14 && !heap->kept_at(183));
            mapped_texture = mapped;
            heap->warm_lazy(nullptr, 6, budget, 4096, 1ull << 30, 1000);
            CHECK(budget.visits == 14);
            ++fixture_epoch;
            heap->warm_lazy(nullptr, 6, budget, 4096, 1ull << 30, 1000);
            CHECK(budget.visits == 28 && heap->kept_at(183) == replacement.Get());
            material_feedback_test(*heap, lazy, replacement.Get());
            partial_clamp = 0.5f;
            CHECK(heap->mirror(28));
            View partial{};
            CHECK(heap->make_listed(l.count + 28, late, 28, &partial).has_value());
            CHECK(partial.pending && partial.srv.Texture2D.ResourceMinLODClamp == 0.5f);
            const auto probes = resolved_probes;
            heap->submit();
            CHECK(resolved_probes > probes);
            partial_clamp = 0;
            const auto before_completion = counted(c_mirrored);
            heap->submit();
            CHECK(counted(c_mirrored) > before_completion);
            View ready{};
            CHECK(heap->make_listed(l.count + 28, late, 28, &ready).has_value());
            CHECK(!ready.pending && ready.srv.Texture2D.ResourceMinLODClamp == 0 && ready.key != partial.key);
        }
        CHECK(heap->shows_none(29) && !heap->kept_at(29));
    } else {
        std::printf("unknown mode %s\n", mode.c_str());
        return 2;
    }
    if (failures) {
        std::printf("startup_gpu_heap %s %s: %d failures\n", backend.c_str(), mode.c_str(), failures);
        return 1;
    }
    std::printf("startup_gpu_heap %s %s: pass\n", backend.c_str(), mode.c_str());
    return 0;
}
