// X2W_D3D12_RELEASE_RETIRED=1 (Slice 2, Refs #2379): slot 133 (CreatePlacedResourceX) committed 2D textures with
// title_memory retire once no recorded list, submission, release-time fence, or tracked slot-18 SRV descriptor slot
// still refers to them, and are cleaned from resource_address_index, texture_uploads and written_textures before
// their desktop ID3D12Resource is destroyed.
//
//   startup_resource_retire_title <warp|hardware> [off]
//
// Cases:
//   1 slot 133 title_memory texture upload/copy (CopyTextureRegion + CopyResource): held behind closed list and
//     gated second submission, read back after copy, then retired to baseline.
//   2 resource_address_index + slot 70 (existing_memory_texture) same-address reuse while T1 is pending behind a
//     held fence, with X2W_RESOURCE_INDEX_CHECK=1 verifying live index consistency.
//   3 written_textures stale bc7_width cleanup on address reuse (BC7 retired -> RGBA8 at same address uploads
//     fresh CPU texels).
//   4 descriptor & view reuse: view_slot<18> + CopyDescriptorsSimple + heap release + descriptor overwrite;
//     stays retained while any tracked slot holds the SRV, retires once overwritten/freed and fences complete.
//   5 switch-off control: an ordinary committed texture (no slot 133, no title memory) copied as a subresource still
//     records its upload, as before the switch (case runs with the switch off, which is what selects the control).
//   6 written_textures generation keeping: a live view keeps the older generation's entry, and a newer generation of
//     the same address survives the older generation's retirement (case runs with the switch on, which is what
//     retires resources).
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <deque>
#include <vector>

using Microsoft::WRL::ComPtr;

struct RetireProbe {
    unsigned cache_entries;
    unsigned list_states;
    unsigned list_holds;
};
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_budget_snapshot(x2w_budget::Snapshot*, SIZE_T);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_retire_probe(void*, void*, RetireProbe*);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_resource_index_probe(void*);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_texture_upload_probe(void*);
extern "C" __declspec(dllimport) BOOL x2w_d3d12_written_probe(void*, unsigned long long*, unsigned*, unsigned*);
extern "C" __declspec(dllimport) void x2w_d3d12_bc7_4k_s_written(void*, UINT, UINT);
extern "C" __declspec(dllimport) void x2w_d3d12_list_copy_texture_stats(LONG*, LONG*);

namespace {
using namespace x2w_budget;

constexpr UINT side = 64, pitch = 256, image_bytes = side * pitch;
constexpr DWORD wait_ms = 10000;
constexpr int settle_limit = 200;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

std::uint8_t pattern_byte(int seed, UINT i) { return static_cast<std::uint8_t>(i * 7 + seed * 13 + 1); }

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

struct Res {
    ComPtr<IUnknown> guest;
    const char* name = "";
    Category category = texture;
    UINT64 bytes = 0;
    bool released = false;
    bool freed = false;
};

struct GuestList {
    ComPtr<IUnknown> allocator, list;
};

struct GuestQueue {
    ComPtr<IUnknown> queue, done;
    ComPtr<ID3D12Fence> done_real;
    UINT64 value = 0;
};

struct MemorySrvInput {
    D3D12_SHADER_RESOURCE_VIEW_DESC view;
    UINT64 address;
};
static_assert(offsetof(MemorySrvInput, address) == 0x28);

struct Context {
    IUnknown* guest;
    ID3D12Device* device;
    ID3D12InfoQueue* debug;
    ComPtr<ID3D12CommandQueue> own_queue;
    ComPtr<ID3D12CommandAllocator> own_allocator;
    ComPtr<ID3D12GraphicsCommandList> own_list;
    ComPtr<ID3D12Fence> own_fence;
    UINT64 own_value = 0;
    HANDLE event = nullptr;
    ComPtr<ID3D12Resource> upload, readback;
    Snapshot base{};
    std::deque<Res> pool;
    GuestQueue direct, gated;
    GuestList drain_list;
    D3D12_RESOURCE_DESC desc{};
    UINT64 texture_bytes = 0;
    std::array<GUID, 1> iid_resource{__uuidof(ID3D12Resource)};

    bool wait(ID3D12Fence* fence, UINT64 value)
    {
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0;
    }
    bool own_run()
    {
        if (!require(own_list->Close(), "own list close")) return false;
        ID3D12CommandList* lists[] = {own_list.Get()};
        own_queue->ExecuteCommandLists(1, lists);
        if (!require(own_queue->Signal(own_fence.Get(), ++own_value), "own signal")) return false;
        if (wait(own_fence.Get(), own_value)) return true;
        std::puts("FAIL: own queue timed out");
        return false;
    }
    D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer, DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM,
                                          UINT width = side) const
    {
        D3D12_TEXTURE_COPY_LOCATION location{buffer, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        location.PlacedFootprint.Footprint = {fmt, width, side, 1, pitch};
        return location;
    }
    bool make_queue(GuestQueue& q)
    {
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        if (!require(invoke(guest, 8, &qd, &__uuidof(ID3D12CommandQueue), q.queue.GetAddressOf()), "queue") ||
            !require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), q.done.GetAddressOf()), "fence"))
            return false;
        q.done_real = desktop_object<ID3D12Fence>(guest, device, q.done.Get());
        return q.done_real != nullptr;
    }
    bool sync(GuestQueue& q)
    {
        if (!require(invoke(q.queue.Get(), 14, q.done.Get(), ++q.value), "queue signal")) return false;
        if (wait(q.done_real.Get(), q.value)) return true;
        std::puts("FAIL: guest queue timed out");
        return false;
    }
    bool make_list(GuestList& l)
    {
        const unsigned list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
        return require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                              l.allocator.GetAddressOf()), "allocator") &&
               require(invoke(guest, 73, list_desc, l.allocator.Get(), nullptr,
                              &__uuidof(ID3D12GraphicsCommandList), l.list.GetAddressOf()), "command list") &&
               require(invoke(l.list.Get(), 139, l.allocator.Get(), nullptr), "list reset");
    }
    bool close(GuestList& l) { return require(invoke(l.list.Get(), 9), "list close"); }
    HRESULT reset_list(GuestList& l)
    {
        ComPtr<IUnknown> next;
        if (!require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator),
                            next.GetAddressOf()), "allocator"))
            return E_FAIL;
        const HRESULT hr = invoke(l.list.Get(), 139, next.Get(), nullptr);
        if (SUCCEEDED(hr)) l.allocator = next;
        return hr;
    }
    void execute(GuestQueue& q, GuestList& l)
    {
        IUnknown* lists[] = {l.list.Get()};
        invoke<void>(q.queue.Get(), 10, 1u, lists);
    }
    Res& make_committed(const char* name, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
    {
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d = desc;
        d.Flags = flags;
        Res& r = pool.emplace_back();
        r.name = name;
        r.bytes = texture_bytes;
        if (!require(invoke(guest, 127, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u,
                            nullptr, &iid_resource[0], r.guest.GetAddressOf()), name))
            throw 4;
        return r;
    }
    Res& make_slot133(const char* name, void* address, DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM, UINT width = side)
    {
        D3D12_RESOURCE_DESC d = desc;
        d.Width = width;
        d.Format = fmt;
        Res& r = pool.emplace_back();
        r.name = name;
        r.bytes = device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        if (!require(invoke(guest, 133, address, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, &iid_resource[0],
                            r.guest.GetAddressOf()), name))
            throw 4;
        return r;
    }
    bool fill_desktop(IUnknown* resource, int seed)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, resource);
        if (!real) return false;
        void* mapped = nullptr;
        const D3D12_RANGE none{};
        if (!require(upload->Map(0, &none, &mapped), "upload map")) return false;
        auto* dst = static_cast<std::uint8_t*>(mapped);
        for (UINT i = 0; i < image_bytes; ++i) dst[i] = pattern_byte(seed, i);
        upload->Unmap(0, nullptr);
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list"))
            return false;
        own_list->CopyTextureRegion(&footprint_tex(real.Get()), 0, 0, 0, &footprint(upload.Get()), nullptr);
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {real.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON};
        own_list->ResourceBarrier(1, &b);
        return own_run();
    }
    D3D12_TEXTURE_COPY_LOCATION footprint_tex(ID3D12Resource* tex) const
    {
        return {tex, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    }
    bool holds(IUnknown* resource, int seed, const char* what, DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM,
               UINT width = side)
    {
        const auto real = desktop_object<ID3D12Resource>(guest, device, resource);
        if (!real) return false;
        if (!require(own_allocator->Reset(), "own allocator") || !require(own_list->Reset(own_allocator.Get(), nullptr), "own list"))
            return false;
        D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_FLAG_NONE, {}};
        b.Transition = {real.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE};
        own_list->ResourceBarrier(1, &b);
        own_list->CopyTextureRegion(&footprint(readback.Get(), fmt, width), 0, 0, 0, &footprint_tex(real.Get()), nullptr);
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        own_list->ResourceBarrier(1, &b);
        if (!own_run()) return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, image_bytes};
        if (!require(readback->Map(0, &all, &mapped), "readback map")) return false;
        const auto* src = static_cast<const std::uint8_t*>(mapped);
        const UINT row_bytes = fmt == DXGI_FORMAT_R8_UNORM ? width : width * 4;
        UINT mismatches = 0;
        for (UINT y = 0; y < side; ++y)
            for (UINT x = 0; x < row_bytes; ++x) {
                const UINT i = y * pitch + x;
                if (src[i] != pattern_byte(seed, i)) ++mismatches;
            }
        readback->Unmap(0, nullptr);
        if (mismatches) {
            std::printf("FAIL: %s readback had %u mismatched bytes\n", what, mismatches);
            return false;
        }
        return true;
    }
    void release(Res& r)
    {
        r.guest.Reset();
        r.released = true;
    }
    void drain_once()
    {
        IUnknown* lists[] = {drain_list.list.Get()};
        invoke<void>(direct.queue.Get(), 10, 1u, lists);
    }
    bool expect(const char* stage, const char* fail) const
    {
        Snapshot now{};
        if (!x2w_d3d12_budget_snapshot(&now, sizeof(now))) return false;
        UINT64 live_b = base.live[texture].bytes, ret_b = base.retained[texture].bytes;
        UINT64 live_n = base.live[texture].objects, ret_n = base.retained[texture].objects;
        for (const Res& r : pool) {
            if (r.category != texture) continue;
            if (!r.released) {
                live_b += r.bytes;
                ++live_n;
            } else if (!r.freed) {
                ret_b += r.bytes;
                ++ret_n;
            }
        }
        const auto& got_live = now.live[texture];
        const auto& got_ret = now.retained[texture];
        if (got_live.bytes == live_b && got_live.objects == live_n &&
            got_ret.bytes == ret_b && got_ret.objects == ret_n)
            return true;
        std::printf("FAIL: %s\n  stage '%s' live=%llu/%llu expected=%llu/%llu retained=%llu/%llu expected=%llu/%llu\n",
                    fail, stage, got_live.bytes, got_live.objects, live_b, live_n,
                    got_ret.bytes, got_ret.objects, ret_b, ret_n);
        return false;
    }
    bool stays(const char* stage, const char* fail)
    {
        drain_once();
        return sync(direct) && expect(stage, fail);
    }
    bool settle(const char* stage, const char* fail)
    {
        for (int i = 0; i < settle_limit; ++i) {
            drain_once();
            Snapshot now{};
            if (!x2w_d3d12_budget_snapshot(&now, sizeof(now))) return false;
            UINT64 ret_n = base.retained[texture].objects;
            for (const Res& r : pool) if (r.released && !r.freed) ++ret_n;
            if (now.retained[texture].objects == ret_n) return expect(stage, fail);
        }
        return expect(stage, fail);
    }
};

void fill_title_memory(void* address, int seed)
{
    auto* dst = static_cast<std::uint8_t*>(address);
    for (UINT i = 0; i < image_bytes; ++i) dst[i] = pattern_byte(seed, i);
}

// 1: slot 133 title_memory texture upload & copy lifetime across closed list and gated submission.
bool case1_slot133_copy_and_upload(Context& c, bool on, void* title_mem)
{
    fill_title_memory(title_mem, 5);
    Res& T1 = c.make_slot133("T1", title_mem);
    auto t1_desktop = desktop_object<ID3D12Resource>(c.guest, c.device, T1.guest.Get());
    if (!t1_desktop) {
        std::puts("FAIL: slot 133 texture has no desktop object");
        return false;
    }
    // The raw pointer only keys the upload cache after retirement: a COM reference here would keep the budget ticket
    // of a retired desktop object alive (the fixture's expected retained count would then not drop).
    ID3D12Resource* const t1_real = t1_desktop.Get();
    t1_desktop.Reset();
    Res& D1 = c.make_committed("D1");
    Res& D2 = c.make_committed("D2");
    if (!c.fill_desktop(D1.guest.Get(), 99) || !c.fill_desktop(D2.guest.Get(), 99)) return false;
    GuestList L1, L2;
    if (!c.make_list(L1) || !c.make_list(L2)) return false;
    const D3D12_TEXTURE_COPY_LOCATION to1{static_cast<ID3D12Resource*>(static_cast<void*>(D1.guest.Get())),
                                          D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const D3D12_TEXTURE_COPY_LOCATION to2{static_cast<ID3D12Resource*>(static_cast<void*>(D2.guest.Get())),
                                          D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(static_cast<void*>(T1.guest.Get())),
                                           D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    invoke<void>(L1.list.Get(), 16, &to1, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    invoke<void>(L2.list.Get(), 17, D2.guest.Get(), T1.guest.Get());
    if (!c.close(L1) || !c.close(L2)) return false;
    // texture_uploads()[T1->real] must be populated when T1 has title_memory.
    RetireProbe probe = {};
    if (on && (!x2w_d3d12_retire_probe(T1.guest.Get(), L1.list.Get(), &probe) || probe.cache_entries == 0)) {
        std::printf("FAIL: slot 133 CopyTextureRegion did not populate texture_uploads (cache_entries=%u)\n", probe.cache_entries);
        return false;
    }
    c.release(T1);
    if (!c.sync(c.direct) || !c.sync(c.gated)) return false;
    if (!x2w_d3d12_texture_upload_probe(t1_real)) {
        std::puts("FAIL: slot 133 CopyTextureRegion recorded no texture_uploads entry for T1");
        return false;
    }
    if (!c.stays("slot 133 released behind closed lists", "early release of slot 133 texture")) return false;
    c.execute(c.direct, L1);
    if (!c.sync(c.direct) || !c.holds(D1.guest.Get(), 5, "D1 CopyTextureRegion from slot 133")) return false;
    if (!SUCCEEDED(c.reset_list(L1))) return false;
    if (!c.stays("slot 133 still held by L2", "early release while L2 closed")) return false;
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate1") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "wait1"))
        return false;
    c.execute(c.gated, L2);
    if (!SUCCEEDED(c.reset_list(L2))) return false;
    if (!c.stays("slot 133 held by gated submission", "early release while gated submission in flight")) return false;
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "signal1") || !c.sync(c.gated)) return false;
    if (on) T1.freed = true;
    if (!c.settle("slot 133 retired", "slot 133 texture kept after fences passed")) return false;
    // The texture_uploads() entry the copy recorded leaves with the desktop object; without the switch nothing
    // retires and the entry stays, as before the switch existed.
    if ((x2w_d3d12_texture_upload_probe(t1_real) != FALSE) == on) {
        std::printf("FAIL: texture_uploads entry for retired T1 is %s\n", on ? "still present" : "missing");
        return false;
    }
    return c.holds(D2.guest.Get(), 5, "D2 CopyResource from slot 133");
}

// 2: resource_address_index + slot 70 existing_memory_texture + same-address reuse while T1 is pending behind fence.
bool case2_same_address_and_slot70_reuse(Context& c, bool on, void* title_mem)
{
    fill_title_memory(title_mem, 11);
    Res& T1 = c.make_slot133("T_addr1", title_mem, DXGI_FORMAT_R8_UNORM, pitch);
    // The wrapper outlives its guest Release (the registry keeps it), so the raw pointer can be probed afterwards.
    IUnknown* const t1_guest = T1.guest.Get();
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate2") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "wait2"))
        return false;
    c.release(T1);
    if (!c.stays("T_addr1 pending behind gate", "early release behind gate2")) return false;
    // Reuse the exact same title_memory address while T1 is in retired_resources.
    fill_title_memory(title_mem, 22);
    Res& T2 = c.make_slot133("T_addr2", title_mem, DXGI_FORMAT_R8_UNORM, pitch);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2}, 0, 0};
    ComPtr<IUnknown> heap;
    if (!require(invoke(c.guest, 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()), "slot70 heap")) return false;
    const UINT inc = c.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT64 h0 = invoke<UINT64>(heap.Get(), 9);
    const UINT64 h1 = h0 + inc;
    D3D12_RESOURCE_DESC r8_desc = c.desc;
    r8_desc.Width = pitch;
    r8_desc.Format = DXGI_FORMAT_R8_UNORM;
    MemorySrvInput srv_in{};
    srv_in.address = reinterpret_cast<UINT64>(title_mem);
    srv_in.view.Format = DXGI_FORMAT_R8_UNORM;
    srv_in.view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_in.view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv_in.view.Texture2D.MipLevels = 1;
    // Slot 70 exercises check_resource_address_index() (X2W_RESOURCE_INDEX_CHECK=1) and existing_memory_texture().
    invoke<void>(c.guest, 70, &r8_desc, &srv_in, h0);
    // Unblock T1: T1 retires cleanly, while T2 (held by slot 70 -> slot 18 SRV in heap[0]) remains live.
    if (!require(invoke(gate.Get(), 10, UINT64{1}), "signal2") || !c.sync(c.gated)) return false;
    if (on) T1.freed = true;
    if (!c.settle("T_addr1 retired while T_addr2 live", "T_addr1 did not retire")) return false;
    // Retirement drops T1's resource_address_index entry (the canonical record stays); without the switch T1 stays
    // live, so its index entry stays too. T2's own entry must be there in both modes.
    if (!x2w_d3d12_resource_index_probe(T2.guest.Get())) {
        std::puts("FAIL: T_addr2 has no resource_address_index entry");
        return false;
    }
    if ((x2w_d3d12_resource_index_probe(t1_guest) != FALSE) == on) {
        std::printf("FAIL: resource_address_index entry for retired T_addr1 is %s\n", on ? "still present" : "missing");
        return false;
    }
    // Verify T2 still uploads seed 22 properly after T1 retired.
    const D3D12_HEAP_PROPERTIES def_heap{D3D12_HEAP_TYPE_DEFAULT};
    Res& D = c.pool.emplace_back();
    D.name = "D_addr2";
    D.bytes = c.device->GetResourceAllocationInfo(0, 1, &r8_desc).SizeInBytes;
    if (!require(invoke(c.guest, 127, &def_heap, D3D12_HEAP_FLAG_NONE, &r8_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                        nullptr, 0u, nullptr, &c.iid_resource[0], D.guest.GetAddressOf()), "D_addr2"))
        return false;
    GuestList L;
    if (!c.make_list(L)) return false;
    const D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(static_cast<void*>(D.guest.Get())),
                                         D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(static_cast<void*>(T2.guest.Get())),
                                          D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    invoke<void>(L.list.Get(), 16, &to, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    if (!c.close(L)) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct) ||
        !c.holds(D.guest.Get(), 22, "T_addr2 after T_addr1 retired", DXGI_FORMAT_R8_UNORM, pitch))
        return false;
    if (!SUCCEEDED(c.reset_list(L))) return false;
    // Also exercise slot 70 once more after T1 retired, then release heap and T2.
    invoke<void>(c.guest, 70, &r8_desc, &srv_in, h1);
    c.release(T2);
    if (!c.stays("T_addr2 kept while slot70 SRV live", "T_addr2 freed while SRV slots live")) return false;
    heap.Reset();
    if (on) T2.freed = true;
    return c.settle("T_addr2 retired after heap released", "T_addr2 kept after heap released");
}

// 3: written_textures stale bc7_width cleanup on address reuse, and preservation of a newer generation.
bool case3_written_textures_cleanup(Context& c, bool on, void* title_mem, void* fallback_mem)
{
    x2w_d3d12_bc7_4k_s_written(title_mem, side, side);
    Res& Tb = c.make_slot133("T_bc7", title_mem, DXGI_FORMAT_BC7_UNORM);
    c.release(Tb);
    if (!c.sync(c.direct) || !c.sync(c.gated)) return false;
    if (on) Tb.freed = true;
    if (!c.settle("T_bc7 retired", "T_bc7 kept")) return false;

    // When retire is on, reusing title_mem for plain RGBA8 (no XG call) must upload fresh texels instead of
    // skipping on stale bc7_width; in off mode Tb stayed retained at title_mem so use fallback_mem.
    void* rgba_mem = on ? title_mem : fallback_mem;
    fill_title_memory(rgba_mem, 33);
    Res& Tc = c.make_slot133("T_rgba_after_bc7", rgba_mem, DXGI_FORMAT_R8G8B8A8_UNORM);
    Res& Dc = c.make_committed("D_rgba_after_bc7");
    if (!c.fill_desktop(Dc.guest.Get(), 99)) return false;
    GuestList L;
    if (!c.make_list(L)) return false;
    const D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(static_cast<void*>(Dc.guest.Get())),
                                         D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(static_cast<void*>(Tc.guest.Get())),
                                          D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    invoke<void>(L.list.Get(), 16, &to, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    if (!c.close(L)) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct) || !c.holds(Dc.guest.Get(), 33, "RGBA8 reuse after BC7 retirement")) return false;
    if (!SUCCEEDED(c.reset_list(L))) return false;
    c.release(Tc);
    if (on) Tc.freed = true;
    return c.settle("T_rgba_after_bc7 retired", "T_rgba_after_bc7 kept");
}

// 4: descriptor & view reuse (slot 18 SRV + CopyDescriptorsSimple + heap release + descriptor overwrite).
bool case4_descriptor_and_view_reuse(Context& c, bool on, void* mem1, void* mem2)
{
    fill_title_memory(mem1, 44);
    fill_title_memory(mem2, 55);
    Res& Tv1 = c.make_slot133("Tv1", mem1);
    Res& Tv2 = c.make_slot133("Tv2", mem2);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2}, 0, 0};
    ComPtr<IUnknown> heap1, heap2;
    if (!require(invoke(c.guest, 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap1.GetAddressOf()), "heap1") ||
        !require(invoke(c.guest, 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap2.GetAddressOf()), "heap2"))
        return false;
    const UINT64 h1 = invoke<UINT64>(heap1.Get(), 9);
    const UINT64 h2 = invoke<UINT64>(heap2.Get(), 9);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    invoke<void>(c.guest, 18, Tv1.guest.Get(), &srv, h1);
    invoke<void>(c.guest, 24, 1u, h2, h1, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    c.release(Tv1);
    if (!c.stays("Tv1 held by heap1 and heap2", "Tv1 freed while two SRV descriptors hold it")) return false;
    heap1.Reset();
    if (!c.stays("Tv1 held by heap2 copy", "Tv1 freed while copied SRV in heap2 holds it")) return false;
    // Overwrite heap2[0] with Tv2: Tv1's last tracked descriptor ref drops to 0 and Tv1 retires.
    invoke<void>(c.guest, 18, Tv2.guest.Get(), &srv, h2);
    if (on) Tv1.freed = true;
    if (!c.settle("Tv1 retired after descriptor overwrite", "Tv1 kept after all SRV descriptors overwritten")) return false;
    c.release(Tv2);
    if (!c.stays("Tv2 held by overwritten heap2[0]", "Tv2 freed while SRV descriptor holds it")) return false;
    heap2.Reset();
    if (on) Tv2.freed = true;
    return c.settle("Tv2 retired after heap2 released", "Tv2 kept after heap2 released");
}

// 5: switch-off control. An ordinary committed texture (slot 127, no title memory, not a slot 133 resource) copied as
// a subresource must still record its upload exactly as before the switch: upload_texture() is gated on the retire
// class, not on title memory. The observable is the per-desktop-object texture_uploads() entry the no-title-memory
// path records (a probe over the same map), plus the forwarded counter of the copy itself.
bool case5_ordinary_upload_switch_off(Context& c)
{
    Res& source = c.make_committed("U_src");
    Res& dest = c.make_committed("U_dst");
    const auto source_real = desktop_object<ID3D12Resource>(c.guest, c.device, source.guest.Get());
    const auto dest_real = desktop_object<ID3D12Resource>(c.guest, c.device, dest.guest.Get());
    if (!source_real || !dest_real) {
        std::puts("FAIL: an ordinary committed texture has no desktop object");
        return false;
    }
    LONG forwarded_before = 0, refused_before = 0;
    x2w_d3d12_list_copy_texture_stats(&forwarded_before, &refused_before);
    GuestList L;
    if (!c.make_list(L)) return false;
    const D3D12_TEXTURE_COPY_LOCATION to{static_cast<ID3D12Resource*>(static_cast<void*>(dest.guest.Get())),
                                         D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    const D3D12_TEXTURE_COPY_LOCATION from{static_cast<ID3D12Resource*>(static_cast<void*>(source.guest.Get())),
                                           D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    invoke<void>(L.list.Get(), 16, &to, 0u, 0u, 0u, &from, static_cast<const D3D12_BOX*>(nullptr));
    if (!c.close(L)) return false;
    c.execute(c.direct, L);
    if (!c.sync(c.direct)) return false;
    LONG forwarded_after = 0, refused_after = 0;
    x2w_d3d12_list_copy_texture_stats(&forwarded_after, &refused_after);
    if (forwarded_after != forwarded_before + 1 || refused_after != refused_before) {
        std::printf("FAIL: ordinary texture copy forwarded %ld refused %ld, expected one more forwarded and none refused\n",
                    forwarded_after - forwarded_before, refused_after - refused_before);
        return false;
    }
    const bool source_cached = x2w_d3d12_texture_upload_probe(source_real.Get()) != FALSE;
    const bool dest_cached = x2w_d3d12_texture_upload_probe(dest_real.Get()) != FALSE;
    if (!source_cached || !dest_cached) {
        std::printf("FAIL: ordinary texture copy recorded no upload (source=%d destination=%d); "
                    "upload_texture is gated on title memory instead of the retire class\n", source_cached, dest_cached);
        return false;
    }
    return true;
}

// 6: written_textures() generation keeping (switch on). A live slot-18 SRV keeps the older resource's entry while it
// is pending; a newer generation of the same address then arrives and must survive the older generation's retirement
// (the snapshot-generation check), so the newer write is not mistaken for the older one's stale state.
bool case6_written_generation_preserved(Context& c, void* title_mem)
{
    x2w_d3d12_bc7_4k_s_written(title_mem, side, side);  // generation 1 of this address
    Res& Told = c.make_slot133("T_gen_old", title_mem);
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1}, 0, 0};
    ComPtr<IUnknown> heap;
    if (!require(invoke(c.guest, 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()), "generation heap"))
        return false;
    const UINT64 handle = invoke<UINT64>(heap.Get(), 9);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    invoke<void>(c.guest, 18, Told.guest.Get(), &srv, handle);
    c.release(Told);
    // The live view keeps the entry: only its generation is snapshotted, the entry is not erased.
    unsigned long long generation = 0;
    unsigned bc7_width = 0;
    if (!x2w_d3d12_written_probe(title_mem, &generation, &bc7_width, nullptr) || generation != 1 ||
        bc7_width != side) {
        std::printf("FAIL: a live view did not keep written_textures entry (found=%d generation=%llu bc7_width=%u)\n",
                    x2w_d3d12_written_probe(title_mem, nullptr, nullptr, nullptr), generation, bc7_width);
        return false;
    }
    // A newer generation of the same address while the older resource is still pending behind its view.
    x2w_d3d12_bc7_4k_s_written(title_mem, side, side);
    heap.Reset();  // drops the view reference: the older resource can retire now
    Told.freed = true;
    if (!c.settle("older generation retired", "older generation kept after its view was released")) return false;
    unsigned long long kept_generation = 0;
    unsigned kept_bc7_width = 0;
    const BOOL found = x2w_d3d12_written_probe(title_mem, &kept_generation, &kept_bc7_width, nullptr);
    if (!found || kept_generation != 2 || kept_bc7_width != side) {
        std::printf("FAIL: retiring the older generation released the newer written generation "
                    "(found=%d generation=%llu bc7_width=%u)\n", found, kept_generation, kept_bc7_width);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "off")))
        return 2;
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    char debug_env[2]{};
    const bool debug_on = GetEnvironmentVariableA("X2W_D3D12_DEBUG", debug_env, sizeof(debug_env)) && debug_env[0] == '1';
    if (debug_on) {
        ComPtr<ID3D12Debug> debug;
        if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
        debug->EnableDebugLayer();
    }
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device") ||
        !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device"))
        return 3;
    ComPtr<ID3D12InfoQueue> debug;
    if (debug_on && FAILED(device.As(&debug))) return 3;
    if (debug) {
        D3D12_MESSAGE_SEVERITY severities[] = {
            D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR, D3D12_MESSAGE_SEVERITY_WARNING};
        D3D12_INFO_QUEUE_FILTER filter{};
        filter.AllowList.NumSeverities = static_cast<UINT>(std::size(severities));
        filter.AllowList.pSeverityList = severities;
        if (FAILED(debug->AddStorageFilterEntries(&filter))) return 3;
        debug->ClearStoredMessages();
    }
    DXGI_ADAPTER_DESC1 adapter_desc{};
    adapter->GetDesc1(&adapter_desc);
    char switch_value[2]{};
    const bool switch_on = GetEnvironmentVariableA("X2W_D3D12_RELEASE_RETIRED", switch_value, sizeof(switch_value)) == 1 &&
        switch_value[0] == '1';
    const bool off_mode = argc == 3;
    if (switch_on == off_mode) return 2;
    const bool on = !off_mode;
    std::printf("adapter=%ls debug_layer=%s mode=%s\n", adapter_desc.Description, debug ? "enabled" : "off", off_mode ? "off" : "on");

    constexpr SIZE_T page = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    auto* mem1 = VirtualAlloc(nullptr, page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    auto* mem2 = VirtualAlloc(nullptr, page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    auto* mem3 = VirtualAlloc(nullptr, page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!mem1 || !mem2 || !mem3) return 4;

    Context c{guest.Get(), device.Get(), debug.Get()};
    try {
        c.desc = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                  D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        c.texture_bytes = device->GetResourceAllocationInfo(0, 1, &c.desc).SizeInBytes;
        c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const D3D12_COMMAND_QUEUE_DESC direct{};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, image_bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                         D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        if (!c.event || !require(device->CreateCommandQueue(&direct, IID_PPV_ARGS(&c.own_queue)), "own queue") ||
            !require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&c.own_allocator)), "own allocator") ||
            !require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c.own_allocator.Get(), nullptr, IID_PPV_ARGS(&c.own_list)), "own list") ||
            !require(c.own_list->Close(), "own list close") ||
            !require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c.own_fence)), "own fence") ||
            !require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nullptr, IID_PPV_ARGS(&c.upload)), "upload") ||
            !require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&c.readback)), "readback"))
            return 4;
        if (!c.make_queue(c.direct) || !c.make_queue(c.gated) || !c.make_list(c.drain_list) || !c.close(c.drain_list)) return 4;
        if (!x2w_d3d12_budget_snapshot(&c.base, sizeof(c.base))) return 4;

        if (!case1_slot133_copy_and_upload(c, on, mem1)) return 5;
        std::puts("1 slot 133 copy and upload: held across closed list and gated submit, readback ok, retired");
        if (!case2_same_address_and_slot70_reuse(c, on, mem1)) return 5;
        std::puts("2 same-address and slot 70 reuse: existing_memory_texture + resource_address_index ok");
        if (!case3_written_textures_cleanup(c, on, mem1, mem2)) return 5;
        std::puts("3 written_textures cleanup: stale bc7_width cleared on address reuse");
        if (!case4_descriptor_and_view_reuse(c, on, mem1, mem2)) return 5;
        std::puts("4 descriptor and view reuse: tracked slot-18 SRV holds and retires on overwrite/heap release");
        if (!on) {
            if (!case5_ordinary_upload_switch_off(c)) return 5;
            std::puts("5 switch-off ordinary texture copy: upload still recorded for a resource outside the retire class");
        } else {
            if (!case6_written_generation_preserved(c, mem3)) return 5;
            std::puts("6 written_textures generation: newer generation of an address kept by the older one's retirement");
        }
    } catch (int code) {
        VirtualFree(mem1, 0, MEM_RELEASE);
        VirtualFree(mem2, 0, MEM_RELEASE);
        VirtualFree(mem3, 0, MEM_RELEASE);
        return code;
    }
    VirtualFree(mem1, 0, MEM_RELEASE);
    VirtualFree(mem2, 0, MEM_RELEASE);
    VirtualFree(mem3, 0, MEM_RELEASE);
    const UINT64 errors = debug ? debug->GetNumStoredMessages() : 0;
    std::printf("debug_errors_or_warnings=%llu\n", errors);
    if (errors || FAILED(device->GetDeviceRemovedReason())) return 4;
    std::puts(on ? "release-retired title checks passed" : "switch-off title checks passed");
    return 0;
}
