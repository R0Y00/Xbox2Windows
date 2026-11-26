#pragma once

#include "gpu_memory.h"
#include <d3dcompiler.h>

namespace x2w_vmemory {
constexpr UINT arena_bytes = 64 << 20;
constexpr UINT arena_pages = arena_bytes / page_bytes;
constexpr UINT max_arena_pages = 3u << 14;

// Length of a version 6 packed structure, including its markers and final 64-byte node.
inline UINT packed_bytes(UINT64 header)
{
    const UINT version = UINT(header);
    const UINT64 nodes = header >> 32;
    const auto size = (nodes + 1) * 64;
    return (version & 65535) == 6 && ((version >> 16) & 0x8100) == 0x8100 && nodes && size <= (64u << 20) ? UINT(size) : 0;
}

struct Arena {
    ComPtr<ID3D12Resource> bytes, previous;
    UINT used = 0;
};
struct Place {
    std::shared_ptr<Arena> arena;
    UINT offset = 0;
    ComPtr<ID3D12Resource> source;
    UINT64 source_offset = 0;
    UINT64 cpu_identity = 0;
};
struct CopyBack {
    Place page;
    UINT offset;
    UINT bytes;
};
struct PageChoice {
    Place overlay;
    ID3D12Resource* source = nullptr;
    UINT64 source_offset = 0;
    const void* owner = nullptr; // a live title buffer holding the entire page
};
struct BuildRecording {
    std::unordered_map<UINT, PageChoice> pages;
    std::unordered_map<UINT, const char*> unavailable_pages;
    std::unordered_set<UINT64> scanned_uploads;
    std::vector<UINT_PTR> reservations;
    bool took_reservations = false;
    UINT cpu_pages = 0;
    UINT64 grant_epoch = 0;
};

// The overlay and previous title dwords stay on the card across lists. Compare at
// execution time, after queue waits for streaming. An unchanged title dword leaves
// the builder's GPU result intact. Refreshes never copy an arena to the title.
class Arenas {
    std::mutex lock_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pso_;
    std::vector<std::shared_ptr<Arena>> arenas_;
    std::unordered_map<UINT, Place> places_;
    // Per structure (old header, last frame, resend decision), updated on the GPU
    // in submission order, including when a closed list is replayed.
    ComPtr<ID3D12Resource> decisions_;
    std::unordered_map<UINT64, UINT> packed_;
    std::unordered_set<const void*> opened_;
    std::atomic<UINT64> grant_epoch_{0};

public:
    bool open(const void* identity, bool grant)
    {
        std::lock_guard<std::mutex> lock(lock_);
        if (grant && opened_.insert(identity).second) ++grant_epoch_;
        return opened_.count(identity) != 0;
    }
    void forget(const void* identity)
    {
        std::lock_guard<std::mutex> lock(lock_);
        if (opened_.erase(identity)) ++grant_epoch_;
    }
    UINT64 grant_epoch() const { return grant_epoch_.load(); }
    bool create(ID3D12Device* d)
    {
        std::lock_guard<std::mutex> lock(lock_);
        if (pso_) return device_.Get() == d;
        static constexpr char code[] = R"(
            ByteAddressBuffer source : register(t0);
            RWByteAddressBuffer oldTitle : register(u0);
            RWByteAddressBuffer overlay : register(u1);
            RWByteAddressBuffer decisions : register(u2);
            cbuffer Args : register(b0) { uint sourceAt, arenaAt, count, mode, frame; }
            [numthreads(64,1,1)] void merge(uint i:SV_DispatchThreadID) {
                if (mode == 0xfffffffe) {
                    if (i < count) oldTitle.Store(sourceAt + i*4,overlay.Load(arenaAt + i*4));
                    return;
                }
                if (mode == 0xffffffff) {
                    if (i != 0) return;
                    uint2 header = source.Load2(sourceAt);
                    uint4 prev = decisions.Load4(arenaAt);
                    bool matches = (header.x & 0xffff) == 6 && ((header.x >> 16) & 0x8100) == 0x8100
                                && header.y > 0 && header.y < 1048576 && ((header.y+1)*64 == count);
                    bool fresh = any(header != prev.xy) || (frame > prev.z && frame - prev.z > 8);
                    decisions.Store4(arenaAt,uint4(header,frame,matches && fresh ? 1 : 0));
                    return;
                }
                if (i >= count) return;
                uint x = source.Load(sourceAt + i*4), at = arenaAt + i*4;
                uint before = oldTitle.Load(at);
                bool resend = mode == 1;
                if (mode >= 2) resend = decisions.Load((mode-2)*16+12) != 0;
                if (resend || x != before) { overlay.Store(at,x); oldTitle.Store(at,x); }
            })";
        ComPtr<ID3DBlob> cs, blob, errors;
        if (FAILED(D3DCompile(code, sizeof(code)-1, nullptr, nullptr, nullptr, "merge", "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &cs, &errors))) return false;
        D3D12_ROOT_PARAMETER args[5]{};
        args[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        args[1].ParameterType = args[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        args[2].Descriptor.ShaderRegister = 1;
        args[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        args[3].Constants.Num32BitValues = 5;
        args[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        args[4].Descriptor.ShaderRegister = 2;
        D3D12_ROOT_SIGNATURE_DESC desc{5, args};
        if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
            FAILED(d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)))) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root_.Get();
        pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        if (!buffer(d, 65536 * 16, D3D12_HEAP_TYPE_DEFAULT, true, &decisions_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
            FAILED(d->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso_)))) return false;
        device_ = d;
        return true;
    }
    bool place(UINT page, ID3D12Resource* source, UINT64 at, Place& out, UINT64 cpu_identity = 0, bool* full = nullptr)
    {
        std::lock_guard<std::mutex> lock(lock_);
        auto found = places_.find(page);
        if (found != places_.end() && (cpu_identity ? found->second.cpu_identity == cpu_identity :
                found->second.source.Get() == source && found->second.source_offset == at)) {
            out = found->second;
            if (cpu_identity) out.source = source; // immutable snapshot for this recording
            return true;
        }
        // Allocate paired payload and preceding-title slabs on demand.
        // A remapped page takes a new slot; recorded GPU work retains its old bytes.
        if (arenas_.empty() || arenas_.back()->used == arena_pages) {
            if (arenas_.size() >= max_arena_pages / arena_pages) { if (full) *full = true; return false; }
            auto a = std::make_shared<Arena>();
            if (!buffer(device_.Get(), arena_bytes, D3D12_HEAP_TYPE_DEFAULT, true, &a->bytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) ||
                !buffer(device_.Get(), arena_bytes, D3D12_HEAP_TYPE_DEFAULT, true, &a->previous, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)) return false;
            arenas_.push_back(a);
        }
        auto a = arenas_.back();
        out = {a, a->used++ * page_bytes, source, at, cpu_identity};
        places_[page] = out;
        if (cpu_identity) places_[page].source.Reset();
        return true;
    }
    bool lookup(UINT page, ID3D12Resource* source, UINT64 at, Place& out)
    {
        std::lock_guard<std::mutex> lock(lock_);
        const auto found = places_.find(page);
        if (found == places_.end() || found->second.source.Get() != source || found->second.source_offset != at) return false;
        out = found->second;
        return true;
    }
    UINT packed_slot(UINT64 address)
    {
        std::lock_guard<std::mutex> lock(lock_);
        if (auto it = packed_.find(address); it != packed_.end()) return it->second;
        const UINT slot = UINT(packed_.size());
        if (slot >= 65536) return UINT32_MAX;
        packed_.emplace(address, slot);
        return slot;
    }
    template<class State> void run(ID3D12GraphicsCommandList* l, const Place& p, UINT source_at,
                                   UINT target, UINT count, UINT mode, UINT frame, State state, ID3D12Resource* zero = nullptr)
    {
        const bool publishing = mode == UINT32_MAX - 1;
        if (!p.cpu_identity)
            state(p.source.Get(), publishing ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        D3D12_RESOURCE_BARRIER order{};
        order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        l->ResourceBarrier(1, &order);
        l->SetComputeRootSignature(root_.Get());
        l->SetPipelineState(pso_.Get());
        l->SetComputeRootShaderResourceView(0, (publishing ? zero : p.source.Get())->GetGPUVirtualAddress());
        l->SetComputeRootUnorderedAccessView(1, (publishing ? p.source.Get() : p.arena->previous.Get())->GetGPUVirtualAddress());
        l->SetComputeRootUnorderedAccessView(2, p.arena->bytes->GetGPUVirtualAddress());
        l->SetComputeRootUnorderedAccessView(4, decisions_->GetGPUVirtualAddress());
        const UINT values[]{source_at, target, count, mode, frame};
        l->SetComputeRoot32BitConstants(3, 5, values, 0);
        l->Dispatch(mode == UINT32_MAX ? 1 : (count + 63)/64, 1, 1);
        l->ResourceBarrier(1, &order);
    }
    template<class State> void publish(ID3D12GraphicsCommandList* l, const CopyBack& range, State state, ID3D12Resource* zero)
    {
        const auto& p = range.page;
        run(l, p, UINT(p.source_offset) + range.offset, p.offset + range.offset, range.bytes / 4, UINT32_MAX-1, 0, state, zero);
    }
    template<class State> void merge(ID3D12GraphicsCommandList* l, const Place& p, UINT off, UINT size, UINT mode, State state)
    {
        run(l, p, UINT(p.source_offset) + off, p.offset + off, size / 4, mode, 0, state);
    }
};
}
