// Indirect signature decoding and GPU argument clamping. Console single-pass dispatch (type 105, stride 12) maps to
// plain dispatch. A host command signature is cached per kind and stride. Signatures that change root arguments are
// refused.
//
// The clamp is a cs_5_1 shader with an embedded root signature. Its destination is a four-byte-aligned offset in
// the indirect ring. Chunks retire after submission. The effective count min(records,count) follows the clamped
// records when bit 2 of has_count is set. The source is rebased to its own GPU address so large window offsets fit.
// Each list's 2^27 budget is cleared on the GPU at its first clamp after reset because the previous recording may
// still be in flight.
#pragma once

#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace x2w_gpu_indirect {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// PC argument numbering (D3D12_INDIRECT_ARGUMENT_TYPE).
constexpr u32 kind_draw = 0;
constexpr u32 kind_indexed = 1;
constexpr u32 kind_dispatch = 2;
constexpr u32 kind_mesh = 10;
// The console's single-pass dispatch: stride 12, the three group counts (its flags live in the signature; the
// title passes none). Run as a plain dispatch.
constexpr u32 console_single_dispatch = 105;

// The budget slot's words (bytes): what the list spent [0], mesh records' largest group count asked [4], mesh
// groups asked in all [8], records refused for want of budget [12], the largest cost >= 2^22 [16].
constexpr u32 budget_slot_bytes = 32;
constexpr u32 budget = 1u << 27;

// The kind a signature runs as, or false when it cannot run: exactly one argument, of a kind the host signature
// takes (draw, indexed draw, dispatch, mesh dispatch, or the console's single-pass dispatch, which becomes a
// dispatch).
inline bool signature_kind(const u32* types, std::size_t count, u32* kind)
{
    if (!types || count != 1) return false;
    const u32 type = types[0] == console_single_dispatch ? kind_dispatch : types[0];
    if (type != kind_draw && type != kind_indexed && type != kind_dispatch && type != kind_mesh) return false;
    *kind = type;
    return true;
}
template <class Description>
bool signature_kind_of(const Description* arguments, std::size_t count, u32* kind)
{
    if (!arguments || count != 1) return false;
    const u32 type = static_cast<u32>(arguments[0].Type);
    return signature_kind(&type, 1, kind);
}

// Minimum record byte stride of `kind`, or 0 for a kind the clamp does not take.
constexpr u32 min_stride(u32 kind)
{
    return kind == kind_draw ? 16 : kind == kind_indexed ? 20 : kind == kind_dispatch || kind == kind_mesh ? 12 : 0;
}

// Validate record alignment, minimum stride, and the 32-bit scratch size stride * records.
inline bool validate_layout(u32 kind, u32 stride, u32 records, u64 src_offset, bool has_count, u64 count_offset, u64* bytes)
{
    const u32 least = min_stride(kind);
    if (!least || stride < least || stride % 4 || src_offset % 4) return false;
    if (has_count && count_offset % 4) return false;
    const u64 total = u64{stride} * records;
    if (total > 0xffffffffull) return false;
    if (bytes) *bytes = total;
    return true;
}

// The groups a direct DispatchMesh runs: the console's mesh programs are one-dimensional and the host takes 65,535
// groups in a dimension.
constexpr u32 mesh_groups(u32 x) { return x < 65535u ? x : 65535u; }

// DispatchX with "use thread dimensions" (console flag 0x20): the counts are threads, run as the groups of the
// program's size that hold them.
constexpr u32 thread_dimensions_flag = 0x20;
inline void dispatch_groups(const u32 threads[3], const u32 size[3], u32 groups[3])
{
    for (int i = 0; i < 3; ++i) groups[i] = size[i] ? (threads[i] + size[i] - 1) / size[i] : threads[i];
}

// The most groups in x an indirect dispatch runs as asked: 2^20 on an AMD card (its dispatch packets take 32-bit
// counts, as the console's), 65,535 elsewhere.
constexpr u32 widest(bool amd, bool wide_on = true) { return amd && wide_on ? (1u << 20) : 65535u; }

// Whether `device` is an AMD card's (vendor 0x1002).
inline bool is_amd(ID3D12Device* device)
{
    if (!device) return false;
    IDXGIFactory4* factory = nullptr;
    if (FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) return false;
    IDXGIAdapter1* adapter = nullptr;
    bool amd = false;
    if (SUCCEEDED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&adapter)))) {
        DXGI_ADAPTER_DESC1 desc{};
        amd = SUCCEEDED(adapter->GetDesc1(&desc)) && desc.VendorId == 0x1002;
        adapter->Release();
    }
    factory->Release();
    return amd;
}

// The clamp writes the effective record count after the records when bit 2 of has_count is set (count_out).
// clamp_text supplies WIDEST and REST.
constexpr char clamp_source[] = R"hlsl(
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2 (X2W): the effective count
    // is stored after the records, for ExecuteIndirect's count buffer.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// The vertices or indices an instance from which a draw record runs empty.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted.
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if ((has_count & 4u) != 0 && id.x == 0) {
        dst_args.Store(records * stride, eff_count);
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        // A record of MOST_DRAW vertices or indices an instance, or more, is
        // no draw of the title's (a slot it has not filled): it runs empty
        // and charges nothing.
        if (want.x >= MOST_DRAW) {
            want = uint2(0, 0);
        }
        uint count = want.x;
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 g = src_args.Load3(src_at);
        // Up to WIDEST groups in x run as asked (`widest`).
        g = min(g, uint3(g.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
        // x * y fits 32 bits where it is looked at; the whole count may not.
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // A record over the most a dispatch runs is a slot the title has
        // not filled (the title leaves all ones there): empty,
        // and not counted below either.
        if (g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        // Diagnostics, in the budget buffer's next words: the largest group
        // count a mesh record asked for, and the groups asked in all.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    // The group's own records: none for a group past a count buffer's
    // count, which touches the budget not at all.
    uint first = id.x - gi;
    uint n = eff_count > first ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest
        // (`REST`).
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next < c) ? (REST ? BUDGET : next) : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        uint refused = 0;
        uint biggest = 0;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at < c) ? (REST ? BUDGET : at) : (at + c);
                refused += 1 - fit;
                biggest = max(biggest, c);
            }
            fits[k] = fit;
        }
        // Diagnostics: the records that run empty for want of budget, and
        // the largest cost there was (small ones are not told).
        uint told;
        if (refused > 0) {
            budget.InterlockedAdd(12, refused, told);
        }
        if (biggest >= (1u << 22)) {
            budget.InterlockedMax(16, biggest, told);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
)hlsl";

// The clamp's text with WIDEST and REST defined.
inline std::string clamp_text(u32 widest_groups, bool rest, const char* source = clamp_source)
{
    char head[96];
    std::snprintf(head, sizeof(head), "#define WIDEST %uu\n#define REST %u\n", widest_groups, rest ? 1u : 0u);
    return std::string(head) + source;
}

struct Clamp {
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr;
    void release()
    {
        if (pso) pso->Release();
        if (root) root->Release();
        pso = nullptr;
        root = nullptr;
    }
};

// Compiles `text` (clamp_text) with FXC as cs_5_1 and makes its root signature (inside the shader) and pipeline.
// `errors` (optional) gets the compiler's message.
inline HRESULT make_clamp(ID3D12Device* device, const std::string& text, Clamp* out, std::string* errors = nullptr)
{
    if (!device || !out) return E_INVALIDARG;
    ID3DBlob* code = nullptr;
    ID3DBlob* messages = nullptr;
    HRESULT hr = D3DCompile(text.data(), text.size(), "gpu_clamp", nullptr, nullptr, "main", "cs_5_1", 0, 0, &code, &messages);
    if (messages) {
        if (errors) errors->assign(static_cast<const char*>(messages->GetBufferPointer()), messages->GetBufferSize());
        messages->Release();
    }
    if (FAILED(hr)) {
        if (code) code->Release();
        return hr;
    }
    Clamp made;
    hr = device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&made.root));
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature = made.root;
        desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&made.pso));
    }
    code->Release();
    if (FAILED(hr)) {
        made.release();
        return hr;
    }
    *out = made;
    return S_OK;
}

inline D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    return b;
}
inline D3D12_RESOURCE_BARRIER uav_barrier(ID3D12Resource* resource = nullptr)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = resource;
    return b;
}

// One clamp.
struct ClampCall {
    u32 kind = 0, stride = 0, records = 0;
    ID3D12Resource* src = nullptr;  // the records, read through a root SRV (any state a root SRV may read)
    u64 src_offset = 0;
    ID3D12Resource* count = nullptr;  // null: no count buffer
    u64 count_offset = 0;
    ID3D12Resource* dst = nullptr;  // the scratch buffer (UAV-capable, device local)
    u64 dst_offset = 0;             // 4-aligned
    D3D12_RESOURCE_STATES dst_before = D3D12_RESOURCE_STATE_COMMON;
    u64 budget_gpu = 0;  // the list's budget slot (budget_slot_bytes)
    // The buffer holding the budget slot, when it rests in COMMON (a pool shared by lists): it is moved to
    // UNORDERED_ACCESS for the clamp and back to COMMON with the clamp's own barriers, so that no list leaves it
    // in another state (buffers decay to COMMON only at the end of a whole ExecuteCommandLists, and the next list
    // of the same call expects COMMON). Null: the slot's buffer is left as it is (tests' mapped buffers).
    ID3D12Resource* budget_resource = nullptr;
    bool ahead = false;      // a window clamped ahead of its draws (unfilled slots empty and free)
    bool count_out = false;  // store min(records, count) at dst_offset + stride * records
};

// The scratch bytes a call needs: the records, and the count after them when count_out.
inline u64 scratch_bytes(const ClampCall& c) { return u64{c.stride} * c.records + (c.count_out ? 4 : 0); }

// Records the clamp of `c` into `list`: a UAV barrier for every earlier shader write (the title's argument and
// count passes), dst before -> UNORDERED_ACCESS, the clamp dispatch, a UAV barrier, dst -> INDIRECT_ARGUMENT.
// Replaces the list's compute root signature and pipeline (the caller binds its own after). False, with nothing
// recorded, when the layout or the addresses do not fit the clamp.
inline bool record_clamp(ID3D12GraphicsCommandList* list, const Clamp& clamp, const ClampCall& c)
{
    u64 bytes = 0;
    if (!list || !clamp.pso || !c.src || !c.dst) return false;
    if (!validate_layout(c.kind, c.stride, c.records, c.src_offset, c.count != nullptr, c.count_offset, &bytes)) return false;
    if (c.dst_offset % 4 || c.budget_gpu % 4 || !c.budget_gpu) return false;
    if (c.count_out && bytes + 4 > 0xffffffffull) return false;
    const u64 src_gpu = c.src->GetGPUVirtualAddress() + c.src_offset;
    const u64 dst_gpu = c.dst->GetGPUVirtualAddress() + c.dst_offset;
    if (src_gpu % 4 || dst_gpu % 4) return false;
    u64 count_gpu = src_gpu;
    u32 has_count = 0;
    if (c.count) {
        count_gpu = c.count->GetGPUVirtualAddress() + c.count_offset;
        if (count_gpu % 4) return false;
        has_count = 1;
    }
    has_count |= (c.ahead ? 2u : 0u) | (c.count_out ? 4u : 0u);
    // The source is rebased to its own address (args_offset 0), the count likewise (count_offset 0).
    const u32 constants[6] = {c.kind, c.stride, c.records, 0, has_count, 0};
    D3D12_RESOURCE_BARRIER first[3] = {uav_barrier(), {}, {}};
    UINT n = 1;
    if (c.dst_before != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        first[n++] = transition(c.dst, c.dst_before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (c.budget_resource)
        first[n++] = transition(c.budget_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->ResourceBarrier(n, first);
    if (c.records > 0) {
        list->SetComputeRootSignature(clamp.root);
        list->SetPipelineState(clamp.pso);
        list->SetComputeRoot32BitConstants(0, 6, constants, 0);
        list->SetComputeRootShaderResourceView(1, src_gpu);
        list->SetComputeRootShaderResourceView(2, count_gpu);
        list->SetComputeRootUnorderedAccessView(3, dst_gpu);
        list->SetComputeRootUnorderedAccessView(4, c.budget_gpu);
        list->Dispatch((c.records + 63) / 64, 1, 1);
    }
    const D3D12_RESOURCE_BARRIER second[3] = {
        uav_barrier(), transition(c.dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        c.budget_resource ? transition(c.budget_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON) : D3D12_RESOURCE_BARRIER{}};
    list->ResourceBarrier(c.budget_resource ? 3 : 2, second);
    return true;
}

// Zeroes a list's budget slot on the GPU at its first clamp after a reset: `pool` (in COMMON, where every list
// leaves it) gets budget_slot_bytes zeros at `offset` from `zeros` (a buffer of at least that many zero bytes, in a
// state a copy reads, e.g. an upload heap's GENERIC_READ), and goes back to COMMON. The clamps that follow name
// the pool as their ClampCall::budget_resource.
inline void record_budget_zero(ID3D12GraphicsCommandList* list, ID3D12Resource* pool, u64 offset, ID3D12Resource* zeros)
{
    const D3D12_RESOURCE_BARRIER to_copy = transition(pool, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(pool, offset, zeros, 0, budget_slot_bytes);
    const D3D12_RESOURCE_BARRIER back = transition(pool, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    list->ResourceBarrier(1, &back);
}

// Host command signatures of one argument of `kind` with records `stride` bytes apart, no root arguments and so no
// root signature, made once per device.
class HostSignatures {
public:
    ID3D12CommandSignature* get(ID3D12Device* device, u32 kind, u32 stride)
    {
        std::lock_guard<std::mutex> hold(lock_);
        for (const auto& m : made_)
            if (m.device == device && m.kind == kind && m.stride == stride) return m.signature;
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(kind);
        const D3D12_COMMAND_SIGNATURE_DESC desc{stride, 1, &argument, 0};
        ID3D12CommandSignature* signature = nullptr;
        if (!device || FAILED(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&signature)))) return nullptr;
        made_.push_back({device, kind, stride, signature});
        return signature;
    }

private:
    struct Made {
        ID3D12Device* device;
        u32 kind, stride;
        ID3D12CommandSignature* signature;
    };
    std::mutex lock_;
    std::vector<Made> made_;
};

}  // namespace x2w_gpu_indirect
