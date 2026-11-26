#pragma once
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cstring>

namespace x2w_compute_trace {
struct IndirectArguments {
    static constexpr UINT limit = 8, words = 5;
    using Record = std::array<UINT, words>;
    struct Snapshot { UINT count{}; std::array<Record, limit> records{}; };
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    UINT kind{}, stride{}, maximum{}, copied{}, fields{};
    bool counted{};

    bool record(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* source,
                UINT64 offset, UINT type, UINT step, UINT max, bool has_count)
    {
        const UINT payload = type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW ? 4 :
            type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED ? 5 : 3;
        if (!source || step < payload * sizeof(UINT) || !max || step % 4) return false;
        const auto width = source->GetDesc().Width;
        const UINT64 bytes = UINT64(step) * max;
        if (offset > width || bytes > width - offset || (has_count && width - offset - bytes < 4)) return false;
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = sizeof(Snapshot);
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&buffer)))) return false;
        kind = type; stride = step; maximum = max; counted = has_count;
        copied = (std::min)(limit, max);
        fields = payload;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {source, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &barrier);
        if (counted) list->CopyBufferRegion(buffer.Get(), 0, source, offset + bytes, 4);
        for (UINT i = 0; i < copied; ++i)
            list->CopyBufferRegion(buffer.Get(), sizeof(UINT) + sizeof(Record) * i, source, offset + UINT64(step)*i, fields*4);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        list->ResourceBarrier(1, &barrier);
        return true;
    }

    bool read(Snapshot& snap) const
    {
        void* data = nullptr;
        D3D12_RANGE range{0, sizeof(Snapshot)};
        if (!buffer || FAILED(buffer->Map(0, &range, &data))) return false;
        const auto* bytes = static_cast<const unsigned char*>(data);
        if (counted) std::memcpy(&snap.count, bytes, sizeof(UINT));
        else snap.count = maximum;
        for (UINT i=0; i<copied; ++i)
            std::memcpy(snap.records[i].data(), bytes + sizeof(UINT) + sizeof(Record)*i, fields*4);
        const D3D12_RANGE none{};
        buffer->Unmap(0, &none);
        return true;
    }
};
}
