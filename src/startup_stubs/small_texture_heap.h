#pragma once

#include "small_texture_4k.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <vector>
#include <utility>

namespace x2w_startup {

// publish_linear_texture serializes access and keeps every published resource.
// Only unused heap tails are allocated; offsets never return to the allocator.
class SmallTextureHeap {
    using Heap = Microsoft::WRL::ComPtr<ID3D12Heap>;
    struct Chunk { Heap heap; UINT64 size{}, used{}; };
    std::vector<Chunk> chunks_;
    ID3D12Device* owner_ = nullptr;
    UINT64 next_size_ = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    static constexpr UINT64 max_chunk = 1u << 20;
    static constexpr GUID backing_id = {0xd5b4b235, 0x9fb8, 0x4924, {0xb2,0x8b,0x69,0xd2,0x72,0x63,0x19,0xa6}};

public:
    UINT64 count{}, committed_equivalent{}, reserved{}, occupied{};

    bool place(ID3D12Device* device, D3D12_RESOURCE_DESC desc, UINT64 standard_bytes, ID3D12Resource** output)
    {
        *output = nullptr;
        if (!standard_bytes || standard_bytes == UINT64_MAX || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
            desc.Layout != D3D12_TEXTURE_LAYOUT_UNKNOWN || desc.Flags != D3D12_RESOURCE_FLAG_NONE ||
            desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality || (owner_ && owner_ != device)) return false;
        const auto tile = small_texture_4k_detail::block(small_texture_4k_bits(desc.Format));
        if (!tile.width || !desc.Width || !desc.Height || desc.Width > 16 * tile.width || desc.Height > 16 * tile.height ||
            ((desc.Width + tile.width - 1) / tile.width) * ((desc.Height + tile.height - 1) / tile.height) > 16) return false;

        desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
        const auto requested = device->GetResourceAllocationInfo(0, 1, &desc);
        const UINT64 bytes = requested.SizeInBytes;
        if (requested.Alignment != desc.Alignment || !bytes || bytes >= standard_bytes || bytes > max_chunk ||
            bytes % desc.Alignment) return false;

        Chunk* chunk = nullptr;
        for (auto& c : chunks_) {
            const auto room = c.size - c.used;
            if (room >= bytes && (!chunk || room < chunk->size - chunk->used)) chunk = &c;
        }
        Chunk fresh;
        if (!chunk) {
            const UINT64 alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            const UINT64 slots = (std::max)(UINT64{1}, next_size_ / bytes);
            fresh.size = (slots * bytes + alignment - 1) & ~(alignment - 1);
            D3D12_HEAP_DESC heap{};
            heap.SizeInBytes = fresh.size;
            heap.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            heap.Properties.CreationNodeMask = heap.Properties.VisibleNodeMask = 1;
            heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            if (FAILED(device->CreateHeap(&heap, IID_PPV_ARGS(&fresh.heap)))) return false;
            chunk = &fresh;
        }
        Microsoft::WRL::ComPtr<ID3D12Resource> placed;
        if (FAILED(device->CreatePlacedResource(chunk->heap.Get(), chunk->used, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&placed))) ||
            FAILED(placed->SetPrivateDataInterface(backing_id, chunk->heap.Get()))) return false;

        chunk->used += bytes;
        if (chunk == &fresh) {
            const UINT64 size = fresh.size;
            if (fresh.used < size) chunks_.push_back(std::move(fresh));
            reserved += size;
            next_size_ = (std::min)(max_chunk, size * 2);
            owner_ = device;
        } else if (chunk->used == chunk->size) {
            if (chunk != &chunks_.back()) *chunk = std::move(chunks_.back());
            chunks_.pop_back();
        }
        committed_equivalent += standard_bytes;
        occupied += bytes;
        ++count;
        *output = placed.Detach();
        return true;
    }
};
}
