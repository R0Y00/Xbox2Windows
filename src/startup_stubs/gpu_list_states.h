#pragma once

#include "startup_stubs/gpu_transition.h"
#include <unordered_map>
#include <vector>

namespace x2w_gpu {

class ListStates {
public:
    struct Entry {
        ID3D12Resource* resource;
        D3D12_RESOURCE_STATES depth = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_STATES stencil = D3D12_RESOURCE_STATE_COMMON;
    };

    const Entry* find(ID3D12Resource* resource) const
    {
        const auto it = indices_.find(resource);
        return it == indices_.end() ? nullptr : &ordered_[it->second];
    }
    D3D12_RESOURCE_STATES state(ID3D12Resource* resource) const
    {
        const auto* entry = find(resource);
        return entry ? entry->depth : D3D12_RESOURCE_STATE_COMMON;
    }
    Entry& track(ID3D12Resource* resource)
    {
        const auto [it, inserted] = indices_.try_emplace(resource, ordered_.size());
        if (inserted) {
            try {
                ordered_.push_back({resource});
            } catch (...) {
                indices_.erase(it);
                throw;
            }
        }
        return ordered_[it->second];
    }
    void require(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                 D3D12_RESOURCE_STATES depth, D3D12_RESOURCE_STATES stencil)
    {
        change(list, track(resource), depth, stencil);
    }
    void close(ID3D12GraphicsCommandList* list)
    {
        for (auto& entry : ordered_)
            change(list, entry, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COMMON);
        clear();
    }
    void clear() { ordered_.clear(); indices_.clear(); }
    std::size_t size() const { return ordered_.size(); }

private:
    static void change(ID3D12GraphicsCommandList* list, Entry& entry,
                       D3D12_RESOURCE_STATES depth, D3D12_RESOURCE_STATES stencil)
    {
        if (entry.depth == depth && entry.stencil == stencil) return;
        if (entry.depth == entry.stencil && depth == stencil) {
            transition_all(list, entry.resource, entry.depth, depth);
        } else {
            const auto desc = entry.resource->GetDesc();
            const UINT stride = UINT(desc.MipLevels) * desc.DepthOrArraySize;
            const auto plane = [&](UINT index, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
                if (before == after) return;
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Transition = {entry.resource, 0, before, after};
                for (UINT i = 0; i < stride; ++i) {
                    barrier.Transition.Subresource = index * stride + i;
                    list->ResourceBarrier(1, &barrier);
                }
            };
            plane(0, entry.depth, depth);
            plane(1, entry.stencil, stencil);
        }
        entry.depth = depth;
        entry.stencil = stencil;
    }

    std::vector<Entry> ordered_;
    std::unordered_map<ID3D12Resource*, std::size_t> indices_;
};

} // namespace x2w_gpu
