#pragma once

#include "gpu_abi.h"
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace x2w_vmemory {
struct BuildRecording;
using Microsoft::WRL::ComPtr;
namespace abi = x2w_gpu::abi;
constexpr UINT page_bytes = 65536;
constexpr UINT slots = 256;
constexpr UINT entries = 65536;
constexpr UINT fault_bytes = abi::memory_faults * sizeof(UINT);
constexpr UINT directory_bytes = abi::memory_pages * 8;

inline bool title_page(UINT64 page) { return page >= 65536 && page < abi::memory_pages - 1; }

enum class MapError { none, page, view, slots, updates };

// Keep missing-page reports separate from confirmed mappings. The reference's
// 128-per-frame and 2048-total budgets count pages it successfully maps.
class FaultPages {
public:
    size_t report(const std::vector<UINT>& faults) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (confirmed_.size() < 2048)
            pending_.insert(faults.begin(), faults.end());
        return confirmed_.size();
    }
    template<class Map> void seed(LONG frame, Map map) {
        std::vector<UINT> known;
        std::vector<UINT> requests;
        size_t remaining{};
        {
            std::lock_guard<std::mutex> lock(mutex_);
            known = confirmed_;
            if (frame > frame_) { frame_ = frame; admitted_ = 0; }
            if (frame == frame_ && admitted_ < 128 && known.size() < 2048) {
                remaining = (std::min)(size_t{128} - admitted_, size_t{2048} - known.size());
                requests.assign(pending_.begin(), pending_.end());
                pending_.clear();
            }
        }
        for (UINT page : known) map(page);
        if (requests.empty()) return;
        std::sort(requests.begin(), requests.end());
        std::vector<UINT> accepted;
        for (UINT page : requests) {
            if (!std::binary_search(known.begin(), known.end(), page) && map(page)) {
                accepted.push_back(page);
                if (accepted.size() == remaining) break;
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (UINT page : accepted) {
            if (frame != frame_ || admitted_ == 128 || confirmed_.size() == 2048) {
                if (confirmed_.size() < 2048) pending_.insert(page);
                continue;
            }
            auto pos = std::lower_bound(confirmed_.begin(), confirmed_.end(), page);
            if (pos == confirmed_.end() || *pos != page) {
                confirmed_.insert(pos, page);
                ++admitted_;
            }
        }
    }
private:
    std::mutex mutex_;
    std::vector<UINT> confirmed_;
    std::unordered_set<UINT> pending_;
    LONG frame_ = -1;
    size_t admitted_ = 0;
};

inline bool buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE type, bool uav, ID3D12Resource** out,
                   D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    hp.CreationNodeMask = hp.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (uav) d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto state = type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ :
                       type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : initial;
    return SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(out)));
}

// Per recording and view heap. The GPU sees directory changes in command order.
// Staging entries are append-only. Their storage and the source windows survive every submit.
struct Recording {
    x2w_gpu::GpuHostHeap* heap = nullptr;
    UINT base = 0;
    ComPtr<ID3D12Resource> directory, faults, results, staging, zeros;
    std::vector<ComPtr<ID3D12Resource>> windows;
    struct ReadAlias { UINT64 original; UINT chunk_slot, offset, rows; };
    std::unordered_map<UINT, ReadAlias> read_aliases;
    std::vector<std::pair<UINT, UINT>> restore_rows;
    std::vector<ID3D12Resource*> ordinary_reads;
    UINT alias_chunk = 0, alias_used = 0;
    std::unordered_set<ID3D12Resource*> permanent_uavs;
    std::unordered_map<UINT, UINT64> addresses;
    std::unordered_set<UINT> absent;
    std::unordered_set<UINT64> ranges;
    std::unordered_set<UINT> refreshed;
    std::unordered_set<UINT64> packed_seen;
    std::shared_ptr<BuildRecording> build;
    UINT updates = 0;
    std::array<UINT, 2>* data = nullptr;
    bool closed = false;
    bool faults_seeded = false;
    bool initialized = false;
    bool writable_mode = false;
    bool claimed = false;
    inline static std::atomic<UINT> active{0};
    ~Recording()
    {
        if (data) staging->Unmap(0, nullptr);
        if (claimed) --active;
    }

    bool create(ID3D12Device* d, bool* at_capacity = nullptr)
    {
        // Each live recording holds a 16 MiB directory. Reaching the limit refuses
        // this bind until a recording retires; nothing in flight is reused.
        constexpr UINT directory_budget = 512u << 20;
        if (active.fetch_add(1) >= directory_budget / directory_bytes) {
            --active;
            if (at_capacity) *at_capacity = true;
            return false;
        }
        claimed = true;
        static std::mutex lock;
        static ComPtr<ID3D12Device> cached_device;
        static ComPtr<ID3D12Resource> cached_zero;
        D3D12_RANGE none{};
        {
            std::lock_guard<std::mutex> guard(lock);
            if (cached_device.Get() != d || !cached_zero) {
                ComPtr<ID3D12Resource> zero;
                void* mapped = nullptr;
                if (!buffer(d, directory_bytes, D3D12_HEAP_TYPE_UPLOAD, false, &zero) ||
                    FAILED(zero->Map(0, &none, &mapped))) return false;
                std::memset(mapped, 0, directory_bytes);
                zero->Unmap(0, nullptr);
                cached_zero = zero;
                cached_device = d;
            }
            zeros = cached_zero;
        }
        return buffer(d, directory_bytes, D3D12_HEAP_TYPE_DEFAULT, false, &directory) &&
               buffer(d, fault_bytes, D3D12_HEAP_TYPE_DEFAULT, true, &faults) &&
               buffer(d, fault_bytes, D3D12_HEAP_TYPE_READBACK, false, &results) &&
               buffer(d, UINT64{entries} * 8, D3D12_HEAP_TYPE_UPLOAD, false, &staging) &&
               SUCCEEDED(staging->Map(0, &none, reinterpret_cast<void**>(&data)));
    }
    template<class State> void begin(ID3D12GraphicsCommandList* list, State state)
    {
        if (initialized) return;
        // A closed native list may be submitted more than once. Its initial map and
        // fault set must be empty every time the recording starts executing.
        state(directory.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(directory.Get(), 0, zeros.Get(), 0, directory_bytes);
        state(faults.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(faults.Get(), 0, zeros.Get(), 0, fault_bytes);
        initialized = true;
    }
    template<class Find, class View, class State>
    void map(ID3D12GraphicsCommandList* list, UINT page, Find find, View view, State state)
    {
        if (!title_page(page) || addresses.count(page) || absent.count(page) || addresses.size() >= entries) return;
        UINT64 offset = 0;
        ID3D12Resource* window = nullptr;
        if (!find(UINT64{page} * page_bytes, &window, &offset) || !window) {
            absent.insert(page);
            return;
        }
        map_to(list, page, window, offset, false, view, state);
    }
    template<class View, class State>
    bool map_to(ID3D12GraphicsCommandList* list, UINT page, ID3D12Resource* window, UINT64 offset,
                bool writable, View view, State state, bool permanent = false, MapError* error = nullptr)
    {
        const auto reject = [&](MapError why) { if (error) *error = why; return false; };
        if (!title_page(page) || !window) return reject(MapError::page);
        const auto desc = window->GetDesc();
        if (!(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || offset % 4 ||
            desc.Width > UINT32_MAX || offset > desc.Width || page_bytes > desc.Width - offset) return reject(MapError::view);
        UINT number = 1;
        for (const auto& w : windows) {
            if (w.Get() == window) break;
            ++number;
        }
        if (number > windows.size()) {
            if (number >= slots) return reject(MapError::slots);
            windows.emplace_back(window);
            view(base + number, window, desc.Width);
        }
        if (permanent) permanent_uavs.insert(window);
        const UINT permission = writable ? 0 : abi::memory_read_only;
        const UINT64 binding = ((UINT64{number} | permission) << 32) | offset;
        if (auto known = addresses.find(page); known != addresses.end() && known->second == binding) return true;
        if (updates == entries) return reject(MapError::updates);
        const auto i = updates++;
        begin(list, state);
        data[i] = {permission | number, UINT(offset)};
        addresses[page] = binding;
        state(directory.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(directory.Get(), UINT64{page} * 8, staging.Get(), i * 8, 8);
        return true;
    }
    template<class State> void restore_read_aliases(ID3D12GraphicsCommandList* list, State state)
    {
        if (!restore_rows.empty()) {
            state(directory.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
            for (const auto& [page, row] : restore_rows)
                list->CopyBufferRegion(directory.Get(), UINT64{page} * 8, staging.Get(), UINT64{row} * 8, 8);
            restore_rows.clear();
        }
        ordinary_reads.clear();
    }

    // The directory shader loads through a UAV even for protected pages. Give it
    // a GPU-time snapshot when a normal shader binding reads the same resource.
    template<class View, class State>
    bool allow_ordinary_reads(ID3D12GraphicsCommandList* list, const std::vector<ID3D12Resource*>& reads,
                              const std::vector<ID3D12Resource*>& writes, D3D12_RESOURCE_STATES read_state, View view, State state)
    {
        if (reads.empty()) return true;
        std::array<bool, slots> conflict{};
        std::vector<ID3D12Resource*> shared;
        for (UINT i = 0; i < windows.size(); ++i) {
            auto* w = windows[i].Get();
            if (std::find(reads.begin(), reads.end(), w) == reads.end()) continue;
            if (writable_mode || permanent_uavs.count(w) || std::find(writes.begin(), writes.end(), w) != writes.end()) return false;
            conflict[i + 1] = true;
            shared.push_back(w);
        }
        if (shared.empty()) return true;
        constexpr UINT chunk_pages = 64, chunk_bytes = chunk_pages * page_bytes;
        std::vector<UINT> pages;
        for (const auto& [page, binding] : addresses) {
            const UINT entry = UINT(binding >> 32);
            if (!conflict[entry & ~abi::memory_read_only]) continue;
            if (!(entry & abi::memory_read_only)) return false;
            auto found = read_aliases.find(page);
            if (found == read_aliases.end() || found->second.original != binding) {
                if (updates > entries - 2) return false;
                if (!alias_chunk || alias_used == chunk_pages) {
                    if (windows.size() >= slots - 1) return false;
                    ComPtr<ID3D12Device> device;
                    ComPtr<ID3D12Resource> target;
                    if (FAILED(directory->GetDevice(IID_PPV_ARGS(&device))) ||
                        !buffer(device.Get(), chunk_bytes, D3D12_HEAP_TYPE_DEFAULT, true, &target)) return false;
                    windows.push_back(target);
                    alias_chunk = UINT(windows.size());
                    alias_used = 0;
                    view(base + alias_chunk, target.Get(), chunk_bytes);
                }
                const UINT offset = alias_used++ * page_bytes;
                read_aliases[page] = {binding, alias_chunk, offset, updates};
                data[updates++] = {alias_chunk | abi::memory_read_only, offset};
                data[updates++] = {entry, UINT(binding)};
            }
            pages.push_back(page);
        }
        // All slots and immutable directory patches exist before changing GPU-visible state.
        if (!pages.empty()) state(directory.Get(), D3D12_RESOURCE_STATE_COPY_DEST);
        for (auto* source : shared) state(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (auto page : pages) {
            const auto& a = read_aliases.at(page);
            const UINT source = UINT(a.original >> 32) & ~abi::memory_read_only;
            auto* dest = windows[a.chunk_slot - 1].Get();
            state(dest, D3D12_RESOURCE_STATE_COPY_DEST);
            list->CopyBufferRegion(dest, a.offset, windows[source - 1].Get(), UINT(a.original), page_bytes);
            list->CopyBufferRegion(directory.Get(), UINT64{page} * 8, staging.Get(), UINT64{a.rows} * 8, 8);
            restore_rows.emplace_back(page, a.rows + 1);
        }
        for (auto* source : shared) state(source, read_state);
        ordinary_reads = std::move(shared);
        return true;
    }

    template<class State> void ready(ID3D12GraphicsCommandList* list, State state)
    {
        // Reads from an unseeded directory return zero and report the missing
        // pages. Clear both buffers before this first dispatch, including on replay.
        begin(list, state);
        state(directory.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        state(faults.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (const auto& w : windows)
            if (!permanent_uavs.count(w.Get()) && std::find(ordinary_reads.begin(), ordinary_reads.end(), w.Get()) == ordinary_reads.end())
                state(w.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    template<class State> void close(ID3D12GraphicsCommandList* list, State state)
    {
        if (!initialized) return;
        state(faults.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(results.Get(), 0, faults.Get(), 0, fault_bytes);
        closed = true;
    }
    // Call only after all fences for this recording passed.
    std::vector<UINT> read_faults() const
    {
        std::vector<UINT> pages;
        if (!initialized || !closed) return pages;
        D3D12_RANGE range{0, fault_bytes};
        UINT* words = nullptr;
        if (FAILED(results->Map(0, &range, reinterpret_cast<void**>(&words)))) return pages;
        for (UINT i = 1; i < abi::memory_faults; ++i)
            if (title_page(words[i])) pages.push_back(words[i]);
        D3D12_RANGE none{};
        results->Unmap(0, &none);
        return pages;
    }
};
}
