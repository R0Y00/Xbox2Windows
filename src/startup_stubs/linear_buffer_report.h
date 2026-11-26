#pragma once

// What d3d12_xs.dll reports about slot 133's opt-in linear buffers (X2W_D3D12_LINEAR_BUFFERS,
// src/startup_stubs/linear_buffer_backing.h) to fixtures through x2w_d3d12_linear_buffer_info and
// x2w_d3d12_linear_buffer_totals. Plain layouts: the DLL and the fixtures include this file alike.

#include <cstdint>

struct X2wLinearBufferInfo {
    std::uint32_t path;             // 1 shared, 2 copied, 3 kept the default backing (refusal_hr says why)
    std::uint32_t use;              // 1 GPU-read, 2 GPU-written
    std::int32_t feature_hr;        // S_OK supported, DXGI_ERROR_UNSUPPORTED not, other: query failed; S_FALSE not queried
    std::int32_t share_hr;          // result of the sharing attempt; S_FALSE when not attempted
    std::int32_t refusal_hr;        // kept default: why the new path could not keep its guarantees; else S_OK
    std::uint32_t stale;            // copied: 1 while the backing holds an older upload than the title's bytes
    std::uint64_t heap_generation;  // shared: generation of the opened heap
    void* backing;                  // the desktop ID3D12Resource
    std::uint64_t refreshes;        // copied: submissions that copied bytes into the backing
    std::uint64_t bytes_copied;     // copied: bytes copied from title memory into the backing
    std::uint64_t last_staging;     // copied: id of the staging buffer of the last refresh (0: none)
    std::uint64_t last_fence;       // copied: internal fence value of the last refresh (0: none)
    std::uint64_t refresh_failures; // copied: submissions that ran without this buffer's refresh
    std::int32_t last_refresh_hr;   // copied: HRESULT of the last failed refresh; S_OK if none
    std::uint32_t gpu_written;      // copied: 1 once a submitted copy wrote it; it is no longer refreshed
};

struct X2wLinearBufferTotals {
    std::uint64_t placements, shared, copied, kept_default;
    std::uint64_t feature_queries, heap_opens, heap_open_failures, heap_cache_hits, place_failures;
    std::uint64_t submissions, refreshes, bytes_copied, staging_created, staging_reused, unreadable;
    std::uint64_t refresh_failures;
    std::uint64_t consumer_waits;   // submissions made to Wait for a refresh on the internal queue
    std::uint64_t gpu_written;      // copied buffers that a submitted copy wrote (held from then on)
    std::uint64_t producer_waits;   // refresh copies made to Wait for a queue's earlier, uncompleted submission
    std::uint64_t unsafe_refresh_refusals;  // submissions refused: their refresh had no safe completion order
    std::uint64_t reordered_publishes;  // an older refresh published after a newer one: `uploaded` dropped
    std::uint64_t copy_faults;      // staging reads of title memory that faulted (caught; that refresh failed)
    std::int32_t last_refusal_hr, last_refresh_hr;
};
