// Cache host pipelines by stage records and graphics state. Title pipeline creation records these inputs. Building
// the root signature and PSO on demand avoids compiling every pipeline the title creates during startup.
#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "startup_stubs/gpu_pipeline_build.h"

namespace x2w_gpu {

struct VpPending {
    // The translated stages, in the order the title gave them. A stage whose record was missing
    // has dxil null (the pipeline is then refused at ready()).
    std::vector<VpShaderView> shaders;
    // Owned copies of the arrays the views point at, so a view never outlives its record.
    std::vector<std::vector<VpStaticSampler>> sampler_store;
    std::vector<std::vector<std::uint8_t>> slot_store;
    std::unique_ptr<VpGraphicsState> graphics;  // null: compute, or the plain graphics description
    std::uint32_t missing = 0;                  // stages without a translated record
    std::uint32_t creation_slot = 0;            // device slot the title created it with (10, 11, 62, 86)

    std::once_flag once;
    std::shared_ptr<const VpHostPipeline> made;
    std::string why;  // set when refused
    std::atomic<bool> done{false};

    // Takes a copy of the view's arrays (the record may live in a mapped file the caller owns,
    // but the pending pipeline must not depend on that).
    void add(const VpShaderView& view)
    {
        VpShaderView v = view;
        if (v.sampler_count) {
            sampler_store.emplace_back(v.samplers, v.samplers + v.sampler_count);
            v.samplers = sampler_store.back().data();
        }
        const auto keep = [this](const std::uint8_t*& p, std::size_t n) {
            if (!n) return;
            slot_store.emplace_back(p, p + n);
            p = slot_store.back().data();
        };
        keep(v.root_cbvs, v.root_cbv_count);
        keep(v.root_srvs, v.root_srv_count);
        keep(v.root_uavs, v.root_uav_count);
        if (!v.dxil || !v.dxil_size) ++missing;
        shaders.push_back(v);
    }
};

// Host pipelines by what they are made of; shared by every pending pipeline with that key.
struct VpPipelineCache {
    std::mutex lock;
    std::map<std::string, std::shared_ptr<const VpHostPipeline>> made;
    std::map<std::string, std::string> refused;
    std::atomic<std::uint64_t> made_count{0}, shared_count{0}, refused_count{0};
};

// What identifies a pending pipeline's host pipeline: each stage's number, record, DXIL bytes and
// binding lists, then the graphics state.
inline std::string vp_pipeline_key(const VpPending& p, const VpMakeOptions& options)
{
    std::string key;
    const auto put = [&key](const void* data, std::size_t n) { key.append(static_cast<const char*>(data), n); };
    const char flags[3] = {options.root.bounded_root_views ? '1' : '0', options.root.heap_indexing ? '1' : '0',
                           options.allow_mesh ? '1' : '0'};
    put(flags, sizeof(flags));
    for (const auto& s : p.shaders) {
        put(&s.stage, sizeof(s.stage));
        // The record itself: two records with the same DXIL can still map user data, used views or threads differently, and
        // the made pipeline hands its stages' records to the bind path. Records are deduplicated by container key, so the
        // same container shares.
        put(&s.tag, sizeof(s.tag));
        const std::uint64_t n = s.dxil_size;
        put(&n, sizeof(n));
        // The DXIL by two 64-bit hashes, not by its bytes (a key per pipeline would hold them all).
        std::uint64_t a = 0xcbf29ce484222325ull, b = 0x84222325cbf29ce4ull;
        for (std::size_t i = 0; s.dxil && i < s.dxil_size; ++i) {
            const unsigned char c = static_cast<const unsigned char*>(s.dxil)[i];
            a = (a ^ c) * 0x100000001b3ull;
            b = (b + c + 1) * 0x9e3779b97f4a7c15ull;
            b ^= b >> 29;
        }
        put(&a, sizeof(a));
        put(&b, sizeof(b));
        put(&s.user_data_dwords, sizeof(s.user_data_dwords));
        const std::uint64_t counts[4] = {s.sampler_count, s.root_cbv_count, s.root_srv_count, s.root_uav_count};
        put(counts, sizeof(counts));
        if (s.sampler_count) put(s.samplers, s.sampler_count * sizeof(VpStaticSampler));
        if (s.root_cbv_count) put(s.root_cbvs, s.root_cbv_count);
        if (s.root_srv_count) put(s.root_srvs, s.root_srv_count);
        if (s.root_uav_count) put(s.root_uavs, s.root_uav_count);
    }
    if (p.graphics) p.graphics->append_key(key);
    return key;
}

// The host pipeline if it is made or can be made now; null if it is refused (p.why says why).
// Thread-safe; the first caller makes it.
inline const VpHostPipeline* vp_ready(VpPending& p, ID3D12Device* device, const VpDeviceCaps& caps, VpPipelineCache& cache,
                                      const VpMakeOptions& options, decltype(&vp_make_host_pipeline) build = vp_make_host_pipeline)
{
    std::call_once(p.once, [&] {
        if (p.missing) {
            p.why = "a stage has no translated record in the export";
            cache.refused_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            const std::string key = vp_pipeline_key(p, options);
            {
                std::lock_guard<std::mutex> held(cache.lock);
                const auto found = cache.made.find(key);
                if (found != cache.made.end()) {
                    p.made = found->second;
                    cache.shared_count.fetch_add(1, std::memory_order_relaxed);
                } else if (const auto no = cache.refused.find(key); no != cache.refused.end()) {
                    p.why = no->second;
                    cache.refused_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!p.made && p.why.empty()) {
                // Made outside the lock: a driver compile takes up to seconds.
                auto host = std::make_shared<VpHostPipeline>();
                std::string why;
                if (build(device, caps, p.shaders.data(), p.shaders.size(), p.graphics.get(), options, *host, why)) {
                    std::lock_guard<std::mutex> held(cache.lock);
                    auto& slot = cache.made[key];
                    if (!slot) {
                        slot = host;
                        cache.made_count.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        cache.shared_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    p.made = slot;
                } else {
                    std::lock_guard<std::mutex> held(cache.lock);
                    cache.refused[key] = why;
                    p.why = why;
                    cache.refused_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        p.done.store(true, std::memory_order_release);
    });
    return p.made.get();
}

}  // namespace x2w_gpu
