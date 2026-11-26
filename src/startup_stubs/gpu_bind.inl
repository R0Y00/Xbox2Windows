// Root argument binding and direct draws for translated shaders. Root arguments are recorded per command list. The
// binder locates title buffers in desktop memory windows and bounds each view by the title allocation.
//
// Table CBVs and misaligned root CBVs are copied at recording time into 256-byte-aligned upload-ring slots. The
// title must have populated the constants by then. Dynamic slots are returned after the last submission fence.
// Other shim operations can replace heaps, roots and PSOs, so cached bindings must be invalidated after such
// operations. Buffer barriers order access to the shared windows with a global UAV barrier.

using x2w_gpu::GpuListState;

namespace vbind {

namespace abi = x2w_gpu::abi;
using x2w_gpu::GpuResourceType;
using x2w_gpu::GpuRootType;
using x2w_gpu::GpuViewKind;

// ---- counters (one line per 300 frames, from gpu_bind_submitted) -------------------------------------------------
enum Drop : int {
    drop_no_pipeline, drop_wrong_kind, drop_no_view_heap, drop_no_slots, drop_memory_shader, drop_root_uav, drop_root_srv,
    drop_index_buffer, drop_rectangles, drop_not_recording, drop_invalid_targets, drop_count
};
LONG dispatched, drawn, dropped[drop_count];
thread_local Drop last_drop;
LONG user_misses, cbv_copies, cbv_copy_failed, cbv_null, root_uav_null, root_uav_unknown, root_uav_cut, root_srv_null,
    root_srv_cut, vertex_unbound, textures_uploaded, foreign_tables, slots_released, slots_kept, buffer_barriers,
    tables_skipped, table_views_missing, missing_sampler_heap;
bool drop(Drop why)
{
    InterlockedIncrement(&dropped[why]);
    if (gpu_drop_trace()) last_drop = why;
    return false;
}

bool bind_cache() { static const bool on = x2w_local::env_on("X2W_GPU_BIND_CACHE"); return on; }
// Caps for root views in MiB (as X2W_SHADERCFG_ROOT_VIEW_MB / X2W_SHADERCFG_ROOT_UAV_MB); unset: the title buffer alone.
std::uint64_t env_mb(const char* name)
{
    char v[24]{};
    const DWORD n = GetEnvironmentVariableA(name, v, sizeof(v));
    const unsigned long long mb = n && n < sizeof(v) ? std::strtoull(v, nullptr, 10) : 0;
    return mb ? mb << 20 : ~0ull;
}
std::uint64_t root_srv_limit() { static const std::uint64_t l = env_mb("X2W_GPU_ROOT_VIEW_MB"); return l; }
std::uint64_t root_uav_limit() { static const std::uint64_t l = env_mb("X2W_GPU_ROOT_UAV_MB"); return l; }
// Maximum readable range when no title buffer record covers an address (X2W_GPU_UNKNOWN_SRV_KB, default 1024;
// zero disables reads). Writes receive a null view.
std::uint64_t unknown_read_bytes()
{
    static const std::uint64_t b = [] {
        char v[24]{};
        const DWORD n = GetEnvironmentVariableA("X2W_GPU_UNKNOWN_SRV_KB", v, sizeof(v));
        return n && n < sizeof(v) ? std::strtoull(v, nullptr, 10) << 10 : (1ull << 20);
    }();
    return b;
}
// Textures uploaded per call from WP5's queue (X2W_GPU_UPLOADS_PER_CALL, default 64).
std::size_t upload_due_most()
{
    static const std::size_t n = [] {
        char v[24]{};
        const DWORD k = GetEnvironmentVariableA("X2W_GPU_UPLOADS_PER_CALL", v, sizeof(v));
        return k && k < sizeof(v) ? static_cast<std::size_t>(std::strtoull(v, nullptr, 10)) : std::size_t{64};
    }();
    return n;
}
ID3D12Resource* window_of(const x2w_vbind::Found& f) { return static_cast<ID3D12Resource*>(f.buffer); }

// ---- per-list state (own().gpu[0] of the list wrapper) ------------------------------------------------------------
GpuListState& state_of(Wrapper* list)
{
    auto& slot = list->own().gpu[0];  // ADAPT to WP0's type of the slot
    if (!slot) slot = new GpuListState();
    return *static_cast<GpuListState*>(slot);
}
GpuListState* state_if(Wrapper* list) { return static_cast<GpuListState*>(list->own().gpu[0]); }

#include "startup_stubs/gpu_compute_trace.inl"

std::optional<bool> prepare_packed_light(Wrapper* self, Wrapper* proxy, ID3D12Resource* destination,
                                         x2w_packed_light::BoundCopy* bound)
{
    if (!packed_light_enabled() || !proxy->own().linear_memory_srv) return {};
    const auto destination_format = destination->GetDesc().Format;
    if (!x2w_packed_light::color_bytes(destination_format)) return {};
    const auto* info = resource_info(proxy);
    if (!info || destination_format != info->console.Format) return {};
    const auto address = reinterpret_cast<u64>(proxy->own().title_memory);
    auto source = packed_light_source(address, info->console);
    if (!source) return {};
    // An explicit write to the view's own host resource already holds its data.
    if (list_wrote(self, destination)) return true;
    const auto& uploaded = texture_uploads();
    if (auto it = uploaded.find(destination); it != uploaded.end() && it->second.gpu_written) return true;
    if (bound && bound->contains(source.Get(), destination)) {
        static LONG64 skipped;
        const auto n = InterlockedIncrement64(&skipped);
        if ((n & (n - 1)) == 0) log("x2w-gpu repeated packed light bindings skipped %lld", n);
        return true;
    }
    auto& copies = state_of(self).light_copies;
    const auto found = std::find_if(copies.begin(), copies.end(), [&](const auto& c) {
        return c->source.Get() == source.Get() && c->destination.Get() == destination;
    });
    std::shared_ptr<x2w_packed_light::Copy> copy;
    if (found != copies.end()) copy = *found;
    else {
        copy = std::make_shared<x2w_packed_light::Copy>();
        auto* scratch = copies.empty() ? nullptr : copies.back()->scratch.Get();
        if (!copy->init(device_of(the_device), source.Get(), destination, scratch)) return false;
        copies.push_back(copy);
    }
    const auto previous = pass_of(self).states.state(source.Get());
    auto require = [&](auto* r, auto state) { require_state(self, r, state); };
    auto* list = recording(self);
    copy->record(list, require, previous);
    const LONG trace_frame = gpu_trace_frame;
    if (trace_frame && trace_frame != copy->last_trace_frame) {
        copy->last_trace_frame = trace_frame;
        log("x2w-frame alias copy frame=%ld list=%p source=%p destination=%p title=0x%llx size=%llux%u",
            trace_frame, static_cast<void*>(self), static_cast<void*>(source.Get()), static_cast<void*>(destination),
            address, info->console.Width, info->console.Height);
    }
    if (bound) *bound = {source.Get(), destination};
    require(destination, x2w_gpu::texture_read_state(list->GetType()));
    static LONG64 recorded;
    const auto n = InterlockedIncrement64(&recorded);
    const auto from = source->GetDesc().Format, to = destination_format;
    static std::atomic<u64> pairs[128][2]{};
    bool first_pair = false;
    if (static_cast<UINT>(from) < 128 && static_cast<UINT>(to) < 128) {
        auto& seen = pairs[from][to / 64];
        const u64 bit = u64{1} << (to % 64);
        if (!(seen.load(std::memory_order_relaxed) & bit))
            first_pair = !(seen.fetch_or(bit, std::memory_order_relaxed) & bit);
    }
    if (first_pair || (n & (n - 1)) == 0)
        log("x2w-gpu packed light GPU copies %lld: title 0x%llx, %llux%u, %zu subresources, format %u to %u",
            n, address, info->console.Width, info->console.Height, copy->source_rows.size(), from, to);
    return true;
}

// Slots of recordings that may still run, given back when their fences have passed.
struct Retiring {
    std::vector<x2w_vbind::Dynamic> slots;
    std::vector<std::pair<ID3D12Fence*, std::uint64_t>> fences;
    std::vector<std::shared_ptr<x2w_depth_twins::Twins>> depth_refs;
    std::vector<std::shared_ptr<x2w_vmemory::Recording>> memories;
    std::vector<std::shared_ptr<x2w_packed_light::Copy>> light_copies;
};
x2w_vmemory::FaultPages memory_fault_pages;
SRWLOCK retire_lock = SRWLOCK_INIT;
std::vector<Retiring>& retiring() { static std::vector<Retiring> v; return v; }
void release_slots(std::vector<x2w_vbind::Dynamic>& slots)
{
    for (const auto& d : slots) {
        gpu_heap_release(d.heap, d.first, d.n);
        InterlockedExchangeAdd(&slots_released, static_cast<LONG>(d.n));
    }
    slots.clear();
}
void retire_passed()
{
    std::vector<Retiring> passed;
    AcquireSRWLockExclusive(&retire_lock);
    auto& r = retiring();
    for (auto it = r.begin(); it != r.end();) {
        if (std::all_of(it->fences.begin(), it->fences.end(), [](const auto& f) { return f.first->GetCompletedValue() >= f.second; })) {
            passed.push_back(std::move(*it));
            it = r.erase(it);
        } else {
            ++it;
        }
    }
    ReleaseSRWLockExclusive(&retire_lock);
    for (auto& p : passed) {
        release_slots(p.slots);
        for (const auto& m : p.memories) {
            const auto faults = m->read_faults();
            const auto known = memory_fault_pages.report(faults);
            if (gpu_drop_trace()) {
                static LONG completed, nonempty;
                const LONG n = InterlockedIncrement(&completed);
                const LONG reports = faults.empty() ? 0 : InterlockedIncrement(&nonempty);
                if ((n & (n - 1)) == 0 || (reports && (reports & (reports - 1)) == 0))
                    log("x2w-gpu memory feedback: recording %ld pages reported %zu first 0x%x known %zu mapped %zu",
                        n, faults.size(), faults.empty() ? 0 : faults.front(), known, m->addresses.size());
            }
        }
    }
}
// The recording ends (reset or release): what it made is forgotten; its slots retire behind its submissions.
void end_recording(GpuListState& s)
{
    if (!s.dynamic.empty() || !s.depth_refs.empty() || !s.light_copies.empty()) {
        if (s.submitted.empty()) {
            release_slots(s.dynamic);  // never submitted: nothing on the GPU reads them
        } else {
            AcquireSRWLockExclusive(&retire_lock);
            retiring().push_back({std::move(s.dynamic), std::move(s.submitted), std::move(s.depth_refs), std::move(s.memories), std::move(s.light_copies)});
            ReleaseSRWLockExclusive(&retire_lock);
        }
    }
    s.dynamic.clear();
    s.trace_request = 0;
    s.trace_current.reset();
    s.trace_captures.clear();
    s.depth_refs.clear();
    s.memories.clear();
    s.memory_back.clear();
    s.light_copies.clear();
    s.writable_depth.clear();
    s.submitted.clear();
    s.fresh_tables.clear();
    s.table_constants.clear();
    s.root_srv_slots.clear();
    s.root_uav_slots.clear();
    s.host_heaps[0] = s.host_heaps[1] = nullptr;
    s.host_roots[0] = s.host_roots[1] = nullptr;
    s.host_pso = nullptr;
}
// Our fence per desktop queue, signalled after a submission that carried a translated recording holding slots.
SRWLOCK fence_lock = SRWLOCK_INIT;
std::unordered_map<ID3D12CommandQueue*, std::pair<ID3D12Fence*, std::uint64_t>>& queue_fences()
{
    static std::unordered_map<ID3D12CommandQueue*, std::pair<ID3D12Fence*, std::uint64_t>> m;
    return m;
}

// ---- heaps ----------------------------------------------------------------------------------------------------------
struct Bound {
    GpuHeapInfo views{}, samplers{};
    x2w_vbind::HeapSpan spans[2]{};  // view heap, sampler heap (start 0: none)
};
Bound bound_heaps(const GpuListState& s)
{
    Bound b;
    for (void* h : s.heaps) {
        GpuHeapInfo info{};
        if (!h || !gpu_heap_for(static_cast<Wrapper*>(h), &info) || !info.heap) continue;
        (info.sampler ? b.samplers : b.views) = info;
        b.spans[info.sampler ? 1 : 0] = {info.title_gpu, info.count, info.sampler};
    }
    return b;
}
// The bound heap a handle lies in, and its slot index there.
const GpuHeapInfo* heap_index(const Bound& b, std::uint64_t handle, std::uint32_t* index)
{
    if (b.views.heap)
        if (const auto i = b.spans[0].index(handle)) return *index = *i, &b.views;
    if (b.samplers.heap)
        if (const auto i = b.spans[1].index(handle)) return *index = *i, &b.samplers;
    return nullptr;
}
// X2W also accepts inline V# and X2WP records. Their format and payload extend past the token prefix.
bool record_words(const GpuHeapInfo& h, std::uint32_t index, std::uint32_t (&words)[8])
{
    if (h.sampler || !h.title_cpu || index >= h.count) return false;
    const auto* at = reinterpret_cast<const void*>(h.title_cpu + std::uint64_t{index} * abi::view_increment);
    static_assert(sizeof(words) == abi::view_increment);
    if (!x2w_local::readable(at, sizeof(words))) return false;
    std::memcpy(words, at, sizeof(words));
    return true;
}
// ---- window states --------------------------------------------------------------------------------------------------
// A window is title memory for every kind of read at once: one read state for vertex, index, constant and shader
// reads (so this path does not move a window between read states), unless the list holds it as UAV: one 256 MiB
// window often carries a UAV of the same call, and one state cannot be both (as the old path's window_read_state).
thread_local std::vector<ID3D12Resource*>* memory_bind_reads;
thread_local std::vector<ID3D12Resource*>* memory_bind_writes;
void window_read(Wrapper* self, ID3D12Resource* window)
{
    if (!window) return;
    if (memory_bind_reads) memory_bind_reads->push_back(window);
    if (pass_of(self).states.state(window) == D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return;
    require_state(self, window, x2w_gpu::buffer_read_state(recording(self)->GetType()));
}
void window_write(Wrapper* self, ID3D12Resource* window)
{
    if (window && memory_bind_writes) memory_bind_writes->push_back(window);
    if (window) require_state(self, window, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

// ---- readable: what a shader reads or writes through a view, made ready ---------------------------------
// Buffers: title memory in place, nothing to copy; their window in the state the access needs. Textures the title
// fills (no target, depth or unordered access): uploaded from title memory, as the old compute path does
// (d3d12_compute.inl texture_view). Textures read: both shader-read states, with DEPTH_READ when it is the depth
// target the draw tests; textures written: UNORDERED_ACCESS.
void readable(Wrapper* self, GpuResourceType kind, const GpuViewInfo& view, bool compute, x2w_packed_light::BoundCopy& bound)
{
    const bool write = kind == GpuResourceType::uav;
    if (view.buffer) {
        x2w_vbind::Found f;
        if (!view.title_address || !find(view.title_address, view.bytes ? view.bytes : 1, ~0ull, unknown_read_bytes(), &f)) return;
        write ? window_write(self, window_of(f)) : window_read(self, window_of(f));
        return;
    }
    if (!view.texture || !view.desktop) return;
    if (write) {
        if (view.kind != GpuViewKind::uav) return;
        bound.invalidate(view.desktop);
        if (view.depth_twins && view.desktop == view.depth_twins->colour.Get()) {
            auto& s = state_of(self);
            auto* pair = view.depth_twins.get();
            if (std::find(s.writable_depth.begin(), s.writable_depth.end(), pair) == s.writable_depth.end()) {
                if (std::find(s.depth_refs.begin(), s.depth_refs.end(), view.depth_twins) == s.depth_refs.end())
                    s.depth_refs.push_back(view.depth_twins);
                s.writable_depth.push_back(pair);
                const auto before = pass_of(self).states.state(pair->depth.Get());
                pair->copy_colour(recording(self), true, [&](auto* r, auto state) { require_state(self, r, state); });
                require_state(self, pair->depth.Get(), before);
            }
        }
        window_write(self, view.desktop);
        return;
    }
    if (view.kind != GpuViewKind::srv) return;
    ID3D12Resource* texture = view.desktop;
    Wrapper* resource = as_wrapper(view.title_resource);
    const D3D12_RESOURCE_DESC rd = texture->GetDesc();
    if (bc_bind_trace::enabled() && ((rd.Format >= DXGI_FORMAT_BC1_TYPELESS && rd.Format <= DXGI_FORMAT_BC5_SNORM) ||
        (rd.Format >= DXGI_FORMAT_BC6H_TYPELESS && rd.Format <= DXGI_FORMAT_BC7_UNORM_SRGB)))
        bc_bind_trace::read(self, resource, texture, rd);
    if (resource && resource->own().kind == resource_kind &&
        !(rd.Flags & (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))) {
        warm_texture(resource);
        memcut_revive(resource);
        const auto alias = prepare_packed_light(self, resource, texture, &bound);
        if (!alias) bound.invalidate(texture);
        if (alias ? *alias : upload_texture(self, resource, texture)) InterlockedIncrement(&textures_uploaded);
    }
    D3D12_RESOURCE_STATES state = x2w_gpu::texture_read_state(recording(self)->GetType());
    if (!compute) require_sampled(self, texture, state);
    else require_state(self, texture, state);
}

// ---- constant buffers: the console wants 16-byte alignment, the host 256 ---------------------------------------------------
bool defer_table_cbv()
{
    static const bool on = x2w_local::env_on("X2W_GPU_DEFER_TABLE_CBV");
    return on;
}
// A copy of the title's constants in the upload ring (record time, see the header): 256-byte steps, the tail past the
// readable bytes zeroed. The ring places at 512 bytes (reserve_upload_ring), which covers a CBV's 256. 0: no copy.
D3D12_GPU_VIRTUAL_ADDRESS constant_copy(std::uint64_t address, std::uint64_t bytes, ID3D12Resource** buffer, std::uint32_t* size,
                                        x2w_gpu::TableConstants* deferred = nullptr)
{
    const std::size_t readable = readable_window(address, static_cast<std::size_t>((std::min<std::uint64_t>)(bytes, 1ull << 16)));
    if (!readable) {
        InterlockedIncrement(&cbv_copy_failed);
        return 0;
    }
    std::uint64_t offset = 0;
    const std::size_t want = x2w_vbind::constant_view_bytes(readable);
    RingTag tag;
    unsigned char* dest = reserve_upload_ring(want, &offset, buffer, deferred ? &tag : nullptr);
    if (!dest || !*buffer || (offset & 255)) {
        InterlockedIncrement(&cbv_copy_failed);
        return 0;
    }
    const void* source = reinterpret_cast<const void*>(address);
    if (deferred) source = deferred->remember(source, readable, {tag.chunk, tag.generation, offset});
    std::memcpy(dest, source, readable);
    std::memset(dest + readable, 0, want - readable);
    InterlockedIncrement(&cbv_copies);
    *size = static_cast<std::uint32_t>(want);
    return (*buffer)->GetGPUVirtualAddress() + offset;
}

bool take_dynamic(GpuListState& s, GpuHostHeap* heap, std::uint32_t n, std::uint32_t* first)
{
    if (!gpu_heap_dynamic(heap, n, first)) return false;
    s.dynamic.push_back({heap, *first, n});
    return true;
}

// ---- tables ------------------------------------------------------------------------------
UINT64 trace_min_width()
{
    static const auto width = gpu_pipeline_setting("X2W_TRACE_TARGET_MIN_WIDTH", 0, D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION);
    return width;
}

bool trace_pass(Wrapper* self, bool compute)
{
    if (compute || !trace_min_width()) return true;
    const Pass& pass = pass_of(self);
    return std::any_of(pass.targets, pass.targets + (std::min)(pass.target_count, 8u), [](ID3D12Resource* target) {
        return target && target->GetDesc().Width >= trace_min_width();
    });
}

bool trace_stage_matches(const char* key)
{
    const auto filter = gpu_trace_stage_key();
    if (!filter || (key && _stricmp(filter->c_str(), key) == 0)) return true;
    const auto second = std::atomic_load(&gpu_scene_second_override);
    return second && key && !_stricmp(second->c_str(), key);
}

bool trace_tables(Wrapper* self, bool compute, const char* key)
{
    return gpu_trace_frame && trace_stage_matches(key) && gpu_trace_per_kind[compute ? 1 : 0] < 64 && trace_pass(self, compute);
}

void light_binding(GpuListState& s, UINT slot, UINT offset, GpuResourceType kind, const GpuViewInfo& view,
                   GpuHostHeap* heap, UINT index)
{
    using x2w_gpu::ambient::Role;
    if (s.light_role == Role::none || !view.texture || !view.desktop) return;
    const auto desc = view.desktop->GetDesc();
    if (s.light_role == Role::resolve && slot == 0 && offset == 464 && kind == GpuResourceType::srv &&
        view.kind == GpuViewKind::srv && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        s.light_views.depth = gpu_heap_gpu_handle(heap, index);
        s.light_depth_width = desc.Width;
        s.light_depth_height = desc.Height;
    }
    const bool volume = s.light_role == Role::air;
    const UINT output = volume ? 32 : s.light_role == Role::radiance ? 16 : 176;
    const auto format = s.light_role == Role::resolve ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R11G11B10_FLOAT;
    if (slot != 1 || offset != output || kind != GpuResourceType::uav || view.kind != GpuViewKind::uav ||
        desc.Format != format || desc.Dimension != (volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D)) return;
    s.light_views.image = gpu_heap_gpu_handle(heap, index);
    s.light_views.width = UINT(desc.Width);
    s.light_views.height = desc.Height;
    s.light_views.slices = desc.DepthOrArraySize;
    if (s.light_role != Role::resolve) s.light_views.depth = s.light_views.image;
}

void finish_light(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    using namespace x2w_gpu::ambient;
    auto& s = state_of(self);
    if (s.light_role == Role::none || !s.light_views.depth.ptr || !s.light_views.image.ptr ||
        (s.light_role == Role::resolve && (s.light_depth_width != s.light_views.width ||
                                          s.light_depth_height != s.light_views.height))) return;
    static std::mutex lock;
    static std::unordered_map<ID3D12Device*, std::unique_ptr<x2w_gpu::ambient::Pass>> passes;
    std::lock_guard<std::mutex> guard(lock);
    auto* device = device_of(the_device);
    auto& pass = passes[device];
    if (!pass) pass = std::make_unique<x2w_gpu::ambient::Pass>();
    if (pass->record(device, list, s.light_role, indirect_light_settings(), s.light_views)) {
        s.host_roots[1] = nullptr;
        s.host_pso = nullptr;
        static LONG calls[4]{};
        const auto n = InterlockedIncrement(&calls[unsigned(s.light_role)]);
        if ((n & (n-1)) == 0) log("x2w-gpu indirect light recorded role=%u count=%ld size=%ux%u",
                                    unsigned(s.light_role), n, s.light_views.width, s.light_views.height);
    } else if (FAILED(pass->error())) {
        static LONG errors;
        if (InterlockedIncrement(&errors) == 1) log("x2w-gpu indirect light init HRESULT %08lx: %s", pass->error(), pass->diagnostic().c_str());
    }
}

void inventory_uav_binding(Wrapper* self, const char* shader, UINT slot, UINT offset, const GpuViewInfo& view)
{
    static const bool enabled = x2w_local::env_on("X2W_TRACE_PASS_INVENTORY");
    const LONG frame = gpu_trace_frame;
    if (!frame || !enabled || !view.desktop) return;
    static std::mutex mutex;
    static LONG observed{};
    static std::set<std::tuple<ID3D12Resource*, const char*, UINT, UINT>> seen;
    {
        std::lock_guard guard(mutex);
        if (frame < observed) return;
        if (frame != observed) { observed = frame; seen.clear(); }
        if (seen.size() >= 4096 || !seen.emplace(view.desktop, shader, slot, offset).second) return;
    }
    const auto d = view.desktop->GetDesc();
    log("x2w-frame UAV binding frame=%ld list=%p shader=%s slot=%u offset=%u resource=%p format=%u size=%llux%u",
        frame, static_cast<void*>(self), shader, slot, offset, static_cast<void*>(view.desktop), UINT(view.view_format), d.Width, d.Height);
}

// Report the materialized descriptor, including dynamically indexed texture slots.
void trace_texture_binding(Wrapper* self, bool compute, const char* key, UINT slot, UINT offset,
                           GpuResourceType kind, const GpuViewInfo& view)
{
    if (!view.texture) return;
    if (gpu_trace_frame && compute && kind == GpuResourceType::uav && view.kind == GpuViewKind::uav)
        inventory_uav_binding(self, key, slot, offset, view);
    if (!compute && pixel_capture_on() && compute_capture_pending.load()) {
        auto& s = state_of(self);
        if (s.pixel_capture_shader && !std::strcmp(s.pixel_capture_shader->key, key) && view.desktop &&
            kind == GpuResourceType::srv && view.kind == GpuViewKind::srv && s.pixel_bindings.size() < 40) {
            const auto& bindings = s.pixel_bindings;
            if (std::none_of(bindings.begin(), bindings.end(), [&](const auto& entry) { return entry.slot == slot && entry.offset == offset; }))
                s.pixel_bindings.push_back({view.desktop, slot, offset, false, false, view.first_mip, view.first_slice, view.view_format, view.plane});
        }
    }
    if (compute && compute_capture_on() && compute_capture_pending.load()) {
        auto& s = state_of(self);
        if (s.trace_request && s.trace_request == compute_capture_pending.load() &&
            ((kind == GpuResourceType::uav && view.kind == GpuViewKind::uav) ||
             (kind == GpuResourceType::srv && view.kind == GpuViewKind::srv)))
        {
            using Role = x2w_gpu::ambient::Role;
            auto* pipeline = static_cast<Wrapper*>(s.pipeline);
            const auto* entry = pipeline ? static_cast<const GpuPipelineEntry*>(pipeline->own().gpu[0]) : nullptr;
            const auto role = entry ? entry->light_role : Role::none;
            const bool priority = (role == Role::resolve && ((slot == 0 && offset == 464) || (slot == 1 && offset == 176))) ||
                (role == Role::radiance && slot == 1 && offset == 16) ||
                (role == Role::air && slot == 1 && offset == 32);
            static const bool wide = x2w_local::env_on("X2W_TRACE_COMPUTE_WIDE");
            s.trace_bindings.add({view.desktop, slot, offset, kind == GpuResourceType::uav, false,
                view.first_mip, view.first_slice, view.view_format, view.plane}, priority, wide);
        }
    }
    if (!gpu_trace_frame || !gpu_trace_stage_key() || !trace_tables(self, compute, key)) return;
    if (InterlockedIncrement(&gpu_trace_bindings) > 256) return;
    const auto d = view.desktop ? view.desktop->GetDesc() : D3D12_RESOURCE_DESC{};
    log("x2w-frame resolved texture frame=%ld list=%p shader=%s slot=%u offset_dwords=%u access=%s "
        "resource=%p format=%u size=%llux%u depth_or_array=%u dim=%u mips=%u", gpu_trace_frame, static_cast<void*>(self), key, slot, offset,
        kind == GpuResourceType::uav ? "UAV" : "SRV", static_cast<void*>(view.desktop), UINT(d.Format), d.Width, d.Height, UINT(d.DepthOrArraySize), UINT(d.Dimension), UINT(d.MipLevels));
}

// What a stage reads through tables: entries mirrored in place, or a fresh table in dynamic slots for tables with a CBV
// and tables past the host heap. `made` gets the fresh tables' byte offsets; false: one found no slots (drop the call).
bool tables(Wrapper* self, GpuListState& s, const Bound& b, const x2w_gpu::GpuShader& sh, bool compute,
            std::vector<x2w_vbind::MadeTable>& made, x2w_packed_light::BoundCopy& bound)
{
    const auto plan = x2w_vbind::plan_tables(s.roots, compute, sh.used.data, sh.used.count, sh.root_indexed_views.data,
                                             sh.root_indexed_views.count, b.spans, 2,
                                             [](const x2w_vbind::HeapSpan& h) { return h.count; });
    if (!plan.skipped.empty()) InterlockedExchangeAdd(&tables_skipped, static_cast<LONG>(plan.skipped.size()));
    for (const std::size_t i : plan.mirrored) {
        const auto& u = sh.used[static_cast<std::uint32_t>(i)];
        const x2w_gpu::GpuRoot* r = s.roots.at(compute, u.slot);
        if (!r || r->kind != GpuRootKind::table) continue;
        const std::uint64_t handle = r->handle + std::uint64_t{u.table_offset_dwords} * 4;
        std::uint32_t index = 0;
        const GpuHeapInfo* h = heap_index(b, handle, &index);
        if (!h) {
            InterlockedIncrement(&foreign_tables);
            continue;
        }
        GpuViewInfo view{};
        const bool mirrored = gpu_heap_mirror(h->heap, index, handle, true, &view);
        if (mirrored && !h->sampler) {
            light_binding(s, u.slot, u.table_offset_dwords, u.kind, view, h->heap, index);
            trace_texture_binding(self, compute, sh.key, u.slot, u.table_offset_dwords, u.kind, view);
            readable(self, u.kind, view, compute, bound);
        }
        if (trace_tables(self, compute, sh.key))
            log("x2w-frame table %s slot=%u offset=%u heap-index=%u mirrored=%d desktop=%p kept=%p",
                sh.key, u.slot, u.table_offset_dwords, index, int(mirrored), static_cast<void*>(view.desktop),
                static_cast<void*>(gpu_heap_kept_at(h->heap, index)));
    }
    for (const auto& f : plan.fresh) {
        const GpuHeapInfo& h = b.views;
        const std::uint32_t title_first = *b.spans[0].index(f.table);
        const auto entry = [&](std::uint32_t k) -> const x2w_gpu::GpuUsedResource* {
            for (const auto& u : sh.used)
                if (x2w_vbind::in_table(u) && u.slot == f.slot && u.table_offset_dwords / 8 == k) return &u;
            return nullptr;
        };
        const auto selected = [&](std::uint32_t k) { return std::binary_search(f.selected.begin(), f.selected.end(), k); };
        const auto kind_of = [&](std::uint32_t k, const GpuViewInfo& view) {
            const auto* u = entry(k);
            return u ? u->kind : view.kind == GpuViewKind::uav ? GpuResourceType::uav : GpuResourceType::srv;
        };
        // Key the table by its address and the first words of its descriptor records.
        auto* mirror = gpu_detail_heap::host_of(h.heap);
        std::vector<std::pair<std::uint32_t, gpu_mirror::View>> late_views;
        const auto late_view = [&](std::uint32_t k) -> const gpu_mirror::View* {
            const auto it = std::lower_bound(late_views.begin(), late_views.end(), k,
                [](const auto& item, std::uint32_t offset) { return item.first < offset; });
            return it != late_views.end() && it->first == k ? &it->second : nullptr;
        };
        const auto table_view = [&](std::uint32_t k, GpuViewInfo& info) {
            const auto* v = late_view(k);
            if (!v) return gpu_heap_view_at(h.heap, title_first + k, &info);
            gpu_detail_heap::view_info(*v, &info);
            return info.kind != GpuViewKind::none;
        };
        x2w_vbind::RecordHash hash;
        for (std::uint32_t k = 0; k < f.n; ++k) {
            std::uint32_t w[8]{};
            record_words(h, title_first + k, w);
            hash.add_record(w);
            if (x2w_late_texture::marker(w[0]) && mirror) {
                auto& texture = late_views.emplace_back(k, gpu_mirror::View{}).second;
                mirror->view_at(title_first + k, w, texture);
                texture.retained = texture.resource;
                const auto host = reinterpret_cast<std::uint64_t>(texture.resource);
                hash.add(k, static_cast<std::uint32_t>(texture.key), static_cast<std::uint32_t>(texture.key >> 32));
                hash.add(k, static_cast<std::uint32_t>(host), static_cast<std::uint32_t>(host >> 32));
            }
            const auto* u = entry(k);
            if (u && u->kind == GpuResourceType::cbv) hash.add(k, 0, 0x434256);
        }
        const x2w_vbind::FreshKey key{f.table, f.n, hash.value};
        if (const auto known = s.fresh_tables.find(key); known != s.fresh_tables.end()) {
            // Made before in this list: what it reads made ready again (states may have moved since).
            for (std::uint32_t k = 0; k < f.n; ++k) {
                const auto* u = entry(k);
                if ((u && u->kind == GpuResourceType::cbv) || (!u && !selected(k))) continue;
                GpuViewInfo view{};
                if (!table_view(k, view)) continue;
                if (view.texture) view.desktop = gpu_heap_kept_at(h.heap, known->second + k);
                if (trace_tables(self, compute, sh.key) && u)
                    log("x2w-frame fresh cached %s slot=%u source=%u dynamic=%u kind=%u address=%llx kept=%p",
                        sh.key, f.slot, title_first + k, known->second + k, static_cast<unsigned>(view.kind),
                        view.title_address, static_cast<void*>(gpu_heap_kept_at(h.heap, known->second + k)));
                light_binding(s, f.slot, k * 8, kind_of(k, view), view, h.heap, known->second + k);
                trace_texture_binding(self, compute, sh.key, f.slot, k * 8, kind_of(k, view), view);
                readable(self, kind_of(k, view), view, compute, bound);
            }
            made.push_back({f.slot, known->second * abi::view_increment});
            continue;
        }
        std::uint32_t first = 0;
        if (!take_dynamic(s, h.heap, f.n, &first)) {
            made.push_back({f.slot, UINT32_MAX});
            continue;
        }
        s.fresh_tables.emplace(key, first);
        // The whole range: a shader may index past the entries it names (descriptor arrays).
        for (std::uint32_t k = 0; k < f.n; ++k) {
            const std::uint32_t index = first + k;
            GpuViewInfo view{};
            if (!table_view(k, view) || view.kind == GpuViewKind::none) {
                gpu_heap_make_null(h.heap, index);
                if (const auto* use = entry(k)) {
                    InterlockedIncrement(&table_views_missing);
                    if (gpu_drop_trace()) {
                        static LONG samples;
                        const LONG n = InterlockedIncrement(&samples);
                        if (n <= 16 || (n & (n - 1)) == 0 || trace_tables(self, compute, sh.key)) {
                            std::uint32_t w[8]{};
                            const bool read = record_words(h, title_first + k, w);
                            const bool empty = std::all_of(std::begin(w), std::end(w), [](UINT word) { return word == 0; });
                            const char* form = !read ? "unreadable" : empty ? "empty" :
                                w[0] == gpu_mirror::token_magic ? "captured" : "inline";
                            log("x2w-gpu missing table view: frame=%ld count=%ld key=%s slot=%u offset=%u kind=%u title-index=%u form=%s words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                                frame_number, n, sh.key, f.slot, use->table_offset_dwords, unsigned(use->kind),
                                title_first + k, form, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
                            if (w[0] == gpu_mirror::token_magic && (w[4] || w[5])) {
                                gpu_mirror::Record r;
                                const bool have = gpu_detail_heap::env_record(nullptr, w[1], &r);
                                Wrapper* owner = have ? as_wrapper(reinterpret_cast<const void*>(r.resource)) : nullptr;
                                UINT fields[2]{};
                                if (r.desc_bytes >= sizeof(fields)) std::memcpy(fields, r.desc, sizeof(fields));
                                log("x2w-gpu unresolved captured view: key=%s id=%x recorded=%d original-created=%d wrapper=%p refs=%ld host=%p desc-bytes=%u format=%u dimension=%u",
                                    sh.key, w[1], int(have), int(have && r.materialized), owner,
                                    owner ? InterlockedCompareExchange(&owner->own().refs, 0, 0) : -1L,
                                    owner ? owner->own().real : nullptr, r.desc_bytes, fields[0], fields[1]);
                            }
                        }
                    }
                }
                continue;
            }
            const auto* usage = entry(k);
            // A console buffer record also describes the byte range of a table constant.
            // The desktop shader accesses that slot with a CBV-typed heap handle.
            const bool constant_buffer = usage && usage->kind == GpuResourceType::cbv &&
                                         view.kind == GpuViewKind::srv && view.buffer;
            if (view.kind == GpuViewKind::cbv || constant_buffer) {
                ID3D12Resource* buffer = nullptr;
                std::uint32_t size = 0;
                const D3D12_GPU_VIRTUAL_ADDRESS gpu = constant_copy(view.title_address, view.bytes, &buffer, &size,
                                                                   defer_table_cbv() ? &s.table_constants : nullptr);
                if (!gpu) {
                    InterlockedIncrement(&cbv_null);
                    gpu_heap_make_null(h.heap, index);
                    continue;
                }
                gpu_heap_constant_at(h.heap, index, buffer, gpu, size);
                if (trace_tables(self, compute, sh.key))
                    log("x2w-frame fresh constant %s slot=%u source=%u dynamic=%u address=%llx bytes=%llu copied=%u",
                        sh.key, f.slot, title_first + k, index, view.title_address, view.bytes, size);
                continue;
            }
            // The title's view, made in the dynamic slot.
            std::uint32_t words[8]{};
            GpuViewInfo made_view{};
            bool materialized = false;
            if (const auto* resolved = late_view(k)) {
                const auto& texture = *resolved;
                std::copy(texture.placement.begin(), texture.placement.end(), words);
                mirror->make_at(index, texture);
                made_view = view;
                made_view.desktop = mirror->kept_at(index);
                materialized = true;
            } else {
                materialized = record_words(h, title_first + k, words) &&
                    gpu_heap_make_listed(h.heap, title_first + k, words, index, &made_view);
            }
            if (trace_tables(self, compute, sh.key) && entry(k))
                log("x2w-frame fresh %s slot=%u source=%u dynamic=%u kind=%u prefix=%08x w3=%08x made=%d address=%llx kept=%p",
                    sh.key, f.slot, title_first + k, index, static_cast<unsigned>(view.kind), words[0], words[3],
                    int(materialized), view.title_address, static_cast<void*>(gpu_heap_kept_at(h.heap, index)));
            if (!materialized) {
                gpu_heap_make_null(h.heap, index);
                continue;
            }
            if (entry(k) || selected(k)) {
                light_binding(s, f.slot, k * 8, kind_of(k, made_view), made_view, h.heap, index);
                trace_texture_binding(self, compute, sh.key, f.slot, k * 8, kind_of(k, made_view), made_view);
                readable(self, kind_of(k, made_view), made_view, compute, bound);
            }
        }
        made.push_back({f.slot, first * abi::view_increment});
    }
    return std::none_of(made.begin(), made.end(), [](const x2w_vbind::MadeTable& m) { return m.offset == UINT32_MAX; });
}

// ---- root views ------------------------------------------------------------
// A root SRV as a one-view table: a raw view from `address` to the end of its title buffer (from the 16 bytes below
// when the shader adds the bias), or a null view. One dynamic slot per address per list.
bool root_srv_table(Wrapper* self, GpuListState& s, const Bound& b, std::uint64_t address, bool biased, std::uint32_t* index,
                    std::uint32_t* bias)
{
    *bias = 0;
    if (!b.views.heap) return false;
    const auto key = std::make_pair(static_cast<const void*>(b.views.heap), address | (std::uint64_t{biased} << 63));
    if (const auto known = s.root_srv_slots.find(key); known != s.root_srv_slots.end()) {
        *index = known->second.index;
        *bias = known->second.bias;
        window_read(self, known->second.window);
        return true;
    }
    if (!take_dynamic(s, b.views.heap, 1, index)) return false;
    x2w_vbind::RootSlot& slot = s.root_srv_slots.emplace(key, x2w_vbind::RootSlot{*index, nullptr, 0}).first->second;
    x2w_vbind::Found f;
    x2w_vbind::RawWindow raw;
    if (address && find(address, 1, root_srv_limit(), unknown_read_bytes(), &f) && x2w_vbind::raw_window(f.offset, f.left, biased, &raw)) {
        if (f.cut_by_window) InterlockedIncrement(&root_srv_cut);
        gpu_heap_raw_at(b.views.heap, *index, window_of(f), raw.first, raw.bytes);
        // Reuse the bias from the window offset at which the view was made.
        *bias = slot.bias = raw.bias;
        slot.window = window_of(f);
        window_read(self, slot.window);
    } else {
        InterlockedIncrement(&root_srv_null);
        gpu_heap_null_raw_at(b.views.heap, *index);
    }
    return true;
}
// A root UAV as a one-view table: a raw UAV from the 16 bytes below `address` to the end of its title buffer, or a null
// view (no address, memory no title buffer record holds, a window shaders cannot write).
bool root_uav_table(Wrapper* self, GpuListState& s, const Bound& b, std::uint64_t address, std::uint32_t* index, std::uint32_t* bias)
{
    *bias = 0;
    if (!b.views.heap) return false;
    const auto key = std::make_pair(static_cast<const void*>(b.views.heap), address);
    if (const auto known = s.root_uav_slots.find(key); known != s.root_uav_slots.end()) {
        *index = known->second.index;
        *bias = known->second.bias;
        window_write(self, known->second.window);
        return true;
    }
    if (!take_dynamic(s, b.views.heap, 1, index)) return false;
    x2w_vbind::RootSlot& slot = s.root_uav_slots.emplace(key, x2w_vbind::RootSlot{*index, nullptr, 0}).first->second;
    x2w_vbind::Found f;
    // No allowance for memory no title buffer record holds (unknown_most 0): a write there could land anywhere.
    const bool found = address && find(address, 1, root_uav_limit(), 0, &f);
    if (address && !found) InterlockedIncrement(f.known_extent ? &root_uav_null : &root_uav_unknown);
    x2w_vbind::RawWindow raw;
    if (found && x2w_vbind::raw_window(f.offset, f.left, true, &raw) &&
        (window_of(f)->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
        if (f.cut_by_window) InterlockedIncrement(&root_uav_cut);
        gpu_heap_raw_uav_at(b.views.heap, *index, window_of(f), raw.first, raw.bytes);
        *bias = slot.bias = raw.bias;
        slot.window = window_of(f);
        window_write(self, slot.window);
    } else {
        gpu_heap_null_raw_uav_at(b.views.heap, *index);
    }
    return true;
}
// A root view bound as a root descriptor (X2W_SHADERCFG_ROOT_SRV_DIRECT=1 pipelines, and root CBVs): its GPU address, from the
// 16 bytes below for a biased SRV or a UAV, with the bias it then needs. 0: nothing to bind.
D3D12_GPU_VIRTUAL_ADDRESS root_direct(Wrapper* self, std::uint64_t address, bool write, bool biased, std::uint32_t* bias)
{
    *bias = 0;
    x2w_vbind::Found f;
    if (!address || !find(address, 1, write ? root_uav_limit() : root_srv_limit(), write ? 0 : unknown_read_bytes(), &f)) return 0;
    write ? window_write(self, window_of(f)) : window_read(self, window_of(f));
    const D3D12_GPU_VIRTUAL_ADDRESS gpu = window_of(f)->GetGPUVirtualAddress() + f.offset;
    if (!write && !biased) return gpu;
    *bias = static_cast<std::uint32_t>(f.offset & 15);
    return gpu - *bias;
}
// A root CBV: in place when 256-aligned (64 KiB at most, cut at its buffer's end), else a copy in the upload ring.
D3D12_GPU_VIRTUAL_ADDRESS root_cbv(Wrapper* self, std::uint64_t address)
{
    x2w_vbind::Found f;
    if (!address || !find(address, 1, 1ull << 16, unknown_read_bytes(), &f)) {
        InterlockedIncrement(&cbv_null);
        return 0;
    }
    const D3D12_GPU_VIRTUAL_ADDRESS gpu = window_of(f)->GetGPUVirtualAddress() + f.offset;
    if (gpu % 256 == 0) {
        window_read(self, window_of(f));
        return gpu;
    }
    ID3D12Resource* buffer = nullptr;
    std::uint32_t size = 0;
    return constant_copy(address, x2w_vbind::root_cbv_bytes(f.left), &buffer, &size);
}

// ---- the bind sequence -----------------------------------------------------------------------------
bool shader_reads_memory(const x2w_gpu::GpuShader& sh)
{
    for (const auto s : sh.root_srvs)
        if (s == abi::slot_memory) return true;
    for (const auto u : sh.root_uavs)
        if (u == abi::slot_faults) return true;
    return false;
}
bool biased_view(const x2w_gpu::GpuShader& sh, std::uint32_t slot)
{
    for (const auto& u : sh.user_data)
        if (u.slot == slot && u.root == GpuRootType::other && u.other == abi::user_view_bias) return true;
    return false;
}

u64 buffer_address(const Bound& b, const GpuListState& s, bool compute, UINT slot, UINT dword)
{
    const auto* r = s.roots.at(compute, slot);
    if (!r || r->kind != GpuRootKind::table) return 0;
    UINT index = 0;
    const auto* heap = heap_index(b, r->handle + u64{dword} * 4, &index);
    return heap && !heap->sampler ? gpu_heap_title_address_at(heap->heap, index) : 0;
}

bool memory_window(u64 at, ID3D12Resource** resource, u64* offset)
{
    using Location = BOOL(WINAPI*)(const void*, HANDLE*, ULONG64*);
    static const auto location = reinterpret_cast<Location>(GetProcAddress(GetModuleHandleW(L"xmem.dll"), "x2w_xmem_gpu_location"));
    HANDLE section = nullptr;
    u64 where = 0, start = 0, end = 0;
    if (!(location && location(reinterpret_cast<const void*>(at), &section, &where)) &&
        !(known_private(at, &start, &end) && end - at >= x2w_vmemory::page_bytes)) {
        if (!builder_memory_on()) return false;
        std::lock_guard<std::mutex> guard(builder_upload_lock);
        if (std::none_of(builder_upload_blocks.begin(), builder_upload_blocks.end(),
                         [&](u64 base) { return at >= base && at - base <= (64ull << 20) - x2w_vmemory::page_bytes; })) return false;
    }
    return desktop_range(at, x2w_vmemory::page_bytes, resource, offset);
}

enum class MemoryRefusal { store, heap, allocation, overlapping_read, output, arena_input, directory_capacity, descriptor_capacity, count };
bool refuse_memory(const x2w_gpu::GpuShader& sh, MemoryRefusal reason)
{
    static constexpr const char* names[] = {"address-store", "view-heap", "directory-allocation",
                                             "window-also-bound-for-read", "unmapped-declared-uav", "arena-input-or-capacity", "live-directory-budget", "dynamic-descriptor-slots"};
    static LONG hits[unsigned(MemoryRefusal::count)]{};
    const LONG n = InterlockedIncrement(&hits[unsigned(reason)]);
    if (n <= 4 || (n & (n - 1)) == 0)
        log("x2w-gpu memory refusal: frame=%ld reason=%s count=%ld key=%s", frame_number, names[unsigned(reason)], n, sh.key);
    return drop(drop_memory_shader);
}

#include "gpu_memory_store.inl"

x2w_vmemory::Recording* memory_ready(Wrapper* self, ID3D12GraphicsCommandList* list, GpuListState& s,
                                     const Bound& b, const x2w_gpu::GpuShader& shader,
                                     std::vector<x2w_vmemory::CopyBack>& pending)
{
    // Declared root UAV instructions can lower to a raw store through the directory
    // even with stores_by_address=false. Read-only directory entries cannot serve them.
    const bool root_store = shader.stores && std::any_of(shader.root_uavs.begin(), shader.root_uavs.end(),
        [](auto slot) { return slot != abi::slot_faults; });
    const bool writes = root_store || shader.builder || shader.stores_by_address;
    static const bool cpu_pages = x2w_local::env_on("X2W_GPU_MEMORY_CPU_PAGES");
    const bool arena = writes ? builder_memory_on() && directory_only_builder(shader) : cpu_pages || builder_memory_on();
    if (!b.views.heap || (writes && !arena)) {
        refuse_memory(shader, b.views.heap ? MemoryRefusal::store : MemoryRefusal::heap);
        return nullptr;
    }
    x2w_vmemory::Recording* m = nullptr;
    for (auto& it : s.memories) if (it->heap == b.views.heap && it->writable_mode == writes) m = it.get();
    if (!m) {
        auto created = std::make_shared<x2w_vmemory::Recording>();
        bool at_capacity = false;
        if (!created->create(device_of(the_device), &at_capacity)) {
            refuse_memory(shader, at_capacity ? MemoryRefusal::directory_capacity : MemoryRefusal::allocation);
            return nullptr;
        }
        if (!take_dynamic(s, b.views.heap, x2w_vmemory::slots, &created->base)) {
            refuse_memory(shader, MemoryRefusal::descriptor_capacity);
            return nullptr;
        }
        created->heap = b.views.heap;
        created->writable_mode = writes;
        gpu_heap_null_raw_uav_at(created->heap, created->base);
        s.memories.push_back(created);
        m = created.get();
    }
    m->restore_read_aliases(list, [&](auto* r, auto state) { require_state(self, r, state); });
    if (arena) {
        if (builder_pages(self,list,s,b,shader,*m,pending)) return m;
        refuse_memory(shader, MemoryRefusal::arena_input);
        return nullptr;
    }
    const auto page = [&](UINT number) {
        m->map(list, number, memory_window,
            [&](UINT index, auto* w, UINT64 size) { gpu_heap_raw_uav_at(m->heap, index, w, 0, size); },
            [&](auto* r, auto state) { require_state(self, r, state); });
    };
    const auto span = [&](u64 at, u64 len) {
        if (at < (1ull << 32) || at >= (u64{abi::memory_pages - 1} << 16)) return;
        const auto first = at >> 16, last = (at + len - 1) >> 16;
        if (m->ranges.insert((first << 32) | last).second)
            for (auto n = first; n <= last; ++n) page(UINT(n));
    };
    for (const auto& root : s.roots.compute) {
        if (shader.builder && x2w_vbind::is_view(root.kind))
            span(root.address, root.kind == GpuRootKind::uav ? 4ull << 20 : 256ull << 10);
        if (root.kind != GpuRootKind::constants) continue;
        for (size_t i = 0; i < root.dwords.size(); ++i) {
            const UINT lo = root.dwords[i];
            if (i + 1 < root.dwords.size() && lo >= 65536 && root.dwords[i + 1] >= 1 && root.dwords[i + 1] <= 31)
                span((u64{root.dwords[i + 1]} << 32) | lo, 1);
            if (shader.builder && lo >= 0x01000000 && lo < 0x20000000) span(u64{lo} << 8, 1);
        }
    }
    for (const auto& u : shader.user_data)
        if (u.root == GpuRootType::other && u.other == abi::user_view_address)
            span(buffer_address(b, s, true, u.slot, u.dword), 256ull << 10);
    if (!m->faults_seeded) {
        memory_fault_pages.seed(frame_number, [&](UINT n) { page(n); return m->addresses.count(n) != 0; });
        m->faults_seeded = true;
    }
    return m;
}

bool bind(Wrapper* self, ID3D12GraphicsCommandList* l, GpuListState& s, const GpuPipeline& p, bool compute)
{
    x2w_packed_light::BoundCopy bound_copy;
    const Bound b = bound_heaps(s);
    static const bool lazy = x2w_local::env_on("X2W_GPU_LAZY_DETILE");
    if (lazy && b.views.heap) {
        static gpu_mirror::LazyBudget budget;
        if (auto* heap = gpu_detail_heap::host_of(b.views.heap)) heap->warm_lazy(self, frame_number, budget);
    }
    x2w_vmemory::Recording* memory = nullptr;
    std::vector<x2w_vmemory::CopyBack> pending;
    bool needs_views = false;
    for (std::uint32_t i = 0; i < p.stage_count; ++i) {
        const auto& st = p.stages[i];
        if (!st.shader) return drop(drop_no_pipeline);
        if (shader_reads_memory(*st.shader)) {
            static const bool on = x2w_local::env_on("X2W_GPU_MEMORY_READS");
            if (!on || !compute || p.stage_count != 1) return drop(drop_memory_shader);
            if (!(memory = memory_ready(self, l, s, b, *st.shader, pending))) return false;
        }
        needs_views |= std::any_of(st.root_srvs.begin(), st.root_srvs.end(), [](const auto& r) { return r.table; }) ||
                       std::any_of(st.root_uavs.begin(), st.root_uavs.end(), [](const auto& r) { return r.table; });
    }
    // Borrow a view heap when the title has none bound.
    if (needs_views && !b.views.heap) return drop(drop_no_view_heap);
    std::vector<ID3D12Resource*> reads, writes;
    struct ReadScope {
        bool enabled;
        ReadScope(bool on, std::vector<ID3D12Resource*>& read, std::vector<ID3D12Resource*>& write) : enabled(on) {
            if (enabled) { memory_bind_reads = &read; memory_bind_writes = &write; }
        }
        ~ReadScope() { if (enabled) { memory_bind_reads = nullptr; memory_bind_writes = nullptr; } }
    } reading(memory != nullptr, reads, writes);
    // Heaps first: a root signature with directly indexed heaps must be set after them.
    ID3D12DescriptorHeap* heaps[2]{};
    UINT n = 0;
    if (b.views.heap) heaps[n++] = b.views.host ? b.views.host : gpu_heap_host(b.views.heap);
    if (b.samplers.heap) heaps[n++] = b.samplers.host ? b.samplers.host : gpu_heap_host(b.samplers.heap);
    if (!b.samplers.heap) InterlockedIncrement(&missing_sampler_heap);
    else {
        static LONG shown;
        if (first_calls(shown, 1)) log("x2w-gpu bind: title sampler heap bound before %s root signature", compute ? "compute" : "graphics");
    }
    if (!bind_cache() || s.host_heaps[0] != heaps[0] || s.host_heaps[1] != heaps[1]) {
        if (n) l->SetDescriptorHeaps(n, heaps);
        s.host_heaps[0] = heaps[0];
        s.host_heaps[1] = heaps[1];
        s.host_roots[0] = s.host_roots[1] = nullptr;
    }
    if (!bind_cache() || s.host_roots[compute] != p.root) {
        if (compute) l->SetComputeRootSignature(p.root);
        else l->SetGraphicsRootSignature(p.root);
        s.host_roots[compute] = p.root;
    }
    if (!bind_cache() || s.host_pso != p.pso) {
        l->SetPipelineState(p.pso);
        s.host_pso = p.pso;
    }
    for (std::uint32_t i = 0; i < p.stage_count; ++i) {
        const auto& st = p.stages[i];
        const x2w_gpu::GpuShader& sh = *st.shader;
        std::vector<x2w_vbind::MadeTable> made;
        if (!tables(self, s, b, sh, compute, made, bound_copy)) return drop(drop_no_slots);
        // Root views before the user data, so the bias each word carries is the one bound.
        std::vector<x2w_vbind::BoundBias> biases;
        struct Table { std::uint32_t param, index; };
        struct Direct { std::uint32_t param; D3D12_GPU_VIRTUAL_ADDRESS gpu; int kind; };  // 0 CBV, 1 SRV, 2 UAV
        std::vector<Table> view_tables;
        std::vector<Direct> directs;
        for (const auto& rs : st.root_cbvs) {
            const x2w_gpu::GpuRoot* r = s.roots.at(compute, rs.title);
            if (!r || !x2w_vbind::is_view(r->kind)) continue;  // "a shader reads a root view the title did not set"
            if (const auto gpu = root_cbv(self, r->address)) directs.push_back({rs.host, gpu, 0});
        }
        for (const auto& rs : st.root_srvs) {
            if (memory && rs.title == abi::slot_memory) {
                directs.push_back({rs.host, memory->directory->GetGPUVirtualAddress(), 1});
                continue;
            }
            if (rs.title == abi::slot_strides) {
                // Ours: the per-slot strides of the heap the shader picks buffers from by index.
                const std::uint64_t gpu = b.views.heap ? gpu_heap_strides_gpu(b.views.heap) : 0;
                if (gpu) directs.push_back({rs.host, gpu, 1});
                continue;
            }
            const x2w_gpu::GpuRoot* r = s.roots.at(compute, rs.title);
            const std::uint64_t address = r && x2w_vbind::is_view(r->kind) ? r->address : 0;
            const bool biased = biased_view(sh, rs.title);
            std::uint32_t bias = 0;
            if (rs.table) {
                std::uint32_t index = 0;
                if (!root_srv_table(self, s, b, address, biased, &index, &bias)) return drop(drop_root_srv);
                view_tables.push_back({rs.host, index});
            } else if (const auto gpu = root_direct(self, address, false, biased, &bias)) {
                directs.push_back({rs.host, gpu, 1});
            }
            biases.push_back({rs.title, bias});
        }
        for (const auto& rs : st.root_uavs) {
            if (memory && rs.title == abi::slot_faults) {
                directs.push_back({rs.host, memory->faults->GetGPUVirtualAddress(), 2});
                continue;
            }
            const x2w_gpu::GpuRoot* r = s.roots.at(compute, rs.title);
            const std::uint64_t address = r && x2w_vbind::is_view(r->kind) ? r->address : 0;
            std::uint32_t bias = 0;
            if (rs.table) {
                std::uint32_t index = 0;
                if (!root_uav_table(self, s, b, address, &index, &bias)) return drop(drop_root_uav);
                if (memory && (!address || !gpu_heap_kept_at(b.views.heap, index))) return refuse_memory(sh, MemoryRefusal::output);
                view_tables.push_back({rs.host, index});
            } else {
                const auto gpu = root_direct(self, address, true, true, &bias);
                if (!gpu) return drop(drop_root_uav);  // A root UAV requires a valid GPU address.
                directs.push_back({rs.host, gpu, 2});
            }
            biases.push_back({rs.title, bias});
        }
        if (st.user_data_param != UINT32_MAX && st.user_data_dwords) {
            unsigned misses = 0;
            const auto values = x2w_vbind::fill_user_data(s.roots, compute, sh.user_data.data, sh.user_data.count, st.user_data_dwords,
                made, b.spans, 2, biases, memory ? memory->base : 0, [&](std::uint32_t slot, std::uint32_t dword) -> std::uint64_t {
                    return buffer_address(b, s, compute, slot, dword);
                }, &misses);
            if (misses) InterlockedExchangeAdd(&user_misses, static_cast<LONG>(misses));
            if (!compute && values.size() >= 15 && sh.material_fields) {
                s.material_heap = b.views.heap;
                s.material_shader = &sh;
                s.material_raw_slot = (std::min)((values[8] + 2752u) >> 5, x2w_material_trace::null_index);
                s.material_base = values[14];
            }
            if (compute) l->SetComputeRoot32BitConstants(st.user_data_param, st.user_data_dwords, values.data(), 0);
            else l->SetGraphicsRoot32BitConstants(st.user_data_param, st.user_data_dwords, values.data(), 0);
        }
        for (const Direct& d : directs) {
            if (compute) {
                if (d.kind == 0) l->SetComputeRootConstantBufferView(d.param, d.gpu);
                else if (d.kind == 1) l->SetComputeRootShaderResourceView(d.param, d.gpu);
                else l->SetComputeRootUnorderedAccessView(d.param, d.gpu);
            } else {
                if (d.kind == 0) l->SetGraphicsRootConstantBufferView(d.param, d.gpu);
                else if (d.kind == 1) l->SetGraphicsRootShaderResourceView(d.param, d.gpu);
                else l->SetGraphicsRootUnorderedAccessView(d.param, d.gpu);
            }
        }
        for (const Table& t : view_tables) {
            const D3D12_GPU_DESCRIPTOR_HANDLE handle = gpu_heap_gpu_handle(b.views.heap, t.index);
            if (compute) l->SetComputeRootDescriptorTable(t.param, handle);
            else l->SetGraphicsRootDescriptorTable(t.param, handle);
        }
    }
    if (memory) {
        const auto state = [&](auto* r, auto next) { require_state(self, r, next); };
        const auto view = [&](UINT slot, auto* r, UINT64 size) { gpu_heap_raw_uav_at(memory->heap, slot, r, 0, size); };
        if (!memory->allow_ordinary_reads(l, reads, writes, x2w_gpu::buffer_read_state(l->GetType()), view, state))
            return refuse_memory(*p.stages[0].shader, MemoryRefusal::overlapping_read);
        if (!memory->ordinary_reads.empty()) {
            static LONG shared;
            const LONG n = InterlockedIncrement(&shared);
            if (n <= 4 || (n & (n - 1)) == 0)
                log("x2w-gpu memory simultaneous reads: frame=%ld count=%ld key=%s pages=%zu windows=%zu",
                    frame_number, n, p.stages[0].shader->key, memory->restore_rows.size(), memory->ordinary_reads.size());
        }
        memory->ready(l, state);
        // Root UAVs and raw directory entries can alias the same title pages across
        // builder dispatches. A transition alone doesn't order an already-UAV window.
        D3D12_RESOURCE_BARRIER order{};
        order.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        l->ResourceBarrier(1, &order);
        s.memory_back = std::move(pending); // published only by an issued command's completion hook
        static LONG dispatched_with_memory;
        const LONG n = InterlockedIncrement(&dispatched_with_memory);
        if ((n & (n - 1)) == 0)
            log("x2w-gpu memory binds %ld: key=%s pages=%zu windows=%zu", n, p.stages[0].shader->key,
                memory->addresses.size(), memory->windows.size());
    }
    return true;
}

// ---- draw state: vertex and index buffers, topology, targets --------------------------------------
bool prepare_draw(Wrapper* self, ID3D12GraphicsCommandList* l, GpuListState& s, const GpuPipeline& p, bool indexed)
{
    const Pass& pass = pass_of(self);
    D3D12_VERTEX_BUFFER_VIEW views[32]{};
    UINT count = 0;
    for (UINT slot = 0; slot < 32; ++slot) {
        if (!(p.vertex_slots & (1u << slot))) continue;
        count = slot + 1;
        x2w_vbind::VertexView v = s.roots.vertex[slot];
        if (!s.vertex_hooked && slot < 8) v = {pass.vb[slot].BufferLocation, pass.vb[slot].SizeInBytes, pass.vb[slot].StrideInBytes};
        if (!v.address || !v.size) continue;
        x2w_vbind::Found f;
        if (!find(v.address, 1, v.size, v.size, &f)) {
            InterlockedIncrement(&vertex_unbound);
            continue;
        }
        views[slot] = {window_of(f)->GetGPUVirtualAddress() + f.offset, static_cast<UINT>((std::min<std::uint64_t>)(v.size, f.left)), v.stride};
        window_read(self, window_of(f));
    }
    if (count) l->IASetVertexBuffers(0, count, views);
    if (indexed) {
        x2w_vbind::IndexView v = s.roots.index;
        if (!s.vertex_hooked) v = {pass.ib.BufferLocation, pass.ib.SizeInBytes, static_cast<std::uint32_t>(pass.ib.Format)};
        x2w_vbind::Found f;
        if (!v.address || !v.size || !find(v.address, 1, v.size, v.size, &f)) return drop(drop_index_buffer);
        const D3D12_INDEX_BUFFER_VIEW view{window_of(f)->GetGPUVirtualAddress() + f.offset,
                                           static_cast<UINT>((std::min<std::uint64_t>)(v.size, f.left)), static_cast<DXGI_FORMAT>(v.format)};
        window_read(self, window_of(f));
        l->IASetIndexBuffer(&view);
    }
    const UINT topology = s.roots.topology_set ? s.roots.topology : pass.topology;
    if (topology) l->IASetPrimitiveTopology(static_cast<D3D12_PRIMITIVE_TOPOLOGY>(x2w_vbind::host_topology(topology)));
    // The targets' states again: a dispatch or a read in between may have moved them.
    for (UINT i = 0; i < pass.target_count && i < 8; ++i)
        if (pass.targets[i]) use_as_target(self, pass.targets[i]);
    if (pass.dsv_resource) {
        constexpr auto shader_bits = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_STATES reads = D3D12_RESOURCE_STATE_COMMON;
        if (const auto* entry = pass.states.find(pass.dsv_resource)) reads = (entry->depth | entry->stencil) & shader_bits;
        require_bound_depth(self, reads);
    }
    return true;
}

// The bound targets, for a pipeline whose stream formats differ from them (WP2 makes a variant).
GpuTargets targets_of(Wrapper* self)
{
    const Pass& pass = pass_of(self);
    GpuTargets t;
    t.count = (std::min)(pass.target_count, 8u);
    for (std::uint32_t i = 0; i < t.count; ++i) t.rtv[i] = pass.target_formats[i];
    t.dsv = pass.dsv_format;
    t.read_depth_plane = pass.read_depth_plane;
    t.read_stencil_plane = pass.read_stencil_plane;
    ID3D12Resource* any = t.count && pass.targets[0] ? pass.targets[0] : pass.dsv_resource;
    t.samples = any ? any->GetDesc().SampleDesc.Count : 1;
    return t;
}

// The pipeline bound to the list for a call of this kind; null (counted) when the call must be dropped.
enum class Call { dispatch, draw, indexed, mesh, indirect };
bool sample_drop(Wrapper* list, Wrapper* pipeline, Call call, GpuPsoDrop pso, Drop why_dropped = last_drop)
{
    if (!gpu_drop_trace()) return false;
    static constexpr const char* calls[] = {"dispatch", "draw", "indexed", "mesh", "indirect"};
    static constexpr const char* bind_reasons[] = {"no-pipeline", "wrong-kind", "no-view-heap", "no-free-slots",
        "memory-shader", "root-uav", "root-srv", "index-buffer", "rectangle", "not-recording", "invalid-targets"};
    static constexpr const char* pso_reasons[] = {"no-pipeline", "async-pending", "missing-shader", "warm-bad",
        "worker-failure", "pipeline-refused"};
    static_assert(std::size(bind_reasons) == drop_count && std::size(pso_reasons) == unsigned(GpuPsoStatus::count));
    const unsigned reason = why_dropped == drop_no_pipeline ? drop_count + unsigned(pso.status) : unsigned(why_dropped);
    static LONG counts[std::size(calls)][drop_count + std::size(pso_reasons)]{};
    const LONG n = InterlockedIncrement(&counts[unsigned(call)][reason]);
    if (n > 4 && (n & (n - 1)) != 0) return false;
    const auto* entry = pipeline ? static_cast<const GpuPipelineEntry*>(pipeline->own().gpu[0]) : nullptr;
    std::string keys;
    if (entry) {
        for (const auto* shader : entry->records)
            if (shader) keys += std::string(" ") + x2w_gpu::gpu_stage_name(shader->stage) + "=" + shader->key;
        if (entry->missing_keys) keys += *entry->missing_keys;
    }
    const char* why = why_dropped == drop_no_pipeline ? pso_reasons[unsigned(pso.status)] : bind_reasons[why_dropped];
    log("x2w-gpu scene drop: frame=%ld call=%s reason=%s count=%ld list=%p pso=%p keys=%s detail=%s",
        frame_number, calls[unsigned(call)], why, n, static_cast<void*>(list), static_cast<void*>(pipeline),
        keys.empty() ? "<none>" : keys.c_str(), pso.detail);
    return false;
}

const GpuPipeline* pipeline_for(Wrapper* self, GpuListState& s, Call call, GpuPsoDrop* failure)
{
    if (!s.pipeline) return drop(drop_no_pipeline), nullptr;
    const GpuTargets targets = call == Call::dispatch ? GpuTargets{} : targets_of(self);
    const GpuPipeline* p = gpu_pipeline_ready(static_cast<Wrapper*>(s.pipeline), call == Call::dispatch ? nullptr : &targets, failure);
    if (!p || !p->pso || !p->root) return drop(drop_no_pipeline), nullptr;
    const bool ok = call == Call::dispatch ? p->compute : call == Call::mesh ? (!p->compute && p->mesh) : (!p->compute && !p->mesh);
    if (!ok) return drop(drop_wrong_kind), nullptr;  // "a draw or mesh dispatch with another kind of pipeline"
    if (call == Call::draw || call == Call::indexed) {
        const UINT topology = s.roots.topology_set ? s.roots.topology : pass_of(self).topology;
        if (topology == x2w_vbind::topology_rectangles) {
            static LONG with_gs, without_gs;
            if (first_calls(p->rects ? with_gs : without_gs, 1))
                log("x2w-gpu rectangle draw: %s, title PSO topology %u, indexed %u",
                    p->rects ? "geometry stage" : "no geometry stage", p->title_topology, call == Call::indexed ? 1u : 0u);
        }
        // Rectangle draws require a rectangle geometry stage.
        if (topology == x2w_vbind::topology_rectangles && !p->rects) return drop(drop_rectangles), nullptr;
    }
    return p;
}
// Everything a call needs set on the host list, without issuing it.
bool prepare(Wrapper* self, ID3D12GraphicsCommandList* l, Call call)
{
    if (!l) {
        drop(drop_not_recording);
        return sample_drop(self, nullptr, call, {});
    }
    GpuListState& s = state_of(self);
    if (call != Call::dispatch && pass_of(self).targets_refused) {
        drop(drop_invalid_targets);
        return sample_drop(self, static_cast<Wrapper*>(s.pipeline), call, {});
    }
    s.writable_depth.clear();
    s.light_role = x2w_gpu::ambient::Role::none;
    if (call == Call::dispatch && s.pipeline && indirect_light_settings().level > 0) {
        const auto* entry = static_cast<const GpuPipelineEntry*>(static_cast<Wrapper*>(s.pipeline)->own().gpu[0]);
        if (entry) {
            if (entry->records.size() == 1 && entry->records.front() && entry->records.front()->builder) return false;
            s.light_role = entry->light_role;
            s.light_views = {};
            s.light_depth_width = s.light_depth_height = 0;
        }
    }
    GpuPsoDrop failure;
    const GpuPipeline* p = pipeline_for(self, s, call, gpu_drop_trace() ? &failure : nullptr);
    s.memory_back.clear();
    if (!p) return sample_drop(self, static_cast<Wrapper*>(s.pipeline), call, failure);
    // Bindless textures WP5 queued (made into mirror slots, or changed): uploaded on this open list first (PLAN 4b).
    gpu_heap_upload_due(self, upload_due_most());
    const bool compute = call == Call::dispatch;
    if (compute_capture_on()) {
        s.trace_request = 0;
        s.trace_shader = compute && p->stage_count ? p->stages[0].shader : nullptr;
        if (s.trace_shader) s.trace_request = match_compute_capture(s.trace_shader->key);
        if (s.trace_request) s.trace_bindings = {};
    }
    s.pixel_capture_shader = nullptr;
    if (!compute && pixel_capture_on() && compute_capture_pending.load()) {
        s.pixel_bindings.clear();
        s.pixel_capture_pipeline = p;
        for (UINT i = 0; i < p->stage_count; ++i) {
            const auto* sh = p->stages[i].shader;
            if (sh && sh->stage == x2w_gpu::GpuStage::pixel) {
                s.trace_request = match_compute_capture(sh->key);
                if (s.trace_request) s.pixel_capture_shader = sh;
            }
        }
    }
    s.material_heap = nullptr;
    if (!bind(self, l, s, *p, compute) || (!compute && !prepare_draw(self, l, s, *p, call == Call::indexed)))
        return sample_drop(self, static_cast<Wrapper*>(s.pipeline), call, failure);
    if (s.material_heap && l->GetType() == D3D12_COMMAND_LIST_TYPE_DIRECT) {
        prepare_material(self, l, s);
        if (gpu_trace_request_id && gpu_trace_stage_key() && trace_tables(self, false, s.material_shader->key))
            material_capture(self, l, s, p->pso);
    }
    return true;
}

void report()
{
    LONG d[drop_count];
    LONG total = 0;
    for (int i = 0; i < drop_count; ++i) total += d[i] = InterlockedExchange(&dropped[i], 0);
    log("x2w-gpu bind 300 frames: dispatched %ld drawn %ld dropped %ld (no pipeline %ld, wrong kind %ld, no view heap %ld, "
        "no free slots %ld, memory shader %ld, root uav %ld, root srv %ld, index buffer %ld, rectangles %ld, not recording %ld, refused targets %ld); "
        "user-data misses %ld, cbv copies %ld (failed %ld, null %ld), root uav null %ld unknown memory %ld cut %ld, root srv null %ld "
        "cut %ld, vertex views unbound %ld, textures uploaded %ld, tables outside the heaps %ld, fresh tables skipped %ld, "
        "table views missing %ld, buffer barriers %ld, slots released %ld kept %ld, binds without sampler heap %ld",
        InterlockedExchange(&dispatched, 0), InterlockedExchange(&drawn, 0), total, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
        d[8], d[9], d[drop_invalid_targets], InterlockedExchange(&user_misses, 0), InterlockedExchange(&cbv_copies, 0), InterlockedExchange(&cbv_copy_failed, 0),
        InterlockedExchange(&cbv_null, 0), InterlockedExchange(&root_uav_null, 0), InterlockedExchange(&root_uav_unknown, 0),
        InterlockedExchange(&root_uav_cut, 0), InterlockedExchange(&root_srv_null, 0), InterlockedExchange(&root_srv_cut, 0),
        InterlockedExchange(&vertex_unbound, 0), InterlockedExchange(&textures_uploaded, 0), InterlockedExchange(&foreign_tables, 0),
        InterlockedExchange(&tables_skipped, 0), InterlockedExchange(&table_views_missing, 0), InterlockedExchange(&buffer_barriers, 0),
        InterlockedExchange(&slots_released, 0), InterlockedExchange(&slots_kept, 0), InterlockedExchange(&missing_sampler_heap, 0));
}

}  // namespace vbind

// ===== the declared hooks (gpu_hooks.h) =================================================================================

// ---- list setters 32-46 (EXCLUSIVE under ABI: X2W's handlers are not called, Pass root state is stale) ----------------------
void gpu_set_heaps(Wrapper* list, UINT count, void* const* heaps)
{
    auto& s = vbind::state_of(list);
    s.heaps[0] = s.heaps[1] = nullptr;
    if (!count || count > 4 || !x2w_local::readable(heaps, std::size_t{count} * sizeof(*heaps))) return;
    for (UINT i = 0; i < count; ++i) {
        GpuHeapInfo info{};
        if (heaps[i] && gpu_heap_for(as_wrapper(heaps[i]), &info) && info.heap) s.heaps[info.sampler ? 1 : 0] = as_wrapper(heaps[i]);
    }
    static LONG selected;
    if (s.heaps[0] && s.heaps[1] && first_calls(selected, 1))
        log("x2w-gpu list slot 32 selected title view and sampler heaps");
}
void gpu_roots_signature(Wrapper*, bool, Wrapper*) {}  // The host root signature belongs to the pipeline.
void gpu_roots_table(Wrapper* list, bool compute, UINT parameter, std::uint64_t handle)
{
    vbind::state_of(list).roots.set_table(compute, parameter, handle);
}
void gpu_roots_constants(Wrapper* list, bool compute, UINT parameter, UINT count, const std::uint32_t* data, UINT offset)
{
    if (!count || !x2w_local::readable(data, std::size_t{count} * 4)) return;
    vbind::state_of(list).roots.set_constants(compute, parameter, data, count, offset);
}
void gpu_roots_view(Wrapper* list, bool compute, UINT parameter, std::uint64_t address, GpuRootKind kind)
{
    vbind::state_of(list).roots.set_view(compute, parameter, address, kind);
}

// ---- non-root draw state (ADDITIVE, after X2W's handler) ----------------------------------------------------------------------
// X2W forwards topologies 1-5 only; the console's own become triangle lists.
void gpu_topology(Wrapper* list, UINT topology)
{
    auto& s = vbind::state_of(list);
    s.roots.topology = topology;
    s.roots.topology_set = true;
    if (auto* l = recording(list)) l->IASetPrimitiveTopology(static_cast<D3D12_PRIMITIVE_TOPOLOGY>(x2w_vbind::host_topology(topology)));
}
// Viewports, scissors and the blend factor: X2W's handlers forward them to the desktop list already.
void gpu_viewports(Wrapper*, UINT, const D3D12_VIEWPORT*) {}
void gpu_scissors(Wrapper*, UINT, const D3D12_RECT*) {}
void gpu_blend_factor(Wrapper*, const float*) {}
// Slot 24 is installed under ABI, also where X2W's own handler is not (a second forward of the same value is harmless).
void gpu_stencil_ref(Wrapper* list, UINT ref)
{
    if (auto* l = recording(list)) l->OMSetStencilRef(ref);
}
void gpu_set_pipeline(Wrapper* list, Wrapper* pipeline) { vbind::state_of(list).pipeline = pipeline; }
// Barriers: X2W forwards UAV barriers on desktop resources. A barrier that names a title BUFFER (or no resource)
// also orders the windows this path binds in its place: one global UAV barrier.
void gpu_barriers(Wrapper* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers)
{
    auto* l = recording(list);
    if (!l || !count || !barriers || !x2w_local::readable(barriers, std::size_t{count} * sizeof(*barriers))) return;
    const auto type = l->GetType();
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) return;
    bool buffer = false;
    for (UINT i = 0; i < count && !buffer; ++i) {
        D3D12_RESOURCE_BARRIER b;
        std::memcpy(&b, barriers + i, sizeof(b));
        if (b.Type == D3D12_RESOURCE_BARRIER_TYPE_ALIASING) continue;
        void* named = b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV ? static_cast<void*>(b.UAV.pResource) : static_cast<void*>(b.Transition.pResource);
        if (!named) {
            buffer = b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV;
            continue;
        }
        Wrapper* w = as_wrapper(named);
        const ResourceInfo* info = w && w->own().kind == resource_kind ? resource_info(w) : nullptr;
        buffer = info && info->console.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
    }
    if (!buffer) return;
    D3D12_RESOURCE_BARRIER all{};
    all.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    all.UAV.pResource = nullptr;
    l->ResourceBarrier(1, &all);
    InterlockedIncrement(&vbind::buffer_barriers);
}
void gpu_ia_index(Wrapper* list, const D3D12_INDEX_BUFFER_VIEW* view)
{
    auto& s = vbind::state_of(list);
    s.vertex_hooked = true;
    if (view && x2w_local::readable(view, sizeof(*view))) {
        const x2w_vbind::IndexView v{view->BufferLocation, view->SizeInBytes, static_cast<std::uint32_t>(view->Format)};
        s.roots.set_index(&v);
    } else {
        s.roots.set_index(nullptr);
    }
}
void gpu_ia_vertex(Wrapper* list, UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views)
{
    auto& s = vbind::state_of(list);
    s.vertex_hooked = true;
    if (views && !x2w_local::readable(views, std::size_t{count} * sizeof(*views))) return;
    for (UINT i = 0; i < count && std::size_t{start} + i < 32; ++i) {
        const x2w_vbind::VertexView v = views ? x2w_vbind::VertexView{views[i].BufferLocation, views[i].SizeInBytes, views[i].StrideInBytes}
                                              : x2w_vbind::VertexView{};
        s.roots.set_vertex(start + i, 1, &v);
    }
}
// Targets: X2W's slot 50 maps the RTV / DSV handles and sets them; the draw reads Pass (formats, states).
void gpu_targets(Wrapper*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL, const D3D12_CPU_DESCRIPTOR_HANDLE*) {}
// Depth bounds: X2W has no handler. Forwarded where the device supports the test.
void gpu_depth_bounds(Wrapper* list, float min, float max)
{
    static const bool supported = [] {
        D3D12_FEATURE_DATA_D3D12_OPTIONS2 o2{};
        return the_device && SUCCEEDED(device_of(the_device)->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS2, &o2, sizeof(o2))) &&
               o2.DepthBoundsTestSupported;
    }();
    auto* l = recording(list);
    if (!l || !supported) return;
    x2w_startup::ComPtr<ID3D12GraphicsCommandList1> l1;
    if (SUCCEEDED(l->QueryInterface(IID_PPV_ARGS(&l1)))) l1->OMSetDepthBounds(min, max);
}

// ---- draws (EXCLUSIVE): true when issued, false when dropped (counted here by reason, by WP0 in total) ---------------------
void gpu_trace_pass_inventory(const GpuPipeline& p, const Pass& pass, const char* command, bool compute,
                              LONG frame, LONG sequence, UINT x, UINT y, UINT z)
{
    struct Entry { ID3D12PipelineState* pso; ID3D12Resource* target; };
    struct Inventory {
        LONG frame = 0;
        std::vector<Entry> entries;
        bool capped = false;
    };
    static std::mutex lock;
    static Inventory seen[2];
    auto* target = !compute && pass.target_count ? pass.targets[0] : nullptr;
    std::lock_guard<std::mutex> guard(lock);
    auto& current = seen[compute ? 1 : 0];
    if (frame < current.frame) return;
    if (frame != current.frame) {
        current.entries.clear();
        current.capped = false;
        current.frame = frame;
    }
    if (current.capped || std::any_of(current.entries.begin(), current.entries.end(),
            [&](const Entry& e) { return e.pso == p.pso && e.target == target; })) return;
    if (current.entries.size() == 128) {
        log("x2w-frame pass inventory %ld %s capped at 128 distinct entries", frame, compute ? "compute" : "graphics");
        current.capped = true;
        return;
    }
    current.entries.push_back({p.pso, target});
    const auto key = [&](UINT i) { return i < p.stage_count && p.stages[i].shader ? p.stages[i].shader->key : "-"; };
    const auto desc = target ? target->GetDesc() : D3D12_RESOURCE_DESC{};
    log("x2w-frame pass inventory %ld #%ld %s stages=%s,%s,%s,%s args=%u,%u,%u RTs=%u RTV0=%p size=%llux%u",
        frame, sequence, command, key(0), key(1), key(2), key(3), x, y, z, compute ? 0u : pass.target_count,
        static_cast<void*>(target), desc.Width, desc.Height);
}

void gpu_trace_issue(Wrapper* self, const char* command, bool compute, UINT a, UINT b, UINT c)
{
    if (compute) {
        static const bool trace_busy = [] {
            char value[16]{};
            return GetEnvironmentVariableA("X2W_D3D12_TRACE_BUSY", value, sizeof(value)) && std::atol(value) > 0;
        }();
        if (trace_busy) InterlockedIncrement(&frame_dispatches);
    }
    auto& scene = gpu_scene_trace();
    if (!compute && scene.waiting.load(std::memory_order_acquire)) {
        const Pass& pass = pass_of(self);
        const auto width = vbind::trace_min_width() ? vbind::trace_min_width() : 2560;
        const auto needed = scene.mrt ? scene.mrt : 5;
        const auto count = (std::min)(pass.target_count, 8u);
        if (count >= static_cast<UINT>(needed) && std::count_if(pass.targets, pass.targets + count,
                [width](ID3D12Resource* target) { return target && target->GetDesc().Width >= width; }) >= needed) {
            std::lock_guard<std::mutex> held(scene.lock);
            const LONG next = InterlockedCompareExchange(&frame_number, 0, 0) + 1;
            if (scene.waiting.load() && !gpu_trace_frame) {
                if (!scene.first) scene.first = next;
                if (scene.pending || next - scene.first >= scene.after) {
                    if ((vbind::compute_capture_on() || vbind::pixel_capture_on()) && scene.pending)
                        vbind::arm_compute_capture(scene.pending, scene.pending_key, scene.second_key, scene.pending_point);
                    std::atomic_store(&gpu_scene_stage_override, std::move(scene.pending_key));
                    std::atomic_store(&gpu_scene_second_override, std::move(scene.second_key));
                    scene.active_id = scene.pending;
                    InterlockedExchange(&gpu_trace_request_id, scene.active_id);
                    scene.pending = 0;
                    scene.active_frame = next;
                    scene.waiting.store(false);
                    reset_gpu_trace_commands();
                    InterlockedExchange(&trace_uav_busy_frame, next);
                    InterlockedExchange(&trace_frame_active, 1);
                    InterlockedExchange(&gpu_trace_frame, next);
                    log("x2w-frame ---- begin deferred frame %ld request %ld first qualifying frame %ld targets %u",
                        next, scene.active_id, scene.first, count);
                }
            }
        }
    }
    const LONG frame = gpu_trace_frame;
    if (!frame) return;
    const LONG n = InterlockedIncrement(&gpu_trace_commands);
    if (!vbind::trace_pass(self, compute)) return;
    UINT topology = 0;
    const GpuPipeline* p = nullptr;
    static const bool inventory = x2w_local::env_on("X2W_TRACE_PASS_INVENTORY");
    if (inventory) {
        p = gpu_bound_pipeline(self, &topology);
        if (p) gpu_trace_pass_inventory(*p, pass_of(self), command, compute, frame, n, a, b, c);
    }
    if (gpu_trace_stage_key()) {
        if (!p) p = gpu_bound_pipeline(self, &topology);
        if (!p || std::none_of(p->stages, p->stages + p->stage_count, [](const auto& stage) {
                return stage.shader && vbind::trace_stage_matches(stage.shader->key);
            })) return;
    }
    const LONG kind_count = InterlockedIncrement(&gpu_trace_per_kind[compute ? 1 : 0]);
    if (kind_count > 64) {
        if (kind_count == 65) log("x2w-frame gpu frame %ld: %s command trace capped at 64", frame, compute ? "compute" : "graphics");
        return;
    }
    if (!p) p = gpu_bound_pipeline(self, &topology);
    if (!p) return;
    const auto key = [&](UINT i) {
        return i < p->stage_count && p->stages[i].shader ? p->stages[i].shader->key : "-";
    };
    const Pass& pass = pass_of(self);
    log("x2w-frame gpu %ld #%ld %s list=%p pso=%p args=%u,%u,%u stage0=%s stage1=%s topology=%u "
        "RTs=%u depth=%p viewport_set=%u xywh=%.1f,%.1f,%.1f,%.1f scissor_set=%u ltrb=%ld,%ld,%ld,%ld",
        frame, n, command, static_cast<void*>(self), static_cast<void*>(p->pso), a, b, c, key(0), key(1), topology,
        compute ? 0u : pass.target_count, compute ? nullptr : static_cast<void*>(pass.dsv_resource),
        UINT(pass.viewport_set), pass.viewport0.TopLeftX, pass.viewport0.TopLeftY, pass.viewport0.Width, pass.viewport0.Height,
        UINT(pass.scissor_set), pass.scissor0.left, pass.scissor0.top, pass.scissor0.right, pass.scissor0.bottom);
    static const bool readback = x2w_local::env_on("X2W_D3D12_TRACE_TARGETS");
    for (UINT i = 0; !compute && i < pass.target_count && i < 8; ++i) {
        ID3D12Resource* rt = pass.targets[i];
        if (!rt) continue;
        const auto desc = rt->GetDesc();
        const auto* tracked = pass.states.find(rt);
        log("x2w-frame gpu %ld #%ld RTV[%u]=%p size=%llux%u resource_fmt=%u view_fmt=%u tracked_state=0x%x",
            frame, n, i, static_cast<void*>(rt), desc.Width, desc.Height, UINT(desc.Format), UINT(pass.target_formats[i]),
            tracked ? UINT(tracked->depth) : UINT_MAX);
        if (readback && desc.Width >= vbind::trace_min_width() && tracked && tracked->depth == D3D12_RESOURCE_STATE_RENDER_TARGET)
            note_traced_target(self, rt, true, D3D12_RESOURCE_STATE_RENDER_TARGET, true, pass.target_formats[i], n, i);
    }
}

// Copies run on the list that used the UAV, including compute lists. Each use starts from the
// original texture and publishes its writes before any following title command can read it.
void gpu_finish_depth_writes(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    auto& s = vbind::state_of(self);
    if (!s.memory_back.empty()) {
        for (const auto& range : s.memory_back)
            vbind::builder_arenas().publish(list,range,[&](auto* r, auto next) { require_state(self,r,next); },
                                             s.memories.front()->zeros.Get());
        s.memory_back.clear();
        s.host_roots[1] = nullptr;
        s.host_pso = nullptr;
    }
    for (auto* pair : s.writable_depth) {
        const auto before = pass_of(self).states.state(pair->depth.Get());
        pair->copy_colour(list, false, [&](auto* r, auto state) { require_state(self, r, state); });
        require_state(self, pair->depth.Get(), before);
        require_state(self, pair->colour.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    s.writable_depth.clear();
}
bool gpu_depth_barrier(Wrapper* self, Wrapper* resource)
{
    std::shared_ptr<x2w_depth_twins::Twins> pair;
    {
        std::lock_guard<std::mutex> guard(gpu_detail_heap::depth_lock);
        const auto& pairs = gpu_detail_heap::depth_pairs();
        const auto it = pairs.find(resource);
        if (it == pairs.end() || !it->second) return false;
        pair = it->second;
    }
    auto* list = recording(self);
    if (!list) return false;
    auto& s = vbind::state_of(self);
    if (std::find(s.depth_refs.begin(), s.depth_refs.end(), pair) == s.depth_refs.end()) s.depth_refs.push_back(pair);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = pair->colour.Get();
    list->ResourceBarrier(1, &barrier);
    return true;
}
void gpu_note_exposure_reset(Wrapper* self)
{
    static std::atomic<bool> seen{false};
    if (!gpu_drop_trace() || seen.load(std::memory_order_relaxed)) return;
    auto* p = static_cast<Wrapper*>(vbind::state_of(self).pipeline);
    auto* entry = p ? static_cast<GpuPipelineEntry*>(p->own().gpu[0]) : nullptr;
    if (entry && entry->exposure_reset && !seen.exchange(true))
        log("x2w-gpu exposure reset draw recorded: frame=%ld list=%p", frame_number, static_cast<void*>(self));
}

bool gpu_draw(Wrapper* self, ID3D12GraphicsCommandList* list, UINT vertices, UINT instances, UINT first_vertex, UINT first_instance)
{
    if (!vbind::prepare(self, list, vbind::Call::draw)) return false;
    if (vbind::pixel_capture_on()) vbind::pixel_capture_before(self, list, "draw");
    list->DrawInstanced(vertices, instances, first_vertex, first_instance);
    if (vbind::pixel_capture_on()) vbind::pixel_capture_after(self, list);
    if (vertices && instances) gpu_note_exposure_reset(self);
    gpu_trace_issue(self, "draw", false, vertices, instances, first_vertex);
    gpu_finish_depth_writes(self, list);
    InterlockedIncrement(&vbind::drawn);
    return true;
}
bool gpu_draw_indexed(Wrapper* self, ID3D12GraphicsCommandList* list, UINT count, UINT instances, UINT start, INT base_vertex,
                          UINT first_instance)
{
    if (!vbind::prepare(self, list, vbind::Call::indexed)) return false;
    if (vbind::pixel_capture_on()) vbind::pixel_capture_before(self, list, "indexed");
    list->DrawIndexedInstanced(count, instances, start, base_vertex, first_instance);
    if (count && instances) gpu_note_exposure_reset(self);
    if (vbind::pixel_capture_on()) vbind::pixel_capture_after(self, list);
    gpu_trace_issue(self, "indexed", false, count, instances, start);
    gpu_finish_depth_writes(self, list);
    InterlockedIncrement(&vbind::drawn);
    return true;
}
bool gpu_dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z)
{
    if (!vbind::prepare(self, list, vbind::Call::dispatch)) return false;
    vbind::compute_capture_before(self, list);
    list->Dispatch(x, y, z);
    if (x && y && z) vbind::finish_light(self, list);
    vbind::compute_capture_after(self, list);
    gpu_trace_issue(self, "dispatch", true, x, y, z);
    gpu_finish_depth_writes(self, list);
    InterlockedIncrement(&vbind::dispatched);
    return true;
}
// The bind sequence without the issue, for WP4 (ExecuteIndirect, DispatchMesh, DispatchX).
bool gpu_bind_draw(Wrapper* self, ID3D12GraphicsCommandList* list, bool indexed)
{
    return vbind::prepare(self, list, indexed ? vbind::Call::indexed : vbind::Call::draw);
}
bool gpu_bind_mesh(Wrapper* self, ID3D12GraphicsCommandList* list) { return vbind::prepare(self, list, vbind::Call::mesh); }
bool gpu_bind_compute(Wrapper* self, ID3D12GraphicsCommandList* list) { return vbind::prepare(self, list, vbind::Call::dispatch); }
// WP4's clamp set its own compute root signature and the list's PSO (vindirect 101900): both are set again at the next
// bind; graphics roots and heaps are untouched. Root arguments are set at every bind anyway.
void gpu_bind_forget_compute(Wrapper* self)
{
    if (auto* s = vbind::state_if(self)) {
        s->host_roots[1] = nullptr;
        s->host_pso = nullptr;
    }
}
// The made pipeline bound on the list (with the bound targets for a graphics one), or null; the console topology last
// set at slot 20 (WP4: kind check, indirect rectangles, DispatchX; vindirect 101523).
const GpuPipeline* gpu_bound_pipeline(Wrapper* self, UINT* title_topology, GpuPsoDrop* drop)
{
    auto* s = vbind::state_if(self);
    if (title_topology) *title_topology = s && s->roots.topology_set ? s->roots.topology : pass_of(self).topology;
    if (!s || !s->pipeline) return nullptr;
    // One call: WP2 ignores the targets for a compute pipeline (no graphics state), so no stream-format variant is made.
    const GpuTargets targets = vbind::targets_of(self);
    return gpu_pipeline_ready(static_cast<Wrapper*>(s->pipeline), &targets, drop);
}

// ---- list lifecycle ---------------------------------------------------------------------------------------------------------
void gpu_bind_list_reset(Wrapper* list)
{
    vbind::retire_passed();
    auto& s = vbind::state_of(list);
    vbind::end_recording(s);
    s.roots.clear();
    s.vertex_hooked = false;
    s.pipeline = nullptr;
    s.heaps[0] = s.heaps[1] = nullptr;
}
void gpu_bind_list_close(Wrapper* self)
{
    auto* s = vbind::state_if(self);
    auto* l = recording(self);
    if (s && l)
        for (const auto& m : s->memories)
            m->close(l, [&](auto* r, auto state) { require_state(self, r, state); });
}
// After ExecuteCommandLists (draw_lock and allocator_lock held): one fence value for the translated recordings that
// hold slots; the 300-frame report line.
void gpu_bind_first_submit(UINT count, Wrapper* const* lists)
{
    if (!vbind::defer_table_cbv()) return;
    for (UINT i = 0; i < count; ++i) {
        auto* s = lists[i] ? vbind::state_if(lists[i]) : nullptr;
        if (!s) continue;
        const auto result = s->table_constants.first_submit([](const x2w_gpu::TableConstantSlice& slice,
                                                                const unsigned char* bytes, size_t size) {
            auto& r = upload_ring_state();
            AcquireSRWLockShared(&r.lock);
            const auto* c = slice.chunk < r.chunks.size() ? &r.chunks[slice.chunk] : nullptr;
            const bool valid = !r.legacy && c && c->mapped && c->generation == slice.generation &&
                c->state == RingChunk::in_use && slice.offset >= c->base && slice.offset - c->base <= c->used &&
                size <= c->used - (slice.offset - c->base);
            if (valid) std::memcpy(c->mapped + (slice.offset - c->base), bytes, size);
            ReleaseSRWLockShared(&r.lock);
            return valid;
        });
        if (result.checked) {
            // Submission holds the draw and allocator locks. Aggregate once per list.
            static UINT64 checked, changed, unreadable, stale, next = 1;
            const bool first_delta = !changed && result.changed;
            checked += result.checked;
            changed += result.changed;
            unreadable += result.unreadable;
            stale += result.stale;
            if (first_delta || checked >= next) {
                while (next <= checked) next *= 2;
                log("x2w-gpu table constants at first submit: frame=%ld checked=%llu changed=%llu unreadable=%llu stale=%llu",
                    frame_number, checked, changed, unreadable, stale);
            }
        }
    }
}

void gpu_bind_submitted(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists)
{
    std::vector<GpuListState*> ours;
    for (UINT i = 0; i < count; ++i)
        if (auto* s = lists[i] ? vbind::state_if(lists[i]) : nullptr; s && (!s->dynamic.empty() || !s->depth_refs.empty() || !s->light_copies.empty() || !s->trace_captures.empty())) ours.push_back(s);
    if (!ours.empty()) {
        AcquireSRWLockExclusive(&vbind::fence_lock);
        auto& [fence, value] = vbind::queue_fences()[queue];
        if (!fence) {
            ID3D12Device* device = nullptr;
            if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device)))) {
                if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) fence = nullptr;
                device->Release();
            }
        }
        const bool signalled = fence && SUCCEEDED(queue->Signal(fence, value + 1));
        if (signalled) ++value;
        const auto mark = std::make_pair(fence, value);
        ReleaseSRWLockExclusive(&vbind::fence_lock);
        for (auto* s : ours) {
            for (const auto& cap : s->trace_captures)
                vbind::compute_capture_submitted(cap, signalled ? fence : nullptr, value);
            if (signalled) {
                s->submitted.push_back(mark);
            } else {
                // Nothing says when the GPU is done with them: the slots are never handed out again (counted).
                for (const auto& d : s->dynamic) InterlockedExchangeAdd(&vbind::slots_kept, static_cast<LONG>(d.n));
                s->dynamic.clear();
                AcquireSRWLockExclusive(&vbind::retire_lock);
                static std::vector<std::shared_ptr<x2w_depth_twins::Twins>> unfenced;
                unfenced.insert(unfenced.end(), s->depth_refs.begin(), s->depth_refs.end());
                s->depth_refs.clear();
                static std::vector<std::shared_ptr<x2w_vmemory::Recording>> unfenced_memory;
                unfenced_memory.insert(unfenced_memory.end(), s->memories.begin(), s->memories.end());
                s->memories.clear();
                static std::vector<std::shared_ptr<x2w_packed_light::Copy>> unfenced_light;
                unfenced_light.insert(unfenced_light.end(), s->light_copies.begin(), s->light_copies.end());
                ReleaseSRWLockExclusive(&vbind::retire_lock);
            }
        }
    }
    vbind::retire_passed();
    static LONG reported = -1;
    const LONG frame = frame_number / 300;
    if (InterlockedExchange(&reported, frame) != frame && frame > 0) vbind::report();
}
void gpu_list_released(Wrapper* list)
{
    if (auto* s = vbind::state_if(list)) {
        vbind::end_recording(*s);
        delete s;
        list->own().gpu[0] = nullptr;
    }
}
