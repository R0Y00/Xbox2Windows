// Indirect draws, mesh dispatch and DispatchX. Device slot 41 records signatures. ExecuteIndirect accepts a single
// draw, indexed draw, dispatch, mesh dispatch or console single-pass dispatch (0x69). Root-changing signatures and
// calls incompatible with the bound pipeline are dropped. Indirect rectangles can use their first triangle when
// there are no per-vertex inputs.
//
// The GPU clamps title records and counts into the indirect ring. Each list has a budget of 2^27 in a 32-byte pool
// slot, cleared at its first clamp after creation or reset. The effective count follows the records. The binder
// restores the compute root and PSO after the clamp, then issues the host command signature. DispatchMesh uses
// min(x,65535) groups in one dimension. For DispatchX, flag 0x20 supplies thread counts and USER_DISPATCH_THREADS_X
// preserves the original x count.
//
// X2W_GPU_INDIRECT=0 drops indirect calls. X2W_GPU_MESH=0 drops mesh dispatches.
// X2W_GPU_INDIRECT_WIDE=0 limits AMD dispatches to 65,535 groups in x. X2W_GPU_INDIRECT_REST=0 drops a
// record that exceeds the remaining budget. X2W_GPU_THREAD_DIMENSIONS=0 treats counts as groups.
// X2W_GPU_INDIRECT_NOTES=1 enables CPU-visible budget diagnostics.

namespace vi = x2w_gpu_indirect;

// WP3's gpu_bound_pipeline (the made pipeline bound on the list) is declared in gpu_hooks.h, defined in
// gpu_bind.inl.

// ---- counters, logged every 300 frames -------------------------------------------------------------------------
enum GpuIndirectKindIndex { vik_draw, vik_indexed, vik_dispatch, vik_mesh, vik_other, vik_kinds };
constexpr const char* gpu_indirect_kind_names[vik_kinds] = {"draw", "indexed", "dispatch", "mesh", "other"};
int gpu_indirect_kind_index(UINT kind)
{
    return kind == vi::kind_draw ? vik_draw : kind == vi::kind_indexed ? vik_indexed : kind == vi::kind_dispatch ? vik_dispatch
           : kind == vi::kind_mesh ? vik_mesh : vik_other;
}
enum GpuIndirectDrop {
    vid_not_recording,
    vid_signature,  // unknown signature, not exactly one argument of a kind the host takes, or no host signature
    vid_mesh_off,   // X2W_GPU_MESH=0
    vid_pipeline,   // No translated pipeline bound, one of another kind, or rectangles it cannot draw
    vid_buffer,     // records or count not found in title memory, or a layout the clamp refuses
    vid_scratch,    // no ring chunk for the clamped records
    vid_clamp,      // the clamp or the budget pool could not be made
    vid_bind,       // WP3 refused to bind the draw state
    vid_reasons
};
constexpr const char* gpu_indirect_drop_names[vid_reasons] = {"not-recording", "signature", "mesh-off", "pipeline",
                                                                  "buffer", "scratch", "clamp", "bind"};
struct GpuIndirectCounts {
    LONG calls[vik_kinds];
    LONG executed[vik_kinds];
    LONG dropped[vid_reasons][vik_kinds];
    LONG console_single;  // calls of signatures of type 0x69
    LONG counted;         // calls with a count buffer
    LONG rect_triangles;  // indirect rectangles drawn as their first triangle
    LONG mesh, mesh_dropped, mesh_cut, mesh_yz;  // direct DispatchMesh: calls, dropped, x > 65,535, y * z != 1
    LONG dispatch_x, dispatch_x_threads, dispatch_x_dropped;
};
GpuIndirectCounts gpu_indirect_counts;
bool gpu_indirect_drop(int kind, GpuIndirectDrop why)
{
    InterlockedIncrement(&gpu_indirect_counts.dropped[why][kind]);
    return false;  // dropped: WP0 counts it and the old path does not run
}
bool gpu_indirect_env_zero(const char* name) { return x2w_gpu::gpu_env_is(name, '0'); }

// ---- the clamp, the host signatures, the budget pool -------------------------------------------------------------
vi::Clamp gpu_clamp;
vi::HostSignatures gpu_host_signatures;
// The budget pool: a list takes the next slot at its first clamp since its creation or reset. A slot is used again only
// after gpu_budget_slots more recordings, far longer than any list stays in flight.
constexpr std::uint32_t gpu_budget_slots = 1u << 16;
ID3D12Resource* gpu_budget_pool;
ID3D12Resource* gpu_budget_zeros;                // upload heap, 256 zero bytes
const volatile std::uint32_t* gpu_budget_words;  // the mapped pool (X2W_GPU_INDIRECT_NOTES=1), or null
volatile LONG gpu_budget_next;

bool gpu_indirect_ready()
{
    static SRWLOCK lock = SRWLOCK_INIT;
    static bool failed;
    if (gpu_budget_pool) return true;
    AcquireSRWLockExclusive(&lock);
    if (!gpu_budget_pool && !failed) {
        ID3D12Device* device = device_of(the_device);
        const bool amd = vi::is_amd(device);
        const bool wide = !gpu_indirect_env_zero("X2W_GPU_INDIRECT_WIDE");
        const bool rest = !gpu_indirect_env_zero("X2W_GPU_INDIRECT_REST");
        const bool notes = x2w_gpu::gpu_env_is("X2W_GPU_INDIRECT_NOTES", '1');
        std::string errors;
        vi::Clamp clamp;
        HRESULT hr = vi::make_clamp(device, vi::clamp_text(vi::widest(amd, wide), rest), &clamp, &errors);
        if (FAILED(hr)) log("x2w-gpu indirect: argument clamp refused 0x%08lx: %.400s", static_cast<unsigned long>(hr), errors.c_str());
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = u64{gpu_budget_slots} * vi::budget_slot_bytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource* pool = nullptr;
        void* mapped = nullptr;
        if (SUCCEEDED(hr) && notes) {
            const D3D12_HEAP_PROPERTIES visible{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
            if (FAILED(device->CreateCommittedResource(&visible, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&pool))) ||
                FAILED(pool->Map(0, nullptr, &mapped))) {
                if (pool) pool->Release();
                pool = nullptr;
                mapped = nullptr;
            }
        }
        if (SUCCEEDED(hr) && !pool) {
            const D3D12_HEAP_PROPERTIES local{D3D12_HEAP_TYPE_DEFAULT};
            hr = device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&pool));
        }
        ID3D12Resource* zeros = nullptr;
        if (SUCCEEDED(hr)) {
            const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
            buffer.Width = 256;
            buffer.Flags = D3D12_RESOURCE_FLAG_NONE;
            hr = device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&zeros));
            void* z = nullptr;
            if (SUCCEEDED(hr)) hr = zeros->Map(0, nullptr, &z);
            if (SUCCEEDED(hr)) {
                std::memset(z, 0, 256);
                zeros->Unmap(0, nullptr);
            }
        }
        log("x2w-gpu indirect: argument clamp %s (widest %u groups in x, %s card, rest %s), budget pool %u slots %s -> 0x%08lx",
            clamp.pso ? "made" : "REFUSED", vi::widest(amd, wide), amd ? "an AMD" : "a non-AMD", rest ? "on" : "off", gpu_budget_slots,
            mapped ? "CPU-visible (notes)" : "device local", static_cast<unsigned long>(hr));
        if (SUCCEEDED(hr) && clamp.pso && pool && zeros) {
            gpu_clamp = clamp;
            gpu_budget_zeros = zeros;
            gpu_budget_words = static_cast<const volatile std::uint32_t*>(mapped);
            MemoryBarrier();
            gpu_budget_pool = pool;
        } else {
            if (pool) pool->Release();
            if (zeros) zeros->Release();
            clamp.release();
            failed = true;
        }
    }
    ReleaseSRWLockExclusive(&lock);
    return gpu_budget_pool != nullptr;
}

// Per list (own().gpu[1], WP4's): 0 = no budget slot since the creation or reset, else the slot + 1.
std::uintptr_t& gpu_indirect_list_word(Wrapper* list) { return reinterpret_cast<std::uintptr_t&>(list->own().gpu[1]); }

// List reset: the list's next recording takes a fresh budget slot, zeroed at its first clamp. A new list starts at 0
// (own().gpu is zero at creation, and wrapper memory is never reused).
void gpu_indirect_list_reset(Wrapper* list)
{
    if (list) gpu_indirect_list_word(list) = 0;
}

// List close (gpu_hooks.h, vhooks fe1f8bd31): a list may record again without a reset, so its next recording
// also takes a fresh budget slot.
void gpu_indirect_list_close(Wrapper* list)
{
    if (list) gpu_indirect_list_word(list) = 0;
}

// The GPU address of the list's budget slot; records the slot's zeroing at the list's first clamp since its reset.
// One thread records one list at a time, so own().gpu[1] needs no lock.
u64 gpu_indirect_budget(Wrapper* self, ID3D12GraphicsCommandList* list)
{
    std::uintptr_t& word = gpu_indirect_list_word(self);
    if (!word) {
        const std::uint32_t slot = static_cast<std::uint32_t>(InterlockedIncrement(&gpu_budget_next)) % gpu_budget_slots;
        vi::record_budget_zero(list, gpu_budget_pool, u64{slot} * vi::budget_slot_bytes, gpu_budget_zeros);
        word = std::uintptr_t{slot} + 1;
    }
    return gpu_budget_pool->GetGPUVirtualAddress() + u64{static_cast<std::uint32_t>(word - 1)} * vi::budget_slot_bytes;
}

// The title records of a buffer wrapper at [offset, offset + bytes), in place.
bool gpu_indirect_records(Wrapper* resource, u64 offset, u64 bytes, ID3D12Resource** window, u64* at)
{
    if (!resource || resource->own().kind != resource_kind || !resource->own().title_memory) return false;
    GpuFound found;
    if (!gpu_find(reinterpret_cast<u64>(resource->own().title_memory) + offset, bytes, &found) || !found.resource ||
        found.bytes_left < bytes)
        return false;
    *window = found.resource;
    *at = found.offset;
    return true;
}

void gpu_indirect_report();

// ---- ExecuteIndirect (list slot 61) ------------------------------------------------------------------------------
bool gpu_indirect(Wrapper* self, ID3D12GraphicsCommandList* list, void* signature, UINT max_count, Wrapper* arguments,
                      std::uint64_t offset, Wrapper* count_resource, std::uint64_t count_offset)
{
    gpu_indirect_report();
    // The signature: its stride and argument types (device slot 41 keeps them, the console's 0x69 included).
    UINT stride = 0;
    std::uint32_t types[2] = {~0u, ~0u};
    std::size_t argument_count = 0;
    AcquireSRWLockShared(&signature_lock);
    const auto found = signature_infos().find(signature);
    if (found != signature_infos().end()) {
        stride = found->second.stride;
        argument_count = found->second.args.size();
        for (std::size_t i = 0; i < argument_count && i < 2; ++i) types[i] = static_cast<std::uint32_t>(found->second.args[i].Type);
    }
    ReleaseSRWLockShared(&signature_lock);
    std::uint32_t kind = ~0u;
    const bool runs = vi::signature_kind(types, argument_count, &kind);
    const int k = runs ? gpu_indirect_kind_index(kind) : vik_other;
    InterlockedIncrement(&gpu_indirect_counts.calls[k]);
    if (argument_count == 1 && types[0] == vi::console_single_dispatch) InterlockedIncrement(&gpu_indirect_counts.console_single);
    if (count_resource) InterlockedIncrement(&gpu_indirect_counts.counted);
    if (!list) return gpu_indirect_drop(k, vid_not_recording);
    if (!runs) {
        static LONG told;
        if (first_calls(told, 8))
            log("x2w-gpu indirect: signature %p dropped (%zu arguments, first type %u, stride %u)", signature, argument_count,
                types[0], stride);
        return gpu_indirect_drop(k, vid_signature);
    }
    if (kind == vi::kind_mesh && !x2w_gpu::gpu_mesh_on()) return gpu_indirect_drop(k, vid_mesh_off);
    if (!max_count) return true;  // nothing to run (D3D12 runs no record either)
    // The pipeline must be of the kind.
    UINT topology = 0;
    GpuPsoDrop failure;
    const GpuPipeline* p = gpu_bound_pipeline(self, &topology, gpu_drop_trace() ? &failure : nullptr);
    if (!p) {
        if (gpu_drop_trace()) {
            const auto* s = vbind::state_if(self);
            vbind::sample_drop(self, s ? static_cast<Wrapper*>(s->pipeline) : nullptr, vbind::Call::indirect,
                               failure, vbind::drop_no_pipeline);
        }
        return gpu_indirect_drop(k, vid_pipeline);
    }
    constexpr UINT rectangles = 17;  // The console's rectangle list
    const bool rect = topology == rectangles;
    bool per_vertex_inputs = false;
    for (const auto& e : p->inputs) per_vertex_inputs |= !e.per_instance;
    const bool rect_triangles = !p->rects && kind == vi::kind_draw && rect && !per_vertex_inputs;
    bool matches = false;
    if (kind == vi::kind_draw || kind == vi::kind_indexed) matches = (!rect || rect_triangles || p->rects) && !p->compute && !p->mesh;
    else if (kind == vi::kind_mesh) matches = !p->compute && p->mesh;
    else matches = p->compute;
    if (!matches) return gpu_indirect_drop(k, vid_pipeline);
    if (rect_triangles) {
        static LONG told;
        if (first_calls(told, 1)) log("x2w-gpu indirect: rectangles without vertex inputs drawn as their first triangle");
        InterlockedIncrement(&gpu_indirect_counts.rect_triangles);
    }
    // The records and the count, in title memory.
    const u64 bytes = u64{stride} * max_count;
    ID3D12Resource* args_window = nullptr;
    ID3D12Resource* count_window = nullptr;
    u64 args_at = 0, count_at = 0;
    if (!gpu_indirect_records(arguments, offset, bytes, &args_window, &args_at) ||
        (count_resource && !gpu_indirect_records(count_resource, count_offset, 4, &count_window, &count_at)) ||
        !vi::validate_layout(kind, stride, max_count, args_at, count_resource != nullptr, count_at, nullptr))
        return gpu_indirect_drop(k, vid_buffer);
    if (!gpu_indirect_ready()) return gpu_indirect_drop(k, vid_clamp);
    ID3D12CommandSignature* host = gpu_host_signatures.get(device_of(the_device), kind, stride);
    if (!host) {
        static LONG told;
        if (first_calls(told, 4)) log("x2w-gpu indirect: host signature (kind %u, stride %u) refused", kind, stride);
        return gpu_indirect_drop(k, vid_signature);
    }
    const bool compute = kind == vi::kind_dispatch;
    vi::ClampCall c;
    c.kind = kind;
    c.stride = stride;
    c.records = max_count;
    c.src = args_window;
    c.src_offset = args_at;
    c.count = count_window;
    c.count_offset = count_at;
    c.count_out = count_resource != nullptr;
    u64 slot = 0;
    ID3D12Resource* scratch = reserve_indirect_ring(vi::scratch_bytes(c), &slot);  // a chunk in COMMON
    if (!scratch) return gpu_indirect_drop(k, vid_scratch);
    c.dst = scratch;
    c.dst_offset = slot;
    c.dst_before = D3D12_RESOURCE_STATE_COMMON;
    c.budget_gpu = gpu_indirect_budget(self, list);
    c.budget_resource = gpu_budget_pool;
    if (!vi::record_clamp(list, gpu_clamp, c)) return gpu_indirect_drop(k, vid_buffer);
    // The clamp set its own compute root signature and pipeline state (the one PSO of the list, graphics too): WP3 sets
    // both again at the bind that follows.
    gpu_bind_forget_compute(self);
    const D3D12_RESOURCE_BARRIER rest = vi::transition(scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON);
    const bool bound = compute ? gpu_bind_compute(self, list)
                       : kind == vi::kind_mesh ? gpu_bind_mesh(self, list)
                                               : gpu_bind_draw(self, list, kind == vi::kind_indexed);
    if (!bound) {
        list->ResourceBarrier(1, &rest);
        return gpu_indirect_drop(k, vid_bind);
    }
    if (compute) vbind::compute_capture_before(self, list);
    else if (vbind::pixel_capture_on()) {
        vbind::pixel_capture_before(self, list, "indirect");
        if (const auto& cap = vbind::state_of(self).trace_current; cap && cap->pixel &&
            !cap->indirect.record(device_of(the_device), list, scratch, slot, kind, stride, max_count, count_resource != nullptr))
            log("x2w-pixel-capture request=%ld frame=%ld shader=%s indirect argument readback unavailable",
                cap->request, cap->frame, cap->key.c_str());
    }
    list->ExecuteIndirect(host, max_count, scratch, slot, count_resource ? scratch : nullptr, count_resource ? slot + bytes : 0);
    if (compute) vbind::compute_capture_after(self, list);
    else if (vbind::pixel_capture_on()) vbind::pixel_capture_after(self, list);
    gpu_trace_issue(self, "indirect", compute, max_count, stride, static_cast<UINT>(kind));
    gpu_finish_depth_writes(self, list);
    list->ResourceBarrier(1, &rest);
    InterlockedIncrement(&gpu_indirect_counts.executed[k]);
    return true;
}

// ---- DispatchMesh (list slot 129) ---------------------------------------------------------------------------------
bool gpu_mesh(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z)
{
    gpu_indirect_report();
    InterlockedIncrement(&gpu_indirect_counts.mesh);
    if (static_cast<u64>(y) * z != 1) {
        InterlockedIncrement(&gpu_indirect_counts.mesh_yz);
        static LONG told;
        if (first_calls(told, 1)) log("x2w-gpu mesh: a mesh dispatch with y or z groups (%u, %u, %u): run as (%u, 1, 1)", x, y, z, vi::mesh_groups(x));
    }
    ID3D12GraphicsCommandList6* list6 = nullptr;
    if (!list || !x2w_gpu::gpu_mesh_on() || !gpu_bind_mesh(self, list) ||
        FAILED(list->QueryInterface(IID_PPV_ARGS(&list6)))) {
        InterlockedIncrement(&gpu_indirect_counts.mesh_dropped);
        return false;
    }
    if (x > 65535u) InterlockedIncrement(&gpu_indirect_counts.mesh_cut);
    list6->DispatchMesh(vi::mesh_groups(x), 1, 1);
    gpu_trace_issue(self, "mesh", false, vi::mesh_groups(x), 1, 1);
    gpu_finish_depth_writes(self, list);
    list6->Release();
    return true;
}

// ---- DispatchX (list slot 80) ------------------------------------------------------------------------------------
bool gpu_dispatch_x(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, UINT flags)
{
    gpu_indirect_report();
    InterlockedIncrement(&gpu_indirect_counts.dispatch_x);
    static const bool thread_dimensions = !gpu_indirect_env_zero("X2W_GPU_THREAD_DIMENSIONS");
    UINT topology = 0;
    const GpuPipeline* p = list ? gpu_bound_pipeline(self, &topology) : nullptr;
    const x2w_gpu::GpuShader* cs = p && p->compute && p->stage_count ? p->stages[0].shader : nullptr;
    if (!(flags & vi::thread_dimensions_flag) || !thread_dimensions || !cs || !cs->has_threads) {
        // Counted in groups: a plain dispatch (WP3).
        const bool ran = gpu_dispatch(self, list, x, y, z);
        if (!ran) InterlockedIncrement(&gpu_indirect_counts.dispatch_x_dropped);
        return ran;
    }
    InterlockedIncrement(&gpu_indirect_counts.dispatch_x_threads);
    if (!gpu_bind_compute(self, list)) {
        InterlockedIncrement(&gpu_indirect_counts.dispatch_x_dropped);
        return false;
    }
    const std::uint32_t threads[3] = {x, y, z};
    std::uint32_t groups[3];
    vi::dispatch_groups(threads, cs->threads, groups);
    // The program is told how many threads run in x, where it asks (its bind set that to all of them).
    if (p->threads_x_param != UINT32_MAX) list->SetComputeRoot32BitConstant(p->threads_x_param, x, p->threads_x_dword);
    vbind::compute_capture_before(self, list);
    list->Dispatch(groups[0], groups[1], groups[2]);
    if (groups[0] && groups[1] && groups[2]) vbind::finish_light(self, list);
    vbind::compute_capture_after(self, list);
    gpu_trace_issue(self, "dispatch-x", true, groups[0], groups[1], groups[2]);
    gpu_finish_depth_writes(self, list);
    return true;
}

// ---- the report, every 300 frames ---------------------------------------------------------------------------------
void gpu_indirect_report()
{
    static volatile LONG reported;
    const LONG frame = frame_number;
    const LONG last = reported;
    if (frame - last < 300 || InterlockedCompareExchange(&reported, frame, last) != last) return;
    GpuIndirectCounts c{};
    for (int t = 0; t < vik_kinds; ++t) {
        c.calls[t] = InterlockedExchange(&gpu_indirect_counts.calls[t], 0);
        c.executed[t] = InterlockedExchange(&gpu_indirect_counts.executed[t], 0);
        for (int r = 0; r < vid_reasons; ++r) c.dropped[r][t] = InterlockedExchange(&gpu_indirect_counts.dropped[r][t], 0);
    }
    c.console_single = InterlockedExchange(&gpu_indirect_counts.console_single, 0);
    c.counted = InterlockedExchange(&gpu_indirect_counts.counted, 0);
    c.rect_triangles = InterlockedExchange(&gpu_indirect_counts.rect_triangles, 0);
    c.mesh = InterlockedExchange(&gpu_indirect_counts.mesh, 0);
    c.mesh_dropped = InterlockedExchange(&gpu_indirect_counts.mesh_dropped, 0);
    c.mesh_cut = InterlockedExchange(&gpu_indirect_counts.mesh_cut, 0);
    c.mesh_yz = InterlockedExchange(&gpu_indirect_counts.mesh_yz, 0);
    c.dispatch_x = InterlockedExchange(&gpu_indirect_counts.dispatch_x, 0);
    c.dispatch_x_threads = InterlockedExchange(&gpu_indirect_counts.dispatch_x_threads, 0);
    c.dispatch_x_dropped = InterlockedExchange(&gpu_indirect_counts.dispatch_x_dropped, 0);
    char dropped[768] = "";
    int at = 0;
    for (int r = 0; r < vid_reasons; ++r) {
        LONG total = 0;
        for (int t = 0; t < vik_kinds; ++t) total += c.dropped[r][t];
        if (!total || at >= static_cast<int>(sizeof(dropped)) - 96) continue;
        at += std::snprintf(dropped + at, sizeof(dropped) - at, "%s%s %ld [", at ? ", " : "", gpu_indirect_drop_names[r], total);
        bool first = true;
        for (int t = 0; t < vik_kinds; ++t)
            if (c.dropped[r][t]) {
                at += std::snprintf(dropped + at, sizeof(dropped) - at, "%s%s %ld", first ? "" : " ", gpu_indirect_kind_names[t],
                                    c.dropped[r][t]);
                first = false;
            }
        at += std::snprintf(dropped + at, sizeof(dropped) - at, "]");
    }
    log("x2w-gpu indirect to frame %ld: calls (draw %ld, indexed %ld, dispatch %ld of which 0x69 %ld, mesh %ld, other %ld), "
        "executed (draw %ld, indexed %ld, dispatch %ld, mesh %ld), dropped (%s); count buffer %ld, rectangles as triangles %ld; "
        "DispatchMesh %ld (dropped %ld, cut to 65535 %ld, y*z != 1 %ld); DispatchX %ld (in threads %ld, dropped %ld)",
        frame, c.calls[0], c.calls[1], c.calls[2], c.console_single, c.calls[3], c.calls[4], c.executed[0], c.executed[1], c.executed[2],
        c.executed[3], at ? dropped : "none", c.counted, c.rect_triangles, c.mesh, c.mesh_dropped, c.mesh_cut, c.mesh_yz, c.dispatch_x,
        c.dispatch_x_threads, c.dispatch_x_dropped);
    if (gpu_budget_words) {
        // Each slot holds the notes of the last recording that used it (zeroed at that recording's first clamp).
        unsigned long long refused = 0, mesh_asked = 0;
        std::uint32_t biggest = 0, largest = 0, spent_full = 0;
        for (std::uint32_t s = 0; s < gpu_budget_slots; ++s) {
            const volatile std::uint32_t* w = gpu_budget_words + u64{s} * (vi::budget_slot_bytes / 4);
            if (w[0] >= vi::budget) ++spent_full;
            largest = w[1] > largest ? w[1] : largest;
            mesh_asked += w[2];
            refused += w[3];
            biggest = w[4] > biggest ? w[4] : biggest;
        }
        log("x2w-gpu indirect notes (last recording per pool slot): lists that spent the whole budget %u, records refused %llu, "
            "largest cost %u, largest mesh group count asked %u, mesh groups asked %llu",
            spent_full, refused, biggest, largest, mesh_asked);
    }
}
