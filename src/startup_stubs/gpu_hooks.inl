// Device and command-list hooks for translated shaders. gpu_install installs these handlers only when the ABI
// is enabled.
//
// Device slots 57 and 67 return E_NOTIMPL and a null output. Slot 15 reports descriptor increments of 32 for views
// and 16 for samplers. Slots 10, 11, 62 and 86 log stage keys when X2W_GPU_KEYLOG is set, record the pipeline,
// and return a placeholder with S_OK. Heap, view and copy hooks extend the shim handler.
//
// List setters 32–46 record heaps and roots. State setters augment the existing handler. Direct and indirect draws,
// mesh and dispatch calls use translated pipelines and count dropped work. Reset and close update the recording
// lifetime.

namespace gpu_detail {

volatile LONG64 counters[x2w_gpu::vc_count];
const char* const counter_names[x2w_gpu::vc_count] = {
    "pipelines", "pipelines kept", "keys", "refused 57", "refused 67", "heaps", "views", "copies", "setters",
    "draws", "draws dropped", "dispatches", "dispatches dropped", "meshes", "meshes dropped", "indirects",
    "indirects dropped", "mesh-x dropped", "resets", "closes", "submits"};

// Counts and, at 1, 2, 4, 8 ... (from 1024 on, every 1024th), logs one line with every counter: there is no exit hook
// and runs end by PID, so the log always holds a recent total.
void bump(int which)
{
    const LONG64 n = InterlockedIncrement64(&counters[which]);
    if (which == x2w_gpu::vc_views || which == x2w_gpu::vc_copies || which == x2w_gpu::vc_setters) return;
    if ((n & (n - 1)) != 0 && (n < 1024 || (n & 1023) != 0)) return;
    char line[1024];
    int used = std::snprintf(line, sizeof(line), "x2w-gpu %s %lld:", counter_names[which], static_cast<long long>(n));
    for (int i = 0; i < x2w_gpu::vc_count && used > 0 && used < static_cast<int>(sizeof(line)); ++i)
        used += std::snprintf(line + used, sizeof(line) - used, " %s=%lld%s", counter_names[i],
                              static_cast<long long>(counters[i]), i + 1 < x2w_gpu::vc_count ? "," : "");
    log("%s", line);
}

// The G0 key log: one line per new (key, stage), written straight to the file (no buffering) so a killed run keeps it.
SRWLOCK key_lock = SRWLOCK_INIT;
std::unordered_set<std::uint64_t>& seen_keys() { static std::unordered_set<std::uint64_t> s; return s; }
HANDLE key_file()
{
    static const HANDLE file = [] {
        char path[MAX_PATH]{};
        const DWORD n = GetEnvironmentVariableA("X2W_GPU_KEYLOG", path, sizeof(path));
        if (!n || n >= sizeof(path)) return INVALID_HANDLE_VALUE;
        const HANDLE h = CreateFileA(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
        log("x2w-gpu key log %s -> %s", path, h == INVALID_HANDLE_VALUE ? "NOT opened" : "opened");
        return h;
    }();
    return file;
}

// Keys `bytes` into `code` and logs a new (key, stage). False: the bytes cannot be read.
bool key_stage(int slot, x2w_gpu::GpuStage stage, const void* bytes, std::size_t size, x2w_gpu::GpuStageCode& code)
{
    code = {};
    code.stage = stage;
    code.bytes = bytes;
    code.size = size;
    if (!bytes || !size || size > (64u << 20) || !x2w_local::readable(bytes, size)) return false;
    x2w_gpu::gpu_key(bytes, size, code.key);
    const std::uint64_t id = std::strtoull(code.key, nullptr, 16) * 9 + static_cast<std::uint64_t>(stage);
    AcquireSRWLockExclusive(&key_lock);
    const bool fresh = seen_keys().insert(id).second;
    ReleaseSRWLockExclusive(&key_lock);
    if (!fresh) return true;
    char line[128];
    const int n = std::snprintf(line, sizeof(line), "%s %s slot%d %zu\r\n", code.key, x2w_gpu::gpu_stage_name(stage),
                                slot, size);
    if (const HANDLE file = key_file(); file != INVALID_HANDLE_VALUE && n > 0) {
        DWORD written = 0;
        WriteFile(file, line, static_cast<DWORD>(n), &written, nullptr);
    }
    static LONG shown;
    if (first_calls(shown, 8)) log("x2w-gpu key %s %s slot%d bytes %zu", code.key, x2w_gpu::gpu_stage_name(stage), slot, size);
    bump(x2w_gpu::vc_keys);
    return true;
}

struct Bytecode {
    const void* code;
    std::size_t size;
};

// A stage at `at` (a D3D12_SHADER_BYTECODE in title memory) into the description.
void add_stage(GpuCreateDesc& desc, x2w_gpu::GpuStage stage, const void* at)
{
    Bytecode code{};
    if (!x2w_local::readable(at, sizeof(code))) {
        desc.unreadable = true;
        return;
    }
    std::memcpy(&code, at, sizeof(code));
    if (!code.code || !code.size) return;  // the stage is absent
    if (desc.stage_count >= 8) return;
    if (!key_stage(desc.slot, stage, code.code, code.size, desc.stages[desc.stage_count])) desc.unreadable = true;
    ++desc.stage_count;
}

// A pipeline subobject's size (type included) and stage: the console numbers amplification and mesh 23 and 24. 0:
// unknown (the walk stops).
std::size_t subobject(std::uint32_t kind, int& stage)
{
    using S = x2w_gpu::GpuStage;
    stage = -1;
    switch (kind) {
    case 1: stage = static_cast<int>(S::vertex); return 24;
    case 2: stage = static_cast<int>(S::pixel); return 24;
    case 3: stage = static_cast<int>(S::domain); return 24;
    case 4: stage = static_cast<int>(S::hull); return 24;
    case 5: stage = static_cast<int>(S::geometry); return 24;
    case 6: stage = static_cast<int>(S::compute); return 24;
    case 23: stage = static_cast<int>(S::amplification); return 24;
    case 24: stage = static_cast<int>(S::mesh); return 24;
    case 0: return 16;
    case 12: case 19: return 24;
    case 7: case 15: return 40;
    case 8: return 336;
    case 10: return 48;
    case 11: return 56;
    case 17: return 16;
    case 21: return 64;
    case 22: return 32;
    case 9: case 13: case 14: case 16: case 18: case 20: return 8;
    default: return 0;
    }
}

// Hands the stages to WP2 and returns the placeholder pipeline (S_OK) the title keeps.
HRESULT hand_out(GpuCreateDesc& desc, const GUID* iid, void** out)
{
    bump(x2w_gpu::vc_pipelines);
    Wrapper* pipeline = wrap(nullptr, pipeline_kind);
    *out = pipeline;
    if (gpu_create_pipeline(pipeline, desc)) bump(x2w_gpu::vc_pipelines_kept);
    (void)iid;
    return S_OK;
}

}  // namespace gpu_detail

long long gpu_count(int counter)
{
    return counter >= 0 && counter < x2w_gpu::vc_count ? static_cast<long long>(gpu_detail::counters[counter]) : -1;
}

void gpu_released(Wrapper* object)
{
    switch (object->own().kind) {
    case pipeline_kind: gpu_pipeline_released(object); break;
    case descriptor_heap_kind: gpu_heap_released(object); break;
    case resource_kind: {
        if (builder_memory_on()) vbind::builder_arenas().forget(object);
        std::lock_guard<std::mutex> guard(gpu_detail_heap::depth_lock);
        gpu_detail_heap::depth_pairs().erase(object);
        break;
    }
    case list_kind: gpu_list_released(object); break;
    default: break;
    }
}

void gpu_list_reset(Wrapper* list)
{
    gpu_detail::bump(x2w_gpu::vc_resets);
    gpu_bind_list_reset(list);
    gpu_indirect_list_reset(list);
}
void gpu_list_close(Wrapper* list)
{
    gpu_detail::bump(x2w_gpu::vc_closes);
    gpu_bind_list_close(list);
    gpu_indirect_list_close(list);
}
void gpu_list_submit(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists)
{
    gpu_detail::bump(x2w_gpu::vc_submits);
    gpu_bind_first_submit(count, lists);
    gpu_heap_submit(queue, count, lists);
}
void gpu_list_submitted(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists)
{
    gpu_bind_submitted(queue, count, lists);
}

// Device slots 10 and 11 take (head, riid, ppv). The graphics head contains the root signature, VS, PS, DS, HS and
// GS. The compute head used by slots 11 and 62 contains the root signature and CS.
template <int N>
HRESULT __stdcall gpu_create_pipeline_slot(Wrapper*, const void* head, const GUID* iid, void** out)
{
    Timed timed(time_pipeline);
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if (!head) return E_INVALIDARG;
    using S = x2w_gpu::GpuStage;
    GpuCreateDesc desc{};
    desc.slot = N;
    desc.desc = head;
    desc.desc_bytes = sizeof(void*) + (N == 10 ? 5 : 1) * sizeof(gpu_detail::Bytecode);
    if (!x2w_local::readable(head, desc.desc_bytes)) return E_INVALIDARG;
    std::memcpy(&desc.title_root, head, sizeof(void*));
    const auto* stages = static_cast<const unsigned char*>(head) + sizeof(void*);
    if constexpr (N == 10) {
        const S order[] = {S::vertex, S::pixel, S::domain, S::hull, S::geometry};
        for (int i = 0; i < 5; ++i) gpu_detail::add_stage(desc, order[i], stages + i * sizeof(gpu_detail::Bytecode));
    } else {
        gpu_detail::add_stage(desc, S::compute, stages);
    }
    return gpu_detail::hand_out(desc, iid, out);
}

// Slot 62 takes the compute descriptor, extension count, extensions, riid and output pointer. Extensions are
// ignored.
HRESULT __stdcall gpu_create_compute_x_slot(Wrapper* self, const D3D12_COMPUTE_PIPELINE_STATE_DESC* head, u64, u64,
                                                const GUID* iid, void** out)
{
    return gpu_create_pipeline_slot<62>(self, head, iid, out);
}

// Slot 86: (stream desc {size, stream}, riid, ppv).
HRESULT __stdcall gpu_create_stream_slot(Wrapper*, const D3D12_PIPELINE_STATE_STREAM_DESC* stream, const GUID* iid, void** out)
{
    Timed timed(time_pipeline);
    if (!out || !iid) return E_POINTER;
    *out = nullptr;
    if (kind_for(*iid) != pipeline_kind) return E_NOINTERFACE;
    if (!stream || !x2w_local::readable(stream, sizeof(*stream)) || !stream->pPipelineStateSubobjectStream) return E_INVALIDARG;
    GpuCreateDesc desc{};
    desc.slot = 86;
    desc.desc = stream->pPipelineStateSubobjectStream;
    desc.desc_bytes = stream->SizeInBytes;
    const auto* bytes = static_cast<const unsigned char*>(stream->pPipelineStateSubobjectStream);
    if (desc.desc_bytes > (1u << 20) || !x2w_local::readable(bytes, desc.desc_bytes)) {
        desc.unreadable = true;
    } else {
        for (std::size_t at = 0; at + 8 <= desc.desc_bytes;) {
            std::uint32_t kind;
            std::memcpy(&kind, bytes + at, sizeof(kind));
            int stage;
            const std::size_t size = gpu_detail::subobject(kind, stage);
            if (!size || at + size > desc.desc_bytes) {
                static LONG unknown;
                if (size == 0 && first_calls(unknown, 4))
                    log("x2w-gpu slot86: subobject %u of unknown size at +0x%zx; the rest is ignored", kind, at);
                break;
            }
            if (kind == 0 && !desc.title_root) std::memcpy(&desc.title_root, bytes + at + 8, sizeof(void*));
            if (stage >= 0) gpu_detail::add_stage(desc, static_cast<x2w_gpu::GpuStage>(stage), bytes + at + 8);
            at += size;
        }
    }
    return gpu_detail::hand_out(desc, iid, out);
}

// Slots 57 and 67: refused, out null. The title goes without the pipeline.
template <int N>
HRESULT __stdcall gpu_refuse_pipeline_slot(Wrapper*, const void* desc, const GUID*, void** out)
{
    if (out) *out = nullptr;
    gpu_detail::bump(N == 57 ? x2w_gpu::vc_refused_57 : x2w_gpu::vc_refused_67);
    static LONG shown;
    if (first_calls(shown, 2)) {
        char bytes[3 * 64 + 1];
        hex_text(desc, 64, bytes, sizeof(bytes));
        log("x2w-gpu Device::slot%d refused (E_NOTIMPL) desc=[%s]", N, bytes);
    }
    return E_NOTIMPL;
}

// Slot 15: the console's increments for views and samplers.
static_assert(x2w_startup::title_sampler_stride == x2w_gpu::abi::sampler_increment);
UINT __stdcall gpu_increment_slot(Wrapper* self, UINT type)
{
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) return x2w_gpu::abi::view_increment;
    if (type == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) return x2w_gpu::abi::sampler_increment;
    return device_descriptor_increment_size(self, type);
}

// Slot 14: X2W's heap (with title_stride 32 / 16 under ABI, device_create_descriptor_heap), then WP5.
HRESULT __stdcall gpu_heap_slot(Wrapper* self, const D3D12_DESCRIPTOR_HEAP_DESC* desc, const GUID* iid, void** out)
{
    const HRESULT hr = device_create_descriptor_heap(self, desc, iid, out);
    if (FAILED(hr) || !out || !*out) return hr;
    Wrapper* heap = static_cast<Wrapper*>(*out);
    const u64 gpu = (desc->Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) && heap->own().real
        ? static_cast<ID3D12DescriptorHeap*>(heap->own().real)->GetGPUDescriptorHandleForHeapStart().ptr : 0;
    gpu_detail::bump(x2w_gpu::vc_heaps);
    gpu_heap_created(self, heap, *desc, heap->own().title_cpu, gpu);
    return hr;
}

// View slots 17-22: X2W's view_slot (token write, desktop view), then WP5.
template <int N, int D>
u64 gpu_view_slot(Wrapper* self, u64 a, u64 b, u64 c, u64 d)
{
    const u64 result = view_slot<N, D>(self, a, b, c, d);
    gpu_detail::bump(x2w_gpu::vc_views);
    gpu_heap_view(self, N, a, b, c, d);
    return result;
}
// Slot 61, the console sampler (desc, dest): X2W has no handler (a no-op, or the census).
u64 __stdcall gpu_sampler_x_slot(Wrapper* self, u64 desc, u64 dest, u64 r9)
{
    if (texture_census_on()) device_slot61_census(self, desc, dest, r9);
    gpu_detail::bump(x2w_gpu::vc_views);
    gpu_heap_view(self, 61, desc, dest, 0, 0);
    return 0;
}
// Slots 70 and 71, the placed SRV / UAV (resource desc, view desc + address, dest).
void __stdcall gpu_placed_srv_slot(Wrapper* self, const D3D12_RESOURCE_DESC* resource, const MemoryShaderResourceView* input,
                                       u64 destination)
{
    device_memory_srv(self, resource, input, destination);
    gpu_detail::bump(x2w_gpu::vc_views);
    gpu_heap_view(self, 70, reinterpret_cast<u64>(resource), reinterpret_cast<u64>(input), destination, 0);
}
void __stdcall gpu_placed_uav_slot(Wrapper* self, const D3D12_RESOURCE_DESC* resource, const MemoryUnorderedAccessView* input,
                                       u64 destination)
{
    if (memory_srv_buffer_on()) device_memory_uav(self, resource, input, destination);
    gpu_detail::bump(x2w_gpu::vc_views);
    gpu_heap_view(self, 71, reinterpret_cast<u64>(resource), reinterpret_cast<u64>(input), destination, 0);
}
// Copy slots 24 and 23.
void __stdcall gpu_copy_simple_slot(Wrapper* self, UINT count, u64 dest, u64 src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    device_copy_descriptors_simple(self, count, dest, src, type);
    gpu_detail::bump(x2w_gpu::vc_copies);
    gpu_heap_copy_simple(self, count, dest, src, type);
}
void __stdcall gpu_copy_slot(Wrapper* self, UINT dest_count, const D3D12_CPU_DESCRIPTOR_HANDLE* dest_starts, const UINT* dest_sizes,
                                 UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE* src_starts, const UINT* src_sizes,
                                 D3D12_DESCRIPTOR_HEAP_TYPE type)
{
    device_copy_descriptors(self, dest_count, dest_starts, dest_sizes, src_count, src_starts, src_sizes, type);
    gpu_detail::bump(x2w_gpu::vc_copies);
    gpu_heap_copy(self, dest_count, dest_starts, dest_sizes, src_count, src_starts, src_sizes, type);
}

// ----- List slots (entered through noop_slot -> list_typed) ------------------------------------------------------
namespace gpu_detail {
ListTyped vh_old_typed[slot_count];  // list_typed as X2W installed it, before gpu_install
inline Wrapper* vh_w(u64 v) { return reinterpret_cast<Wrapper*>(v); }
inline u64 vh_old(int slot, u64 a, u64 b, u64 c, u64 d, u64 e, u64 f, u64 g, u64 h)
{
    return vh_old_typed[slot] ? vh_old_typed[slot](a, b, c, d, e, f, g, h) : 0;
}
#define X2W_GPU_ARGS u64 self, u64 a1, u64 a2, u64 a3, u64 s0, u64 s1, u64 s2, u64 s3
#define X2W_GPU_OLD(slot) vh_old(slot, self, a1, a2, a3, s0, s1, s2, s3)

// Root setters 32-46: the hook only. X2W's handlers would issue desktop root calls (list_graphics_root's
// SetGraphicsRootSignature, list_root_srv's SetGraphicsRootShaderResourceView, the *_desktop compute variants) with the
// title's parameter indices onto the list WP3 binds with its own host roots, so under ABI they are not called.
template <int N>
u64 __stdcall vh_setter(X2W_GPU_ARGS)
{
    bump(x2w_gpu::vc_setters);
    Wrapper* list = vh_w(self);
    const UINT p = static_cast<UINT>(a1);
    switch (N) {
    case 32: gpu_set_heaps(list, static_cast<UINT>(a1), reinterpret_cast<void* const*>(a2)); break;
    case 33: case 34: gpu_roots_signature(list, N == 33, as_wrapper(reinterpret_cast<const void*>(a1))); break;
    case 35: case 36: gpu_roots_table(list, N == 35, p, a2); break;
    case 37: case 38: {  // (parameter, value, offset)
        const std::uint32_t value = static_cast<std::uint32_t>(a2);
        gpu_roots_constants(list, N == 37, p, 1, &value, static_cast<UINT>(a3));
        break;
    }
    case 39: case 40:  // (parameter, count, data, offset)
        gpu_roots_constants(list, N == 39, p, static_cast<UINT>(a2), reinterpret_cast<const std::uint32_t*>(a3),
                                static_cast<UINT>(s0));
        break;
    case 41: case 42: gpu_roots_view(list, N == 41, p, a2, GpuRootKind::cbv); break;
    case 43: case 44: gpu_roots_view(list, N == 43, p, a2, GpuRootKind::srv); break;
    case 45: case 46: gpu_roots_view(list, N == 45, p, a2, GpuRootKind::uav); break;
    default: break;
    }
    (void)s1, (void)s2, (void)s3;
    return 0;
}

// Other draw state: X2W's handler first, then the hook.
template <int N>
u64 __stdcall vh_state(X2W_GPU_ARGS)
{
    const u64 result = X2W_GPU_OLD(N);
    Wrapper* list = vh_w(self);
    switch (N) {
    case 20: gpu_topology(list, static_cast<UINT>(a1)); break;
    case 21: gpu_viewports(list, static_cast<UINT>(a1), reinterpret_cast<const D3D12_VIEWPORT*>(a2)); break;
    case 22: gpu_scissors(list, static_cast<UINT>(a1), reinterpret_cast<const D3D12_RECT*>(a2)); break;
    case 23: gpu_blend_factor(list, reinterpret_cast<const float*>(a1)); break;
    case 24: gpu_stencil_ref(list, static_cast<UINT>(a1)); break;
    case 25: gpu_set_pipeline(list, as_wrapper(reinterpret_cast<const void*>(a1))); break;
    case 26: gpu_barriers(list, static_cast<UINT>(a1), reinterpret_cast<const D3D12_RESOURCE_BARRIER*>(a2)); break;
    case 47: gpu_ia_index(list, reinterpret_cast<const D3D12_INDEX_BUFFER_VIEW*>(a1)); break;
    case 48: gpu_ia_vertex(list, static_cast<UINT>(a1), static_cast<UINT>(a2), reinterpret_cast<const D3D12_VERTEX_BUFFER_VIEW*>(a3)); break;
    case 50:
        gpu_targets(list, static_cast<UINT>(a1), reinterpret_cast<const D3D12_CPU_DESCRIPTOR_HANDLE*>(a2), static_cast<BOOL>(a3 & 0xff),
                        reinterpret_cast<const D3D12_CPU_DESCRIPTOR_HANDLE*>(s0));
        break;
    default: break;
    }
    return result;
}

void vh_counted(bool issued, int done, int dropped) { bump(issued ? done : dropped); }

// Draws and dispatches: only the translated shader path.
u64 __stdcall vh_draw(X2W_GPU_ARGS)
{
    (void)s1, (void)s2, (void)s3;
    Wrapper* s = vh_w(self);
    vh_counted(gpu_draw(s, recording(s), static_cast<UINT>(a1), static_cast<UINT>(a2), static_cast<UINT>(a3), static_cast<UINT>(s0)),
            x2w_gpu::vc_draws, x2w_gpu::vc_draws_dropped);
    return 0;
}
u64 __stdcall vh_draw_indexed(X2W_GPU_ARGS)
{
    (void)s2, (void)s3;
    Wrapper* s = vh_w(self);
    vh_counted(gpu_draw_indexed(s, recording(s), static_cast<UINT>(a1), static_cast<UINT>(a2), static_cast<UINT>(a3),
                                 static_cast<INT>(static_cast<UINT>(s0)), static_cast<UINT>(s1)),
            x2w_gpu::vc_draws, x2w_gpu::vc_draws_dropped);
    return 0;
}
u64 __stdcall vh_dispatch(X2W_GPU_ARGS)
{
    (void)s0, (void)s1, (void)s2, (void)s3;
    Wrapper* s = vh_w(self);
    vh_counted(gpu_dispatch(s, recording(s), static_cast<UINT>(a1), static_cast<UINT>(a2), static_cast<UINT>(a3)),
            x2w_gpu::vc_dispatches, x2w_gpu::vc_dispatches_dropped);
    return 0;
}
u64 __stdcall vh_dispatch_x(X2W_GPU_ARGS)  // slot 80: (x, y, z, flags)
{
    (void)s1, (void)s2, (void)s3;
    Wrapper* s = vh_w(self);
    vh_counted(gpu_dispatch_x(s, recording(s), static_cast<UINT>(a1), static_cast<UINT>(a2), static_cast<UINT>(a3), static_cast<UINT>(s0)),
            x2w_gpu::vc_dispatches, x2w_gpu::vc_dispatches_dropped);
    return 0;
}
u64 __stdcall vh_mesh(X2W_GPU_ARGS)  // slot 129: (x, y, z)
{
    (void)s0, (void)s1, (void)s2, (void)s3;
    Wrapper* s = vh_w(self);
    const bool issued = x2w_gpu::gpu_mesh_on() &&
        gpu_mesh(s, recording(s), static_cast<UINT>(a1), static_cast<UINT>(a2), static_cast<UINT>(a3));
    vh_counted(issued, x2w_gpu::vc_meshes, x2w_gpu::vc_meshes_dropped);
    return 0;
}
u64 __stdcall vh_mesh_x(X2W_GPU_ARGS)  // slot 100: DispatchMeshX(x, flags, indirect arguments, payload), dropped
{
    (void)self, (void)a1, (void)a2, (void)a3, (void)s0, (void)s1, (void)s2, (void)s3;
    bump(x2w_gpu::vc_mesh_x_dropped);
    return 0;
}
u64 __stdcall vh_indirect(X2W_GPU_ARGS)  // slot 61: (signature, max count, arguments, offset, count, count offset)
{
    (void)s3;
    Wrapper* s = vh_w(self);
    const bool issued = x2w_gpu::gpu_indirect_on() &&
        gpu_indirect(s, recording(s), reinterpret_cast<void*>(a1), static_cast<UINT>(a2), vh_w(a3), s0, vh_w(s1), s2);
    vh_counted(issued, x2w_gpu::vc_indirects, x2w_gpu::vc_indirects_dropped);
    return 0;
}
// List reset (139) and close (9).
u64 __stdcall vh_reset(X2W_GPU_ARGS)
{
    const u64 result = X2W_GPU_OLD(139);
    if (SUCCEEDED(static_cast<HRESULT>(result))) gpu_list_reset(vh_w(self));
    return result;
}
u64 __stdcall vh_close(X2W_GPU_ARGS)
{
    gpu_list_close(vh_w(self));
    return X2W_GPU_OLD(9);
}
#undef X2W_GPU_OLD
#undef X2W_GPU_ARGS

// Slot 70, depth bounds (min, max): float arguments, so the table entry itself (noop_slot passes integer registers).
void __stdcall vh_depth_bounds(Wrapper* self, float min, float max) { gpu_depth_bounds(self, min, max); }

template <int... N>
void vh_install_setters(std::integer_sequence<int, N...>)
{
    ((list_typed[N] = reinterpret_cast<ListTyped>(&vh_setter<N>)), ...);
}
template <int... N>
void vh_install_state(std::integer_sequence<int, N...>)
{
    ((list_typed[N] = reinterpret_cast<ListTyped>(&vh_state<N>)), ...);
}
}  // namespace gpu_detail

// Called by build_tables when gpu_abi_on(), after X2W's own entries are in.
void gpu_install()
{
    using namespace gpu_detail;
    void** device = tables[device_kind];
    device[10] = reinterpret_cast<void*>(&gpu_create_pipeline_slot<10>);
    device[11] = reinterpret_cast<void*>(&gpu_create_pipeline_slot<11>);
    device[62] = reinterpret_cast<void*>(&gpu_create_compute_x_slot);
    device[86] = reinterpret_cast<void*>(&gpu_create_stream_slot);
    device[57] = reinterpret_cast<void*>(&gpu_refuse_pipeline_slot<57>);
    device[67] = reinterpret_cast<void*>(&gpu_refuse_pipeline_slot<67>);
    device[15] = reinterpret_cast<void*>(&gpu_increment_slot);
    device[14] = reinterpret_cast<void*>(&gpu_heap_slot);
    device[17] = reinterpret_cast<void*>(&gpu_view_slot<17, 1>);
    device[18] = reinterpret_cast<void*>(&gpu_view_slot<18, 2>);
    device[19] = reinterpret_cast<void*>(&gpu_view_slot<19, 3>);
    device[20] = reinterpret_cast<void*>(&gpu_view_slot<20, 2>);
    device[21] = reinterpret_cast<void*>(&gpu_view_slot<21, 2>);
    device[22] = reinterpret_cast<void*>(&gpu_view_slot<22, 1>);
    device[23] = reinterpret_cast<void*>(&gpu_copy_slot);
    device[24] = reinterpret_cast<void*>(&gpu_copy_simple_slot);
    device[61] = reinterpret_cast<void*>(&gpu_sampler_x_slot);
    device[70] = reinterpret_cast<void*>(&gpu_placed_srv_slot);
    device[71] = reinterpret_cast<void*>(&gpu_placed_uav_slot);

    std::copy(std::begin(list_typed), std::end(list_typed), std::begin(vh_old_typed));
    vh_install_setters(std::integer_sequence<int, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46>{});
    vh_install_state(std::integer_sequence<int, 20, 21, 22, 23, 24, 25, 26, 47, 48, 50>{});
    list_typed[12] = reinterpret_cast<ListTyped>(&vh_draw);
    list_typed[13] = reinterpret_cast<ListTyped>(&vh_draw_indexed);
    list_typed[14] = reinterpret_cast<ListTyped>(&vh_dispatch);
    list_typed[x2w_list_slots::execute_indirect_slot] = reinterpret_cast<ListTyped>(&vh_indirect);
    // Newer list layouts shift slots by two starting at slot 59. DispatchX uses 80, DispatchMeshX uses 100 and
    // DispatchMesh uses 129.
    list_typed[80] = reinterpret_cast<ListTyped>(&vh_dispatch_x);
    list_typed[100] = reinterpret_cast<ListTyped>(&vh_mesh_x);
    list_typed[129] = reinterpret_cast<ListTyped>(&vh_mesh);
    if (vh_old_typed[x2w_list_slots::reset_slot]) list_typed[x2w_list_slots::reset_slot] = reinterpret_cast<ListTyped>(&vh_reset);
    if (vh_old_typed[9]) list_typed[9] = reinterpret_cast<ListTyped>(&vh_close);
    if (list_noop()) tables[list_kind][70] = reinterpret_cast<void*>(&vh_depth_bounds);
    log("x2w-gpu ABI on: slots 57/67 refused, slot 15 answers %u/%u, slots 10/11/62/86 keyed%s; indirect %s, mesh %s, twins %s",
        x2w_gpu::abi::view_increment, x2w_gpu::abi::sampler_increment,
        key_file() != INVALID_HANDLE_VALUE ? " (key log open)" : " (no X2W_GPU_KEYLOG)",
        x2w_gpu::gpu_indirect_on() ? "on" : "off", x2w_gpu::gpu_mesh_on() ? "on" : "off",
        x2w_gpu::gpu_twins_on() ? "on" : "off");
}
